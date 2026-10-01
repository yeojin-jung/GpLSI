"""Build the processed MERFISH TREM2/5xFAD mouse-brain H5AD read by ``configs/merfish/`` (data/merfish/).

Source (public, CC BY-SA 4.0; Johnston et al., Mol Psychiatry 2025): Brain
Image Library dataset ace-ear-nap (submission b72faf9d87d7fc00),
``Data_Repository/MERFISH_Data.h5ad`` (5,483,600,151 bytes; SHA-256 below,
computed on the copy downloaded 2026-10-01; the library publishes no checksum).
The server is slow per connection, so ``--download`` fetches 8 byte ranges in
parallel.

Output: raw integer counts (dense ``layers/RNA``; the normalized layers are
dropped) of all 432,794 cells x 300 genes, cell centres (``center_x``,
``center_y``) as coordinates, and

* ``unit`` (``gen``): fitting unit, one animal (15);
* ``stratum`` (``gen_fine``): coronal half-section, the graph boundary (19);
* ``genotype`` (``gen_coarse``: WT, 5xFAD, Trem2, Trem2_5xFAD), constant within
  an animal, for cross-animal analyses only;
* evaluation labels, never used in fitting: ``cell_type_coarse`` (9),
  ``cell_type`` (37), ``region_coarse`` (11), ``region`` (17);
* ``plaque_distance`` (no documented unit; meaningful only in the 8 animals
  carrying 5xFAD, around 3,000 elsewhere), for the plaque-proximity task, and
  ``leiden`` (released clusters, not scored).

Usage::

    python scripts/data/prepare_merfish.py --download
    python scripts/data/prepare_merfish.py --raw path/to/MERFISH_Data.h5ad
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "data" / "merfish" / "merfish_trem2_5xfad.h5ad"
RAW = REPO_ROOT / "data" / "merfish" / "raw" / "MERFISH_Data.h5ad"
URL = ("https://download.brainimagelibrary.org/b7/2f/b72faf9d87d7fc00/repository/"
       "Data_Repository/MERFISH_Data.h5ad")
SIZE = 5_483_600_151
SHA256 = "d6876020cfdb5cb2fdeb848550e8334cfc2684b6ce6f7755b862cac567f06a38"
COLUMNS = {
    "gen": "unit",
    "gen_fine": "stratum",
    "gen_coarse": "genotype",
    "cluster_coarse": "cell_type_coarse",
    "cluster": "cell_type",
    "region_labels_coarse": "region_coarse",
    "region_labels": "region",
    "leiden": "leiden",
}
EVALUATION_LABELS = ["cell_type_coarse", "cell_type", "region_coarse", "region"]
# Published dimensions (GpLSI benchmark note, section 3.2) checked on build.
EXPECTED = {"cells": 432_794, "genes": 300, "units": 15, "strata": 19,
            "nonzero": 30_608_835, "molecules": 139_256_236}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw: Path, connections: int = 8) -> None:
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists() and raw.stat().st_size == SIZE and _sha256(raw) == SHA256:
        return
    chunk = -(-SIZE // connections)
    parts = [raw.with_name(f"{raw.name}.part{i}") for i in range(connections)]

    def fetch(i: int) -> None:
        start, end = i * chunk, min(SIZE, (i + 1) * chunk) - 1
        for _ in range(5):
            subprocess.run(["curl", "-sL", "--fail", "-r", f"{start}-{end}", "-o", str(parts[i]), URL], check=False)
            if parts[i].exists() and parts[i].stat().st_size == end - start + 1:
                return
        raise RuntimeError(f"byte range {start}-{end} failed after 5 attempts")

    with ThreadPoolExecutor(connections) as pool:
        list(pool.map(fetch, range(connections)))
    with raw.open("wb") as output:
        for part in parts:
            output.write(part.read_bytes())
            part.unlink()
    if _sha256(raw) != SHA256:
        raise RuntimeError(f"{raw} does not match the recorded SHA-256 {SHA256}")


def build(raw: Path, output: Path) -> ad.AnnData:
    source = ad.read_h5ad(raw, backed="r")
    obs = source.obs[list(COLUMNS)].rename(columns=COLUMNS).astype(str)
    obs["plaque_distance"] = source.obs["plaque_distance"].to_numpy(dtype=float)
    blocks = []
    with h5py.File(raw, "r") as handle:
        dense = handle["layers/RNA"]
        for start in range(0, dense.shape[0], 50_000):
            block = dense[start : start + 50_000]
            if not np.array_equal(block, np.round(block)) or block.min() < 0:
                raise ValueError("layers/RNA is not a nonnegative integer matrix")
            blocks.append(csr_matrix(block.astype(np.int32)))
    counts = vstack(blocks).tocsr()
    coordinates = source.obs[["center_x", "center_y"]].to_numpy(dtype=float)

    observed = {"cells": counts.shape[0], "genes": counts.shape[1], "units": obs["unit"].nunique(),
                "strata": obs["stratum"].nunique(), "nonzero": int(counts.nnz), "molecules": int(counts.sum())}
    if observed != EXPECTED:
        raise ValueError(f"unexpected dimensions {observed}; expected {EXPECTED}")
    for column in ("genotype",):
        if (obs.groupby("unit")[column].nunique() != 1).any():
            raise ValueError(f"an animal spans several values of {column}")
    if (obs.groupby("stratum")["unit"].nunique() != 1).any():
        raise ValueError("a section spans several animals")

    adata = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=source.var_names.astype(str)))
    for column in COLUMNS.values():
        adata.obs[column] = adata.obs[column].astype("category")
    adata.obs_names = source.obs_names.astype(str)
    adata.obsm["spatial"] = coordinates
    adata.uns["benchmark_unit_column"] = "unit"
    adata.uns["graph_unit_column"] = "stratum"
    adata.uns["evaluation_label_columns"] = EVALUATION_LABELS
    adata.uns["fit_forbidden_obs_columns"] = EVALUATION_LABELS + ["genotype", "plaque_distance", "leiden"]
    adata.uns["coordinate_convention"] = "center_x, center_y of the source obs"
    adata.uns["source"] = json.dumps({"url": URL, "sha256": SHA256, "bil_dataset": "ace-ear-nap"})
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression="gzip")
    summary = {**observed,
               "cells_per_unit": obs["unit"].value_counts().sort_index().to_dict(),
               "sections_per_unit": obs.groupby("unit")["stratum"].nunique().to_dict(),
               "genotype_per_unit": obs.groupby("unit")["genotype"].first().to_dict()}
    (output.parent / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch MERFISH_Data.h5ad first (5.5 GB)")
    parser.add_argument("--raw", type=Path, default=RAW, help="source MERFISH_Data.h5ad")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.download:
        download(args.raw)
    adata = build(args.raw, args.output)
    print(f"wrote {args.output}: {adata.n_obs:,} cells x {adata.n_vars} genes, "
          f"{adata.obs['unit'].nunique()} animals, {adata.obs['stratum'].nunique()} sections")


if __name__ == "__main__":
    main()
