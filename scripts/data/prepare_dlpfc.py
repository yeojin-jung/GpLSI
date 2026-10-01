"""Build the processed Visium DLPFC H5AD read by ``configs/dlpfc/`` (data/dlpfc/).

Sources (public, Maynard et al. 2021 / LIBD HumanPilot):

* counts: ``https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/<id>_filtered_feature_bc_matrix.h5``
* spot positions: ``LieberInstitute/HumanPilot/10X/<id>/tissue_positions_list.txt``
* manual layers: ``ground_truth`` column of
  ``LieberInstitute/HumanPilot/outputs/SpatialDE_clustering/cluster_labels_<id>.csv``
  (the ``layer_guess_reordered`` annotation distributed by spatialLIBD).

The output keeps raw integer UMI counts for all 33,538 genes; gene panels are
chosen later from training counts only, inside each task. Layer labels and the
donor are stored as evaluation-only columns.

Usage::

    python scripts/data/prepare_dlpfc.py --download
    python scripts/data/prepare_dlpfc.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


SECTIONS = {
    "Br5292": ["151507", "151508", "151509", "151510"],
    "Br5595": ["151669", "151670", "151671", "151672"],
    "Br8100": ["151673", "151674", "151675", "151676"],
}
URLS = {
    "filtered_feature_bc_matrix.h5": "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{s}_filtered_feature_bc_matrix.h5",
    "tissue_positions_list.txt": "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{s}/tissue_positions_list.txt",
    "cluster_labels.csv": "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/outputs/SpatialDE_clustering/cluster_labels_{s}.csv",
}
LAYER_NAMES = {f"Layer_{i}": f"Layer{i}" for i in range(1, 7)} | {"WM": "WM"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for sections in SECTIONS.values():
        for s in sections:
            for suffix, url in URLS.items():
                target = raw / f"{s}_{suffix}"
                if not target.is_file():
                    subprocess.run(["curl", "-sfL", "-o", str(target), url.format(s=s)], check=True)


def read_10x_h5(path: Path) -> tuple[csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        genes_by_spots = csc_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])
        )
        barcodes = m["barcodes"][:].astype(str)
        gene_ids = m["features/id"][:].astype(str)
        symbols = m["features/name"][:].astype(str)
    return genes_by_spots, barcodes, gene_ids, symbols


def build(raw: Path, output: Path) -> ad.AnnData:
    blocks, frames = [], []
    reference_ids = None
    for donor, sections in SECTIONS.items():
        for position, s in enumerate(sections):
            matrix, barcodes, gene_ids, symbols = read_10x_h5(raw / f"{s}_filtered_feature_bc_matrix.h5")
            if reference_ids is None:
                reference_ids, reference_symbols = gene_ids, symbols
            elif not np.array_equal(gene_ids, reference_ids):
                raise ValueError(f"{s}: gene order differs from the first section")
            positions = pd.read_csv(
                raw / f"{s}_tissue_positions_list.txt", header=None,
                names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
            ).set_index("barcode")
            labels = pd.read_csv(raw / f"{s}_cluster_labels.csv", usecols=["key", "ground_truth"])
            labels["barcode"] = labels["key"].str.split("_", n=1).str[1]
            labels = labels.set_index("barcode")["ground_truth"]
            obs = positions.loc[barcodes].copy()
            if not (obs["in_tissue"] == 1).all():
                raise ValueError(f"{s}: filtered matrix contains out-of-tissue spots")
            obs["layer_guess_reordered"] = labels.reindex(barcodes).map(LAYER_NAMES).fillna("")
            obs["sample_id"] = s
            obs["subject"] = donor
            # Adjacent pairs 0/1 and 2/3 are 300 um apart (Maynard et al. 2021).
            obs["position"] = "0" if position < 2 else "300"
            obs["replicate"] = str(position % 2 + 1)
            obs.index = [f"{s}_{b}" for b in barcodes]
            frames.append(obs)
            blocks.append(matrix.T.tocsr())
    from scipy.sparse import vstack

    X = vstack(blocks).tocsr().astype(np.int32)
    X.sum_duplicates()
    X.sort_indices()
    obs = pd.concat(frames)
    var = pd.DataFrame({"gene_id": reference_ids, "symbol": reference_symbols}, index=reference_ids)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    # Visium spots lie on a hexagonal lattice: columns step by 2 within a row and
    # rows are offset by one column. Scaling rows by sqrt(3) makes all six
    # lattice neighbours equidistant, so a 6-NN graph is exactly the hex grid.
    adata.obsm["spatial"] = np.column_stack(
        [obs["array_col"].to_numpy(float), np.sqrt(3.0) * obs["array_row"].to_numpy(float)]
    )
    adata.obsm["spatial_pixel"] = obs[["pxl_col", "pxl_row"]].to_numpy(float)
    adata.uns["benchmark_unit_column"] = "sample_id"
    adata.uns["graph_unit_column"] = "sample_id"
    adata.uns["fit_forbidden_obs_columns"] = ["layer_guess_reordered"]
    adata.uns["coordinate_convention"] = "x=array_col, y=sqrt(3)*array_row (Visium hex lattice)"
    adata.uns["source_sha256"] = {
        p.name: _sha256(p) for p in sorted(raw.iterdir()) if p.is_file()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression="gzip")
    summary = {
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "nnz": int(X.nnz),
        "total_umis": int(X.sum()),
        "labeled_spots": int((obs["layer_guess_reordered"] != "").sum()),
        "spots_per_section": obs["sample_id"].value_counts().sort_index().to_dict(),
        "layers_per_section": {
            s: sorted(set(g["layer_guess_reordered"]) - {""}) for s, g in obs.groupby("sample_id")
        },
    }
    (output.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/dlpfc/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/dlpfc/visium_dlpfc.h5ad"))
    parser.add_argument("--download", action="store_true", help="download missing source files first")
    args = parser.parse_args()
    if args.download:
        download(args.raw)
    build(args.raw, args.output)


if __name__ == "__main__":
    main()
