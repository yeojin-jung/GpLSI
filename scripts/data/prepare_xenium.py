"""Build the processed Xenium ulcerative-colitis H5AD read by ``configs/xenium/`` (data/xenium/).

Source (public, CC BY 4.0; Mennillo et al., J Clin Invest 2026): Figshare
article 27327813 v1, Dataset 1 (custom 290-gene panel, integrated replicates),
annotated/QC object ``25_11_22_Xenium_Dataset1_290_IntReps1and2_Annotated.h5ad``
(MD5 419cab98a3be238d310db6e3fc92bc42).

Output: raw integer counts (``layers/raw_counts``; the normalized layer is
dropped) of the 581,967 cells assigned to a patient-condition/timepoint unit
(the 221 ``unassigned`` cells are excluded), cell centroids (µm) as
coordinates, and

* ``unit`` (``24_01_17_HS``): fitting unit, one patient at one condition/timepoint (25);
* ``stratum`` (``Patient_ID_cores_combined``): slide/core, the graph boundary (114);
* ``patient`` (HS31-HS50, 20) and ``condition`` (HC, PRE/POST_VDZ_R/NR):
  constant within a unit, for cross-unit analyses only;
* evaluation labels, never used in fitting: ``compartment`` (5 broad cell
  classes), ``cell_type_coarse`` (13), ``cell_type_fine`` (34), and
  ``leiden_annotation`` (26 Leiden-derived clusters; kept but not scored, being
  derived from the same cells' expression);
* ``neighborhood`` (8, scored): cellular neighbourhoods built here from the
  annotated cell types, the only spatial label (the release has no tissue
  regions). For every cell, the coarse cell-type composition of its
  ``NEIGHBORHOOD_WINDOW`` = 10 nearest cells in the same slide/core (itself
  included; Schurch et al., Cell 2020); then k-means with
  ``NEIGHBORHOOD_COUNT`` = 8 clusters (the authors' number of CellCharter
  neighbourhoods) over all cells jointly, seed 0, 10 restarts. Uses only the
  cell-type annotations, never expression; fixed before any topic model was
  fitted. Clusters are numbered by size (N1 largest) and described in
  ``uns["neighborhood_composition"]``.

Usage::

    python scripts/data/prepare_xenium.py --download
    python scripts/data/prepare_xenium.py --raw path/to/annotated.h5ad
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
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "data" / "xenium" / "xenium_uc.h5ad"
RAW = REPO_ROOT / "data" / "xenium" / "raw" / "25_11_22_Xenium_Dataset1_290_IntReps1and2_Annotated.h5ad"
URL = "https://ndownloader.figshare.com/files/59799746"
MD5 = "419cab98a3be238d310db6e3fc92bc42"
COLUMNS = {
    "24_01_17_HS": "unit",
    "Patient_ID_cores_combined": "stratum",
    "24_01_17_Condition": "condition",
    "25_06_11_Compartments": "compartment",
    "25_06_11_Common_Coarse_Xenium_Combinedv3": "cell_type_coarse",
    "24_05_29_Fine_annotations_Xenium_combined": "cell_type_fine",
    "24_01_08_EM_combined_from_leiden": "leiden_annotation",
}
EVALUATION_LABELS = ["compartment", "cell_type_coarse", "cell_type_fine", "neighborhood"]
NEIGHBORHOOD_WINDOW = 10
NEIGHBORHOOD_COUNT = 8
NEIGHBORHOOD_SEED = 0
# Published dimensions (GpLSI benchmark note, section 3.3) checked on build.
EXPECTED = {"cells": 581_967, "genes": 290, "units": 25, "strata": 114, "patients": 20}


def _digest(path: Path, name: str) -> str:
    digest = hashlib.new(name)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw: Path) -> None:
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists() or _digest(raw, "md5") != MD5:
        subprocess.run(["curl", "-L", "--fail", "-o", str(raw), URL], check=True)
    if _digest(raw, "md5") != MD5:
        raise RuntimeError(f"{raw} does not match the Figshare MD5 {MD5}")


def cellular_neighborhoods(
    coordinates: np.ndarray, strata: np.ndarray, cell_types: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster cells by the annotated cell-type mix of their nearest neighbours.

    Returns labels ``N1``..``N<count>`` (N1 = largest) and each cluster's mean
    composition (rows: neighbourhoods, columns: coarse cell types).
    """

    types = np.unique(cell_types)
    one_hot = (cell_types[:, None] == types[None, :]).astype(float)
    windows = np.zeros_like(one_hot)
    for stratum in np.unique(strata):
        index = np.flatnonzero(strata == stratum)
        neighbors = min(NEIGHBORHOOD_WINDOW, index.size)
        _, nearest = NearestNeighbors(n_neighbors=neighbors).fit(coordinates[index]).kneighbors(coordinates[index])
        windows[index] = one_hot[index][nearest].mean(axis=1)
    clusters = KMeans(NEIGHBORHOOD_COUNT, n_init=10, random_state=NEIGHBORHOOD_SEED).fit_predict(windows)
    order = np.argsort(-np.bincount(clusters, minlength=NEIGHBORHOOD_COUNT), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    labels = np.array([f"N{r + 1}" for r in rank[clusters]])
    composition = pd.DataFrame(windows, columns=types).groupby(labels).mean()
    return labels, composition


def build(raw: Path, output: Path) -> ad.AnnData:
    source = ad.read_h5ad(raw, backed="r")
    obs = source.obs[list(COLUMNS)].rename(columns=COLUMNS).astype(str)
    keep = (obs["unit"] != "unassigned").to_numpy()
    with h5py.File(raw, "r") as handle:
        group = handle["layers/raw_counts"]
        counts = csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(group.attrs["shape"]),
        )
    if not np.array_equal(counts.data, np.round(counts.data)) or counts.data.min() < 0:
        raise ValueError("layers/raw_counts is not a nonnegative integer matrix")
    counts = counts[keep].astype(np.int32)
    obs = obs.loc[keep].copy()
    obs["patient"] = obs["unit"].str.split("_").str[0]
    coordinates = np.asarray(source.obsm["spatial"], dtype=float)[keep]

    observed = {
        "cells": counts.shape[0], "genes": counts.shape[1], "units": obs["unit"].nunique(),
        "strata": obs["stratum"].nunique(), "patients": obs["patient"].nunique(),
    }
    if observed != EXPECTED:
        raise ValueError(f"unexpected dimensions {observed}; expected {EXPECTED}")
    if (obs.groupby("unit")["condition"].nunique() != 1).any():
        raise ValueError("a unit spans several conditions")
    obs["neighborhood"], composition = cellular_neighborhoods(
        coordinates, obs["stratum"].to_numpy(), obs["cell_type_coarse"].to_numpy()
    )

    adata = ad.AnnData(
        X=counts,
        obs=obs.astype("category"),
        var=pd.DataFrame(index=source.var_names.astype(str)),
    )
    adata.obs_names = source.obs_names[keep].astype(str)
    adata.obsm["spatial"] = coordinates
    adata.uns["benchmark_unit_column"] = "unit"
    adata.uns["graph_unit_column"] = "stratum"
    adata.uns["evaluation_label_columns"] = EVALUATION_LABELS
    adata.uns["fit_forbidden_obs_columns"] = EVALUATION_LABELS + ["leiden_annotation", "condition", "patient"]
    adata.uns["coordinate_convention"] = "x_centroid, y_centroid in µm (obsm['spatial'] of the source)"
    adata.uns["neighborhood_composition"] = json.dumps(composition.round(4).to_dict(orient="index"))
    adata.uns["neighborhood_definition"] = (
        f"coarse cell-type composition of the {NEIGHBORHOOD_WINDOW} nearest cells in the same slide/core "
        f"(self included); k-means, {NEIGHBORHOOD_COUNT} clusters, seed {NEIGHBORHOOD_SEED}, 10 restarts; "
        "N1 = largest"
    )
    adata.uns["source"] = json.dumps({"url": URL, "md5": MD5, "file": raw.name, "figshare_article": "27327813 v1"})
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression="gzip")
    summary = {**observed, "molecules": int(counts.sum()), "nonzero": int(counts.nnz),
               "cells_per_unit": obs["unit"].value_counts().sort_index().to_dict(),
               "units_per_condition": obs.groupby("condition")["unit"].nunique().to_dict(),
               "neighborhood_cells": obs["neighborhood"].value_counts().sort_index().to_dict(),
               "neighborhood_composition": composition.round(3).to_dict(orient="index")}
    (output.parent / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch the Figshare file first (1.5 GB)")
    parser.add_argument("--raw", type=Path, default=RAW, help="annotated source H5AD")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.download:
        download(args.raw)
    adata = build(args.raw, args.output)
    print(f"wrote {args.output}: {adata.n_obs:,} cells x {adata.n_vars} genes, "
          f"{adata.obs['unit'].nunique()} units, {adata.obs['stratum'].nunique()} graph strata")


if __name__ == "__main__":
    main()
