"""Processed-data loader with an explicit fit/evaluation separation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


DATA_FILES = {
    "visium_dlpfc": "visium_dlpfc/visium_dlpfc.h5ad",
    "merfish_trem2_5xfad": "merfish_trem2_5xfad/merfish_trem2_5xfad.h5ad",
    "xenium_uc": "xenium_uc/xenium_uc.h5ad",
}


@dataclass
class SpatialSlice:
    dataset: str
    unit_id: str
    counts: csr_matrix
    coordinates: np.ndarray
    observation_ids: np.ndarray
    feature_ids: np.ndarray
    external: pd.DataFrame
    graph_unit_ids: np.ndarray
    metadata: dict


def _deterministic_spatial_subset(coordinates: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Farthest-point subset, seeded reproducibly, for spatial coverage."""

    if n >= coordinates.shape[0]:
        return np.arange(coordinates.shape[0])
    xy = np.asarray(coordinates, dtype=float)
    rng = np.random.default_rng(seed)
    chosen = np.empty(n, dtype=int)
    chosen[0] = int(rng.integers(xy.shape[0]))
    nearest = np.sum((xy - xy[chosen[0]]) ** 2, axis=1)
    for i in range(1, n):
        chosen[i] = int(np.argmax(nearest))
        nearest = np.minimum(nearest, np.sum((xy - xy[chosen[i]]) ** 2, axis=1))
    return np.sort(chosen)


def load_spatial_slice(
    processed_root: str | Path,
    dataset: str,
    unit_id: str,
    *,
    max_observations: int | None = None,
    seed: int = 0,
) -> SpatialSlice:
    """Load counts/coordinates for fitting; keep labels in an external-only frame."""

    if dataset not in DATA_FILES:
        raise ValueError(f"unknown dataset {dataset!r}")
    obj = ad.read_h5ad(Path(processed_root) / DATA_FILES[dataset])
    unit_column = str(obj.uns["benchmark_unit_column"])
    mask = np.asarray(obj.obs[unit_column].astype(str) == str(unit_id))
    if not np.any(mask):
        raise ValueError(f"unit {unit_id!r} was not found in {dataset}")
    sub = obj[mask].copy()
    counts = csr_matrix(sub.X) if issparse(sub.X) else csr_matrix(np.asarray(sub.X))
    xy = np.asarray(sub.obsm["spatial"], dtype=float)
    indices = np.arange(sub.n_obs)
    if max_observations and sub.n_obs > max_observations:
        indices = _deterministic_spatial_subset(xy, max_observations, seed)
        sub = sub[indices].copy()
        counts = counts[indices]
        xy = xy[indices]
    forbidden = [str(value) for value in obj.uns["fit_forbidden_obs_columns"]]
    external_columns = [column for column in forbidden if column in sub.obs]
    external = sub.obs[external_columns].copy().reset_index(drop=True)
    graph_column = str(obj.uns.get("graph_unit_column", unit_column))
    graph_units = sub.obs[graph_column].astype(str).to_numpy()
    if np.any(counts.data < 0) or not np.all(counts.data == np.floor(counts.data)):
        raise ValueError("processed X is not an integer count matrix")
    return SpatialSlice(
        dataset=dataset,
        unit_id=str(unit_id),
        counts=counts.astype(np.int32),
        coordinates=xy,
        observation_ids=sub.obs_names.astype(str).to_numpy(),
        feature_ids=sub.var_names.astype(str).to_numpy(),
        external=external,
        graph_unit_ids=graph_units,
        metadata={
            "processed_file": str(Path(processed_root) / DATA_FILES[dataset]),
            "unit_column": unit_column,
            "graph_unit_column": graph_column,
            "source_observations_in_unit": int(np.count_nonzero(mask)),
            "selected_observations": int(sub.n_obs),
            "selection": "all" if indices.size == np.count_nonzero(mask) else "seeded_farthest_point",
            "fit_forbidden_obs_columns": forbidden,
        },
    )
