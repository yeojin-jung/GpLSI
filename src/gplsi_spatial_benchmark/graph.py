"""Spatial graphs built only from coordinates and declared modeling units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SpatialGraph:
    edges: pd.DataFrame
    adjacency: csr_matrix
    metadata: dict


def build_within_unit_knn_graph(
    coordinates: np.ndarray,
    unit_ids: np.ndarray,
    *,
    k: int = 6,
) -> SpatialGraph:
    """Return a symmetrized kNN graph with no edge crossing a unit boundary."""

    xy = np.asarray(coordinates, dtype=float)
    units = np.asarray(unit_ids).astype(str)
    if xy.ndim != 2 or xy.shape[1] != 2 or units.shape != (xy.shape[0],):
        raise ValueError("coordinates must be n by 2 and unit_ids must have length n")
    if not np.isfinite(xy).all() or k < 1:
        raise ValueError("coordinates must be finite and k must be positive")
    pairs: dict[tuple[int, int], float] = {}
    for unit in sorted(np.unique(units)):
        idx = np.flatnonzero(units == unit)
        if idx.size < 2:
            continue
        neighbors = min(k + 1, idx.size)
        distances, local = NearestNeighbors(n_neighbors=neighbors).fit(xy[idx]).kneighbors(xy[idx])
        positive = distances[:, 1:][distances[:, 1:] > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        scale = max(scale, np.finfo(float).eps)
        for i in range(idx.size):
            for distance, j in zip(distances[i, 1:], local[i, 1:]):
                src, tgt = sorted((int(idx[i]), int(idx[j])))
                if src == tgt:
                    continue
                weight = float(np.exp(-((float(distance) / scale) ** 2)))
                pairs[(src, tgt)] = max(weight, pairs.get((src, tgt), 0.0))
    if not pairs:
        raise ValueError("no spatial edges could be constructed")
    ordered = sorted(pairs)
    src = np.asarray([pair[0] for pair in ordered], dtype=np.int64)
    tgt = np.asarray([pair[1] for pair in ordered], dtype=np.int64)
    weight = np.asarray([pairs[pair] for pair in ordered], dtype=float)
    if np.any(units[src] != units[tgt]):
        raise AssertionError("a graph edge crossed a modeling-unit boundary")
    edge_frame = pd.DataFrame({"src": src, "tgt": tgt, "weight": weight})
    adjacency = csr_matrix((weight, (src, tgt)), shape=(xy.shape[0], xy.shape[0]))
    return SpatialGraph(
        edge_frame,
        adjacency,
        {
            "kind": "symmetric_knn_unique_undirected_edges",
            "k": int(k),
            "unit_count": int(np.unique(units).size),
            "edge_count": int(len(edge_frame)),
            "weight": "exp(-(distance/within_unit_median_positive_knn_distance)^2)",
        },
    )
