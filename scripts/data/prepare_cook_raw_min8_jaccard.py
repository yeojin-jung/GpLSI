#!/usr/bin/env python3
"""Freeze the raw What's Cooking >=8-recipe subset and its Jaccard graph.

Rows are selected from the audited literal-token raw corpus using their raw
ingredient-count total before feature thresholding, count thinning, or graph
construction.  The complete 5,346-feature vocabulary and row order are kept.
On the retained rows, each recipe chooses five neighbors by standard binary-set
Jaccard similarity among its own cuisine and configured neighboring cuisines.
Directed choices are preserved for audit and unioned into an undirected,
unit-weight modeling graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "cook" / "dataset"
DEFAULT_BASE = DATASET_ROOT / "raw_jaccard_v1"
DEFAULT_OUTPUT = DATASET_ROOT / "raw_min8_jaccard_v1"
DEFAULT_NEIGHBOR_MAP = DATASET_ROOT / "neighbor_countries_mapping.pkl"
MIN_RAW_INGREDIENT_COUNT = 8
TOP_K = 5
BLOCK_ROWS = 128


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Any) -> str:
    encoded = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_numeric(value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_dense_view(
    matrix: sparse.csr_matrix,
    dtype: str,
    *,
    row_divisors: np.ndarray | None = None,
    block_rows: int = 128,
) -> str:
    output_dtype = np.dtype(dtype)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(matrix.shape), separators=(",", ":")).encode())
    digest.update(output_dtype.str.encode())
    for start in range(0, matrix.shape[0], block_rows):
        stop = min(start + block_rows, matrix.shape[0])
        block = matrix[start:stop].toarray().astype(output_dtype, copy=False)
        if row_divisors is not None:
            block = np.ascontiguousarray(
                block / row_divisors[start:stop, None], dtype=output_dtype
            )
        digest.update(np.ascontiguousarray(block).tobytes(order="C"))
    return digest.hexdigest()


def _verify_base_artifacts(base: Path, manifest: dict[str, Any]) -> None:
    for name in ("counts_csr.npz", "recipes.csv.gz", "feature_ids.json"):
        expected = str(manifest["artifacts"][name]["sha256"])
        observed = _sha256_file(base / name)
        if observed != expected:
            raise ValueError(f"base raw artifact {name} changed: {observed} != {expected}")


def build_jaccard_graph(
    counts: sparse.csr_matrix,
    cuisines: np.ndarray,
    observation_ids: np.ndarray,
    base_rows: np.ndarray,
    neighbor_mapping: dict[str, list[str]],
    *,
    top_k: int = TOP_K,
    block_rows: int = BLOCK_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic directed choices and their undirected union."""

    if top_k < 1 or block_rows < 1:
        raise ValueError("top_k and block_rows must be positive")
    presence = counts.tocsr().astype(np.int8, copy=True)
    presence.data.fill(1)
    cardinality = np.diff(presence.indptr).astype(float)
    labels = np.asarray(cuisines, dtype=str)
    ids = np.asarray(observation_ids)
    original_rows = np.asarray(base_rows, dtype=np.int64)
    n = presence.shape[0]
    if not (labels.size == ids.size == original_rows.size == n):
        raise ValueError("graph metadata are not row-aligned")
    if len(np.unique(ids)) != n or len(np.unique(original_rows)) != n:
        raise ValueError("recipe ids and base rows must be unique")
    cuisine_rows = {
        cuisine: np.flatnonzero(labels == cuisine)
        for cuisine in sorted(set(labels.tolist()))
    }
    selected: dict[tuple[int, int], float] = {}
    directed_records: list[dict[str, Any]] = []
    for cuisine, query_indices in cuisine_rows.items():
        eligible = {cuisine, *map(str, neighbor_mapping.get(cuisine, []))}
        candidate_indices = np.sort(
            np.concatenate(
                [
                    cuisine_rows[value]
                    for value in sorted(eligible)
                    if value in cuisine_rows
                ]
            )
        )
        if candidate_indices.size <= top_k:
            raise ValueError(
                f"cuisine {cuisine!r} has only {candidate_indices.size} eligible rows"
            )
        candidate_presence = presence[candidate_indices]
        candidate_cardinality = cardinality[candidate_indices]
        for start in range(0, query_indices.size, block_rows):
            query = query_indices[start : start + block_rows]
            intersections = (presence[query] @ candidate_presence.T).toarray()
            unions = (
                cardinality[query, None]
                + candidate_cardinality[None, :]
                - intersections
            )
            similarities = np.divide(
                intersections,
                unions,
                out=np.zeros_like(intersections, dtype=float),
                where=unions > 0,
            )
            for local_row, source in enumerate(query):
                scores = similarities[local_row]
                scores[candidate_indices == source] = -np.inf
                # Preserve the current raw/Jaccard experiment's frozen row tie rule.
                order = np.lexsort((candidate_indices, -scores))
                for rank, position in enumerate(order[:top_k], start=1):
                    target = int(candidate_indices[position])
                    similarity = float(scores[position])
                    edge = (min(int(source), target), max(int(source), target))
                    selected[edge] = max(selected.get(edge, -np.inf), similarity)
                    directed_records.append(
                        {
                            "src": int(source),
                            "tgt": target,
                            "rank": rank,
                            "src_base_row": int(original_rows[source]),
                            "tgt_base_row": int(original_rows[target]),
                            "src_recipe_id": ids[source],
                            "tgt_recipe_id": ids[target],
                            "jaccard_similarity": similarity,
                            "jaccard_distance": 1.0 - similarity,
                        }
                    )
        print(
            f"finished {cuisine}: {query_indices.size} queries, "
            f"{candidate_indices.size} eligible recipes",
            file=sys.stderr,
            flush=True,
        )
    edge_records = [
        {
            "src": source,
            "tgt": target,
            "weight": 1.0,
            "jaccard_similarity": selected[(source, target)],
            "jaccard_distance": 1.0 - selected[(source, target)],
        }
        for source, target in sorted(selected)
    ]
    return pd.DataFrame.from_records(edge_records), pd.DataFrame.from_records(
        directed_records
    )


def prepare(
    base: Path,
    output: Path,
    neighbor_map_path: Path,
    *,
    minimum_count: int = MIN_RAW_INGREDIENT_COUNT,
    top_k: int = TOP_K,
    block_rows: int = BLOCK_ROWS,
) -> dict[str, Any]:
    started = time.perf_counter()
    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    base_manifest = json.loads((base / "manifest.json").read_text())
    _verify_base_artifacts(base, base_manifest)
    base_counts = sparse.load_npz(base / "counts_csr.npz").tocsr()
    base_recipes = pd.read_csv(base / "recipes.csv.gz")
    feature_ids = json.loads((base / "feature_ids.json").read_text())
    if base_counts.shape != (len(base_recipes), len(feature_ids)):
        raise ValueError("base raw count and metadata dimensions disagree")
    base_lengths = np.asarray(base_counts.sum(axis=1)).reshape(-1).astype(np.int64)
    retained_base_rows = np.flatnonzero(base_lengths >= minimum_count)
    counts = base_counts[retained_base_rows].tocsr()
    recipes = base_recipes.iloc[retained_base_rows].reset_index(drop=True).copy()
    recipes["base_row_index"] = retained_base_rows
    recipes["raw_literal_ingredient_count"] = base_lengths[retained_base_rows]
    lengths = np.asarray(counts.sum(axis=1)).reshape(-1).astype(float)
    positive_features = np.asarray(counts.sum(axis=0)).reshape(-1) > 0
    with neighbor_map_path.open("rb") as handle:
        neighbor_mapping = pickle.load(handle)

    graph, directed = build_jaccard_graph(
        counts,
        recipes["cuisine"].astype(str).to_numpy(),
        recipes["id"].to_numpy(),
        retained_base_rows,
        neighbor_mapping,
        top_k=top_k,
        block_rows=block_rows,
    )
    if len(directed) != counts.shape[0] * top_k:
        raise ValueError("directed Jaccard table does not contain top_k rows per recipe")
    per_source = directed.groupby("src", sort=False).size().to_numpy()
    if (
        per_source.size != counts.shape[0]
        or np.any(per_source != top_k)
        or np.any(directed["src"].to_numpy() == directed["tgt"].to_numpy())
    ):
        raise ValueError("directed Jaccard table has missing, duplicate, or self choices")
    endpoints = graph[["src", "tgt"]].to_numpy(dtype=np.int64)
    if len(graph) == 0 or np.any(endpoints[:, 0] >= endpoints[:, 1]):
        raise ValueError("Jaccard graph is empty or not canonical undirected src < tgt")
    adjacency = sparse.csr_matrix(
        (graph["weight"], (graph["src"], graph["tgt"])),
        shape=(counts.shape[0], counts.shape[0]),
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    component_count, component_labels = connected_components(
        adjacency, directed=False
    )
    degree = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1)

    output.mkdir(parents=True, exist_ok=True)
    counts_path = output / "counts_csr.npz"
    recipes_path = output / "recipes.csv.gz"
    features_path = output / "feature_ids.json"
    edges_path = output / "jaccard_edges.csv.gz"
    directed_path = output / "directed_jaccard_neighbors.csv.gz"
    sparse.save_npz(counts_path, counts, compressed=True)
    recipes.to_csv(
        recipes_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    features_path.write_text(json.dumps(feature_ids, ensure_ascii=False, indent=2) + "\n")
    graph.to_csv(
        edges_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    directed.to_csv(
        directed_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )

    contract_hashes = {
        "D_sha256": _sha256_dense_view(counts, "<i8"),
        "X_sha256": _sha256_dense_view(counts, "<f8", row_divisors=lengths),
        "N_i_sha256": _sha256_numeric(lengths, "<f8"),
        "feature_ids_sha256": _sha256_strings(feature_ids),
        "observation_ids_sha256": _sha256_strings(recipes["id"]),
        "group_ids_sha256": _sha256_strings(recipes["cuisine"]),
        "edges_sha256": _sha256_numeric(endpoints, "<i8"),
        "edge_weights_sha256": _sha256_numeric(
            graph["weight"].to_numpy(), "<f8"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "whats_cooking_raw_min8_jaccard",
        "canonical_scope": (
            "audited cuisine-balanced literal-token raw corpus, restricted before "
            "all modeling to recipes with raw ingredient-count total >= 8"
        ),
        "base_raw_contract": {
            "relative_path": "../raw_jaccard_v1/manifest.json",
            "dataset": base_manifest["dataset"],
            "identity": {
                key: base_manifest["contract_hashes"][key]
                for key in (
                    "D_sha256",
                    "X_sha256",
                    "N_i_sha256",
                    "feature_ids_sha256",
                    "observation_ids_sha256",
                    "group_ids_sha256",
                )
            },
        },
        "row_filter": {
            "rule": "raw literal ingredient-count row total >= 8",
            "minimum_inclusive": int(minimum_count),
            "applied_before": [
                "Jaccard graph construction",
                "count thinning",
                "Tran feature thresholding",
                "Ke weighting",
            ],
            "before_n": int(base_counts.shape[0]),
            "after_n": int(counts.shape[0]),
            "removed_n": int(base_counts.shape[0] - counts.shape[0]),
            "retained_base_rows_sha256": _sha256_numeric(
                retained_base_rows, "<i8"
            ),
        },
        "matrix": {
            "shape": list(counts.shape),
            "n": int(counts.shape[0]),
            "p": int(counts.shape[1]),
            "nnz": int(counts.nnz),
            "positive_feature_count": int(positive_features.sum()),
            "zero_feature_count": int((~positive_features).sum()),
            "N_mean": float(lengths.mean()),
            "N_min": float(lengths.min()),
            "N_max": float(lengths.max()),
            "corpus_tokens": int(lengths.sum()),
            "feature_thresholding": "none in canonical contract",
            "token_parser": "literal mapped ingredient strings; duplicates retained as counts",
        },
        "graph": {
            "definition": (
                "standard binary set Jaccard similarity |S_i intersection S_j| / "
                "|S_i union S_j| on the >=8 recipe subset; for each recipe choose "
                "the five most similar other recipes among its own cuisine and the "
                "cuisines in the supplied neighboring-country map; break ties by "
                "retained row order inherited from the frozen raw corpus; union "
                "directed choices and use unit edge weights"
            ),
            "neighbor_mapping_sha256": _sha256_file(neighbor_map_path),
            "top_k_directed": int(top_k),
            "candidate_scope": "same cuisine plus configured neighboring cuisines",
            "tie_break": "ascending retained row, preserving frozen base-row order",
            "edge_count_undirected": int(len(graph)),
            "component_count": int(component_count),
            "component_sizes": np.bincount(component_labels).astype(int).tolist(),
            "degree_min": int(degree.min()),
            "degree_median": float(np.median(degree)),
            "degree_max": int(degree.max()),
            "similarity_min": float(graph["jaccard_similarity"].min()),
            "similarity_median": float(graph["jaccard_similarity"].median()),
            "similarity_max": float(graph["jaccard_similarity"].max()),
            "weights": "unweighted unit edges after neighbor selection",
        },
        "model_preprocessing": {
            "requested_tran_alpha": 0.005,
            "application": "P1/P3 column preprocessing after count thinning",
            "canonical_rows_removed_by_tran": 0,
        },
        "contract_hashes": contract_hashes,
        "artifacts": {},
        "runtime_seconds": time.perf_counter() - started,
    }
    artifacts = [counts_path, recipes_path, features_path, edges_path, directed_path]
    manifest["artifacts"] = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbor-map", type=Path, default=DEFAULT_NEIGHBOR_MAP)
    parser.add_argument("--minimum-count", type=int, default=MIN_RAW_INGREDIENT_COUNT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--block-rows", type=int, default=BLOCK_ROWS)
    args = parser.parse_args()
    manifest = prepare(
        args.base,
        args.output,
        args.neighbor_map,
        minimum_count=args.minimum_count,
        top_k=args.top_k,
        block_rows=args.block_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
