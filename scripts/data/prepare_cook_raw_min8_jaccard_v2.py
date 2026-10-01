#!/usr/bin/env python3
"""Freeze the v2 raw What's Cooking >=8/Jaccard data contract.

The transformation order is part of the contract:

1. start from the frozen literal-token ``raw_jaccard_v1`` corpus;
2. retain recipes whose raw ingredient-count total is at least eight;
3. remove ingredient columns whose corpus count is zero on those recipes; and
4. rebuild the same binary-set Jaccard graph on the resulting matrix.

The older ``raw_min8_jaccard_v1`` contract deliberately retained the zero
columns.  This script writes a distinct v2 directory and never modifies v1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import the frozen v1 graph builder so v2 changes only the requested matrix
# contract.  In particular, candidate scope, tie-breaking, and edge unioning
# remain byte-for-byte governed by the existing implementation.
import prepare_cook_raw_min8_jaccard as v1  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "cook" / "dataset"
DEFAULT_BASE = DATASET_ROOT / "raw_jaccard_v1"
DEFAULT_OUTPUT = DATASET_ROOT / "raw_min8_jaccard_v2"
DEFAULT_NEIGHBOR_MAP = DATASET_ROOT / "neighbor_countries_mapping.pkl"
MIN_RAW_INGREDIENT_COUNT = 8
TOP_K = v1.TOP_K
BLOCK_ROWS = v1.BLOCK_ROWS

_sha256_file = v1._sha256_file
_sha256_strings = v1._sha256_strings
_sha256_numeric = v1._sha256_numeric
_sha256_dense_view = v1._sha256_dense_view
_verify_base_artifacts = v1._verify_base_artifacts
build_jaccard_graph = v1.build_jaccard_graph


@dataclass(frozen=True)
class ContractSelection:
    """Row/column selection produced before graph construction."""

    counts: sparse.csr_matrix
    recipes: pd.DataFrame
    feature_ids: list[str]
    retained_base_rows: np.ndarray
    retained_base_columns: np.ndarray
    removed_base_columns: np.ndarray
    base_row_lengths: np.ndarray
    retained_corpus_counts: np.ndarray


def select_min8_positive_columns(
    base_counts: sparse.spmatrix,
    base_recipes: pd.DataFrame,
    base_feature_ids: Sequence[object],
    *,
    minimum_count: int = MIN_RAW_INGREDIENT_COUNT,
) -> ContractSelection:
    """Apply the inclusive row filter, then remove exactly zero-count columns."""

    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    counts = sparse.csr_matrix(base_counts)
    feature_ids = list(map(str, base_feature_ids))
    if counts.shape != (len(base_recipes), len(feature_ids)):
        raise ValueError("base raw count and metadata dimensions disagree")
    if not {"id", "cuisine"}.issubset(base_recipes.columns):
        raise ValueError("base recipe metadata require id and cuisine columns")
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("base feature ids must be unique")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts.data < 0):
        raise ValueError("base raw counts must be nonnegative integers")

    base_row_lengths = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.int64)
    retained_base_rows = np.flatnonzero(base_row_lengths >= minimum_count)
    if retained_base_rows.size == 0:
        raise ValueError("minimum-count rule retained no recipes")

    row_filtered = counts[retained_base_rows].tocsr()
    corpus_counts_before = (
        np.asarray(row_filtered.sum(axis=0)).reshape(-1).astype(np.int64)
    )
    retained_base_columns = np.flatnonzero(corpus_counts_before > 0)
    removed_base_columns = np.flatnonzero(corpus_counts_before == 0)
    if retained_base_columns.size == 0:
        raise ValueError("all ingredient columns are zero after the row filter")

    selected_counts = row_filtered[:, retained_base_columns].tocsr()
    selected_counts.sort_indices()
    selected_counts.eliminate_zeros()
    retained_corpus_counts = (
        np.asarray(selected_counts.sum(axis=0)).reshape(-1).astype(np.int64)
    )
    if np.any(retained_corpus_counts <= 0):
        raise ValueError("zero-corpus ingredient survived the v2 column filter")
    selected_row_lengths = np.asarray(selected_counts.sum(axis=1)).reshape(-1)
    if not np.array_equal(
        selected_row_lengths,
        base_row_lengths[retained_base_rows],
    ):
        raise ValueError("zero-column removal unexpectedly changed recipe lengths")

    recipes = base_recipes.iloc[retained_base_rows].reset_index(drop=True).copy()
    recipes["base_row_index"] = retained_base_rows
    recipes["raw_literal_ingredient_count"] = base_row_lengths[retained_base_rows]
    selected_features = [feature_ids[index] for index in retained_base_columns]
    return ContractSelection(
        counts=selected_counts,
        recipes=recipes,
        feature_ids=selected_features,
        retained_base_rows=retained_base_rows.astype(np.int64, copy=False),
        retained_base_columns=retained_base_columns.astype(np.int64, copy=False),
        removed_base_columns=removed_base_columns.astype(np.int64, copy=False),
        base_row_lengths=base_row_lengths,
        retained_corpus_counts=retained_corpus_counts,
    )


def _validate_graph(
    graph: pd.DataFrame,
    directed: pd.DataFrame,
    *,
    n: int,
    top_k: int,
) -> tuple[sparse.csr_matrix, int, np.ndarray, np.ndarray]:
    if len(directed) != n * top_k:
        raise ValueError("directed Jaccard table does not contain top_k rows per recipe")
    per_source = directed.groupby("src", sort=False).size().reindex(range(n), fill_value=0)
    if np.any(per_source.to_numpy() != top_k):
        raise ValueError("directed Jaccard table is not complete by source recipe")
    if np.any(directed["src"].to_numpy() == directed["tgt"].to_numpy()):
        raise ValueError("directed Jaccard table contains a self choice")
    expected_ranks = np.tile(np.arange(1, top_k + 1), n)
    ordered_ranks = directed.sort_values(["src", "rank"])["rank"].to_numpy()
    if not np.array_equal(ordered_ranks, expected_ranks):
        raise ValueError("directed Jaccard ranks are incomplete or duplicated")

    endpoints = graph[["src", "tgt"]].to_numpy(dtype=np.int64)
    if (
        len(graph) == 0
        or np.any(endpoints[:, 0] < 0)
        or np.any(endpoints[:, 1] >= n)
        or np.any(endpoints[:, 0] >= endpoints[:, 1])
    ):
        raise ValueError("Jaccard graph is empty, out of range, or noncanonical")
    similarity = graph["jaccard_similarity"].to_numpy(dtype=float)
    distance = graph["jaccard_distance"].to_numpy(dtype=float)
    if (
        np.any((similarity < 0) | (similarity > 1))
        or not np.allclose(distance, 1.0 - similarity, rtol=0.0, atol=1e-15)
        or not np.all(graph["weight"].to_numpy(dtype=float) == 1.0)
    ):
        raise ValueError("Jaccard similarities, distances, or unit weights are invalid")

    adjacency = sparse.csr_matrix(
        (graph["weight"], (graph["src"], graph["tgt"])),
        shape=(n, n),
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    component_count, component_labels = connected_components(adjacency, directed=False)
    degree = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1)
    return adjacency, int(component_count), component_labels, degree


def prepare(
    base: Path,
    output: Path,
    neighbor_map_path: Path,
    *,
    minimum_count: int = MIN_RAW_INGREDIENT_COUNT,
    top_k: int = TOP_K,
    block_rows: int = BLOCK_ROWS,
) -> dict[str, Any]:
    """Create and hash the standalone v2 data bundle."""

    started = time.perf_counter()
    if top_k < 1 or block_rows < 1:
        raise ValueError("top_k and block_rows must be positive")
    base_manifest_path = base / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    _verify_base_artifacts(base, base_manifest)
    base_counts = sparse.load_npz(base / "counts_csr.npz").tocsr()
    base_recipes = pd.read_csv(base / "recipes.csv.gz")
    base_feature_ids = json.loads((base / "feature_ids.json").read_text())
    selection = select_min8_positive_columns(
        base_counts,
        base_recipes,
        base_feature_ids,
        minimum_count=minimum_count,
    )
    counts = selection.counts
    recipes = selection.recipes
    feature_ids = selection.feature_ids
    lengths = np.asarray(counts.sum(axis=1)).reshape(-1).astype(float)
    with neighbor_map_path.open("rb") as handle:
        neighbor_mapping = pickle.load(handle)

    graph, directed = build_jaccard_graph(
        counts,
        recipes["cuisine"].astype(str).to_numpy(),
        recipes["id"].to_numpy(),
        selection.retained_base_rows,
        neighbor_mapping,
        top_k=top_k,
        block_rows=block_rows,
    )
    _, component_count, component_labels, degree = _validate_graph(
        graph,
        directed,
        n=counts.shape[0],
        top_k=top_k,
    )

    output.mkdir(parents=True, exist_ok=True)
    counts_path = output / "counts_csr.npz"
    recipes_path = output / "recipes.csv.gz"
    features_path = output / "feature_ids.json"
    dropped_features_path = output / "dropped_feature_ids.json"
    feature_provenance_path = output / "feature_provenance.csv.gz"
    edges_path = output / "jaccard_edges.csv.gz"
    directed_path = output / "directed_jaccard_neighbors.csv.gz"

    sparse.save_npz(counts_path, counts, compressed=True)
    recipes.to_csv(
        recipes_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    features_path.write_text(json.dumps(feature_ids, ensure_ascii=False, indent=2) + "\n")
    dropped_feature_ids = [
        str(base_feature_ids[index]) for index in selection.removed_base_columns
    ]
    dropped_features_path.write_text(
        json.dumps(dropped_feature_ids, ensure_ascii=False, indent=2) + "\n"
    )
    feature_provenance = pd.DataFrame(
        {
            "retained_column_index": np.arange(counts.shape[1], dtype=np.int64),
            "base_column_index": selection.retained_base_columns,
            "feature_id": feature_ids,
            "corpus_count": selection.retained_corpus_counts,
        }
    )
    feature_provenance.to_csv(
        feature_provenance_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
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

    endpoints = graph[["src", "tgt"]].to_numpy(dtype=np.int64)
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
        "schema_version": 2,
        "contract_version": "raw_min8_jaccard_v2",
        "dataset": "whats_cooking_raw_min8_jaccard_v2",
        "canonical_scope": (
            "audited cuisine-balanced literal-token raw corpus; retain recipes "
            "with raw ingredient-count total >= 8, then remove ingredient "
            "columns with zero corpus count among the retained recipes"
        ),
        "supersedes_without_modifying": {
            "relative_path": "../raw_min8_jaccard_v1/manifest.json",
            "difference": (
                "v2 removes columns that become all-zero after the >=8 row filter; "
                "v1 and all of its artifacts remain immutable"
            ),
        },
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
            "before_n": int(base_counts.shape[0]),
            "after_n": int(counts.shape[0]),
            "removed_n": int(base_counts.shape[0] - counts.shape[0]),
            "retained_base_rows_sha256": _sha256_numeric(
                selection.retained_base_rows, "<i8"
            ),
            "applied_before": [
                "positive-corpus ingredient column filtering",
                "Jaccard graph construction",
                "count thinning",
                "all fitted denoising or weighting",
            ],
        },
        "column_filter": {
            "rule": "corpus count among retained recipes > 0",
            "applied_after": "raw literal ingredient-count row total >= 8",
            "applied_before": [
                "Jaccard graph construction",
                "count thinning",
                "all fitted denoising or weighting",
            ],
            "before_p": int(base_counts.shape[1]),
            "after_p": int(counts.shape[1]),
            "removed_p": int(selection.removed_base_columns.size),
            "zero_count_before": int(selection.removed_base_columns.size),
            "zero_count_after": 0,
            "retained_base_columns_sha256": _sha256_numeric(
                selection.retained_base_columns, "<i8"
            ),
            "removed_base_columns_sha256": _sha256_numeric(
                selection.removed_base_columns, "<i8"
            ),
            "dropped_feature_ids_sha256": _sha256_strings(dropped_feature_ids),
        },
        "matrix": {
            "shape": list(counts.shape),
            "n": int(counts.shape[0]),
            "p": int(counts.shape[1]),
            "nnz": int(counts.nnz),
            "positive_feature_count": int(counts.shape[1]),
            "zero_feature_count": 0,
            "N_mean": float(lengths.mean()),
            "N_min": float(lengths.min()),
            "N_max": float(lengths.max()),
            "corpus_tokens": int(lengths.sum()),
            "canonical_feature_filtering": (
                "only zero-corpus columns removed after the >=8 row filter"
            ),
            "token_parser": (
                "literal mapped ingredient strings; duplicates retained as counts"
            ),
        },
        "graph": {
            "definition": (
                "standard binary set Jaccard similarity |S_i intersection S_j| / "
                "|S_i union S_j| on the >=8 recipe, positive-corpus-feature "
                "matrix; for each recipe choose the five most similar other "
                "recipes among its own cuisine and the cuisines in the supplied "
                "neighboring-country map; break ties by retained row order "
                "inherited from the frozen raw corpus; union directed choices "
                "and use unit edge weights"
            ),
            "invariance_note": (
                "removing globally zero columns does not change any binary-set "
                "Jaccard intersection, union, similarity, or neighbor ordering"
            ),
            "neighbor_mapping_sha256": _sha256_file(neighbor_map_path),
            "top_k_directed": int(top_k),
            "candidate_scope": "same cuisine plus configured neighboring cuisines",
            "tie_break": "ascending retained row, preserving frozen base-row order",
            "edge_count_undirected": int(len(graph)),
            "component_count": component_count,
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
            "canonical_only": "drop zero-corpus columns after row filtering",
            "fitted_denoising": "selected inside the experiment by cross-validation",
            "count_thinning": "performed only after this canonical contract is loaded",
        },
        "contract_hashes": contract_hashes,
        "artifacts": {},
        "runtime_seconds": time.perf_counter() - started,
    }
    artifacts = (
        counts_path,
        recipes_path,
        features_path,
        dropped_features_path,
        feature_provenance_path,
        edges_path,
        directed_path,
    )
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
