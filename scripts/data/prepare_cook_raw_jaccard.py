#!/usr/bin/env python3
"""Prepare the full pre-threshold What's Cooking corpus and Jaccard graph.

The modeling artifact uses literal mapped ingredient tokens, rather than the
historical ``CountVectorizer`` comma-tokenizer path that silently lowercases or
splits some ingredient names.  A separate, read-only reconstruction of that
historical path is included in the audit manifest and figure because it is the
only supplied-data pipeline that reproduces the manuscript's 1,716 retained
columns and 13,887 retained recipes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Iterable
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress
from sklearn.feature_extraction.text import CountVectorizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "cook" / "dataset"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "raw_jaccard_v1"
DEFAULT_FIGURE = (
    REPO_ROOT
    / "figures"
    / "cook"
    / "prethreshold_frequencies"
)
TRAN_ALPHA = 0.1
SAMPLE_CAP = 2_000
SAMPLE_SEED = 1
TOP_K = 5
GRAPH_BLOCK_ROWS = 128


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _sha256_strings(values: Iterable[object]) -> str:
    encoded = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_numeric(value: np.ndarray, dtype: str) -> str:
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
    block_rows: int = 256,
) -> str:
    """Hash the logical dense row-major matrix without materializing it whole."""

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


def _sample_cuisine(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) > SAMPLE_CAP:
        return group.sample(SAMPLE_CAP, random_state=SAMPLE_SEED)
    return group


def _load_balanced_mapped(
    source: Path,
) -> tuple[pd.DataFrame, list[list[str]], dict[str, list[str]]]:
    raw = pd.read_json(source / "train.json")
    with (source / "ingredient_mapping.pkl").open("rb") as handle:
        ingredient_mapping = pickle.load(handle)
    with (source / "neighbor_countries_mapping.pkl").open("rb") as handle:
        neighbor_mapping = pickle.load(handle)
    reverse = {
        raw_name: canonical
        for canonical, raw_names in ingredient_mapping.items()
        for raw_name in raw_names
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sampled = (
            raw.groupby("cuisine", group_keys=False)
            .apply(_sample_cuisine)
            .reset_index(drop=True)
        )
    mapped = [
        [reverse.get(ingredient, ingredient) for ingredient in ingredients]
        for ingredients in sampled["ingredients"]
    ]
    return sampled[["id", "cuisine"]].copy(), mapped, neighbor_mapping


def _literal_count_matrix(
    mapped: list[list[str]],
) -> tuple[sparse.csr_matrix, list[str]]:
    vocabulary = list(
        dict.fromkeys(ingredient for ingredients in mapped for ingredient in ingredients)
    )
    index = {ingredient: position for position, ingredient in enumerate(vocabulary)}
    indptr = np.zeros(len(mapped) + 1, dtype=np.int64)
    indices: list[int] = []
    values: list[int] = []
    for row, ingredients in enumerate(mapped):
        local = Counter(index[ingredient] for ingredient in ingredients)
        for column in sorted(local):
            indices.append(column)
            values.append(local[column])
        indptr[row + 1] = len(indices)
    counts = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.int64),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(mapped), len(vocabulary)),
    )
    counts.sort_indices()
    return counts, vocabulary


def _standard_jaccard_graph(
    counts: sparse.csr_matrix,
    cuisines: np.ndarray,
    neighbor_mapping: dict[str, list[str]],
    *,
    top_k: int = TOP_K,
    block_rows: int = GRAPH_BLOCK_ROWS,
) -> pd.DataFrame:
    """Return a symmetric, unweighted top-k graph chosen by set Jaccard."""

    presence = counts.copy().astype(np.int8)
    presence.data.fill(1)
    cardinality = np.diff(presence.indptr).astype(float)
    cuisine_rows = {
        cuisine: np.flatnonzero(cuisines == cuisine)
        for cuisine in sorted(set(map(str, cuisines)))
    }
    selected: dict[tuple[int, int], float] = {}
    for cuisine, query_indices in cuisine_rows.items():
        eligible_cuisines = {cuisine, *map(str, neighbor_mapping.get(cuisine, []))}
        candidate_indices = np.sort(
            np.concatenate(
                [
                    cuisine_rows[value]
                    for value in sorted(eligible_cuisines)
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
                # Primary key: descending similarity. Tie key: ascending global row.
                order = np.lexsort((candidate_indices, -scores))
                for position in order[:top_k]:
                    target = int(candidate_indices[position])
                    edge = (min(int(source), target), max(int(source), target))
                    selected[edge] = max(selected.get(edge, 0.0), float(scores[position]))
    records = [
        {
            "src": source,
            "tgt": target,
            "weight": 1.0,
            "jaccard_similarity": selected[(source, target)],
            "jaccard_distance": 1.0 - selected[(source, target)],
        }
        for source, target in sorted(selected)
    ]
    return pd.DataFrame.from_records(records)


def _legacy_manuscript_reproduction(
    mapped: list[list[str]], vocabulary: list[str]
) -> dict[str, object]:
    """Reproduce the quoted 5,346 -> 1,716 -> 13,887 sequence exactly."""

    strings = [",".join(ingredients) for ingredients in mapped]
    vectorizer = CountVectorizer(
        tokenizer=lambda value: value.split(","), vocabulary=vocabulary
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = vectorizer.fit_transform(strings).tocsr()
    row_totals = np.asarray(legacy.sum(axis=1)).reshape(-1)
    mean_counts = np.asarray(legacy.mean(axis=0)).reshape(-1)
    maximum_length = float(row_totals.max())
    threshold = float(
        TRAN_ALPHA
        * np.sqrt(
            np.log(max(legacy.shape))
            / (legacy.shape[0] * maximum_length)
        )
    )
    retained = mean_counts > threshold
    retained_row_totals = np.asarray(legacy[:, retained].sum(axis=1)).reshape(-1)
    rows_after = retained_row_totals >= 10
    final = legacy[rows_after][:, retained]
    positive_after = int(np.count_nonzero(np.asarray(final.sum(axis=0)).reshape(-1)))
    return {
        "matrix": legacy,
        "mean_counts": mean_counts,
        "threshold": threshold,
        "nominal_prethreshold_columns": int(legacy.shape[1]),
        "positive_prethreshold_columns": int(np.count_nonzero(mean_counts)),
        "silent_zero_prethreshold_columns": int(np.count_nonzero(mean_counts == 0)),
        "retained_columns": int(retained.sum()),
        "retained_rows": int(rows_after.sum()),
        "positive_columns_after_row_filter": positive_after,
        "silent_zero_columns_after_row_filter": int(retained.sum()) - positive_after,
        "mean_length_after": float(np.asarray(final.sum(axis=1)).mean()),
        "N_used": maximum_length,
        "row_rule": "retain rows with at least 10 counts after feature thresholding",
        "parser": (
            "historical CountVectorizer comma tokenizer with default lowercase=True; "
            "retained only to reproduce manuscript dimensions"
        ),
    }


def _frequency_table(
    counts: sparse.csr_matrix,
    vocabulary: list[str],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    lengths = np.asarray(counts.sum(axis=1)).reshape(-1).astype(float)
    frequencies = counts.multiply((1.0 / lengths)[:, None]).tocsr()
    eta = np.asarray(frequencies.mean(axis=0)).reshape(-1)
    mean_counts = np.asarray(counts.mean(axis=0)).reshape(-1)
    threshold = float(
        TRAN_ALPHA
        * np.sqrt(
            np.log(max(counts.shape)) / (counts.shape[0] * float(lengths.mean()))
        )
    )
    order = np.argsort(-eta, kind="stable")
    rank = np.arange(1, counts.shape[1] + 1, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "rank": rank,
            "feature": np.asarray(vocabulary, dtype=object)[order],
            "document_balanced_frequency": eta[order],
            "mean_count_per_recipe": mean_counts[order],
            "corpus_count": np.asarray(counts.sum(axis=0)).reshape(-1)[order],
            "tran_alpha": TRAN_ALPHA,
            "tran_cutoff_document_balanced": threshold,
            "tran_retained_document_balanced": eta[order] > threshold,
        }
    )
    fit = linregress(np.log10(rank), np.log10(eta[order]))
    summary: dict[str, float | int] = {
        "n": int(counts.shape[0]),
        "p": int(counts.shape[1]),
        "N_mean": float(lengths.mean()),
        "N_min": float(lengths.min()),
        "N_max": float(lengths.max()),
        "corpus_tokens": int(lengths.sum()),
        "tran_alpha": TRAN_ALPHA,
        "tran_cutoff_document_balanced": threshold,
        "tran_retained_document_balanced": int(np.sum(eta > threshold)),
        "log_log_slope": float(fit.slope),
        "log_log_intercept": float(fit.intercept),
        "log_log_r_squared": float(fit.rvalue**2),
    }
    return frame, summary


def _save_frequency_figure(
    raw: pd.DataFrame,
    raw_summary: dict[str, float | int],
    legacy: dict[str, object],
    output_stem: Path,
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    rank = raw["rank"].to_numpy(dtype=float)
    eta = raw["document_balanced_frequency"].to_numpy(dtype=float)
    slope = float(raw_summary["log_log_slope"])
    intercept = float(raw_summary["log_log_intercept"])
    fitted = np.power(10.0, intercept + slope * np.log10(rank))
    cutoff = float(raw_summary["tran_cutoff_document_balanced"])
    axes[0].plot(rank, eta, color="#0072B2", linewidth=1.6, label="Raw empirical frequency")
    axes[0].plot(rank, fitted, color="#D55E00", linestyle="--", linewidth=2.0,
                 label="Log–log OLS fit")
    axes[0].axhline(cutoff, color="#7A1F5C", linestyle=":", linewidth=2.0,
                    label=rf"Tran cutoff ($\alpha={TRAN_ALPHA:g}$)")
    axes[0].set(xscale="log", yscale="log", xlabel="Ingredient rank (log scale)",
                ylabel=r"$\hat\eta_j=n^{-1}\sum_i D_{ij}/N_i$ (log scale)",
                title="(A) Literal mapped-token corpus")
    axes[0].grid(True, which="both", alpha=0.18)
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].text(
        0.97,
        0.96,
        rf"$n={int(raw_summary['n']):,}$, $p={int(raw_summary['p']):,}$"
        "\n"
        rf"$N={float(raw_summary['N_mean']):.3f}$; retained "
        rf"{int(raw_summary['tran_retained_document_balanced']):,}/{int(raw_summary['p']):,}"
        "\n"
        rf"$\beta={slope:.3f}$, $R^2={float(raw_summary['log_log_r_squared']):.3f}$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92},
    )

    legacy_means = np.asarray(legacy["mean_counts"], dtype=float)
    positive = legacy_means > 0
    legacy_ranked = np.sort(legacy_means[positive])[::-1]
    legacy_rank = np.arange(1, legacy_ranked.size + 1)
    legacy_cutoff = float(legacy["threshold"])
    axes[1].plot(
        legacy_rank,
        legacy_ranked,
        color="#0072B2",
        linewidth=1.6,
        label="Historical pre-threshold column mean",
    )
    axes[1].axhline(
        legacy_cutoff,
        color="#7A1F5C",
        linestyle=":",
        linewidth=2.0,
        label=rf"Historical cutoff ($\alpha={TRAN_ALPHA:g}$)",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Positive nominal-column rank (log scale)",
        ylabel=r"$n^{-1}\sum_i D_{ij}$ (log scale)",
        title="(B) Exact manuscript-dimension reconstruction",
    )
    axes[1].grid(True, which="both", alpha=0.18)
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].text(
        0.97,
        0.96,
        rf"5,346 nominal columns; {int(legacy['silent_zero_prethreshold_columns']):,} parser-zero"
        "\n"
        rf"cutoff retains {int(legacy['retained_columns']):,}; row rule leaves "
        rf"{int(legacy['retained_rows']):,} recipes"
        "\n"
        rf"final positive columns: {int(legacy['positive_columns_after_row_filter']):,}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92},
    )

    fig.suptitle(
        "What's Cooking ingredient frequencies before Tran thresholding\n"
        "Full cuisine-balanced source; corrected literal-token analysis and exact historical reconstruction",
        fontsize=14,
    )
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def prepare(source: Path, output: Path, figure_stem: Path) -> dict[str, object]:
    started = time.perf_counter()
    rows, mapped, neighbor_mapping = _load_balanced_mapped(source)
    counts, vocabulary = _literal_count_matrix(mapped)
    if counts.shape != (24_856, 5_346):
        raise ValueError(f"unexpected full raw shape {counts.shape}; expected (24856, 5346)")
    if np.any(np.asarray(counts.sum(axis=0)).reshape(-1) <= 0):
        raise ValueError("literal-token raw matrix unexpectedly contains a zero column")
    legacy = _legacy_manuscript_reproduction(mapped, vocabulary)
    if (legacy["retained_columns"], legacy["retained_rows"]) != (1_716, 13_887):
        raise ValueError(
            "historical reconstruction no longer matches manuscript dimensions: "
            f"{legacy['retained_columns']} columns, {legacy['retained_rows']} rows"
        )

    graph = _standard_jaccard_graph(
        counts,
        rows["cuisine"].astype(str).to_numpy(),
        neighbor_mapping,
    )
    adjacency = sparse.csr_matrix(
        (graph["weight"], (graph["src"], graph["tgt"])), shape=(counts.shape[0],) * 2
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    components, labels = connected_components(adjacency, directed=False)
    degree = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1)

    output.mkdir(parents=True, exist_ok=True)
    counts_path = output / "counts_csr.npz"
    rows_path = output / "recipes.csv.gz"
    features_path = output / "feature_ids.json"
    edges_path = output / "jaccard_edges.csv.gz"
    frequencies_path = output / "prethreshold_rank_frequencies.csv"
    sparse.save_npz(counts_path, counts, compressed=True)
    rows.to_csv(
        rows_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    features_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n"
    )
    graph.to_csv(
        edges_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )

    frequency_frame, frequency_summary = _frequency_table(counts, vocabulary)
    frequency_frame.to_csv(frequencies_path, index=False, float_format="%.12g")
    figure_paths = _save_frequency_figure(
        frequency_frame, frequency_summary, legacy, figure_stem
    )

    lengths = np.asarray(counts.sum(axis=1)).reshape(-1).astype(float)
    endpoints = graph[["src", "tgt"]].to_numpy(dtype=np.int64)
    weights = graph["weight"].to_numpy(dtype=float)
    contract_hashes = {
        "D_sha256": _sha256_dense_view(counts, "<i8"),
        "X_sha256": _sha256_dense_view(counts, "<f8", row_divisors=lengths),
        "N_i_sha256": _sha256_numeric(lengths, "<f8"),
        "feature_ids_sha256": _sha256_strings(vocabulary),
        "observation_ids_sha256": _sha256_strings(rows["id"]),
        "group_ids_sha256": _sha256_strings(rows["cuisine"]),
        "edges_sha256": _sha256_numeric(endpoints, "<i8"),
        "edge_weights_sha256": _sha256_numeric(weights, "<f8"),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "whats_cooking_raw_jaccard",
        "canonical_scope": (
            "full cuisine-balanced, ingredient-mapped, pre-feature-threshold corpus; "
            "literal mapped tokens"
        ),
        "source": {
            "train_json": _portable_path(source / "train.json"),
            "train_json_sha256": _sha256_file(source / "train.json"),
            "ingredient_mapping_sha256": _sha256_file(source / "ingredient_mapping.pkl"),
            "neighbor_mapping_sha256": _sha256_file(
                source / "neighbor_countries_mapping.pkl"
            ),
            "original_recipe_count": 39_774,
            "cuisine_count": 20,
        },
        "sampling": {
            "rule": "cap each cuisine at 2,000 recipes",
            "seed": SAMPLE_SEED,
            "recipe_count": int(counts.shape[0]),
        },
        "matrix": {
            **frequency_summary,
            "shape": list(counts.shape),
            "nnz": int(counts.nnz),
            "token_parser": "literal mapped ingredient strings; duplicates retained as counts",
            "feature_thresholding": "none",
            "row_filtering_after_cuisine_balance": "none",
        },
        "graph": {
            "definition": (
                "standard binary set Jaccard similarity |S_i intersection S_j| / "
                "|S_i union S_j|; choose five smallest Jaccard distances among "
                "same-cuisine plus configured neighboring-cuisine candidates; "
                "symmetrize and use unit edge weights"
            ),
            "top_k_directed": TOP_K,
            "edge_count_undirected": int(len(graph)),
            "component_count": int(components),
            "component_sizes": np.bincount(labels).astype(int).tolist(),
            "degree_min": int(degree.min()),
            "degree_median": float(np.median(degree)),
            "degree_max": int(degree.max()),
            "distance_min": float(graph["jaccard_distance"].min()),
            "distance_median": float(graph["jaccard_distance"].median()),
            "distance_max": float(graph["jaccard_distance"].max()),
            "weights": "unweighted unit edges after neighbor selection",
        },
        "manuscript_dimension_reproduction": {
            key: value for key, value in legacy.items() if key not in {"matrix", "mean_counts"}
        },
        "contract_hashes": contract_hashes,
        "artifacts": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in [counts_path, rows_path, features_path, edges_path, frequencies_path]
        },
        "figures": [_portable_path(path) for path in figure_paths],
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.figure_stem)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
