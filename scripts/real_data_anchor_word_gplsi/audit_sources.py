#!/usr/bin/env python3
"""Create immutable manifests and descriptive checks for the three real datasets.

This is an audit utility, not an experiment runner.  It reads the source data,
reconstructs the published canonical preprocessing where possible, and writes
only hashes and summaries.  It never writes a processed data cache.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GPLSI_DATA_ROOT", str(REPO_ROOT / "data")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_numeric(value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_strings(value: Any) -> str:
    encoded = json.dumps(
        [str(item) for item in value], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def quantiles(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=float)
    points = np.quantile(array, [0, 0.25, 0.5, 0.75, 1])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, points)))


def preprocessing_summary(D: np.ndarray, alpha: float = 0.005) -> dict[str, Any]:
    lengths = D.sum(axis=1).astype(float)
    X = D / lengths[:, None]
    n, p = X.shape
    N = float(lengths.mean())
    eta = X.mean(axis=0)
    threshold = float(alpha * np.sqrt(np.log(max(n, p)) / (n * N)))
    selected = np.flatnonzero(eta > threshold)
    fallback = False
    if selected.size < 0.1 * p:
        selected = np.argsort(-eta, kind="stable")[: int(np.ceil(0.1 * p))]
        fallback = True
    positive = np.flatnonzero(eta > 0)
    weights = eta[positive] ** -0.5
    return {
        "n": int(n),
        "p": int(p),
        "N_definition": "mean of per-document canonical count totals",
        "N_mean": N,
        "N_i": quantiles(lengths),
        "P0_positive_feature_count": int(positive.size),
        "P1_alpha": alpha,
        "P1_threshold": threshold,
        "P1_strict_greater_retained": int(selected.size),
        "P1_top_10_percent_fallback": fallback,
        "P2_weight_definition": "eta_hat_j**(-1/2); no cap/floor/rescaling",
        "P2_weight_quantiles": quantiles(weights),
        "D_sha256": sha256_numeric(D, "<i8"),
        "X_sha256": sha256_numeric(X, "<f8"),
        "N_i_sha256": sha256_numeric(lengths, "<f8"),
    }


def audit_crc() -> dict[str, Any]:
    root = DATA_ROOT / "stanford-crc"
    source = root / "output" / "output_3hop"
    meta = pd.read_csv(root / "charville_labels.csv")
    selected_meta = meta.loc[meta["primary_outcome"].notna()].copy()
    all_counts: list[np.ndarray] = []
    observation_ids: list[str] = []
    region_ids: list[str] = []
    all_coords: list[np.ndarray] = []
    corrected_edges: list[np.ndarray] = []
    historical_ids: list[int] = []
    corrected_offset = 0
    historical_offset = 0

    for region in selected_meta["region_id"].astype(str):
        prefix = source / region
        D_frame = pd.read_csv(f"{prefix}.D.csv", index_col=0)
        raw_cell_ids = np.asarray(
            [int(ast.literal_eval(item)[1]) for item in D_frame.index], dtype=int
        )
        keep = D_frame.sum(axis=1).to_numpy() >= 10
        retained_cells = raw_cell_ids[keep]
        retained_D = D_frame.loc[keep].to_numpy(dtype=np.int64)

        coord = pd.read_csv(f"{prefix}.coord.csv", index_col=0).set_index("CELL_ID")
        ordered_coord = coord.loc[retained_cells, ["X", "Y"]].to_numpy(dtype=float)
        edge = pd.read_csv(f"{prefix}.edge.csv", index_col=0)
        retained_set = set(map(int, retained_cells))
        edge = edge.loc[
            edge["src"].isin(retained_set) & edge["tgt"].isin(retained_set)
        ]
        corrected_map = {
            int(cell): corrected_offset + index
            for index, cell in enumerate(retained_cells)
        }
        corrected_edges.append(
            np.column_stack(
                (
                    edge["src"].map(corrected_map).to_numpy(dtype=np.int64),
                    edge["tgt"].map(corrected_map).to_numpy(dtype=np.int64),
                )
            )
        )

        raw_to_historical = {
            int(cell): historical_offset + index
            for index, cell in enumerate(raw_cell_ids)
        }
        historical_ids.extend(raw_to_historical[int(cell)] for cell in retained_cells)
        historical_offset += retained_D.shape[0]
        corrected_offset += retained_D.shape[0]
        all_counts.append(retained_D)
        all_coords.append(ordered_coord)
        observation_ids.extend(f"{region}::{cell}" for cell in retained_cells)
        region_ids.extend([region] * retained_D.shape[0])

    D = np.vstack(all_counts)
    coords = np.vstack(all_coords)
    edges = np.vstack(corrected_edges)
    historical_ids_array = np.asarray(historical_ids, dtype=np.int64)
    summary = preprocessing_summary(D)
    summary.update(
        {
            "dataset": "stanford_crc_codex",
            "canonical_scope": "196 regions with nonmissing primary_outcome; rows with count total >=10",
            "source_region_count": int(meta.shape[0]),
            "selected_region_count": int(selected_meta.shape[0]),
            "feature_ids": list(D_frame.columns),
            "observation_ids_sha256": sha256_strings(observation_ids),
            "group_ids_definition": "region_id; no patient identifier is supplied",
            "group_ids_sha256": sha256_strings(region_ids),
            "coordinates_sha256": sha256_numeric(coords, "<f8"),
            "corrected_edges_sha256": sha256_numeric(edges, "<i8"),
            "corrected_edge_count": int(edges.shape[0]),
            "historical_index_unique_count": int(np.unique(historical_ids_array).size),
            "historical_index_duplicate_count": int(
                historical_ids_array.size - np.unique(historical_ids_array).size
            ),
            "historical_index_max": int(historical_ids_array.max()),
            "historical_indices_out_of_bounds": int(
                np.count_nonzero(historical_ids_array >= D.shape[0])
            ),
            "canonical_repair": "assign contiguous global row ids after the >=10 filter and remap each retained edge through that row map",
            "outcomes_available": [
                column
                for column in ("primary_outcome", "recurrence", "alive", "OS", "RFS")
                if column in meta.columns
            ],
        }
    )
    return summary


def audit_spleen() -> dict[str, Any]:
    root = DATA_ROOT / "spleen" / "dataset"
    D_all = pd.read_pickle(root / "merged_D.pkl")
    coords_all = pd.read_pickle(root / "merged_coord.pkl")
    edges_all = pd.read_pickle(root / "merged_data.pkl")
    groups: dict[str, Any] = {}
    for tumor in D_all.index.get_level_values(0).unique():
        D = np.asarray(D_all.loc[tumor], dtype=np.int64)
        coords = np.asarray(coords_all.loc[tumor], dtype=float)
        edges_frame = edges_all.loc[tumor]
        edges = edges_frame[["src", "dst"]].to_numpy(dtype=np.int64)
        item = preprocessing_summary(D)
        item.update(
            {
                "coordinates_sha256": sha256_numeric(coords, "<f8"),
                "edges_sha256": sha256_numeric(edges, "<i8"),
                "edge_count": int(edges.shape[0]),
                "observation_ids_sha256": sha256_strings(D_all.loc[tumor].index),
            }
        )
        groups[str(tumor)] = item
    pointer = (root / "spleen_cells_features.pkl").read_bytes()
    return {
        "dataset": "mouse_spleen_codex",
        "canonical_scope": "three BALB/c spleens; published cached results cover BALBc-1",
        "feature_ids": list(map(str, D_all.columns)),
        "group_ids": list(groups),
        "groups": groups,
        "spleen_cells_features_is_git_lfs_pointer": pointer.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ),
        "spleen_cells_features_pointer_text": pointer.decode("utf-8", errors="replace"),
    }


def _sample_cuisine(group: pd.DataFrame) -> pd.DataFrame:
    return group.sample(2000, random_state=1) if len(group) > 2000 else group


def audit_cook() -> dict[str, Any]:
    root = DATA_ROOT / "whats-cooking" / "dataset"
    raw = pd.read_json(root / "train.json")
    with (root / "ingredient_mapping.pkl").open("rb") as handle:
        ingredient_mapping = pickle.load(handle)
    reverse_mapping = {
        value: key for key, values in ingredient_mapping.items() for value in values
    }
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sampled = (
            raw.groupby("cuisine", group_keys=False)
            .apply(_sample_cuisine)
            .reset_index(drop=True)
        )
        sampled["ingredients"] = sampled["ingredients"].apply(
            lambda values: [reverse_mapping.get(value, value) for value in values]
        )
        ingredient_strings = sampled["ingredients"].apply(lambda x: ",".join(x))
        counts = Counter(
            ingredient for values in sampled["ingredients"] for ingredient in values
        )
        vocabulary = [ingredient for ingredient, count in counts.items() if count >= 10]
        vectorizer = CountVectorizer(
            tokenizer=lambda value: value.split(","), vocabulary=vocabulary
        )
        matrix = vectorizer.fit_transform(ingredient_strings)
        frame = pd.DataFrame(
            matrix.toarray(), columns=vectorizer.get_feature_names_out()
        )
    row_keep = frame.sum(axis=1).to_numpy() >= 10
    frame = frame.loc[row_keep]
    sampled = sampled.loc[row_keep]
    column_keep = frame.sum(axis=0).to_numpy() >= 10
    frame = frame.loc[:, column_keep].reset_index(drop=True)
    sampled = sampled.reset_index(drop=True)
    D = frame.to_numpy(dtype=np.int64)
    edge = pd.read_pickle(root / "processed_edge_df.pkl")
    summary = preprocessing_summary(D)
    summary.update(
        {
            "dataset": "whats_cooking",
            "canonical_scope": "historical source: cap each cuisine at 2000, map ingredients, corpus count >=10, row total >=10, then corpus count >=10 again",
            "raw_recipe_count": int(raw.shape[0]),
            "cuisine_count": int(raw["cuisine"].nunique()),
            "balanced_recipe_count": int(sum(min(size, 2000) for size in raw.groupby("cuisine").size())),
            "pre_refilter_vocabulary_count": int(len(vocabulary)),
            "feature_ids": list(map(str, frame.columns)),
            "observation_ids_sha256": sha256_strings(sampled["id"]),
            "group_ids_definition": "cuisine",
            "group_ids_sha256": sha256_strings(sampled["cuisine"]),
            "edge_count": int(edge.shape[0]),
            "edges_sha256": sha256_numeric(edge[["src", "tgt"]], "<i8"),
            "edge_weights_sha256": sha256_numeric(edge["weight"], "<f8"),
            "graph_definition": "historical cached unweighted top-five Jaccard graph restricted by cuisine-neighbor table",
            "coordinates": None,
            "preprocessing_warnings": [str(item.message) for item in captured],
        }
    )
    return summary


def raw_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roots = {
        "stanford_crc_codex": DATA_ROOT / "stanford-crc",
        "mouse_spleen_codex": DATA_ROOT / "spleen",
        "whats_cooking": DATA_ROOT / "whats-cooking",
    }
    for dataset, root in roots.items():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append(
                {
                    "dataset": dataset,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "real_data_anchor_word_gplsi" / "audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = raw_manifest()
    manifest.to_csv(args.output_dir / "source_data_file_manifest.csv", index=False)
    summary = {
        "schema_version": 1,
        "audit_only": True,
        "canonical_hash_contract": {
            "numeric": "sha256(JSON compact shape || NumPy dtype string || C-contiguous bytes)",
            "strings": "sha256(UTF-8 compact JSON array)",
            "experimental_preprocessing_excluded": True,
        },
        "datasets": {
            "stanford_crc_codex": audit_crc(),
            "mouse_spleen_codex": audit_spleen(),
            "whats_cooking": audit_cook(),
        },
    }
    (args.output_dir / "canonical_data_contract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"files_hashed": len(manifest), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
