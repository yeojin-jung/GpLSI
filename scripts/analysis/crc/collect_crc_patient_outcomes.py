#!/usr/bin/env python3
"""Collect sharded CRC patient prediction and CCA outputs.

The collection step is intentionally separate from model evaluation so the
Slurm jobs remain independent.  It deduplicates W-level records, applies the
Benjamini--Hochberg correction to CCA permutation tests across all collected
fits within each phenotype universe, and writes dashboard-sized summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONFIG_COLUMNS = [
    "dataset",
    "K",
    "estimator_family",
    "spectral_geometry",
    "vertex_hunter",
    "preprocessing",
    "W_recovery_method",
    "graph_svd_version",
]
METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "brier",
]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(set(paths)):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    observed = values.dropna().astype(float)
    if observed.empty:
        return result
    order = observed.sort_values(kind="stable").index
    ranked = observed.loc[order].to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def _deduplicate(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing deduplication columns {sorted(missing)}")
    duplicate = frame.duplicated(columns, keep=False)
    if duplicate.any():
        comparison_columns = [
            column
            for column in frame.columns
            if column not in {"evaluated_at_utc", "source_artifact_path"}
        ]
        for _, group in frame.loc[duplicate].groupby(columns, dropna=False):
            if len(group[comparison_columns].drop_duplicates()) > 1:
                raise ValueError(f"conflicting duplicate rows in {name}")
        frame = frame.drop_duplicates(columns, keep="last")
    return frame.reset_index(drop=True)


def _outcome_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes
    group_columns = [
        *CONFIG_COLUMNS,
        "summary",
        "target",
        "classifier",
    ]
    rows: list[dict[str, object]] = []
    for key, group in outcomes.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_complete": int(len(group)),
                "patient_count": int(group["patient_count"].iloc[0]),
                "positive_count": int(group["positive_count"].iloc[0]),
                "region_count": int(group["region_count"].iloc[0]),
                "cell_count": int(group["cell_count"].iloc[0]),
                "cell_universe": str(group["cell_universe"].iloc[0]),
                "retained_cell_count": int(group["retained_cell_count"].iloc[0]),
                "predictor_dimension": int(group["predictor_dimension"].iloc[0]),
                "majority_accuracy": float(group["majority_accuracy"].iloc[0]),
                "evaluation_version": str(group["evaluation_version"].iloc[0]),
                "summary_definition": str(group["summary_definition"].iloc[0]),
                "classifier_definition": str(group["classifier_definition"].iloc[0]),
            }
        )
        for column in (
            "composition_dimension",
            "cell_type_features",
            "source_cell_count",
            "source_region_count",
            "aggregation_scope",
            "source_artifact_path",
        ):
            if column in group:
                values = group[column].dropna().unique()
                if len(values) == 1:
                    row[column] = values[0]
        for metric in METRICS:
            values = pd.to_numeric(group[f"{metric}_mean"], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_between_seed_std"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            )
            within = pd.to_numeric(group[f"{metric}_std"], errors="coerce")
            row[f"{metric}_mean_cv_std"] = float(within.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def collect(input_root: Path, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()

    def inputs(filename: str) -> list[Path]:
        return [
            path
            for path in input_root.rglob(filename)
            if output_resolved not in path.resolve().parents
        ]

    outcomes = _read_many(inputs("outcome_rows.csv"))
    outcomes = _deduplicate(
        outcomes,
        ["task_config_hash", "W_fit_id", "summary", "target", "classifier"],
        "outcome rows",
    )
    if not outcomes.empty:
        _atomic_csv(outcomes, output_dir / "outcome_rows.csv")
        _atomic_csv(_outcome_summary(outcomes), output_dir / "outcome_summary.csv")

    cca_summary = _read_many(inputs("cca_summary.csv"))
    cca_summary = _deduplicate(
        cca_summary,
        ["task_config_hash", "W_fit_id", "phenotype_cell_universe"],
        "CCA summaries",
    )
    if not cca_summary.empty:
        cca_summary["permutation_q_value"] = np.nan
        for _, indices in cca_summary.groupby(
            "phenotype_cell_universe", dropna=False
        ).groups.items():
            subset = cca_summary.loc[indices, "permutation_p_value"]
            cca_summary.loc[indices, "permutation_q_value"] = _bh_adjust(subset)
        _atomic_csv(cca_summary, output_dir / "cca_summary.csv")

    components = _read_many(inputs("cca_components.csv"))
    components = _deduplicate(
        components,
        [
            "task_config_hash",
            "W_fit_id",
            "phenotype_cell_universe",
            "component",
        ],
        "CCA components",
    )
    loadings = _read_many(inputs("cca_loadings.csv"))
    loadings = _deduplicate(
        loadings,
        [
            "task_config_hash",
            "W_fit_id",
            "phenotype_cell_universe",
            "component",
            "side",
            "part_index",
        ],
        "CCA loadings",
    )
    if not cca_summary.empty:
        q_columns = [
            "task_config_hash",
            "W_fit_id",
            "phenotype_cell_universe",
            "permutation_p_value",
            "permutation_q_value",
        ]
        if not components.empty:
            components = components.merge(
                cca_summary[q_columns],
                on=["task_config_hash", "W_fit_id", "phenotype_cell_universe"],
                how="left",
                validate="many_to_one",
            )
        if not loadings.empty:
            loadings = loadings.merge(
                cca_summary[q_columns],
                on=["task_config_hash", "W_fit_id", "phenotype_cell_universe"],
                how="left",
                validate="many_to_one",
            )
    if not components.empty:
        _atomic_csv(components, output_dir / "cca_components.csv")
    if not loadings.empty:
        _atomic_csv(loadings, output_dir / "cca_loadings.csv")

    contracts = inputs("patient_contract.csv")
    prediction_splits = inputs("prediction_cv_splits.csv")
    cca_splits = inputs("cca_cv_splits.csv")
    phenotype_features = inputs("cca_phenotype_patient_features.csv")
    for paths, filename in [
        (contracts, "patient_contract.csv"),
        (prediction_splits, "prediction_cv_splits.csv"),
        (cca_splits, "cca_cv_splits.csv"),
        (phenotype_features, "cca_phenotype_patient_features.csv"),
    ]:
        if paths:
            first = pd.read_csv(paths[0], low_memory=False)
            for path in paths[1:]:
                candidate = pd.read_csv(path, low_memory=False)
                if not first.equals(candidate):
                    raise ValueError(f"shards disagree on shared {filename}")
            _atomic_csv(first, output_dir / filename)

    print(
        f"collected outcomes={len(outcomes)}, CCA summaries={len(cca_summary)}, "
        f"components={len(components)}, loadings={len(loadings)} into {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
