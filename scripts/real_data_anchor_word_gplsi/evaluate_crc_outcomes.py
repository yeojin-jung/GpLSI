#!/usr/bin/env python3
"""Post-process fitted CRC topic weights into region-level outcome metrics.

This evaluator is deliberately separate from the unsupervised fit.  It never
passes CRC outcomes into topic estimation or hyperparameter selection.  Each
saved ``W_hat`` is aggregated within region and evaluated on shared,
deterministic region-level splits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gplsi.real_data import load_real_data  # noqa: E402


CONFIG_COLUMNS = [
    "dataset",
    "K",
    "estimator_family",
    "spectral_geometry",
    "vertex_hunter",
    "preprocessing",
    "A_recovery",
    "W_recovery_method",
]
TARGETS = ("primary_outcome", "recurrence")
REPRESENTATIONS = ("soft_mean", "published_hard")
FEATURE_TRANSFORMS = ("ilr", "log")
CLASSIFIERS = ("ridge_logistic", "random_forest")
LOG_FLOOR = 1e-8
RANDOM_FOREST_ESTIMATORS = 100
EVALUATION_VERSIONS = {
    ("ilr", "ridge_logistic"): "region_rskf5x5_ilr_ridge_v1",
    ("log", "ridge_logistic"): "region_rskf5x5_log_floor1e-8_ridge_v1",
    ("ilr", "random_forest"): "region_rskf5x5_ilr_rf100_v1",
    ("log", "random_forest"): "region_rskf5x5_log_floor1e-8_rf100_v1",
}
CELL_TYPE_BASELINE_FAMILY = "cell_type_baseline"
CELL_TYPE_BASELINE_K_VALUES = (1, 2, 3, 4, 5, 6)


def _artifact_path(raw_path: str, artifact_root: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    if artifact_root is not None:
        marker = "results/real_data_anchor_word_gplsi/"
        if marker in raw_path:
            candidate = artifact_root / marker / raw_path.split(marker, 1)[1]
            if candidate.is_file():
                return candidate
        candidate = artifact_root / raw_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(raw_path)


def _region_contract(
    bundle: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if bundle is None:
        bundle = load_real_data("crc")
    group_values = bundle.group_ids.astype(str)
    regions, inverse = np.unique(group_values, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(regions)).astype(float)
    if bundle.outcomes is None:
        raise RuntimeError("CRC bundle has no outcome table")
    outcomes = bundle.outcomes.copy()
    outcomes["__region"] = group_values
    labels: dict[str, np.ndarray] = {}
    for target in TARGETS:
        grouped = outcomes.groupby("__region", sort=True)[target]
        inconsistent = grouped.nunique(dropna=False) > 1
        if inconsistent.any():
            raise RuntimeError(f"{target} is not constant within CRC region")
        labels[target] = grouped.first().reindex(regions).to_numpy(dtype=float)
    return inverse, counts, labels


def _cell_type_composition(
    bundle: Any,
    inverse: np.ndarray,
    region_counts: np.ndarray,
    representation: str,
) -> np.ndarray:
    """Treat the observed cell-type features as fixed topics.

    The canonical CRC rows contain an eight-part local cell-type composition.
    Setting ``A = I`` and ``W = X`` therefore gives a non-learned control with
    exactly the same region aggregation and classifier as every fitted topic
    model.  The hard representation uses each row's dominant observed cell
    type, matching the argmax representation used for fitted topics.
    """

    frequencies = np.asarray(bundle.frequencies, dtype=float)
    if frequencies.ndim != 2 or frequencies.shape[1] != len(bundle.feature_ids):
        raise ValueError("CRC cell-type feature matrix is not row/feature aligned")
    return _aggregate_W(frequencies, inverse, region_counts, representation)


def _aggregate_W(
    W: np.ndarray,
    inverse: np.ndarray,
    region_counts: np.ndarray,
    representation: str,
) -> np.ndarray:
    K = W.shape[1]
    output = np.zeros((len(region_counts), K), dtype=float)
    if representation == "soft_mean":
        np.add.at(output, inverse, W)
    elif representation == "published_hard":
        hard = np.argmax(W, axis=1)
        np.add.at(output, (inverse, hard), 1.0)
    else:
        raise ValueError(f"unknown representation: {representation}")
    output /= region_counts[:, None]
    output = np.maximum(output, 0.0)
    row_sums = output.sum(axis=1, keepdims=True)
    output /= np.maximum(row_sums, np.finfo(float).eps)
    return output


def _ilr(composition: np.ndarray) -> np.ndarray:
    if composition.shape[1] == 1:
        return np.zeros((composition.shape[0], 1), dtype=float)
    adjusted = np.maximum(composition, LOG_FLOOR)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    basis = helmert(composition.shape[1], full=False)
    # The matrices here are tiny; an explicit contraction avoids spurious
    # floating-point warnings emitted by some macOS BLAS builds for matmul.
    transformed = np.einsum(
        "ij,kj->ik", np.log(adjusted), basis, optimize=False
    )
    if not np.isfinite(transformed).all():
        raise ValueError("ILR transform produced a non-finite predictor")
    return transformed


def _log_composition(composition: np.ndarray) -> np.ndarray:
    """Apply an elementwise natural log without a log-ratio projection.

    The zero floor and re-closure match the preparation used by ``_ilr`` so
    this comparison isolates the effect of retaining the original composition
    coordinates instead of projecting them onto a Helmert basis.
    """

    if composition.ndim != 2 or composition.shape[1] < 1:
        raise ValueError(f"composition must be a nonempty matrix, got {composition.shape}")
    if not np.isfinite(composition).all():
        raise ValueError("composition contains a non-finite value")
    if float(np.min(composition)) < -1e-10:
        raise ValueError("composition contains a materially negative value")
    adjusted = np.maximum(composition, LOG_FLOOR)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    transformed = np.log(adjusted)
    if not np.isfinite(transformed).all():
        raise ValueError("log transform produced a non-finite predictor")
    return transformed


def _transform_composition(
    composition: np.ndarray,
    feature_transform: str,
) -> np.ndarray:
    if feature_transform == "ilr":
        return _ilr(composition)
    if feature_transform == "log":
        return _log_composition(composition)
    raise ValueError(
        f"unknown feature transform {feature_transform!r}; "
        f"expected one of {FEATURE_TRANSFORMS}"
    )


def _transform_definition(feature_transform: str) -> str:
    if feature_transform == "ilr":
        return "ILR coordinates after flooring composition entries at 1e-8 and re-closing"
    if feature_transform == "log":
        return (
            "elementwise natural log after flooring composition entries at 1e-8 "
            "and re-closing; no log-ratio projection"
        )
    raise ValueError(f"unknown feature transform: {feature_transform}")


def _classifier_model(classifier: str, *, split_index: int = 0) -> Any:
    if classifier == "ridge_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=260908,
            ),
        )
    if classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=RANDOM_FOREST_ESTIMATORS,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=1,
            random_state=260908 + int(split_index),
        )
    raise ValueError(f"unknown classifier {classifier!r}; expected one of {CLASSIFIERS}")


def _classifier_definition(classifier: str) -> str:
    if classifier == "ridge_logistic":
        return (
            "training-fold StandardScaler plus L2 logistic regression "
            "(C=1, liblinear, unweighted classes)"
        )
    if classifier == "random_forest":
        return (
            f"untuned random forest ({RANDOM_FOREST_ESTIMATORS} trees, Gini, "
            "sqrt features, unlimited depth, min leaf 1, unweighted classes, no scaling)"
        )
    raise ValueError(f"unknown classifier: {classifier}")


def _evaluate_binary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classifier: str = "ridge_logistic",
) -> dict[str, float | int]:
    observed = np.isfinite(y)
    X = X[observed]
    y = y[observed].astype(int)
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=260908,
    )
    metrics: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
        "brier": [],
    }
    for split_index, (train, test) in enumerate(splitter.split(X, y)):
        model = _classifier_model(classifier, split_index=split_index)
        model.fit(X[train], y[train])
        probability = model.predict_proba(X[test])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        truth = y[test]
        metrics["roc_auc"].append(float(roc_auc_score(truth, probability)))
        metrics["pr_auc"].append(float(average_precision_score(truth, probability)))
        metrics["accuracy"].append(float(accuracy_score(truth, prediction)))
        metrics["balanced_accuracy"].append(
            float(balanced_accuracy_score(truth, prediction))
        )
        metrics["sensitivity"].append(
            float(recall_score(truth, prediction, zero_division=0))
        )
        metrics["specificity"].append(
            float(recall_score(truth, prediction, pos_label=0, zero_division=0))
        )
        metrics["f1"].append(float(f1_score(truth, prediction, zero_division=0)))
        metrics["brier"].append(float(brier_score_loss(truth, probability)))

    result: dict[str, float | int] = {
        "region_count": int(len(y)),
        "positive_count": int(y.sum()),
        "split_count": int(len(metrics["accuracy"])),
    }
    for name, values in metrics.items():
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values, ddof=1))
    return result


def evaluate_fit_table(
    fit_table: Path,
    output: Path,
    *,
    artifact_root: Path | None = None,
    allow_missing_artifacts: bool = False,
    feature_transform: str = "ilr",
    classifier: str = "ridge_logistic",
) -> None:
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError(f"unknown feature transform: {feature_transform}")
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown classifier: {classifier}")
    frame = pd.read_csv(fit_table, low_memory=False)
    frame = frame[
        (frame["dataset"] == "stanford_crc_codex")
        & (frame["status"] == "ok")
        & frame["artifacts.estimate.path"].notna()
    ].copy()
    if frame.empty:
        raise ValueError(f"no successful CRC fits in {fit_table}")

    inverse, region_counts, labels = _region_contract()
    evaluated_at = datetime.now(timezone.utc).isoformat()
    output_rows: list[dict[str, Any]] = []
    missing = 0
    for (_, W_fit_id), shared_rows in frame.groupby(
        ["task_config_hash", "W_fit_id"], sort=False, dropna=False
    ):
        source_row = shared_rows.iloc[0]
        try:
            artifact = _artifact_path(
                str(source_row["artifacts.estimate.path"]), artifact_root
            )
        except FileNotFoundError:
            if allow_missing_artifacts:
                missing += 1
                continue
            raise
        with np.load(artifact, allow_pickle=False) as arrays:
            W = np.asarray(arrays["W_hat"], dtype=float)
        if W.shape[0] != len(inverse):
            raise ValueError(
                f"W row count {W.shape[0]} does not match canonical CRC rows {len(inverse)}"
            )
        evaluated: dict[tuple[str, str], dict[str, float | int]] = {}
        for representation in REPRESENTATIONS:
            X = _transform_composition(
                _aggregate_W(W, inverse, region_counts, representation),
                feature_transform,
            )
            for target in TARGETS:
                evaluated[(target, representation)] = {
                    "predictor_dimension": int(X.shape[1]),
                    **_evaluate_binary(X, labels[target], classifier=classifier),
                }

        for _, fit_row in shared_rows.iterrows():
            base = {
                column: fit_row.get(column)
                for column in CONFIG_COLUMNS
            }
            base.update(
                {
                    "fit_id": fit_row.get("fit_id"),
                    "W_fit_id": W_fit_id,
                    "task_config_hash": fit_row.get("task_config_hash"),
                    "seed": fit_row.get("seed"),
                    "biological_group": fit_row.get("biological_group"),
                    "evaluation_version": EVALUATION_VERSIONS[
                        (feature_transform, classifier)
                    ],
                    "evaluation_scope": "transductive_unsupervised_fit_region_cv",
                    "evaluated_at_utc": evaluated_at,
                    "predictor_transform": feature_transform,
                    "transform_definition": _transform_definition(feature_transform),
                    "transform_floor": LOG_FLOOR,
                    "zero_replacement": LOG_FLOOR,
                    "closure_after_zero_replacement": True,
                    "log_base": "e",
                    "classifier": classifier,
                    "classifier_definition": _classifier_definition(classifier),
                    "classifier_random_state": (
                        "260908 + repeated-CV split index"
                        if classifier == "random_forest"
                        else 260908
                    ),
                }
            )
            for (target, representation), result in evaluated.items():
                output_rows.append(
                    {
                        **base,
                        "target": target,
                        "representation": representation,
                        **result,
                    }
                )

    result_frame = pd.DataFrame(output_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result_frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(
        f"wrote {len(result_frame)} outcome rows to {output}; "
        f"skipped {missing} missing W artifacts"
    )


def evaluate_cell_type_baseline(
    output: Path,
    *,
    K_values: tuple[int, ...] = CELL_TYPE_BASELINE_K_VALUES,
    feature_transform: str = "ilr",
    classifier: str = "ridge_logistic",
) -> None:
    """Evaluate the fixed observed-cell-type control on the shared CRC splits.

    ``K`` is repeated only as a plotting coordinate so the fixed control can be
    compared with learned topic models at each requested topic count.  The
    predictor itself always has one component per observed CRC cell type.
    """

    if not K_values or any(int(value) <= 0 for value in K_values):
        raise ValueError("K_values must contain positive integers")
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError(f"unknown feature transform: {feature_transform}")
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown classifier: {classifier}")
    bundle = load_real_data("crc")
    inverse, region_counts, labels = _region_contract(bundle)
    composition_dimension = int(len(bundle.feature_ids))
    evaluated_at = datetime.now(timezone.utc).isoformat()
    evaluated: dict[tuple[str, str], dict[str, float | int]] = {}
    for representation in REPRESENTATIONS:
        composition = _cell_type_composition(
            bundle,
            inverse,
            region_counts,
            representation,
        )
        X = _transform_composition(composition, feature_transform)
        for target in TARGETS:
            evaluated[(target, representation)] = {
                "predictor_dimension": int(X.shape[1]),
                **_evaluate_binary(X, labels[target], classifier=classifier),
            }

    output_rows: list[dict[str, Any]] = []
    for K in dict.fromkeys(map(int, K_values)):
        base = {
            "dataset": "stanford_crc_codex",
            "K": K,
            "estimator_family": CELL_TYPE_BASELINE_FAMILY,
            "spectral_geometry": "observed_cell_type_composition",
            "vertex_hunter": "not_applicable",
            "preprocessing": "native_baseline",
            "A_recovery": "not_used_for_outcome",
            "W_recovery_method": "observed_cell_type_composition",
            "fit_id": "observed_cell_type_composition",
            "W_fit_id": "observed_cell_type_composition",
            "task_config_hash": f"cell_type_baseline_K{K}",
            "seed": 260908,
            "biological_group": "all",
            "evaluation_version": EVALUATION_VERSIONS[
                (feature_transform, classifier)
            ],
            "evaluation_scope": "fixed_observed_composition_region_cv",
            "evaluated_at_utc": evaluated_at,
            "predictor_transform": feature_transform,
            "transform_definition": _transform_definition(feature_transform),
            "transform_floor": LOG_FLOOR,
            "zero_replacement": LOG_FLOOR,
            "closure_after_zero_replacement": True,
            "log_base": "e",
            "classifier": classifier,
            "classifier_definition": _classifier_definition(classifier),
            "classifier_random_state": (
                "260908 + repeated-CV split index"
                if classifier == "random_forest"
                else 260908
            ),
            "composition_dimension": composition_dimension,
            "cell_type_features": "|".join(map(str, bundle.feature_ids)),
            "baseline_definition": (
                "A=I and W=canonical row-normalized observed cell-type counts; "
                "K is repeated only for comparison with learned topic-model curves"
            ),
        }
        for (target, representation), result in evaluated.items():
            output_rows.append(
                {
                    **base,
                    "target": target,
                    "representation": representation,
                    **result,
                }
            )

    result_frame = pd.DataFrame(output_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result_frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(
        f"wrote {len(result_frame)} fixed cell-type baseline rows to {output}; "
        f"composition dimension={composition_dimension}, "
        f"feature transform={feature_transform}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-table", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--cell-type-baseline-only", action="store_true")
    parser.add_argument(
        "--feature-transform",
        choices=FEATURE_TRANSFORMS,
        default="ilr",
        help="Region-level composition feature transform used before scaling/classification.",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        default="ridge_logistic",
        help="Downstream CRC outcome classifier.",
    )
    parser.add_argument(
        "--K-values",
        type=int,
        nargs="+",
        default=list(CELL_TYPE_BASELINE_K_VALUES),
    )
    args = parser.parse_args()
    if args.cell_type_baseline_only:
        if args.output is None:
            parser.error("--cell-type-baseline-only requires --output")
        evaluate_cell_type_baseline(
            args.output,
            K_values=tuple(args.K_values),
            feature_transform=args.feature_transform,
            classifier=args.classifier,
        )
        return
    if args.fit_table is None:
        parser.error("--fit-table is required unless --cell-type-baseline-only is set")
    if args.classifier == "ridge_logistic":
        default_name = (
            "crc_outcome_rows.csv"
            if args.feature_transform == "ilr"
            else f"crc_outcome_rows_{args.feature_transform}.csv"
        )
    else:
        default_name = (
            f"crc_outcome_rows_{args.feature_transform}_{args.classifier}.csv"
        )
    output = args.output or args.fit_table.with_name(default_name)
    evaluate_fit_table(
        args.fit_table,
        output,
        artifact_root=args.artifact_root,
        allow_missing_artifacts=args.allow_missing_artifacts,
        feature_transform=args.feature_transform,
        classifier=args.classifier,
    )


if __name__ == "__main__":
    main()
