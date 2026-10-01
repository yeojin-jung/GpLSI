#!/usr/bin/env python3
"""Evaluate CRC topics once per patient and relate topics to tumor phenotypes.

The Stanford CRC topic models are fitted to cells from tissue regions.  Some
patients contribute two regions, so downstream clinical evaluation must pool
those regions before constructing predictors.  This script implements that
patient-level contract and deliberately keeps outcome labels out of topic
fitting and feature construction.

For every unique saved ``W_hat`` it evaluates three patient summaries:

``hard_argmax_log``
    Log of the patient-level fractions of cells assigned to each argmax topic.
``mean_cell_ilr``
    Patient mean of the per-cell ILR coordinates of the soft topic weights.
``soft_mean_raw``
    Patient-level mean soft topic proportions, left on their original scale.

Both ridge logistic regression and a fixed random forest are evaluated on the
same deterministic repeated patient folds.  The script also runs a separate
CCA between logged patient-average topic proportions and the seven supplied
tumor-cell phenotype proportions.  CCA is performed in ILR coordinates, with
both all directly annotated cells (primary analysis) and W-matched retained
cells (sensitivity analysis).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gplsi.real_data import load_real_data  # noqa: E402


CRC_DATA_ROOT = REPO_ROOT / "data" / "crc"
TARGETS = ("primary_outcome", "recurrence")
SUMMARIES = ("hard_argmax_log", "mean_cell_ilr", "soft_mean_raw")
CLASSIFIERS = ("ridge_logistic", "random_forest")
PHENOTYPE_UNIVERSES = ("all_direct_cells", "W_matched_retained_cells")
CELL_TYPE_BASELINE_K_VALUES = (1, 2, 3, 4, 5, 6)
DIRECT_EIGHT_CELL_TYPE_BASELINE_FAMILY = "patient_direct_8_cell_type_baseline"
DIRECT_EIGHT_CELL_TYPES = (
    "B cell",
    "Blood vessel",
    "CD4 T cell",
    "CD8 T cell",
    "Granulocyte",
    "Macrophage",
    "Other",
    "Stroma",
)
EIGHT_CELL_TYPE_SUMMARY = "observed_direct_8_cell_type_composition_raw"
CONFIG_COLUMNS = (
    "dataset",
    "K",
    "estimator_family",
    "spectral_geometry",
    "vertex_hunter",
    "preprocessing",
    "W_recovery_method",
    "graph_svd_version",
)
LOG_FLOOR = 1e-8
CCA_PSEUDOCOUNT = 0.5
CV_RANDOM_STATE = 260908
RANDOM_FOREST_ESTIMATORS = 100
EVALUATION_VERSION = "patient_pool_rskf5x5_ridge_rf100_v1"
CCA_VERSION = "patient_pool_ilr_cca_rkf5x5_perm_v1"
PATIENT_MAPPING_VERSION = "sample_label_prefix_v1"
# Production CRC fits were created under this frozen contract identifier.  The
# observation-id digest directly pins the W row order used for patient pooling.
EXPECTED_CRC_W_CONTRACT_HASH = (
    "e8cc8cb6dcbce75de0fa44eeffabc4d9ef40d9319a061e988d66025efe582dfe"
)
EXPECTED_CRC_OBSERVATION_IDS_SHA256 = (
    "efdfae24e69f60a6509a84a11ae04d89e83b98a798b33be0e60f6dbf5fc5b202"
)


@dataclass(frozen=True)
class PatientContract:
    patient_keys: np.ndarray
    patient_labels: np.ndarray
    inverse: np.ndarray
    cell_counts: np.ndarray
    region_counts: np.ndarray
    labels: dict[str, np.ndarray]
    region_to_patient: dict[str, int]
    region_metadata: pd.DataFrame


@dataclass(frozen=True)
class CcaFit:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray
    correlations: np.ndarray
    x_rank: int
    y_rank: int


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(value: str) -> tuple[Any, ...]:
    import re

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def _unique_constant(frame: pd.DataFrame, group: str, value: str) -> pd.Series:
    grouped = frame.groupby(group, sort=False, dropna=False)[value]
    inconsistent = grouped.nunique(dropna=True) > 1
    if inconsistent.any():
        bad = ", ".join(map(str, inconsistent[inconsistent].index[:5]))
        raise ValueError(f"{value} is inconsistent within {group}: {bad}")
    return grouped.first()


def _patient_contract(bundle: Any | None = None) -> PatientContract:
    """Build the source-validated patient mapping for the analyzed CRC rows.

    The public data contain no explicit patient-id column.  Supplementary
    Table 1 of the SPACE-GM paper is exactly reproduced by the prefix before
    the underscore in ``sample_label_visualizer``: 196 primary-labelled
    regions from 109 patients and 186 recurrence-labelled regions from 103
    patients.  In particular, prefixes 24130 and 24131 remain distinct.
    """

    if bundle is None:
        bundle = load_real_data("crc")
    if bundle.outcomes is None:
        raise RuntimeError("CRC bundle has no outcome table")
    outcomes = bundle.outcomes.copy()
    required = {"region_id", "sample_label_visualizer", *TARGETS}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"CRC outcomes are missing columns {sorted(missing)}")
    outcomes["region_id"] = outcomes["region_id"].astype(str)
    outcomes["sample_label_visualizer"] = outcomes[
        "sample_label_visualizer"
    ].astype(str)

    region_sample = _unique_constant(
        outcomes, "region_id", "sample_label_visualizer"
    )
    region_metadata = region_sample.rename("sample_label_visualizer").reset_index()
    region_metadata["patient_key"] = region_metadata[
        "sample_label_visualizer"
    ].str.split("_", n=1).str[0]
    if region_metadata["patient_key"].str.len().eq(0).any():
        raise ValueError("empty CRC patient prefix")

    patient_keys = np.asarray(
        sorted(region_metadata["patient_key"].unique(), key=_natural_key),
        dtype=object,
    )
    patient_to_index = {str(key): i for i, key in enumerate(patient_keys)}
    patient_labels = np.asarray(
        [f"P{i + 1:03d}" for i in range(len(patient_keys))], dtype=object
    )
    region_metadata["patient_index"] = region_metadata["patient_key"].map(
        patient_to_index
    )
    region_to_patient = dict(
        zip(
            region_metadata["region_id"].astype(str),
            region_metadata["patient_index"].astype(int),
        )
    )

    group_values = np.asarray(bundle.group_ids, dtype=str)
    unknown = sorted(set(group_values).difference(region_to_patient))
    if unknown:
        raise ValueError(f"CRC rows contain unmapped regions: {unknown[:5]}")
    inverse = np.asarray([region_to_patient[x] for x in group_values], dtype=int)
    cell_counts = np.bincount(inverse, minlength=len(patient_keys)).astype(int)
    region_counts = (
        region_metadata.groupby("patient_index", sort=True)["region_id"]
        .nunique()
        .reindex(range(len(patient_keys)), fill_value=0)
        .to_numpy(dtype=int)
    )

    cell_meta = outcomes[["region_id", *TARGETS]].copy()
    cell_meta["patient_index"] = cell_meta["region_id"].map(region_to_patient)
    labels: dict[str, np.ndarray] = {}
    for target in TARGETS:
        patient_values = _unique_constant(cell_meta, "patient_index", target)
        labels[target] = patient_values.reindex(range(len(patient_keys))).to_numpy(
            dtype=float
        )

    # These are the class-specific counts reported in Supplementary Table 1.
    expected = {
        "primary_outcome": (196, 109, 68, 41),
        "recurrence": (186, 103, 72, 31),
    }
    region_level = outcomes.drop_duplicates("region_id")
    for target, (n_region, n_patient, n_zero, n_one) in expected.items():
        observed_patients = np.isfinite(labels[target])
        y = labels[target][observed_patients].astype(int)
        observed_regions = int(region_level[target].notna().sum())
        actual = (
            observed_regions,
            int(observed_patients.sum()),
            int((y == 0).sum()),
            int((y == 1).sum()),
        )
        if actual != (n_region, n_patient, n_zero, n_one):
            raise ValueError(
                f"CRC patient mapping does not reproduce published {target} counts: "
                f"got {actual}, expected {(n_region, n_patient, n_zero, n_one)}"
            )

    return PatientContract(
        patient_keys=patient_keys,
        patient_labels=patient_labels,
        inverse=inverse,
        cell_counts=cell_counts,
        region_counts=region_counts,
        labels=labels,
        region_to_patient=region_to_patient,
        region_metadata=region_metadata,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"expected a nonempty matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if float(np.min(matrix)) < -1e-8:
        raise ValueError("matrix contains materially negative values")
    matrix = np.maximum(matrix, 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("matrix contains a zero-sum row")
    return matrix / totals


def _ilr(composition: np.ndarray, *, floor: float = LOG_FLOOR) -> np.ndarray:
    composition = np.asarray(composition, dtype=float)
    if composition.ndim != 2 or composition.shape[1] < 1:
        raise ValueError(f"invalid composition shape {composition.shape}")
    if composition.shape[1] == 1:
        return np.empty((composition.shape[0], 0), dtype=float)
    adjusted = np.maximum(composition, floor)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    basis = helmert(adjusted.shape[1], full=False)
    transformed = np.einsum(
        "ij,kj->ik", np.log(adjusted), basis, optimize=False
    )
    if not np.isfinite(transformed).all():
        raise ValueError("ILR transform produced a non-finite value")
    return transformed


def _clr(composition: np.ndarray) -> np.ndarray:
    logged = np.log(np.asarray(composition, dtype=float))
    return logged - logged.mean(axis=1, keepdims=True)


def _aggregate_sum(
    matrix: np.ndarray, inverse: np.ndarray, n_patient: int
) -> np.ndarray:
    output = np.zeros((n_patient, matrix.shape[1]), dtype=float)
    np.add.at(output, inverse, matrix)
    return output


def _patient_topic_summaries(
    W: np.ndarray, contract: PatientContract
) -> dict[str, tuple[np.ndarray, int, str]]:
    """Pool all retained cells before constructing each patient summary."""

    W = _normalize_rows(W)
    if W.shape[0] != len(contract.inverse):
        raise ValueError(
            f"W has {W.shape[0]} rows; CRC patient contract has "
            f"{len(contract.inverse)} retained cells"
        )
    n_patient = len(contract.patient_keys)
    K = W.shape[1]
    soft_sums = _aggregate_sum(W, contract.inverse, n_patient)
    soft_mean = soft_sums / contract.cell_counts[:, None]
    soft_mean = _normalize_rows(soft_mean)

    hard_counts = np.zeros((n_patient, K), dtype=float)
    np.add.at(hard_counts, (contract.inverse, np.argmax(W, axis=1)), 1.0)
    hard_proportions = hard_counts / contract.cell_counts[:, None]
    hard_adjusted = np.maximum(hard_proportions, LOG_FLOOR)
    hard_adjusted /= hard_adjusted.sum(axis=1, keepdims=True)
    hard_log = np.log(hard_adjusted)

    cell_ilr = _ilr(W)
    if cell_ilr.shape[1]:
        mean_cell_ilr = _aggregate_sum(
            cell_ilr, contract.inverse, n_patient
        ) / contract.cell_counts[:, None]
    else:
        mean_cell_ilr = np.empty((n_patient, 0), dtype=float)

    return {
        "hard_argmax_log": (
            hard_log,
            0 if K == 1 else K,
            "natural log of pooled patient argmax-topic proportions after "
            "flooring at 1e-8 and re-closing",
        ),
        "mean_cell_ilr": (
            mean_cell_ilr,
            max(K - 1, 0),
            "cell-level Helmert ILR after flooring at 1e-8, averaged over all "
            "retained cells from all regions of the patient",
        ),
        "soft_mean_raw": (
            soft_mean,
            0 if K == 1 else K,
            "raw patient mean soft topic proportions after pooling retained cells "
            "from all patient regions",
        ),
    }


def _patient_direct_eight_cell_type_composition(
    contract: PatientContract,
    *,
    data_root: Path = CRC_DATA_ROOT,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool direct non-tumor cell annotations once per CRC patient.

    The official SPACE-GM raw archive supplies one ``cell_types.csv`` file per
    region containing every segmented cell.  The eight non-tumor labels are
    counted across every region assigned to a patient and normalized only after
    pooling.  Tumor 1--7 annotations are deliberately excluded.  This control
    never reads the derived 3-hop neighborhood-count matrices.

    Returns
    -------
    composition
        One row per patient and one column per ``DIRECT_EIGHT_CELL_TYPES``
        entry, in that fixed order.
    direct_cell_totals
        Number of direct non-tumor annotations contributing to each patient.
    """

    source = Path(data_root) / "raw_data"
    level_to_index = {
        label: index for index, label in enumerate(DIRECT_EIGHT_CELL_TYPES)
    }
    counts = np.zeros(
        (len(contract.patient_keys), len(DIRECT_EIGHT_CELL_TYPES)), dtype=float
    )
    for region in sorted(contract.region_to_patient, key=_natural_key):
        path = source / f"{region}.cell_types.csv"
        try:
            frame = pd.read_csv(path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"missing direct CRC cell annotations for region {region}: {path}; "
                "extract raw_data/*.cell_types.csv from the official SPACE-GM "
                "charville_raw_data.zip archive"
            ) from error
        required = {"CELL_ID", "CELL_TYPE"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"direct CRC cell annotation file is empty: {path}")
        if frame["CELL_ID"].isna().any() or frame["CELL_ID"].duplicated().any():
            raise ValueError(f"invalid direct CRC CELL_ID values in {path}")
        if frame["CELL_TYPE"].isna().any():
            raise ValueError(f"missing direct CRC CELL_TYPE values in {path}")

        labels = frame["CELL_TYPE"].astype(str)
        unexpected = sorted(
            {
                label
                for label in labels.unique()
                if label not in level_to_index and not label.startswith("Tumor ")
            }
        )
        if unexpected:
            raise ValueError(
                f"unexpected non-tumor CRC cell types in {path}: {unexpected}"
            )
        patient_index = contract.region_to_patient[region]
        region_counts = labels.value_counts(sort=False)
        for label, index in level_to_index.items():
            counts[patient_index, index] += float(region_counts.get(label, 0))

    direct_cell_totals = counts.sum(axis=1)
    if np.any(direct_cell_totals <= 0):
        bad = np.flatnonzero(direct_cell_totals <= 0)[:5].tolist()
        raise ValueError(f"CRC patients have no direct eight-type cells: {bad}")
    composition = counts / direct_cell_totals[:, None]
    if not np.isfinite(composition).all() or not np.allclose(
        composition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("direct CRC eight-cell-type compositions are invalid")
    return composition, direct_cell_totals.astype(int)


def _prediction_splits(
    contract: PatientContract,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], pd.DataFrame]:
    all_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        observed = np.flatnonzero(np.isfinite(contract.labels[target]))
        y = contract.labels[target][observed].astype(int)
        splitter = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=5, random_state=CV_RANDOM_STATE
        )
        target_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for split_index, (train_local, test_local) in enumerate(
            splitter.split(np.zeros((len(y), 1)), y)
        ):
            train = observed[train_local]
            test = observed[test_local]
            target_splits.append((train, test))
            repeat = split_index // 5 + 1
            fold = split_index % 5 + 1
            test_set = set(map(int, test))
            for patient_index in observed:
                rows.append(
                    {
                        "target": target,
                        "repeat": repeat,
                        "fold": fold,
                        "patient_id": contract.patient_labels[patient_index],
                        "role": "test" if int(patient_index) in test_set else "train",
                        "outcome": int(contract.labels[target][patient_index]),
                    }
                )
        all_splits[target] = target_splits
    return all_splits, pd.DataFrame(rows)


def _classifier_model(classifier: str, split_index: int) -> Any:
    if classifier == "ridge_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=CV_RANDOM_STATE,
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
            random_state=CV_RANDOM_STATE + split_index,
        )
    raise ValueError(f"unknown classifier {classifier}")


def _classifier_definition(classifier: str) -> str:
    if classifier == "ridge_logistic":
        return (
            "training-fold StandardScaler plus L2 logistic regression "
            "(C=1, liblinear, unweighted classes)"
        )
    if classifier == "random_forest":
        return (
            "untuned random forest (100 trees, Gini, sqrt features, unlimited "
            "depth, min leaf 1, unweighted classes, no scaling)"
        )
    raise ValueError(classifier)


def _evaluate_binary(
    X: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    classifier: str,
) -> dict[str, float | int]:
    informative_dimension = X.shape[1]
    if informative_dimension == 0:
        X = np.zeros((len(labels), 1), dtype=float)
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
    for split_index, (train, test) in enumerate(splits):
        y_train = labels[train].astype(int)
        y_test = labels[test].astype(int)
        if informative_dimension == 0:
            # K=1 contains no compositional predictor.  Use the unpenalized
            # training-fold intercept MLE rather than letting liblinear shrink
            # an arbitrary dummy-feature intercept.
            probability = np.full(len(test), float(y_train.mean()), dtype=float)
        else:
            model = _classifier_model(classifier, split_index)
            model.fit(X[train], y_train)
            probability = model.predict_proba(X[test])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics["roc_auc"].append(float(roc_auc_score(y_test, probability)))
        metrics["pr_auc"].append(float(average_precision_score(y_test, probability)))
        metrics["accuracy"].append(float(accuracy_score(y_test, prediction)))
        metrics["balanced_accuracy"].append(
            float(balanced_accuracy_score(y_test, prediction))
        )
        metrics["sensitivity"].append(
            float(recall_score(y_test, prediction, zero_division=0))
        )
        metrics["specificity"].append(
            float(recall_score(y_test, prediction, pos_label=0, zero_division=0))
        )
        metrics["f1"].append(float(f1_score(y_test, prediction, zero_division=0)))
        metrics["brier"].append(float(brier_score_loss(y_test, probability)))

    observed = np.isfinite(labels)
    y = labels[observed].astype(int)
    result: dict[str, float | int] = {
        "patient_count": int(len(y)),
        "positive_count": int(y.sum()),
        "split_count": int(len(splits)),
        "predictor_dimension": int(informative_dimension),
    }
    for name, values in metrics.items():
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values, ddof=1))
        # Split-repeat SE: SD of the five repeat means over sqrt(5).  It
        # describes the repeated-split estimate on this fixed cohort and W,
        # not patient-sampling or topic-fitting uncertainty.
        repeat_means = np.asarray(values).reshape(-1, 5).mean(axis=1)
        result[f"{name}_repeat_se"] = float(
            np.std(repeat_means, ddof=1) / np.sqrt(len(repeat_means))
        )
    return result


def _tumor_phenotype_counts(
    bundle: Any,
    contract: PatientContract,
    *,
    data_root: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Count tumor phenotypes for all cells and the W-matched subset."""

    region_to_observation_ids: dict[str, set[str]] = {}
    for observation_id in np.asarray(bundle.observation_ids, dtype=str):
        region, cell_id = observation_id.rsplit("::", 1)
        region_to_observation_ids.setdefault(region, set()).add(cell_id)

    source = data_root / "output" / "output_3hop"
    cached: list[tuple[int, pd.DataFrame, np.ndarray]] = []
    levels: set[str] = set()
    n_matched = 0
    for region in sorted(contract.region_to_patient, key=_natural_key):
        path = source / f"{region}.type.csv"
        frame = pd.read_csv(path)
        required = {"CELL_ID", "CELL_TYPE"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        ids = frame["CELL_ID"].astype(str)
        if ids.duplicated().any() or frame["CELL_TYPE"].isna().any():
            raise ValueError(f"invalid CELL_ID/CELL_TYPE values in {path}")
        labels = frame["CELL_TYPE"].astype(str).to_numpy()
        matched = ids.isin(region_to_observation_ids.get(region, set())).to_numpy()
        expected_matches = len(region_to_observation_ids.get(region, set()))
        if int(matched.sum()) != expected_matches:
            raise ValueError(
                f"retained-cell match failure in {region}: "
                f"{int(matched.sum())} != {expected_matches}"
            )
        patient_index = contract.region_to_patient[region]
        cached.append((patient_index, frame, matched))
        levels.update(map(str, labels))
        n_matched += int(matched.sum())

    if n_matched != len(bundle.observation_ids):
        raise ValueError(
            f"matched {n_matched} phenotype cells, expected {len(bundle.observation_ids)}"
        )
    ordered_levels = np.asarray(sorted(levels, key=_natural_key), dtype=object)
    if len(ordered_levels) != 7:
        raise ValueError(f"expected seven tumor phenotypes, got {ordered_levels.tolist()}")
    level_to_index = {str(level): i for i, level in enumerate(ordered_levels)}
    n_patient = len(contract.patient_keys)
    all_counts = np.zeros((n_patient, len(ordered_levels)), dtype=float)
    matched_counts = np.zeros_like(all_counts)
    for patient_index, frame, matched in cached:
        labels = frame["CELL_TYPE"].astype(str).to_numpy()
        for label in labels:
            all_counts[patient_index, level_to_index[label]] += 1.0
        for label in labels[matched]:
            matched_counts[patient_index, level_to_index[label]] += 1.0
    if np.any(all_counts.sum(axis=1) <= 0) or np.any(
        matched_counts.sum(axis=1) <= 0
    ):
        raise ValueError("a CRC patient has no tumor phenotype cells")
    counts = {
        "all_direct_cells": all_counts,
        "W_matched_retained_cells": matched_counts,
    }
    totals = {name: value.sum(axis=1).astype(int) for name, value in counts.items()}
    return ordered_levels, counts, totals


def _smooth_composition(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    adjusted = np.asarray(counts, dtype=float) + float(pseudocount)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _matrix_rank_and_invsqrt(
    covariance: np.ndarray,
) -> tuple[int, np.ndarray]:
    values, vectors = np.linalg.eigh(covariance)
    tolerance = max(covariance.shape) * np.finfo(float).eps * max(
        float(np.max(values)), 1.0
    )
    keep = values > tolerance
    rank = int(keep.sum())
    if rank == 0:
        return 0, np.zeros_like(covariance)
    invsqrt = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    return rank, invsqrt


def _fit_cca(X: np.ndarray, Y: np.ndarray) -> CcaFit:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] != Y.shape[0] or X.shape[0] < 3:
        raise ValueError("CCA matrices are not row-aligned or have too few patients")
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1)
    y_scale = Y.std(axis=0, ddof=1)
    x_scale[x_scale <= np.finfo(float).eps] = 1.0
    y_scale[y_scale <= np.finfo(float).eps] = 1.0
    Xs = (X - x_mean) / x_scale
    Ys = (Y - y_mean) / y_scale
    denominator = X.shape[0] - 1
    cxx = Xs.T @ Xs / denominator
    cyy = Ys.T @ Ys / denominator
    cxy = Xs.T @ Ys / denominator
    x_rank, x_inv = _matrix_rank_and_invsqrt(cxx)
    y_rank, y_inv = _matrix_rank_and_invsqrt(cyy)
    if min(x_rank, y_rank) == 0:
        raise ValueError("CCA has zero-rank input")
    left, singular, right_t = np.linalg.svd(x_inv @ cxy @ y_inv)
    m = min(x_rank, y_rank)
    x_weights = x_inv @ left[:, :m]
    y_weights = y_inv @ right_t.T[:, :m]
    return CcaFit(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        x_weights=x_weights,
        y_weights=y_weights,
        correlations=np.clip(singular[:m], 0.0, 1.0),
        x_rank=x_rank,
        y_rank=y_rank,
    )


def _cca_scores(
    fit: CcaFit, X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Xs = (X - fit.x_mean) / fit.x_scale
    Ys = (Y - fit.y_mean) / fit.y_scale
    return Xs @ fit.x_weights, Ys @ fit.y_weights


def _corr_columns(matrix: np.ndarray, scores: np.ndarray) -> np.ndarray:
    output = np.empty((matrix.shape[1], scores.shape[1]), dtype=float)
    for i in range(matrix.shape[1]):
        for j in range(scores.shape[1]):
            if np.std(matrix[:, i]) <= np.finfo(float).eps or np.std(
                scores[:, j]
            ) <= np.finfo(float).eps:
                output[i, j] = np.nan
            else:
                output[i, j] = float(np.corrcoef(matrix[:, i], scores[:, j])[0, 1])
    return output


def _orient_cca(
    fit: CcaFit,
    X: np.ndarray,
    Y: np.ndarray,
    phenotype_clr: np.ndarray,
) -> tuple[CcaFit, np.ndarray, np.ndarray]:
    x_scores, y_scores = _cca_scores(fit, X, Y)
    phenotype_structure = _corr_columns(phenotype_clr, y_scores)
    x_weights = fit.x_weights.copy()
    y_weights = fit.y_weights.copy()
    for component in range(y_scores.shape[1]):
        values = phenotype_structure[:, component]
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            anchor = finite[np.argmax(np.abs(values[finite]))]
            if values[anchor] < 0:
                x_weights[:, component] *= -1
                y_weights[:, component] *= -1
                x_scores[:, component] *= -1
                y_scores[:, component] *= -1
    oriented = CcaFit(
        x_mean=fit.x_mean,
        x_scale=fit.x_scale,
        y_mean=fit.y_mean,
        y_scale=fit.y_scale,
        x_weights=x_weights,
        y_weights=y_weights,
        correlations=fit.correlations,
        x_rank=fit.x_rank,
        y_rank=fit.y_rank,
    )
    return oriented, x_scores, y_scores


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= np.finfo(float).eps or np.std(
        y
    ) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cca_cv_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    phenotype_clr: np.ndarray,
) -> list[list[float]]:
    splitter = RepeatedKFold(
        n_splits=5, n_repeats=5, random_state=CV_RANDOM_STATE
    )
    m = min(X.shape[1], Y.shape[1])
    values: list[list[float]] = [[] for _ in range(m)]
    for train, test in splitter.split(X):
        fit = _fit_cca(X[train], Y[train])
        fit, _, _ = _orient_cca(
            fit, X[train], Y[train], phenotype_clr[train]
        )
        x_test, y_test = _cca_scores(fit, X[test], Y[test])
        for component in range(min(m, x_test.shape[1])):
            values[component].append(
                _safe_correlation(x_test[:, component], y_test[:, component])
            )
    return values


def _wilks_statistic(correlations: np.ndarray) -> float:
    squared = np.minimum(np.asarray(correlations, dtype=float) ** 2, 1 - 1e-12)
    return float(-np.log1p(-squared).sum())


def _permutation_pvalue(
    X: np.ndarray,
    Y: np.ndarray,
    observed_statistic: float,
    *,
    n_permutations: int,
    random_state: int,
) -> float:
    if n_permutations <= 0:
        return float("nan")
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(n_permutations):
        fit = _fit_cca(X, Y[rng.permutation(len(Y))])
        if _wilks_statistic(fit.correlations) >= observed_statistic - 1e-12:
            exceedances += 1
    return float((1 + exceedances) / (1 + n_permutations))


def _stable_seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode("utf-8")).digest()
    return CV_RANDOM_STATE + int.from_bytes(digest[:4], "big") % 1_000_000_000


def _base_fit_metadata(source_row: pd.Series, artifact: Path) -> dict[str, Any]:
    base = {column: source_row.get(column) for column in CONFIG_COLUMNS}
    base.update(
        {
            "A_recovery": "not_used_for_patient_analysis",
            "source_fit_id": source_row.get("fit_id"),
            "W_fit_id": source_row.get("W_fit_id"),
            "task_config_hash": source_row.get("task_config_hash"),
            "seed": source_row.get("seed"),
            "biological_group": source_row.get("biological_group"),
            "source_artifact_path": str(artifact),
        }
    )
    return base


def _select_W_rows(
    fit_tables: Iterable[Path],
    *,
    artifact_root: Path | None,
    include_estimators: set[str] | None,
    allow_missing_artifacts: bool,
) -> tuple[list[tuple[pd.Series, Path]], int]:
    frames = [pd.read_csv(path, low_memory=False) for path in fit_tables]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[
        frame["dataset"].eq("stanford_crc_codex")
        & frame["status"].eq("ok")
        & frame["artifacts.estimate.path"].notna()
    ].copy()
    if include_estimators:
        frame = frame[frame["estimator_family"].astype(str).isin(include_estimators)]
    if frame.empty:
        raise ValueError("no successful CRC W fits matched the requested tables")
    # Prefer the current-A artifact only to make selection deterministic.  W is
    # explicitly shared across A recoveries and is evaluated exactly once.
    priority = {"A_current": 0, "native_baseline": 1, "A_full_Pois": 2}
    frame["__priority"] = frame["A_recovery"].map(priority).fillna(9)
    frame = frame.sort_values(
        ["task_config_hash", "W_fit_id", "__priority", "fit_id"], kind="stable"
    )
    selected: list[tuple[pd.Series, Path]] = []
    missing = 0
    for _, group in frame.groupby(
        ["task_config_hash", "W_fit_id"], sort=False, dropna=False
    ):
        resolved: tuple[pd.Series, Path] | None = None
        for _, row in group.iterrows():
            try:
                artifact = _artifact_path(
                    str(row["artifacts.estimate.path"]), artifact_root
                )
            except FileNotFoundError:
                continue
            resolved = (row, artifact)
            break
        if resolved is None:
            if allow_missing_artifacts:
                missing += 1
                continue
            raise FileNotFoundError(
                "no artifact resolved for W fit " + str(group.iloc[0]["W_fit_id"])
            )
        selected.append(resolved)
    return selected, missing


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate(
    fit_tables: Iterable[Path],
    output_dir: Path,
    *,
    artifact_root: Path | None = None,
    data_root: Path = CRC_DATA_ROOT,
    include_estimators: set[str] | None = None,
    allow_missing_artifacts: bool = False,
    cca_permutations: int = 999,
    run_prediction: bool = True,
    run_cca: bool = True,
    include_cell_type_baseline: bool = True,
    cell_type_baselines_only: bool = False,
) -> None:
    bundle = load_real_data("crc")
    contract = _patient_contract(bundle)
    canonical_hashes = bundle.metadata.get("canonical_contract_hashes")
    if not isinstance(canonical_hashes, dict):
        raise ValueError("CRC bundle has no validated canonical-contract hashes")
    if (
        canonical_hashes.get("observation_ids_sha256")
        != EXPECTED_CRC_OBSERVATION_IDS_SHA256
    ):
        raise ValueError("loaded CRC observation order does not match production W rows")
    if cell_type_baselines_only:
        selected: list[tuple[pd.Series, Path]] = []
        missing_artifacts = 0
    else:
        selected, missing_artifacts = _select_W_rows(
            fit_tables,
            artifact_root=artifact_root,
            include_estimators=include_estimators,
            allow_missing_artifacts=allow_missing_artifacts,
        )
    prediction_splits, split_frame = _prediction_splits(contract)
    evaluated_at = datetime.now(timezone.utc).isoformat()

    patient_contract_rows: list[dict[str, Any]] = []
    for patient_index, patient_key in enumerate(contract.patient_keys):
        patient_contract_rows.append(
            {
                "patient_id": contract.patient_labels[patient_index],
                "patient_key_sha256": hashlib.sha256(
                    str(patient_key).encode("utf-8")
                ).hexdigest(),
                "region_count": int(contract.region_counts[patient_index]),
                "retained_cell_count": int(contract.cell_counts[patient_index]),
                **{
                    target: contract.labels[target][patient_index]
                    for target in TARGETS
                },
                "mapping_version": PATIENT_MAPPING_VERSION,
            }
        )
    _atomic_csv(pd.DataFrame(patient_contract_rows), output_dir / "patient_contract.csv")
    _atomic_csv(split_frame, output_dir / "prediction_cv_splits.csv")

    phenotype_levels: np.ndarray | None = None
    phenotype_counts: dict[str, np.ndarray] = {}
    phenotype_totals: dict[str, np.ndarray] = {}
    if run_cca or (run_prediction and include_cell_type_baseline):
        phenotype_levels, phenotype_counts, phenotype_totals = _tumor_phenotype_counts(
            bundle, contract, data_root=data_root
        )
    if run_cca:
        cca_splitter = RepeatedKFold(
            n_splits=5, n_repeats=5, random_state=CV_RANDOM_STATE
        )
        cca_split_rows: list[dict[str, Any]] = []
        for split_index, (train, test) in enumerate(
            cca_splitter.split(np.arange(len(contract.patient_keys)))
        ):
            repeat = split_index // 5 + 1
            fold = split_index % 5 + 1
            test_set = set(map(int, test))
            for patient_index in range(len(contract.patient_keys)):
                cca_split_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "patient_id": contract.patient_labels[patient_index],
                        "role": "test" if patient_index in test_set else "train",
                    }
                )
        _atomic_csv(pd.DataFrame(cca_split_rows), output_dir / "cca_cv_splits.csv")

    outcome_rows: list[dict[str, Any]] = []
    cca_summary_rows: list[dict[str, Any]] = []
    cca_component_rows: list[dict[str, Any]] = []
    cca_loading_rows: list[dict[str, Any]] = []

    # Fixed comparison 1: the same classifiers applied to the patient-pooled
    # direct annotations of the eight non-tumor cell types.  This is a no-fit
    # control sourced from the official raw SPACE-GM cell annotation files and
    # never uses the derived 3-hop neighborhood-count matrices.
    if run_prediction and include_cell_type_baseline:
        assert phenotype_levels is not None
        (
            eight_type_composition,
            eight_type_cell_totals,
        ) = _patient_direct_eight_cell_type_composition(
            contract, data_root=data_root
        )
        cell_type_composition = _normalize_rows(phenotype_counts["all_direct_cells"])
        K_values = list(CELL_TYPE_BASELINE_K_VALUES)
        eight_type_results = {
            (target, classifier): _evaluate_binary(
                eight_type_composition,
                contract.labels[target],
                prediction_splits[target],
                classifier=classifier,
            )
            for target in TARGETS
            for classifier in CLASSIFIERS
        }
        direct_type_results = {
            (target, classifier): _evaluate_binary(
                cell_type_composition,
                contract.labels[target],
                prediction_splits[target],
                classifier=classifier,
            )
            for target in TARGETS
            for classifier in CLASSIFIERS
        }
        for K in K_values:
            baseline_base = {
                "dataset": "stanford_crc_codex",
                "K": K,
                "estimator_family": DIRECT_EIGHT_CELL_TYPE_BASELINE_FAMILY,
                "spectral_geometry": "direct_observed_cell_type_composition",
                "vertex_hunter": "not_applicable",
                "preprocessing": "not_applicable",
                "W_recovery_method": "not_applicable",
                "graph_svd_version": "not_applicable",
                "A_recovery": "not_used_for_patient_analysis",
                "source_fit_id": "observed_direct_8_cell_type_composition",
                "W_fit_id": "observed_direct_8_cell_type_composition",
                "task_config_hash": f"patient_direct_8_cell_type_baseline_K{K}",
                "seed": CV_RANDOM_STATE,
                "biological_group": "all",
                "source_artifact_path": "data/crc/raw_data/*.cell_types.csv",
                "source_archive_doi": "10.5281/zenodo.13179600",
                "source_archive_file": "charville_raw_data.zip",
                "uses_3hop_counts": False,
            }
            for target in TARGETS:
                for classifier in CLASSIFIERS:
                    result = eight_type_results[(target, classifier)]
                    observed = np.isfinite(contract.labels[target])
                    outcome_rows.append(
                        {
                            **baseline_base,
                            "evaluation_version": EVALUATION_VERSION,
                            "evaluation_scope": (
                                "fixed_direct_8_cell_type_composition_patient_cv"
                            ),
                            "evaluated_at_utc": evaluated_at,
                            "patient_mapping_version": PATIENT_MAPPING_VERSION,
                            "aggregation_scope": (
                                "direct non-tumor cell annotations summed across all "
                                "regions sharing the patient prefix, then normalized "
                                "once per patient"
                            ),
                            "summary": EIGHT_CELL_TYPE_SUMMARY,
                            "summary_definition": (
                                "one patient-level composition obtained by counting "
                                "the eight direct non-tumor CELL_TYPE annotations over "
                                "all patient regions and normalizing once; Tumor 1--7 "
                                "are excluded; no 3-hop counts and no topic fit"
                            ),
                            "target": target,
                            "classifier": classifier,
                            "classifier_definition": _classifier_definition(classifier),
                            "region_count": int(
                                contract.region_metadata.loc[
                                    contract.region_metadata["patient_index"].isin(
                                        np.flatnonzero(observed)
                                    ),
                                    "region_id",
                                ].nunique()
                            ),
                            "retained_cell_count": int(
                                eight_type_cell_totals[observed].sum()
                            ),
                            "cell_count": int(
                                eight_type_cell_totals[observed].sum()
                            ),
                            "cell_universe": "all_direct_non_tumor_cells",
                            "majority_accuracy": float(
                                max(
                                    np.mean(contract.labels[target][observed] == 0),
                                    np.mean(contract.labels[target][observed] == 1),
                                )
                            ),
                            "composition_dimension": int(
                                eight_type_composition.shape[1]
                            ),
                            "cell_type_features": "|".join(
                                DIRECT_EIGHT_CELL_TYPES
                            ),
                            **result,
                        }
                    )

        # Fixed comparison 2: direct patient proportions of the seven observed
        # focal-tumor phenotypes.  This uses CELL_TYPE annotations and no 3-hop
        # neighborhood counts.
        for K in K_values:
            baseline_base = {
                "dataset": "stanford_crc_codex",
                "K": K,
                "estimator_family": "raw_cell_type_baseline",
                "spectral_geometry": "not_applicable",
                "vertex_hunter": "not_applicable",
                "preprocessing": "not_applicable",
                "W_recovery_method": "not_applicable",
                "graph_svd_version": "not_applicable",
                "A_recovery": "not_used_for_patient_analysis",
                "source_fit_id": "observed_focal_tumor_type_proportions",
                "W_fit_id": "observed_focal_tumor_type_proportions",
                "task_config_hash": f"patient_cell_type_baseline_K{K}",
                "seed": CV_RANDOM_STATE,
                "biological_group": "all",
                "source_artifact_path": "data/crc/output/output_3hop/*.type.csv",
            }
            for target in TARGETS:
                for classifier in CLASSIFIERS:
                    result = direct_type_results[(target, classifier)]
                    observed = np.isfinite(contract.labels[target])
                    outcome_rows.append(
                        {
                            **baseline_base,
                            "evaluation_version": EVALUATION_VERSION,
                            "evaluation_scope": "fixed_observed_composition_patient_cv",
                            "evaluated_at_utc": evaluated_at,
                            "patient_mapping_version": PATIENT_MAPPING_VERSION,
                            "aggregation_scope": (
                                "all direct focal-tumor cells pooled across all "
                                "regions sharing the patient prefix"
                            ),
                            "summary": "observed_tumor_type_proportion_raw",
                            "summary_definition": (
                                "raw patient proportions of the seven supplied focal-"
                                "tumor CELL_TYPE annotations; no 3-hop counts"
                            ),
                            "target": target,
                            "classifier": classifier,
                            "classifier_definition": _classifier_definition(classifier),
                            "region_count": int(
                                contract.region_metadata.loc[
                                    contract.region_metadata["patient_index"].isin(
                                        np.flatnonzero(observed)
                                    ),
                                    "region_id",
                                ].nunique()
                            ),
                            "retained_cell_count": int(
                                phenotype_totals["all_direct_cells"][observed].sum()
                            ),
                            "cell_count": int(
                                phenotype_totals["all_direct_cells"][observed].sum()
                            ),
                            "cell_universe": "all_direct_focal_tumor_cells",
                            "majority_accuracy": float(
                                max(
                                    np.mean(contract.labels[target][observed] == 0),
                                    np.mean(contract.labels[target][observed] == 1),
                                )
                            ),
                            "composition_dimension": int(len(phenotype_levels)),
                            "cell_type_features": "|".join(map(str, phenotype_levels)),
                            **result,
                        }
                    )

    for fit_index, (source_row, artifact) in enumerate(selected, start=1):
        source_contract_hash = str(source_row.get("canonical_contract_hash", ""))
        if source_contract_hash != EXPECTED_CRC_W_CONTRACT_HASH:
            raise ValueError(
                "saved W canonical-contract hash does not match the loaded CRC "
                "production contract: "
                f"{source_contract_hash} != {EXPECTED_CRC_W_CONTRACT_HASH}"
            )
        declared_artifact_hash = str(
            source_row.get("artifacts.estimate.sha256", "")
        )
        if declared_artifact_hash and declared_artifact_hash != "nan":
            observed_artifact_hash = _sha256_file(artifact)
            if observed_artifact_hash != declared_artifact_hash:
                raise ValueError(
                    f"saved W artifact checksum mismatch for {artifact}: "
                    f"{observed_artifact_hash} != {declared_artifact_hash}"
                )
        with np.load(artifact, allow_pickle=False) as arrays:
            W = _normalize_rows(np.asarray(arrays["W_hat"], dtype=float))
        expected_K = int(source_row.get("K"))
        if W.shape[1] != expected_K:
            raise ValueError(
                f"saved W has K={W.shape[1]}, but fit row declares K={expected_K}"
            )
        source_n = source_row.get("n")
        if pd.notna(source_n) and int(source_n) != len(contract.inverse):
            raise ValueError(
                f"saved fit declares n={int(source_n)}, expected {len(contract.inverse)}"
            )
        summaries = _patient_topic_summaries(W, contract)
        base = _base_fit_metadata(source_row, artifact)

        if run_prediction:
            for summary_name in SUMMARIES:
                X, recorded_dimension, definition = summaries[summary_name]
                # K=1 is an explicit intercept-only comparison.
                if recorded_dimension == 0:
                    X_for_prediction = np.empty((len(X), 0), dtype=float)
                else:
                    X_for_prediction = X
                for target in TARGETS:
                    for classifier in CLASSIFIERS:
                        result = _evaluate_binary(
                            X_for_prediction,
                            contract.labels[target],
                            prediction_splits[target],
                            classifier=classifier,
                        )
                        observed = np.isfinite(contract.labels[target])
                        outcome_rows.append(
                            {
                                **base,
                                "evaluation_version": EVALUATION_VERSION,
                                "evaluation_scope": (
                                    "transductive_outcome_free_topic_fit_patient_cv"
                                ),
                                "evaluated_at_utc": evaluated_at,
                                "patient_mapping_version": PATIENT_MAPPING_VERSION,
                                "aggregation_scope": (
                                    "all retained cells pooled across all regions "
                                    "sharing the patient prefix"
                                ),
                                "summary": summary_name,
                                "summary_definition": definition,
                                "target": target,
                                "classifier": classifier,
                                "classifier_definition": _classifier_definition(
                                    classifier
                                ),
                                "region_count": int(
                                    contract.region_metadata.loc[
                                        contract.region_metadata["patient_index"].isin(
                                            np.flatnonzero(observed)
                                        ),
                                        "region_id",
                                    ].nunique()
                                ),
                                "retained_cell_count": int(
                                    contract.cell_counts[observed].sum()
                                ),
                                "cell_count": int(contract.cell_counts[observed].sum()),
                                "cell_universe": "topic_model_retained_cells",
                                "pr_auc_definition": "average precision",
                                "majority_accuracy": float(
                                    max(
                                        np.mean(contract.labels[target][observed] == 0),
                                        np.mean(contract.labels[target][observed] == 1),
                                    )
                                ),
                                **result,
                            }
                        )

        if run_cca:
            assert phenotype_levels is not None
            K = W.shape[1]
            common = {
                **base,
                "cca_version": CCA_VERSION,
                "evaluated_at_utc": evaluated_at,
                "patient_mapping_version": PATIENT_MAPPING_VERSION,
                "topic_representation": "log_patient_soft_mean",
                "topic_internal_coordinates": "Helmert_ILR",
                "phenotype_internal_coordinates": "Helmert_ILR",
                "topic_pseudocount": CCA_PSEUDOCOUNT,
                "phenotype_pseudocount": CCA_PSEUDOCOUNT,
                "patient_count": len(contract.patient_keys),
                "region_count": int(contract.region_metadata["region_id"].nunique()),
                "retained_topic_cell_count": int(len(W)),
                "topic_fit_scope": "transductive_outcome_free_unsupervised_fit",
            }
            if K == 1:
                for universe in PHENOTYPE_UNIVERSES:
                    cca_summary_rows.append(
                        {
                            **common,
                            "phenotype_cell_universe": universe,
                            "phenotype_cell_count": int(
                                phenotype_totals[universe].sum()
                            ),
                            "status": "not_applicable",
                            "reason": "K=1 has no topic log-ratio coordinate",
                            "topic_dimension": 0,
                            "phenotype_dimension": 6,
                            "topic_rank": 0,
                            "phenotype_rank": 6,
                            "component_count": 0,
                            "wilks_lambda": np.nan,
                            "wilks_log_statistic": np.nan,
                            "permutation_count": cca_permutations,
                            "permutation_p_value": np.nan,
                            "permutation_q_value": np.nan,
                        }
                    )
                continue

            topic_sums = _aggregate_sum(W, contract.inverse, len(contract.patient_keys))
            topic_composition = _smooth_composition(topic_sums, CCA_PSEUDOCOUNT)
            topic_ilr = _ilr(topic_composition, floor=np.finfo(float).tiny)
            topic_clr = _clr(topic_composition)

            for universe in PHENOTYPE_UNIVERSES:
                phenotype_composition = _smooth_composition(
                    phenotype_counts[universe], CCA_PSEUDOCOUNT
                )
                phenotype_ilr = _ilr(
                    phenotype_composition, floor=np.finfo(float).tiny
                )
                phenotype_clr = _clr(phenotype_composition)
                fit = _fit_cca(topic_ilr, phenotype_ilr)
                fit, x_scores, y_scores = _orient_cca(
                    fit, topic_ilr, phenotype_ilr, phenotype_clr
                )
                wilks_log = _wilks_statistic(fit.correlations)
                wilks_lambda = float(np.exp(-wilks_log))
                permutation_p = _permutation_pvalue(
                    topic_ilr,
                    phenotype_ilr,
                    wilks_log,
                    n_permutations=cca_permutations,
                    random_state=_stable_seed(
                        str(base.get("task_config_hash")),
                        str(base.get("W_fit_id")),
                        universe,
                    ),
                )
                cca_summary_rows.append(
                    {
                        **common,
                        "phenotype_cell_universe": universe,
                        "phenotype_cell_count": int(
                            phenotype_totals[universe].sum()
                        ),
                        "status": "ok",
                        "reason": "",
                        "topic_dimension": int(topic_ilr.shape[1]),
                        "phenotype_dimension": int(phenotype_ilr.shape[1]),
                        "topic_rank": fit.x_rank,
                        "phenotype_rank": fit.y_rank,
                        "component_count": int(len(fit.correlations)),
                        "wilks_lambda": wilks_lambda,
                        "wilks_log_statistic": wilks_log,
                        "permutation_count": cca_permutations,
                        "permutation_p_value": permutation_p,
                        # Filled globally after all task outputs are collected.
                        "permutation_q_value": np.nan,
                    }
                )

                cv_values = _cca_cv_correlations(
                    topic_ilr, phenotype_ilr, phenotype_clr
                )
                topic_structure = _corr_columns(topic_clr, x_scores)
                topic_cross = _corr_columns(topic_clr, y_scores)
                phenotype_structure = _corr_columns(phenotype_clr, y_scores)
                phenotype_cross = _corr_columns(phenotype_clr, x_scores)
                for component, rho in enumerate(fit.correlations):
                    heldout = np.asarray(cv_values[component], dtype=float)
                    finite = heldout[np.isfinite(heldout)]
                    fisher = (
                        float(np.tanh(np.mean(np.arctanh(np.clip(finite, -0.999999, 0.999999)))))
                        if len(finite)
                        else np.nan
                    )
                    cca_component_rows.append(
                        {
                            **common,
                            "phenotype_cell_universe": universe,
                            "component": component + 1,
                            "full_data_canonical_correlation": float(rho),
                            "heldout_correlation_fisher_mean": fisher,
                            "heldout_correlation_median": (
                                float(np.median(finite)) if len(finite) else np.nan
                            ),
                            "heldout_correlation_std": (
                                float(np.std(finite, ddof=1))
                                if len(finite) > 1
                                else np.nan
                            ),
                            "heldout_split_count": int(len(finite)),
                        }
                    )
                    for part in range(K):
                        cca_loading_rows.append(
                            {
                                **common,
                                "phenotype_cell_universe": universe,
                                "component": component + 1,
                                "side": "topic",
                                "part_index": part + 1,
                                "part_name": f"Topic {part + 1}",
                                "part_scale": "centered_log_ratio",
                                "structure_correlation": float(
                                    topic_structure[part, component]
                                ),
                                "cross_loading": float(
                                    topic_cross[part, component]
                                ),
                            }
                        )
                    for part, name in enumerate(phenotype_levels):
                        cca_loading_rows.append(
                            {
                                **common,
                                "phenotype_cell_universe": universe,
                                "component": component + 1,
                                "side": "phenotype",
                                "part_index": part + 1,
                                "part_name": str(name),
                                "part_scale": "centered_log_ratio",
                                "structure_correlation": float(
                                    phenotype_structure[part, component]
                                ),
                                "cross_loading": float(
                                    phenotype_cross[part, component]
                                ),
                            }
                        )

        if fit_index % 10 == 0 or fit_index == len(selected):
            print(f"evaluated {fit_index}/{len(selected)} unique W fits", flush=True)

    if run_prediction:
        _atomic_csv(pd.DataFrame(outcome_rows), output_dir / "outcome_rows.csv")
    if run_cca:
        _atomic_csv(pd.DataFrame(cca_summary_rows), output_dir / "cca_summary.csv")
        component_frame = pd.DataFrame(cca_component_rows)
        if component_frame.empty:
            component_frame = pd.DataFrame(
                columns=[
                    "task_config_hash",
                    "W_fit_id",
                    "phenotype_cell_universe",
                    "component",
                ]
            )
        loading_frame = pd.DataFrame(cca_loading_rows)
        if loading_frame.empty:
            loading_frame = pd.DataFrame(
                columns=[
                    "task_config_hash",
                    "W_fit_id",
                    "phenotype_cell_universe",
                    "component",
                    "side",
                    "part_index",
                ]
            )
        _atomic_csv(component_frame, output_dir / "cca_components.csv")
        _atomic_csv(loading_frame, output_dir / "cca_loadings.csv")
        audit_rows: list[dict[str, Any]] = []
        assert phenotype_levels is not None
        for universe in PHENOTYPE_UNIVERSES:
            composition = _smooth_composition(
                phenotype_counts[universe], CCA_PSEUDOCOUNT
            )
            for patient_index in range(len(contract.patient_keys)):
                for part, name in enumerate(phenotype_levels):
                    audit_rows.append(
                        {
                            "patient_id": contract.patient_labels[patient_index],
                            "phenotype_cell_universe": universe,
                            "phenotype": str(name),
                            "count": int(phenotype_counts[universe][patient_index, part]),
                            "smoothed_proportion": float(
                                composition[patient_index, part]
                            ),
                            "total_cells": int(
                                phenotype_totals[universe][patient_index]
                            ),
                        }
                    )
        _atomic_csv(
            pd.DataFrame(audit_rows), output_dir / "cca_phenotype_patient_features.csv"
        )

    print(
        f"completed {len(selected)} unique W fits in {output_dir}; "
        f"missing artifacts skipped={missing_artifacts}; "
        f"patient counts primary=109, recurrence=103",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-table", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--crc-data-root", type=Path, default=CRC_DATA_ROOT)
    parser.add_argument("--include-estimator", action="append")
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--cca-permutations", type=int, default=999)
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--cca-only", action="store_true")
    parser.add_argument(
        "--skip-cell-type-baseline",
        action="store_true",
        help="Do not emit the fixed 8-feature and seven-phenotype controls.",
    )
    parser.add_argument(
        "--cell-type-baselines-only",
        action="store_true",
        help=(
            "Evaluate only the two fixed patient-level composition controls; "
            "no saved W artifacts or CCA are required."
        ),
    )
    args = parser.parse_args()
    if args.prediction_only and args.cca_only:
        parser.error("--prediction-only and --cca-only are mutually exclusive")
    if args.cell_type_baselines_only and args.cca_only:
        parser.error("--cell-type-baselines-only cannot be combined with --cca-only")
    if args.cell_type_baselines_only and args.skip_cell_type_baseline:
        parser.error(
            "--cell-type-baselines-only cannot be combined with "
            "--skip-cell-type-baseline"
        )
    if not args.cell_type_baselines_only and not args.fit_table:
        parser.error("at least one --fit-table is required")
    if args.cca_permutations < 0:
        parser.error("--cca-permutations must be nonnegative")
    evaluate(
        args.fit_table,
        args.output_dir,
        artifact_root=args.artifact_root,
        data_root=args.crc_data_root,
        include_estimators=(
            set(args.include_estimator) if args.include_estimator else None
        ),
        allow_missing_artifacts=args.allow_missing_artifacts,
        cca_permutations=args.cca_permutations,
        run_prediction=not args.cca_only,
        run_cca=not (args.prediction_only or args.cell_type_baselines_only),
        include_cell_type_baseline=not args.skip_cell_type_baseline,
        cell_type_baselines_only=args.cell_type_baselines_only,
    )


if __name__ == "__main__":
    main()
