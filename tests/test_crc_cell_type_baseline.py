from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_module(
    "evaluate_crc_outcomes_for_test",
    "scripts/real_data_anchor_word_gplsi/evaluate_crc_outcomes.py",
)
DASHBOARD = _load_module(
    "build_results_dashboard_data_for_test",
    "scripts/real_data_anchor_word_gplsi/build_results_dashboard_data.py",
)


def _fake_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        frequencies=np.asarray(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.2, 0.8],
            ],
            dtype=float,
        ),
        feature_ids=np.asarray(["T cell", "B cell"], dtype=object),
        group_ids=np.asarray(["r1", "r1", "r2", "r2"], dtype=object),
        outcomes=pd.DataFrame(
            {
                "primary_outcome": [0, 0, 1, 1],
                "recurrence": [1, 1, 0, 0],
            }
        ),
    )


def _fake_metrics(
    _X: np.ndarray,
    y: np.ndarray,
    *,
    classifier: str = "ridge_logistic",
) -> dict[str, float | int]:
    assert classifier in {"ridge_logistic", "random_forest"}
    return {
        "region_count": int(len(y)),
        "positive_count": int(np.sum(y)),
        "split_count": 25,
        "roc_auc_mean": 0.71,
        "roc_auc_std": 0.01,
        "pr_auc_mean": 0.63,
        "pr_auc_std": 0.02,
        "accuracy_mean": 0.68,
        "accuracy_std": 0.03,
        "balanced_accuracy_mean": 0.67,
        "balanced_accuracy_std": 0.03,
        "sensitivity_mean": 0.66,
        "sensitivity_std": 0.04,
        "specificity_mean": 0.69,
        "specificity_std": 0.04,
        "f1_mean": 0.65,
        "f1_std": 0.03,
        "brier_mean": 0.21,
        "brier_std": 0.01,
    }


def test_cell_type_composition_matches_topic_representation_contract() -> None:
    bundle = _fake_bundle()
    inverse = np.asarray([0, 0, 1, 1], dtype=int)
    counts = np.asarray([2.0, 2.0])

    soft = EVALUATOR._cell_type_composition(bundle, inverse, counts, "soft_mean")
    hard = EVALUATOR._cell_type_composition(
        bundle, inverse, counts, "published_hard"
    )

    np.testing.assert_allclose(soft, [[0.45, 0.55], [0.40, 0.60]])
    np.testing.assert_allclose(hard, [[0.50, 0.50], [0.50, 0.50]])


def test_log_composition_is_elementwise_log_after_floor_and_reclosure() -> None:
    composition = np.asarray([[0.0, 0.25, 0.75], [1.0, 0.0, 0.0]])
    transformed = EVALUATOR._log_composition(composition)

    assert transformed.shape == composition.shape
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(np.exp(transformed).sum(axis=1), 1.0)
    np.testing.assert_allclose(
        EVALUATOR._log_composition(np.ones((3, 1))),
        np.zeros((3, 1)),
    )


def test_cell_type_baseline_is_a_single_fixed_control_across_K(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _fake_bundle()
    monkeypatch.setattr(EVALUATOR, "load_real_data", lambda _dataset: bundle)
    monkeypatch.setattr(EVALUATOR, "_evaluate_binary", _fake_metrics)
    output_paths = {
        ("ilr", "ridge_logistic"): tmp_path / "crc_cell_type_outcome_rows.csv",
        ("log", "ridge_logistic"): tmp_path / "crc_cell_type_outcome_rows_log.csv",
        ("ilr", "random_forest"): tmp_path
        / "crc_cell_type_outcome_rows_ilr_random_forest.csv",
        ("log", "random_forest"): tmp_path
        / "crc_cell_type_outcome_rows_log_random_forest.csv",
    }
    frames = []
    for (transform, classifier), output in output_paths.items():
        EVALUATOR.evaluate_cell_type_baseline(
            output,
            K_values=(1, 2, 3, 4, 5, 6),
            feature_transform=transform,
            classifier=classifier,
        )
        frame = pd.read_csv(output)
        assert len(frame) == 24
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    assert set(rows["K"]) == {1, 2, 3, 4, 5, 6}
    assert set(rows["target"]) == {"primary_outcome", "recurrence"}
    assert set(rows["representation"]) == {"soft_mean", "published_hard"}
    assert set(rows["estimator_family"]) == {"cell_type_baseline"}
    assert set(rows["composition_dimension"]) == {2}
    assert set(rows.loc[rows["predictor_transform"] == "ilr", "predictor_dimension"]) == {1}
    assert set(rows.loc[rows["predictor_transform"] == "log", "predictor_dimension"]) == {2}
    assert set(rows["predictor_transform"]) == {"ilr", "log"}
    assert set(rows["classifier"]) == {"ridge_logistic", "random_forest"}
    assert rows.groupby(["K", "predictor_transform", "classifier"])["task_config_hash"].nunique().eq(1).all()

    summary, contract = DASHBOARD._load_crc_outcome_summary(tmp_path)
    assert len(summary) == 96
    assert summary["n_complete"].eq(1).all()
    assert summary["planned_n"].eq(1).all()
    assert summary["coverage"].eq(1.0).all()
    assert set(summary["predictor_transform"]) == {"ilr", "log"}
    assert set(summary["classifier"]) == {"ridge_logistic", "random_forest"}
    assert summary["variant_label"].eq(
        "Observed 3-hop cell-type composition (8 fixed topics; no topic fit)"
    ).all()
    assert contract["cell_type_baseline_status"] == "complete"
    assert contract["cell_type_baseline_K_values"] == [1, 2, 3, 4, 5, 6]
    assert contract["cell_type_baseline_predictor_dimension"] == 2
    assert contract["cell_type_baseline_composition_dimension"] == 2
    assert contract["cell_type_baseline_predictor_dimension_by_transform"] == {
        "ilr": 1,
        "log": 2,
    }
    assert contract["cell_type_baseline_status_by_transform"] == {
        "ilr": "complete",
        "log": "complete",
    }
    assert contract["cell_type_baseline_status_by_evaluation"] == {
        "ilr::ridge_logistic": "complete",
        "ilr::random_forest": "complete",
        "log::ridge_logistic": "complete",
        "log::random_forest": "complete",
    }
    assert contract["evaluated_tasks"] == 0


def test_dashboard_infers_ilr_for_legacy_rows_without_transform(
    tmp_path: Path,
) -> None:
    legacy = {
        "dataset": "stanford_crc_codex",
        "K": 3,
        "estimator_family": "plsi",
        "spectral_geometry": "native_baseline",
        "vertex_hunter": "not_applicable",
        "preprocessing": "native_baseline",
        "A_recovery": "A_current",
        "W_recovery_method": "native_baseline",
        "fit_id": "legacy-fit",
        "W_fit_id": "legacy-W",
        "task_config_hash": "legacy-task",
        "seed": 1,
        "evaluation_version": "region_rskf5x5_ilr_ridge_v1",
        "evaluated_at_utc": "2026-01-01T00:00:00+00:00",
        "target": "primary_outcome",
        "representation": "soft_mean",
        "region_count": 20,
        "positive_count": 8,
    }
    for name in (
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "brier",
    ):
        legacy[f"{name}_mean"] = 0.6
    pd.DataFrame([legacy]).to_csv(tmp_path / "crc_outcome_rows.csv", index=False)

    summary, contract = DASHBOARD._load_crc_outcome_summary(tmp_path)

    assert len(summary) == 1
    assert summary.iloc[0]["predictor_transform"] == "ilr"
    assert contract["predictor_transforms"] == ["ilr"]
    assert contract["status_by_transform"] == {"ilr": "partial", "log": "pending"}


def test_dashboard_keeps_classifiers_separate_for_the_same_W(
    tmp_path: Path,
) -> None:
    rows = []
    for classifier, value, version in (
        ("ridge_logistic", 0.41, "region_rskf5x5_log_floor1e-8_ridge_v1"),
        ("random_forest", 0.79, "region_rskf5x5_log_floor1e-8_rf100_v1"),
    ):
        row = {
            "dataset": "stanford_crc_codex",
            "K": 3,
            "estimator_family": "document_gplsi",
            "spectral_geometry": "document_U",
            "vertex_hunter": "spa_current",
            "preprocessing": "P0_raw",
            "A_recovery": "A_current",
            "W_recovery_method": "stable_document_vertex_solve",
            "fit_id": "same-fit",
            "W_fit_id": "same-W",
            "task_config_hash": "same-task",
            "seed": 1,
            "evaluation_version": version,
            "evaluated_at_utc": "2026-01-01T00:00:00+00:00",
            "predictor_transform": "log",
            "classifier": classifier,
            "target": "primary_outcome",
            "representation": "soft_mean",
            "region_count": 20,
            "positive_count": 8,
        }
        for name in (
            "roc_auc",
            "pr_auc",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "brier",
        ):
            row[f"{name}_mean"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "crc_outcome_rows_log.csv", index=False)

    summary, _ = DASHBOARD._load_crc_outcome_summary(tmp_path)

    assert len(summary) == 2
    observed = dict(zip(summary["classifier"], summary["accuracy_mean"], strict=True))
    assert observed == {"ridge_logistic": 0.41, "random_forest": 0.79}
