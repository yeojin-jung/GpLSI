from pathlib import Path
import json
import numpy as np

from gplsi_joint_v2.config import default_config
from gplsi_joint_v2.artifacts import atomic_json, commit_stage
from gplsi_joint_v2.results_reporting import (flatten_evaluation, expected_endpoint_subjects,
                                            summarize_scores, paired_contrasts, report_results)


def split_fixture():
    return {"split_id": "section_rotation_01", "protocol": "section_holdout",
            "assignments": [{"bio_id": bio, "train_sections": [bio + "a1", bio + "b1"],
                             "primary_test": [bio + "a2"], "additional_test": [bio + "b2"]}
                            for bio in ["donor1", "donor2", "donor3"]]}


def task_fixture(method="gplsi_document__P0_raw__spa_current__selected", key="evaluation_a"):
    spec = {"dataset": "visium_dlpfc", "protocol": "section_holdout", "outer_split_id": "section_rotation_01",
            "K": 7, "panel_requested": 2000, "retention": 1.0, "molecule_seed": 7, "estimator_seed": 8,
            "method": method, "recovery": "A_full_Pois", "vocabulary": "native", "control": "selected",
            "preprocessing": "P0_raw", "family": "document", "hunter": "spa_current"}
    return {"task_id": key, "stage": "evaluation", "dataset": "visium_dlpfc", "variant": method,
            "spec": spec, "parents": ["prepare"], "base_ids": ["base"]}


def evaluation_fixture(add=0):
    def role(value, mass, name):
        return {"per_biological_unit": [{"biological_id": bio, "observations": 2,
                    "molecules": mass, "deviance_per_molecule": value + add,
                    "log_likelihood_per_molecule": -value - add, "inference_failures": 0}
                    for bio in ["donor1", "donor2", "donor3"]],
                "evaluation_vocabulary_sha256": "genes", "evaluation_mask_sha256": name,
                "foldin_runtime_seconds": 3, "foldin_converged_observations": 6,
                "foldin": {"count_eligible_rows": 6}, "foldin_status_counts": {"ok": 6}}
    return {"within_training_molecule_prediction": role(1, 50, "train"),
            "transfer": {"primary_test": role(2, 10, "primary"), "additional_test": role(10, 90, "additional")},
            "fit_converged": True, "fit_status": "ok", "training_coverage": {"panel_actual": 1978}}


def test_expected_missing_scores_retained_and_equal_section_combination():
    expected = expected_endpoint_subjects(split_fixture())
    complete = flatten_evaluation(task_fixture(), evaluation_fixture(), expected, {"base": ["core"]})
    assert len(complete) == 12
    combined = [r for r in complete if r["endpoint"] == "section_transfer_equal_section_combined"]
    assert all(r["deviance_per_molecule"] == 6 for r in combined)
    assert all(r["deviance_total"] == 920 for r in combined)
    assert all(r["panel_actual"] == 1978 for r in complete)
    missing = flatten_evaluation(task_fixture(), None, expected, {"base": ["core"]}, state="blocked_parent")
    assert len(missing) == 12
    assert not any(r["success"] for r in missing)
    summaries, units = summarize_scores(complete)
    combined_summary = next(r for r in summaries if r["endpoint"] == "section_transfer_equal_section_combined")
    assert combined_summary["equal_biological_deviance"] == 6
    assert combined_summary["molecule_weighted_deviance"] == 9.2
    assert all(r["bootstrap_standard_error"] is None for r in summaries)
    assert len(combined_summary["biological_unit_values"]) == 3


def test_json_infinite_scores_are_scientific_results_and_not_dropped():
    evaluation = evaluation_fixture()
    evaluation["transfer"]["primary_test"]["per_biological_unit"][0]["deviance_per_molecule"] = "Infinity"
    records = flatten_evaluation(task_fixture(), evaluation, expected_endpoint_subjects(split_fixture()), {"base": ["core"]})
    primary = [r for r in records if r["endpoint"] == "primary_test"]
    assert all(r["success"] for r in primary)
    assert primary[0]["status"] == "support_failure"
    summaries, _ = summarize_scores(records)
    summary = next(r for r in summaries if r["endpoint"] == "primary_test")
    assert summary["equal_biological_deviance"] == np.inf
    assert summary["successful_biological_evaluations"] == 3
    assert summary["infinite_score_evaluations"] == 1


def test_paired_contrasts_use_frozen_support_and_never_visium_repeat_se():
    expected = expected_endpoint_subjects(split_fixture())
    baseline = flatten_evaluation(task_fixture(), evaluation_fixture(), expected, {"base": ["core"]})
    candidate = flatten_evaluation(task_fixture("candidate", "evaluation_b"), evaluation_fixture(add=1), expected, {"base": ["core"]})
    output = paired_contrasts(baseline + candidate, bootstrap_draws=100)
    assert len(output) == 8  # four endpoints, unrestricted and convergence-restricted
    assert all(row["mean_paired_difference"] == 1 for row in output)
    assert all(row["bootstrap_standard_error"] is None for row in output)
    assert all(row["expected_paired_evaluations"] == 3 for row in output)
    assert all(row["support_hash_compatibility_verified"] for row in output)
    assert all("paired_evaluations" not in row for row in output)
    candidate[0]["evaluation_mask_sha256"] = "different_count_mask"
    output = paired_contrasts(baseline + candidate, bootstrap_draws=100)
    changed = [row for row in output if row["endpoint"] == "within_training"]
    assert all(row["matched_successful_evaluations"] == 2 for row in changed)


def test_report_partial_manifest_complete_index_compact_storage(tmp_path):
    root = tmp_path / "benchmark"
    manifest_path = root / "data/manifests/joint_v2/test_run/dag.json"
    manifest_path.parent.mkdir(parents=True)
    split_directory = root / "data/manifests/joint_v2/splits/visium_dlpfc"
    atomic_json(split_directory / "splits.json", [split_fixture()])
    atomic_json(split_directory / "inventory.json", {"units": [{"bio_id": d} for d in ["donor1", "donor2", "donor3"]]})
    first, missing = task_fixture(), task_fixture("candidate", "evaluation_b")
    prepare = {"task_id": "prepare", "stage": "prepare", "dataset": "visium_dlpfc", "variant": "prepare",
               "spec": first["spec"], "parents": [], "base_ids": ["base"]}
    atomic_json(manifest_path, {"config": default_config(), "base_memberships": {"base": ["core"]},
                "bases": [{"base_id": "base", "dataset": "visium_dlpfc"}], "tasks": [prepare, first, missing]})
    for task, prefix in [(prepare, "data/interim/joint_v2/stages"), (first, "results/joint_v2/stages")]:
        directory = root / prefix / task["stage"] / task["task_id"]
        eff = {"runtime_seconds": 5, "peak_rss_bytes": 1024, "slurm_job_id": "test_only"}
        atomic_json(directory / "effective.json", eff)
        files = ["effective.json"]
        if task["stage"] == "evaluation":
            atomic_json(directory / "evaluation.json", evaluation_fixture())
            files.append("evaluation.json")
        commit_stage(directory, task["task_id"], eff, files)
    report = report_results(root, manifest_path, run_profiles=False, make_plots=False)
    assert report["expected_stages"] == 3
    assert report["stage_state_counts"] == {"complete": 2, "pending_or_not_submitted": 1}
    assert report["expected_biological_score_rows_including_tier_membership"] == 24
    assert report["successful_biological_score_rows"] == 12
    output = Path(report["tables_directory"])
    assert (output / "result_index.parquet").exists()
    assert not (output / "result_index.csv").exists()
    assert (output / "method_summaries.csv").exists()
    assert (output / "REPORT_STATUS.json").exists()
