"""Metadata-only regressions for the separately versioned reporting revision."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

import gplsi_joint_v2.results_reporting as reporting
from gplsi_joint_v2.artifacts import sha256_file
from gplsi_joint_v2.config import fingerprint


REFERENCE = "gplsi_document__P0_raw__spa_current__selected"


def task(method=REFERENCE, seed=1, dataset="xenium_uc"):
    spec = {"dataset": dataset, "protocol": "patient_holdout", "outer_split_id": "patient_split_01",
            "K": 12, "panel_requested": 290, "retention": 1.0, "molecule_seed": 7,
            "estimator_seed": seed, "method": method, "recovery": "A_full_Pois", "vocabulary": "native"}
    return {"task_id": method + "_" + str(seed), "stage": "evaluation", "dataset": dataset,
            "variant": method, "spec": spec, "parents": [], "base_ids": ["base"]}


def payload(biological_ids=("patient0", "patient1"), value=2.0, *, eligible=5, converged=5):
    return {"per_biological_unit": [{"biological_id": bio, "observations": 2, "molecules": 10,
              "deviance_per_molecule": value, "log_likelihood_per_molecule": -value,
              "inference_failures": 0} for bio in biological_ids],
            "evaluation_vocabulary_sha256": "same-ordered-genes", "evaluation_mask_sha256": "same-count-mask",
            "foldin": {"count_eligible_rows": eligible}, "foldin_converged_observations": converged,
            "foldin_status_counts": {"ok": converged, "max_iter_reached": eligible - converged,
                                     "no_adaptation_counts": 2}}


def evaluation(*, converged=5):
    return {"within_training_molecule_prediction": payload(), "transfer": {"unseen_patient": payload(converged=converged)},
            "fit_converged": True, "fit_status": "ok", "training_coverage": {"panel_actual": 290}}


def records(method=REFERENCE, seed=1, *, output="complete"):
    value = evaluation() if output == "complete" else None
    return reporting.flatten_evaluation(task(method, seed), value,
        {"within_training": ["patient0", "patient1"], "unseen_patient": ["patient0", "patient1"]},
        {"base": ["stability"]}, state=output)


def test_manifest_seed_intersection_does_not_call_unrequested_spatial_lda_seeds_missing():
    baseline = sum([records(seed=seed) for seed in (1, 2, 3)], [])
    candidate = records("spatial_lda_tuned", 1)
    contrasts = reporting.paired_contrasts(baseline + candidate, bootstrap_draws=20)
    assert len(contrasts) == 4
    for row in contrasts:
        assert row["expected_paired_evaluations"] == row["matched_successful_evaluations"] == 2
        assert row["planned_reference_evaluations"] == 6
        assert row["planned_candidate_evaluations"] == 2
        assert row["manifest_scope_exclusion_counts"] == {"not_requested_for_candidate": 4, "not_requested_for_reference": 0}
        assert row["unrequested_estimator_seed_evaluations"]["not_requested_for_candidate"] == {"2": 2, "3": 2}
        assert row["manifest_scope_exclusions_are_failures"] is False
        assert row["exclusion_counts"] == {}


@pytest.mark.parametrize("state", ["failed", "pending_or_not_submitted", "blocked_parent"])
def test_planned_failed_or_missing_placeholders_remain_in_common_denominator(state):
    baseline = sum([records(seed=seed) for seed in (1, 2, 3)], [])
    candidate = records("spatial_lda_tuned", 1, output=state)
    contrasts = reporting.paired_contrasts(baseline + candidate, bootstrap_draws=20)
    for row in contrasts:
        assert row["expected_paired_evaluations"] == 2
        assert row["matched_successful_evaluations"] == 0
        assert row["exclusion_counts"] == {"estimator_or_inference_failure": 2}
        assert row["manifest_scope_exclusion_counts"]["not_requested_for_candidate"] == 4
        assert row["status"] == "no_common_successful_support"


def test_common_expected_infinite_scores_are_not_removed():
    baseline, candidate = records(), records("candidate")
    for row in candidate:
        row["deviance_per_molecule"] = np.inf
    contrasts = reporting.paired_contrasts(baseline + candidate, bootstrap_draws=20)
    assert all(row["matched_successful_evaluations"] == 2 for row in contrasts)
    assert all(row["mean_paired_difference"] == np.inf for row in contrasts)
    assert all(row["bootstrap_standard_error"] is None for row in contrasts)


def test_duplicate_planned_identity_rejected_even_outside_common_seed_scope():
    duplicate = records(seed=2)
    with pytest.raises(ValueError, match="duplicate manifest-expected"):
        reporting.paired_contrasts(records() + duplicate + duplicate + records("spatial_lda_tuned"))


def test_all_adaptation_eligible_rows_must_converge_even_without_score_counts():
    value = evaluation(converged=4)  # Four scored rows, five eligible adaptation rows.
    rows = reporting.flatten_evaluation(task(), value,
        {"within_training": ["patient0", "patient1"], "unseen_patient": ["patient0", "patient1"]}, {"base": ["core"]})
    training = [row for row in rows if row["endpoint"] == "within_training"]
    transfer = [row for row in rows if row["endpoint"] == "unseen_patient"]
    assert all(row["fit_converged"] and row["converged"] for row in training)
    assert all(row["foldin_required"] is False and row["foldin_converged"] is None for row in training)
    assert all(row["success"] and row["fit_converged"] and not row["converged"] for row in transfer)
    assert all(row["foldin_converged"] is False for row in transfer)
    summaries, _ = reporting.summarize_scores(rows, bootstrap_draws=20, require_converged=True)
    restricted = next(row for row in summaries if row["endpoint"] == "unseen_patient")
    assert restricted["successful_biological_evaluations"] == 0
    assert restricted["fit_nonconverged_evaluations"] == 0
    assert restricted["foldin_nonconverged_evaluations"] == 2


def test_zero_adaptation_rows_do_not_prevent_endpoint_convergence():
    result = reporting._endpoint_foldin_convergence(payload(), "unseen_patient")
    assert result["foldin_converged"] is True
    assert result["foldin_count_eligible_observations"] == 5


@pytest.mark.parametrize("missing", ["foldin", "foldin_status_counts", "foldin_converged_observations"])
def test_missing_foldin_evidence_is_unknown_not_converged(missing):
    value = evaluation()
    del value["transfer"]["unseen_patient"][missing]
    row = reporting.flatten_evaluation(task(), value, {"unseen_patient": ["patient0", "patient1"]}, {"base": ["core"]})[0]
    assert row["fit_converged"] is True and row["success"] is True
    assert row["foldin_converged"] is None and row["converged"] is False
    assert row["foldin_convergence_status"] == "unknown_missing_evidence"


def test_inconsistent_foldin_counts_do_not_certify_convergence():
    value = payload()
    value["foldin_status_counts"]["max_iter_reached"] = 1
    result = reporting._endpoint_foldin_convergence(value, "unseen_patient")
    assert result["foldin_converged"] is None
    assert result["foldin_convergence_status"] == "unknown_inconsistent_evidence"


@pytest.mark.parametrize("second_converged,unknown,expected", [(5, False, True), (4, False, False), (5, True, None)])
def test_combined_visium_endpoint_requires_both_role_certificates(second_converged, unknown, expected):
    value = evaluation()
    value["transfer"] = {"primary_test": payload(), "additional_test": payload(converged=second_converged)}
    if unknown:
        del value["transfer"]["additional_test"]["foldin_status_counts"]
    fit = task(dataset="visium_dlpfc")
    fit["spec"]["protocol"] = "section_holdout"
    ids = ["patient0", "patient1"]
    rows = reporting.flatten_evaluation(fit, value,
        {"primary_test": ids, "additional_test": ids, "section_transfer_equal_section_combined": ids}, {"base": ["core"]})
    combined = [row for row in rows if row["endpoint"] == "section_transfer_equal_section_combined"]
    assert all(row["foldin_converged"] is expected for row in combined)
    assert all(row["converged"] is (expected is True) for row in combined)
    assert all(row["fit_converged"] is True for row in combined)
    assert all(set(row["foldin_component_convergence"]) == {"primary_test", "additional_test"} for row in combined)


@pytest.mark.parametrize("bad_hash", [None, "", "   ", 123])
def test_combined_endpoint_does_not_manufacture_missing_support_hash(bad_hash):
    parts = [{"hash": "valid"}, {"hash": bad_hash}]
    assert reporting._combined_support_hash(parts, "hash") is None


def test_plot_series_keeps_gaps_and_annotates_unavailable_and_partial_failure():
    rows = [{"K": k, "equal_biological_deviance": value,
             "expected_biological_evaluations": 2, "successful_biological_evaluations": available,
             "failure_status_counts": {"failed": 2 - available} if available < 2 else {}}
            for k, value, available in [(7, 1.0, 2), (10, np.nan, 0), (12, np.inf, 2), (15, 2.0, 1)]]
    x, y, notes = reporting._plot_score_series(rows, "K")
    assert x == [7, 10, 12, 15]
    assert y[0] == 1.0 and y[-1] == 2.0 and np.isnan(y[1:3]).all()
    assert any("unavailable/NaN at K=10" in note for note in notes)
    assert any("infinite at K=12" in note for note in notes)
    assert any("10: 0/2" in note and "15: 1/2" in note and "failed" in note for note in notes)


def test_reporting_identity_hashes_actual_code_caller_cannot_override(monkeypatch):
    def fake_git(command, **kwargs):
        text = "reporting-commit\n" if "rev-parse" in command else "results_reporting.py\naggregation.py\n" if "ls-files" in command else ""
        return SimpleNamespace(returncode=0, stdout=text)
    monkeypatch.setattr(reporting.subprocess, "run", fake_git)
    identity = reporting._reporting_identity({"git_commit": "not-the-actual-commit", "code_digest": "wrong"})
    assert identity["git_commit"] == "reporting-commit"
    assert identity["reporting_modules_tracked"] is True
    assert identity["package_python_sha256"]["results_reporting.py"] == sha256_file(Path(reporting.__file__))
    assert identity["code_digest"] == fingerprint(identity["package_python_sha256"])
    assert identity["caller_metadata"]["code_digest"] == "wrong"
    assert identity["fitting_or_scoring_recomputed"] is False


def test_report_status_separates_frozen_model_provenance_and_versioned_output(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    manifest = root / "data/manifests/joint_v2/revision_case/dag.json"
    manifest.parent.mkdir(parents=True)
    original = {"tasks": [], "bases": [], "base_memberships": {}, "config": {"bootstrap": {"draws": 20}},
                "code_hash": "frozen-fit-code", "config_hash": "frozen-fit-config",
                "source_hashes": {"source_commit": "2be1800-frozen"}}
    manifest.write_text(json.dumps(original))
    frozen_output = root / "reports/joint_v2/revision_case/scientific/REPORT_STATUS.json"
    frozen_output.parent.mkdir(parents=True)
    frozen_output.write_text("do not overwrite")
    identity = {"git_commit": "new-reporting-commit", "code_digest": "a" * 64, "revision": "test"}
    monkeypatch.setattr(reporting, "_reporting_identity", lambda caller: deepcopy(identity))
    monkeypatch.setattr(reporting, "_write_table", lambda *args, **kwargs: None)
    result = reporting.report_results(root, manifest, run_profiles=False, make_plots=False)
    saved = json.loads((Path(result["tables_directory"]) / "REPORT_STATUS.json").read_text())
    assert saved["reporting_provenance"] == identity
    assert saved["model_manifest_provenance"]["code_hash"] == "frozen-fit-code"
    assert saved["model_manifest_provenance"]["config_hash"] == "frozen-fit-config"
    assert saved["model_manifest_provenance"]["manifest_sha256"] == sha256_file(manifest)
    assert saved["model_manifest_provenance"]["scientific_task_identities_changed"] is False
    assert saved["output_tag"] == "scientific_reporting_" + "a" * 12
    assert Path(saved["tables_directory"]).name == saved["output_tag"]
    assert frozen_output.read_text() == "do not overwrite"
    assert json.loads(manifest.read_text()) == original
