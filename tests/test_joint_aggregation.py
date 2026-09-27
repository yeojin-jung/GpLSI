import numpy as np
import pytest

from gplsi_joint_v2.aggregation import (audit_result_index, paired_method_contrast,
                                      aggregate_topic_summaries)


def records_fixture():
    result = []
    # Subject one has many repeats; the subject-balanced contrast remains two,
    # not the repeat-weighted contrast close to one.
    for bio, repeats, difference in [("one", 10, 1), ("two", 1, 3)]:
        for split in range(repeats):
            for method, value in [("A", 5 + difference), ("B", 5)]:
                result.append({"biological_id": bio, "split": split, "method": method,
                               "success": True, "converged": True, "deviance_per_molecule": value,
                               "evaluation_vocabulary_sha256": "same_genes", "evaluation_mask_sha256": "same_mask"})
    return result


def test_paired_bootstrap_averages_repeats_before_resampling_subjects():
    result = paired_method_contrast(records_fixture(), "A", "B", match_fields=["split"], n_bootstrap=500)
    assert result["mean_paired_difference"] == 2
    assert result["biological_units"] == 2
    assert result["matched_successful_evaluations"] == 11
    assert result["confidence_interval_95"] == [1, 3]
    assert result["subjects_not_repeated_splits_are_resampled"]
    assert not result["training_and_tuning_refit_uncertainty"]
    assert result["support_hash_compatibility_verified"]


def test_missing_failure_nonconvergence_and_target_mismatch_are_audited():
    records = records_fixture()
    expected = [{"biological_id": r["biological_id"], "split": r["split"]} for r in records if r["method"] == "A"]
    records.pop(0)
    records[2]["success"] = False
    records[4]["converged"] = False
    records[6]["evaluation_mask_sha256"] = "different_rows"
    result = paired_method_contrast(records, "A", "B", match_fields=["split"], expected_pairs=expected,
                                     require_converged=True, n_bootstrap=20)
    assert result["expected_paired_evaluations"] == 11
    assert result["matched_successful_evaluations"] == 7
    assert result["expected_denominator_source"] == "frozen_manifest"
    assert result["exclusion_counts"] == {"missing_method_result": 1,
        "estimator_or_inference_failure": 1, "convergence_restricted_exclusion": 1,
        "incompatible_evaluation_mask_sha256": 1}


def test_infinite_support_scores_are_not_dropped_from_biological_contrasts():
    records = records_fixture()
    records[0]["deviance_per_molecule"] = np.inf
    result = paired_method_contrast(records, "A", "B", match_fields=["split"])
    assert result["mean_paired_difference"] == np.inf
    assert result["status"] == "infinite_contrast_retained"
    assert result["matched_successful_evaluations"] == 11
    assert result["confidence_interval_95"] is None
    records[1]["deviance_per_molecule"] = np.inf
    result = paired_method_contrast(records, "A", "B", match_fields=["split"])
    assert np.isnan(result["mean_paired_difference"])
    assert result["status"] == "undefined_infinite_contrast_retained"
    assert result["matched_successful_evaluations"] == 11


def test_full_result_index_audits_duplicate_missing_and_failed_records():
    expected = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    observed = [{"id": "a", "status": "ok", "success": True},
                {"id": "a", "status": "ok", "success": True},
                {"id": "b", "status": "failed", "success": False, "converged": False}]
    result = audit_result_index(expected, observed, key_fields=["id"])
    assert result["duplicate_keys"] == [("a",)]
    assert result["missing_keys"] == [("c",)]
    assert result["explicit_nonconvergence"] == 1
    assert not result["complete_identity_inventory"]
    assert not result["all_successful"]
    with pytest.raises(ValueError, match="duplicate"):
        paired_method_contrast(records_fixture() * 2, "A", "B", match_fields=["split"])


def test_equal_section_unit_summaries_keep_patient_timepoints_blocked():
    w = np.array([[1, 0]] * 10 + [[0, 1]] + [[.2, .8]] * 3)
    biological = np.full(14, "patient1")
    sections = ["s1"] * 10 + ["s2"] + ["s3"] * 3
    conditions = ["pre"] * 11 + ["post"] * 3
    result = aggregate_topic_summaries(w, biological, sections, condition_ids=conditions)
    assert result["units"][0]["biological_id"] == result["units"][1]["biological_id"] == "patient1"
    assert [u["condition_id"] for u in result["units"]] == ["post", "pre"]
    assert np.allclose(result["W_summary"][1], [.5, .5])
    assert not result["changes_training_likelihood_weights"]
