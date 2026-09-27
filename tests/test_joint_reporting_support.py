"""Metadata-only regression tests for fail-closed paired-score support checks."""
from copy import deepcopy

import numpy as np
import pytest

from gplsi_joint_v2.aggregation import paired_method_contrast


SUPPORT = ("evaluation_vocabulary_sha256", "evaluation_mask_sha256")


def records_fixture():
    return [{"method": method, "biological_id": biological, "split": "frozen_split",
             "success": True, "converged": True, "deviance_per_molecule": value,
             "evaluation_vocabulary_sha256": "frozen_vocab_digest",
             "evaluation_mask_sha256": "frozen_mask_digest"}
            for biological, difference in [("one", 1.), ("two", 2.), ("three", 3.)]
            for method, value in [("A", 5. + difference), ("B", 5.)]]


def compare(records, **kwargs):
    expected = [{"biological_id": row["biological_id"], "split": row["split"]}
                for row in records if row["method"] == "A"]
    return paired_method_contrast(records, "A", "B", match_fields=("split",),
                                  expected_pairs=expected, n_bootstrap=64, **kwargs)


@pytest.mark.parametrize("field", SUPPORT)
@pytest.mark.parametrize("absent", ["missing", None, "", " \t\n"])
@pytest.mark.parametrize("side", ["A", "B"])
def test_missing_or_empty_requested_hash_produces_no_estimate_or_uncertainty(field, absent, side):
    records = records_fixture()
    for row in records:
        if row["method"] == side:
            if absent == "missing":
                row.pop(field)
            else:
                row[field] = absent
    result = compare(records)
    assert result["expected_paired_evaluations"] == 3
    assert result["matched_successful_evaluations"] == result["biological_units"] == 0
    assert result["paired_evaluations"] == result["per_biological_unit"] == []
    assert np.isnan(result["mean_paired_difference"])
    assert result["bootstrap_standard_error"] is result["confidence_interval_95"] is None
    assert result["status"] == "no_common_successful_support"
    assert result["exclusion_counts"] == {"missing_or_empty_" + field: 3}
    assert result["support_hash_compatibility_verified"] is False
    assert result["support_hash_verification_requested"] is True


@pytest.mark.parametrize("invalid", [0, False, [], {}, float("nan")])
def test_nonstring_hash_is_not_treated_as_verified_support(invalid):
    records = records_fixture()
    for row in records:
        row[SUPPORT[0]] = deepcopy(invalid)
    result = compare(records)
    assert result["exclusion_counts"] == {"invalid_" + SUPPORT[0]: 3}
    assert result["matched_successful_evaluations"] == 0
    assert np.isnan(result["mean_paired_difference"])
    assert result["bootstrap_standard_error"] is None


def test_only_verified_pairs_contribute_to_estimate_or_bootstrap():
    records = records_fixture()
    records[0].pop(SUPPORT[1])
    records[0]["deviance_per_molecule"] = 1e12
    result = compare(records)
    assert result["matched_successful_evaluations"] == result["biological_units"] == 2
    assert result["mean_paired_difference"] == 2.5
    assert {row["biological_id"] for row in result["paired_evaluations"]} == {"two", "three"}
    assert result["confidence_interval_95"][0] >= 2.
    assert result["confidence_interval_95"][1] <= 3.
    assert result["exclusion_counts"] == {"missing_or_empty_" + SUPPORT[1]: 1}


def test_present_mismatch_reason_and_verified_infinite_scores_are_preserved():
    records = records_fixture()
    records[0][SUPPORT[1]] = "another_nonempty_digest"
    records[2]["deviance_per_molecule"] = np.inf
    result = compare(records)
    assert result["exclusion_counts"] == {"incompatible_" + SUPPORT[1]: 1}
    assert result["matched_successful_evaluations"] == 2
    assert result["mean_paired_difference"] == np.inf
    assert result["status"] == "infinite_contrast_retained"
    assert result["bootstrap_standard_error"] is None


def test_empty_support_fields_is_explicit_optout_not_false_verification_claim():
    records = records_fixture()
    for row in records:
        for field in SUPPORT:
            row.pop(field)
    result = compare(records, support_fields=())
    assert result["mean_paired_difference"] == 2.
    assert result["matched_successful_evaluations"] == 3
    assert result["support_hash_verification_requested"] is False
    assert result["support_hash_compatibility_verified"] is False
    assert result["requested_support_fields"] == []


def test_default_requested_fields_and_complete_verified_support():
    result = compare(records_fixture())
    assert result["requested_support_fields"] == list(SUPPORT)
    assert result["support_hash_compatibility_verified"] is True
    assert result["mean_paired_difference"] == 2.
    assert result["exclusion_counts"] == {}
