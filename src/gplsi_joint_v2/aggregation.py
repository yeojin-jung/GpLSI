"""Paired biological-unit summaries with explicitly conditional uncertainty.

These functions consume saved scores. They never refit models and therefore
cannot estimate uncertainty from repeated training/tuning. Missing expected
tasks, incompatible score targets, nonconvergence and infinite scores remain
in the returned audit rather than disappearing through nanmean/dropna.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np


def _key(record, fields):
    absent = [field for field in fields if field not in record]
    if absent:
        raise ValueError(f"record is missing required key fields {absent}")
    return tuple(str(record[field]) for field in fields)


def audit_result_index(expected, observed, *, key_fields):
    expected_keys = [_key(row, key_fields) for row in expected]
    observed_keys = [_key(row, key_fields) for row in observed]
    expected_counts, observed_counts = Counter(expected_keys), Counter(observed_keys)
    if any(n != 1 for n in expected_counts.values()):
        raise ValueError("expected scientific manifest contains duplicate identities")
    duplicates = [key for key, count in observed_counts.items() if count > 1]
    expected_set, observed_set = set(expected_keys), set(observed_keys)
    statuses = Counter(str(row.get("status", "status_missing")) for row in observed)
    return {"expected_records": len(expected_keys), "observed_records": len(observed_keys),
            "unique_observed_records": len(observed_set),
            "missing_keys": sorted(expected_set - observed_set),
            "unexpected_keys": sorted(observed_set - expected_set), "duplicate_keys": sorted(duplicates),
            "status_counts": dict(statuses),
            "explicit_nonconvergence": sum(row.get("converged") is False for row in observed),
            "complete_identity_inventory": expected_set == observed_set and not duplicates,
            "all_successful": expected_set == observed_set and not duplicates and all(row.get("success") is True for row in observed)}


def paired_method_contrast(records, method_a, method_b, *, match_fields,
                           value_field="deviance_per_molecule", biological_field="biological_id",
                           expected_pairs=None, support_fields=("evaluation_vocabulary_sha256", "evaluation_mask_sha256"),
                           require_converged=False, n_bootstrap=2000, seed=260910):
    """Average matched evaluations within a subject, then contrast subjects.

    Each record must explicitly contain method, success, biological_id, the
    declared matching fields and value_field. A support failure is success=True
    with an infinite score, not a missing fit. `expected_pairs`, when supplied,
    is the full expected table of matching keys and subjects from the manifest.
    Convergence-restricted sensitivity requires converged=True in both records.
    Requested support hashes must be nonempty strings in both records and
    match exactly. Missing/blank/invalid hashes exclude that pair before any
    estimate or bootstrap. ``support_fields=()`` is an explicit verification
    opt-out, never a claim that support-hash compatibility was checked.
    """
    if method_a == method_b or n_bootstrap < 1:
        raise ValueError("distinct methods and positive bootstrap count required")
    fields = tuple(match_fields) + (biological_field,)
    if len(set(fields)) != len(fields):
        raise ValueError("biological key must not also appear among matching fields")
    by_method = {method_a: {}, method_b: {}}
    for record in records:
        method = record.get("method")
        if method not in by_method:
            continue
        if "success" not in record or value_field not in record:
            raise ValueError("every score record needs explicit success and score fields")
        key = _key(record, fields)
        if key in by_method[method]:
            raise ValueError(f"duplicate score identity for {method}: {key}")
        by_method[method][key] = record
    observed_union = set(by_method[method_a]) | set(by_method[method_b])
    expected = observed_union if expected_pairs is None else {_key(row, fields) for row in expected_pairs}
    if expected_pairs is not None and len(expected) != len(expected_pairs):
        raise ValueError("expected paired score manifest contains duplicate identities")
    unexpected = observed_union - expected
    if unexpected:
        raise ValueError(f"scores outside expected comparison manifest: {sorted(unexpected)[:3]}")
    support_fields = tuple(support_fields)
    audit, paired, support_verified = [], [], bool(support_fields)
    for key in sorted(expected):
        left, right = by_method[method_a].get(key), by_method[method_b].get(key)
        reason = None
        if left is None or right is None:
            reason = "missing_method_result"
        elif not left["success"] or not right["success"]:
            reason = "estimator_or_inference_failure"
        elif require_converged and (left.get("converged") is not True or right.get("converged") is not True):
            reason = "convergence_restricted_exclusion"
        else:
            for field in support_fields:
                values = (left.get(field), right.get(field))
                if any(value is None or (isinstance(value, str) and not value.strip()) for value in values):
                    support_verified = False
                    reason = f"missing_or_empty_{field}"
                    break
                if any(not isinstance(value, str) for value in values):
                    support_verified = False
                    reason = f"invalid_{field}"
                    break
                if left[field] != right[field]:
                    reason = f"incompatible_{field}"
                    break
        if reason is None:
            a, b = float(left[value_field]), float(right[value_field])
            if np.isnan(a) or np.isnan(b):
                reason = "score_unavailable"
            else:
                # inf-inf is undefined and retained; it is not an exclusion.
                delta = np.nan if np.isinf(a) and a == b else a - b
                paired.append({"key": key, "biological_id": str(left[biological_field]),
                               "score_a": a, "score_b": b, "difference_a_minus_b": delta,
                               "converged_a": left.get("converged"), "converged_b": right.get("converged"),
                               "scored_observations_a": left.get("scored_observations"),
                               "scored_observations_b": right.get("scored_observations"),
                               "scored_molecules_a": left.get("scored_molecules"),
                               "scored_molecules_b": right.get("scored_molecules")})
        if reason is not None:
            audit.append({"key": key, "reason": reason})
    groups = defaultdict(list)
    for record in paired:
        groups[record["biological_id"]].append(record)
    per_unit = []
    for unit in sorted(groups):
        rows = groups[unit]
        differences = np.asarray([row["difference_a_minus_b"] for row in rows])
        per_unit.append({"biological_id": unit, "matched_evaluations": len(rows),
                         "mean_a": float(np.mean([row["score_a"] for row in rows])),
                         "mean_b": float(np.mean([row["score_b"] for row in rows])),
                         "mean_difference_a_minus_b": float(np.mean(differences)),
                         "repeated_evaluation_difference_min": float(np.min(differences)),
                         "repeated_evaluation_difference_max": float(np.max(differences)),
                         "repeated_evaluations_are_independent_subjects": False})
    values = np.asarray([row["mean_difference_a_minus_b"] for row in per_unit])
    estimate = float(np.mean(values)) if len(values) else np.nan
    interval, standard_error = None, None
    if not len(values):
        status = "no_common_successful_support"
    elif np.any(np.isnan(values)):
        status = "undefined_infinite_contrast_retained"
    elif np.any(~np.isfinite(values)):
        status = "infinite_contrast_retained"
    elif len(values) < 2:
        status = "insufficient_biological_units_for_uncertainty"
    else:
        status = "ok"
        rng = np.random.default_rng(seed)
        sample = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
        interval = np.quantile(sample, [.025, .975]).tolist()
        standard_error = float(sample.std(ddof=1)) if n_bootstrap > 1 else None
    return {"method_a": method_a, "method_b": method_b, "difference_direction": "a_minus_b",
            "value_field": value_field, "status": status, "mean_paired_difference": estimate,
            "biological_units": len(values), "per_biological_unit": per_unit,
            "expected_paired_evaluations": len(expected), "matched_successful_evaluations": len(paired),
            "expected_denominator_source": "frozen_manifest" if expected_pairs is not None else "observed_union_manifest_incomplete",
            "support_hash_compatibility_verified": support_verified and bool(paired),
            "support_hash_verification_requested": bool(support_fields),
            "requested_support_fields": list(support_fields),
            "excluded_evaluations": audit, "exclusion_counts": dict(Counter(row["reason"] for row in audit)),
            "paired_evaluations": paired, "require_converged": require_converged,
            "confidence_interval_95": interval, "bootstrap_standard_error": standard_error,
            "bootstrap_replicates": n_bootstrap, "bootstrap_seed": seed,
            "uncertainty_scope": "biological_unit_bootstrap_of_out_of_fold_scores_conditional_on_fitted_models",
            "training_and_tuning_refit_uncertainty": False,
            "small_biological_sample": len(values) <= 3, "subjects_not_repeated_splits_are_resampled": True}


def aggregate_topic_summaries(W, biological_ids, section_ids, *, condition_ids=None):
    """Equal-section W summaries within each animal/patient-condition unit.

    This is evaluation only; it introduces no likelihood weights. For Xenium,
    condition IDs preserve timepoints while biological IDs preserve patient
    blocking in downstream cross-validation and paired pre/post contrasts.
    """
    w = np.asarray(W, dtype=float)
    bio, section = np.asarray(biological_ids, str), np.asarray(section_ids, str)
    condition = np.full(len(w), "all") if condition_ids is None else np.asarray(condition_ids, str)
    if w.ndim != 2 or not np.isfinite(w).all() or any(len(value) != len(w) for value in [bio, section, condition]):
        raise ValueError("aligned finite topic memberships and grouping IDs required")
    records, summaries = [], []
    for unit in sorted(set(zip(bio, condition))):
        mask = (bio == unit[0]) & (condition == unit[1])
        sections = np.unique(section[mask])
        section_means = [w[mask & (section == s)].mean(axis=0) for s in sections]
        summaries.append(np.mean(section_means, axis=0))
        records.append({"biological_id": unit[0], "condition_id": unit[1],
                        "sections": sections.tolist(), "section_count": len(sections),
                        "observations": int(mask.sum())})
    return {"W_summary": np.asarray(summaries), "units": records,
            "aggregation": "equal_section_mean_within_biological_unit_and_condition",
            "changes_training_likelihood_weights": False}
