"""Manifest-complete scientific reporting for joint_v2 stage artifacts.

No scientific score is selected for being finite or favorable. Available-fit
means are descriptive; matched method contrasts use identical per-split targets.
Bootstrap intervals concern saved out-of-fold scores conditional on fitted
models, never refitting uncertainty or independent repeated-split replicates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import json
import sqlite3
import subprocess

import numpy as np
import pandas as pd

from .artifacts import atomic_json, json_safe, compatible_completed, sha256_file
from .config import fingerprint
from .aggregation import paired_method_contrast
from .metrics import align_profiles

SUMMARY_FIELDS = ("dataset", "protocol", "direction", "tier", "K", "panel_requested", "retention",
                  "method", "preprocessing", "hunter", "family", "recovery", "control", "vocabulary", "endpoint")
REPORTING_REVISION = "reporting_only_20260910_v1"
FOLDIN_CONVERGENCE_SCOPE = "conservative_all_count_eligible_adaptation_rows_in_endpoint"


class ManifestReader:
    """Use the indexed manifest when available; never duplicate its full DAG."""
    def __init__(self, path):
        self.path = Path(path)
        database = self.path.parent / "tasks.sqlite"
        self.db = None
        if database.exists():
            self.db = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
            self.metadata = {k: json.loads(v) for k, v in self.db.execute("SELECT key,value FROM metadata")}
            self.rows = None
        else:
            dag = json.loads(self.path.read_text())
            self.rows = {row["task_id"]: row for row in dag.pop("tasks")}
            self.metadata = dag

    def tasks(self, stage=None):
        if self.db:
            query = "SELECT value FROM tasks" + (" WHERE stage=?" if stage else "")
            for row, in self.db.execute(query, (stage,) if stage else ()):
                yield json.loads(row)
        else:
            yield from (row for row in self.rows.values() if stage is None or row["stage"] == stage)

    @lru_cache(maxsize=4096)
    def get(self, key):
        if self.db:
            result = self.db.execute("SELECT value FROM tasks WHERE task_id=?", (key,)).fetchone()
            if result is None:
                raise KeyError(f"missing parent in manifest: {key}")
            return json.loads(result[0])
        return self.rows[key]

    def close(self):
        if self.db:
            self.db.close()


def _directory(root, task):
    prefix = "data/interim/joint_v2/stages" if task["stage"] == "prepare" else "results/joint_v2/stages"
    return Path(root) / prefix / task["stage"] / task["task_id"]


def _number(value):
    return np.nan if value is None else float(value)


def _load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def expected_endpoint_subjects(split, cohort_biological_ids=()):
    protocol = split["protocol"]
    if protocol == "section_holdout":
        ids = sorted(str(row["bio_id"]) for row in split["assignments"])
        return {"within_training": ids, "primary_test": ids, "additional_test": ids,
                "section_transfer_equal_section_combined": ids}
    if protocol == "spatial_half":
        ids = sorted(map(str, cohort_biological_ids))
        if not ids:
            raise ValueError("frozen cohort inventory required for spatial-half subjects")
        return {"within_training": ids, "spatial_half": ids}
    result = {"within_training": list(map(str, split["train_biological_ids"])),
              "unseen_animal" if protocol == "animal_holdout" else "unseen_patient": list(map(str, split["test_biological_ids"]))}
    if protocol == "animal_holdout" and split.get("seen_animal_new_section"):
        result["seen_animal_new_section"] = sorted(map(str, split["seen_animal_new_section"]))
    return result


def _direction(spec):
    if spec["protocol"] != "spatial_half":
        return "not_applicable"
    return "horizontal" if spec["outer_split_id"] in ("spatial_left", "spatial_right") else "vertical"


def _endpoint_foldin_convergence(payload, endpoint):
    """Use frozen endpoint totals, never pretend to have per-subject certificates.

    A single unconverged adaptation-eligible row excludes the whole endpoint
    from convergence-restricted summaries, including rows without score counts.
    Zero-adaptation rows are not eligible. Missing/inconsistent evidence is
    unknown and cannot qualify as converged.
    """
    required = endpoint != "within_training"
    result = {"foldin_required": required, "foldin_converged": None,
              "foldin_convergence_scope": FOLDIN_CONVERGENCE_SCOPE if required else "not_applicable",
              "foldin_convergence_status": "unknown_missing_evidence" if required else "not_applicable",
              "foldin_count_eligible_observations": None}
    if not required or not isinstance(payload, dict):
        return result
    foldin = payload.get("foldin")
    counts = payload.get("foldin_status_counts")
    eligible = foldin.get("count_eligible_rows") if isinstance(foldin, dict) else None
    converged = payload.get("foldin_converged_observations")
    def is_count(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0
    if not is_count(eligible) or not is_count(converged) or not isinstance(counts, dict):
        return result
    if any(not isinstance(status, str) or not is_count(count) for status, count in counts.items()):
        result["foldin_convergence_status"] = "unknown_invalid_evidence"
        return result
    result["foldin_count_eligible_observations"] = int(eligible)
    eligible_status_count = sum(count for status, count in counts.items() if status != "no_adaptation_counts")
    if eligible_status_count != eligible or converged > eligible or counts.get("ok", 0) != converged:
        result["foldin_convergence_status"] = "unknown_inconsistent_evidence"
        return result
    result["foldin_converged"] = bool(converged == eligible)
    result["foldin_convergence_status"] = "all_count_eligible_converged" if converged == eligible else "not_all_count_eligible_converged"
    return result


def _combined_support_hash(parts, field):
    values = [row.get(field) for row in parts]
    # Hashing [None, None] would manufacture apparently valid support evidence.
    return fingerprint(values) if all(isinstance(value, str) and value.strip() for value in values) else None


def flatten_evaluation(task, evaluation, expected_subjects, memberships, *, state="complete", error=None):
    """One row per expected endpoint and biological subject, even before output."""
    spec = task["spec"]
    tiers = sorted({tier for base in task["base_ids"] for tier in memberships.get(base, [])})
    if not tiers:
        raise ValueError("evaluation task has no core/stability/thinning membership")
    header = {**spec, "task_id": task["task_id"], "direction": _direction(spec),
              "method": spec["method"], "preprocessing": spec.get("preprocessing", "native"),
              "hunter": spec.get("hunter", "native"), "family": spec.get("family", "competitor"),
              "control": spec.get("control", "native"), "stage_state": state,
              "manifest_expected": True}
    payloads = {} if evaluation is None else {"within_training": evaluation["within_training_molecule_prediction"],
                                              **evaluation.get("transfer", {})}
    records = []
    for endpoint, ids in expected_subjects.items():
        if endpoint == "section_transfer_equal_section_combined":
            continue
        payload = payloads.get(endpoint)
        foldin = _endpoint_foldin_convergence(payload, endpoint)
        fit_converged = bool(evaluation and evaluation.get("fit_converged") is True)
        endpoint_converged = fit_converged and (not foldin["foldin_required"] or foldin["foldin_converged"] is True)
        by_subject = {} if payload is None else {str(row["biological_id"]): row for row in payload.get("per_biological_unit", [])}
        unexpected = set(by_subject) - set(ids)
        if unexpected:
            raise ValueError(f"unexpected biological score rows in {task['task_id']}: {unexpected}")
        for bio in ids:
            subject = by_subject.get(str(bio))
            sections = [] if payload is None else [s for s in payload.get("per_section", [])
                                                   if str(s["biological_id"]) == str(bio)]
            section_dev = [_number(s["deviance"]) / _number(s["scored_molecules"])
                           if _number(s["scored_molecules"]) > 0 else np.nan for s in sections]
            section_ll = [_number(s["log_likelihood"]) / _number(s["scored_molecules"])
                          if _number(s["scored_molecules"]) > 0 else np.nan for s in sections]
            mass = _number(subject.get("molecules")) if subject else np.nan
            dev = _number(subject.get("deviance_per_molecule")) if subject else np.nan
            ll = _number(subject.get("log_likelihood_per_molecule")) if subject else np.nan
            failures = int(subject.get("inference_failures", 0)) if subject else 0
            success = bool(subject is not None and mass > 0 and not failures and not np.isnan(dev) and not np.isnan(ll))
            status = ("inference_failed" if failures else "support_failure" if success and np.isinf(dev) else
                      "ok" if success else "zero_score_or_unavailable" if subject is not None else
                      "missing_endpoint_subject" if payload is not None else state)
            for tier in tiers:
                records.append({**header, "tier": tier, "endpoint": endpoint, "biological_id": str(bio),
                    "success": success, "status": status, "error": error,
                    "fit_converged": fit_converged, **foldin, "converged": endpoint_converged,
                    "fit_status": evaluation.get("fit_status") if evaluation else None,
                    "scored_observations": int(subject["observations"]) if subject else 0,
                    "scored_molecules": mass, "deviance_per_molecule": dev,
                    "log_likelihood_per_molecule": ll,
                    "deviance_total": dev * mass if subject and mass > 0 else np.nan,
                    "log_likelihood_total": ll * mass if subject and mass > 0 else np.nan,
                    "inference_failures": failures,
                    "section_count": len(sections),
                    "equal_section_deviance": float(np.mean(section_dev)) if sections else np.nan,
                    "equal_section_log_likelihood": float(np.mean(section_ll)) if sections else np.nan,
                    "evaluation_vocabulary_sha256": payload.get("evaluation_vocabulary_sha256") if payload else None,
                    "evaluation_mask_sha256": payload.get("evaluation_mask_sha256") if payload else None,
                    "panel_actual": (evaluation.get("training_coverage", {}).get("panel_actual") if evaluation else None),
                    "foldin_converged_observations": payload.get("foldin_converged_observations") if payload else None,
                    "foldin_runtime_seconds": payload.get("foldin_runtime_seconds", 0) if payload else None})
    if "section_transfer_equal_section_combined" in expected_subjects:
        for tier in tiers:
            for bio in expected_subjects["section_transfer_equal_section_combined"]:
                parts = [r for r in records if r["tier"] == tier and r["biological_id"] == str(bio)
                         and r["endpoint"] in ("primary_test", "additional_test")]
                if len(parts) != 2:
                    raise ValueError("equal-section endpoint requires both prespecified transfer roles")
                success = all(row["success"] for row in parts)
                combined = {**parts[0], "endpoint": "section_transfer_equal_section_combined", "success": success,
                            "status": "ok" if success else "component_endpoint_unavailable",
                            "scored_molecules": sum(row["scored_molecules"] for row in parts),
                            "scored_observations": sum(row["scored_observations"] for row in parts),
                            "evaluation_mask_sha256": _combined_support_hash(parts, "evaluation_mask_sha256"),
                            "evaluation_vocabulary_sha256": _combined_support_hash(parts, "evaluation_vocabulary_sha256"),
                            "fit_converged": all(row["fit_converged"] for row in parts),
                            "converged": all(row["converged"] for row in parts)}
                combined["foldin_converged"] = (False if any(row["foldin_converged"] is False for row in parts) else
                                                  True if all(row["foldin_converged"] is True for row in parts) else None)
                combined["foldin_convergence_status"] = ("both_component_endpoints_converged" if combined["foldin_converged"] is True else
                                                          "component_endpoint_not_converged" if combined["foldin_converged"] is False else
                                                          "unknown_component_endpoint_evidence")
                combined["foldin_component_convergence"] = {row["endpoint"]: row["foldin_convergence_status"] for row in parts}
                for field in ("foldin_count_eligible_observations", "foldin_converged_observations"):
                    counts = [row[field] for row in parts]
                    valid_counts = all(isinstance(count, (int, np.integer)) and
                                       not isinstance(count, (bool, np.bool_)) and count >= 0 for count in counts)
                    combined[field] = int(sum(counts)) if valid_counts else None
                for field in ("deviance_per_molecule", "log_likelihood_per_molecule"):
                    section_field = "equal_section_deviance" if field == "deviance_per_molecule" else "equal_section_log_likelihood"
                    if all(row["section_count"] > 0 for row in parts):
                        combined[field] = float(np.average([row[section_field] for row in parts],
                                                  weights=[row["section_count"] for row in parts])) if success else np.nan
                        combined["equal_section_source"] = "per_section_sufficient_statistics"
                    else:
                        combined[field] = float(np.mean([row[field] for row in parts])) if success else np.nan
                        combined["equal_section_source"] = "one_prespecified_section_per_role_and_donor"
                combined["section_count"] = sum(row["section_count"] for row in parts)
                if success and (np.isnan(combined["deviance_per_molecule"]) or np.isnan(combined["log_likelihood_per_molecule"])):
                    combined["success"] = False
                    combined["status"] = "section_component_score_unavailable"
                for field in ("deviance_total", "log_likelihood_total"):
                    combined[field] = sum(row[field] for row in parts) if success else np.nan
                if success and np.isinf(combined["deviance_per_molecule"]):
                    combined["status"] = "support_failure"
                records.append(combined)
    return records


def _group_key(record, fields):
    return tuple(record[field] for field in fields)


def summarize_scores(records, *, bootstrap_draws=2000, seed=26091001, require_converged=False):
    """Available-fit means with complete denominators and subject-first averages."""
    groups = defaultdict(list)
    for row in records:
        groups[_group_key(row, SUMMARY_FIELDS)].append(row)
        if row["protocol"] == "spatial_half":
            combined = {**row, "direction": "combined"}
            groups[_group_key(combined, SUMMARY_FIELDS)].append(combined)
    summaries, unit_rows = [], []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        identity = dict(zip(SUMMARY_FIELDS, key))
        eligible = [r for r in rows if r["success"] and (r["converged"] or not require_converged)]
        subjects = sorted({row["biological_id"] for row in rows})
        per_subject = []
        for bio in subjects:
            complete = [r for r in eligible if r["biological_id"] == bio]
            all_rows = [r for r in rows if r["biological_id"] == bio]
            values = np.array([r["deviance_per_molecule"] for r in complete])
            ll = np.array([r["log_likelihood_per_molecule"] for r in complete])
            unit = {**identity, "biological_id": bio, "expected_evaluations": len(all_rows),
                    "successful_evaluations": len(complete),
                    "deviance_per_molecule": float(values.mean()) if len(values) else np.nan,
                    "log_likelihood_per_molecule": float(ll.mean()) if len(ll) else np.nan,
                    "repeated_split_min": float(values.min()) if len(values) else np.nan,
                    "repeated_split_max": float(values.max()) if len(values) else np.nan,
                    "repeated_splits_not_independent": True}
            per_subject.append(unit)
            unit_rows.append(unit)
        available = [u for u in per_subject if u["successful_evaluations"]]
        values = np.asarray([u["deviance_per_molecule"] for u in available])
        total_mass = sum(r["scored_molecules"] for r in eligible)
        ci, se = None, None
        if identity["dataset"] != "visium_dlpfc" and len(values) >= 2 and np.isfinite(values).all():
            rng = np.random.default_rng(seed)
            draws = values[rng.integers(len(values), size=(bootstrap_draws, len(values)))].mean(axis=1)
            ci = np.quantile(draws, [.025, .975]).tolist()
            se = float(draws.std(ddof=1)) if bootstrap_draws > 1 else None
        summaries.append({**identity, "require_converged": require_converged,
            "expected_biological_evaluations": len(rows), "successful_biological_evaluations": len(eligible),
            "expected_task_count": len({r["task_id"] for r in rows}),
            "successful_task_count": len({r["task_id"] for r in eligible}),
            "expected_biological_units": len(subjects), "available_biological_units": len(available),
            "failure_status_counts": dict(Counter(r["status"] for r in rows if not r["success"])),
            "nonconverged_evaluations": sum(not row["converged"] for row in rows if row["success"]),
            "fit_nonconverged_evaluations": sum(not row["fit_converged"] for row in rows if row["success"]),
            "foldin_nonconverged_evaluations": sum(row["foldin_converged"] is False for row in rows if row["success"] and row["foldin_required"]),
            "foldin_unknown_evaluations": sum(row["foldin_converged"] is None for row in rows if row["success"] and row["foldin_required"]),
            "convergence_restriction": "training_fit_and_applicable_endpoint_foldin",
            "foldin_convergence_scope": FOLDIN_CONVERGENCE_SCOPE,
            "infinite_score_evaluations": sum(np.isinf(r["deviance_per_molecule"]) for r in eligible),
            "equal_biological_deviance": float(values.mean()) if len(values) else np.nan,
            "equal_biological_log_likelihood": float(np.mean([u["log_likelihood_per_molecule"] for u in available])) if available else np.nan,
            "molecule_weighted_deviance": float(sum(r["deviance_total"] for r in eligible) / total_mass) if total_mass > 0 else np.nan,
            "molecule_weighted_log_likelihood": float(sum(r["log_likelihood_total"] for r in eligible) / total_mass) if total_mass > 0 else np.nan,
            "scored_molecule_evaluations": total_mass, "biological_unit_values": [{"biological_id": u["biological_id"], "deviance": u["deviance_per_molecule"]} for u in per_subject],
            "bootstrap_standard_error": se, "confidence_interval_95": ci,
            "uncertainty_scope": "no_Visium_SE_three_donors_shown" if identity["dataset"] == "visium_dlpfc" else "biological_score_bootstrap_conditional_on_fitted_models",
            "method_specific_available_support": True,
            "native_panel_scores_comparable_across_panel_sizes": False,
            "reference_vocabulary_hash_count": len({r["evaluation_vocabulary_sha256"] for r in eligible}),
            "evaluation_mask_hash_count": len({r["evaluation_mask_sha256"] for r in eligible})})
    return summaries, unit_rows


def paired_contrasts(records, *, reference_method="gplsi_document__P0_raw__spa_current__selected",
                     bootstrap_draws=2000, seed=26091001):
    fields = ("dataset", "protocol", "direction", "tier", "K", "panel_requested", "retention", "vocabulary", "endpoint")
    groups = defaultdict(list)
    for row in records:
        groups[_group_key(row, fields)].append(row)
        if row["protocol"] == "spatial_half":
            combined = {**row, "direction": "combined"}
            groups[_group_key(combined, fields)].append(combined)
    output = []
    for key, rows in groups.items():
        identity = dict(zip(fields, key))
        reference_recovery = "A_full_Pois"
        baseline = [r for r in rows if r["method"] == reference_method and r["recovery"] == reference_recovery]
        if not baseline:
            continue
        methods = sorted({(r["method"], r["recovery"]) for r in rows})
        match_fields = ("outer_split_id", "molecule_seed", "estimator_seed")
        identity_fields = match_fields + ("biological_id",)
        def planned_index(method_rows):
            index = {}
            for row in method_rows:
                identity_key = tuple(str(row[field]) for field in identity_fields)
                if identity_key in index:
                    raise ValueError(f"duplicate manifest-expected method identity: {identity_key}")
                index[identity_key] = row
            return index
        baseline_index = planned_index(baseline)
        for method, recovery in methods:
            if (method, recovery) == (reference_method, reference_recovery):
                continue
            selected = [r for r in rows if (r["method"], r["recovery"]) == (method, recovery)]
            candidate_index = planned_index(selected)
            common = set(candidate_index) & set(baseline_index)
            # Rows are manifest placeholders even when no output exists. Never
            # use success/finite-score filters to define the planned overlap.
            expected = [{field: baseline_index[key][field] for field in identity_fields} for key in sorted(common)]
            left = [{**candidate_index[key], "method": "candidate"} for key in sorted(common)]
            right = [{**baseline_index[key], "method": "reference"} for key in sorted(common)]
            unrequested = {"not_requested_for_candidate": set(baseline_index) - common,
                           "not_requested_for_reference": set(candidate_index) - common}
            scope_audit = {"planned_candidate_evaluations": len(candidate_index),
                           "planned_reference_evaluations": len(baseline_index),
                           "planned_common_evaluations": len(common),
                           "manifest_scope_exclusion_counts": {reason: len(keys) for reason, keys in unrequested.items()},
                           "unrequested_estimator_seed_evaluations": {
                               reason: dict(Counter(key[2] for key in keys)) for reason, keys in unrequested.items()},
                           "manifest_scope_exclusions_are_failures": False}
            for restricted in (False, True):
                contrast = paired_method_contrast(left + right, "candidate", "reference", match_fields=match_fields,
                    expected_pairs=expected, require_converged=restricted, n_bootstrap=bootstrap_draws, seed=seed)
                contrast.update(scope_audit)
                contrast["expected_denominator_source"] = "intersection_of_each_methods_manifest_expected_identities"
                contrast["convergence_restriction"] = "training_fit_and_applicable_endpoint_foldin"
                contrast["foldin_convergence_scope"] = FOLDIN_CONVERGENCE_SCOPE
                if identity["dataset"] == "visium_dlpfc":
                    contrast["bootstrap_standard_error"], contrast["confidence_interval_95"] = None, None
                    contrast["uncertainty_scope"] = "three_donor_contrasts_and_repeated_split_ranges_no_pseudoreplicated_SE"
                # The compact endpoint index is the single copy of paired
                # task scores; contrasts retain subject values and audit counts.
                contrast.pop("paired_evaluations", None)
                contrast.pop("excluded_evaluations", None)
                output.append({**identity, "method": method, "recovery": recovery,
                               "reference_method": reference_method, "reference_recovery": reference_recovery, **contrast})
    return output


def _write_table(path, records, *, csv=False):
    frame = pd.DataFrame(records)
    for column in frame.columns:
        if any(isinstance(value, (dict, list, tuple, np.ndarray)) for value in frame[column]):
            frame[column] = frame[column].map(lambda value: json.dumps(json_safe(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    if csv:
        frame.to_csv(path.with_suffix(".csv"), index=False)


def _plot_score_series(selected, xfield):
    """Keep unavailable positions as gaps and provide explicit coverage notes."""
    x = [row[xfield] for row in selected]
    values = np.asarray([row["equal_biological_deviance"] for row in selected], dtype=float)
    y = np.where(np.isfinite(values), values, np.nan).tolist()
    notes = []
    for name, mask in (("infinite", np.isinf(values)), ("unavailable/NaN", np.isnan(values))):
        if np.any(mask):
            notes.append(f"{name} at {xfield}=" + ",".join(str(value) for value, keep in zip(x, mask) if keep))
    unavailable = []
    for row in selected:
        expected = row.get("expected_biological_evaluations", 0)
        observed = row.get("successful_biological_evaluations", expected)
        if observed < expected:
            unavailable.append(f"{row[xfield]}: {observed}/{expected} available ({json.dumps(row.get('failure_status_counts', {}), sort_keys=True)})")
    if unavailable:
        notes.append("coverage by " + xfield + " [" + "; ".join(unavailable) + "]")
    return x, y, notes


def _plot_scores(summaries, runtime_rows, figure_directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_directory.mkdir(parents=True, exist_ok=True)
    files = []
    allowed = [r for r in summaries if r["tier"] == "core" and not r["require_converged"]]
    for xfield, purpose in [("K", "deviance_vs_K"), ("panel_requested", "reference_deviance_vs_HVG")]:
        groups = defaultdict(list)
        for row in allowed:
            if xfield == "panel_requested" and (row["dataset"] != "visium_dlpfc" or row["vocabulary"] != "common_reference"):
                continue
            fixed = "panel_requested" if xfield == "K" else "K"
            key = tuple(row[f] for f in ["dataset", "protocol", "direction", "endpoint", "vocabulary", fixed])
            groups[key].append(row)
        for key, rows in groups.items():
            labels = sorted({r["method"] + "/" + r["recovery"] for r in rows})
            for page in range(0, len(labels), 12):
                fig, ax = plt.subplots(figsize=(10, 6))
                notes = []
                for label in labels[page:page + 12]:
                    selected = sorted([r for r in rows if r["method"] + "/" + r["recovery"] == label], key=lambda r: r[xfield])
                    x, y, issues = _plot_score_series(selected, xfield)
                    ax.plot(x, y, marker="o", label=label)
                    if issues:
                        notes.append(f"{label}: " + "; ".join(issues))
                ax.set_xticks(sorted({r[xfield] for r in rows}))
                ax.set(xlabel=xfield, ylabel="Equal biological-unit deviance / molecule", title=" | ".join(map(str, key)))
                ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1, 1))
                if notes:
                    fig.text(.01, .01, "\n".join(notes), fontsize=5)
                fig.tight_layout()
                path = figure_directory / f"{purpose}_{fingerprint(key)[:12]}_{page // 12}.png"
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                files.append(str(path))
    if runtime_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        for dataset in sorted({r["dataset"] for r in runtime_rows}):
            rows = [r for r in runtime_rows if r["dataset"] == dataset and r["standalone_ancestor_runtime_seconds"] > 0]
            ax.scatter([r["standalone_ancestor_runtime_seconds"] for r in rows],
                       [r["peak_ancestor_memory_bytes"] / 2**30 for r in rows], s=12, alpha=.5, label=dataset)
        ax.set(xlabel="Standalone observed ancestor wall time (s)", ylabel="Maximum ancestor peak RSS (GiB)", xscale="log")
        ax.legend()
        path = figure_directory / "runtime_memory.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))
    return files


def _reporting_identity(caller_metadata=None):
    """Hash reporting code separately from the immutable fitted-model manifest."""
    package = Path(__file__).resolve().parent
    files = sorted(package.rglob("*.py"))
    hashes = {str(path.relative_to(package)): sha256_file(path) for path in files}
    def git_output(arguments):
        try:
            process = subprocess.run(["git", "-C", str(package), *arguments],
                                     capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return process.stdout.strip() if process.returncode == 0 else None
    commit = git_output(["rev-parse", "HEAD"])
    tracked = git_output(["ls-files", "--error-unmatch", "--", "results_reporting.py", "aggregation.py"])
    status = git_output(["status", "--porcelain", "--untracked-files=all", "--", *hashes]) if commit else None
    return {"revision": REPORTING_REVISION, "git_commit": commit,
            "reporting_modules_tracked": tracked is not None,
            "working_tree_status": status,
            "package_python_sha256": hashes, "code_digest": fingerprint(hashes),
            "digest_scope": "all_Python_files_in_checked_out_gplsi_joint_v2_package",
            "caller_metadata": {} if caller_metadata is None else caller_metadata,
            "fitting_or_scoring_recomputed": False}


def report_results(root, manifest_path, *, verify_hashes=True, run_profiles=True, make_plots=True,
                   reference_method="gplsi_document__P0_raw__spa_current__selected", reporting_provenance=None):
    root, manifest_path = Path(root), Path(manifest_path)
    manifest = ManifestReader(manifest_path)
    config = manifest.metadata["config"]
    reporting_identity = _reporting_identity(reporting_provenance)
    output_tag = "scientific_reporting_" + reporting_identity["code_digest"][:12]
    output = root / "reports/joint_v2" / manifest_path.parent.name / output_tag
    figures = root / "figures/joint_v2" / manifest_path.parent.name / output_tag
    output.mkdir(parents=True, exist_ok=True)
    memberships = manifest.metadata["base_memberships"]
    splits, cohort_ids = {}, {}
    for dataset in {base["dataset"] for base in manifest.metadata["bases"]}:
        directory = root / "data/manifests/joint_v2/splits" / dataset
        splits[dataset] = {r["split_id"]: r for r in _load(directory / "splits.json", [])}
        inventory = _load(directory / "inventory.json", {"units": []})
        cohort_ids[dataset] = sorted({str(r["bio_id"]) for r in inventory["units"]})
    shared_consumers = Counter()
    for task in manifest.tasks():
        if task["stage"] in ("geometry", "competitor"):
            for parent in task["parents"]:
                if manifest.get(parent)["stage"] == "spectral":
                    shared_consumers[parent] += 1

    @lru_cache(maxsize=8192)
    def status(key):
        task = manifest.get(key)
        directory = _directory(root, task)
        if compatible_completed(directory, key, verify_hashes=verify_hashes):
            return "complete", None
        if (directory / "complete.json").exists():
            return "invalid_completion", "completion marker or artifact checksum mismatch"
        attempts = [_load(p) for p in (directory / "attempts").glob("*.json")]
        if attempts:
            last = max(attempts, key=lambda r: r.get("recorded_at", 0))
            return ("resource_blocked" if last.get("type") == "GeometryResourceBlocked" else "failed"), last.get("error")
        if any(status(parent)[0] in ("failed", "resource_blocked", "blocked_parent", "invalid_completion") for parent in task["parents"]):
            return "blocked_parent", "one or more required ancestors failed"
        return "pending_or_not_submitted", None

    @lru_cache(maxsize=8192)
    def effective(key):
        return _load(_directory(root, manifest.get(key)) / "effective.json", {})

    def ancestors(key):
        seen, pending = set(), [key]
        while pending:
            item = pending.pop()
            if item not in seen:
                seen.add(item)
                pending.extend(manifest.get(item)["parents"])
        return seen

    index, records, runtime_rows, profile_tasks = [], [], [], []
    for task in manifest.tasks():
        state, error = status(task["task_id"])
        eff = effective(task["task_id"]) if state == "complete" else {}
        index.append({"task_id": task["task_id"], "stage": task["stage"], "dataset": task["dataset"],
                      "variant": task["variant"], "state": state, "error": error,
                      "runtime_seconds": eff.get("runtime_seconds"), "peak_rss_bytes": eff.get("peak_rss_bytes"),
                      "slurm_job_id": eff.get("slurm_job_id"), "partition": eff.get("partition"),
                      "spec": task["spec"], "base_ids": task["base_ids"]})
        if task["stage"] != "evaluation":
            continue
        spec = task["spec"]
        split = splits.get(spec["dataset"], {}).get(spec["outer_split_id"])
        if split is None:
            raise ValueError(f"expected frozen outer split not found: {spec['outer_split_id']}")
        expected = expected_endpoint_subjects(split, cohort_ids[spec["dataset"]])
        evaluation = _load(_directory(root, task) / "evaluation.json") if state == "complete" else None
        if state == "complete" and evaluation is None:
            raise ValueError("complete evaluation task has no evaluation.json")
        records.extend(flatten_evaluation(task, evaluation, expected, memberships, state=state, error=error))
        if state == "complete":
            chain = ancestors(task["task_id"])
            stage_costs = defaultdict(float)
            for parent in chain:
                stage_costs[manifest.get(parent)["stage"]] += float(effective(parent).get("runtime_seconds", 0))
            standalone = sum(stage_costs.values())
            shared = stage_costs.get("spectral", 0)
            amortized_shared = sum(float(effective(p).get("runtime_seconds", 0)) / max(shared_consumers[p], 1)
                                  for p in chain if manifest.get(p)["stage"] == "spectral")
            runtime_rows.append({"task_id": task["task_id"], **spec,
                "standalone_ancestor_runtime_seconds": standalone, "shared_spectral_seconds": shared,
                "amortized_shared_spectral_seconds": amortized_shared,
                "amortized_total_seconds": standalone - shared + amortized_shared,
                "recovery_seconds": stage_costs.get("recovery", 0) + stage_costs.get("reference_recovery", 0),
                "evaluation_seconds": stage_costs.get("evaluation", 0),
                "transfer_inference_seconds": sum(float(p.get("foldin_runtime_seconds", 0)) for p in evaluation.get("transfer", {}).values()),
                "peak_ancestor_memory_bytes": max((effective(p).get("peak_rss_bytes", 0) for p in chain), default=0),
                "observed_stage_seconds": dict(stage_costs), "queue_time_included": False,
                "amortization_denominator": "expected_direct_geometry_or_competitor_consumers_of_shared_spectral_stage"})
            profile_tasks.append(task)
    summaries, unit_rows = summarize_scores(records, bootstrap_draws=config["bootstrap"]["draws"])
    restricted, _ = summarize_scores(records, bootstrap_draws=config["bootstrap"]["draws"], require_converged=True)
    contrasts = paired_contrasts(records, reference_method=reference_method, bootstrap_draws=config["bootstrap"]["draws"])
    for name, rows in [("result_index", index), ("expected_and_observed_scores", records),
                       ("method_summaries", summaries + restricted), ("biological_unit_scores", unit_rows),
                       ("paired_method_contrasts", contrasts), ("runtime_costs", runtime_rows)]:
        _write_table(output / name, rows, csv=name in ("method_summaries", "paired_method_contrasts"))
    profiles = profile_stability(root, manifest, profile_tasks, config) if run_profiles else []
    _write_table(output / "profile_stability", profiles)
    plots = _plot_scores(summaries, runtime_rows, figures) if make_plots and summaries else []
    scientific_figures = None
    if make_plots:
        from .figures import generate_figures
        scientific_figures = generate_figures(root, manifest, figures, tables_directory=output,
                                              profile_rows=profiles, verify_hashes=verify_hashes)
    report = {"expected_stages": len(index), "stage_state_counts": dict(Counter(r["state"] for r in index)),
              "expected_biological_score_rows_including_tier_membership": len(records),
              "successful_biological_score_rows": sum(r["success"] for r in records),
              "infinite_deviance_rows": sum(np.isinf(r["deviance_per_molecule"]) for r in records),
              "summary_rows": len(summaries), "paired_contrasts": len(contrasts), "profile_pairs": len(profiles),
              "tables_directory": str(output), "figures": plots, "full_artifact_checksums_verified": verify_hashes,
              "scientific_figures": scientific_figures,
              "uncertainty": "conditional biological score bootstrap; Visium donor values, no pseudo-SE",
              "method_means_support": "available fits with complete expected coverage; infer using matched contrasts",
              "reporting_provenance": reporting_identity, "output_tag": output_tag,
              "model_manifest_provenance": {"manifest_path": str(manifest_path),
                  "manifest_sha256": sha256_file(manifest_path),
                  "code_hash": manifest.metadata.get("code_hash"),
                  "config_hash": manifest.metadata.get("config_hash"),
                  "source_hashes": manifest.metadata.get("source_hashes"),
                  "scientific_task_identities_changed": False},
              "historical_results_included": False}
    atomic_json(output / "REPORT_STATUS.json", report)
    manifest.close()
    return report


def profile_stability(root, manifest, evaluation_tasks, config):
    """Prespecified primary K/panel profile panels, separated by perturbation."""
    @lru_cache(maxsize=64)
    def profile(task_id):
        task = manifest.get(task_id)
        parent = [manifest.get(p) for p in task["parents"] if manifest.get(p)["stage"] != "prepare"]
        if len(parent) != 1:
            raise ValueError("profile evaluation requires one factor/recovery parent")
        directory = _directory(root, parent[0])
        path = directory / ("A.npz" if (directory / "A.npz").exists() else "factors.npz")
        with np.load(path, allow_pickle=False) as saved:
            return saved["A"], saved["feature_ids"].astype(str)
    selected = []
    for task in evaluation_tasks:
        spec = task["spec"]
        if spec["K"] == config["primary_K"][spec["dataset"]] and spec["panel_requested"] == config["primary_panel"][spec["dataset"]]:
            selected.append(task)
    common = ("dataset", "protocol", "K", "panel_requested", "method", "recovery", "vocabulary")
    groups = defaultdict(list)
    for task in selected:
        groups[tuple(task["spec"][field] for field in common)].append(task)
    records = []
    for key, tasks in groups.items():
        for left, right in combinations(tasks, 2):
            a, b = left["spec"], right["spec"]
            same_outer = a["outer_split_id"] == b["outer_split_id"]
            same_estimator = a["estimator_seed"] == b["estimator_seed"]
            same_molecules = a["molecule_seed"] == b["molecule_seed"]
            same_retention = a["retention"] == b["retention"]
            if same_outer and same_molecules and same_retention and a["retention"] == 1 and not same_estimator:
                kind = "same_data_initialization_stability"
            elif same_outer and same_molecules and same_estimator and not same_retention and 1 in (a["retention"], b["retention"]):
                kind = "nested_count_thinning_stability"
            elif not same_outer and same_molecules and same_estimator and same_retention and a["retention"] == 1 and a["estimator_seed"] == config["seeds"]["estimator_seed"]:
                kind = "section_training_split_stability" if a["protocol"] == "section_holdout" else "spatial_half_training_split_stability" if a["protocol"] == "spatial_half" else "biological_training_split_stability"
            else:
                continue
            aa, ga = profile(left["task_id"])
            ab, gb = profile(right["task_id"])
            aligned = align_profiles(aa, ab, ga, gb)
            if "common_gene_ids" in aligned:
                aligned["common_gene_support_sha256"] = fingerprint(aligned.pop("common_gene_ids"))
            records.append({**dict(zip(common, key)), "comparison": kind,
                            "task_a": left["task_id"], "task_b": right["task_id"],
                            "outer_split_a": a["outer_split_id"], "outer_split_b": b["outer_split_id"],
                            "retention_a": a["retention"], "retention_b": b["retention"],
                            "estimator_seed_a": a["estimator_seed"], "estimator_seed_b": b["estimator_seed"],
                            "independent_pair_standard_error": None, **aligned})
    return records
