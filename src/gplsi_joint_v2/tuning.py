"""Five-fold training-only spatial-competitor tuning, independent of graph-SVD CV.

The criterion is mean conditional composition deviance per scored molecule
across all five folds. Fold rows are excluded from profiles/graphs/HVGs, then
fold counts (drawn only from outer fitting molecules) are split into adaptation
and scoring counts. No outer D_score or outer test set is read by this module.
"""
from __future__ import annotations

from time import perf_counter
import hashlib

import numpy as np
from scipy import sparse

from .data import training_only_feature_ranking, molecule_split, seed_namespace
from .graph import build_graph
from .spectral import graph_fold_ids


class CompetitorTuningError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _compact_metadata(value):
    """Retain optimizer evidence without repeating all n fitted depths per fold."""
    if isinstance(value, np.ndarray):
        if value.size <= 1000:
            return value.tolist()
        numeric = np.asarray(value, float)
        finite = numeric[np.isfinite(numeric)]
        return {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "quantiles": np.quantile(finite, [0, .25, .5, .75, 1]).tolist() if len(finite) else [],
                "nonfinite": int(np.count_nonzero(~np.isfinite(numeric)))}
    if isinstance(value, dict):
        return {key: _compact_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_metadata(item) for item in value]
    return value


def select_spatial_penalty(records, penalties):
    """Every candidate needs five valid folds; +inf is a valid bad score."""
    curve = []
    for penalty in sorted(set(float(x) for x in penalties)):
        rows = records[penalty]
        complete = len(rows) == 5 and {row["fold"] for row in rows} == set(range(5))
        eligible = complete and all(row.get("valid_for_selection", False) and
                                    not np.isnan(row.get("score", np.nan)) for row in rows)
        curve.append({"penalty": penalty, "five_fold_complete": complete,
                      "selectable": bool(eligible),
                      "fold_scores": [row.get("score") for row in sorted(rows, key=lambda row: row["fold"])],
                      "mean_five_fold_deviance_per_molecule": float(np.mean([row["score"] for row in rows])) if eligible else None})
    selectable = [row for row in curve if row["selectable"]]
    if not selectable:
        return None, curve
    best = min(selectable, key=lambda row: (row["mean_five_fold_deviance_per_molecule"], row["penalty"]))
    return best["penalty"], curve


def fit_tuned_spatial_competitor(method, prepared, G, K, seed, config):
    from .competitors import fit_graph_kl_nmf, fit_spatial_lda
    from .likelihood import fold_in_fixed_A, recover_A_poisson
    from .metrics import score_counts
    if method not in ("graph_kl_nmf_tuned", "spatial_lda_tuned"):
        raise ValueError(f"unknown tunable spatial competitor {method}")
    tuning = config["competitor_tuning"]
    if tuning.get("folds", 5) != 5:
        raise ValueError("joint_v2 competitor tuning requires all five folds")
    spatial_lda = method == "spatial_lda_tuned"
    penalties = sorted(set(float(x) for x in tuning[
        "spatial_lda_inverse_penalties" if spatial_lda else "penalties"]))
    if not penalties or any(not np.isfinite(x) or x < 0 or (spatial_lda and x == 0) for x in penalties):
        raise ValueError("invalid frozen competitor penalty grid; Calico inverse penalty must be strictly positive")
    budgets = config.get("competitor_budgets", {})
    solver = config["solvers"]
    D = sparse.csr_matrix(prepared.train_rank_counts)
    coordinates, strata = np.asarray(prepared.train_coords), np.asarray(prepared.train_graph_ids)
    if D.shape[0] != len(coordinates) or len(strata) != len(coordinates):
        raise ValueError("prepared training rank counts and coordinate rows differ")
    outer = prepared.split_metadata["outer_split"]
    dataset = outer["dataset"]
    split_id = outer["split_id"]
    requested = prepared.split_metadata["panel_requested"] if dataset == "visium_dlpfc" else None
    molecule_seed = int(prepared.split_metadata.get("molecule_seed", config["seeds"]["molecule_seed"]))
    fold_seed = int(config["seeds"]["graph_cv_seed"])
    folds = graph_fold_ids(G, fold_seed, nfolds=5)
    records = {penalty: [] for penalty in penalties}
    fold_inventory = []
    started = perf_counter()
    def fit_one(counts, graph, coords, graph_ids, penalty):
        if spatial_lda:
            return fit_spatial_lda(counts, coords, graph_ids, K, seed, penalty=penalty,
                                   outer_iterations=budgets.get("spatial_lda_outer_iterations", 3),
                                   lda_iterations=budgets.get("spatial_lda_inner_iterations", 5),
                                   admm_iterations=budgets.get("spatial_lda_admm_iterations", 15))
        return fit_graph_kl_nmf(counts, graph, K, seed, penalty=penalty,
                                max_iter=budgets.get("nmf_iterations", 500),
                                chunk_size=solver["entry_chunk_size"])
    for fold in range(5):
        training = folds != fold
        validation = ~training
        try:
            if requested is None:
                features = np.arange(D.shape[1])
                feature_rule = {"method": "source_targeted_panel"}
            else:
                ranking = training_only_feature_ranking(D[training])
                features = np.sort(ranking["ranked_indices"][:requested])
                feature_rule = {"method": "fold_training_raw_variance_to_mean", "requested": requested,
                                "detection_threshold": ranking["detection_threshold"],
                                "n_eligible_training_observations": ranking["n_eligible_training_observations"]}
            if len(features) < K:
                raise ValueError(f"inner panel has {len(features)} genes for K={K}")
            candidate = D[training][:, features]
            keep = np.asarray(candidate.sum(axis=1)).ravel() > 0
            counts = candidate[keep]
            fit_rows = np.flatnonzero(training)[keep]
            inner_graph, graph_metadata = build_graph(coordinates[fit_rows], strata[fit_rows],
                                                       k=config["graph"]["neighbors"])
            namespace_seed = seed_namespace(molecule_seed, dataset, split_id,
                                             "spatial_competitor_inner_validation", fold)
            # Split on the full raw vocabulary before selecting columns so the
            # same count-defined targets are used across candidate panel sizes.
            full_adapt, full_score = molecule_split(D[validation], namespace_seed, training_fraction=.8)
            adapt, score = full_adapt[:, features], full_score[:, features]
            eligible = np.asarray(adapt.sum(axis=1)).ravel() > 0
            scored = eligible & (np.asarray(score.sum(axis=1)).ravel() > 0)
            if not np.any(scored):
                raise ValueError("fold has no positive adaptation-and-score observations")
            fold_inventory.append({"fold": fold, "training_rows": int(len(fit_rows)),
                                   "validation_rows": int(validation.sum()),
                                   "training_indices_sha256": hashlib.sha256(fit_rows.astype(np.int64).tobytes()).hexdigest(),
                                   "excluded_zero_panel_training_indices": np.flatnonzero(training)[~keep].tolist(),
                                   "zero_panel_training_rows": int((~keep).sum()),
                                   "feature_indices": features.tolist(), "feature_rule": feature_rule,
                                   "graph": graph_metadata, "validation_molecule_seed": namespace_seed,
                                   "validation_eligible_rows": int(eligible.sum()), "scored_rows": int(scored.sum()),
                                   "zero_adaptation_rows": int((~eligible).sum()),
                                   "scored_molecules": int(score[eligible].sum())})
        except Exception as exc:
            for penalty in penalties:
                records[penalty].append({"fold": fold, "status": "fold_preparation_failed",
                                         "valid_for_selection": False, "error": str(exc), "score": None})
            continue
        for penalty in penalties:
            row = {"fold": fold, "penalty": penalty, "valid_for_selection": False, "score": None}
            try:
                fitted = fit_one(counts, inner_graph, coordinates[fit_rows], strata[fit_rows], penalty)
                accepted_schedule = spatial_lda and fitted.status == "fixed_schedule_completed_uncertified"
                fit_eligible = fitted.converged or accepted_schedule
                profile = recover_A_poisson(fitted.W, counts, initial_A=None,
                                            max_iter=solver["A_max_iter"], tolerance=solver["A_tolerance"],
                                            chunk_size=solver["entry_chunk_size"])
                foldin = fold_in_fixed_A(profile.A_hat, adapt, max_iter=solver["foldin_max_iter"],
                                          tolerance=solver["foldin_tolerance"],
                                          row_chunk_size=solver["row_chunk_size"],
                                          entry_chunk_size=solver["entry_chunk_size"])
                scored_result = score_counts(foldin.W, profile.A_hat, score, eligibility_mask=eligible,
                                               inference_valid=foldin.inference_valid,
                                               entry_chunk_size=solver["entry_chunk_size"])
                summary = scored_result["summary"]
                value = summary["deviance_per_molecule"]
                foldin_eligible = bool(np.all(foldin.row_converged[scored]))
                valid = bool(fit_eligible and profile.converged and foldin_eligible and not np.isnan(value))
                row.update(score=float(value), valid_for_selection=valid,
                           status="ok" if valid and np.isfinite(value) else
                                  "support_infinite" if valid else "fit_profile_or_foldin_nonconverged_or_failed",
                           estimator_converged=fitted.converged,
                           estimator_schedule_accepted_without_certificate=accepted_schedule,
                           estimator_status=fitted.status, estimator_metadata=_compact_metadata(fitted.metadata),
                           profile_recovery={"method": "A_full_Pois", "converged": profile.converged,
                                             "status": profile.status, "iterations": profile.iterations,
                                             "normalized_optimality_gap": profile.normalized_optimality_gap,
                                             "initializer": "pooled_fitting_frequencies_then_interiorization",
                                             "native_A_used": False, "W_fixed": True,
                                             "counts": "inner_training_fit_only"},
                           foldin_converged_scored_rows=int(foldin.row_converged[scored].sum()),
                           expected_scored_rows=int(scored.sum()), foldin_metadata=foldin.metadata,
                           score_coverage=summary)
            except Exception as exc:
                row.update(status="candidate_fit_failed", error=str(exc))
            records[penalty].append(row)
    selected, aggregate = select_spatial_penalty(records, penalties)
    diagnostics = {"version": "joint_v2_spatial_competitor_inner_five_fold_poisson_v2",
                   "method": method, "penalty_grid": penalties, "fold_curves": records,
                   "aggregate": aggregate, "selected_penalty": selected,
                   "fold_inventory": fold_inventory, "graph_cv_seed": fold_seed,
                   "fold_assignments_in_prepared_training_row_order": folds.tolist(),
                   "inner_adaptation_seed_source": molecule_seed, "estimator_seed": seed,
                   "loss": "mean_of_five_conditional_composition_deviances_per_scored_molecule",
                   "fold_weights": [.2] * 5, "scoring_folds": list(range(5)),
                   "tie_rule": "smallest_numeric_penalty", "infinite_support_scores": "valid_scientific_outcomes",
                   "outer_scoring_counts_used": False, "labels_used": False,
                   "all_candidate_profiles": "A_full_Pois_given_inner_training_W",
                   "candidate_native_A_scored": False, "A_current_used": False,
                   "inner_graphs": "rebuilt_coordinate_only_after_excluding_fold_rows",
                   "penalty_convention": "Calico_inverse_difference_penalty" if spatial_lda else "penalty/2*Tr(U.T@L@U)",
                   "Calico_zero_penalty": "not_defined_by_inverse_penalty_API" if spatial_lda else None,
                   "Calico_fixed_schedule_is_not_convergence_certificate": spatial_lda,
                   "tuning_runtime_seconds": perf_counter() - started}
    if selected is None:
        raise CompetitorTuningError("no competitor penalty has five valid fitted-and-scored folds", diagnostics)
    refit_started = perf_counter()
    fitted = fit_one(prepared.train_fit, G, coordinates, strata, selected)
    diagnostics["refit_runtime_seconds"] = perf_counter() - refit_started
    fitted.metadata.update(tuning=diagnostics, selected_penalty=selected,
                           runtime_seconds=perf_counter() - started,
                           native_fit_runtime_seconds=diagnostics["refit_runtime_seconds"],
                           joint_profiles=True)
    return fitted
