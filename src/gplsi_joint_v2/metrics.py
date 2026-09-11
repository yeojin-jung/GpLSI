"""Exact sparse prediction, section-centered spatial diagnostics and evaluation.

Likelihood fitting is deliberately imported from a separate module. Scoring
counts and annotations have no path back to fit or fold-in functions.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy, rel_entr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .likelihood import count_csr, simplex_matrix
from .graph import neighbor_indices


def score_counts(W, A, score_counts, *, eligibility_mask=None, inference_valid=None,
                 entry_chunk_size=250000, floor=None):
    """Conditional composition score, with count-defined coverage and exact zeros.

    Per-row arrays preserve support failures as infinities. Inference failures
    remain unavailable results within the eligible denominator, rather than
    being silently removed. The separately named optional floor diagnostic is
    renormalized over all genes in bounded row blocks.
    """
    w, a = simplex_matrix(W, "W"), simplex_matrix(A, "A")
    d = count_csr(score_counts)
    if d.shape != (w.shape[0], a.shape[1]) or w.shape[1] != a.shape[0]:
        raise ValueError("score dimensions do not match W and A")
    if entry_chunk_size <= 0 or (floor is not None and not 0 < floor < 1):
        raise ValueError("invalid score chunk size or diagnostic floor")
    n = d.shape[0]
    eligible = np.ones(n, bool) if eligibility_mask is None else np.asarray(eligibility_mask, bool)
    valid = np.ones(n, bool) if inference_valid is None else np.asarray(inference_valid, bool)
    if eligible.shape != (n,) or valid.shape != (n,):
        raise ValueError("eligibility and inference masks must match observation rows")
    depth = np.asarray(d.sum(axis=1), dtype=float).ravel()
    scored = eligible & (depth > 0)
    ll = np.zeros(n)
    dev = np.zeros(n)
    bad_entries = np.zeros(n, np.int64)
    bad_molecules = np.zeros(n)
    if floor is not None:
        floor_normalizer = np.empty(n)
        # This diagnostic alone needs all feature probabilities, in 128-row blocks.
        for start in range(0, n, 128):
            stop = min(n, start + 128)
            floor_normalizer[start:stop] = np.maximum(w[start:stop] @ a, floor).sum(axis=1)
        floored_ll, floored_dev = np.zeros(n), np.zeros(n)
    for start in range(0, d.nnz, entry_chunk_size):
        stop = min(d.nnz, start + entry_chunk_size)
        rows = np.searchsorted(d.indptr, np.arange(start, stop), side="right") - 1
        cols = d.indices[start:stop]
        y = d.data[start:stop].astype(float, copy=False)
        probability = np.einsum("ik,ik->i", w[rows], a[:, cols].T)
        supported = probability > 0
        selected_rows = rows[supported]
        ll += np.bincount(selected_rows, weights=y[supported] * np.log(probability[supported]), minlength=n)
        dev += np.bincount(selected_rows, weights=2 * y[supported] *
                           (np.log(y[supported]) - np.log(depth[selected_rows]) - np.log(probability[supported])), minlength=n)
        bad_entries += np.bincount(rows[~supported], minlength=n)
        bad_molecules += np.bincount(rows[~supported], weights=y[~supported], minlength=n)
        if floor is not None:
            p_floor = np.maximum(probability, floor) / floor_normalizer[rows]
            floored_ll += np.bincount(rows, weights=y * np.log(p_floor), minlength=n)
            floored_dev += np.bincount(rows, weights=2 * y * (np.log(y) - np.log(depth[rows]) - np.log(p_floor)), minlength=n)
    # Sum of means equals observed row depth exactly on the simplex, so the
    # zero-count contribution and linear terms cancel without dense predictions.
    slack = 256 * np.finfo(float).eps * np.maximum(depth, 1.0)
    if np.any((bad_entries == 0) & (dev < -slack)):
        raise ValueError("conditional deviance is negative beyond floating-point roundoff")
    dev = np.maximum(dev, 0.0)
    ll[bad_entries > 0], dev[bad_entries > 0] = -np.inf, np.inf
    unavailable = scored & ~valid
    ll[~scored | unavailable], dev[~scored | unavailable] = np.nan, np.nan
    statuses = np.full(n, "ok", dtype="U48")
    statuses[bad_entries > 0] = "positive_score_count_has_zero_probability"
    statuses[unavailable] = "inference_failed"
    statuses[depth == 0] = "no_score_counts"
    statuses[~eligible] = "count_ineligible"
    scored_molecules = float(depth[scored].sum())
    any_impossible = bool(np.any(scored & valid & (bad_entries > 0)))
    any_unavailable = bool(np.any(unavailable))
    total_ll = -np.inf if any_impossible else (np.nan if any_unavailable else float(np.sum(ll[scored])))
    total_dev = np.inf if any_impossible else (np.nan if any_unavailable else float(np.sum(dev[scored])))
    if not np.any(scored):
        total_ll, total_dev = np.nan, np.nan
    summary = {"endpoint": "conditional_composition_prediction",
               "status": "no_scored_rows" if not np.any(scored) else
                         "inference_failed" if any_unavailable else
                         "support_failure" if any_impossible else "ok",
               "log_likelihood": float(total_ll), "deviance": float(total_dev),
               "log_likelihood_per_molecule": float(total_ll / scored_molecules) if scored_molecules else np.nan,
               "deviance_per_molecule": float(total_dev / scored_molecules) if scored_molecules else np.nan,
               "total_observations": n, "count_eligible_observations": int(eligible.sum()),
               "scored_observations": int(scored.sum()), "scored_molecules": scored_molecules,
               "zero_score_observations": int(np.count_nonzero(eligible & (depth == 0))),
               "inference_failed_observations": int(unavailable.sum()),
               "inference_failed_molecules": float(depth[unavailable].sum()),
               "supported_prediction_observations": int(np.count_nonzero(scored & valid & (bad_entries == 0))),
               "support_violation_entries": int(bad_entries[scored & valid].sum()),
               "support_violation_molecules": float(bad_molecules[scored & valid].sum()),
               "support_violation_observations": int(np.count_nonzero(scored & valid & (bad_entries > 0))),
               "depth_offset": "observed_scoring_row_total", "probability_floor_primary": None,
               "factor_storage_roundoff_renormalized": True}
    per_row = {"depth": depth, "eligible": eligible, "scored": scored, "inference_valid": valid,
               "status": statuses, "log_likelihood": ll, "deviance": dev,
               "support_violation_entries": bad_entries, "support_violation_molecules": bad_molecules}
    if floor is not None:
        floored_ll[~scored | unavailable], floored_dev[~scored | unavailable] = np.nan, np.nan
        summary["diagnostic_floor"] = floor
        summary["diagnostic_floored_log_likelihood"] = float(np.sum(floored_ll[scored]))
        summary["diagnostic_floored_deviance"] = float(np.sum(floored_dev[scored]))
        per_row["diagnostic_floored_log_likelihood"] = floored_ll
        per_row["diagnostic_floored_deviance"] = floored_dev
    return {"summary": summary, "per_row": per_row}


def biological_score_summary(per_row, biological_ids):
    ids = np.asarray(biological_ids, dtype=str)
    if len(ids) != len(per_row["depth"]):
        raise ValueError("biological IDs do not match scoring rows")
    output = []
    for unit in np.unique(ids):
        selected = (ids == unit) & per_row["scored"]
        mass = float(per_row["depth"][selected].sum())
        ll, dev = np.sum(per_row["log_likelihood"][selected]), np.sum(per_row["deviance"][selected])
        output.append({"biological_id": unit, "observations": int(selected.sum()), "molecules": mass,
                       "log_likelihood_per_molecule": float(ll / mass) if mass else np.nan,
                       "deviance_per_molecule": float(dev / mass) if mass else np.nan,
                       "inference_failures": int(np.count_nonzero(selected & ~per_row["inference_valid"]))})
    return {"per_biological_unit": output,
            "equal_biological_unit_deviance": float(np.mean([x["deviance_per_molecule"] for x in output])) if output else np.nan,
            "equal_biological_unit_log_likelihood": float(np.mean([x["log_likelihood_per_molecule"] for x in output])) if output else np.nan,
            "standard_error": None, "uncertainty": "requires paired biological-unit aggregation across repeated splits"}


def pas10_neighbors(coordinates, strata):
    coordinates, strata = np.asarray(coordinates, float), np.asarray(strata, str)
    if coordinates.shape != (len(strata), 2) or not np.isfinite(coordinates).all():
        raise ValueError("finite 2D coordinates and matching strata required")
    neighbors, _ = neighbor_indices(coordinates, strata, k=10)
    # An incomplete neighborhood is wholly undefined, never a smaller-k PAS.
    neighbors[np.any(neighbors < 0, axis=1)] = -1
    return neighbors


def pas10_from_neighbors(labels, neighbors, strata):
    labels, neighbors, strata = np.asarray(labels), np.asarray(neighbors), np.asarray(strata, str)
    if neighbors.shape != (len(labels), 10) or len(strata) != len(labels):
        raise ValueError("PAS_10 requires exactly ten neighbor slots per observation")
    valid = np.all(neighbors >= 0, axis=1)
    if np.any(neighbors[valid] >= len(labels)):
        raise ValueError("PAS neighbor is out of range")
    ids = np.flatnonzero(valid)
    if np.any(neighbors[valid] == ids[:, None]):
        raise ValueError("PAS neighbors must exclude the observation itself")
    if np.any(strata[neighbors[valid]] != strata[ids, None]):
        raise ValueError("PAS neighbors cross a forbidden spatial stratum")
    if any(len(np.unique(row)) != 10 for row in neighbors[valid]):
        raise ValueError("PAS neighbors must be distinct")
    disagreement = np.full(len(labels), -1, np.int16)
    disagreement[valid] = np.sum(labels[neighbors[valid]] != labels[ids, None], axis=1)
    abnormal = valid & (disagreement >= 6)
    records = []
    for stratum in np.unique(strata):
        selected = strata == stratum
        covered = selected & valid
        count = int(covered.sum())
        records.append({"stratum": stratum, "observations": int(selected.sum()), "valid_observations": count,
                        "one_minus_PAS_10": float(1 - abnormal[covered].mean()) if count else np.nan,
                        "status": "ok" if count else "fewer_than_11_observations"})
    return {"one_minus_PAS_10": float(1 - abnormal[valid].mean()) if np.any(valid) else np.nan,
            "valid_observations": int(valid.sum()), "total_observations": len(labels),
            "per_stratum": records, "disagreeing_neighbors": disagreement, "abnormal": abnormal,
            "valid_mask": valid, "k": 10, "abnormal_disagreement_threshold": 6}


def spatial_metrics(W, adjacency, coordinates, strata, *, variance_tolerance=1e-14,
                    pas_neighbors=None):
    w, strata = simplex_matrix(W, "W"), np.asarray(strata, str)
    graph = sparse.csr_matrix(adjacency, dtype=float)
    if graph.shape != (len(w), len(w)) or len(strata) != len(w):
        raise ValueError("spatial graph dimensions do not match observations")
    edges = graph.tocoo()
    if not np.isfinite(edges.data).all() or np.any(edges.data < 0):
        raise ValueError("spatial edge weights must be finite and nonnegative")
    asymmetry = graph - graph.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12:
        raise ValueError("spatial evaluation graph must be symmetric")
    if np.any(strata[edges.row] != strata[edges.col]):
        raise ValueError("spatial graph crosses forbidden strata")
    if np.any(edges.row == edges.col):
        raise ValueError("spatial evaluation graph must not contain self edges")
    records, roughness_numerator, weight_sum = [], 0.0, 0.0
    for stratum in np.unique(strata):
        ids = np.flatnonzero(strata == stratum)
        local, g = w[ids], graph[ids][:, ids]
        centered = local - local.mean(axis=0)
        variances = np.mean(centered ** 2, axis=0)
        s0 = float(g.sum())
        cross = np.sum(centered * (g @ centered), axis=0)
        for topic in range(w.shape[1]):
            reason = "no_edges" if s0 <= 0 else "near_constant_topic" if variances[topic] <= variance_tolerance else "ok"
            value = float(cross[topic] / (s0 * variances[topic])) if reason == "ok" else np.nan
            records.append({"stratum": stratum, "topic": topic, "observations": int(len(ids)),
                            "topic_variance": float(variances[topic]), "edge_weight_sum": s0,
                            "moran_I": value, "status": reason})
        degree = np.asarray(g.sum(axis=1)).ravel()
        # Symmetric adjacency counts each edge twice; this is the average
        # squared difference under exactly those same directed edge weights.
        roughness_numerator += float(2 * np.sum(degree[:, None] * local ** 2) - 2 * np.sum(local * (g @ local)))
        weight_sum += s0
    valid = [r for r in records if r["status"] == "ok"]
    per_stratum_means = [np.mean([r["moran_I"] for r in valid if r["stratum"] == s])
                        for s in np.unique(strata) if any(r["stratum"] == s for r in valid)]
    labels = w.argmax(axis=1)
    sizes = np.bincount(labels, minlength=w.shape[1])
    neighbors = pas10_neighbors(coordinates, strata) if pas_neighbors is None else pas_neighbors
    pas = pas10_from_neighbors(labels, neighbors, strata)
    return {"moran_I_equal_stratum_mean": float(np.mean(per_stratum_means)) if per_stratum_means else np.nan,
            "moran_valid_topic_strata": len(valid), "moran_expected_topic_strata": len(records),
            "moran_I_observation_weighted": float(np.average([r["moran_I"] for r in valid], weights=[r["observations"] for r in valid])) if valid else np.nan,
            "moran_per_stratum_topic": records, "variance_tolerance": variance_tolerance,
            "W_edge_squared_difference": max(0.0, roughness_numerator / weight_sum) if weight_sum else np.nan,
            "occupied_topics": int(np.count_nonzero(sizes)), "topic_sizes": sizes.tolist(),
            "one_minus_PAS_10": pas["one_minus_PAS_10"], "PAS_10": pas,
            "moran_centering": "within_original_spatial_stratum"}


def hard_label_metrics(W, labels):
    w, labels = simplex_matrix(W, "W"), np.asarray(labels, dtype=str)
    valid = ~np.isin(labels, ["", "nan", "None", "<NA>"])
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"status": "insufficient_label_classes", "labeled_observations": int(valid.sum())}
    return {"status": "ok", "labeled_observations": int(valid.sum()),
            "NMI": float(normalized_mutual_info_score(labels[valid], w[valid].argmax(axis=1))),
            "ARI": float(adjusted_rand_score(labels[valid], w[valid].argmax(axis=1)))}


def grouped_label_transfer(Xtrain, ytrain, groupstrain, Xtest, ytest, *, seed=260910,
                           C_grid=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Tune a metadata decoder on training groups, then score outer test labels.

    For genotype/clinical outcomes callers must first average sections within
    animals or timepoint-specific rows within patients; no cell-level outcome
    decoding is created here. Group counts and unseen classes remain explicit.
    """
    x, xt = np.asarray(Xtrain, float), np.asarray(Xtest, float)
    y, yt, groups = np.asarray(ytrain, str), np.asarray(ytest, str), np.asarray(groupstrain, str)
    if x.ndim != 2 or xt.ndim != 2 or len(x) != len(y) or len(y) != len(groups) or len(xt) != len(yt):
        raise ValueError("incompatible metadata decoder arrays")
    if x.shape[1] != xt.shape[1] or not np.isfinite(x).all() or not np.isfinite(xt).all():
        raise ValueError("metadata decoder features must be finite and aligned")
    observed = ~np.isin(y, ["", "nan", "None", "<NA>"])
    observed_test = ~np.isin(yt, ["", "nan", "None", "<NA>"])
    x, y, groups = x[observed], y[observed], groups[observed]
    classes = np.unique(y)
    metadata = {"training_groups": int(len(np.unique(groups))), "training_classes": classes.tolist(),
                "test_classes": np.unique(yt[observed_test]).tolist(),
                "test_unseen_classes": sorted(set(yt[observed_test]) - set(classes)),
                "test_labeled_rows": int(observed_test.sum()), "test_labels_used_for_tuning": False}
    if len(classes) < 2 or not np.any(observed_test):
        return {**metadata, "status": "insufficient_class_coverage", "balanced_accuracy": np.nan}
    groups_per_class = [len(np.unique(groups[y == label])) for label in classes]
    folds = min(5, min(groups_per_class), len(np.unique(groups)))
    candidates = sorted(set(float(c) for c in C_grid))
    if not candidates or candidates[0] <= 0:
        raise ValueError("positive prespecified decoder C candidates required")
    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed))
    curves, selected = [], 1.0
    if folds >= 2:
        splits = list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(x, y, groups))
        for c in candidates:
            scores, failures = [], []
            for index, (train, test) in enumerate(splits):
                try:
                    if len(np.unique(y[train])) < 2:
                        raise ValueError("inner training split has only one class")
                    predictor = model(c).fit(x[train], y[train])
                    scores.append(float(balanced_accuracy_score(y[test], predictor.predict(x[test]))))
                except ValueError as error:
                    failures.append({"fold": index, "error": str(error)})
            curves.append({"C": c, "fold_scores": scores, "failures": failures,
                           "complete": len(scores) == folds,
                           "mean": float(np.mean(scores)) if len(scores) == folds else None})
        selectable = [r for r in curves if r["complete"]]
        if not selectable:
            return {**metadata, "status": "all_grouped_decoder_candidates_failed", "cv_curves": curves, "balanced_accuracy": np.nan}
        selected = min(selectable, key=lambda r: (-r["mean"], r["C"]))["C"]
    predictor = model(selected).fit(x, y)
    predictions = predictor.predict(xt)
    return {**metadata, "status": "ok" if folds >= 2 else "prespecified_C_insufficient_grouped_CV",
            "selected_C": selected, "inner_folds": folds, "cv_curves": curves,
            "balanced_accuracy": float(balanced_accuracy_score(yt[observed_test], predictions[observed_test])),
            "predictions": predictions, "class_coverage_complete": not metadata["test_unseen_classes"]}


def align_profiles(A, B, genesA, genesB, *, common_support=None, top_n=20):
    """A-only equal-K Hungarian JSD matching on explicit common gene support."""
    a, b = simplex_matrix(A, "A"), simplex_matrix(B, "B")
    ga, gb = np.asarray(genesA, str), np.asarray(genesB, str)
    if len(ga) != a.shape[1] or len(gb) != b.shape[1] or a.shape[0] != b.shape[0]:
        raise ValueError("profile stability requires equal K and matching gene IDs")
    if len(set(ga)) != len(ga) or len(set(gb)) != len(gb):
        raise ValueError("profile gene IDs must be unique")
    support = sorted(set(ga) & set(gb)) if common_support is None else list(map(str, common_support))
    if len(support) != len(set(support)) or not set(support) <= set(ga) & set(gb):
        raise ValueError("declared common support must be unique and present in both profiles")
    if not support:
        return {"status": "empty_common_support", "support_size": 0, "labels_used": False}
    ia, ib = {g: j for j, g in enumerate(ga)}, {g: j for j, g in enumerate(gb)}
    ac, bc = a[:, [ia[g] for g in support]], b[:, [ib[g] for g in support]]
    ma, mb = ac.sum(axis=1), bc.sum(axis=1)
    base = {"support_size": len(support), "common_gene_ids": support,
            "retained_mass_A": ma.tolist(), "retained_mass_B": mb.tolist(), "labels_used": False}
    if np.any(ma == 0) or np.any(mb == 0):
        return {**base, "status": "topic_has_zero_mass_on_common_support"}
    ac, bc = ac / ma[:, None], bc / mb[:, None]
    # Compute divergence directly. scipy's sqrt(divergence)**2 can produce
    # NaN when a theoretically zero divergence rounds slightly negative.
    cost = np.empty((len(ac), len(bc)))
    for i, x in enumerate(ac):
        for j, y in enumerate(bc):
            middle = .5 * (x + y)
            divergence = float((np.sum(rel_entr(x, middle)) + np.sum(rel_entr(y, middle))) / (2 * np.log(2)))
            if divergence < -1e-12:
                raise ValueError("JSD unexpectedly negative beyond floating-point roundoff")
            cost[i, j] = max(0.0, divergence)
    rows, columns = linear_sum_assignment(cost)
    overlaps, cosines = [], []
    for i, j in zip(rows, columns):
        sa = set(np.argsort(-ac[i], kind="stable")[:min(top_n, len(support))])
        sb = set(np.argsort(-bc[j], kind="stable")[:min(top_n, len(support))])
        overlaps.append(len(sa & sb) / len(sa | sb))
        cosines.append(float(np.dot(ac[i], bc[j]) / (np.linalg.norm(ac[i]) * np.linalg.norm(bc[j]))))
    return {**base, "status": "ok", "assignment_A": rows.tolist(), "assignment_B": columns.tolist(),
            "JSD_base2": cost[rows, columns].tolist(), "JSD_mean": float(cost[rows, columns].mean()),
            "cosine_mean": float(np.mean(cosines)), "top_gene_jaccard_mean": float(np.mean(overlaps)),
            "top_n": min(top_n, len(support)), "alignment_objective": "minimum_sum_base2_JSD_A_only"}


def topic_profile_metrics(A):
    a = simplex_matrix(A, "A")
    entropy = -np.sum(xlogy(a, a), axis=1)
    if a.shape[1] > 1:
        entropy /= np.log(a.shape[1])
    column_mass = a.sum(axis=0)
    exclusivity = np.divide(a, column_mass[None, :], out=np.zeros_like(a), where=column_mass[None, :] > 0)
    return {"normalized_entropy": entropy.tolist(), "mean_entropy": float(entropy.mean()),
            "topic_feature_exclusivity": exclusivity, "zero_frequency_features": int(np.count_nonzero(column_mass == 0))}
