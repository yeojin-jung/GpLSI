"""Scores for one fitted (W, A) pair.

* :func:`fit_diagnostics` - training-count fit, simplex checks, and graph
  smoothness of W;
* :func:`heldout_count_metrics` - multinomial likelihood of held-out counts;
* :func:`external_structure_metrics` - agreement of W with evaluation-only
  labels (e.g. DLPFC layers), which are never used for fitting;
* :func:`spatial_metrics` - W smoothness on the graph (roughness, Moran's I,
  neighbour agreement) and, with coordinates, CHAOS and PAS of hard topics
  (:func:`spatial_domain_metrics`);
* :func:`topic_profile_metrics` - interpretability and distinctness of A
  (entropy, exclusivity, pairwise cosine, topic diversity);
* :func:`topic_prevalence_metrics` - topic sizes from W.

:func:`evaluate_fit` combines them for the experiment runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..real_data import RealDataBundle
from ..recovery import poisson_objective_and_gradient


HELDOUT_SMOOTHING = (1e-4, 1e-3, 1e-2)


def heldout_count_metrics(W: np.ndarray, A: np.ndarray, test: np.ndarray) -> dict[str, float]:
    probability = np.asarray(W, dtype=float) @ np.asarray(A, dtype=float)
    if not np.isfinite(probability).all() or np.any(probability < -1e-12):
        raise ValueError("held-out probabilities must be finite and nonnegative")
    probability = np.maximum(probability, 0.0)
    row_mass = probability.sum(axis=1, keepdims=True)
    probability = np.divide(
        probability,
        row_mass,
        out=np.zeros_like(probability),
        where=row_mass > 0,
    )
    test = np.asarray(test, dtype=float)
    lengths = test.sum(axis=1)
    keep = lengths > 0
    if not np.any(keep):
        return {"heldout_status": "no_positive_test_rows"}
    y = test[keep]
    p = probability[keep]
    n = lengths[keep]
    mean = n[:, None] * p
    positive = y > 0
    impossible = positive & (p <= 0)
    impossible_entries = int(np.count_nonzero(impossible))
    impossible_molecules = int(y[impossible].sum())
    impossible_rows = int(np.count_nonzero(np.any(impossible, axis=1)))
    if impossible_molecules:
        log_likelihood = float("-inf")
        poisson_deviance = float("inf")
        status = "positive_test_count_has_zero_probability"
    else:
        log_likelihood = float(np.sum(y[positive] * np.log(p[positive])))
        poisson_deviance = float(2 * (
            np.sum(y[positive] * np.log(y[positive] / mean[positive]))
            - np.sum(y - mean)
        ))
        status = "ok"

    # Smoothed deviance: mix the prediction with the uniform distribution,
    # p_eps = (1 - eps) p + eps / p_vocab.  Finite for every fit, so methods
    # whose A has exact zeros stay comparable; a zero-probability count still
    # costs about log(p_vocab / eps).  It also caps tiny positive probabilities,
    # which otherwise dominate the exact deviance.  eps = 1e-3 is the
    # protocol's main value.
    smoothed: dict[str, float] = {}
    for eps in HELDOUT_SMOOTHING:
        p_eps = (1.0 - eps) * p + eps / p.shape[1]
        mean_eps = n[:, None] * p_eps
        deviance = 2 * (np.sum(y[positive] * np.log(y[positive] / mean_eps[positive])) - np.sum(y - mean_eps))
        smoothed[f"heldout_poisson_deviance_per_molecule_smoothed_{eps:.0e}"] = float(deviance / y.sum())

    # Preserve the former floored calculation as an explicitly named
    # diagnostic.  It is never substituted for the exact predictive score.
    floored = np.maximum(p, 1e-12)
    floored /= floored.sum(axis=1, keepdims=True)
    floored_mean = n[:, None] * floored
    floored_log_likelihood = float(np.sum(y * np.log(floored)))
    floored_deviance = float(2 * (
        np.sum(y[positive] * np.log(y[positive] / floored_mean[positive]))
        - np.sum(y - floored_mean)
    ))
    return {
        "heldout_status": status,
        "heldout_log_likelihood_without_constant": log_likelihood,
        "heldout_log_likelihood_per_molecule": float(log_likelihood / y.sum()),
        "heldout_poisson_deviance": poisson_deviance,
        "heldout_poisson_deviance_per_molecule": float(poisson_deviance / y.sum()),
        "heldout_zero_probability_positive_entries": impossible_entries,
        "heldout_zero_probability_molecules": impossible_molecules,
        "heldout_zero_probability_rows": impossible_rows,
        **smoothed,
        "heldout_log_likelihood_without_constant_floored_1e-12": floored_log_likelihood,
        "heldout_log_likelihood_per_molecule_floored_1e-12": float(
            floored_log_likelihood / y.sum()
        ),
        "heldout_poisson_deviance_floored_1e-12": floored_deviance,
        "heldout_poisson_deviance_per_molecule_floored_1e-12": float(
            floored_deviance / y.sum()
        ),
        "heldout_molecules": int(y.sum()),
        "heldout_rows": int(np.count_nonzero(keep)),
    }


def external_structure_metrics(W: np.ndarray, external: pd.DataFrame, seed: int) -> dict[str, float]:
    """Measure label recovery without exposing any external column to fitting."""

    output: dict[str, float] = {}
    hard = np.argmax(W, axis=1)
    for column in external.columns:
        values = external[column]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_array = numeric.to_numpy(dtype=float, na_value=np.nan)
        usable_numeric = np.isfinite(numeric_array)
        categorical = values.astype("string").fillna("").astype(str).to_numpy()
        usable_category = categorical != ""
        unique_category = np.unique(categorical[usable_category]).size
        if 2 <= unique_category <= min(100, max(2, np.count_nonzero(usable_category) // 3)):
            y = categorical[usable_category]
            x = W[usable_category]
            counts = pd.Series(y).value_counts()
            output[f"external__{column}__nmi"] = float(normalized_mutual_info_score(y, hard[usable_category]))
            output[f"external__{column}__ari"] = float(adjusted_rand_score(y, hard[usable_category]))
            folds = int(min(5, counts.min()))
            if folds >= 2:
                model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
                cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
                try:
                    score = cross_val_score(model, x, y, cv=cv, scoring="balanced_accuracy")
                    output[f"external__{column}__cv_balanced_accuracy"] = float(score.mean())
                except ValueError:
                    pass
        elif np.count_nonzero(usable_numeric) >= 10 and np.unique(numeric_array[usable_numeric]).size >= 5:
            correlations = [
                abs(float(spearmanr(W[usable_numeric, topic], numeric_array[usable_numeric]).statistic))
                for topic in range(W.shape[1])
            ]
            output[f"external__{column}__max_abs_spearman"] = float(np.nanmax(correlations))
    return output


def spatial_metrics(
    W: np.ndarray,
    edge_frame: pd.DataFrame,
    coordinates: np.ndarray | None = None,
    units: np.ndarray | None = None,
) -> dict[str, float]:
    endpoints = edge_frame[["src", "tgt"]].to_numpy(dtype=int)
    weights = edge_frame["weight"].to_numpy(dtype=float)
    if endpoints.size == 0:
        return {"spatial_status": "no_edges"}
    delta = W[endpoints[:, 0]] - W[endpoints[:, 1]]
    labels = np.argmax(W, axis=1)
    morans = []
    n = W.shape[0]
    for topic in range(W.shape[1]):
        centered = W[:, topic] - W[:, topic].mean()
        denominator = centered @ centered
        if denominator > 0:
            morans.append(n / weights.sum() * np.sum(weights * centered[endpoints[:, 0]] * centered[endpoints[:, 1]]) / denominator)
    output = {
        "spatial_status": "ok",
        "spatial_W_edge_squared_difference": float(np.average(np.sum(delta**2, axis=1), weights=weights)),
        "spatial_hard_topic_neighbor_agreement": float(np.average(labels[endpoints[:, 0]] == labels[endpoints[:, 1]], weights=weights)),
        "spatial_topic_moran_mean": float(np.mean(morans)) if morans else float("nan"),
    }
    if coordinates is not None:
        output.update(spatial_domain_metrics(labels, coordinates, units))
    return output


def spatial_domain_metrics(
    labels: np.ndarray,
    coordinates: np.ndarray,
    units: np.ndarray | None = None,
    *,
    pas_neighbors: int = 10,
    pas_threshold: int = 6,
) -> dict[str, float]:
    """CHAOS and PAS of hard topics, as defined for SpatialPCA (Shang & Zhou, 2022).

    Both use coordinates within each unit (region, spleen, section), never
    across units. CHAOS: for every topic, each of its points contributes the
    distance to its nearest point of the same topic; the sum is divided by n.
    Distances are in units of the unit's median nearest-neighbour spacing, so
    values compare across tissues. PAS: share of points whose topic differs
    from at least ``pas_threshold`` of their ``pas_neighbors`` nearest points.
    Lower is spatially more coherent for both. Points alone in their topic
    within a unit have no same-topic neighbour and are counted separately.
    """

    from sklearn.neighbors import NearestNeighbors

    xy = np.asarray(coordinates, dtype=float)
    labels = np.asarray(labels)
    units = np.zeros(len(labels), dtype=int) if units is None else np.asarray(units).astype(str)
    chaos_sum, abnormal, evaluated, singletons = 0.0, 0, 0, 0
    for unit in np.unique(units):
        index = np.flatnonzero(units == unit)
        if index.size < 2:
            continue
        points = xy[index]
        spacing, _ = NearestNeighbors(n_neighbors=2).fit(points).kneighbors(points)
        scale = float(np.median(spacing[:, 1]))
        scale = scale if scale > 0 else 1.0
        unit_labels = labels[index]
        for topic in np.unique(unit_labels):
            members = points[unit_labels == topic]
            if len(members) < 2:
                singletons += len(members)
                continue
            distances, _ = NearestNeighbors(n_neighbors=2).fit(members).kneighbors(members)
            chaos_sum += float(distances[:, 1].sum()) / scale
        neighbors = min(pas_neighbors, index.size - 1)
        _, nearest = NearestNeighbors(n_neighbors=neighbors + 1).fit(points).kneighbors(points)
        different = (unit_labels[nearest[:, 1:]] != unit_labels[:, None]).sum(axis=1)
        abnormal += int(np.sum(different >= min(pas_threshold, neighbors)))
        evaluated += index.size
    return {
        "spatial_CHAOS": chaos_sum / len(labels),
        "spatial_PAS": abnormal / evaluated if evaluated else float("nan"),
        "spatial_CHAOS_singleton_points": int(singletons),
    }


def topic_profile_metrics(A: np.ndarray, feature_ids: np.ndarray, top_n: int = 20) -> tuple[dict, list[list[str]]]:
    A = np.maximum(np.asarray(A, dtype=float), 1e-15)
    A /= A.sum(axis=1, keepdims=True)
    entropy = -np.sum(A * np.log(A), axis=1) / np.log(A.shape[1])
    across = A / np.maximum(A.sum(axis=0, keepdims=True), 1e-15)
    exclusivity = np.max(across, axis=0)
    top = np.argsort(-A, axis=1)[:, : min(top_n, A.shape[1])]
    genes = [[str(feature_ids[index]) for index in row] for row in top]
    top_exclusivity = [float(np.mean(exclusivity[row])) for row in top]
    return {
        "topic_entropy_mean": float(entropy.mean()),
        "topic_entropy_sd": float(entropy.std()),
        "top_gene_exclusivity_mean": float(np.mean(top_exclusivity)),
        "topic_minimum_mass": float(A.sum(axis=1).min()),
        **topic_overlap_metrics(A),
    }, genes


def topic_overlap_metrics(A: np.ndarray, top: tuple[int, ...] = (3, 10, 25)) -> dict[str, float]:
    """How distinct the topics (rows of A) are.

    * pairwise cosine similarity of topic rows: mean and max over pairs (a max
      near 1 flags a duplicated topic; Cao et al., 2009);
    * topic diversity: share of unique features among each topic's top t
      (Dieng, Ruiz & Blei, 2020, use t = 25), with t capped at p.
    """

    A = np.asarray(A, dtype=float)
    K, p = A.shape
    output: dict[str, float] = {}
    if K > 1:
        unit = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
        cosine = (unit @ unit.T)[np.triu_indices(K, 1)]
        output.update(topic_cosine_mean=float(cosine.mean()), topic_cosine_max=float(cosine.max()))
    for t in top:
        t_used = min(t, p)
        ranked = np.argsort(-A, axis=1, kind="stable")[:, :t_used]
        output[f"topic_diversity_top{t}"] = float(np.unique(ranked).size / (K * t_used))
    return output


def topic_prevalence_metrics(W: np.ndarray) -> dict[str, Any]:
    """Topic sizes from W, in the fit's own topic order.

    ``argmax``: share of documents whose dominant topic is k (ties to the
    lowest topic); ``mean_W``: average topic weight. ``effective_number`` is
    the exponential of the entropy of the argmax shares.
    """

    W = np.asarray(W, dtype=float)
    K = W.shape[1]
    argmax = np.bincount(np.argmax(W, axis=1), minlength=K) / W.shape[0]
    positive = argmax[argmax > 0]
    return {
        "topic_prevalence_argmax": argmax.tolist(),
        "topic_prevalence_mean_W": W.mean(axis=0).tolist(),
        "topic_effective_number": float(np.exp(-np.sum(positive * np.log(positive)))),
        "empty_topic_count": int(np.sum(argmax == 0)),
    }


def fit_diagnostics(
    bundle: RealDataBundle,
    W_hat: np.ndarray,
    A_hat: np.ndarray,
) -> dict[str, float]:
    W = np.asarray(W_hat, dtype=float)
    A = np.asarray(A_hat, dtype=float)
    reconstruction = W @ A
    residual = bundle.frequencies - reconstruction
    poisson, gradient = poisson_objective_and_gradient(
        A, W, bundle.counts, bundle.document_lengths, 1e-12
    )
    endpoints = bundle.edge_df[["src", "tgt"]].to_numpy(dtype=int)
    edge_weights = bundle.edge_df["weight"].to_numpy(dtype=float)
    if endpoints.size:
        differences = W[endpoints[:, 0]] - W[endpoints[:, 1]]
        smoothness = float(np.average(np.sum(differences**2, axis=1), weights=edge_weights))
        labels = np.argmax(W, axis=1)
        neighbor_agreement = float(
            np.average(labels[endpoints[:, 0]] == labels[endpoints[:, 1]], weights=edge_weights)
        )
        centered_numeric = labels.astype(float) - float(np.mean(labels))
        numeric_denominator = float(np.sum(centered_numeric**2) * np.sum(edge_weights))
        historical_moran = (
            np.nan
            if numeric_denominator <= 0
            else float(
                bundle.n
                * np.sum(
                    edge_weights
                    * centered_numeric[endpoints[:, 0]]
                    * centered_numeric[endpoints[:, 1]]
                )
                / numeric_denominator
            )
        )
        incident: dict[int, list[float]] = {}
        disagreements = labels[endpoints[:, 0]] != labels[endpoints[:, 1]]
        for side in (0, 1):
            for node in np.unique(endpoints[:, side]):
                incident.setdefault(int(node), []).append(
                    float(np.mean(disagreements[endpoints[:, side] == node]))
                )
        summed_incident = np.asarray([sum(values) for values in incident.values()])
        historical_one_minus_pas = float(1.0 - np.mean(summed_incident >= 0.6))
        one_hot_moran: list[float] = []
        for topic in range(W.shape[1]):
            indicator = (labels == topic).astype(float)
            centered = indicator - indicator.mean()
            denominator = float(np.sum(centered**2) * np.sum(edge_weights))
            if denominator > 0:
                one_hot_moran.append(
                    float(
                        bundle.n
                        * np.sum(
                            edge_weights
                            * centered[endpoints[:, 0]]
                            * centered[endpoints[:, 1]]
                        )
                        / denominator
                    )
                )
    else:
        smoothness = np.nan
        neighbor_agreement = np.nan
        historical_moran = np.nan
        historical_one_minus_pas = np.nan
        one_hot_moran = []
    return {
        "reconstruction_fro": float(np.linalg.norm(residual, ord="fro")),
        "reconstruction_l1": float(np.sum(np.abs(residual))),
        "poisson_objective": float(poisson),
        "poisson_gradient_norm": float(np.linalg.norm(gradient)),
        "graph_W_smoothness": smoothness,
        "graph_neighbor_topic_agreement": neighbor_agreement,
        "historical_numeric_label_moran": historical_moran,
        "historical_1_minus_PAS": historical_one_minus_pas,
        "permutation_invariant_one_hot_moran_mean": (
            np.nan if not one_hot_moran else float(np.mean(one_hot_moran))
        ),
        "W_simplex_error": float(np.max(np.abs(W.sum(axis=1) - 1.0))),
        "W_min": float(W.min()),
        "A_simplex_error": float(np.max(np.abs(A.sum(axis=1) - 1.0))),
        "A_min": float(A.min()),
        "W_rank": int(np.linalg.matrix_rank(W)),
        "A_rank": int(np.linalg.matrix_rank(A)),
    }


def evaluate_fit(data, W: np.ndarray, A: np.ndarray, seed: int) -> dict:
    """All scores for one fit on a :class:`~gplsi.pipeline.datasets.TaskData`."""

    output = {"diagnostics": fit_diagnostics(data.bundle, W, A)}
    if data.heldout is not None:
        heldout = heldout_count_metrics(W, A, data.heldout)
        if data.reference_columns is not None:
            # Common vocabulary for comparisons across panel sizes (the
            # conditional composition on the reference panel).
            ref = data.reference_columns
            reference = heldout_count_metrics(W, A[:, ref], data.heldout[:, ref])
            heldout.update({f"reference_panel__{key}": value for key, value in reference.items()})
        output["heldout_metrics"] = heldout
    bundle = data.bundle
    metrics = spatial_metrics(W, bundle.edge_df, bundle.coordinates, bundle.group_ids)
    metrics.update(topic_prevalence_metrics(W))
    if data.labels is not None:
        metrics.update(external_structure_metrics(W, data.labels, seed))
    profile, top_features = topic_profile_metrics(A, data.feature_names)
    metrics.update(profile)
    output["metrics"] = metrics
    output["top_features"] = top_features
    return output
