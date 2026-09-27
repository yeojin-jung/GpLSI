"""Primary predictive, external-structure, interpretability, and spatial metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


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


def spatial_metrics(W: np.ndarray, edge_frame: pd.DataFrame) -> dict[str, float]:
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
    return {
        "spatial_status": "ok",
        "spatial_W_edge_squared_difference": float(np.average(np.sum(delta**2, axis=1), weights=weights)),
        "spatial_hard_topic_neighbor_agreement": float(np.average(labels[endpoints[:, 0]] == labels[endpoints[:, 1]], weights=weights)),
        "spatial_topic_moran_mean": float(np.mean(morans)) if morans else float("nan"),
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
    }, genes
