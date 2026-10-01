#!/usr/bin/env python3
"""Downstream evaluation of spleen embeddings against manual compartments.

The labels used here are never passed to a topic-model fit.  The joint-model
linear probe is intentionally leave-one-spleen-out: its classifier sees the
labels from two biological spleens and is evaluated on the third.  Because W
itself was fitted without labels on all three spleens, this is a transductive
embedding evaluation rather than an inductive prediction experiment.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    completeness_score,
    confusion_matrix,
    f1_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COMPARTMENTS = ("B-zone", "marginal zone", "PALS", "red pulp")
UNLABELED = "NoAnnotation"


def _row_normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"embedding must be a nonempty matrix, received {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding contains nonfinite values")
    if float(matrix.min(initial=0.0)) < -1e-9:
        raise ValueError("embedding contains materially negative values")
    matrix = np.maximum(matrix, 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= np.finfo(float).eps):
        raise ValueError("embedding contains an all-zero row")
    return matrix / totals


def _stratified_indices(
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic, approximately class-balanced silhouette sample."""

    rng = np.random.default_rng(seed)
    observed = [label for label in COMPARTMENTS if np.any(labels == label)]
    if not observed:
        return np.asarray([], dtype=np.int64)
    per_class = max(2, max_samples // len(observed))
    selected: list[np.ndarray] = []
    for label in observed:
        candidates = np.flatnonzero(labels == label)
        take = min(len(candidates), per_class)
        selected.append(np.sort(rng.choice(candidates, size=take, replace=False)))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def _label_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[float | None, int]:
    indices = _stratified_indices(labels, max_samples=max_samples, seed=seed)
    sampled_labels = labels[indices]
    if indices.size < 3 or np.unique(sampled_labels).size < 2:
        return None, int(indices.size)
    return (
        float(silhouette_score(embedding[indices], sampled_labels, metric="euclidean")),
        int(indices.size),
    )


def _alignment_metrics(hard_topics: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "ami": float(adjusted_mutual_info_score(labels, hard_topics)),
        "ari": float(adjusted_rand_score(labels, hard_topics)),
        "homogeneity": float(homogeneity_score(labels, hard_topics)),
        "completeness": float(completeness_score(labels, hard_topics)),
    }


def _hungarian_exact_match(
    hard_topics: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_topics: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Align four hard topics to the four compartments exactly once.

    The permutation is selected on all labeled cells.  Per-spleen scores then
    reuse that global mapping; they never optimize a separate permutation for
    each spleen.
    """

    unavailable = {
        "hungarian_exact_match_accuracy": None,
        "hungarian_exact_match_matched_cells": None,
        "hungarian_topic_to_compartment": None,
        "hungarian_exact_match_mapping_scope": None,
    }
    if int(n_topics) != len(COMPARTMENTS):
        return unavailable, {}

    topics = np.asarray(hard_topics, dtype=int)
    if np.any((topics < 0) | (topics >= n_topics)):
        raise ValueError(
            "hard topic assignments fall outside the declared topic count"
        )

    label_indices = {label: index for index, label in enumerate(COMPARTMENTS)}
    unexpected = sorted(set(map(str, np.unique(labels))) - set(COMPARTMENTS))
    if unexpected:
        raise ValueError(f"unexpected labeled compartments: {unexpected}")

    contingency = np.zeros((n_topics, len(COMPARTMENTS)), dtype=np.int64)
    for topic, compartment in zip(topics, labels, strict=True):
        contingency[int(topic), label_indices[str(compartment)]] += 1

    topic_indices, compartment_indices = linear_sum_assignment(-contingency)
    topic_to_compartment = {
        int(topic): COMPARTMENTS[int(compartment)]
        for topic, compartment in zip(
            topic_indices, compartment_indices, strict=True
        )
    }
    predicted = np.asarray(
        [topic_to_compartment[int(topic)] for topic in topics], dtype=str
    )
    matched = predicted == labels
    mapping = [
        {
            "topic_index_zero_based": int(topic),
            "compartment": str(topic_to_compartment[topic]),
            "matched_cells": int(
                contingency[topic, label_indices[topic_to_compartment[topic]]]
            ),
        }
        for topic in range(n_topics)
    ]
    global_result = {
        "hungarian_exact_match_accuracy": float(np.mean(matched)),
        "hungarian_exact_match_matched_cells": int(matched.sum()),
        "hungarian_topic_to_compartment": mapping,
        "hungarian_exact_match_mapping_scope": "all_labeled_cells_global",
    }
    per_group = {}
    for group in sorted(map(str, np.unique(groups))):
        mask = groups == group
        per_group[group] = {
            "hungarian_exact_match_accuracy_global_mapping": float(
                np.mean(matched[mask])
            ),
            "hungarian_exact_match_matched_cells_global_mapping": int(
                matched[mask].sum()
            ),
        }
    return global_result, per_group


def _topic_compartment_profiles(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    hard_topics = np.argmax(probabilities, axis=1)
    mean_w: list[list[float]] = []
    argmax_proportions: list[list[float]] = []
    contingency = np.zeros((probabilities.shape[1], len(COMPARTMENTS)), dtype=int)
    for class_index, compartment in enumerate(COMPARTMENTS):
        mask = labels == compartment
        mean_w.append(probabilities[mask].mean(axis=0).tolist())
        counts = np.bincount(hard_topics[mask], minlength=probabilities.shape[1])
        argmax_proportions.append((counts / max(1, counts.sum())).tolist())
        contingency[:, class_index] = counts
    topic_mass = np.zeros((probabilities.shape[1], len(COMPARTMENTS)), dtype=float)
    for class_index, compartment in enumerate(COMPARTMENTS):
        topic_mass[:, class_index] = probabilities[labels == compartment].sum(axis=0)
    denominators = topic_mass.sum(axis=1, keepdims=True)
    topic_composition = np.divide(
        topic_mass,
        denominators,
        out=np.zeros_like(topic_mass),
        where=denominators > 0,
    )
    return {
        "compartments": list(COMPARTMENTS),
        "compartment_mean_w": np.round(mean_w, 7).tolist(),
        "compartment_argmax_topic_proportions": np.round(
            argmax_proportions, 7
        ).tolist(),
        "topic_compartment_composition": np.round(topic_composition, 7).tolist(),
        "argmax_contingency": contingency.tolist(),
    }


def _leave_one_spleen_out_probe(
    embedding: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any] | None:
    spleens = sorted(map(str, np.unique(groups)))
    if len(spleens) != 3:
        return None
    folds: list[dict[str, Any]] = []
    for heldout in spleens:
        train = groups != heldout
        test = groups == heldout
        if np.unique(labels[train]).size != len(COMPARTMENTS):
            raise ValueError(f"training fold for {heldout} does not contain all compartments")
        classifier = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=1.0, class_weight="balanced"),
        )
        # Some macOS Accelerate builds emit spurious overflow diagnostics in
        # otherwise finite small-dimensional BLAS products.  The fitted
        # inputs and predictions are validated, so keep dashboard builds quiet.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            classifier.fit(embedding[train], labels[train])
            prediction = classifier.predict(embedding[test])
        folds.append(
            {
                "heldout_spleen": heldout,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "accuracy": float(accuracy_score(labels[test], prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test], prediction)
                ),
                "macro_f1": float(
                    f1_score(
                        labels[test],
                        prediction,
                        labels=list(COMPARTMENTS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "majority_accuracy": float(
                    max(np.mean(labels[test] == value) for value in COMPARTMENTS)
                ),
                "confusion_matrix": confusion_matrix(
                    labels[test], prediction, labels=list(COMPARTMENTS)
                ).tolist(),
            }
        )
    return {
        "classifier": "leave-one-spleen-out ridge linear probe",
        "folds": folds,
        "accuracy": float(np.mean([row["accuracy"] for row in folds])),
        "balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in folds])
        ),
        "macro_f1": float(np.mean([row["macro_f1"] for row in folds])),
        "accuracy_std": float(np.std([row["accuracy"] for row in folds], ddof=1)),
        "balanced_accuracy_std": float(
            np.std([row["balanced_accuracy"] for row in folds], ddof=1)
        ),
        "macro_f1_std": float(np.std([row["macro_f1"] for row in folds], ddof=1)),
    }


def evaluate_spleen_embedding(
    values: np.ndarray,
    compartments: np.ndarray,
    group_ids: np.ndarray,
    *,
    hard_topics: np.ndarray | None = None,
    hard_topic_count: int | None = None,
    compute_loso: bool = False,
    silhouette_max_samples: int = 800,
    seed: int = 260912,
) -> dict[str, Any]:
    """Evaluate one nonnegative embedding without exposing labels to its fit."""

    probabilities = _row_normalize(values)
    labels_all = np.asarray(compartments, dtype=str)
    groups_all = np.asarray(group_ids, dtype=str)
    if len(labels_all) != len(probabilities) or len(groups_all) != len(probabilities):
        raise ValueError("embedding, labels, and group ids must be row-aligned")
    labeled = labels_all != UNLABELED
    labels = labels_all[labeled]
    groups = groups_all[labeled]
    probabilities = probabilities[labeled]
    embedding = np.sqrt(probabilities)
    if hard_topics is None:
        hard = np.argmax(probabilities, axis=1)
        n_hard_topics = int(probabilities.shape[1])
    else:
        supplied = np.asarray(hard_topics)
        if len(supplied) == len(labeled):
            supplied = supplied[labeled]
        if len(supplied) != len(labels):
            raise ValueError("hard topic assignments are not row-aligned")
        hard = supplied.astype(int)
        n_hard_topics = (
            int(hard_topic_count)
            if hard_topic_count is not None
            else int(probabilities.shape[1])
        )

    exact_match, exact_match_per_group = _hungarian_exact_match(
        hard,
        labels,
        groups,
        n_topics=n_hard_topics,
    )

    output: dict[str, Any] = {
        "n_total": int(len(labeled)),
        "n_labeled": int(labeled.sum()),
        "annotation_coverage": float(labeled.mean()),
        **_alignment_metrics(hard, labels),
        **exact_match,
        **_topic_compartment_profiles(probabilities, labels),
    }
    silhouette, sample_n = _label_silhouette(
        embedding,
        labels,
        max_samples=silhouette_max_samples,
        seed=seed,
    )
    output["label_silhouette"] = silhouette
    output["silhouette_sample_n"] = sample_n

    per_spleen: list[dict[str, Any]] = []
    for offset, group in enumerate(sorted(map(str, np.unique(groups)))):
        mask = groups == group
        group_silhouette, group_sample_n = _label_silhouette(
            embedding[mask],
            labels[mask],
            max_samples=max(200, silhouette_max_samples // 3),
            seed=seed + offset + 1,
        )
        per_spleen.append(
            {
                "spleen": group,
                "n_labeled": int(mask.sum()),
                **_alignment_metrics(hard[mask], labels[mask]),
                **exact_match_per_group.get(
                    group,
                    {
                        "hungarian_exact_match_accuracy_global_mapping": None,
                        "hungarian_exact_match_matched_cells_global_mapping": None,
                    },
                ),
                "label_silhouette": group_silhouette,
                "silhouette_sample_n": group_sample_n,
            }
        )
    output["per_spleen"] = per_spleen
    output["loso_probe"] = (
        _leave_one_spleen_out_probe(embedding, labels, groups)
        if compute_loso
        else None
    )
    return output


def evaluate_raw_feature_baseline(
    frequencies: np.ndarray,
    compartments: np.ndarray,
    group_ids: np.ndarray,
    *,
    K: int,
    compute_loso: bool,
    silhouette_max_samples: int = 800,
    seed: int = 260912,
) -> dict[str, Any]:
    """Evaluate the canonical 24-D neighborhood composition without a topic fit."""

    probabilities = _row_normalize(frequencies)
    clusterer = MiniBatchKMeans(
        n_clusters=int(K),
        random_state=seed + int(K),
        batch_size=4096,
        n_init=5,
        max_iter=100,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        hard_topics = clusterer.fit_predict(np.sqrt(probabilities))
    return evaluate_spleen_embedding(
        probabilities,
        compartments,
        group_ids,
        hard_topics=hard_topics,
        hard_topic_count=int(K),
        compute_loso=compute_loso,
        silhouette_max_samples=silhouette_max_samples,
        seed=seed,
    )
