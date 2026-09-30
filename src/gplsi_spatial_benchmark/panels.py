"""Training-only gene panels for whole-transcriptome platforms (e.g. Visium)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse


def rank_features_by_dispersion(train: np.ndarray, *, detection_fraction: float = 0.01) -> dict:
    """Rank genes by raw variance-to-mean ratio using training counts only.

    Genes must be detected in at least ``max(1, floor(detection_fraction * n))``
    training rows. Ties keep the original column order, so panels of different
    sizes are nested prefixes of one ranking.
    """

    counts = csr_matrix(train, dtype=np.float64) if issparse(train) else csr_matrix(np.asarray(train, dtype=np.float64))
    counts.eliminate_zeros()
    n = counts.shape[0]
    if n == 0:
        raise ValueError("no training rows")
    threshold = max(1, int(detection_fraction * n))
    detected = np.bincount(counts.indices, minlength=counts.shape[1])
    mean = np.asarray(counts.sum(axis=0)).ravel() / n
    second = np.asarray(counts.multiply(counts).sum(axis=0)).ravel() / n
    variance = np.maximum(second - mean**2, 0.0)
    ratio = np.divide(variance, mean, out=np.zeros_like(mean), where=mean > 0)
    eligible = np.flatnonzero((detected >= threshold) & (mean > 0))
    ranked = eligible[np.argsort(-ratio[eligible], kind="stable")]
    return {
        "ranked_indices": ranked,
        "detection_threshold": threshold,
        "eligible_count": int(eligible.size),
    }


def panel_indices(ranking: dict, size: int) -> np.ndarray:
    ranked = ranking["ranked_indices"]
    if size > ranked.size:
        raise ValueError(f"requested panel of {size} genes but only {ranked.size} are eligible")
    return np.asarray(ranked[:size], dtype=int)
