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


def tran_vocabulary(train, *, alpha: float) -> dict:
    """Genes kept by the Tran threshold on training counts (sparse, all genes).

    With X = D / N the training row frequencies, eta_j = mean_i X_ij and
    N_bar the mean training length, keep j when
    ``eta_j >= alpha * sqrt(log(max(n, p)) / (n * N_bar))`` (the paper rule,
    ``select_feature_columns(method="tran_paper_exact")``). There is no top-10%
    fallback: at alpha = 0.1 on all Visium genes fewer than 10% pass, and the
    fallback would replace the cut by a fixed-size panel. Rows with no training
    count are ignored. Indices are in the original gene order.
    """

    counts = csr_matrix(train, dtype=np.float64) if issparse(train) else csr_matrix(np.asarray(train, dtype=np.float64))
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    counts = counts[lengths > 0]
    lengths = lengths[lengths > 0]
    n, p = counts.shape
    if n == 0:
        raise ValueError("no training rows")
    eta = np.asarray(counts.multiply(1.0 / lengths[:, None]).sum(axis=0)).ravel() / n
    threshold = float(alpha * np.sqrt(np.log(max(n, p)) / (n * lengths.mean())))
    kept = np.flatnonzero(eta >= threshold)
    if kept.size == 0:
        raise ValueError(f"the Tran threshold at alpha={alpha} keeps no gene")
    return {
        "indices": kept,
        "alpha": float(alpha),
        "threshold": threshold,
        "kept_count": int(kept.size),
        "kept_mass": float(np.asarray(counts[:, kept].sum()) / lengths.sum()),
    }
