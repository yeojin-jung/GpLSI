"""Column thresholding and corpus-frequency weighting for GpLSI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


THRESHOLD_METHODS = (
    "none",
    "tran",
    "tran_script_exact",
    "tran_paper_exact",
)
WEIGHT_METHODS = (
    "none",
    "ke_empirical",
    "ke_empirical_additive_floor",
    "ke_empirical_capped",
    "oracle_eta",
    "oracle_h",
)


class PreprocessingError(ValueError):
    """Raised when preprocessing is undefined or destroys required rank."""


@dataclass
class ThresholdResult:
    retained_indices: np.ndarray
    discarded_indices: np.ndarray
    threshold_value: float
    alpha: float | None
    eta_hat: np.ndarray
    retained_feature_count: int
    retained_feature_fraction: float
    retained_row_mass: np.ndarray
    retained_topic_mass: np.ndarray | None = None
    retained_population_rank: int | None = None
    rank_ok: bool | None = None
    requested_method: str = "none"
    effective_method: str = "none"
    N_used: float | None = None
    unequal_document_lengths: bool = False
    fallback_active: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class WeightResult:
    weights: np.ndarray
    requested_method: str
    effective_method: str
    tau: float
    cap: float | None
    cap_active: bool
    common_scale: str
    common_scale_factor: float
    quantiles: dict[str, float]
    maximum_to_median_ratio: float
    effective_variance: float
    transformed_singular_values: np.ndarray
    transformed_condition_number: float


@dataclass
class PreprocessingResult:
    X_original: np.ndarray
    X_retained: np.ndarray
    X_transformed: np.ndarray
    threshold: ThresholdResult
    weighting: WeightResult


def _validate_X(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise PreprocessingError(f"X must be a matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise PreprocessingError("X contains non-finite entries")
    if np.any(array < 0):
        raise PreprocessingError("X contains negative entries")
    return array


def _document_length_summary(N: float | np.ndarray, n: int) -> tuple[float, bool]:
    values = np.asarray(N, dtype=float)
    if values.ndim == 0:
        summary = float(values)
        unequal = False
    else:
        values = values.reshape(-1)
        if values.size != n:
            raise PreprocessingError(
                f"document-length vector has length {values.size}, expected n={n}"
            )
        summary = float(np.mean(values))
        unequal = not np.allclose(values, values[0])
    if not np.isfinite(summary) or summary <= 0:
        raise PreprocessingError("document length N must be positive and finite")
    return summary, unequal


def select_feature_columns(
    X: np.ndarray,
    N: float | np.ndarray,
    *,
    method: str = "none",
    alpha: float = 0.005,
    K: int | None = None,
    true_A: np.ndarray | None = None,
) -> ThresholdResult:
    """Select whole vocabulary columns without row renormalization.

    ``tran`` maps to the literal paper rule (``tran_paper_exact``).  Use
    ``tran_script_exact`` explicitly to reproduce the R source's strict
    inequality and top-10-percent fallback.
    """

    matrix = _validate_X(X)
    n, p = matrix.shape
    if method not in THRESHOLD_METHODS:
        raise PreprocessingError(
            f"unknown threshold method {method!r}; expected {THRESHOLD_METHODS}"
        )
    N_used, unequal = _document_length_summary(N, n)
    eta_hat = np.mean(matrix, axis=0)
    warnings: list[str] = []
    if unequal:
        warnings.append("threshold_uses_mean_document_length")

    fallback = False
    if method == "none":
        retained = np.arange(p, dtype=int)
        threshold = 0.0
        effective = "none"
        output_alpha: float | None = None
    else:
        if alpha < 0 or not np.isfinite(alpha):
            raise PreprocessingError("alpha must be finite and nonnegative")
        threshold = float(alpha * np.sqrt(np.log(max(p, n)) / (n * N_used)))
        effective = "tran_paper_exact" if method == "tran" else method
        if effective == "tran_script_exact":
            retained = np.flatnonzero(eta_hat > threshold)
            if retained.size < 0.1 * p:
                # R's decreasing sort with its first-index tie convention.
                order = np.argsort(-eta_hat, kind="stable")
                retained = order[: int(np.ceil(0.1 * p))]
                fallback = True
                warnings.append("tran_script_top_10_percent_fallback")
        else:
            retained = np.flatnonzero(eta_hat >= threshold)
        output_alpha = float(alpha)

    retained = np.asarray(retained, dtype=int)
    discarded = np.setdiff1d(np.arange(p, dtype=int), retained, assume_unique=True)
    retained_row_mass = matrix[:, retained].sum(axis=1)
    retained_topic_mass: np.ndarray | None = None
    retained_rank: int | None = None
    rank_ok: bool | None = None
    if true_A is not None:
        topics = np.asarray(true_A, dtype=float)
        if topics.ndim != 2:
            raise PreprocessingError("true_A must be a matrix")
        if topics.shape[1] == p:
            topic_by_feature = topics
        elif topics.shape[0] == p:
            topic_by_feature = topics.T
        else:
            raise PreprocessingError(
                f"neither true_A dimension matches p={p}: {topics.shape}"
            )
        retained_topic_mass = topic_by_feature[:, retained].sum(axis=1)
        retained_rank = int(np.linalg.matrix_rank(topic_by_feature[:, retained]))
        expected_rank = topic_by_feature.shape[0] if K is None else K
        rank_ok = retained_rank >= expected_rank
    elif K is not None:
        rank_ok = None

    if retained.size == 0:
        warnings.append("no_features_retained")
    if K is not None and retained.size < K:
        warnings.append("retained_feature_count_below_K")
        rank_ok = False
    if rank_ok is False:
        warnings.append("retained_population_rank_below_K")

    return ThresholdResult(
        retained_indices=retained,
        discarded_indices=discarded,
        threshold_value=threshold,
        alpha=output_alpha,
        eta_hat=eta_hat,
        retained_feature_count=int(retained.size),
        retained_feature_fraction=float(retained.size / p),
        retained_row_mass=retained_row_mass,
        retained_topic_mass=retained_topic_mass,
        retained_population_rank=retained_rank,
        rank_ok=rank_ok,
        requested_method=method,
        effective_method=effective,
        N_used=N_used,
        unequal_document_lengths=unequal,
        fallback_active=fallback,
        warnings=warnings,
    )


def frequency_weights(
    X_retained: np.ndarray,
    *,
    method: str = "none",
    tau: float = 0.0,
    cap: float | None = None,
    common_scale: str = "none",
    population_M_retained: np.ndarray | None = None,
    true_A_retained: np.ndarray | None = None,
) -> WeightResult:
    matrix = _validate_X(X_retained)
    if method not in WEIGHT_METHODS:
        raise PreprocessingError(
            f"unknown weight method {method!r}; expected {WEIGHT_METHODS}"
        )
    if tau < 0 or not np.isfinite(tau):
        raise PreprocessingError("tau must be finite and nonnegative")
    if cap is not None and (cap <= 0 or not np.isfinite(cap)):
        raise PreprocessingError("weight cap must be positive and finite")
    if common_scale not in {"none", "rms_one"}:
        raise PreprocessingError("common_scale must be 'none' or 'rms_one'")

    eta_hat = np.mean(matrix, axis=0)
    effective_method = method
    if method == "none":
        base = np.ones(matrix.shape[1])
    elif method in {
        "ke_empirical",
        "ke_empirical_additive_floor",
        "ke_empirical_capped",
    }:
        denominators = eta_hat + tau
        if np.any(denominators <= 0):
            raise PreprocessingError(
                "empirical inverse-square-root weights are undefined for zero-frequency columns"
            )
        base = denominators ** -0.5
    elif method == "oracle_eta":
        if population_M_retained is None:
            raise PreprocessingError("oracle_eta is synthetic-only and requires population_M_retained")
        population = _validate_X(population_M_retained)
        if population.shape != matrix.shape:
            raise PreprocessingError("population_M_retained must match X_retained")
        denominators = population.mean(axis=0) + tau
        if np.any(denominators <= 0):
            raise PreprocessingError("oracle eta contains a nonpositive denominator")
        base = denominators ** -0.5
    else:
        if true_A_retained is None:
            raise PreprocessingError("oracle_h is synthetic-only and requires true_A_retained")
        topics = np.asarray(true_A_retained, dtype=float)
        if topics.ndim != 2:
            raise PreprocessingError("true_A_retained must be a matrix")
        if topics.shape[1] == matrix.shape[1]:
            topic_by_feature = topics
        elif topics.shape[0] == matrix.shape[1]:
            topic_by_feature = topics.T
        else:
            raise PreprocessingError("true_A_retained feature dimension does not match X")
        denominators = topic_by_feature.sum(axis=0) + tau
        if np.any(denominators <= 0):
            raise PreprocessingError("oracle h contains a nonpositive denominator")
        base = denominators ** -0.5

    requested_cap = cap
    if method == "ke_empirical_capped" and cap is None:
        raise PreprocessingError("ke_empirical_capped requires cap")
    if cap is not None:
        capped = np.minimum(base, cap)
        cap_active = bool(np.any(capped < base))
        weights = capped
    else:
        weights = base
        cap_active = False

    scale_factor = 1.0
    if common_scale == "rms_one":
        rms = float(np.sqrt(np.mean(weights**2)))
        if rms <= 0:
            raise PreprocessingError("cannot RMS-rescale zero weights")
        scale_factor = 1.0 / rms
        weights = weights * scale_factor

    quantile_values = np.quantile(weights, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    quantiles = {
        key: float(value)
        for key, value in zip(
            ("min", "q25", "median", "q75", "q90", "q99", "max"),
            quantile_values,
        )
    }
    transformed = matrix * weights
    singular_values = np.linalg.svd(transformed, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).eps]
    condition = np.inf if positive.size == 0 else float(positive[0] / positive[-1])
    median = quantiles["median"]
    max_to_median = np.inf if median == 0 else quantiles["max"] / median
    effective_variance = float(np.sum(eta_hat * weights**2))

    return WeightResult(
        weights=weights,
        requested_method=method,
        effective_method=effective_method,
        tau=float(tau),
        cap=requested_cap,
        cap_active=cap_active,
        common_scale=common_scale,
        common_scale_factor=scale_factor,
        quantiles=quantiles,
        maximum_to_median_ratio=float(max_to_median),
        effective_variance=effective_variance,
        transformed_singular_values=singular_values,
        transformed_condition_number=condition,
    )


def preprocess_features(
    X: np.ndarray,
    N: float | np.ndarray,
    *,
    threshold_method: str = "none",
    alpha: float = 0.005,
    weight_method: str = "none",
    tau: float = 0.0,
    weight_cap: float | None = None,
    weight_common_scale: str = "none",
    K: int | None = None,
    true_A: np.ndarray | None = None,
    population_M: np.ndarray | None = None,
    fail_on_rank_loss: bool = True,
) -> PreprocessingResult:
    matrix = _validate_X(X)
    threshold = select_feature_columns(
        matrix, N, method=threshold_method, alpha=alpha, K=K, true_A=true_A
    )
    if threshold.retained_feature_count == 0:
        raise PreprocessingError("thresholding retained no feature columns")
    if fail_on_rank_loss and threshold.rank_ok is False:
        raise PreprocessingError(
            "thresholding destroyed the required rank K; see threshold diagnostics"
        )
    retained = matrix[:, threshold.retained_indices]

    population_retained = None
    if population_M is not None:
        population = _validate_X(population_M)
        if population.shape != matrix.shape:
            raise PreprocessingError("population_M must match X")
        population_retained = population[:, threshold.retained_indices]
    topics_retained = None
    if true_A is not None:
        topics = np.asarray(true_A, dtype=float)
        if topics.shape[1] == matrix.shape[1]:
            topics_retained = topics[:, threshold.retained_indices]
        else:
            topics_retained = topics[threshold.retained_indices, :]

    weighting = frequency_weights(
        retained,
        method=weight_method,
        tau=tau,
        cap=weight_cap,
        common_scale=weight_common_scale,
        population_M_retained=population_retained,
        true_A_retained=topics_retained,
    )
    transformed = retained * weighting.weights
    return PreprocessingResult(
        X_original=matrix,
        X_retained=retained,
        X_transformed=transformed,
        threshold=threshold,
        weighting=weighting,
    )


def weighted_debiased_correction(
    eta_hat_retained: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    N: float,
) -> np.ndarray:
    """Diagonal of ``(n/N) R diag(eta_hat) R``."""

    eta = np.asarray(eta_hat_retained, dtype=float)
    r = np.asarray(weights, dtype=float)
    if eta.shape != r.shape:
        raise PreprocessingError("eta and weight vectors must have the same shape")
    if N <= 0:
        raise PreprocessingError("N must be positive")
    return (float(n) / float(N)) * eta * r**2
