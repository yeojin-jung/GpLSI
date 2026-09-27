"""Numerically explicit W and A recovery on weighted and original scales."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, issparse

from .utils import _euclidean_proj_simplex


class RecoveryError(RuntimeError):
    """Raised when a factor cannot be recovered on the requested scale."""


@dataclass
class WRecoveryResult:
    raw: np.ndarray
    truncated_normalized: np.ndarray
    simplex_projected: np.ndarray
    condition_number: float
    smallest_singular_value: float
    stable: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class ARecoveryResult:
    A_hat: np.ndarray
    method: str
    converged: bool
    iterations: int
    objective_history: list[float]
    gradient_norm: float | None = None
    projected_gradient_norm: float | None = None
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    optimality_gap: float | None = None
    normalized_optimality_gap: float | None = None
    solver: str | None = None
    objective_name: str | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedPoissonCounts:
    """Canonical sparse representation reused by fixed-W Poisson refits."""

    row_indices: np.ndarray
    column_indices: np.ndarray
    values: np.ndarray
    row_totals: np.ndarray
    shape: tuple[int, int]
    total_count: float


def project_simplex(vector: np.ndarray, total: float = 1.0) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise RecoveryError("simplex projection expects a vector")
    if total <= 0:
        raise RecoveryError("simplex total must be positive")
    input_sum = float(values.sum())
    exact_tolerance = 10.0 * np.finfo(float).eps * max(1.0, abs(total))
    if np.all(values >= 0) and abs(input_sum - total) <= exact_tolerance:
        return values * (total / input_sum)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - total
    indices = np.arange(1, len(values) + 1)
    support = np.flatnonzero(ordered - cumulative / indices > 0)
    if support.size == 0:
        return np.full_like(values, total / len(values))
    rho = support[-1]
    theta = cumulative[rho] / (rho + 1.0)
    projected = np.maximum(values - theta, 0.0)
    projected_sum = float(projected.sum())
    if projected_sum <= np.finfo(float).eps:
        return np.full_like(values, total / len(values))
    # Large raw barycentric coordinates can lose a few ulps through
    # cancellation in theta.  Enforce the requested affine constraint after
    # nonnegative projection so saved W rows are numerically on the simplex.
    return projected * (total / projected_sum)


def project_rows_simplex(matrix: np.ndarray) -> np.ndarray:
    """Row-wise :func:`project_simplex` (unit total), vectorized over rows."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        return np.vstack([project_simplex(row) for row in values])
    n, p = values.shape
    input_sum = values.sum(axis=1)
    exact_tolerance = 10.0 * np.finfo(float).eps
    exact = np.all(values >= 0, axis=1) & (np.abs(input_sum - 1.0) <= exact_tolerance)
    ordered = -np.sort(-values, axis=1)
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    positive = ordered - cumulative / np.arange(1, p + 1) > 0
    has_support = positive.any(axis=1)
    rho = p - 1 - np.argmax(positive[:, ::-1], axis=1)
    theta = cumulative[np.arange(n), rho] / (rho + 1.0)
    projected = np.maximum(values - theta[:, None], 0.0)
    projected_sum = projected.sum(axis=1)
    usable = has_support & (projected_sum > np.finfo(float).eps)
    out = np.full_like(values, 1.0 / p)
    out[usable] = projected[usable] * (1.0 / projected_sum[usable])[:, None]
    out[exact] = values[exact] * (1.0 / input_sum[exact])[:, None]
    return out


def recover_W(
    embedding: np.ndarray,
    vertices: np.ndarray,
    *,
    condition_threshold: float = 1e12,
) -> WRecoveryResult:
    U = np.asarray(embedding, dtype=float)
    H = np.asarray(vertices, dtype=float)
    if U.ndim != 2 or H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise RecoveryError(
            f"W recovery requires a square vertex matrix; got U={U.shape}, H={H.shape}"
        )
    if U.shape[1] != H.shape[1]:
        raise RecoveryError("embedding and vertex dimensions do not match")
    singular_values = np.linalg.svd(H, compute_uv=False)
    smallest = float(singular_values[-1])
    condition = np.inf if smallest == 0 else float(singular_values[0] / smallest)
    warnings: list[str] = []
    stable = bool(np.isfinite(condition) and condition <= condition_threshold)
    if not stable:
        warnings.append("ill_conditioned_vertex_matrix")
    try:
        raw = np.linalg.solve(H.T, U.T).T
    except np.linalg.LinAlgError as error:
        raise RecoveryError(f"singular vertex matrix: {error}") from error

    truncated = np.maximum(raw, 0.0)
    row_sums = truncated.sum(axis=1, keepdims=True)
    zero_rows = np.flatnonzero(row_sums[:, 0] <= np.finfo(float).eps)
    if zero_rows.size:
        warnings.append("zero_rows_after_nonnegative_truncation")
        truncated[zero_rows] = 1.0 / H.shape[0]
        row_sums = truncated.sum(axis=1, keepdims=True)
    truncated_normalized = truncated / row_sums
    simplex = project_rows_simplex(raw)
    return WRecoveryResult(
        raw=raw,
        truncated_normalized=truncated_normalized,
        simplex_projected=simplex,
        condition_number=condition,
        smallest_singular_value=smallest,
        stable=stable,
        warnings=warnings,
    )


def spectral_A_unweighted(
    vertices: np.ndarray,
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    weights: np.ndarray,
    retained_indices: np.ndarray,
    full_feature_count: int,
) -> ARecoveryResult:
    H = np.asarray(vertices, dtype=float)
    L = np.asarray(singular_values, dtype=float)
    V = np.asarray(right_vectors, dtype=float)
    r = np.asarray(weights, dtype=float)
    retained = np.asarray(retained_indices, dtype=int)
    if L.ndim == 1:
        L = np.diag(L)
    if V.shape[0] != retained.size or r.size != retained.size:
        raise RecoveryError("weighted SVD factors do not match retained features")
    if np.any(r <= 0):
        raise RecoveryError("weights must be positive before topic unweighting")

    weighted_topics = H @ L @ V.T
    # Undo R before either truncation or normalization.
    retained_topics = weighted_topics / r[None, :]
    retained_topics = np.maximum(retained_topics, 0.0)
    row_sums = retained_topics.sum(axis=1, keepdims=True)
    warnings: list[str] = []
    if np.any(row_sums <= np.finfo(float).eps):
        raise RecoveryError("spectral topic recovery produced an all-zero topic")
    retained_topics /= row_sums
    full = np.zeros((H.shape[0], full_feature_count))
    full[:, retained] = retained_topics
    return ARecoveryResult(
        A_hat=full,
        method="A_spectral_unweighted",
        converged=True,
        iterations=0,
        objective_history=[],
        warnings=warnings,
    )


def _initialize_A(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    unconstrained, _, _, _ = np.linalg.lstsq(W, X, rcond=None)
    return project_rows_simplex(unconstrained)


def refit_A_current(W: np.ndarray, X: np.ndarray) -> ARecoveryResult:
    """Wrap the historical ``GpLSI.get_A_hat`` calculation without changes.

    This deliberately uses the same normal-equation inverse and the historical
    simplex projection helper.  It is kept separate from the numerically more
    defensive full-L2 refit so regression comparisons remain meaningful.
    """

    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    if W.ndim != 2 or X.ndim != 2 or W.shape[0] != X.shape[0]:
        raise RecoveryError("W and X dimensions are incompatible")
    try:
        projector = np.linalg.inv(W.T @ W) @ W.T
    except np.linalg.LinAlgError as error:
        raise RecoveryError(f"historical current A recovery is singular: {error}") from error
    unconstrained = projector @ X
    if not np.isfinite(unconstrained).all():
        raise RecoveryError("historical current A recovery produced non-finite coordinates")
    try:
        A_hat = np.asarray([_euclidean_proj_simplex(row) for row in unconstrained])
    except (IndexError, ValueError, FloatingPointError) as error:
        raise RecoveryError(
            f"historical current A simplex projection failed: {error}"
        ) from error
    residual = W @ A_hat - X
    return ARecoveryResult(
        A_hat=A_hat,
        method="current",
        converged=True,
        iterations=0,
        objective_history=[float(np.sum(residual**2))],
        gradient_norm=None,
        projected_gradient_norm=None,
        status="ok",
    )


def refit_A_full_l2(
    W: np.ndarray,
    X: np.ndarray,
    *,
    initial_A: np.ndarray | None = None,
    max_iter: int = 2_000,
    tolerance: float = 1e-8,
) -> ARecoveryResult:
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    if W.ndim != 2 or X.ndim != 2 or W.shape[0] != X.shape[0]:
        raise RecoveryError("W and X dimensions are incompatible")
    A = _initialize_A(W, X) if initial_A is None else project_rows_simplex(initial_A)
    if A.shape != (W.shape[1], X.shape[1]):
        raise RecoveryError("initial_A has the wrong shape")
    lipschitz = 2.0 * float(np.linalg.norm(W, 2) ** 2)
    if lipschitz <= np.finfo(float).eps:
        raise RecoveryError("W has zero spectral norm")
    step = 1.0 / lipschitz
    # Sufficient statistics: the gradient and objective depend on X only through
    # W^T W, W^T X, and ||X||^2, so each iteration costs O(K^2 p) instead of O(nKp).
    gram = W.T @ W
    cross = W.T @ X
    x_norm = float(np.sum(X**2))

    def objective(value: np.ndarray) -> float:
        return float(x_norm - 2.0 * np.sum(value * cross) + np.sum(value * (gram @ value)))

    history = [objective(A)]
    converged = False
    gradient_norm = np.nan
    for iteration in range(1, max_iter + 1):
        gradient = 2.0 * (gram @ A - cross)
        gradient_norm = float(np.linalg.norm(gradient))
        candidate = project_rows_simplex(A - step * gradient)
        candidate_objective = objective(candidate)
        local_step = step
        while candidate_objective > history[-1] + 1e-12 and local_step > 1e-16:
            local_step *= 0.5
            candidate = project_rows_simplex(A - local_step * gradient)
            candidate_objective = objective(candidate)
        change = np.linalg.norm(candidate - A) / max(np.linalg.norm(A), 1.0)
        A = candidate
        history.append(candidate_objective)
        if change <= tolerance:
            converged = True
            break
    return ARecoveryResult(
        A_hat=A,
        method="A_full_L2",
        converged=converged,
        iterations=iteration,
        objective_history=history,
        gradient_norm=gradient_norm,
        projected_gradient_norm=float(
            np.linalg.norm(A - project_rows_simplex(A - gradient))
        ),
        status="ok" if converged else "max_iter_reached",
    )


def poisson_objective_and_gradient(
    A: np.ndarray,
    W: np.ndarray,
    counts: np.ndarray,
    document_lengths: np.ndarray,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    mean = W @ A
    stabilized = mean + epsilon
    objective = float(
        np.sum(document_lengths[:, None] * mean - counts * np.log(stabilized))
    )
    gradient = W.T @ (
        document_lengths[:, None] - counts / stabilized
    )
    return objective, gradient


def prepare_poisson_counts(counts: np.ndarray) -> PreparedPoissonCounts:
    """Validate counts and store only positive entries in canonical order."""

    if issparse(counts):
        sparse = csr_matrix(counts, dtype=float, copy=True)
        sparse.sum_duplicates()
        sparse.eliminate_zeros()
        if sparse.ndim != 2:
            raise RecoveryError("counts must be a matrix")
        if not np.isfinite(sparse.data).all() or np.any(sparse.data < 0):
            raise RecoveryError("counts must be finite and nonnegative")
    else:
        dense = np.asarray(counts)
        if dense.ndim != 2:
            raise RecoveryError("counts must be a matrix")
        if not np.isfinite(dense).all() or np.any(dense < 0):
            raise RecoveryError("counts must be finite and nonnegative")
        sparse = csr_matrix(dense, dtype=float)
        sparse.eliminate_zeros()

    row_counts = np.diff(sparse.indptr)
    rows = np.repeat(np.arange(sparse.shape[0], dtype=np.int64), row_counts)
    columns = sparse.indices.astype(np.int64, copy=True)
    values = sparse.data.astype(float, copy=True)
    row_totals = np.asarray(sparse.sum(axis=1)).reshape(-1)
    total = float(values.sum())
    if total <= 0:
        raise RecoveryError("counts must contain at least one positive observation")
    return PreparedPoissonCounts(
        row_indices=rows,
        column_indices=columns,
        values=values,
        row_totals=row_totals,
        shape=sparse.shape,
        total_count=total,
    )


def _poisson_em_statistics(
    A: np.ndarray,
    W: np.ndarray,
    prepared: PreparedPoissonCounts,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, float, float]:
    """Return EM scores, log likelihood, and the smallest observed probability."""

    K, feature_count = A.shape
    scores = np.zeros((K, feature_count), dtype=float)
    log_likelihood = 0.0
    minimum_probability = np.inf
    nnz = prepared.values.size
    for start in range(0, nnz, chunk_size):
        stop = min(start + chunk_size, nnz)
        rows = prepared.row_indices[start:stop]
        columns = prepared.column_indices[start:stop]
        values = prepared.values[start:stop]
        local_W = W[rows]
        probabilities = np.einsum(
            "ek,ek->e", local_W, A[:, columns].T, optimize=True
        )
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise RecoveryError(
                "the fitted model assigns nonpositive probability to a positive count"
            )
        minimum_probability = min(minimum_probability, float(probabilities.min()))
        log_likelihood += float(np.dot(values, np.log(probabilities)))
        ratio = values / probabilities
        for topic in range(K):
            scores[topic] += np.bincount(
                columns,
                weights=ratio * local_W[:, topic],
                minlength=feature_count,
            )
    if not np.isfinite(scores).all() or not np.isfinite(log_likelihood):
        raise RecoveryError("Poisson EM statistics became non-finite")
    return scores, log_likelihood, minimum_probability


def _poisson_optimality_gap(
    A: np.ndarray,
    scores: np.ndarray,
    *,
    prior_strength: float,
    prior_base: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    gradient_scores = scores
    if prior_strength > 0:
        if prior_base is None or np.any(A <= 0):
            raise RecoveryError("MAP optimality requires a positive A and prior base")
        gradient_scores = scores + prior_strength * prior_base[None, :] / A
    row_average = np.sum(A * gradient_scores, axis=1)
    row_best = np.max(gradient_scores, axis=1)
    gap = max(0.0, float(np.sum(row_best - row_average)))
    return gap, gradient_scores


def refit_A_full_poisson(
    W: np.ndarray,
    counts: np.ndarray | PreparedPoissonCounts,
    document_lengths: float | np.ndarray,
    *,
    initial_A: np.ndarray | None = None,
    epsilon: float = 1e-12,
    max_iter: int = 2_000,
    tolerance: float = 1e-8,
    initial_step: float = 1e-2,
    interior_mass: float = 1e-6,
    chunk_size: int = 250_000,
    prior_strength: float = 0.0,
    prior_base: np.ndarray | None = None,
) -> ARecoveryResult:
    """Refit topic profiles with fixed ``W`` by sparse, monotone EM.

    With row-simplex ``W`` and ``A``, the linear term in the full Poisson
    objective is constant over feasible ``A``.  The EM update therefore solves
    the equivalent fixed-W multinomial likelihood.  Convergence is certified
    by the Frank--Wolfe/KKT gap, not by a small parameter change.

    ``prior_strength=0`` is the exact MLE.  A positive value requests a
    separately labelled MAP sensitivity with a full-support base distribution.
    ``epsilon`` and ``initial_step`` remain in the signature only for backwards
    compatibility; neither is used to smooth or step the exact EM objective.
    """

    W = np.asarray(W, dtype=float)
    prepared = counts if isinstance(counts, PreparedPoissonCounts) else prepare_poisson_counts(counts)
    if W.ndim != 2 or W.shape[0] != prepared.shape[0]:
        raise RecoveryError("W and count dimensions are incompatible")
    if not np.isfinite(W).all() or np.any(W < 0):
        raise RecoveryError("W must be finite and nonnegative")
    W_row_sums = W.sum(axis=1)
    if np.any(W_row_sums <= 0):
        raise RecoveryError("W rows must have positive mass")
    topic_exposure = W.sum(axis=0)
    active_topics = topic_exposure > np.finfo(float).eps
    inactive_topic_indices = np.flatnonzero(~active_topics)
    lengths = np.asarray(document_lengths, dtype=float)
    if lengths.ndim == 0:
        lengths = np.full(W.shape[0], float(lengths))
    lengths = lengths.reshape(-1)
    if lengths.size != W.shape[0] or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise RecoveryError("document lengths must be positive with one value per row")
    if epsilon <= 0 or max_iter < 0 or tolerance <= 0 or initial_step <= 0:
        raise RecoveryError("optimizer controls must be positive")
    if not 0 < interior_mass < 1 or chunk_size <= 0 or prior_strength < 0:
        raise RecoveryError("invalid EM interior, chunk, or prior control")

    feature_count = prepared.shape[1]
    if initial_A is None:
        feature_totals = np.bincount(
            prepared.column_indices,
            weights=prepared.values,
            minlength=feature_count,
        )
        A = np.tile(feature_totals / prepared.total_count, (W.shape[1], 1))
        initializer_source = "pooled_feature_frequencies"
    else:
        supplied = np.asarray(initial_A, dtype=float)
        if supplied.ndim != 2 or not np.isfinite(supplied).all():
            raise RecoveryError("initial_A must be a finite matrix")
        A = project_rows_simplex(supplied)
        initializer_source = "supplied_initial_A"
    if A.shape != (W.shape[1], feature_count):
        raise RecoveryError("initial_A has the wrong shape")

    original_initial_A = A.copy()

    # Multiplicative EM cannot reopen an exact zero.  This one-time
    # interiorization is an optimization device, not a prior or output floor.
    A = (1.0 - interior_mass) * A + interior_mass / feature_count
    A /= A.sum(axis=1, keepdims=True)

    base = None
    if prior_strength > 0:
        if prior_base is None:
            base = np.full(feature_count, 1.0 / feature_count)
        else:
            base = np.asarray(prior_base, dtype=float).reshape(-1)
            if (
                base.size != feature_count
                or not np.isfinite(base).all()
                or np.any(base <= 0)
                or float(base.sum()) <= 0
            ):
                raise RecoveryError("prior_base must be a positive feature distribution")
            base = base / base.sum()

    linear_term = float(np.dot(lengths, W_row_sums))

    def evaluate(value_A: np.ndarray):
        scores, log_likelihood, minimum_probability = _poisson_em_statistics(
            value_A, W, prepared, chunk_size=chunk_size
        )
        objective = linear_term - log_likelihood
        if prior_strength > 0:
            objective -= float(prior_strength * np.sum(base[None, :] * np.log(value_A)))
        gap, gradient_scores = _poisson_optimality_gap(
            value_A,
            scores,
            prior_strength=prior_strength,
            prior_base=base,
        )
        normalizer = prepared.total_count + W.shape[1] * prior_strength
        return objective, scores, gradient_scores, gap, gap / normalizer, minimum_probability

    try:
        (
            original_initial_objective,
            original_initial_scores,
            original_initial_gradient_scores,
            original_initial_gap,
            original_initial_normalized_gap,
            original_initial_minimum_probability,
        ) = evaluate(original_initial_A)
    except RecoveryError:
        original_initial_objective = float("inf")
        original_initial_scores = None
        original_initial_gradient_scores = None
        original_initial_gap = float("inf")
        original_initial_normalized_gap = float("inf")
        original_initial_minimum_probability = 0.0

    value, scores, gradient_scores, gap, normalized_gap, minimum_probability = evaluate(A)
    interiorized_initial_objective = value
    history = [value]
    converged = normalized_gap <= tolerance
    status = "ok" if converged else "max_iter_reached"
    warnings: list[str] = []
    if inactive_topic_indices.size:
        warnings.append("inactive_W_topics_preserved")
    attempted_iterations = 0
    accepted_iterations = 0
    maximum_increase = 0.0
    for attempt in range(1, max_iter + 1):
        if converged:
            break
        attempted_iterations = attempt
        numerator = A * scores
        if prior_strength > 0:
            numerator = numerator + prior_strength * base[None, :]
        row_mass = numerator.sum(axis=1, keepdims=True)
        update_topics = np.ones(W.shape[1], dtype=bool) if prior_strength > 0 else active_topics
        if (
            np.any(row_mass[update_topics] <= 0)
            or not np.isfinite(row_mass[update_topics]).all()
        ):
            status = "degenerate_em_update"
            warnings.append(status)
            break
        candidate = A.copy()
        candidate[update_topics] = (
            numerator[update_topics] / row_mass[update_topics]
        )
        try:
            (
                candidate_value,
                candidate_scores,
                candidate_gradient_scores,
                candidate_gap,
                candidate_normalized_gap,
                candidate_minimum_probability,
            ) = evaluate(candidate)
        except RecoveryError as error:
            status = "numerical_failure"
            warnings.append(f"numerical_failure: {error}")
            break
        increase = candidate_value - value
        maximum_increase = max(maximum_increase, increase)
        monotonic_slack = 128.0 * np.finfo(float).eps * max(1.0, abs(value))
        if not np.isfinite(candidate_value) or increase > monotonic_slack:
            status = "objective_increase"
            warnings.append(status)
            break
        A = candidate
        value = candidate_value
        scores = candidate_scores
        gradient_scores = candidate_gradient_scores
        gap = candidate_gap
        normalized_gap = candidate_normalized_gap
        minimum_probability = candidate_minimum_probability
        history.append(value)
        accepted_iterations += 1
        if normalized_gap <= tolerance:
            converged = True
            status = "ok"
            break

    if not converged and status == "max_iter_reached":
        warnings.append("maximum_iterations_reached")
    comparison_slack = 128.0 * np.finfo(float).eps * max(
        1.0, abs(original_initial_objective)
    )
    if (
        np.isfinite(original_initial_objective)
        and value > original_initial_objective + comparison_slack
    ):
        A = original_initial_A
        value = original_initial_objective
        scores = original_initial_scores
        gradient_scores = original_initial_gradient_scores
        gap = original_initial_gap
        normalized_gap = original_initial_normalized_gap
        minimum_probability = original_initial_minimum_probability
        converged = normalized_gap <= tolerance
        status = "ok" if converged else "initial_A_retained"
        warnings.append("initial_A_had_lower_objective_than_em_iterate")
        if not history or history[-1] != value:
            history.append(value)
    identified_subproblem_converged = converged
    if inactive_topic_indices.size and prior_strength == 0:
        converged = False
        status = "unidentified_inactive_topics"
    raw_gradient = W.T @ lengths[:, None] - gradient_scores
    normalized_gradient = raw_gradient / (
        prepared.total_count + W.shape[1] * prior_strength
    )
    method = "A_full_Pois" if prior_strength == 0 else "A_full_Pois_MAP"
    return ARecoveryResult(
        A_hat=A,
        method=method,
        converged=converged,
        iterations=accepted_iterations,
        objective_history=history,
        gradient_norm=float(np.linalg.norm(raw_gradient)),
        projected_gradient_norm=float(
            np.linalg.norm(A - project_rows_simplex(A - normalized_gradient))
        ),
        optimality_gap=gap,
        normalized_optimality_gap=normalized_gap,
        solver="sparse_monotone_em",
        objective_name=(
            "full_poisson_nll_omitting_count_only_constants"
            if prior_strength == 0
            else "full_poisson_negative_log_posterior_omitting_count_only_constants"
        ),
        diagnostics={
            "training_count_total": prepared.total_count,
            "training_nonzero_entries": int(prepared.values.size),
            "poisson_linear_term": linear_term,
            "training_log_likelihood_without_constant": linear_term - value
            if prior_strength == 0
            else None,
            "minimum_probability_at_positive_count": minimum_probability,
            "initializer_source": initializer_source,
            "original_initial_objective": original_initial_objective,
            "interiorized_initial_objective": interiorized_initial_objective,
            "final_objective_no_worse_than_original_initial": bool(
                not np.isfinite(original_initial_objective)
                or value <= original_initial_objective + comparison_slack
            ),
            "interior_mass": interior_mass,
            "chunk_size": int(chunk_size),
            "optimizer_version": "sparse_monotone_em_v1",
            "attempted_iterations": int(attempted_iterations),
            "accepted_iterations": int(accepted_iterations),
            "inactive_topic_indices": inactive_topic_indices.tolist(),
            "identified_subproblem_converged": bool(identified_subproblem_converged),
            "prior_strength": float(prior_strength),
            "prior_base": None if base is None else base.tolist(),
            "maximum_objective_increase": maximum_increase,
            "objective_monotone_within_roundoff": bool(
                all(
                    later - earlier
                    <= 128.0 * np.finfo(float).eps * max(1.0, abs(earlier))
                    for earlier, later in zip(history, history[1:])
                )
            ),
            "legacy_epsilon_used": False,
            "legacy_initial_step_used": False,
            "legacy_epsilon_requested": float(epsilon),
            "legacy_initial_step_requested": float(initial_step),
            "projected_gradient_scaling": "raw_gradient_divided_by_total_count_plus_prior_mass",
            "optimality_gap_normalization": "total_count_plus_K_times_prior_strength",
            "final_objective": value,
            "maximum_W_row_sum_deviation_from_one": float(
                np.max(np.abs(W_row_sums - 1.0))
            ),
            "maximum_document_length_count_mismatch": float(
                np.max(np.abs(lengths - prepared.row_totals))
            ),
        },
        status=status,
        warnings=warnings,
    )


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float],
    value: np.ndarray,
    *,
    step: float = 1e-6,
) -> np.ndarray:
    estimate = np.empty_like(value, dtype=float)
    for index in np.ndindex(value.shape):
        plus = value.copy()
        minus = value.copy()
        plus[index] += step
        minus[index] -= step
        estimate[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return estimate
