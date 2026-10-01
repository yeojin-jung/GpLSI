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
    """Row-wise :func:`project_simplex` (unit total), vectorized over row blocks.

    Bitwise identical to projecting each row separately; blocks bound the
    temporary sort/cumsum arrays to about one million entries.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        return np.vstack([project_simplex(row) for row in values])
    n, p = values.shape
    output = np.empty_like(values)
    step = min(8192, max(1, 1_000_000 // p))
    exact_tolerance = 10.0 * np.finfo(float).eps
    divisors = np.arange(1, p + 1)
    for start in range(0, n, step):
        block = values[start:start + step]
        rows = block.shape[0]
        input_sum = block.sum(axis=1)
        exact = np.all(block >= 0, axis=1) & (np.abs(input_sum - 1.0) <= exact_tolerance)
        ordered = -np.sort(-block, axis=1)
        cumulative = np.cumsum(ordered, axis=1) - 1.0
        positive = ordered - cumulative / divisors > 0
        has_support = positive.any(axis=1)
        rho = p - 1 - np.argmax(positive[:, ::-1], axis=1)
        theta = cumulative[np.arange(rows), rho] / (rho + 1.0)
        projected = np.maximum(block - theta[:, None], 0.0)
        projected_sum = projected.sum(axis=1)
        usable = has_support & (projected_sum > np.finfo(float).eps)
        out = np.full_like(block, 1.0 / p)
        out[usable] = projected[usable] * (1.0 / projected_sum[usable])[:, None]
        out[exact] = block[exact] * (1.0 / input_sum[exact])[:, None]
        output[start:start + step] = out
    return output


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
    lengths = np.asarray(document_lengths, dtype=float).reshape(-1)
    if counts.size <= 10_000_000:
        mean = W @ A
        stabilized = mean + epsilon
        objective = float(
            np.sum(lengths[:, None] * mean - counts * np.log(stabilized))
        )
        gradient = W.T @ (lengths[:, None] - counts / stabilized)
        return objective, gradient

    objective = 0.0
    gradient = np.zeros_like(A, dtype=float)
    for start in range(0, W.shape[0], 512):
        stop = min(start + 512, W.shape[0])
        mean = W[start:stop] @ A
        stabilized = mean + epsilon
        local_counts = counts[start:stop]
        local_lengths = lengths[start:stop]
        objective += float(
            np.sum(
                local_lengths[:, None] * mean
                - local_counts * np.log(stabilized)
            )
        )
        gradient += W[start:stop].T @ (
            local_lengths[:, None] - local_counts / stabilized
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


POISSON_ROW_TILED_MAX_FEATURES = 512
POISSON_ROW_TILE_SIZE = 256
POISSON_PREDICTION_ENTRY_CAP = 1_000_000


def _poisson_prepared_csr(prepared: PreparedPoissonCounts) -> csr_matrix:
    """One sparse row-indexed view per optimized solve; inputs are not mutated."""
    return csr_matrix(
        (prepared.values, (prepared.row_indices, prepared.column_indices)),
        shape=prepared.shape,
    )


def _poisson_em_statistics_row_tiled(
    A: np.ndarray,
    W: np.ndarray,
    prepared: PreparedPoissonCounts,
    *,
    counts_csr: csr_matrix | None = None,
    row_chunk_size: int = POISSON_ROW_TILE_SIZE,
    prediction_entry_cap: int = POISSON_PREDICTION_ENTRY_CAP,
) -> tuple[np.ndarray, float, float]:
    """Same exact-support statistics using bounded GEMM and sparse products.

    Targeted-panel path (<=512 features), motivated by the measured full-size
    300-feature probe. Dense tiles have <=256 rows and <=1,000,000 entries;
    neither a full n-by-p prediction nor a full-nnz ratio buffer is formed.
    Input A, W, counts and PreparedPoissonCounts arrays remain unchanged.
    """
    if A.shape[1] > POISSON_ROW_TILED_MAX_FEATURES:
        raise RecoveryError("row-tiled Poisson statistics are restricted to <=512 features")
    if not 1 <= row_chunk_size <= POISSON_ROW_TILE_SIZE:
        raise RecoveryError("invalid Poisson row-tile size")
    if not 1 <= prediction_entry_cap <= POISSON_PREDICTION_ENTRY_CAP:
        raise RecoveryError("invalid Poisson prediction-entry cap")
    counts = _poisson_prepared_csr(prepared) if counts_csr is None else counts_csr
    scores = np.zeros(A.shape, dtype=float)
    log_likelihood = 0.0
    minimum_probability = np.inf
    row_step = min(row_chunk_size, prediction_entry_cap)
    for start in range(0, prepared.shape[0], row_step):
        stop = min(start + row_step, prepared.shape[0])
        local_W = W[start:stop]
        local_counts = counts[start:stop]
        columns_per_tile = max(1, prediction_entry_cap // (stop - start))
        for first in range(0, A.shape[1], columns_per_tile):
            last = min(A.shape[1], first + columns_per_tile)
            local = local_counts if first == 0 and last == A.shape[1] else local_counts[:, first:last]
            if local.nnz == 0:
                continue
            probabilities = local_W @ A[:, first:last]
            if probabilities.shape[0] > POISSON_ROW_TILE_SIZE or probabilities.size > prediction_entry_cap:
                raise RecoveryError("Poisson prediction tile exceeded its allocation bound")
            rows = np.repeat(np.arange(stop - start), np.diff(local.indptr))
            observed = probabilities[rows, local.indices]
            del probabilities
            if not np.isfinite(observed).all() or np.any(observed <= 0):
                raise RecoveryError(
                    "the fitted model assigns nonpositive probability to a positive count"
                )
            values = local.data.astype(float, copy=False)
            ratios = csr_matrix((values / observed, local.indices, local.indptr), shape=local.shape)
            scores[:, first:last] += np.asarray(ratios.T @ local_W).T
            log_likelihood += float(np.dot(values, np.log(observed)))
            minimum_probability = min(minimum_probability, float(observed.min()))
    if not np.isfinite(scores).all() or not np.isfinite(log_likelihood):
        raise RecoveryError("Poisson EM statistics became non-finite")
    return scores, log_likelihood, minimum_probability


@dataclass
class PreparedCSRPoissonCounts:
    counts: object
    row_totals: np.ndarray
    feature_totals: np.ndarray
    total_count: float
    shape: tuple
    values: np.ndarray


def _is_disk_csr(value) -> bool:
    """True for the memory-mapped row store of the optional acceleration package.

    That package (``gplsi_joint_v2.acceleration``) lives on a separate branch;
    without it every count matrix is an ordinary in-memory CSR.
    """

    try:
        from gplsi_joint_v2.acceleration.storage import DiskCSR
    except ImportError:
        return False
    return isinstance(value, DiskCSR)


def _csr_poisson_blocks(source, row_limit=256, nnz_limit=250000):
    start=0
    while start<source.shape[0]:
        stop=min(start+row_limit,source.shape[0]);limit=int(source.indptr[start])+nnz_limit
        if source.indptr[stop]>limit:stop=max(start,int(np.searchsorted(source.indptr,limit,side="right")-1))
        if stop<=start:raise RecoveryError("single observation exceeds sparse recovery block budget")
        yield start,stop,(source.rows(start,stop) if _is_disk_csr(source) else source[start:stop])
        start=stop


def _prepare_csr_poisson(counts):
    """One CSR layout or mapped row store; no full-nnz row/column copies."""
    from scipy import sparse
    if _is_disk_csr(counts):
        source = counts
    else:
        source = sparse.csr_matrix(counts, copy=True)
        source.sum_duplicates(); source.eliminate_zeros(); source.sort_indices()
    n, p = source.shape
    totals = np.empty(n, dtype=float); features = np.zeros(p, dtype=float)
    for start, stop, block in _csr_poisson_blocks(source):
        if not np.isfinite(block.data).all() or np.any(block.data < 0):
            raise RecoveryError("counts must be finite nonnegative")
        totals[start:stop] = np.asarray(block.sum(axis=1)).ravel()
        features += np.asarray(block.sum(axis=0)).ravel()
    total = float(features.sum())
    if total <= 0: raise RecoveryError("counts must contain at least one positive observation")
    return PreparedCSRPoissonCounts(source, totals, features, total, source.shape, source.data)


def _poisson_em_statistics_csr(A, W, prepared, *, chunk_size):
    """Sparse nonzero log/gradient terms, bounded row tiles, full linear term in caller."""
    source = prepared.counts
    scores = np.zeros(A.shape, dtype=float); log_likelihood = 0.; minimum_probability = np.inf
    # A canonical row has at most p entries. Entry chunks below p still
    # split its likelihood contributions; they are not a whole-row limit.
    for start, stop, block in _csr_poisson_blocks(source, nnz_limit=max(chunk_size,source.shape[1])):
        if block.nnz == 0: continue
        local_W = W[start:stop]
        predictions = local_W @ A if A.shape[1] <= POISSON_ROW_TILED_MAX_FEATURES else None
        for lo in range(0, block.nnz, chunk_size):
            hi = min(lo+chunk_size, block.nnz)
            rows = np.searchsorted(block.indptr, np.arange(lo, hi), side="right") - 1
            columns = block.indices[lo:hi]; values = block.data[lo:hi].astype(float, copy=False)
            probability = (predictions[rows, columns] if predictions is not None else
                           np.einsum("ik,ik->i", local_W[rows], A[:, columns].T))
            if not np.isfinite(probability).all() or np.any(probability <= 0):
                raise RecoveryError("the fitted model assigns nonpositive probability to a positive count")
            minimum_probability = min(minimum_probability, float(probability.min()))
            log_likelihood += float(np.dot(values, np.log(probability)))
            ratio = csr_matrix((values/probability, (rows, columns)), shape=block.shape)
            scores += np.asarray(ratio.T @ local_W).T
    if not np.isfinite(scores).all() or not np.isfinite(log_likelihood):
        raise RecoveryError("Poisson EM statistics became non-finite")
    return scores, log_likelihood, minimum_probability


def _select_poisson_statistics_backend(requested: str, feature_count: int) -> str:
    if requested not in {"auto", "sparse_entry", "row_tiled", "csr_streamed"}:
        raise RecoveryError("unknown Poisson statistics backend")
    if requested == "auto":
        return "row_tiled" if feature_count <= POISSON_ROW_TILED_MAX_FEATURES else "sparse_entry"
    if requested == "row_tiled" and feature_count > POISSON_ROW_TILED_MAX_FEATURES:
        raise RecoveryError("row-tiled Poisson statistics are restricted to <=512 features")
    return requested


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
    statistics_backend: str = "auto",
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
    ``statistics_backend`` changes only the floating-point evaluation kernel:
    auto uses bounded row tiles for <=512 features, original sparse-entry
    statistics otherwise. Initializer, updates and stopping criteria are shared.
    """

    W = np.asarray(W, dtype=float)
    if statistics_backend == "csr_streamed":
        prepared = counts if isinstance(counts, PreparedCSRPoissonCounts) else _prepare_csr_poisson(counts)
    else:
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
    selected_statistics_backend = _select_poisson_statistics_backend(statistics_backend, feature_count)
    statistics_counts_csr = (
        _poisson_prepared_csr(prepared) if selected_statistics_backend == "row_tiled" else None
    )
    if initial_A is None:
        feature_totals = (prepared.feature_totals if isinstance(prepared, PreparedCSRPoissonCounts) else np.bincount(
            prepared.column_indices, weights=prepared.values, minlength=feature_count))
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
        if selected_statistics_backend == "csr_streamed":
            scores, log_likelihood, minimum_probability = _poisson_em_statistics_csr(
                value_A, W, prepared, chunk_size=chunk_size)
        elif selected_statistics_backend == "row_tiled":
            scores, log_likelihood, minimum_probability = _poisson_em_statistics_row_tiled(
                value_A, W, prepared, counts_csr=statistics_counts_csr
            )
        else:
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
            "statistics_backend_requested": statistics_backend,
            "statistics_backend": selected_statistics_backend,
            "statistics_kernel_revision": "targeted_row_tiled_v1",
            "statistics_auto_panel_threshold": POISSON_ROW_TILED_MAX_FEATURES,
            "statistics_row_tile_size": POISSON_ROW_TILE_SIZE if selected_statistics_backend == "row_tiled" else None,
            "statistics_prediction_entry_cap": POISSON_PREDICTION_ENTRY_CAP if selected_statistics_backend == "row_tiled" else None,
            "statistics_full_dense_prediction_allocated": False,
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


def _poisson_squarem_em_map(
    A: np.ndarray, scores: np.ndarray, active_topics: np.ndarray
) -> np.ndarray:
    """One unchanged MLE EM update; preserve inactive topic rows."""
    numerator = A * scores
    row_mass = numerator.sum(axis=1, keepdims=True)
    if (
        np.any(row_mass[active_topics] <= 0)
        or not np.isfinite(row_mass[active_topics]).all()
    ):
        raise RecoveryError("degenerate_em_update")
    candidate = A.copy()
    candidate[active_topics] = numerator[active_topics] / row_mass[active_topics]
    return candidate


def _poisson_squarem_step_length(
    A0: np.ndarray, A1: np.ndarray, A2: np.ndarray, step_max: float
) -> float:
    """Bounded SqS3 step; zero/underflowed curvature safely gives plain EM."""
    r = A1 - A0
    v = (A2 - A1) - r
    # Scaling avoids squaring subnormal displacements. It affects only the
    # proposal, never the objective, gradient, EM map or KKT stopping rule.
    r_scale = float(np.max(np.abs(r)))
    v_scale = float(np.max(np.abs(v)))
    if not np.isfinite(r_scale) or not np.isfinite(v_scale) or r_scale == 0 or v_scale == 0:
        return 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r_norm_scaled = float(np.linalg.norm(r / r_scale))
        v_norm_scaled = float(np.linalg.norm(v / v_scale))
        alpha = (r_scale / v_scale) * (r_norm_scaled / v_norm_scaled)
    if not np.isfinite(alpha):
        return 1.0
    return float(min(step_max, max(1.0, alpha)))


def _poisson_squarem_extrapolate(
    A0: np.ndarray, A1: np.ndarray, A2: np.ndarray, alpha: float
) -> np.ndarray:
    """Raw proposal only; the caller independently checks every constraint."""
    if alpha == 1.0:
        return A2.copy()
    r = A1 - A0
    v = (A2 - A1) - r
    with np.errstate(over="ignore", invalid="ignore"):
        # Algebraically A0 + 2*alpha*r + alpha**2*v. This form retreats to
        # the already evaluated A2 exactly at alpha=1.
        return A2 + (alpha - 1.0) * (2.0 * r + (alpha + 1.0) * v)


def _poisson_squarem_validate_candidate(
    candidate: np.ndarray, ordinary_A2: np.ndarray, active_topics: np.ndarray,
    *, normalize: bool = True,
) -> np.ndarray:
    """Reject unsafe extrapolation; never clip/project/floor its entries.

    Positive row scaling only removes affine roundoff. Additional absorbing
    zeros relative to the validated ordinary candidate are forbidden before
    and after scaling. Inactive rows remain byte-identical to that candidate.
    """
    matrix = np.asarray(candidate, dtype=float)
    if matrix.shape != ordinary_A2.shape:
        raise RecoveryError("invalid_shape")
    if not np.isfinite(matrix).all():
        raise RecoveryError("nonfinite_candidate")
    if np.any(matrix < 0):
        raise RecoveryError("negative_candidate")
    if np.any((ordinary_A2 > 0) & (matrix == 0)):
        raise RecoveryError("additional_zero")
    if not np.array_equal(matrix[~active_topics], ordinary_A2[~active_topics]):
        raise RecoveryError("inactive_topic_changed")
    row_mass = matrix.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(row_mass).all()
        or np.any(row_mass <= 0)
        or np.max(np.abs(row_mass - 1.0)) > 128.0 * np.finfo(float).eps
    ):
        raise RecoveryError("invalid_simplex_mass")
    normalized = matrix.copy()
    if normalize:
        normalized[active_topics] = matrix[active_topics] / row_mass[active_topics]
    if np.any((ordinary_A2 > 0) & (normalized == 0)):
        raise RecoveryError("additional_zero")
    if (
        not np.isfinite(normalized).all()
        or np.any(normalized < 0)
        or np.max(np.abs(normalized.sum(axis=1) - 1.0)) > 128.0 * np.finfo(float).eps
    ):
        raise RecoveryError("simplex_roundoff_failure")
    return normalized


def refit_A_full_poisson_squarem(
    W: np.ndarray,
    counts: np.ndarray | PreparedPoissonCounts | PreparedCSRPoissonCounts,
    document_lengths: float | np.ndarray,
    *,
    initial_A: np.ndarray | None = None,
    epsilon: float = 1e-12,
    max_evaluations: int = 10_002,
    tolerance: float = 1e-8,
    initial_step: float = 1e-2,
    interior_mass: float = 1e-6,
    chunk_size: int = 250_000,
    prior_strength: float = 0.0,
    prior_base: np.ndarray | None = None,
    statistics_backend: str = "auto",
    acceleration: bool = True,
    step_max: float = 64.0,
    max_backtracks: int = 8,
) -> ARecoveryResult:
    """Opt-in, safeguarded SQUAREM-style acceleration of the identical MLE.

    The original refit_A_full_poisson and all its numerical helpers are
    unchanged. Disabled acceleration delegates to that function with
    max_iter=max_evaluations-2 and returns its original result unchanged,
    including MAP support. Enabled acceleration is explicitly MLE-only.

    Two evaluated ordinary EM steps are retained before extrapolation. A
    proposal is backed toward the second ordinary point, checked for
    feasibility/support, stabilized by one unchanged EM step, and accepted
    only if no worse than that second ordinary point under the original
    roundoff allowance. A rejected trial cannot replace the accepted state.
    Convergence uses only the original full-gradient normalized FW/KKT gap.

    The hard budget counts EVERY attempted statistics call: both initial
    evaluations, ordinary EM, rejected support trials and stabilization.
    Two calls are reserved before a trial; exhaustion always returns an
    already evaluated point, never an unevaluated extrapolation. Iterations
    count accepted ordinary plus accelerated transitions, not outer cycles.

    This is an independently implemented constrained/monotone variant of the
    SqS3 idea, not an unmodified call to the SQUAREM package. Algorithm:
    Varadhan & Roland (2008), doi:10.1111/j.1467-9469.2007.00585.x.
    There are no priors, output floors, randomization or new dense n-by-p
    arrays; the unchanged selected sparse statistics kernel is reused.
    """
    if not isinstance(acceleration, (bool, np.bool_)):
        raise RecoveryError("acceleration must be boolean")
    if (
        not isinstance(max_evaluations, (int, np.integer))
        or isinstance(max_evaluations, (bool, np.bool_))
        or max_evaluations < 2
    ):
        raise RecoveryError("max_evaluations must be an integer >=2")
    max_evaluations = int(max_evaluations)
    if not acceleration:
        # The original path (including its complete result metadata) remains
        # bit-for-bit intact. It can attempt at most two initial evaluations
        # plus max_iter candidate evaluations, even if a kernel rejects one.
        return refit_A_full_poisson(
            W, counts, document_lengths, initial_A=initial_A, epsilon=epsilon,
            max_iter=max_evaluations - 2, tolerance=tolerance,
            initial_step=initial_step, interior_mass=interior_mass,
            chunk_size=chunk_size, prior_strength=prior_strength,
            prior_base=prior_base, statistics_backend=statistics_backend,
        )
    if prior_strength != 0 or prior_base is not None:
        raise RecoveryError("SQUAREM candidate supports only prior-free Poisson MLE")
    if (
        not isinstance(step_max, (int, float, np.integer, np.floating))
        or isinstance(step_max, (bool, np.bool_))
        or not np.isfinite(step_max)
        or step_max < 1
    ):
        raise RecoveryError("step_max must be finite and >=1")
    if (
        not isinstance(max_backtracks, (int, np.integer))
        or isinstance(max_backtracks, (bool, np.bool_))
        or not 0 <= max_backtracks <= 64
    ):
        raise RecoveryError("max_backtracks must be an integer in [0,64]")
    if not all(np.isfinite(x) for x in (epsilon, tolerance, initial_step, interior_mass)):
        raise RecoveryError("optimizer controls must be finite")
    step_max = float(step_max)
    max_backtracks = int(max_backtracks)

    W = np.asarray(W, dtype=float)
    if statistics_backend == "csr_streamed":
        prepared = (
            counts
            if isinstance(counts, PreparedCSRPoissonCounts)
            else _prepare_csr_poisson(counts)
        )
    else:
        prepared = (
            counts
            if isinstance(counts, PreparedPoissonCounts)
            else prepare_poisson_counts(counts)
        )
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
    if epsilon <= 0 or tolerance <= 0 or initial_step <= 0:
        raise RecoveryError("optimizer controls must be positive")
    if not 0 < interior_mass < 1 or chunk_size <= 0 or prior_strength < 0:
        raise RecoveryError("invalid EM interior, chunk, or prior control")

    feature_count = prepared.shape[1]
    selected_statistics_backend = _select_poisson_statistics_backend(statistics_backend, feature_count)
    statistics_counts_csr = (
        _poisson_prepared_csr(prepared) if selected_statistics_backend == "row_tiled" else None
    )
    if initial_A is None:
        feature_totals = (
            prepared.feature_totals
            if isinstance(prepared, PreparedCSRPoissonCounts)
            else np.bincount(
                prepared.column_indices,
                weights=prepared.values,
                minlength=feature_count,
            )
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
    linear_term = float(np.dot(lengths, W_row_sums))
    evaluations_attempted = 0
    evaluations_succeeded = 0
    evaluations_failed = 0
    evaluation_roles = {
        "original_initial": 0, "interiorized_initial": 0,
        "ordinary_em": 0, "extrapolation": 0, "stabilization": 0,
    }

    def evaluate(value_A: np.ndarray, role: str):
        nonlocal evaluations_attempted, evaluations_succeeded, evaluations_failed
        if evaluations_attempted >= max_evaluations:
            # Defensive invariant: ordinary calls and two-call trial
            # reservations below must prevent entry here.
            raise RuntimeError("Poisson SQUAREM statistics budget invariant violated")
        evaluations_attempted += 1
        evaluation_roles[role] += 1
        try:
            if selected_statistics_backend == "csr_streamed":
                local_scores, log_likelihood, minimum_probability = (
                    _poisson_em_statistics_csr(
                        value_A, W, prepared, chunk_size=chunk_size
                    )
                )
            elif selected_statistics_backend == "row_tiled":
                local_scores, log_likelihood, minimum_probability = _poisson_em_statistics_row_tiled(
                    value_A, W, prepared, counts_csr=statistics_counts_csr
                )
            else:
                local_scores, log_likelihood, minimum_probability = _poisson_em_statistics(
                    value_A, W, prepared, chunk_size=chunk_size
                )
            objective = linear_term - log_likelihood
            local_gap, local_gradient_scores = _poisson_optimality_gap(
                value_A, local_scores, prior_strength=prior_strength, prior_base=base
            )
            normalizer = prepared.total_count + W.shape[1] * prior_strength
            normalized_gap = local_gap / normalizer
            if (
                not np.isfinite(objective)
                or not np.isfinite(local_gap)
                or not np.isfinite(normalized_gap)
                or not np.isfinite(minimum_probability)
                or minimum_probability <= 0
            ):
                raise RecoveryError("Poisson SQUAREM evaluation became non-finite or unsupported")
        except RecoveryError:
            evaluations_failed += 1
            raise
        evaluations_succeeded += 1
        return (objective, local_scores, local_gradient_scores,
                local_gap, normalized_gap, minimum_probability)

    try:
        (
            original_initial_objective,
            original_initial_scores,
            original_initial_gradient_scores,
            original_initial_gap,
            original_initial_normalized_gap,
            original_initial_minimum_probability,
        ) = evaluate(original_initial_A, "original_initial")
    except RecoveryError:
        original_initial_objective = float("inf")
        original_initial_scores = None
        original_initial_gradient_scores = None
        original_initial_gap = float("inf")
        original_initial_normalized_gap = float("inf")
        original_initial_minimum_probability = 0.0

    state = evaluate(A, "interiorized_initial")
    interiorized_initial_objective = state[0]
    history = [state[0]]
    converged = state[4] <= tolerance
    status = "ok" if converged else "max_evaluations_reached"
    warnings: list[str] = []
    if inactive_topic_indices.size:
        warnings.append("inactive_W_topics_preserved")
    ordinary_attempted = 0
    ordinary_accepted = 0
    accelerated_accepted = 0
    cycles = 0
    trials = 0
    rejections = 0
    backtracks = 0
    rejection_reasons: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}
    accepted_step_lengths: list[float] = []
    maximum_increase = 0.0
    ordinary_failure = False

    def reject(reason: str):
        nonlocal rejections
        rejections += 1
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    def skip(reason: str):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    while not converged and evaluations_attempted < max_evaluations:
        cycle_A0 = A
        ordinary_points = []
        for _ in range(2):
            if converged or evaluations_attempted >= max_evaluations:
                break
            ordinary_attempted += 1
            try:
                candidate = _poisson_squarem_em_map(A, state[1], active_topics)
            except RecoveryError:
                status = "degenerate_em_update"
                warnings.append(status)
                ordinary_failure = True
                break
            try:
                candidate_state = evaluate(candidate, "ordinary_em")
            except RecoveryError as error:
                status = "numerical_failure"
                warnings.append(f"numerical_failure: {error}")
                ordinary_failure = True
                break
            increase = candidate_state[0] - state[0]
            maximum_increase = max(maximum_increase, increase)
            monotonic_slack = 128.0 * np.finfo(float).eps * max(1.0, abs(state[0]))
            if not np.isfinite(candidate_state[0]) or increase > monotonic_slack:
                status = "objective_increase"
                warnings.append(status)
                ordinary_failure = True
                break
            A = candidate
            state = candidate_state
            ordinary_points.append(A)
            history.append(state[0])
            ordinary_accepted += 1
            if state[4] <= tolerance:
                converged = True
                status = "ok"
                break
        if len(ordinary_points) == 2:
            cycles += 1
        if ordinary_failure or converged or evaluations_attempted >= max_evaluations:
            break
        if len(ordinary_points) != 2:
            # Only a partially completed ordinary pair can exhaust the budget.
            break
        if max_evaluations - evaluations_attempted < 2:
            skip("insufficient_trial_budget")
            continue
        cycle_A1, ordinary_A2 = ordinary_points
        alpha = _poisson_squarem_step_length(cycle_A0, cycle_A1, ordinary_A2, step_max)
        if alpha <= 1.0 or not np.isfinite(alpha):
            skip("plain_em_step_length")
            continue
        # A, state already contain the two evaluated, monotone ordinary EM
        # updates. Any rejected trial leaves those exact objects untouched.
        for backtrack_index in range(max_backtracks + 1):
            if alpha <= 1.0 or max_evaluations - evaluations_attempted < 2:
                break
            trials += 1
            trial = None
            try:
                raw_trial = _poisson_squarem_extrapolate(cycle_A0, cycle_A1, ordinary_A2, alpha)
                trial = _poisson_squarem_validate_candidate(raw_trial, ordinary_A2, active_topics)
            except RecoveryError as error:
                reject("proposal_" + str(error))
            if trial is not None:
                trial_state = None
                try:
                    trial_state = evaluate(trial, "extrapolation")
                except RecoveryError:
                    reject("extrapolation_statistics_failure")
                if trial_state is not None:
                    stabilized = None
                    try:
                        stabilized = _poisson_squarem_em_map(trial, trial_state[1], active_topics)
                    except RecoveryError:
                        reject("degenerate_stabilization")
                    if stabilized is not None:
                        try:
                            stabilized = _poisson_squarem_validate_candidate(
                                stabilized, ordinary_A2, active_topics, normalize=False
                            )
                        except RecoveryError as error:
                            reject("stabilization_" + str(error))
                            stabilized = None
                    if stabilized is not None:
                        stabilized_state = None
                        try:
                            stabilized_state = evaluate(stabilized, "stabilization")
                        except RecoveryError:
                            reject("stabilization_statistics_failure")
                        if stabilized_state is not None:
                            increase = stabilized_state[0] - state[0]
                            maximum_increase = max(maximum_increase, increase)
                            monotonic_slack = 128.0 * np.finfo(float).eps * max(1.0, abs(state[0]))
                            if increase <= monotonic_slack:
                                A = stabilized
                                state = stabilized_state
                                history.append(state[0])
                                accelerated_accepted += 1
                                accepted_step_lengths.append(float(alpha))
                                if state[4] <= tolerance:
                                    converged = True
                                    status = "ok"
                                break
                            reject("accelerated_objective_increase")
            if backtrack_index < max_backtracks:
                alpha = 1.0 + 0.5 * (alpha - 1.0)
                backtracks += 1

    if not converged and status == "max_evaluations_reached":
        warnings.append("maximum_statistics_evaluations_reached")
    termination_before_initial_comparison = status
    value, scores, gradient_scores, gap, normalized_gap, minimum_probability = state
    attempted_iterations = ordinary_attempted + trials
    accepted_iterations = ordinary_accepted + accelerated_accepted
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
    result = ARecoveryResult(
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
        solver="sparse_monotone_squarem_em",
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
            "statistics_backend_requested": statistics_backend,
            "statistics_backend": selected_statistics_backend,
            "statistics_kernel_revision": (
                "csr_streamed_v1"
                if selected_statistics_backend == "csr_streamed"
                else "targeted_row_tiled_v1"
            ),
            "statistics_auto_panel_threshold": POISSON_ROW_TILED_MAX_FEATURES,
            "statistics_row_tile_size": POISSON_ROW_TILE_SIZE if selected_statistics_backend == "row_tiled" else None,
            "statistics_prediction_entry_cap": POISSON_PREDICTION_ENTRY_CAP if selected_statistics_backend == "row_tiled" else None,
            "statistics_full_dense_prediction_allocated": False,
            "optimizer_version": "sparse_monotone_squarem_em_v1",
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

    result.diagnostics.update({
        "acceleration_enabled": True,
        "max_evaluations": max_evaluations,
        "statistics_evaluations_attempted": evaluations_attempted,
        "statistics_evaluations_succeeded": evaluations_succeeded,
        "statistics_evaluations_failed": evaluations_failed,
        "statistics_evaluations_by_role": evaluation_roles,
        "statistics_budget_includes_failed_calls": True,
        "statistics_budget_exhausted": evaluations_attempted >= max_evaluations,
        "ordinary_em_steps_attempted": ordinary_attempted,
        "ordinary_em_steps_accepted": ordinary_accepted,
        "accelerated_steps_accepted": accelerated_accepted,
        "completed_ordinary_pairs": cycles,
        "extrapolation_trials": trials,
        "extrapolation_rejections": rejections,
        "extrapolation_rejection_reasons": rejection_reasons,
        "extrapolation_skips": skip_reasons,
        "extrapolation_backtrack_steps": backtracks,
        "accepted_extrapolation_step_lengths": accepted_step_lengths,
        "squarem_step_max": step_max,
        "squarem_max_backtracks": max_backtracks,
        "squarem_stabilization": "one_unchanged_EM_step",
        "squarem_acceptance_reference": "validated_second_ordinary_EM_point",
        "additional_zeros_relative_to_ordinary_candidate_allowed": False,
        "extrapolation_feasibility": "reject_then_positive_row_scaling_no_projection_or_floor",
        "iteration_count_semantics":
            "accepted_ordinary_EM_plus_accepted_acceleration_transitions_excludes_initial_fallback",
        "attempted_iteration_count_semantics":
            "ordinary_EM_map_attempts_plus_extrapolation_proposals_not_statistics_evaluations",
        "termination_reason_before_initial_comparison": termination_before_initial_comparison,
        "convergence_certificate": "unchanged_full_gradient_normalized_Frank_Wolfe_KKT_gap",
        "returned_point_evaluated": True,
    })
    return result
