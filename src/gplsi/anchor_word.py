"""Word-profile geometry and W recovery for anchor-word GpLSI.

The primary point cloud is built only after undoing feature weights.  All
functions use document-by-word matrices and topic-by-word A matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .recovery import RecoveryError, project_rows_simplex, project_simplex


@dataclass
class WordProfileResult:
    Z_hat: np.ndarray
    M_hat_retained: np.ndarray
    eta_hat_retained: np.ndarray
    original_scale_cross_product: np.ndarray
    identity_error: float
    orthonormality_error: float
    profile_source: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnchorWordWRecoveryResult:
    G_hat: np.ndarray
    b_hat: np.ndarray
    prevalence_residual: float
    raw: np.ndarray
    truncated_normalized: np.ndarray
    simplex_projected: np.ndarray
    condition_number: float
    smallest_singular_value: float
    converged: bool
    iterations: int
    objective_history: list[float]
    stable: bool
    warnings: list[str] = field(default_factory=list)


def population_word_geometry(
    W: np.ndarray,
    A: np.ndarray,
    U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``b, eta, Pi, H_word`` for the population identity."""

    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)
    U = np.asarray(U, dtype=float)
    if W.ndim != 2 or A.ndim != 2 or U.ndim != 2:
        raise RecoveryError("W, A, and U must be matrices")
    if W.shape != U.shape or W.shape[1] != A.shape[0]:
        raise RecoveryError("population word-geometry dimensions are incompatible")
    b = W.mean(axis=0)
    M = W @ A
    eta = M.mean(axis=0)
    if np.any(b <= 0) or np.any(eta <= 0):
        raise RecoveryError("population word geometry requires positive b and eta")
    Pi = (A.T * b[None, :]) / eta[:, None]
    H_word = (W.T @ U) / b[:, None]
    return b, eta, Pi, H_word


def build_word_profile(
    X_original: np.ndarray,
    U_hat: np.ndarray,
    V_hat: np.ndarray,
    singular_values: np.ndarray,
    weights: np.ndarray,
    retained_indices: np.ndarray,
    *,
    profile_source: str = "denoised_original_scale",
    orthonormal_tolerance: float = 1e-8,
) -> WordProfileResult:
    """Construct the K-dimensional normalized word-profile point cloud.

    For the primary source this evaluates

    ``diag(eta_J)^-1 R^-1 Vhat Lambdahat``

    and also forms ``Mhat_J = Uhat Lambdahat Vhat.T R^-1`` for diagnostics.
    No factor of ``1/n`` or arbitrary cloud normalization is introduced.
    """

    X = np.asarray(X_original, dtype=float)
    U = np.asarray(U_hat, dtype=float)
    V = np.asarray(V_hat, dtype=float)
    singular = np.asarray(singular_values, dtype=float)
    if singular.ndim == 2:
        singular = np.diag(singular)
    r = np.asarray(weights, dtype=float).reshape(-1)
    retained = np.asarray(retained_indices, dtype=int).reshape(-1)
    if X.ndim != 2 or U.ndim != 2 or V.ndim != 2:
        raise RecoveryError("X, Uhat, and Vhat must be matrices")
    if U.shape[0] != X.shape[0] or U.shape[1] != V.shape[1]:
        raise RecoveryError("spectral factor dimensions are incompatible")
    if V.shape[0] != retained.size or r.size != retained.size:
        raise RecoveryError("retained features and weights do not match Vhat")
    if singular.size != U.shape[1] or np.any(r <= 0):
        raise RecoveryError("singular values or feature weights are invalid")
    if np.any(retained < 0) or np.any(retained >= X.shape[1]):
        raise RecoveryError("retained feature indices are out of bounds")
    if profile_source not in {"denoised_original_scale", "observed_cross_product"}:
        raise RecoveryError(f"unknown profile source {profile_source!r}")

    gram_error = float(np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord="fro"))
    if gram_error > orthonormal_tolerance:
        raise RecoveryError(
            "anchor-word profile requires orthonormal Uhat; "
            f"||Uhat.T Uhat-I||_F={gram_error:.6g}"
        )

    weighted_cross = V * singular[None, :]
    original_cross = weighted_cross / r[:, None]
    M_hat = (U * singular[None, :]) @ V.T
    M_hat = M_hat / r[None, :]
    direct_cross = M_hat.T @ U
    identity_error = float(np.linalg.norm(direct_cross - original_cross, ord="fro"))

    eta = X[:, retained].mean(axis=0)
    if np.any(eta <= 0):
        bad = retained[np.flatnonzero(eta <= 0)]
        raise RecoveryError(
            "word-profile normalization is undefined for zero-frequency columns: "
            f"{bad.tolist()}"
        )
    if profile_source == "denoised_original_scale":
        numerator = original_cross
    else:
        numerator = X[:, retained].T @ U
    Z = numerator / eta[:, None]
    warnings: list[str] = []
    if identity_error > 100 * np.finfo(float).eps * max(1.0, np.linalg.norm(original_cross)):
        warnings.append("original_scale_cross_product_identity_roundoff")
    return WordProfileResult(
        Z_hat=Z,
        M_hat_retained=M_hat,
        eta_hat_retained=eta,
        original_scale_cross_product=original_cross,
        identity_error=identity_error,
        orthonormality_error=gram_error,
        profile_source=profile_source,
        warnings=warnings,
    )


def solve_topic_prevalence(
    G: np.ndarray,
    *,
    max_iter: int = 10_000,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, bool, int, list[float]]:
    """Solve ``min_{q in simplex} ||Gq-1||_2^2`` by projected gradient."""

    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or not np.isfinite(G).all():
        raise RecoveryError("G must be a finite matrix")
    spectral_sq = float(np.linalg.norm(G, 2) ** 2)
    if spectral_sq <= np.finfo(float).eps:
        raise RecoveryError("prevalence design matrix has zero spectral norm")
    target = np.ones(G.shape[0])
    q = np.full(G.shape[1], 1.0 / G.shape[1])
    step = 1.0 / spectral_sq

    def objective(value: np.ndarray) -> float:
        residual = G @ value - target
        return 0.5 * float(residual @ residual)

    history = [objective(q)]
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        gradient = G.T @ (G @ q - target)
        candidate = project_simplex(q - step * gradient)
        candidate_objective = objective(candidate)
        local_step = step
        while candidate_objective > history[-1] + 1e-14 and local_step > 1e-18:
            local_step *= 0.5
            candidate = project_simplex(q - local_step * gradient)
            candidate_objective = objective(candidate)
        change = float(np.linalg.norm(candidate - q))
        q = candidate
        history.append(candidate_objective)
        step = min(local_step * 1.1, 1.0 / spectral_sq)
        if change <= tolerance:
            converged = True
            break
    return q, converged, iteration, history


def recover_W_from_word_vertices(
    U_hat: np.ndarray,
    H_word: np.ndarray,
    *,
    condition_threshold: float = 1e12,
    max_iter: int = 10_000,
    tolerance: float = 1e-12,
) -> AnchorWordWRecoveryResult:
    """Recover W from word vertices without inverting ``H_word``."""

    U = np.asarray(U_hat, dtype=float)
    H = np.asarray(H_word, dtype=float)
    if U.ndim != 2 or H.ndim != 2 or H.shape != (U.shape[1], U.shape[1]):
        raise RecoveryError(
            f"word-side W recovery requires U=n x K and H=K x K; got {U.shape}, {H.shape}"
        )
    singular = np.linalg.svd(H, compute_uv=False)
    smallest = float(singular[-1])
    condition = np.inf if smallest == 0 else float(singular[0] / smallest)
    warnings: list[str] = []
    stable = bool(np.isfinite(condition) and condition <= condition_threshold)
    if not stable:
        warnings.append("ill_conditioned_word_vertex_matrix")

    G = U @ H.T
    b_hat, converged, iterations, history = solve_topic_prevalence(
        G, max_iter=max_iter, tolerance=tolerance
    )
    residual = float(np.linalg.norm(G @ b_hat - np.ones(U.shape[0])))
    raw = G * b_hat[None, :]
    truncated = np.maximum(raw, 0.0)
    row_sums = truncated.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= np.finfo(float).eps
    if np.any(zero_rows):
        warnings.append("zero_rows_after_word_W_nonnegative_truncation")
        truncated[zero_rows] = 1.0 / H.shape[0]
        row_sums = truncated.sum(axis=1, keepdims=True)
    truncated /= row_sums
    simplex = project_rows_simplex(raw)
    if not converged:
        warnings.append("topic_prevalence_max_iter_reached")
    return AnchorWordWRecoveryResult(
        G_hat=G,
        b_hat=b_hat,
        prevalence_residual=residual,
        raw=raw,
        truncated_normalized=truncated,
        simplex_projected=simplex,
        condition_number=condition,
        smallest_singular_value=smallest,
        converged=converged,
        iterations=iterations,
        objective_history=history,
        stable=stable,
        warnings=warnings,
    )
