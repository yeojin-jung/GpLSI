"""Topic-SCORE adapters preserving the local Tran ratio geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .recovery import RecoveryError, project_rows_simplex


@dataclass
class TopicScoreResult:
    W_hat: np.ndarray
    A_hat: np.ndarray
    ratio_cloud: np.ndarray
    vertices: np.ndarray
    Pi: np.ndarray
    word_vectors: np.ndarray
    selected_word_indices: np.ndarray
    runtime_seconds: float
    method: str
    converged: bool = True
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _tran_successive_projection(
    ratio_cloud: np.ndarray, K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic no-tie branch of local Tran ``successiveProj``."""

    R = np.asarray(ratio_cloud, dtype=float)
    if R.ndim != 2 or R.shape[0] < K or R.shape[1] != max(K - 1, 1):
        raise RecoveryError(f"invalid Topic-SCORE ratio cloud {R.shape} for K={K}")
    if K == 1:
        return R[[int(np.argmax(R[:, 0]))]], np.array([int(np.argmax(R[:, 0]))])
    Y = np.column_stack((np.ones(R.shape[0]), R))
    selected: list[int] = []
    for _ in range(K):
        squared_norms = np.sum(Y**2, axis=1)
        index = int(np.argmax(squared_norms))
        selected.append(index)
        residual = Y[index]
        norm_sq = float(residual @ residual)
        if norm_sq <= np.finfo(float).eps:
            raise RecoveryError("Topic-SCORE successive projection encountered zero residual")
        Y = Y - np.outer(Y @ residual, residual) / norm_sq
    indices = np.asarray(selected, dtype=int)
    return R[indices], indices


def _recover_W_from_A(
    A_hat: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 5_000,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, bool, int]:
    """Fit document proportions on the simplex for a fixed Topic-SCORE A."""

    A = np.asarray(A_hat, dtype=float)
    X = np.asarray(X, dtype=float)
    W = np.full((X.shape[0], A.shape[0]), 1.0 / A.shape[0])
    lipschitz = 2.0 * float(np.linalg.norm(A, 2) ** 2)
    if lipschitz <= np.finfo(float).eps:
        raise RecoveryError("Topic-SCORE A has zero spectral norm")
    step = 1.0 / lipschitz
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        gradient = 2.0 * (W @ A - X) @ A.T
        candidate = project_rows_simplex(W - step * gradient)
        change = float(np.linalg.norm(candidate - W) / max(1.0, np.linalg.norm(W)))
        W = candidate
        if change <= tolerance:
            converged = True
            break
    return W, converged, iteration


def _from_word_vectors(
    word_vectors: np.ndarray,
    eta_hat: np.ndarray,
    X_original: np.ndarray,
    K: int,
    *,
    method: str,
    source_metadata: dict[str, Any] | None = None,
) -> TopicScoreResult:
    started = perf_counter()
    Xi = np.asarray(word_vectors, dtype=float).copy()
    eta = np.asarray(eta_hat, dtype=float).reshape(-1)
    X = np.asarray(X_original, dtype=float)
    if Xi.shape != (X.shape[1], K) or eta.size != X.shape[1]:
        raise RecoveryError("Topic-SCORE word vectors or eta have incompatible dimensions")
    if np.any(eta <= 0) or not np.isfinite(Xi).all():
        raise RecoveryError("Topic-SCORE requires positive word frequencies and finite vectors")
    Xi[:, 0] = np.abs(Xi[:, 0])
    if np.any(Xi[:, 0] <= np.finfo(float).eps):
        raise RecoveryError("Topic-SCORE first word singular vector contains numerical zeros")

    if K == 1:
        ratio = Xi[:, [0]]
        vertices = ratio[[int(np.argmax(ratio[:, 0]))]]
        selected = np.array([int(np.argmax(ratio[:, 0]))])
        Pi = np.ones((X.shape[1], 1))
    else:
        ratio = Xi[:, 1:K] / Xi[:, [0]]
        vertices, selected = _tran_successive_projection(ratio, K)
        augmented_vertices = np.column_stack((vertices, np.ones(K)))
        augmented_cloud = np.column_stack((ratio, np.ones(ratio.shape[0])))
        if np.linalg.matrix_rank(augmented_vertices) < K:
            Pi = augmented_cloud @ np.linalg.pinv(augmented_vertices)
        else:
            Pi = np.linalg.solve(augmented_vertices.T, augmented_cloud.T).T
        Pi = np.maximum(Pi, 0.0)
        sums = Pi.sum(axis=1, keepdims=True)
        if np.any(sums <= np.finfo(float).eps):
            raise RecoveryError("Topic-SCORE barycentric recovery produced a zero row")
        Pi /= sums

    A_feature_by_topic = (np.sqrt(eta) * Xi[:, 0])[:, None] * Pi
    topic_masses = A_feature_by_topic.sum(axis=0)
    if np.any(topic_masses <= np.finfo(float).eps):
        raise RecoveryError("Topic-SCORE topic reconstruction produced zero mass")
    A_feature_by_topic /= topic_masses[None, :]
    A_hat = A_feature_by_topic.T
    W_hat, W_converged, W_iterations = _recover_W_from_A(A_hat, X)
    warnings: list[str] = []
    if not W_converged:
        warnings.append("topicscore_W_refit_max_iter_reached")
    metadata = dict(source_metadata or {})
    metadata.update(
        {
            "orientation": "document_by_word_input",
            "normalization": "norm",
            "Mquantile": 0.0,
            "vertex_hunter": "Tran_SP",
            "W_refit": "simplex_constrained_L2_equivalent_to_local_QP",
            "W_refit_converged": W_converged,
            "W_refit_iterations": W_iterations,
        }
    )
    return TopicScoreResult(
        W_hat=W_hat,
        A_hat=A_hat,
        ratio_cloud=ratio,
        vertices=vertices,
        Pi=Pi,
        word_vectors=Xi,
        selected_word_indices=selected,
        runtime_seconds=perf_counter() - started,
        method=method,
        converged=W_converged,
        warnings=warnings,
        metadata=metadata,
    )


def fit_topicscore_raw(X_original: np.ndarray, K: int) -> TopicScoreResult:
    """Faithful direct-raw port of the local Tran Topic-SCORE P0 call."""

    started = perf_counter()
    X = np.asarray(X_original, dtype=float)
    if X.ndim != 2 or np.any(X < 0):
        raise RecoveryError("Topic-SCORE input must be a nonnegative document-by-word matrix")
    eta_full = X.mean(axis=0)
    retained = np.flatnonzero(eta_full > 0)
    if retained.size < K:
        raise RecoveryError("raw Topic-SCORE has fewer than K positive-frequency words")
    X_retained = X[:, retained]
    eta = eta_full[retained]
    normalized_word_document = X_retained.T / np.sqrt(eta)[:, None]
    Xi, singular, _ = np.linalg.svd(normalized_word_document, full_matrices=False)
    result = _from_word_vectors(
        Xi[:, :K],
        eta,
        X_retained,
        K,
        method="topicscore_raw",
        source_metadata={
            "source": "local Tran r/score.r::score",
            "singular_values": singular[:K].tolist(),
            "native_preprocessing": "no threshold; frequency normalization eta^-1/2",
            "zero_frequency_handling": (
                "mechanical removal before eta^-1/2 normalization; zero rows "
                "reinserted in A on the original vocabulary"
            ),
            "zero_frequency_feature_count": int(X.shape[1] - retained.size),
            "retained_feature_count": int(retained.size),
        },
    )
    if retained.size != X.shape[1]:
        full_A = np.zeros((K, X.shape[1]))
        full_A[:, retained] = result.A_hat
        full_A /= full_A.sum(axis=1, keepdims=True)
        W_hat, converged, iterations = _recover_W_from_A(full_A, X)
        result.A_hat = full_A
        result.W_hat = W_hat
        result.selected_word_indices = retained[result.selected_word_indices]
        result.metadata["retained_word_indices"] = retained.tolist()
        result.metadata["W_refit_converged_full_vocabulary"] = converged
        result.metadata["W_refit_iterations_full_vocabulary"] = iterations
        result.converged = result.converged and converged
        if not converged:
            result.warnings.append("topicscore_W_full_vocabulary_refit_max_iter_reached")
    result.runtime_seconds = perf_counter() - started
    return result


def fit_topicscore_graph_denoised(
    X_original: np.ndarray,
    U_hat: np.ndarray,
    V_hat: np.ndarray,
    singular_values: np.ndarray,
    *,
    retained_indices: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> TopicScoreResult:
    """Run Topic-SCORE geometry on a graph-denoised rank-K factorization.

    The normalized word singular vectors are obtained directly from
    ``diag(eta)^-1/2 R^-1 Vhat Lambdahat``.  This avoids passing a possibly
    signed low-rank reconstruction into code that assumes counts.
    """

    started = perf_counter()
    X = np.asarray(X_original, dtype=float)
    U = np.asarray(U_hat, dtype=float)
    V = np.asarray(V_hat, dtype=float)
    singular = np.asarray(singular_values, dtype=float)
    if singular.ndim == 2:
        singular = np.diag(singular)
    retained = (
        np.arange(X.shape[1], dtype=int)
        if retained_indices is None
        else np.asarray(retained_indices, dtype=int)
    )
    r = np.ones(retained.size) if weights is None else np.asarray(weights, dtype=float)
    if V.shape != (retained.size, U.shape[1]) or singular.size != U.shape[1]:
        raise RecoveryError("graph Topic-SCORE factors have incompatible dimensions")
    if np.linalg.norm(U.T @ U - np.eye(U.shape[1])) > 1e-8:
        raise RecoveryError("graph Topic-SCORE requires orthonormal Uhat")
    eta = X[:, retained].mean(axis=0)
    if np.any(eta <= 0) or np.any(r <= 0):
        raise RecoveryError("graph Topic-SCORE requires positive eta and weights")
    original_word_factor = (V * singular[None, :]) / r[:, None]
    normalized_factor = original_word_factor / np.sqrt(eta)[:, None]
    Xi, normalized_singular, _ = np.linalg.svd(normalized_factor, full_matrices=False)
    retained_result = _from_word_vectors(
        Xi[:, : U.shape[1]],
        eta,
        X[:, retained],
        U.shape[1],
        method="topicscore_graph_denoised",
        source_metadata={
            "source": "graph-aligned Uhat/Vhat/Lambdahat",
            "normalized_factor_singular_values": normalized_singular[: U.shape[1]].tolist(),
            "profile_scale": "original_unweighted",
            "zero_frequency_feature_count": int(X.shape[1] - retained.size),
            "retained_feature_count": int(retained.size),
        },
    )
    if retained.size == X.shape[1] and np.array_equal(retained, np.arange(X.shape[1])):
        retained_result.runtime_seconds = perf_counter() - started
        return retained_result
    full_A = np.zeros((U.shape[1], X.shape[1]))
    full_A[:, retained] = retained_result.A_hat
    full_A /= full_A.sum(axis=1, keepdims=True)
    W_hat, converged, iterations = _recover_W_from_A(full_A, X)
    retained_result.A_hat = full_A
    retained_result.W_hat = W_hat
    retained_result.selected_word_indices = retained[retained_result.selected_word_indices]
    retained_result.metadata["W_refit_converged_full_vocabulary"] = converged
    retained_result.metadata["W_refit_iterations_full_vocabulary"] = iterations
    retained_result.converged = retained_result.converged and converged
    if not converged:
        retained_result.warnings.append(
            "topicscore_W_full_vocabulary_refit_max_iter_reached"
        )
    retained_result.runtime_seconds = perf_counter() - started
    return retained_result
