"""Sparse joint-cohort competitors with explicit objective/provenance contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.exceptions import ConvergenceWarning

from gplsi.recovery import project_rows_simplex
from gplsi.topicscore import _tran_successive_projection
from .likelihood import normalize_nmf_composition, count_csr


@dataclass
class CompetitorResult:
    W: np.ndarray
    A: np.ndarray
    converged: bool
    status: str
    metadata: dict = field(default_factory=dict)


def _counts(value):
    D = count_csr(value)
    if np.any(np.asarray(D.sum(axis=1)).ravel() <= 0):
        raise ValueError("zero count rows must be recorded in shared eligibility masks")
    return D


def _frequencies(D):
    lengths = np.asarray(D.sum(axis=1)).ravel()
    return D.multiply((1 / lengths)[:, None]).tocsr()


def sparse_simplex_l2_refit(A, X, max_iter=5000, tolerance=1e-10):
    """Same Topic-SCORE least-squares objective using n×K and K×K only."""
    gram = A @ A.T
    rhs = np.asarray(X @ A.T)
    W = np.full(rhs.shape, 1 / A.shape[0])
    step = 1 / max(2 * np.linalg.norm(gram, 2), np.finfo(float).tiny)
    for iteration in range(1, max_iter + 1):
        candidate = project_rows_simplex(W - 2 * step * (W @ gram - rhs))
        change = np.linalg.norm(candidate - W) / max(1.0, np.linalg.norm(W))
        W = candidate
        if change <= tolerance:
            return W, True, iteration
    return W, False, max_iter


def _topicscore_profile(Xi, eta, K):
    Xi = Xi.copy()
    Xi[:, 0] = np.abs(Xi[:, 0])
    if np.any(Xi[:, 0] <= np.finfo(float).eps):
        raise ValueError("Topic-SCORE first word vector has numerical zero entries")
    if K == 1:
        Pi = np.ones((len(eta), 1))
        vertices = [int(np.argmax(Xi[:, 0]))]
    else:
        ratio = Xi[:, 1:] / Xi[:, [0]]
        hull, vertices = _tran_successive_projection(ratio, K)
        H = np.column_stack((hull, np.ones(K)))
        cloud = np.column_stack((ratio, np.ones(len(eta))))
        rank = np.linalg.matrix_rank(H)
        Pi = cloud @ np.linalg.pinv(H) if rank < K else np.linalg.solve(H.T, cloud.T).T
        Pi = np.maximum(Pi, 0)
        if np.any(Pi.sum(axis=1) == 0):
            raise ValueError("Topic-SCORE barycentric recovery produced a zero row")
        Pi /= Pi.sum(axis=1, keepdims=True)
    A = ((np.sqrt(eta) * Xi[:, 0])[:, None] * Pi).T
    if np.any(A.sum(axis=1) <= 0):
        raise ValueError("Topic-SCORE recovered an empty topic")
    A /= A.sum(axis=1, keepdims=True)
    return A, np.asarray(vertices)


def fit_topicscore(counts, K, seed, *, spectral=None, max_iter=5000):
    started = perf_counter()
    D = _counts(counts)
    X = _frequencies(D)
    eta_full = np.asarray(X.mean(axis=0)).ravel()
    retained = np.flatnonzero(eta_full > 0)
    if len(retained) <= K:
        raise ValueError("partial SVD requires more than K positive-frequency features")
    eta = eta_full[retained]
    if spectral is None:
        normalized = X[:, retained].multiply((1 / np.sqrt(eta))[None, :]).tocsr()
        _, singular, Vt = svds(normalized, k=K, which="LM", random_state=seed)
        order = np.argsort(singular)[::-1]
        Xi = Vt[order].T
        singular = singular[order]
        implementation = "legacy_Tran_ratio_geometry_sparse_partial_SVD"
    else:
        # Supplied native vocabulary mapping, factors from the shared P0 block.
        U = np.asarray(spectral["U"])
        if U.shape != (D.shape[0], K) or not np.allclose(U.T @ U, np.eye(K), atol=1e-8):
            raise ValueError("graph Topic-SCORE requires matched orthonormal U")
        retained = np.asarray(spectral["retained_indices"], dtype=int)
        eta = eta_full[retained]
        V = spectral["V"]
        singular = np.asarray(spectral["singular_values"])
        weights = np.asarray(spectral.get("weights", np.ones(len(retained))))
        normalized = (V * singular[None, :]) / (weights * np.sqrt(eta))[:, None]
        Xi, singular, _ = np.linalg.svd(normalized, full_matrices=False)
        Xi = Xi[:, :K]
        implementation = "legacy_Tran_ratio_geometry_low_rank_graph_denoised"
    local_A, vertices = _topicscore_profile(Xi, eta, K)
    A = np.zeros((K, D.shape[1]))
    A[:, retained] = local_A
    W, converged, iterations = sparse_simplex_l2_refit(A, X, max_iter=max_iter)
    return CompetitorResult(W, A, converged, "ok" if converged else "max_iter_reached",
                            {"implementation": implementation, "iterations": iterations,
                             "selected_gene_indices": retained[vertices].tolist(),
                             "singular_values": singular.tolist(), "runtime_seconds": perf_counter() - started,
                             "W_train_recovery": "historical_simplex_L2", "joint_profiles": True})


def fit_lda(counts, K, seed, max_iter=50):
    started = perf_counter()
    D = _counts(counts)
    model = LatentDirichletAllocation(n_components=K, learning_method="batch", n_jobs=1,
                                     random_state=seed, max_iter=max_iter,
                                     evaluate_every=1, perp_tol=1e-3)
    W = model.fit_transform(D)
    A = model.components_ / model.components_.sum(axis=1, keepdims=True)
    converged = model.n_iter_ < max_iter
    return CompetitorResult(W, A, converged, "ok" if converged else "max_iter_reached",
                            {"implementation": "sklearn_batch_variational_LDA", "iterations": model.n_iter_,
                             "runtime_seconds": perf_counter() - started, "joint_profiles": True})


def fit_kl_nmf(counts, K, seed, max_iter=500, tolerance=1e-4):
    started = perf_counter()
    model = NMF(n_components=K, init="nndsvda", solver="mu", beta_loss="kullback-leibler",
                max_iter=max_iter, tol=tolerance, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        U = model.fit_transform(_counts(counts))
    W, A, normalization = normalize_nmf_composition(U, model.components_)
    converged = model.n_iter_ < max_iter
    return CompetitorResult(W, A, converged, "ok" if converged else "max_iter_reached",
                            {"implementation": "sklearn_sparse_KL_NMF", "iterations": model.n_iter_,
                             "runtime_seconds": perf_counter() - started, "joint_profiles": True,
                             "normalization": normalization, "warnings": [str(w.message) for w in caught]})


def _ratio_nnz(D, U, H, chunk_size=100000):
    data = np.empty(D.nnz, dtype=float)
    nll = float(U.sum(axis=0) @ H.sum(axis=1))
    for begin in range(0, D.nnz, chunk_size):
        end = min(D.nnz, begin + chunk_size)
        rows = np.searchsorted(D.indptr, np.arange(begin, end), side="right") - 1
        mu = np.einsum("ik,ki->i", U[rows], H[:, D.indices[begin:end]])
        if np.any(mu <= 0):
            raise FloatingPointError("KL-NMF assigned zero intensity to positive training counts")
        data[begin:end] = D.data[begin:end] / mu
        nll -= float(D.data[begin:end] @ np.log(mu))
    return sparse.csr_matrix((data, D.indices, D.indptr), shape=D.shape), nll


def fit_graph_kl_nmf(counts, adjacency, K, seed, *, penalty=.25, max_iter=500,
                     tolerance=1e-5, chunk_size=100000):
    """Legacy multiplicative update, sparse; stopping monitors its full objective.

    The historical update corresponds to (penalty/2) Tr(U' L U), with
    unnormalized U. The scale ambiguity is reported, not silently removed.
    """
    started = perf_counter()
    D = _counts(counts)
    G = sparse.csr_matrix(adjacency)
    if penalty < 0 or max_iter < 1 or chunk_size < 1 or tolerance <= 0:
        raise ValueError("nonnegative penalty and positive solver budgets required")
    if G.shape != (D.shape[0], D.shape[0]) or np.any(G.data < 0) or not np.isfinite(G.data).all():
        raise ValueError("invalid count-matched nonnegative graph")
    if (G - G.T).nnz:
        raise ValueError("symmetric adjacency required")
    degree = np.asarray(G.sum(axis=1)).ravel()
    initial = NMF(n_components=K, init="nndsvda", max_iter=1, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        U = np.maximum(initial.fit_transform(D), 1e-10)
    H = np.maximum(initial.components_, 1e-10)
    ratio, nll = _ratio_nnz(D, U, H, chunk_size)
    objective = nll + .5 * penalty * float(np.sum(U * (degree[:, None] * U - G @ U)))
    history = [objective]
    converged = False
    for iteration in range(1, max_iter + 1):
        H *= np.asarray(U.T @ ratio) / np.maximum(U.sum(axis=0)[:, None], 1e-300)
        ratio, _ = _ratio_nnz(D, U, H, chunk_size)
        U *= (ratio @ H.T + penalty * (G @ U)) / np.maximum(
            H.sum(axis=1)[None, :] + penalty * degree[:, None] * U, 1e-300)
        ratio, nll = _ratio_nnz(D, U, H, chunk_size)
        new_objective = nll + .5 * penalty * float(np.sum(U * (degree[:, None] * U - G @ U)))
        if new_objective > history[-1] + 1e-8 * max(1, abs(history[-1])):
            raise FloatingPointError("graph-KL-NMF full objective increased")
        history.append(new_objective)
        if abs(history[-2] - history[-1]) <= tolerance * max(1, abs(history[-2])):
            converged = True
            break
    W, A, normalization = normalize_nmf_composition(U, H)
    return CompetitorResult(W, A, converged, "ok" if converged else "max_iter_reached",
                            {"implementation": "legacy_graph_KL_NMF_sparse_full_objective_stop_v2",
                             "penalty": penalty, "penalty_definition": "penalty/2*Tr(U.T@L@U)",
                             "legacy_graph_parity": "old upper-triangle adjacency + transpose equals v2 symmetric G; no doubling",
                             "scale_nonidentifiability": "unnormalized U/H scale affects graph penalty",
                             "iterations": iteration, "objective_history": history,
                             "normalization": normalization, "joint_profiles": True,
                             "runtime_seconds": perf_counter() - started})


class SparseFeatureView:
    """The pinned Calico train/_update_xis interface needs only values and index."""
    def __init__(self, counts, strata):
        self.values = sparse.csr_matrix(counts)
        self.index = pd.Index([(str(s), int(i)) for i, s in enumerate(strata)], tupleize_cols=False)


def fit_spatial_lda(counts, coordinates, strata, K, seed, *, penalty=.25,
                    outer_iterations=3, lda_iterations=5, admm_iterations=15):
    """One shared Calico model; original native Voronoi/MST priors per stratum.

    Pinned source functions are wrapped; counts remain CSR through .values.
    Train uses a fixed internal seed 0 in the pinned implementation, which is
    reported explicitly. A separate seed-sensitive wrapper is not invented.
    """
    if penalty <= 0:
        raise ValueError("Calico difference_penalty is inverse weight and must be strictly positive")
    utilities = str(Path(__file__).resolve().parents[2] / "utils")
    if utilities not in sys.path:
        sys.path.insert(0, utilities)
    from spatial_lda import model as calico
    from spatial_lda.featurization import make_merged_difference_matrices
    started = perf_counter()
    D = _counts(counts)
    strata = np.asarray(strata).astype(str)
    features = SparseFeatureView(D, strata)
    # A zero-column frame provides the original graph API with indices only.
    index_frame = pd.DataFrame(index=features.index)
    coordinate_frames = {s: pd.DataFrame(np.asarray(coordinates)[strata == s],
                                        index=np.flatnonzero(strata == s), columns=["x", "y"])
                         for s in sorted(set(strata))}
    if any(len(frame) < 4 for frame in coordinate_frames.values()):
        raise ValueError("Calico native Voronoi/MST requires >=4 noncollinear observations per stratum")
    differences = make_merged_difference_matrices(index_frame, coordinate_frames, "x", "y")
    model = calico.train(features, differences, K, difference_penalty=penalty,
                         max_lda_iter=lda_iterations, max_admm_iter=admm_iterations,
                         n_iters=outer_iterations, n_parallel_processes=1, verbosity=0)
    W = np.asarray(model.topic_weights.values, dtype=float)
    A = np.asarray(model.components_, dtype=float)
    W /= W.sum(axis=1, keepdims=True)
    A /= A.sum(axis=1, keepdims=True)
    return CompetitorResult(W, A, False, "fixed_schedule_completed_uncertified",
                            {"implementation": "Calico_de6b00e_sparse_joint_interface_wrapper",
                             "graph": "native_Voronoi_MST_independently_within_stratum",
                             "difference_penalty": penalty, "effective_inverse_penalty": 1 / penalty,
                             "requested_seed": seed, "effective_internal_seed": 0,
                             "joint_profiles": True, "strata": len(differences),
                             "convergence_certificate": "not_exposed_by_pinned_Calico_training_loop",
                             "runtime_seconds": perf_counter() - started})
