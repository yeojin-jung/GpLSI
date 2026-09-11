"""Sparse composition likelihoods; adaptation and scoring are separate APIs.

The fixed-W implementation is the validated 9a23a18 repair in gplsi.recovery.
No function in this module accepts biological annotations or scoring counts as
an input to an estimator. Dense objects are at most n-by-K, K-by-p, or bounded
nonzero-entry chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.linalg import qr, solve_triangular

from gplsi.recovery import ARecoveryResult, RecoveryError, project_rows_simplex
from gplsi.recovery import refit_A_full_poisson


def count_csr(counts) -> sparse.csr_matrix:
    """Canonical nonnegative integer counts, without dense conversion."""
    x = sparse.csr_matrix(counts, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if x.ndim != 2 or not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(x.data != np.floor(x.data)):
        raise ValueError("raw integer counts are required")
    return x


def simplex_matrix(value, name: str, *, storage_tolerance=2e-6) -> np.ndarray:
    """Validate a simplex factor; undo only documented storage-roundoff drift."""
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or 0 in a.shape or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must be a nonempty finite nonnegative matrix")
    totals = a.sum(axis=1)
    if np.max(np.abs(totals - 1)) > storage_tolerance:
        raise ValueError(f"{name} rows must sum to one")
    return a / totals[:, None]


def recover_A_current(W, counts, *, rank_rtol=None) -> ARecoveryResult:
    """Historical LS-then-simplex A_current, with QR rather than an inverse.

    This is the same full-rank estimator, without ridge or minimum-norm
    substitution. Singular W is an explicit failure. Input frequencies are
    formed from the full raw recovery vocabulary and matching row depths.
    """
    w = simplex_matrix(W, "W")
    d = count_csr(counts)
    lengths = np.asarray(d.sum(axis=1), dtype=float).ravel()
    if len(lengths) != w.shape[0] or np.any(lengths <= 0):
        raise RecoveryError("A_current requires matching positive-count training rows")
    x = d.astype(np.float64).multiply((1.0 / lengths)[:, None]).tocsr()
    q, r = qr(w, mode="economic", check_finite=False)
    singular = np.linalg.svd(r, compute_uv=False)
    cutoff = (max(w.shape) * np.finfo(float).eps if rank_rtol is None else rank_rtol)
    rank = int(np.count_nonzero(singular > cutoff * singular[0]))
    if rank != w.shape[1]:
        raise RecoveryError(f"A_current rank-deficient W: rank={rank}, K={w.shape[1]}, "
                            f"smallest_singular_value={singular[-1]:.9g}; no ridge applied")
    rhs = np.asarray(x.T @ q).T
    unconstrained = solve_triangular(r, rhs, check_finite=False)
    a = project_rows_simplex(unconstrained)
    wtx = np.asarray(x.T @ w).T
    squared_error = max(0.0, float(np.sum(a * ((w.T @ w) @ a))
                                  - 2 * np.sum(a * wtx) + np.dot(x.data, x.data)))
    return ARecoveryResult(
        A_hat=a, method="A_current", converged=True, iterations=0,
        objective_history=[squared_error], status="ok", solver="full_rank_QR_then_simplex",
        objective_name="row_frequency_squared_error_after_historical_simplex_projection",
        diagnostics={"rank": rank, "rank_relative_tolerance": float(cutoff),
                     "condition_number": float(singular[0] / singular[-1]),
                     "smallest_singular_value": float(singular[-1]), "ridge": 0.0,
                     "is_constrained_LS_optimum": False,
                     "training_count_total": float(lengths.sum()),
                     "full_dense_count_or_prediction_allocated": False},
    )


def recover_A_poisson(W, counts, *, initial_A=None, max_iter=2000,
                      tolerance=1e-8, chunk_size=250000) -> ARecoveryResult:
    """Enforce joint contracts, then call the preserved certified sparse repair."""
    w = simplex_matrix(W, "W")
    d = count_csr(counts)
    lengths = np.asarray(d.sum(axis=1), dtype=float).ravel()
    if len(lengths) != w.shape[0] or np.any(lengths <= 0):
        raise RecoveryError("Poisson recovery requires matching positive-count training rows")
    initial = None if initial_A is None else simplex_matrix(initial_A, "initial_A")
    fit = refit_A_full_poisson(w, d, lengths, initial_A=initial,
                              max_iter=max_iter, tolerance=tolerance, chunk_size=chunk_size)
    simplex_matrix(fit.A_hat, "recovered A")
    fit.diagnostics.update({"source_repair_commit": "9a23a1822408217219d1b113d2d79200b7b0157b",
                            "joint_contract_version": "joint_v2", "raw_integer_counts": True,
                            "lengths_computed_from_recovery_vocabulary": True})
    if fit.converged and not fit.diagnostics["final_objective_no_worse_than_original_initial"]:
        raise RecoveryError("certified Poisson solution is worse than feasible supplied A")
    return fit


def normalize_nmf_composition(U, H):
    """Preserve normalized U@H by compensating both factor scales.

    An inactive zero-mass topic receives a uniform, unidentified A row and
    zero W column. A zero fitted-depth observation is a method failure.
    """
    u, h = np.asarray(U, dtype=float), np.asarray(H, dtype=float)
    if u.ndim != 2 or h.ndim != 2 or u.shape[1] != h.shape[0]:
        raise ValueError("incompatible NMF factors")
    if not np.isfinite(u).all() or not np.isfinite(h).all() or np.any(u < 0) or np.any(h < 0):
        raise ValueError("NMF factors must be finite and nonnegative")
    mass = h.sum(axis=1)
    active = mass > 0
    a = np.full(h.shape, 1 / h.shape[1], dtype=float)
    a[active] = h[active] / mass[active, None]
    scaled = u * mass[None, :]
    depths = scaled.sum(axis=1)
    if np.any(depths <= 0):
        raise RecoveryError(f"NMF predicts zero depth in {np.count_nonzero(depths <= 0)} rows")
    w = scaled / depths[:, None]
    return w, a, {"normalization": "compensated_topic_scales_then_row_composition",
                  "inactive_topic_indices": np.flatnonzero(~active).tolist(),
                  "topic_scales": mass.tolist(), "fitted_depth": depths,
                  "inactive_profile_identified": False if np.any(~active) else True}


@dataclass
class FoldInResult:
    W: np.ndarray
    row_status: np.ndarray
    row_converged: np.ndarray
    normalized_gap: np.ndarray
    iterations: np.ndarray
    metadata: dict

    @property
    def inference_valid(self):
        return np.isin(self.row_status, ["ok", "max_iter_reached"])


def _fold_statistics(w, a, d, rows, entry_chunk_size):
    probabilities = np.empty(d.nnz, dtype=float)
    for start in range(0, d.nnz, entry_chunk_size):
        stop = min(start + entry_chunk_size, d.nnz)
        probabilities[start:stop] = np.einsum(
            "ik,ik->i", w[rows[start:stop]], a[:, d.indices[start:stop]].T)
    if np.any(probabilities <= 0) or not np.isfinite(probabilities).all():
        raise RecoveryError("fixed-A likelihood has unsupported positive adaptation counts")
    ratio = sparse.csr_matrix((d.data / probabilities, d.indices, d.indptr), shape=d.shape)
    score = np.asarray(ratio @ a.T)
    nll = np.bincount(rows, weights=-d.data * np.log(probabilities), minlength=d.shape[0])
    gap = np.maximum(0.0, score.max(axis=1) - np.sum(w * score, axis=1))
    return score, nll, gap


def fold_in_fixed_A(A, adapt_counts, *, max_iter=2000, tolerance=1e-7,
                    row_chunk_size=256, entry_chunk_size=250000) -> FoldInResult:
    """Infer W from adaptation counts with A fixed, using row-separable EM.

    Stopping uses each row's count-normalized Frank-Wolfe gap. Rows having
    unsupported adaptation genes are explicitly failed, with uniform W only
    as a storage placeholder; inference_valid must accompany later scoring.
    Zero adaptation rows are a method-independent count exclusion. No scoring
    counts, labels, graph, or data-dependent initializer enters this routine.
    """
    a = simplex_matrix(A, "A")
    d = count_csr(adapt_counts)
    if d.shape[1] != a.shape[1] or max_iter < 0 or tolerance <= 0:
        raise ValueError("incompatible fold-in dimensions or controls")
    if row_chunk_size <= 0 or entry_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    n, k = d.shape[0], a.shape[0]
    depths = np.asarray(d.sum(axis=1), dtype=float).ravel()
    w_all = np.full((n, k), 1 / k)
    status = np.full(n, "no_adaptation_counts", dtype="U40")
    converged = np.zeros(n, bool)
    gaps = np.full(n, np.nan)
    iterations = np.zeros(n, np.int32)
    unsupported_gene = a.sum(axis=0) == 0
    unsupported_rows = np.asarray(d[:, unsupported_gene].sum(axis=1)).ravel() > 0
    status[unsupported_rows] = "unsupported_adaptation_counts"
    eligible = (depths > 0) & ~unsupported_rows
    max_increase = 0.0
    chunk_summaries = []
    for chunk_start in range(0, n, row_chunk_size):
        ids = np.arange(chunk_start, min(n, chunk_start + row_chunk_size))
        ids = ids[eligible[ids]]
        if ids.size == 0:
            continue
        local = d[ids].astype(float)
        rows = np.repeat(np.arange(ids.size), np.diff(local.indptr))
        w = w_all[ids].copy()
        scores, nll, gap = _fold_statistics(w, a, local, rows, entry_chunk_size)
        normalized = gap / depths[ids]
        done = normalized <= tolerance
        failed = np.zeros(len(ids), bool)
        initial_nll = float(nll.sum())
        for iteration in range(1, max_iter + 1):
            active = ~done & ~failed
            if not np.any(active):
                break
            candidate = w.copy()
            candidate[active] *= scores[active]
            candidate[active] /= candidate[active].sum(axis=1, keepdims=True)
            new_scores, new_nll, new_gap = _fold_statistics(candidate, a, local, rows, entry_chunk_size)
            increase = new_nll - nll
            max_increase = max(max_increase, float(increase.max()))
            slack = 128 * np.finfo(float).eps * np.maximum(1.0, np.abs(nll))
            bad = active & ((increase > slack) | ~np.isfinite(new_nll))
            failed |= bad
            accept = active & ~bad
            w[accept] = candidate[accept]
            scores[accept] = new_scores[accept]
            nll[accept] = new_nll[accept]
            gap[accept] = new_gap[accept]
            normalized = gap / depths[ids]
            iterations[ids[accept]] = iteration
            done = normalized <= tolerance
        w_all[ids] = w
        gaps[ids] = normalized
        converged[ids] = done & ~failed
        status[ids] = np.where(failed, "objective_increase", np.where(done, "ok", "max_iter_reached"))
        chunk_summaries.append({"rows": int(len(ids)), "initial_nll": initial_nll,
                               "final_nll": float(nll.sum()), "max_iterations": int(iterations[ids].max())})
    return FoldInResult(w_all, status, converged, gaps, iterations,
                        {"solver": "fixed_A_sparse_row_simplex_EM_v1", "tolerance": tolerance,
                         "max_iter": max_iter, "row_chunk_size": row_chunk_size,
                         "entry_chunk_size": entry_chunk_size, "initializer": "uniform_all_topics",
                         "optimality_certificate": "row_Frank_Wolfe_gap_divided_by_adaptation_depth",
                         "count_eligible_rows": int(np.count_nonzero(depths > 0)),
                         "unsupported_adaptation_rows": int(np.count_nonzero(unsupported_rows)),
                         "unsupported_adaptation_molecules": int(d[:, unsupported_gene].sum()),
                         "maximum_objective_increase": max_increase, "chunk_summaries": chunk_summaries,
                         "A_updated": False, "scoring_counts_used": False})
