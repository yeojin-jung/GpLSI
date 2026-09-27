"""Sparse, training-only, five-fold graph spectral fitting for joint_v2.

The graph smoother is the pinned SSNAL implementation and its objective remains
0.5 ||C-XV||_F^2 + lambda sum_{undirected e} w_e ||C_i-C_j||_2.
The CV loss remains ||X_validation V-C_validation||_F / n_validation,
summed with equal weight over exactly five folds. ``fold_fitted_v2`` removes
the old full-data feature/initialization leak: each candidate's entire spectral
fit sees fold training data and coordinate-only interpolation only.

Only sparse count/frequency matrices and dense n-by-K or p-by-K factors are
formed. Reconstruction changes are evaluated by low-rank trace identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
from scipy.sparse.linalg import LinearOperator, aslinearoperator, svds


PREPROCESSINGS = ("P0_raw", "P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke")


@dataclass(frozen=True)
class SpectralConfig:
    nfolds: int = 5
    initial_grid: tuple[float, ...] = tuple(sorted(set([0.0, 1e-6] + [1e-4 * 1.7**j for j in range(29)])))
    grid_growth: float = 1.7
    extension_batch_size: int = 4
    max_candidates: int = 47
    lambda_ceiling: float = 1e6
    plateau_relative_tolerance: float = 1e-4
    plateau_points: int = 3
    max_iterations: int = 300
    reconstruction_tolerance: float = 1e-5
    stable_iterations: int = 2
    ssnal_max_iterations: int = 1000
    ssnal_admm_iterations: int = 100
    ssnal_tolerance: float = 1e-6
    svd_tolerance: float = 1e-8
    initialization: str = "current"
    detection_fraction: float = 0.01
    tran_alpha: float = 0.005
    fold_retries: int = 0


@dataclass
class FeaturePlan:
    panel_indices: np.ndarray
    retained_indices: np.ndarray
    weights: np.ndarray
    eta: np.ndarray
    row_totals: np.ndarray
    metadata: dict


@dataclass
class Embedding:
    U: np.ndarray
    V: np.ndarray
    singular_values: np.ndarray
    U_bar: np.ndarray
    lambda_value: float
    converged: bool
    iterations: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SpectralResult:
    selected: Embedding | None
    zero: Embedding
    features: FeaturePlan
    fold_ids: np.ndarray
    metadata: dict


def _csr_counts(counts):
    D = sparse.csr_matrix(counts, dtype=np.float64)
    if D.ndim != 2 or not np.isfinite(D.data).all() or np.any(D.data < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(D.data != np.floor(D.data)):
        raise ValueError("raw integer counts are required")
    return D


def row_frequencies(counts):
    D = sparse.csr_matrix(counts, dtype=np.float64)
    totals = np.asarray(D.sum(axis=1)).ravel()
    inverse = np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)
    return sparse.diags(inverse) @ D, totals


def learn_feature_plan(counts, training_mask, preprocessing, *, requested_panel=None,
                       candidate_gene_indices=None, panel_indices=None, config=None):
    """Learn ranking, detection, threshold and weights from allowed rows only.

    Input columns are the full outer-training source assay. The candidate mask
    may exclude source-unusable genes, but must not be learned from scoring data.
    ``requested_panel=None`` retains the source targeted panel. The public plan
    records canonical column indices; neither test data nor annotations enter.
    """
    config = config or SpectralConfig()
    if preprocessing not in PREPROCESSINGS:
        raise ValueError(f"unknown preprocessing {preprocessing}")
    D = _csr_counts(counts)
    train = np.asarray(training_mask, dtype=bool)
    if train.shape != (D.shape[0],) or not train.any():
        raise ValueError("training_mask must identify at least one training row")
    candidates = np.arange(D.shape[1]) if candidate_gene_indices is None else np.asarray(candidate_gene_indices, dtype=int)
    if candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate gene indices must be unique")
    candidate_counts = D[train][:, candidates]
    nonzero_training = np.asarray(candidate_counts.sum(axis=1)).ravel() > 0
    candidate_counts = candidate_counts[nonzero_training]
    n_train = candidate_counts.shape[0]
    if n_train == 0:
        raise ValueError("no nonzero allowed training rows")
    minimum_detection = max(1, int(config.detection_fraction * n_train))
    detection = np.asarray((candidate_counts > 0).sum(axis=0)).ravel()
    mean = np.asarray(candidate_counts.mean(axis=0)).ravel()
    second = np.asarray(candidate_counts.power(2).mean(axis=0)).ravel()
    # ddof=1 is the documented variance; its common factor does not alter rank.
    variance = np.maximum(second - mean**2, 0.) * n_train / max(n_train - 1, 1)
    vtm = np.divide(variance, mean, out=np.zeros_like(mean), where=mean > 0)
    if panel_indices is not None:
        panel = np.asarray(panel_indices, dtype=int)
        if len(np.unique(panel)) != len(panel) or not np.isin(panel, candidates).all():
            raise ValueError("frozen panel indices must be unique permitted source columns")
        ranking = np.argsort(-vtm, kind="stable")
    elif requested_panel is None:
        panel = candidates.copy()
        ranking = np.argsort(-vtm, kind="stable")
    else:
        if requested_panel <= 0:
            raise ValueError("requested_panel must be positive")
        eligible = np.flatnonzero(detection >= minimum_detection)
        ranking = eligible[np.argsort(-vtm[eligible], kind="stable")]
        panel = np.sort(candidates[ranking[:requested_panel]])
    if not len(panel):
        raise ValueError("no genes satisfy the training-relative detection rule")
    X, totals = row_frequencies(D[:, panel])
    eligible_train = train & (totals > 0)
    if not eligible_train.any():
        raise ValueError("no positive-count training rows in selected panel")
    eta_full = np.asarray(X[eligible_train].mean(axis=0)).ravel()
    positive = np.flatnonzero(eta_full > 0)
    # The audited baseline mechanically removes zero-frequency columns before
    # applying Tran's p-dependent rule; retain that convention explicitly.
    eta_positive = eta_full[positive]
    n = int(eligible_train.sum())
    Nbar = float(totals[eligible_train].mean())
    threshold = 0.0
    fallback = False
    retained = np.arange(len(positive))
    if preprocessing in ("P1_tran_alpha_0p005", "P3_tran_then_ke"):
        threshold = float(config.tran_alpha * np.sqrt(np.log(max(len(positive), n)) / (n * Nbar)))
        retained = np.flatnonzero(eta_positive > threshold)
        if len(retained) < 0.1 * len(positive):
            retained = np.argsort(-eta_positive, kind="stable")[:int(np.ceil(0.1 * len(positive)))]
            fallback = True
    local_retained = positive[retained]
    eta = eta_full[local_retained]
    weights = 1. / np.sqrt(eta) if preprocessing in ("P2_ke_weighted", "P3_tran_then_ke") else np.ones_like(eta)
    metadata = {
        "preprocessing": preprocessing, "requested_panel": requested_panel,
        "actual_panel": int(len(panel)), "training_rows": n_train,
        "positive_panel_training_rows": n, "zero_panel_training_rows": int((train & (totals == 0)).sum()),
        "minimum_detection": minimum_detection, "detection_fraction": config.detection_fraction,
        "panel_selection": "targeted_source_panel" if requested_panel is None else "training_raw_variance_to_mean_stable_ties",
        "frozen_outer_panel_indices": panel_indices is not None,
        "zero_frequency_genes": int(len(panel) - len(positive)),
        "positive_vocabulary_before_tran": int(len(positive)), "spectral_features": int(len(eta)),
        "tran_threshold": threshold, "tran_alpha": config.tran_alpha,
        "tran_strict_inequality": True, "tran_top10percent_fallback": fallback,
        "N_mean": Nbar, "row_renormalized_after_tran": False,
        "weight_floor": None, "weight_cap": None,
        "initialization_frequency_scope": "fold_training_rows_only",
    }
    return FeaturePlan(panel, panel[local_retained], weights, eta, totals, metadata)


def transformed_frequencies(counts, plan):
    D = sparse.csr_matrix(counts, dtype=np.float64)
    inv = np.divide(1., plan.row_totals, out=np.zeros_like(plan.row_totals), where=plan.row_totals > 0)
    return (sparse.diags(inv) @ D[:, plan.retained_indices] @ sparse.diags(plan.weights)).tocsr()


def validate_adjacency(adjacency, n):
    G = sparse.csr_matrix(adjacency, dtype=np.float64)
    if G.shape != (n, n) or not np.isfinite(G.data).all() or np.any(G.data < 0):
        raise ValueError("invalid sparse graph")
    if np.any(G.diagonal() != 0):
        raise ValueError("graph must exclude self edges")
    delta = G - G.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-12:
        raise ValueError("graph must already be symmetric; no implicit rescaling")
    return G


def graph_fold_ids(adjacency, seed, nfolds=5):
    """Coordinate-graph-only spanning-tree depth modulo five.

    scipy's deterministic unweighted spanning forest replaces NetworkX's much
    larger graph objects. The seeded root rule is preserved; tied MST edges can
    differ and are recorded as a versioned fold construction.
    """
    if nfolds != 5:
        raise ValueError("joint_v2 requires exactly five graph folds")
    G = sparse.csr_matrix(adjacency)
    binary = G.copy()
    binary.data[:] = 1.
    n_components, components = connected_components(binary, directed=False)
    forest = minimum_spanning_tree(binary).tocsr()
    forest = (forest + forest.T).tocsr()
    rng = np.random.default_rng(seed)
    folds = np.full(G.shape[0], -1, dtype=np.int8)
    for component in range(n_components):
        nodes = np.flatnonzero(components == component)
        root = int(rng.choice(nodes))
        order, predecessors = breadth_first_order(forest, root, directed=False)
        depth = np.zeros(G.shape[0], dtype=np.int32)
        for node in order:
            parent = predecessors[node]
            if parent >= 0:
                depth[node] = depth[parent] + 1
            folds[node] = depth[node] % nfolds
    if np.any(folds < 0) or len(np.unique(folds)) != 5:
        raise ValueError("coordinate graph does not yield five nonempty folds")
    return folds


def interpolation_operator(adjacency, permitted_mask):
    """Identity on permitted rows, unweighted neighbor means on masked rows.

    A masked row without an observed neighbor is an explicit fold failure. No
    validation expression or fallback expression-neighbor connection is used.
    """
    G = sparse.csr_matrix(adjacency)
    allowed = np.asarray(permitted_mask, dtype=bool)
    observed = np.flatnonzero(allowed)
    masked = np.flatnonzero(~allowed)
    neighbors = G[masked].multiply(allowed[None, :]).tocsr()
    neighbors.eliminate_zeros()
    neighbors.data[:] = 1.
    degree = np.asarray(neighbors.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError(f"{int((degree == 0).sum())} masked graph rows have no permitted immediate neighbor")
    neighbors = sparse.diags(1. / degree) @ neighbors
    coo = neighbors.tocoo()
    return sparse.csr_matrix((np.concatenate([np.ones(len(observed)), coo.data]),
                              (np.concatenate([observed, masked[coo.row]]),
                               np.concatenate([observed, coo.col]))), shape=G.shape)


def _masked_operator(X, interpolation):
    def matmat(V):
        return interpolation @ (X @ V)
    def rmatmat(U):
        return X.T @ (interpolation.T @ U)
    return LinearOperator(X.shape, matvec=lambda v: matmat(v), rmatvec=lambda u: rmatmat(u),
                          matmat=matmat, rmatmat=rmatmat, dtype=np.float64)


def _partial_svd(operator, K, seed, tolerance):
    op = aslinearoperator(operator)
    if K > min(op.shape):
        raise ValueError(f"rank {K} exceeds operator shape {op.shape}")
    if K == min(op.shape):
        # Here the dense side is exactly K, hence this is an allowed n-by-K or
        # p-by-K factor, not a full large count/prediction matrix.
        if op.shape[1] == K:
            U, s, Vt = np.linalg.svd(op @ np.eye(K), full_matrices=False)
        else:
            V, s, Ut = np.linalg.svd(op.T @ np.eye(K), full_matrices=False)
            U, Vt = Ut.T, V.T
    else:
        U, s, Vt = svds(op, k=K, tol=tolerance, random_state=np.random.default_rng(seed))
        order = np.argsort(s)[::-1]
        U, s, Vt = U[:, order], s[order], Vt[order]
    # Deterministic sign from largest absolute entry in each right vector.
    pivot = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(K), pivot]); signs[signs == 0] = 1.
    return U * signs, s, Vt.T * signs


def initialize_factors(operator, K, seed, config, correction):
    op = aslinearoperator(operator)
    U_direct, s_direct, V_direct = _partial_svd(op, K, seed, config.svd_tolerance)
    if config.initialization == "direct_svd":
        return U_direct, s_direct, V_direct
    if config.initialization != "current":
        raise ValueError("joint_v2 implements audited current and direct_svd initializations only")
    correction = np.asarray(correction, dtype=np.float64)
    def cov_matmat(V):
        if V.ndim == 1:
            return op.T @ (op @ V) - correction * V
        return op.T @ (op @ V) - correction[:, None] * V
    cov = LinearOperator((op.shape[1], op.shape[1]), matvec=cov_matmat,
                         rmatvec=cov_matmat, matmat=cov_matmat, rmatmat=cov_matmat,
                         dtype=np.float64)
    _, _, V = _partial_svd(cov, K, seed + 1, config.svd_tolerance)
    return U_direct, s_direct, V


def graph_penalty(factors, adjacency, chunk_size=50000):
    edges = sparse.triu(adjacency, k=1).tocoo()
    total = 0.
    for start in range(0, edges.nnz, chunk_size):
        stop = min(start + chunk_size, edges.nnz)
        difference = factors[edges.row[start:stop]] - factors[edges.col[start:stop]]
        total += float(np.dot(edges.data[start:stop], np.linalg.norm(difference, axis=1)))
    return total


def graph_roughness(factors, adjacency, chunk_size=50000):
    edges = sparse.triu(adjacency, k=1).tocoo()
    total = 0.
    for start in range(0, edges.nnz, chunk_size):
        stop = min(start + chunk_size, edges.nnz)
        difference = factors[edges.row[start:stop]] - factors[edges.col[start:stop]]
        total += float(np.dot(edges.data[start:stop], np.einsum("ij,ij->i", difference, difference)))
    return total / float(edges.data.sum()) if edges.nnz and edges.data.sum() > 0 else float("nan")


def smooth_fixed_input(Y, adjacency, lambda_value, config=None):
    config = config or SpectralConfig()
    if lambda_value < 0 or not np.isfinite(lambda_value):
        raise ValueError("lambda must be finite and nonnegative")
    if lambda_value == 0 or adjacency.nnz == 0:
        return Y.copy(), {"converged": True, "iterations": 0, "certificate": 0.,
                          "termination": "exact_zero_identity" if lambda_value == 0 else "empty_graph_identity",
                          "objective": 0., "graph_penalty": graph_penalty(Y, adjacency)}
    from pycvxcluster.pycvxcluster import SSNAL
    solver = SSNAL(gamma=float(lambda_value), maxiter=config.ssnal_max_iterations,
                   admm_iter=config.ssnal_admm_iterations, stoptol=config.ssnal_tolerance, verbose=0)
    solver.fit(X=Y, weight_matrix=adjacency, save_centers=True, save_labels=False)
    centers = np.asarray(solver.centers_.T, dtype=np.float64)
    eta = float(solver.eta_) if solver.eta_ is not None else float("inf")
    successful = bool(solver.termination_ == 1 and np.isfinite(eta) and eta <= config.ssnal_tolerance)
    return centers, {"converged": successful, "iterations": int(solver.iter_),
                     "certificate": eta, "termination": int(solver.termination_),
                     "objective": float(solver.primobj_), "dual_objective": float(solver.dualobj_),
                     "graph_penalty": graph_penalty(centers, adjacency)}


def lowrank_difference(U, s, V, U_previous, s_previous, V_previous):
    """Frobenius change without constructing either n-by-p reconstruction."""
    core = (U.T @ U_previous) * (s[:, None] * s_previous[None, :])
    cross = float(np.sum(core * (V.T @ V_previous)))
    squared = max(0., float(s @ s + s_previous @ s_previous - 2 * cross))
    return np.sqrt(squared)


def fit_fixed_lambda(operator, adjacency, K, lambda_value, *, seed, config=None,
                     correction=None, smoother=None):
    config = config or SpectralConfig()
    op = aslinearoperator(operator)
    if K < 1 or K > min(op.shape):
        raise ValueError("insufficient rows or features for requested K")
    if correction is None:
        correction = np.zeros(op.shape[1])
    U, _, V = initialize_factors(op, K, seed, config, correction)
    # Canonical aligned low-rank initial reconstruction, preserving current V's
    # span and the source's direct-SVD U initialization.
    core = U.T @ (op @ V)
    left, s, right = np.linalg.svd(core, full_matrices=False)
    U_previous, V_previous = U @ left, V @ right.T
    s_previous = s
    history = []
    stable = 0
    converged = False
    smoother = smoother or smooth_fixed_input
    U_bar = U.copy()
    initial_roughness = graph_roughness(U, adjacency)
    started = perf_counter()
    for iteration in range(1, config.max_iterations + 1):
        projected = op @ V
        U_bar, solve = smoother(projected, adjacency, lambda_value, config)
        if not np.isfinite(U_bar).all():
            raise FloatingPointError("nonfinite graph-smoothed embedding")
        Q, _, _ = np.linalg.svd(U_bar, full_matrices=False)
        V_new, s_new, rotation = np.linalg.svd(op.T @ Q, full_matrices=False)
        # Source drops this K-by-K rotation. Keeping it aligns U,s,V and makes
        # anchor/Topic-SCORE reconstruction identities exact; spans are equal.
        U_new = Q @ rotation.T
        change = lowrank_difference(U_new, s_new, V_new, U_previous, s_previous, V_previous)
        relative = change / max(float(np.linalg.norm(s_previous)), np.finfo(float).eps)
        history.append({"iteration": iteration, "relative_reconstruction_change": relative,
                        "smoother": solve})
        U, V, s = U_new, V_new, s_new
        if relative <= config.reconstruction_tolerance and solve["converged"]:
            stable += 1
        else:
            stable = 0
        U_previous, V_previous, s_previous = U, V, s
        if stable >= config.stable_iterations:
            converged = True
            break
    if not history:
        raise ValueError("max_iterations must be positive")
    metadata = {"history": history, "runtime_seconds": perf_counter() - started,
                "unregularized_initial_embedding_roughness": initial_roughness,
                "regularized_embedding_roughness": graph_roughness(U, adjacency),
                "unorthogonalized_embedding_roughness": graph_roughness(U_bar, adjacency),
                "stopping_rule": "full_lowrank_relative_Frobenius_change_with_consecutive_stability",
                "all_inner_smoothers_converged": all(row["smoother"]["converged"] for row in history),
                "aligned_factor_rotation_correction": True,
                "convergence_status": "converged" if converged else "maximum_iterations_or_smoother_failure"}
    return Embedding(U, V, s, U_bar, float(lambda_value), converged, iteration, metadata)


def _aggregate_candidates(records, grid):
    aggregate = []
    for lam in grid:
        folds = records[float(lam)]
        complete = len(folds) == 5 and {row["fold"] for row in folds} == set(range(5))
        valid = complete and all(row.get("converged", False) and np.isfinite(row.get("score", np.inf)) for row in folds)
        aggregate.append({"lambda": float(lam), "complete": bool(complete), "selectable": bool(valid),
                          "score": float(sum(row["score"] for row in folds)) if valid else None,
                          "fold_scores": [row.get("score") for row in sorted(folds, key=lambda row: row["fold"])]})
    return aggregate


def tune_lambda(evaluate_batch: Callable, config=None):
    """Evaluate all five folds; expand unresolved upper endpoints deterministically."""
    config = config or SpectralConfig()
    if config.nfolds != 5 or config.grid_growth <= 1 or config.extension_batch_size < 1:
        raise ValueError("invalid all-five-fold/grid-expansion configuration")
    grid = sorted(set(float(x) for x in config.initial_grid))
    if not grid or grid[0] != 0 or any(x < 0 or not np.isfinite(x) for x in grid):
        raise ValueError("initial lambda grid must include exact zero and finite nonnegative candidates")
    if len(grid) > config.max_candidates:
        raise ValueError("candidate budget smaller than frozen initial grid")
    records = evaluate_batch(grid)
    expansion = []
    boundary_unresolved = False
    plateau = False
    while True:
        aggregate = _aggregate_candidates(records, grid)
        eligible = [item for item in aggregate if item["selectable"]]
        if not eligible:
            return {"selected_lambda": None, "status": "no_complete_converged_candidate",
                    "grid": grid, "fold_curves": records, "aggregate": aggregate,
                    "expansion_history": expansion, "boundary_unresolved": True}
        best = min(eligible, key=lambda row: (row["score"], row["lambda"]))
        positive = [row for row in aggregate if row["lambda"] > 0]
        at_upper = bool(positive and best["lambda"] == positive[-1]["lambda"])
        last = positive[-config.plateau_points:]
        plateau = len(last) == config.plateau_points and all(row["selectable"] for row in last)
        if plateau:
            losses = np.array([row["score"] for row in last])
            plateau = bool(np.ptp(losses) <= config.plateau_relative_tolerance * max(np.max(np.abs(losses)), np.finfo(float).eps))
        if not at_upper or plateau:
            break
        extras = []
        value = grid[-1]
        for _ in range(config.extension_batch_size):
            value *= config.grid_growth
            if value > config.lambda_ceiling or len(grid) + len(extras) >= config.max_candidates:
                break
            extras.append(value)
        if not extras:
            boundary_unresolved = True
            break
        expansion.append({"previous_upper": grid[-1], "trigger_best_score": best["score"], "appended": extras})
        records.update(evaluate_batch(extras))
        grid.extend(extras)
    return {"selected_lambda": best["lambda"], "status": "ok", "grid": grid,
            "fold_curves": records, "aggregate": aggregate, "expansion_history": expansion,
            "boundary_unresolved": boundary_unresolved, "upper_plateau": plateau,
            "selected_zero": best["lambda"] == 0, "tie_rule": "minimum_loss_then_smallest_lambda",
            "candidate_incomplete_count": sum(not row["selectable"] for row in aggregate),
            "loss": "sum_f ||X_validation,f V_f - C_validation,f||_F / n_validation,f",
            "fold_weights": [1., 1., 1., 1., 1.], "scoring_folds": [0, 1, 2, 3, 4]}


def fit_shared_spectral(counts, adjacency, K, preprocessing, *, graph_cv_seed,
                        estimator_seed, requested_panel=None, candidate_gene_indices=None,
                        final_panel_indices=None, config=None, fold_ids=None, smoother=None):
    """Shared spectral stage: return selected and matched exact-zero embeddings.

    Counts must contain outer TRAINING molecules only (no D_score, no outer test
    rows). Raw candidate genes must be retained here for fold-specific HVGs.
    The caller persists row IDs and the canonical gene IDs with returned arrays.
    """
    started = perf_counter()
    config = config or SpectralConfig()
    D = _csr_counts(counts)
    G = validate_adjacency(adjacency, D.shape[0])
    folds = graph_fold_ids(G, graph_cv_seed, config.nfolds) if fold_ids is None else np.asarray(fold_ids)
    if folds.shape != (D.shape[0],) or set(np.unique(folds)) != set(range(5)):
        raise ValueError("exactly five nonempty graph folds required")
    fold_plans = {}
    def evaluate_batch(grid):
        output = {float(lam): [] for lam in grid}
        for fold in range(5):
            train = folds != fold
            try:
                plan = learn_feature_plan(D, train, preprocessing, requested_panel=requested_panel,
                                          candidate_gene_indices=candidate_gene_indices, config=config)
                if len(plan.retained_indices) < K:
                    raise ValueError(f"fold has {len(plan.retained_indices)} spectral columns for K={K}")
                X = transformed_frequencies(D, plan)
                allowed = train & (plan.row_totals > 0)
                interpolation = interpolation_operator(G, allowed)
                operator = _masked_operator(X, interpolation)
                validation = (~train) & (plan.row_totals > 0)
                if not validation.any():
                    raise ValueError("fold has no count-eligible validation rows")
                # The correction uses only measured fold training frequencies.
                correction = np.asarray(X[allowed].sum(axis=0)).ravel() / plan.metadata["N_mean"]
                fold_plans[fold] = {**plan.metadata, "panel_indices": plan.panel_indices.tolist(),
                                    "retained_indices": plan.retained_indices.tolist(),
                                    "validation_rows": int(validation.sum()),
                                    "zero_count_validation_rows": int(((~train) & ~validation).sum())}
            except Exception as exc:
                for lam in grid:
                    output[float(lam)].append({"fold": fold, "score": None, "converged": False,
                                               "status": "fold_preparation_failed", "error": str(exc)})
                continue
            for lam in grid:
                record = {"fold": fold, "score": None, "converged": False}
                for attempt in range(config.fold_retries + 1):
                    try:
                        fit = fit_fixed_lambda(operator, G, K, lam, seed=estimator_seed + 1009 * fold,
                                               config=config, correction=correction, smoother=smoother)
                        # U_bar corresponds to the final smoother input V. After
                        # the last right-factor update, refit its same convex
                        # subproblem for an internally consistent validation loss.
                        fitted_centers, final_solve = (smoother or smooth_fixed_input)(operator @ fit.V, G, lam, config)
                        residual = X[validation] @ fit.V - fitted_centers[validation]
                        score = float(np.linalg.norm(residual) / validation.sum())
                        record.update(score=score, converged=fit.converged and final_solve["converged"],
                                      status="ok" if fit.converged and final_solve["converged"] else "nonconverged",
                                      iterations=fit.iterations, metadata=fit.metadata,
                                      final_smoother=final_solve, attempt=attempt)
                        if record["converged"]:
                            break
                    except Exception as exc:
                        record.update(status="solver_failed", error=str(exc), attempt=attempt)
                output[float(lam)].append(record)
        return output
    cv = tune_lambda(evaluate_batch, config)
    plan = learn_feature_plan(D, np.ones(D.shape[0], dtype=bool), preprocessing,
                              requested_panel=requested_panel, candidate_gene_indices=candidate_gene_indices,
                              panel_indices=final_panel_indices, config=config)
    if np.any(plan.row_totals == 0):
        raise ValueError("outer native-panel zero-count rows must be excluded by the frozen count eligibility mask")
    X = transformed_frequencies(D, plan)
    correction = np.asarray(X.sum(axis=0)).ravel() / plan.metadata["N_mean"]
    zero = fit_fixed_lambda(X, G, K, 0., seed=estimator_seed, config=config, correction=correction, smoother=smoother)
    if cv["selected_lambda"] is None:
        selected = None
    elif cv["selected_lambda"] == 0:
        selected = zero
    else:
        try:
            selected = fit_fixed_lambda(X, G, K, cv["selected_lambda"], seed=estimator_seed,
                                        config=config, correction=correction, smoother=smoother)
        except Exception as exc:
            selected = None
            cv["refit_status"] = "failed"
            cv["refit_error"] = str(exc)
    edges = sparse.triu(G, k=1).data
    metadata = {"version": "joint_v2_fold_fitted_cv_aligned_sparse_spectral",
                "configuration": asdict(config), "K": K, "preprocessing": preprocessing,
                "graph_cv_seed": graph_cv_seed, "estimator_seed": estimator_seed,
                "cv": cv, "fold_preprocessing": fold_plans,
                "graph": {"n_observations": D.shape[0], "undirected_edges": int(len(edges)),
                          "weight_sum": float(edges.sum()), "weight_mean": float(edges.mean()) if len(edges) else None,
                          "weight_min": float(edges.min()) if len(edges) else None,
                          "weight_max": float(edges.max()) if len(edges) else None},
                "objective": "0.5*sum((C-XV)^2) + lambda*sum_undirected_edges(weight*L2(C_i-C_j))",
                "penalty_rescaling": "none", "fold_construction": "scipy_unweighted_MST_depth_mod5_seeded_roots",
                "runtime_seconds": perf_counter() - started,
                "selected_converged": selected.converged if selected is not None else False,
                "selection_status": "refit_failed" if cv.get("refit_status") == "failed" else cv["status"],
                "zero_converged": zero.converged, "training_only": True}
    return SpectralResult(selected, zero, plan, folds, metadata)


def spectral_arrays(result):
    arrays = {"panel_indices": result.features.panel_indices,
              "retained_indices": result.features.retained_indices, "feature_weights": result.features.weights,
              "eta": result.features.eta, "row_totals": result.features.row_totals,
              "fold_ids": result.fold_ids}
    for name in ("selected", "zero"):
        block = getattr(result, name)
        if block is None:
            continue
        arrays.update({f"{name}_U": block.U, f"{name}_V": block.V,
                       f"{name}_singular_values": block.singular_values, f"{name}_U_bar": block.U_bar})
    return arrays
