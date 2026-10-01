from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd
from scipy.sparse.linalg import svds

from .utils import (
    get_folds_disconnected_G, interpolate_X
)

from multiprocessing import Pool
import pycvxcluster.pycvxcluster

def graphSVD(
    X: np.ndarray,
    N: float,
    K: int,
    edge_df,
    weights,
    lamb_start: float,
    step_size: float,
    grid_len: int,
    maxiter: int,
    eps: float,
    verbose: int,
    initialize: bool,
    initialization: str = "current",
    debias_correction: np.ndarray | None = None,
    return_metadata: bool = False,
    random_state: int | None = None,
    nfolds: int = 5,
    cv_fold_mode: str = "legacy_first_three",
    n_jobs: int = 3,
    lambda_selection_mode: str = "cv_each_iteration",
):
    """
    Graph-aligned SVD (graphSVD) for GpLSI.

    This routine computes a graph-regularized low-rank representation of X:

        X ≈ U L V^T

    where the left singular vectors U are encouraged to be smooth over a graph
    defined on the samples (rows of X).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Row-normalized data matrix (e.g., document-term or cell-by-gene).
        Each row corresponds to a node in the graph. In your use cases:
        - DLPFC / spleen / CRC: rows = spots/cells
        - WhatsCooking: rows = cuisines or region-level samples

    N : float
        Average document length (mean row sum of the original count matrix D).
        Passed for API consistency with other pieces; not used directly here,
        but kept in case you extend graphSVD to use it later.

    K : int
        Target rank / number of latent components (topics).

    edge_df : pandas.DataFrame
        Graph edge list over the n_samples nodes, with **integer** indices.

        Required columns
        ----------------
        - "src": int
            Source node index in [0, n_samples - 1].
        - "tgt": int
            Target node index in [0, n_samples - 1].
        - "weight": float
            Edge weight (e.g. exp(-distance / phi), or 1.0 for unweighted).

        Notes
        -----
        - The graph is treated as undirected; you only need to store each edge once.
        - All node indices in "src" and "tgt" must line up with the row indices
          of X. That is, if X has shape (n_samples, p), valid indices are
          0, 1, ..., n_samples - 1.

    weights : scipy.sparse.spmatrix, shape (n_samples, n_samples)
        Sparse adjacency/weight matrix built from edge_df, typically:

        .. code-block:: python

            weights = csr_matrix(
                (edge_df["weight"].values,
                 (edge_df["src"].values, edge_df["tgt"].values)),
                shape=(n_samples, n_samples),
            )

        This is used internally (via functions in utils) to construct graph
        Laplacians and to perform graph-regularized updates of U.

    lamb_start : float
        Starting value for the lambda grid (graph regularization strength).

    step_size : float
        Multiplicative step between successive lambda values on the grid.
        The grid is roughly:

            lamb_start * step_size**j  for  j = 0, 1, ..., grid_len-1

        plus a tiny value 1e-6 inserted at the beginning.

    grid_len : int
        Number of lambda values in the main grid.

    maxiter : int
        Maximum number of outer iterations of the alternating procedure:

            U <- update_U_tilde(...)
            V, L <- update_V_L_tilde(...)

    eps : float
        Convergence tolerance. The algorithm stops when the relative change
        in reconstructed X_hat on a subsample of rows is below eps, or
        when maxiter iterations are reached.

    verbose : int
        If 1, print progress and errors at each iteration.
        If 0, run quietly.

    initialize : bool
        If True, run an SVD-based initialization for U, L, V using a covariance
        approximation and a separate SVD of X for U_init.
        If False, just run truncated SVD on X for all factors.

    lambda_selection_mode : {"cv_each_iteration", "cv_once"}
        ``"cv_each_iteration"`` preserves the historical behavior: rerun graph
        cross-validation after every update of the right singular subspace.
        ``"cv_once"`` runs the same cross-validation exactly once, using the
        initialized right singular subspace, and holds the selected lambda fixed
        for every alternating update in this graph-SVD fit.

    Returns
    -------
    U : np.ndarray, shape (n_samples, K)
        Final left singular vectors (graph-regularized).

    V : np.ndarray, shape (n_features, K)
        Final right singular vectors.

    L : np.ndarray, shape (K, K)
        Diagonal matrix of singular values.

    U_init : np.ndarray or None
        Initial left singular vectors used for warm starting (if initialize=True),
        otherwise None.

    V_init : np.ndarray or None
        Initial right singular vectors used for warm starting (if initialize=True),
        otherwise None.

    L_init : np.ndarray or None
        Initial diagonal matrix of singular values used for warm starting
        (if initialize=True), otherwise None.

    lambd : float
        Selected regularization parameter (lambda) chosen during update_U_tilde
        (e.g. by cross-validation over folds).

    lambd_errs : dict
        Fold-level and summed cross-validation errors for each lambda in the
        grid. In ``"cv_once"`` mode these are the sole selection diagnostics;
        in historical mode they are from the final outer iteration.

    niter : int
        Number of outer iterations performed.

    Notes
    -----
    - Internally, we:
        1. Build a disconnected graph structure using get_folds_disconnected_G(edge_df).
        2. Construct a grid of lambda values (lambd_grid).
        3. Optionally do an SVD-based initialization (initialize=True).
        4. Alternate between:
            - update_U_tilde (graph-regularized U, optionally with CV over lambda)
            - update_V_L_tilde (update V, L given U)
      until the reconstruction stabilizes.

    - Convergence criterion uses a random subsample of up to 1000 rows to
      measure the change in P_U X P_V, where P_U and P_V are projection matrices.
    """
    valid_lambda_selection_modes = {"cv_each_iteration", "cv_once"}
    if lambda_selection_mode not in valid_lambda_selection_modes:
        raise ValueError(
            f"unknown lambda_selection_mode={lambda_selection_mode!r}; expected one of "
            f"{sorted(valid_lambda_selection_modes)}"
        )
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")

    n = X.shape[0]
    rng = None if random_state is None else np.random.default_rng(random_state)
    _, folds, G, _ = get_folds_disconnected_G(edge_df, nfolds=nfolds, rng=rng)
    nonempty_folds = {key: value for key, value in folds.items() if value}
    if len(nonempty_folds) < 2:
        raise ValueError(
            "graph cross-validation requires at least two nonempty folds; "
            f"observed {len(nonempty_folds)}"
        )

    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)

    lambd_grid_init = (0.0001 * np.power(1.5, np.arange(10))).tolist()
    lambd_grid_init.insert(0, 1e-06)

    if initialize and initialization == "current":
        print('Initializing...')
        colsums = np.sum(X, axis=0)
        cov = X.T @ X - np.diag(colsums/N)
        U, L, V = _svds(cov, K, rng)
        V  = V.T
        L = np.diag(L)
        V_init = V
        L_init = L
        U, _, _ = _svds(X, K, rng)
        U_init = U
    elif initialize and initialization in {
        "weighted_debiased",
        "weighted_debiased_mean_N_approx",
    }:
        print('Initializing with weighted diagonal debiasing...')
        if debias_correction is None:
            raise ValueError(
                f"initialization={initialization!r} requires debias_correction"
            )
        correction = np.asarray(debias_correction, dtype=float)
        if correction.shape != (X.shape[1],):
            raise ValueError(
                "debias_correction must have one entry per transformed feature"
            )
        cov = X.T @ X - np.diag(correction)
        U, L, V = _svds(cov, K, rng)
        V = V.T
        L = np.diag(L)
        V_init = V
        L_init = L
        U, _, _ = _svds(X, K, rng)
        U_init = U
    elif initialization == "direct_svd" or not initialize:
        U, L, V = _svds(X, K, rng)
        V  = V.T
        L = np.diag(L)
        U_init = None
        V_init = None
        L_init = None
    else:
        raise ValueError(f"unknown initialization mode: {initialization!r}")

    score = float("inf")
    niter = 0
    score_history = []
    lambd_history = []
    cv_history = []
    U_bar_history = []
    U_hat_history = []
    V_hat_history = []
    singular_value_history = []

    # Select before the first alternating update so CV is based on the same
    # initialized V that the historical implementation uses on iteration zero.
    # Keeping selection separate from fitting makes it impossible for later V
    # updates to silently trigger another CV pass in ``cv_once`` mode.
    if lambda_selection_mode == "cv_once":
        lambd, lambd_errs = select_lambda_by_cv(
            X,
            V,
            L,
            G,
            weights,
            nonempty_folds,
            lambd_grid,
            cv_fold_mode=cv_fold_mode,
            n_jobs=n_jobs,
        )
        cv_history.append(lambd_errs)
        print(f"Optimal lambda is {lambd}...")

    while score > eps and niter < maxiter:
        if n > 1000:
            idx = (
                np.random.choice(range(n), 1000, replace=False)
                if rng is None
                else rng.choice(n, 1000, replace=False)
            )
        else:
            idx = range(n)
        
        U_samp = U[idx,:]
        P_U_old = np.dot(U_samp, U_samp.T)
        P_V_old = np.dot(V, V.T)
        X_hat_old = (P_U_old @ X[idx,:]) @ P_V_old
        if lambda_selection_mode == "cv_once":
            if return_metadata:
                U, U_bar = update_U_tilde_fixed_lambda(
                    X,
                    V,
                    weights,
                    lambd,
                    return_unorthogonalized=True,
                )
                U_bar_history.append(U_bar.copy())
            else:
                U = update_U_tilde_fixed_lambda(X, V, weights, lambd)
        else:
            if return_metadata:
                U, lambd, lambd_errs, U_bar = update_U_tilde(
                    X, V, L, G, weights, nonempty_folds, lambd_grid,
                    return_unorthogonalized=True,
                    cv_fold_mode=cv_fold_mode,
                    n_jobs=n_jobs,
                )
                U_bar_history.append(U_bar.copy())
            else:
                U, lambd, lambd_errs = update_U_tilde(
                    X, V, L, G, weights, nonempty_folds, lambd_grid,
                    cv_fold_mode=cv_fold_mode,
                    n_jobs=n_jobs,
                )
            cv_history.append(lambd_errs)
        V, L = update_V_L_tilde(X, U)
        if return_metadata:
            U_hat_history.append(U.copy())
            V_hat_history.append(V.copy())
            singular_value_history.append(np.diag(L).copy())

        P_U = np.dot(U[idx,:], U[idx,:].T)
        P_V = np.dot(V, V.T)
        X_hat = (P_U @ X[idx,:]) @ P_V
        score = norm(X_hat-X_hat_old)/n
        score_history.append(float(score))
        lambd_history.append(float(lambd))
        niter += 1
        if verbose == 1:
            print(f"Error is {score}")
    
    print(f"Graph-aligned SVD ran for {niter} steps.")

    output = (U, V, L, U_init, V_init, L_init, lambd, lambd_errs, niter)
    if not return_metadata:
        return output
    metadata = {
        "U_bar_history": U_bar_history,
        "U_bar": U_bar_history[-1] if U_bar_history else U.copy(),
        "U_hat_history": U_hat_history,
        "V_hat_history": V_hat_history,
        "singular_value_history": singular_value_history,
        "score_history": score_history,
        "lambd_history": lambd_history,
        "cv_history": cv_history,
        "lambd_grid": lambd_grid,
        "initialization": initialization,
        "random_state": random_state,
        "nfolds_requested": int(nfolds),
        "nfolds_nonempty": int(len(nonempty_folds)),
        "cv_fold_mode": cv_fold_mode,
        "n_jobs": int(n_jobs),
        "lambda_selection_mode": lambda_selection_mode,
        "lambda_selection_basis": (
            "initial_V_before_first_alternating_update"
            if lambda_selection_mode == "cv_once"
            else "current_V_at_each_alternating_update"
        ),
        "lambda_cv_evaluations": int(len(cv_history)),
        "lambda_fixed_across_iterations": lambda_selection_mode == "cv_once",
        "debias_correction": None
        if debias_correction is None
        else np.asarray(debias_correction, dtype=float),
        "objective_convention": (
            "0.5*||U-XV||_F^2 + gamma*sum_e(weight_e*||(Gamma U)_e||_2); "
            "GpLSI passes lambda-grid values directly as pycvxcluster gamma"
        ),
    }
    return output + (metadata,)


def _svds(matrix, K, rng):
    """Keep the historical ARPACK call unless an audited seed is supplied."""

    if rng is None:
        return svds(matrix, k=K)
    try:
        # SciPy >=1.15 uses the SPEC-007 ``rng`` keyword.
        return svds(matrix, k=K, rng=rng)
    except TypeError as error:
        if "unexpected keyword argument 'rng'" not in str(error):
            raise
        # SciPy <=1.14 exposes the same deterministic ARPACK seed through
        # ``random_state``. Keep this compatibility path explicit so runtime
        # provenance records which SciPy implementation was used.
        return svds(matrix, k=K, random_state=rng)


def lambda_search(j, folds, X, V, L, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    X_tildeV = X_tilde @ V
    X_j = X[fold, :] @ V
  
    errs = []
    best_err = float("inf")
    U_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=X_tildeV,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        U_tilde = ssnal.centers_.T
        E = U_tilde
        err = norm(X_j - E[fold, :])/len(fold)
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            U_best = U_tilde
            best_err = err
    return j, errs, U_best, lambd_best


def update_U_tilde(
    X,
    V,
    L,
    G,
    weights,
    folds,
    lambd_grid,
    return_unorthogonalized=False,
    cv_fold_mode="legacy_first_three",
    n_jobs=3,
):
    lambd_cv, lambd_errs = select_lambda_by_cv(
        X,
        V,
        L,
        G,
        weights,
        folds,
        lambd_grid,
        cv_fold_mode=cv_fold_mode,
        n_jobs=n_jobs,
    )
    fitted = update_U_tilde_fixed_lambda(
        X,
        V,
        weights,
        lambd_cv,
        return_unorthogonalized=return_unorthogonalized,
    )
    print(f"Optimal lambda is {lambd_cv}...")
    if return_unorthogonalized:
        U_hat, U_tilde = fitted
        return U_hat, lambd_cv, lambd_errs, U_tilde
    return fitted, lambd_cv, lambd_errs


def select_lambda_by_cv(
    X,
    V,
    L,
    G,
    weights,
    folds,
    lambd_grid,
    cv_fold_mode="legacy_first_three",
    n_jobs=3,
):
    """Select one graph penalty from held-out-node reconstruction error."""

    lambd_errs = {"fold_errors": {}, "final_errors": []}

    tasks = [(j, folds, X, V, L, G, weights, lambd_grid) for j in sorted(folds)]
    if n_jobs == 1:
        results = [lambda_search(*task) for task in tasks]
    elif n_jobs > 1:
        with Pool(n_jobs) as p:
            results = p.starmap(lambda_search, tasks)
    else:
        raise ValueError("n_jobs must be at least 1")
    for result in results:
        j, errs, _, _ = result
        lambd_errs["fold_errors"][j] = errs

    fold_ids = sorted(lambd_errs["fold_errors"])
    if cv_fold_mode == "legacy_first_three":
        if not all(index in lambd_errs["fold_errors"] for index in range(3)):
            raise ValueError("legacy_first_three requires nonempty folds 0, 1, and 2")
        scoring_folds = list(range(3))
    elif cv_fold_mode == "all":
        scoring_folds = fold_ids
    else:
        raise ValueError(f"unknown cv_fold_mode={cv_fold_mode!r}")
    cv_errs = np.sum(
        [lambd_errs["fold_errors"][index] for index in scoring_folds], axis=0
    )
    lambd_errs["scoring_folds"] = scoring_folds
    lambd_errs["summed_cv_errors"] = np.asarray(cv_errs, dtype=float).tolist()
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    return lambd_cv, lambd_errs


def update_U_tilde_fixed_lambda(
    X,
    V,
    weights,
    lambd,
    return_unorthogonalized=False,
):
    """Update the graph-regularized left factor without running CV."""

    lambd = float(lambd)
    if not np.isfinite(lambd) or lambd < 0:
        raise ValueError("lambd must be finite and nonnegative")
    XV = X @ V

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
    U_tilde = ssnal.centers_.T

    U_hat, _, _ = svd(U_tilde, full_matrices=False)

    if return_unorthogonalized:
        return U_hat, U_tilde
    return U_hat


def update_V_L_tilde(X, U_tilde):
    V_mul = np.dot(X.T, U_tilde)
    V_hat, L_hat, _ = svd(V_mul, full_matrices=False)
    L_hat = np.diag(L_hat)
    return V_hat, L_hat
