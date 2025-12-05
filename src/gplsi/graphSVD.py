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

    lambd_errs : list[float]
        List of cross-validation errors for each lambda in the grid.

    niter : int
        Number of outer iterations performed.

    Notes
    -----
    - Internally, we:
        1. Build a disconnected graph structure using get_folds_disconnected_G(edge_df).
        2. Construct a grid of lambda values (lambd_grid).
        3. Optionally do an SVD-based initialization (initialize=True).
        4. Alternate between:
            - update_U_tilde (graph-regularized U, with CV over lambda)
            - update_V_L_tilde (update V, L given U)
      until the reconstruction stabilizes.

    - Convergence criterion uses a random subsample of up to 1000 rows to
      measure the change in P_U X P_V, where P_U and P_V are projection matrices.
    """
    n = X.shape[0]
    _, folds, G, _ = get_folds_disconnected_G(edge_df)

    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)

    lambd_grid_init = (0.0001 * np.power(1.5, np.arange(10))).tolist()
    lambd_grid_init.insert(0, 1e-06)

    if initialize:
        print('Initializing...')
        colsums = np.sum(X, axis=0)
        cov = X.T @ X - np.diag(colsums/N)
        U, L, V  =svds(cov, k=K)
        V  = V.T
        L = np.diag(L)
        V_init = V
        L_init = L
        U, _, _ = svds(X, k=K)
        U_init = U
    else:
        U, L, V = svds(X, k=K)
        V  = V.T
        L = np.diag(L)
        U_init = None
        V_init = None
        L_init = None

    score = 1
    niter = 0
    while score > eps and niter < maxiter:
        if n > 1000:
            idx = np.random.choice(range(n),1000,replace=False)
        else:
            idx = range(n)
        
        U_samp = U[idx,:]
        P_U_old = np.dot(U_samp, U_samp.T)
        P_V_old = np.dot(V, V.T)
        X_hat_old = (P_U_old @ X[idx,:]) @ P_V_old
        U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid)
        V, L = update_V_L_tilde(X, U)

        P_U = np.dot(U[idx,:], U[idx,:].T)
        P_V = np.dot(V, V.T)
        X_hat = (P_U @ X[idx,:]) @ P_V
        score = norm(X_hat-X_hat_old)/n
        niter += 1
        if verbose == 1:
            print(f"Error is {score}")
    
    print(f"Graph-aligned SVD ran for {niter} steps.")

    return U, V, L, U_init, V_init, L_init, lambd, lambd_errs, niter


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


def update_U_tilde(X, V, L, G, weights, folds, lambd_grid):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    L_inv = 1/np.diag(L)
    XVL_inv = X @ V

    with Pool(3) as p:
        results = p.starmap(
            lambda_search,
            [(j, folds, X, V, L, G, weights, lambd_grid) for j in folds.keys()],
        )
    for result in results:
        j, errs, _, lambd_best = result
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.sum([lambd_errs["fold_errors"][i] for i in range(3)], axis=0)
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=XVL_inv, weight_matrix=weights, save_centers=True)
    U_tilde = ssnal.centers_.T

    U_hat, _, _ = svd(U_tilde, full_matrices=False)

    print(f"Optimal lambda is {lambd_cv}...")
    return U_hat, lambd_cv, lambd_errs


def update_V_L_tilde(X, U_tilde):
    V_mul = np.dot(X.T, U_tilde)
    V_hat, L_hat, _ = svd(V_mul, full_matrices=False)
    L_hat = np.diag(L_hat)
    return V_hat, L_hat