import os
import sys
import time
import random
import pickle
import numpy as np
from numpy.linalg import norm, svd, solve
from scipy.linalg import inv, sqrtm
import networkx as nx

from scipy.sparse.linalg import svds

from GpLSI.utils import *

import pycvxcluster.pycvxcluster

# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"
from multiprocessing import Pool


def graphSVD(
    X,
    N,
    K,
    edge_df,
    weights,
    lamb_start,
    step_size,
    grid_len,
    maxiter,
    eps,
    verbose,
    initialize
):
    n = X.shape[0]
    srn, folds, G, mst = get_folds_disconnected_G(edge_df)

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