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
from GpLSI import cfg

import pycvxcluster.pycvxcluster

# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"
from multiprocessing import Pool


def graphSVD(
    X,
    K,
    edge_df,
    weights,
    lamb_start,
    step_size,
    grid_len,
    maxiter,
    eps,
    verbose,
    initialize,
    initialize2,
):
    n = X.shape[0]
    srn, folds, G, mst = get_folds_disconnected_G(edge_df)

    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)

    lambd_grid_init = (0.0001 * np.power(1.5, np.arange(10))).tolist()
    lambd_grid_init.insert(0, 1e-06)

    if initialize:
        if initialize2:
            print('Initializing using XTX...')
            X_centered = X - np.mean(X, axis=0)
            cov = X_centered.T @ X_centered
            U, L, V  =svds(cov/n, k=K)
            V  = V.T
            L = np.diag(L)
            V_init = V
            L_init = L
            U, _, _ = svds(X, k=K)
            U_init = U
            time_init = None
            lambd_init =  None
        else:
            print('Initializing..')
            start_time = time.time()
            M, lambd_init, lambd_errs = initial_svd(X, G, weights, folds, lambd_grid_init)
            time_init = time.time() - start_time
            print(f'Initializing time: {time_init}')
            U, L, V = svds(M, k=K)
            V  = V.T
            L = np.diag(L)
            U_init = U
            V_init = V
            L_init = L
            niter=0
    else:
        U, L, V = svds(X, k=K)
        V  = V.T
        L = np.diag(L)
        U_init = None
        V_init = None
        L_init = None
        time_init = None
        lambd_init =  None

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

    return U, V, L, U_init, V_init, L_init, lambd, lambd_errs, niter, time_init, lambd_init


def lambda_search_init(j, folds, X, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    X_j = X[fold, :]
  
    errs = []
    best_err = float("inf")
    M_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=X_tilde,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        M_hat = ssnal.centers_.T
        err = norm(X_j - M_hat[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            M_best = M_hat
            best_err = err
    return j, errs, M_best, lambd_best


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


def initial_svd(X, G, weights, folds, lambd_grid):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    
    with Pool(3) as p:
        results = p.starmap(
            lambda_search_init,
            [(j, folds, X, G, weights, lambd_grid) for j in folds.keys()],
        )
    for result in results:
        j, errs, _, lambd_best = result
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.sum([lambd_errs["fold_errors"][i] for i in range(3)], axis=0)
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=X, weight_matrix=weights, save_centers=True)
    M_hat = ssnal.centers_.T

    print(f"Optimal lambda is {lambd_cv}...")
    return M_hat, lambd_cv, lambd_errs


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

def plot_fold_cv(lambd_grid, lambd_errs, lambd, N):
    cv_1 = np.round(lambd_grid[np.argmin(lambd_errs["fold_errors"][0])], 5)
    cv_2 = np.round(lambd_grid[np.argmin(lambd_errs["fold_errors"][1])], 5)
    cv_final = lambd
    for j, fold_errs in lambd_errs["fold_errors"].items():
        plt.plot(np.log(lambd_grid), fold_errs, label=f"Fold {j}", marker="o")
        # plt.vlines(cv_1, 18.90, 18.25, color = "blue")
        # plt.vlines(cv_2, 18.90, 18.25, color = "orange")

    # plt.plot(lambd_grid, new_list, label='Final Errors', linestyle='--', linewidth=2)
    plt.xlabel("Lambda")
    plt.ylabel("Errors")
    plt.text(cv_1, lambd_errs["fold_errors"][0][0], cv_1, color="blue")
    plt.text(cv_1, lambd_errs["fold_errors"][1][0], cv_2, color="orange")
    plt.title(f"Lambda CV = {cv_final}")
    plt.legend()
    plt.show()