import sys
import os
import time
import pickle
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

import pycvxcluster.pycvxcluster

from GpLSI import generate_topic_model as gen_model
from GpLSI.utils import *
from GpLSI import gplsi
import utils.spatial_lda.model

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

from collections import defaultdict

def _euclidean_proj_simplex(v, s=1):
        n = v.shape[0]
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w

def calculate_norm(matrix):
    return norm(matrix, axis=1)


def run_simul(
    nsim=50,
    N=100,
    n=1000,
    p=30,
    K=3,
    rt=0.05,
    m=5,
    n_clusters=30,
    phi=0.1,
    lamb_start=0.0001,
    step_size=1.25,
    grid_len=29,
    eps=1e-05,
    start_seed=None,
):
    results = defaultdict(list)
    models = {'gplsi': [], 'gplsi_XTX':[], 'plsi': [], 'tscore': [], 'lda': [], 'slda': []}

    # Activate automatic conversion of numpy objects to rpy2 objects
    numpy2ri.activate()

    # Import the necessary R packages
    nnls = rpackages.importr('nnls')
    rARPACK = rpackages.importr('rARPACK')
    quadprog = rpackages.importr('quadprog')
    Matrix = rpackages.importr('Matrix')

    r = robjects.r
    r['source']('topicscore.R')


    # print(f"Running simulations for N={N}, n={n}, p={p}, K={K}, r={r}, m={m}, phi={phi}...")
    model_save_loc = os.path.join(os.getcwd(), "sim_models_final_final")
    file_base = os.path.join(model_save_loc, f"simul_N={N}_n={n}_K={K}_p={p}")
    extensions = ['.csv', '.pkl']
    if not os.path.exists(model_save_loc):
        os.makedirs(model_save_loc, exist_ok=True)
    
    for trial in range(nsim):
        pkl_loc = f"{file_base}_trial={trial}{extensions[1]}"
        #if os.path.exists(pkl_loc):
        #    continue
        os.system(f"echo Running trial {trial}...")
        # Generate topic data and graph
        regens = 0
        if not (start_seed is None):
            np.random.seed(start_seed + trial)
        while True:
            try:
                coords_df, W, A, X = gen_model.generate_data(N, n, p, K, rt, n_clusters)
                weights, edge_df = gen_model.generate_weights_edge(coords_df, m, phi)
                D = N * X.T

                # TopicScore
                start_time = time.time()
                Mquantile = 0
                K0 = int(np.ceil(1.5*K))
                c = min(10*K, int(np.ceil(D.shape[0]*0.7)))
                D_r = robjects.r.matrix(D, nrow=D.shape[0], ncol=D.shape[1])  # Convert to R matrix
                norm_score = r['norm_score']
                A_hat_ts = norm_score(K, K0, c, D_r)
                M = np.mean(D, axis=1)
                M_trunk = np.minimum(M, np.quantile(M, Mquantile))
                S = np.diag(np.sqrt(1/M_trunk))
                H = S @ A_hat_ts 
                projector = np.linalg.inv(H.T @ H) @ H.T
                theta_W = (projector @ S) @ D
                W_hat_ts= np.array([_euclidean_proj_simplex(x) for x in theta_W.T])
                time_ts = time.time() - start_time
                A_hat_ts = A_hat_ts.T

                break
            except Exception as e:
                print(f"Regenerating dataset due to error: {e}")
                regens += 1


        # Vanilla SVD
        start_time = time.time()
        model_plsi = gplsi.GpLSI_(
            method="pLSI"
        )
        model_plsi.fit(X, K, edge_df, weights)
        time_plsi = time.time() - start_time

        # GpLSI
        start_time = time.time()
        model_gplsi = gplsi.GpLSI_(
            lamb_start=lamb_start,
            step_size=step_size,
            grid_len=grid_len,
            eps=eps
        )
        model_gplsi.fit(X, K, edge_df, weights)
        time_gplsi= time.time() - start_time
        print(f"CV Lambda is {model_gplsi.lambd}")

        # GpLSI using XTX/n for initial V0
        start_time = time.time()
        model_gplsi_XTX = gplsi.GpLSI_(
            lamb_start=lamb_start,
            step_size=step_size,
            grid_len=grid_len,
            eps=eps,
            initialize2=True
        )
        model_gplsi_XTX.fit(X, K, edge_df, weights)
        time_gplsi_XTX= time.time() - start_time
        print(f"CV Lambda is {model_gplsi_XTX.lambd}")

        # LDA 
        print("Running LDA...")
        start_time = time.time()
        model_lda = LatentDirichletAllocation(n_components=K, random_state=0)
        model_lda.fit(D.T)
        W_hat_lda = model_lda.transform(D.T)
        A_hat_lda_unnormalized = model_lda.components_
        A_hat_lda = A_hat_lda_unnormalized / A_hat_lda_unnormalized.sum(axis=1, keepdims=True)
        time_lda = time.time() - start_time

        # SLDA
        start_time = time.time()
        model_slda = utils.spatial_lda.model.run_simulation(X, K, coords_df)
        A_hat_slda = model_slda.components_
        row_sums = A_hat_slda.sum(axis=1, keepdims=True)
        A_hat_slda = (A_hat_slda / row_sums).T
        time_slda = time.time() - start_time

        M = X[edge_df['src']] - X[edge_df['tgt']]
        s = sum(calculate_norm(M) * np.sqrt(edge_df['weight']))

        # Record errors
        P_plsi = get_component_mapping(model_plsi.W_hat.T, W)
        P_gplsi = get_component_mapping(model_gplsi.W_hat.T, W)
        P_gplsi_onestep = get_component_mapping(model_gplsi.W_hat_init.T, W)
        P_gplsi_XTX = get_component_mapping(model_gplsi_XTX.W_hat.T, W)
        P_ts = get_component_mapping(W_hat_ts.T, W)
        P_lda = get_component_mapping(W_hat_lda.T, W)
        P_slda = get_component_mapping(model_slda.topic_weights.values.T, W)
        
        # permute W's
        W_hat_plsi = model_plsi.W_hat @ P_plsi
        W_hat_gplsi = model_gplsi.W_hat @ P_gplsi
        W_hat_gplsi_onestep = model_gplsi.W_hat_init @ P_gplsi_onestep
        W_hat_gplsi_XTX = model_gplsi_XTX.W_hat @ P_gplsi_XTX
        W_hat_ts = W_hat_ts @ P_ts
        W_hat_lda = W_hat_lda @ P_lda
        W_hat_slda = model_slda.topic_weights.values @ P_slda
       
        # permute A's
        P_plsi_A = get_component_mapping(model_plsi.A_hat, A.T)
        P_gplsi_A = get_component_mapping(model_gplsi.A_hat, A.T)
        P_gplsi_A_onestep = get_component_mapping(model_gplsi.A_hat_init, A.T)
        P_gplsi_XTX_A = get_component_mapping(model_gplsi_XTX.A_hat, A.T)
        P_ts_A = get_component_mapping(A_hat_ts, A.T)
        P_lda_A = get_component_mapping(A_hat_lda, A.T)
        P_slda_A = get_component_mapping(A_hat_slda.T, A.T)

        A_hat_plsi = P_plsi_A.T @ model_plsi.A_hat
        A_hat_gplsi = P_gplsi_A.T @ model_gplsi.A_hat
        A_hat_gplsi_onestep = P_gplsi_A_onestep.T @ model_gplsi.A_hat_init
        A_hat_gplsi_XTX = P_gplsi_XTX_A.T @ model_gplsi.A_hat
        A_hat_ts = P_ts_A.T @ A_hat_ts
        A_hat_lda = P_lda_A.T @ A_hat_lda
        A_hat_slda = P_slda_A.T @ A_hat_slda.T

        err_acc_plsi = [
            get_F_err(W_hat_plsi, W),
            get_l1_err(W_hat_plsi, W),
            get_F_err(A_hat_plsi, A),
            get_l1_err(A_hat_plsi, A),
            get_accuracy(coords_df, n, W_hat_plsi),
        ]
        err_acc_gplsi = [
            get_F_err(W_hat_gplsi, W),
            get_l1_err(W_hat_gplsi, W),
            get_F_err(A_hat_gplsi, A),
            get_l1_err(A_hat_gplsi, A),
            get_accuracy(coords_df, n, W_hat_gplsi),
        ]
        err_acc_gplsi_onestep = [
            get_F_err(W_hat_gplsi_onestep, W),
            get_l1_err(W_hat_gplsi_onestep, W),
            get_F_err(A_hat_gplsi_onestep, A),
            get_l1_err(A_hat_gplsi_onestep, A),
            get_accuracy(coords_df, n, W_hat_gplsi_onestep),
        ]
        err_acc_gplsi_XTX = [
            get_F_err(W_hat_gplsi_XTX, W),
            get_l1_err(W_hat_gplsi_XTX, W),
            get_F_err(A_hat_gplsi_XTX, A),
            get_l1_err(A_hat_gplsi_XTX, A),
            get_accuracy(coords_df, n, W_hat_gplsi_XTX),
        ]
        err_acc_ts = [
            get_F_err(W_hat_ts, W),
            get_l1_err(W_hat_ts, W),
            get_F_err(A_hat_ts, A),
            get_l1_err(A_hat_ts, A),
            get_accuracy(coords_df, n, W_hat_ts),
        ]
        err_acc_lda = [
            get_F_err(W_hat_lda, W),
            get_l1_err(W_hat_lda, W),
            get_F_err(A_hat_lda, A),
            get_l1_err(A_hat_lda, A),
            get_accuracy(coords_df, n, W_hat_lda),
        ]
        err_acc_slda = [
            get_F_err(W_hat_slda, W),
            get_l1_err(W_hat_slda, W),
            get_F_err(A_hat_slda, A),
            get_l1_err(A_hat_slda, A),
            get_accuracy(coords_df, n, W_hat_slda),
        ]

        results["trial"].append(trial)
        results["jump"].append(s/n)
        results["N"].append(N)
        results["n"].append(n)
        results["p"].append(p)
        results["K"].append(K)

        results["plsi_err"].append(err_acc_plsi[0])
        results["plsi_l1_err"].append(err_acc_plsi[1])
        results["A_plsi_err"].append(err_acc_plsi[2])
        results["A_plsi_l1_err"].append(err_acc_plsi[3])
        results["plsi_acc"].append(err_acc_plsi[4])
        print(results['plsi_err'])
        print(results["A_plsi_err"])

        results["gplsi_err"].append(err_acc_gplsi[0])
        results["gplsi_l1_err"].append(err_acc_gplsi[1])
        results["A_gplsi_err"].append(err_acc_gplsi[2])
        results["A_gplsi_l1_err"].append(err_acc_gplsi[3])
        results["gplsi_acc"].append(err_acc_gplsi[4])

        results["gplsi_onestep_err"].append(err_acc_gplsi_onestep[0])
        results["gplsi_onestep_l1_err"].append(err_acc_gplsi_onestep[1])
        results["A_gplsi_onestep_err"].append(err_acc_gplsi_onestep[2])
        results["A_gplsi_onestep_l1_err"].append(err_acc_gplsi_onestep[3])
        results["gplsi_onestep_acc"].append(err_acc_gplsi_onestep[4])

        results["gplsi_XTX_err"].append(err_acc_gplsi_XTX[0])
        results["gplsi_XTX_l1_err"].append(err_acc_gplsi_XTX[1])
        results["A_gplsi_XTX_err"].append(err_acc_gplsi_XTX[2])
        results["A_gplsi_XTX_l1_err"].append(err_acc_gplsi_XTX[3])
        results["gplsi_XTX_acc"].append(err_acc_gplsi_XTX[4])

        results["opt_gamma"].append(model_gplsi.lambd)
        results["opt_gamma_init"].append(model_gplsi.lambd_init)
        results["opt_gamma_XTX"].append(model_gplsi_XTX.lambd)

        print(results['gplsi_err'])
        print(results["A_gplsi_err"])

        print(results['gplsi_onestep_err'])
        print(results['A_gplsi_onestep_err'])

        results["ts_err"].append(err_acc_ts[0])
        results["ts_l1_err"].append(err_acc_ts[1])
        results["A_ts_err"].append(err_acc_ts[2])
        results["A_ts_l1_err"].append(err_acc_ts[3])
        results["ts_acc"].append(err_acc_ts[4])

        results["lda_err"].append(err_acc_lda[0])
        results["lda_l1_err"].append(err_acc_lda[1])
        results["A_lda_err"].append(err_acc_lda[2])
        results["A_lda_l1_err"].append(err_acc_lda[3])
        results["lda_acc"].append(err_acc_lda[4])

        results["slda_err"].append(err_acc_slda[0])
        results["slda_l1_err"].append(err_acc_slda[1])
        results["A_slda_err"].append(err_acc_slda[2])
        results["A_slda_l1_err"].append(err_acc_slda[3])
        results["slda_acc"].append(err_acc_slda[4])

        results["plsi_time"].append(time_plsi)
        results["gplsi_time"].append(time_gplsi)
        results['gplsi_onestep_time'].append(model_gplsi.time_init)
        results['gplsi_XTX_time'].append(time_gplsi_XTX)
        results["ts_time"].append(time_ts)
        results["lda_time"].append(time_lda)
        results["slda_time"].append(time_slda)

        models['plsi'].append(model_plsi)
        models['gplsi'].append(model_gplsi)
        models['gplsi_XTX'].append(model_gplsi_XTX)
        models['tscore'].append(A_hat_ts.T)
        models['lda'].append(model_lda)
        models['slda'].append(model_slda)

        with open(pkl_loc, "wb") as f:
            pickle.dump(models, f)

    results = pd.DataFrame(results)
    return results

