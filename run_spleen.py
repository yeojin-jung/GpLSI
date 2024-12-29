import sys
import os
import gc
import time
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/")
import pycvxcluster.pycvxcluster

import logging
logging.captureWarnings(True)

from GpLSI.utils import *
from utils.data_helpers import *
from GpLSI import gplsi

import utils.spatial_lda.model
from utils.spatial_lda.featurization import make_merged_difference_matrices

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri


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

def preprocess_spleen(coord_df, edge_df, D, phi):
    # normalize coordinate to (0,1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)

    # get weight
    edge_df["weight"] = dist_to_exp_weight(
        edge_df, coord_df, phi
    )

    # edge, coord, X, weights
    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    X = D.div(row_sums, axis=0)  # normalize
    n = X.shape[0]
    weights = csr_matrix(
        (edge_df["weight"].values, (edge_df["src"].values, edge_df["tgt"].values)),
        shape=(n, n),
    )

    return X, edge_df, coord_df, weights, n, nodes


if __name__ == "__main__":
    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])
    ntumor = int(sys.argv[6])

    # Activate automatic conversion of numpy objects to rpy2 objects
    numpy2ri.activate()

    # Import the necessary R packages
    nnls = rpackages.importr('nnls')
    rARPACK = rpackages.importr('rARPACK')
    quadprog = rpackages.importr('quadprog')
    Matrix = rpackages.importr('Matrix')

    r = robjects.r
    r['source']('topicscore.R')

    root_path = os.path.join(os.getcwd(), "data/spleen")
    dataset_root = os.path.join(root_path, "dataset")
    model_root = os.path.join(root_path, "model")
    fig_root = os.path.join(root_path, "fig")
    path_to_D = os.path.join(dataset_root, "merged_D.pkl")
    path_to_edge = os.path.join(dataset_root, "merged_data.pkl")
    path_to_coord = os.path.join(dataset_root, "merged_coord.pkl")
    
    tumors = ["BALBc-1", "BALBc-2", "BALBc-3"]
    tumor = tumors[ntumor]

    spleen_D = pd.read_pickle(path_to_D).loc[tumor]
    spleen_edge = pd.read_pickle(path_to_edge).loc[tumor]
    spleen_coord = pd.read_pickle(path_to_coord).loc[tumor]
    
    spleen_coord.columns = ["x", "y"]
    spleen_edge.columns = ["src", "tgt", "distance"]

    X, edge_df, coord_df, weights, n, nodes = preprocess_spleen(spleen_coord, spleen_edge, spleen_D, phi=0.1)
    del spleen_edge, spleen_coord

    # TopicScore
    D = spleen_D.values
    D = D.T
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
    
    # GpLSI 
    start_time = time.time()
    model_gplsi = gplsi.GpLSI_(
        lamb_start=lamb_start,
        step_size=step_size,
        grid_len=grid_len,
        eps=eps
    )
    model_gplsi.fit(X.values, K, edge_df, weights)
    time_gplsi = time.time() - start_time

    # pLSI
    start_time = time.time()
    model_plsi = gplsi.GpLSI_(
       method="pLSI"
    )
    model_plsi.fit(X.values, K, edge_df, weights)
    time_plsi = time.time() - start_time

    # LDA
    start_time = time.time()
    model_lda = LatentDirichletAllocation(n_components=K, random_state=0)
    model_lda.fit(spleen_D.values)
    time_lda = time.time() - start_time

    # SLDA
    cell_org = [x[1] for x in X.index]
    cell_dict = dict(zip(range(len(X)), cell_org))
    samp_coord_ = coord_df.copy()
    samp_coord_.index = coord_df.index.map(cell_dict)
    samp_coord__ = {tumor: samp_coord_}
    difference_matrices = make_merged_difference_matrices(X, samp_coord__, "x", "y")
    print("Running SLDA...")
    start_time = time.time()
    model_slda = utils.spatial_lda.model.train(
        sample_features=X,
        difference_matrices=difference_matrices,
        difference_penalty=0.25,
        n_topics=K,
        n_parallel_processes=2,
        verbosity=1,
        admm_rho=0.1,
        primal_dual_mu=1e5,
    )
    time_slda = time.time() - start_time

    results = []

    W_gplsi = model_gplsi.W_hat
    gmoran_gplsi, moran_gplsi = moran(W_gplsi, edge_df)
    gchaos_gplsi, chaos_gplsi = get_CHAOS(W_gplsi, nodes, coord_df, n, K)
    pas_gplsi = get_PAS(W_gplsi, edge_df)

    W_plsi = model_plsi.W_hat
    gmoran_plsi, moran_plsi = moran(W_plsi, edge_df)
    gchaos_plsi, chaos_plsi = get_CHAOS(W_plsi, nodes, coord_df, n, K)
    pas_plsi = get_PAS(W_plsi, edge_df)

    gmoran_ts, moran_plsi = moran(W_hat_ts, edge_df)
    gchaos_ts, chaos_plsi = get_CHAOS(W_hat_ts, nodes, coord_df, n, K)
    pas_ts = get_PAS(W_hat_ts, edge_df)

    W_lda = model_lda.transform(spleen_D.values)
    gmoran_lda, moran_lda = moran(W_lda, edge_df)
    gchaos_lda, chaos_lda = get_CHAOS(W_lda, nodes, coord_df, n, K)
    pas_lda = get_PAS(W_lda, edge_df)

    W_slda = model_slda.topic_weights.values
    gmoran_slda, moran_slda = moran(W_slda, edge_df)
    gchaos_slda, chaos_slda = get_CHAOS(W_slda, nodes, coord_df, n, K)
    pas_slda = get_PAS(W_slda, edge_df)

    A_hat_gplsi = model_gplsi.A_hat.T
    A_hat_plsi = model_plsi.A_hat.T
    A_hat_lda = model_lda.components_ / model_lda.components_.sum(axis=1)[:, np.newaxis]
    A_hat_slda = model_slda.components_
    row_sums = A_hat_slda.sum(axis=1, keepdims=True)
    A_hat_slda = (A_hat_slda / row_sums).T

    names = ["GpLSI", "pLSI", "TopicScore", "LDA", "SLDA"]
    morans = [gmoran_gplsi, gmoran_plsi, gmoran_ts, gmoran_lda, gmoran_slda]
    chaoss = [gchaos_gplsi, gchaos_plsi, gchaos_ts, gchaos_lda, gchaos_slda]
    pas = [pas_gplsi, pas_plsi, pas_ts, pas_lda, pas_slda]
    times = [time_gplsi, time_plsi, time_ts, time_lda, time_slda]
    Whats = [W_gplsi, W_plsi, W_hat_ts, W_lda, W_slda]
    Ahats = [A_hat_gplsi, A_hat_plsi, A_hat_ts, A_hat_lda, A_hat_slda]

    results.append(
        {
            "Whats": Whats,
            "Ahats": Ahats,
            "chaoss": chaoss,
            "morans": morans,
            "pas": pas,
            "times": times,
            "coord_df": coord_df,
            "edge_df": edge_df,
        }
    )
    save_path = os.path.join(model_root, f'{tumor}_spleen_model_results_{K}_.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(results, f)

