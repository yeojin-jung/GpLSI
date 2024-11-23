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

    root_path = os.path.join(os.getcwd(), "data/spleen")
    dataset_root = os.path.join(root_path, "dataset")
    model_root = os.path.join(root_path, "model")
    fig_root = os.path.join(root_path, "fig")
    path_to_D = os.path.join(dataset_root, "merged_D.pkl")
    path_to_edge = os.path.join(dataset_root, "merged_data.pkl")
    path_to_coord = os.path.join(dataset_root, "merged_coord.pkl")
    
    tumors = ["BALBc-1", "BALBc-2", "BALBc-3"]

    D_all = pd.DataFrame()
    edge_all = pd.DataFrame()
    coords_all = pd.DataFrame()

    s = 0
    for tumor in tumors:
            spleen_D = pd.read_pickle(path_to_D).loc[tumor]
            gc.collect()
            spleen_edge = pd.read_pickle(path_to_edge).loc[tumor]
            gc.collect()
            spleen_coord = pd.read_pickle(path_to_coord).loc[tumor]
            gc.collect()
            
            ntumor = spleen_D.shape[0]
            spleen_coord.columns = ["x", "y"]
            spleen_edge.columns = ["src", "tgt", "distance"]
            cell_to_idx_dict = dict(zip(range(ntumor), [i+s for i in range(ntumor)]))

            spleen_edge["src"] = spleen_edge["src"].map(cell_to_idx_dict)
            spleen_edge["tgt"] = spleen_edge["tgt"].map(cell_to_idx_dict)

            D_all = pd.concat([D_all, spleen_D], axis=0, ignore_index=False)
            edge_all = pd.concat([edge_all, spleen_edge], axis=0, ignore_index=True)
            coords_all = pd.concat([coords_all, spleen_coord], axis=0, ignore_index=True)
            s+= ntumor
    print(f"D has {s} rows.")
    
    coords_all = coords_all.reset_index()

    X, edge_df, coord_df, weights, n, nodes = preprocess_spleen(coords_all, edge_all, D_all, phi=0.1)
    
    # GpLSI
    start_time = time.time()
    model_gplsi = gplsi.GpLSI_(
        lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, eps=eps
    )
    model_gplsi.fit(X.values, K, edge_df, weights)
    time_gplsi = time.time() - start_time

    # PLSI
    start_time = time.time()
    model_plsi = gplsi.GpLSI_(
        method='pLSI',
    )
    model_plsi.fit(X.values, K, edge_df, weights)
    time_plsi = time.time() - start_time

    # LDA
    start_time = time.time()
    model_lda = LatentDirichletAllocation(n_components=K, random_state=0)
    model_lda.fit(D_all.values)
    time_lda = time.time() - start_time

    save_path = os.path.join(model_root, f'spleen_model_gplsi_all_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(model_gplsi, f)

    save_path = os.path.join(model_root, f'spleen_model_plsi_all_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(model_plsi, f)

    save_path = os.path.join(model_root, f'spleen_model_lda_all_{K}.pkl')
    with open(save_path, "wb") as f:
            pickle.dump(model_lda, f)

    # align models
    results = []

    W_gplsi = model_gplsi.W_hat
    gmoran_gplsi, moran_gplsi = moran(W_gplsi, edge_all)
    gchaos_gplsi, chaos_gplsi = get_CHAOS(W_gplsi, nodes, coords_all, n, K)
    pas_gplsi = get_PAS(W_gplsi, edge_all)

    W_plsi = model_plsi.W_hat
    gmoran_plsi, moran_plsi = moran(W_plsi, edge_all)
    gchaos_plsi, chaos_plsi = get_CHAOS(W_plsi, nodes, coords_all, n, K)
    pas_plsi = get_PAS(W_plsi, edge_all)

    W_lda = model_lda.transform(D_all.values)
    gmoran_lda, moran_lda = moran(W_lda, edge_df)
    gchaos_lda, chaos_lda = get_CHAOS(W_lda, nodes, coords_all, n, K)
    pas_lda = get_PAS(W_lda, edge_all)

    # Align A_hat
    A_hat_gplsi = model_gplsi.A_hat.T
    A_hat_plsi = model_plsi.A_hat.T
    A_hat_lda = W_lda.components_ / W_lda.components_.sum(axis=1)[:, np.newaxis]

    # Plot
    names = ["gplsi", "PLSI", "LDA"]
    morans = [gmoran_gplsi, gmoran_plsi, gmoran_lda]
    moran_locals = [moran_gplsi, moran_plsi, gmoran_lda]
    chaoss = [gchaos_gplsi, gchaos_plsi, gchaos_lda]
    chaos_locals = [chaos_gplsi, chaos_plsi, chaos_lda]
    pas = [pas_gplsi, pas_plsi, pas_lda]
    times = [time_gplsi, time_plsi, time_lda]
    Whats = [W_gplsi, W_plsi, W_lda]
    Ahats = [A_hat_gplsi, A_hat_plsi, A_hat_lda]

    print(times)

    # fig, axes = plt.subplots(1,3, figsize=(18,6))
    # for j, ax in enumerate(axes):
    #    w = np.argmax(Whats[j], axis=1)
    #    samp_coord_ = coord_df.copy()
    #    samp_coord_['tpc'] = w
    #    sns.scatterplot(x='x',y='y',hue='tpc', data=samp_coord_, palette='viridis', ax=ax, s=20)
    #    name = names[j]
    #    ax.set_title(f'{name} (chaos:{np.round(chaoss[j],8)}, moran:{np.round(morans[j],3)}, time:{np.round(times[j],2)})')
    # plt.tight_layout()
    # plt.show()

    results.append(
        {
            "Whats": Whats,
            "Ahats": Ahats,
            "chaoss": chaoss,
            "chaos_locals": chaos_locals,
            "morans": morans,
            "moran_locals": moran_locals,
            "pas": pas,
            "times": times,
            "coord_df": coord_df,
            "edge_df": edge_df,
        }
    )
    save_path = os.path.join(model_root, f'spleen_model_results_{K}_.pkl')
    results = pd.DataFrame(results)
    results.to_csv(save_path)

