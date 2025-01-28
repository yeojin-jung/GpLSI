import sys
import os
import time
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict

from scipy.sparse import csr_matrix

# !git clone https://github.com/dx-li/pycvxcluster.git
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([
    os.path.join(parent_dir, "pycvxcluster"),
    parent_dir 
])

import logging
logging.captureWarnings(True)

import pycvxcluster.pycvxcluster
from GpLSI.utils import *
from utils.data_helpers import *
from GpLSI import gplsi

from sklearn.decomposition import LatentDirichletAllocation

def preprocess_crc(coord_df, edge_df, D, phi):
    new_columns = [col.replace("X", "x").replace("Y", "y") for col in coord_df.columns]
    coord_df.columns = new_columns

    # get cell index
    cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], range(coord_df.shape[0])))

    # normalize coordinate to (0,1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)

    # get weight
    edge_df_ = edge_df.copy()
    edge_df_["src"] = edge_df["src"].map(cell_to_idx_dict)
    edge_df_["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
    edge_df_["weight"] = dist_to_exp_weight(edge_df_, coord_df, phi)

    # edge, coord, X, weights
    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    N = row_sums.mean()
    X = D.div(row_sums, axis=0)
    n = X.shape[0]
    weights = csr_matrix(
        (edge_df_["weight"].values, (edge_df_["src"].values, edge_df_["tgt"].values)),
        shape=(n, n),
    )

    return X, N, edge_df_, coord_df, weights, n, nodes


def save_model_results(results, model_name, model, A_hat=None, W_hat=None):
    results[model_name].append(model)
    if A_hat is not None:
        results[f'A_hat_{model_name}'].append(A_hat)
    if W_hat is not None:
        results[f'W_hat_{model_name}'].append(W_hat)


if __name__ == "__main__":
    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])

    root_path = os.path.join(parent_dir, "data/stanford-crc")
    dataset_root = os.path.join(root_path, "output/output_3hop")
    model_root = os.path.join(parent_dir, "output/stanford-crc")
    
    meta_path = os.path.join(root_path, "charville_labels.csv")
    meta = pd.read_csv(meta_path)
    meta_sub = meta[~pd.isna(meta['primary_outcome'])]
    filenames = meta_sub['region_id'].to_list()
    
    D_all = pd.DataFrame()
    edge_all = pd.DataFrame()
    coords_all = pd.DataFrame()

    s = 0
    for filename in filenames:
            paths = {kind: os.path.join(dataset_root, f"{filename}.{kind}.csv") for kind in ['D', 'edge', 'coord', 'type', 'model']}
            D = pd.read_csv(paths['D'], index_col=0, converters={0: tuple_converter})
            D['filename'] = filename 
            edge_df = pd.read_csv(paths['edge'], index_col=0)
            coord_df = pd.read_csv(paths['coord'], index_col=0)
            type_df = pd.read_csv(paths['type'], index_col=0)
            coords_df = pd.merge(coord_df, type_df).reset_index(drop=True)

            cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], [i+s for i in range(coord_df.shape[0])]))

            edge_df["src"] = edge_df["src"].map(cell_to_idx_dict)
            edge_df["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
            coords_df["CELL_ID"] = coords_df["CELL_ID"].map(cell_to_idx_dict)
            new_index = [(x,cell_to_idx_dict[y]) for x,y in D.index]
            D.index = new_index

            D = D[D.iloc[:, :-1].sum(axis=1) >= 10]
            idx = [y for x,y in D.index]
            edge_df = edge_df[(edge_df['src'].isin(idx)) & (edge_df['tgt'].isin(idx))]
            coords_df = coords_df[coords_df['CELL_ID'].isin(idx)]

            D_all = pd.concat([D_all, D], axis=0, ignore_index=False)
            edge_all = pd.concat([edge_all, edge_df], axis=0, ignore_index=True)
            coords_all = pd.concat([coords_all, coords_df], axis=0, ignore_index=True)
            s+= D.shape[0]
    print(f"D has {s} rows.")

    patient_id_path = os.path.join(root_path, "selected_patient_id.csv")
    if not os.path.exists(patient_id_path):
          patient_id_df = D_all['filename']
          patient_id_df.to_csv(patient_id_path, index=False)

    D_all = D_all.drop(columns=['filename'])
    X, N, edge_df, coord_df, weights, n, nodes = preprocess_crc(coords_all, edge_all, D_all, phi=0.1)
    
    # run GpLSI
    start_time = time.time()
    model_gplsi = gplsi.GpLSI_(
        lamb_start=lamb_start,
        step_size=step_size,
        grid_len=grid_len,
        eps=eps
    )
    model_gplsi.fit(X.values, N, K, edge_df, weights)
    time_gplsi = time.time() - start_time

    # run pLSI
    start_time = time.time()
    model_plsi = gplsi.GpLSI_(
        method='pLSI'
    )
    model_plsi.fit(X.values, N, K, edge_df, weights)
    time_plsi = time.time() - start_time

    # run LDA
    start_time = time.time()
    lda = LatentDirichletAllocation(n_components=K, random_state=0)
    lda.fit(D_all.values)
    time_lda = time.time() - start_time
    W_hat_lda = lda.transform(D_all.values)
    A_hat_lda = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

    # save files
    results = defaultdict(list)
    save_model_results(results, 'gplsi', model_gplsi, model_gplsi.A_hat, model_gplsi.W_hat)
    save_model_results(results, 'plsi', model_plsi, model_plsi.A_hat, model_plsi.W_hat)
    save_model_results(results, 'lda', lda, A_hat_lda, W_hat_lda)

    save_path = os.path.join(model_root, f'results_crc_{K}.pkl')

    with open(save_path, "wb") as f:
            pickle.dump(results, f)
