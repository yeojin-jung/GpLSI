import sys
import os
import gc
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

from scipy.sparse import csr_matrix
from collections import defaultdict

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/")
import pycvxcluster.pycvxcluster

import logging
logging.captureWarnings(True)

from GpLSI.utils import *
from utils.data_helpers import *
from GpLSI import gplsi

from sklearn.decomposition import LatentDirichletAllocation

from mpi4py import MPI


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
    X = D.div(row_sums, axis=0)  # normalize
    n = X.shape[0]
    weights = csr_matrix(
        (edge_df_["weight"].values, (edge_df_["src"].values, edge_df_["tgt"].values)),
        shape=(n, n),
    )

    return D, X.values, edge_df_, coord_df, weights, n, nodes


def divide_folds(filenames, num_parts=2):
    filenames = np.random.shuffle(filenames)
    avg_length = len(filenames) / float(num_parts)

    divided_folds = []
    last = 0.0

    while last < len(filenames):
        divided_folds.append(filenames[int(last):int(last + avg_length)])
        last += avg_length

    return divided_folds


def shuffle_folds(filenames, dataset_root, nfolds):
    data_inputs = []

    divided_folds = divide_folds(filenames, nfolds)

    for i in range(nfolds):
        D_fold = pd.DataFrame()
        edge_fold = pd.DataFrame()
        coords_fold = pd.DataFrame()

        set = divided_folds[i]
        s = 0
        for filename in set:
            paths = {kind: os.path.join(dataset_root, f"{filename}.{kind}.csv") for kind in ['D', 'edge', 'coord', 'type', 'model']}
            D = pd.read_csv(paths['D'], index_col=0, converters={0: tuple_converter})
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

            D = D[D.sum(axis=1)>=10]
            idx = [y for x,y in D.index]
            edge_df = edge_df[(edge_df['src'].isin(idx)) & (edge_df['tgt'].isin(idx))]
            coords_df = coords_df[coords_df['CELL_ID'].isin(idx)]

            D_fold = pd.concat([D_fold, D], axis=0, ignore_index=False)
            edge_fold = pd.concat([edge_fold, edge_df], axis=0, ignore_index=True)
            coords_fold = pd.concat([coords_fold, coords_df], axis=0, ignore_index=True)
            del D, edge_df, coords_df
            gc.collect()

        D, X, edge_df, _, weights, _, _ = preprocess_crc(coords_fold, edge_fold, D_fold, phi=0.1)
        data_inputs.append((D, X, edge_df, weights))
        del D_fold, X, edge_df, weights
        gc.collect()

    return data_inputs


def get_resolutions(all_Ahats, method_name):
    results = defaultdict(list)
    for i, j in combinations(range(len(all_Ahats[0])), 2):
        P = get_component_mapping(all_Ahats[i][0], all_Ahats[j][0])
        A_1 = all_Ahats[i][0]
        A_2 = all_Ahats[j][0]
        A_1 = P.T @ A_1
        K = A_1.shape[0]

        # l1 distance
        l1_dist = np.sum(np.abs(A_1 - A_2))

        # cosine similarity
        A_1_norm = A_1 / norm(A_1, axis=1, keepdims=True)
        A_2_norm = A_2 / norm(A_2, axis=1, keepdims=True)
        diag_cos = np.mean(np.diag(A_1_norm @ A_2_norm.T))

        # cosine similarity ratio
        off_cos = (np.sum(A_1_norm @ A_2_norm.T)-np.sum(np.diag(A_1_norm @ A_2_norm.T)))/(K*K - K)
        r = diag_cos/off_cos

        results['K'].append(K)
        results['l1_dist'].append(l1_dist/K)
        results['cos_sim'].append(diag_cos)
        results['cos_sim_ratio'].append(r)
        results['method'].append(method_name)
    
    return results


def plot_folds(all_Ahats, K, id, method_name):
    vmin = min(matrix[0].min() for matrix in all_Ahats)
    vmax = max(matrix[0].max() for matrix in all_Ahats)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3)) 
    for ax, matrix in zip(axes.flatten(), all_Ahats):
        P = get_component_mapping(matrix[0], all_Ahats[0][0])
        matrix[0] = P.T @ matrix[0]
        row_names = ["T"+str(i) for i in range(1, K+1)]
        column_names = ['CD4 T cell', 'CD8 T cell', 'B cell', 'Macrophage', 'Granulocyte', 'Blood vessel', 'Stroma', 'Other']
        im = ax.imshow(matrix[0].T, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(row_names)))
        ax.set_yticks(np.arange(len(column_names)))
        ax.set_xticklabels(row_names)
        ax.set_yticklabels(column_names)
        ax.tick_params(axis='x', labeltop=True, labelbottom=False)

    fig.subplots_adjust(right=2)  
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    id = np.random.choice(100,1)[0]
    plt.tight_layout()
    plt.savefig(f'crc_folds_{K}_{id})_{method_name}.png', format='png', dpi=300)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size() 
    rank = comm.Get_rank() 

    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])

    if rank == 0:
        print("Processing data...")
        nfolds = 5
        root_path = os.path.join(os.getcwd(), "data/stanford-crc")
        dataset_root = os.path.join(root_path, "output_3hop")
        model_root = os.path.join(root_path, "model_3hop")
        fig_root = os.path.join(model_root, "fig")
        os.makedirs(model_root, exist_ok=True)
        os.makedirs(fig_root, exist_ok=True)

        filenames = sorted(set(f.split('.')[0] for f in os.listdir(dataset_root)))
        filenames = [f for f in filenames if f]
        data_inputs = shuffle_folds(filenames, dataset_root, nfolds)

        X_full = np.vstack([input[1] for input in data_inputs])
        chunks = [[] for _ in range(size)]
        for i, data in enumerate(data_inputs):
            chunks[i % size].append(data)
        del data_inputs
        gc.collect()

    else:
        chunks = None
        X_full = None

    print("Scattering inputs...")
    tasks = comm.scatter(chunks, root=0)

    # Run in each node
    start_time = time.time()
    print(f"Process {rank} calculating alignment score...")

    local_As_gplsi = []
    local_As_plsi = []
    local_As_lda = []
    for task in tasks:
        D_fold, X, edge_df, weights = task

        # run GpLSI
        model_gplsi = gplsi.GpLSI_(
                lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, verbose=0, eps=eps
            )
        model_gplsi.fit(X, K, edge_df, weights)
        local_As_gplsi.append(model_gplsi.A_hat)

        # run pLSI
        model_plsi = gplsi.GpLSI_(
            method='pLSI'
        )
        model_plsi.fit(X.values, K, edge_df, weights)
        local_As_plsi.append(model_plsi.A_hat)
        
        # run LDA
        lda = LatentDirichletAllocation(n_components=K, random_state=0)
        lda.fit(D_fold.values)
        W_hat_lda = lda.transform(D_fold.values)
        A_hat_lda = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        local_As_lda.append(A_hat_lda)

    all_Ahats_gplsi = comm.gather(local_As_gplsi, root=0)
    all_Ahats_plsi = comm.gather(local_As_plsi, root=0)
    all_Ahats_lda = comm.gather(local_As_lda, root=0)

    if rank == 0:
        results = defaultdict(list)
        results_gplsi = get_resolutions(all_Ahats_gplsi, 'gplsi')
        results_plsi = get_resolutions(all_Ahats_plsi, 'plsi')
        results_lda = get_resolutions(all_Ahats_lda, 'lda')

        for key, value in results_gplsi.items():
            results[key].extend(value)

        for key, value in results_plsi.items():
            results[key].extend(value)

        for key, value in results_lda.items():
            results[key].extend(value)

        save_path = os.path.join(model_root, 'crc_chooseK_results.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv('crc_chooseK_results.csv')

        plot_folds(all_Ahats_gplsi, K, 'gplsi')
        plot_folds(all_Ahats_plsi, K, 'plsi')
        plot_folds(all_Ahats_lda, K, 'lda')
