import sys
import os
import gc
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([
    os.path.join(parent_dir, "pycvxcluster"),
    parent_dir 
])

from GpLSI.utils import *
from utils.data_helpers import *


def _euclidean_proj_simplex(v, s=1):
        (n,) = v.shape
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w

def get_A_hat(W_hat, M):
        projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
        theta = projector.dot(M)
        theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

def align_matrices(ntopics_list, matrices_list_A, matrices_list_W):
    for ntopic1, ntopic2 in zip(ntopics_list[:-1], ntopics_list[1:]):
            src = matrices_list_A[ntopic1-1]
            tgt = matrices_list_A[ntopic2-1]
            P1 = get_component_mapping(tgt, src)
            P = np.zeros((ntopic2, ntopic2))
            P[:, :ntopic1] = P1
            col_ind = np.where(np.all(P == 0, axis=0))[0].tolist()
            row_ind = np.where(np.all(P == 0, axis=1))[0].tolist()
            for i, col in enumerate(col_ind):
                P[row_ind[i], col] = 1
            matrices_list_A[ntopic2-1] = (
                P.T @ matrices_list_A[ntopic2-1]
            )
            matrices_list_W[ntopic2-1] = (
                matrices_list_W[ntopic2-1] @ P
            )
    return matrices_list_A, matrices_list_W

def plot_Ahats(method, aligned_A_matrices, ntopics, fig_root):
    column_names = ['CD4 T cell', 'CD8 T cell', 'B cell', 'Macrophage', 'Granulocyte', 'Blood vessel', 'Stroma', 'Other']
    save_path = os.path.join(fig_root, f'Ahat_{method}_aligned.png')
    method_A_matrices = aligned_A_matrices[method]

    fig, axes = plt.subplots(1, ntopics, figsize=(22.5, 2))
    vmin = min(matrix.min() for matrix in method_A_matrices)
    vmax = max(matrix.max() for matrix in method_A_matrices)
    k = 1
    for ax, matrix in zip(axes.flatten(), method_A_matrices):
        row_names = ["T"+str(i) for i in range(1, k+1)]
        im = ax.imshow(matrix.T, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(row_names)))
        ax.set_yticks(np.arange(len(column_names)))
        ax.set_xticklabels(row_names)
        ax.set_yticklabels(column_names)
        ax.tick_params(axis='x', labeltop=True, labelbottom=False)
        k +=1

    fig.subplots_adjust(right=2)  
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def get_survival_data(method, meta, coords_all):
    topic_counts = coords_all.groupby(['region_id', f'topics_{method}']).size()
    topic_proportions = topic_counts.groupby(level=0).transform(lambda x: x / x.sum())

    survival_df = topic_proportions.unstack(fill_value=0)
    survival_df.columns = [f'Topic{col}' for col in survival_df.columns]

    survival_df_merged = survival_df.merge(meta[['primary_outcome', 'recurrence', 'length_of_disease_free_survival']], left_index=True, right_index=True)
    survival_df_merged.to_csv(os.path.join(model_root, f'survival_{method}.csv'))


if __name__ == "__main__":
    root_path = os.path.join(parent_dir, "data/stanford-crc")
    model_root = os.path.join(parent_dir, "output/stanford-crc")
    fig_root = os.path.join(model_root, "fig")
    
    coords_all = pd.read_csv(os.path.join(model_root, 'crc_coords_all.csv'), index_col=None)
    D_all = pd.read_csv(os.path.join(model_root, 'crc_D_all.csv'))
    meta = pd.read_csv(os.path.join(root_path, 'charville_labels.csv'))

    model_names = ['gplsi', 'plsi', 'lda']
    ntopics = 6
    ntopics_list = [i+1 for i in range(ntopics)]

    # load A and W
    matrices_A_gplsi = []
    matrices_W_gplsi = []

    matrices_A_plsi = []
    matrices_W_plsi = []

    matrices_A_lda = []
    matrices_W_lda = []

    for i in range(ntopics):
        k = i+1
        save_path = os.path.join(model_root, f'results_crc_{k}.pkl')
        with open(save_path, "rb") as f:
            results = pickle.load(f)

        row_sums = D_all.iloc[:, 1:9].sum(axis=1)
        X = D_all.iloc[:, 1:9].div(row_sums, axis=0)
        print(X.shape)

        W = results['W_hat_gplsi'][0]
        matrices_W_gplsi.append(W)
        Ahat = get_A_hat(W, X)
        matrices_A_gplsi.append(Ahat)
        
        W = results['W_hat_plsi'][0]
        matrices_W_plsi.append(W)
        Ahat = get_A_hat(W, X)
        matrices_A_plsi.append(Ahat)

        matrices_A_lda.append(results['A_hat_lda'][0])
        matrices_W_lda.append(results['W_hat_lda'][0])

    # align pLSI, LDA with GpLSI
    src = matrices_A_gplsi[1]

    tgt = matrices_A_plsi[1]
    P = get_component_mapping(tgt, src)
    matrices_A_plsi[1] = (
        P.T @ matrices_A_plsi[1]
            )
    matrices_W_plsi[1] = (
        matrices_W_plsi[1] @ P
    )
    
    tgt = matrices_A_lda[1]
    P = get_component_mapping(tgt, src)
    matrices_A_lda[1] = (
        P.T @ matrices_A_lda[1]
            )
    matrices_W_lda[1] = (
        matrices_W_lda[1] @ P
    )
    
    matrices_A_gplsi_aligned, matrices_W_gplsi_aligned = align_matrices(ntopics_list, matrices_A_gplsi, matrices_W_gplsi)
    matrices_A_plsi_aligned, matrices_W_plsi_aligned = align_matrices(ntopics_list, matrices_A_plsi, matrices_W_plsi)
    matrices_A_lda_aligned, matrices_W_lda_aligned = align_matrices(ntopics_list, matrices_A_lda, matrices_W_lda)

    # save aligned matrices
    aligned_A_dir = os.path.join(model_root, 'Ahats_aligned')
    os.makedirs(aligned_A_dir, exist_ok=True)
    aligned_W_dir = os.path.join(model_root, 'Whats_aligned')
    os.makedirs(aligned_W_dir, exist_ok=True)

    methods = ['gplsi', 'plsi', 'lda']
    aligned_A_matrices = {
        'gplsi': matrices_A_gplsi_aligned,
        'plsi': matrices_A_plsi_aligned,
        'lda': matrices_A_lda_aligned}
    aligned_W_matrices = {
        'gplsi': matrices_W_gplsi_aligned,
        'plsi': matrices_W_plsi_aligned,
        'lda': matrices_W_lda_aligned}
    
    for method in methods:   
        method_A_dir = os.path.join(aligned_A_dir, f'Ahats_{method}')
        os.makedirs(method_A_dir, exist_ok=True)
        method_W_dir = os.path.join(aligned_W_dir, f'Whats_{method}')
        os.makedirs(method_W_dir, exist_ok=True)
        
        for k, matrix in enumerate(aligned_A_matrices[method]):
            save_path = os.path.join(method_A_dir, f'Ahat_{method}_{k+1}_aligned.csv')
            matrix = pd.DataFrame(matrix)
            matrix.to_csv(save_path, index=False, header=False)
        for k, matrix in enumerate(aligned_W_matrices[method]): 
            save_path = os.path.join(method_W_dir, f'What_{method}_{k+1}_aligned.csv')
            matrix = pd.DataFrame(matrix)
            matrix.to_csv(save_path, index=False, header=False)
        plot_Ahats(method, aligned_A_matrices, 6, fig_root)
    
    src = matrices_W_gplsi_aligned[5]
    tgt = matrices_W_plsi_aligned[5]
    P = get_component_mapping(tgt.T, src.T)
    matrices_W_plsi_aligned[5] = (
        matrices_W_plsi_aligned[5] @ P
            )
    matrices_A_plsi_aligned[5] = (
        P @ matrices_A_plsi_aligned[5]
            )

    src = matrices_W_gplsi_aligned[5]
    tgt = matrices_W_lda_aligned[5]
    P = get_component_mapping(tgt.T, src.T)
    matrices_W_lda_aligned[5] = (
        matrices_W_lda_aligned[5] @ P
            )
    matrices_A_lda_aligned[5] = (
        P @ matrices_A_lda_aligned[5]
            )
    
    topics_gplsi = pd.DataFrame(matrices_W_gplsi_aligned[5])
    topics_plsi = pd.DataFrame(matrices_W_plsi_aligned[5])
    topics_lda = pd.DataFrame(matrices_W_lda_aligned[5])

    coords_all['topics_gplsi'] = topics_gplsi.idxmax(axis=1)
    coords_all['topics_plsi'] = topics_plsi.idxmax(axis=1)
    coords_all['topics_lda'] = topics_lda.idxmax(axis=1)
    coords_all['CELL_TYPE'] = coords_all['CELL_TYPE'].replace('Tumor 2 (Ki67 Proliferating)', 'Tumor 2')
    coords_all['CELL_TYPE'] = coords_all['CELL_TYPE'].replace('Tumor 6 / DC', 'Tumor 6')

    patient_id = D_all['filename'].unique()
    coords_all['region_id'] = D_all['filename']

    meta.set_index('region_id', inplace=True)

    for method in model_names:
        get_survival_data(method, meta, coords_all)