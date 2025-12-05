from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

from . import gplsi
from .utils import normaliza_coords, dist_to_exp_weight, tuple_converter

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def preprocess_crc(
    coord_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    D: pd.DataFrame,
    phi: float,
) -> Tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame, csr_matrix, int, List[Any]]:
    """
    Normalize CRC coordinates, remap cell IDs to indices, and build graph.

    Parameters
    ----------
    coord_df : pd.DataFrame
        DataFrame with columns including X/Y coordinates and CELL_ID.
    edge_df : pd.DataFrame
        DataFrame with columns ['src', 'tgt', ...], where src/tgt are cell IDs.
    D : pd.DataFrame
        Count matrix with MultiIndex (filename, CELL_ID) and gene columns.
    phi : float
        Decay parameter for distance-to-weight mapping.

    Returns
    -------
    X : pd.DataFrame
        Row-normalized counts (rows sum to 1).
    N : float
        Average row sum of D (used as "document length").
    edge_df_ : pd.DataFrame
        Edge DataFrame with remapped integer indices and weights.
    coord_df : pd.DataFrame
        Coordinate DataFrame with normalized x, y and CELL_ID as integer index.
    weights : csr_matrix
        Sparse matrix of edge weights (n x n).
    n : int
        Number of nodes.
    nodes : list
        List of node IDs (coord_df index).
    """
    # standardize column names X/Y -> x/y
    new_columns = [col.replace("X", "x").replace("Y", "y") for col in coord_df.columns]
    coord_df.columns = new_columns

    # map CELL_ID to integer indices 0..n-1
    cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], range(coord_df.shape[0])))

    # normalize coordinates to (0,1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)

    edge_df_ = edge_df.copy()
    edge_df_["src"] = edge_df["src"].map(cell_to_idx_dict)
    edge_df_["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)

    # distance -> weights
    edge_df_["weight"] = dist_to_exp_weight(edge_df_, coord_df, phi)

    # normalize counts row-wise
    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    N = float(row_sums.mean())
    X = D.div(row_sums, axis=0)
    n = X.shape[0]

    weights = csr_matrix(
        (edge_df_["weight"].values, (edge_df_["src"].values, edge_df_["tgt"].values)),
        shape=(n, n),
    )

    return X, N, edge_df_, coord_df, weights, n, nodes


def run_crc_analysis(
    K: int,
    lamb_start: float,
    step_size: float,
    grid_len: int,
    eps: float,
    data_root: Path,
    phi: float = 0.1,
    min_count: int = 10,
) -> Dict[str, List[Any]]:
    """
    Run GpLSI, pLSI, and LDA on the Stanford CRC dataset.

    This function aggregates all regions with non-missing primary outcome,
    builds a single global graph, and fits the models.

    Parameters
    ----------
    K : int
        Number of topics.
    lamb_start, step_size, grid_len, eps :
        GpLSI hyperparameters (lambda path search).
    data_root : Path
        Path to CRC data root (e.g., 'data/stanford-crc').
    phi : float, optional
        Distance-to-weight decay parameter. Default 0.1.
    min_count : int, optional
        Minimum total count per row to retain. Default 10.

    Returns
    -------
    results : dict
        Dictionary of lists with keys:
        - 'gplsi', 'plsi', 'lda'          (models)
        - 'A_hat_gplsi', 'A_hat_plsi', 'A_hat_lda'
        - 'W_hat_gplsi', 'W_hat_plsi', 'W_hat_lda'
    """
    root_path = data_root
    dataset_root = root_path / "output" / "output_3hop"

    meta_path = root_path / "charville_labels.csv"
    meta = pd.read_csv(meta_path)
    meta_sub = meta[~pd.isna(meta["primary_outcome"])]
    filenames = meta_sub["region_id"].to_list()

    D_all = pd.DataFrame()
    edge_all = pd.DataFrame()
    coords_all = pd.DataFrame()

    # If tuple_converter is defined in data_helpers, you can import and use it.
    # Here we assume index is already parsed as tuples in CSV. If not, adjust.
    total_rows = 0
    for filename in filenames:
        paths = {
            kind: dataset_root / f"{filename}.{kind}.csv"
            for kind in ["D", "edge", "coord", "type", "model"]
        }

        # D: index has (something, CELL_ID)
        D = pd.read_csv(paths["D"], index_col=0, converters={0: tuple_converter})
        D["filename"] = filename

        edge_df = pd.read_csv(paths["edge"], index_col=0)
        coord_df = pd.read_csv(paths["coord"], index_col=0)
        type_df = pd.read_csv(paths["type"], index_col=0)
        coords_df = pd.merge(coord_df, type_df).reset_index(drop=True)

        # global cell index (offset by total_rows)
        cell_to_idx_dict = dict(
            zip(coord_df["CELL_ID"], [i + total_rows for i in range(coord_df.shape[0])])
        )

        edge_df["src"] = edge_df["src"].map(cell_to_idx_dict)
        edge_df["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
        coords_df["CELL_ID"] = coords_df["CELL_ID"].map(cell_to_idx_dict)
        new_index = [(x, cell_to_idx_dict[y]) for x, y in D.index]
        D.index = new_index

        # filter low-count rows (all gene columns except 'filename')
        gene_cols = D.columns[:-1]
        D = D[D[gene_cols].sum(axis=1) >= min_count]

        idx = [y for x, y in D.index]
        edge_df = edge_df[(edge_df["src"].isin(idx)) & (edge_df["tgt"].isin(idx))]
        coords_df = coords_df[coords_df["CELL_ID"].isin(idx)]

        D_all = pd.concat([D_all, D], axis=0, ignore_index=False)
        edge_all = pd.concat([edge_all, edge_df], axis=0, ignore_index=True)
        coords_all = pd.concat([coords_all, coords_df], axis=0, ignore_index=True)

        total_rows += D.shape[0]

    print(f"[CRC] D has {total_rows} rows after filtering.")

    D_all = D_all.drop(columns=["filename"])
    X, N, edge_df, coord_df, weights, n, nodes = preprocess_crc(
        coords_all,
        edge_all,
        D_all,
        phi=phi,
    )

    #patient_id_path = os.path.join(root_path, "selected_patient_id.csv")
    #if not os.path.exists(patient_id_path):
    #      patient_id_df = D_all['filename']
    #      patient_id_df.to_csv(patient_id_path, index=False)

    # ----------------------- GpLSI -------------------------------------
    start_time = time.time()
    model_gplsi = gplsi.GpLSI(
        lamb_start=lamb_start,
        step_size=step_size,
        grid_len=grid_len,
        eps=eps,
    )
    model_gplsi.fit(X.values, N, K, edge_df, weights)
    time_gplsi = time.time() - start_time
    print(f"[CRC] GpLSI done in {time_gplsi:.2f} seconds.")

    # ----------------------- pLSI --------------------------------------
    start_time = time.time()
    model_plsi = gplsi.GpLSI(method="pLSI")
    model_plsi.fit(X.values, N, K, edge_df, weights)
    time_plsi = time.time() - start_time
    print(f"[CRC] pLSI done in {time_plsi:.2f} seconds.")

    # ----------------------- LDA (sklearn) -----------------------------
    start_time = time.time()
    lda = LatentDirichletAllocation(
        n_components=K,
        random_state=0,
    )
    lda.fit(D_all.values)
    time_lda = time.time() - start_time
    print(f"[CRC] LDA done in {time_lda:.2f} seconds.")

    W_hat_lda = lda.transform(D_all.values)
    A_hat_lda = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

    # ----------------------- Collect results ---------------------------
    results: Dict[str, List[Any]] = defaultdict(list)

    # GpLSI
    results["gplsi"].append(model_gplsi)
    results["A_hat_gplsi"].append(model_gplsi.A_hat)
    results["W_hat_gplsi"].append(model_gplsi.W_hat)

    # pLSI
    results["plsi"].append(model_plsi)
    results["A_hat_plsi"].append(model_plsi.A_hat)
    results["W_hat_plsi"].append(model_plsi.W_hat)

    # LDA
    results["lda"].append(lda)
    results["A_hat_lda"].append(A_hat_lda)
    results["W_hat_lda"].append(W_hat_lda)

    # meta info
    results["K"].append(K)
    results["N"].append(N)
    results["n"].append(n)
    results["time_gplsi"].append(time_gplsi)
    results["time_plsi"].append(time_plsi)
    results["time_lda"].append(time_lda)

    return results, D_all, coords_all