from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

# Package-local imports
from . import gplsi
from .utils import (
    normaliza_coords, 
    dist_to_exp_weight,
    _euclidean_proj_simplex,
    moran,
    get_CHAOS,
    get_PAS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "utils"
if UTILS_DIR.exists():
    sys.path.append(str(UTILS_DIR))

try:
    from spatial_lda import model as spatial_lda_model  # type: ignore[import]
    from spatial_lda.featurization import (  # type: ignore[import]
        make_merged_difference_matrices,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Could not import 'spatial_lda' from utils/. "
        "Make sure external spatial LDA code is in utils/spatial_lda/."
    ) from e

# download folder 'spatial_lda' from https://github.com/calico/spatial_lda.git

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def preprocess_spleen(
    coord_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    D: pd.DataFrame,
    phi: float,
) -> Tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame, csr_matrix, int, List[Any]]:
    """
    Normalize coordinates, compute edge weights, and return features / graph.

    Parameters
    ----------
    coord_df : pd.DataFrame
        DataFrame with columns ['x', 'y'] indexed by node IDs.
    edge_df : pd.DataFrame
        DataFrame with columns ['src', 'tgt', 'distance'].
    D : pd.DataFrame
        Count matrix, rows = cells/nodes, columns = features.
    phi : float
        Decay parameter for distance-to-weight mapping.

    Returns
    -------
    X : pd.DataFrame
        Row-normalized feature matrix (rows sum to 1).
    N : float
        Average row sum of D (used as "document length").
    edge_df : pd.DataFrame
        Edge dataframe with 'weight' column added.
    coord_df : pd.DataFrame
        Normalized coordinates (in [0,1]).
    weights : csr_matrix
        Sparse matrix of edge weights (n x n).
    n : int
        Number of nodes.
    nodes : list
        List of node IDs (coord_df index).
    """
    # normalize coordinates to (0, 1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)

    # distance -> exponential weights
    edge_df["weight"] = dist_to_exp_weight(edge_df, coord_df, phi)

    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    N = float(row_sums.mean())
    X = D.div(row_sums, axis=0)  # normalize rows
    n = X.shape[0]

    weights = csr_matrix(
        (edge_df["weight"].values, (edge_df["src"].values, edge_df["tgt"].values)),
        shape=(n, n),
    )

    return X, N, edge_df, coord_df, weights, n, nodes

def run_spleen_analysis(
    K: int,
    lamb_start: float,
    step_size: float,
    grid_len: int,
    eps: float,
    ntumor: int,
    data_root: Path,
    phi: float = 0.1,
    difference_penalty: float = 0.25,
    n_parallel_processes: int = 2,
    admm_rho: float = 0.1,
    primal_dual_mu: float = 1e5,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Run GpLSI + baselines on a single spleen tumor.

    Parameters
    ----------
    K : int
        Number of topics.
    lamb_start, step_size, grid_len, eps :
        Hyperparameters for GpLSI path search.
    ntumor : int
        Index into the tumor list (0, 1, 2) -> ["BALBc-1", "BALBc-2", "BALBc-3"].
    data_root : Path
        Path to spleen data root (containing 'dataset/').
        For example: Path('data/spleen').
    phi : float, optional
        Distance-to-weight decay parameter. Default 0.1.
    difference_penalty : float, optional
        Penalty for spatial LDA. Default 0.25.
    n_parallel_processes : int, optional
        Number of parallel processes for spatial LDA. Default 2.
    admm_rho : float, optional
        ADMM rho parameter for spatial LDA. Default 0.1.
    primal_dual_mu : float, optional
        Primal-dual parameter for spatial LDA. Default 1e5.

    Returns
    -------
    results : list of dict
        Single-element list containing a dict with W/A, metrics, etc.
    tumor : str
        Tumor name used (e.g., 'BALBc-1').
    """
    # ----------------------- load data ---------------------------------
    dataset_root = data_root / "dataset"
    path_to_D = dataset_root / "merged_D.pkl"
    path_to_edge = dataset_root / "merged_data.pkl"
    path_to_coord = dataset_root / "merged_coord.pkl"

    tumors = ["BALBc-1", "BALBc-2", "BALBc-3"]
    tumor = tumors[ntumor]

    spleen_D = pd.read_pickle(path_to_D).loc[tumor]
    spleen_edge = pd.read_pickle(path_to_edge).loc[tumor]
    spleen_coord = pd.read_pickle(path_to_coord).loc[tumor]

    spleen_coord.columns = ["x", "y"]
    spleen_edge.columns = ["src", "tgt", "distance"]

    X, N, edge_df, coord_df, weights, n, nodes = preprocess_spleen(
        spleen_coord,
        spleen_edge,
        spleen_D,
        phi=phi,
    )

    # ----------------------- TopicScore (R) -----------------------------
    numpy2ri.activate()

    rpackages.importr("nnls")
    rpackages.importr("rARPACK")
    rpackages.importr("quadprog")
    rpackages.importr("Matrix")

    r = robjects.r

    topicscore_path = REPO_ROOT / "utils" / "topicscore.R"
    if not topicscore_path.exists():
        raise FileNotFoundError(
            f"topicscore.R not found at {topicscore_path}. "
            "Make sure the external TopicScore code is in utils/topicscore.R."
        )

    r["source"](str(topicscore_path))
    norm_score = r["norm_score"]

    D = spleen_D.values.T  # p × n
    start_time = time.time()
    Mquantile = 0
    K0 = int(np.ceil(1.5 * K))
    c = min(10 * K, int(np.ceil(D.shape[0] * 0.7)))

    D_r = robjects.r.matrix(D, nrow=D.shape[0], ncol=D.shape[1])  # R matrix
    A_hat_ts = np.asarray(norm_score(K, K0, c, D_r))
    M = np.mean(D, axis=1)
    M_trunk = np.minimum(M, np.quantile(M, Mquantile))
    S = np.diag(np.sqrt(1 / M_trunk))
    H = S @ A_hat_ts
    projector = np.linalg.inv(H.T @ H) @ H.T
    theta_W = (projector @ S) @ D
    W_hat_ts = np.array([_euclidean_proj_simplex(x) for x in theta_W.T])
    time_ts = time.time() - start_time
    A_hat_ts = A_hat_ts.T  # K × p

    # ----------------------- GpLSI -------------------------------------
    start_time = time.time()
    model_gplsi = gplsi.GpLSI_(
        lamb_start=lamb_start,
        step_size=step_size,
        grid_len=grid_len,
        eps=eps,
    )
    model_gplsi.fit(X.values, N, K, edge_df, weights)
    time_gplsi = time.time() - start_time

    # ----------------------- pLSI --------------------------------------
    start_time = time.time()
    model_plsi = gplsi.GpLSI_(method="pLSI")
    model_plsi.fit(X.values, N, K, edge_df, weights)
    time_plsi = time.time() - start_time

    # ----------------------- LDA (sklearn) -----------------------------
    start_time = time.time()
    model_lda = LatentDirichletAllocation(
        n_components=K,
        random_state=0,
    )
    model_lda.fit(spleen_D.values)
    time_lda = time.time() - start_time

    # ----------------------- spatial LDA -------------------------------
    cell_org = [x[1] for x in X.index]
    cell_dict = dict(zip(range(len(X)), cell_org))
    samp_coord_ = coord_df.copy()
    samp_coord_.index = coord_df.index.map(cell_dict)
    samp_coord__ = {tumor: samp_coord_}

    difference_matrices = make_merged_difference_matrices(
        X,
        samp_coord__,
        "x",
        "y",
    )
    print("Running SLDA...")
    start_time = time.time()
    model_slda = spatial_lda_model.train(
        sample_features=X,
        difference_matrices=difference_matrices,
        difference_penalty=difference_penalty,
        n_topics=K,
        n_parallel_processes=n_parallel_processes,
        verbosity=1,
        admm_rho=admm_rho,
        primal_dual_mu=primal_dual_mu,
    )
    time_slda = time.time() - start_time

    # ----------------------- Metrics -----------------------------------
    results: List[Dict[str, Any]] = []

    # W matrices
    W_gplsi = model_gplsi.W_hat
    W_plsi = model_plsi.W_hat
    W_ts = W_hat_ts
    W_lda = model_lda.transform(spleen_D.values)
    W_slda = model_slda.topic_weights.values

    # Moran, CHAOS, PAS for each method
    gmoran_gplsi, moran_gplsi = moran(W_gplsi, edge_df)
    gchaos_gplsi, chaos_gplsi = get_CHAOS(W_gplsi, nodes, coord_df, n, K)
    pas_gplsi = get_PAS(W_gplsi, edge_df)

    gmoran_plsi, moran_plsi = moran(W_plsi, edge_df)
    gchaos_plsi, chaos_plsi = get_CHAOS(W_plsi, nodes, coord_df, n, K)
    pas_plsi = get_PAS(W_plsi, edge_df)

    gmoran_ts, moran_ts = moran(W_ts, edge_df)
    gchaos_ts, chaos_ts = get_CHAOS(W_ts, nodes, coord_df, n, K)
    pas_ts = get_PAS(W_ts, edge_df)

    gmoran_lda, moran_lda = moran(W_lda, edge_df)
    gchaos_lda, chaos_lda = get_CHAOS(W_lda, nodes, coord_df, n, K)
    pas_lda = get_PAS(W_lda, edge_df)

    gmoran_slda, moran_slda = moran(W_slda, edge_df)
    gchaos_slda, chaos_slda = get_CHAOS(W_slda, nodes, coord_df, n, K)
    pas_slda = get_PAS(W_slda, edge_df)

    # A matrices
    A_hat_gplsi = model_gplsi.A_hat.T
    A_hat_plsi = model_plsi.A_hat.T
    A_hat_lda = model_lda.components_ / model_lda.components_.sum(axis=1)[
        :, np.newaxis
    ]
    A_hat_slda = model_slda.components_
    row_sums = A_hat_slda.sum(axis=1, keepdims=True)
    A_hat_slda = (A_hat_slda / row_sums).T  # K × p

    names = ["GpLSI", "pLSI", "TopicScore", "LDA", "SLDA"]
    morans = [gmoran_gplsi, gmoran_plsi, gmoran_ts, gmoran_lda, gmoran_slda]
    chaoss = [gchaos_gplsi, gchaos_plsi, gchaos_ts, gchaos_lda, gchaos_slda]
    pas = [pas_gplsi, pas_plsi, pas_ts, pas_lda, pas_slda]
    times = [time_gplsi, time_plsi, time_ts, time_lda, time_slda]
    Whats = [W_gplsi, W_plsi, W_ts, W_lda, W_slda]
    Ahats = [A_hat_gplsi, A_hat_plsi, A_hat_ts, A_hat_lda, A_hat_slda]

    result = {
        "names": names,
        "Whats": Whats,
        "Ahats": Ahats,
        "chaoss": chaoss,
        "morans": morans,
        "pas": pas,
        "times": times,
        "coord_df": coord_df,
        "edge_df": edge_df,
        "tumor": tumor,
        "K": K,
    }
    results.append(result)

    return results, tumor