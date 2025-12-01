from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple

import os
import sys
import time

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import LatentDirichletAllocation

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

# Package-local imports
from . import gplsi
from . import generate_topic_model as gen_model
from .utils import (
    _euclidean_proj_simplex,
    get_component_mapping,
    get_F_err,
    get_l1_err,
    get_accuracy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "utils"
if UTILS_DIR.exists():
    sys.path.append(str(UTILS_DIR))

try:
    from spatial_lda import model as spatial_lda_model # type: ignore[import]
except ImportError as e:
    raise ImportError(
        "Could not import 'spatial_lda' from utils/. "
        "Make sure the external spatial LDA code is in utils/spatial_lda."
    ) from e

# download folder 'spatial_lda' from https://github.com/calico/spatial_lda.git

def calculate_norm(matrix: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norm."""
    return norm(matrix, axis=1)


# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------

@dataclass
class SimulationConfig:
    nsim: int
    N: int
    n: int
    K: int
    p: int
    rt: float = 0.05
    n_clusters: int = 30
    m: int = 5
    phi: float = 0.1
    lamb_start: float = 1e-4
    step_size: float = 1.25
    grid_len: int = 29
    eps: float = 1e-5
    start_seed: int | None = None


def run_single_sim(cfg: SimulationConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run nsim simulation trials for a given configuration.

    This compares:
      - GpLSI
      - pLSI
      - TopicScore (R)
      - LDA (sklearn)
      - spatial LDA (Python implementation)

    Returns
    -------
    results_df : pd.DataFrame
        One row per trial with errors, accuracies, and timings.
    models : dict
        Dictionary with lists of fitted model objects for each method.
    """
    # containers
    results: Dict[str, list] = defaultdict(list)
    models: Dict[str, list] = {
        "gplsi": [],
        "plsi": [],
        "tscore": [],
        "lda": [],
        "slda": [],
    }

    # rpy2 setup (one-time)
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

    # where to save models if you want
    model_save_loc = Path(os.getcwd()) / "sim_models"
    file_base = model_save_loc / f"simul_N={cfg.N}_n={cfg.n}_K={cfg.K}_p={cfg.p}"
    extensions = [".csv", ".pkl"]
    model_save_loc.mkdir(parents=True, exist_ok=True)

    for trial in range(cfg.nsim):
        pkl_loc = f"{file_base}_trial={trial}{extensions[1]}"
        os.system(f"echo Running trial {trial}...")

        # ------------------------------------------------------------------
        # Generate topic data and graph
        # ------------------------------------------------------------------
        regens = 0
        if cfg.start_seed is not None:
            np.random.seed(cfg.start_seed + trial)

        while True:
            try:
                coords_df, W, A, X = gen_model.generate_data(
                    cfg.N,
                    cfg.n,
                    cfg.p,
                    cfg.K,
                    cfg.rt,
                    cfg.n_clusters,
                )
                weights, edge_df = gen_model.generate_weights_edge(
                    coords_df,
                    cfg.m,
                    cfg.phi,
                )
                D = cfg.N * X.T
                break
            except Exception as e:  # noqa: BLE001
                print(f"Regenerating dataset due to error: {e}")
                regens += 1

        # ------------------------------------------------------------------
        # TopicScore (R)
        # ------------------------------------------------------------------
        start_time = time.time()
        Mquantile = 0
        K0 = int(np.ceil(1.5 * cfg.K))
        c = min(10 * cfg.K, int(np.ceil(D.shape[0] * 0.7)))
        D_r = robjects.r.matrix(D, nrow=D.shape[0], ncol=D.shape[1])  # R matrix

        norm_score = r["norm_score"]
        A_hat_ts = norm_score(cfg.K, K0, c, D_r)
        A_hat_ts = np.asarray(A_hat_ts)

        M = np.mean(D, axis=1)
        M_trunk = np.minimum(M, np.quantile(M, Mquantile))
        S = np.diag(np.sqrt(1 / M_trunk))
        H = S @ A_hat_ts
        projector = np.linalg.inv(H.T @ H) @ H.T
        theta_W = (projector @ S) @ D
        W_hat_ts = np.array([_euclidean_proj_simplex(x) for x in theta_W.T])
        time_ts = time.time() - start_time
        A_hat_ts = A_hat_ts.T  # K × p

        # ------------------------------------------------------------------
        # pLSI
        # ------------------------------------------------------------------
        start_time = time.time()
        model_plsi = gplsi.GpLSI_(method="pLSI")
        model_plsi.fit(X, cfg.N, cfg.K, edge_df, weights)
        time_plsi = time.time() - start_time

        # ------------------------------------------------------------------
        # GpLSI
        # ------------------------------------------------------------------
        start_time = time.time()
        model_gplsi = gplsi.GpLSI_(
            lamb_start=cfg.lamb_start,
            step_size=cfg.step_size,
            grid_len=cfg.grid_len,
            eps=cfg.eps,
        )
        model_gplsi.fit(X, cfg.N, cfg.K, edge_df, weights)
        time_gplsi = time.time() - start_time
        print(f"CV Lambda is {model_gplsi.lambd}")

        # ------------------------------------------------------------------
        # LDA (sklearn)
        # ------------------------------------------------------------------
        print("Running LDA...")
        start_time = time.time()
        model_lda = LatentDirichletAllocation(
            n_components=cfg.K,
            random_state=0,
        )
        model_lda.fit(D.T)
        W_hat_lda = model_lda.transform(D.T)
        A_hat_lda_unnormalized = model_lda.components_
        A_hat_lda = A_hat_lda_unnormalized / A_hat_lda_unnormalized.sum(
            axis=1,
            keepdims=True,
        )
        time_lda = time.time() - start_time

        # ------------------------------------------------------------------
        # spatial LDA
        # ------------------------------------------------------------------
        start_time = time.time()
        model_slda = spatial_lda_model.run_simulation(X, cfg.K, coords_df)
        A_hat_slda = model_slda.components_
        row_sums = A_hat_slda.sum(axis=1, keepdims=True)
        A_hat_slda = (A_hat_slda / row_sums).T  # p × K transpose later
        time_slda = time.time() - start_time

        # ------------------------------------------------------------------
        # Jump size on edges (graph smoothness)
        # ------------------------------------------------------------------
        M_edges = X[edge_df["src"]] - X[edge_df["tgt"]]
        s = np.sum(calculate_norm(M_edges) * np.sqrt(edge_df["weight"]))

        # ------------------------------------------------------------------
        # Align and compute errors/accuracies
        # ------------------------------------------------------------------
        # W alignment
        P_plsi = get_component_mapping(model_plsi.W_hat.T, W)
        P_gplsi = get_component_mapping(model_gplsi.W_hat.T, W)
        P_ts = get_component_mapping(W_hat_ts.T, W)
        P_lda = get_component_mapping(W_hat_lda.T, W)
        P_slda = get_component_mapping(model_slda.topic_weights.values.T, W)

        W_hat_plsi = model_plsi.W_hat @ P_plsi
        W_hat_gplsi = model_gplsi.W_hat @ P_gplsi
        W_hat_ts = W_hat_ts @ P_ts
        W_hat_lda = W_hat_lda @ P_lda
        W_hat_slda = model_slda.topic_weights.values @ P_slda

        # A alignment
        P_plsi_A = get_component_mapping(model_plsi.A_hat, A.T)
        P_gplsi_A = get_component_mapping(model_gplsi.A_hat, A.T)
        P_ts_A = get_component_mapping(A_hat_ts, A.T)
        P_lda_A = get_component_mapping(A_hat_lda, A.T)
        P_slda_A = get_component_mapping(A_hat_slda.T, A.T)

        A_hat_plsi = P_plsi_A.T @ model_plsi.A_hat
        A_hat_gplsi = P_gplsi_A.T @ model_gplsi.A_hat
        A_hat_ts = P_ts_A.T @ A_hat_ts
        A_hat_lda = P_lda_A.T @ A_hat_lda
        A_hat_slda = P_slda_A.T @ A_hat_slda.T

        # metric helpers
        def _err_acc(W_hat: np.ndarray, A_hat_: np.ndarray) -> list[float]:
            return [
                get_F_err(W_hat, W),
                get_l1_err(W_hat, W),
                get_F_err(A_hat_, A),
                get_l1_err(A_hat_, A),
                get_accuracy(coords_df, cfg.n, W_hat),
            ]

        err_acc_plsi = _err_acc(W_hat_plsi, A_hat_plsi)
        err_acc_gplsi = _err_acc(W_hat_gplsi, A_hat_gplsi)
        err_acc_ts = _err_acc(W_hat_ts, A_hat_ts)
        err_acc_lda = _err_acc(W_hat_lda, A_hat_lda)
        err_acc_slda = _err_acc(W_hat_slda, A_hat_slda)

        # ------------------------------------------------------------------
        # Store metrics
        # ------------------------------------------------------------------
        results["trial"].append(trial)
        results["jump"].append(s / cfg.n)
        results["N"].append(cfg.N)
        results["n"].append(cfg.n)
        results["p"].append(cfg.p)
        results["K"].append(cfg.K)

        results["plsi_err"].append(err_acc_plsi[0])
        results["plsi_l1_err"].append(err_acc_plsi[1])
        results["A_plsi_err"].append(err_acc_plsi[2])
        results["A_plsi_l1_err"].append(err_acc_plsi[3])
        results["plsi_acc"].append(err_acc_plsi[4])

        results["gplsi_err"].append(err_acc_gplsi[0])
        results["gplsi_l1_err"].append(err_acc_gplsi[1])
        results["A_gplsi_err"].append(err_acc_gplsi[2])
        results["A_gplsi_l1_err"].append(err_acc_gplsi[3])
        results["gplsi_acc"].append(err_acc_gplsi[4])
        results["opt_gamma"].append(model_gplsi.lambd)

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
        results["ts_time"].append(time_ts)
        results["lda_time"].append(time_lda)
        results["slda_time"].append(time_slda)

        # store models
        models["plsi"].append(model_plsi)
        models["gplsi"].append(model_gplsi)
        models["tscore"].append(A_hat_ts.T)
        models["lda"].append(model_lda)
        models["slda"].append(model_slda)

    results_df = pd.DataFrame(results)
    return results_df, models


def run_simul(
    nsim: int,
    N: int,
    n: int,
    p: int,
    K: int,
    rt: float = 0.05,
    n_clusters: int = 30,
    m: int = 5,
    phi: float = 0.1,
    lamb_start: float = 1e-4,
    step_size: float = 1.25,
    grid_len: int = 29,
    eps: float = 1e-5,
    start_seed: int | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper around `run_single_sim`.

    Returns only the results DataFrame (drops the models dict).
    """
    cfg = SimulationConfig(
        nsim=nsim,
        N=N,
        n=n,
        K=K,
        p=p,
        rt=rt,
        n_clusters=n_clusters,
        m=m,
        phi=phi,
        lamb_start=lamb_start,
        step_size=step_size,
        grid_len=grid_len,
        eps=eps,
        start_seed=start_seed,
    )
    results_df, _ = run_single_sim(cfg)
    return results_df


# ----------------------------------------------------------------------
# Grid-based runner (for SLURM arrays etc.)
# ----------------------------------------------------------------------

def run_simulation_grid(
    grid: pd.DataFrame,
    task_id: int,
    start_seed: int = 50,
) -> pd.DataFrame:
    """
    Run simulations for one row of the grid (selected by task_id).

    Parameters
    ----------
    grid : pd.DataFrame
        Configuration table with columns:
        ["task_id", "nsim", "N", "n", "K", "p"].
    task_id : int
        The task_id identifying which row of the grid to use.
    start_seed : int, optional
        Base random seed to use for this task. Each trial will use
        start_seed + trial.

    Returns
    -------
    results : pd.DataFrame
        DataFrame of simulation results for this (N, n, K, p) setting.
    """
    row = grid.loc[grid["task_id"] == task_id]
    if row.empty:
        raise ValueError(f"task_id {task_id} not found in grid")

    row = row.iloc[0]

    nsim = int(row["nsim"])
    N = int(row["N"])
    n = int(row["n"])
    K = int(row["K"])
    p = int(row["p"])

    cfg = SimulationConfig(
        nsim=nsim,
        N=N,
        n=n,
        K=K,
        p=p,
        start_seed=start_seed,
    )

    results_df, _ = run_single_sim(cfg)
    results_df["task_id"] = task_id
    return results_df