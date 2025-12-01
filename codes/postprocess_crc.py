from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

from .utils import get_component_mapping, tuple_converter, _euclidean_proj_simplex

import cvxpy as cp


def preprocess_U(U: np.ndarray, K: int) -> np.ndarray:
    """Make first row of each column in U non-negative."""
    U = U.copy()
    for k in range(K):
        if U[0, k] < 0:
            U[:, k] = -1 * U[:, k]
    return U


def precondition_M(M: np.ndarray, K: int) -> np.ndarray:
    """
    Solve the Klopp preconditioning problem:

        max log_det(Q) s.t. ||Q M_j||_2 <= 1  for all columns j.
    """
    Q = cp.Variable((K, K), symmetric=True)
    objective = cp.Maximize(cp.log_det(Q))
    constraints = [cp.norm(Q @ M, axis=0) <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    Q_value = Q.value
    return Q_value


def preconditioned_spa(U: np.ndarray, K: int, precondition: bool = True):
    """
    SPA with optional preconditioning (as in Klopp et al.).

    Parameters
    ----------
    U : np.ndarray
        Matrix of size (n, K) whose columns correspond to topics/components.
    K : int
        Number of topics/components.
    precondition : bool
        Whether to apply Klopp preconditioning.

    Returns
    -------
    J : list[int]
        Indices of selected anchor rows.
    H_hat : np.ndarray
        Anchor matrix of shape (K, K).
    """
    J: List[int] = []
    M = preprocess_U(U, K).T  # K x n
    if precondition:
        L = precondition_M(M, K)
        S = L @ M
    else:
        S = M

    for _ in range(K):
        maxind = int(np.argmax(norm(S, axis=0)))
        s = np.reshape(S[:, maxind], (K, 1))
        S1 = (np.eye(K) - np.dot(s, s.T) / norm(s) ** 2).dot(S)
        S = S1
        J.append(maxind)

    H_hat = U[J, :]
    return J, H_hat


def get_A_hat_klopp(L: np.ndarray, V: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Recover A_hat using Klopp-style reparametrization:

        theta = (H L) V^T  (then project onto simplex row-wise)
    """
    theta = (H @ L) @ V.T
    theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj


def get_A_hat(W_hat: np.ndarray, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Least-squares projection to recover A_hat given W_hat and X.

    Parameters
    ----------
    W_hat : np.ndarray
        n x K topic proportion matrix.
    X : array-like
        n x p document-term matrix (row-normalized counts).

    Returns
    -------
    A_hat : np.ndarray
        K x p topic loading matrix with rows on simplex (approximately).
    """
    X_arr = np.asarray(X)
    projector = np.linalg.inv(W_hat.T @ W_hat) @ W_hat.T
    theta = projector @ X_arr
    theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj


def align_matrices(
    ntopics_list: list[int],
    matrices_list_A: list[np.ndarray],
    matrices_list_W: list[np.ndarray],
):
    """
    Align topic matrices across increasing K using permutation maps.

    For each consecutive pair (K1, K2), we:
      - find P1 that maps A(K2) to A(K1)
      - embed P1 into a K2 x K2 permutation matrix P
      - use P to reorder A(K2), W(K2).
    """
    for ntopic1, ntopic2 in zip(ntopics_list[:-1], ntopics_list[1:]):
        src = matrices_list_A[ntopic1 - 1]
        tgt = matrices_list_A[ntopic2 - 1]

        P1 = get_component_mapping(tgt, src)
        P = np.zeros((ntopic2, ntopic2))
        P[:, :ntopic1] = P1

        col_ind = np.where(np.all(P == 0, axis=0))[0].tolist()
        row_ind = np.where(np.all(P == 0, axis=1))[0].tolist()
        for i, col in enumerate(col_ind):
            P[row_ind[i], col] = 1

        matrices_list_A[ntopic2 - 1] = P.T @ matrices_list_A[ntopic2 - 1]
        matrices_list_W[ntopic2 - 1] = matrices_list_W[ntopic2 - 1] @ P

    return matrices_list_A, matrices_list_W


def plot_Ahats(
    method: str,
    aligned_A_matrices: Dict[str, List[np.ndarray]],
    ntopics: int,
    fig_root: Path,
) -> None:
    """
    Plot aligned A_hat matrices for a given method across K=1..ntopics.
    """
    column_names = [
        "CD4 T cell",
        "CD8 T cell",
        "B cell",
        "Macrophage",
        "Granulocyte",
        "Blood vessel",
        "Stroma",
        "Other",
    ]

    fig_root.mkdir(parents=True, exist_ok=True)
    save_path = fig_root / f"Ahat_{method}_aligned.png"

    method_A_matrices = aligned_A_matrices[method]

    fig, axes = plt.subplots(1, ntopics, figsize=(22.5, 2))
    vmin = min(matrix.min() for matrix in method_A_matrices)
    vmax = max(matrix.max() for matrix in method_A_matrices)
    k = 1
    for ax, matrix in zip(axes.flatten(), method_A_matrices):
        row_names = [f"T{i}" for i in range(1, k + 1)]
        im = ax.imshow(
            matrix.T,
            cmap="Blues",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(row_names)))
        ax.set_yticks(np.arange(len(column_names)))
        ax.set_xticklabels(row_names)
        ax.set_yticklabels(column_names)
        ax.tick_params(axis="x", labeltop=True, labelbottom=False)
        k += 1

    fig.subplots_adjust(right=2)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def get_survival_data(
    method: str,
    meta: pd.DataFrame,
    coords_all: pd.DataFrame,
    ntopic: int,
    model_root: Path,
) -> None:
    """
    Aggregate topic proportions per region and merge with survival metadata.
    """
    topic_counts = coords_all.groupby(["region_id", f"topics_{method}"]).size()
    topic_proportions = topic_counts.groupby(level=0).transform(lambda x: x / x.sum())

    survival_df = topic_proportions.unstack(fill_value=0)
    survival_df.columns = [f"Topic{col}" for col in survival_df.columns]

    survival_df_merged = survival_df.merge(
        meta[["primary_outcome", "recurrence", "length_of_disease_free_survival"]],
        left_index=True,
        right_index=True,
    )

    model_root.mkdir(parents=True, exist_ok=True)
    out_path = model_root / f"survival_{method}_K={ntopic}.csv"
    survival_df_merged.to_csv(out_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process CRC models: align topics across K and methods, "
            "save aligned matrices, plot A_hat heatmaps, and create survival CSVs."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/stanford-crc"),
        help="Path to CRC data root (contains charville_labels.csv).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output/stanford-crc"),
        help=(
            "Directory containing model outputs (results_crc_K.pkl, "
            "crc_coords_all.csv, crc_D_all.csv). Default: output/stanford-crc"
        ),
    )
    parser.add_argument(
        "--ntopics",
        type=int,
        default=6,
        help="Maximum number of topics K to postprocess (assumes 1..K exist).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_path: Path = args.data_dir
    model_root: Path = args.model_dir
    fig_root: Path = model_root / "fig"

    coords_all = pd.read_csv(model_root / "crc_coords_all.csv", index_col=None)
    D_all = pd.read_csv(model_root / "crc_D_all.csv")
    meta = pd.read_csv(root_path / "charville_labels.csv")

    model_names = ["gplsi", "plsi", "lda"]
    ntopics = args.ntopics # 6
    ntopics_list = [i + 1 for i in range(ntopics)]

    # ------------------------------------------------------------------
    # Load A and W for each K and reconstruct A_hat for GpLSI/pLSI
    # ------------------------------------------------------------------
    matrices_A_gplsi: List[np.ndarray] = []
    matrices_W_gplsi: List[np.ndarray] = []

    matrices_A_plsi: List[np.ndarray] = []
    matrices_W_plsi: List[np.ndarray] = []

    matrices_A_lda: List[np.ndarray] = []
    matrices_W_lda: List[np.ndarray] = []

    # we assume the first column of D_all is CELL_ID
    row_sums = D_all.iloc[:, 1:9].sum(axis=1)
    X = D_all.iloc[:, 1:9].div(row_sums, axis=0)
    print(f"[CRC postprocess] X shape: {X.shape}")

    for i in range(ntopics):
        k = i + 1
        save_path = model_root / f"results_crc_{k}.pkl"
        with open(save_path, "rb") as f:
            results = pickle.load(f)

        # GpLSI
        W = results["W_hat_gplsi"][0]
        matrices_W_gplsi.append(W)
        Ahat = get_A_hat(W, X)
        matrices_A_gplsi.append(Ahat)

        # pLSI
        W = results["W_hat_plsi"][0]
        matrices_W_plsi.append(W)
        J, H_hat = preconditioned_spa(results["plsi"][0].U, k, precondition=False)
        Ahat_plsi = get_A_hat_klopp(
            results["plsi"][0].L,
            results["plsi"][0].V,
            H_hat,
        )
        matrices_A_plsi.append(Ahat_plsi)

        # LDA
        matrices_A_lda.append(results["A_hat_lda"][0])
        matrices_W_lda.append(results["W_hat_lda"][0])

    # ------------------------------------------------------------------
    # Align pLSI and LDA at K=2 to the GpLSI topics at K=2
    # ------------------------------------------------------------------
    src = matrices_A_gplsi[1]  # K=2 index

    tgt = matrices_A_plsi[1]
    P = get_component_mapping(tgt, src)
    matrices_A_plsi[1] = P.T @ matrices_A_plsi[1]
    matrices_W_plsi[1] = matrices_W_plsi[1] @ P

    tgt = matrices_A_lda[1]
    P = get_component_mapping(tgt, src)
    matrices_A_lda[1] = P.T @ matrices_A_lda[1]
    matrices_W_lda[1] = matrices_W_lda[1] @ P

    # ------------------------------------------------------------------
    # Align across K (1..ntopics) for each method
    # ------------------------------------------------------------------
    matrices_A_gplsi_aligned, matrices_W_gplsi_aligned = align_matrices(
        ntopics_list,
        matrices_A_gplsi,
        matrices_W_gplsi,
    )
    matrices_A_plsi_aligned, matrices_W_plsi_aligned = align_matrices(
        ntopics_list,
        matrices_A_plsi,
        matrices_W_plsi,
    )
    matrices_A_lda_aligned, matrices_W_lda_aligned = align_matrices(
        ntopics_list,
        matrices_A_lda,
        matrices_W_lda,
    )

    # ------------------------------------------------------------------
    # Save aligned matrices to CSV + plot heatmaps
    # ------------------------------------------------------------------
    aligned_A_dir = model_root / "Ahats_aligned"
    aligned_W_dir = model_root / "Whats_aligned"
    aligned_A_dir.mkdir(parents=True, exist_ok=True)
    aligned_W_dir.mkdir(parents=True, exist_ok=True)

    aligned_A_matrices = {
        "gplsi": matrices_A_gplsi_aligned,
        "plsi": matrices_A_plsi_aligned,
        "lda": matrices_A_lda_aligned,
    }
    aligned_W_matrices = {
        "gplsi": matrices_W_gplsi_aligned,
        "plsi": matrices_W_plsi_aligned,
        "lda": matrices_W_lda_aligned,
    }

    for method in model_names:
        method_A_dir = aligned_A_dir / f"Ahats_{method}"
        method_W_dir = aligned_W_dir / f"Whats_{method}"
        method_A_dir.mkdir(parents=True, exist_ok=True)
        method_W_dir.mkdir(parents=True, exist_ok=True)

        for k, matrix in enumerate(aligned_A_matrices[method]):
            save_path = method_A_dir / f"Ahat_{method}_{k+1}_aligned.csv"
            pd.DataFrame(matrix).to_csv(save_path, index=False, header=False)

        for k, matrix in enumerate(aligned_W_matrices[method]):
            save_path = method_W_dir / f"What_{method}_{k+1}_aligned.csv"
            pd.DataFrame(matrix).to_csv(save_path, index=False, header=False)

        plot_Ahats(method, aligned_A_matrices, ntopics, fig_root)

    # ------------------------------------------------------------------
    # Extra alignment at K = ntopics (e.g. K=6) across methods
    # ------------------------------------------------------------------
    final_idx = ntopics - 1 
    src = matrices_W_gplsi_aligned[final_idx]

    tgt = matrices_W_plsi_aligned[final_idx]
    P = get_component_mapping(tgt.T, src.T)
    matrices_W_plsi_aligned[final_idx] = matrices_W_plsi_aligned[final_idx] @ P
    matrices_A_plsi_aligned[final_idx] = P @ matrices_A_plsi_aligned[final_idx]

    tgt = matrices_W_lda_aligned[final_idx]
    P = get_component_mapping(tgt.T, src.T)
    matrices_W_lda_aligned[final_idx] = matrices_W_lda_aligned[final_idx] @ P
    matrices_A_lda_aligned[final_idx] = P @ matrices_A_lda_aligned[final_idx]

    # ------------------------------------------------------------------
    # Build survival data CSVs for each K and method
    # ------------------------------------------------------------------
    for i in range(ntopics):
        topics_gplsi = pd.DataFrame(matrices_W_gplsi_aligned[i])
        topics_plsi = pd.DataFrame(matrices_W_plsi_aligned[i])
        topics_lda = pd.DataFrame(matrices_W_lda_aligned[i])

        coords_all_ = coords_all.copy()
        meta_ = meta.copy()

        coords_all_["topics_gplsi"] = topics_gplsi.idxmax(axis=1)
        coords_all_["topics_plsi"] = topics_plsi.idxmax(axis=1)
        coords_all_["topics_lda"] = topics_lda.idxmax(axis=1)
        coords_all_["CELL_TYPE"] = coords_all_["CELL_TYPE"].replace(
            "Tumor 2 (Ki67 Proliferating)",
            "Tumor 2",
        )
        coords_all_["CELL_TYPE"] = coords_all_["CELL_TYPE"].replace(
            "Tumor 6 / DC",
            "Tumor 6",
        )

        # region IDs from D_all
        patient_id = D_all["filename"].unique()
        coords_all_["region_id"] = D_all["filename"]

        meta_.set_index("region_id", inplace=True)

        for method in model_names:
            get_survival_data(
                method=method,
                meta=meta_,
                coords_all=coords_all_,
                ntopic=i + 1,
                model_root=model_root,
            )

    print(f"[CRC postprocess] Done. Outputs in: {model_root}")


if __name__ == "__main__":
    main()