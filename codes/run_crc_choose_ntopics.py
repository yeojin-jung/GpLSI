from __future__ import annotations

import argparse
from pathlib import Path
import os
import gc
import pickle
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

from mpi4py import MPI

from gplsi.utils import get_component_mapping, normaliza_coords, dist_to_exp_weight, tuple_converter


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def preprocess_crc(
    coord_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    D: pd.DataFrame,
    phi: float,
):
    """
    Normalize CRC coordinates, map cell IDs to integer indices,
    and build graph + row-normalized X.
    """
    new_columns = [col.replace("X", "x").replace("Y", "y") for col in coord_df.columns]
    coord_df.columns = new_columns

    # map CELL_ID -> 0..n-1
    cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], range(coord_df.shape[0])))

    # normalize coords to (0, 1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)

    # remap edges + weights
    edge_df_ = edge_df.copy()
    edge_df_["src"] = edge_df_["src"].map(cell_to_idx_dict)
    edge_df_["tgt"] = edge_df_["tgt"].map(cell_to_idx_dict)
    edge_df_["weight"] = dist_to_exp_weight(edge_df_, coord_df, phi)

    # row-normalize counts
    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    N = float(row_sums.mean())
    X = D.div(row_sums, axis=0)
    n = X.shape[0]

    weights = csr_matrix(
        (edge_df_["weight"].values, (edge_df_["src"].values, edge_df_["tgt"].values)),
        shape=(n, n),
    )

    return D, X.to_numpy(), N, edge_df_, coord_df, weights, n, nodes


def divide_folds(filenames: list[str], num_parts: int = 2) -> list[list[str]]:
    """
    Randomly shuffle filenames and split into num_parts folds.
    """
    filenames = filenames.copy()
    np.random.shuffle(filenames)

    avg_length = len(filenames) / float(num_parts)
    divided_folds: list[list[str]] = []
    last = 0.0

    while last < len(filenames):
        divided_folds.append(filenames[int(last) : int(last + avg_length)])
        last += avg_length

    return divided_folds


def shuffle_folds(
    filenames: list[str],
    dataset_root: Path,
    nfolds: int,
    phi: float = 0.1,
):
    """
    Build nfolds folds of (D, X, edge_df, weights, N) for cross-validation-like
    evaluation of topic stability.
    """
    data_inputs = []

    divided_folds = divide_folds(filenames, nfolds)

    for i in range(nfolds):
        D_fold = pd.DataFrame()
        edge_fold = pd.DataFrame()
        coords_fold = pd.DataFrame()

        current_set = divided_folds[i]
        s = 0
        for filename in current_set:
            paths = {
                kind: dataset_root / f"{filename}.{kind}.csv"
                for kind in ["D", "edge", "coord", "type", "model"]
            }
            D = pd.read_csv(paths["D"], index_col=0, converters={0: tuple_converter})
            edge_df = pd.read_csv(paths["edge"], index_col=0)
            coord_df = pd.read_csv(paths["coord"], index_col=0)
            type_df = pd.read_csv(paths["type"], index_col=0)
            coords_df = pd.merge(coord_df, type_df).reset_index(drop=True)

            cell_to_idx_dict = dict(
                zip(coord_df["CELL_ID"], [j + s for j in range(coord_df.shape[0])])
            )

            edge_df["src"] = edge_df["src"].map(cell_to_idx_dict)
            edge_df["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
            coords_df["CELL_ID"] = coords_df["CELL_ID"].map(cell_to_idx_dict)
            new_index = [(x, cell_to_idx_dict[y]) for x, y in D.index]
            D.index = new_index

            # keep rows with sufficient counts
            D = D[D.sum(axis=1) >= 10]
            idx = [y for x, y in D.index]
            edge_df = edge_df[(edge_df["src"].isin(idx)) & (edge_df["tgt"].isin(idx))]
            coords_df = coords_df[coords_df["CELL_ID"].isin(idx)]

            D_fold = pd.concat([D_fold, D], axis=0, ignore_index=False)
            edge_fold = pd.concat([edge_fold, edge_df], axis=0, ignore_index=True)
            coords_fold = pd.concat([coords_fold, coords_df], axis=0, ignore_index=True)
            del D, edge_df, coords_df
            gc.collect()

        D_fold, X, N, edge_df, _, weights, _, _ = preprocess_crc(
            coords_fold,
            edge_fold,
            D_fold,
            phi=phi,
        )
        data_inputs.append((D_fold, X, edge_df, weights, N))
        del D_fold, X, edge_df, weights
        gc.collect()

    return data_inputs


def get_resolutions(all_Ahats: list[np.ndarray], method_name: str) -> dict[str, list]:
    """
    Given a list of A_hat matrices (one per fold), compute resolution metrics
    across all pairs of folds:
      - average L1 distance per topic
      - mean diagonal cosine similarity
      - cosine similarity ratio (diag / off-diagonal)
    """
    results: dict[str, list] = defaultdict(list)

    for i, j in combinations(range(len(all_Ahats)), 2):
        A_1 = all_Ahats[i]
        A_2 = all_Ahats[j]

        # align topics in A_2 to A_1
        P = get_component_mapping(A_2, A_1)
        A_1_aligned = P.T @ A_1
        K = A_1_aligned.shape[0]

        # L1 distance
        l1_dist = np.sum(np.abs(A_1_aligned - A_2))

        # cosine similarity
        A_1_norm = A_1_aligned / norm(A_1_aligned, axis=1, keepdims=True)
        A_2_norm = A_2 / norm(A_2, axis=1, keepdims=True)
        cos_mat = A_1_norm @ A_2_norm.T
        diag_cos = float(np.mean(np.diag(cos_mat)))

        # cosine similarity ratio
        off_sum = float(np.sum(cos_mat) - np.sum(np.diag(cos_mat)))
        off_count = K * K - K
        off_cos = off_sum / off_count if off_count > 0 else 0.0
        r = diag_cos / off_cos if off_cos > 0 else np.inf

        results["K"].append(K)
        results["l1_dist"].append(l1_dist / K)
        results["cos_sim"].append(diag_cos)
        results["cos_sim_ratio"].append(r)
        results["method"].append(method_name)

    return results


def plot_folds(
    all_Ahats: list[np.ndarray],
    K: int,
    method_name: str,
    fig_root: Path,
) -> None:
    """
    Plot A_hat matrices across folds as heatmaps, aligned to the first fold.
    """
    fig_root.mkdir(parents=True, exist_ok=True)

    # align all to first fold for visualization
    ref = all_Ahats[0]
    aligned = []
    for A in all_Ahats:
        P = get_component_mapping(A, ref)
        aligned.append(P.T @ A)

    vmin = min(matrix.min() for matrix in aligned)
    vmax = max(matrix.max() for matrix in aligned)

    n_folds = len(aligned)
    fig, axes = plt.subplots(1, n_folds, figsize=(3 * n_folds, 3))
    if n_folds == 1:
        axes = [axes]

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

    for ax, matrix in zip(axes, aligned):
        row_names = [f"T{i}" for i in range(1, K + 1)]
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

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()

    rand_id = int(np.random.choice(100, 1)[0])
    save_path = fig_root / f"crc_folds_K={K}_id={rand_id}_{method_name}.png"
    plt.savefig(save_path, format="png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose number of topics K for CRC via cross-fold stability "
            "using GpLSI, pLSI, and LDA with MPI."
        )
    )

    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Number of topics to evaluate.",
    )
    parser.add_argument(
        "--lamb-start",
        type=float,
        default=1e-4,
        help="Initial lambda for GpLSI path search. Default: 1e-4",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.25,
        help="Multiplicative step size for lambda grid. Default: 1.25",
    )
    parser.add_argument(
        "--grid-len",
        type=int,
        default=29,
        help="Number of lambda grid points. Default: 29",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Convergence tolerance for GpLSI. Default: 1e-5",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=5,
        help="Number of folds to split regions into. Default: 5",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/stanford-crc"),
        help="Path to CRC data root (contains 'output/output_3hop').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/stanford-crc"),
        help="Directory where outputs (csv, figs) will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    args = parse_args()
    K = args.K

    # ------------------------------------------------------------------
    # Rank 0: prepare data folds and scatter tasks
    # ------------------------------------------------------------------
    if rank == 0:
        print("[MPI] Preparing data folds...")
        root_path = args.data_dir
        dataset_root = root_path / "output" / "output_3hop"
        model_root = args.output_dir
        fig_root = model_root / "fig"
        model_root.mkdir(parents=True, exist_ok=True)
        fig_root.mkdir(parents=True, exist_ok=True)

        filenames = sorted({f.split(".")[0] for f in os.listdir(dataset_root) if f})
        data_inputs = shuffle_folds(
            filenames,
            dataset_root=dataset_root,
            nfolds=args.nfolds,
            phi=0.1,
        )

        # vertical stack of all X, not strictly needed but kept for reference
        X_full = np.vstack([entry[1] for entry in data_inputs])
        print(f"[MPI] Total rows across folds: {X_full.shape[0]}")

        # assign folds to processes
        chunks: list[list[tuple]] = [[] for _ in range(size)]
        for i, data in enumerate(data_inputs):
            chunks[i % size].append(data)

        del data_inputs
        gc.collect()
    else:
        chunks = None
        model_root = None
        fig_root = None

    model_root = comm.bcast(model_root, root=0)
    fig_root = comm.bcast(fig_root, root=0)

    print(f"[MPI] Rank {rank} scattering inputs...")
    tasks = comm.scatter(chunks, root=0)

    # ------------------------------------------------------------------
    # Each rank: run models on its assigned folds
    # ------------------------------------------------------------------
    from gplsi import gplsi as gplsi_mod 

    local_As_gplsi: list[np.ndarray] = []
    local_As_plsi: list[np.ndarray] = []
    local_As_lda: list[np.ndarray] = []

    print(f"[MPI] Rank {rank} starting computation on {len(tasks)} folds...")
    for D_fold, X, edge_df, weights, N in tasks:
        # GpLSI
        model_gplsi = gplsi_mod.GpLSI_(
            lamb_start=args.lamb_start,
            step_size=args.step_size,
            grid_len=args.grid_len,
            eps=args.eps,
        )
        model_gplsi.fit(X, N, K, edge_df, weights)
        local_As_gplsi.append(model_gplsi.A_hat)

        # pLSI
        model_plsi = gplsi_mod.GpLSI_(method="pLSI")
        model_plsi.fit(X, N, K, edge_df, weights)
        local_As_plsi.append(model_plsi.A_hat)

        # LDA
        lda = LatentDirichletAllocation(n_components=K, random_state=0)
        lda.fit(D_fold.values)
        A_hat_lda = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
        local_As_lda.append(A_hat_lda)

    # gather all A_hats at rank 0
    all_Ahats_gplsi = comm.gather(local_As_gplsi, root=0)
    all_Ahats_plsi = comm.gather(local_As_plsi, root=0)
    all_Ahats_lda = comm.gather(local_As_lda, root=0)

    # ------------------------------------------------------------------
    # Rank 0: aggregate results, compute resolutions, and plot
    # ------------------------------------------------------------------
    if rank == 0:
        print("[MPI] Gathering and computing resolutions...")

        flat_gplsi = [A for worker in all_Ahats_gplsi for A in worker]
        flat_plsi = [A for worker in all_Ahats_plsi for A in worker]
        flat_lda = [A for worker in all_Ahats_lda for A in worker]

        results: dict[str, list] = defaultdict(list)

        results_gplsi = get_resolutions(flat_gplsi, "gplsi")
        results_plsi = get_resolutions(flat_plsi, "plsi")
        results_lda = get_resolutions(flat_lda, "lda")

        for key, value in results_gplsi.items():
            results[key].extend(value)
        for key, value in results_plsi.items():
            results[key].extend(value)
        for key, value in results_lda.items():
            results[key].extend(value)

        results_df = pd.DataFrame(results)
        csv_path = model_root / f"crc_chooseK_results_{K}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"[MPI] Saved resolution metrics to {csv_path}")

        plot_folds(flat_gplsi, K, "gplsi", fig_root)
        plot_folds(flat_plsi, K, "plsi", fig_root)
        plot_folds(flat_lda, K, "lda", fig_root)
        print(f"[MPI] Plots saved under {fig_root}")


if __name__ == "__main__":
    main()