from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from gplsi.realdata_spleen import run_spleen_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GpLSI and baselines on the spleen dataset."
    )

    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Number of topics.",
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
        "--ntumor",
        type=int,
        default=0,
        help=(
            "Index of tumor to analyze (0, 1, or 2) corresponding to "
            "['BALBc-1', 'BALBc-2', 'BALBc-3']. Default: 0"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/spleen"),
        help="Path to spleen data root (containing 'dataset/'). Default: data/spleen",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/spleen"),
        help="Directory where results will be saved. Default: output/spleen",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.1,
        help="Distance-to-weight decay parameter for spleen graph. Default: 0.1",
    )
    parser.add_argument(
        "--difference-penalty",
        type=float,
        default=0.25,
        help="Spatial LDA difference penalty. Default: 0.25",
    )
    parser.add_argument(
        "--n-parallel-processes",
        type=int,
        default=2,
        help="Number of parallel processes for spatial LDA. Default: 2",
    )
    parser.add_argument(
        "--admm-rho",
        type=float,
        default=0.1,
        help="ADMM rho parameter for spatial LDA. Default: 0.1",
    )
    parser.add_argument(
        "--primal-dual-mu",
        type=float,
        default=1e5,
        help="Primal-dual parameter for spatial LDA. Default: 1e5",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="spleen_model_results",
        help="Base filename for saved pickle. Default: spleen_model_results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results, tumor = run_spleen_analysis(
        K=args.K,
        lamb_start=args.lamb_start,
        step_size=args.step_size,
        grid_len=args.grid_len,
        eps=args.eps,
        ntumor=args.ntumor,
        data_root=args.data_dir,
        phi=args.phi,
        difference_penalty=args.difference_penalty,
        n_parallel_processes=args.n_parallel_processes,
        admm_rho=args.admm_rho,
        primal_dual_mu=args.primal_dual_mu,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        args.output_dir
        / f"{tumor}_{args.output_prefix}_K={args.K}.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[INFO] Saved spleen results to: {save_path}")


if __name__ == "__main__":
    main()