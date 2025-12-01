from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from gplsi.realdata_crc import run_crc_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GpLSI, pLSI, and LDA on the Stanford CRC dataset."
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
        "--data-dir",
        type=Path,
        default=Path("data/stanford-crc"),
        help="Path to CRC data root (contains 'output/output_3hop', labels CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/stanford-crc"),
        help="Directory where model results will be saved. Default: output/stanford-crc",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.1,
        help="Distance-to-weight decay parameter. Default: 0.1",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum total count per row to retain. Default: 10",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="results_crc",
        help="Base filename for saved pickle. Default: results_crc",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results, D_all, coords_all = run_crc_analysis(
        K=args.K,
        lamb_start=args.lamb_start,
        step_size=args.step_size,
        grid_len=args.grid_len,
        eps=args.eps,
        data_root=args.data_dir,
        phi=args.phi,
        min_count=args.min_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"{args.output_prefix}_{args.K}.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    
    D_all.to_csv(args.output_dir / "crc_D_all.csv", index=False)
    coords_all.to_csv(args.output_dir / "crc_coords_all.csv", index=False)

    print(f"[INFO] Saved CRC results to: {save_path}")


if __name__ == "__main__":
    main()