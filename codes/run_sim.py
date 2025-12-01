from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd

from gplsi.simulation import run_simulation_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulation experiments for one configuration row "
            "from the simulation grid."
        )
    )

    parser.add_argument(
        "--grid-path",
        type=Path,
        default=Path("codes/config.txt"),
        help=(
            "Path to the simulation grid file (e.g. space-separated text with "
            "columns: task_id, nsim, N, n, K, p). Default: codes/config.txt"
        ),
    )

    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task ID identifying which row in the grid to run.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sim"),
        help="Directory where simulation results will be saved. Default: output/sim",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="sim_results",
        help="Base name for the output file(s). Default: sim_results",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["pkl", "csv", "both"],
        default="pkl",
        help=(
            "Output format: 'pkl' (pickle), 'csv', or 'both'. "
            "Default: pkl"
        ),
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=50,
        help=(
            "Base random seed for this task. Each trial will use "
            "start_seed + trial. Default: 50"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {args.grid_path}")

    grid = pd.read_csv(args.grid_path, sep=r"\s+")

    # ------------------------------------------------------------------
    # Run simulations for this task_id
    # ------------------------------------------------------------------
    results = run_simulation_grid(
        grid=grid,
        task_id=args.task_id,
        start_seed=args.start_seed,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = args.output_dir / f"{args.output_prefix}_task={args.task_id}"

    if args.format in ("pkl", "both"):
        pkl_path = base.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[INFO] Saved pickle results to: {pkl_path}")

    if args.format in ("csv", "both"):
        csv_path = base.with_suffix(".csv")
        results.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV results to: {csv_path}")


if __name__ == "__main__":
    main()