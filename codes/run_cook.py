from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from gplsi.realdata_cook import run_cook_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GpLSI, pLSI, and LDA on the WhatsCooking dataset."
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
        default=Path("data/whats-cooking"),
        help="Path to WhatsCooking data root (contains 'dataset/').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/whats-cooking"),
        help="Directory where model results will be saved.",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="If set, use ingredient frequency thresholding in preprocessing.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="cook",
        help="Base filename prefix for saved pickles. Default: 'cook'",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results, models = run_cook_analysis(
        K=args.K,
        lamb_start=args.lamb_start,
        step_size=args.step_size,
        grid_len=args.grid_len,
        eps=args.eps,
        data_root=args.data_dir,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual models
    for name in ["gplsi", "plsi", "lda"]:
        model = models.get(name)
        if model is not None:
            model_path = (
                args.output_dir / f"{args.output_prefix}_model_{name}_K={args.K}.pkl"
            )
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[Cook] Saved {name} model to: {model_path}")

    # Save combined results
    results_path = (
        args.output_dir / f"{args.output_prefix}_model_results_K={args.K}.pkl"
    )
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[Cook] Saved results to: {results_path}")


if __name__ == "__main__":
    main()