"""Generate the immutable one-seed pre-refactor GpLSI regression fixture.

Run this script only when intentionally establishing a new baseline.  The
default output path is protected against accidental overwrite.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn

from gplsi.generate_topic_model import generate_data, generate_weights_edge
from gplsi.gplsi import GpLSI
from gplsi.utils import get_F_err, get_component_mapping, get_l1_err


# The untouched estimator calls the NumPy 1.x alias ``np.alltrue``.  Preserve
# that historical behavior in this fixture generator without modifying the
# estimator before the baseline has been captured.
if not hasattr(np, "alltrue"):
    np.alltrue = np.all  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("gplsi_baseline_seed_950.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite baseline fixture: {args.output}")

    config = {
        "seed": 950,
        "N": 40,
        "n": 36,
        "p": 18,
        "K": 3,
        "rt": 0.05,
        "n_clusters": 6,
        "nearest_n": 3,
        "phi": 0.1,
        "lamb_start": 1e-4,
        "step_size": 1.25,
        "grid_len": 3,
        "maxiter": 3,
        "eps": 1e-5,
        "precondition": False,
        "initialize": True,
    }

    np.random.seed(config["seed"])
    coords, W_true_t, A_true_t, X = generate_data(
        config["N"],
        config["n"],
        config["p"],
        config["K"],
        config["rt"],
        config["n_clusters"],
    )
    weights, edge_df = generate_weights_edge(
        coords, config["nearest_n"], config["phi"]
    )

    model = GpLSI(
        lamb_start=config["lamb_start"],
        step_size=config["step_size"],
        grid_len=config["grid_len"],
        maxiter=config["maxiter"],
        eps=config["eps"],
        precondition=config["precondition"],
        initialize=config["initialize"],
    )
    started = time.perf_counter()
    model.fit(X, config["N"], config["K"], edge_df, weights)
    runtime = time.perf_counter() - started

    W_true = W_true_t.T
    A_true = A_true_t.T
    P_w = get_component_mapping(model.W_hat.T, W_true_t)
    P_a = get_component_mapping(model.A_hat, A_true)
    W_aligned = model.W_hat @ P_w
    A_aligned = P_a.T @ model.A_hat
    metrics = {
        "W_frobenius_error": float(get_F_err(W_aligned, W_true_t)),
        "W_l1_error": float(get_l1_err(W_aligned, W_true_t)),
        "A_frobenius_error": float(get_F_err(A_aligned, A_true_t)),
        "A_l1_error": float(get_l1_err(A_aligned, A_true_t)),
        "reconstruction_frobenius": float(np.linalg.norm(X - model.W_hat @ model.A_hat)),
        "selected_rho": float(model.lambd),
        "iterations": int(model.used_iters),
        "runtime_seconds": float(runtime),
    }
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "source_state": "pre-vertex-hunting-weighting-refactor",
        "comparison_atol": 1e-8,
        "comparison_rtol": 1e-7,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        W_true=W_true,
        A_true=A_true,
        U=model.U,
        V=model.V,
        singular_values=np.diag(model.L),
        U_init=model.U_init,
        V_init=model.V_init,
        singular_values_init=np.diag(model.L_init),
        W_hat=model.W_hat,
        A_hat=model.A_hat,
        anchor_indices=np.asarray(model.anchor_indices, dtype=int),
        edge_src=edge_df["src"].to_numpy(dtype=int),
        edge_tgt=edge_df["tgt"].to_numpy(dtype=int),
        edge_weight=edge_df["weight"].to_numpy(dtype=float),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        metrics_json=np.asarray(json.dumps(metrics, sort_keys=True)),
        cv_json=np.asarray(json.dumps(model.lambd_errs, sort_keys=True)),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {"config": config, "metrics": metrics, "provenance": provenance},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
