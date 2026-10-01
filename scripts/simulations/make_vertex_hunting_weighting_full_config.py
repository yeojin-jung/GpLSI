#!/usr/bin/env python3
"""Materialize one restartable full-synthetic configuration from scripts/simulations/config.txt.

This prepares, but does not submit, the 42-task published GpLSI grid.  K=2/3
uses source-adaptive SVS selection for SVS*.  Larger K uses the explicitly
labeled fixed L=2K sensitivity because exhaustive adaptive SVS is not a safe
production default at K=5/7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--grid-path", type=Path, default=Path("scripts/simulations/config.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=50)
    args = parser.parse_args()

    grid = pd.read_csv(args.grid_path, sep=r"\s+")
    selected = grid.loc[grid.task_id == args.task_id]
    if selected.empty:
        raise ValueError(f"task_id {args.task_id} is absent from {args.grid_path}")
    row = selected.iloc[0]
    K = int(row.K)
    n = int(row.n)
    nsim = int(row.nsim)
    if K <= 3:
        svs_star_parameters = {
            "L_mode": "mixedscore_adaptive",
            "max_simplexes": 100000,
        }
        svs_star_scope = "source_adaptive_primary"
    else:
        svs_star_parameters = {
            "L_mode": "fixed",
            "L": min(2 * K, n),
            "max_simplexes": 100000,
        }
        svs_star_scope = "fixed_2K_sensitivity_pending_adaptive_feasibility"

    task_root = args.output_root.resolve() / f"task_{args.task_id:02d}"
    config = {
        "schema_version": 1,
        "stage": f"full_synthetic_task_{args.task_id:02d}",
        "output_directory": str(task_root / "results"),
        "figure_directory": str(task_root / "figures"),
        "condition_threshold": 1.0e12,
        "notes": {
            "source_grid": str(args.grid_path),
            "source_task_id": args.task_id,
            "svs_star_scope": svs_star_scope,
            "submission_status": "prepared_not_submitted",
        },
        "graphSVD": {
            "lamb_start": 0.0001,
            "step_size": 1.25,
            "grid_len": 29,
            "maxiter": 50,
            "eps": 1.0e-5,
            "verbose": 0,
        },
        "datasets": [
            {
                "name": f"gplsi_grid_task_{args.task_id:02d}",
                "experiment_family": "gplsi_published_synthetic_grid",
                "generator": "gplsi_current",
                "seeds": list(range(args.start_seed, args.start_seed + nsim)),
                "N": int(row.N),
                "n": n,
                "p": int(row.p),
                "K": K,
                "rt": 0.05,
                "n_clusters": 30,
                "nearest_n": 5,
                "phi": 0.1,
                "generator_method": "strong",
                "graph_setting": "published_current_rbf_knn",
            }
        ],
        "preprocessing_variants": [
            {
                "order": 0,
                "name": "P0",
                "threshold_method": "none",
                "weight_method": "none",
                "initialization": "current",
            },
            {
                "order": 1,
                "name": "P1",
                "threshold_method": "tran",
                "alpha": 0.005,
                "weight_method": "none",
                "initialization": "current",
            },
            {
                "order": 2,
                "name": "P2",
                "threshold_method": "none",
                "weight_method": "ke_empirical",
                "tau": 0.0,
                "weight_common_scale": "none",
                "initialization": "weighted_debiased",
            },
            {
                "order": 3,
                "name": "P3",
                "threshold_method": "tran",
                "alpha": 0.005,
                "weight_method": "ke_empirical",
                "tau": 0.0,
                "weight_common_scale": "none",
                "initialization": "weighted_debiased",
            },
        ],
        "primary_vertex_hunters": [
            {
                "method": "spa_current",
                "embedding_source": "U_hat",
                "parameters": {"precondition": False},
            },
            {
                "method": "svs_star",
                "embedding_source": "U_hat",
                "parameters": svs_star_parameters,
            },
            {
                "method": "pp_spa",
                "embedding_source": "U_hat",
                "parameters": {
                    "radius_divisor": 20.0,
                    "m_neighbors": 4,
                    "min_neighbors": 3,
                },
            },
        ],
        "additional_vertex_runs": [
            {
                "method": "svs_star",
                "embedding_source": "U_bar",
                "preprocessing_variants": ["P0"],
                "parameters": svs_star_parameters,
            },
            {
                "method": "pp_spa",
                "embedding_source": "U_bar",
                "preprocessing_variants": ["P0"],
                "parameters": {
                    "radius_divisor": 20.0,
                    "m_neighbors": 4,
                    "min_neighbors": 3,
                },
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
