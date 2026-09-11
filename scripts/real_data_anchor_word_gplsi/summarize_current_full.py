#!/usr/bin/env python3
"""Summarize completed full real-data GpLSI task directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC = "heldout_metrics.heldout_multinomial_deviance"
COUNT = "heldout_metrics.heldout_count_total"


def load_completed(result_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    complete_files = [
        path
        for path in result_root.rglob("complete.json")
        if "/full_" in path.as_posix()
    ]
    fit_files = [path.parent / "fit_rows.csv" for path in complete_files]
    missing = [path for path in fit_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fit_rows.csv files: {missing[:3]}")
    if not fit_files:
        raise FileNotFoundError(f"No completed full tasks under {result_root}")
    return pd.concat(
        [pd.read_csv(path, low_memory=False) for path in fit_files],
        ignore_index=True,
    ), complete_files


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def paired_palm(ok: pd.DataFrame) -> pd.DataFrame:
    methods = ok[ok["vertex_hunter"].isin(["palm", "palm_accelerated"])].copy()
    keys = [
        "dataset",
        "biological_group",
        "K",
        "seed",
        "estimator_family",
        "spectral_geometry",
        "preprocessing",
        "A_recovery",
    ]
    methods["dev_per_count"] = methods[METRIC] / methods[COUNT]
    metric = methods.pivot_table(
        index=keys,
        columns="vertex_hunter",
        values="dev_per_count",
        aggfunc="first",
        dropna=False,
    ).dropna(subset=["palm", "palm_accelerated"])
    runtime = methods.pivot_table(
        index=keys,
        columns="vertex_hunter",
        values="runtime_vertex_hunting_seconds",
        aggfunc="first",
        dropna=False,
    ).dropna(subset=["palm", "palm_accelerated"])
    paired = metric.join(runtime, lsuffix="_dev", rsuffix="_runtime", how="inner").reset_index()
    paired["accelerated_minus_palm_dev_per_count"] = (
        paired["palm_accelerated_dev"] - paired["palm_dev"]
    )
    paired["palm_over_accelerated_runtime"] = (
        paired["palm_runtime"] / paired["palm_accelerated_runtime"]
    )
    rows = []
    for dataset, group in paired.groupby("dataset"):
        delta = group["accelerated_minus_palm_dev_per_count"]
        ratio = group["palm_over_accelerated_runtime"].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "dataset": dataset,
                "n_pairs": len(group),
                "accelerated_deviance_win_fraction": float((delta < 0).mean()),
                "accelerated_deviance_tie_fraction_1e-10": float(
                    np.isclose(delta, 0.0, atol=1e-10, rtol=0.0).mean()
                ),
                "mean_accelerated_minus_palm_dev_per_count": float(delta.mean()),
                "median_accelerated_minus_palm_dev_per_count": float(delta.median()),
                "median_palm_over_accelerated_runtime": float(ratio.median()),
                "accelerated_runtime_win_fraction": float((ratio > 1).mean()),
                "median_palm_runtime_seconds": float(group["palm_runtime"].median()),
                "median_accelerated_runtime_seconds": float(
                    group["palm_accelerated_runtime"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_a_recovery(ok: pd.DataFrame) -> pd.DataFrame:
    candidates = ok[ok["A_recovery"].isin(["A_current", "A_full_Pois"])].copy()
    candidates["dev_per_count"] = candidates[METRIC] / candidates[COUNT]
    keys = ["dataset", "biological_group", "K", "seed", "W_fit_id"]
    paired = candidates.pivot_table(
        index=keys,
        columns="A_recovery",
        values="dev_per_count",
        aggfunc="first",
        dropna=False,
    ).dropna(subset=["A_current", "A_full_Pois"]).reset_index()
    paired["current_minus_poisson_dev_per_count"] = (
        paired["A_current"] - paired["A_full_Pois"]
    )
    rows = []
    for dataset, group in paired.groupby("dataset"):
        delta = group["current_minus_poisson_dev_per_count"]
        rows.append(
            {
                "dataset": dataset,
                "n_pairs": len(group),
                "poisson_deviance_win_fraction": float((delta > 0).mean()),
                "poisson_deviance_tie_fraction_1e-10": float(
                    np.isclose(delta, 0.0, atol=1e-10, rtol=0.0).mean()
                ),
                "mean_current_minus_poisson_dev_per_count": float(delta.mean()),
                "median_current_minus_poisson_dev_per_count": float(delta.median()),
            }
        )
    return pd.DataFrame(rows)


def best_complete_configuration_by_k(ok: pd.DataFrame) -> pd.DataFrame:
    candidates = ok.dropna(subset=[METRIC, COUNT]).copy()
    candidates = candidates[candidates[COUNT] > 0]
    candidates["dev_per_count"] = candidates[METRIC] / candidates[COUNT]
    task_keys = ["dataset", "biological_group", "K", "seed"]
    expected = (
        candidates[task_keys]
        .drop_duplicates()
        .groupby(["dataset", "K"], dropna=False)
        .size()
        .rename("expected_n")
        .reset_index()
    )
    config = [
        "dataset",
        "K",
        "estimator_family",
        "spectral_geometry",
        "vertex_hunter",
        "preprocessing",
        "A_recovery",
        "W_recovery_method",
    ]
    grouped = (
        candidates.groupby(config, dropna=False)["dev_per_count"]
        .agg(n="size", mean="mean", median="median")
        .reset_index()
        .merge(expected, on=["dataset", "K"], how="left")
    )
    complete = grouped[grouped["n"] == grouped["expected_n"]]
    best = (
        complete.sort_values(["dataset", "K", "mean"])
        .groupby(["dataset", "K"], as_index=False)
        .first()
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/real_data_anchor_word_gplsi"),
    )
    args = parser.parse_args()
    data, complete_files = load_completed(args.result_root)
    ok = data[data["status"] == "ok"].copy()

    task_coverage = (
        data[["dataset", "biological_group", "K", "seed"]]
        .drop_duplicates()
        .groupby(["dataset", "biological_group", "K"], dropna=False)
        .size()
        .rename("completed_tasks")
        .reset_index()
    )
    status = data.groupby("status", dropna=False).size().rename("rows").reset_index()
    palm_status = (
        data[data["vertex_hunter"].isin(["palm", "palm_accelerated"])]
        .groupby(["dataset", "vertex_hunter", "status"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    failures = (
        data[data["status"] == "failed"]
        .groupby("failure_reason", dropna=False)
        .size()
        .sort_values(ascending=False)
        .rename("rows")
        .reset_index()
    )

    output = {
        "completed_task_directories": len(complete_files),
        "fit_rows": len(data),
        "task_coverage": records(task_coverage),
        "status": records(status),
        "palm_status": records(palm_status),
        "paired_palm": records(paired_palm(ok)),
        "paired_a_recovery": records(paired_a_recovery(ok)),
        "best_complete_configuration_by_k": records(
            best_complete_configuration_by_k(ok)
        ),
        "best_complete_gplsi_configuration_by_k": records(
            best_complete_configuration_by_k(
                ok[
                    ok["estimator_family"].isin(
                        ["document_gplsi", "anchor_feature_gplsi"]
                    )
                ]
            )
        ),
        "failure_reasons": records(failures),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
