#!/usr/bin/env python3
"""Aggregate completed real-data pilot tasks into preliminary tables and plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results/real_data_anchor_word_gplsi"
FIGURE_ROOT = REPO_ROOT / "figures/real_data_anchor_word_gplsi"
DEFAULT_RUNS = (
    "pilot_crc_5seed",
    "pilot_spleen_BALBc-1_5seed",
    "pilot_cook_5seed",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _task_directories(run_dirs: list[Path]) -> list[Path]:
    pattern = re.compile(r"__[0-9a-f]{10}$")
    tasks = [
        child
        for run_dir in run_dirs
        for child in run_dir.iterdir()
        if child.is_dir()
        and pattern.search(child.name)
        and (child / "complete.json").exists()
        and (child / "task_manifest.json").exists()
    ]
    latest: dict[tuple[str, int, int], Path] = {}
    for task in tasks:
        manifest = json.loads((task / "task_manifest.json").read_text())
        key = (
            str(manifest["config"]["dataset"])
            + "::"
            + str(manifest["config"].get("group")),
            int(manifest["K"]),
            int(manifest["seed"]),
        )
        previous = latest.get(key)
        if previous is None or task.stat().st_mtime_ns > previous.stat().st_mtime_ns:
            latest[key] = task
    return sorted(latest.values())


def _load_rows(tasks: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for task in tasks:
        manifest = json.loads((task / "task_manifest.json").read_text())
        complete = json.loads((task / "complete.json").read_text())
        task_hash = manifest["task_config_hash"]
        if complete["task_config_hash"] != task_hash:
            raise RuntimeError(f"completion hash mismatch in {task}")
        manifests.append({"task": str(task), **manifest})
        for path in sorted((task / "rows").glob("*.json")):
            row = json.loads(path.read_text())
            if row["task_config_hash"] != task_hash:
                raise RuntimeError(f"stale fit row {path}")
            row["_row_path"] = str(path)
            rows.append(row)
    return rows, manifests


def _estimate(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(row["artifacts"]["estimate"]["path"])
    if _hash_file(path) != row["artifacts"]["estimate"]["sha256"]:
        raise RuntimeError(f"estimate artifact hash mismatch: {path}")
    with np.load(path) as data:
        return data["W_hat"].copy(), data["A_hat"].copy()


def _align_topics(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-15)
    cand_norm = candidate / np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-15)
    rows, columns = linear_sum_assignment(1.0 - ref_norm @ cand_norm.T)
    order = np.empty(reference.shape[0], dtype=int)
    order[rows] = columns
    return order


def _topic_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    order = _align_topics(reference, candidate)
    aligned = candidate[order]
    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-15)
    cand_norm = aligned / np.maximum(np.linalg.norm(aligned, axis=1, keepdims=True), 1e-15)
    mixture = 0.5 * (reference + aligned)
    epsilon = 1e-15
    js = 0.5 * np.sum(reference * np.log((reference + epsilon) / (mixture + epsilon)), axis=1)
    js += 0.5 * np.sum(aligned * np.log((aligned + epsilon) / (mixture + epsilon)), axis=1)
    top = min(10, reference.shape[1])
    overlaps = []
    for topic in range(reference.shape[0]):
        left = set(np.argpartition(reference[topic], -top)[-top:])
        right = set(np.argpartition(aligned[topic], -top)[-top:])
        overlaps.append(len(left & right) / top)
    return {
        "mean_topic_cosine": float(np.mean(np.sum(ref_norm * cand_norm, axis=1))),
        "mean_topic_l1": float(np.mean(np.sum(np.abs(reference - aligned), axis=1))),
        "mean_topic_jensen_shannon": float(np.mean(js)),
        "mean_topic_hellinger": float(
            np.mean(np.linalg.norm(np.sqrt(reference) - np.sqrt(aligned), axis=1) / np.sqrt(2))
        ),
        "mean_top10_overlap": float(np.mean(overlaps)),
    }


def stability_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    successful = [row for row in rows if row["status"] == "ok" and "estimate" in row.get("artifacts", {})]
    keys = (
        "dataset",
        "biological_group",
        "K",
        "estimator_family",
        "spectral_geometry",
        "vertex_hunter",
        "preprocessing",
        "A_recovery",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in successful:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        group.sort(key=lambda row: int(row["seed"]))
        if len(group) < 2:
            continue
        _, reference = _estimate(group[0])
        for row in group[1:]:
            _, candidate = _estimate(row)
            output.append(
                {
                    **dict(zip(keys, key)),
                    "reference_seed": int(group[0]["seed"]),
                    "candidate_seed": int(row["seed"]),
                    **_topic_metrics(reference, candidate),
                }
            )
    return pd.DataFrame(output)


def paired_A_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    successful = [
        row
        for row in rows
        if row["status"] == "ok"
        and row["A_recovery"] in {"A_current", "A_full_Pois"}
        and "estimate" in row.get("artifacts", {})
    ]
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in successful:
        groups.setdefault((row["task_config_hash"], row["W_fit_id"]), {})[
            row["A_recovery"]
        ] = row
    output: list[dict[str, Any]] = []
    for (_, W_fit_id), pair in groups.items():
        if set(pair) != {"A_current", "A_full_Pois"}:
            continue
        current = pair["A_current"]
        poisson = pair["A_full_Pois"]
        W_current, A_current = _estimate(current)
        W_poisson, A_poisson = _estimate(poisson)
        if not np.array_equal(W_current, W_poisson):
            raise RuntimeError(f"paired A records do not share bitwise-identical W: {W_fit_id}")
        comparison = _topic_metrics(A_current, A_poisson)
        output.append(
            {
                "dataset": current["dataset"],
                "biological_group": current.get("biological_group"),
                "seed": current["seed"],
                "K": current["K"],
                "estimator_family": current["estimator_family"],
                "spectral_geometry": current["spectral_geometry"],
                "vertex_hunter": current["vertex_hunter"],
                "preprocessing": current["preprocessing"],
                "W_fit_id": W_fit_id,
                "W_bitwise_identical": True,
                "current_A_heldout_deviance": current.get("heldout_metrics", {}).get("heldout_poisson_deviance"),
                "poisson_A_heldout_deviance": poisson.get("heldout_metrics", {}).get("heldout_poisson_deviance"),
                "heldout_deviance_difference_poisson_minus_current": (
                    poisson.get("heldout_metrics", {}).get("heldout_poisson_deviance")
                    - current.get("heldout_metrics", {}).get("heldout_poisson_deviance")
                ),
                "current_A_reconstruction_fro": current.get("diagnostics", {}).get("reconstruction_fro"),
                "poisson_A_reconstruction_fro": poisson.get("diagnostics", {}).get("reconstruction_fro"),
                "runtime_difference_poisson_minus_current": (
                    poisson.get("runtime_A_recovery_seconds", 0.0)
                    - current.get("runtime_A_recovery_seconds", 0.0)
                ),
                "poisson_converged": poisson.get("poisson_converged"),
                "poisson_kkt_projected_gradient_norm": poisson.get(
                    "poisson_kkt_projected_gradient_norm"
                ),
                **comparison,
            }
        )
    return pd.DataFrame(output)


def candidate_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for row in rows:
        for candidate in row.get("selected_anchor_feature_candidates") or []:
            output.append(
                {
                    "dataset": row["dataset"],
                    "biological_group": row.get("biological_group"),
                    "seed": row["seed"],
                    "K": row["K"],
                    "preprocessing": row["preprocessing"],
                    "vertex_hunter": row["vertex_hunter"],
                    "A_recovery": row["A_recovery"],
                    "W_fit_id": row["W_fit_id"],
                    **candidate,
                }
            )
    frame = pd.DataFrame(output)
    if not frame.empty:
        counts = (
            frame.drop_duplicates(
                ["dataset", "seed", "preprocessing", "vertex_hunter", "feature_index"]
            )
            .groupby(["dataset", "feature_index", "feature_name"], dropna=False)
            .size()
            .rename("selection_count_across_seed_preprocessing_hunter")
            .reset_index()
        )
        frame = frame.merge(counts, how="left")
    return frame


def _failure_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    keys = [
        "dataset",
        "estimator_family",
        "spectral_geometry",
        "vertex_hunter",
        "preprocessing",
        "A_recovery",
    ]
    return (
        frame.assign(failed=frame["status"].eq("failed"))
        .groupby(keys, dropna=False)
        .agg(runs=("status", "size"), failures=("failed", "sum"))
        .reset_index()
        .assign(failure_rate=lambda value: value["failures"] / value["runs"])
    )


def _plot_summaries(
    rows: list[dict[str, Any]], stability: pd.DataFrame, failures: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    failed = failures.groupby(["dataset", "estimator_family"], as_index=False)[
        ["failures", "runs"]
    ].sum()
    failed["failure_rate"] = failed["failures"] / failed["runs"]
    pivot = failed.pivot(index="estimator_family", columns="dataset", values="failure_rate").fillna(0)
    axis = pivot.plot.bar(figsize=(11, 5), ylim=(0, 1), rot=35)
    axis.set_title("Preliminary pilot: recorded configuration failure rates")
    axis.set_ylabel("Failure rate")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "preliminary_failure_rates.png", dpi=180)
    plt.close(axis.figure)

    if not stability.empty:
        summary = stability.groupby(["dataset", "estimator_family"], as_index=False)[
            "mean_topic_cosine"
        ].mean()
        pivot = summary.pivot(
            index="estimator_family", columns="dataset", values="mean_topic_cosine"
        )
        axis = pivot.plot.bar(figsize=(11, 5), ylim=(0, 1), rot=35)
        axis.set_title("Preliminary pilot: full-feature topic stability across splits")
        axis.set_ylabel("Mean aligned topic cosine")
        axis.figure.tight_layout()
        axis.figure.savefig(output / "preliminary_topic_stability.png", dpi=180)
        plt.close(axis.figure)

    successful = [
        row
        for row in rows
        if row["status"] == "ok" and row.get("heldout_metrics") is not None
    ]
    heldout = pd.DataFrame(
        {
            "dataset": [row["dataset"] for row in successful],
            "estimator_family": [row["estimator_family"] for row in successful],
            "heldout_deviance_per_count": [
                row["heldout_metrics"]["heldout_poisson_deviance"]
                / row["heldout_metrics"]["heldout_count_total"]
                for row in successful
            ],
        }
    )
    if not heldout.empty:
        summary = heldout.groupby(["dataset", "estimator_family"], as_index=False)[
            "heldout_deviance_per_count"
        ].median()
        pivot = summary.pivot(
            index="estimator_family", columns="dataset", values="heldout_deviance_per_count"
        )
        axis = pivot.plot.bar(figsize=(11, 5), rot=35)
        axis.set_title("Preliminary pilot: held-out count deviance (median)")
        axis.set_ylabel("Poisson deviance / held-out count")
        axis.figure.tight_layout()
        axis.figure.savefig(output / "preliminary_heldout_deviance.png", dpi=180)
        plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path)
    args = parser.parse_args()
    run_dirs = args.run_dir or [RESULT_ROOT / name for name in DEFAULT_RUNS]
    tasks = _task_directories(run_dirs)
    rows, manifests = _load_rows(tasks)
    if not rows:
        raise SystemExit("no current hash-suffixed pilot tasks found")
    manifest_hash = hashlib.sha256(
        json.dumps(
            [manifest["task_config_hash"] for manifest in manifests],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = RESULT_ROOT / f"pilot_summary_{manifest_hash}"
    figures = FIGURE_ROOT / f"pilot_summary_{manifest_hash}"
    output.mkdir(parents=True, exist_ok=True)
    normalized = pd.json_normalize(rows, sep=".")
    normalized.to_csv(output / "pilot_fit_rows.csv", index=False)
    failures = _failure_table(rows)
    failures.to_csv(output / "pilot_failure_summary.csv", index=False)
    stability = stability_table(rows)
    stability.to_csv(output / "pilot_topic_stability.csv", index=False)
    paired = paired_A_table(rows)
    paired.to_csv(output / "pilot_current_vs_poisson_A.csv", index=False)
    candidates = candidate_table(rows)
    candidates.to_csv(output / "pilot_anchor_feature_candidates.csv", index=False)

    spectral_records: list[dict[str, Any]] = []
    for task in tasks:
        for path in sorted((task / "spectral").glob("*.json")):
            record = json.loads(path.read_text())
            spectral_records.append(
                {
                    "task": str(task),
                    "preprocessing": path.stem,
                    "rho_selected": record["rho_selected"],
                    "rho_grid_min": min(record["rho_grid"]),
                    "rho_grid_max": max(record["rho_grid"]),
                    "rho_at_lower_boundary": record["rho_selected"] == min(record["rho_grid"]),
                    "rho_at_upper_boundary": record["rho_selected"] == max(record["rho_grid"]),
                    "p_spectral": record["p_spectral"],
                    "transformed_condition_number": record["transformed_condition_number"],
                }
            )
    spectral = pd.DataFrame(spectral_records)
    spectral.to_csv(output / "pilot_spectral_diagnostics.csv", index=False)
    poisson_rows = [row for row in rows if row.get("A_recovery") == "A_full_Pois"]
    flags = {
        "label": "preliminary",
        "input_task_count": len(tasks),
        "input_task_hashes": [manifest["task_config_hash"] for manifest in manifests],
        "fit_row_count": len(rows),
        "status_counts": pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
        "rho_lower_boundary_fraction": float(spectral["rho_at_lower_boundary"].mean()),
        "rho_upper_boundary_fraction": float(spectral["rho_at_upper_boundary"].mean()),
        "poisson_fit_count": len(poisson_rows),
        "poisson_converged_count": int(sum(row.get("poisson_converged") is True for row in poisson_rows)),
        "rank_deficient_success_count": int(
            sum(
                row["status"] == "ok"
                and row.get("diagnostics", {}).get("W_rank", row["K"]) < row["K"]
                for row in rows
            )
        ),
        "paired_A_count": int(len(paired)),
        "paired_W_bitwise_identity_all": bool(
            paired.empty or paired["W_bitwise_identical"].all()
        ),
        "notes": [
            "Pilot subsets are graph-stratified and all fits use 80/20 within-count thinning.",
            "These are computational/scientific pilot results, not full-data conclusions.",
            "Failures are retained and summarized; no downstream labels were used for fitting or tuning.",
        ],
    }
    (output / "pilot_diagnostic_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True) + "\n"
    )
    _plot_summaries(rows, stability, failures, figures)
    print(
        json.dumps(
            {"output": str(output), "figures": str(figures), **flags},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
