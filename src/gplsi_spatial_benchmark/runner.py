"""One-task benchmark runner with atomic, provenance-rich outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from gplsi.real_data import RealDataBundle

from .data import load_spatial_slice
from .graph import build_within_unit_knn_graph
from .methods import fit_method_suite
from .metrics import external_structure_metrics, heldout_count_metrics, spatial_metrics, topic_profile_metrics
from .splits import thin_and_split_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        # A packaged installation or standalone data directory may have no Git metadata.
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_task(root: Path, tier: str, task: dict) -> Path:
    started = perf_counter()
    config_path = root / "configs" / "benchmark.json"
    config = json.loads(config_path.read_text())
    seed = int(task["seed"])
    maximum = task.get("max_observations")
    maximum = None if maximum is None or pd.isna(maximum) or str(maximum) == "" else int(maximum)
    spatial = load_spatial_slice(
        root / "data" / "processed", task["dataset"], str(task["unit_id"]),
        max_observations=maximum,
        seed=seed,
    )
    split = thin_and_split_counts(
        spatial.counts,
        retained_fraction=float(task["retained_fraction"]),
        test_fraction=float(config["heldout_test_fraction"]),
        seed=seed,
    )
    keep = np.asarray(split.metadata["row_keep_mask"], dtype=bool)
    coordinates = spatial.coordinates[keep]
    graph_units = spatial.graph_unit_ids[keep]
    external = spatial.external.loc[keep].reset_index(drop=True)
    graph = build_within_unit_knn_graph(coordinates, graph_units, k=int(config["graph"]["k"]))
    lengths = split.train.sum(axis=1).astype(float)
    frequencies = split.train / lengths[:, None]
    bundle = RealDataBundle(
        dataset=f"{task['dataset']}::{task['unit_id']}",
        counts=split.train,
        frequencies=frequencies,
        document_lengths=lengths,
        feature_ids=spatial.feature_ids,
        observation_ids=spatial.observation_ids[keep],
        group_ids=graph_units,
        edge_df=graph.edges,
        weights=graph.adjacency,
        coordinates=coordinates,
        outcomes=None,
        metadata={"external_columns_withheld": list(external.columns)},
    ).validate()
    factorial = config["gplsi_factorial"]
    smoke = tier == "smoke"
    # Smoke verifies every declared document-side cell with bounded iteration counts.
    spectral_parameters = {
        "n_jobs": max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    }
    if smoke:
        spectral_parameters.update({"grid_len": 5, "maxiter": 4, "nfolds": 3})
    fits = fit_method_suite(
        bundle, coordinates, int(task["K"]), seed,
        document_preprocessings=tuple(factorial["document_preprocessing"]),
        document_vertex_hunters=tuple(factorial["document_vertex_hunting"]),
        anchor_preprocessings=tuple(factorial["anchor_preprocessing"]),
        anchor_vertex_hunters=tuple(factorial["anchor_vertex_hunting"]),
        initialization=str(factorial["spectral_initialization"]),
        spectral_parameters=spectral_parameters,
    )
    records = []
    arrays = {}
    for index, fit in enumerate(fits):
        record = {
            "method": fit.method,
            "status": fit.status,
            "runtime_seconds": fit.runtime_seconds,
            "warnings": fit.warnings,
            "metadata": fit.metadata,
        }
        if fit.status == "ok" and fit.W is not None and fit.A is not None:
            record["metrics"] = {
                **heldout_count_metrics(fit.W, fit.A, split.test),
                **external_structure_metrics(fit.W, external, seed),
                **spatial_metrics(fit.W, graph.edges),
            }
            profile_metrics, top_features = topic_profile_metrics(fit.A, spatial.feature_ids)
            record["metrics"].update(profile_metrics)
            record["top_features"] = top_features
            arrays[f"W_{index}"] = fit.W.astype(np.float32)
            arrays[f"A_{index}"] = fit.A.astype(np.float32)
        records.append(record)
    identity = "__".join(
        [str(task["dataset"]), str(task["unit_id"]), f"K{task['K']}", f"r{task['retained_fraction']}", f"s{seed}"]
    ).replace("/", "-").replace(" ", "_")
    output_dir = root / "results" / tier / str(task["dataset"])
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{identity}.npz"
    json_path = output_dir / f"{identity}.json"
    npz_partial = npz_path.with_suffix(".partial")
    with npz_partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    npz_partial.replace(npz_path)
    payload = {
        "schema_version": 1,
        "task": task,
        "data": spatial.metadata,
        "split": {key: value for key, value in split.metadata.items() if key != "row_keep_mask"},
        "graph": graph.metadata,
        "results": records,
        "provenance": {
            "config_sha256": _sha256(config_path),
            "processed_h5ad_sha256": _sha256(Path(spatial.metadata["processed_file"])),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
            "gplsi_worktree_commit": _git_head(Path(__file__).resolve().parents[2]),
            "orchestration_commit": _git_head(root),
            "python_lock_sha256": (
                _sha256(root / "environments" / "python-requirements.lock.txt")
                if (root / "environments" / "python-requirements.lock.txt").is_file() else None
            ),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == "darwin" else 1)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": perf_counter() - started,
            "array_file": npz_path.name,
        },
    }
    partial = json_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    partial.replace(json_path)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tier", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.task_manifest, dtype={"unit_id": str})
    if not 0 <= args.task_index < len(frame):
        raise IndexError(f"task index {args.task_index} outside [0,{len(frame)})")
    output = run_task(args.root, args.tier, frame.iloc[args.task_index].to_dict())
    print(output)


if __name__ == "__main__":
    main()
