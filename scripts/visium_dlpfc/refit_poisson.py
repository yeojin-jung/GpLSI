"""Refit A_full_Pois from the W saved by each completed ablation task.

The task's split, training-only panel, spot mask, and graph are rebuilt
deterministically, so the Poisson profiles are paired with exactly the same W
(and the same training/test counts) as A_current and A_full_L2. Results go to
``results/visium_dlpfc/<design>/poisson/<task>.json|.npz`` and are merged by
``summarize.py``.

    python scripts/visium_dlpfc/refit_poisson.py --designs core panel lambda_wide --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gplsi.recovery import prepare_poisson_counts, refit_A_full_poisson  # noqa: E402
from gplsi_spatial_benchmark.ablation_runner import prepare_task_data, score_fit  # noqa: E402
from gplsi_spatial_benchmark.runner import _json_safe  # noqa: E402


RECOVERY_SUFFIXES = ("__A_current", "__A_full_L2")


def geometry_sources(results: list[dict], saved_keys) -> list[tuple[str, int | None, int | None]]:
    """One (geometry, W index, A_current index) per GpLSI geometry, in result order.

    All recoveries of a geometry share one W, so W is taken from the first record that
    saved it; a failed A_current (e.g. a singular anchor fit) must not block the refit.
    """
    saved_keys = set(saved_keys)
    order, w_index, current_index = [], {}, {}
    for index, record in enumerate(results):
        method = record["method"]
        suffix = next((s for s in RECOVERY_SUFFIXES if method.endswith(s)), None)
        if not method.startswith("gplsi_") or suffix is None:
            continue
        geometry = method[: -len(suffix)]
        if geometry not in w_index:
            order.append(geometry)
            w_index[geometry] = None
        if w_index[geometry] is None and f"W_{index}" in saved_keys:
            w_index[geometry] = index
        if suffix == "__A_current":
            current_index[geometry] = index
    return [(g, w_index[g], current_index.get(g)) for g in order]


def refit_task(config_path: str, result_json: str) -> str:
    config = json.loads(Path(config_path).read_text())
    settings = config["poisson_refit"]
    payload = json.loads(Path(result_json).read_text())
    task, identity = payload["task"], payload["identity"]
    out_dir = Path(result_json).parent / "poisson"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"{identity}.json"
    prepared = prepare_task_data(config, task, REPO)
    bundle = prepared["bundle"]
    counts = prepare_poisson_counts(bundle.counts)
    records, arrays = [], {}
    with np.load(Path(result_json).with_suffix(".npz"), allow_pickle=True) as saved:
        if not np.array_equal(saved["observation_ids"].astype(str), prepared["observation_ids"].astype(str)) or not np.array_equal(
            saved["feature_ids"].astype(str), prepared["feature_ids"].astype(str)
        ):
            raise ValueError(f"{identity}: rebuilt task data do not match the saved fit")
        for geometry, index, current_index in geometry_sources(payload["results"], saved.keys()):
            started = perf_counter()
            entry = {"method": f"{geometry}__A_full_Pois", "status": "failed", "metadata": {}}
            try:
                if index is None:
                    raise ValueError("no saved W for this geometry")
                record = payload["results"][index]
                W = saved[f"W_{index}"].astype(float)
                W /= W.sum(axis=1, keepdims=True)
                initial = None
                if settings["initial"] == "A_current":
                    if current_index is None or f"A_{current_index}" not in saved:
                        raise ValueError("initial='A_current' but this geometry has no saved A_current")
                    initial = saved[f"A_{current_index}"].astype(float)
                result = refit_A_full_poisson(
                    W, counts, bundle.document_lengths, initial_A=initial,
                    max_iter=int(settings["max_iter"]), tolerance=float(settings["tolerance"]),
                    interior_mass=float(settings["interior_mass"]),
                )
                entry.update(
                    status="ok" if result.converged else result.status,
                    runtime_seconds=perf_counter() - started,
                    metadata={
                        **{k: record["metadata"].get(k) for k in ("spectral_preprocessing", "selected_rho", "retained_feature_count")},
                        "A_recovery": {
                            "method": result.method, "status": result.status, "converged": result.converged,
                            "iterations": result.iterations, "normalized_optimality_gap": result.normalized_optimality_gap,
                            "solver": result.solver, "settings": settings,
                        },
                    },
                    warnings=result.warnings,
                )
                entry["metrics"], entry["top_features"] = score_fit(W, result.A_hat, prepared, int(task["seed"]))
                arrays[f"A_{len(records)}"] = result.A_hat.astype(np.float32)
            except Exception as exc:  # recorded, never silently dropped
                entry["metadata"] = {"exception_type": type(exc).__name__, "exception": str(exc)}
                entry["runtime_seconds"] = perf_counter() - started
            entry["source_array_index"] = index
            records.append(entry)
    np.savez_compressed(out_dir / f"{identity}.npz", **arrays)
    out = {"task": task, "identity": identity, "settings": settings, "results": records}
    partial = out_json.with_suffix(".json.partial")
    partial.write_text(json.dumps(_json_safe(out), indent=2, sort_keys=True) + "\n")
    partial.replace(out_json)
    return f"{identity}: {sum(r['status'] == 'ok' for r in records)}/{len(records)} converged"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs/visium_dlpfc/ablation.json"))
    parser.add_argument("--designs", nargs="+", default=["core", "panel", "lambda_wide"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    jobs = []
    for design in args.designs:
        directory = REPO / config["output_root"] / design
        for path in sorted(directory.glob("*.json")):
            if args.rerun or not (directory / "poisson" / path.name).is_file():
                jobs.append(str(path))
    print(f"{len(jobs)} tasks to refit", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(refit_task, args.config, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                print(f"FAILED {Path(futures[future]).name}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
