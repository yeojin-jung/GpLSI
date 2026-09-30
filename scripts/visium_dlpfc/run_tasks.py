"""Run DLPFC ablation tasks locally in parallel, skipping completed ones.

Each task runs in its own process with single-threaded BLAS; logs go to
logs/visium_dlpfc/<identity>.log.

    python scripts/visium_dlpfc/run_tasks.py --tasks configs/visium_dlpfc/tasks_core.csv --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gplsi_spatial_benchmark.ablation_runner import task_identity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs/visium_dlpfc/ablation.json")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--graph-n-jobs", type=int, default=1, help="CV-fold processes inside each task")
    parser.add_argument("--rerun", action="store_true", help="rerun tasks that already have results")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    frame = pd.read_csv(args.tasks, dtype={"unit_id": str})
    pending = []
    for index, task in frame.iterrows():
        identity = task_identity(task.to_dict())
        done = REPO / config["output_root"] / task["design"] / f"{identity}.json"
        if args.rerun or not done.is_file():
            pending.append((index, identity))
    print(f"{len(frame) - len(pending)} done, {len(pending)} to run with {args.workers} workers", flush=True)

    logs = REPO / "logs/visium_dlpfc"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "src"),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        GPLSI_GRAPH_N_JOBS=str(args.graph_n_jobs),
    )

    def run(item):
        index, identity = item
        started = perf_counter()
        with (logs / f"{identity}.log").open("w") as log:
            code = subprocess.call(
                [sys.executable, "-m", "gplsi_spatial_benchmark.ablation_runner",
                 "--config", str(args.config), "--tasks", str(args.tasks), "--index", str(index)],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        return identity, code, perf_counter() - started

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, item) for item in pending]):
            identity, code, seconds = future.result()
            failures += code != 0
            print(f"[{'ok' if code == 0 else f'FAILED ({code})'}] {identity} {seconds / 60:.1f} min", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
