#!/usr/bin/env python3
"""Run tasks of an experiment config.

    python scripts/run_experiment.py configs/crc/production.json --list
    python scripts/run_experiment.py configs/crc/production.json --task 3
    python scripts/run_experiment.py configs/crc/production.json            # all tasks
    python scripts/run_experiment.py CONFIG --task $SLURM_ARRAY_TASK_ID --stride 10

With ``--stride S`` the job runs tasks i, i+S, i+2S, ... so S array jobs cover
the whole grid (useful under a per-user job limit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gplsi.pipeline import expand_tasks, load_config, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--task", type=int, help="task index (default: all tasks)")
    parser.add_argument("--stride", type=int, default=0, help="also run task+stride, task+2*stride, ...")
    parser.add_argument("--list", action="store_true", help="print the task grid and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    tasks = expand_tasks(config)
    if args.list:
        for index, task in enumerate(tasks):
            print(index, json.dumps(task))
        return
    if args.task is None:
        indices = range(len(tasks))
    else:
        if not 0 <= args.task < len(tasks):
            raise SystemExit(f"task index must be in [0, {len(tasks) - 1}]")
        indices = range(args.task, len(tasks), args.stride) if args.stride > 0 else [args.task]
    for index in indices:
        path = run_task(config, tasks[index])
        print(json.dumps({"task_index": index, **tasks[index], "output": str(path)}), flush=True)


if __name__ == "__main__":  # guard needed: graph CV may use a multiprocessing pool
    main()
