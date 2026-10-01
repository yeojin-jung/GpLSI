#!/usr/bin/env python3
"""Refit A from the W saved by ``run_experiment.py`` (W is kept fixed).

    python scripts/refit_A.py configs/dlpfc/ablation/core.json            # config's posthoc_refit
    python scripts/refit_A.py CONFIG --recovery A_full_Pois_SQUAREM --poisson-start pooled
    python scripts/refit_A.py CONFIG --task $SLURM_ARRAY_TASK_ID --stride 10

Defaults come from the config's ``posthoc_refit`` block; flags override them.
Task indices count tasks once even when a config splits them into parts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gplsi.pipeline import expand_tasks, load_config
from gplsi.pipeline.config import task_id
from gplsi.pipeline.refit import refit_task
from gplsi.real_experiment import A_RECOVERIES, POISSON_STARTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--task", type=int)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--recovery", choices=A_RECOVERIES)
    parser.add_argument("--poisson-start", choices=POISSON_STARTS)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    defaults = dict(config.get("posthoc_refit", {}))
    defaults.pop("note", None)
    recovery = args.recovery or defaults.pop("A_recovery", "A_full_Pois")
    defaults.pop("A_recovery", None)
    settings = {"poisson_start": "pooled", "poisson_max_iter": 2000, "poisson_tolerance": 1e-8, **defaults}
    for key, value in (("poisson_start", args.poisson_start), ("poisson_max_iter", args.max_iter),
                       ("poisson_tolerance", args.tolerance)):
        if value is not None:
            settings[key] = value

    tasks = list({task_id(config, task): task for task in expand_tasks(config)}.values())
    if args.task is None:
        indices = range(len(tasks))
    elif args.task >= len(tasks):
        print(f"task {args.task} >= {len(tasks)} unique tasks; nothing to do")
        return
    else:
        indices = range(args.task, len(tasks), args.stride) if args.stride > 0 else [args.task]
    for index in indices:
        task = {key: value for key, value in tasks[index].items() if key != "part"}
        path = refit_task(config, task, recovery, settings)
        print(json.dumps({"task_index": index, "recovery": recovery, "output": str(path)}), flush=True)


if __name__ == "__main__":
    main()
