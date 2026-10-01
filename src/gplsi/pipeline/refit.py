"""Refit A afterwards from the W saved by an earlier run (W is never changed).

Used when an A recovery is too slow to run inline or needs different settings,
e.g. the DLPFC Poisson refit from pooled frequencies or the SQUAREM
comparison.  The task's data are rebuilt deterministically and checked against
the hashes recorded when W was fitted; each new row records its source row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..real_data import REPO_ROOT
from ..real_experiment import recover_A
from .config import config_for_task, task_id
from .datasets import prepare_task_data
from .metrics import evaluate_fit
from .results import TaskDirectory, canonical_json, fit_id, load_arrays, method_name, sha256_text
from .runner import _error_text, _recovery_info


# Families whose W is refit; W is shared by every A recovery of one fit.
REFIT_FAMILIES = ("document_gplsi", "anchor_feature_gplsi", "plsi")
PREFERRED_SOURCE = ("A_current", "A_full_L2", "A_full_Pois")


def source_rows(task_dir: Path) -> list[dict[str, Any]]:
    """One finished row per saved W (preferring its A_current row)."""

    by_W: dict[str, dict[str, Any]] = {}
    for path in sorted((task_dir / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        if row["estimator_family"] not in REFIT_FAMILIES or row["status"] != "ok":
            continue
        if row.get("refit_source") is not None:
            continue
        current = by_W.get(row["W_fit_id"])
        rank = PREFERRED_SOURCE.index(row["A_recovery"]) if row["A_recovery"] in PREFERRED_SOURCE else 99
        if current is None or rank < PREFERRED_SOURCE.index(current["A_recovery"]):
            by_W[row["W_fit_id"]] = row
    return list(by_W.values())


def refit_task(config: dict[str, Any], task: dict[str, Any], recovery: str, settings: dict[str, Any]) -> Path:
    resolved = config_for_task(config, task)
    task_dir = REPO_ROOT / resolved.get("output_root", "results") / resolved["name"] / task_id(resolved, task)
    manifests = sorted(task_dir.glob("task*.json"))
    if not manifests:
        raise FileNotFoundError(f"no fitted task at {task_dir}")
    recorded = json.loads(manifests[0].read_text())["data_hashes"]
    data = prepare_task_data(resolved, task)
    if data.hashes() != recorded:
        raise RuntimeError(f"rebuilt data differ from the data W was fitted on: {task_dir}")

    for source in source_rows(task_dir):
        row = {key: value for key, value in source.items()
               if key not in ("artifacts", "diagnostics", "heldout_metrics", "metrics", "top_features",
                              "A_recovery_info", "runtime", "warnings", "traceback", "failure_reason")}
        row.update(A_recovery=recovery, status="pending", converged=None, failure_reason=None, warnings=[])
        row["method"] = method_name(row)
        row["fit_id"] = fit_id(row)
        row["refit_source"] = source["fit_id"]
        row["refit_settings"] = settings
        row["cache_key"] = sha256_text(canonical_json([source["cache_key"], recovery, settings]))
        output = TaskDirectory(task_dir, row["cache_key"])
        if output.cached(row["fit_id"]):
            continue
        arrays = load_arrays(source, task_dir)
        try:
            result, seconds = recover_A(data.bundle, arrays["W_hat"], recovery, **settings)
            row.update(evaluate_fit(data, arrays["W_hat"], result.A_hat, int(task["seed"])))
            row.update(
                status="ok",
                converged=bool(result.converged),
                A_recovery_info=_recovery_info(result),
                runtime={"A_recovery": seconds},
                warnings=list(result.warnings),
            )
            output.write_row(row, {**arrays, "A_hat": result.A_hat})
        except Exception as error:
            row["failure_reason"], row["traceback"] = _error_text(error)
            row["status"] = "failed"
            output.write_row(row)
    TaskDirectory(task_dir, "").aggregate()
    return task_dir
