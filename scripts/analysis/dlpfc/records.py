"""Read DLPFC pipeline results in the per-task form the analysis scripts use.

``scripts/run_experiment.py configs/dlpfc/ablation/<design>.json`` writes one directory
per task under ``results/dlpfc/<design>/`` (see ``gplsi.pipeline.results``).
:func:`load_tasks` turns each into ``(payload, arrays)``:

* ``payload["identity"]``: ``<design>__<task directory>``; ``payload["task"]``: section (``unit_id``), K, panel_size, retained_fraction, seed;
* ``payload["results"]``: one record per method, with ``method`` named
  ``gplsi_document__<preprocessing>__<hunter>__<A recovery>``,
  ``gplsi_anchor__...`` or the baseline name, plus ``status``, ``metrics``
  (held-out, layer, spatial, and topic-profile scores), ``metadata`` and
  ``runtime_seconds`` and ``npz_path`` (its W_hat/A_hat file);
* ``payload["array_index"]``: method -> position in ``results``;
* ``arrays``: ``W_<i>``/``A_<i>`` of record ``i`` (loaded on demand) and the
  task's ``observation_ids``, ``coordinates``, ``feature_ids``,
  ``feature_symbols``.

A GpLSI record's ``status`` is ``"ok"`` only when its A recovery converged
(otherwise the recovery's status, e.g. ``max_iter_reached``).
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "dlpfc"
DIAGNOSTICS = RESULTS / "diagnostics"
SUMMARY = RESULTS / "summary"
H5AD = REPO / "data" / "dlpfc" / "visium_dlpfc.h5ad"

FAMILY_PREFIX = {"document_gplsi": "gplsi_document", "anchor_feature_gplsi": "gplsi_anchor"}


def legacy_method(row: dict[str, Any]) -> str:
    prefix = FAMILY_PREFIX.get(row["estimator_family"])
    if prefix is None:
        return row["estimator_family"]
    return "__".join([prefix, row["preprocessing"], row["vertex_hunter"], row["A_recovery"]])


def _record(row: dict[str, Any]) -> dict[str, Any]:
    info = row.get("A_recovery_info") or {}
    spectral = row.get("spectral") or {}
    vertex = (row.get("geometry") or {}).get("vertex_parameters") or {}
    status = row["status"]
    if status == "ok" and row.get("converged") is False and row["estimator_family"] in FAMILY_PREFIX:
        status = info.get("status") or "not_converged"
    metadata = {
        "selected_rho": spectral.get("rho_selected"),
        "spectral_iterations": spectral.get("iterations"),
        "retained_feature_count": spectral.get("p_spectral"),
        "A_recovery": info,
        "vertex_hunting": {"optimizer_converged": vertex.get("optimizer_converged")},
        **(row.get("metadata") or {}),
    }
    if row["status"] == "failed":
        exception_type, _, exception = (row.get("failure_reason") or "").partition(": ")
        metadata.update(exception_type=exception_type, exception=exception)
    record = {
        "method": legacy_method(row),
        "status": status,
        "runtime_seconds": (row.get("runtime") or {}).get("total"),
        "warnings": row.get("warnings", []),
        "metadata": metadata,
        "refit_source": row.get("refit_source"),
        "npz_path": ((row.get("artifacts") or {}).get("estimate") or {}).get("path", ""),  # made absolute in load_task
    }
    if row["status"] == "ok":
        record["metrics"] = {**(row.get("heldout_metrics") or {}), **(row.get("metrics") or {})}
        record["top_features"] = row.get("top_features")
    return record


class TaskArrays(Mapping):
    """``W_<i>``/``A_<i>`` per record (read on first use) plus task-level arrays."""

    def __init__(self, task_dir: Path, rows: list[dict[str, Any]]):
        with np.load(task_dir / "data.npz", allow_pickle=False) as data:
            self._static = {
                "observation_ids": data["observation_ids"],
                "feature_ids": data["feature_ids"],
                "feature_symbols": data["feature_names"],
                **({"coordinates": data["coordinates"]} if "coordinates" in data.files else {}),
            }
        self._paths = {
            index: task_dir / row["artifacts"]["estimate"]["path"]
            for index, row in enumerate(rows)
            if row["status"] == "ok" and row.get("artifacts")
        }
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def _estimate(self, index: int) -> dict[str, np.ndarray]:
        if index not in self._cache:
            with np.load(self._paths[index], allow_pickle=False) as arrays:
                self._cache[index] = {"W": arrays["W_hat"], "A": arrays["A_hat"]}
        return self._cache[index]

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._static:
            return self._static[key]
        kind, _, index = key.partition("_")
        if kind in ("W", "A") and index.isdigit() and int(index) in self._paths:
            return self._estimate(int(index))[kind]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from self._static
        for index in self._paths:
            yield f"W_{index}"
            yield f"A_{index}"

    def __len__(self) -> int:
        return len(self._static) + 2 * len(self._paths)


def load_task(task_dir: Path) -> tuple[dict[str, Any], TaskArrays]:
    rows = [json.loads(path.read_text()) for path in sorted((task_dir / "rows").glob("*.json"))]
    task = json.loads(next(task_dir.glob("task*.json")).read_text())["task"]
    records = [_record(row) for row in rows]
    for record in records:
        if record["npz_path"]:
            record["npz_path"] = str(task_dir / record["npz_path"])
    payload = {
        "identity": f"{task_dir.parent.name}__{task_dir.name}",
        "task_dir": str(task_dir),
        "task": {
            "unit_id": str(task["section"]),
            "K": int(task["K"]),
            "panel_size": int(task["panel_size"]),
            "retained_fraction": float(task.get("retained_fraction", 1.0)),
            "seed": int(task["seed"]),
        },
        "results": records,
        "array_index": {record["method"]: index for index, record in enumerate(records)},
    }
    return payload, TaskArrays(task_dir, rows)


def load_tasks(design: str) -> list[tuple[dict[str, Any], TaskArrays]]:
    """All finished tasks of a design, ordered by task directory name."""

    return [load_task(path) for path in sorted((RESULTS / design).iterdir()) if (path / "rows").is_dir()]
