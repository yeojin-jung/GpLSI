"""On-disk layout of experiment results, and helpers to read them back.

``<output_root>/<run name>/<task id>/`` holds one task (all of its parts)::

    task.json                  task, resolved config, data summary, hashes
    data.npz                   row/feature ids, feature names, coordinates
    rows/<fit_id>.json         one row per fit (status, settings, scores)
    arrays/<fit_id>.npz        W_hat, A_hat and geometry arrays of that fit
    spectral/<preproc>.npz     graph-SVD factors shared by a preprocessing
    fit_rows.csv               all rows of the task, flattened

A row is reused on rerun only if its ``cache_key`` (data hashes + the settings
that determine the numbers) matches and its arrays still have the recorded
hash, so interrupted jobs resume and changed settings recompute.  Artifact
paths are stored relative to the task directory, so results can be moved
(e.g. copied off the cluster); :func:`load_rows` makes them absolute.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 3
IDENTITY_FIELDS = ("estimator_family", "spectral_geometry", "vertex_hunter", "preprocessing", "A_recovery")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def method_name(fields: dict[str, Any]) -> str:
    """Readable method label, e.g. ``document_gplsi__document_U__svs__P0_raw__A_current``."""

    return "__".join(str(fields[key]) for key in IDENTITY_FIELDS)


def fit_id(fields: dict[str, Any]) -> str:
    suffix = sha256_text(canonical_json({key: fields[key] for key in IDENTITY_FIELDS}))[:10]
    return f"{method_name(fields)}__{suffix}".replace("/", "-")


def W_fit_id(fields: dict[str, Any]) -> str:
    """Identifies the W shared by all A recoveries of one geometry."""

    keys = IDENTITY_FIELDS[:-1]
    return "W__" + sha256_text(canonical_json({key: fields[key] for key in keys}))[:16]


def _partial(path: Path) -> Path:
    # Per-process name: parts of one task may run concurrently in one directory.
    return path.with_name(f"{path.name}.{os.getpid()}.partial")


def write_json(path: Path, payload: Any) -> None:
    """Write atomically, so a killed job never leaves a truncated file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial(path)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    partial.replace(path)


def save_arrays(path: Path, *, relative_to: Path | None = None, **arrays: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial(path)
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(path)
    stored = path.relative_to(relative_to) if relative_to is not None else path.resolve()
    return {"path": str(stored), "sha256": sha256_file(path)}


class TaskDirectory:
    """Rows and arrays of one task."""

    def __init__(self, path: Path, cache_key: str):
        self.path = Path(path)
        self.cache_key = cache_key
        self.path.mkdir(parents=True, exist_ok=True)

    def row_path(self, fid: str) -> Path:
        return self.path / "rows" / f"{fid}.json"

    def cached(self, fid: str) -> dict[str, Any] | None:
        """A finished row with the same cache key and intact arrays, else None."""

        path = self.row_path(fid)
        if not path.exists():
            return None
        try:
            row = json.loads(path.read_text())
            if row.get("cache_key") != self.cache_key or row.get("status") == "failed":
                return None
            for artifact in row.get("artifacts", {}).values():
                if sha256_file(self.path / artifact["path"]) != artifact["sha256"]:
                    return None
            return row
        except (OSError, ValueError, KeyError):
            return None

    def write_row(self, row: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        if arrays:
            row["artifacts"] = {
                "estimate": save_arrays(
                    self.path / "arrays" / f"{row['fit_id']}.npz", relative_to=self.path, **arrays
                )
            }
        write_json(self.row_path(row["fit_id"]), row)
        return row

    def aggregate(self) -> pd.DataFrame:
        rows = [json.loads(path.read_text()) for path in sorted((self.path / "rows").glob("*.json"))]
        frame = pd.json_normalize(rows, sep=".")
        partial = _partial(self.path / "fit_rows.csv")
        frame.to_csv(partial, index=False)
        partial.replace(self.path / "fit_rows.csv")
        return frame


def load_rows(run_dir: str | Path, *, flatten: bool = True) -> pd.DataFrame | list[dict[str, Any]]:
    """Every row of every task under ``run_dir`` (one experiment's results).

    Each row gains ``task_dir``, and its artifact paths are made absolute.
    """

    rows = []
    for path in sorted(Path(run_dir).resolve().glob("*/rows/*.json")):
        row = json.loads(path.read_text())
        task_dir = path.parent.parent
        row["task_dir"] = str(task_dir)
        for artifact in row.get("artifacts", {}).values():
            artifact["path"] = str(task_dir / artifact["path"])
        rows.append(row)
    return pd.json_normalize(rows, sep=".") if flatten else rows


def load_arrays(row: dict[str, Any] | pd.Series, task_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    """The saved arrays (``W_hat``, ``A_hat``, ...) of one row."""

    if "artifacts.estimate.path" in row:
        path = row["artifacts.estimate.path"]
    else:
        path = row["artifacts"]["estimate"]["path"]
    base = task_dir if task_dir is not None else row.get("task_dir")
    path = Path(base) / path if base is not None else Path(path)
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}
