"""Experiment configuration: loading, inheritance, and task expansion.

A config is one JSON file describing one experiment (see ``configs/README.md``).
It may ``extend`` a base file; nested dictionaries are merged and everything
else (including lists, and ``parts`` as a whole) is replaced.  ``grid`` lists the task axes; every
combination is one task.  Optional ``parts`` split a task into independently
schedulable pieces: each part is a set of overrides (for example a subset of
preprocessings, or extra seeds for LDA only) and may carry its own ``grid``.
"""

from __future__ import annotations

import copy
from itertools import product
import json
from pathlib import Path
from typing import Any


# Task axes other than K and seed, in the order they appear in task ids.
EXTRA_AXES = ("section", "panel_size", "retained_fraction")

# Keys whose values only choose *which* methods run.  They are excluded from
# the settings hash so that splitting a run into parts does not invalidate
# rows computed by another part.
METHOD_SELECTION_KEYS = ("gplsi", "A_recoveries", "baselines")


# A child's parts describe its own split of the work, never an extension of
# the parent's.
REPLACED_WHOLE = ("parts",)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in REPLACED_WHOLE and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_with_parents(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = _read_with_parents(path.parent / parent)
    base.pop("name", None)  # a name identifies one experiment; never inherit it
    return deep_merge(base, config)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a config, resolving ``extends`` chains relative to each file.

    ``name`` (the results directory) defaults to the file name.
    """

    path = Path(path)
    config = _read_with_parents(path)
    config.setdefault("name", path.stem)
    return config


def _grid_tasks(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if "K" not in grid or "seed" not in grid:
        raise ValueError("config grid must list both K and seed")
    axes = ["K", "seed"] + [axis for axis in EXTRA_AXES if axis in grid]
    unknown = set(grid) - set(axes)
    if unknown:
        raise ValueError(f"unknown grid axes {sorted(unknown)}; allowed: K, seed, {EXTRA_AXES}")
    return [dict(zip(axes, values)) for values in product(*(grid[axis] for axis in axes))]


def expand_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """All tasks of a config, in a stable order (the Slurm array index)."""

    parts = config.get("parts")
    if not parts:
        return _grid_tasks(config["grid"])
    tasks = []
    for part, overrides in parts.items():
        grid = {**config["grid"], **overrides.get("grid", {})}
        tasks.extend({**task, "part": part} for task in _grid_tasks(grid))
    return tasks


def config_for_task(config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Apply a task's part overrides (grid overrides are already expanded)."""

    part = task.get("part")
    resolved = {key: value for key, value in config.items() if key != "parts"}
    if part is None:
        return resolved
    overrides = {key: value for key, value in config["parts"][part].items() if key != "grid"}
    return deep_merge(resolved, overrides)


def task_id(config: dict[str, Any], task: dict[str, Any]) -> str:
    """Directory name shared by every part of the same task."""

    dataset = config["dataset"]
    tokens = [str(dataset["name"])]
    if dataset.get("group"):
        tokens.append(str(dataset["group"]))
    if "section" in task:
        tokens.append(str(task["section"]))
    tokens += [f"K{int(task['K'])}"]
    if dataset.get("vocabulary") == "tran":
        tokens.append(f"tran{float(dataset['tran_alpha']):g}")
    elif "panel_size" in task:
        tokens.append(f"p{int(task['panel_size'])}")
    if "retained_fraction" in task:
        tokens.append(f"r{float(task['retained_fraction'])}")
    tokens.append(f"seed{int(task['seed'])}")
    return "__".join(tokens).replace("/", "-")


def settings_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    """The part of a resolved config that determines each fit's numbers."""

    ignored = {"name", "description", "output_root", "grid", "parts", *METHOD_SELECTION_KEYS}
    return {key: value for key, value in config.items() if key not in ignored}
