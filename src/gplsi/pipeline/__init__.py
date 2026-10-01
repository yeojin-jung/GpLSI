"""One experiment pipeline for every dataset (CRC, spleen, Cooking, Visium DLPFC).

``scripts/run_experiment.py CONFIG --task i`` runs task ``i`` of a config:

* :mod:`.config` - load a JSON config and expand its task grid;
* :mod:`.datasets` - build the training data and held-out counts of a task;
* :mod:`.runner` - fit the GpLSI grid and baselines, one result row per method;
* :mod:`.metrics` - scores attached to every row;
* :mod:`.results` - on-disk layout, caching, and loaders for analysis.
"""

from .config import expand_tasks, load_config
from .results import load_arrays, load_rows
from .runner import run_task

__all__ = ["expand_tasks", "load_arrays", "load_config", "load_rows", "run_task"]
