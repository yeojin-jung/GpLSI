"""Common result interface for the anchor-word experiment family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExperimentalEstimate:
    estimator_family: str
    W_hat: np.ndarray | None
    A_hat: np.ndarray | None
    U_hat: np.ndarray | None = None
    U_bar: np.ndarray | None = None
    V_hat: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    vertices: np.ndarray | None = None
    selected_observation_indices: np.ndarray | None = None
    selected_vocabulary_indices: np.ndarray | None = None
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)
    graph_tuning_metadata: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)
    runtimes: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"
    failure_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "ok"
