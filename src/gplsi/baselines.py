"""Count-native adapters for the existing LDA and spatial-LDA baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


@dataclass
class BaselineResult:
    method: str
    W_hat: np.ndarray
    A_hat: np.ndarray
    runtime_seconds: float
    converged: bool
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_lda(
    counts: np.ndarray,
    K: int,
    *,
    random_state: int = 0,
) -> BaselineResult:
    """Fit the ordinary sklearn LDA baseline used by current GpLSI simulations."""

    D = np.asarray(counts)
    started = perf_counter()
    model = LatentDirichletAllocation(n_components=K, random_state=random_state)
    model.fit(D)
    W_hat = model.transform(D)
    A_hat = model.components_.astype(float)
    A_hat /= A_hat.sum(axis=1, keepdims=True)
    return BaselineResult(
        method="lda",
        W_hat=W_hat,
        A_hat=A_hat,
        runtime_seconds=perf_counter() - started,
        converged=bool(model.n_iter_ < model.max_iter),
        metadata={
            "implementation": "sklearn.decomposition.LatentDirichletAllocation",
            "random_state": random_state,
            "n_iter": int(model.n_iter_),
            "max_iter": int(model.max_iter),
            "learning_method": model.learning_method,
        },
    )


def fit_spatial_lda(
    counts: np.ndarray,
    K: int,
    coordinates,
    *,
    sample_ids: np.ndarray | None = None,
    parameters: dict[str, Any] | None = None,
) -> BaselineResult:
    """Fit the vendored Calico spatial-LDA implementation on counts."""

    repo_root = Path(__file__).resolve().parents[2]
    utilities = repo_root / "utils"
    if str(utilities) in sys.path:
        sys.path.remove(str(utilities))
    # The environment may contain a different package with the same name.
    # Put the audited vendored implementation first.
    sys.path.insert(0, str(utilities))
    from spatial_lda import model as spatial_lda_model

    effective = {
        "difference_penalty": 0.25,
        "max_primal_dual_iter": 400,
        "max_dirichlet_iter": 20,
        "max_dirichlet_ls_iter": 10,
        "max_lda_iter": 5,
        "max_admm_iter": 15,
        "n_iters": 3,
        "n_parallel_processes": 1,
        "verbosity": 0,
        "primal_dual_mu": 2,
        "admm_rho": 1.0,
        "primal_tol": 1e-3,
        "threshold": None,
    }
    effective.update(parameters or {})
    started = perf_counter()
    count_array = np.asarray(counts, dtype=float)
    coordinate_array = np.asarray(coordinates, dtype=float)
    coordinate_frame = pd.DataFrame(coordinate_array, columns=["x", "y"])
    sample_count = 1
    if sample_ids is None:
        model = spatial_lda_model.run_simulation(
            count_array, K, coordinate_frame, **effective
        )
        regularization_scope = "single coordinate panel"
    else:
        sample_values = np.asarray(sample_ids, dtype=object).reshape(-1)
        if sample_values.size != count_array.shape[0]:
            raise ValueError("sample_ids must contain one value per count row")
        ordered_samples = list(dict.fromkeys(map(str, sample_values)))
        sample_count = len(ordered_samples)
        if sample_count == 1:
            model = spatial_lda_model.run_simulation(
                count_array, K, coordinate_frame, **effective
            )
            regularization_scope = "single coordinate panel"
        else:
            from spatial_lda.featurization import make_merged_difference_matrices

            sample_strings = np.asarray(list(map(str, sample_values)), dtype=object)
            global_indices = np.arange(count_array.shape[0], dtype=int)
            row_index = pd.MultiIndex.from_arrays(
                [sample_strings, global_indices], names=["sample", "observation"]
            )
            feature_frame = pd.DataFrame(count_array, index=row_index)
            coordinate_frames = {
                sample: pd.DataFrame(
                    coordinate_array[sample_strings == sample],
                    index=global_indices[sample_strings == sample],
                    columns=["x", "y"],
                )
                for sample in ordered_samples
            }
            difference_matrices = make_merged_difference_matrices(
                feature_frame, coordinate_frames, "x", "y"
            )
            model = spatial_lda_model.train(
                sample_features=feature_frame,
                difference_matrices=difference_matrices,
                n_topics=K,
                **effective,
            )
            regularization_scope = "separate within-sample coordinate panels"
    W_hat = np.asarray(model.topic_weights.values, dtype=float)
    W_hat /= W_hat.sum(axis=1, keepdims=True)
    A_hat = np.asarray(model.components_, dtype=float)
    A_hat /= A_hat.sum(axis=1, keepdims=True)
    return BaselineResult(
        method="spatial_lda",
        W_hat=W_hat,
        A_hat=A_hat,
        runtime_seconds=perf_counter() - started,
        converged=True,
        metadata={
            "implementation": "vendored calico/spatial_lda",
            "input_scale": "counts",
            "sample_count": sample_count,
            "spatial_regularization_scope": regularization_scope,
            **effective,
        },
    )
