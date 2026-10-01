"""Count-native adapters for the existing LDA and spatial-LDA baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
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

    started = perf_counter()
    if issparse(counts):
        D = counts.tocsr().astype(float, copy=False)
        model_input = D
        input_storage = "csr"
    else:
        D = np.asarray(counts)
        model_input = csr_matrix(D) if D.size > 10_000_000 else D
        input_storage = "csr" if model_input is not D else "dense"
    model = LatentDirichletAllocation(n_components=K, random_state=random_state)
    model.fit(model_input)
    W_hat = model.transform(model_input)
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
            "input_storage": input_storage,
        },
    )


def fit_spatial_lda(
    counts: np.ndarray,
    K: int,
    coordinates,
    *,
    sample_ids: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    parameters: dict[str, Any] | None = None,
) -> BaselineResult:
    """Fit the vendored Calico spatial-LDA implementation on counts.

    With coordinates, spatial LDA builds its own graph (Voronoi neighbours
    reduced to a minimum spanning tree, per sample). Without coordinates
    (Cooking), ``edges`` (n_edges x 2 row indices) are penalized directly:
    every given edge enters the difference matrix, no tree reduction.
    """

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
    sample_count = 1
    if coordinates is None:
        if edges is None:
            raise ValueError("spatial LDA needs coordinates or graph edges")
        from spatial_lda.featurization import make_difference_matrix

        endpoints = np.asarray(edges, dtype=int)
        rows = count_array.shape[0]
        feature_frame = pd.DataFrame(
            count_array,
            index=pd.MultiIndex.from_arrays(
                [np.full(rows, "corpus", dtype=object), np.arange(rows)], names=["sample", "observation"]
            ),
        )
        model = spatial_lda_model.train(
            sample_features=feature_frame,
            difference_matrices={"corpus": make_difference_matrix(rows, endpoints[:, 0], endpoints[:, 1])},
            n_topics=K,
            **effective,
        )
        regularization_scope = "given graph edges (no spanning-tree reduction)"
        return _spatial_lda_result(model, started, sample_count, regularization_scope, effective)
    coordinate_array = np.asarray(coordinates, dtype=float)
    coordinate_frame = pd.DataFrame(coordinate_array, columns=["x", "y"])
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
    return _spatial_lda_result(model, started, sample_count, regularization_scope, effective)


def _spatial_lda_result(model, started: float, sample_count: int, scope: str, effective: dict) -> BaselineResult:
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
            "spatial_regularization_scope": scope,
            **effective,
        },
    )


def _normalize_nmf_factors(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale nonnegative factors so rows of W and of A lie on the simplex."""

    W = np.maximum(np.asarray(W, dtype=float), 0)
    H = np.maximum(np.asarray(H, dtype=float), 0)
    topic_mass = np.maximum(H.sum(axis=1), np.finfo(float).eps)
    A = H / topic_mass[:, None]
    W = W * topic_mass[None, :]
    W /= np.maximum(W.sum(axis=1, keepdims=True), np.finfo(float).eps)
    return W, A


def fit_kl_nmf(counts: np.ndarray, K: int, *, random_state: int = 0) -> BaselineResult:
    """Poisson (KL) NMF by sklearn multiplicative updates."""

    from sklearn.decomposition import NMF

    started = perf_counter()
    model = NMF(
        n_components=K,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=500,
        tol=1e-4,
        random_state=random_state,
    )
    W, A = _normalize_nmf_factors(model.fit_transform(counts), model.components_)
    converged = model.n_iter_ < model.max_iter
    return BaselineResult(
        method="kl_nmf",
        W_hat=W,
        A_hat=A,
        runtime_seconds=perf_counter() - started,
        converged=converged,
        warnings=[] if converged else ["maximum_iterations_reached"],
        metadata={"implementation": "sklearn.NMF", "iterations": int(model.n_iter_)},
    )


def fit_graph_kl_nmf(
    counts: np.ndarray,
    adjacency: csr_matrix,
    K: int,
    *,
    random_state: int = 0,
    graph_penalty: float = 0.25,
    max_iter: int = 500,
    tolerance: float = 1e-5,
) -> BaselineResult:
    """Poisson/KL NMF with a Laplacian penalty Tr(W^T L W) on the row factors."""

    from sklearn.decomposition import NMF

    started = perf_counter()
    X = np.asarray(counts, dtype=float)
    initial = NMF(n_components=K, init="nndsvda", max_iter=1, random_state=random_state)
    try:
        W = np.maximum(initial.fit_transform(X), 1e-10)
        H = np.maximum(initial.components_, 1e-10)
    except Exception:
        rng = np.random.default_rng(random_state)
        W = rng.gamma(1.0, 1.0, size=(X.shape[0], K)) + 1e-10
        H = rng.gamma(1.0, 1.0, size=(K, X.shape[1])) + 1e-10
    symmetric = (adjacency + adjacency.T).tocsr()
    degree = np.asarray(symmetric.sum(axis=1)).ravel()
    previous = np.inf
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        ratio = X / np.maximum(W @ H, 1e-10)
        H *= (W.T @ ratio) / np.maximum(W.sum(axis=0)[:, None], 1e-10)
        ratio = X / np.maximum(W @ H, 1e-10)
        numerator = ratio @ H.T + graph_penalty * (symmetric @ W)
        denominator = H.sum(axis=1)[None, :] + graph_penalty * degree[:, None] * W
        W *= numerator / np.maximum(denominator, 1e-10)
        if iteration % 10 == 0 or iteration == max_iter:
            reconstruction = np.maximum(W @ H, 1e-10)
            objective = float(np.sum(reconstruction - X * np.log(reconstruction)))
            if np.isfinite(previous) and abs(previous - objective) <= tolerance * max(1.0, abs(previous)):
                converged = True
                break
            previous = objective
    W, A = _normalize_nmf_factors(W, H)
    return BaselineResult(
        method="graph_kl_nmf",
        W_hat=W,
        A_hat=A,
        runtime_seconds=perf_counter() - started,
        converged=converged,
        warnings=[] if converged else ["maximum_iterations_reached"],
        metadata={
            "implementation": "multiplicative KL-NMF with Tr(W^T L W)",
            "graph_penalty": float(graph_penalty),
            "iterations": int(iteration),
        },
    )
