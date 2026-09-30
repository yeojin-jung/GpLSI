"""Estimator registry for the predeclared benchmark matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import traceback

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

from gplsi.baselines import fit_lda, fit_spatial_lda
from gplsi.real_experiment import (
    fit_geometry,
    fit_graph_topicscore,
    fit_spectral_block,
    recover_A_for_geometry,
)
from gplsi.topicscore import fit_topicscore_raw
from gplsi.recovery import prepare_poisson_counts


@dataclass
class Estimate:
    method: str
    W: np.ndarray | None
    A: np.ndarray | None
    runtime_seconds: float
    status: str = "ok"
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize_factors(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = np.maximum(np.asarray(W, dtype=float), 0)
    H = np.maximum(np.asarray(H, dtype=float), 0)
    topic_mass = np.maximum(H.sum(axis=1), np.finfo(float).eps)
    A = H / topic_mass[:, None]
    W = W * topic_mass[None, :]
    W /= np.maximum(W.sum(axis=1, keepdims=True), np.finfo(float).eps)
    return W, A


def fit_kl_nmf(counts: np.ndarray, K: int, seed: int) -> Estimate:
    started = perf_counter()
    model = NMF(
        n_components=K,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=500,
        tol=1e-4,
        random_state=seed,
    )
    W, A = _normalize_factors(model.fit_transform(counts), model.components_)
    return Estimate(
        "kl_nmf", W, A, perf_counter() - started,
        metadata={"implementation": "sklearn.NMF", "iterations": int(model.n_iter_)},
        warnings=[] if model.n_iter_ < model.max_iter else ["maximum_iterations_reached"],
    )


def fit_graph_kl_nmf(
    counts: np.ndarray,
    adjacency: csr_matrix,
    K: int,
    seed: int,
    *,
    graph_penalty: float = 0.25,
    max_iter: int = 500,
    tolerance: float = 1e-5,
) -> Estimate:
    """Poisson/KL NMF with a transparent Laplacian penalty on cell factors."""

    started = perf_counter()
    X = np.asarray(counts, dtype=float)
    initial = NMF(n_components=K, init="nndsvda", max_iter=1, random_state=seed)
    try:
        W = np.maximum(initial.fit_transform(X), 1e-10)
        H = np.maximum(initial.components_, 1e-10)
    except Exception:
        rng = np.random.default_rng(seed)
        W = rng.gamma(1.0, 1.0, size=(X.shape[0], K)) + 1e-10
        H = rng.gamma(1.0, 1.0, size=(K, X.shape[1])) + 1e-10
    sym = (adjacency + adjacency.T).tocsr()
    degree = np.asarray(sym.sum(axis=1)).ravel()
    previous = np.inf
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        ratio = X / np.maximum(W @ H, 1e-10)
        H *= (W.T @ ratio) / np.maximum(W.sum(axis=0)[:, None], 1e-10)
        ratio = X / np.maximum(W @ H, 1e-10)
        numerator = ratio @ H.T + graph_penalty * (sym @ W)
        denominator = H.sum(axis=1)[None, :] + graph_penalty * degree[:, None] * W
        W *= numerator / np.maximum(denominator, 1e-10)
        if iteration % 10 == 0 or iteration == max_iter:
            reconstruction = np.maximum(W @ H, 1e-10)
            objective = float(np.sum(reconstruction - X * np.log(reconstruction)))
            differences = W[:, None, :]  # do not materialize pairwise distances
            del differences
            if np.isfinite(previous) and abs(previous - objective) <= tolerance * max(1.0, abs(previous)):
                converged = True
                break
            previous = objective
    W, A = _normalize_factors(W, H)
    return Estimate(
        "graph_kl_nmf", W, A, perf_counter() - started,
        metadata={
            "implementation": "benchmark multiplicative KL-NMF with Tr(W^T L W)",
            "graph_penalty": float(graph_penalty),
            "iterations": int(iteration),
        },
        warnings=[] if converged else ["maximum_iterations_reached"],
    )


def _failed(method: str, started: float, exception: Exception) -> Estimate:
    return Estimate(
        method, None, None, perf_counter() - started, status="failed",
        metadata={"exception_type": type(exception).__name__, "exception": str(exception),
                  "traceback": traceback.format_exc(limit=12)},
    )


def fit_method_suite(
    bundle,
    coordinates: np.ndarray,
    K: int,
    seed: int,
    *,
    document_preprocessings: tuple[str, ...] = (
        "P0_raw", "P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke"
    ),
    initialization: str = "current",
    document_vertex_hunters: tuple[str, ...] = (
        "spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"
    ),
    anchor_preprocessings: tuple[str, ...] = ("P0_raw",),
    anchor_vertex_hunters: tuple[str, ...] = ("spa_current",),
    include_spatial_lda: bool = True,
    spectral_parameters: dict | None = None,
    A_recoveries: tuple[str, ...] = ("A_current", "A_full_Pois"),
    vertex_parameters: dict[str, dict] | None = None,
    competitors: tuple[str, ...] | None = None,
    recovery_parameters: dict | None = None,
) -> list[Estimate]:
    """Fit every declared estimator; failures become explicit result records.

    ``A_recoveries`` are all applied to the identical W of each geometry.
    ``A_current`` is always computed first when requested, because it is the
    warm start for the full-vocabulary refits. ``competitors=None`` keeps every
    non-GpLSI baseline; otherwise only the named ones are fitted.
    """

    results: list[Estimate] = []
    A_recoveries = tuple(sorted(dict.fromkeys(A_recoveries), key=lambda name: name != "A_current"))
    vertex_parameters = vertex_parameters or {}
    recovery_parameters = recovery_parameters or {}

    def wanted(name: str) -> bool:
        return competitors is None or name in competitors
    prepared_poisson_counts = prepare_poisson_counts(bundle.counts)
    all_preprocessings = tuple(dict.fromkeys(document_preprocessings + anchor_preprocessings))
    first_spectral = None
    for preprocessing in all_preprocessings:
        spectral = None
        started = perf_counter()
        try:
            spectral = fit_spectral_block(
                bundle, K, preprocessing, seed=seed, initialization=initialization,
                **(spectral_parameters or {}),
            )
            if first_spectral is None:
                first_spectral = spectral
        except Exception as exc:
            results.append(_failed(f"gplsi_shared_spectral_block__{preprocessing}", started, exc))
            if preprocessing in document_preprocessings:
                for hunter in document_vertex_hunters:
                    for recovery in A_recoveries:
                        results.append(_failed(
                            f"gplsi_document__{preprocessing}__{hunter}__{recovery}", started, exc
                        ))
            if preprocessing in anchor_preprocessings:
                for hunter in anchor_vertex_hunters:
                    for recovery in A_recoveries:
                        results.append(_failed(
                            f"gplsi_anchor__{preprocessing}__{hunter}__{recovery}", started, exc
                        ))
            continue
        geometry_grid = []
        if preprocessing in document_preprocessings:
            geometry_grid.extend(("document", "document_U", hunter) for hunter in document_vertex_hunters)
        if preprocessing in anchor_preprocessings:
            geometry_grid.extend(("anchor", "word_Z", hunter) for hunter in anchor_vertex_hunters)
        for geometry_name, geometry, hunter in geometry_grid:
            geometry_method = f"gplsi_{geometry_name}__{preprocessing}__{hunter}"
            started = perf_counter()
            try:
                fitted = fit_geometry(
                    bundle, spectral, K, geometry=geometry, vertex_hunter=hunter, seed=seed,
                    vertex_parameters=vertex_parameters.get(hunter),
                )
                paired_current_A = None
                for recovery in A_recoveries:
                    recovery_started = perf_counter()
                    try:
                        recovered, a_runtime = recover_A_for_geometry(
                            bundle,
                            fitted,
                            recovery,
                            poisson_counts=prepared_poisson_counts,
                            poisson_initial_A=paired_current_A,
                            **recovery_parameters,
                        )
                        if recovery == "A_current":
                            paired_current_A = recovered.A_hat
                        results.append(Estimate(
                            f"{geometry_method}__{recovery}", fitted.W_hat, recovered.A_hat,
                            spectral.runtime_seconds + sum(fitted.runtimes.values()) + a_runtime,
                            status="ok" if recovered.converged else recovered.status,
                            metadata={
                                "spectral_preprocessing": preprocessing,
                                "spectral_initialization": initialization,
                                "selected_rho": spectral.selected_rho,
                                "spectral_iterations": spectral.used_iterations,
                                "geometry": fitted.geometry_diagnostics,
                                "A_recovery": {
                                    "method": recovered.method,
                                    "status": recovered.status,
                                    "converged": recovered.converged,
                                    "iterations": recovered.iterations,
                                    "gradient_norm": recovered.gradient_norm,
                                    "projected_gradient_norm": recovered.projected_gradient_norm,
                                    "optimality_gap": recovered.optimality_gap,
                                    "normalized_optimality_gap": recovered.normalized_optimality_gap,
                                    "solver": recovered.solver,
                                    "objective_name": recovered.objective_name,
                                    "diagnostics": recovered.diagnostics,
                                },
                                "vertex_hunting": {
                                    "status": fitted.vertex_result.status,
                                    "runtime_seconds": fitted.runtimes.get("vertex_hunting"),
                                    "optimizer_converged": fitted.vertex_result.parameters.get(
                                        "optimizer_converged"
                                    ),
                                },
                                "retained_feature_count": int(spectral.retained_canonical_indices.size),
                                "shared_spectral_runtime_seconds": spectral.runtime_seconds,
                            },
                            warnings=spectral.warnings + fitted.warnings + recovered.warnings,
                        ))
                    except Exception as exc:
                        results.append(_failed(f"{geometry_method}__{recovery}", recovery_started, exc))
            except Exception as exc:
                for recovery in A_recoveries:
                    results.append(_failed(f"{geometry_method}__{recovery}", started, exc))
    if first_spectral is not None and wanted("topicscore_graph_denoised"):
        started = perf_counter()
        try:
            fitted = fit_graph_topicscore(bundle, first_spectral)
            results.append(Estimate(
                "topicscore_graph_denoised", fitted.W_hat, fitted.A_hat,
                first_spectral.runtime_seconds + fitted.runtime_seconds,
                metadata={**fitted.metadata, "shared_preprocessing": first_spectral.preprocessing_name},
                warnings=fitted.warnings,
            ))
        except Exception as exc:
            results.append(_failed("topicscore_graph_denoised", started, exc))

    independent = [
        ("topicscore_raw", lambda: fit_topicscore_raw(bundle.frequencies, K)),
        ("lda", lambda: fit_lda(bundle.counts, K, random_state=seed)),
        ("kl_nmf", lambda: fit_kl_nmf(bundle.counts, K, seed)),
        ("graph_kl_nmf", lambda: fit_graph_kl_nmf(bundle.counts, bundle.weights, K, seed)),
    ]
    if include_spatial_lda:
        independent.append(("spatial_lda", lambda: fit_spatial_lda(bundle.counts, K, coordinates)))
    independent = [(name, callback) for name, callback in independent if wanted(name)]
    for name, callback in independent:
        started = perf_counter()
        try:
            fitted = callback()
            if isinstance(fitted, Estimate):
                results.append(fitted)
            else:
                results.append(Estimate(
                    name, fitted.W_hat, fitted.A_hat, fitted.runtime_seconds,
                    status=fitted.status, metadata=fitted.metadata, warnings=fitted.warnings,
                ))
        except Exception as exc:
            results.append(_failed(name, started, exc))
    return results
