"""Composable experiment blocks for audited real-data GpLSI comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .anchor_word import build_word_profile, recover_W_from_word_vertices
from .graphSVD import graphSVD
from .preprocessing import PreprocessingResult, preprocess_features, weighted_debiased_correction
from .real_data import RealDataBundle
from .recovery import (
    ARecoveryResult,
    PreparedPoissonCounts,
    RecoveryError,
    WRecoveryResult,
    poisson_objective_and_gradient,
    project_rows_simplex,
    recover_W,
    refit_A_current,
    refit_A_full_l2,
    refit_A_full_poisson,
)
from .topicscore import TopicScoreResult, fit_topicscore_graph_denoised
from .vertex_hunting import VertexHuntResult, vertex_hunt


PREPROCESSING_SPECS: dict[str, dict[str, Any]] = {
    "P0_raw": {"threshold_method": "none", "weight_method": "none"},
    "P1_tran_alpha_0p005": {
        "threshold_method": "tran_script_exact",
        "weight_method": "none",
    },
    "P2_ke_weighted": {"threshold_method": "none", "weight_method": "ke_empirical"},
    "P3_tran_then_ke": {
        "threshold_method": "tran_script_exact",
        "weight_method": "ke_empirical",
    },
}


@dataclass
class SpectralBlock:
    preprocessing_name: str
    preprocessing: PreprocessingResult
    positive_canonical_indices: np.ndarray
    retained_canonical_indices: np.ndarray
    U_hat: np.ndarray
    V_hat: np.ndarray
    singular_values: np.ndarray
    U_bar: np.ndarray
    graph_metadata: dict[str, Any]
    selected_rho: float
    cv_errors: dict[str, Any]
    used_iterations: int
    runtime_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class GeometryFit:
    estimator_family: str
    spectral_geometry: str
    vertex_hunter: str
    W_hat: np.ndarray
    vertices: np.ndarray
    vertex_result: VertexHuntResult
    W_recovery: WRecoveryResult | Any
    selected_vocabulary_indices: np.ndarray | None
    selected_profile_indices: np.ndarray | None
    selected_candidate_distances: np.ndarray | None
    runtimes: dict[str, float]
    geometry_diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def fit_spectral_block(
    bundle: RealDataBundle,
    K: int,
    preprocessing_name: str,
    *,
    seed: int,
    lamb_start: float = 1e-4,
    step_size: float = 1.2,
    grid_len: int = 29,
    maxiter: int = 50,
    eps: float = 1e-5,
    nfolds: int = 5,
    n_jobs: int = 1,
    initialization: str = "current",
) -> SpectralBlock:
    """Preprocess once, retune rho, and return reusable graph-SVD factors."""

    if preprocessing_name not in PREPROCESSING_SPECS:
        raise ValueError(
            f"unknown preprocessing {preprocessing_name!r}; "
            f"expected {sorted(PREPROCESSING_SPECS)}"
        )
    X = np.asarray(bundle.frequencies, dtype=float)
    positive = np.flatnonzero(X.mean(axis=0) > 0)
    if positive.size < K:
        raise ValueError(f"only {positive.size} positive-frequency columns remain for K={K}")
    spec = PREPROCESSING_SPECS[preprocessing_name]
    processed = preprocess_features(
        X[:, positive],
        bundle.document_lengths,
        alpha=0.005,
        K=K,
        fail_on_rank_loss=True,
        weight_cap=None,
        weight_common_scale="none",
        tau=0.0,
        **spec,
    )
    retained = positive[processed.threshold.retained_indices]
    eta = X[:, retained].mean(axis=0)
    correction = None
    if initialization in {"weighted_debiased", "weighted_debiased_mean_N_approx"}:
        correction = weighted_debiased_correction(
            eta,
            processed.weighting.weights,
            n=bundle.n,
            N=bundle.N_mean,
        )
    started = perf_counter()
    (
        U,
        V,
        L,
        _,
        _,
        _,
        rho,
        errors,
        iterations,
        graph_metadata,
    ) = graphSVD(
        processed.X_transformed,
        bundle.N_mean,
        K,
        bundle.edge_df,
        bundle.weights,
        lamb_start,
        step_size,
        grid_len,
        maxiter,
        eps,
        0,
        True,
        initialization=initialization,
        debias_correction=correction,
        return_metadata=True,
        random_state=seed,
        nfolds=nfolds,
        cv_fold_mode="all",
        n_jobs=n_jobs,
    )
    warning_messages = list(processed.threshold.warnings)
    if positive.size != bundle.p:
        warning_messages.append(
            f"mechanically_removed_{bundle.p - positive.size}_zero_frequency_columns"
        )
    return SpectralBlock(
        preprocessing_name=preprocessing_name,
        preprocessing=processed,
        positive_canonical_indices=positive,
        retained_canonical_indices=retained,
        U_hat=U,
        V_hat=V,
        singular_values=np.diag(L).copy(),
        U_bar=graph_metadata["U_bar"],
        graph_metadata=graph_metadata,
        selected_rho=float(rho),
        cv_errors=errors,
        used_iterations=int(iterations),
        runtime_seconds=perf_counter() - started,
        warnings=warning_messages,
    )


def _vertex_parameters(
    method: str,
    K: int,
    point_count: int,
    configured: dict[str, Any] | None,
) -> dict[str, Any]:
    parameters = dict(configured or {})
    if method in {"svs", "svs_star"}:
        parameters.setdefault("L_mode", "fixed")
        parameters.setdefault("L", min(K + 2, point_count))
        parameters.setdefault("max_simplexes", 200_000)
    elif method == "pp_spa":
        parameters.setdefault("radius_divisor", 20.0)
        parameters.setdefault("m_neighbors", 4)
        parameters.setdefault("min_neighbors", 3)
    elif method == "spa_current":
        parameters.setdefault("precondition", False)
    elif method in {"palm", "palm_accelerated"}:
        parameters.setdefault("lambda_", 1.0)
        parameters.setdefault("max_iterations", 300)
        parameters.setdefault("tolerance", 1e-7)
        parameters.setdefault("c_h", 1.1)
        parameters.setdefault("c_w", 1.1)
        parameters.setdefault("projection_tolerance", 1e-10)
        parameters.setdefault("projection_max_iterations", 10_000)
        parameters.setdefault("hull_reduction", "none")
        parameters.setdefault("initialize_weights_iterations", 200)
        parameters.setdefault("final_weight_refit_iterations", 200)
        parameters.setdefault("initialization_precondition", False)
    return parameters


def fit_geometry(
    bundle: RealDataBundle,
    block: SpectralBlock,
    K: int,
    *,
    geometry: str,
    vertex_hunter: str,
    seed: int,
    vertex_parameters: dict[str, Any] | None = None,
    condition_threshold: float = 1e12,
) -> GeometryFit:
    """Fit one document- or anchor-feature-side vertex geometry."""

    selected_words: np.ndarray | None = None
    selected_profile: np.ndarray | None = None
    selected_distances: np.ndarray | None = None
    geometry_diagnostics: dict[str, Any] = {}
    profile_runtime = 0.0
    if geometry == "document_U":
        point_cloud = block.U_hat
        family = "document_gplsi"
        word_profile = None
    elif geometry == "word_Z":
        started = perf_counter()
        word_profile = build_word_profile(
            bundle.frequencies,
            block.U_hat,
            block.V_hat,
            block.singular_values,
            block.preprocessing.weighting.weights,
            block.retained_canonical_indices,
        )
        profile_runtime = perf_counter() - started
        point_cloud = word_profile.Z_hat
        family = "anchor_feature_gplsi"
    else:
        raise ValueError("geometry must be 'document_U' or 'word_Z'")

    parameters = _vertex_parameters(
        vertex_hunter, K, point_cloud.shape[0], vertex_parameters
    )

    vertex = vertex_hunt(
        point_cloud,
        K,
        vertex_hunter,
        random_state=seed,
        condition_threshold=condition_threshold,
        raise_on_failure=True,
        **parameters,
    )
    recovery_started = perf_counter()
    if geometry == "document_U":
        recovery_embedding = vertex.embedding_used
        if recovery_embedding is None:
            recovery_embedding = block.U_hat
        W_result = recover_W(
            recovery_embedding, vertex.vertices, condition_threshold=condition_threshold
        )
    else:
        U_for_recovery = block.U_hat
        if vertex_hunter == "spa_current":
            signs = np.asarray(vertex.parameters["coordinate_signs"], dtype=float)
            U_for_recovery = U_for_recovery * signs[None, :]
        W_result = recover_W_from_word_vertices(
            U_for_recovery,
            vertex.vertices,
            condition_threshold=condition_threshold,
        )
        distances = np.linalg.norm(
            point_cloud[:, None, :] - vertex.vertices[None, :, :], axis=2
        )
        selected_profile = np.argmin(distances, axis=0).astype(int)
        selected_distances = distances[selected_profile, np.arange(K)]
        selected_words = block.retained_canonical_indices[selected_profile]
        try:
            barycentric = np.linalg.solve(vertex.vertices.T, point_cloud.T).T
            projected_barycentric = project_rows_simplex(barycentric)
            containment_distances = np.linalg.norm(
                projected_barycentric @ vertex.vertices - point_cloud, axis=1
            )
            affine_residual = np.abs(barycentric.sum(axis=1) - 1.0)
            outside = (barycentric < -1e-8).any(axis=1) | (affine_residual > 1e-8)
            pairwise = np.linalg.norm(
                vertex.vertices[:, None, :] - vertex.vertices[None, :, :], axis=2
            )
            pairwise[pairwise == 0] = np.inf
            geometry_diagnostics = {
                "feature_simplex_affine_residual_max": float(affine_residual.max()),
                "feature_simplex_containment_residual_mean": float(
                    containment_distances.mean()
                ),
                "feature_simplex_containment_residual_max": float(
                    containment_distances.max()
                ),
                "feature_simplex_outside_fraction": float(outside.mean()),
                "minimum_pairwise_vertex_distance": float(pairwise.min()),
                "unique_selected_candidate_count": int(np.unique(selected_words).size),
            }
        except np.linalg.LinAlgError:
            geometry_diagnostics = {
                "feature_simplex_diagnostic_status": "singular_vertex_matrix"
            }
    recovery_runtime = perf_counter() - recovery_started
    return GeometryFit(
        estimator_family=family,
        spectral_geometry=geometry,
        vertex_hunter=vertex_hunter,
        W_hat=W_result.simplex_projected,
        vertices=vertex.vertices,
        vertex_result=vertex,
        W_recovery=W_result,
        selected_vocabulary_indices=selected_words,
        selected_profile_indices=selected_profile,
        selected_candidate_distances=selected_distances,
        runtimes={
            "word_profile": profile_runtime,
            "vertex_hunting": vertex.runtime_seconds,
            "W_recovery": recovery_runtime,
        },
        geometry_diagnostics=geometry_diagnostics,
        warnings=list(vertex.warnings) + list(W_result.warnings),
    )


def recover_A_for_geometry(
    bundle: RealDataBundle,
    geometry_fit: GeometryFit,
    method: str,
    *,
    poisson_max_iter: int = 2_000,
    poisson_tolerance: float = 1e-8,
    poisson_counts: PreparedPoissonCounts | None = None,
    poisson_initial_A: np.ndarray | None = None,
) -> tuple[ARecoveryResult, float]:
    """Apply A-current, simplex least-squares, or Poisson A to the same W."""

    started = perf_counter()
    if method == "A_current":
        result = refit_A_current(geometry_fit.W_hat, bundle.frequencies)
    elif method == "A_full_L2":
        # Same warm start as the Poisson refit, so the two full-vocabulary
        # estimators differ only in their loss.
        result = refit_A_full_l2(
            geometry_fit.W_hat,
            bundle.frequencies,
            initial_A=poisson_initial_A,
            max_iter=poisson_max_iter,
            tolerance=poisson_tolerance,
        )
    elif method == "A_full_Pois":
        # Use the exact paired historical estimate as a deterministic warm
        # start.  The Poisson routine interiorizes it once so EM can reopen
        # coordinates that the simplex projection set to zero.
        warm_start = poisson_initial_A
        warm_start_warning = None
        if warm_start is None:
            try:
                warm_start = refit_A_current(geometry_fit.W_hat, bundle.frequencies).A_hat
            except RecoveryError as error:
                warm_start_warning = (
                    "paired_A_current_warm_start_failed; used pooled feature frequencies: "
                    f"{error}"
                )
        result = refit_A_full_poisson(
            geometry_fit.W_hat,
            bundle.counts if poisson_counts is None else poisson_counts,
            bundle.document_lengths,
            initial_A=warm_start,
            max_iter=poisson_max_iter,
            tolerance=poisson_tolerance,
        )
        result.diagnostics["warm_start"] = (
            "paired_A_current" if warm_start is not None else "pooled_feature_frequencies"
        )
        if warm_start_warning is not None:
            result.warnings.append(warm_start_warning)
    else:
        raise ValueError("A recovery must be 'A_current', 'A_full_L2', or 'A_full_Pois'")
    return result, perf_counter() - started


def fit_graph_topicscore(
    bundle: RealDataBundle, block: SpectralBlock
) -> TopicScoreResult:
    return fit_topicscore_graph_denoised(
        bundle.frequencies,
        block.U_hat,
        block.V_hat,
        block.singular_values,
        retained_indices=block.retained_canonical_indices,
        weights=block.preprocessing.weighting.weights,
    )


def fit_diagnostics(
    bundle: RealDataBundle,
    W_hat: np.ndarray,
    A_hat: np.ndarray,
) -> dict[str, float]:
    W = np.asarray(W_hat, dtype=float)
    A = np.asarray(A_hat, dtype=float)
    reconstruction = W @ A
    residual = bundle.frequencies - reconstruction
    poisson, gradient = poisson_objective_and_gradient(
        A, W, bundle.counts, bundle.document_lengths, 1e-12
    )
    endpoints = bundle.edge_df[["src", "tgt"]].to_numpy(dtype=int)
    edge_weights = bundle.edge_df["weight"].to_numpy(dtype=float)
    if endpoints.size:
        differences = W[endpoints[:, 0]] - W[endpoints[:, 1]]
        smoothness = float(np.average(np.sum(differences**2, axis=1), weights=edge_weights))
        labels = np.argmax(W, axis=1)
        neighbor_agreement = float(
            np.average(labels[endpoints[:, 0]] == labels[endpoints[:, 1]], weights=edge_weights)
        )
        centered_numeric = labels.astype(float) - float(np.mean(labels))
        numeric_denominator = float(np.sum(centered_numeric**2) * np.sum(edge_weights))
        historical_moran = (
            np.nan
            if numeric_denominator <= 0
            else float(
                bundle.n
                * np.sum(
                    edge_weights
                    * centered_numeric[endpoints[:, 0]]
                    * centered_numeric[endpoints[:, 1]]
                )
                / numeric_denominator
            )
        )
        incident: dict[int, list[float]] = {}
        disagreements = labels[endpoints[:, 0]] != labels[endpoints[:, 1]]
        for side in (0, 1):
            for node in np.unique(endpoints[:, side]):
                incident.setdefault(int(node), []).append(
                    float(np.mean(disagreements[endpoints[:, side] == node]))
                )
        summed_incident = np.asarray([sum(values) for values in incident.values()])
        historical_one_minus_pas = float(1.0 - np.mean(summed_incident >= 0.6))
        one_hot_moran: list[float] = []
        for topic in range(W.shape[1]):
            indicator = (labels == topic).astype(float)
            centered = indicator - indicator.mean()
            denominator = float(np.sum(centered**2) * np.sum(edge_weights))
            if denominator > 0:
                one_hot_moran.append(
                    float(
                        bundle.n
                        * np.sum(
                            edge_weights
                            * centered[endpoints[:, 0]]
                            * centered[endpoints[:, 1]]
                        )
                        / denominator
                    )
                )
    else:
        smoothness = np.nan
        neighbor_agreement = np.nan
        historical_moran = np.nan
        historical_one_minus_pas = np.nan
        one_hot_moran = []
    return {
        "reconstruction_fro": float(np.linalg.norm(residual, ord="fro")),
        "reconstruction_l1": float(np.sum(np.abs(residual))),
        "poisson_objective": float(poisson),
        "poisson_gradient_norm": float(np.linalg.norm(gradient)),
        "graph_W_smoothness": smoothness,
        "graph_neighbor_topic_agreement": neighbor_agreement,
        "historical_numeric_label_moran": historical_moran,
        "historical_1_minus_PAS": historical_one_minus_pas,
        "permutation_invariant_one_hot_moran_mean": (
            np.nan if not one_hot_moran else float(np.mean(one_hot_moran))
        ),
        "W_simplex_error": float(np.max(np.abs(W.sum(axis=1) - 1.0))),
        "W_min": float(W.min()),
        "A_simplex_error": float(np.max(np.abs(A.sum(axis=1) - 1.0))),
        "A_min": float(A.min()),
        "W_rank": int(np.linalg.matrix_rank(W)),
        "A_rank": int(np.linalg.matrix_rank(A)),
    }


def heldout_count_diagnostics(
    W_hat: np.ndarray,
    A_hat: np.ndarray,
    heldout_counts: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Evaluate a train-fitted W/A pair on untouched thinned counts."""

    W = np.asarray(W_hat, dtype=float)
    A = np.asarray(A_hat, dtype=float)
    test = np.asarray(heldout_counts, dtype=float)
    if test.shape != (W.shape[0], A.shape[1]) or np.any(test < 0):
        raise ValueError("held-out counts do not align with W and A")
    lengths = test.sum(axis=1)
    evaluated = lengths > 0
    if not np.any(evaluated):
        raise ValueError("held-out evaluation has no positive-count rows")
    test = test[evaluated]
    W = W[evaluated]
    lengths = lengths[evaluated]
    probabilities = np.maximum(W @ A, 0.0)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), epsilon)
    stabilized = probabilities + epsilon
    means = lengths[:, None] * probabilities + epsilon
    positive = test > 0
    log_likelihood = float(np.sum(test * np.log(stabilized)))
    poisson_deviance = float(
        2.0
        * (
            np.sum(test[positive] * np.log(test[positive] / means[positive]))
            - np.sum(test - means)
        )
    )
    empirical = test / lengths[:, None]
    residual = empirical - probabilities
    return {
        "heldout_multinomial_log_likelihood_without_constant": log_likelihood,
        "heldout_poisson_deviance": poisson_deviance,
        "heldout_multinomial_deviance": poisson_deviance,
        "heldout_frequency_fro": float(np.linalg.norm(residual, ord="fro")),
        "heldout_frequency_l1": float(np.sum(np.abs(residual))),
        "heldout_count_total": float(test.sum()),
        "heldout_evaluated_row_count": int(np.count_nonzero(evaluated)),
        "heldout_zero_count_row_count": int(evaluated.size - np.count_nonzero(evaluated)),
    }
