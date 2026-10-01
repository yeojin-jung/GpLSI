"""GpLSI building blocks shared by all experiments.

One spectral block (preprocessing + graph-aligned SVD) is fitted per
preprocessing and reused by every vertex hunter and geometry; every A recovery
is then applied to the identical W.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np

from .anchor_word import build_word_profile, recover_W_from_word_vertices
from .graphSVD import graphSVD
from .preprocessing import (
    PreprocessingResult,
    ThresholdResult,
    preprocess_features,
    weighted_debiased_correction,
)
from .real_data import RealDataBundle
from .recovery import (
    ARecoveryResult,
    PreparedPoissonCounts,
    RecoveryError,
    WRecoveryResult,
    project_rows_simplex,
    recover_W,
    refit_A_current,
    refit_A_full_l2,
    refit_A_full_poisson,
    refit_A_full_poisson_squarem,
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
    "P1_tran_alpha_0p1": {
        "threshold_method": "tran_paper_exact",
        "weight_method": "none",
        "alpha": 0.1,
    },
    "P3_tran_alpha_0p1_then_ke": {
        "threshold_method": "tran_paper_exact",
        "weight_method": "ke_empirical",
        "alpha": 0.1,
    },
    "P1_tran_alpha_0p1_drop_zero_rows": {
        "threshold_method": "tran_paper_exact",
        "weight_method": "none",
        "alpha": 0.1,
    },
    "P3_tran_alpha_0p1_then_ke_drop_zero_rows": {
        "threshold_method": "tran_paper_exact",
        "weight_method": "ke_empirical",
        "alpha": 0.1,
    },
    "P1_tran_alpha_0p01_drop_zero_rows": {
        "threshold_method": "tran_paper_exact",
        "weight_method": "none",
        "alpha": 0.01,
    },
    "P3_tran_alpha_0p01_then_ke_drop_zero_rows": {
        "threshold_method": "tran_paper_exact",
        "weight_method": "ke_empirical",
        "alpha": 0.01,
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
    cv_fold_mode: str = "all",
    lambda_selection_mode: str = "cv_each_iteration",
) -> SpectralBlock:
    """Preprocess once, retune rho, and return reusable graph-SVD factors."""

    if preprocessing_name not in PREPROCESSING_SPECS:
        raise ValueError(
            f"unknown preprocessing {preprocessing_name!r}; "
            f"expected {sorted(PREPROCESSING_SPECS)}"
        )
    X = np.asarray(bundle.frequencies, dtype=float)
    spec = dict(PREPROCESSING_SPECS[preprocessing_name])
    alpha = float(spec.pop("alpha", 0.005))
    frozen = bundle.metadata.get("frozen_tran_feature_support")
    if frozen and preprocessing_name in frozen.get("matching_preprocessings", []):
        positive = np.asarray(
            frozen["positive_canonical_indices"], dtype=np.int64
        )
        retained = np.asarray(
            frozen["retained_canonical_indices"], dtype=np.int64
        )
        if (
            positive.ndim != 1
            or retained.ndim != 1
            or positive.size < K
            or retained.size < K
            or np.any(positive < 0)
            or np.any(positive >= bundle.p)
            or np.any(retained < 0)
            or np.any(retained >= bundle.p)
        ):
            raise ValueError("frozen Tran support is invalid for this bundle and K")
        positive_position = {int(value): index for index, value in enumerate(positive)}
        try:
            retained_relative = np.asarray(
                [positive_position[int(value)] for value in retained], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError("frozen retained support is not contained in positive support") from error
        requested_method = str(spec.pop("threshold_method", "none"))
        processed_base = preprocess_features(
            X[:, retained],
            bundle.document_lengths,
            threshold_method="none",
            alpha=alpha,
            K=K,
            fail_on_rank_loss=True,
            weight_cap=None,
            weight_common_scale="none",
            tau=0.0,
            **spec,
        )
        retained_row_mass = X[:, retained].sum(axis=1)
        if np.any(retained_row_mass <= 0):
            raise ValueError(
                "frozen Tran support contains a zero-mass recipe after row filtering"
            )
        frozen_threshold = ThresholdResult(
            retained_indices=retained_relative,
            discarded_indices=np.setdiff1d(
                np.arange(positive.size, dtype=np.int64), retained_relative
            ),
            threshold_value=float(frozen["threshold_value"]),
            alpha=float(frozen["alpha"]),
            eta_hat=np.asarray(frozen["eta_hat_positive"], dtype=float),
            retained_feature_count=int(retained.size),
            retained_feature_fraction=float(retained.size / positive.size),
            retained_row_mass=retained_row_mass,
            retained_topic_mass=None,
            retained_population_rank=None,
            rank_ok=None,
            requested_method=requested_method,
            effective_method=str(frozen["threshold_method"]),
            N_used=float(frozen["N_used"]),
            unequal_document_lengths=bool(
                frozen.get("unequal_document_lengths", False)
            ),
            fallback_active=bool(frozen.get("top_10_percent_fallback", False)),
            warnings=[
                *[str(value) for value in frozen.get("threshold_warnings", [])],
                "frozen_tran_support_after_zero_training_mass_row_filter",
            ],
        )
        processed = replace(
            processed_base,
            X_original=X[:, positive],
            threshold=frozen_threshold,
        )
    else:
        positive = np.flatnonzero(X.mean(axis=0) > 0)
        if positive.size < K:
            raise ValueError(
                f"only {positive.size} positive-frequency columns remain for K={K}"
            )
        processed = preprocess_features(
            X[:, positive],
            bundle.document_lengths,
            alpha=alpha,
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
        cv_fold_mode=cv_fold_mode,
        n_jobs=n_jobs,
        lambda_selection_mode=lambda_selection_mode,
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


A_RECOVERIES = ("A_current", "A_full_L2", "A_full_Pois", "A_full_Pois_SQUAREM")
POISSON_STARTS = ("A_current", "pooled")


def recover_A(
    bundle: RealDataBundle,
    W: np.ndarray,
    method: str,
    *,
    poisson_max_iter: int = 2_000,
    poisson_tolerance: float = 1e-8,
    poisson_start: str = "A_current",
    poisson_counts: PreparedPoissonCounts | None = None,
    poisson_initial_A: np.ndarray | None = None,
    statistics_backend: str = "auto",
) -> tuple[ARecoveryResult, float]:
    """Apply one A recovery to a fixed W (shared by all recoveries of a fit).

    ``A_full_L2`` and the Poisson refits start from ``poisson_initial_A`` when
    given; otherwise ``poisson_start`` chooses between the paired A-current
    estimate ("A_current") and pooled feature frequencies ("pooled").
    """

    if method not in A_RECOVERIES:
        raise ValueError(f"A recovery must be one of {A_RECOVERIES}, not {method!r}")
    if poisson_start not in POISSON_STARTS:
        raise ValueError(f"poisson_start must be one of {POISSON_STARTS}")
    started = perf_counter()
    if method == "A_current":
        return refit_A_current(W, bundle.frequencies), perf_counter() - started

    warm_start = poisson_initial_A
    warm_start_warning = None
    if warm_start is None and poisson_start == "A_current":
        try:
            warm_start = refit_A_current(W, bundle.frequencies).A_hat
        except RecoveryError as error:
            warm_start_warning = (
                "paired_A_current_warm_start_failed; used pooled feature frequencies: "
                f"{error}"
            )
    if method == "A_full_L2":
        result = refit_A_full_l2(
            W,
            bundle.frequencies,
            initial_A=warm_start,
            max_iter=poisson_max_iter,
            tolerance=poisson_tolerance,
        )
    else:
        # The Poisson routines interiorize the start once so EM can reopen
        # coordinates that a simplex projection set to zero.
        counts = bundle.counts if poisson_counts is None else poisson_counts
        if method == "A_full_Pois":
            result = refit_A_full_poisson(
                W,
                counts,
                bundle.document_lengths,
                initial_A=warm_start,
                max_iter=poisson_max_iter,
                tolerance=poisson_tolerance,
                statistics_backend=statistics_backend,
            )
        else:
            result = refit_A_full_poisson_squarem(
                W,
                counts,
                bundle.document_lengths,
                initial_A=warm_start,
                max_evaluations=poisson_max_iter + 2,
                tolerance=poisson_tolerance,
                statistics_backend=statistics_backend,
            )
    result.diagnostics["warm_start"] = (
        "paired_A_current" if warm_start is not None else "pooled_feature_frequencies"
    )
    if warm_start_warning is not None:
        result.warnings.append(warm_start_warning)
    return result, perf_counter() - started


def recover_A_for_geometry(
    bundle: RealDataBundle,
    geometry_fit: GeometryFit,
    method: str,
    **settings: Any,
) -> tuple[ARecoveryResult, float]:
    """:func:`recover_A` applied to the W of one vertex-hunting geometry."""

    return recover_A(bundle, geometry_fit.W_hat, method, **settings)


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
