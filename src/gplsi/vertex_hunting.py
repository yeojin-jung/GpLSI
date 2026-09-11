"""Common vertex-hunting interface for GpLSI experiments.

References
----------
* MixedSCORE ``mixedSCORE.R`` at commit
  a2112c355c191a04f41d00ba7730f7e40e8111be (SVS).
* VertexHunting at commit e3826fca5ee08916e36ae936d6fbe791a2126c18
  (official pp-SPA research implementation).

The snapshots contain no explicit license file.  This module therefore uses
independently written, source-parity implementations and records the pinned
source revisions rather than copying either file wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from math import comb, factorial
from time import perf_counter
from typing import Any, MutableMapping

import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from .accelerated_palm_aa import palm_aa_vertex_hunt


MIXEDSCORE_COMMIT = "a2112c355c191a04f41d00ba7730f7e40e8111be"
PPSPA_COMMIT = "e3826fca5ee08916e36ae936d6fbe791a2126c18"
PALM_NMF_SOURCE_SHA256 = "f895da417e58af010312a2c8920602cd28f292bdb335fd1e170c7e2d9dc4793b"
PALM_ACCELERATED_SOURCE_SHA256 = (
    "2a067439426a3dcc661811fcad2bdd1a0e486d15ac8ae9402b13667b8bf393e8"
)
METHODS = (
    "spa_current",
    "svs",
    "svs_star",
    "pp_spa",
    "palm",
    "palm_accelerated",
)


class VertexHuntingError(RuntimeError):
    """Raised when a requested vertex-hunting fit cannot be computed."""


@dataclass
class VertexHuntResult:
    method: str
    vertices: np.ndarray
    embedding_used: np.ndarray | None = None
    selected_observation_indices: np.ndarray | None = None
    selected_center_indices: np.ndarray | None = None
    centers: np.ndarray | None = None
    pseudo_points: np.ndarray | None = None
    projected_points: np.ndarray | None = None
    cluster_assignments: np.ndarray | None = None
    neighbor_indices: list[np.ndarray] | None = None
    neighborhood_sizes: np.ndarray | None = None
    retained_point_indices: np.ndarray | None = None
    discarded_point_indices: np.ndarray | None = None
    initialization_observation_indices: np.ndarray | None = None
    observation_weights: np.ndarray | None = None
    archetype_to_data_weights: np.ndarray | None = None
    archetype_projection_points: np.ndarray | None = None
    initialization_vertices: np.ndarray | None = None
    initialization_weights: np.ndarray | None = None
    objective_trace: np.ndarray | None = None
    reconstruction_trace: np.ndarray | None = None
    penalty_trace: np.ndarray | None = None
    relative_step_trace: np.ndarray | None = None
    gamma_h_trace: np.ndarray | None = None
    gamma_w_trace: np.ndarray | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    condition_number: float = np.nan
    smallest_singular_value: float = np.nan
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    failure_flags: list[str] = field(default_factory=list)
    status: str = "ok"
    failure_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "ok"


def _vertex_diagnostics(vertices: np.ndarray) -> tuple[float, float]:
    if vertices.size == 0:
        return np.inf, 0.0
    singular_values = np.linalg.svd(vertices, compute_uv=False)
    if singular_values.size == 0:
        return np.inf, 0.0
    smallest = float(singular_values[-1])
    condition = np.inf if smallest == 0 else float(singular_values[0] / smallest)
    return condition, smallest


def _spa_current(
    embedding: np.ndarray,
    K: int,
    *,
    precondition: bool = False,
    mutate_signs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Behavioral port of ``GpLSI.preconditioned_spa``.

    The current estimator has embedding dimension K.  Keeping that restriction
    explicit prevents this baseline from being silently replaced by affine SPA.
    """

    U = embedding if mutate_signs else embedding.copy()
    if U.ndim != 2 or U.shape[1] != K:
        raise VertexHuntingError(
            f"spa_current expects an n x K embedding; got {U.shape} for K={K}"
        )
    for k in range(K):
        if U[0, k] < 0:
            U[:, k] *= -1
    M = U.T
    if precondition:
        import cvxpy as cp

        Q = cp.Variable((K, K), symmetric=True)
        problem = cp.Problem(
            cp.Maximize(cp.log_det(Q)), [cp.norm(Q @ M, axis=0) <= 1]
        )
        problem.solve(solver=cp.SCS, verbose=False)
        if Q.value is None:
            raise VertexHuntingError("spa_current preconditioner did not converge")
        S = Q.value @ M
    else:
        S = M.copy()

    selected: list[int] = []
    for _ in range(K):
        maxind = int(np.argmax(norm(S, axis=0)))
        s = S[:, maxind].reshape(K, 1)
        denom = float(norm(s) ** 2)
        if denom <= np.finfo(float).eps:
            raise VertexHuntingError("spa_current encountered a zero residual")
        S = (np.eye(K) - (s @ s.T) / denom) @ S
        selected.append(maxind)
    indices = np.asarray(selected, dtype=int)
    return U[indices], indices, U


def _affine_spa(points: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Official VertexHunting ``SuccessiveProj`` convention, with indices."""

    R = np.asarray(points, dtype=float)
    if R.ndim != 2:
        raise VertexHuntingError("affine SPA expects a two-dimensional point cloud")
    if R.shape[0] < K:
        raise VertexHuntingError(f"affine SPA needs at least K={K} points")
    Y = np.column_stack((np.ones(R.shape[0]), R))
    selected: list[int] = []
    for _ in range(K):
        index = int(np.argmax(np.sum(Y**2, axis=1)))
        residual = Y[index]
        size = float(np.linalg.norm(residual))
        if size <= np.finfo(float).eps:
            raise VertexHuntingError("affine SPA encountered a zero residual")
        selected.append(index)
        direction = residual / size
        Y -= np.outer(Y @ direction, direction)
    indices = np.asarray(selected, dtype=int)
    return R[indices], indices


def _fit_kmeans(
    points: np.ndarray,
    L: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if L < 1 or L > points.shape[0]:
        raise VertexHuntingError(f"invalid number of centers L={L} for n={points.shape[0]}")
    unique_count = np.unique(points, axis=0).shape[0]
    if unique_count < L:
        raise VertexHuntingError(
            f"cannot form L={L} distinct centers from {unique_count} unique observations"
        )
    # Canonical row order makes fixed-seed output invariant to observation
    # permutations (sklearn's k-means++ otherwise samples row positions).
    sort_keys = tuple(points[:, column] for column in reversed(range(points.shape[1])))
    canonical_order = np.lexsort(sort_keys)
    canonical_points = points[canonical_order]
    fit = KMeans(
        n_clusters=L,
        max_iter=100,
        n_init=100,
        random_state=random_state,
        algorithm="lloyd",
    ).fit(canonical_points)
    labels = np.empty(points.shape[0], dtype=int)
    labels[canonical_order] = fit.labels_.astype(int)
    warnings: list[str] = []
    if np.unique(labels).size != L:
        raise VertexHuntingError("k-means returned one or more empty clusters")
    if np.unique(fit.cluster_centers_, axis=0).shape[0] != L:
        warnings.append("duplicate_kmeans_centers")
    return fit.cluster_centers_, labels, warnings


def _centers_for_L(
    points: np.ndarray,
    L: int,
    random_state: int | None,
    cache: MutableMapping[int, tuple[np.ndarray, np.ndarray, list[str]]],
    *,
    bypass_kmeans: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if L in cache:
        centers, labels, warnings = cache[L]
        return centers.copy(), labels.copy(), list(warnings)
    if bypass_kmeans:
        if L != points.shape[0]:
            raise VertexHuntingError("bypass_kmeans requires L=n")
        result = (points.copy(), np.arange(points.shape[0], dtype=int), [])
    else:
        result = _fit_kmeans(points, L, random_state)
    cache[L] = (result[0].copy(), result[1].copy(), list(result[2]))
    return result


def _distance_to_convex_hull(point: np.ndarray, vertices: np.ndarray) -> float:
    # Active-set enumeration is exact for this small-K problem and is more
    # reliable near a simplex face than a generic inequality optimizer.  Every
    # optimum lies in the relative interior of some nonempty vertex subset.
    K = vertices.shape[0]
    best = np.inf
    for subset_size in range(1, K + 1):
        for subset in combinations(range(K), subset_size):
            face = vertices[np.asarray(subset)]
            gram = face @ face.T
            kkt = np.block(
                [
                    [gram, np.ones((subset_size, 1))],
                    [np.ones((1, subset_size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.concatenate((face @ point, [1.0]))
            solution, _, _, _ = np.linalg.lstsq(kkt, rhs, rcond=None)
            weights = solution[:subset_size]
            if np.all(weights >= -1e-10):
                weights = np.maximum(weights, 0.0)
                weights /= weights.sum()
                best = min(best, float(np.linalg.norm(weights @ face - point)))
    if not np.isfinite(best):
        raise VertexHuntingError("convex-hull active-set projection failed")
    return best


def exhaustive_vertex_search(
    centers: np.ndarray,
    K: int,
    *,
    max_simplexes: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Faithful exhaustive MixedSCORE simplex search on a fixed center matrix."""

    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    L = centers.shape[0]
    if not (K < L):
        raise VertexHuntingError(f"SVS requires L>K; got L={L}, K={K}")
    candidate_count = comb(L, K)
    if candidate_count > max_simplexes:
        raise VertexHuntingError(
            "exhaustive SVS is computationally infeasible: "
            f"C({L},{K})={candidate_count:,} exceeds max_simplexes={max_simplexes:,}"
        )

    best_indices: np.ndarray | None = None
    best_distance = np.inf
    all_indices = np.arange(L)
    evaluated = 0
    for candidate in combinations(range(L), K):
        candidate_indices = np.asarray(candidate, dtype=int)
        vertices = centers[candidate_indices]
        mask = np.ones(L, dtype=bool)
        mask[candidate_indices] = False
        distances = [
            _distance_to_convex_hull(point, vertices) for point in centers[mask]
        ]
        distance = max(distances) if distances else 0.0
        evaluated += 1
        # Strict comparison preserves R's which.min(...)[1] tie behavior.
        if distance < best_distance:
            best_distance = distance
            best_indices = candidate_indices
    if best_indices is None:
        raise VertexHuntingError("SVS did not evaluate a candidate simplex")
    return centers[best_indices], best_indices, float(best_distance), evaluated


def _permutation_stability(current: np.ndarray, previous: np.ndarray) -> float:
    K = current.shape[0]
    if factorial(K) > 1_000_000:
        raise VertexHuntingError(
            f"faithful MixedSCORE stability search requires {K}! permutations"
        )
    best = np.inf
    for order in permutations(range(K)):
        # Preserve the R source literally: ``rowSums(V.L - V.Lminus1)^2``
        # squares each row sum; it is not a sum of squared coordinates.
        discrepancy = np.max(
            np.sum(current[np.asarray(order)] - previous, axis=1) ** 2
        )
        best = min(best, float(discrepancy))
    return best


def _run_svs(
    points: np.ndarray,
    K: int,
    *,
    L_mode: str,
    L: int | None,
    random_state: int | None,
    center_cache: MutableMapping[int, tuple[np.ndarray, np.ndarray, list[str]]],
    max_simplexes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    if L_mode == "fixed":
        if L is None:
            raise VertexHuntingError("fixed SVS requires L")
        centers, labels, warnings = _centers_for_L(points, L, random_state, center_cache)
        vertices, selected, objective, evaluated = exhaustive_vertex_search(
            centers, K, max_simplexes=max_simplexes
        )
        details = {
            "L": int(L),
            "L_mode": L_mode,
            "simplex_fitting_objective": objective,
            "candidate_simplexes_evaluated": evaluated,
            "source_commit": MIXEDSCORE_COMMIT,
        }
        return vertices, selected, centers, labels, details, warnings

    if L_mode != "mixedscore_adaptive":
        raise VertexHuntingError(f"unknown SVS L_mode={L_mode!r}")
    if points.shape[0] < K + 1:
        raise VertexHuntingError("adaptive SVS needs at least K+1 observations")

    centers_k, _, warnings = _centers_for_L(points, K, random_state, center_cache)
    previous_vertices = centers_k
    fits: list[dict[str, Any]] = []
    total_evaluated = 0
    for candidate_L in range(K + 1, min(3 * K, points.shape[0]) + 1):
        centers, labels, center_warnings = _centers_for_L(
            points, candidate_L, random_state, center_cache
        )
        warnings.extend(center_warnings)
        vertices, selected, objective, evaluated = exhaustive_vertex_search(
            centers, K, max_simplexes=max_simplexes
        )
        total_evaluated += evaluated
        delta = _permutation_stability(vertices, previous_vertices) / (1.0 + objective)
        fits.append(
            {
                "L": candidate_L,
                "vertices": vertices,
                "selected": selected,
                "centers": centers,
                "labels": labels,
                "objective": objective,
                "evaluated": evaluated,
                "delta": delta,
            }
        )
        previous_vertices = vertices
    chosen = min(fits, key=lambda item: item["delta"])
    details = {
        "L": int(chosen["L"]),
        "L_mode": L_mode,
        "simplex_fitting_objective": float(chosen["objective"]),
        "stability_score": float(chosen["delta"]),
        "candidate_simplexes_evaluated": int(total_evaluated),
        "candidate_L_details": [
            {
                "L": int(item["L"]),
                "simplex_fitting_objective": float(item["objective"]),
                "stability_score": float(item["delta"]),
                "candidate_simplexes_evaluated": int(item["evaluated"]),
            }
            for item in fits
        ],
        "source_commit": MIXEDSCORE_COMMIT,
    }
    return (
        chosen["vertices"],
        chosen["selected"],
        chosen["centers"],
        chosen["labels"],
        details,
        warnings,
    )


def _run_svs_star(
    points: np.ndarray,
    K: int,
    *,
    L_mode: str,
    L: int | None,
    random_state: int | None,
    center_cache: MutableMapping[int, tuple[np.ndarray, np.ndarray, list[str]]],
    max_simplexes: int,
    bypass_kmeans: bool,
    preselected_svs_details: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    adaptive_details: dict[str, Any] = {}
    if preselected_svs_details is not None:
        if L_mode != "mixedscore_adaptive":
            raise VertexHuntingError(
                "preselected SVS details require L_mode='mixedscore_adaptive'"
            )
        adaptive_details = dict(preselected_svs_details)
        if "L" not in adaptive_details:
            raise VertexHuntingError("preselected SVS details do not contain L")
        warnings = []
        selected_L = int(adaptive_details["L"])
        effective_mode = "svs_star_L_reused_from_same_task_svs"
    elif L_mode == "mixedscore_adaptive":
        _, _, _, _, adaptive_details, warnings = _run_svs(
            points,
            K,
            L_mode=L_mode,
            L=None,
            random_state=random_state,
            center_cache=center_cache,
            max_simplexes=max_simplexes,
        )
        selected_L = int(adaptive_details["L"])
        effective_mode = "svs_star_L_selected_by_svs"
    elif L_mode == "fixed":
        if L is None:
            raise VertexHuntingError("fixed SVS* requires L")
        selected_L = int(L)
        effective_mode = "fixed"
        warnings = []
    else:
        raise VertexHuntingError(f"unknown SVS* L_mode={L_mode!r}")

    centers, labels, center_warnings = _centers_for_L(
        points,
        selected_L,
        random_state,
        center_cache,
        bypass_kmeans=bypass_kmeans,
    )
    warnings.extend(center_warnings)
    _, selected, _ = _spa_current(centers, K, mutate_signs=False)
    # Column sign normalization in the historical SPA affects coordinates but
    # not selected row indices.  Return the original centers at those indices,
    # as required by the SVS* definition.
    vertices = centers[selected]
    details = {
        "L": selected_L,
        "L_mode": L_mode,
        "effective_L_mode": effective_mode,
        "second_stage": "spa_current_on_shared_kmeans_centers",
        "bypass_kmeans": bool(bypass_kmeans),
        "source_commit_for_L_selection": MIXEDSCORE_COMMIT,
    }
    if adaptive_details:
        details["svs_L_selection"] = adaptive_details
    return vertices, selected, centers, labels, details, warnings


def _project_affine(points: np.ndarray, K: int) -> np.ndarray:
    """Official ``get_projected_points`` convention, returned as n x d."""

    samples = points.T
    means = np.mean(samples, axis=1, keepdims=True)
    centered = samples - means
    # Only the left singular vectors define the affine projection. Economy
    # SVD preserves that subspace without allocating an unused n-by-n Vh.
    U, _, _ = np.linalg.svd(centered, full_matrices=False)
    U_s = U[:, : K - 1]
    projection = U_s @ U_s.T
    return (means + projection @ centered).T


def _pp_pseudo_points(
    projected: np.ndarray,
    *,
    epsilon: float,
    m_neighbors: int,
    min_neighbors: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(projected)
    radius_neighbors = tree.query_ball_tree(tree, r=epsilon)
    pseudo_points: list[np.ndarray] = []
    used_neighbors: list[np.ndarray] = []
    neighborhood_sizes = np.asarray([len(indices) for indices in radius_neighbors], dtype=int)
    retained: list[int] = []
    discarded: list[int] = []
    for index, point in enumerate(projected):
        candidates = np.asarray(radius_neighbors[index], dtype=int)
        if candidates.size < min_neighbors:
            discarded.append(index)
            continue
        if candidates.size <= m_neighbors:
            chosen = candidates
        else:
            _, queried = tree.query(point, k=m_neighbors)
            chosen = np.atleast_1d(queried).astype(int)
        pseudo_points.append(np.mean(projected[chosen], axis=0))
        used_neighbors.append(chosen)
        retained.append(index)
    if not pseudo_points:
        raise VertexHuntingError("pp-SPA discarded every projected observation")
    return (
        np.asarray(pseudo_points),
        used_neighbors,
        neighborhood_sizes,
        np.asarray(retained, dtype=int),
        np.asarray(discarded, dtype=int),
    )


def vertex_hunt(
    embedding: np.ndarray,
    K: int,
    method: str,
    random_state: int | None = None,
    **method_parameters: Any,
) -> VertexHuntResult:
    """Estimate K vertices from a row-oriented point cloud.

    Set ``raise_on_failure=True`` to turn a structured failure result into a
    ``VertexHuntingError``.  A mutable ``center_cache`` may be shared between
    SVS and SVS* calls so fixed-L comparisons use bitwise-identical centers.
    """

    started = perf_counter()
    points = np.asarray(embedding, dtype=float)
    raise_on_failure = bool(method_parameters.pop("raise_on_failure", False))
    condition_threshold = float(method_parameters.pop("condition_threshold", 1e12))
    try:
        if method not in METHODS:
            raise VertexHuntingError(
                f"unknown vertex hunter {method!r}; expected one of {METHODS}"
            )
        if points.ndim != 2 or points.shape[0] < K or K < 1:
            raise VertexHuntingError(
                f"invalid embedding shape {points.shape} for K={K}"
            )
        if not np.isfinite(points).all():
            raise VertexHuntingError("embedding contains non-finite values")

        result = VertexHuntResult(method=method, vertices=np.empty((0, points.shape[1])))
        if method == "spa_current":
            precondition = bool(method_parameters.pop("precondition", False))
            vertices, selected, embedding_used = _spa_current(
                points, K, precondition=precondition
            )
            result.vertices = vertices
            result.embedding_used = embedding_used
            result.selected_observation_indices = selected
            result.parameters = {
                "precondition": precondition,
                "implementation": "GpLSI.preconditioned_spa_behavioral_port",
                "coordinate_signs": np.where(points[0] < 0, -1.0, 1.0).tolist(),
            }

        elif method in {"svs", "svs_star"}:
            L_mode = str(method_parameters.pop("L_mode", "mixedscore_adaptive"))
            L_value = method_parameters.pop("L", None)
            L_value = None if L_value is None else int(L_value)
            max_simplexes = int(method_parameters.pop("max_simplexes", 100_000))
            center_cache = method_parameters.pop("center_cache", None)
            if center_cache is None:
                center_cache = {}
            if method == "svs":
                vertices, selected, centers, labels, details, warnings = _run_svs(
                    points,
                    K,
                    L_mode=L_mode,
                    L=L_value,
                    random_state=random_state,
                    center_cache=center_cache,
                    max_simplexes=max_simplexes,
                )
            else:
                bypass = bool(method_parameters.pop("bypass_kmeans", False))
                preselected_svs_details = method_parameters.pop(
                    "preselected_svs_details", None
                )
                vertices, selected, centers, labels, details, warnings = _run_svs_star(
                    points,
                    K,
                    L_mode=L_mode,
                    L=L_value,
                    random_state=random_state,
                    center_cache=center_cache,
                    max_simplexes=max_simplexes,
                    bypass_kmeans=bypass,
                    preselected_svs_details=preselected_svs_details,
                )
                if bypass:
                    result.selected_observation_indices = selected
            result.vertices = vertices
            result.embedding_used = points
            result.selected_center_indices = selected
            result.centers = centers
            result.cluster_assignments = labels
            result.parameters = details
            result.warnings.extend(warnings)

        elif method == "pp_spa":
            radius_divisor = float(method_parameters.pop("radius_divisor", 20.0))
            m_neighbors = int(method_parameters.pop("m_neighbors", 4))
            min_neighbors = int(method_parameters.pop("min_neighbors", 3))
            if radius_divisor <= 0 or m_neighbors < 1 or min_neighbors < 1:
                raise VertexHuntingError("invalid pp-SPA neighborhood parameters")
            projected = _project_affine(points, K)
            max_distance = float(np.max(pdist(projected))) if len(projected) > 1 else 0.0
            epsilon = max_distance / radius_divisor
            pseudo, neighbors, sizes, retained, discarded = _pp_pseudo_points(
                projected,
                epsilon=epsilon,
                m_neighbors=m_neighbors,
                min_neighbors=min_neighbors,
            )
            vertices, selected_pseudo = _affine_spa(pseudo, K)
            result.vertices = vertices
            result.embedding_used = points
            result.selected_center_indices = selected_pseudo
            result.selected_observation_indices = retained[selected_pseudo]
            result.projected_points = projected
            result.pseudo_points = pseudo
            result.neighbor_indices = neighbors
            result.neighborhood_sizes = sizes
            result.retained_point_indices = retained
            result.discarded_point_indices = discarded
            result.parameters = {
                "radius_divisor": radius_divisor,
                "epsilon": epsilon,
                "m_neighbors": m_neighbors,
                "official_parameter_N": m_neighbors,
                "min_neighbors": min_neighbors,
                "official_parameter_t": min_neighbors,
                "affine_dimension": K - 1,
                "source_commit": PPSPA_COMMIT,
            }

        else:
            acceleration = (
                "none" if method == "palm" else "monotone_restart"
            )
            configured_acceleration = str(
                method_parameters.pop("acceleration", acceleration)
            )
            if configured_acceleration != acceleration:
                raise VertexHuntingError(
                    f"{method} fixes acceleration={acceleration!r}; "
                    f"got {configured_acceleration!r}"
                )
            initialization_precondition = bool(
                method_parameters.pop("initialization_precondition", False)
            )
            _, initialization_indices, _ = _spa_current(
                points,
                K,
                precondition=initialization_precondition,
                mutate_signs=False,
            )
            # SPA's sign normalization does not change selected row indices.
            # PALM is optimized in the original point-cloud coordinates, so
            # initialize from the original rows rather than sign-flipped rows.
            initial_vertices = points[initialization_indices].copy()
            palm_parameters = dict(method_parameters)
            method_parameters.clear()
            palm_result = palm_aa_vertex_hunt(
                points,
                K,
                random_state=random_state,
                H_init=initial_vertices,
                acceleration=acceleration,
                **palm_parameters,
            )
            if palm_result["failure"]:
                raise VertexHuntingError(
                    f"{method} returned non-finite vertices "
                    f"(optimizer status={palm_result['status']})"
                )
            result.vertices = np.asarray(palm_result["vertices"], dtype=float)
            result.embedding_used = points
            result.initialization_observation_indices = initialization_indices
            result.observation_weights = np.asarray(
                palm_result["observation_weights"], dtype=float
            )
            result.archetype_to_data_weights = (
                None
                if palm_result["archetype_to_data_weights"] is None
                else np.asarray(palm_result["archetype_to_data_weights"], dtype=float)
            )
            result.archetype_projection_points = (
                None
                if palm_result["archetype_projection_points"] is None
                else np.asarray(palm_result["archetype_projection_points"], dtype=float)
            )
            result.initialization_vertices = np.asarray(
                palm_result["initialization_vertices"], dtype=float
            )
            result.initialization_weights = np.asarray(
                palm_result["initialization_weights"], dtype=float
            )
            fit_result = palm_result["fit_result"]
            result.objective_trace = np.asarray(palm_result["objective_trace"], dtype=float)
            result.reconstruction_trace = np.asarray(
                palm_result["reconstruction_trace"], dtype=float
            )
            result.penalty_trace = np.asarray(palm_result["penalty_trace"], dtype=float)
            result.relative_step_trace = np.asarray(
                palm_result["relative_step_trace"], dtype=float
            )
            result.gamma_h_trace = np.asarray(fit_result.gamma_h_trace, dtype=float)
            result.gamma_w_trace = np.asarray(fit_result.gamma_w_trace, dtype=float)
            result.warnings.extend(palm_result["warnings"])
            if palm_result["status"] != "converged_relative_step":
                result.warnings.append(f"palm_optimizer_status={palm_result['status']}")
            result.parameters = {
                **palm_result["parameters"],
                "implementation": "corrected_palm_archetypal_analysis",
                "method_definition": (
                    "plain_PALM_from_NMF.py"
                    if method == "palm"
                    else "accelerated_PALM_monotone_restart"
                ),
                "initialization_method": "spa_current_selected_original_rows",
                "initialization_precondition": initialization_precondition,
                "optimizer_status": palm_result["status"],
                "optimizer_converged": bool(fit_result.converged),
                "optimizer_iterations": int(palm_result["iteration_count"]),
                "initial_objective": float(palm_result["initial_objective"]),
                "final_objective": float(palm_result["final_objective"]),
                "stationarity": float(palm_result["stationarity"]),
                "accepted_accelerated_steps": int(
                    palm_result["accepted_accelerated_steps"]
                ),
                "rejected_accelerated_steps": int(
                    fit_result.rejected_accelerated_steps
                ),
                "restart_count": int(palm_result["restart_count"]),
                "backtracking_count": int(fit_result.backtracking_count),
                "hull_projection_calls": int(fit_result.hull_projection_calls),
                "hull_projection_iterations": int(
                    fit_result.hull_projection_iterations
                ),
                "hull_projection_failures": int(fit_result.hull_projection_failures),
                "hull_reduction_status": fit_result.hull_reduction,
                "hull_input_size": int(fit_result.hull_input_size),
                "internal_runtime_seconds": float(palm_result["runtime"]),
                "PALM_NMF_source_sha256": PALM_NMF_SOURCE_SHA256,
                "PALM_accelerated_source_sha256": PALM_ACCELERATED_SOURCE_SHA256,
                "source_attribution": (
                    "author-supplied Non-negative Matrix Factorization via "
                    "Archetypal Analysis implementation"
                ),
            }

        if method_parameters:
            unknown = ", ".join(sorted(method_parameters))
            raise VertexHuntingError(f"unused {method} parameters: {unknown}")
        result.condition_number, result.smallest_singular_value = _vertex_diagnostics(
            result.vertices
        )
        if not np.isfinite(result.condition_number) or result.condition_number > condition_threshold:
            result.failure_flags.append("ill_conditioned_vertex_matrix")
            result.warnings.append(
                f"vertex condition number {result.condition_number:.6g} exceeds "
                f"threshold {condition_threshold:.6g}"
            )
        result.parameters["random_state"] = random_state
        result.parameters["condition_threshold"] = condition_threshold
        result.runtime_seconds = perf_counter() - started
        return result
    except Exception as error:  # structured failure is part of the experiment record
        if raise_on_failure:
            if isinstance(error, VertexHuntingError):
                raise
            raise VertexHuntingError(str(error)) from error
        return VertexHuntResult(
            method=method,
            vertices=np.empty((0, points.shape[1] if points.ndim == 2 else 0)),
            runtime_seconds=perf_counter() - started,
            status="failed",
            failure_reason=f"{type(error).__name__}: {error}",
            failure_flags=["vertex_hunting_failed"],
            parameters={"random_state": random_state},
        )
