"""Corrected and accelerated PALM archetypal analysis.

This module implements the objective used in the authors' supplied code for
"Non-negative Matrix Factorization via Archetypal Analysis":

    F(W, H) = 1/2 ||X - W H||_F^2
              + lambda_/2 * sum_k dist(H[k], conv(X))^2,

subject to every row of W belonging to the probability simplex.

Data orientation
----------------
X : (n_samples, embedding_dimension)
W : (n_samples, n_archetypes), row stochastic
H : (n_archetypes, embedding_dimension), rows are archetypes/vertices

The implementation keeps the authors' accelerated extrapolation formula, but
corrects the block gradients and matrix orientations and adds a monotone
restart safeguard. It also removes several large avoidable costs:

* batched Euclidean projection of all W rows onto the simplex;
* direct point-to-convex-hull projection, without materializing X - query;
* warm starts for the convex-hull active sets;
* candidate-specific spectral Lipschitz constants;
* cached evaluation of the archetype penalty after the proximal H update;
* no post-fit lambda-selection projections when lambda_ is supplied.

The default ``acceleration='monotone_restart'`` tries one extrapolated PALM
candidate and computes a plain PALM candidate only when the extrapolated step
fails the monotonicity safeguard. ``acceleration='source_best'`` always
constructs both candidates and selects the lower-objective one, matching the
structure of the authors' accelerated routine more closely.

Attribution
-----------
Adapted from the author-supplied Python implementation accompanying
"Non-negative Matrix Factorization via Archetypal Analysis". Preserve the
original attribution and verify the authors' software license before public
redistribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - SciPy is expected in the project env.
    ConvexHull = None  # type: ignore[assignment]
    QhullError = Exception  # type: ignore[assignment,misc]


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AccelerationMode = Literal["none", "monotone_restart", "source_best"]
HullReductionMode = Literal["none", "auto", "exact"]


@dataclass
class HullProjectionState:
    """Warm-start state for one point-to-convex-hull projection."""

    active_indices: IntArray
    weights: FloatArray

    def copy(self) -> "HullProjectionState":
        return HullProjectionState(
            active_indices=self.active_indices.copy(),
            weights=self.weights.copy(),
        )


@dataclass
class HullProjectionResult:
    """Result of projecting one query onto a data convex hull."""

    point: FloatArray
    residual: FloatArray
    squared_distance: float
    state: HullProjectionState
    iterations: int
    dual_gap: float
    converged: bool


@dataclass
class PALMCandidate:
    """One complete H-then-W PALM candidate."""

    W: FloatArray
    H: FloatArray
    objective: float
    reconstruction_loss: float
    archetype_penalty: float
    projection_states: List[HullProjectionState]
    gamma_h: float
    gamma_w: float
    projection_iterations: int
    projection_failures: int


@dataclass
class PALMResult:
    """Fit returned by :func:`accelerated_palm_aa`."""

    W: FloatArray
    H: FloatArray
    W_initial: FloatArray
    H_initial: FloatArray
    archetype_projection_weights: Optional[FloatArray]
    archetype_projection_points: Optional[FloatArray]
    converged: bool
    status: str
    n_iter: int
    lambda_: float
    acceleration: str
    objective_trace: FloatArray
    reconstruction_trace: FloatArray
    penalty_trace: FloatArray
    relative_step_trace: FloatArray
    stationarity: float
    gamma_h_trace: FloatArray
    gamma_w_trace: FloatArray
    accepted_accelerated_steps: int
    rejected_accelerated_steps: int
    restart_count: int
    backtracking_count: int
    hull_projection_calls: int
    hull_projection_iterations: int
    hull_projection_failures: int
    runtime_seconds: float
    vertex_condition_number: float
    vertex_smallest_singular_value: float
    initialization: str
    hull_reduction: str
    hull_input_size: int
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def vertices(self) -> FloatArray:
        return self.H

    @property
    def weights(self) -> FloatArray:
        return self.W

    def to_dict(self, include_arrays: bool = True) -> Dict[str, Any]:
        if include_arrays:
            return asdict(self)
        array_fields = {
            "W",
            "H",
            "W_initial",
            "H_initial",
            "archetype_projection_weights",
            "archetype_projection_points",
            "objective_trace",
            "reconstruction_trace",
            "penalty_trace",
            "relative_step_trace",
            "gamma_h_trace",
            "gamma_w_trace",
        }
        return {
            item.name: getattr(self, item.name)
            for item in dataclass_fields(self)
            if item.name not in array_fields
        }


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------


def project_simplex_rows(values: ArrayLike) -> FloatArray:
    """Project one vector or every row of a matrix onto the probability simplex.

    Uses the exact sorting-based Euclidean projection. Unlike clipping followed
    by normalization, this is the true Euclidean proximal map.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError("values must be a one- or two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains non-finite entries")

    one_dimensional = array.ndim == 1
    matrix = array[None, :] if one_dimensional else array
    if matrix.shape[1] == 0:
        raise ValueError("cannot project an empty vector onto the simplex")

    ordered = np.sort(matrix, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    denominators = np.arange(1, matrix.shape[1] + 1, dtype=float)
    positive = ordered - cumulative / denominators[None, :] > 0.0
    rho = positive.sum(axis=1) - 1
    theta = cumulative[np.arange(matrix.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(matrix - theta[:, None], 0.0)

    # Remove tiny accumulated floating-point error without changing support.
    row_sums = projected.sum(axis=1, keepdims=True)
    projected /= row_sums
    return projected[0] if one_dimensional else projected


def _largest_eigenvalue_symmetric(matrix: FloatArray, floor: float) -> float:
    """Largest eigenvalue of a small symmetric PSD matrix."""

    sym = 0.5 * (matrix + matrix.T)
    value = float(np.linalg.eigvalsh(sym)[-1]) if sym.size else 0.0
    return max(value, floor)


def _squared_frobenius(array: FloatArray) -> float:
    return float(np.sum(array * array))


def _affine_minimizer(points: FloatArray, query: FloatArray) -> FloatArray:
    """Minimum-distance affine combination of ``points`` with coefficients summing to 1.

    Nonnegativity is deliberately omitted here. The outer active-set routine
    moves toward this affine minimizer until feasibility is reached.
    """

    m = points.shape[0]
    if m == 1:
        return np.ones(1, dtype=float)

    gram = points @ points.T
    rhs = points @ query
    kkt = np.empty((m + 1, m + 1), dtype=float)
    kkt[:m, :m] = gram
    kkt[:m, m] = 1.0
    kkt[m, :m] = 1.0
    kkt[m, m] = 0.0
    target = np.concatenate([rhs, np.ones(1, dtype=float)])

    # lstsq is robust to affinely dependent active points.
    solution, *_ = np.linalg.lstsq(kkt, target, rcond=None)
    beta = solution[:m]
    correction = (1.0 - float(beta.sum())) / m
    return beta + correction


def _sanitize_projection_state(
    state: Optional[HullProjectionState],
    n_vertices: int,
    zero_tolerance: float,
) -> Optional[HullProjectionState]:
    if state is None:
        return None

    indices = np.asarray(state.active_indices, dtype=np.int64).reshape(-1)
    weights = np.asarray(state.weights, dtype=float).reshape(-1)
    if len(indices) != len(weights) or len(indices) == 0:
        return None
    valid = (indices >= 0) & (indices < n_vertices) & np.isfinite(weights)
    indices = indices[valid]
    weights = weights[valid]
    if len(indices) == 0:
        return None

    # Merge duplicate indices, which can appear after a user-provided warm start.
    unique, inverse = np.unique(indices, return_inverse=True)
    merged = np.zeros(len(unique), dtype=float)
    np.add.at(merged, inverse, np.maximum(weights, 0.0))
    keep = merged > zero_tolerance
    unique = unique[keep]
    merged = merged[keep]
    if len(unique) == 0 or not np.isfinite(merged.sum()) or merged.sum() <= 0.0:
        return None
    merged /= merged.sum()
    return HullProjectionState(unique.astype(np.int64), merged)


class ConvexHullProjector:
    """Fully-corrective active-set projector onto the convex hull of data rows.

    The method is a numerically straightforward Wolfe-style minimum-norm point
    algorithm. It scans all hull input points for the most violated vertex and
    solves the small affine projection problem on the active face. Warm starts
    reuse the previous active face for each archetype.
    """

    def __init__(
        self,
        points: ArrayLike,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 10_000,
        zero_tolerance: float = 1e-12,
    ) -> None:
        data = np.asarray(points, dtype=float)
        if data.ndim != 2:
            raise ValueError("points must be a two-dimensional array")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("points must be non-empty")
        if not np.all(np.isfinite(data)):
            raise ValueError("points contains non-finite entries")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.points = np.ascontiguousarray(data)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.zero_tolerance = float(zero_tolerance)
        self.max_point_norm = float(
            max(1.0, np.sqrt(np.max(np.sum(self.points * self.points, axis=1))))
        )
        self.calls = 0
        self.total_iterations = 0
        self.failures = 0

    def project(
        self,
        query: ArrayLike,
        state: Optional[HullProjectionState] = None,
    ) -> HullProjectionResult:
        q = np.asarray(query, dtype=float).reshape(-1)
        if q.shape[0] != self.points.shape[1]:
            raise ValueError(
                f"query has dimension {q.shape[0]}, expected {self.points.shape[1]}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("query contains non-finite entries")

        self.calls += 1
        warm = _sanitize_projection_state(
            state, self.points.shape[0], self.zero_tolerance
        )
        if warm is None:
            distances = np.sum((self.points - q[None, :]) ** 2, axis=1)
            active = np.array([int(np.argmin(distances))], dtype=np.int64)
            weights = np.ones(1, dtype=float)
        else:
            active = warm.active_indices.copy()
            weights = warm.weights.copy()

        converged = False
        dual_gap = np.inf
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            point = weights @ self.points[active]
            gradient_point = point - q
            linear_scores = self.points @ gradient_point
            new_index = int(np.argmin(linear_scores))
            current_score = float(point @ gradient_point)
            dual_gap = current_score - float(linear_scores[new_index])
            scale = max(
                1.0,
                abs(current_score),
                float(np.linalg.norm(gradient_point)) * self.max_point_norm,
            )
            if dual_gap <= self.tolerance * scale:
                converged = True
                break

            if not np.any(active == new_index):
                active = np.append(active, np.int64(new_index))
                weights = np.append(weights, 0.0)

            # Minor cycles: reach a feasible affine minimizer on the active face.
            while True:
                beta = _affine_minimizer(self.points[active], q)
                if np.all(beta >= -self.zero_tolerance):
                    beta = np.maximum(beta, 0.0)
                    total = float(beta.sum())
                    if not np.isfinite(total) or total <= 0.0:
                        # Extremely degenerate affine solve. Retain the safest
                        # feasible active point rather than emitting NaNs.
                        nearest_local = int(
                            np.argmin(
                                np.sum((self.points[active] - q[None, :]) ** 2, axis=1)
                            )
                        )
                        active = active[[nearest_local]]
                        weights = np.ones(1, dtype=float)
                    else:
                        weights = beta / total
                    break

                direction = beta - weights
                negative_target = beta < -self.zero_tolerance
                denominators = weights[negative_target] - beta[negative_target]
                admissible = denominators > 0.0
                if not np.any(admissible):
                    # Defensive fallback for a rank-degenerate active face.
                    drop = int(np.argmin(beta))
                    active = np.delete(active, drop)
                    weights = np.delete(weights, drop)
                    if len(active) == 0:
                        distances = np.sum((self.points - q[None, :]) ** 2, axis=1)
                        active = np.array([int(np.argmin(distances))], dtype=np.int64)
                        weights = np.ones(1, dtype=float)
                        break
                    weights = np.maximum(weights, 0.0)
                    weights /= weights.sum()
                    continue

                candidate_theta = (
                    weights[negative_target][admissible] / denominators[admissible]
                )
                theta = float(np.clip(np.min(candidate_theta), 0.0, 1.0))
                weights = weights + theta * direction
                weights[np.abs(weights) <= self.zero_tolerance] = 0.0
                keep = weights > self.zero_tolerance
                if not np.any(keep):
                    keep[int(np.argmax(weights))] = True
                active = active[keep]
                weights = weights[keep]
                weights = np.maximum(weights, 0.0)
                weights /= weights.sum()

        point = weights @ self.points[active]
        residual = point - q
        squared_distance = _squared_frobenius(residual)
        self.total_iterations += iteration
        if not converged:
            self.failures += 1

        return HullProjectionResult(
            point=np.asarray(point, dtype=float),
            residual=np.asarray(residual, dtype=float),
            squared_distance=squared_distance,
            state=HullProjectionState(active.copy(), weights.copy()),
            iterations=iteration,
            dual_gap=float(dual_gap),
            converged=converged,
        )


# ---------------------------------------------------------------------------
# Optional exact hull reduction
# ---------------------------------------------------------------------------


def _affine_coordinates(
    X: FloatArray,
    rank_tolerance: Optional[float] = None,
) -> Tuple[FloatArray, int]:
    centered = X - X.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 0), dtype=float), 0
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((X.shape[0], 0), dtype=float), 0
    if rank_tolerance is None:
        rank_tolerance = (
            np.finfo(float).eps
            * max(centered.shape)
            * max(float(singular_values[0]), 1.0)
        )
    rank = int(np.sum(singular_values > rank_tolerance))
    if rank == 0:
        return np.zeros((X.shape[0], 0), dtype=float), 0
    coordinates = centered @ right_vectors[:rank].T
    return coordinates, rank


def reduce_to_convex_hull_vertices(
    X: ArrayLike,
    *,
    mode: HullReductionMode = "auto",
    max_exact_dimension: int = 8,
    min_reduction_fraction: float = 0.02,
) -> Tuple[FloatArray, IntArray, str]:
    """Optionally replace X by an exact set of convex-hull vertices.

    Returns ``(reduced_X, original_indices, status)``. No perturbing Qhull
    options are used, so successful reduction preserves the exact data hull up
    to numerical affine-rank determination.
    """

    data = np.asarray(X, dtype=float)
    n = data.shape[0]
    all_indices = np.arange(n, dtype=np.int64)
    if mode == "none" or n <= 2:
        return data, all_indices, "none"
    if ConvexHull is None:
        if mode == "exact":
            raise RuntimeError("SciPy ConvexHull is unavailable")
        return data, all_indices, "unavailable"

    coordinates, rank = _affine_coordinates(data)
    if rank == 0:
        return data[[0]], np.array([0], dtype=np.int64), "exact_rank0"
    if rank == 1:
        values = coordinates[:, 0]
        indices = np.unique([int(np.argmin(values)), int(np.argmax(values))]).astype(
            np.int64
        )
        return data[indices], indices, "exact_rank1"
    if mode == "auto" and rank > max_exact_dimension:
        return data, all_indices, f"skipped_rank_{rank}"

    try:
        hull = ConvexHull(coordinates)
    except QhullError as exc:
        if mode == "exact":
            raise RuntimeError("exact convex-hull reduction failed") from exc
        return data, all_indices, "qhull_failed"

    indices = np.unique(np.asarray(hull.vertices, dtype=np.int64))
    reduction_fraction = 1.0 - len(indices) / n
    if mode == "auto" and reduction_fraction < min_reduction_fraction:
        return data, all_indices, "skipped_no_material_reduction"
    return data[indices], indices, f"exact_rank{rank}"


# ---------------------------------------------------------------------------
# Initialization and objective
# ---------------------------------------------------------------------------


def successive_projection_init(X: ArrayLike, n_archetypes: int) -> FloatArray:
    """Vectorized version of the source successive-projection initialization."""

    data = np.asarray(X, dtype=float)
    n, d = data.shape
    if not 1 <= n_archetypes <= n:
        raise ValueError("n_archetypes must lie between 1 and n_samples")

    selected: List[int] = [int(np.argmax(np.linalg.norm(data, axis=1)))]
    if n_archetypes == 1:
        return data[selected].copy()

    distances = np.linalg.norm(data - data[selected[0]], axis=1)
    selected.append(int(np.argmax(distances)))

    while len(selected) < n_archetypes:
        base = data[selected[0]]
        differences = data[selected[1:]] - base
        if differences.size == 0:
            residual_sq = np.sum((data - base) ** 2, axis=1)
        else:
            _, singular_values, right_vectors = np.linalg.svd(
                differences, full_matrices=False
            )
            rank_tol = (
                np.finfo(float).eps
                * max(differences.shape)
                * max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
            )
            rank = int(np.sum(singular_values > rank_tol))
            centered = data - base
            if rank == 0:
                residual_sq = np.sum(centered * centered, axis=1)
            else:
                basis = right_vectors[:rank]
                projection = centered @ basis.T
                residual_sq = np.sum(centered * centered, axis=1) - np.sum(
                    projection * projection, axis=1
                )
        residual_sq[np.asarray(selected, dtype=int)] = -np.inf
        selected.append(int(np.argmax(residual_sq)))

    return data[np.asarray(selected, dtype=int)].copy()


def solve_weights_projected_gradient(
    X: ArrayLike,
    H: ArrayLike,
    *,
    W_init: Optional[ArrayLike] = None,
    max_iterations: int = 250,
    tolerance: float = 1e-9,
    accelerated: bool = True,
) -> FloatArray:
    """Solve min_W 1/2 ||X-WH||^2 with row-simplex constraints.

    This batched projected-gradient/FISTA solve is used for a good initial W
    and may also be called as a high-accuracy final weight refit.
    """

    data = np.asarray(X, dtype=float)
    archetypes = np.asarray(H, dtype=float)
    n = data.shape[0]
    k = archetypes.shape[0]
    if W_init is None:
        try:
            unconstrained = data @ np.linalg.pinv(archetypes)
            W = project_simplex_rows(unconstrained)
        except np.linalg.LinAlgError:
            W = np.full((n, k), 1.0 / k, dtype=float)
    else:
        W = project_simplex_rows(np.asarray(W_init, dtype=float))
        if W.shape != (n, k):
            raise ValueError(f"W_init must have shape {(n, k)}, got {W.shape}")

    lipschitz = _largest_eigenvalue_symmetric(
        archetypes @ archetypes.T, floor=1e-12
    )
    auxiliary = W.copy()
    t = 1.0
    previous_objective = 0.5 * _squared_frobenius(data - W @ archetypes)

    for _ in range(max_iterations):
        gradient = (auxiliary @ archetypes - data) @ archetypes.T
        candidate = project_simplex_rows(auxiliary - gradient / lipschitz)
        objective = 0.5 * _squared_frobenius(data - candidate @ archetypes)

        # Monotone FISTA restart.
        if objective > previous_objective + 1e-13 * max(1.0, previous_objective):
            auxiliary = W
            t = 1.0
            gradient = (auxiliary @ archetypes - data) @ archetypes.T
            candidate = project_simplex_rows(auxiliary - gradient / lipschitz)
            objective = 0.5 * _squared_frobenius(data - candidate @ archetypes)

        relative_change = float(
            np.linalg.norm(candidate - W) / max(1.0, np.linalg.norm(W))
        )
        if accelerated:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            auxiliary = candidate + ((t - 1.0) / t_new) * (candidate - W)
            t = t_new
        else:
            auxiliary = candidate
        W = candidate
        previous_objective = objective
        if relative_change <= tolerance:
            break

    return W


def _objective_with_projection(
    X: FloatArray,
    W: FloatArray,
    H: FloatArray,
    lambda_: float,
    projector: ConvexHullProjector,
    states: Optional[Sequence[Optional[HullProjectionState]]] = None,
) -> Tuple[float, float, float, List[HullProjectionState], int, int]:
    reconstruction = 0.5 * _squared_frobenius(X - W @ H)
    penalty_sq = 0.0
    output_states: List[HullProjectionState] = []
    projection_iterations = 0
    projection_failures = 0

    if lambda_ > 0.0:
        for index, archetype in enumerate(H):
            state = None if states is None else states[index]
            result = projector.project(archetype, state=state)
            penalty_sq += result.squared_distance
            output_states.append(result.state)
            projection_iterations += result.iterations
            projection_failures += int(not result.converged)
    else:
        output_states = [
            HullProjectionState(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float))
            for _ in range(H.shape[0])
        ]

    penalty = 0.5 * lambda_ * penalty_sq
    return (
        reconstruction + penalty,
        reconstruction,
        penalty,
        output_states,
        projection_iterations,
        projection_failures,
    )


def _make_candidate(
    X: FloatArray,
    H_base: FloatArray,
    W_base: FloatArray,
    lambda_: float,
    projector: ConvexHullProjector,
    projection_states: Sequence[Optional[HullProjectionState]],
    *,
    c_h: float,
    c_w: float,
    eps_step: float,
    step_multiplier: float = 1.0,
) -> PALMCandidate:
    """Perform one sequential H proximal step and W projected-gradient step."""

    gram_w = W_base.T @ W_base
    gamma_h = step_multiplier * c_h * _largest_eigenvalue_symmetric(
        gram_w, floor=eps_step
    )
    residual = W_base @ H_base - X
    H_gradient_point = H_base - (W_base.T @ residual) / gamma_h

    H_new = np.empty_like(H_base)
    states_new: List[HullProjectionState] = []
    projection_iterations = 0
    projection_failures = 0
    penalty_sq = 0.0

    if lambda_ == 0.0:
        H_new[:] = H_gradient_point
        states_new = [
            HullProjectionState(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float))
            for _ in range(H_base.shape[0])
        ]
    else:
        alpha = lambda_ / (lambda_ + gamma_h)
        distance_scale_sq = (gamma_h / (lambda_ + gamma_h)) ** 2
        for index, point in enumerate(H_gradient_point):
            result = projector.project(point, state=projection_states[index])
            H_new[index] = point + alpha * result.residual
            # If p=P_C(point), then p is also P_C((1-alpha)point+alpha p).
            penalty_sq += distance_scale_sq * result.squared_distance
            states_new.append(result.state)
            projection_iterations += result.iterations
            projection_failures += int(not result.converged)

    gram_h = H_new @ H_new.T
    gamma_w = step_multiplier * c_w * _largest_eigenvalue_symmetric(
        gram_h, floor=eps_step
    )
    residual_new = W_base @ H_new - X
    W_gradient_point = W_base - (residual_new @ H_new.T) / gamma_w
    W_new = project_simplex_rows(W_gradient_point)

    reconstruction = 0.5 * _squared_frobenius(X - W_new @ H_new)
    penalty = 0.5 * lambda_ * penalty_sq
    objective = reconstruction + penalty

    return PALMCandidate(
        W=W_new,
        H=H_new,
        objective=float(objective),
        reconstruction_loss=float(reconstruction),
        archetype_penalty=float(penalty),
        projection_states=states_new,
        gamma_h=float(gamma_h),
        gamma_w=float(gamma_w),
        projection_iterations=projection_iterations,
        projection_failures=projection_failures,
    )


def _source_extrapolation(
    H: FloatArray,
    H_previous: FloatArray,
    H_auxiliary: FloatArray,
    W: FloatArray,
    W_previous: FloatArray,
    W_auxiliary: FloatArray,
    t: float,
    t_previous: float,
) -> Tuple[FloatArray, FloatArray]:
    """The extrapolation formula used by the supplied accelerated routine."""

    G = (
        H
        + (t_previous / t) * (H_auxiliary - H)
        + ((t_previous - 1.0) / t) * (H - H_previous)
    )
    V = (
        W
        + (t_previous / t) * (W_auxiliary - W)
        + ((t_previous - 1.0) / t) * (W - W_previous)
    )
    return G, V


def _relative_pair_step(
    W_old: FloatArray,
    H_old: FloatArray,
    W_new: FloatArray,
    H_new: FloatArray,
) -> float:
    numerator = np.sqrt(
        _squared_frobenius(W_new - W_old) + _squared_frobenius(H_new - H_old)
    )
    denominator = max(
        1.0,
        np.sqrt(_squared_frobenius(W_old) + _squared_frobenius(H_old)),
    )
    return float(numerator / denominator)


def _vertex_conditioning(H: FloatArray) -> Tuple[float, float]:
    if H.shape[0] > H.shape[1]:
        # The K rows cannot be linearly independent when K>d, so a left-side
        # simplex solve of the type used by GpLSI is necessarily singular.
        singular_values = np.linalg.svd(H, compute_uv=False)
        smallest = float(singular_values[-1]) if singular_values.size else 0.0
        return np.inf, smallest
    singular_values = np.linalg.svd(H, compute_uv=False)
    if singular_values.size == 0:
        return np.inf, 0.0
    smallest = float(singular_values[-1])
    largest = float(singular_values[0])
    if smallest <= np.finfo(float).eps * max(H.shape) * max(largest, 1.0):
        return np.inf, smallest
    return largest / smallest, smallest


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def accelerated_palm_aa(
    X: ArrayLike,
    n_archetypes: int,
    lambda_: float,
    *,
    H_init: Optional[ArrayLike] = None,
    W_init: Optional[ArrayLike] = None,
    acceleration: AccelerationMode = "monotone_restart",
    max_iterations: int = 300,
    tolerance: float = 1e-7,
    objective_tolerance: float = 1e-12,
    c_h: float = 1.1,
    c_w: float = 1.1,
    eps_step: float = 1e-10,
    max_backtracking: int = 8,
    backtracking_factor: float = 2.0,
    projection_tolerance: float = 1e-10,
    projection_max_iterations: int = 10_000,
    hull_reduction: HullReductionMode = "none",
    hull_max_exact_dimension: int = 8,
    initialize_weights_iterations: int = 200,
    final_weight_refit_iterations: int = 0,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> PALMResult:
    """Fit the corrected accelerated PALM archetypal-analysis estimator.

    Parameters
    ----------
    X:
        Row-oriented point cloud of shape ``(n, d)``.
    n_archetypes:
        Number of rows in the estimated vertex matrix H.
    lambda_:
        Nonnegative penalty on distance of each H row to ``conv(X)``. A fixed
        value is required deliberately; outer validation or a separate lambda
        path should select it. This avoids the source routine's expensive
        post-fit lambda diagnostic when lambda is already known.
    H_init:
        Optional ``(K, d)`` initialization, normally SPA/SVS*/pp-SPA vertices.
    W_init:
        Optional ``(n, K)`` row-simplex initialization. If omitted, W is
        initialized from H by a batched constrained least-squares solve.
    acceleration:
        ``'none'`` for plain PALM; ``'monotone_restart'`` to try one inertial
        step and compute a plain step only after rejection; ``'source_best'``
        to construct both candidates each iteration and retain the better one.
    hull_reduction:
        ``'none'``, ``'auto'``, or ``'exact'``. Successful reduction uses only
        exact convex-hull vertices and therefore preserves the objective.

    Returns
    -------
    PALMResult
        Includes vertices, barycentric weights, traces, conditioning, restart
        counts, projection counts, and failure diagnostics.
    """

    del random_state  # Current implementation is deterministic; kept for API parity.
    start_time = perf_counter()
    warning_messages: List[str] = []

    data = np.asarray(X, dtype=float)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    n, d = data.shape
    if n == 0 or d == 0:
        raise ValueError("X must be non-empty")
    if not np.all(np.isfinite(data)):
        raise ValueError("X contains non-finite entries")
    if not 1 <= n_archetypes <= n:
        raise ValueError("n_archetypes must lie between 1 and n_samples")
    if lambda_ < 0.0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be finite and nonnegative")
    if acceleration not in ("none", "monotone_restart", "source_best"):
        raise ValueError(f"unsupported acceleration mode: {acceleration}")
    if c_h <= 1.0 or c_w <= 1.0:
        raise ValueError("c_h and c_w must exceed 1 for PALM majorization")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be nonnegative")
    if max_backtracking < 0:
        raise ValueError("max_backtracking must be nonnegative")
    if backtracking_factor <= 1.0:
        raise ValueError("backtracking_factor must exceed 1")

    hull_points, hull_indices, hull_status = reduce_to_convex_hull_vertices(
        data,
        mode=hull_reduction,
        max_exact_dimension=hull_max_exact_dimension,
    )
    projector = ConvexHullProjector(
        hull_points,
        tolerance=projection_tolerance,
        max_iterations=projection_max_iterations,
    )

    if H_init is None:
        H = successive_projection_init(data, n_archetypes)
        initialization = "successive_projection"
    else:
        H = np.asarray(H_init, dtype=float).copy()
        if H.shape != (n_archetypes, d):
            raise ValueError(
                f"H_init must have shape {(n_archetypes, d)}, got {H.shape}"
            )
        if not np.all(np.isfinite(H)):
            raise ValueError("H_init contains non-finite entries")
        initialization = "supplied_vertices"

    if W_init is None:
        W = solve_weights_projected_gradient(
            data,
            H,
            max_iterations=initialize_weights_iterations,
            tolerance=min(tolerance, 1e-9),
            accelerated=True,
        )
        initialization += "+barycentric_W"
    else:
        W = project_simplex_rows(np.asarray(W_init, dtype=float))
        if W.shape != (n, n_archetypes):
            raise ValueError(
                f"W_init must have shape {(n, n_archetypes)}, got {W.shape}"
            )
        initialization += "+supplied_W"

    H_initial = H.copy()
    W_initial = W.copy()

    (
        current_objective,
        current_reconstruction,
        current_penalty,
        projection_states,
        initial_projection_iterations,
        initial_projection_failures,
    ) = _objective_with_projection(data, W, H, lambda_, projector)

    objective_trace = [current_objective]
    reconstruction_trace = [current_reconstruction]
    penalty_trace = [current_penalty]
    relative_step_trace: List[float] = []
    gamma_h_trace: List[float] = []
    gamma_w_trace: List[float] = []

    H_previous = H.copy()
    W_previous = W.copy()
    H_auxiliary = H.copy()
    W_auxiliary = W.copy()
    t = 1.0
    t_previous = 0.0

    accepted_accelerated = 0
    rejected_accelerated = 0
    restarts = 0
    backtracking_count = 0
    del initial_projection_iterations, initial_projection_failures
    converged = False
    status = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        W_before = W
        H_before = H
        states_before: List[Optional[HullProjectionState]] = [
            state.copy() for state in projection_states
        ]

        chosen: Optional[PALMCandidate] = None
        accelerated_candidate: Optional[PALMCandidate] = None
        used_accelerated = False

        if acceleration != "none":
            G, V = _source_extrapolation(
                H,
                H_previous,
                H_auxiliary,
                W,
                W_previous,
                W_auxiliary,
                t,
                t_previous,
            )
            accelerated_candidate = _make_candidate(
                data,
                G,
                V,
                lambda_,
                projector,
                states_before,
                c_h=c_h,
                c_w=c_w,
                eps_step=eps_step,
            )
            monotone_bound = current_objective + objective_tolerance * max(
                1.0, abs(current_objective)
            )
            accelerated_is_acceptable = (
                np.isfinite(accelerated_candidate.objective)
                and accelerated_candidate.objective <= monotone_bound
            )

            if acceleration == "monotone_restart" and accelerated_is_acceptable:
                chosen = accelerated_candidate
                used_accelerated = True
                accepted_accelerated += 1

        plain_candidate: Optional[PALMCandidate] = None
        need_plain = acceleration == "none" or acceleration == "source_best" or chosen is None
        if need_plain:
            plain_accepted = False
            for backtrack in range(max_backtracking + 1):
                plain_candidate = _make_candidate(
                    data,
                    H,
                    W,
                    lambda_,
                    projector,
                    states_before,
                    c_h=c_h,
                    c_w=c_w,
                    eps_step=eps_step,
                    step_multiplier=backtracking_factor**backtrack,
                )
                descent_bound = current_objective + objective_tolerance * max(
                    1.0, abs(current_objective)
                )
                if (
                    np.isfinite(plain_candidate.objective)
                    and plain_candidate.objective <= descent_bound
                ):
                    plain_accepted = True
                    break
                backtracking_count += 1

            if plain_candidate is None or not np.isfinite(plain_candidate.objective):
                status = "non_finite_plain_candidate"
                warning_messages.append(status)
                break
            if not plain_accepted:
                status = "plain_candidate_failed_descent"
                warning_messages.append(status)
                break

            if acceleration == "source_best" and accelerated_candidate is not None:
                if (
                    np.isfinite(accelerated_candidate.objective)
                    and accelerated_candidate.objective <= plain_candidate.objective
                ):
                    chosen = accelerated_candidate
                    used_accelerated = True
                    accepted_accelerated += 1
                else:
                    chosen = plain_candidate
                    rejected_accelerated += 1
                    restarts += 1
            else:
                chosen = plain_candidate
                if acceleration == "monotone_restart":
                    rejected_accelerated += 1
                    restarts += 1

        if chosen is None:  # Defensive guard.
            status = "no_candidate"
            warning_messages.append(status)
            break

        if not (
            np.all(np.isfinite(chosen.W))
            and np.all(np.isfinite(chosen.H))
            and np.isfinite(chosen.objective)
        ):
            status = "non_finite_iterate"
            warning_messages.append(status)
            break

        relative_step = _relative_pair_step(W, H, chosen.W, chosen.H)
        W = chosen.W
        H = chosen.H
        projection_states = chosen.projection_states
        current_objective = chosen.objective
        current_reconstruction = chosen.reconstruction_loss
        current_penalty = chosen.archetype_penalty

        objective_trace.append(current_objective)
        reconstruction_trace.append(current_reconstruction)
        penalty_trace.append(current_penalty)
        relative_step_trace.append(relative_step)
        gamma_h_trace.append(chosen.gamma_h)
        gamma_w_trace.append(chosen.gamma_w)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        if acceleration == "none":
            H_previous = H.copy()
            W_previous = W.copy()
            H_auxiliary = H.copy()
            W_auxiliary = W.copy()
            t = 1.0
            t_previous = 0.0
        elif used_accelerated and accelerated_candidate is not None:
            H_previous = H_before.copy()
            W_previous = W_before.copy()
            H_auxiliary = accelerated_candidate.H.copy()
            W_auxiliary = accelerated_candidate.W.copy()
            t_previous = t
            t = t_new
        else:
            # Objective-based restart: erase momentum so the next extrapolated
            # point is exactly the accepted current iterate.
            H_previous = H.copy()
            W_previous = W.copy()
            H_auxiliary = H.copy()
            W_auxiliary = W.copy()
            t = 1.0
            t_previous = 0.0

        if verbose:
            print(
                f"iter={iteration:4d} objective={current_objective:.8e} "
                f"reconstruction={current_reconstruction:.8e} "
                f"penalty={current_penalty:.8e} rel_step={relative_step:.3e} "
                f"accelerated={used_accelerated}"
            )

        if relative_step <= tolerance:
            converged = True
            status = "converged_relative_step"
            break

    n_iter = len(relative_step_trace)

    if final_weight_refit_iterations > 0:
        W_refit = solve_weights_projected_gradient(
            data,
            H,
            W_init=W,
            max_iterations=final_weight_refit_iterations,
            tolerance=min(tolerance, 1e-10),
            accelerated=True,
        )
        refit_reconstruction = 0.5 * _squared_frobenius(data - W_refit @ H)
        refit_objective = refit_reconstruction + current_penalty
        if refit_objective <= current_objective + objective_tolerance * max(
            1.0, abs(current_objective)
        ):
            W = W_refit
            current_objective = refit_objective
            current_reconstruction = refit_reconstruction
            objective_trace.append(current_objective)
            reconstruction_trace.append(current_reconstruction)
            penalty_trace.append(current_penalty)

    row_sum_error = float(np.max(np.abs(W.sum(axis=1) - 1.0)))
    minimum_weight = float(np.min(W))
    if row_sum_error > 1e-8 or minimum_weight < -1e-10:
        message = (
            "weight feasibility tolerance exceeded: "
            f"max row-sum error={row_sum_error:.3e}, min W={minimum_weight:.3e}"
        )
        warning_messages.append(message)
        warnings.warn(message, RuntimeWarning)

    # Reproject final archetypes once. This both verifies the cached proximal
    # penalty and supplies sparse archetype-to-data hull coordinates.
    final_states: List[HullProjectionState] = []
    final_penalty_sq = 0.0
    if lambda_ > 0.0:
        for archetype, state in zip(H, projection_states):
            final_projection = projector.project(archetype, state=state)
            final_states.append(final_projection.state)
            final_penalty_sq += final_projection.squared_distance
        verified_penalty = 0.5 * lambda_ * final_penalty_sq
        verified_reconstruction = 0.5 * _squared_frobenius(data - W @ H)
        verified_objective = verified_reconstruction + verified_penalty
        cached_error = abs(verified_objective - current_objective)
        cached_scale = max(1.0, abs(verified_objective), abs(current_objective))
        if cached_error > 100.0 * projection_tolerance * cached_scale:
            warning_messages.append(
                "cached proximal objective differed from final direct projection "
                f"by {cached_error:.3e}"
            )
        current_objective = verified_objective
        current_reconstruction = verified_reconstruction
        current_penalty = verified_penalty
        objective_trace[-1] = current_objective
        reconstruction_trace[-1] = current_reconstruction
        penalty_trace[-1] = current_penalty
        projection_states = final_states

        archetype_projection_weights = np.zeros((n_archetypes, n), dtype=float)
        for k, state in enumerate(projection_states):
            original_indices = hull_indices[state.active_indices]
            archetype_projection_weights[k, original_indices] = state.weights
        archetype_projection_points = archetype_projection_weights @ data
    else:
        archetype_projection_weights = None
        archetype_projection_points = None

    condition_number, smallest_singular_value = _vertex_conditioning(H)
    if not np.isfinite(condition_number):
        warning_messages.append("final vertex matrix is rank deficient or numerically singular")

    # A one-step PALM fixed-point residual is a more meaningful stationarity
    # diagnostic than merely repeating the last accepted iterate difference.
    stationarity_candidate = _make_candidate(
        data,
        H,
        W,
        lambda_,
        projector,
        projection_states,
        c_h=c_h,
        c_w=c_w,
        eps_step=eps_step,
    )
    stationarity = _relative_pair_step(
        W, H, stationarity_candidate.W, stationarity_candidate.H
    )

    if projector.failures > 0:
        warning_messages.append(
            f"{projector.failures} convex-hull projections reached their iteration limit"
        )

    runtime = perf_counter() - start_time
    return PALMResult(
        W=W,
        H=H,
        W_initial=W_initial,
        H_initial=H_initial,
        archetype_projection_weights=archetype_projection_weights,
        archetype_projection_points=archetype_projection_points,
        converged=converged,
        status=status,
        n_iter=n_iter,
        lambda_=float(lambda_),
        acceleration=acceleration,
        objective_trace=np.asarray(objective_trace, dtype=float),
        reconstruction_trace=np.asarray(reconstruction_trace, dtype=float),
        penalty_trace=np.asarray(penalty_trace, dtype=float),
        relative_step_trace=np.asarray(relative_step_trace, dtype=float),
        stationarity=stationarity,
        gamma_h_trace=np.asarray(gamma_h_trace, dtype=float),
        gamma_w_trace=np.asarray(gamma_w_trace, dtype=float),
        accepted_accelerated_steps=accepted_accelerated,
        rejected_accelerated_steps=rejected_accelerated,
        restart_count=restarts,
        backtracking_count=backtracking_count,
        hull_projection_calls=projector.calls,
        hull_projection_iterations=projector.total_iterations,
        hull_projection_failures=projector.failures,
        runtime_seconds=float(runtime),
        vertex_condition_number=float(condition_number),
        vertex_smallest_singular_value=float(smallest_singular_value),
        initialization=initialization,
        hull_reduction=hull_status,
        hull_input_size=int(hull_points.shape[0]),
        warnings=warning_messages,
        metadata={
            "n": n,
            "d": d,
            "K": n_archetypes,
            "c_h": c_h,
            "c_w": c_w,
            "eps_step": eps_step,
            "projection_tolerance": projection_tolerance,
            "projection_max_iterations": projection_max_iterations,
            "hull_original_indices": (
                None
                if len(hull_indices) == n
                and np.array_equal(hull_indices, np.arange(n, dtype=np.int64))
                else hull_indices.tolist()
            ),
            "objective_scaling": "0.5_reconstruction_plus_0.5_lambda_penalty",
            "source_costfun_scaling": "twice_reported_objective",
            "row_sum_error": row_sum_error,
            "minimum_weight": minimum_weight,
            "fixed_lambda": True,
            "stationarity_type": "one_plain_palm_fixed_point_residual",
        },
    )


# ---------------------------------------------------------------------------
# GpLSI/common vertex-hunter adapter
# ---------------------------------------------------------------------------


def palm_aa_vertex_hunt(
    embedding: ArrayLike,
    K: int,
    *,
    random_state: Optional[int] = None,
    H_init: Optional[ArrayLike] = None,
    **method_parameters: Any,
) -> Dict[str, Any]:
    """Adapter suitable for a common ``vertex_hunt`` interface.

    Returns a dictionary instead of a dataclass so it can be serialized in the
    same way as other vertex hunters.
    """

    if "lambda_" not in method_parameters:
        raise ValueError("palm_aa_vertex_hunt requires an explicit lambda_")
    result = accelerated_palm_aa(
        embedding,
        K,
        H_init=H_init,
        random_state=random_state,
        **method_parameters,
    )
    return {
        "vertices": result.H,
        "selected_indices": None,
        "observation_weights": result.W,
        "archetype_to_data_weights": result.archetype_projection_weights,
        "archetype_projection_points": result.archetype_projection_points,
        "initialization_vertices": result.H_initial,
        "initialization_weights": result.W_initial,
        "initial_objective": float(result.objective_trace[0]),
        "final_objective": float(result.objective_trace[-1]),
        "iteration_count": result.n_iter,
        "stationarity": result.stationarity,
        "accepted_accelerated_steps": result.accepted_accelerated_steps,
        "restart_count": result.restart_count,
        "initialization": result.initialization,
        "objective_trace": result.objective_trace,
        "reconstruction_trace": result.reconstruction_trace,
        "penalty_trace": result.penalty_trace,
        "relative_step_trace": result.relative_step_trace,
        "condition_number": result.vertex_condition_number,
        "smallest_singular_value": result.vertex_smallest_singular_value,
        "runtime": result.runtime_seconds,
        "status": result.status,
        "failure": not np.all(np.isfinite(result.H)),
        "warnings": result.warnings,
        "parameters": {
            "K": K,
            "lambda_": result.lambda_,
            "acceleration": result.acceleration,
            **{
                key: value
                for key, value in method_parameters.items()
                if key != "lambda_"
            },
        },
        "metadata": result.to_dict(include_arrays=False),
        "fit_result": result,
    }


# ---------------------------------------------------------------------------
# Compatibility wrappers with source-like naming
# ---------------------------------------------------------------------------


def acc_palm_nmf_corrected(
    X: ArrayLike,
    r: int,
    l: float,
    *,
    H_init: Optional[ArrayLike] = None,
    W_init: Optional[ArrayLike] = None,
    maxiter: int = 300,
    delta: float = 1e-7,
    c1: float = 1.1,
    c2: float = 1.1,
    acceleration: AccelerationMode = "monotone_restart",
    **kwargs: Any,
) -> PALMResult:
    """Source-style argument names, returning the richer :class:`PALMResult`."""

    return accelerated_palm_aa(
        X,
        n_archetypes=r,
        lambda_=l,
        H_init=H_init,
        W_init=W_init,
        acceleration=acceleration,
        max_iterations=maxiter,
        tolerance=delta,
        c_h=c1,
        c_w=c2,
        **kwargs,
    )


__all__ = [
    "AccelerationMode",
    "ConvexHullProjector",
    "HullProjectionResult",
    "HullProjectionState",
    "PALMResult",
    "acc_palm_nmf_corrected",
    "accelerated_palm_aa",
    "palm_aa_vertex_hunt",
    "project_simplex_rows",
    "reduce_to_convex_hull_vertices",
    "solve_weights_projected_gradient",
    "successive_projection_init",
]
