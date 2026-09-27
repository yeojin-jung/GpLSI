"""Tests for accelerated_palm_aa.py.

Run from the directory containing both files with:

    pytest -q test_accelerated_palm_aa.py
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from gplsi.accelerated_palm_aa import (
    ConvexHullProjector,
    HullProjectionState,
    _make_candidate,
    accelerated_palm_aa,
    palm_aa_vertex_hunt,
    project_simplex_rows,
    reduce_to_convex_hull_vertices,
    solve_weights_projected_gradient,
    successive_projection_init,
)


def source_project_simplex_reference(x: np.ndarray) -> np.ndarray:
    """Python 3 transcription of the authors' scalar-row routine."""

    x = np.asarray(x, dtype=float)
    n = len(x)
    xord = -np.sort(-x)
    sx = np.sum(x)
    lam = (sx - 1.0) / n
    if lam <= xord[n - 1]:
        return x - lam
    k = n - 1
    flag = False
    while (not flag) and k > 0:
        sx -= xord[k]
        lam = (sx - 1.0) / k
        if xord[k] <= lam <= xord[k - 1]:
            flag = True
        k -= 1
    return np.maximum(x - lam, 0.0)


def slsqp_hull_projection(X: np.ndarray, query: np.ndarray) -> np.ndarray:
    n = len(X)

    def objective(weights: np.ndarray) -> float:
        residual = weights @ X - query
        return 0.5 * float(residual @ residual)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return X @ (weights @ X - query)

    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-13, "maxiter": 2_000},
    )
    assert result.success, result.message
    return result.x @ X


def synthetic_simplex(seed: int = 0, n: int = 250, K: int = 3):
    rng = np.random.default_rng(seed)
    H = np.eye(K)
    W = rng.dirichlet(np.full(K, 0.35), size=n)
    X = W @ H + 0.005 * rng.normal(size=(n, K))
    return X, W, H


def finite_difference(
    function,
    point: np.ndarray,
    direction: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    return float(
        (function(point + epsilon * direction) - function(point - epsilon * direction))
        / (2.0 * epsilon)
    )


def test_batched_simplex_projection_matches_source_reference():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(200, 9))
    expected = np.vstack([source_project_simplex_reference(row) for row in values])
    observed = project_simplex_rows(values)
    assert_allclose(observed, expected, atol=2e-14, rtol=2e-14)
    assert np.min(observed) >= 0.0
    assert_allclose(observed.sum(axis=1), 1.0, atol=2e-15)


def test_batched_simplex_projection_accepts_one_vector():
    observed = project_simplex_rows(np.array([-2.0, 0.1, 1.2]))
    assert observed.shape == (3,)
    assert_allclose(observed.sum(), 1.0)
    assert np.min(observed) >= 0.0


@pytest.mark.parametrize("seed", range(5))
def test_convex_hull_projection_matches_slsqp(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(25, 4))
    query = 2.0 * rng.normal(size=4)
    projector = ConvexHullProjector(X, tolerance=1e-12, max_iterations=2_000)
    result = projector.project(query)
    expected = slsqp_hull_projection(X, query)
    assert result.converged
    assert_allclose(result.point, expected, atol=1e-6, rtol=1e-6)
    assert_allclose(
        result.squared_distance,
        np.sum((expected - query) ** 2),
        atol=1e-10,
        rtol=1e-8,
    )


def test_convex_hull_projection_warm_start_preserves_answer():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 5))
    first_query = rng.normal(size=5)
    second_query = first_query + 1e-3 * rng.normal(size=5)
    projector = ConvexHullProjector(X, tolerance=1e-12)
    first = projector.project(first_query)
    warm = projector.project(second_query, state=first.state)
    cold = ConvexHullProjector(X, tolerance=1e-12).project(second_query)
    assert_allclose(warm.point, cold.point, atol=1e-8, rtol=1e-8)
    assert warm.iterations <= cold.iterations + 2


def test_hull_reduction_is_projection_equivalent():
    rng = np.random.default_rng(4)
    outer = rng.normal(size=(20, 3))
    mixing = rng.dirichlet(np.ones(len(outer)), size=100)
    interior = mixing @ outer
    X = np.vstack([outer, interior])
    reduced, indices, status = reduce_to_convex_hull_vertices(X, mode="exact")
    assert status.startswith("exact")
    assert len(indices) <= len(X)

    query = rng.normal(size=3) * 2.0
    full_result = ConvexHullProjector(X, tolerance=1e-12).project(query)
    reduced_result = ConvexHullProjector(reduced, tolerance=1e-12).project(query)
    assert_allclose(full_result.point, reduced_result.point, atol=1e-7, rtol=1e-7)


def test_reconstruction_block_gradients_by_finite_difference():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(9, 4))
    W = project_simplex_rows(rng.normal(size=(9, 3)))
    H = rng.normal(size=(3, 4))

    H_direction = rng.normal(size=H.shape)
    H_gradient = W.T @ (W @ H - X)
    numerical_h = finite_difference(
        lambda candidate: 0.5 * np.sum((X - W @ candidate) ** 2),
        H,
        H_direction,
    )
    analytic_h = float(np.sum(H_gradient * H_direction))
    assert_allclose(numerical_h, analytic_h, atol=1e-7, rtol=1e-6)

    W_direction = rng.normal(size=W.shape)
    W_gradient = (W @ H - X) @ H.T
    numerical_w = finite_difference(
        lambda candidate: 0.5 * np.sum((X - candidate @ H) ** 2),
        W,
        W_direction,
    )
    analytic_w = float(np.sum(W_gradient * W_direction))
    assert_allclose(numerical_w, analytic_w, atol=1e-7, rtol=1e-6)


def test_cached_candidate_objective_matches_direct_projection():
    X, _, _ = synthetic_simplex(seed=6, n=100, K=3)
    H = successive_projection_init(X, 3)
    W = solve_weights_projected_gradient(X, H, max_iterations=50)
    projector = ConvexHullProjector(X, tolerance=1e-12)
    states = [None, None, None]
    candidate = _make_candidate(
        X,
        H,
        W,
        lambda_=0.7,
        projector=projector,
        projection_states=states,
        c_h=1.1,
        c_w=1.1,
        eps_step=1e-12,
    )

    direct_penalty_sq = 0.0
    for archetype, state in zip(candidate.H, candidate.projection_states):
        result = projector.project(archetype, state=state)
        direct_penalty_sq += result.squared_distance
    direct_reconstruction = 0.5 * np.sum((X - candidate.W @ candidate.H) ** 2)
    direct_objective = direct_reconstruction + 0.5 * 0.7 * direct_penalty_sq
    assert_allclose(candidate.objective, direct_objective, atol=2e-9, rtol=2e-8)


def test_plain_palm_objective_is_monotone():
    X, _, _ = synthetic_simplex(seed=7)
    H_init = successive_projection_init(X, 3)
    result = accelerated_palm_aa(
        X,
        3,
        lambda_=1.0,
        H_init=H_init,
        acceleration="none",
        max_iterations=40,
        tolerance=1e-12,
        hull_reduction="none",
    )
    differences = np.diff(result.objective_trace)
    assert np.max(differences) <= 1e-10
    assert result.objective_trace[-1] < result.objective_trace[0]


def test_acceleration_is_safeguarded_and_feasible():
    X, _, _ = synthetic_simplex(seed=8)
    H_init = successive_projection_init(X, 3)
    result = accelerated_palm_aa(
        X,
        3,
        lambda_=0.5,
        H_init=H_init,
        acceleration="monotone_restart",
        max_iterations=60,
        tolerance=1e-10,
        hull_reduction="none",
    )
    assert np.max(np.diff(result.objective_trace)) <= 1e-10
    assert np.min(result.W) >= -1e-14
    assert_allclose(result.W.sum(axis=1), 1.0, atol=2e-14)
    assert result.accepted_accelerated_steps + result.rejected_accelerated_steps >= 1
    assert result.hull_projection_calls < X.shape[0] * max(1, result.n_iter)
    assert np.isfinite(result.stationarity)
    assert result.stationarity >= 0.0
    assert result.archetype_projection_weights is not None
    assert_allclose(
        result.archetype_projection_weights.sum(axis=1), 1.0, atol=2e-12
    )
    assert_allclose(
        result.archetype_projection_points,
        result.archetype_projection_weights @ X,
        atol=2e-12,
    )


def test_max_iterations_zero_preserves_supplied_initialization():
    X, _, _ = synthetic_simplex(seed=9, n=50, K=3)
    H_init = successive_projection_init(X, 3)
    W_init = project_simplex_rows(np.arange(150, dtype=float).reshape(50, 3))
    result = accelerated_palm_aa(
        X,
        3,
        lambda_=1.0,
        H_init=H_init,
        W_init=W_init,
        max_iterations=0,
        hull_reduction="none",
    )
    assert_allclose(result.H, H_init)
    assert_allclose(result.W, W_init)
    assert result.n_iter == 0


def test_vertex_hunter_adapter_has_expected_fields():
    X, _, _ = synthetic_simplex(seed=10, n=80, K=3)
    H_init = successive_projection_init(X, 3)
    output = palm_aa_vertex_hunt(
        X,
        3,
        H_init=H_init,
        lambda_=0.5,
        max_iterations=5,
        hull_reduction="none",
    )
    assert output["vertices"].shape == (3, 3)
    assert output["observation_weights"].shape == (80, 3)
    assert "condition_number" in output
    assert "objective_trace" in output
    assert output["parameters"]["lambda_"] == 0.5
    assert output["initialization_vertices"].shape == (3, 3)
    assert output["archetype_to_data_weights"].shape == (3, 80)


def test_rank_deficient_vertices_are_flagged():
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 1.0, 60)
    X = np.column_stack([x, 2.0 * x, 3.0 * x])
    H_init = X[[0, 20, 59]]
    result = accelerated_palm_aa(
        X,
        3,
        lambda_=1.0,
        H_init=H_init,
        max_iterations=3,
        hull_reduction="none",
    )
    assert not np.isfinite(result.vertex_condition_number)
    assert any("rank deficient" in message for message in result.warnings)
