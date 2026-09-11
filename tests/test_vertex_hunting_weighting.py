from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from gplsi.generate_topic_model import generate_data, generate_weights_edge
from gplsi.gplsi import GpLSI
from gplsi.preprocessing import (
    PreprocessingError,
    frequency_weights,
    preprocess_features,
    select_feature_columns,
    weighted_debiased_correction,
)
from gplsi.recovery import (
    finite_difference_gradient,
    poisson_objective_and_gradient,
    project_simplex,
    recover_W,
    refit_A_full_l2,
    refit_A_full_poisson,
    spectral_A_unweighted,
)
from gplsi.vertex_hunting import exhaustive_vertex_search, vertex_hunt


FIXTURES = Path(__file__).parent / "fixtures"


def _metadata(path: Path) -> dict[str, str]:
    return dict(
        line.strip().split("=", 1)
        for line in path.read_text().splitlines()
        if line.strip()
    )


def _sort_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    keys = tuple(matrix[:, column] for column in reversed(range(matrix.shape[1])))
    return matrix[np.lexsort(keys)]


def test_baseline_noop_regression() -> None:
    fixture = np.load(FIXTURES / "gplsi_baseline_seed_950.npz")
    config = json.loads(str(fixture["config_json"]))
    np.random.seed(config["seed"])
    coords, _, _, X = generate_data(
        config["N"],
        config["n"],
        config["p"],
        config["K"],
        config["rt"],
        config["n_clusters"],
    )
    weights, edge_df = generate_weights_edge(
        coords, config["nearest_n"], config["phi"]
    )
    model = GpLSI(
        lamb_start=config["lamb_start"],
        step_size=config["step_size"],
        grid_len=config["grid_len"],
        maxiter=config["maxiter"],
        eps=config["eps"],
        precondition=config["precondition"],
        initialize=config["initialize"],
        threshold_method="none",
        weight_method="none",
        vertex_hunter="spa_current",
        A_recovery="current",
        initialization="current",
    ).fit(X, config["N"], config["K"], edge_df, weights)

    provenance = json.loads(str(fixture["provenance_json"]))
    atol, rtol = provenance["comparison_atol"], provenance["comparison_rtol"]
    np.testing.assert_allclose(model.U, fixture["U"], atol=atol, rtol=rtol)
    np.testing.assert_allclose(model.V, fixture["V"], atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        np.diag(model.L), fixture["singular_values"], atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(model.W_hat, fixture["W_hat"], atol=atol, rtol=rtol)
    np.testing.assert_allclose(model.A_hat, fixture["A_hat"], atol=atol, rtol=rtol)
    np.testing.assert_array_equal(model.anchor_indices, fixture["anchor_indices"])
    assert model.lambd == json.loads(str(fixture["metrics_json"]))["selected_rho"]


def test_spa_wrapper_matches_current_vertices_and_W() -> None:
    rng = np.random.default_rng(73)
    embedding, _ = np.linalg.qr(rng.normal(size=(30, 3)))
    old_model = GpLSI()
    current_embedding = embedding.copy()
    old_indices, old_vertices = old_model.preconditioned_spa(
        current_embedding, 3, precondition=False
    )
    wrapped = vertex_hunt(embedding, 3, "spa_current")
    np.testing.assert_array_equal(wrapped.selected_observation_indices, old_indices)
    np.testing.assert_allclose(wrapped.vertices, old_vertices, atol=0, rtol=0)
    current_W = old_model.get_W_hat(current_embedding, old_vertices)
    stable_W = recover_W(wrapped.embedding_used, wrapped.vertices).simplex_projected
    np.testing.assert_allclose(stable_W, current_W, atol=1e-12, rtol=1e-12)


def test_exact_tran_threshold_fixture_parity() -> None:
    eta_fixture = pd.read_csv(FIXTURES / "tran_threshold_eta.csv")
    matrix_fixture = pd.read_csv(FIXTURES / "tran_threshold_matrix.csv")
    metadata = _metadata(FIXTURES / "tran_threshold_metadata.txt")
    counts = np.array(
        [
            [8, 1, 4, 3, 1, 1, 1, 1, 0, 0],
            [7, 1, 4, 3, 1, 1, 1, 1, 1, 0],
            [6, 2, 4, 3, 1, 1, 1, 1, 1, 0],
            [7, 1, 4, 3, 1, 1, 1, 1, 1, 0],
            [1, 8, 4, 3, 1, 1, 1, 1, 0, 0],
            [1, 7, 4, 3, 1, 1, 1, 1, 1, 0],
            [2, 6, 4, 3, 1, 1, 1, 1, 1, 0],
            [1, 7, 4, 3, 1, 1, 1, 1, 1, 0],
        ],
        dtype=float,
    )
    X = counts / 20.0
    result = select_feature_columns(
        X, 20, method="tran_script_exact", alpha=float(metadata["alpha"])
    )
    np.testing.assert_allclose(result.eta_hat, eta_fixture["eta_hat"])
    np.testing.assert_allclose(result.threshold_value, float(metadata["threshold"]))
    expected = eta_fixture.loc[eta_fixture.retained, "feature_index_r"].to_numpy() - 1
    np.testing.assert_array_equal(result.retained_indices, expected)
    np.testing.assert_allclose(
        X[:, result.retained_indices].T,
        matrix_fixture.iloc[:, 1:].to_numpy(),
    )


def test_tran_script_and_paper_modes_are_distinct() -> None:
    X = np.array(
        [
            [0.4, 0.3, 0.2, 0.1, 0.0],
            [0.4, 0.3, 0.2, 0.1, 0.0],
            [0.4, 0.3, 0.2, 0.1, 0.0],
        ]
    )
    script = select_feature_columns(X, 1, method="tran_script_exact", alpha=10.0)
    paper = select_feature_columns(X, 1, method="tran_paper_exact", alpha=10.0)
    assert script.fallback_active
    assert script.retained_feature_count == 1
    assert paper.retained_feature_count == 0
    assert select_feature_columns(X, 1, method="tran", alpha=10.0).effective_method == "tran_paper_exact"


def test_thresholding_is_columnwise_without_row_renormalization() -> None:
    X = np.array([[0.7, 0.2, 0.1], [0.8, 0.2, 0.0]])
    result = preprocess_features(
        X,
        N=10,
        threshold_method="tran_paper_exact",
        alpha=1.0,
        weight_method="none",
    )
    np.testing.assert_allclose(
        result.X_retained, X[:, result.threshold.retained_indices]
    )
    np.testing.assert_allclose(
        result.threshold.retained_row_mass, result.X_retained.sum(axis=1)
    )
    assert np.any(result.threshold.retained_row_mass < 1.0)


def test_weighted_factorization_geometry_and_graph_support() -> None:
    W = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.2, 0.3, 0.5]],
        dtype=float,
    )
    A = np.array(
        [[0.4, 0.2, 0.1, 0.2, 0.1], [0.1, 0.4, 0.2, 0.1, 0.2], [0.2, 0.1, 0.4, 0.2, 0.1]]
    )
    J = np.array([0, 1, 2, 3])
    weights = np.array([0.8, 1.5, 0.7, 2.0])
    transformed = (W @ A[:, J]) * weights
    np.testing.assert_allclose(transformed, W @ (A[:, J] * weights))
    U, _, _ = np.linalg.svd(transformed, full_matrices=False)
    U = U[:, :3]
    H, _, _, _ = np.linalg.lstsq(W, U, rcond=None)
    np.testing.assert_allclose(U, W @ H, atol=1e-12)
    assert np.linalg.matrix_rank(transformed) == 3
    projector_W = W @ np.linalg.pinv(W)
    projector_U = U @ U.T
    np.testing.assert_allclose(projector_U, projector_W, atol=1e-12)
    Gamma = np.array([[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 0, 1, -1]])
    support_W = np.linalg.norm(Gamma @ W, axis=1) > 1e-12
    support_U = np.linalg.norm(Gamma @ U, axis=1) > 1e-12
    np.testing.assert_array_equal(support_U, support_W)


def test_weighted_diagonal_correction_identity_and_nonidentity() -> None:
    eta = np.array([0.25, 0.1, 0.04])
    weights = eta ** -0.5
    np.testing.assert_allclose(weights**2 * eta, np.ones(3))
    correction = weighted_debiased_correction(eta, weights, n=20, N=10)
    np.testing.assert_allclose(correction, np.full(3, 2.0))
    floored = (eta + 0.02) ** -0.5
    correction_floor = weighted_debiased_correction(eta, floored, n=20, N=10)
    assert not np.allclose(correction_floor, np.full(3, 2.0))


def test_frequency_weight_metadata_and_cap() -> None:
    X = np.array([[0.8, 0.19, 0.01], [0.7, 0.29, 0.01]])
    uncapped = frequency_weights(X, method="ke_empirical")
    capped = frequency_weights(
        X, method="ke_empirical_capped", cap=float(np.median(uncapped.weights))
    )
    assert capped.cap_active
    assert capped.quantiles["max"] <= capped.cap
    np.testing.assert_allclose(uncapped.effective_variance, X.shape[1])


def test_simplex_projection_enforces_sum_under_large_cancellation() -> None:
    projected = project_simplex(np.array([-2.0e10, 1.0e10 + 0.25, 1.0e10]))
    assert np.all(projected >= 0)
    np.testing.assert_allclose(projected.sum(), 1.0, atol=2e-15, rtol=0)
    almost_affine = project_simplex(np.array([0.3047468, 0.3271020, 0.3681575]))
    np.testing.assert_allclose(almost_affine.sum(), 1.0, atol=2e-15, rtol=0)


def test_topic_unweighting_noiseless() -> None:
    W = np.array([[1, 0], [0, 1], [0.4, 0.6], [0.7, 0.3]], dtype=float)
    A = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]], dtype=float)
    weights = np.array([0.5, 2.0, 1.3])
    transformed = (W @ A) * weights
    U, singular, Vt = np.linalg.svd(transformed, full_matrices=False)
    U, singular, Vt = U[:, :2], singular[:2], Vt[:2]
    H = U[:2]
    result = spectral_A_unweighted(
        H, singular, Vt.T, weights, np.arange(3), 3
    )
    np.testing.assert_allclose(result.A_hat, A, atol=1e-12)


def test_integrated_weighted_spectral_recovery_handles_spa_signs() -> None:
    W = np.array(
        [[1, 0], [0, 1], [0.25, 0.75], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]],
        dtype=float,
    )
    A = np.array([[0.5, 0.3, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]])
    X = W @ A
    model = GpLSI(
        method="pLSI",
        threshold_method="none",
        weight_method="ke_empirical",
        vertex_hunter="spa_current",
        A_recovery="A_spectral_unweighted",
        initialization="direct_svd",
        random_state=2,
    ).fit(X, 1000, 2, None, None)
    distances = np.linalg.norm(model.A_hat[:, None] - A[None, :], axis=2)
    rows, columns = linear_sum_assignment(distances)
    np.testing.assert_allclose(model.A_hat[rows], A[columns], atol=1e-12)
    np.testing.assert_allclose(model.W_hat.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(model.W_hat @ model.A_hat, X, atol=1e-12)


@pytest.mark.parametrize("case_name", ["k2", "k3"])
def test_svs_fixed_center_source_parity(case_name: str) -> None:
    centers_frame = pd.read_csv(FIXTURES / f"mixedscore_svs_{case_name}_centers.csv")
    result_frame = pd.read_csv(FIXTURES / f"mixedscore_svs_{case_name}_result.csv")
    metadata = _metadata(FIXTURES / f"mixedscore_svs_{case_name}_metadata.txt")
    centers = centers_frame.iloc[:, 1:].to_numpy()
    K = int(metadata["K"])
    vertices, indices, objective, evaluated = exhaustive_vertex_search(centers, K)
    np.testing.assert_array_equal(indices, result_frame.selected_index_r.to_numpy() - 1)
    np.testing.assert_allclose(vertices, result_frame.iloc[:, 1:].to_numpy(), atol=1e-8)
    np.testing.assert_allclose(objective, float(metadata["objective"]), atol=2e-7)
    assert evaluated == int(metadata["candidate_count"])


def test_svs_adaptive_source_parity_on_identical_R_centers() -> None:
    metadata = _metadata(FIXTURES / "mixedscore_adaptive_metadata.txt")
    K = int(metadata["K"])
    cache = {}
    for L in range(K, 3 * K + 1):
        frame = pd.read_csv(FIXTURES / f"mixedscore_adaptive_centers_L{L}.csv")
        centers = frame.iloc[:, 1:].to_numpy()
        cache[L] = (centers, np.zeros(15, dtype=int), [])
    point_cloud = pd.read_csv(FIXTURES / "mixedscore_adaptive_point_cloud.csv").iloc[:, 1:].to_numpy()
    result = vertex_hunt(
        point_cloud,
        K,
        "svs",
        L_mode="mixedscore_adaptive",
        center_cache=cache,
        max_simplexes=1000,
        raise_on_failure=True,
    )
    assert result.parameters["L"] == int(metadata["selected_L"])
    expected = pd.read_csv(FIXTURES / "mixedscore_adaptive_selected_vertices.csv").iloc[:, 1:].to_numpy()
    np.testing.assert_allclose(result.vertices, expected, atol=2e-7)
    r_summary = pd.read_csv(FIXTURES / "mixedscore_adaptive_summary.csv")
    py_summary = pd.DataFrame(result.parameters["candidate_L_details"])
    np.testing.assert_allclose(
        py_summary.simplex_fitting_objective, r_summary.objective, atol=2e-7
    )
    np.testing.assert_allclose(py_summary.stability_score, r_summary.stability, atol=2e-7)


def test_svs_and_svs_star_share_centers_but_are_distinct() -> None:
    rng = np.random.default_rng(111)
    points = rng.normal(size=(50, 3))
    cache = {}
    svs = vertex_hunt(
        points, 3, "svs", random_state=7, L_mode="fixed", L=7, center_cache=cache
    )
    star = vertex_hunt(
        points, 3, "svs_star", random_state=7, L_mode="fixed", L=7, center_cache=cache
    )
    assert svs.success and star.success
    np.testing.assert_array_equal(svs.centers, star.centers)
    assert svs.parameters["candidate_simplexes_evaluated"] == 35
    assert star.parameters["second_stage"].startswith("spa_current")
    assert set(map(tuple, star.vertices)).issubset(set(map(tuple, star.centers)))


def test_svs_star_can_reuse_the_exact_adaptive_svs_selection() -> None:
    rng = np.random.default_rng(112)
    points = rng.normal(size=(30, 2))
    cache = {}
    svs = vertex_hunt(
        points,
        2,
        "svs",
        random_state=9,
        L_mode="mixedscore_adaptive",
        max_simplexes=1000,
        center_cache=cache,
        raise_on_failure=True,
    )
    reused = vertex_hunt(
        points,
        2,
        "svs_star",
        random_state=9,
        L_mode="mixedscore_adaptive",
        max_simplexes=1000,
        center_cache=cache,
        preselected_svs_details=svs.parameters,
        raise_on_failure=True,
    )
    direct = vertex_hunt(
        points,
        2,
        "svs_star",
        random_state=9,
        L_mode="mixedscore_adaptive",
        max_simplexes=1000,
        center_cache=cache,
        raise_on_failure=True,
    )
    np.testing.assert_allclose(reused.vertices, direct.vertices, atol=0, rtol=0)
    assert reused.parameters["effective_L_mode"] == (
        "svs_star_L_reused_from_same_task_svs"
    )
    assert reused.parameters["L"] == svs.parameters["L"]


def test_svs_star_L_equals_n_bypass_reduces_to_spa() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(20, 3))
    spa = vertex_hunt(points, 3, "spa_current")
    star = vertex_hunt(
        points,
        3,
        "svs_star",
        L_mode="fixed",
        L=len(points),
        bypass_kmeans=True,
    )
    np.testing.assert_array_equal(
        star.selected_observation_indices, spa.selected_observation_indices
    )
    signs = np.where(points[0] < 0, -1.0, 1.0)
    np.testing.assert_allclose(star.vertices * signs, spa.vertices, atol=0, rtol=0)


def test_svs_reports_combinatorial_infeasibility() -> None:
    centers = np.arange(30.0).reshape(10, 3)
    with pytest.raises(Exception, match="computationally infeasible"):
        exhaustive_vertex_search(centers, 5, max_simplexes=10)


def test_ppspa_official_source_parity() -> None:
    fixture = np.load(FIXTURES / "ppspa_official_fixture.npz")
    result = vertex_hunt(
        fixture["samples"],
        3,
        "pp_spa",
        radius_divisor=20,
        m_neighbors=4,
        min_neighbors=3,
        raise_on_failure=True,
    )
    np.testing.assert_allclose(result.projected_points, fixture["projected_points"], atol=1e-12)
    np.testing.assert_allclose(result.pseudo_points, fixture["pseudo_points"], atol=1e-12)
    np.testing.assert_allclose(result.vertices, fixture["vertices"], atol=1e-12)
    assert result.parameters["official_parameter_N"] == 4
    assert result.parameters["official_parameter_t"] == 3


def test_vertex_methods_reproducible_and_row_permutation_invariant() -> None:
    rng = np.random.default_rng(98)
    points = rng.normal(size=(60, 3))
    first = vertex_hunt(points, 3, "svs_star", random_state=42, L_mode="fixed", L=8)
    second = vertex_hunt(points, 3, "svs_star", random_state=42, L_mode="fixed", L=8)
    np.testing.assert_allclose(first.centers, second.centers, atol=0, rtol=0)
    np.testing.assert_allclose(first.vertices, second.vertices, atol=0, rtol=0)
    order = rng.permutation(len(points))
    permuted = vertex_hunt(
        points[order], 3, "svs_star", random_state=42, L_mode="fixed", L=8
    )
    np.testing.assert_allclose(_sort_rows(first.vertices), _sort_rows(permuted.vertices), atol=1e-10)


def test_rank_failure_is_explicit() -> None:
    X = np.array([[0.49, 0.49, 0.01, 0.01], [0.48, 0.50, 0.01, 0.01]])
    true_A = np.array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    with pytest.raises(PreprocessingError, match="rank"):
        preprocess_features(
            X,
            10,
            threshold_method="tran_paper_exact",
            alpha=0.5,
            weight_method="none",
            K=2,
            true_A=true_A,
            fail_on_rank_loss=True,
        )


def test_full_l2_refit_constraints_and_monotonicity() -> None:
    rng = np.random.default_rng(6)
    W = rng.dirichlet(np.ones(3), size=30)
    A = rng.dirichlet(np.ones(8), size=3)
    X = W @ A
    result = refit_A_full_l2(W, X, max_iter=5000)
    assert result.converged
    assert np.all(result.A_hat >= -1e-14)
    np.testing.assert_allclose(result.A_hat.sum(axis=1), 1.0)
    assert np.all(np.diff(result.objective_history) <= 1e-10)
    np.testing.assert_allclose(result.A_hat, A, atol=1e-6)


def test_poisson_gradient_and_objective_monotonicity() -> None:
    rng = np.random.default_rng(8)
    W = rng.dirichlet(np.ones(2), size=5)
    A = rng.dirichlet(np.ones(4), size=2)
    lengths = np.array([20, 25, 30, 35, 40], dtype=float)
    counts = np.vstack(
        [rng.multinomial(int(lengths[i]), W[i] @ A) for i in range(len(W))]
    )
    value, analytic = poisson_objective_and_gradient(A, W, counts, lengths, 1e-9)
    numeric = finite_difference_gradient(
        lambda candidate: poisson_objective_and_gradient(
            candidate, W, counts, lengths, 1e-9
        )[0],
        A,
    )
    assert np.isfinite(value)
    np.testing.assert_allclose(analytic, numeric, atol=2e-5, rtol=2e-5)
    result = refit_A_full_poisson(W, counts, lengths, max_iter=500)
    assert np.all(result.A_hat >= -1e-14)
    np.testing.assert_allclose(result.A_hat.sum(axis=1), 1.0)
    assert np.all(np.diff(result.objective_history) <= 1e-9)
