from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from gplsi.anchor_word import (
    build_word_profile,
    population_word_geometry,
    recover_W_from_word_vertices,
)
from gplsi.preprocessing import select_feature_columns
from gplsi.recovery import refit_A_current, refit_A_full_poisson
from gplsi.topicscore import fit_topicscore_graph_denoised, fit_topicscore_raw
from gplsi.vertex_hunting import vertex_hunt


FIXTURE = Path(__file__).parent / "fixtures" / "anchor_word_gplsi" / "tran_mixed_seed_4132"


def _metadata() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (FIXTURE / "metadata.txt").read_text().splitlines()
        if line.strip()
    )


def _population_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260903)
    W = rng.dirichlet(np.array([1.2, 0.8, 1.5]), size=24)
    A = np.array(
        [
            [0.40, 0.00, 0.00, 0.20, 0.15, 0.15, 0.10],
            [0.00, 0.35, 0.00, 0.10, 0.20, 0.15, 0.20],
            [0.00, 0.00, 0.45, 0.10, 0.10, 0.20, 0.15],
        ],
        dtype=float,
    )
    U, _ = np.linalg.qr(W)
    M = W @ A
    return W, A, U, M


def _align(A_left: np.ndarray, A_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(A_left[:, None] - A_right[None, :], axis=2)
    left, right = linear_sum_assignment(distances)
    return left, right


def test_exact_tran_mixed_decay_fixture_dimensions_seed_and_anchors() -> None:
    metadata = _metadata()
    counts = pd.read_csv(FIXTURE / "counts_document_by_word.csv").to_numpy(float)
    A_feature_topic = pd.read_csv(FIXTURE / "A_feature_by_topic.csv").to_numpy(float)
    W_topic_document = pd.read_csv(FIXTURE / "W_topic_by_document.csv").to_numpy(float)
    population = pd.read_csv(FIXTURE / "population_document_by_word.csv").to_numpy(float)
    vocab = pd.read_csv(FIXTURE / "vocabulary.csv").vocab_index_r.to_numpy(int)

    assert metadata["seed"] == "4132"
    assert metadata["a_zipf"] == "1"
    assert metadata["offset_zipf"] == "2.7"
    assert metadata["n_anchors"] == "2"
    assert counts.shape == (36, 78)
    assert A_feature_topic.shape == (78, 3)
    assert W_topic_document.shape == (3, 36)
    np.testing.assert_allclose(counts.sum(axis=1), 60)
    np.testing.assert_allclose(A_feature_topic.sum(axis=0), 1)
    np.testing.assert_allclose(W_topic_document.sum(axis=0), 1)
    reconstructed_from_returned_factors = W_topic_document.T @ A_feature_topic.T
    np.testing.assert_allclose(reconstructed_from_returned_factors.sum(axis=1), 1)
    # Exact local-source behavior: D0 is restricted after the count draw, while
    # returned A is additionally topic-normalized.  The two returned objects
    # are therefore intentionally not an exact factorization after zero-word
    # deletion.
    assert population.sum(axis=1).min() < 1
    np.testing.assert_allclose(
        np.max(np.abs(population - reconstructed_from_returned_factors)),
        0.00013186226282004,
        atol=1e-15,
    )
    for original_index in range(1, 7):
        retained = int(np.flatnonzero(vocab == original_index)[0])
        topic = (original_index - 1) // 2
        assert A_feature_topic[retained, topic] > 0
        np.testing.assert_allclose(
            np.delete(A_feature_topic[retained], topic), 0, atol=0, rtol=0
        )


def test_anchor_word_population_geometry_and_exact_vertices() -> None:
    W, A, U, M = _population_case()
    b, eta, Pi, H_word = population_word_geometry(W, A, U)
    Z = (M.T @ U) / eta[:, None]
    assert np.all(Pi >= -1e-14)
    np.testing.assert_allclose(Pi.sum(axis=1), 1, atol=1e-13)
    np.testing.assert_allclose(Z, Pi @ H_word, atol=1e-12)
    for topic, anchor_word in enumerate([0, 1, 2]):
        np.testing.assert_allclose(Pi[anchor_word], np.eye(3)[topic], atol=1e-12)
        np.testing.assert_allclose(Z[anchor_word], H_word[topic], atol=1e-12)
    np.testing.assert_allclose(U @ H_word.T, W / b[None, :], atol=1e-12)


def test_exact_W_recovery_rotation_and_topic_permutation() -> None:
    W, A, U, _ = _population_case()
    b, _, _, H_word = population_word_geometry(W, A, U)
    recovered = recover_W_from_word_vertices(U, H_word)
    np.testing.assert_allclose(recovered.G_hat, W / b[None, :], atol=1e-12)
    np.testing.assert_allclose(recovered.G_hat @ b, 1, atol=1e-12)
    np.testing.assert_allclose(recovered.b_hat, b, atol=1e-9)
    np.testing.assert_allclose(recovered.simplex_projected, W, atol=1e-9)

    rng = np.random.default_rng(91)
    O, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = recover_W_from_word_vertices(U @ O, H_word @ O)
    np.testing.assert_allclose(rotated.G_hat, recovered.G_hat, atol=1e-12)
    np.testing.assert_allclose(rotated.simplex_projected, W, atol=1e-9)

    permutation = np.array([2, 0, 1])
    permuted = recover_W_from_word_vertices(U, H_word[permutation])
    np.testing.assert_allclose(
        permuted.simplex_projected, W[:, permutation], atol=1e-9
    )
    current = refit_A_current(permuted.simplex_projected, W @ A)
    np.testing.assert_allclose(current.A_hat, A[permutation], atol=1e-9)


def test_weighted_original_scale_word_profile_and_exact_recovery() -> None:
    W, A, _, M = _population_case()
    retained = np.arange(A.shape[1])
    weights = np.array([0.7, 1.4, 0.8, 2.1, 1.3, 0.9, 1.7])
    transformed = M * weights[None, :]
    U, singular, Vt = np.linalg.svd(transformed, full_matrices=False)
    U, singular, V = U[:, :3], singular[:3], Vt[:3].T
    profile = build_word_profile(M, U, V, singular, weights, retained)
    np.testing.assert_allclose(profile.M_hat_retained, M, atol=1e-12)
    np.testing.assert_allclose(
        profile.M_hat_retained.T @ U,
        (V * singular[None, :]) / weights[:, None],
        atol=1e-12,
    )
    recovered = recover_W_from_word_vertices(U, profile.Z_hat[[0, 1, 2]])
    np.testing.assert_allclose(recovered.simplex_projected, W, atol=1e-9)


def test_observed_cross_product_is_explicit_secondary_profile() -> None:
    W, A, _, M = _population_case()
    U, singular, Vt = np.linalg.svd(M, full_matrices=False)
    primary = build_word_profile(
        M, U[:, :3], Vt[:3].T, singular[:3], np.ones(M.shape[1]), np.arange(M.shape[1])
    )
    observed = build_word_profile(
        M,
        U[:, :3],
        Vt[:3].T,
        singular[:3],
        np.ones(M.shape[1]),
        np.arange(M.shape[1]),
        profile_source="observed_cross_product",
    )
    np.testing.assert_allclose(primary.Z_hat, observed.Z_hat, atol=1e-12)
    assert observed.profile_source == "observed_cross_product"


def test_threshold_anchor_survival_and_failure_diagnostic() -> None:
    W_exact, A_exact, _, M_exact = _population_case()
    retained_exact = select_feature_columns(
        M_exact, 1000, method="tran_paper_exact", alpha=0.0, K=3, true_A=A_exact
    ).retained_indices
    assert all(anchor in retained_exact for anchor in [0, 1, 2])
    transformed = M_exact[:, retained_exact]
    U_exact, singular_exact, Vt_exact = np.linalg.svd(transformed, full_matrices=False)
    profile = build_word_profile(
        M_exact,
        U_exact[:, :3],
        Vt_exact[:3].T,
        singular_exact[:3],
        np.ones(len(retained_exact)),
        retained_exact,
    )
    anchor_rows = [int(np.flatnonzero(retained_exact == value)[0]) for value in [0, 1, 2]]
    recovered = recover_W_from_word_vertices(
        U_exact[:, :3], profile.Z_hat[anchor_rows]
    )
    np.testing.assert_allclose(recovered.simplex_projected, W_exact, atol=1e-9)

    W = np.tile(np.array([[0.45, 0.35, 0.20]]), (30, 1))
    A = np.array(
        [
            [0.40, 0.00, 0.00, 0.20, 0.20, 0.20],
            [0.00, 0.40, 0.00, 0.20, 0.20, 0.20],
            [0.00, 0.00, 0.03, 0.32, 0.32, 0.33],
        ]
    )
    X = W @ A
    retained = select_feature_columns(
        X, 100, method="tran_paper_exact", alpha=0.5, K=3, true_A=A
    ).retained_indices
    anchors = [np.array([0]), np.array([1]), np.array([2])]
    survival = [int(np.intersect1d(group, retained).size) for group in anchors]
    assert survival[:2] == [1, 1]
    assert survival[2] == 0
    anchor_survival_failure = any(value == 0 for value in survival)
    assert anchor_survival_failure


@pytest.mark.parametrize(
    "method",
    ["spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"],
)
def test_word_cloud_vertex_hunters_return_original_K_coordinates(method: str) -> None:
    rng = np.random.default_rng(177)
    vertices = np.array([[2.0, -0.5, 0.2], [-1.0, 1.7, 0.3], [0.1, -0.2, 2.2]])
    clouds = []
    for vertex in vertices:
        clouds.append(vertex + rng.normal(scale=0.01, size=(18, 3)))
    cloud = np.vstack(clouds)
    parameters: dict[str, object] = {}
    if method in {"svs", "svs_star"}:
        parameters.update(L_mode="fixed", L=6, max_simplexes=1000)
    if method == "pp_spa":
        parameters.update(radius_divisor=8, m_neighbors=4, min_neighbors=3)
    if method in {"palm", "palm_accelerated"}:
        parameters.update(
            lambda_=1.0,
            max_iterations=8,
            tolerance=1e-10,
            initialize_weights_iterations=30,
            final_weight_refit_iterations=20,
            hull_reduction="none",
        )
    result = vertex_hunt(
        cloud, 3, method, random_state=12, raise_on_failure=True, **parameters
    )
    assert result.vertices.shape == (3, 3)
    assert result.embedding_used.shape[1] == cloud.shape[1]
    if method == "pp_spa":
        assert result.projected_points.shape[1] == cloud.shape[1]
        assert result.pseudo_points.shape[1] == cloud.shape[1]
    if method in {"palm", "palm_accelerated"}:
        assert result.initialization_observation_indices.shape == (3,)
        assert result.observation_weights.shape == (len(cloud), 3)
        assert result.initialization_weights.shape == (len(cloud), 3)
        assert result.objective_trace[-1] <= result.objective_trace[0] + 1e-10
        assert result.parameters["acceleration"] == (
            "none" if method == "palm" else "monotone_restart"
        )
        assert result.parameters["PALM_NMF_source_sha256"]
        assert result.parameters["PALM_accelerated_source_sha256"]


def test_raw_topicscore_matches_exact_local_R_fixture() -> None:
    counts = pd.read_csv(FIXTURE / "counts_document_by_word.csv").to_numpy(float)
    expected_A = pd.read_csv(FIXTURE / "topicscore_A_feature_by_topic.csv").to_numpy(float).T
    expected_W = pd.read_csv(FIXTURE / "topicscore_W_topic_by_document.csv").to_numpy(float).T
    result = fit_topicscore_raw(counts / 60.0, 3)
    left, right = _align(result.A_hat, expected_A)
    np.testing.assert_allclose(result.A_hat[left], expected_A[right], atol=1e-10)
    np.testing.assert_allclose(result.W_hat[:, left], expected_W[:, right], atol=1e-7)


def test_raw_topicscore_mechanically_removes_and_reinserts_zero_words() -> None:
    counts = pd.read_csv(FIXTURE / "counts_document_by_word.csv").to_numpy(float)
    with_zeros = np.insert(counts / 60.0, [2, 17], 0.0, axis=1)
    reference = fit_topicscore_raw(counts / 60.0, 3)
    result = fit_topicscore_raw(with_zeros, 3)
    expected_A = np.insert(reference.A_hat, [2, 17], 0.0, axis=1)
    left, right = _align(result.A_hat, expected_A)
    np.testing.assert_allclose(result.A_hat[left], expected_A[right], atol=1e-10)
    np.testing.assert_allclose(result.W_hat[:, left], reference.W_hat[:, right], atol=1e-8)
    assert result.metadata["zero_frequency_feature_count"] == 2
    np.testing.assert_allclose(result.A_hat[:, [2, 18]], 0.0, atol=0, rtol=0)


def test_graph_topicscore_identity_case_matches_raw_up_to_permutation() -> None:
    W, A, _, M = _population_case()
    raw = fit_topicscore_raw(M, 3)
    U, singular, Vt = np.linalg.svd(M, full_matrices=False)
    graph = fit_topicscore_graph_denoised(
        M, U[:, :3], Vt[:3].T, singular[:3]
    )
    left, right = _align(graph.A_hat, raw.A_hat)
    np.testing.assert_allclose(graph.A_hat[left], raw.A_hat[right], atol=1e-10)
    np.testing.assert_allclose(graph.W_hat[:, left], raw.W_hat[:, right], atol=1e-8)


def test_A_recovery_methods_receive_and_preserve_identical_W() -> None:
    rng = np.random.default_rng(603)
    W = rng.dirichlet(np.ones(3), size=18)
    A = rng.dirichlet(np.ones(12), size=3)
    lengths = np.full(18, 80)
    counts = np.vstack([rng.multinomial(80, W[i] @ A) for i in range(18)])
    before = W.tobytes()
    current = refit_A_current(W, counts / lengths[:, None])
    assert W.tobytes() == before
    poisson = refit_A_full_poisson(W, counts, lengths, max_iter=1000)
    assert W.tobytes() == before
    assert current.A_hat.shape == poisson.A_hat.shape == A.shape
    assert np.isfinite(current.A_hat).all() and np.isfinite(poisson.A_hat).all()
