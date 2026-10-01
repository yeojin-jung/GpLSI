import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import coo_matrix

from gplsi.recovery import refit_A_full_poisson
from gplsi.pipeline.metrics import heldout_count_metrics


def _oracle_fixture():
    W = np.array(
        [
            [.9, .1],
            [.75, .25],
            [.6, .4],
            [.4, .6],
            [.25, .75],
            [.1, .9],
        ]
    )
    counts = np.array(
        [
            [18, 5, 3, 4],
            [12, 7, 5, 6],
            [10, 8, 7, 5],
            [6, 7, 10, 7],
            [4, 5, 12, 9],
            [3, 4, 8, 15],
        ],
        dtype=float,
    )
    return W, counts


def _slsqp_oracle(W, counts):
    K = W.shape[1]
    p = counts.shape[1]
    constraint = np.zeros((K, K * p))
    for topic in range(K):
        constraint[topic, topic * p : (topic + 1) * p] = 1.0

    def objective(flat):
        A = flat.reshape(K, p)
        return float(-np.sum(counts * np.log(W @ A)))

    def gradient(flat):
        A = flat.reshape(K, p)
        return (-(W.T @ (counts / (W @ A)))).reshape(-1)

    fit = minimize(
        objective,
        np.full(K * p, 1.0 / p),
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(1e-12, 1.0),
        constraints=LinearConstraint(constraint, np.ones(K), np.ones(K)),
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    assert fit.success, fit.message
    return fit.fun, fit.x.reshape(K, p)


def test_sparse_em_matches_independent_slsqp_oracle():
    W, counts = _oracle_fixture()
    oracle_value, oracle_A = _slsqp_oracle(W, counts)
    fit = refit_A_full_poisson(
        W,
        counts,
        counts.sum(axis=1),
        initial_A=np.array([[0, .5, .5, 0], [.5, 0, 0, .5]]),
        max_iter=10_000,
        tolerance=1e-8,
        chunk_size=3,
    )
    fitted_nll = fit.objective_history[-1] - fit.diagnostics["poisson_linear_term"]
    assert fit.converged
    assert fit.status == "ok"
    assert fit.solver == "sparse_monotone_em"
    assert abs(fitted_nll - oracle_value) / counts.sum() <= 5e-8
    assert np.max(np.abs(W @ fit.A_hat - W @ oracle_A)) <= 2e-6
    assert fit.normalized_optimality_gap <= 1.01e-8
    assert np.allclose(fit.A_hat.sum(axis=1), 1.0, atol=2e-13)
    assert fit.A_hat[0, 0] > 1e-3
    assert fit.A_hat[0, 3] > 1e-3
    assert fit.A_hat[1, 1] > 1e-3
    assert fit.A_hat[1, 2] > 1e-3


def test_sparse_em_recovers_closed_form_and_unsupported_feature():
    W = np.ones((4, 1))
    counts = np.array([[4, 0, 1, 0], [0, 3, 2, 0], [1, 1, 0, 0], [5, 0, 0, 0]])
    fit = refit_A_full_poisson(
        W,
        counts,
        counts.sum(axis=1),
        initial_A=np.array([[0, 0, 1, 0]], dtype=float),
        max_iter=10,
        tolerance=1e-12,
    )
    expected = np.array([[10, 4, 3, 0]], dtype=float) / 17
    assert fit.converged
    assert fit.iterations <= 2
    assert np.allclose(fit.A_hat, expected, atol=2e-14, rtol=0)
    assert fit.A_hat[0, -1] == 0
    fallback = refit_A_full_poisson(
        W,
        coo_matrix(counts),
        counts.sum(axis=1),
        max_iter=10,
        tolerance=1e-12,
    )
    assert fallback.converged
    assert fallback.diagnostics["initializer_source"] == "pooled_feature_frequencies"
    assert np.allclose(fallback.A_hat, expected, atol=2e-14, rtol=0)


def test_sparse_em_recovers_one_hot_topic_profiles():
    W = np.repeat(np.eye(3), 2, axis=0)
    counts = np.array(
        [
            [4, 1, 0, 0, 0],
            [1, 3, 1, 0, 0],
            [0, 1, 4, 0, 0],
            [0, 0, 1, 3, 1],
            [0, 0, 0, 2, 3],
            [1, 0, 0, 1, 3],
        ]
    )
    expected = np.array([[5, 4, 1, 0, 0], [0, 1, 5, 3, 1], [1, 0, 0, 3, 6]]) / 10
    fit = refit_A_full_poisson(
        W,
        counts,
        counts.sum(axis=1),
        initial_A=np.full((3, 5), .2),
        max_iter=10,
        tolerance=1e-12,
    )
    assert fit.converged
    assert fit.iterations <= 2
    assert np.allclose(fit.A_hat, expected, atol=2e-14, rtol=0)


def test_nonconvergence_is_not_reported_as_ok_and_history_is_monotone():
    W, counts = _oracle_fixture()
    fit = refit_A_full_poisson(
        W,
        counts * 1_000_000,
        counts.sum(axis=1) * 1_000_000,
        initial_A=np.full((2, 4), .25),
        max_iter=1,
        tolerance=1e-14,
    )
    assert not fit.converged
    assert fit.status == "max_iter_reached"
    assert fit.iterations == 1
    assert fit.normalized_optimality_gap > 1e-14
    for earlier, later in zip(fit.objective_history, fit.objective_history[1:]):
        slack = 128 * np.finfo(float).eps * max(1.0, abs(earlier))
        assert later - earlier <= slack


def test_sparse_duplicates_and_chunk_sizes_match_dense_path():
    W, counts = _oracle_fixture()
    rows, columns = np.nonzero(counts)
    values = counts[rows, columns]
    duplicate = coo_matrix(
        (
            np.repeat(values / 2, 2),
            (np.repeat(rows, 2), np.repeat(columns, 2)),
        ),
        shape=counts.shape,
    )
    kwargs = dict(
        document_lengths=counts.sum(axis=1),
        initial_A=np.full((2, 4), .25),
        max_iter=10_000,
        tolerance=1e-8,
    )
    dense_fit = refit_A_full_poisson(W, counts, chunk_size=2, **kwargs)
    sparse_fit = refit_A_full_poisson(W, duplicate, chunk_size=10_000, **kwargs)
    assert dense_fit.converged and sparse_fit.converged
    assert np.allclose(W @ dense_fit.A_hat, W @ sparse_fit.A_hat, atol=2e-12, rtol=2e-12)


def test_map_has_distinct_label_and_closed_form():
    W = np.ones((4, 1))
    counts = np.array([[4, 0, 1, 0], [0, 3, 2, 0], [1, 1, 0, 0], [5, 0, 0, 0]])
    fit = refit_A_full_poisson(
        W,
        counts,
        counts.sum(axis=1),
        initial_A=np.full((1, 4), .25),
        max_iter=10,
        tolerance=1e-12,
        prior_strength=2.0,
        prior_base=np.full(4, .25),
    )
    expected = np.array([[10.5, 4.5, 3.5, .5]]) / 19
    assert fit.converged
    assert fit.method == "A_full_Pois_MAP"
    assert fit.diagnostics["prior_strength"] == 2.0
    assert np.allclose(fit.A_hat, expected, atol=2e-14, rtol=0)


def test_inactive_W_topic_is_preserved_instead_of_failing_the_fit():
    W = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
    counts = np.array([[3, 1, 0], [0, 2, 2], [1, 0, 3]], dtype=float)
    initial = np.array([[.2, .3, .5], [.7, .2, .1]])
    fit = refit_A_full_poisson(
        W,
        counts,
        counts.sum(axis=1),
        initial_A=initial,
        max_iter=10,
        tolerance=1e-12,
    )
    assert not fit.converged
    assert fit.status == "unidentified_inactive_topics"
    assert fit.diagnostics["identified_subproblem_converged"]
    assert fit.diagnostics["inactive_topic_indices"] == [1]
    expected_inactive = (1 - 1e-6) * initial[1] + 1e-6 / 3
    assert np.allclose(fit.A_hat[1], expected_inactive, atol=2e-14, rtol=0)
    assert "inactive_W_topics_preserved" in fit.warnings


def test_heldout_metrics_report_impossible_events_without_silent_flooring():
    W = np.array([[1, 0], [0, 1], [.5, .5]], dtype=float)
    A = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    test = np.array([[2, 0, 1], [0, 3, 2], [1, 1, 4]], dtype=float)
    metrics = heldout_count_metrics(W, A, test)
    assert metrics["heldout_status"] == "positive_test_count_has_zero_probability"
    assert metrics["heldout_zero_probability_positive_entries"] == 3
    assert metrics["heldout_zero_probability_molecules"] == 7
    assert metrics["heldout_zero_probability_rows"] == 3
    assert metrics["heldout_molecules"] == 14
    assert metrics["heldout_log_likelihood_without_constant"] == float("-inf")
    assert metrics["heldout_poisson_deviance"] == float("inf")
    assert np.isfinite(metrics["heldout_poisson_deviance_floored_1e-12"])


def test_heldout_metrics_remain_exact_when_all_events_have_support():
    W = np.array([[1, 0], [0, 1], [.5, .5]], dtype=float)
    A = np.array([[.9, .05, .05], [.05, .9, .05]], dtype=float)
    test = np.array([[2, 0, 1], [0, 3, 2], [1, 1, 4]], dtype=float)
    metrics = heldout_count_metrics(W, A, test)
    assert metrics["heldout_status"] == "ok"
    assert metrics["heldout_zero_probability_molecules"] == 0
    assert np.isclose(
        metrics["heldout_log_likelihood_without_constant"],
        -22.98580944306206,
        atol=1e-12,
    )
    assert np.isclose(metrics["heldout_poisson_deviance"], 25.011658464485144, atol=1e-12)
