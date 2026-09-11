import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from gplsi.recovery import RecoveryError, prepare_poisson_counts, _poisson_em_statistics
from gplsi_joint_v2.likelihood import (recover_A_current, recover_A_poisson,
                                      fold_in_fixed_A, normalize_nmf_composition)
from gplsi_joint_v2.metrics import score_counts


def fixture():
    w = np.array([[.9, .1], [.75, .25], [.6, .4], [.4, .6], [.25, .75], [.1, .9]])
    d = np.array([[18, 5, 3, 4], [12, 7, 5, 6], [10, 8, 7, 5],
                  [6, 7, 10, 7], [4, 5, 12, 9], [3, 4, 8, 15]])
    return w, d


def test_current_qr_matches_historical_full_rank_estimator_without_ridge():
    from gplsi.recovery import refit_A_current
    w, d = fixture()
    old = refit_A_current(w, d / d.sum(axis=1)[:, None])
    fit = recover_A_current(w, sparse.csr_matrix(d))
    assert np.allclose(fit.A_hat, old.A_hat, atol=5e-14)
    assert fit.method == "A_current"
    assert fit.diagnostics["ridge"] == 0
    assert fit.diagnostics["rank"] == 2
    assert fit.diagnostics["is_constrained_LS_optimum"] is False


def test_current_reports_rank_failure_without_replacing_estimator():
    w, d = fixture()
    w[:] = .5
    with pytest.raises(RecoveryError, match="rank-deficient"):
        recover_A_current(w, d)


def test_repair_wrapper_retained_and_no_worse_than_feasible_current():
    w, d = fixture()
    before = w.copy()
    current = recover_A_current(w, d)
    fit = recover_A_poisson(w, sparse.csr_matrix(d), initial_A=current.A_hat,
                           max_iter=10000, tolerance=1e-8, chunk_size=5)
    assert fit.converged
    assert np.array_equal(w, before)
    assert fit.diagnostics["source_repair_commit"].startswith("9a23a18")
    assert fit.diagnostics["final_objective_no_worse_than_original_initial"]
    assert fit.normalized_optimality_gap <= 1e-8
    assert np.all(np.diff(fit.objective_history) <= 1e-9)
    assert np.allclose(fit.A_hat.sum(axis=1), 1)


def test_exact_sparse_poisson_gradient_matches_finite_difference():
    w, d = fixture()
    a = np.array([[.45, .25, .2, .1], [.1, .2, .25, .45]])
    prepared = prepare_poisson_counts(sparse.csr_matrix(d))
    score, ll, _ = _poisson_em_statistics(a, w, prepared, chunk_size=7)
    lengths = d.sum(axis=1)
    analytic = w.T @ lengths[:, None] - score
    def objective(value):
        p = w @ value
        return np.sum(lengths[:, None] * p - d * np.log(p))
    numerical = np.zeros_like(a)
    for index in np.ndindex(a.shape):
        plus, minus = a.copy(), a.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (objective(plus) - objective(minus)) / 2e-6
    assert np.allclose(analytic, numerical, atol=2e-7, rtol=2e-7)
    assert np.isclose(ll, np.sum(d * np.log(w @ a)))


def test_raw_contract_rejects_fractional_counts_and_invalid_simplex():
    w, d = fixture()
    with pytest.raises(ValueError, match="integer"):
        recover_A_poisson(w, d + .1)
    with pytest.raises(ValueError, match="sum to one"):
        recover_A_poisson(2 * w, d)
    d[0] = 0
    with pytest.raises(RecoveryError, match="positive-count"):
        recover_A_poisson(w, d)


def test_near_boundary_sparse_solution_and_deliberate_nonconvergence():
    w, d = fixture()
    nonconverged = recover_A_poisson(w, d, max_iter=0, tolerance=1e-14)
    assert not nonconverged.converged
    assert nonconverged.status == "max_iter_reached"
    counts = np.array([[1000000, 1, 0], [1000000, 0, 0]])
    closed = recover_A_poisson(np.ones((2, 1)), counts, tolerance=1e-12)
    assert closed.converged
    assert closed.A_hat[0, 2] == 0
    assert np.isclose(closed.A_hat[0, 1], 1 / 2000001)


def test_fold_in_matches_independent_constrained_optimizer_and_chunk_parity():
    a = np.array([[.7, .2, .1], [.1, .2, .7]])
    d = np.array([[7, 3, 9], [9, 5, 7], [1, 1, 12]])
    fits = [fold_in_fixed_A(a, x, tolerance=1e-9, max_iter=10000,
                            row_chunk_size=chunk, entry_chunk_size=2)
            for x, chunk in [(d, 1), (sparse.csr_matrix(d), 100)]]
    for fit in fits:
        assert fit.row_converged.all()
        assert np.max(fit.normalized_gap) <= 1e-9
        assert fit.metadata["A_updated"] is False
        for index, row in enumerate(d):
            result = minimize(lambda w: -np.dot(row, np.log(w @ a)), [.5, .5],
                              method="SLSQP", bounds=[(0, 1), (0, 1)],
                              constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                              options={"ftol": 1e-12, "maxiter": 1000})
            assert result.success
            assert abs(-np.dot(row, np.log(fit.W[index] @ a)) - result.fun) < 1e-7
    assert np.allclose(fits[0].W, fits[1].W, atol=1e-12)


def test_fold_in_support_zero_depth_and_nonconvergence_are_explicit():
    a = np.array([[.9, .1, 0], [.1, .9, 0]])
    d = np.array([[9, 2, 0], [1, 2, 3], [0, 0, 0]])
    fit = fold_in_fixed_A(a, d, max_iter=0)
    assert fit.row_status.tolist() == ["max_iter_reached", "unsupported_adaptation_counts", "no_adaptation_counts"]
    assert fit.inference_valid.tolist() == [True, False, False]
    assert np.allclose(fit.W.sum(axis=1), 1)
    score = score_counts(fit.W, a, np.ones((3, 3), int), eligibility_mask=d.sum(axis=1) > 0,
                         inference_valid=fit.inference_valid)
    assert score["summary"]["scored_observations"] == 2
    assert score["summary"]["inference_failed_observations"] == 1
    assert score["per_row"]["status"][1] == "inference_failed"


def test_scoring_count_mutation_cannot_change_fit_or_adaptation():
    w, train = fixture()
    fit = recover_A_poisson(w, train, max_iter=10000)
    adapt = train[:3]
    folded = fold_in_fixed_A(fit.A_hat, adapt)
    original_w, original_a, original_test_w = w.copy(), fit.A_hat.copy(), folded.W.copy()
    score_counts(w, fit.A_hat, train)
    changed_score = train.copy()
    changed_score[:, 0] += 100000
    score_counts(w, fit.A_hat, changed_score)
    score_counts(folded.W, fit.A_hat, changed_score[:3])
    assert np.array_equal(w, original_w)
    assert np.array_equal(fit.A_hat, original_a)
    assert np.array_equal(folded.W, original_test_w)
    repeated = fold_in_fixed_A(fit.A_hat, adapt)
    assert np.array_equal(repeated.W, original_test_w)


def test_nmf_compensated_conversion_preserves_compositions_and_inactive_topics():
    u = np.array([[3, 1, 2], [1, 4, 3]], float)
    h = np.array([[2, 1, 4], [10, 2, 1], [0, 0, 0]], float)
    expected = u @ h
    expected /= expected.sum(axis=1)[:, None]
    w, a, metadata = normalize_nmf_composition(u, h)
    assert np.allclose(w @ a, expected)
    assert np.allclose(w.sum(axis=1), 1) and np.allclose(a.sum(axis=1), 1)
    assert metadata["inactive_topic_indices"] == [2]
    assert np.all(w[:, 2] == 0)
    with pytest.raises(RecoveryError, match="zero depth"):
        normalize_nmf_composition(np.zeros_like(u), h)


def test_sparse_paths_do_not_call_full_toarray(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("full sparse count densification")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    w, d = fixture()
    d = sparse.csr_matrix(d)
    current = recover_A_current(w, d)
    poisson = recover_A_poisson(w, d, initial_A=current.A_hat, max_iter=3)
    fold = fold_in_fixed_A(poisson.A_hat, d, max_iter=3)
    metrics = score_counts(fold.W, poisson.A_hat, d, entry_chunk_size=3)
    assert metrics["summary"]["scored_observations"] == len(w)
