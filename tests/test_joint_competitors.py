import numpy as np
import pytest
from scipy import sparse

from gplsi_joint_v2.competitors import (_ratio_nnz, fit_lda, fit_kl_nmf,
                                       fit_graph_kl_nmf, fit_topicscore)


def counts_fixture():
    rng = np.random.default_rng(751)
    w = rng.dirichlet([.7, .7], size=24)
    a = np.array([[.65, .2, .1, .04, .01], [.01, .04, .1, .2, .65]])
    return np.vstack([rng.multinomial(100, row) for row in w @ a])


def test_sparse_kl_nnz_ratio_and_objective_match_dense_independent_formula():
    d = np.array([[4, 0, 1], [0, 3, 4], [1, 2, 0]], float)
    u = np.array([[1, 2], [3, 1], [1, 1]], float)
    h = np.array([[1, 2, 3], [3, 2, 1]], float)
    ratio, nll = _ratio_nnz(sparse.csr_matrix(d), u, h, chunk_size=2)
    mu = u @ h
    assert np.allclose(ratio.toarray(), d / mu)
    assert np.isclose(nll, np.sum(mu - d * np.log(mu)))


def test_graph_kl_sparse_full_objective_is_monotone_and_returns_compositions():
    d = sparse.csr_matrix(counts_fixture())
    graph = sparse.diags([np.ones(23), np.ones(23)], [-1, 1], shape=(24, 24)).tocsr()
    result = fit_graph_kl_nmf(d, graph, 2, 54, max_iter=30, tolerance=1e-10, chunk_size=5)
    assert np.all(np.diff(result.metadata["objective_history"]) <= 1e-6)
    assert np.allclose(result.W.sum(axis=1), 1)
    assert np.allclose(result.A.sum(axis=1), 1)
    assert result.metadata["joint_profiles"]


def test_joint_lda_and_kl_nmf_share_one_profile_matrix():
    d = sparse.csr_matrix(counts_fixture())
    for result in [fit_lda(d, 2, 77, max_iter=3), fit_kl_nmf(d, 2, 77, max_iter=5)]:
        assert result.W.shape == (24, 2)
        assert result.A.shape == (2, 5)
        assert np.allclose(result.W.sum(axis=1), 1)
        assert np.allclose(result.A.sum(axis=1), 1)
        assert result.metadata["joint_profiles"]


def test_sparse_topicscore_matches_legacy_dense_composition_up_to_topic_order():
    from gplsi.topicscore import fit_topicscore_raw
    d = counts_fixture()
    x = d / d.sum(axis=1)[:, None]
    old = fit_topicscore_raw(x, 2)
    new = fit_topicscore(sparse.csr_matrix(d), 2, 139)
    assert np.allclose(new.W @ new.A, old.W_hat @ old.A_hat, atol=2e-6)


def test_sparse_spectral_competitors_do_not_densify_counts(monkeypatch):
    original = sparse.csr_matrix.toarray
    def no_cohort_dense(self, *args, **kwargs):
        if self.shape == (24, 5):
            raise AssertionError("entire sparse cohort densified")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", no_cohort_dense)
    d = sparse.csr_matrix(counts_fixture())
    graph = sparse.diags([np.ones(23), np.ones(23)], [-1, 1], shape=(24, 24)).tocsr()
    fit_topicscore(d, 2, 139, max_iter=10)
    fit_graph_kl_nmf(d, graph, 2, 139, max_iter=3)
