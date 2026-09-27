import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from gplsi.graphSVD import _svds
from gplsi.vertex_hunting import _project_affine
from gplsi_spatial_benchmark.graph import build_within_unit_knn_graph
from gplsi_spatial_benchmark.metrics import external_structure_metrics, topic_profile_metrics
from gplsi_spatial_benchmark.splits import thin_and_split_counts


def test_graph_never_crosses_declared_units():
    xy = np.array([[0, 0], [1, 0], [0, 1], [100, 100], [101, 100], [100, 101]], float)
    units = np.array(["a", "a", "a", "b", "b", "b"])
    graph = build_within_unit_knn_graph(xy, units, k=2)
    edges = graph.edges[["src", "tgt"]].to_numpy(int)
    assert np.all(units[edges[:, 0]] == units[edges[:, 1]])
    assert np.all(edges[:, 0] < edges[:, 1])


def test_seeded_sparse_svd_uses_pinned_scipy_compatibility_path():
    left, values, right = _svds(np.diag(np.arange(1.0, 9.0)), 2, np.random.default_rng(11))
    assert left.shape == (8, 2)
    assert values.shape == (2,)
    assert right.shape == (2, 8)


def test_affine_projection_uses_economy_svd_without_changing_projection(monkeypatch):
    rng = np.random.default_rng(13)
    points = rng.normal(size=(41, 5))
    samples = points.T
    means = samples.mean(axis=1, keepdims=True)
    centered = samples - means
    original_svd = np.linalg.svd
    full_u, _, _ = original_svd(centered, full_matrices=True)
    expected = (means + full_u[:, :4] @ full_u[:, :4].T @ centered).T
    seen = {}

    def checked_svd(matrix, *args, **kwargs):
        seen["full_matrices"] = kwargs.get("full_matrices")
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", checked_svd)
    projected = _project_affine(points, K=5)

    assert seen["full_matrices"] is False
    assert np.allclose(projected, expected, atol=1e-12, rtol=1e-12)


def test_count_split_is_exact_and_reproducible():
    counts = csr_matrix(np.array([[4, 2, 0], [3, 1, 8], [2, 5, 1]], dtype=int))
    first = thin_and_split_counts(counts, retained_fraction=1, test_fraction=.2, seed=7)
    second = thin_and_split_counts(counts, retained_fraction=1, test_fraction=.2, seed=7)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.test, second.test)
    assert np.array_equal(first.train + first.test, counts.toarray())


def test_retention_levels_are_nested_for_a_fixed_seed():
    counts = np.full((5, 4), 20, dtype=int)
    high = thin_and_split_counts(counts, retained_fraction=.75, test_fraction=.2, seed=9)
    low = thin_and_split_counts(counts, retained_fraction=.25, test_fraction=.2, seed=9)
    assert np.all(low.train + low.test <= high.train + high.test)


def test_external_metrics_are_topic_permutation_invariant():
    W = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
    external = pd.DataFrame({"label": ["a", "a", "b", "b"]})
    before = external_structure_metrics(W, external, 4)
    after = external_structure_metrics(W[:, ::-1], external, 4)
    assert before["external__label__nmi"] == after["external__label__nmi"]
    assert before["external__label__ari"] == after["external__label__ari"]


def test_topic_profiles_are_simplex_normalized():
    metrics, genes = topic_profile_metrics(np.array([[2, 1, 0], [0, 1, 4.]]), np.array(["a", "b", "c"]))
    assert len(genes) == 2
    assert metrics["topic_minimum_mass"] == 1.0
