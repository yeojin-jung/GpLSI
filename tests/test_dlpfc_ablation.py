import numpy as np
from scipy.sparse import csr_matrix

from gplsi.real_experiment import GeometryFit, recover_A_for_geometry
from gplsi.recovery import refit_A_current
from gplsi_spatial_benchmark.panels import panel_indices, rank_features_by_dispersion
from gplsi_spatial_benchmark.splits import thin_and_split_sparse_counts


def _counts(seed=0, n=60, p=40):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(rng.gamma(0.5, 2.0, size=(n, p)))
    dense[:, -3:] = 0  # undetected genes
    return dense


def test_sparse_split_conserves_molecules_and_nests_retention():
    counts = csr_matrix(_counts())
    splits = {r: thin_and_split_sparse_counts(counts, retained_fraction=r, test_fraction=0.2, seed=7)
              for r in (1.0, 0.5, 0.25)}
    full = splits[1.0]
    assert np.array_equal((full.train + full.test).toarray(), counts.toarray())
    retained = {r: (s.train + s.test).toarray() for r, s in splits.items()}
    assert np.all(retained[0.25] <= retained[0.5]) and np.all(retained[0.5] <= retained[1.0])
    again = thin_and_split_sparse_counts(counts, retained_fraction=1.0, test_fraction=0.2, seed=7)
    assert np.array_equal(again.test.toarray(), full.test.toarray())


def test_panels_are_training_only_nested_prefixes():
    counts = _counts()
    ranking = rank_features_by_dispersion(csr_matrix(counts), detection_fraction=0.05)
    dense_ranking = rank_features_by_dispersion(counts, detection_fraction=0.05)
    assert np.array_equal(ranking["ranked_indices"], dense_ranking["ranked_indices"])
    small, large = panel_indices(ranking, 10), panel_indices(ranking, 20)
    assert np.array_equal(large[:10], small)
    assert not set(large) & {37, 38, 39}
    ratio = counts.var(axis=0) / counts.mean(axis=0).clip(1e-12)
    assert np.all(np.diff(ratio[large]) <= 1e-12)


def test_A_full_L2_uses_same_W_and_returns_simplex_rows():
    rng = np.random.default_rng(3)
    W = rng.dirichlet(np.ones(3), size=50)
    A_true = rng.dirichlet(np.ones(12), size=3)
    counts = np.vstack([rng.multinomial(400, row) for row in W @ A_true])
    lengths = counts.sum(axis=1).astype(float)

    class Bundle:
        frequencies = counts / lengths[:, None]
        document_lengths = lengths

    Bundle.counts = counts
    geometry = GeometryFit("document_gplsi", "document_U", "spa_current", W, np.eye(3), None, None,
                           None, None, None, {})
    warm = refit_A_current(W, Bundle.frequencies).A_hat
    result, _ = recover_A_for_geometry(Bundle, geometry, "A_full_L2", poisson_initial_A=warm)
    assert result.method == "A_full_L2"
    assert np.allclose(result.A_hat.sum(axis=1), 1) and np.all(result.A_hat >= -1e-12)
    residual = lambda A: np.sum((W @ A - Bundle.frequencies) ** 2)  # noqa: E731
    assert residual(result.A_hat) <= residual(warm) + 1e-10
    assert np.abs(result.A_hat - A_true).max() < 0.1


def test_vectorized_row_simplex_projection_matches_rowwise_projection():
    from gplsi.recovery import project_rows_simplex, project_simplex

    rng = np.random.default_rng(0)
    dirichlet = rng.dirichlet(np.ones(7), size=100)
    cases = [rng.normal(size=(200, 7)) * scale for scale in (1e-3, 1.0, 1e6)]
    cases += [dirichlet, np.vstack([dirichlet, -np.ones((3, 7)), np.zeros((2, 7)), np.eye(7)]),
              rng.normal(size=(20, 1))]
    for matrix in cases:
        expected = np.vstack([project_simplex(row) for row in matrix])
        np.testing.assert_array_equal(project_rows_simplex(matrix), expected)


def _reference_full_l2(W, X, A, max_iter=2000, tolerance=1e-8):
    """The original residual-based projected-gradient loop, kept as an oracle."""
    from gplsi.recovery import project_rows_simplex

    step = 1.0 / (2.0 * np.linalg.norm(W, 2) ** 2)
    objective = lambda V: float(np.sum((W @ V - X) ** 2))  # noqa: E731
    value = objective(A)
    for _ in range(max_iter):
        gradient = 2.0 * W.T @ (W @ A - X)
        local = step
        candidate = project_rows_simplex(A - local * gradient)
        while objective(candidate) > value + 1e-12 and local > 1e-16:
            local *= 0.5
            candidate = project_rows_simplex(A - local * gradient)
        change = np.linalg.norm(candidate - A) / max(np.linalg.norm(A), 1.0)
        A, value = candidate, objective(candidate)
        if change <= tolerance:
            break
    return A


def test_gram_form_full_l2_matches_residual_form():
    from gplsi.recovery import refit_A_current, refit_A_full_l2

    rng = np.random.default_rng(11)
    for n, K, p in [(80, 3, 25), (300, 6, 120)]:
        W = rng.dirichlet(np.ones(K) * 0.5, size=n)
        X = rng.dirichlet(np.ones(p) * 0.3, size=n)
        start = refit_A_current(W, X).A_hat
        fast = refit_A_full_l2(W, X, initial_A=start)
        np.testing.assert_allclose(fast.A_hat, _reference_full_l2(W, X, start), atol=1e-9)
        residual = np.sum((W @ fast.A_hat - X) ** 2)
        np.testing.assert_allclose(fast.objective_history[-1], residual, rtol=1e-10)


def _load_script(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "visium_dlpfc" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dlpfc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_poisson_refit_takes_W_from_any_recovery_of_a_geometry():
    geometry_sources = _load_script("refit_poisson").geometry_sources
    results = [
        {"method": "gplsi_document__P0_raw__spa_current__A_current"},
        {"method": "gplsi_document__P0_raw__spa_current__A_full_L2"},
        {"method": "gplsi_anchor__P0_raw__spa_current__A_current"},  # failed: no W saved
        {"method": "gplsi_anchor__P0_raw__spa_current__A_full_L2"},
        {"method": "gplsi_document__P2_ke_weighted__palm__A_current"},  # failed at geometry: no W at all
        {"method": "gplsi_document__P2_ke_weighted__palm__A_full_L2"},
        {"method": "topicscore_raw"},
    ]
    saved = {"W_0", "A_0", "W_1", "A_1", "W_3", "A_3", "W_6", "A_6"}
    assert geometry_sources(results, saved) == [
        ("gplsi_document__P0_raw__spa_current", 0, 0),
        ("gplsi_anchor__P0_raw__spa_current", 3, 2),
        ("gplsi_document__P2_ke_weighted__palm", None, 4),
    ]
