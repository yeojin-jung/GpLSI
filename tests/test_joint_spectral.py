"""Small algebraic and leakage checks; all production scale work uses Slurm."""
import importlib.util
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist

from gplsi_joint_v2.spectral import (
    SpectralConfig, learn_feature_plan, transformed_frequencies, interpolation_operator,
    _masked_operator, fit_fixed_lambda, fit_shared_spectral, lowrank_difference,
    smooth_fixed_input, graph_penalty, tune_lambda,
)
from gplsi_joint_v2.geometry import exact_diameter, vertex_resource_preflight, GeometryResourceBlocked
from gplsi_joint_v2.diagnostics import embedding_center_diagnostics


def ring(n):
    rows = np.arange(n)
    graph = sparse.csr_matrix((np.ones(n), (rows, (rows + 1) % n)), shape=(n, n))
    return (graph + graph.T).tocsr()


def fold_records(scores):
    return [{"fold": fold, "score": float(score), "converged": True}
            for fold, score in enumerate(scores)]


class SpectralTests(unittest.TestCase):
    def test_frozen_initial_grid_has_31_candidates_and_exact_zero(self):
        grid = SpectralConfig().initial_grid
        self.assertEqual(len(grid), 31)
        self.assertEqual(grid[0], 0.)
        np.testing.assert_allclose(grid[2:], 1e-4 * 1.7**np.arange(29))

    def test_training_only_hvgs_and_weights(self):
        rng = np.random.default_rng(10)
        D = sparse.csr_matrix(rng.poisson(np.arange(1, 13), size=(30, 12)))
        train = np.arange(30) < 20
        changed = D.toarray(); changed[~train] *= 1000; changed[~train, 0] = 10**6
        for preprocessing in ("P0_raw", "P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke"):
            first = learn_feature_plan(D, train, preprocessing, requested_panel=7)
            second = learn_feature_plan(sparse.csr_matrix(changed), train, preprocessing, requested_panel=7)
            np.testing.assert_array_equal(first.panel_indices, second.panel_indices)
            np.testing.assert_array_equal(first.retained_indices, second.retained_indices)
            np.testing.assert_array_equal(first.weights, second.weights)
            np.testing.assert_array_equal(first.eta, second.eta)
        smaller = learn_feature_plan(D, train, "P0_raw", requested_panel=4)
        self.assertTrue(set(smaller.panel_indices).issubset(set(first.panel_indices)))

    def test_inner_masked_operator_has_no_validation_expression(self):
        rng = np.random.default_rng(21)
        D = sparse.csr_matrix(rng.integers(1, 20, size=(25, 9)))
        train = np.arange(25) % 5 != 1
        altered = D.toarray(); altered[~train] = rng.integers(100, 10000, size=(5, 9))
        first = learn_feature_plan(D, train, "P2_ke_weighted", requested_panel=7)
        second = learn_feature_plan(altered, train, "P2_ke_weighted", requested_panel=7)
        T = interpolation_operator(ring(25), train)
        op1 = _masked_operator(transformed_frequencies(D, first), T)
        op2 = _masked_operator(transformed_frequencies(altered, second), T)
        V = rng.normal(size=(7, 3)); U = rng.normal(size=(25, 3))
        np.testing.assert_allclose(op1 @ V, op2 @ V, atol=1e-14)
        np.testing.assert_allclose(op1.T @ U, op2.T @ U, atol=1e-14)

    def test_lowrank_difference_matches_dense(self):
        rng = np.random.default_rng(12)
        U = np.linalg.qr(rng.normal(size=(19, 3)))[0]
        V = np.linalg.qr(rng.normal(size=(7, 3)))[0]
        U2 = np.linalg.qr(rng.normal(size=(19, 3)))[0]
        V2 = np.linalg.qr(rng.normal(size=(7, 3)))[0]
        s, s2 = np.array([3., 2., 1.]), np.array([4., 2., .5])
        self.assertAlmostEqual(lowrank_difference(U, s, V, U2, s2, V2),
                               np.linalg.norm((U * s) @ V.T - (U2 * s2) @ V2.T), places=12)

    def test_exact_zero_identity_and_svd_parity(self):
        rng = np.random.default_rng(22)
        X = sparse.csr_matrix(rng.gamma(1., size=(30, 8)))
        graph = ring(30)
        Y = rng.normal(size=(30, 3))
        identity, metadata = smooth_fixed_input(Y, graph, 0.)
        np.testing.assert_array_equal(Y, identity)
        self.assertEqual(metadata["termination"], "exact_zero_identity")
        config = replace(SpectralConfig(), reconstruction_tolerance=1e-7, max_iterations=400)
        fitted = fit_fixed_lambda(X, graph, 3, 0., seed=13, config=config,
                                  correction=np.asarray(X.sum(axis=0)).ravel() / 20.)
        U, s, Vt = np.linalg.svd(X.toarray(), full_matrices=False)
        expected = (U[:, :3] * s[:3]) @ Vt[:3]
        np.testing.assert_allclose((fitted.U * fitted.singular_values) @ fitted.V.T, expected, atol=1e-5)
        self.assertTrue(fitted.converged)

    def test_final_two_folds_change_selected_lambda(self):
        config = replace(SpectralConfig(), initial_grid=(0., 1., 2.), max_candidates=3)
        def first(grid):
            curves = {0.: [0, 0, 0, 0, 0], 1.: [1, 1, 1, 0, 0], 2.: [10]*5}
            return {lam: fold_records(curves[lam]) for lam in grid}
        def second(grid):
            curves = {0.: [0, 0, 0, 10, 10], 1.: [1, 1, 1, 0, 0], 2.: [10]*5}
            return {lam: fold_records(curves[lam]) for lam in grid}
        self.assertEqual(tune_lambda(first, config)["selected_lambda"], 0.)
        self.assertEqual(tune_lambda(second, config)["selected_lambda"], 1.)
        self.assertEqual(tune_lambda(second, config)["scoring_folds"], list(range(5)))

    def test_grid_expands_and_capped_endpoint_is_flagged(self):
        config = replace(SpectralConfig(), initial_grid=(0., 1.), grid_growth=2., max_candidates=6,
                         extension_batch_size=2, plateau_points=2, plateau_relative_tolerance=0.)
        def interior(grid):
            return {lam: fold_records([(lam - 4)**2 + 1]*5) for lam in grid}
        result = tune_lambda(interior, config)
        self.assertEqual(result["selected_lambda"], 4.)
        self.assertTrue(result["expansion_history"])
        self.assertFalse(result["boundary_unresolved"])
        capped = tune_lambda(lambda grid: {lam: fold_records([1/(lam+1)]*5) for lam in grid}, config)
        self.assertTrue(capped["boundary_unresolved"])
        self.assertEqual(len(capped["grid"]), 6)

    def test_candidate_requires_all_five_converged_folds(self):
        config = replace(SpectralConfig(), initial_grid=(0., 1., 2.), max_candidates=3)
        def evaluate(grid):
            result = {lam: fold_records([5 if lam == 0 else 0]*5) for lam in grid}
            result[1.] = result[1.][:4]
            result[2.][4]["converged"] = False
            return result
        result = tune_lambda(evaluate, config)
        self.assertEqual(result["selected_lambda"], 0.)
        self.assertEqual(result["candidate_incomplete_count"], 2)

    def test_sparse_shared_fit_never_converts_count_matrix_to_dense(self):
        rng = np.random.default_rng(9)
        D = sparse.csr_matrix(rng.poisson(5., size=(30, 8)))
        config = replace(SpectralConfig(), initial_grid=(0.,), max_candidates=1,
                         initialization="direct_svd", reconstruction_tolerance=1e-6)
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("count densification")):
            fitted = fit_shared_spectral(D, ring(30), 2, "P0_raw", graph_cv_seed=2,
                                          estimator_seed=3, config=config, fold_ids=np.arange(30) % 5)
        self.assertTrue(fitted.selected.converged)
        self.assertIs(fitted.selected, fitted.zero)
        self.assertEqual(len(fitted.metadata["cv"]["aggregate"][0]["fold_scores"]), 5)

    def test_deliberate_nonconvergence_is_not_success(self):
        rng = np.random.default_rng(9)
        X = sparse.csr_matrix(rng.random((25, 8)))
        fitted = fit_fixed_lambda(X, ring(25), 2, 0., seed=1,
                                  config=replace(SpectralConfig(), max_iterations=1))
        self.assertFalse(fitted.converged)

    def test_failed_cv_preserves_separately_identified_zero_control(self):
        rng = np.random.default_rng(9)
        counts = sparse.csr_matrix(rng.poisson(8., size=(30, 8)))
        result = fit_shared_spectral(counts, ring(30), 2, "P0_raw", graph_cv_seed=2,
                                     estimator_seed=3, fold_ids=np.arange(30) % 5,
                                     config=replace(SpectralConfig(), initial_grid=(0.,),
                                                    max_candidates=1, max_iterations=1))
        self.assertIsNone(result.selected)
        self.assertEqual(result.zero.lambda_value, 0.)
        self.assertEqual(result.metadata["selection_status"], "no_complete_converged_candidate")
        self.assertFalse(result.metadata["selected_converged"])

    @unittest.skipUnless(importlib.util.find_spec("pycvxcluster"), "requires pinned Midway SSNAL environment")
    def test_fixed_input_convex_smoother_penalty_and_source_parity(self):
        from pycvxcluster.pycvxcluster import SSNAL
        rng = np.random.default_rng(17)
        Y = rng.normal(size=(15, 3)); graph = ring(15)
        config = replace(SpectralConfig(), ssnal_tolerance=1e-7)
        low, lm = smooth_fixed_input(Y, graph, .1, config)
        high, hm = smooth_fixed_input(Y, graph, 1., config)
        source = SSNAL(gamma=.1, maxiter=config.ssnal_max_iterations,
                       admm_iter=config.ssnal_admm_iterations, stoptol=config.ssnal_tolerance, verbose=0)
        source.fit(Y, weight_matrix=graph, save_centers=True, save_labels=False)
        np.testing.assert_allclose(low, source.centers_.T, atol=1e-10)
        self.assertTrue(lm["converged"] and hm["converged"])
        self.assertLessEqual(graph_penalty(high, graph), graph_penalty(low, graph) + 1e-5)

    def test_exact_diameter_is_pairwise_exact_and_budget_is_explicit(self):
        rng = np.random.default_rng(100)
        X = rng.normal(size=(120, 5))
        diameter, metadata = exact_diameter(X)
        self.assertAlmostEqual(diameter, float(pdist(X).max()), places=12)
        self.assertTrue(metadata["exact"])
        with self.assertRaises(GeometryResourceBlocked):
            exact_diameter(X, max_pair_comparisons=1)

    def test_exact_svs_face_complexity_preflight(self):
        expected = {7: 9144, 10: 135036, 12: 745290, 15: 8912624, 20: 484441650}
        for K, face_solves in expected.items():
            record = vertex_resource_preflight(K, 500000, "svs")
            self.assertEqual(record["exact_active_face_solves"], face_solves)
            self.assertEqual(record["status"], "resource_blocked" if K >= 15 else "feasible_preflight")

    def test_omitted_center_roundtrip_preserves_exact_diagnostic_scalars(self):
        from gplsi_joint_v2.spectral import Embedding, graph_roughness
        rng = np.random.default_rng(44)
        U = rng.normal(size=(30, 3))
        graph = ring(30)
        penalty = graph_penalty(U, graph)
        roughness = graph_roughness(U, graph)
        smoother = {"graph_penalty": penalty, "objective": 1.23}
        metadata = {"history": [{"smoother": smoother}],
                    "unorthogonalized_embedding_roughness": roughness}
        full = Embedding(U, np.eye(3), np.ones(3), U.copy(), .5, True, 2, metadata)
        compact = Embedding(U, np.eye(3), np.ones(3), None, .5, True, 2,
                            json.loads(json.dumps(metadata)))
        self.assertEqual(embedding_center_diagnostics(full, graph),
                         embedding_center_diagnostics(compact, graph))
        # Also support a compact final_smoother record with no full history.
        compact.metadata["final_smoother"] = compact.metadata.pop("history")[-1]["smoother"]
        self.assertEqual(embedding_center_diagnostics(full, graph),
                         embedding_center_diagnostics(compact, graph))
        compact.metadata.pop("unorthogonalized_embedding_roughness")
        with self.assertRaisesRegex(ValueError, "exact saved"):
            embedding_center_diagnostics(compact, graph)

    @unittest.skipUnless(importlib.util.find_spec("pycvxcluster"), "requires pinned GpLSI environment")
    def test_bounded_pp_spa_matches_source(self):
        from gplsi.vertex_hunting import vertex_hunt
        from gplsi_joint_v2.geometry import _pp_spa_bounded
        rng = np.random.default_rng(5)
        centers = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        points = np.vstack([center + rng.normal(scale=.005, size=(25, 3)) for center in centers])
        params = {"radius_divisor": 20., "m_neighbors": 4, "min_neighbors": 3}
        original = vertex_hunt(points, 3, "pp_spa", raise_on_failure=True, **params)
        bounded, _ = _pp_spa_bounded(points, 3, params, 1_000_000)
        np.testing.assert_allclose(bounded.vertices, original.vertices, atol=1e-13)
        np.testing.assert_array_equal(bounded.neighborhood_sizes, original.neighborhood_sizes)
        np.testing.assert_allclose(bounded.pseudo_points, original.pseudo_points, atol=1e-13)

    @unittest.skipUnless(importlib.util.find_spec("pycvxcluster"), "requires pinned metric environment")
    def test_lambda_diagnostics_reuse_and_resource_failure_are_explicit(self):
        from types import SimpleNamespace
        from gplsi_joint_v2.diagnostics import run_lambda_diagnostics
        from gplsi_joint_v2.spectral import Embedding
        from gplsi_joint_v2.geometry import GeometryResult
        rng = np.random.default_rng(8)
        counts = sparse.csr_matrix(rng.poisson(5., size=(30, 7)))
        W = rng.dirichlet([1., 1.], size=30)
        U = np.linalg.qr(W)[0]
        V = np.linalg.qr(rng.normal(size=(7, 2)))[0]
        block = Embedding(U, V, np.ones(2), U, 0., True, 2,
                          {"unregularized_initial_embedding_roughness": .5,
                           "history": [{"smoother": {"objective": 0.}}], "convergence_status": "converged"})
        def fake_geometry(*args, hunter, **kwargs):
            if hunter == "svs":
                raise GeometryResourceBlocked("frozen test budget", {"work": 10})
            return GeometryResult(W, np.eye(2), W, {"runtime_seconds": .1})
        callbacks = []
        coordinates = np.c_[np.cos(np.arange(30)), np.sin(np.arange(30))]
        with patch("gplsi_joint_v2.diagnostics.fit_geometry", side_effect=fake_geometry):
            result = run_lambda_diagnostics(counts, ring(30), coordinates, np.repeat("section", 30),
                                            2, "P0_raw", lambdas=[0.], estimator_seed=4,
                                            hunters=("spa_current", "svs"), reused_embeddings={0.: block},
                                            record_callback=callbacks.append)
        self.assertTrue(result["spectral_path"][0]["reused_shared_embedding"])
        self.assertEqual(result["geometry_path"][1]["status"], "resource_blocked")
        self.assertEqual(len(callbacks), 3)
        self.assertFalse(result["used_for_lambda_selection"])
        self.assertNotIn("W", result["geometry_path"][0])


if __name__ == "__main__":
    unittest.main()
