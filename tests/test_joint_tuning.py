"""Competitor tuning selection and exclusion-of-scoring-data contracts."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from gplsi_joint_v2.tuning import select_spatial_penalty, fit_tuned_spatial_competitor, CompetitorTuningError
from gplsi_joint_v2.config import default_config


def records(values):
    return [{"fold": fold, "score": score, "valid_for_selection": True}
            for fold, score in enumerate(values)]


class CompetitorTuningTests(unittest.TestCase):
    def test_last_two_folds_change_selected_penalty(self):
        curves = {0.: records([0., 0., 0., 0., 0.]), .25: records([1., 1., 1., 0., 0.])}
        self.assertEqual(select_spatial_penalty(curves, [0., .25])[0], 0.)
        curves[0.] = records([0., 0., 0., 10., 10.])
        self.assertEqual(select_spatial_penalty(curves, [0., .25])[0], .25)

    def test_incomplete_nonconverged_and_infinite_candidates(self):
        curves = {0.: records([5.] * 5), .1: records([0.] * 4), .25: records([0.] * 5)}
        curves[.25][3]["valid_for_selection"] = False
        self.assertEqual(select_spatial_penalty(curves, curves)[0], 0.)
        curves = {0.: records([np.inf] * 5), .25: records([np.inf] * 5)}
        chosen, table = select_spatial_penalty(curves, curves)
        self.assertEqual(chosen, 0.)
        self.assertTrue(all(row["selectable"] for row in table))
        self.assertTrue(np.isinf(table[0]["mean_five_fold_deviance_per_molecule"]))

    @unittest.skipUnless(importlib.util.find_spec("pycvxcluster"), "requires pinned likelihood environment")
    def test_tuning_excludes_scoring_counts_and_rebuilds_each_inner_graph(self):
        from gplsi_joint_v2.competitors import CompetitorResult
        from gplsi_joint_v2.likelihood import recover_A_poisson
        class Prepared:
            @property
            def train_score(self):
                raise AssertionError("outer scoring counts leaked into tuning")
            @property
            def eval_sets(self):
                raise AssertionError("outer test sets leaked into tuning")
        rng = np.random.default_rng(90)
        prepared = Prepared()
        prepared.train_rank_counts = sparse.csr_matrix(rng.poisson(8., size=(30, 10)))
        prepared.train_fit = prepared.train_rank_counts[:, :7]
        prepared.train_coords = np.c_[np.cos(np.arange(30)), np.sin(np.arange(30))]
        prepared.train_graph_ids = np.repeat("section", 30)
        prepared.split_metadata = {"outer_split": {"dataset": "visium_dlpfc", "split_id": "test"},
                                   "panel_requested": 7, "molecule_seed": 22}
        rows = np.arange(30)
        graph = sparse.csr_matrix((np.ones(30), (rows, (rows+1) % 30)), shape=(30, 30))
        graph = (graph + graph.T).tocsr()
        calls = []
        def fake_nmf(counts, adjacency, K, seed, *, penalty, **kwargs):
            calls.append((counts.shape, adjacency.copy(), penalty))
            # This intentionally unsupported native profile would fail fold-in
            # if tuning accidentally bypassed the fixed-W Poisson recovery.
            A = np.zeros((K, counts.shape[1]))
            A[:, 0] = 1.
            return CompetitorResult(np.full((counts.shape[0], K), 1/K), A, True, "ok", {})
        config = default_config()
        config["competitor_tuning"]["penalties"] = [0., .25]
        config["solvers"]["foldin_max_iter"] = 50
        with patch("gplsi_joint_v2.competitors.fit_graph_kl_nmf", side_effect=fake_nmf), \
             patch("gplsi_joint_v2.likelihood.recover_A_poisson", wraps=recover_A_poisson) as recovery:
            fit = fit_tuned_spatial_competitor("graph_kl_nmf_tuned", prepared, graph, 2, 8, config)
        self.assertEqual(recovery.call_count, 10)
        self.assertTrue(all(call.args[1].shape[0] < 30 for call in recovery.call_args_list))
        self.assertEqual(len(calls), 11)
        self.assertTrue(all(shape[0] < 30 and shape[1] == 7 for shape, _, _ in calls[:-1]))
        self.assertEqual(calls[-1][0], (30, 7))
        self.assertEqual(len(fit.metadata["tuning"]["fold_inventory"]), 5)
        self.assertFalse(fit.metadata["tuning"]["outer_scoring_counts_used"])
        self.assertTrue(all(row["five_fold_complete"] for row in fit.metadata["tuning"]["aggregate"]))
        self.assertFalse(fit.metadata["tuning"]["candidate_native_A_scored"])
        self.assertFalse(fit.metadata["tuning"]["A_current_used"])
        for curve in fit.metadata["tuning"]["fold_curves"].values():
            self.assertTrue(all(row["profile_recovery"]["method"] == "A_full_Pois" for row in curve))
            self.assertTrue(all(row["profile_recovery"]["converged"] for row in curve))
        def uncertified(*args, **kwargs):
            return replace(recover_A_poisson(*args, **kwargs), converged=False, status="test_nonconverged")
        with patch("gplsi_joint_v2.competitors.fit_graph_kl_nmf", side_effect=fake_nmf), \
             patch("gplsi_joint_v2.likelihood.recover_A_poisson", side_effect=uncertified):
            with self.assertRaises(CompetitorTuningError) as caught:
                fit_tuned_spatial_competitor("graph_kl_nmf_tuned", prepared, graph, 2, 8, config)
        self.assertTrue(all(not row["selectable"] for row in caught.exception.diagnostics["aggregate"]))

    @unittest.skipUnless(importlib.util.find_spec("pycvxcluster"), "requires pinned likelihood environment")
    def test_calico_zero_is_rejected_as_undefined_inverse_penalty(self):
        config = default_config()
        config["competitor_tuning"]["spatial_lda_inverse_penalties"] = [0., .25]
        with self.assertRaisesRegex(ValueError, "inverse penalty"):
            fit_tuned_spatial_competitor("spatial_lda_tuned", None, None, 2, 1, config)


if __name__ == "__main__":
    unittest.main()
