import numpy as np
import pytest
from scipy import sparse

from gplsi_joint_v2.metrics import (score_counts, pas10_neighbors, pas10_from_neighbors,
    spatial_metrics, align_profiles, grouped_label_transfer, biological_score_summary,
    hard_label_metrics)


def test_exact_score_dense_formula_sparse_parity_and_support_failures():
    w = np.array([[1, 0], [0, 1], [.5, .5]], float)
    a = np.array([[.9, .05, .05], [.05, .9, .05]])
    d = np.array([[2, 0, 1], [0, 3, 2], [1, 1, 4]])
    dense = score_counts(w, a, d, entry_chunk_size=2)
    csr = score_counts(w, a, sparse.csr_matrix(d), entry_chunk_size=100)
    assert np.isclose(dense["summary"]["log_likelihood"], -22.98580944306206)
    assert np.isclose(dense["summary"]["deviance"], 25.011658464485144)
    assert dense["summary"] == csr["summary"]
    zero = np.array([[1, 0, 0], [0, 1, 0]], float)
    failed = score_counts(w, zero, d, floor=1e-12)
    assert failed["summary"]["support_violation_molecules"] == 7
    assert failed["summary"]["support_violation_entries"] == 3
    assert failed["summary"]["deviance"] == np.inf
    assert np.isfinite(failed["summary"]["diagnostic_floored_deviance"])
    assert np.isposinf(failed["per_row"]["deviance"]).all()


def test_count_mask_does_not_hide_method_failure_and_zero_score_rows():
    w, a = np.ones((4, 1)), np.array([[.5, .5]])
    d = np.array([[1, 1], [2, 2], [0, 0], [5, 5]])
    result = score_counts(w, a, d, eligibility_mask=[True, True, True, False],
                          inference_valid=[True, False, True, True])
    assert result["summary"]["scored_observations"] == 2
    assert result["summary"]["scored_molecules"] == 6
    assert result["summary"]["inference_failed_observations"] == 1
    assert result["summary"]["zero_score_observations"] == 1
    assert np.isnan(result["summary"]["deviance"])
    aggregate = biological_score_summary(result["per_row"], ["a", "b", "a", "b"])
    assert aggregate["standard_error"] is None
    assert np.isnan(aggregate["equal_biological_unit_deviance"])


def test_float32_factor_storage_preserves_composition_scores_to_tolerance():
    rng = np.random.default_rng(5)
    w = rng.dirichlet(np.ones(7), size=23)
    a = rng.dirichlet(np.ones(29), size=7)
    d = np.vstack([rng.multinomial(100, row) for row in w @ a])
    original = score_counts(w, a, d)["summary"]
    stored = score_counts(w.astype(np.float32), a.astype(np.float32), d)["summary"]
    assert abs(original["deviance_per_molecule"] - stored["deviance_per_molecule"]) < 1e-7
    assert abs(original["log_likelihood_per_molecule"] - stored["log_likelihood_per_molecule"]) < 1e-7
    assert stored["factor_storage_roundoff_renormalized"]


def test_pas_five_vs_six_disagreements_and_alternatives_need_not_match():
    coords = np.column_stack([np.arange(11), np.zeros(11)])
    strata = np.full(11, "section")
    neighbors = pas10_neighbors(coords, strata)
    labels = np.zeros(11, int)
    labels[1:6] = np.arange(1, 6)
    five = pas10_from_neighbors(labels, neighbors, strata)
    assert five["disagreeing_neighbors"][0] == 5
    assert not five["abnormal"][0]
    labels[6] = 6
    six = pas10_from_neighbors(labels, neighbors, strata)
    assert six["disagreeing_neighbors"][0] == 6
    assert six["abnormal"][0]
    permuted = pas10_from_neighbors(100 - labels, neighbors, strata)
    assert permuted["one_minus_PAS_10"] == six["one_minus_PAS_10"]
    constant = pas10_from_neighbors(np.zeros(11), neighbors, strata)
    assert constant["one_minus_PAS_10"] == 1


def test_pas_self_exclusion_duplicate_coordinates_small_strata_and_boundaries():
    coordinates = np.zeros((32, 2))
    strata = np.array(["a"] * 11 + ["b"] * 11 + ["c"] * 10)
    neighbors = pas10_neighbors(coordinates, strata)
    assert np.all(neighbors[:22] != np.arange(22)[:, None])
    assert np.all(strata[neighbors[:22]] == strata[:22, None])
    assert np.all(neighbors[22:] == -1)
    result = pas10_from_neighbors(np.zeros(32), neighbors, strata)
    assert result["valid_observations"] == 22
    assert result["per_stratum"][2]["status"] == "fewer_than_11_observations"
    invalid = neighbors.copy()
    invalid[0, 0] = 0
    with pytest.raises(ValueError, match="itself"):
        pas10_from_neighbors(np.zeros(32), invalid, strata)
    invalid[0, 0] = 12
    with pytest.raises(ValueError, match="cross"):
        pas10_from_neighbors(np.zeros(32), invalid, strata)


def line_graph(n):
    return sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n)).tocsr()


def test_moran_centering_detects_within_stratum_constants_not_cohort_means():
    w = np.array([[.1, .9]] * 4 + [[.9, .1]] * 4)
    strata = np.array(["a"] * 4 + ["b"] * 4)
    coords = np.column_stack([np.tile(np.arange(4), 2), np.zeros(8)])
    graph = sparse.block_diag([line_graph(4), line_graph(4)], format="csr")
    result = spatial_metrics(w, graph, coords, strata)
    assert result["moran_valid_topic_strata"] == 0
    assert np.isnan(result["moran_I_equal_stratum_mean"])
    assert all(r["status"] == "near_constant_topic" for r in result["moran_per_stratum_topic"])
    assert result["W_edge_squared_difference"] < 1e-14
    cross = graph.tolil()
    cross[0, 4] = cross[4, 0] = 1
    with pytest.raises(ValueError, match="crosses"):
        spatial_metrics(w, cross.tocsr(), coords, strata)


def test_moran_formula_and_paired_A_W_only_invariants():
    values = np.linspace(.05, .95, 12)
    w = np.column_stack([values, 1 - values])
    coords = np.column_stack([np.arange(12), np.zeros(12)])
    graph, strata = line_graph(12), np.full(12, "section")
    metrics = spatial_metrics(w, graph, coords, strata)
    centered = values - values.mean()
    expected = len(w) / graph.sum() * centered @ (graph @ centered) / (centered @ centered)
    assert np.isclose(metrics["moran_I_equal_stratum_mean"], expected)
    assert metrics["occupied_topics"] == 2
    # A is intentionally absent from both W_train-only APIs.
    assert hard_label_metrics(w, np.arange(12) > 5) == hard_label_metrics(w.copy(), np.arange(12) > 5)
    repeated = spatial_metrics(w.copy(), graph, coords, strata)
    assert repeated["moran_I_equal_stratum_mean"] == metrics["moran_I_equal_stratum_mean"]
    assert repeated["one_minus_PAS_10"] == metrics["one_minus_PAS_10"]


def test_profile_matching_is_label_free_and_reports_support_and_retained_mass():
    a = np.array([[.6, .3, .1], [.1, .3, .6]])
    b = a[::-1, ::-1]
    result = align_profiles(a, b, ["a", "b", "c"], ["c", "b", "a"])
    assert result["assignment_B"] == [1, 0]
    assert result["JSD_mean"] < 1e-15
    assert result["labels_used"] is False
    restricted = align_profiles(a, b, ["a", "b", "c"], ["c", "b", "a"], common_support=["a", "b"])
    assert restricted["support_size"] == 2
    assert np.allclose(restricted["retained_mass_A"], [.9, .4])
    with pytest.raises(ValueError, match="equal K"):
        align_profiles(a, b[:1], ["a", "b", "c"], ["c", "b", "a"])


def test_outer_test_metadata_cannot_select_decoder_or_change_predictions():
    rng = np.random.default_rng(17)
    y = np.repeat(["a", "b"] * 5, 4)
    groups = np.repeat(np.arange(10).astype(str), 4)
    x = np.column_stack([(y == "a").astype(float), rng.normal(0, .1, len(y))])
    xt = np.array([[1, 0], [0, 0], [1, .1], [0, .1]])
    first = grouped_label_transfer(x, y, groups, xt, ["a", "b", "a", "b"])
    altered = grouped_label_transfer(x, y, groups, xt, ["b", "a", "b", "a"])
    assert first["inner_folds"] == 5
    assert first["selected_C"] == altered["selected_C"]
    assert first["cv_curves"] == altered["cv_curves"]
    assert np.array_equal(first["predictions"], altered["predictions"])
    assert first["balanced_accuracy"] > altered["balanced_accuracy"]
    assert first["test_labels_used_for_tuning"] is False
