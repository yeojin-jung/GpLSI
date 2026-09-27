"""Compact plots use frozen selection and exact shared training-row contracts."""
import json
import numpy as np
import pandas as pd
import pytest

from gplsi_joint_v2.config import default_config
from gplsi_joint_v2.figures import (plot_diagnostic_path, prespecified_map_sources,
                                  plot_training_map, top_profile_rows, plot_profile_stability,
                                  available_symbols)
from gplsi_joint_v2.metrics import topic_profile_metrics
from gplsi_joint_v2.splits import array_hash


def test_lambda_plot_retains_negative_moran_and_undefined_metric_points(tmp_path):
    spectral = [{"lambda": value, "CV_five_scores": [1, 2, 3, 4, 5],
                 "CV_aggregate_score": 15, "optimized_unscaled_edge_group_norm": 10 - i,
                 "optimized_lambda_times_edge_group_norm": value * (10 - i), "status": "ok"}
                for i, value in enumerate([0., .01, .1])]
    geometry = [{"lambda": value, "hunter": hunter, "W_edge_squared_difference": 2,
                 "moran_I_equal_stratum_mean": moran, "one_minus_PAS_10": .2,
                 "occupied_topics": 3}
                for hunter in ("spa_current", "palm")
                for value, moran in zip([0., .01, .1], [-.4, None, .5])]
    path = tmp_path / "path.png"
    result = plot_diagnostic_path({"spectral_path": spectral, "geometry_path": geometry}, path)
    assert path.stat().st_size > 1000
    assert result["lambda_includes_exact_zero"] is True
    assert result["used_for_lambda_selection"] is False
    for hunter in ("spa_current", "palm"):
        row = result["metric_coverage"][hunter + "/moran_I_equal_stratum_mean"]
        assert row == {"expected_points": 3, "finite_points": 2,
                       "undefined_or_nonfinite_points": 1, "negative_points": 1}
    assert json.loads(path.with_suffix(".json").read_text())["used_for_lambda_selection"] is False


def test_prespecified_map_sources_never_pick_more_favorable_complete_split():
    config = default_config()
    dataset = "visium_dlpfc"
    spec = {"dataset": dataset, "outer_split_id": "a_first", "K": config["primary_K"][dataset],
            "panel_requested": config["primary_panel"][dataset], "estimator_seed": config["seeds"]["estimator_seed"],
            "retention": 1., "preprocessing": "P0_raw", "hunter": "spa_current", "control": "selected"}
    tasks = [{"task_id": family, "stage": "geometry", "spec": {**spec, "family": family}}
             for family in ("document", "anchor")]
    tasks += [{"task_id": method, "stage": "competitor", "spec": {**spec, "method": method}}
              for method in config["competitors"]]
    tasks += [{"task_id": "later_but_complete", "stage": "geometry", "state": "complete",
               "spec": {**spec, "family": "document", "outer_split_id": "z_later"}},
              {"task_id": "thin", "stage": "geometry", "spec": {**spec, "family": "document", "retention": .5}},
              {"task_id": "zero", "stage": "geometry", "spec": {**spec, "family": "document", "control": "lambda_zero"}},
              {"task_id": "recovery", "stage": "recovery", "spec": spec}]
    bases = [{"dataset": dataset, "outer_split_id": outer} for outer in ["z_later", "a_first"]]
    selected = prespecified_map_sources(iter(tasks), bases, config)
    assert {t["task_id"] for t in selected} == {"document", "anchor", *config["competitors"]}


def test_map_uses_all_rows_in_first_coordinate_only_stratum_and_validates_order(tmp_path):
    obs = pd.DataFrame({"obs_id": ["r3", "r1", "r2", "r4"], "graph_id": ["z", "a", "a", "z"],
                        "x": [0, 1, 2, 3], "y": [0, 0, 1, 1], "phenotype": ["bad", "bad", "good", "good"]})
    w = np.array([[.1, .9], [.8, .2], [.1, .9], [.6, .4]])
    result = plot_training_map(w, obs, tmp_path / "map.png", expected_row_hash=array_hash(obs.obs_id), W_parent="same_W_parent")
    assert result["graph_stratum"] == "a"
    assert result["plotted_observations"] == result["eligible_stratum_observations"] == 2
    assert result["hard_topic_counts"] == [1, 1]
    assert result["labels_used_for_selection"] is False
    assert result["plotting_subsampling"] is result["fitting_subsampling"] is False
    assert result["no_per_cell_plot_data_persisted"] is True
    assert {p.suffix for p in tmp_path.iterdir()} == {".png", ".json"}
    with pytest.raises(ValueError, match="row hash mismatch"):
        plot_training_map(w, obs.iloc[::-1], tmp_path / "wrong.png", expected_row_hash=array_hash(obs.obs_id))


def test_top_profiles_preserve_ids_use_only_available_symbols_and_shared_metrics(tmp_path):
    a = np.array([[.6, .3, .1], [.1, .3, .6]])
    rows = top_profile_rows(a, ["id1", "id2", "id3"], symbols={"id1": "ACTUAL1"}, top_n=2)
    assert [r["feature_id"] for r in rows] == ["id1", "id2", "id3", "id2"]
    assert rows[0]["gene_symbol"] == "ACTUAL1"
    assert rows[1]["gene_symbol"] is rows[2]["gene_symbol"] is None
    assert rows[0]["exclusivity"] == pytest.approx(.6 / .7)
    assert rows[0]["normalized_topic_entropy"] == pytest.approx(topic_profile_metrics(a)["normalized_entropy"][0])
    symbols, provenance = available_symbols(str(tmp_path), "no_known_source")
    assert symbols == {} and provenance == []


def test_profile_plot_discloses_undefined_and_has_no_pair_based_standard_errors(tmp_path):
    rows = [{"dataset": "visium_dlpfc", "comparison": "initialization", "status": "ok", "JSD_mean": x}
            for x in [.1, .3, None]]
    rows.append({"status": "empty_common_support"})
    result = plot_profile_stability(rows, tmp_path / "profiles.png")
    assert result["profile_pairs"] == 2
    assert result["failed_or_undefined_pairs"] == 2
    assert result["pairwise_SE_reported"] is False
    assert result["labels_used_for_alignment"] is False
