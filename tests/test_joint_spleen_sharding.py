from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/real_data_anchor_word_gplsi/run_experiment.py"
SPEC = importlib.util.spec_from_file_location("joint_spleen_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

DASHBOARD_BUILDER_PATH = (
    ROOT / "scripts/real_data_anchor_word_gplsi/build_results_dashboard_data.py"
)
DASHBOARD_SPEC = importlib.util.spec_from_file_location(
    "results_dashboard_builder", DASHBOARD_BUILDER_PATH
)
assert DASHBOARD_SPEC is not None and DASHBOARD_SPEC.loader is not None
DASHBOARD_BUILDER = importlib.util.module_from_spec(DASHBOARD_SPEC)
DASHBOARD_SPEC.loader.exec_module(DASHBOARD_BUILDER)


def test_joint_spleen_shards_partition_the_full_method_grid() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/real_data_anchor_word_gplsi/spleen/full_joint.json"
        ).read_text()
    )
    assert config["geometries"] == ["document_U"]
    assert config["default_seeds"] == [26090401]
    assert config["seeds_by_estimator_family"] == {
        "lda": [26090401, 26090402, 26090403],
        "spatial_lda": [26090401, 26090402, 26090403],
    }
    assert len(config["execution_shards"]) == 12
    resolved = {
        shard: RUNNER._for_execution_shard(config, shard)
        for shard in config["execution_shards"]
    }

    fast = resolved["P0_raw__fast"]
    assert fast["preprocessings"] == ["P0_raw"]
    assert fast["vertex_hunters"] == [
        "spa_current",
        "palm",
        "palm_accelerated",
    ]
    assert fast["geometries"] == ["document_U"]
    assert fast["graph_topicscore_preprocessings"] == ["P0_raw"]
    assert fast["baselines"] == []

    slow = resolved["P2_ke_weighted__document_U__svs_family"]
    assert slow["preprocessings"] == ["P2_ke_weighted"]
    assert slow["geometries"] == ["document_U"]
    assert slow["vertex_hunters"] == ["svs", "svs_star"]
    assert slow["graph_topicscore_preprocessings"] == []
    assert slow["baselines"] == []

    assert resolved["baseline_plsi"]["baselines"] == [
        "published_baseline",
        "plsi",
    ]
    assert resolved["baseline_spatial_lda"]["baselines"] == ["spatial_lda"]
    assert RUNNER._declared_vertex_unavailability(config, 7, "svs") is None
    assert "computationally unavailable" in RUNNER._declared_vertex_unavailability(
        config, 10, "svs"
    )
    assert "computationally unavailable" in RUNNER._declared_vertex_unavailability(
        config, 10, "svs_star"
    )

    # Four fast preprocessing shards contribute 4*(1*3*2), four SVS-family
    # shards contribute 4*(2*2), P0 contributes graph TopicScore once, and
    # the four baseline shards contribute six rows in total.
    expected_rows = 4 * (1 * 3 * 2) + 4 * (2 * 2) + 1 + 6
    assert expected_rows == 47


def test_joint_spleen_dashboard_plan_uses_method_specific_seed_counts() -> None:
    tasks = DASHBOARD_BUILDER._load_plan_tasks()
    joint = tasks[
        (tasks["dataset"] == "mouse_spleen_codex")
        & (tasks["summary_group"] == "joint")
    ]
    assert len(joint) == 12
    assert set(joint["seed"]) == {26090401, 26090402, 26090403}
    default_tasks = joint[joint["seed"] == 26090401]
    assert set(default_tasks["planned_estimator_families"]) == {("*",)}
    extra_tasks = joint[joint["seed"].isin([26090402, 26090403])]
    assert set(extra_tasks["planned_estimator_families"]) == {
        ("lda", "spatial_lda")
    }
    assert set(extra_tasks["required_execution_shards"]) == {
        ("baseline_lda", "baseline_spatial_lda")
    }

    planned = DASHBOARD_BUILDER._planned_configurations(tasks)
    joint_k5 = planned[
        (planned["dataset"] == "mouse_spleen_codex")
        & (planned["summary_group"] == "joint")
        & (planned["K"] == 5)
    ]
    assert not (joint_k5["estimator_family"] == "anchor_feature_gplsi").any()
    assert set(
        joint_k5.loc[
            joint_k5["estimator_family"] == "document_gplsi", "planned_n"
        ]
    ) == {1}
    assert set(
        joint_k5.loc[
            joint_k5["estimator_family"].isin(["lda", "spatial_lda"]),
            "planned_n",
        ]
    ) == {3}
    assert set(
        joint_k5.loc[joint_k5["estimator_family"] == "plsi", "planned_n"]
    ) == {1}


def test_joint_spleen_runner_enforces_seed_policy() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/real_data_anchor_word_gplsi/spleen/full_joint.json"
        ).read_text()
    )
    unrestricted = RUNNER._for_seed_policy(config, 26090401)
    assert unrestricted is config

    extra_seed = RUNNER._for_seed_policy(config, 26090402)
    assert extra_seed["baselines"] == ["lda", "spatial_lda"]
    assert extra_seed["preprocessings"] == []
    assert extra_seed["vertex_hunters"] == []

    lda_shard = RUNNER._for_execution_shard(config, "baseline_lda")
    assert RUNNER._for_seed_policy(lda_shard, 26090403)["baselines"] == [
        "lda"
    ]
    fast_shard = RUNNER._for_execution_shard(config, "P0_raw__fast")
    with pytest.raises(ValueError, match="not shard"):
        RUNNER._for_seed_policy(fast_shard, 26090402)


def test_cooking_k7_uses_k_specific_shards_without_replanning_completed_k() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/real_data_anchor_word_gplsi/cook/full_manuscript_K5-7.json"
        ).read_text()
    )
    assert config["K_values"] == [5, 6, 7]
    k7_shards = config["execution_shards_by_K"]["7"]
    assert len(k7_shards) == 16
    resolved = {
        shard: RUNNER._for_execution_shard(config, shard) for shard in k7_shards
    }
    assert resolved["P0_raw__fast"]["vertex_hunters"] == [
        "spa_current",
        "palm",
        "palm_accelerated",
    ]
    assert resolved["P3_tran_then_ke__word_Z__svs_family"][
        "vertex_hunters"
    ] == ["svs", "svs_star"]
    assert resolved["baseline_plsi"]["baselines"] == [
        "published_baseline",
        "plsi",
    ]

    planned = DASHBOARD_BUILDER._load_plan_tasks()
    cooking = planned[planned["dataset"] == "whats_cooking"]
    assert sorted(cooking["K"].unique().tolist()) == [5, 6, 7]
    assert set(cooking.loc[cooking["K"].isin([5, 6]), "required_execution_shards"]) == {
        ("all",)
    }
    assert set(cooking.loc[cooking["K"] == 7, "required_execution_shards"]) == {
        tuple(k7_shards)
    }


def test_topic_explorer_retains_every_exact_completed_configuration() -> None:
    base = {
        "dataset": "stanford_crc_codex",
        "summary_group": "all",
        "K": 5,
        "estimator_family": "document_gplsi",
        "spectral_geometry": "document_U",
        "vertex_hunter": "spa_current",
        "W_recovery_method": "stable_document_vertex_solve",
        "graph_svd_version": "iterative_two_step",
    }
    completed = [
        {**base, "preprocessing": preprocessing, "A_recovery": A_recovery}
        for preprocessing in ("P0_raw", "P1_tran_alpha_0p005")
        for A_recovery in ("A_current", "A_full_Pois")
    ]
    published = {
        **base,
        "estimator_family": "published_baseline",
        "spectral_geometry": "not_applicable",
        "vertex_hunter": "not_applicable",
        "preprocessing": "native_baseline",
        "A_recovery": "native_baseline",
        "W_recovery_method": "native_baseline",
        "graph_svd_version": "not_applicable",
    }

    summary_rows = []
    fit_rows = []
    for config_index, config in enumerate(completed):
        variant_id = DASHBOARD_BUILDER._variant_id(config)
        summary_rows.append(
            {
                **config,
                "variant_id": variant_id,
                "variant_label": f"completed-{config_index}",
                "n_ok": 2,
                "mean": 1.5 + config_index,
                "median": 1.5 + config_index,
                "planned_n": 2,
                "full_coverage": True,
            }
        )
        for seed_offset, dev_per_count in enumerate((1.0, 2.0)):
            fit_rows.append(
                {
                    **config,
                    "biological_group": None,
                    "status": "ok",
                    DASHBOARD_BUILDER.METRIC: dev_per_count * 100,
                    DASHBOARD_BUILDER.COUNT: 100,
                    "artifacts.estimate.path": (
                        "results/real_data_anchor_word_gplsi/"
                        f"config-{config_index}-seed-{seed_offset}.npz"
                    ),
                    "seed": 100 + seed_offset,
                    "fit_id": f"fit-{config_index}-{seed_offset}",
                    "task_config_hash": f"task-{config_index}-{seed_offset}",
                }
            )

    summary_rows.append(
        {
            **published,
            "variant_id": DASHBOARD_BUILDER._variant_id(published),
            "variant_label": "published reference",
            "n_ok": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "planned_n": 1,
            "full_coverage": False,
        }
    )
    fit_rows.append(
        {
            **published,
            "biological_group": None,
            "status": "published_artifact_only",
            DASHBOARD_BUILDER.METRIC: float("nan"),
            DASHBOARD_BUILDER.COUNT: float("nan"),
            "artifacts.estimate.path": (
                "results/real_data_anchor_word_gplsi/published.npz"
            ),
            "seed": 0,
            "fit_id": "published-fit",
            "task_config_hash": "published-task",
        }
    )

    selections = DASHBOARD_BUILDER._select_representative_fits(
        pd.DataFrame(fit_rows), pd.DataFrame(summary_rows)
    )

    assert {
        (selection["preprocessing"], selection["A_recovery"])
        for selection in selections
    } == {
        ("P0_raw", "A_current"),
        ("P0_raw", "A_full_Pois"),
        ("P1_tran_alpha_0p005", "A_current"),
        ("P1_tran_alpha_0p005", "A_full_Pois"),
    }
    assert len(selections) == len(completed)
    assert len({selection["id"] for selection in selections}) == len(selections)
    assert all(
        selection["estimator_family"] != "published_baseline"
        for selection in selections
    )
