"""Launch membership must not alter the frozen scientific experiment."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from gplsi_joint_v2.config import default_config
from gplsi_joint_v2.manifests import (
    build_base_manifests, build_dag, split_launch_manifests,
    validate_task_identity, write_launch_manifests,
)


def _all_splits():
    return ([{"dataset": "visium_dlpfc", "protocol": "section_holdout",
              "split_id": f"section_rotation_{i:02d}"} for i in range(1, 4)]
            + [{"dataset": "visium_dlpfc", "protocol": "spatial_half",
                "split_id": f"spatial_{side}"} for side in ("left", "right", "top", "bottom")]
            + [{"dataset": "merfish_trem2_5xfad", "protocol": "animal_holdout",
                "split_id": f"leave_2_out_{i:02d}"} for i in range(1, 16)]
            + [{"dataset": "xenium_uc", "protocol": "patient_holdout",
                "split_id": f"leave_3_out_{i:02d}"} for i in range(1, 21)])


def _tiny_grid():
    config = default_config()
    config.update(K_values=[7], visium_panels=[500],
                  primary_K={dataset: 7 for dataset in config["primary_K"]},
                  primary_panel={dataset: 500 for dataset in config["primary_panel"]},
                  initialization_seeds=[config["seeds"]["estimator_seed"]],
                  retention_secondary=[], preprocessings=["P0_raw"],
                  hunters=["spa_current"], competitors=["lda"])
    splits = [_all_splits()[i] for i in (0, 3, 7, 22)]
    return config, build_base_manifests(splits, config)


class LaunchSelectionTests(unittest.TestCase):
    def test_all_frozen_bases_preserved_with_exact_counts(self):
        config = default_config()
        original = build_base_manifests(_all_splits(), config)
        before = deepcopy(original)
        selected = split_launch_manifests(original)
        self.assertEqual(original, before)
        expected = {
            "initial": {"core": 250, "stability": 114, "thinning": 114},
            "deferred_visium_spatial_half": {"core": 100, "stability": 12, "thinning": 12},
        }
        for name, manifests in selected.items():
            self.assertEqual({tier: len(rows) for tier, rows in manifests.items()}, expected[name])
        for tier, rows in original.items():
            combined = selected["initial"][tier] + selected["deferred_visium_spatial_half"][tier]
            self.assertEqual(Counter(row["base_id"] for row in rows),
                             Counter(row["base_id"] for row in combined))
        initial = {row["base_id"]: row for rows in selected["initial"].values() for row in rows}
        deferred = {row["base_id"]: row for rows in selected["deferred_visium_spatial_half"].values() for row in rows}
        self.assertEqual(len(initial), 440)
        self.assertEqual(len(deferred), 120)
        self.assertFalse(set(initial) & set(deferred))
        self.assertEqual(Counter(row["dataset"] for row in initial.values()),
                         {"visium_dlpfc": 90, "merfish_trem2_5xfad": 150, "xenium_uc": 200})
        self.assertEqual({row["protocol"] for row in deferred.values()}, {"spatial_half"})
        # Deferral is scheduling-only: all three configured stability seeds remain.
        stability = selected["initial"]["stability"]
        self.assertEqual({row["estimator_seed"] for row in stability}, set(config["initialization_seeds"]))
        self.assertEqual(len({(row["dataset"], row["outer_split_id"]) for row in stability}), 38)

    def test_exclusion_requires_both_visium_and_spatial_half(self):
        records = [
            {"base_id": "a", "dataset": "visium_dlpfc", "protocol": "spatial_half"},
            {"base_id": "b", "dataset": "visium_dlpfc", "protocol": "section_holdout"},
            {"base_id": "c", "dataset": "xenium_uc", "protocol": "spatial_half"},
        ]
        result = split_launch_manifests({"core": records})
        self.assertEqual([r["base_id"] for r in result["initial"]["core"]], ["b", "c"])
        self.assertEqual([r["base_id"] for r in result["deferred_visium_spatial_half"]["core"]], ["a"])

    def test_partition_preserves_exact_full_dag_ids_and_parent_closure(self):
        config, manifests = _tiny_grid()
        full = build_dag(manifests, config, "same_snapshot", {"counts": "frozen"})
        full_tasks = {task["task_id"]: task for task in full["tasks"]}
        collected = {}
        counts = {}
        for name, subset in split_launch_manifests(manifests).items():
            dag = build_dag(subset, config, "same_snapshot", {"counts": "frozen"})
            ids = {task["task_id"] for task in dag["tasks"]}
            self.assertFalse(ids & set(collected))
            self.assertTrue(all(set(task["parents"]) <= ids for task in dag["tasks"]))
            for task in dag["tasks"]:
                self.assertEqual(task, full_tasks[task["task_id"]])
                self.assertTrue(validate_task_identity(task, dag))
                collected[task["task_id"]] = task
            counts[name] = len(dag["tasks"])
        self.assertEqual(collected, full_tasks)
        self.assertEqual(counts, {"initial": 64, "deferred_visium_spatial_half": 28})

    def test_all_competitors_have_poisson_recovery_and_old_grid_is_rejected(self):
        config, manifests = _tiny_grid()
        config["competitors"] = default_config()["competitors"]
        dag = build_dag(manifests, config, "snapshot", {})
        mapping = {task["task_id"]: task for task in dag["tasks"]}
        for task in dag["tasks"]:
            if task["stage"] == "evaluation":
                self.assertEqual(task["spec"]["recovery"], "A_full_Pois")
                self.assertTrue(any(mapping[p]["stage"] in ("recovery", "reference_recovery")
                                    for p in task["parents"]))
            if task["stage"] == "competitor":
                recoveries = [child for child in dag["tasks"] if child["stage"] == "recovery"
                              and task["task_id"] in child["parents"]]
                self.assertEqual(len(recoveries), 1)
                self.assertEqual(recoveries[0]["spec"]["recovery"], "A_full_Pois")
        config["recoveries"] = ["A_current", "A_full_Pois"]
        with self.assertRaisesRegex(ValueError, "requires A_full_Pois"):
            build_dag(manifests, config, "snapshot", {})

    def test_writer_keeps_deferred_inventory_without_duplicate_full_dag(self):
        config, manifests = _tiny_grid()
        with tempfile.TemporaryDirectory(prefix="joint-launch-selection-") as tmp:
            result = write_launch_manifests(tmp, manifests, config, "snapshot", {})
            self.assertEqual(result, write_launch_manifests(tmp, manifests, config, "snapshot", {}))
            self.assertEqual(json.loads(Path(result["full_inventory"]).read_text()), manifests)
            self.assertEqual(result["initial_workload"]["stage_tasks_total"], 64)
            self.assertEqual(result["deferred_workload"]["stage_tasks_total"], 28)
            self.assertEqual(result["full_design_workload"]["stage_tasks_total"], 92)
            self.assertFalse(result["launch_policy"]["deferred_launch_authorized"])
            self.assertEqual(len(list(Path(tmp).rglob("dag.json"))), 2)
            self.assertEqual(len(list(Path(tmp).rglob("tasks.sqlite"))), 2)
            initial = json.loads(Path(result["initial_manifest"]).read_text())
            deferred = json.loads(Path(result["deferred_manifest"]).read_text())
            self.assertEqual(initial["config_hash"], deferred["config_hash"])
            self.assertEqual(initial["launch_selection"]["membership"], "initial")
            self.assertEqual(deferred["launch_selection"]["membership"], "deferred_visium_spatial_half")
            self.assertEqual(Path(result["initial_manifest"]).parent.name,
                             "initial_" + Path(tmp).name)
            self.assertEqual(Path(result["deferred_manifest"]).parent.name,
                             "deferred_visium_spatial_half_" + Path(tmp).name)
            self.assertTrue(all(validate_task_identity(task, initial) for task in initial["tasks"]))

    def test_three_seed_scope_excludes_only_spatial_lda_extra_runs(self):
        config = default_config()
        config.update(K_values=[7], visium_panels=[2000], retention_secondary=[])
        split = _all_splits()[0]
        manifests = build_base_manifests([split], config)
        dag = build_dag(manifests, config, "snapshot", {})
        self.assertEqual(len(dag["bases"]), 3)
        for seed in config["initialization_seeds"]:
            tasks = [task for task in dag["tasks"] if task["spec"].get("estimator_seed") == seed]
            competitors = {task["spec"]["method"] for task in tasks if task["stage"] == "competitor"}
            expected = config["competitors"] if seed == config["seeds"]["estimator_seed"] else config["initialization_scope"]["competitors"]
            self.assertEqual(competitors, set(expected))
            self.assertEqual(sum(task["stage"] == "geometry" for task in tasks), 26)
            self.assertEqual({task["spec"]["hunter"] for task in tasks
                              if task["stage"] == "geometry" and task["spec"]["family"] == "document"},
                             set(config["hunters"]))
            if seed != config["seeds"]["estimator_seed"]:
                self.assertFalse(any(task["stage"] == "diagnostics" for task in tasks))
        # Extracting stability-only bases changes no scientific IDs or methods.
        stability = build_dag({"stability": manifests["stability"]}, config, "snapshot", {})
        self.assertEqual({task["task_id"] for task in stability["tasks"]},
                         {task["task_id"] for task in dag["tasks"]})


if __name__ == "__main__":
    unittest.main()
