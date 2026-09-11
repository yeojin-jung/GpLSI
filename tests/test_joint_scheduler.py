"""Mocked Slurm dispatch tests: exact fan-out, bounds, gates and crash safety."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from gplsi_joint_v2 import scheduler
from gplsi_joint_v2.artifacts import atomic_json, commit_stage
from gplsi_joint_v2.runner import stage_path


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="joint_scheduler_test_")
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifests/run_v2/dag.json"
        self.resources = self.root / "resources.json"
        self.config = {"resources": {"account": "pi-test", "partitions": ["caslake", "cdonnat"],
                                     "default_partition": "caslake", "global_concurrency": 4,
                                     "max_submitted_stage_tasks": 8, "array_cap": 2}}
        atomic_json(self.resources, {"default": {"cpus": 2, "memory_gb": 4, "time": "00:10:00"}})
        self.slurm_calls = []
        self.active_output = ""
        self.next_job = 1000

    def tearDown(self):
        self.temporary.cleanup()

    @property
    def ledger_file(self):
        return self.root / "slurm/joint_v2/run_v2/dispatch.json"

    def task(self, key, parents=(), stage="prepare"):
        return {"task_id": key, "parents": list(parents), "stage": stage,
                "dataset": "visium_dlpfc", "variant": key, "spec": {"dataset": "visium_dlpfc"}}

    def write_manifest(self, tasks):
        dag = {"tasks": tasks, "code_hash": "tested_source", "config_hash": "tested_config",
               "config": self.config}
        atomic_json(self.manifest, dag)
        return dag

    def write_ledger(self, submitted=None, blocked=None, **extra):
        atomic_json(self.ledger_file, {"manifest": str(self.manifest), "code_hash": "tested_source",
                                     "submitted": submitted or {}, "blocked": blocked or {}, **extra})

    def slurm(self, arguments):
        self.slurm_calls.append(arguments)
        if arguments[0] == "squeue":
            return self.active_output
        if arguments[0] == "sacct":
            return ""
        if arguments[0] == "sbatch":
            self.next_job += 1
            return str(self.next_job)
        raise AssertionError(f"unexpected scheduler mutation: {arguments}")

    def call_dispatch(self, **kwargs):
        return scheduler.dispatch(self.root, self.manifest, self.resources, gate_kind="pilot", **kwargs)

    def complete(self, task, value=None):
        directory = stage_path(self.root, task)
        atomic_json(directory / "result.json", {"deviance": value})
        commit_stage(directory, task["task_id"], {}, ["result.json"])

    def test_array_indices_map_to_exact_immutable_task_list(self):
        self.write_manifest([self.task("a"), self.task("b"), self.task("c")])
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            result = self.call_dispatch(maximum_new=3)
        self.assertEqual(len(result["submitted"]), 3)
        calls = [args for args in self.slurm_calls if args[0] == "sbatch"]
        self.assertEqual(len(calls), 2)
        self.assertIn("--array=0-1%2", calls[0])
        self.assertIn("--array=0-0%2", calls[1])
        for row in result["submitted"]:
            array = json.loads(Path(row["array_manifest"]).read_text())
            index = int(row["job_id"].split("_")[1])
            self.assertEqual(array["task_ids"][index], row["task_id"])
            self.assertFalse(any("aftercorr" in token for token in row["command"]))
        ledger = json.loads(self.ledger_file.read_text())
        self.assertEqual(set(ledger["submitted"]), {"a", "b", "c"})

    def test_global_outstanding_cap_ignores_unrelated_jobs(self):
        self.config["resources"]["global_concurrency"] = 3
        self.write_manifest([self.task(x) for x in ("old_a", "old_b", "new_a", "new_b")])
        self.write_ledger({"old_a": {"job_id": "20_0"}, "old_b": {"job_id": "20_1"}})
        self.active_output = "20_0|RUNNING\n20_1|PENDING\n999_0|RUNNING"
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            result = self.call_dispatch()
        self.assertEqual(result["active_before_dispatch"], 2)
        self.assertEqual(len(result["submitted"]), 1)
        self.assertEqual(result["submitted"][0]["task_id"], "new_a")

    def test_ready_only_children_wait_for_running_parent_artifact(self):
        parent = self.task("parent")
        child = self.task("child", ["parent"], "spectral")
        unrelated = self.task("unrelated")
        self.write_manifest([parent, child, unrelated])
        self.write_ledger({"parent": {"job_id": "30_0"}})
        self.active_output = "30_0|RUNNING"
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            result = self.call_dispatch()
        self.assertEqual([row["task_id"] for row in result["submitted"]], ["unrelated"])
        self.assertNotIn("child", json.loads(self.ledger_file.read_text())["blocked"])

    def test_failed_parent_blocks_only_its_descendants(self):
        a, c = self.task("failed_parent"), self.task("good_parent")
        b, d = self.task("blocked_child", ["failed_parent"], "spectral"), self.task("good_child", ["good_parent"], "spectral")
        self.write_manifest([a, b, c, d])
        self.write_ledger({"failed_parent": {"job_id": "40_0"}})
        self.complete(c)
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            result = self.call_dispatch()
        self.assertEqual([row["task_id"] for row in result["submitted"]], ["good_child"])
        blocked = json.loads(self.ledger_file.read_text())["blocked"]
        self.assertEqual(blocked["blocked_child"]["parents"], ["failed_parent"])
        self.assertNotIn("good_child", blocked)

    def test_completed_infinite_score_is_scientific_output_not_parent_failure(self):
        parent = self.task("support_infinite", stage="evaluation")
        child = self.task("summary", ["support_infinite"], "aggregation")
        self.write_manifest([parent, child])
        self.complete(parent, float("inf"))
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            result = self.call_dispatch()
        self.assertEqual([row["task_id"] for row in result["submitted"]], ["summary"])
        value = json.loads((stage_path(self.root, parent) / "result.json").read_text())
        self.assertEqual(value["deviance"], "Infinity")

    def test_missing_and_stale_production_gates_prevent_submission(self):
        dag = self.write_manifest([self.task("a")])
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            with self.assertRaisesRegex(RuntimeError, "gated"):
                scheduler.dispatch(self.root, self.manifest, self.resources)
        gate = {"status": "passed", "code_hash": "old_source", "config_hash": dag["config_hash"],
                "correctness_passed": True, "smoke_passed": True,
                "full_size_pilots_passed": True, "storage_passed": True}
        atomic_json(self.root / "reports/joint_v2/PRODUCTION_GATE.json", gate)
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            with self.assertRaisesRegex(RuntimeError, "exact source"):
                scheduler.dispatch(self.root, self.manifest, self.resources)
        self.assertFalse(self.slurm_calls)

    def test_resume_does_not_query_purged_historical_job_ids(self):
        self.write_manifest([self.task("old"), self.task("new")])
        self.write_ledger({"old": {"job_id": "1_0"}})
        def slurm(arguments):
            if arguments[0] == "squeue" and "-j" in arguments:
                raise AssertionError("querying purged job IDs can make Slurm reject the entire resume")
            return self.slurm(arguments)
        with patch.object(scheduler, "_slurm", side_effect=slurm):
            result = self.call_dispatch()
        self.assertEqual([row["task_id"] for row in result["submitted"]], ["new"])

    def test_submission_intent_is_durable_before_sbatch_and_uncertainty_blocks_retry(self):
        self.write_manifest([self.task("a")])
        checked_intent = []
        def interrupted(arguments):
            if arguments[0] == "sbatch":
                ledger = json.loads(self.ledger_file.read_text())
                intents = [row for row in ledger["intents"].values() if row["status"] == "dispatching"]
                self.assertEqual(len(intents), 1)
                comment = next(token.split("=", 1)[1] for token in arguments if token.startswith("--comment="))
                self.assertEqual(intents[0]["unique_comment"], comment)
                checked_intent.append(comment)
                raise RuntimeError("connection interrupted after Slurm may have accepted submission")
            return self.slurm(arguments)
        with patch.object(scheduler, "_slurm", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.call_dispatch()
        self.assertEqual(len(checked_intent), 1)
        self.assertFalse(json.loads(self.ledger_file.read_text())["submitted"])
        self.slurm_calls.clear()
        with patch.object(scheduler, "_slurm", side_effect=self.slurm):
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                self.call_dispatch()
        self.assertFalse(any(args[0] == "sbatch" for args in self.slurm_calls))

    def test_successful_first_array_survives_interrupted_second_submission(self):
        self.write_manifest([self.task("a"), self.task("b"), self.task("c")])
        submitted_count = 0
        def second_interrupt(arguments):
            nonlocal submitted_count
            if arguments[0] == "sbatch":
                submitted_count += 1
                if submitted_count == 2:
                    ledger = json.loads(self.ledger_file.read_text())
                    self.assertEqual(set(ledger["submitted"]), {"a", "b"})
                    raise RuntimeError("uncertain second array acceptance")
            return self.slurm(arguments)
        with patch.object(scheduler, "_slurm", side_effect=second_interrupt):
            with self.assertRaisesRegex(RuntimeError, "uncertain second"):
                self.call_dispatch()
        ledger = json.loads(self.ledger_file.read_text())
        self.assertEqual(set(ledger["submitted"]), {"a", "b"})
        self.assertEqual(sum(row["status"] == "dispatching" for row in ledger["intents"].values()), 1)


if __name__ == "__main__":
    unittest.main()
