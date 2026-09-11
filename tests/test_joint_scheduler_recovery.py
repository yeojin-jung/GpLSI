"""Crash recovery and resource retry are ledger-only, exact-ID operations."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from gplsi_joint_v2 import scheduler_recovery as recovery
from gplsi_joint_v2.artifacts import atomic_json, commit_stage, sha256_file
from gplsi_joint_v2.runner import stage_path


class SchedulerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="joint_recovery_test_")
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifests/run_v2/dag.json"
        self.ledger_path = self.root / "slurm/joint_v2/run_v2/dispatch.json"
        self.queue = ""
        self.accounting = ""
        self.calls = []
        self.task_map = {}
        self.write_manifest([self.task("a"), self.task("b")])

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def task(key, parents=(), stage="prepare"):
        return {"task_id": key, "parents": list(parents), "stage": stage,
                "dataset": "visium_dlpfc", "spec": {"dataset": "visium_dlpfc"}, "variant": key}

    def write_manifest(self, tasks):
        self.task_map = {task["task_id"]: task for task in tasks}
        atomic_json(self.manifest, {"tasks": tasks, "config_hash": "config_unchanged",
                                    "code_hash": "source_unchanged", "config": {}})
        atomic_json(self.ledger_path, {"manifest": str(self.manifest),
                    "manifest_sha256": sha256_file(self.manifest), "config_hash": "config_unchanged",
                    "code_hash": "source_unchanged", "submitted": {}, "blocked": {}, "intents": {}})

    def ledger(self):
        return json.loads(self.ledger_path.read_text())

    def make_intent(self, keys, token="a" * 32, *, registered=False, base="100"):
        array_path = self.ledger_path.parent / ("array_" + token + ".json")
        resources = {"partition": "caslake", "cpus": 2, "memory_gb": 8, "time": "00:10:00"}
        atomic_json(array_path, {"manifest": str(self.manifest), "task_ids": keys,
                                "resource_request": resources, "exact_parent_jobs": []})
        stage = self.task_map[keys[0]]["stage"]
        name, comment = "jv2_" + stage[:8] + "_" + token[:12], "joint_v2:" + token
        command = ["sbatch", "--parsable", "--job-name=" + name, "--comment=" + comment,
                   "--array=0-" + str(len(keys)-1) + "%8", "run_action.sh", "array",
                   "--array-manifest", str(array_path)]
        intent = {"status": "submitted" if registered else "dispatching", "created_at": 1789000000,
                  "array_manifest": str(array_path), "task_ids": keys, "command": command,
                  "unique_comment": comment}
        ledger = self.ledger()
        if registered:
            intent["job_id"] = base
            for index, key in enumerate(keys):
                ledger["submitted"][key] = {"job_id": f"{base}_{index}", "submitted_at": intent["created_at"],
                    "array_manifest": str(array_path), "resource_request": resources,
                    "exact_parent_jobs": [], "command": command}
        ledger["intents"][token] = intent
        atomic_json(self.ledger_path, ledger)
        return {"token": token, "name": name, "comment": comment, "array": array_path, "intent": intent}

    @staticmethod
    def qrow(job_id, state, item, *, comment=None, name=None):
        return "|".join((job_id, state, item["name"] if name is None else name,
                         item["comment"] if comment is None else comment))

    @staticmethod
    def arow(job_id, state, item, *, comment=None, name=None):
        return "|".join((job_id, state, item["name"] if name is None else name,
                         item["comment"] if comment is None else comment,
                         "2026-09-09T12:00:00", "2026-09-09T12:01:00", "2026-09-09T12:05:00", "1:0"))

    def slurm(self, arguments):
        self.calls.append(arguments)
        self.assertIn(arguments[0], {"squeue", "sacct"}, "recovery must never submit or cancel")
        if arguments[0] == "squeue":
            self.assertNotIn("-j", arguments, "purged historical queue IDs must not break resume")
            return self.queue
        self.assertIn("--array", arguments)
        self.assertIn("-X", arguments)
        self.assertIn("-S", arguments)
        self.assertIn("JobID%64", arguments[arguments.index("-o")+1])
        self.assertNotIn("JobIDRaw", arguments[arguments.index("-o")+1])
        return self.accounting

    def reconcile(self, **kwargs):
        with patch.object(recovery, "_slurm", side_effect=self.slurm):
            return recovery.reconcile_intents(self.root, self.manifest, **kwargs)

    def retry(self, keys=("a",), **kwargs):
        with patch.object(recovery, "_slurm", side_effect=self.slurm):
            return recovery.retry_failed_tasks(self.root, self.manifest, keys,
                                               reason="increase measured memory only", **kwargs)

    def complete(self, key):
        directory = stage_path(self.root, self.task_map[key])
        atomic_json(directory / "result.json", {"scientific_output": "keep me"})
        commit_stage(directory, key, {}, ["result.json"])
        return directory

    def assert_unchanged(self, digest):
        self.assertEqual(sha256_file(self.ledger_path), digest)

    def test_reconcile_read_only_then_apply_combines_queue_and_accounting(self):
        item = self.make_intent(["a", "b"])
        self.queue = self.qrow("123_1", "RUNNING", item) + "\n999_0|RUNNING|unrelated|unrelated|comment"
        self.accounting = self.arow("123_0", "COMPLETED", item)
        before = sha256_file(self.ledger_path)
        report = self.reconcile()
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["applied"])
        self.assert_unchanged(before)
        self.assertEqual(report["proposals"][0]["task_job_ids"], {"a": "123_0", "b": "123_1"})
        applied = self.reconcile(apply=True)
        self.assertTrue(applied["applied"])
        ledger = self.ledger()
        self.assertEqual(ledger["submitted"]["a"]["job_id"], "123_0")
        self.assertEqual(ledger["submitted"]["b"]["job_id"], "123_1")
        self.assertEqual(ledger["intents"][item["token"]]["status"], "submitted")
        self.assertEqual(ledger["config_hash"], "config_unchanged")
        self.assertEqual(ledger["code_hash"], "source_unchanged")

    def test_reconcile_accounting_unique_name_fallback_when_comments_not_stored(self):
        item = self.make_intent(["a", "b"])
        self.accounting = "\n".join(self.arow(f"222_{i}", "FAILED", item, comment="") for i in range(2))
        report = self.reconcile(apply=True)
        matches = report["proposals"][0]["evidence"]
        self.assertTrue(all(row["identity_match"] == "exact_unique_name_comment_not_stored" for row in matches))

    def test_reconcile_missing_jobs_does_not_authorize_resubmission(self):
        self.make_intent(["a", "b"])
        before = sha256_file(self.ledger_path)
        report = self.reconcile()
        self.assertEqual(report["status"], "blocked")
        with self.assertRaises(recovery.RecoveryRefused):
            self.reconcile(apply=True)
        self.assert_unchanged(before)
        self.assertFalse(self.ledger()["submitted"])

    def test_reconcile_requires_full_exact_index_set(self):
        item = self.make_intent(["a", "b"])
        before = sha256_file(self.ledger_path)
        for indices in ([0], [0, 1, 2], [1, 2]):
            with self.subTest(indices=indices):
                self.queue = "\n".join(self.qrow(f"333_{i}", "PENDING", item) for i in indices)
                self.accounting = self.arow("333_1.batch", "COMPLETED", item)
                with self.assertRaises(recovery.RecoveryRefused):
                    self.reconcile(apply=True)
                self.assert_unchanged(before)

    def test_reconcile_multiple_arrays_with_same_token_is_ambiguous(self):
        item = self.make_intent(["a"])
        self.queue = self.qrow("111_0", "PENDING", item)
        self.accounting = self.arow("222_0", "FAILED", item)
        self.assertEqual(self.reconcile()["unresolved"][0]["observed_array_ids"], ["111", "222"])

    def test_reconcile_conflicting_comment_or_name_fails_closed(self):
        item = self.make_intent(["a"])
        for overrides in ({"comment": "foreign"}, {"name": "foreign"}):
            with self.subTest(overrides=overrides):
                self.queue = self.qrow("111_0", "PENDING", item, **overrides)
                self.assertEqual(self.reconcile()["status"], "blocked")

    def test_reconcile_all_selected_intents_commit_atomically(self):
        first = self.make_intent(["a"])
        self.make_intent(["b"], token="b"*32)
        self.accounting = self.arow("222_0", "FAILED", first)
        before = sha256_file(self.ledger_path)
        with self.assertRaises(recovery.RecoveryRefused) as caught:
            self.reconcile(apply=True)
        self.assertEqual(len(caught.exception.report["proposals"]), 1)
        self.assert_unchanged(before)
        self.reconcile(intent_ids=[first["token"]], apply=True)
        self.assertEqual(set(self.ledger()["submitted"]), {"a"})

    def test_reconcile_two_intents_cannot_claim_same_task(self):
        first = self.make_intent(["a"])
        second = self.make_intent(["a"], token="b"*32)
        self.accounting = self.arow("111_0", "FAILED", first) + "\n" + self.arow("222_0", "FAILED", second)
        before = sha256_file(self.ledger_path)
        with self.assertRaises(recovery.RecoveryRefused):
            self.reconcile(apply=True)
        self.assert_unchanged(before)

    def test_reconcile_manifest_or_array_tampering_is_rejected_before_query(self):
        item = self.make_intent(["a", "b"])
        array = json.loads(item["array"].read_text())
        array["task_ids"] = ["b", "a"]
        atomic_json(item["array"], array)
        with self.assertRaisesRegex(recovery.RecoveryRefused, "order differ"):
            self.reconcile(apply=True)
        self.assertFalse(self.calls)
        dag = json.loads(self.manifest.read_text())
        dag["code_hash"] = "new_source"
        atomic_json(self.manifest, dag)
        with self.assertRaisesRegex(recovery.RecoveryRefused, "mismatch"):
            self.reconcile()
        self.assertFalse(self.calls)

    def test_retry_queries_exact_registered_ids_and_preserves_old_attempt_and_files(self):
        item = self.make_intent(["a", "b"], registered=True)
        directory = stage_path(self.root, self.task_map["a"])
        atomic_json(directory / "partial.json", {"partial": "retain"})
        partial_hash = sha256_file(directory / "partial.json")
        self.queue = self.qrow("100_1", "RUNNING", item) + "\n999_0|RUNNING|unrelated|"
        self.accounting = self.arow("100_0", "OUT_OF_MEMORY", item)
        before = sha256_file(self.ledger_path)
        old_submission = self.ledger()["submitted"]["a"]
        self.assertEqual(self.retry()["status"], "ready")
        self.assert_unchanged(before)
        report = self.retry(apply=True)
        self.assertTrue(report["applied"])
        ledger = self.ledger()
        self.assertEqual(set(ledger["submitted"]), {"b"})
        self.assertEqual(ledger["attempt_history"]["a"][0]["previous_submission"], old_submission)
        self.assertEqual(ledger["retry_authorizations"]["a"]["previous_job_id"], "100_0")
        self.assertEqual(ledger["code_hash"], "source_unchanged")
        self.assertEqual(ledger["config_hash"], "config_unchanged")
        self.assertEqual(sha256_file(directory / "partial.json"), partial_hash)
        for call in self.calls:
            if call[0] == "sacct":
                self.assertEqual(call[call.index("-j")+1], "100_0")

    def test_retry_rejects_queue_live_and_accounting_live_or_unknown(self):
        item = self.make_intent(["a"], registered=True)
        before = sha256_file(self.ledger_path)
        self.queue = self.qrow("100_0", "COMPLETING", item)
        self.accounting = self.arow("100_0", "FAILED", item)
        self.assertEqual(self.retry()["status"], "blocked")
        self.queue = ""
        for state in ("RUNNING", "REQUEUED", "SUSPENDED", "UNKNOWN", "OUT_OF_ME+"):
            with self.subTest(state=state):
                self.accounting = self.arow("100_0", state, item)
                with self.assertRaises(recovery.RecoveryRefused):
                    self.retry(apply=True)
                self.assert_unchanged(before)

    def test_retry_absent_or_conflicting_accounting_never_assumes_failure(self):
        item = self.make_intent(["a"], registered=True)
        before = sha256_file(self.ledger_path)
        for output in ("", self.arow("100_0", "FAILED", item) + "\n" + self.arow("100_0", "COMPLETED", item),
                       self.arow("100_0", "FAILED", item, name="recycled_job_id")):
            with self.subTest(output=output):
                self.accounting = output
                with self.assertRaises(recovery.RecoveryRefused):
                    self.retry(apply=True)
                self.assert_unchanged(before)

    def test_retry_valid_successful_artifact_always_protected(self):
        item = self.make_intent(["a"], registered=True)
        directory = self.complete("a")
        before = sha256_file(self.ledger_path)
        marker_hash = sha256_file(directory / "complete.json")
        self.accounting = self.arow("100_0", "TIMEOUT", item)
        with self.assertRaises(recovery.RecoveryRefused) as caught:
            self.retry(apply=True, allow_corrupt_output=True, allow_missing_output=True)
        self.assertIn("protected", caught.exception.report["refused"][0]["reason"])
        self.assert_unchanged(before)
        self.assertEqual(sha256_file(directory / "complete.json"), marker_hash)

    def test_retry_corrupt_output_is_protected_even_with_flag_and_keeps_files(self):
        item = self.make_intent(["a"], registered=True)
        directory = self.complete("a")
        atomic_json(directory / "result.json", {"corrupted": True})
        self.accounting = self.arow("100_0", "FAILED", item)
        before = sha256_file(self.ledger_path)
        self.assertEqual(self.retry()["status"], "blocked")
        self.assert_unchanged(before)
        with self.assertRaises(recovery.RecoveryRefused) as caught:
            self.retry(apply=True, allow_corrupt_output=True)
        self.assertEqual(caught.exception.report["refused"][0]["artifact"]["status"], "corrupt")
        self.assert_unchanged(before)
        self.assertTrue((directory / "result.json").is_file())
        self.assertTrue((directory / "complete.json").is_file())

    def test_retry_protects_valid_completed_descendants_when_parent_output_missing(self):
        self.write_manifest([self.task("a"), self.task("child", ["a"], "spectral")])
        item = self.make_intent(["a"], registered=True)
        self.complete("child")
        self.accounting = self.arow("100_0", "FAILED", item)
        before = sha256_file(self.ledger_path)
        with self.assertRaises(recovery.RecoveryRefused) as caught:
            self.retry(apply=True)
        self.assertIn("descendants", caught.exception.report["refused"][0]["reason"])
        self.assert_unchanged(before)

    def test_retry_removes_only_exact_targets_from_cached_completed_inventory(self):
        item = self.make_intent(["a", "b"], registered=True)
        ledger = self.ledger()
        ledger.update(initial_inventory_scanned=True, completed_artifact_keys=["a", "b"])
        atomic_json(self.ledger_path, ledger)
        self.accounting = self.arow("100_0", "TIMEOUT", item)
        report = self.retry(apply=True)
        self.assertEqual(report["completed_cache_keys_removed"], ["a"])
        self.assertEqual(self.ledger()["completed_artifact_keys"], ["b"])
        self.assertTrue(self.ledger()["initial_inventory_scanned"])
        self.assertTrue(self.ledger()["retry_authorizations"]["a"]["verify_completion_hashes"])

    def test_retry_completed_missing_output_requires_explicit_flag(self):
        item = self.make_intent(["a"], registered=True)
        self.accounting = self.arow("100_0", "COMPLETED", item)
        self.assertEqual(self.retry()["status"], "blocked")
        self.assertEqual(self.retry(allow_corrupt_output=True)["status"], "blocked")
        self.assertTrue(self.retry(apply=True, allow_missing_output=True)["applied"])

    def test_retry_clears_only_affected_blocked_descendants_and_preserves_other_failure(self):
        tasks = [self.task("a"), self.task("b"), self.task("child", ["a"], "spectral"),
                 self.task("grandchild", ["child"], "geometry"),
                 self.task("join", ["a", "b"], "spectral"),
                 self.task("other", ["b"], "spectral"),
                 self.task("manual_hold", ["a"], "spectral")]
        self.write_manifest(tasks)
        item = self.make_intent(["a", "b"], registered=True)
        ledger = self.ledger()
        ledger["blocked"] = {"child": {"status": "blocked_parent", "parents": ["a"]},
                             "grandchild": {"status": "blocked_parent", "parents": ["child"]},
                             "join": {"status": "blocked_parent", "parents": ["a", "b"]},
                             "other": {"status": "blocked_parent", "parents": ["b"]},
                             "manual_hold": {"status": "manual_review", "parents": ["a"]}}
        atomic_json(self.ledger_path, ledger)
        self.accounting = self.arow("100_0", "TIMEOUT", item)
        report = self.retry(apply=True)
        blocked = self.ledger()["blocked"]
        self.assertEqual(set(blocked), {"join", "other", "manual_hold"})
        self.assertEqual(blocked["join"]["parents"], ["b"])
        self.assertEqual(blocked["other"], ledger["blocked"]["other"])
        self.assertEqual(blocked["manual_hold"], ledger["blocked"]["manual_hold"])
        self.assertEqual(self.ledger()["submitted"]["b"], ledger["submitted"]["b"])
        self.assertEqual(set(report["descendant_block_updates"]), {"child", "grandchild", "join"})

    def test_retry_batch_is_atomic_when_one_target_live(self):
        item = self.make_intent(["a", "b"], registered=True)
        self.accounting = self.arow("100_0", "FAILED", item) + "\n" + self.arow("100_1", "RUNNING", item)
        before = sha256_file(self.ledger_path)
        with self.assertRaises(recovery.RecoveryRefused):
            self.retry(keys=["a", "b"], apply=True)
        self.assert_unchanged(before)

    def test_retry_rechecks_queue_before_commit_when_job_is_requeued_during_accounting(self):
        item = self.make_intent(["a"], registered=True)
        self.accounting = self.arow("100_0", "FAILED", item)
        before = sha256_file(self.ledger_path)
        queue_calls = 0
        def changed_queue(arguments):
            nonlocal queue_calls
            if arguments[0] == "squeue":
                queue_calls += 1
                if queue_calls == 2:
                    self.queue = self.qrow("100_0", "PENDING", item)
            return self.slurm(arguments)
        with patch.object(recovery, "_slurm", side_effect=changed_queue):
            with self.assertRaisesRegex(recovery.RecoveryRefused, "became visible"):
                recovery.retry_failed_tasks(self.root, self.manifest, ["a"], reason="resource-only retry", apply=True)
        self.assertEqual(queue_calls, 2)
        self.assert_unchanged(before)

    def test_retry_rejects_unregistered_duplicate_wrong_index_and_uncertain_targets(self):
        item = self.make_intent(["a"], registered=True)
        for keys in (["b"], ["a", "a"]):
            with self.subTest(keys=keys):
                with self.assertRaises(recovery.RecoveryRefused):
                    self.retry(keys=keys, apply=True)
        ledger = self.ledger()
        ledger["submitted"]["a"]["job_id"] = "100_9"
        atomic_json(self.ledger_path, ledger)
        with self.assertRaisesRegex(recovery.RecoveryRefused, "index"):
            self.retry(apply=True)
        ledger["submitted"]["a"]["job_id"] = "100_0"
        ledger["intents"][item["token"]]["status"] = "dispatching"
        atomic_json(self.ledger_path, ledger)
        with self.assertRaisesRegex(recovery.RecoveryRefused, "uncertain"):
            self.retry(apply=True)
        self.assertFalse(self.calls)

    def test_read_only_query_failure_leaves_ledger_and_intents_intact(self):
        self.make_intent(["a"])
        before = sha256_file(self.ledger_path)
        with patch.object(recovery, "_slurm", side_effect=RuntimeError("accounting unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                recovery.reconcile_intents(self.root, self.manifest, apply=True)
        self.assert_unchanged(before)


if __name__ == "__main__":
    unittest.main()
