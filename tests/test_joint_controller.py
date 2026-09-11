"""Slurm controller lifecycle tests use a fake clock and no scheduler actions."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from gplsi_joint_v2 import controller, pilots
from gplsi_joint_v2.artifacts import atomic_json, sha256_file


class FakeClock:
    def __init__(self, on_sleep=None):
        self.elapsed = 0.0
        self.sleeps = []
        self.on_sleep = on_sleep

    def monotonic(self):
        return self.elapsed

    def time(self):
        return 1789080000 + self.elapsed

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="joint_controller_test_")
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifests/pilot_v2/dag.json"
        self.resources = self.root / "resources.json"
        self.directory = self.root / "slurm/joint_v2/pilot_v2"
        self.ledger_path = self.directory / "dispatch.json"
        self.gate = self.root / "reports/joint_v2/PRODUCTION_GATE.json"
        atomic_json(self.manifest, {"identity": "unchanged_manifest"})
        atomic_json(self.resources, {"resource_profile": "unchanged"})
        self.write_ledger(submitted={"a": {"job_id": "100_0"}})
        self.clock = FakeClock()
        self.environment = patch.dict(os.environ, {"SLURM_JOB_ID": "9999"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def write_ledger(self, *, submitted=None, completed=(), blocked=None):
        atomic_json(self.ledger_path, {"submitted": submitted or {}, "blocked": blocked or {},
                                      "completed_artifact_keys": list(completed)})

    def run_controller(self, *, dispatch_result=None, live=None, clock=None, gate_kind="pilot", **kwargs):
        dispatch_result = dispatch_result or {"submitted": [], "expected_tasks": 2}
        if live is None:
            live = {"100_0": "RUNNING"}
        with (patch.object(controller, "time", clock or self.clock),
              patch.object(controller, "dispatch", return_value=dispatch_result) as dispatch,
              patch.object(controller, "_active_submission_ids", return_value=live) as active,
              patch.object(controller, "audit_manifest", return_value={"state_counts": {"failed": 1}}) as audit,
              patch.object(pilots, "assess_pilots", return_value={"production_approved": False}) as assess,
              redirect_stdout(StringIO())):
            result = controller.run_controller(self.root, self.manifest, self.resources,
                                               gate_kind=gate_kind, **kwargs)
        return result, dispatch, active, audit, assess

    def test_rejects_non_slurm_execution_before_any_dispatch(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            with patch.object(controller, "dispatch") as dispatch:
                with self.assertRaisesRegex(RuntimeError, "Slurm, not on a login node"):
                    controller.run_controller(self.root, self.manifest, self.resources, gate_kind="pilot")
                dispatch.assert_not_called()
        self.assertFalse((self.directory / "controller_status.json").exists())

    def test_stop_marker_prevents_dispatch_and_keeps_registered_jobs_unchanged(self):
        atomic_json(self.directory / "STOP_CONTROLLER.json", {"reason": "user stop"})
        before = sha256_file(self.ledger_path)
        result, dispatch, active, audit, assess = self.run_controller()
        self.assertEqual(result["state"], "stopped_by_request")
        self.assertTrue(result["leaves_running_jobs_unchanged"])
        for call in (dispatch, active, audit, assess):
            call.assert_not_called()
        self.assertEqual(sha256_file(self.ledger_path), before)
        self.assertFalse(self.gate.exists())

    def test_stop_between_waves_does_not_cancel_or_submit_another_wave(self):
        before = sha256_file(self.ledger_path)
        clock = FakeClock(lambda: atomic_json(self.directory / "STOP_CONTROLLER.json", {}))
        result, dispatch, active, audit, assess = self.run_controller(clock=clock, interval_seconds=2)
        self.assertEqual(result["state"], "stopped_by_request")
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(active.call_count, 1)
        audit.assert_not_called()
        assess.assert_not_called()
        self.assertEqual(sha256_file(self.ledger_path), before)

    def test_time_budget_returns_without_cancelling_live_jobs_or_approving(self):
        before = sha256_file(self.ledger_path)
        result, dispatch, active, audit, assess = self.run_controller(max_hours=4/3600, interval_seconds=2)
        self.assertEqual(result["state"], "controller_time_budget_reached")
        self.assertTrue(result["stage_jobs_not_cancelled"])
        self.assertEqual(result["resume"], "same immutable manifest and dispatch ledger")
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(active.call_count, 2)
        self.assertEqual(self.clock.sleeps, [2, 2])
        audit.assert_not_called()
        assess.assert_not_called()
        self.assertEqual(sha256_file(self.ledger_path), before)
        self.assertFalse(self.gate.exists())

    def test_zero_budget_performs_no_dispatch(self):
        result, dispatch, active, audit, assess = self.run_controller(max_hours=0)
        self.assertEqual(result["state"], "controller_time_budget_reached")
        for call in (dispatch, active, audit, assess):
            call.assert_not_called()
        self.assertFalse(self.clock.sleeps)

    def test_terminal_pilot_audits_failed_and_blocked_tasks_without_approval(self):
        self.write_ledger(submitted={"a": {"job_id": "100_0"}, "failed": {"job_id": "100_1"}},
                          completed=["a"], blocked={"child": {"status": "blocked_parent", "parents": ["failed"]}})
        result, dispatch, active, audit, assess = self.run_controller(
            live={}, dispatch_result={"submitted": [], "expected_tasks": 3})
        self.assertEqual(result["state"], "terminal_manifest_audited")
        self.assertTrue(result["scientific_success_not_implied_by_scheduler_completion"])
        self.assertEqual(result["registered_submissions"], 2)
        self.assertEqual(result["blocked_descendants"], 1)
        self.assertEqual(result["completed_artifacts"], 1)
        audit.assert_called_once_with(self.root, self.manifest)
        assess.assert_called_once_with(self.root, self.manifest)
        self.assertFalse(result["pilot_review"]["production_approved"])
        self.assertFalse(self.gate.exists())
        self.assertFalse(self.clock.sleeps)
        self.assertEqual(json.loads((self.directory / "controller_status.json").read_text()), result)

    def test_empty_queue_is_not_terminal_when_expected_tasks_remain_unaccounted(self):
        self.write_ledger(submitted={})
        result, dispatch, active, audit, assess = self.run_controller(live={}, max_hours=2/3600, interval_seconds=2)
        self.assertEqual(result["state"], "controller_time_budget_reached")
        dispatch.assert_called_once_with(self.root, self.manifest, self.resources, gate_kind="pilot")
        active.assert_not_called()
        audit.assert_not_called()
        assess.assert_not_called()
        self.assertFalse(self.gate.exists())

    def test_production_terminal_audits_but_does_not_write_or_reapprove_gate(self):
        atomic_json(self.gate, {"status": "passed", "reviewed_independently": True})
        before = sha256_file(self.gate)
        self.write_ledger(submitted={}, completed=["a"])
        result, dispatch, active, audit, assess = self.run_controller(
            live={}, dispatch_result={"submitted": [], "expected_tasks": 1}, gate_kind="production")
        self.assertEqual(result["state"], "terminal_manifest_audited")
        dispatch.assert_called_once_with(self.root, self.manifest, self.resources, gate_kind="production")
        audit.assert_called_once_with(self.root, self.manifest)
        assess.assert_not_called()
        self.assertEqual(sha256_file(self.gate), before)


if __name__ == "__main__":
    unittest.main()
