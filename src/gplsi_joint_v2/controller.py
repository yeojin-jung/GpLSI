"""A small Slurm-resident controller for bounded ready-only scientific waves.

This never approves production. It dispatches only the explicit pilot manifest
or an exact-source production manifest whose gates were independently passed.
All failures remain in the expected result index; no automatic scientific
parameter change or cancellation occurs.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import time

from .artifacts import atomic_json, stage_lock
from .scheduler import dispatch, _active_submission_ids
from .runner import audit_manifest


def run_controller(root, manifest, resources, *, gate_kind, interval_seconds=45, max_hours=12):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the persistent controller must run through Slurm, not on a login node")
    root, manifest = Path(root), Path(manifest)
    directory = root / "slurm/joint_v2" / manifest.parent.name
    directory.mkdir(parents=True, exist_ok=True)
    # Separate lock from dispatch's per-wave lock. Kernel unlocks after death.
    with stage_lock(directory / "controller_lock", blocking=False):
        start = time.monotonic()
        status_path = directory / "controller_status.json"
        cycle = 0
        while time.monotonic() - start < max_hours * 3600:
            if (directory / "STOP_CONTROLLER.json").exists():
                status = {"state": "stopped_by_request", "leaves_running_jobs_unchanged": True}
                atomic_json(status_path, status)
                return status
            result = dispatch(root, manifest, resources, gate_kind=gate_kind)
            cycle += 1
            ledger = json.loads((directory / "dispatch.json").read_text())
            live = _active_submission_ids(ledger) if ledger["submitted"] else {}
            completed = set(ledger.get("completed_artifact_keys", []))
            submitted = set(ledger["submitted"])
            blocked = set(ledger["blocked"])
            # A registered job absent from squeue is terminal for dispatch;
            # its accounting and scientific status are audited independently.
            accounted = submitted | blocked | completed
            status = {"state": "running", "cycle": cycle, "manifest": str(manifest),
                      "controller_job_id": os.environ["SLURM_JOB_ID"], "gate_kind": gate_kind,
                      "active_stage_jobs": len(live), "completed_artifacts": len(completed),
                      "registered_submissions": len(submitted), "blocked_descendants": len(blocked),
                      "expected_stages": result["expected_tasks"], "new_submissions": len(result["submitted"]),
                      "elapsed_seconds": time.monotonic() - start, "updated_at": time.time()}
            atomic_json(status_path, status)
            print(json.dumps(status), flush=True)
            if not live and len(accounted) == result["expected_tasks"]:
                audit = audit_manifest(root, manifest)
                status.update(state="terminal_manifest_audited", audit=audit,
                              scientific_success_not_implied_by_scheduler_completion=True)
                if gate_kind == "pilot":
                    from .pilots import assess_pilots
                    status["pilot_review"] = assess_pilots(root, manifest)
                atomic_json(status_path, status)
                return status
            time.sleep(interval_seconds)
        status = {"state": "controller_time_budget_reached", "manifest": str(manifest),
                  "stage_jobs_not_cancelled": True, "resume": "same immutable manifest and dispatch ledger"}
        atomic_json(status_path, status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--gate-kind", choices=["pilot", "production"], required=True)
    parser.add_argument("--interval-seconds", type=float, default=45)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    print(json.dumps(run_controller(args.root, args.manifest, args.resources, gate_kind=args.gate_kind,
                                   interval_seconds=args.interval_seconds, max_hours=args.max_hours), indent=2))


if __name__ == "__main__":
    main()
