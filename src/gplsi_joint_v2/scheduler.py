"""Bounded Slurm fan-out with exact parent dependencies and immutable arrays.

Only this experiment's job IDs are inspected; unrelated jobs are never changed.
The dispatch ledger is append-safe and limited by both global outstanding tasks
and per-array throttles. A controller can invoke dispatch periodically; an empty
queue is not completion unless the full expected manifest has terminal states.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import json
import os
import pwd
import subprocess
import time
import uuid
from functools import lru_cache

from .artifacts import atomic_json, stage_lock, compatible_completed
from .artifacts import sha256_file
from .runner import stage_path, run_stage


def _slurm(arguments):
    result = subprocess.run(arguments, text=True, capture_output=True, check=True)
    return result.stdout.strip()


@lru_cache(maxsize=2)
def _manifest_snapshot(path, size, modified_ns):
    # The manifest is immutable. Long-lived dispatchers avoid reparsing the
    # complete 200k-stage DAG every minute; changed stat identity invalidates.
    return json.loads(Path(path).read_text()), sha256_file(path)


def require_gate(root, dag, resources):
    gate = Path(root) / "reports/joint_v2" / "PRODUCTION_GATE.json"
    if not gate.exists():
        raise RuntimeError("production is gated: correctness, smoke, pilots and resource review not yet approved")
    record = json.loads(gate.read_text())
    if record.get("status") != "passed" or record.get("code_hash") != dag["code_hash"]:
        raise RuntimeError("production gate must pass for this exact source snapshot")
    if record.get("config_hash") != dag["config_hash"]:
        raise RuntimeError("production gate scientific configuration mismatch")
    if not all(record.get(key) for key in ["correctness_passed", "smoke_passed", "full_size_pilots_passed", "storage_passed"]):
        raise RuntimeError("incomplete production gate")
    if not resources:
        raise RuntimeError("measured per-platform/stage resource profiles required")


def _active_submission_ids(ledger):
    ids = {r["job_id"] for r in ledger["submitted"].values()}
    active = {}
    text = _slurm(["squeue", "-h", "-r", "-u", pwd.getpwuid(os.getuid()).pw_name, "-o", "%i|%T"])
    for line in text.splitlines():
        job_id, state = line.split("|", 1)
        if job_id in ids:
            active[job_id] = state
    return active


def dispatch(root, manifest_path, resource_file, *, gate_kind="production", maximum_new=None):
    root, manifest_path = Path(root), Path(manifest_path)
    stat = manifest_path.stat()
    dag, manifest_digest = _manifest_snapshot(str(manifest_path), stat.st_size, stat.st_mtime_ns)
    resources = json.loads(Path(resource_file).read_text())
    config = dag["config"]["resources"]
    if gate_kind == "production":
        require_gate(root, dag, resources)
    elif gate_kind != "pilot":
        raise ValueError("gate_kind must be production or pilot")
    directory = root / "slurm/joint_v2" / manifest_path.parent.name
    ledger_file = directory / "dispatch.json"
    with stage_lock(directory):
        ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {
            "manifest": str(manifest_path), "manifest_sha256": manifest_digest,
            "config_hash": dag["config_hash"], "code_hash": dag["code_hash"], "submitted": {}, "blocked": {}, "intents": {}}
        ledger.setdefault("intents", {})
        uncertain = [key for key, value in ledger["intents"].items() if value["status"] == "dispatching"]
        if uncertain:
            # Absence from squeue is not proof of non-submission: the job may
            # already be in sacct. Refuse duplicate work until reconciled.
            raise RuntimeError("uncertain sbatch intent requires squeue/sacct reconciliation before dispatch: " + ",".join(uncertain))
        if ledger["code_hash"] != dag["code_hash"]:
            raise ValueError("dispatch ledger source mismatch")
        if ledger["manifest"] != str(manifest_path):
            raise ValueError("dispatch ledger points to a different manifest")
        if ledger.get("manifest_sha256", manifest_digest) != manifest_digest:
            raise ValueError("dispatch ledger manifest changed in place")
        if ledger.get("config_hash", dag["config_hash"]) != dag["config_hash"]:
            raise ValueError("dispatch ledger scientific configuration mismatch")
        active = _active_submission_ids(ledger) if ledger["submitted"] else {}
        outstanding = sum(r["job_id"] in active for r in ledger["submitted"].values())
        slots = max(0, min(config["global_concurrency"], config["max_submitted_stage_tasks"]) - outstanding)
        if maximum_new is not None:
            slots = min(slots, maximum_new)
        tasks = {x["task_id"]: x for x in dag["tasks"]}
        complete = set(ledger.get("completed_artifact_keys", []))
        # Discover reused pilot artifacts once. Subsequent waves check only
        # newly submitted/retry-authorized work, not 200k filesystem paths.
        inspect_keys = set(tasks) if not ledger.get("initial_inventory_scanned") else set(ledger["submitted"]) - complete
        retry_keys = set(ledger.get("retry_authorizations", {}))
        inspect_keys.update(retry_keys)
        complete.difference_update(retry_keys)
        for key in inspect_keys:
            if compatible_completed(stage_path(root, tasks[key]), key, verify_hashes=key in retry_keys):
                complete.add(key)
        ledger["initial_inventory_scanned"] = True
        ledger["completed_artifact_keys"] = sorted(complete)
        groups = defaultdict(list)
        # The ordered DAG adds parents before their descendants. Groups contain
        # only configurations with exactly the same required parent job IDs.
        for key, task in tasks.items():
            if key in complete or key in ledger["submitted"] or key in ledger["blocked"]:
                continue
            dependencies, ready, failed = [], True, []
            for parent in task["parents"]:
                if parent in complete:
                    continue
                entry = ledger["submitted"].get(parent)
                if entry is None:
                    ready = False
                    if parent in ledger["blocked"]:
                        failed.append(parent)
                elif entry["job_id"] in active:
                    # Ready-only fan-out: exact parent artifact completion is
                    # required BEFORE submission. No dormant afterok children
                    # can consume the global slots after a parent fails.
                    ready = False
                else:
                    failed.append(parent)
                    ready = False
            if failed:
                ledger["blocked"][key] = {"status": "blocked_parent", "parents": failed}
            if not ready or slots <= 0:
                continue
            profile_key = task["dataset"] + "/" + task["stage"]
            resource_spec = resources.get(profile_key, resources.get("default"))
            if resource_spec is None:
                raise ValueError(f"no measured resource profile for {profile_key}")
            partition = resource_spec.get("partition", config["default_partition"])
            if config.get("partitions") and partition not in config["partitions"]:
                raise ValueError("partition not in audited authorized list")
            resource_key = json.dumps({**resource_spec, "partition": partition}, sort_keys=True)
            groups[(resource_key, tuple(sorted(dependencies)), task["stage"])].append(key)
            slots -= 1
        added = []
        for (resource_key, dependencies, stage), keys in groups.items():
            request = json.loads(resource_key)
            for begin in range(0, len(keys), config["array_cap"]):
                chunk = keys[begin:begin+config["array_cap"]]
                array_id = uuid.uuid4().hex
                array_manifest = directory / f"array_{array_id}.json"
                atomic_json(array_manifest, {"manifest": str(manifest_path), "task_ids": chunk,
                           "resource_request": request, "exact_parent_jobs": dependencies,
                           "satisfied_parent_artifact_keys": sorted({p for key in chunk for p in tasks[key]["parents"]})})
                cmd = ["sbatch", "--parsable", "--job-name=jv2_" + stage[:8] + "_" + array_id[:12],
                       "--comment=joint_v2:" + array_id,
                       "--cpus-per-task=" + str(request["cpus"]), "--mem=" + str(request["memory_gb"]) + "G",
                       "--time=" + request["time"], "--array=0-" + str(len(chunk)-1) + "%" + str(config["array_cap"]),
                       "--output=" + str(root / "logs/joint_v2" / f"{stage}_%A_%a.out")]
                if config.get("account"):
                    cmd.append("--account=" + config["account"])
                if request.get("partition"):
                    cmd.append("--partition=" + request["partition"])
                if dependencies:
                    cmd.append("--dependency=afterok:" + ":".join(dependencies))
                launcher = config.get("run_action") or str(Path(__file__).resolve().parents[2] / "joint_v2_scripts/run_action.sh")
                cmd += [str(launcher), "array", "--root", str(root), "--array-manifest", str(array_manifest)]
                ledger["intents"][array_id] = {"status": "dispatching", "created_at": time.time(),
                           "array_manifest": str(array_manifest), "task_ids": chunk, "command": cmd,
                           "unique_comment": "joint_v2:" + array_id}
                atomic_json(ledger_file, ledger)
                response = _slurm(cmd)
                last = response.splitlines()[-1].split(";")[0]
                if not last.isdigit():
                    raise RuntimeError(f"unrecognized sbatch response; inspect before retry: {response}")
                for index, key in enumerate(chunk):
                    entry = {"job_id": f"{last}_{index}", "submitted_at": time.time(),
                             "array_manifest": str(array_manifest), "resource_request": request,
                             "exact_parent_jobs": list(dependencies), "command": cmd}
                    ledger["submitted"][key] = entry
                    added.append({"task_id": key, **entry})
                ledger["intents"][array_id].update(status="submitted", job_id=last)
                # Record after EACH successful sbatch, not just the whole wave.
                atomic_json(ledger_file, ledger)
        atomic_json(ledger_file, ledger)
    return {"submitted": added, "active_before_dispatch": outstanding,
            "already_complete": len(complete), "expected_tasks": len(tasks), "ledger": str(ledger_file)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["dispatch", "run-array"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--gate-kind", choices=["pilot", "production"], default="production")
    parser.add_argument("--array-manifest", type=Path)
    args = parser.parse_args()
    os.environ["GPLSI_BENCHMARK_ROOT"] = str(args.root.resolve())
    if args.action == "run-array":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("scientific stages require a Slurm allocation")
        array = json.loads(args.array_manifest.read_text())
        task_id = array["task_ids"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
        result = run_stage(args.root, array["manifest"], task_id)
    else:
        result = dispatch(args.root, args.manifest, args.resources, gate_kind=args.gate_kind)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
