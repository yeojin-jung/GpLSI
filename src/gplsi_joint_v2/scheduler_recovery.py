"""Conservative, manually invoked recovery of this benchmark's dispatch ledger.

The Slurm operations here are read-only (squeue and sacct). ``apply=False`` also
leaves the ledger unchanged. ``apply=True`` changes only the locked dispatch
ledger; it never submits, cancels, deletes, or changes scientific task IDs.

An absent job is not evidence that sbatch failed. Interrupted submission intents
are registered only after a unique comment/name identifies exactly one array
and every expected array index has been observed. Accounting comments are not
always retained (Slurm AccountingStoreFlags must include job_comment), so the
exact, UUID-bearing job name is an audited fallback. We use sacct JobID, *not*
JobIDRaw: only the former retains ArrayJobID_ArrayTaskID notation.

See https://slurm.schedmd.com/sacct.html and
https://slurm.schedmd.com/squeue.html for the read-only fields and array flags.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import os
import pwd
import re
import subprocess
import time

from .artifacts import atomic_json, sha256_file, stage_lock
from .runner import stage_path


_ARRAY_TASK = re.compile(r"^(\d+)_(\d+)$")
_ARRAY_BASE = re.compile(r"^(\d+)(?:_|$)")
_FAILURE_STATES = frozenset({"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED",
                             "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"})


class RecoveryRefused(RuntimeError):
    """No ledger mutation occurred; ``report`` contains the audit evidence."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def _slurm(arguments):
    # No shell and no write-capable Slurm command are accepted by this module.
    if not arguments or arguments[0] not in {"squeue", "sacct"}:
        raise ValueError("recovery is restricted to read-only Slurm queries")
    return subprocess.run(arguments, text=True, capture_output=True, check=True).stdout.strip()


def _username():
    return pwd.getpwuid(os.getuid()).pw_name


def _parse_rows(output, *, accounting):
    fields = ("job_id", "state", "job_name", "comment", "submit", "start", "end", "exit_code") if accounting else (
        "job_id", "state", "job_name", "comment")
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        # Unrelated queue comments may contain a pipe; keep the whole comment
        # rather than making unrelated user jobs break this scoped recovery.
        parts = line.split("|") if accounting else line.split("|", 3)
        if accounting and len(parts) == len(fields) + 1 and not parts[-1].strip():
            parts.pop()
        if len(parts) != len(fields):
            raise RecoveryRefused("malformed Slurm output; refusing to infer job identity")
        row = dict(zip(fields, (part.strip() for part in parts)))
        row["source"] = "sacct" if accounting else "squeue"
        rows.append(row)
    return rows


def _queue_rows():
    # Query the current user's queue once and filter locally. squeue -j with a
    # purged historical ID can fail the whole request, even for valid IDs.
    return _parse_rows(_slurm(["squeue", "-h", "-r", "-u", _username(),
                              "-o", "%i|%T|%j|%k"]), accounting=False)


def _accounting_rows(*, earliest, names=None, job_ids=None):
    if bool(names) == bool(job_ids):
        raise ValueError("accounting lookup needs exactly one nonempty identity filter")
    # sacct defaults to today's records. Use a bounded explicit local-time
    # window covering submission, including a day for clock/zone uncertainty.
    start = datetime.fromtimestamp(max(0.0, float(earliest) - 86400)).strftime("%Y-%m-%dT%H:%M:%S")
    arguments = ["sacct", "-n", "-P", "-X", "--array", "-u", _username(), "-S", start,
                 "-o", "JobID%64,State%64,JobName%128,Comment%256,Submit,Start,End,ExitCode"]
    arguments += ["--name", ",".join(names)] if names else ["-j", ",".join(job_ids)]
    return _parse_rows(_slurm(arguments), accounting=True)


def _load(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path)
    directory = root / "slurm/joint_v2" / manifest_path.parent.name
    ledger_path = directory / "dispatch.json"
    if not ledger_path.is_file():
        raise RecoveryRefused("no dispatch ledger exists for this manifest")
    dag = json.loads(manifest_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    expected = {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                "code_hash": dag["code_hash"], "config_hash": dag["config_hash"]}
    for field, value in expected.items():
        if ledger.get(field) != value:
            raise RecoveryRefused(f"dispatch ledger {field} mismatch; scientific identities must not change")
    tasks = {task["task_id"]: task for task in dag["tasks"]}
    if len(tasks) != len(dag["tasks"]):
        raise RecoveryRefused("duplicate scientific task IDs in manifest")
    for task in tasks.values():
        if any(parent not in tasks for parent in task["parents"]):
            raise RecoveryRefused("manifest contains an unknown parent")
    return ledger_path, dag, tasks, ledger


def _selected_ids(values, available, *, kind):
    chosen = list(available) if values is None else list(values)
    if len(set(chosen)) != len(chosen):
        raise RecoveryRefused(f"duplicate {kind} IDs are not permitted")
    if any(key not in available for key in chosen):
        raise RecoveryRefused(f"unknown {kind} ID; exact registered IDs are required")
    return chosen


def _array_record(array_path, *, ledger_path, manifest_path, tasks):
    path = Path(array_path)
    if not path.resolve().is_relative_to(ledger_path.parent.resolve()) or not path.is_file():
        raise RecoveryRefused("array manifest must exist inside this dispatch ledger directory")
    record = json.loads(path.read_text())
    task_ids = record.get("task_ids", [])
    if record.get("manifest") != str(manifest_path):
        raise RecoveryRefused("array manifest points to a different scientific manifest")
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(key not in tasks for key in task_ids):
        raise RecoveryRefused("array manifest has missing, duplicate, or foreign task IDs")
    if not isinstance(record.get("resource_request"), dict):
        raise RecoveryRefused("array manifest resource request is missing")
    return record


def _intent_identity(intent_id, intent, *, ledger_path, manifest_path, tasks):
    if not re.fullmatch(r"[0-9a-f]{32}", intent_id):
        raise RecoveryRefused("submission intent is not a complete UUID token")
    expected_comment = "joint_v2:" + intent_id
    if intent.get("unique_comment") != expected_comment:
        raise RecoveryRefused("submission intent comment does not match its unique token")
    array = _array_record(intent["array_manifest"], ledger_path=ledger_path,
                          manifest_path=manifest_path, tasks=tasks)
    if array["task_ids"] != intent.get("task_ids"):
        raise RecoveryRefused("intent and array manifest task order differ")
    stages = {tasks[key]["stage"] for key in array["task_ids"]}
    if len(stages) != 1:
        raise RecoveryRefused("submission array unexpectedly spans stages")
    expected_name = "jv2_" + next(iter(stages))[:8] + "_" + intent_id[:12]
    command = intent.get("command", [])
    if command.count("--comment=" + expected_comment) != 1 or command.count("--job-name=" + expected_name) != 1:
        raise RecoveryRefused("intent command lacks its exact unique name/comment")
    array_options = [part for part in command if part.startswith("--array=")]
    expected_array = rf"--array=0-{len(array['task_ids'])-1}%[1-9][0-9]*"
    if len(array_options) != 1 or not re.fullmatch(expected_array, array_options[0]):
        raise RecoveryRefused("intent array range does not match its exact task count")
    if command.count("--array-manifest") != 1:
        raise RecoveryRefused("intent command must name exactly one array manifest")
    pos = command.index("--array-manifest")
    if pos + 1 >= len(command) or command[pos + 1] != intent["array_manifest"]:
        raise RecoveryRefused("intent command references a different array manifest")
    return expected_name, expected_comment, array


def _match_intent(intent_id, intent, identity, rows):
    name, comment, array = identity
    matched, conflicts = [], []
    for row in rows:
        # Job steps are not distinct array tasks, even if accounting repeats
        # their parent name/comment. Never use them to satisfy index coverage.
        if "." in row["job_id"]:
            continue
        if row["comment"] == comment:
            if row["job_name"] != name:
                conflicts.append({**row, "reason": "matching token but unexpected job name"})
            else:
                matched.append({**row, "identity_match": "comment_and_name"})
        elif row["job_name"] == name:
            if row["comment"] in {"", "(null)", "N/A", "None"}:
                matched.append({**row, "identity_match": "exact_unique_name_comment_not_stored"})
            else:
                conflicts.append({**row, "reason": "matching unique name but conflicting comment"})
    bases = {match.group(1) for row in matched
             if (match := _ARRAY_BASE.match(row["job_id"])) is not None}
    report = {"intent_id": intent_id, "task_ids": array["task_ids"], "unique_name": name,
              "unique_comment": comment, "evidence": matched, "conflicts": conflicts}
    if conflicts:
        return {**report, "status": "unresolved", "reason": "conflicting Slurm identity evidence"}
    if len(bases) != 1:
        return {**report, "status": "unresolved", "reason": "no unique array ID proven",
                "observed_array_ids": sorted(bases)}
    base = next(iter(bases))
    indices = {int(match.group(2)) for row in matched
               if (match := _ARRAY_TASK.fullmatch(row["job_id"])) is not None and match.group(1) == base}
    expected = set(range(len(array["task_ids"])))
    if indices != expected:
        return {**report, "status": "unresolved", "reason": "exact array task count/index coverage not proven",
                "array_job_id": base, "expected_indices": sorted(expected), "observed_indices": sorted(indices)}
    return {**report, "status": "reconciled", "array_job_id": base,
            "task_job_ids": {key: f"{base}_{index}" for index, key in enumerate(array["task_ids"])}}


def reconcile_intents(root, manifest_path, *, intent_ids=None, apply=False):
    """Prove uncertain sbatch acceptance; optionally register only proven jobs.

    All selected intents must be proven before any ledger update is committed.
    No match, ambiguous arrays, missing indices, or conflicting existing task
    submissions leave the intent uncertain. In particular, no match never
    authorizes resubmission. Non-applied reports can be saved by a caller.
    """
    root, manifest_path = Path(root), Path(manifest_path)
    directory = root / "slurm/joint_v2" / manifest_path.parent.name
    with stage_lock(directory):
        ledger_path, dag, tasks, ledger = _load(root, manifest_path)
        uncertain = {key: value for key, value in ledger.get("intents", {}).items()
                     if value.get("status") == "dispatching"}
        chosen = _selected_ids(intent_ids, uncertain, kind="uncertain intent")
        report = {"ledger": str(ledger_path), "applied": False, "code_hash": dag["code_hash"],
                  "config_hash": dag["config_hash"], "proposals": [], "unresolved": []}
        if not chosen:
            return {**report, "status": "nothing_to_reconcile"}
        identities = {key: _intent_identity(key, uncertain[key], ledger_path=ledger_path,
                                           manifest_path=manifest_path, tasks=tasks) for key in chosen}
        if len({identity[0] for identity in identities.values()}) != len(identities):
            raise RecoveryRefused("selected intents have colliding unique job names; comment-only manual audit required")
        rows = _queue_rows() + _accounting_rows(
            earliest=min(uncertain[key]["created_at"] for key in chosen),
            names=[identities[key][0] for key in chosen])
        planned_jobs = {key: value["job_id"] for key, value in ledger["submitted"].items()}
        planned_owners = {value["job_id"]: key for key, value in ledger["submitted"].items()}
        for key in chosen:
            proposal = _match_intent(key, uncertain[key], identities[key], rows)
            if proposal["status"] == "reconciled":
                for task_id, job_id in proposal["task_job_ids"].items():
                    current = ledger["submitted"].get(task_id)
                    if task_id in planned_jobs and planned_jobs[task_id] != job_id:
                        proposal.update(status="unresolved", reason="task already registered to a different job")
                    elif job_id in planned_owners and planned_owners[job_id] != task_id:
                        proposal.update(status="unresolved", reason="array index already registered to a different task")
                    elif current is not None and current.get("array_manifest") != uncertain[key]["array_manifest"]:
                        proposal.update(status="unresolved", reason="existing registration references a different array manifest")
                if proposal["status"] == "reconciled":
                    planned_jobs.update(proposal["task_job_ids"])
                    planned_owners.update({job_id: task_id for task_id, job_id in proposal["task_job_ids"].items()})
                report["proposals" if proposal["status"] == "reconciled" else "unresolved"].append(proposal)
            else:
                report["unresolved"].append(proposal)
        report["status"] = "blocked" if report["unresolved"] else "ready"
        if apply and report["unresolved"]:
            raise RecoveryRefused("uncertain intents remain unproven; no ledger changes made", report)
        if apply:
            now = time.time()
            for proposal in report["proposals"]:
                key = proposal["intent_id"]
                intent, array = ledger["intents"][key], identities[key][2]
                for task_id, job_id in proposal["task_job_ids"].items():
                    # Preserve an identical partial registration rather than
                    # losing any older timestamps or audit fields it contains.
                    if task_id not in ledger["submitted"]:
                        ledger["submitted"][task_id] = {
                            "job_id": job_id, "submitted_at": intent["created_at"],
                            "array_manifest": intent["array_manifest"],
                            "resource_request": array["resource_request"],
                            "exact_parent_jobs": array.get("exact_parent_jobs", []),
                            "command": intent["command"], "reconciled_at": now,
                            "reconciled_intent": key}
                intent.update(status="submitted", job_id=proposal["array_job_id"],
                              reconciled_at=now, reconciliation_evidence=proposal["evidence"])
            ledger.setdefault("recovery_history", []).append({"action": "reconcile_intents", "at": now,
                                                              "intent_ids": chosen})
            atomic_json(ledger_path, ledger)
            report.update(applied=True, status="applied")
        return report


def _artifact_evidence(directory, task_id):
    directory = Path(directory)
    marker = directory / "complete.json"
    result = {"directory": str(directory), "marker": str(marker), "status": "missing", "problems": []}
    if not marker.exists():
        return result
    result["marker_sha256"] = sha256_file(marker)
    try:
        record = json.loads(marker.read_text())
        if record.get("cache_key") != task_id or record.get("state") != "complete":
            result["problems"].append("completion marker is incompatible with this exact task")
        files = record.get("files")
        if not isinstance(files, dict):
            result["problems"].append("completion file inventory is malformed")
        else:
            for relative, digest in files.items():
                path = directory / relative
                if Path(relative).is_absolute() or not path.resolve().is_relative_to(directory.resolve()):
                    result["problems"].append({"file": relative, "problem": "path is outside stage directory"})
                elif not path.is_file():
                    result["problems"].append({"file": relative, "problem": "missing file"})
                elif sha256_file(path) != digest:
                    result["problems"].append({"file": relative, "problem": "checksum mismatch"})
    except (ValueError, TypeError, OSError, AttributeError) as exc:
        result["problems"].append("unreadable/malformed completion record: " + type(exc).__name__)
    result["status"] = "corrupt" if result["problems"] else "complete"
    return result


def _state(value):
    # sacct may append " by <UID>" to CANCELLED; widths above prevent normal
    # state truncation. An unknown or truncated state fails closed below.
    return value.split()[0] if value.split() else ""


def _registration_identity(entry):
    command = entry.get("command", [])
    names = [part.split("=", 1)[1] for part in command if part.startswith("--job-name=")]
    comments = [part.split("=", 1)[1] for part in command if part.startswith("--comment=")]
    if len(names) != 1 or len(comments) != 1 or not re.fullmatch(r"joint_v2:[0-9a-f]{32}", comments[0]):
        raise RecoveryRefused("registered submission lacks an unambiguous benchmark job identity")
    if not re.fullmatch(r"jv2_[a-z_]{1,8}_" + comments[0].split(":")[1][:12], names[0]):
        raise RecoveryRefused("registered job name/comment tokens do not agree")
    return names[0], comments[0]


def _affected_tasks(tasks, origins):
    children = defaultdict(set)
    for key, task in tasks.items():
        for parent in task["parents"]:
            children[parent].add(key)
    affected, frontier = set(origins), list(origins)
    while frontier:
        for child in children[frontier.pop()]:
            if child not in affected:
                affected.add(child)
                frontier.append(child)
    return affected


def _descendant_block_updates(tasks, blocked, retried, *, affected=None):
    affected = _affected_tasks(tasks, retried) if affected is None else affected
    # Remove only blocked_parent causes coming through the recovered branch.
    # A join with a separate failed parent remains blocked for that parent.
    changes = {}
    for key in affected - set(retried):
        if key not in blocked or blocked[key].get("status") != "blocked_parent":
            continue
        entry = blocked[key]
        remaining = [parent for parent in entry.get("parents", []) if parent not in affected]
        if remaining != entry.get("parents", []):
            changes[key] = {**entry, "parents": remaining} if remaining else None
    return changes


def retry_failed_tasks(root, manifest_path, task_ids, *, reason,
                       allow_corrupt_output=False, allow_missing_output=False, apply=False):
    """Make exact, terminal failed tasks eligible for resource-only dispatch.

    Source/config hashes, scientific task IDs, old attempt metadata and files
    are preserved. Only Slurm terminal-failure evidence permits the ordinary
    path. COMPLETED without an output requires explicit missing-output
    authorization. Any existing completion marker, corrupt or valid, is
    protected; repairing/quarantining an artifact is outside this workflow.
    ``allow_corrupt_output`` is retained for API compatibility but does not
    authorize replacing a completion marker. Valid completed descendants are
    also protected, even when their parent's Slurm allocation eventually
    failed. This function never retries an uncertain intent or changes
    submitted descendant jobs.

    The scheduler must checksum-validate IDs in ``retry_authorizations`` when
    scanning completion; this function also removes these targets from its
    cached completed-key inventory. Files are never deleted or quarantined.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise RecoveryRefused("an explicit audit reason is required for every retry")
    root, manifest_path = Path(root), Path(manifest_path)
    directory = root / "slurm/joint_v2" / manifest_path.parent.name
    with stage_lock(directory):
        ledger_path, dag, tasks, ledger = _load(root, manifest_path)
        chosen = _selected_ids(task_ids, ledger["submitted"], kind="registered task")
        if not chosen:
            raise RecoveryRefused("resource retry requires at least one exact registered task ID")
        uncertain_tasks = {key for value in ledger.get("intents", {}).values()
                           if value.get("status") == "dispatching" for key in value["task_ids"]}
        if set(chosen) & uncertain_tasks:
            raise RecoveryRefused("target task belongs to an uncertain intent; reconcile it first")
        target_jobs = {}
        identities = {}
        for key in chosen:
            if key not in tasks:
                raise RecoveryRefused("registered task is not in this exact manifest")
            entry = ledger["submitted"][key]
            identities[key] = _registration_identity(entry)
            if key in ledger.get("blocked", {}):
                raise RecoveryRefused("target is both submitted and blocked; inspect inconsistent ledger before retry")
            match = _ARRAY_TASK.fullmatch(entry["job_id"])
            if match is None:
                raise RecoveryRefused("registered job ID is not an exact array task")
            array = _array_record(entry["array_manifest"], ledger_path=ledger_path,
                                  manifest_path=manifest_path, tasks=tasks)
            index = int(match.group(2))
            if index >= len(array["task_ids"]) or array["task_ids"][index] != key:
                raise RecoveryRefused("registered array index does not map to the requested task")
            if entry["job_id"] in target_jobs:
                raise RecoveryRefused("two scientific tasks are registered to the same array index")
            target_jobs[entry["job_id"]] = key
        queue = [row for row in _queue_rows() if row["job_id"] in target_jobs]
        accounting = [row for row in _accounting_rows(
            earliest=min(ledger["submitted"][key]["submitted_at"] for key in chosen),
            job_ids=list(target_jobs)) if row["job_id"] in target_jobs]
        report = {"ledger": str(ledger_path), "applied": False, "code_hash": dag["code_hash"],
                  "config_hash": dag["config_hash"], "reason": reason.strip(),
                  "proposals": [], "refused": []}
        for key in chosen:
            entry = ledger["submitted"][key]
            queue_evidence = [row for row in queue if row["job_id"] == entry["job_id"]]
            evidence = [row for row in accounting if row["job_id"] == entry["job_id"]]
            artifact = _artifact_evidence(stage_path(root, tasks[key]), key)
            proposal = {"task_id": key, "previous_submission": deepcopy(entry),
                        "queue_evidence": queue_evidence, "accounting_evidence": evidence,
                        "artifact": artifact}
            states = {_state(row["state"]) for row in evidence}
            submit_times = {row["submit"] for row in evidence if row["submit"] not in {"", "Unknown"}}
            name, comment = identities[key]
            identity_conflict = any(row["job_name"] != name or row["comment"] not in {
                comment, "", "(null)", "N/A", "None"} for row in evidence)
            refusal = None
            if queue_evidence:
                refusal = "registered job is still present in squeue; never duplicate live work"
            elif artifact["status"] == "complete":
                refusal = "checksum-valid successful artifact is protected"
            elif identity_conflict:
                refusal = "accounting identity conflicts with this registered submission"
            elif len(states) != 1 or len(submit_times) > 1:
                refusal = "no unambiguous terminal accounting state is proven"
            elif artifact["status"] == "corrupt":
                refusal = "existing corrupt completion marker requires separate artifact repair review; resource retry cannot overwrite it"
            elif states <= _FAILURE_STATES:
                pass
            elif states == {"COMPLETED"} and artifact["status"] == "missing" and allow_missing_output:
                pass
            elif states == {"COMPLETED"}:
                refusal = "Slurm completed successfully; missing output needs explicit authorization"
            else:
                refusal = "registered job is live or has an unknown/nonterminal accounting state"
            if refusal:
                report["refused"].append({**proposal, "reason": refusal})
            else:
                report["proposals"].append(proposal)
        affected = _affected_tasks(tasks, chosen)
        protected_descendants = []
        if not report["refused"]:
            for key in sorted(affected - set(chosen)):
                path = stage_path(root, tasks[key])
                if (path / "complete.json").is_file():
                    descendant = _artifact_evidence(path, key)
                    if descendant["status"] == "complete":
                        protected_descendants.append({"task_id": key, "artifact": descendant})
            if protected_descendants:
                report["refused"].append({"task_ids": chosen, "reason": "checksum-valid completed descendants are protected",
                                          "completed_descendants": protected_descendants})
        report["status"] = "blocked" if report["refused"] else "ready"
        if report["refused"]:
            if apply:
                raise RecoveryRefused("retry refused; no ledger changes made", report)
            return report
        changes = _descendant_block_updates(tasks, ledger.get("blocked", {}), chosen, affected=affected)
        report["descendant_block_updates"] = changes
        if apply:
            # Recheck just before committing: a manually requeued allocation
            # may have become live while accounting was queried. The ledger
            # lock prevents our own dispatcher racing this operation; external
            # Slurm changes cannot be locked, so narrow that window explicitly.
            recheck = [row for row in _queue_rows() if row["job_id"] in target_jobs]
            if recheck:
                report.update(status="blocked", precommit_queue_evidence=recheck)
                raise RecoveryRefused("target job became visible before retry commit; no ledger changes made", report)
            for proposal in report["proposals"]:
                latest = _artifact_evidence(stage_path(root, tasks[proposal["task_id"]]), proposal["task_id"])
                if latest["status"] in {"complete", "corrupt"}:
                    report.update(status="blocked", precommit_artifact=latest)
                    raise RecoveryRefused("artifact changed before retry commit; no ledger changes made", report)
            now = time.time()
            for proposal in report["proposals"]:
                key = proposal["task_id"]
                history = {**proposal, "action": "resource_only_retry_authorized", "at": now,
                           "reason": reason.strip(), "code_hash": dag["code_hash"],
                           "config_hash": dag["config_hash"], "allow_corrupt_output": bool(allow_corrupt_output),
                           "allow_missing_output": bool(allow_missing_output)}
                ledger.setdefault("attempt_history", {}).setdefault(key, []).append(history)
                ledger.setdefault("retry_authorizations", {})[key] = {
                    "at": now, "previous_job_id": proposal["previous_submission"]["job_id"],
                    "reason": reason.strip(), "code_hash": dag["code_hash"], "config_hash": dag["config_hash"],
                    "verify_completion_hashes": True, "artifact_status": proposal["artifact"]["status"]}
                del ledger["submitted"][key]
            old_completed = set(ledger.get("completed_artifact_keys", []))
            ledger["completed_artifact_keys"] = sorted(old_completed - set(chosen))
            report["completed_cache_keys_removed"] = sorted(old_completed & set(chosen))
            for key, replacement in changes.items():
                if replacement is None:
                    del ledger["blocked"][key]
                else:
                    ledger["blocked"][key] = replacement
            ledger.setdefault("recovery_history", []).append({"action": "resource_only_retry", "at": now,
                                                              "task_ids": chosen, "reason": reason.strip(),
                                                              "descendant_block_updates": changes})
            atomic_json(ledger_path, ledger)
            report.update(applied=True, status="applied")
        return report
