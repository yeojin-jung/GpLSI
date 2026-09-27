import json
from multiprocessing import Process
from pathlib import Path

import pytest

from gplsi_joint_v2.artifacts import (atomic_json, commit_stage, compatible_completed,
                                    record_failed_attempt, stage_key, stage_lock)
from gplsi_joint_v2.config import default_config
from gplsi_joint_v2.manifests import build_base_manifests, build_dag


def test_cache_requires_complete_and_compatible_content(tmp_path):
    spec = {"mask": "a", "vocabulary": "b", "solver": "c", "seed": 1}
    key = stage_key(spec, code_hash="code1", input_hashes=["input1"], parent_keys=[])
    atomic_json(tmp_path / "result.json", {"score": float("inf")})
    assert not compatible_completed(tmp_path, key)
    commit_stage(tmp_path, key, {}, ["result.json"])
    assert compatible_completed(tmp_path, key)
    assert json.loads((tmp_path / "result.json").read_text())["score"] == "Infinity"
    assert not compatible_completed(tmp_path, stage_key(spec, code_hash="code2", input_hashes=["input1"], parent_keys=[]))
    atomic_json(tmp_path / "result.json", {"score": 0})
    assert not compatible_completed(tmp_path, key)


def test_failed_attempt_does_not_make_completed_cache(tmp_path):
    record_failed_attempt(tmp_path, RuntimeError("interrupted"), {"task": "x"})
    assert not compatible_completed(tmp_path, "x")
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1


def _hold_lock(path):
    import time
    with stage_lock(path):
        atomic_json(Path(path) / "locked.json", {})
        time.sleep(30)


def test_killed_process_releases_lock(tmp_path):
    import time
    worker = Process(target=_hold_lock, args=(tmp_path,))
    worker.start()
    for _ in range(100):
        if (tmp_path / "locked.json").exists():
            break
        time.sleep(.02)
    worker.terminate()
    worker.join(5)
    with stage_lock(tmp_path, blocking=False):
        assert not compatible_completed(tmp_path, "none")


def test_stages_share_embedding_and_use_poisson_for_all_w_sources():
    config = default_config()
    config["K_values"] = [7]
    config["visium_panels"] = [500]
    config["primary_panel"]["visium_dlpfc"] = 500
    config["initialization_seeds"] = [config["seeds"]["estimator_seed"]]
    config["retention_secondary"] = []
    splits = [{"dataset": "visium_dlpfc", "protocol": "section_holdout", "split_id": "s1"}]
    manifests = build_base_manifests(splits, config)
    dag = build_dag(manifests, config, "commit", {"visium": "checksum"})
    assert len(dag["bases"]) == 1
    tasks = dag["tasks"]
    assert sum(t["stage"] == "prepare" for t in tasks) == 1
    assert sum(t["stage"] == "spectral" for t in tasks) == 4
    assert sum(t["stage"] == "geometry" for t in tasks) == 26
    assert sum(t["stage"] == "recovery" for t in tasks) == 34
    assert sum(t["stage"] == "reference_recovery" for t in tasks) == 34
    task_map = {t["task_id"]: t for t in tasks}
    for task in tasks:
        assert all(parent in task_map for parent in task["parents"])
        if task["stage"] == "recovery":
            assert sum(task_map[p]["stage"] in ("geometry", "competitor") for p in task["parents"]) == 1
        if task["stage"] in ("recovery", "reference_recovery", "evaluation"):
            assert task["spec"]["recovery"] == "A_full_Pois"
        if task["stage"] == "evaluation":
            assert sum(task_map[p]["stage"] in ("recovery", "reference_recovery") for p in task["parents"]) == 1
