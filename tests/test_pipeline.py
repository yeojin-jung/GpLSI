"""End-to-end checks of the experiment pipeline on a small synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gplsi.generate_topic_model import generate_data, generate_weights_edge
from gplsi.pipeline import config as config_module
from gplsi.pipeline import expand_tasks, load_config, load_rows, run_task
from gplsi.pipeline import refit as refit_module
from gplsi.pipeline import runner as runner_module
from gplsi.pipeline.datasets import TaskData
from gplsi.real_data import RealDataBundle


def _synthetic_task_data(config, task) -> TaskData:
    np.random.seed(int(task["seed"]))
    coords, W, A, X = generate_data(200, 90, 12, int(task["K"]), 0.05, 6)
    weights, edge_df = generate_weights_edge(coords, 5, 0.1)
    rng = np.random.default_rng(int(task["seed"]))
    counts = np.vstack([rng.multinomial(200, row / row.sum()) for row in X]).astype(np.int64)
    train = rng.binomial(counts, 0.8).astype(np.int64)
    train[train.sum(axis=1) == 0, 0] = 1
    heldout = np.maximum(counts - train, 0)
    lengths = train.sum(axis=1).astype(float)
    bundle = RealDataBundle(
        dataset="synthetic",
        counts=train,
        frequencies=train / lengths[:, None],
        document_lengths=lengths,
        feature_ids=np.array([f"f{j}" for j in range(train.shape[1])]),
        observation_ids=np.array([f"o{i}" for i in range(train.shape[0])]),
        group_ids=np.array(["g"] * train.shape[0]),
        edge_df=edge_df[["src", "tgt", "weight"]],
        weights=weights,
        coordinates=np.asarray(coords[["x", "y"]], dtype=float),
    ).validate()
    return TaskData(bundle=bundle, heldout=heldout, feature_names=bundle.feature_ids.astype(str))


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(runner_module, "prepare_task_data", _synthetic_task_data)
    monkeypatch.setattr(refit_module, "prepare_task_data", _synthetic_task_data)


def _config(tmp_path: Path, **overrides) -> dict:
    config = {
        "name": "synthetic",
        "output_root": str(tmp_path / "results"),
        "dataset": {"name": "synthetic"},
        "grid": {"K": [3], "seed": [7]},
        "heldout_fraction": 0.2,
        "spectral": {"grid_len": 3, "maxiter": 3, "nfolds": 3, "lambda_selection_mode": "cv_once"},
        "gplsi": [
            {"geometry": "document_U", "preprocessings": ["P0_raw", "P2_ke_weighted"],
             "vertex_hunters": ["spa_current", "svs"]},
            {"geometry": "word_Z", "preprocessings": ["P0_raw"], "vertex_hunters": ["spa_current"]},
        ],
        "A_recoveries": ["A_current", "A_full_L2"],
        "recovery": {"poisson_max_iter": 50},
        "baselines": ["plsi", "topicscore_raw", "topicscore_graph_denoised", "lda"],
    }
    config.update(overrides)
    return config


def _rows(tmp_path: Path, name: str = "synthetic") -> dict[str, dict]:
    return {row["method"]: row for row in load_rows(tmp_path / "results" / name, flatten=False)}


def test_every_requested_method_gets_one_row_with_scores(tmp_path, synthetic):
    config = _config(tmp_path)
    task_dir = run_task(config, expand_tasks(config)[0])
    rows = _rows(tmp_path)
    # 2 document preprocessings x 2 hunters + 1 anchor, each with 2 A recoveries;
    # pLSI with 2 recoveries; 3 single baselines.
    assert len(rows) == 5 * 2 + 2 + 3
    assert all(row["status"] == "ok" for row in rows.values()), {
        name: row["failure_reason"] for name, row in rows.items() if row["status"] != "ok"
    }
    for row in rows.values():
        arrays = np.load(row["artifacts"]["estimate"]["path"])
        assert np.allclose(arrays["W_hat"].sum(axis=1), 1)
        assert np.allclose(arrays["A_hat"].sum(axis=1), 1)
        assert "heldout_poisson_deviance_per_molecule" in row["heldout_metrics"]
        assert "reconstruction_fro" in row["diagnostics"]
    # All A recoveries of one geometry share W.
    geometry = [r for r in rows.values() if r["vertex_hunter"] == "svs" and r["preprocessing"] == "P0_raw"]
    assert len({r["W_fit_id"] for r in geometry}) == 1
    Ws = [np.load(r["artifacts"]["estimate"]["path"])["W_hat"] for r in geometry]
    assert np.array_equal(*Ws)
    assert (task_dir / "fit_rows.csv").exists() and (task_dir / "data.npz").exists()
    assert rows["document_gplsi__document_U__svs__P0_raw__A_current"]["spectral"]["lambda_selection_mode"] == "cv_once"


def test_rerun_reuses_rows_and_changed_settings_recompute(tmp_path, synthetic, monkeypatch):
    config = _config(tmp_path, baselines=[], gplsi=[
        {"geometry": "document_U", "preprocessings": ["P0_raw"], "vertex_hunters": ["spa_current"]}])
    task = expand_tasks(config)[0]
    run_task(config, task)
    calls = []
    original = runner_module.fit_spectral_block
    monkeypatch.setattr(runner_module, "fit_spectral_block", lambda *a, **k: calls.append(1) or original(*a, **k))
    run_task(config, task)
    assert calls == []
    changed = _config(tmp_path, baselines=[], gplsi=config["gplsi"],
                      spectral={**config["spectral"], "grid_len": 2})
    run_task(changed, task)
    assert calls == [1]


def test_unavailable_hunters_and_failures_are_explicit_rows(tmp_path, synthetic):
    config = _config(
        tmp_path,
        baselines=["spatial_lda"],
        spatial_lda_parameters={"not_a_parameter": 1},
        unavailable=[{"K": [3], "vertex_hunters": ["svs"], "reason": "declared too slow"}],
    )
    run_task(config, expand_tasks(config)[0])
    rows = _rows(tmp_path)
    svs = [row for row in rows.values() if row["vertex_hunter"] == "svs"]
    assert svs and all(row["status"] == "not_available" for row in svs)
    assert svs[0]["failure_reason"] == "declared too slow"
    assert rows["spatial_lda__not_applicable__not_applicable__not_applicable__native"]["status"] == "failed"


def test_parts_split_one_task_directory(tmp_path, synthetic):
    config = _config(tmp_path, parts={
        "gplsi": {"baselines": []},
        "baselines": {"gplsi": [], "baselines": ["lda"]},
        "more_seeds": {"grid": {"seed": [8]}, "gplsi": [], "baselines": ["lda"]},
    })
    tasks = expand_tasks(config)
    assert [(t["part"], t["seed"]) for t in tasks] == [("gplsi", 7), ("baselines", 7), ("more_seeds", 8)]
    for task in tasks:
        run_task(config, task)
    run_dir = tmp_path / "results" / "synthetic"
    assert sorted(p.name for p in run_dir.iterdir()) == ["synthetic__K3__seed7", "synthetic__K3__seed8"]
    assert (run_dir / "synthetic__K3__seed7" / "task__gplsi.json").exists()
    frame = load_rows(run_dir)
    assert set(frame.loc[frame.seed == 8, "estimator_family"]) == {"lda"}
    assert "lda" in set(frame.loc[frame.seed == 7, "estimator_family"])


def test_refit_keeps_W_and_records_its_source(tmp_path, synthetic):
    config = _config(tmp_path, baselines=["plsi"], gplsi=[
        {"geometry": "document_U", "preprocessings": ["P0_raw"], "vertex_hunters": ["spa_current"]}])
    task = expand_tasks(config)[0]
    run_task(config, task)
    settings = {"poisson_start": "pooled", "poisson_max_iter": 30, "poisson_tolerance": 1e-8}
    refit_module.refit_task(config, task, "A_full_Pois", settings)
    rows = _rows(tmp_path)
    refit = rows["document_gplsi__document_U__spa_current__P0_raw__A_full_Pois"]
    source = rows["document_gplsi__document_U__spa_current__P0_raw__A_current"]
    assert refit["refit_source"] == source["fit_id"]
    assert refit["A_recovery_info"]["diagnostics"]["warm_start"] == "pooled_feature_frequencies"
    W_refit = np.load(refit["artifacts"]["estimate"]["path"])["W_hat"]
    assert np.array_equal(W_refit, np.load(source["artifacts"]["estimate"]["path"])["W_hat"])
    assert "plsi__document_U__not_applicable__not_applicable__A_full_Pois" in rows


def test_config_inheritance_replaces_lists_and_parts(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({
        "spectral": {"grid_len": 29, "maxiter": 50},
        "baselines": ["lda", "plsi"],
        "parts": {"a": {}, "b": {}},
    }))
    (tmp_path / "child.json").write_text(json.dumps({
        "extends": "base.json",
        "dataset": {"name": "crc"},
        "grid": {"K": [2, 3], "seed": [1]},
        "spectral": {"grid_len": 5},
        "baselines": ["lda"],
        "parts": {"c": {}},
    }))
    config = load_config(tmp_path / "child.json")
    assert config["name"] == "child"
    assert config["spectral"] == {"grid_len": 5, "maxiter": 50}
    assert config["baselines"] == ["lda"]
    assert list(config["parts"]) == ["c"]
    assert [t["K"] for t in expand_tasks(config)] == [2, 3]
    assert config_module.task_id(config, {"K": 2, "seed": 1}) == "crc__K2__seed1"
    with pytest.raises(ValueError, match="unknown grid axes"):
        expand_tasks({**config, "grid": {"K": [2], "seed": [1], "alpha": [0.1]}})


@pytest.mark.parametrize("path", sorted(Path(__file__).resolve().parents[1].glob("configs/[!s]*/*.json")))
def test_shipped_configs_expand(path):
    if path.name == "base.json":
        return
    config = load_config(path)
    tasks = expand_tasks(config)
    assert tasks
    for task in tasks:
        resolved = config_module.config_for_task(config, task)
        blocks = resolved.get("gplsi", [])
        assert blocks or resolved.get("baselines"), f"{path.name}: task {task} runs nothing"
        for block in blocks:
            assert block["geometry"] in runner_module.GEOMETRY_FAMILY
        assert set(resolved.get("baselines", [])) <= set(runner_module.BASELINES)
