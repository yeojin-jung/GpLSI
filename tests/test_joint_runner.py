"""Tiny actual stage handlers, compact persistence, scoring and reporting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from gplsi_joint_v2.artifacts import atomic_json, sha256_file
from gplsi_joint_v2.config import default_config, fingerprint
from gplsi_joint_v2.data import PreparedSplit, TransferSet
from gplsi_joint_v2.runner import run_stage, stage_path
from gplsi_joint_v2.results_reporting import report_results
from gplsi_joint_v2.manifests import scientific_task_key, validate_task_identity
from gplsi_joint_v2.splits import array_hash


def prepared_fixture():
    rng = np.random.default_rng(32)
    source_genes = np.array([f"gene_{j}" for j in range(5)])
    annotations = []
    def role_data(role, section_suffixes):
        rows, profiles = [], []
        for donor in ("d1", "d2", "d3"):
            for suffix in section_suffixes:
                section = donor + suffix
                for i in range(4):
                    row_id = f"{donor}/{section}/{role}/{i}"
                    rows.append({"obs_id": row_id, "bio_id": donor, "section_id": section,
                                 "graph_id": section, "x": i % 2, "y": i // 2,
                                 "source_row": len(annotations)})
                    label = "Layer1" if i < 2 else "Layer2"
                    annotations.append({"obs_id": row_id, "layer_guess": label,
                                        "layer_guess_reordered": label})
                    profiles.append([.6, .15, .1, .1, .05] if i < 2 else [.1, .1, .15, .6, .05])
        raw = np.vstack([rng.multinomial(100, p) for p in profiles])
        fit = rng.binomial(raw, .8)
        return sparse.csr_matrix(fit), sparse.csr_matrix(raw - fit), pd.DataFrame(rows)
    train_full, score_full, train_obs = role_data("train", ["a1", "b1"])
    native, reference = {}, {}
    for role, suffix in [("primary_test", "a2"), ("additional_test", "b2")]:
        adapt, score, obs = role_data(role, [suffix])
        for panel, output in [(np.arange(4), native), (np.arange(5), reference)]:
            a, s = adapt[:, panel], score[:, panel]
            output[role] = TransferSet(a, s, obs, obs.source_row.to_numpy(),
                                      np.asarray(a.sum(axis=1)).ravel() == 0,
                                      np.asarray(s.sum(axis=1)).ravel() == 0)
    split = {"split_id": "section_rotation_01", "protocol": "section_holdout",
             "assignments": [{"bio_id": d, "train_sections": [d + "a1", d + "b1"],
                              "primary_test": [d + "a2"], "additional_test": [d + "b2"]}
                             for d in ["d1", "d2", "d3"]]}
    metadata = {"cache_key": "tiny_synthetic_count_cache", "panel_requested": 4, "panel_actual": 4,
                "reference_panel_actual": 5, "outer_split": split,
                "feature_order_sha256": array_hash(source_genes[:4]),
                "training_row_order_sha256": array_hash(train_obs.obs_id)}
    prepared = PreparedSplit(train_fit=train_full[:, :4], train_score=score_full[:, :4],
        train_rank_counts=train_full, full_feature_ids=source_genes, feature_indices=np.arange(4),
        feature_ids=source_genes[:4], reference_indices=np.arange(5), reference_feature_ids=source_genes,
        reference_train_fit=train_full, reference_train_score=score_full,
        eval_sets=native, reference_eval_sets=reference, train_obs=train_obs,
        source_train_rows=train_obs.source_row.to_numpy(), native_train_mask=np.ones(len(train_obs), bool),
        ranking={}, split_metadata=metadata)
    return prepared, pd.DataFrame(annotations), split


def test_actual_runner_compact_factor_recovery_evaluation_report(tmp_path, monkeypatch):
    import gplsi_joint_v2.runner as runner
    import gplsi_joint_v2.cli as cli
    import gplsi_joint_v2.likelihood as likelihood
    prepared, annotations, split = prepared_fixture()
    root = tmp_path / "benchmark"
    monkeypatch.setattr(runner, "prepared_for", lambda root, task: prepared)
    monkeypatch.setattr(cli, "source_identity", lambda: {"source_tree_hash": "tiny_test_code"})
    def no_current_profile(*args, **kwargs):
        raise AssertionError("A_current must not be fitted by production stages")
    monkeypatch.setattr(likelihood, "recover_A_current", no_current_profile)
    annotation_path = root / "data/processed/joint_v2/visium_dlpfc/evaluation_annotations.parquet"
    annotation_path.parent.mkdir(parents=True)
    annotations.to_parquet(annotation_path, index=False)
    contract = {"dataset": "visium_dlpfc", "sources": [], "artifacts": {
        "evaluation_annotations.parquet": {"sha256": sha256_file(annotation_path),
                                           "size_bytes": annotation_path.stat().st_size}}}
    source_hashes = {"visium_dlpfc": contract}
    contract_path = annotation_path.parent / "contract.json"
    atomic_json(contract_path, contract)
    split_dir = root / "data/manifests/joint_v2/splits/visium_dlpfc"
    atomic_json(split_dir / "splits.json", [split])
    atomic_json(split_dir / "inventory.json", {"units": [{"bio_id": d} for d in ["d1", "d2", "d3"]]})
    config = default_config()
    config["bootstrap"]["draws"] = 30
    config["solvers"].update(A_max_iter=300, foldin_max_iter=300, entry_chunk_size=5, row_chunk_size=4)
    config["competitor_budgets"] = {"lda_iterations": 3}
    config["primary_K"]["visium_dlpfc"] = 2
    config["primary_panel"]["visium_dlpfc"] = 4
    config["seeds"]["estimator_seed"] = 29
    spec = {"dataset": "visium_dlpfc", "protocol": "section_holdout", "outer_split_id": split["split_id"],
            "panel_requested": 4, "K": 2, "retention": 1.0, "molecule_seed": 23, "estimator_seed": 29,
            "method": "lda"}
    tasks, aliases = [], {}
    def task(key, stage, parents, **extra):
        parents = sorted(aliases[parent] for parent in parents)
        task_spec = {**spec, **extra}
        identity = scientific_task_key(stage, task_spec, parents, code_hash="tiny_test_code",
                                       config=config, source_hashes=source_hashes)
        aliases[key] = identity
        row = {"task_id": identity, "stage": stage, "dataset": spec["dataset"], "spec": task_spec,
               "parents": parents, "base_ids": ["tiny_base"], "variant": key}
        tasks.append(row)
        return row
    task("prepare", "prepare", [])
    task("lda", "competitor", ["prepare"])
    task("poisson", "recovery", ["prepare", "lda"], recovery="A_full_Pois", vocabulary="native")
    task("reference", "reference_recovery", ["prepare", "lda"], recovery="A_full_Pois", vocabulary="common_reference")
    task("eval_poisson", "evaluation", ["prepare", "poisson"], recovery="A_full_Pois", vocabulary="native")
    task("eval_reference", "evaluation", ["prepare", "reference"], recovery="A_full_Pois", vocabulary="common_reference")
    manifest = root / "data/manifests/joint_v2/tiny_run/dag.json"
    dag = {"tasks": tasks, "config": config, "code_hash": "tiny_test_code", "source_hashes": source_hashes,
           "base_memberships": {"tiny_base": ["core"]}, "bases": [{"base_id": "tiny_base", **spec}]}
    assert all(validate_task_identity(node, dag) for node in tasks)
    atomic_json(manifest, dag)
    for node in tasks:
        assert run_stage(root, manifest, node["task_id"])["state"] == "complete"
    factor_path = stage_path(root, tasks[1]) / "factors.npz"
    digest = sha256_file(factor_path)
    with np.load(factor_path, allow_pickle=False) as saved:
        assert set(saved.files) == {"W"}
        w = saved["W"]
    assert run_stage(root, manifest, aliases["lda"])["state"] == "cache_hit"
    assert sha256_file(factor_path) == digest
    fits = [json.loads((stage_path(root, task) / "fit.json").read_text()) for task in tasks[2:4]]
    assert len({fit["W_parent_sha256"] for fit in fits}) == 1
    assert all(fit["profile_recovery"] == "A_full_Pois" and fit["W_train_refitted"] is False for fit in fits)
    assert all(fit["diagnostics"]["initialization_audit"]["A_current_used"] is False for fit in fits)
    for node in tasks[4:]:
        directory = stage_path(root, node)
        assert not list(directory.glob("W_transfer*"))
        assert not list(directory.glob("scores_*.parquet"))
        payload = json.loads((directory / "evaluation.json").read_text())
        assert payload["W_parent_sha256"] == digest
        assert payload["profile_recovery"] == "A_full_Pois"
        assert payload["reported_method"] == "lda__A_full_Pois"
        for endpoint in payload["transfer"].values():
            sections = endpoint["per_section"]
            assert len(sections) == 3
            assert all({"biological_id", "section_id", "scored_molecules", "log_likelihood", "deviance"} <= set(s) for s in sections)
            assert sum(s["scored_molecules"] for s in sections) == endpoint["scored_molecules"]
            assert endpoint["evaluation_vocabulary_sha256"]
            assert endpoint["evaluation_mask_sha256"]
        assert payload["external_biology"]["labels_used_during_fitting"] is False
    report = report_results(root, manifest, make_plots=True, run_profiles=True, reference_method="lda")
    assert report["stage_state_counts"] == {"complete": 6}
    assert report["expected_biological_score_rows_including_tier_membership"] == 24
    assert report["successful_biological_score_rows"] == 24
    summaries = pd.read_parquet(Path(report["tables_directory"]) / "method_summaries.parquet")
    assert summaries.loc[summaries.dataset == "visium_dlpfc", "bootstrap_standard_error"].isna().all()
    assert set(summaries.recovery) == {"A_full_Pois"}
    assert np.allclose(w.sum(axis=1), 1)
    figures = report["scientific_figures"]
    assert figures["failures"] == []
    assert len(figures["prespecified_training_maps"]) == 1
    assert len(figures["profile_tables"]) == 2
    for table in figures["profile_tables"]:
        assert set(pd.read_csv(table["path"]).recovery) == {"A_full_Pois"}
    assert figures["prespecified_training_maps"][0]["W_parent"] == str(stage_path(root, tasks[1]))
    assert figures["W_parent_maps_deduplicated_across_A_recoveries"] is True
    # Even an existing complete marker cannot bypass scientific input identity.
    atomic_json(contract_path, {**contract, "changed_after_manifest": True})
    with pytest.raises(ValueError, match="cohort contract changed"):
        run_stage(root, manifest, aliases["lda"])
    atomic_json(contract_path, contract)
    assert sha256_file(factor_path) == digest


@pytest.mark.parametrize("forbidden", ["A_current", "native"])
def test_runner_refuses_nonpoisson_recovery_and_evaluation(tmp_path, forbidden):
    from gplsi_joint_v2.runner import _recover, _evaluate
    task = {"spec": {"recovery": forbidden}}
    with pytest.raises(ValueError, match="A_full_Pois"):
        _recover(task, None, tmp_path, tmp_path, {})
    with pytest.raises(ValueError, match="A_full_Pois"):
        _evaluate(task, None, None, tmp_path, tmp_path, {})
