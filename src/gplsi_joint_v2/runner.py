"""One independently restartable scientific stage per invocation/Slurm element."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import os
import resource
import sys
import time
import traceback
import sqlite3

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import (atomic_json, atomic_npz, stage_lock, compatible_completed,
                        commit_stage, record_failed_attempt, sha256_file)
from .config import fingerprint
from .data import prepare_split, frequency_statistics
from .graph import build_graph


def stage_path(root, task):
    base = "data/interim/joint_v2/stages" if task["stage"] == "prepare" else "results/joint_v2/stages"
    return Path(root) / base / task["stage"] / task["task_id"]


def prepared_for(root, task):
    spec = task["spec"]
    return prepare_split(root, spec["dataset"], spec["outer_split_id"], spec["panel_requested"],
                         spec["retention"], spec["molecule_seed"], expected_split_hash=spec["split_hash"])


def _spectral_config(config):
    from .spectral import SpectralConfig
    cv = config["graph_cv"]
    return SpectralConfig(nfolds=cv["folds"], initial_grid=tuple(cv["initial_grid"]),
                          grid_growth=cv["growth"], extension_batch_size=cv["extension_batch"],
                          max_candidates=cv["max_candidates"], lambda_ceiling=cv["lambda_ceiling"],
                          plateau_relative_tolerance=cv["plateau_relative_tolerance"],
                          max_iterations=cv["max_spectral_iterations"],
                          reconstruction_tolerance=cv["spectral_tolerance"])


def _read_embedding(directory, control):
    from .spectral import Embedding, FeaturePlan
    meta = json.loads((directory / "spectral.json").read_text())
    name = "zero" if control == "lambda_zero" else "selected"
    if meta["blocks"].get(name) is None:
        raise RuntimeError("selected spectral fit unavailable; exact-zero control is a separate artifact")
    with np.load(directory / "spectral.npz", allow_pickle=False) as data:
        blockmeta = meta["blocks"][name]
        stored = meta.get("factor_aliases", {}).get(name, name)
        block = Embedding(data[f"{stored}_U"], data[f"{stored}_V"], data[f"{stored}_singular_values"],
                          data[f"{stored}_U_bar"] if f"{stored}_U_bar" in data else None,
                          blockmeta["lambda_value"], blockmeta["converged"],
                          blockmeta["iterations"], blockmeta["metadata"])
        plan = FeaturePlan(data["panel_indices"], data["retained_indices"], data["feature_weights"],
                           data["eta"], data["row_totals"], meta["features"])
    return block, plan, meta


def _parent_of(task, tasks, stages):
    matched = [tasks[p] for p in task["parents"] if tasks[p]["stage"] in stages]
    if len(matched) != 1:
        raise ValueError(f"expected one parent in {stages}, got {len(matched)}")
    return matched[0]


def _write_factors(directory, W, A, prepared, metadata, *, raw_W=None, vertices=None):
    from .likelihood import simplex_matrix
    arrays = {"W": simplex_matrix(W, "W")}
    metadata = {**metadata, "row_contract": prepared.split_metadata["row_contract"],
                "training_row_order_sha256": prepared.split_metadata["training_row_order_sha256"]}
    if A is not None:
        arrays.update(A=simplex_matrix(A, "A"), feature_ids=prepared.feature_ids.astype(str))
    if raw_W is not None:
        arrays["unconstrained_W"] = raw_W
    if vertices is not None:
        arrays["vertices"] = vertices
    atomic_npz(directory / "factors.npz", arrays)
    atomic_json(directory / "fit.json", metadata)
    return ["factors.npz", "fit.json"]


def _fit_competitor(task, prepared, G, config, spectral_directory=None):
    from .competitors import fit_topicscore, fit_lda, fit_kl_nmf, fit_graph_kl_nmf, fit_spatial_lda
    spec = task["spec"]
    method, K, seed = spec["method"], spec["K"], spec["estimator_seed"]
    D = prepared.train_fit
    budgets = config.get("competitor_budgets", {})
    if method == "topicscore_raw":
        return fit_topicscore(D, K, seed, max_iter=budgets.get("topicscore_iterations", 5000))
    if method == "topicscore_graph_denoised":
        block, plan, meta = _read_embedding(spectral_directory, "selected")
        mapping = {int(full): native for native, full in enumerate(prepared.feature_indices)}
        result = fit_topicscore(D, K, seed, spectral={"U": block.U, "V": block.V,
                 "singular_values": block.singular_values, "weights": plan.weights,
                 "retained_indices": np.asarray([mapping[int(i)] for i in plan.retained_indices])},
                 max_iter=budgets.get("topicscore_iterations", 5000))
        result.metadata.update(shared_spectral_converged=block.converged,
                               selected_lambda=block.lambda_value,
                               shared_spectral_runtime=meta["metadata"]["runtime_seconds"])
        result.converged = result.converged and block.converged
        return result
    if method == "lda":
        return fit_lda(D, K, seed, max_iter=budgets.get("lda_iterations", 50))
    if method == "kl_nmf":
        return fit_kl_nmf(D, K, seed, max_iter=budgets.get("nmf_iterations", 500))
    if method.endswith("_tuned"):
        from .tuning import fit_tuned_spatial_competitor
        return fit_tuned_spatial_competitor(method, prepared, G, K, seed, config)
    if method == "graph_kl_nmf_legacy_0p25":
        return fit_graph_kl_nmf(D, G, K, seed, penalty=.25, max_iter=budgets.get("nmf_iterations", 500))
    if method == "spatial_lda_legacy_0p25":
        return fit_spatial_lda(D, prepared.train_coords, prepared.train_graph_ids, K, seed, penalty=.25,
                               outer_iterations=budgets.get("spatial_lda_outer_iterations", 3),
                               lda_iterations=budgets.get("spatial_lda_inner_iterations", 5),
                               admm_iterations=budgets.get("spatial_lda_admm_iterations", 15))
    raise ValueError(f"unsupported named competitor: {method}")


def _recover(task, prepared, source_directory, directory, config):
    from .likelihood import recover_A_poisson
    spec = task["spec"]
    if spec.get("recovery") != "A_full_Pois":
        raise ValueError("reported profiles require A_full_Pois recovery; A_current is validation-only")
    with np.load(source_directory / "factors.npz", allow_pickle=False) as factors:
        W = factors["W"]
    source_metadata = json.loads((source_directory / "fit.json").read_text())
    if source_metadata["training_row_order_sha256"] != prepared.split_metadata["training_row_order_sha256"]:
        raise ValueError("saved W row contract does not match prepared training counts")
    reference = spec["vocabulary"] == "common_reference"
    counts = prepared.reference_train_fit if reference else prepared.train_fit
    gene_ids = prepared.reference_feature_ids if reference else prepared.feature_ids
    # The preserved Poisson solver starts from pooled fitting-count feature
    # frequencies and interiorizes once; this is not a prior or output floor.
    # No A_current fit or competitor-native profile enters reported recovery.
    fit = recover_A_poisson(W, counts, initial_A=None,
                            max_iter=config["solvers"]["A_max_iter"],
                            tolerance=config["solvers"]["A_tolerance"],
                            chunk_size=config["solvers"]["entry_chunk_size"])
    fit.diagnostics["initialization_audit"] = {
        "initializer": "pooled_fitting_count_feature_frequencies_then_strict_interiorization",
        "A_current_used": False, "competitor_native_A_used": False,
        "scoring_or_adaptation_counts_used": False}
    # Do not duplicate n-by-K W in native/reference children. Its exact parent
    # path and file checksum are the pairing invariant.
    atomic_npz(directory / "A.npz", {"A": fit.A_hat, "feature_ids": gene_ids.astype(str)})
    metadata = asdict(fit)
    metadata.pop("A_hat")
    metadata.update(W_parent=str(source_directory), W_parent_sha256=sha256_file(source_directory / "factors.npz"),
                    vocabulary=spec["vocabulary"], feature_order_sha256=fingerprint(gene_ids.astype(str).tolist()),
                    training_rows=len(W), training_molecules=int(counts.sum()),
                    profile_recovery="A_full_Pois", W_source_method=spec["method"],
                    reported_method=spec["method"] + "__A_full_Pois", W_train_refitted=False)
    atomic_json(directory / "fit.json", metadata)
    return ["A.npz", "fit.json"]


def _evaluate(task, prepared, G, source_directory, directory, config):
    from .likelihood import fold_in_fixed_A
    from .metrics import score_counts, biological_score_summary, spatial_metrics, topic_profile_metrics
    spec = task["spec"]
    if spec.get("recovery") != "A_full_Pois":
        raise ValueError("evaluation requires A_full_Pois; native and A_current profiles are not reportable")
    from .data import _verify_files
    cohort = Path(task["root"]) / "data/processed/joint_v2" / spec["dataset"]
    contract = json.loads((cohort / "contract.json").read_text())
    _verify_files(cohort, {"evaluation_annotations.parquet": contract["artifacts"]["evaluation_annotations.parquet"]})
    metadata = json.loads((source_directory / "fit.json").read_text())
    reference = spec["vocabulary"] == "common_reference"
    if metadata.get("profile_recovery") != "A_full_Pois" or not (source_directory / "A.npz").is_file():
        raise ValueError("evaluation parent must be a completed fixed-W Poisson profile recovery")
    with np.load(source_directory / "A.npz", allow_pickle=False) as f:
        A, gene_ids = f["A"], f["feature_ids"]
    W_directory = Path(metadata["W_parent"])
    if sha256_file(W_directory / "factors.npz") != metadata["W_parent_sha256"]:
        raise ValueError("paired W parent changed after A recovery")
    with np.load(W_directory / "factors.npz", allow_pickle=False) as f:
        W = f["W"]
    source_fit = json.loads((W_directory / "fit.json").read_text())
    if source_fit["training_row_order_sha256"] != prepared.split_metadata["training_row_order_sha256"]:
        raise ValueError("evaluation row-contract mismatch")
    source_converged = bool(source_fit.get("converged", source_fit.get("recovery_stable", False)))
    source_converged = source_converged and bool(source_fit.get("spectral_converged", True))
    recovery_converged = bool(metadata.get("converged", source_converged))
    expected_genes = prepared.reference_feature_ids if reference else prepared.feature_ids
    if not np.array_equal(gene_ids, expected_genes.astype(str)):
        raise ValueError("evaluation vocabulary mismatch")
    solver = config["solvers"]
    score_options = {"entry_chunk_size": solver["entry_chunk_size"], "floor": solver["scoring_floor_diagnostic"]}
    training_score = prepared.reference_train_score if reference else prepared.train_score
    within = score_counts(W, A, training_score, **score_options)
    summary = {"within_training_molecule_prediction": {**within["summary"],
               **biological_score_summary(within["per_row"], prepared.train_bio_ids)},
               "training_coverage": prepared.split_metadata,
               "fit_converged": source_converged and recovery_converged,
               "W_source_converged": source_converged, "A_recovery_converged": recovery_converged,
               "fit_status": metadata.get("status", metadata.get("vertex_status")),
               "vocabulary": spec["vocabulary"], "W_parent": str(W_directory),
               "profile_recovery": "A_full_Pois", "reported_method": spec["method"] + "__A_full_Pois",
               "W_parent_sha256": sha256_file(W_directory / "factors.npz"),
               "profile": topic_profile_metrics(A), "transfer": {}}
    vocabulary_hash = fingerprint(gene_ids.astype(str).tolist())
    summary["within_training_molecule_prediction"].update(
        evaluation_vocabulary_sha256=vocabulary_hash,
        evaluation_mask_sha256=fingerprint({"obs": prepared.train_ids.astype(str).tolist(),
                                            "mask": within["per_row"]["scored"].tolist()}))
    files = []
    summary["within_training_molecule_prediction"]["per_section"] = _section_sufficient_statistics(within["per_row"], prepared.train_obs)
    # W_train spatial metrics are calculated once for the immutable W parent,
    # never separately from an A-dependent W_test.
    diagnostic_directory = W_directory / "W_diagnostics"
    with stage_lock(diagnostic_directory):
        marker = diagnostic_directory / "spatial.json"
        if not marker.exists():
            atomic_json(marker, spatial_metrics(W, G, prepared.train_coords, prepared.train_graph_ids))
    summary["W_train_spatial"] = str(marker)
    transfers = prepared.reference_eval_sets if reference else prepared.eval_sets
    inferred = {}
    for name, transfer in transfers.items():
        started = time.perf_counter()
        fold = fold_in_fixed_A(A, transfer.adapt, max_iter=solver["foldin_max_iter"],
                               tolerance=solver["foldin_tolerance"], row_chunk_size=solver["row_chunk_size"],
                               entry_chunk_size=solver["entry_chunk_size"])
        elapsed = time.perf_counter() - started
        score = score_counts(fold.W, A, transfer.score, eligibility_mask=~transfer.zero_adaptation_mask,
                              inference_valid=fold.inference_valid, **score_options)
        summary["transfer"][name] = {**score["summary"],
               **biological_score_summary(score["per_row"], transfer.biological_ids),
               "adaptation_zero_observations": int(transfer.zero_adaptation_mask.sum()),
               "adaptation_zero_scoring_molecules": int(transfer.score[transfer.zero_adaptation_mask].sum()),
               "foldin": fold.metadata, "foldin_runtime_seconds": elapsed,
               "foldin_converged_observations": int(fold.row_converged.sum()),
               "evaluation_vocabulary_sha256": vocabulary_hash,
               "evaluation_mask_sha256": fingerprint({"obs": transfer.observation_ids.astype(str).tolist(),
                          "mask": (~transfer.zero_adaptation_mask & ~transfer.zero_score_mask).tolist()})}
        summary["transfer"][name]["per_section"] = _section_sufficient_statistics(score["per_row"], transfer.obs)
        summary["transfer"][name]["foldin_status_counts"] = {str(s): int(np.sum(fold.row_status == s)) for s in np.unique(fold.row_status)}
        summary["transfer"][name]["foldin_gap_quantiles"] = np.quantile(fold.normalized_gap[np.isfinite(fold.normalized_gap)], [0, .5, .9, .99, 1]).tolist() if np.isfinite(fold.normalized_gap).any() else []
        inferred[name] = fold
    from .metadata_evaluation import evaluate_metadata
    summary["external_biology"] = evaluate_metadata(Path(task["root"]), spec["dataset"], prepared, W, inferred,
                                                    reference=reference, seed=spec["estimator_seed"])
    atomic_json(directory / "evaluation.json", summary)
    return files + ["evaluation.json"]


def _section_sufficient_statistics(per_row, obs):
    """Small exact sums support section-balanced and biological score analyses.

    Counts/IDs/masks remain in shared input contracts. Per-observation score
    vectors and inferred W_test are intentionally not persisted per method.
    """
    output = []
    for (bio, section), group in obs.groupby(["bio_id", "section_id"], sort=True):
        rows = group.index.to_numpy()
        selected = rows[per_row["scored"][rows]]
        depth = float(per_row["depth"][selected].sum())
        ll = float(per_row["log_likelihood"][selected].sum())
        dev = float(per_row["deviance"][selected].sum())
        output.append({"biological_id": str(bio), "section_id": str(section),
                       "total_observations": len(rows), "scored_observations": len(selected),
                       "scored_molecules": depth, "log_likelihood": ll, "deviance": dev,
                       "deviance_per_molecule": dev / depth if depth else np.nan,
                       "log_likelihood_per_molecule": ll / depth if depth else np.nan,
                       "inference_failures": int((~per_row["inference_valid"][selected]).sum()),
                       "support_violation_entries": int(per_row["support_violation_entries"][selected].sum()),
                       "support_violation_molecules": float(per_row["support_violation_molecules"][selected].sum())})
    return output


def run_stage(root, manifest_path, task_id):
    root = Path(root)
    database = Path(manifest_path).parent / "tasks.sqlite"
    if database.exists():
        with sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True) as db:
            dag = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata WHERE key IN ('config','code_hash','source_hashes')")}
            task = json.loads(db.execute("SELECT value FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0])
            tasks = {task_id: task}
            for key in task["parents"]:
                tasks[key] = json.loads(db.execute("SELECT value FROM tasks WHERE task_id=?", (key,)).fetchone()[0])
    else:
        dag = json.loads(Path(manifest_path).read_text())
        tasks = {x["task_id"]: x for x in dag["tasks"]}
        task = tasks[task_id]
    task["root"] = str(root)
    directory = stage_path(root, task)
    with stage_lock(directory):
        started = time.perf_counter()
        from .resource_monitor import PeakMemory
        memory_monitor = PeakMemory().start()
        try:
            from .cli import source_identity
            from .manifests import validate_task_identity
            validate_task_identity(task, dag)
            for parent in task["parents"]:
                validate_task_identity(tasks[parent], dag)
            if source_identity()["source_tree_hash"] != dag["code_hash"]:
                raise ValueError("source snapshot changed; regenerate a new immutable manifest")
            actual_contract = json.loads((root / "data/processed/joint_v2" / task["dataset"] / "contract.json").read_text())
            if fingerprint(actual_contract) != fingerprint(dag["source_hashes"][task["dataset"]]):
                raise ValueError("cohort contract changed after immutable manifest generation")
            if compatible_completed(directory, task_id):
                memory_monitor.stop()
                return {"task_id": task_id, "state": "cache_hit", "path": str(directory)}
            if (directory / "complete.json").exists():
                raise RuntimeError("completed artifact failed integrity checks; preserve it and audit before any overwrite")
            for key in task["parents"]:
                if not compatible_completed(stage_path(root, tasks[key]), key):
                    raise RuntimeError(f"parent is not complete and checksum-compatible: {key}")
            spec, config = task["spec"], dag["config"]
            prepared = prepared_for(root, task)
            prepare_task = task if task["stage"] == "prepare" else _parent_of(task, tasks, {"prepare"})
            prepared.split_metadata["row_contract"] = str(stage_path(root, prepare_task) / "train_observations.parquet")
            if task["stage"] == "prepare":
                G, graphmeta = build_graph(prepared.train_coords, prepared.train_graph_ids, k=config["graph"]["neighbors"])
                sparse.save_npz(directory / "graph.npz", G, compressed=True)
                atomic_json(directory / "prepared.json", prepared.split_metadata)
                atomic_json(directory / "graph.json", graphmeta)
                prepared.train_obs.to_parquet(directory / "train_observations.parquet", index=False)
                frequencies = frequency_statistics(prepared.train_fit, prepared.train_bio_ids)
                atomic_json(directory / "frequencies.json", frequencies)
                from .reporting_data import frequency_report
                atomic_json(directory / "frequency_report.json", frequency_report(root, spec["dataset"], prepared))
                files = ["graph.npz", "prepared.json", "graph.json", "frequencies.json", "frequency_report.json", "train_observations.parquet"]
            else:
                parent = _parent_of(task, tasks, {"prepare"})
                G = sparse.load_npz(stage_path(root, parent) / "graph.npz")
                if task["stage"] == "spectral":
                    from .spectral import fit_shared_spectral, spectral_arrays
                    fitted = fit_shared_spectral(prepared.train_rank_counts, G, spec["K"], spec["preprocessing"],
                        graph_cv_seed=spec["graph_cv_seed"], estimator_seed=spec["estimator_seed"],
                        requested_panel=spec["panel_requested"] if spec["dataset"] == "visium_dlpfc" else None,
                        final_panel_indices=prepared.feature_indices, config=_spectral_config(config))
                    arrays = spectral_arrays(fitted)
                    arrays.pop("selected_U_bar", None)
                    arrays.pop("zero_U_bar", None)
                    aliases = {}
                    if fitted.selected is fitted.zero:
                        aliases["selected"] = "zero"
                        arrays = {key: value for key, value in arrays.items() if not key.startswith("selected_")}
                    atomic_npz(directory / "spectral.npz", arrays)
                    blocks = {}
                    for name in ("selected", "zero"):
                        block = getattr(fitted, name)
                        blocks[name] = None if block is None else {k: getattr(block, k)
                                     for k in ["lambda_value", "converged", "iterations", "metadata"]}
                    atomic_json(directory / "spectral.json", {"blocks": blocks, "features": fitted.features.metadata,
                                "metadata": fitted.metadata, "factor_aliases": aliases,
                                "centers_persisted": False,
                                "center_diagnostics": "exact scalars in block metadata",
                                "row_contract": prepared.split_metadata["row_contract"],
                                "training_row_order_sha256": prepared.split_metadata["training_row_order_sha256"]})
                    files = ["spectral.npz", "spectral.json"]
                elif task["stage"] == "geometry":
                    from .geometry import fit_geometry
                    parent = _parent_of(task, tasks, {"spectral"})
                    block, plan, spectral_meta = _read_embedding(stage_path(root, parent), spec["control"])
                    fit = fit_geometry(block, plan, G, hunter=spec["hunter"], estimator_seed=spec["estimator_seed"],
                                       family=spec["family"], max_svs_face_solves=config["solvers"]["svs_face_solve_budget"])
                    # Raw W summaries are saved in fit.json; selected U and
                    # vertices permit reconstruction without duplicating every
                    # n-by-K unconstrained membership array on disk.
                    files = _write_factors(directory, fit.W, None, prepared, fit.metadata, vertices=fit.vertices)
                elif task["stage"] == "diagnostics":
                    from .diagnostics import run_lambda_diagnostics
                    parent = _parent_of(task, tasks, {"spectral"})
                    spectral_directory = stage_path(root, parent)
                    zero, _, spectral_meta = _read_embedding(spectral_directory, "lambda_zero")
                    reused = {0.0: zero}
                    if spectral_meta["blocks"].get("selected") is not None:
                        selected, _, _ = _read_embedding(spectral_directory, "selected")
                        reused[selected.lambda_value] = selected
                    cv = spectral_meta["metadata"]["cv"]
                    grid = [r["lambda"] for r in cv["aggregate"]]
                    records = []
                    def checkpoint(record):
                        records.append(record)
                        atomic_json(directory / "diagnostic_progress.json", records)
                    diagnostic = run_lambda_diagnostics(prepared.train_rank_counts, G,
                        prepared.train_coords, prepared.train_graph_ids, spec["K"], spec["preprocessing"],
                        lambdas=grid, estimator_seed=spec["estimator_seed"],
                        requested_panel=spec["panel_requested"] if spec["dataset"] == "visium_dlpfc" else None,
                        final_panel_indices=prepared.feature_indices, config=_spectral_config(config),
                        hunters=config["hunters"], cv_metadata=cv, reused_embeddings=reused,
                        max_svs_face_solves=config["solvers"]["svs_face_solve_budget"], record_callback=checkpoint)
                    atomic_json(directory / "diagnostics.json", diagnostic)
                    files = ["diagnostics.json"]
                elif task["stage"] == "competitor":
                    spectral_parent = [tasks[p] for p in task["parents"] if tasks[p]["stage"] == "spectral"]
                    spectral_directory = stage_path(root, spectral_parent[0]) if spectral_parent else None
                    fit = _fit_competitor(task, prepared, G, config, spectral_directory)
                    files = _write_factors(directory, fit.W, None, prepared,
                                            {**fit.metadata, "status": fit.status, "converged": fit.converged,
                                             "native_A_role": "internal_method_fit_only_not_persisted_or_reported",
                                             "reported_profiles_require": "A_full_Pois"})
                elif task["stage"] in ("recovery", "reference_recovery"):
                    parent = _parent_of(task, tasks, {"geometry", "competitor"})
                    files = _recover(task, prepared, stage_path(root, parent), directory, config)
                elif task["stage"] == "evaluation":
                    parent = _parent_of(task, tasks, {"recovery", "reference_recovery"})
                    files = _evaluate(task, prepared, G, stage_path(root, parent), directory, config)
                else:
                    raise ValueError(f"unknown stage: {task['stage']}")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            process_memory = memory_monitor.stop()
            effective = {"task": task, "config": config, "prepared": prepared.split_metadata,
                         "runtime_seconds": time.perf_counter() - started,
                         "peak_rss_bytes": max(int(peak * (1 if sys.platform == "darwin" else 1024)),
                                               process_memory.get("peak_process_tree_rss_bytes") or 0),
                         "process_memory": process_memory,
                         "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                         "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                         "partition": os.environ.get("SLURM_JOB_PARTITION"),
                         "threads": {k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}}
            atomic_json(directory / "effective.json", effective)
            result = commit_stage(directory, task_id, effective, files + ["effective.json"])
            return {"task_id": task_id, "state": result["state"], "path": str(directory)}
        except Exception as exc:
            extra = {"task_id": task_id, "runtime_seconds": time.perf_counter() - started,
                     "traceback": traceback.format_exc(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "process_memory": memory_monitor.stop()}
            for field in ["cv_metadata", "diagnostics", "metadata"]:
                if hasattr(exc, field):
                    extra[field] = getattr(exc, field)
            record_failed_attempt(directory, exc, extra)
            raise


def audit_manifest(root, manifest_path):
    dag = json.loads(Path(manifest_path).read_text())
    rows = []
    for task in dag["tasks"]:
        directory = stage_path(root, task)
        complete = compatible_completed(directory, task["task_id"], verify_hashes=False)
        attempts = sorted((directory / "attempts").glob("*.json"))
        row = {"task_id": task["task_id"], "stage": task["stage"], "dataset": task["dataset"],
               "variant": task["variant"], "state": "complete" if complete else "failed" if attempts else "pending",
               "attempts": len(attempts), "path": str(directory)}
        if attempts and not complete:
            details = json.loads(attempts[-1].read_text())
            row.update(error_type=details["type"], error=details["error"])
        rows.append(row)
    output = Path(root) / "reports/joint_v2" / Path(manifest_path).parent.name
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output / "result_index.parquet", index=False)
    from collections import Counter
    summary = {"expected_stages": len(rows), "state_counts": dict(Counter(x["state"] for x in rows)),
               "result_index": str(output / "result_index.parquet"), "audit_runs_after_failures": True}
    atomic_json(output / "audit.json", summary)
    return summary
