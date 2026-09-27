#!/usr/bin/env python3
"""Restartable runner for audited CRC, spleen, and What's Cooking experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any
from time import perf_counter

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gplsi.baselines import fit_lda, fit_spatial_lda  # noqa: E402
from gplsi.gplsi import GpLSI  # noqa: E402
from gplsi.real_data import RealDataBundle, load_real_data  # noqa: E402
from gplsi.real_experiment import (  # noqa: E402
    PREPROCESSING_SPECS,
    fit_diagnostics,
    fit_geometry,
    fit_graph_topicscore,
    fit_spectral_block,
    heldout_count_diagnostics,
    recover_A_for_geometry,
)
from gplsi.topicscore import fit_topicscore_raw  # noqa: E402
from gplsi.recovery import refit_A_full_poisson  # noqa: E402


SCHEMA_VERSION = 2
EXECUTION_SHARD_BASELINES = {
    "baseline_plsi": "plsi",
    "baseline_topicscore_raw": "topicscore_raw",
    "baseline_lda": "lda",
    "baseline_spatial_lda": "spatial_lda",
}
PRIMARY_K = {
    "stanford_crc_codex": 6,
    "mouse_spleen_codex": 5,
    "whats_cooking": 7,
}
CODE_FILES = [
    REPO_ROOT / "src/gplsi/__init__.py",
    REPO_ROOT / "src/gplsi/utils.py",
    REPO_ROOT / "src/gplsi/gplsi.py",
    REPO_ROOT / "src/gplsi/graphSVD.py",
    REPO_ROOT / "src/gplsi/preprocessing.py",
    REPO_ROOT / "src/gplsi/accelerated_palm_aa.py",
    REPO_ROOT / "src/gplsi/vertex_hunting.py",
    REPO_ROOT / "src/gplsi/anchor_word.py",
    REPO_ROOT / "src/gplsi/recovery.py",
    REPO_ROOT / "src/gplsi/real_data.py",
    REPO_ROOT / "src/gplsi/real_experiment.py",
    REPO_ROOT / "src/gplsi/baselines.py",
    REPO_ROOT / "src/gplsi/topicscore.py",
    Path(__file__).resolve(),
] + sorted((REPO_ROOT / "utils/spatial_lda").glob("*.py"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _code_hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): _hash_file(path) for path in CODE_FILES}


def _runtime_provenance() -> dict[str, Any]:
    import cvxpy
    import numpy
    import pandas
    import pycvxcluster
    import scipy
    import sklearn

    package_root = Path(pycvxcluster.__file__).resolve().parent
    pycvxcluster_hashes = {
        str(path.relative_to(package_root)): _hash_file(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    distributions = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "pycvxcluster"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "python": platform.python_version(),
        "versions": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "cvxpy": cvxpy.__version__,
            **distributions,
        },
        "pycvxcluster_package_root": str(package_root),
        "pycvxcluster_source_hashes": pycvxcluster_hashes,
    }


def _task_id(dataset: str, group: str | None, K: int, seed: int) -> str:
    parts = [dataset]
    if group:
        parts.append(group)
    parts.extend((f"K{K}", f"seed{seed}"))
    return "__".join(parts).replace("/", "-")


def _for_seed_policy(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Restrict non-default seeds to explicitly declared estimator families."""

    configured_seeds = [int(value) for value in config.get("seeds", [])]
    default_seeds = {
        int(value) for value in config.get("default_seeds", configured_seeds)
    }
    if int(seed) in default_seeds:
        return config

    allowed = {
        str(estimator)
        for estimator, seeds in config.get(
            "seeds_by_estimator_family", {}
        ).items()
        if int(seed) in {int(value) for value in seeds}
    }
    if not allowed:
        raise ValueError(
            f"seed {seed} is not planned for any estimator under this config"
        )

    shard = config.get("execution_shard")
    if shard is not None:
        baseline = EXECUTION_SHARD_BASELINES.get(str(shard))
        if baseline not in allowed:
            raise ValueError(
                f"seed {seed} is only planned for {sorted(allowed)}, "
                f"not shard {shard!r}"
            )

    output = json.loads(json.dumps(config))
    output["record_published_placeholder"] = False
    output["preprocessings"] = []
    output["graph_topicscore_preprocessings"] = []
    output["vertex_hunters"] = []
    output["baselines"] = [
        baseline
        for baseline in output.get("baselines", [])
        if str(baseline) in allowed
    ]
    return output


def _method_id(fields: dict[str, Any]) -> str:
    keys = (
        "estimator_family",
        "spectral_geometry",
        "vertex_hunter",
        "preprocessing",
        "A_recovery",
    )
    readable = "__".join(str(fields[key]) for key in keys)
    suffix = _hash_bytes(_canonical_json({key: fields[key] for key in keys}).encode())[:10]
    return f"{readable}__{suffix}".replace("/", "-")


def _W_fit_id(fields: dict[str, Any]) -> str:
    keys = (
        "estimator_family",
        "spectral_geometry",
        "vertex_hunter",
        "preprocessing",
    )
    return "W__" + _hash_bytes(
        _canonical_json({key: fields[key] for key in keys}).encode()
    )[:16]


def _safe_error(error: BaseException) -> tuple[str, str]:
    reason = f"{type(error).__name__}: {error}"
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return reason, trace


def _artifact_valid(row_path: Path, task_hash: str) -> dict[str, Any] | None:
    if not row_path.exists():
        return None
    try:
        row = json.loads(row_path.read_text())
        if row.get("task_config_hash") != task_hash:
            return None
        for artifact in row.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if not path.exists() or _hash_file(path) != artifact["sha256"]:
                return None
        return row
    except (OSError, ValueError, KeyError):
        return None


def _save_npz(path: Path, **arrays: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(path.resolve()), "sha256": _hash_file(path)}


def _vertex_artifact_arrays(vertex) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in (
        "selected_observation_indices",
        "selected_center_indices",
        "centers",
        "pseudo_points",
        "projected_points",
        "cluster_assignments",
        "neighborhood_sizes",
        "retained_point_indices",
        "discarded_point_indices",
        "initialization_observation_indices",
        "observation_weights",
        "archetype_to_data_weights",
        "archetype_projection_points",
        "initialization_vertices",
        "initialization_weights",
        "objective_trace",
        "reconstruction_trace",
        "penalty_trace",
        "relative_step_trace",
        "gamma_h_trace",
        "gamma_w_trace",
    ):
        value = getattr(vertex, name, None)
        if value is not None:
            output[name] = np.asarray(value)
    neighbors = getattr(vertex, "neighbor_indices", None)
    if neighbors is not None:
        sizes = np.asarray([len(value) for value in neighbors], dtype=np.int64)
        output["neighbor_indices_offsets"] = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(sizes))
        )
        output["neighbor_indices_flat"] = (
            np.concatenate(neighbors).astype(np.int64, copy=False)
            if sizes.sum()
            else np.array([], dtype=np.int64)
        )
    return output


def _write_row(task_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = task_dir / "rows" / f"{row['fit_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True, default=_json_default) + "\n")
    return row


def _base_row(
    bundle: RealDataBundle,
    *,
    canonical_bundle: RealDataBundle,
    K: int,
    seed: int,
    task_hash: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    canonical_hashes = canonical_bundle.hashes()
    execution_hashes = bundle.hashes()
    canonical_contract_hash = _hash_bytes(_canonical_json(canonical_hashes).encode())
    execution_data_hash = _hash_bytes(_canonical_json(execution_hashes).encode())
    family = fields["estimator_family"]
    geometry = fields["spectral_geometry"]
    row = {
        "schema_version": SCHEMA_VERSION,
        "dataset": bundle.dataset,
        "data_version": canonical_bundle.metadata.get("canonical_scope"),
        "biological_group": bundle.metadata.get("biological_group"),
        "canonical_contract_hash": canonical_contract_hash,
        "execution_data_hash": execution_data_hash,
        "input_hash": execution_data_hash,
        "graph_hash": _hash_bytes(
            _canonical_json(
                {
                    "edges": execution_hashes["edges_sha256"],
                    "weights": execution_hashes["edge_weights_sha256"],
                }
            ).encode()
        ),
        "graph_setting": canonical_bundle.metadata.get("graph", "spatial_coordinate_graph"),
        "experiment_family": "real_data_anchor_word_gplsi",
        "run_id": task_hash,
        "subset_of_canonical": bool(bundle.metadata.get("subset_of_canonical", False)),
        "subset_seed": bundle.metadata.get("subset_seed"),
        "split_id": (
            "full"
            if bundle.n == canonical_bundle.n and not bundle.metadata.get("count_thinning")
            else (
                f"{bundle.metadata.get('subset_strategy', 'connected')}_n{bundle.n}"
                + (
                    f"_thin{bundle.metadata.get('thinning_test_fraction')}"
                    if bundle.metadata.get("count_thinning")
                    else ""
                )
            )
        ),
        "group_split_unit": bundle.metadata.get("independent_unit"),
        "subset_group_counts": bundle.metadata.get("subset_group_counts"),
        "thinning_test_fraction": bundle.metadata.get("thinning_test_fraction"),
        "seed": int(seed),
        "K": int(K),
        "K_role": "primary" if int(K) == PRIMARY_K.get(bundle.dataset) else "published_K_sensitivity",
        "n": bundle.n,
        "p_canonical": canonical_bundle.p,
        "p_execution": bundle.p,
        "N_definition": "mean of per-document canonical count totals",
        "N_mean": bundle.N_mean,
        "N_summary": {
            "min": float(np.min(bundle.document_lengths)),
            "mean": float(np.mean(bundle.document_lengths)),
            "median": float(np.median(bundle.document_lengths)),
            "max": float(np.max(bundle.document_lengths)),
        },
        "graph_svd_version": (
            "iterative_two_step"
            if family in {
                "document_gplsi",
                "anchor_feature_gplsi",
                "topicscore_graph_denoised",
            }
            else "not_applicable"
        ),
        "simplex_side": (
            "document" if geometry == "document_U" else "feature" if geometry == "word_Z" else "not_applicable"
        ),
        "profile_source": "denoised_original_scale" if geometry == "word_Z" else "not_applicable",
        "W_recovery_method": (
            "stable_document_vertex_solve"
            if geometry == "document_U" and family == "document_gplsi"
            else "Ghat_simplex_prevalence"
            if geometry == "word_Z"
            else "native_baseline"
        ),
        "task_config_hash": task_hash,
        "execution_shard": bundle.metadata.get("execution_shard", "all"),
        "status": "pending",
        "failure_reason": None,
        "warnings": [],
        "artifacts": {},
        **fields,
    }
    row["fit_id"] = _method_id(row)
    row["W_fit_id"] = _W_fit_id(row)
    return row


def _attach_diagnostics(
    row: dict[str, Any],
    bundle: RealDataBundle,
    W: np.ndarray,
    A: np.ndarray,
    heldout_counts: np.ndarray | None,
) -> None:
    row["diagnostics"] = fit_diagnostics(bundle, W, A)
    row["heldout_metrics"] = (
        None
        if heldout_counts is None
        else heldout_count_diagnostics(W, A, heldout_counts)
    )


def _anchor_candidate_records(
    bundle: RealDataBundle,
    geometry_fit,
    A: np.ndarray,
) -> list[dict[str, Any]] | None:
    indices = geometry_fit.selected_vocabulary_indices
    if indices is None:
        return None
    eta = bundle.frequencies.mean(axis=0)
    order = np.argsort(-eta, kind="stable")
    ranks = np.empty(bundle.p, dtype=int)
    ranks[order] = np.arange(1, bundle.p + 1)
    records: list[dict[str, Any]] = []
    for topic, feature in enumerate(np.asarray(indices, dtype=int)):
        across_topics = float(np.sum(A[:, feature]))
        records.append(
            {
                "topic": int(topic),
                "feature_index": int(feature),
                "feature_name": str(bundle.feature_ids[feature]),
                "corpus_frequency": float(eta[feature]),
                "frequency_rank": int(ranks[feature]),
                "distance_to_fitted_vertex": float(
                    geometry_fit.selected_candidate_distances[topic]
                ),
                "topic_specificity": None
                if across_topics <= 0
                else float(A[topic, feature] / across_topics),
                "topic_exclusivity": None
                if across_topics <= 0
                else float(np.max(A[:, feature]) / across_topics),
            }
        )
    return records


def _failure_row(row: dict[str, Any], error: BaseException | str) -> dict[str, Any]:
    if isinstance(error, BaseException):
        reason, trace = _safe_error(error)
        row["traceback"] = trace
    else:
        reason = error
    row["status"] = "failed"
    row["failure_reason"] = reason
    return row


def _declared_vertex_unavailability(
    config: dict[str, Any], K: int, vertex_hunter: str
) -> str | None:
    for record in config.get("known_vertex_hunter_unavailability", []):
        if (
            int(K) in {int(value) for value in record.get("K_values", [])}
            and vertex_hunter in set(map(str, record.get("vertex_hunters", [])))
        ):
            return str(record["reason"])
    return None


def _record_fit(
    task_dir: Path,
    row: dict[str, Any],
    W: np.ndarray,
    A: np.ndarray,
    *,
    vertices: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    arrays = {"W_hat": W, "A_hat": A}
    if vertices is not None:
        arrays["vertices"] = vertices
    arrays.update(extra_arrays or {})
    artifact = _save_npz(task_dir / "arrays" / f"{row['fit_id']}.npz", **arrays)
    row["artifacts"] = {"estimate": artifact}
    row["status"] = "ok"
    return _write_row(task_dir, row)


def _aggregate_rows(task_dir: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text()) for path in sorted((task_dir / "rows").glob("*.json"))]
    frame = pd.json_normalize(rows, sep=".")
    frame.to_csv(task_dir / "fit_rows.csv", index=False)
    return frame


def _published_artifact_row(base: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    row = dict(base)
    if (
        row["dataset"] == "mouse_spleen_codex"
        and row.get("biological_group") != "BALBc-1"
    ):
        row["status"] = "not_available"
        row["failure_reason"] = "frozen published spleen artifacts cover BALBc-1 only"
        row["warnings"] = []
        return _write_row(task_dir, row)
    row["status"] = "published_artifact_only"
    row["warnings"] = [
        "historical graph-GpLSI is retained for comparison but is not claimed bitwise reproducible"
    ]
    row["published_fixture_directory"] = str(
        (REPO_ROOT / "tests/fixtures/real_data_anchor_word_gplsi").resolve()
    )
    return _write_row(task_dir, row)


def _preprocessing_metadata(block, p_execution: int) -> dict[str, Any]:
    threshold = block.preprocessing.threshold
    weighting = block.preprocessing.weighting
    return {
        "p_spectral": int(block.retained_canonical_indices.size),
        "retained_feature_count": int(block.retained_canonical_indices.size),
        "retained_feature_fraction": float(
            block.retained_canonical_indices.size / p_execution
        ),
        "removed_zero_feature_count": int(
            p_execution - block.positive_canonical_indices.size
        ),
        "retained_feature_indices": block.retained_canonical_indices.tolist(),
        "discarded_feature_indices": np.setdiff1d(
            np.arange(p_execution),
            block.retained_canonical_indices,
        ).tolist(),
        "alpha": threshold.alpha,
        "threshold_method": threshold.effective_method,
        "threshold_value": threshold.threshold_value,
        "threshold_strict_greater": threshold.effective_method == "tran_script_exact",
        "top_10_percent_fallback": threshold.fallback_active,
        "retained_row_mass_quantiles": dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                map(float, np.quantile(threshold.retained_row_mass, [0, .25, .5, .75, 1])),
            )
        ),
        "weight_rule": weighting.effective_method,
        "weight_method": weighting.effective_method,
        "tau": weighting.tau,
        "weight_quantiles": weighting.quantiles,
        "weight_maximum_to_median_ratio": weighting.maximum_to_median_ratio,
        "effective_variance": weighting.effective_variance,
        "weight_cap": weighting.cap,
        "weight_common_scale": weighting.common_scale,
        "transformed_singular_values": weighting.transformed_singular_values.tolist(),
        "transformed_condition_number": weighting.transformed_condition_number,
        "rho_selected": block.selected_rho,
        "rho_grid": block.graph_metadata["lambd_grid"],
        "rho_path": block.graph_metadata["lambd_grid"],
        "rho_cv": block.cv_errors,
        "graph_fold_mode": block.graph_metadata["cv_fold_mode"],
        "graph_nfolds_nonempty": block.graph_metadata["nfolds_nonempty"],
        "spectral_iterations": block.used_iterations,
        "spectral_runtime_seconds": block.runtime_seconds,
        "spectral_singular_values": block.singular_values.tolist(),
    }


def _run_baselines(
    bundle: RealDataBundle,
    canonical: RealDataBundle,
    config: dict[str, Any],
    task_dir: Path,
    K: int,
    seed: int,
    task_hash: str,
    heldout_counts: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    requested = list(config.get("baselines", ["published_baseline", "plsi", "topicscore_raw", "lda", "spatial_lda"]))
    definitions = {
        "published_baseline": ("published_baseline", "not_applicable"),
        "plsi": ("plsi", "document_U"),
        "topicscore_raw": ("topicscore_raw", "topicscore_ratio_raw"),
        "lda": ("lda", "not_applicable"),
        "spatial_lda": ("spatial_lda", "not_applicable"),
    }
    for name in requested:
        family, geometry = definitions[name]
        fields = {
            "estimator_family": family,
            "spectral_geometry": geometry,
            "vertex_hunter": "not_applicable",
            "preprocessing": "native_baseline",
            "A_recovery": "native_baseline",
        }
        if name == "plsi":
            pair_rows: dict[str, dict[str, Any]] = {}
            for recovery in config.get("A_recoveries", ["A_current", "A_full_Pois"]):
                pair_fields = dict(fields)
                pair_fields["A_recovery"] = recovery
                pair_rows[recovery] = _base_row(
                    bundle,
                    canonical_bundle=canonical,
                    K=K,
                    seed=seed,
                    task_hash=task_hash,
                    fields=pair_fields,
                )
            cached_pair = {
                recovery: _artifact_valid(
                    task_dir / "rows" / f"{row['fit_id']}.json", task_hash
                )
                for recovery, row in pair_rows.items()
            }
            if all(value is not None for value in cached_pair.values()):
                rows.extend(value for value in cached_pair.values() if value is not None)
                continue
            try:
                started = perf_counter()
                model = GpLSI(method="pLSI", random_state=seed)
                model.fit(
                    bundle.frequencies,
                    bundle.document_lengths,
                    K,
                    bundle.edge_df,
                    bundle.weights,
                    counts=bundle.counts,
                )
                W = model.W_hat.copy()
                spectral_runtime = perf_counter() - started
                for recovery, row in pair_rows.items():
                    if cached_pair[recovery] is not None:
                        rows.append(cached_pair[recovery])
                        continue
                    A_started = perf_counter()
                    if recovery == "A_current":
                        A = model.A_hat.copy()
                        A_result = None
                    elif recovery == "A_full_Pois":
                        A_result = refit_A_full_poisson(
                            W,
                            bundle.counts,
                            bundle.document_lengths,
                            max_iter=int(config.get("poisson_max_iter", 2000)),
                            tolerance=float(config.get("poisson_tolerance", 1e-8)),
                        )
                        A = A_result.A_hat
                    else:
                        raise ValueError(f"unsupported pLSI A recovery {recovery!r}")
                    row["runtime_spectral_seconds"] = spectral_runtime
                    row["runtime_A_recovery_seconds"] = perf_counter() - A_started
                    row["runtime_total_seconds"] = (
                        spectral_runtime + row["runtime_A_recovery_seconds"]
                    )
                    row["metadata"] = {
                        "implementation": "audited current GpLSI pLSI path",
                        "W_shared_across_A_recoveries": True,
                    }
                    if A_result is not None:
                        row["poisson_iterations"] = A_result.iterations
                        row["poisson_converged"] = A_result.converged
                        row["poisson_status"] = A_result.status
                        row["poisson_kkt_projected_gradient_norm"] = (
                            A_result.projected_gradient_norm
                        )
                    _attach_diagnostics(row, bundle, W, A, heldout_counts)
                    rows.append(_record_fit(task_dir, row, W, A))
            except Exception as error:
                for recovery, row in pair_rows.items():
                    if cached_pair[recovery] is not None:
                        rows.append(cached_pair[recovery])
                    else:
                        rows.append(_write_row(task_dir, _failure_row(row, error)))
            continue
        row = _base_row(bundle, canonical_bundle=canonical, K=K, seed=seed, task_hash=task_hash, fields=fields)
        row_path = task_dir / "rows" / f"{row['fit_id']}.json"
        cached = _artifact_valid(row_path, task_hash)
        if cached is not None:
            rows.append(cached)
            continue
        if name == "published_baseline":
            rows.append(_published_artifact_row(row, task_dir))
            continue
        try:
            if name == "topicscore_raw":
                result = fit_topicscore_raw(bundle.frequencies, K)
                W, A = result.W_hat, result.A_hat
                row["runtime_total_seconds"] = result.runtime_seconds
                row["metadata"] = result.metadata
                row["warnings"] = result.warnings
            elif name == "lda":
                result = fit_lda(bundle.counts, K, random_state=seed)
                W, A = result.W_hat, result.A_hat
                row["runtime_total_seconds"] = result.runtime_seconds
                row["metadata"] = result.metadata
                row["warnings"] = result.warnings
            else:
                if bundle.coordinates is None:
                    row["status"] = "not_applicable"
                    row["failure_reason"] = "no_valid_spatial_graph"
                    rows.append(_write_row(task_dir, row))
                    continue
                result = fit_spatial_lda(
                    bundle.counts,
                    K,
                    bundle.coordinates,
                    sample_ids=(
                        bundle.group_ids
                        if np.unique(bundle.group_ids).size > 1
                        else None
                    ),
                    parameters=config.get("spatial_lda_parameters", {}),
                )
                W, A = result.W_hat, result.A_hat
                row["runtime_total_seconds"] = result.runtime_seconds
                row["metadata"] = result.metadata
                row["warnings"] = result.warnings
            _attach_diagnostics(row, bundle, W, A, heldout_counts)
            rows.append(_record_fit(task_dir, row, W, A))
        except Exception as error:
            rows.append(_write_row(task_dir, _failure_row(row, error)))
    return rows


def execute_task(config: dict[str, Any], *, K: int, seed: int) -> Path:
    config = _for_seed_policy(config, seed)
    dataset = str(config["dataset"])
    group = config.get("group", "BALBc-1") if dataset in {"spleen", "mouse_spleen_codex"} else None
    canonical = load_real_data(dataset, group=group or "BALBc-1")
    subset_n = config.get("subset_n")
    if subset_n is None:
        sampled_bundle = canonical
    elif config.get("subset_strategy") == "graph_stratified":
        sampled_bundle = canonical.graph_stratified_subset(
            int(subset_n),
            seed=seed,
            max_groups=config.get("subset_max_groups"),
        )
    else:
        sampled_bundle = canonical.connected_subset(int(subset_n), seed=seed)
    heldout_counts = None
    if config.get("thinning_test_fraction") is not None:
        bundle, heldout_counts = sampled_bundle.binomial_thinning(
            float(config["thinning_test_fraction"]), seed=seed
        )
    else:
        bundle = sampled_bundle
    bundle.metadata["execution_shard"] = config.get("execution_shard", "all")
    code_hashes = _code_hashes()
    task_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "K": K,
        "seed": seed,
        "canonical_hashes": canonical.hashes(),
        "execution_hashes": bundle.hashes(),
        "sampled_pre_thinning_hashes": sampled_bundle.hashes(),
        "heldout_D_sha256": bundle.metadata.get("heldout_D_sha256"),
        "code_hashes": code_hashes,
        "runtime_provenance": _runtime_provenance(),
    }
    task_hash = _hash_bytes(_canonical_json(task_payload).encode())
    output_root = Path(
        os.environ.get(
            "GPLSI_TASK_OUTPUT_ROOT",
            config.get("output_root", REPO_ROOT / "results/real_data_anchor_word_gplsi"),
        )
    )
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    task_dir = (
        output_root
        / str(config["run_name"])
        / f"{_task_id(dataset, group, K, seed)}__{task_hash[:10]}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_manifest.json").write_text(
        json.dumps({**task_payload, "task_config_hash": task_hash}, indent=2, sort_keys=True, default=_json_default) + "\n"
    )

    published_fields = {
        "estimator_family": "published_baseline",
        "spectral_geometry": "not_applicable",
        "vertex_hunter": "not_applicable",
        "preprocessing": "native_baseline",
        "A_recovery": "native_baseline",
    }
    if (
        config.get("record_published_placeholder", True)
        and "published_baseline" not in config.get("baselines", [])
    ):
        row = _base_row(bundle, canonical_bundle=canonical, K=K, seed=seed, task_hash=task_hash, fields=published_fields)
        _published_artifact_row(row, task_dir)
    _run_baselines(
        bundle, canonical, config, task_dir, K, seed, task_hash, heldout_counts
    )

    graph_parameters = dict(config.get("graph", {}))
    preprocessings = config.get("preprocessings", list(PREPROCESSING_SPECS))
    hunters = config.get("vertex_hunters", ["spa_current", "svs", "svs_star", "pp_spa"])
    geometries = config.get("geometries", ["document_U", "word_Z"])
    recoveries = config.get("A_recoveries", ["A_current", "A_full_Pois"])
    for preprocessing_name in preprocessings:
        unavailable_hunters = {
            hunter: _declared_vertex_unavailability(config, K, hunter)
            for hunter in hunters
        }
        if (
            hunters
            and all(unavailable_hunters.values())
            and preprocessing_name
            not in config.get("graph_topicscore_preprocessings", ["P0_raw"])
        ):
            for geometry in geometries:
                family = (
                    "document_gplsi"
                    if geometry == "document_U"
                    else "anchor_feature_gplsi"
                )
                for hunter in hunters:
                    for recovery in recoveries:
                        fields = {
                            "estimator_family": family,
                            "spectral_geometry": geometry,
                            "vertex_hunter": hunter,
                            "preprocessing": preprocessing_name,
                            "A_recovery": recovery,
                        }
                        row = _base_row(
                            bundle,
                            canonical_bundle=canonical,
                            K=K,
                            seed=seed,
                            task_hash=task_hash,
                            fields=fields,
                        )
                        row["status"] = "not_available"
                        row["failure_reason"] = unavailable_hunters[hunter]
                        row["metadata"] = {
                            "availability_decision": "declared_before_spectral_fit",
                            "scientific_grid_retained": True,
                        }
                        _write_row(task_dir, row)
            continue
        try:
            block = fit_spectral_block(
                bundle,
                K,
                preprocessing_name,
                seed=seed,
                **graph_parameters,
            )
            spectral_artifact = _save_npz(
                task_dir / "spectral" / f"{preprocessing_name}.npz",
                U_hat=block.U_hat,
                U_bar=block.U_bar,
                V_hat=block.V_hat,
                singular_values=block.singular_values,
                retained_canonical_indices=block.retained_canonical_indices,
                feature_weights=block.preprocessing.weighting.weights,
                eta_hat=block.preprocessing.threshold.eta_hat,
            )
            spectral_metadata = _preprocessing_metadata(block, bundle.p)
            (task_dir / "spectral" / f"{preprocessing_name}.json").write_text(
                json.dumps(
                    {"artifact": spectral_artifact, **spectral_metadata},
                    indent=2,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
        except Exception as error:
            if preprocessing_name in config.get(
                "graph_topicscore_preprocessings", ["P0_raw"]
            ):
                ts_fields = {
                    "estimator_family": "topicscore_graph_denoised",
                    "spectral_geometry": "topicscore_ratio_graph",
                    "vertex_hunter": "not_applicable",
                    "preprocessing": preprocessing_name,
                    "A_recovery": "native_baseline",
                }
                ts_row = _base_row(
                    bundle,
                    canonical_bundle=canonical,
                    K=K,
                    seed=seed,
                    task_hash=task_hash,
                    fields=ts_fields,
                )
                _write_row(task_dir, _failure_row(ts_row, error))
            for geometry in geometries:
                for hunter in hunters:
                    for recovery in recoveries:
                        fields = {
                            "estimator_family": "document_gplsi" if geometry == "document_U" else "anchor_feature_gplsi",
                            "spectral_geometry": geometry,
                            "vertex_hunter": hunter,
                            "preprocessing": preprocessing_name,
                            "A_recovery": recovery,
                        }
                        row = _base_row(bundle, canonical_bundle=canonical, K=K, seed=seed, task_hash=task_hash, fields=fields)
                        _write_row(task_dir, _failure_row(row, error))
            continue

        if preprocessing_name in config.get(
            "graph_topicscore_preprocessings", ["P0_raw"]
        ):
            ts_fields = {
            "estimator_family": "topicscore_graph_denoised",
            "spectral_geometry": "topicscore_ratio_graph",
            "vertex_hunter": "not_applicable",
            "preprocessing": preprocessing_name,
            "A_recovery": "native_baseline",
            }
            ts_row = _base_row(bundle, canonical_bundle=canonical, K=K, seed=seed, task_hash=task_hash, fields=ts_fields)
            try:
                result = fit_graph_topicscore(bundle, block)
                ts_row.update(
                    {
                        **spectral_metadata,
                        "runtime_total_seconds": block.runtime_seconds + result.runtime_seconds,
                        "metadata": result.metadata,
                        "warnings": block.warnings + result.warnings,
                    }
                )
                _attach_diagnostics(
                    ts_row, bundle, result.W_hat, result.A_hat, heldout_counts
                )
                _record_fit(task_dir, ts_row, result.W_hat, result.A_hat, vertices=result.vertices)
            except Exception as error:
                _write_row(task_dir, _failure_row(ts_row, error))

        for geometry in geometries:
            # SVS* uses the L selected by the corresponding adaptive SVS run.
            # Share both the fitted k-means centers and the exact selection
            # details inside a geometry shard instead of repeating the full
            # exhaustive simplex sweep.
            svs_center_cache: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}
            selected_svs_details: dict[str, Any] | None = None
            for hunter in hunters:
                family = "document_gplsi" if geometry == "document_U" else "anchor_feature_gplsi"
                unavailability_reason = unavailable_hunters[hunter]
                if unavailability_reason is not None:
                    for recovery in recoveries:
                        fields = {
                            "estimator_family": family,
                            "spectral_geometry": geometry,
                            "vertex_hunter": hunter,
                            "preprocessing": preprocessing_name,
                            "A_recovery": recovery,
                        }
                        row = _base_row(
                            bundle,
                            canonical_bundle=canonical,
                            K=K,
                            seed=seed,
                            task_hash=task_hash,
                            fields=fields,
                        )
                        row["status"] = "not_available"
                        row["failure_reason"] = unavailability_reason
                        row["metadata"] = {
                            "availability_decision": "declared_method_limit",
                            "scientific_grid_retained": True,
                        }
                        _write_row(task_dir, row)
                    continue
                vertex_parameters = dict(
                    config.get("vertex_parameters", {}).get(hunter, {})
                )
                if hunter in {"svs", "svs_star"}:
                    vertex_parameters["center_cache"] = svs_center_cache
                if hunter == "svs_star" and selected_svs_details is not None:
                    vertex_parameters["preselected_svs_details"] = selected_svs_details
                try:
                    geometry_fit = fit_geometry(
                        bundle,
                        block,
                        K,
                        geometry=geometry,
                        vertex_hunter=hunter,
                        seed=seed,
                        vertex_parameters=vertex_parameters,
                        condition_threshold=float(config.get("condition_threshold", 1e12)),
                    )
                    if hunter == "svs":
                        selected_svs_details = dict(
                            geometry_fit.vertex_result.parameters
                        )
                except Exception as error:
                    for recovery in recoveries:
                        fields = {
                            "estimator_family": family,
                            "spectral_geometry": geometry,
                            "vertex_hunter": hunter,
                            "preprocessing": preprocessing_name,
                            "A_recovery": recovery,
                        }
                        row = _base_row(bundle, canonical_bundle=canonical, K=K, seed=seed, task_hash=task_hash, fields=fields)
                        row.update(spectral_metadata)
                        _write_row(task_dir, _failure_row(row, error))
                    continue
                for recovery in recoveries:
                    fields = {
                        "estimator_family": family,
                        "spectral_geometry": geometry,
                        "vertex_hunter": hunter,
                        "preprocessing": preprocessing_name,
                        "A_recovery": recovery,
                    }
                    row = _base_row(bundle, canonical_bundle=canonical, K=K, seed=seed, task_hash=task_hash, fields=fields)
                    row_path = task_dir / "rows" / f"{row['fit_id']}.json"
                    cached = _artifact_valid(row_path, task_hash)
                    if cached is not None:
                        continue
                    try:
                        A_result, A_runtime = recover_A_for_geometry(
                            bundle,
                            geometry_fit,
                            recovery,
                            poisson_max_iter=int(config.get("poisson_max_iter", 2000)),
                            poisson_tolerance=float(config.get("poisson_tolerance", 1e-8)),
                        )
                        row.update(
                            {
                                **spectral_metadata,
                                "runtime_word_profile_seconds": geometry_fit.runtimes["word_profile"],
                                "runtime_vertex_hunting_seconds": geometry_fit.runtimes["vertex_hunting"],
                                "runtime_W_recovery_seconds": geometry_fit.runtimes["W_recovery"],
                                "runtime_A_recovery_seconds": A_runtime,
                                "runtime_total_seconds": block.runtime_seconds + sum(geometry_fit.runtimes.values()) + A_runtime,
                                "vertex_condition": geometry_fit.vertex_result.condition_number,
                                "smallest_vertex_singular_value": geometry_fit.vertex_result.smallest_singular_value,
                                "prevalence_residual": getattr(geometry_fit.W_recovery, "prevalence_residual", None),
                                "geometry_diagnostics": geometry_fit.geometry_diagnostics,
                                "poisson_iterations": A_result.iterations if recovery == "A_full_Pois" else None,
                                "poisson_converged": A_result.converged if recovery == "A_full_Pois" else None,
                                "poisson_status": A_result.status if recovery == "A_full_Pois" else None,
                                "A_optimizer_gradient_norm": A_result.gradient_norm,
                                "poisson_kkt_projected_gradient_norm": (
                                    A_result.projected_gradient_norm
                                    if recovery == "A_full_Pois"
                                    else None
                                ),
                                "warnings": block.warnings + geometry_fit.warnings + A_result.warnings,
                                "selected_vocabulary_indices": None
                                if geometry_fit.selected_vocabulary_indices is None
                                else geometry_fit.selected_vocabulary_indices.tolist(),
                                "vertex_parameters": geometry_fit.vertex_result.parameters,
                                "selected_vertex_indices": (
                                    None
                                    if geometry_fit.vertex_result.selected_observation_indices is None
                                    else geometry_fit.vertex_result.selected_observation_indices.tolist()
                                ),
                                "selected_feature_indices": (
                                    None
                                    if geometry_fit.selected_vocabulary_indices is None
                                    else geometry_fit.selected_vocabulary_indices.tolist()
                                ),
                                "selected_feature_names": (
                                    None
                                    if geometry_fit.selected_vocabulary_indices is None
                                    else [
                                        str(bundle.feature_ids[value])
                                        for value in geometry_fit.selected_vocabulary_indices
                                    ]
                                ),
                            }
                        )
                        row["selected_anchor_feature_candidates"] = _anchor_candidate_records(
                            bundle, geometry_fit, A_result.A_hat
                        )
                        _attach_diagnostics(
                            row,
                            bundle,
                            geometry_fit.W_hat,
                            A_result.A_hat,
                            heldout_counts,
                        )
                        _record_fit(
                            task_dir,
                            row,
                            geometry_fit.W_hat,
                            A_result.A_hat,
                            vertices=geometry_fit.vertices,
                            extra_arrays={
                                "W_raw": geometry_fit.W_recovery.raw,
                                "W_truncated_normalized": geometry_fit.W_recovery.truncated_normalized,
                                "G_hat": np.asarray(
                                    getattr(geometry_fit.W_recovery, "G_hat", []),
                                    dtype=float,
                                ),
                                "b_hat": np.asarray(
                                    getattr(geometry_fit.W_recovery, "b_hat", []),
                                    dtype=float,
                                ),
                                "selected_vocabulary_indices": np.asarray(
                                    [] if geometry_fit.selected_vocabulary_indices is None else geometry_fit.selected_vocabulary_indices,
                                    dtype=int,
                                ),
                                **_vertex_artifact_arrays(geometry_fit.vertex_result),
                            },
                        )
                    except Exception as error:
                        row.update(
                            {
                                **spectral_metadata,
                                "runtime_word_profile_seconds": geometry_fit.runtimes["word_profile"],
                                "runtime_vertex_hunting_seconds": geometry_fit.runtimes["vertex_hunting"],
                                "runtime_W_recovery_seconds": geometry_fit.runtimes["W_recovery"],
                                "vertex_condition": geometry_fit.vertex_result.condition_number,
                                "smallest_vertex_singular_value": geometry_fit.vertex_result.smallest_singular_value,
                                "prevalence_residual": getattr(
                                    geometry_fit.W_recovery, "prevalence_residual", None
                                ),
                                "geometry_diagnostics": geometry_fit.geometry_diagnostics,
                                "warnings": block.warnings + geometry_fit.warnings,
                                "vertex_parameters": geometry_fit.vertex_result.parameters,
                            }
                        )
                        row["artifacts"] = {
                            "geometry": _save_npz(
                                task_dir / "arrays" / f"{row['fit_id']}__geometry.npz",
                                W_hat=geometry_fit.W_hat,
                                W_raw=geometry_fit.W_recovery.raw,
                                W_truncated_normalized=geometry_fit.W_recovery.truncated_normalized,
                                vertices=geometry_fit.vertices,
                                **_vertex_artifact_arrays(geometry_fit.vertex_result),
                            )
                        }
                        _write_row(task_dir, _failure_row(row, error))
    frame = _aggregate_rows(task_dir)
    complete = {
        "task_config_hash": task_hash,
        "execution_shard": config.get("execution_shard", "all"),
        "row_count": int(len(frame)),
        "ok_count": int((frame["status"] == "ok").sum()),
        "failed_count": int((frame["status"] == "failed").sum()),
        "status_counts": frame["status"].value_counts().to_dict(),
    }
    (task_dir / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    return task_dir


def _for_execution_shard(config: dict[str, Any], shard: str | None) -> dict[str, Any]:
    if shard is None:
        return config
    declared = list(config.get("execution_shards", []))
    for K_shards in config.get("execution_shards_by_K", {}).values():
        declared.extend(map(str, K_shards))
    declared = list(dict.fromkeys(map(str, declared)))
    if not declared:
        raise ValueError(
            "this config does not declare execution_shards or execution_shards_by_K"
        )
    if shard not in declared:
        raise ValueError(f"unknown execution shard {shard!r}; expected one of {declared}")
    output = json.loads(json.dumps(config))
    output["execution_shard"] = shard
    output["record_published_placeholder"] = False
    shard_parts = shard.split("__")
    if len(shard_parts) == 2 and shard_parts[1] == "fast":
        preprocessing = shard_parts[0]
        if preprocessing not in output.get("preprocessings", []):
            raise ValueError(f"unknown preprocessing in execution shard {shard!r}")
        output["preprocessings"] = [preprocessing]
        output["baselines"] = []
        output["vertex_hunters"] = [
            hunter
            for hunter in output.get("vertex_hunters", [])
            if hunter not in {"svs", "svs_star"}
        ]
        output["graph_topicscore_preprocessings"] = (
            [preprocessing]
            if preprocessing in output.get("graph_topicscore_preprocessings", [])
            else []
        )
        return output
    if len(shard_parts) == 3 and shard_parts[2] == "svs_family":
        preprocessing, geometry, _ = shard_parts
        if preprocessing not in output.get("preprocessings", []):
            raise ValueError(f"unknown preprocessing in execution shard {shard!r}")
        if geometry not in output.get("geometries", []):
            raise ValueError(f"unknown geometry in execution shard {shard!r}")
        output["preprocessings"] = [preprocessing]
        output["geometries"] = [geometry]
        output["vertex_hunters"] = ["svs", "svs_star"]
        output["baselines"] = []
        output["graph_topicscore_preprocessings"] = []
        return output
    if shard in output.get("preprocessings", []):
        output["preprocessings"] = [shard]
        output["baselines"] = []
        output["graph_topicscore_preprocessings"] = (
            [shard]
            if shard in output.get("graph_topicscore_preprocessings", [])
            else []
        )
        return output
    baseline = EXECUTION_SHARD_BASELINES.get(shard)
    if baseline is None:
        raise ValueError(f"execution shard {shard!r} is not a preprocessing or baseline")
    output["preprocessings"] = []
    output["graph_topicscore_preprocessings"] = []
    output["baselines"] = [baseline]
    if baseline == "plsi":
        output["baselines"].insert(0, "published_baseline")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--execution-shard")
    args = parser.parse_args()
    config = _for_execution_shard(
        json.loads(args.config.read_text()), args.execution_shard
    )
    tasks = [(int(K), int(seed)) for K in config["K_values"] for seed in config["seeds"]]
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise SystemExit(f"task index must be in [0,{len(tasks) - 1}]")
        tasks = [tasks[args.task_index]]
    for K, seed in tasks:
        output = execute_task(config, K=K, seed=seed)
        print(json.dumps({"K": K, "seed": seed, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
