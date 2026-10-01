"""Run one task: every requested GpLSI variant and baseline on one dataset/K/seed.

The GpLSI grid is organised to share work:

* one spectral block (preprocessing + graph-aligned SVD) per preprocessing;
* one vertex-hunting geometry per (geometry, hunter) on that block;
* every A recovery applied to the identical W of that geometry.

Every requested method gets exactly one row, including failures (with the
traceback) and methods declared unavailable for a K, so result tables never
silently lose a cell of the design.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import traceback
from typing import Any, Callable

import numpy as np

from ..baselines import fit_graph_kl_nmf, fit_kl_nmf, fit_lda, fit_spatial_lda
from ..gplsi import GpLSI
from ..real_data import REPO_ROOT
from ..real_experiment import (
    SpectralBlock,
    fit_geometry,
    fit_graph_topicscore,
    fit_spectral_block,
    recover_A,
)
from ..topicscore import fit_topicscore_raw
from .config import config_for_task, settings_for_hash, task_id
from .datasets import TaskData, prepare_task_data
from .metrics import evaluate_fit
from .results import (
    SCHEMA_VERSION,
    TaskDirectory,
    W_fit_id,
    canonical_json,
    fit_id,
    method_name,
    save_arrays,
    sha256_file,
    sha256_text,
    write_json,
)


BASELINES = (
    "plsi",
    "topicscore_raw",
    "topicscore_graph_denoised",
    "lda",
    "spatial_lda",
    "kl_nmf",
    "graph_kl_nmf",
)
GEOMETRY_FAMILY = {"document_U": "document_gplsi", "word_Z": "anchor_feature_gplsi"}
# Source files whose content is part of every cache key.
CODE_FILES = sorted((REPO_ROOT / "src" / "gplsi").rglob("*.py")) + sorted(
    (REPO_ROOT / "utils" / "spatial_lda").glob("*.py")
)


# --------------------------------------------------------------------------
# Task setup
# --------------------------------------------------------------------------


def run_task(config: dict[str, Any], task: dict[str, Any]) -> Path:
    """Fit every requested method of one task and return its directory."""

    started = perf_counter()
    resolved = config_for_task(config, task)
    spectral = dict(resolved.get("spectral", {}))
    if os.environ.get("GPLSI_GRAPH_N_JOBS"):
        spectral["n_jobs"] = int(os.environ["GPLSI_GRAPH_N_JOBS"])
    resolved["spectral"] = spectral

    data = prepare_task_data(resolved, task)
    data_hashes = data.hashes()
    identity = {key: value for key, value in task.items() if key != "part"}
    cache_key = sha256_text(
        canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "task": identity,
                "settings": settings_for_hash(resolved),
                "data": data_hashes,
                "code": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in CODE_FILES},
            }
        )
    )
    run_dir = REPO_ROOT / resolved.get("output_root", "results") / resolved["name"]
    output = TaskDirectory(run_dir / task_id(resolved, task), cache_key)
    manifest = "task.json" if task.get("part") is None else f"task__{task['part']}.json"
    write_json(
        output.path / manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "config": resolved,
            "cache_key": cache_key,
            "data_summary": data.summary,
            "data_hashes": data_hashes,
            "provenance": _provenance(),
        },
    )

    bundle = data.bundle
    save_arrays(
        output.path / "data.npz",
        observation_ids=np.asarray(bundle.observation_ids).astype(str),
        feature_ids=np.asarray(bundle.feature_ids).astype(str),
        feature_names=data.feature_names,
        group_ids=np.asarray(bundle.group_ids).astype(str),
        **({} if bundle.coordinates is None else {"coordinates": np.asarray(bundle.coordinates)}),
    )
    fitter = TaskFitter(resolved, task, data, output)
    fitter.run_baselines()
    fitter.run_gplsi()
    frame = output.aggregate()
    write_json(
        output.path / manifest.replace("task", "complete", 1),
        {
            "cache_key": cache_key,
            "elapsed_seconds": perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 * 1024 if sys.platform == "darwin" else 1024),
            "status_counts": frame["status"].value_counts().to_dict() if len(frame) else {},
        },
    )
    return output.path


def _provenance() -> dict[str, Any]:
    versions = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "pycvxcluster"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "host": platform.node(),
        "versions": versions,
        "git_commit": commit,
        "git_dirty": dirty,
        "code_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in CODE_FILES},
    }


def _error_text(error: BaseException) -> tuple[str, str]:
    return f"{type(error).__name__}: {error}", "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


@dataclass
class TaskFitter:
    config: dict[str, Any]
    task: dict[str, Any]
    data: TaskData
    output: TaskDirectory

    @property
    def K(self) -> int:
        return int(self.task["K"])

    @property
    def seed(self) -> int:
        return int(self.task["seed"])

    # ---- rows -------------------------------------------------------------

    def new_row(self, **fields: Any) -> dict[str, Any]:
        bundle = self.data.bundle
        row = {
            "schema_version": SCHEMA_VERSION,
            "run": self.config["name"],
            "dataset": self.config["dataset"]["name"],
            "group": self.config["dataset"].get("group"),
            "task": {key: value for key, value in self.task.items() if key != "part"},
            "part": self.task.get("part"),
            "K": self.K,
            "seed": self.seed,
            "n": bundle.n,
            "p": bundle.p,
            "N_mean": bundle.N_mean,
            "cache_key": self.output.cache_key,
            "status": "pending",
            "converged": None,
            "failure_reason": None,
            "warnings": [],
            "runtime": {},
            "artifacts": {},
            **fields,
        }
        row["method"] = method_name(row)
        row["fit_id"] = fit_id(row)
        row["W_fit_id"] = W_fit_id(row)
        return row

    def record_fit(
        self, row: dict[str, Any], W: np.ndarray, A: np.ndarray, extra_arrays: dict | None = None
    ) -> dict[str, Any]:
        row.update(evaluate_fit(self.data, W, A, self.seed))
        row["status"] = "ok"
        arrays = {"W_hat": W, "A_hat": A, **(extra_arrays or {})}
        return self.output.write_row(row, arrays)

    def record_failure(self, row: dict[str, Any], error: BaseException | str) -> dict[str, Any]:
        if isinstance(error, BaseException):
            row["failure_reason"], row["traceback"] = _error_text(error)
        else:
            row["failure_reason"] = error
        row["status"] = "failed"
        return self.output.write_row(row)

    def unavailable_reason(self, hunter: str) -> str | None:
        for rule in self.config.get("unavailable", []):
            if self.K in rule.get("K", []) and hunter in rule.get("vertex_hunters", []):
                return str(rule["reason"])
        return None

    # ---- baselines ----------------------------------------------------------

    def run_baselines(self) -> None:
        requested = list(self.config.get("baselines", []))
        unknown = set(requested) - set(BASELINES)
        if unknown:
            raise ValueError(f"unknown baselines {sorted(unknown)}; expected {BASELINES}")
        bundle = self.data.bundle
        simple: dict[str, Callable[[], Any]] = {
            "topicscore_raw": lambda: fit_topicscore_raw(bundle.frequencies, self.K),
            "lda": lambda: fit_lda(bundle.counts, self.K, random_state=self.seed),
            "kl_nmf": lambda: fit_kl_nmf(bundle.counts, self.K, random_state=self.seed),
            "graph_kl_nmf": lambda: fit_graph_kl_nmf(
                bundle.counts, bundle.weights, self.K, random_state=self.seed
            ),
            "spatial_lda": self._fit_spatial_lda,
        }
        for name in requested:
            if name == "plsi":
                self._run_plsi()
            elif name in simple:
                self._run_simple_baseline(name, simple[name])
            # topicscore_graph_denoised reuses a spectral block: see run_gplsi.

    def _baseline_row(self, name: str, geometry: str = "not_applicable", recovery: str = "native") -> dict:
        return self.new_row(
            estimator_family=name,
            spectral_geometry=geometry,
            vertex_hunter="not_applicable",
            preprocessing="not_applicable",
            A_recovery=recovery,
        )

    def _fit_spatial_lda(self):
        bundle = self.data.bundle
        if bundle.coordinates is None:
            # No coordinates (Cooking): penalize the dataset's own graph.
            return fit_spatial_lda(
                bundle.counts, self.K, None,
                edges=bundle.edge_df[["src", "tgt"]].to_numpy(dtype=int),
                parameters=self.config.get("spatial_lda_parameters", {}),
            )
        multiple = np.unique(bundle.group_ids).size > 1
        return fit_spatial_lda(
            bundle.counts,
            self.K,
            bundle.coordinates,
            sample_ids=bundle.group_ids if multiple else None,
            parameters=self.config.get("spatial_lda_parameters", {}),
        )

    def _run_simple_baseline(self, name: str, fit: Callable[[], Any]) -> None:
        row = self._baseline_row(name, "topicscore_ratio_raw" if name == "topicscore_raw" else "not_applicable")
        if self.output.cached(row["fit_id"]):
            return
        try:
            result = fit()
            row.update(
                converged=bool(result.converged),
                runtime={"total": result.runtime_seconds},
                metadata=result.metadata,
                warnings=list(result.warnings),
            )
            self.record_fit(row, result.W_hat, result.A_hat)
        except Exception as error:
            self.record_failure(row, error)

    def _run_plsi(self) -> None:
        """Graph-free pLSI; every A recovery shares its W."""

        recoveries = self._recoveries()
        rows = {r: self._baseline_row("plsi", "document_U", r) for r in recoveries}
        todo = {r: row for r, row in rows.items() if not self.output.cached(row["fit_id"])}
        if not todo:
            return
        bundle = self.data.bundle
        try:
            started = perf_counter()
            model = GpLSI(method="pLSI", random_state=self.seed)
            model.fit(bundle.frequencies, bundle.document_lengths, self.K, bundle.edge_df,
                      bundle.weights, counts=bundle.counts)
            W = model.W_hat.copy()
            spectral_seconds = perf_counter() - started
        except Exception as error:
            for row in todo.values():
                self.record_failure(row, error)
            return
        for recovery, row in todo.items():
            try:
                if recovery == "A_current":
                    A, a_seconds, info = model.A_hat.copy(), 0.0, None
                else:
                    initial = model.A_hat if self._recovery_settings()["poisson_start"] == "A_current" else None
                    result, a_seconds = recover_A(
                        bundle, W, recovery, poisson_initial_A=initial, **self._recovery_settings()
                    )
                    A, info = result.A_hat, _recovery_info(result)
                row.update(
                    converged=None if info is None else info["converged"],
                    A_recovery_info=info,
                    runtime={"spectral": spectral_seconds, "A_recovery": a_seconds,
                             "total": spectral_seconds + a_seconds},
                )
                self.record_fit(row, W, A)
            except Exception as error:
                self.record_failure(row, error)

    # ---- GpLSI grid ---------------------------------------------------------

    def _recoveries(self) -> list[str]:
        # A_current first: its estimate is the warm start of the refits.
        requested = list(dict.fromkeys(self.config.get("A_recoveries", ["A_current"])))
        return sorted(requested, key=lambda name: name != "A_current")

    def _recovery_settings(self) -> dict[str, Any]:
        settings = {"poisson_max_iter": 2000, "poisson_tolerance": 1e-8, "poisson_start": "A_current"}
        settings.update(self.config.get("recovery", {}))
        return settings

    def _gplsi_rows(self, preprocessing: str, geometry: str, hunter: str) -> dict[str, dict]:
        return {
            recovery: self.new_row(
                estimator_family=GEOMETRY_FAMILY[geometry],
                spectral_geometry=geometry,
                vertex_hunter=hunter,
                preprocessing=preprocessing,
                A_recovery=recovery,
            )
            for recovery in self._recoveries()
        }

    def run_gplsi(self) -> None:
        blocks = self.config.get("gplsi", [])
        for block in blocks:
            if block["geometry"] not in GEOMETRY_FAMILY:
                raise ValueError(f"geometry must be one of {sorted(GEOMETRY_FAMILY)}")
        topicscore_on = None
        if "topicscore_graph_denoised" in self.config.get("baselines", []):
            topicscore_on = self.config.get("topicscore_graph_preprocessing", "P0_raw")
        preprocessings = list(dict.fromkeys(
            [name for block in blocks for name in block["preprocessings"]]
            + ([topicscore_on] if topicscore_on else [])
        ))
        for preprocessing in preprocessings:
            plan = [
                (block["geometry"], hunter)
                for block in blocks
                if preprocessing in block["preprocessings"]
                for hunter in block["vertex_hunters"]
            ]
            self._run_preprocessing(preprocessing, plan, with_topicscore=preprocessing == topicscore_on)

    def _run_preprocessing(self, preprocessing: str, plan: list[tuple[str, str]], *, with_topicscore: bool) -> None:
        pending = [
            (geometry, hunter, rows)
            for geometry, hunter in plan
            for rows in [self._gplsi_rows(preprocessing, geometry, hunter)]
            if not all(self.output.cached(row["fit_id"]) for row in rows.values())
        ]
        topicscore_row = None
        if with_topicscore:
            topicscore_row = self.new_row(
                estimator_family="topicscore_graph_denoised",
                spectral_geometry="topicscore_ratio_graph",
                vertex_hunter="not_applicable",
                preprocessing=preprocessing,
                A_recovery="native",
            )
            if self.output.cached(topicscore_row["fit_id"]):
                topicscore_row = None
        if not pending and topicscore_row is None:
            return

        # Hunters declared unavailable at this K get explicit rows, no fit.
        available = []
        for geometry, hunter, rows in pending:
            reason = self.unavailable_reason(hunter)
            if reason is None:
                available.append((geometry, hunter, rows))
                continue
            for row in rows.values():
                row.update(status="not_available", failure_reason=reason)
                self.output.write_row(row)
        if not available and topicscore_row is None:
            return

        try:
            block = fit_spectral_block(
                self.data.bundle, self.K, preprocessing, seed=self.seed, **self.config.get("spectral", {})
            )
        except Exception as error:
            for _, _, rows in available:
                for row in rows.values():
                    self.record_failure(row, error)
            if topicscore_row is not None:
                self.record_failure(topicscore_row, error)
            return
        spectral_info = self._save_spectral_block(block)

        if topicscore_row is not None:
            try:
                result = fit_graph_topicscore(self.data.bundle, block)
                topicscore_row.update(
                    spectral=spectral_info,
                    runtime={"spectral": block.runtime_seconds, "total": block.runtime_seconds + result.runtime_seconds},
                    metadata=result.metadata,
                    warnings=block.warnings + result.warnings,
                )
                self.record_fit(topicscore_row, result.W_hat, result.A_hat, {"vertices": result.vertices})
            except Exception as error:
                self.record_failure(topicscore_row, error)

        # Within a geometry, SVS* reuses the k-means centers and the adaptive
        # L selection of SVS instead of repeating the exhaustive sweep.
        shared: dict[str, dict[str, Any]] = {}
        for geometry, hunter, rows in available:
            state = shared.setdefault(geometry, {"centers": {}, "svs_details": None})
            self._run_geometry(block, spectral_info, geometry, hunter, rows, state)

    def _save_spectral_block(self, block: SpectralBlock) -> dict[str, Any]:
        artifact = save_arrays(
            self.output.path / "spectral" / f"{block.preprocessing_name}.npz",
            relative_to=self.output.path,
            U_hat=block.U_hat,
            U_bar=block.U_bar,
            V_hat=block.V_hat,
            singular_values=block.singular_values,
            retained_canonical_indices=block.retained_canonical_indices,
            feature_weights=block.preprocessing.weighting.weights,
            eta_hat=block.preprocessing.threshold.eta_hat,
        )
        threshold = block.preprocessing.threshold
        weighting = block.preprocessing.weighting
        graph = block.graph_metadata
        return {
            "artifact": artifact,
            "p_spectral": int(block.retained_canonical_indices.size),
            "removed_zero_feature_count": int(self.data.bundle.p - block.positive_canonical_indices.size),
            "threshold_method": threshold.effective_method,
            "threshold_value": threshold.threshold_value,
            "alpha": threshold.alpha,
            "top_10_percent_fallback": threshold.fallback_active,
            "weight_method": weighting.effective_method,
            "weight_quantiles": weighting.quantiles,
            "rho_selected": block.selected_rho,
            "rho_grid": graph["lambd_grid"],
            "rho_path": graph["lambd_history"],
            "rho_cv": block.cv_errors,
            "lambda_selection_mode": graph["lambda_selection_mode"],
            "cv_fold_mode": graph["cv_fold_mode"],
            "nfolds_nonempty": graph["nfolds_nonempty"],
            "iterations": block.used_iterations,
            "runtime_seconds": block.runtime_seconds,
            "singular_values": block.singular_values.tolist(),
        }

    def _run_geometry(self, block, spectral_info, geometry, hunter, rows, state) -> None:
        parameters = dict(self.config.get("vertex_parameters", {}).get(hunter, {}))
        if hunter in {"svs", "svs_star"}:
            parameters["center_cache"] = state["centers"]
        adaptive = parameters.get("L_mode") == "mixedscore_adaptive"
        if hunter == "svs_star" and adaptive and state["svs_details"] is not None:
            parameters["preselected_svs_details"] = state["svs_details"]
        try:
            fitted = fit_geometry(
                self.data.bundle, block, self.K,
                geometry=geometry, vertex_hunter=hunter, seed=self.seed,
                vertex_parameters=parameters,
                condition_threshold=float(self.config.get("condition_threshold", 1e12)),
            )
        except Exception as error:
            for row in rows.values():
                row["spectral"] = spectral_info
                self.record_failure(row, error)
            return
        if hunter == "svs":
            state["svs_details"] = dict(fitted.vertex_result.parameters)

        vertex = fitted.vertex_result
        geometry_info = {
            "vertex_parameters": vertex.parameters,
            "vertex_condition": vertex.condition_number,
            "smallest_vertex_singular_value": vertex.smallest_singular_value,
            "geometry_diagnostics": fitted.geometry_diagnostics,
            "selected_vertex_indices": vertex.selected_observation_indices,
            "selected_feature_names": (
                None if fitted.selected_vocabulary_indices is None
                else [str(self.data.bundle.feature_ids[i]) for i in fitted.selected_vocabulary_indices]
            ),
        }
        arrays = {
            "vertices": fitted.vertices,
            "W_raw": fitted.W_recovery.raw,
            "selected_vocabulary_indices": np.asarray(
                [] if fitted.selected_vocabulary_indices is None else fitted.selected_vocabulary_indices, dtype=int
            ),
            **_vertex_arrays(vertex),
        }
        settings = self._recovery_settings()
        paired_current = None
        for recovery, row in rows.items():
            cached = self.output.cached(row["fit_id"])
            if cached is not None:
                continue
            try:
                initial = paired_current if settings["poisson_start"] == "A_current" else None
                result, a_seconds = recover_A(
                    self.data.bundle, fitted.W_hat, recovery, poisson_initial_A=initial, **settings
                )
                if recovery == "A_current":
                    paired_current = result.A_hat
                row.update(
                    spectral=spectral_info,
                    geometry=geometry_info,
                    converged=bool(result.converged),
                    A_recovery_info=_recovery_info(result),
                    runtime={
                        "spectral": block.runtime_seconds,
                        **fitted.runtimes,
                        "A_recovery": a_seconds,
                        "total": block.runtime_seconds + sum(fitted.runtimes.values()) + a_seconds,
                    },
                    warnings=block.warnings + fitted.warnings + result.warnings,
                )
                self.record_fit(row, fitted.W_hat, result.A_hat, arrays)
            except Exception as error:
                row.update(spectral=spectral_info, geometry=geometry_info)
                self.record_failure(row, error)


def _recovery_info(result) -> dict[str, Any]:
    return {
        "method": result.method,
        "status": result.status,
        "converged": bool(result.converged),
        "iterations": result.iterations,
        "gradient_norm": result.gradient_norm,
        "projected_gradient_norm": result.projected_gradient_norm,
        "optimality_gap": result.optimality_gap,
        "normalized_optimality_gap": result.normalized_optimality_gap,
        "solver": result.solver,
        "diagnostics": result.diagnostics,
    }


def _vertex_arrays(vertex) -> dict[str, np.ndarray]:
    names = (
        "selected_observation_indices", "selected_center_indices", "centers", "pseudo_points",
        "projected_points", "cluster_assignments", "neighborhood_sizes", "retained_point_indices",
        "discarded_point_indices", "initialization_observation_indices", "observation_weights",
        "archetype_to_data_weights", "archetype_projection_points", "initialization_vertices",
        "initialization_weights", "objective_trace", "reconstruction_trace", "penalty_trace",
        "relative_step_trace", "gamma_h_trace", "gamma_w_trace",
    )
    return {name: np.asarray(getattr(vertex, name)) for name in names if getattr(vertex, name, None) is not None}
