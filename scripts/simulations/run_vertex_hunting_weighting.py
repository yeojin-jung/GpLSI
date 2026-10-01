#!/usr/bin/env python3
"""Restartable controlled smoke and pilot experiments.

Each data/preprocessing pair is factorized once.  Every requested vertex hunter
then receives the same saved embedding, so vertex comparisons cannot regenerate
data, graph folds, graph-CV paths, or singular vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from gplsi.generate_topic_model import (
    generate_W_strong,
    generate_data,
    generate_graph,
    generate_weights_edge,
)
from gplsi.graphSVD import graphSVD
from gplsi.preprocessing import preprocess_features, weighted_debiased_correction
from gplsi.recovery import recover_W, refit_A_full_l2
from gplsi.vertex_hunting import vertex_hunt


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAN_SOURCE = Path(os.environ.get(
    "GPLSI_TRAN_ROOT", str(REPO_ROOT / "external_references/topic-modeling")
)) / "r/experiments/synthetic/synthetic_dataset.R"


@dataclass
class Dataset:
    name: str
    family: str
    seed: int
    X: np.ndarray
    counts: np.ndarray
    W_true: np.ndarray
    A_true: np.ndarray
    population_M: np.ndarray
    N: float
    graph_setting: str
    edge_df: pd.DataFrame | None
    graph_weights: Any
    factorization: str
    design_axis: str | None
    design_value: float | None
    requested_p: int
    a_zipf: float | None
    generator_name: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=float)


def _recorded_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _generate_gplsi(spec: dict[str, Any], seed: int) -> Dataset:
    np.random.seed(seed)
    coords, W_topic_by_doc, A_feature_by_topic, X = generate_data(
        spec["N"],
        spec["n"],
        spec["p"],
        spec["K"],
        spec["rt"],
        spec["n_clusters"],
        method=spec.get("generator_method", "strong"),
    )
    graph_weights, edge_df = generate_weights_edge(
        coords, spec["nearest_n"], spec["phi"]
    )
    W_true = W_topic_by_doc.T
    A_true = A_feature_by_topic.T
    return Dataset(
        name=spec["name"],
        family=spec["experiment_family"],
        seed=seed,
        X=X,
        counts=np.rint(X * spec["N"]),
        W_true=W_true,
        A_true=A_true,
        population_M=W_true @ A_true,
        N=float(spec["N"]),
        graph_setting=spec["graph_setting"],
        edge_df=edge_df,
        graph_weights=graph_weights,
        factorization="graphSVD",
        design_axis=spec.get("design_axis"),
        design_value=spec.get("design_value"),
        requested_p=int(spec["p"]),
        a_zipf=spec.get("a_zipf"),
        generator_name=spec["generator"],
    )


def _generate_tran(spec: dict[str, Any], seed: int, data_root: Path) -> Dataset:
    target = data_root / f"{spec['name']}_seed_{seed}"
    required = [
        target / "counts.csv",
        target / "A_feature_by_topic.csv",
        target / "W_topic_by_document.csv",
        target / "population_document_by_feature.csv",
    ]
    if not all(path.exists() for path in required):
        command = [
            "Rscript",
            str(REPO_ROOT / "codes" / "generate_tran_pilot_data.R"),
            str(TRAN_SOURCE),
            str(target),
            str(seed),
            str(spec["n"]),
            str(spec["p"]),
            str(spec["N"]),
            str(spec["K"]),
            str(spec["alpha_dirichlet"]),
            str(spec["n_anchors"]),
            str(spec["delta_anchor"]),
            str(spec["a_zipf"]),
            str(spec["offset_zipf"]),
        ]
        subprocess.run(command, cwd=TRAN_SOURCE.parents[3], check=True)
    counts = _read_csv_matrix(target / "counts.csv")
    A_true = _read_csv_matrix(target / "A_feature_by_topic.csv").T
    W_true = _read_csv_matrix(target / "W_topic_by_document.csv").T
    population = _read_csv_matrix(target / "population_document_by_feature.csv")
    return Dataset(
        name=spec["name"],
        family=spec["experiment_family"],
        seed=seed,
        X=counts / float(spec["N"]),
        counts=counts,
        W_true=W_true,
        A_true=A_true,
        population_M=population,
        N=float(spec["N"]),
        graph_setting=spec["graph_setting"],
        edge_df=None,
        graph_weights=None,
        factorization="pLSI",
        design_axis=spec.get("design_axis"),
        design_value=spec.get("design_value"),
        requested_p=int(spec["p"]),
        a_zipf=float(spec["a_zipf"]),
        generator_name=spec["generator"],
    )


def _generate_mixed_tran_gplsi(
    spec: dict[str, Any], seed: int, data_root: Path
) -> Dataset:
    target = data_root / f"{spec['name']}_seed_{seed}"
    required = [
        target / "counts.csv",
        target / "A_topic_by_feature.csv",
        target / "W_document_by_topic.csv",
        target / "population_document_by_feature.csv",
        target / "coordinates.csv",
    ]
    if not all(path.exists() for path in required):
        target.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        coordinates = generate_graph(
            spec["N"],
            spec["n"],
            spec["p"],
            spec["K"],
            spec["rt"],
            spec["n_clusters"],
        )
        W_topic_by_document = generate_W_strong(
            coordinates,
            spec["N"],
            spec["n"],
            spec["p"],
            spec["K"],
            spec["rt"],
        )
        W_true = W_topic_by_document.T
        topic_path = target / "tran_A_full_feature_by_topic.csv"
        subprocess.run(
            [
                "Rscript",
                str(REPO_ROOT / "codes" / "generate_tran_topic_matrix.R"),
                str(TRAN_SOURCE),
                str(topic_path),
                str(seed),
                str(spec["n"]),
                str(spec["p"]),
                str(spec["K"]),
                str(spec["alpha_dirichlet"]),
                str(spec["n_anchors"]),
                str(spec["delta_anchor"]),
                str(spec["a_zipf"]),
                str(spec["offset_zipf"]),
            ],
            cwd=TRAN_SOURCE.parents[3],
            check=True,
        )
        A_full = _read_csv_matrix(topic_path).T
        population_full = W_true @ A_full
        count_rng = np.random.default_rng(seed + 10_000_019)
        counts_full = np.vstack(
            [
                count_rng.multinomial(int(spec["N"]), population_full[index])
                for index in range(spec["n"])
            ]
        ).astype(float)
        observed = counts_full.sum(axis=0) > 0
        counts = counts_full[:, observed]
        population = population_full[:, observed]
        A_true = A_full[:, observed]
        A_true /= A_true.sum(axis=1, keepdims=True)
        pd.DataFrame(counts).to_csv(target / "counts.csv", index=False)
        pd.DataFrame(A_true).to_csv(target / "A_topic_by_feature.csv", index=False)
        pd.DataFrame(W_true).to_csv(target / "W_document_by_topic.csv", index=False)
        pd.DataFrame(population).to_csv(
            target / "population_document_by_feature.csv", index=False
        )
        coordinates.to_csv(target / "coordinates.csv", index=False)
        (target / "metadata.json").write_text(
            json.dumps(
                {
                    "generator": "tran_topic_frequencies_plus_gplsi_graph_W",
                    "tran_source": str(TRAN_SOURCE),
                    "tran_source_commit": "7ce72d5bd9a183d6bbc83436e79f5b213a42eeb0",
                    "seed": seed,
                    "requested_p": spec["p"],
                    "observed_p": int(observed.sum()),
                    "count_seed_offset": 10_000_019,
                },
                indent=2,
            )
            + "\n"
        )
    counts = _read_csv_matrix(target / "counts.csv")
    A_true = _read_csv_matrix(target / "A_topic_by_feature.csv")
    W_true = _read_csv_matrix(target / "W_document_by_topic.csv")
    population = _read_csv_matrix(target / "population_document_by_feature.csv")
    coordinates = pd.read_csv(target / "coordinates.csv")
    graph_weights, edge_df = generate_weights_edge(
        coordinates, spec["nearest_n"], spec["phi"]
    )
    return Dataset(
        name=spec["name"],
        family=spec["experiment_family"],
        seed=seed,
        X=counts / float(spec["N"]),
        counts=counts,
        W_true=W_true,
        A_true=A_true,
        population_M=population,
        N=float(spec["N"]),
        graph_setting=spec["graph_setting"],
        edge_df=edge_df,
        graph_weights=graph_weights,
        factorization="graphSVD",
        design_axis=spec.get("design_axis"),
        design_value=spec.get("design_value"),
        requested_p=int(spec["p"]),
        a_zipf=float(spec["a_zipf"]),
        generator_name=spec["generator"],
    )


def generate_dataset(spec: dict[str, Any], seed: int, data_root: Path) -> Dataset:
    if spec["generator"] == "gplsi_current":
        return _generate_gplsi(spec, seed)
    if spec["generator"] == "tran_source_exact":
        return _generate_tran(spec, seed, data_root)
    if spec["generator"] == "tran_A_gplsi_W":
        return _generate_mixed_tran_gplsi(spec, seed, data_root)
    raise ValueError(f"unknown generator {spec['generator']!r}")


def expanded_dataset_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    expanded = [dict(spec) for spec in config.get("datasets", [])]
    for design in config.get("dataset_design_grids", []):
        for family in design["families"]:
            for axis, values in design["vary"].items():
                for value in values:
                    spec = dict(design["base_parameters"])
                    spec.update(family)
                    spec[axis] = value
                    spec["design_axis"] = axis
                    spec["design_value"] = value
                    value_label = str(value).replace(".", "p")
                    spec["name"] = f"{family['name_prefix']}_{axis}_{value_label}"
                    expanded.append(spec)
    return expanded


def _factorize(
    dataset: Dataset,
    preprocessing: dict[str, Any],
    graph_config: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], float]:
    prep = preprocess_features(
        dataset.X,
        dataset.N,
        threshold_method=preprocessing["threshold_method"],
        alpha=preprocessing.get("alpha", 0.005),
        weight_method=preprocessing["weight_method"],
        tau=preprocessing.get("tau", 0.0),
        weight_cap=preprocessing.get("weight_cap"),
        weight_common_scale=preprocessing.get("weight_common_scale", "none"),
        K=dataset.W_true.shape[1],
        true_A=dataset.A_true,
        population_M=dataset.population_M,
        fail_on_rank_loss=True,
    )
    K = dataset.W_true.shape[1]
    started = time.perf_counter()
    if dataset.factorization == "pLSI":
        U_full, singular_full, Vt_full = np.linalg.svd(
            prep.X_transformed, full_matrices=False
        )
        U = U_full[:, :K]
        singular = singular_full[:K]
        V = Vt_full[:K].T
        U_bar = U.copy()
        metadata = {
            "initialization": "direct_svd",
            "score_history": [],
            "lambd_history": [],
            "cv_history": [],
            "lambd_grid": [],
            "U_bar_history": [U.copy()],
            "U_hat_history": [U.copy()],
            "V_hat_history": [V.copy()],
            "singular_value_history": [singular.copy()],
        }
    else:
        initialization = preprocessing.get("initialization", "current")
        correction = None
        if initialization in {
            "weighted_debiased",
            "weighted_debiased_mean_N_approx",
        }:
            eta = dataset.X[:, prep.threshold.retained_indices].mean(axis=0)
            correction = weighted_debiased_correction(
                eta,
                prep.weighting.weights,
                n=dataset.X.shape[0],
                N=dataset.N,
            )
        np.random.seed(dataset.seed + 1009 * int(preprocessing["order"]))
        output = graphSVD(
            prep.X_transformed,
            dataset.N,
            K,
            dataset.edge_df,
            dataset.graph_weights,
            graph_config["lamb_start"],
            graph_config["step_size"],
            graph_config["grid_len"],
            graph_config["maxiter"],
            graph_config["eps"],
            graph_config.get("verbose", 0),
            True,
            initialization=initialization,
            debias_correction=correction,
            return_metadata=True,
        )
        U, V, L, _, _, _, _, _, _, metadata = output
        singular = np.diag(L)
        U_bar = metadata["U_bar"]
    runtime = time.perf_counter() - started
    return prep, U, U_bar, V, singular, metadata, runtime


def _align_topics(
    W_hat: np.ndarray,
    A_hat: np.ndarray,
    W_true: np.ndarray,
    A_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarity = W_hat.T @ W_true
    estimated, truth = linear_sum_assignment(1.0 - np.abs(similarity))
    mapping = np.zeros((W_hat.shape[1], W_true.shape[1]))
    mapping[estimated, truth] = 1.0
    return W_hat @ mapping, mapping.T @ A_hat, mapping


def _vertex_error(
    embedding: np.ndarray, vertices: np.ndarray, W_true: np.ndarray
) -> tuple[float, float]:
    oracle, _, _, _ = np.linalg.lstsq(W_true, embedding, rcond=None)
    distances = np.linalg.norm(vertices[:, None, :] - oracle[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(distances)
    difference = vertices[rows] - oracle[columns]
    return float(np.linalg.norm(difference)), float(np.sqrt(np.mean(difference**2)))


def _subspace_error(embedding: np.ndarray, W_true: np.ndarray) -> float:
    Q_embedding, _ = np.linalg.qr(embedding)
    Q_truth, _ = np.linalg.qr(W_true)
    return float(np.linalg.norm(Q_embedding @ Q_embedding.T - Q_truth @ Q_truth.T))


def _poisson_deviance(counts: np.ndarray, means: np.ndarray, lengths: float) -> float:
    expected = np.maximum(lengths * means, 1e-12)
    observed = counts
    positive = observed > 0
    terms = expected - observed
    terms[positive] += observed[positive] * np.log(observed[positive] / expected[positive])
    return float(2.0 * np.sum(terms))


def _base_record(
    dataset: Dataset,
    preprocessing: dict[str, Any],
    prep: Any,
    embedding_source: str,
    method: str,
    metadata: dict[str, Any],
    factor_runtime: float,
) -> dict[str, Any]:
    threshold = prep.threshold
    weighting = prep.weighting
    return {
        "dataset": dataset.name,
        "experiment_family": dataset.family,
        "seed": dataset.seed,
        "n": dataset.X.shape[0],
        "p": dataset.X.shape[1],
        "N": dataset.N,
        "K": dataset.W_true.shape[1],
        "requested_p": dataset.requested_p,
        "a_zipf": dataset.a_zipf,
        "design_axis": dataset.design_axis,
        "design_value": dataset.design_value,
        "data_generator": dataset.generator_name,
        "graph_setting": dataset.graph_setting,
        "preprocessing_variant": preprocessing["name"],
        "threshold_method": threshold.effective_method,
        "alpha": threshold.alpha,
        "threshold_value": threshold.threshold_value,
        "retained_feature_count": threshold.retained_feature_count,
        "retained_feature_fraction": threshold.retained_feature_fraction,
        "retained_row_mass_min": float(np.min(threshold.retained_row_mass)),
        "retained_row_mass_max": float(np.max(threshold.retained_row_mass)),
        "retained_topic_mass": json.dumps(_jsonable(threshold.retained_topic_mass)),
        "retained_population_rank": threshold.retained_population_rank,
        "weight_method": weighting.effective_method,
        "tau": weighting.tau,
        "weight_cap": weighting.cap,
        "weight_common_scale": weighting.common_scale,
        "weight_quantiles": json.dumps(weighting.quantiles, sort_keys=True),
        "weight_max_to_median": weighting.maximum_to_median_ratio,
        "effective_variance": weighting.effective_variance,
        "transformed_condition_number": weighting.transformed_condition_number,
        "transformed_singular_values": json.dumps(
            _jsonable(weighting.transformed_singular_values)
        ),
        "vertex_hunter": method,
        "embedding_source": embedding_source,
        "A_recovery_method": "A_full_L2",
        "rho_path": json.dumps(_jsonable(metadata.get("lambd_history", []))),
        "selected_rho": (
            metadata.get("lambd_history", [])[-1]
            if metadata.get("lambd_history", [])
            else None
        ),
        "cv_path": json.dumps(_jsonable(metadata.get("cv_history", []))),
        "graph_iteration_count": len(metadata.get("score_history", [])),
        "graph_convergence_scores": json.dumps(
            _jsonable(metadata.get("score_history", []))
        ),
        "factorization_runtime_seconds": factor_runtime,
    }


def _run_hunter(
    dataset: Dataset,
    preprocessing: dict[str, Any],
    prep: Any,
    U: np.ndarray,
    U_bar: np.ndarray,
    V: np.ndarray,
    singular: np.ndarray,
    metadata: dict[str, Any],
    factor_runtime: float,
    method_spec: dict[str, Any],
    condition_threshold: float,
    center_cache: dict[int, Any],
    fit_path: Path,
) -> dict[str, Any]:
    method = method_spec["method"]
    embedding_source = method_spec.get("embedding_source", "U_hat")
    embedding = U if embedding_source == "U_hat" else U_bar
    record = _base_record(
        dataset,
        preprocessing,
        prep,
        embedding_source,
        method,
        metadata,
        factor_runtime,
    )
    parameters = dict(method_spec.get("parameters", {}))
    if method in {"svs", "svs_star"}:
        parameters["center_cache"] = center_cache
    started = time.perf_counter()
    result = vertex_hunt(
        embedding.copy(),
        dataset.W_true.shape[1],
        method,
        random_state=dataset.seed,
        condition_threshold=condition_threshold,
        **parameters,
    )
    record.update(
        {
            "L": result.parameters.get("L"),
            "L_mode": result.parameters.get("L_mode"),
            "pp_spa_parameters": json.dumps(
                result.parameters if method == "pp_spa" else {}, sort_keys=True
            ),
            "vertex_parameters": json.dumps(_jsonable(result.parameters), sort_keys=True),
            "vertex_condition_number": result.condition_number,
            "vertex_smallest_singular_value": result.smallest_singular_value,
            "vertex_runtime_seconds": result.runtime_seconds,
            "warning_flags": json.dumps(result.warnings),
            "failure_flags": json.dumps(result.failure_flags),
            "status": result.status,
            "failure_reason": result.failure_reason,
            "selected_observation_indices": json.dumps(
                _jsonable(result.selected_observation_indices)
            ),
            "selected_center_indices": json.dumps(
                _jsonable(result.selected_center_indices)
            ),
            "center_count": None if result.centers is None else len(result.centers),
            "pseudo_point_count": (
                None if result.pseudo_points is None else len(result.pseudo_points)
            ),
            "retained_point_count": (
                None
                if result.retained_point_indices is None
                else len(result.retained_point_indices)
            ),
            "discarded_point_count": (
                None
                if result.discarded_point_indices is None
                else len(result.discarded_point_indices)
            ),
        }
    )
    vertex_artifacts = {
        "vertices": result.vertices,
        "embedding_used": (
            result.embedding_used
            if result.embedding_used is not None
            else np.empty((0, embedding.shape[1]))
        ),
        "selected_observation_indices": (
            result.selected_observation_indices
            if result.selected_observation_indices is not None
            else np.empty(0, dtype=int)
        ),
        "selected_center_indices": (
            result.selected_center_indices
            if result.selected_center_indices is not None
            else np.empty(0, dtype=int)
        ),
        "centers": (
            result.centers
            if result.centers is not None
            else np.empty((0, embedding.shape[1]))
        ),
        "pseudo_points": (
            result.pseudo_points
            if result.pseudo_points is not None
            else np.empty((0, embedding.shape[1]))
        ),
        "projected_points": (
            result.projected_points
            if result.projected_points is not None
            else np.empty((0, embedding.shape[1]))
        ),
        "cluster_assignments": (
            result.cluster_assignments
            if result.cluster_assignments is not None
            else np.empty(0, dtype=int)
        ),
        "neighborhood_sizes": (
            result.neighborhood_sizes
            if result.neighborhood_sizes is not None
            else np.empty(0, dtype=int)
        ),
        "retained_point_indices": (
            result.retained_point_indices
            if result.retained_point_indices is not None
            else np.empty(0, dtype=int)
        ),
        "discarded_point_indices": (
            result.discarded_point_indices
            if result.discarded_point_indices is not None
            else np.empty(0, dtype=int)
        ),
        "neighbor_indices_json": np.asarray(
            json.dumps(_jsonable(result.neighbor_indices))
        ),
        "vertex_parameters_json": np.asarray(
            json.dumps(_jsonable(result.parameters), sort_keys=True)
        ),
    }
    if not result.success:
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fit_path, **vertex_artifacts)
        record["fit_file"] = _recorded_path(fit_path)
        record["runtime_seconds"] = time.perf_counter() - started + factor_runtime
        return record

    try:
        recovery_embedding = (
            result.embedding_used if result.embedding_used is not None else embedding
        )
        W_result = recover_W(
            recovery_embedding,
            result.vertices,
            condition_threshold=condition_threshold,
        )
        A_result = refit_A_full_l2(W_result.simplex_projected, dataset.X)
        W_hat = W_result.simplex_projected
        A_hat = A_result.A_hat
        W_aligned, A_aligned, _ = _align_topics(
            W_hat, A_hat, dataset.W_true, dataset.A_true
        )
        W_difference = W_aligned - dataset.W_true
        A_difference = A_aligned - dataset.A_true
        vertex_fro, vertex_rmse = _vertex_error(
            recovery_embedding, result.vertices, dataset.W_true
        )
        reconstruction = W_hat @ A_hat
        transformed_reconstruction = np.linalg.norm(
            prep.X_transformed - U @ np.diag(singular) @ V.T
        )
        flags = list(result.failure_flags)
        warnings = list(result.warnings) + list(W_result.warnings) + list(A_result.warnings)
        status = "ok"
        if not W_result.stable:
            status = "unstable"
            flags.append("unstable_W_recovery")
        if not A_result.converged:
            warnings.append("A_full_L2_max_iter_reached")
        record.update(
            {
                "status": status,
                "failure_reason": None,
                "warning_flags": json.dumps(sorted(set(warnings))),
                "failure_flags": json.dumps(sorted(set(flags))),
                "W_recovery_condition_number": W_result.condition_number,
                "W_recovery_smallest_singular_value": W_result.smallest_singular_value,
                "W_row_sum_min": float(W_hat.sum(axis=1).min()),
                "W_row_sum_max": float(W_hat.sum(axis=1).max()),
                "W_minimum_entry": float(W_hat.min()),
                "A_row_sum_min": float(A_hat.sum(axis=1).min()),
                "A_row_sum_max": float(A_hat.sum(axis=1).max()),
                "A_minimum_entry": float(A_hat.min()),
                "A_recovery_converged": A_result.converged,
                "A_recovery_iterations": A_result.iterations,
                "A_objective_initial": A_result.objective_history[0],
                "A_objective_final": A_result.objective_history[-1],
                "W_frobenius_error": float(np.linalg.norm(W_difference)),
                "W_l1_error": float(np.abs(W_difference).sum()),
                "W_rmse": float(np.sqrt(np.mean(W_difference**2))),
                "A_frobenius_error": float(np.linalg.norm(A_difference)),
                "A_l1_error": float(np.abs(A_difference).sum()),
                "A_rmse": float(np.sqrt(np.mean(A_difference**2))),
                "vertex_frobenius_error": vertex_fro,
                "vertex_rmse": vertex_rmse,
                "subspace_projection_error": _subspace_error(
                    recovery_embedding, dataset.W_true
                ),
                "reconstruction_frobenius": float(
                    np.linalg.norm(dataset.X - reconstruction)
                ),
                "reconstruction_rmse": float(
                    np.sqrt(np.mean((dataset.X - reconstruction) ** 2))
                ),
                "transformed_reconstruction_error": float(transformed_reconstruction),
                "original_scale_poisson_deviance": _poisson_deviance(
                    dataset.counts, reconstruction, dataset.N
                ),
                "held_out_deviance": np.nan,
                "all_finite": bool(
                    np.isfinite(W_hat).all()
                    and np.isfinite(A_hat).all()
                    and np.isfinite(result.vertices).all()
                ),
            }
        )
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fit_path,
            **vertex_artifacts,
            W_raw=W_result.raw,
            W_truncated_normalized=W_result.truncated_normalized,
            W_simplex_projected=W_result.simplex_projected,
            A_hat=A_hat,
            A_objective_history=np.asarray(A_result.objective_history),
        )
        record["fit_file"] = _recorded_path(fit_path)
    except Exception as error:  # retain numerical failures in the tidy output
        record.update(
            {
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
                "failure_flags": json.dumps(["downstream_recovery_failed"]),
            }
        )
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fit_path, **vertex_artifacts)
        record["fit_file"] = _recorded_path(fit_path)
    record["runtime_seconds"] = time.perf_counter() - started + factor_runtime
    return record


def _method_specs(config: dict[str, Any], preprocessing_name: str) -> list[dict[str, Any]]:
    methods = [dict(item) for item in config["primary_vertex_hunters"]]
    for addition in config.get("additional_vertex_runs", []):
        if preprocessing_name in addition["preprocessing_variants"]:
            methods.append(
                {
                    "method": addition["method"],
                    "embedding_source": addition.get("embedding_source", "U_hat"),
                    "parameters": addition.get("parameters", {}),
                }
            )
    return methods


def _write_plot(frame: pd.DataFrame, path: Path, stage: str) -> None:
    if "design_axis" in frame and frame.design_axis.notna().any():
        return
    valid = frame.loc[
        (frame.status == "ok")
        & (frame.embedding_source == "U_hat")
        & frame.vertex_hunter.isin(["spa_current", "svs_star", "pp_spa"])
    ].copy()
    if valid.empty:
        return
    groups = (
        valid.groupby(
            ["experiment_family", "preprocessing_variant", "vertex_hunter"],
            as_index=False,
        )[["W_rmse", "A_rmse"]]
        .median()
        .sort_values(["experiment_family", "preprocessing_variant", "vertex_hunter"])
    )
    families = groups.experiment_family.unique().tolist()
    figure, axes = plt.subplots(
        len(families), 2, figsize=(12, max(4.2, 3.8 * len(families))), squeeze=False
    )
    methods = ["spa_current", "svs_star", "pp_spa"]
    method_labels = {"spa_current": "SPA", "svs_star": "SVS*", "pp_spa": "pp-SPA"}
    colors = {"spa_current": "#4C78A8", "svs_star": "#F58518", "pp_spa": "#54A24B"}
    width = 0.24
    for family_index, family in enumerate(families):
        family_frame = groups.loc[groups.experiment_family == family]
        variants = sorted(family_frame.preprocessing_variant.unique())
        x = np.arange(len(variants), dtype=float)
        for metric_index, (metric, label) in enumerate(
            [("W_rmse", "W RMSE"), ("A_rmse", "A RMSE")]
        ):
            axis = axes[family_index, metric_index]
            for method_index, method in enumerate(methods):
                values = (
                    family_frame.loc[family_frame.vertex_hunter == method]
                    .set_index("preprocessing_variant")
                    .reindex(variants)[metric]
                    .to_numpy()
                )
                axis.bar(
                    x + (method_index - 1) * width,
                    values,
                    width=width,
                    label=method_labels[method],
                    color=colors[method],
                )
            axis.set_xticks(x, variants)
            axis.set_ylabel(label)
            axis.set_title(f"{family}: median {label}")
            axis.grid(axis="y", alpha=0.25)
            if family_index == 0 and metric_index == 1:
                axis.legend(frameon=False)
    figure.suptitle(
        f"{stage.capitalize()} primary comparisons — preliminary, not final findings"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--only-family")
    parser.add_argument(
        "--force", action="store_true", help="recompute and replace this stage's records"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = REPO_ROOT / config["output_directory"]
    figure_root = REPO_ROOT / config["figure_directory"]
    record_root = output_root / "records"
    embedding_root = output_root / "embeddings"
    fit_root = output_root / "fits"
    data_root = output_root / "generated_data"
    for directory in (record_root, embedding_root, fit_root, data_root, figure_root):
        directory.mkdir(parents=True, exist_ok=True)

    for family_spec in expanded_dataset_specs(config):
        if args.only_family and family_spec["experiment_family"] != args.only_family:
            continue
        seeds = family_spec["seeds"]
        if args.seed_limit is not None:
            seeds = seeds[: args.seed_limit]
        for seed in seeds:
            dataset = generate_dataset(family_spec, int(seed), data_root)
            for preprocessing in config["preprocessing_variants"]:
                factor_key = f"{dataset.name}_seed_{seed}_{preprocessing['name']}"
                method_specs = _method_specs(config, preprocessing["name"])
                embedding_path = embedding_root / f"{factor_key}.npz"
                expected_records = [
                    record_root
                    / (
                        f"{factor_key}_{item.get('embedding_source', 'U_hat')}_"
                        f"{item['method']}.json"
                    )
                    for item in method_specs
                ]
                expected_fits = [fit_root / f"{path.stem}.npz" for path in expected_records]
                if (
                    not args.force
                    and embedding_path.exists()
                    and all(path.exists() for path in expected_records)
                    and all(path.exists() for path in expected_fits)
                ):
                    print(f"resuming: complete {factor_key}", flush=True)
                    continue
                print(f"factorizing {factor_key}", flush=True)
                try:
                    prep, U, U_bar, V, singular, metadata, factor_runtime = _factorize(
                        dataset, preprocessing, config["graphSVD"]
                    )
                except Exception as error:
                    failure = {
                        "dataset": dataset.name,
                        "experiment_family": dataset.family,
                        "seed": seed,
                        "n": dataset.X.shape[0],
                        "p": dataset.X.shape[1],
                        "N": dataset.N,
                        "K": dataset.W_true.shape[1],
                        "requested_p": dataset.requested_p,
                        "a_zipf": dataset.a_zipf,
                        "design_axis": dataset.design_axis,
                        "design_value": dataset.design_value,
                        "data_generator": dataset.generator_name,
                        "graph_setting": dataset.graph_setting,
                        "preprocessing_variant": preprocessing["name"],
                        "threshold_method": preprocessing["threshold_method"],
                        "alpha": preprocessing.get("alpha"),
                        "weight_method": preprocessing["weight_method"],
                        "vertex_hunter": None,
                        "embedding_source": None,
                        "A_recovery_method": "A_full_L2",
                        "status": "failed",
                        "failure_reason": f"{type(error).__name__}: {error}",
                        "failure_flags": json.dumps(["factorization_failed"]),
                    }
                    path = record_root / f"{factor_key}_factorization_failed.json"
                    path.write_text(json.dumps(_jsonable(failure), indent=2) + "\n")
                    continue
                np.savez_compressed(
                    embedding_path,
                    U_hat=U,
                    U_bar=U_bar,
                    V=V,
                    singular_values=singular,
                    retained_indices=prep.threshold.retained_indices,
                    discarded_indices=prep.threshold.discarded_indices,
                    feature_weights=prep.weighting.weights,
                    eta_hat=prep.threshold.eta_hat,
                    retained_row_mass=prep.threshold.retained_row_mass,
                    rho_path_json=np.asarray(
                        json.dumps(_jsonable(metadata.get("lambd_history", [])))
                    ),
                    cv_path_json=np.asarray(
                        json.dumps(_jsonable(metadata.get("cv_history", [])))
                    ),
                    convergence_json=np.asarray(
                        json.dumps(_jsonable(metadata.get("score_history", [])))
                    ),
                    U_bar_history=np.asarray(metadata.get("U_bar_history", [U_bar])),
                    U_hat_history=np.asarray(metadata.get("U_hat_history", [U])),
                    V_hat_history=np.asarray(metadata.get("V_hat_history", [V])),
                    singular_value_history=np.asarray(
                        metadata.get("singular_value_history", [singular])
                    ),
                )
                caches: dict[str, dict[int, Any]] = {"U_hat": {}, "U_bar": {}}
                for method_spec in method_specs:
                    embedding_source = method_spec.get("embedding_source", "U_hat")
                    method = method_spec["method"]
                    record_key = (
                        f"{factor_key}_{embedding_source}_{method}"
                    )
                    record_path = record_root / f"{record_key}.json"
                    fit_path = fit_root / f"{record_key}.npz"
                    if record_path.exists() and not args.force:
                        print(f"resuming: kept {record_path.name}", flush=True)
                        continue
                    record = _run_hunter(
                        dataset,
                        preprocessing,
                        prep,
                        U,
                        U_bar,
                        V,
                        singular,
                        metadata,
                        factor_runtime,
                        method_spec,
                        config["condition_threshold"],
                        caches[embedding_source],
                        fit_path,
                    )
                    record["embedding_file"] = _recorded_path(embedding_path)
                    record_path.write_text(json.dumps(_jsonable(record), indent=2) + "\n")
                    print(
                        f"  {embedding_source}/{method}: {record['status']}", flush=True
                    )

    records = [json.loads(path.read_text()) for path in sorted(record_root.glob("*.json"))]
    frame = pd.DataFrame(records)
    tidy_path = output_root / f"{config['stage']}_tidy.csv"
    frame.to_csv(tidy_path, index=False)
    (output_root / f"{config['stage']}_tidy.json").write_text(
        json.dumps(_jsonable(records), indent=2) + "\n"
    )
    if not frame.empty and "W_rmse" in frame:
        summary = (
            frame.groupby(
                [
                    "experiment_family",
                    "design_axis",
                    "design_value",
                    "preprocessing_variant",
                    "vertex_hunter",
                    "embedding_source",
                ],
                dropna=False,
            )
            .agg(
                runs=("status", "size"),
                successful=("status", lambda values: int((values == "ok").sum())),
                failures=("status", lambda values: int((values == "failed").sum())),
                unstable=("status", lambda values: int((values == "unstable").sum())),
                W_rmse_median=("W_rmse", "median"),
                W_rmse_mean=("W_rmse", "mean"),
                W_rmse_q25=("W_rmse", lambda values: values.quantile(0.25)),
                W_rmse_q75=("W_rmse", lambda values: values.quantile(0.75)),
                A_rmse_median=("A_rmse", "median"),
                A_rmse_mean=("A_rmse", "mean"),
                A_rmse_q25=("A_rmse", lambda values: values.quantile(0.25)),
                A_rmse_q75=("A_rmse", lambda values: values.quantile(0.75)),
                runtime_median=("runtime_seconds", "median"),
            )
            .reset_index()
        )
        summary.to_csv(output_root / f"{config['stage']}_summary.csv", index=False)
        _write_plot(
            frame,
            figure_root / f"{config['stage']}_preliminary_errors.png",
            config["stage"],
        )
    print(tidy_path, flush=True)


if __name__ == "__main__":
    main()
