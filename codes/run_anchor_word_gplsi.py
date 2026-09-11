#!/usr/bin/env python3
"""Restartable Tran word-decay experiments for document- and word-side GpLSI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from gplsi.anchor_word import build_word_profile, recover_W_from_word_vertices
from gplsi.baselines import fit_lda, fit_spatial_lda
from gplsi.generate_topic_model import (
    generate_W_strong,
    generate_graph,
    generate_weights_edge,
)
from gplsi.graphSVD import graphSVD
from gplsi.preprocessing import preprocess_features, weighted_debiased_correction
from gplsi.recovery import (
    recover_W,
    refit_A_current,
    refit_A_full_poisson,
)
from gplsi.topicscore import (
    fit_topicscore_graph_denoised,
    fit_topicscore_raw,
)
from gplsi.vertex_hunting import vertex_hunt


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAN_ROOT = Path(os.environ.get(
    "GPLSI_TRAN_ROOT", str(REPO_ROOT / "external_references/topic-modeling")
))
TRAN_GENERATOR = TRAN_ROOT / "r/experiments/synthetic/synthetic_dataset.R"


@dataclass
class Dataset:
    name: str
    design_variant: str
    experiment_family: str
    seed: int
    replicate: int
    counts: np.ndarray
    train_counts: np.ndarray
    test_counts: np.ndarray
    train_N: int
    test_N: int
    W_true: np.ndarray
    A_true: np.ndarray
    factor_population: np.ndarray
    source_population: np.ndarray
    vocabulary_r: np.ndarray
    anchor_indices_r_by_topic: list[np.ndarray]
    coordinates: pd.DataFrame | None
    edge_df: pd.DataFrame | None
    graph_weights: Any
    graph_setting: str
    factorization: str
    requested_p: int
    word_decay_parameter: float
    n_anchors: int
    delta_anchor: float
    split_seed: int

    @property
    def X(self) -> np.ndarray:
        return self.train_counts / float(self.train_N)


@dataclass
class SpectralBlock:
    prep: Any
    positive_feature_indices: np.ndarray
    retained_indices: np.ndarray
    U_hat: np.ndarray
    U_bar: np.ndarray
    V_hat: np.ndarray
    singular_values: np.ndarray
    M_hat_retained: np.ndarray
    metadata: dict[str, Any]
    runtime_seconds: float


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=float)


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _anchor_groups(K: int, n_anchors: int) -> list[np.ndarray]:
    return [
        np.arange(k * n_anchors + 1, (k + 1) * n_anchors + 1, dtype=int)
        for k in range(K)
    ]


def _fixed_count_split(
    counts: np.ndarray, seed: int, test_fraction: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    D = np.asarray(np.rint(counts), dtype=int)
    totals = D.sum(axis=1)
    if not np.all(totals == totals[0]):
        raise ValueError("fixed-count thinning requires equal document lengths")
    N = int(totals[0])
    test_N = max(1, int(round(test_fraction * N)))
    train_N = N - test_N
    if train_N < 1:
        raise ValueError("count split leaves no training observations")
    rng = np.random.default_rng(seed)
    train = np.vstack(
        [rng.multivariate_hypergeometric(row, train_N) for row in D]
    )
    test = D - train
    return train.astype(float), test.astype(float), train_N, test_N


def _load_or_create_split(
    target: Path,
    counts: np.ndarray,
    seed: int,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    path = target / "count_split.npz"
    if path.exists():
        saved = np.load(path)
        return (
            saved["train_counts"],
            saved["test_counts"],
            int(saved["train_N"]),
            int(saved["test_N"]),
        )
    train, test, train_N, test_N = _fixed_count_split(counts, seed, test_fraction)
    np.savez_compressed(
        path,
        train_counts=train,
        test_counts=test,
        train_N=train_N,
        test_N=test_N,
        split_seed=seed,
    )
    return train, test, train_N, test_N


def _create_exact_tran(spec: dict[str, Any], target: Path) -> None:
    _run(
        [
            "Rscript",
            str(REPO_ROOT / "codes/generate_tran_pilot_data.R"),
            str(TRAN_GENERATOR),
            str(target),
            str(spec["seed"]),
            str(spec["n"]),
            str(spec["p"]),
            str(spec["N"]),
            str(spec["K"]),
            str(spec["alpha_dirichlet"]),
            str(spec["n_anchors"]),
            str(spec["delta_anchor"]),
            str(spec["a_zipf"]),
            str(spec["offset_zipf"]),
        ],
        cwd=TRAN_ROOT,
    )


def _create_graph_adapted_tran(spec: dict[str, Any], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(spec["seed"]))
    coordinates = generate_graph(
        spec["N"], spec["n"], spec["p"], spec["K"], spec["rt"], spec["n_clusters"]
    )
    W_topic_document = generate_W_strong(
        coordinates, spec["N"], spec["n"], spec["p"], spec["K"], spec["rt"]
    )
    W_document_topic = W_topic_document.T
    coordinates.to_csv(target / "coordinates.csv", index=False)
    pd.DataFrame(W_document_topic).to_csv(
        target / "W_document_by_topic_input.csv", index=False
    )
    A_full_path = target / "A_full_feature_by_topic.csv"
    _run(
        [
            "Rscript",
            str(REPO_ROOT / "codes/generate_tran_topic_matrix.R"),
            str(TRAN_GENERATOR),
            str(A_full_path),
            str(spec["seed"]),
            str(spec["n"]),
            str(spec["p"]),
            str(spec["K"]),
            str(spec["alpha_dirichlet"]),
            str(spec["n_anchors"]),
            str(spec["delta_anchor"]),
            str(spec["a_zipf"]),
            str(spec["offset_zipf"]),
        ],
        cwd=TRAN_ROOT,
    )
    count_seed = int(spec["seed"]) + int(spec.get("count_seed_offset", 10_000_019))
    _run(
        [
            "Rscript",
            str(REPO_ROOT / "codes/generate_tran_counts_from_factors.R"),
            str(A_full_path),
            str(target / "W_document_by_topic_input.csv"),
            str(target),
            str(count_seed),
            str(spec["N"]),
            str(spec["n_anchors"]),
        ]
    )
    (target / "graph_adaptation_metadata.json").write_text(
        json.dumps(
            {
                "Tran_A_source": str(TRAN_GENERATOR),
                "GpLSI_graph_W_source": "src/gplsi/generate_topic_model.py",
                "seed": spec["seed"],
                "count_seed": count_seed,
                "graph_does_not_use_true_W_or_A": True,
            },
            indent=2,
        )
        + "\n"
    )


def generate_dataset(
    spec: dict[str, Any], data_root: Path, test_fraction: float
) -> Dataset:
    value_label = f"{float(spec['a_zipf']):g}".replace(".", "p")
    sweep_label = ""
    if spec.get("sweep_parameter") is not None:
        sweep_value = f"{float(spec['sweep_value']):g}".replace(".", "p")
        sweep_label = f"_{spec['sweep_parameter']}_{sweep_value}"
    name = (
        f"{spec['design_variant']}{sweep_label}_decay_{value_label}_seed_{spec['seed']}"
    )
    target = data_root / name
    target.mkdir(parents=True, exist_ok=True)
    required = [target / "counts.csv", target / "vocabulary.csv"]
    if not all(path.exists() for path in required):
        if spec["design_variant"] == "tran_mixed_word_decay_exact":
            _create_exact_tran(spec, target)
        elif spec["design_variant"] == "tran_mixed_word_decay_graph_adapted":
            _create_graph_adapted_tran(spec, target)
        else:
            raise ValueError(f"unknown design variant {spec['design_variant']!r}")

    if spec["design_variant"] == "tran_mixed_word_decay_exact":
        counts = _read_csv(target / "counts.csv")
        A_true = _read_csv(target / "A_feature_by_topic.csv").T
        W_true = _read_csv(target / "W_topic_by_document.csv").T
        source_population = _read_csv(target / "population_document_by_feature.csv")
        coordinates = None
        edge_df = None
        graph_weights = None
        graph_setting = "none_exact_Tran"
        factorization = "pLSI"
    else:
        counts = _read_csv(target / "counts.csv")
        A_true = _read_csv(target / "A_feature_by_topic.csv").T
        W_true = _read_csv(target / "W_document_by_topic.csv")
        source_population = _read_csv(target / "source_population_document_by_word.csv")
        coordinates = pd.read_csv(target / "coordinates.csv")
        graph_weights, edge_df = generate_weights_edge(
            coordinates, int(spec["nearest_n"]), float(spec["phi"])
        )
        edge_path = target / "graph_edges.csv"
        if not edge_path.exists():
            edge_df.to_csv(edge_path, index=False)
        graph_setting = "GpLSI_coordinate_RBF_nearest_neighbor"
        factorization = "graphSVD"
    factor_population = W_true @ A_true
    vocabulary = pd.read_csv(target / "vocabulary.csv").iloc[:, 0].to_numpy(int)
    split_seed = int(spec["seed"]) + int(spec.get("split_seed_offset", 20_000_039))
    train, test, train_N, test_N = _load_or_create_split(
        target, counts, split_seed, test_fraction
    )
    return Dataset(
        name=name,
        design_variant=spec["design_variant"],
        experiment_family=spec["experiment_family"],
        seed=int(spec["seed"]),
        replicate=int(spec["replicate"]),
        counts=counts,
        train_counts=train,
        test_counts=test,
        train_N=train_N,
        test_N=test_N,
        W_true=W_true,
        A_true=A_true,
        factor_population=factor_population,
        source_population=source_population,
        vocabulary_r=vocabulary,
        anchor_indices_r_by_topic=_anchor_groups(int(spec["K"]), int(spec["n_anchors"])),
        coordinates=coordinates,
        edge_df=edge_df,
        graph_weights=graph_weights,
        graph_setting=graph_setting,
        factorization=factorization,
        requested_p=int(spec["p"]),
        word_decay_parameter=float(spec["a_zipf"]),
        n_anchors=int(spec["n_anchors"]),
        delta_anchor=float(spec["delta_anchor"]),
        split_seed=split_seed,
    )


def expand_design(config: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in config["seeds"]]
    sweep = config.get("sweep")
    if sweep is None:
        sweep_points: list[tuple[str | None, float | None]] = [(None, None)]
    else:
        parameter = str(sweep["parameter"])
        if parameter not in {"n", "N", "p", "a_zipf"}:
            raise ValueError(f"unsupported sweep parameter {parameter!r}")
        sweep_points = [(parameter, float(value)) for value in sweep["values"]]
    for design in config["designs"]:
        for replicate, seed in enumerate(seeds, start=1):
            for decay in config["word_decay_grid"]:
                for sweep_parameter, sweep_value in sweep_points:
                    spec = dict(config["base_parameters"])
                    spec.update(design)
                    spec.update(seed=seed, replicate=replicate, a_zipf=float(decay))
                    if sweep_parameter is not None and sweep_value is not None:
                        cast_value: int | float
                        cast_value = int(sweep_value) if sweep_parameter in {"n", "N", "p"} else sweep_value
                        spec[sweep_parameter] = cast_value
                        spec["sweep_parameter"] = sweep_parameter
                        spec["sweep_value"] = cast_value
                    output.append(spec)
    return output


def factorize(
    dataset: Dataset,
    preprocessing: dict[str, Any],
    graph_config: dict[str, Any],
    target: Path,
) -> SpectralBlock:
    npz_path = target / "spectral_factors.npz"
    metadata_path = target / "metadata.json"
    X = dataset.X
    positive = np.flatnonzero(X.mean(axis=0) > 0)
    if positive.size < dataset.W_true.shape[1]:
        raise ValueError("fewer than K positive-frequency training words")
    true_A_positive = dataset.A_true[:, positive]
    population_positive = dataset.factor_population[:, positive]
    prep = preprocess_features(
        X[:, positive],
        dataset.train_N,
        threshold_method=preprocessing["threshold_method"],
        alpha=float(preprocessing.get("alpha", 0.005)),
        weight_method=preprocessing["weight_method"],
        tau=float(preprocessing.get("tau", 0.0)),
        weight_cap=preprocessing.get("weight_cap"),
        weight_common_scale=preprocessing.get("weight_common_scale", "none"),
        K=dataset.W_true.shape[1],
        true_A=true_A_positive,
        population_M=population_positive,
        fail_on_rank_loss=True,
    )
    retained = positive[prep.threshold.retained_indices]
    if npz_path.exists() and metadata_path.exists():
        saved = np.load(npz_path)
        return SpectralBlock(
            prep=prep,
            positive_feature_indices=positive,
            retained_indices=saved["retained_indices"],
            U_hat=saved["U_hat"],
            U_bar=saved["U_bar"],
            V_hat=saved["V_hat"],
            singular_values=saved["singular_values"],
            M_hat_retained=saved["M_hat_retained"],
            metadata=json.loads(metadata_path.read_text()),
            runtime_seconds=float(saved["runtime_seconds"]),
        )

    K = dataset.W_true.shape[1]
    started = time.perf_counter()
    if dataset.factorization == "pLSI":
        U_full, singular_full, Vt_full = np.linalg.svd(
            prep.X_transformed, full_matrices=False
        )
        U_hat = U_full[:, :K]
        V_hat = Vt_full[:K].T
        singular = singular_full[:K]
        U_bar = U_hat.copy()
        metadata = {
            "graph_svd_version": "not_applicable_pLSI",
            "initialization": "direct_svd",
            "rho": None,
            "rho_path": [],
            "cv_path": [],
            "score_history": [],
        }
    else:
        initialization = preprocessing.get("initialization", "current")
        correction = None
        if initialization in {"weighted_debiased", "weighted_debiased_mean_N_approx"}:
            eta = X[:, retained].mean(axis=0)
            correction = weighted_debiased_correction(
                eta,
                prep.weighting.weights,
                n=X.shape[0],
                N=dataset.train_N,
            )
        np.random.seed(dataset.seed + 1009 * int(preprocessing["order"]))
        output = graphSVD(
            prep.X_transformed,
            dataset.train_N,
            K,
            dataset.edge_df,
            dataset.graph_weights,
            float(graph_config["lamb_start"]),
            float(graph_config["step_size"]),
            int(graph_config["grid_len"]),
            int(graph_config["maxiter"]),
            float(graph_config["eps"]),
            int(graph_config.get("verbose", 0)),
            True,
            initialization=initialization,
            debias_correction=correction,
            return_metadata=True,
        )
        U_hat, V_hat, L, _, _, _, rho, cv_errors, niter, raw_metadata = output
        U_bar = raw_metadata["U_bar"]
        singular = np.diag(L)
        metadata = {
            "graph_svd_version": "iterative",
            "initialization": initialization,
            "rho": float(rho),
            "rho_path": _jsonable(raw_metadata["lambd_history"]),
            "cv_path": _jsonable(raw_metadata["cv_history"]),
            "rho_grid": _jsonable(raw_metadata["lambd_grid"]),
            "score_history": _jsonable(raw_metadata["score_history"]),
            "n_iterations": int(niter),
            "final_cv_errors": _jsonable(cv_errors),
        }
    runtime = time.perf_counter() - started
    M_hat = ((U_hat * singular[None, :]) @ V_hat.T) / prep.weighting.weights[None, :]
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        retained_indices=retained,
        positive_feature_indices=positive,
        U_hat=U_hat,
        U_bar=U_bar,
        V_hat=V_hat,
        singular_values=singular,
        M_hat_retained=M_hat,
        runtime_seconds=runtime,
    )
    all_features = np.arange(dataset.X.shape[1], dtype=int)
    discarded = np.setdiff1d(all_features, retained, assume_unique=True)
    np.savez_compressed(
        target / "preprocessing.npz",
        retained_indices=retained,
        discarded_indices=discarded,
        zero_frequency_indices=np.setdiff1d(
            all_features, positive, assume_unique=True
        ),
        eta_hat=dataset.X.mean(axis=0),
        retained_row_mass=dataset.X[:, retained].sum(axis=1),
        retained_topic_mass=dataset.A_true[:, retained].sum(axis=1),
        weights=prep.weighting.weights,
        transformed_singular_values=prep.weighting.transformed_singular_values,
    )
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2) + "\n")
    return SpectralBlock(
        prep=prep,
        positive_feature_indices=positive,
        retained_indices=retained,
        U_hat=U_hat,
        U_bar=U_bar,
        V_hat=V_hat,
        singular_values=singular,
        M_hat_retained=M_hat,
        metadata=metadata,
        runtime_seconds=runtime,
    )


def _poisson_deviance(counts: np.ndarray, probabilities: np.ndarray, N: int) -> float:
    expected = np.maximum(float(N) * probabilities, 1e-12)
    observed = np.asarray(counts, dtype=float)
    terms = expected - observed
    positive = observed > 0
    terms[positive] += observed[positive] * np.log(observed[positive] / expected[positive])
    return float(2 * np.sum(terms))


def _graph_total_variation(W: np.ndarray, edge_df: pd.DataFrame | None) -> float:
    if edge_df is None or len(edge_df) == 0:
        return np.nan
    differences = W[edge_df.src.to_numpy(int)] - W[edge_df.tgt.to_numpy(int)]
    weights = np.sqrt(edge_df.weight.to_numpy(float))
    return float(np.sum(np.linalg.norm(differences, axis=1) * weights))


def _top_word_recovery(A_hat: np.ndarray, A_true: np.ndarray, top_n: int = 10) -> float:
    size = min(top_n, A_true.shape[1])
    scores = []
    for topic in range(A_true.shape[0]):
        truth = set(np.argsort(-A_true[topic])[:size].tolist())
        estimate = set(np.argsort(-A_hat[topic])[:size].tolist())
        scores.append(len(truth & estimate) / size)
    return float(np.mean(scores))


def _aligned_metrics(
    W_hat: np.ndarray,
    A_hat: np.ndarray,
    dataset: Dataset,
    retained_indices: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    similarity = np.asarray(W_hat).T @ dataset.W_true
    estimated, truth = linear_sum_assignment(1.0 - np.abs(similarity))
    mapping = np.zeros((W_hat.shape[1], dataset.W_true.shape[1]))
    mapping[estimated, truth] = 1.0
    W = W_hat @ mapping
    A = mapping.T @ A_hat
    W_diff = W - dataset.W_true
    A_diff = A - dataset.A_true
    row_errors = np.sum(np.abs(W_diff), axis=1)
    purity = np.max(dataset.W_true, axis=1)

    def group_error(mask: np.ndarray) -> float:
        return float(np.mean(row_errors[mask])) if np.any(mask) else np.nan

    W_metrics = {
        "W_frobenius_error": float(np.linalg.norm(W_diff)),
        "W_normalized_frobenius_error": float(
            np.linalg.norm(W_diff) / max(np.linalg.norm(dataset.W_true), 1e-15)
        ),
        "W_rmse": float(np.sqrt(np.mean(W_diff**2))),
        "W_rowwise_l1_mean": float(np.mean(row_errors)),
        "W_max_row_l1_error": float(np.max(row_errors)),
        "W_graph_total_variation": _graph_total_variation(W, dataset.edge_df),
        "W_pure_document_l1": group_error(purity >= 0.9),
        "W_near_pure_document_l1": group_error((purity >= 0.6) & (purity < 0.9)),
        "W_highly_mixed_document_l1": group_error(purity < 0.6),
    }
    eta = dataset.X.mean(axis=0)
    common = eta >= np.quantile(eta, 0.75)
    tail = eta <= np.quantile(eta, 0.25)
    root_true = np.sqrt(np.maximum(dataset.A_true, 0))
    root_hat = np.sqrt(np.maximum(A, 0))
    midpoint = 0.5 * (dataset.A_true + A)
    eps = 1e-15
    js = 0.5 * np.sum(dataset.A_true * np.log((dataset.A_true + eps) / (midpoint + eps)))
    js += 0.5 * np.sum(A * np.log((A + eps) / (midpoint + eps)))
    A_metrics = {
        "A_frobenius_error": float(np.linalg.norm(A_diff)),
        "A_normalized_frobenius_error": float(
            np.linalg.norm(A_diff) / max(np.linalg.norm(dataset.A_true), 1e-15)
        ),
        "A_rowwise_l1_mean": float(np.mean(np.sum(np.abs(A_diff), axis=1))),
        "A_mean_topic_TV": float(np.sum(np.abs(A_diff)) / (2 * A.shape[0])),
        "A_hellinger_mean": float(np.mean(np.linalg.norm(root_true - root_hat, axis=1) / np.sqrt(2))),
        "A_jensen_shannon_total": float(js),
        "A_top_word_recovery": _top_word_recovery(A, dataset.A_true),
        "A_common_word_rmse": float(np.sqrt(np.mean(A_diff[:, common] ** 2))),
        "A_tail_word_rmse": float(np.sqrt(np.mean(A_diff[:, tail] ** 2))),
    }
    probabilities = np.maximum(W @ A, 0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    fit_metrics = {
        "original_scale_reconstruction_frobenius": float(
            np.linalg.norm(dataset.X - probabilities)
        ),
        "training_poisson_deviance": _poisson_deviance(
            dataset.train_counts, probabilities, dataset.train_N
        ),
        "held_out_poisson_deviance": _poisson_deviance(
            dataset.test_counts, probabilities, dataset.test_N
        ),
    }
    if retained_indices is not None:
        retained = np.asarray(retained_indices, dtype=int)
        fit_metrics["retained_topic_mass_min"] = float(
            np.min(dataset.A_true[:, retained].sum(axis=1))
        )
    return W_metrics, A_metrics, fit_metrics, W, A


def _anchor_diagnostics(
    dataset: Dataset, retained: np.ndarray, selected: np.ndarray | None
) -> dict[str, Any]:
    retained_original = dataset.vocabulary_r[np.asarray(retained, dtype=int)]
    counts = [
        int(np.intersect1d(group, retained_original).size)
        for group in dataset.anchor_indices_r_by_topic
    ]
    failure = bool(dataset.n_anchors > 0 and any(value == 0 for value in counts))
    if selected is None or len(selected) == 0:
        purities: list[float] = []
        exact: list[bool] = []
        selected_original: list[int] = []
    else:
        selected = np.asarray(selected, dtype=int)
        selected_original = dataset.vocabulary_r[selected].astype(int).tolist()
        word_topics = dataset.A_true[:, selected].T
        denominators = word_topics.sum(axis=1)
        purities = (np.max(word_topics, axis=1) / np.maximum(denominators, 1e-15)).tolist()
        exact = [bool(np.count_nonzero(row > 1e-14) == 1) for row in word_topics]
    return {
        "retained_anchor_count_by_topic": counts,
        "anchor_survival_failure": failure,
        "selected_vocabulary_indices_r": selected_original,
        "selected_word_purity": purities,
        "selected_words_exact_anchors": exact,
    }


def _vertex_truth_error(
    dataset: Dataset,
    U: np.ndarray,
    vertices: np.ndarray,
    side: str,
) -> tuple[float, float]:
    if side == "anchor_word":
        b = dataset.W_true.mean(axis=0)
        truth_vertices = (dataset.W_true.T @ U) / b[:, None]
    else:
        truth_vertices, _, _, _ = np.linalg.lstsq(dataset.W_true, U, rcond=None)
    distances = np.linalg.norm(
        vertices[:, None, :] - truth_vertices[None, :, :], axis=2
    )
    estimated, truth = linear_sum_assignment(distances)
    difference = vertices[estimated] - truth_vertices[truth]
    return float(np.linalg.norm(difference)), float(np.sqrt(np.mean(difference**2)))


def _base_record(
    dataset: Dataset,
    preprocessing: dict[str, Any] | None,
    block: SpectralBlock | None,
) -> dict[str, Any]:
    if preprocessing is None or block is None:
        prep_values = {
            "preprocessing_variant": "native",
            "threshold_method": "native",
            "alpha": None,
            "threshold_value": None,
            "retained_feature_count": dataset.X.shape[1],
            "retained_feature_fraction": 1.0,
            "zero_frequency_training_columns_removed": 0,
            "weight_method": "native",
            "tau": None,
            "weight_cap": None,
            "weight_quantiles": None,
            "transformed_condition_number": None,
            "rho": None,
            "rho_path": None,
            "graph_svd_version": "not_applicable",
            "runtime_spectral": 0.0,
        }
    else:
        threshold = block.prep.threshold
        weighting = block.prep.weighting
        prep_values = {
            "preprocessing_variant": preprocessing["name"],
            "threshold_method": threshold.effective_method,
            "alpha": threshold.alpha,
            "threshold_value": threshold.threshold_value,
            "retained_feature_count": int(len(block.retained_indices)),
            "retained_feature_fraction": float(len(block.retained_indices) / dataset.X.shape[1]),
            "zero_frequency_training_columns_removed": int(
                dataset.X.shape[1] - len(block.positive_feature_indices)
            ),
            "weight_method": weighting.effective_method,
            "tau": weighting.tau,
            "weight_cap": weighting.cap,
            "weight_quantiles": json.dumps(_jsonable(weighting.quantiles)),
            "transformed_condition_number": weighting.transformed_condition_number,
            "rho": block.metadata.get("rho"),
            "rho_path": json.dumps(_jsonable(block.metadata.get("rho_path", []))),
            "graph_svd_version": block.metadata.get("graph_svd_version"),
            "runtime_spectral": block.runtime_seconds,
        }
    return {
        "dataset": dataset.name,
        "design_variant": dataset.design_variant,
        "experiment_family": dataset.experiment_family,
        "seed": dataset.seed,
        "replicate": dataset.replicate,
        "n": dataset.X.shape[0],
        "p": dataset.X.shape[1],
        "requested_p": dataset.requested_p,
        "N": int(dataset.counts.sum(axis=1)[0]),
        "train_N": dataset.train_N,
        "test_N": dataset.test_N,
        "K": dataset.W_true.shape[1],
        "word_decay_parameter": dataset.word_decay_parameter,
        "a_zipf": dataset.word_decay_parameter,
        "n_anchors": dataset.n_anchors,
        "delta_anchor": dataset.delta_anchor,
        "graph_setting": dataset.graph_setting,
        "split_seed": dataset.split_seed,
        **prep_values,
    }


def _failed_record(base: dict[str, Any], *, family: str, reason: str, **values: Any) -> dict[str, Any]:
    return {
        **base,
        "estimator_family": family,
        "simplex_side": values.pop("simplex_side", "not_applicable"),
        "profile_source": values.pop("profile_source", None),
        "vertex_hunter": values.pop("vertex_hunter", "not_applicable"),
        "A_recovery_method": values.pop("A_recovery_method", "native"),
        "W_recovery_method": values.pop("W_recovery_method", "native"),
        "W_fit_id": values.pop("W_fit_id", None),
        "status": "failed",
        "failure_reason": reason,
        "runtime_vertex": values.pop("runtime_vertex", 0.0),
        "runtime_A_recovery": values.pop("runtime_A_recovery", 0.0),
        "runtime_total": values.pop("runtime_total", base.get("runtime_spectral", 0.0)),
        **values,
    }


def _method_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record.get(name)
        for name in (
            "dataset",
            "estimator_family",
            "simplex_side",
            "preprocessing_variant",
            "vertex_hunter",
            "A_recovery_method",
        )
    )


def _expected_gplsi_keys(
    dataset: Dataset,
    preprocessing: dict[str, Any],
    vertex_specs: list[dict[str, Any]],
    recovery_config: dict[str, Any],
) -> set[tuple[Any, ...]]:
    """Return the complete result-key grid for one saved spectral block."""

    keys: set[tuple[Any, ...]] = set()
    for side, family in (
        ("document", "gplsi_document"),
        ("anchor_word", "gplsi_anchor_word_profile"),
    ):
        for vertex_spec in vertex_specs:
            for A_method in recovery_config["A_recovery_methods"]:
                keys.add(
                    _method_key(
                        {
                            "dataset": dataset.name,
                            "estimator_family": family,
                            "simplex_side": side,
                            "preprocessing_variant": preprocessing["name"],
                            "vertex_hunter": vertex_spec["method"],
                            "A_recovery_method": A_method,
                        }
                    )
                )
    return keys


def run_gplsi_block(
    dataset: Dataset,
    preprocessing: dict[str, Any],
    block: SpectralBlock,
    vertex_specs: list[dict[str, Any]],
    recovery_config: dict[str, Any],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = _base_record(dataset, preprocessing, block)
    profile = build_word_profile(
        dataset.X,
        block.U_hat,
        block.V_hat,
        block.singular_values,
        block.prep.weighting.weights,
        block.retained_indices,
        profile_source="denoised_original_scale",
    )
    clouds = {
        "document": (block.U_hat, None),
        "anchor_word": (profile.Z_hat, profile),
    }
    caches: dict[str, dict[int, Any]] = {"document": {}, "anchor_word": {}}
    for side, (cloud, profile_result) in clouds.items():
        for vertex_spec in vertex_specs:
            method = vertex_spec["method"]
            parameters = dict(vertex_spec.get("parameters", {}))
            if method in {"svs", "svs_star"}:
                parameters["center_cache"] = caches[side]
            vertex = vertex_hunt(
                cloud,
                dataset.W_true.shape[1],
                method,
                random_state=dataset.seed,
                condition_threshold=float(recovery_config["condition_threshold"]),
                **parameters,
            )
            family = "gplsi_document" if side == "document" else "gplsi_anchor_word_profile"
            selected_local = vertex.selected_observation_indices
            selected_global = (
                None
                if selected_local is None
                else (
                    np.asarray(selected_local, dtype=int)
                    if side == "document"
                    else block.retained_indices[np.asarray(selected_local, dtype=int)]
                )
            )
            anchor_diag = _anchor_diagnostics(
                dataset,
                block.retained_indices,
                selected_global if side == "anchor_word" else None,
            )
            W_fit_id = hashlib.sha256(
                f"{dataset.name}|{preprocessing['name']}|{side}|{method}".encode()
            ).hexdigest()[:20]
            common = {
                "simplex_side": side,
                "profile_source": (
                    profile_result.profile_source if profile_result is not None else None
                ),
                "vertex_hunter": method,
                "L": vertex.parameters.get("L"),
                "L_mode": vertex.parameters.get("L_mode"),
                "pp_spa_parameters": json.dumps(_jsonable(vertex.parameters))
                if method == "pp_spa"
                else None,
                "selected_vertex_indices": json.dumps(
                    _jsonable(vertex.selected_observation_indices)
                ),
                "selected_vocabulary_indices": json.dumps(
                    anchor_diag["selected_vocabulary_indices_r"]
                ),
                "selected_word_purity": json.dumps(anchor_diag["selected_word_purity"]),
                "selected_words_exact_anchors": json.dumps(
                    anchor_diag["selected_words_exact_anchors"]
                ),
                "retained_anchor_count_by_topic": json.dumps(
                    anchor_diag["retained_anchor_count_by_topic"]
                ),
                "anchor_survival_failure": anchor_diag["anchor_survival_failure"],
                "vertex_condition_number": vertex.condition_number,
                "smallest_vertex_singular_value": vertex.smallest_singular_value,
                "word_profile_identity_error": (
                    profile_result.identity_error if profile_result is not None else None
                ),
                "runtime_vertex": vertex.runtime_seconds,
                "W_fit_id": W_fit_id,
            }
            if not vertex.success:
                for A_method in recovery_config["A_recovery_methods"]:
                    records.append(
                        _failed_record(
                            base,
                            family=family,
                            reason=vertex.failure_reason or "vertex hunting failed",
                            A_recovery_method=A_method,
                            W_recovery_method=(
                                "document_vertex_solve" if side == "document" else "word_G_b_simplex"
                            ),
                            **common,
                        )
                    )
                continue
            try:
                if side == "document":
                    W_result = recover_W(
                        vertex.embedding_used,
                        vertex.vertices,
                        condition_threshold=float(recovery_config["condition_threshold"]),
                    )
                    W_hat = W_result.simplex_projected
                    W_recovery_method = "document_vertex_solve"
                    prevalence_residual = None
                    b_hat = None
                    recovery_warnings = W_result.warnings
                else:
                    coordinate_signs = np.asarray(
                        vertex.parameters.get(
                            "coordinate_signs", np.ones(block.U_hat.shape[1])
                        ),
                        dtype=float,
                    )
                    U_for_recovery = block.U_hat * coordinate_signs[None, :]
                    W_result = recover_W_from_word_vertices(
                        U_for_recovery,
                        vertex.vertices,
                        condition_threshold=float(recovery_config["condition_threshold"]),
                    )
                    W_hat = W_result.simplex_projected
                    W_recovery_method = "word_G_b_simplex"
                    prevalence_residual = W_result.prevalence_residual
                    b_hat = W_result.b_hat
                    recovery_warnings = W_result.warnings
            except Exception as error:
                for A_method in recovery_config["A_recovery_methods"]:
                    records.append(
                        _failed_record(
                            base,
                            family=family,
                            reason=f"{type(error).__name__}: {error}",
                            A_recovery_method=A_method,
                            W_recovery_method=(
                                "document_vertex_solve" if side == "document" else "word_G_b_simplex"
                            ),
                            **common,
                        )
                    )
                continue

            artifact_root.mkdir(parents=True, exist_ok=True)
            W_artifact = artifact_root / f"{W_fit_id}_W_and_vertices.npz"
            W_payload: dict[str, Any] = {
                "W_hat_raw": W_result.raw,
                "W_hat_truncated_normalized": W_result.truncated_normalized,
                "W_hat_simplex_projected": W_hat,
                "vertices": vertex.vertices,
                "selected_observation_indices": (
                    np.asarray(vertex.selected_observation_indices, dtype=int)
                    if vertex.selected_observation_indices is not None
                    else np.array([], dtype=int)
                ),
                "selected_vocabulary_indices": (
                    np.asarray(selected_global, dtype=int)
                    if side == "anchor_word" and selected_global is not None
                    else np.array([], dtype=int)
                ),
            }
            if side == "anchor_word":
                W_payload.update(
                    G_hat=W_result.G_hat,
                    b_hat=W_result.b_hat,
                    prevalence_objective_history=np.asarray(W_result.objective_history),
                    Z_hat=profile_result.Z_hat,
                )
            np.savez_compressed(W_artifact, **W_payload)
            U_for_vertex_truth = (
                block.U_hat * coordinate_signs[None, :]
                if side == "anchor_word"
                else vertex.embedding_used
            )
            vertex_fro, vertex_rmse = _vertex_truth_error(
                dataset, U_for_vertex_truth, vertex.vertices, side
            )

            W_bytes = W_hat.tobytes()
            for A_method in recovery_config["A_recovery_methods"]:
                started = time.perf_counter()
                try:
                    if A_method == "current":
                        A_result = refit_A_current(W_hat, dataset.X)
                    elif A_method == "poisson_full":
                        A_result = refit_A_full_poisson(
                            W_hat,
                            dataset.train_counts,
                            dataset.train_N,
                            epsilon=float(recovery_config.get("poisson_epsilon", 1e-12)),
                            max_iter=int(recovery_config.get("poisson_max_iter", 2000)),
                            tolerance=float(recovery_config.get("poisson_tolerance", 1e-8)),
                        )
                    else:
                        raise ValueError(f"unknown A recovery {A_method!r}")
                    if W_hat.tobytes() != W_bytes:
                        raise RuntimeError("A recovery mutated the shared W estimate")
                    A_runtime = time.perf_counter() - started
                    A_artifact = artifact_root / f"{W_fit_id}_{A_method}_A.npz"
                    np.savez_compressed(
                        A_artifact,
                        A_hat=A_result.A_hat,
                        objective_history=np.asarray(A_result.objective_history),
                        W_fit_id=np.asarray(W_fit_id),
                    )
                    W_metrics, A_metrics, fit_metrics, _, _ = _aligned_metrics(
                        W_hat, A_result.A_hat, dataset, block.retained_indices
                    )
                    warnings = list(vertex.warnings) + list(recovery_warnings) + list(A_result.warnings)
                    status = "ok" if A_result.converged else "unstable"
                    record = {
                        **base,
                        **common,
                        "estimator_family": family,
                        "A_recovery_method": A_method,
                        "W_recovery_method": W_recovery_method,
                        "b_hat": json.dumps(_jsonable(b_hat)),
                        "prevalence_residual": prevalence_residual,
                        "vertex_error_to_population_frobenius": vertex_fro,
                        "vertex_error_to_population_rmse": vertex_rmse,
                        "W_artifact": str(W_artifact.relative_to(REPO_ROOT)),
                        "A_artifact": str(A_artifact.relative_to(REPO_ROOT)),
                        "A_converged": A_result.converged,
                        "A_iterations": A_result.iterations,
                        "A_objective_start": (
                            A_result.objective_history[0] if A_result.objective_history else None
                        ),
                        "A_objective_end": (
                            A_result.objective_history[-1] if A_result.objective_history else None
                        ),
                        "status": status,
                        "failure_reason": None,
                        "warnings": json.dumps(warnings),
                        "runtime_A_recovery": A_runtime,
                        "runtime_total": block.runtime_seconds + vertex.runtime_seconds + A_runtime,
                        **W_metrics,
                        **A_metrics,
                        **fit_metrics,
                        "all_W_metrics": json.dumps(_jsonable(W_metrics)),
                        "all_A_metrics": json.dumps(_jsonable(A_metrics)),
                        "all_fit_metrics": json.dumps(_jsonable(fit_metrics)),
                    }
                    records.append(record)
                except Exception as error:
                    records.append(
                        _failed_record(
                            base,
                            family=family,
                            reason=f"{type(error).__name__}: {error}",
                            A_recovery_method=A_method,
                            W_recovery_method=W_recovery_method,
                            runtime_A_recovery=time.perf_counter() - started,
                            **common,
                        )
                    )
    return records


def _baseline_record(
    dataset: Dataset,
    method: str,
    W_hat: np.ndarray,
    A_hat: np.ndarray,
    runtime: float,
    metadata: dict[str, Any],
    warnings: list[str],
    status: str = "ok",
) -> dict[str, Any]:
    base = _base_record(dataset, None, None)
    if "retained_feature_count" in metadata:
        retained_count = int(metadata["retained_feature_count"])
        base["retained_feature_count"] = retained_count
        base["retained_feature_fraction"] = retained_count / dataset.X.shape[1]
        base["zero_frequency_training_columns_removed"] = int(
            metadata.get(
                "zero_frequency_feature_count", dataset.X.shape[1] - retained_count
            )
        )
    W_metrics, A_metrics, fit_metrics, _, _ = _aligned_metrics(W_hat, A_hat, dataset)
    return {
        **base,
        "estimator_family": method,
        "simplex_side": "not_applicable",
        "profile_source": None,
        "vertex_hunter": metadata.get("vertex_hunter", "native"),
        "A_recovery_method": "native",
        "W_recovery_method": metadata.get("W_refit", "native_joint"),
        "W_fit_id": hashlib.sha256(f"{dataset.name}|{method}".encode()).hexdigest()[:20],
        "status": status,
        "failure_reason": None,
        "warnings": json.dumps(warnings),
        "baseline_metadata": json.dumps(_jsonable(metadata)),
        "runtime_vertex": 0.0,
        "runtime_A_recovery": 0.0,
        "runtime_total": runtime,
        **W_metrics,
        **A_metrics,
        **fit_metrics,
        "all_W_metrics": json.dumps(_jsonable(W_metrics)),
        "all_A_metrics": json.dumps(_jsonable(A_metrics)),
        "all_fit_metrics": json.dumps(_jsonable(fit_metrics)),
    }


def run_baselines(
    dataset: Dataset,
    P0_block: SpectralBlock,
    config: dict[str, Any],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    methods = config["baselines"]
    for method in methods:
        started = time.perf_counter()
        try:
            if method == "topicscore_raw":
                fit = fit_topicscore_raw(dataset.X, dataset.W_true.shape[1])
            elif method == "topicscore_graph_denoised":
                fit = fit_topicscore_graph_denoised(
                    dataset.X,
                    P0_block.U_hat,
                    P0_block.V_hat,
                    P0_block.singular_values,
                    retained_indices=P0_block.retained_indices,
                    weights=P0_block.prep.weighting.weights,
                )
            elif method == "lda":
                fit = fit_lda(dataset.train_counts, dataset.W_true.shape[1], random_state=0)
            elif method == "spatial_lda":
                if dataset.coordinates is None:
                    base = _base_record(dataset, None, None)
                    records.append(
                        _failed_record(
                            base,
                            family=method,
                            reason="not_applicable: exact Tran design has no coordinates or graph",
                        )
                    )
                    continue
                fit = fit_spatial_lda(
                    dataset.train_counts,
                    dataset.W_true.shape[1],
                    dataset.coordinates,
                    parameters=config.get("spatial_lda_parameters"),
                )
            else:
                raise ValueError(f"unknown baseline {method!r}")
            artifact_root.mkdir(parents=True, exist_ok=True)
            artifact = artifact_root / f"{method}_native.npz"
            np.savez_compressed(artifact, W_hat=fit.W_hat, A_hat=fit.A_hat)
            record = _baseline_record(
                    dataset,
                    method,
                    fit.W_hat,
                    fit.A_hat,
                    fit.runtime_seconds
                    + (P0_block.runtime_seconds if method == "topicscore_graph_denoised" else 0.0),
                    fit.metadata,
                    fit.warnings,
                    "ok" if fit.converged else "unstable",
                )
            record["W_artifact"] = str(artifact.relative_to(REPO_ROOT))
            record["A_artifact"] = str(artifact.relative_to(REPO_ROOT))
            records.append(record)
        except Exception as error:
            records.append(
                _failed_record(
                    _base_record(dataset, None, None),
                    family=method,
                    reason=f"{type(error).__name__}: {error}",
                    runtime_total=time.perf_counter() - started,
                )
            )
    return records


def _write_results(records: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(records)
    preferred = [
        "dataset",
        "design_variant",
        "experiment_family",
        "seed",
        "replicate",
        "n",
        "p",
        "N",
        "K",
        "word_decay_parameter",
        "graph_setting",
        "graph_svd_version",
        "estimator_family",
        "simplex_side",
        "profile_source",
        "preprocessing_variant",
        "threshold_method",
        "alpha",
        "threshold_value",
        "retained_feature_count",
        "retained_feature_fraction",
        "retained_anchor_count_by_topic",
        "weight_method",
        "tau",
        "weight_cap",
        "weight_quantiles",
        "rho",
        "rho_path",
        "vertex_hunter",
        "L",
        "L_mode",
        "pp_spa_parameters",
        "selected_vertex_indices",
        "selected_vocabulary_indices",
        "selected_word_purity",
        "vertex_condition_number",
        "smallest_vertex_singular_value",
        "A_recovery_method",
        "W_recovery_method",
        "W_fit_id",
        "status",
        "failure_reason",
        "runtime_spectral",
        "runtime_vertex",
        "runtime_A_recovery",
        "runtime_total",
        "all_W_metrics",
        "all_A_metrics",
        "all_fit_metrics",
    ]
    columns = [column for column in preferred if column in frame] + [
        column for column in frame if column not in preferred
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame[columns].to_csv(temporary, index=False)
    temporary.replace(path)


def run_experiment(
    config_path: Path,
    output_override: Path | None = None,
    seed_override: int | None = None,
    baselines_only: bool = False,
) -> Path:
    config_text = config_path.read_text()
    config = json.loads(config_text)
    if seed_override is not None:
        available_seeds = [int(seed) for seed in config["seeds"]]
        if seed_override not in available_seeds:
            raise ValueError(
                f"seed {seed_override} is not present in {config_path}; "
                f"available range is {min(available_seeds)}--{max(available_seeds)}"
            )
        config["seeds"] = [int(seed_override)]
        config["seed_shard"] = int(seed_override)
    effective_config_text = (
        config_text if seed_override is None else json.dumps(config, sort_keys=True)
    )
    config_hash = hashlib.sha256(effective_config_text.encode()).hexdigest()[:10]
    if output_override is None:
        output_root = REPO_ROOT / config["output_directory"] / f"{config['stage']}_{config_hash}"
    else:
        output_root = output_override if output_override.is_absolute() else REPO_ROOT / output_override
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    data_root = output_root / "data"
    spectral_root = output_root / "spectral"
    artifact_root = output_root / "estimates"
    data_root.mkdir(exist_ok=True)
    spectral_root.mkdir(exist_ok=True)
    artifact_root.mkdir(exist_ok=True)
    results_path = output_root / "tidy_results.csv"
    records = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    completed = {_method_key(record) for record in records}

    specs = expand_design(config)
    for dataset_index, spec in enumerate(specs, start=1):
        sweep_message = ""
        if spec.get("sweep_parameter") is not None:
            sweep_message = f" {spec['sweep_parameter']}={spec['sweep_value']}"
        print(
            f"[{dataset_index}/{len(specs)}] {spec['design_variant']}"
            f"{sweep_message} decay={spec['a_zipf']} seed={spec['seed']}"
        )
        dataset = generate_dataset(spec, data_root, float(config["test_fraction"]))
        blocks: dict[str, SpectralBlock] = {}
        preprocessing_grid = config["preprocessing_variants"]
        if baselines_only:
            preprocessing_grid = [
                item for item in preprocessing_grid if item["name"] == "P0_raw"
            ]
        for preprocessing in preprocessing_grid:
            block_target = spectral_root / dataset.name / preprocessing["name"]
            try:
                block = factorize(dataset, preprocessing, config["graphSVD"], block_target)
                blocks[preprocessing["name"]] = block
                if baselines_only:
                    new_records = []
                else:
                    expected_keys = _expected_gplsi_keys(
                        dataset,
                        preprocessing,
                        config["vertex_hunters"],
                        config["recovery"],
                    )
                    if expected_keys.issubset(completed):
                        new_records = []
                    else:
                        new_records = run_gplsi_block(
                            dataset,
                            preprocessing,
                            block,
                            config["vertex_hunters"],
                            config["recovery"],
                            artifact_root / dataset.name / preprocessing["name"],
                        )
            except Exception as error:
                if baselines_only:
                    raise
                base = _base_record(dataset, None, None)
                base["preprocessing_variant"] = preprocessing["name"]
                base["threshold_method"] = preprocessing["threshold_method"]
                base["weight_method"] = preprocessing["weight_method"]
                new_records = []
                for side in ("document", "anchor_word"):
                    family = "gplsi_document" if side == "document" else "gplsi_anchor_word_profile"
                    for vertex_spec in config["vertex_hunters"]:
                        for A_method in config["recovery"]["A_recovery_methods"]:
                            new_records.append(
                                _failed_record(
                                    base,
                                    family=family,
                                    reason=f"spectral/preprocessing failure: {type(error).__name__}: {error}",
                                    simplex_side=side,
                                    vertex_hunter=vertex_spec["method"],
                                    A_recovery_method=A_method,
                                )
                            )
            for record in new_records:
                key = _method_key(record)
                if key not in completed:
                    records.append(record)
                    completed.add(key)
            _write_results(records, results_path)

        if "P0_raw" in blocks:
            existing_families = {
                str(record.get("estimator_family"))
                for record in records
                if record.get("dataset") == dataset.name
            }
            missing_baselines = [
                method for method in config["baselines"] if method not in existing_families
            ]
            if missing_baselines:
                baseline_config = dict(config)
                baseline_config["baselines"] = missing_baselines
                for record in run_baselines(
                    dataset,
                    blocks["P0_raw"],
                    baseline_config,
                    artifact_root / dataset.name / "baselines",
                ):
                    key = _method_key(record)
                    if key not in completed:
                        records.append(record)
                        completed.add(key)
                _write_results(records, results_path)

    frame = pd.DataFrame(records)
    summary_columns = [
        "design_variant",
        "n",
        "p",
        "N",
        "word_decay_parameter",
        "estimator_family",
        "simplex_side",
        "preprocessing_variant",
        "vertex_hunter",
        "A_recovery_method",
    ]
    metrics = [
        "W_rmse",
        "A_mean_topic_TV",
        "A_tail_word_rmse",
        "held_out_poisson_deviance",
        "runtime_total",
        "vertex_condition_number",
        "retained_feature_fraction",
    ]
    successful = frame[frame.status.isin(["ok", "unstable"])].copy()
    if not successful.empty:
        def q25(values):
            return values.quantile(0.25)

        def q75(values):
            return values.quantile(0.75)

        summary = successful.groupby(summary_columns, dropna=False)[metrics].agg(
            ["count", "mean", "median", "std", q25, q75]
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.reset_index().to_csv(output_root / "summary.csv", index=False)
    failures = frame[~frame.status.isin(["ok", "unstable"])]
    failures.to_csv(output_root / "failures.csv", index=False)
    validation = {
        "stage": config["stage"],
        "config_hash": config_hash,
        "dataset_count": len(specs),
        "record_count": int(len(frame)),
        "status_counts": frame.status.value_counts(dropna=False).to_dict(),
        "unique_W_fit_ids": int(frame.W_fit_id.dropna().nunique()),
        "paired_A_recovery_W_ids": int(
            frame[frame.A_recovery_method.isin(["current", "poisson_full"])]
            .groupby("W_fit_id")
            .A_recovery_method.nunique()
            .eq(2)
            .sum()
        ),
        "manuscript_files_modified": False,
    }
    (output_root / "validation.json").write_text(
        json.dumps(_jsonable(validation), indent=2) + "\n"
    )
    print(output_root)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        help="run only one seed listed in the configuration (for job-array shards)",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="reuse saved P0 factors and fill missing baseline rows only",
    )
    args = parser.parse_args()
    run_experiment(args.config, args.output_dir, args.seed, args.baselines_only)


if __name__ == "__main__":
    main()
