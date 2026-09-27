"""Synthetic, disconnected-cohort integration smoke; never a real-data pilot."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
import resource
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy import sparse

from .artifacts import atomic_json, atomic_npz
from .config import fingerprint
from .graph import build_graph
from .likelihood import recover_A_poisson, fold_in_fixed_A
from .metrics import score_counts, spatial_metrics, pas10_neighbors
from .spectral import fit_shared_spectral, SpectralConfig


def synthetic_fixture(seed=26091002):
    """Three spatial blocks, shared separable profiles, untouched test molecules."""
    rng = np.random.default_rng(seed)
    p, k, per_section = 30, 3, 50
    a = np.full((k, p), .002)
    for topic in range(k):
        a[topic, topic * 10:(topic + 1) * 10] = .096
    a /= a.sum(axis=1, keepdims=True)
    coordinates = np.tile(np.array([(x, y) for y in range(5) for x in range(10)], float), (3, 1))
    strata = np.repeat(["bio0/section0", "bio1/section0", "bio2/section0"], per_section)
    biological_ids = np.repeat(["bio0", "bio1", "bio2"], per_section)
    hard = (coordinates[:, 0] // 4).astype(int)
    w = np.eye(k)[hard]
    mixed = np.arange(len(w)) % 11 == 0
    w[mixed] = rng.dirichlet([2, 2, 2], mixed.sum())
    raw = np.vstack([rng.multinomial(2000, row) for row in w @ a])
    train = rng.binomial(raw, .8)
    score = raw - train
    test_w = rng.dirichlet([.6, .6, .6], 24)
    test_raw = np.vstack([rng.multinomial(1200, row) for row in test_w @ a])
    adapt = rng.binomial(test_raw, .8)
    return {"train": sparse.csr_matrix(train), "score": sparse.csr_matrix(score),
            "adapt": sparse.csr_matrix(adapt), "test_score": sparse.csr_matrix(test_raw - adapt),
            "coordinates": coordinates, "strata": strata, "biological_ids": biological_ids,
            "true_A": a, "true_W": w, "true_W_test": test_w, "annotations": hard.astype(str),
            "gene_ids": np.array([f"synthetic_gene_{j}" for j in range(p)])}


@contextmanager
def forbid_full_sparse_densification(forbidden_shapes):
    """Permit n-by-K working factors; reject full counts or graph densification."""
    shapes = set(tuple(shape) for shape in forbidden_shapes)
    with ExitStack() as stack:
        for cls in (sparse.csr_matrix, sparse.csc_matrix):
            original = cls.toarray
            def guarded(self, *args, _original=original, **kwargs):
                if self.shape in shapes:
                    raise AssertionError(f"full sparse matrix densified: {self.shape}")
                return _original(self, *args, **kwargs)
            stack.enter_context(patch.object(cls, "toarray", guarded))
        yield


def smoke_spectral_config():
    return SpectralConfig(initial_grid=(0.0, 1e-4), max_candidates=2, max_iterations=80,
                          reconstruction_tolerance=1e-4, stable_iterations=2,
                          ssnal_max_iterations=300, ssnal_admm_iterations=50)


def fit_synthetic_training(payload, *, preprocessing="P0_raw", seed=26091004):
    """Only training counts and coordinates enter; scores/labels are inaccessible."""
    graph, _ = build_graph(payload["coordinates"], payload["strata"])
    fit = fit_shared_spectral(payload["train"], graph, 3, preprocessing,
                              graph_cv_seed=26091003, estimator_seed=seed,
                              requested_panel=24, config=smoke_spectral_config())
    return fit, graph


def check_leakage_sentinel(payload, *, seed=26091004):
    first, graph = fit_synthetic_training(payload, seed=seed)
    changed = dict(payload)
    changed["score"] = payload["score"].copy()
    changed["score"].data += 12345
    changed["test_score"] = payload["test_score"].copy()
    changed["test_score"].data += 54321
    changed["annotations"] = np.array(["adversarial_annotation"] * len(payload["annotations"]))
    second, _ = fit_synthetic_training(changed, seed=seed)
    if first.selected is None or second.selected is None:
        raise AssertionError("smoke all-five-fold tuning has no converged candidate")
    assert first.metadata["cv"]["selected_lambda"] == second.metadata["cv"]["selected_lambda"]
    assert np.array_equal(first.features.panel_indices, second.features.panel_indices)
    assert np.array_equal(first.features.retained_indices, second.features.retained_indices)
    assert np.array_equal(first.features.weights, second.features.weights)
    assert np.allclose(first.selected.U, second.selected.U, atol=2e-10, rtol=2e-10)
    from .geometry import fit_geometry
    geometry = [fit_geometry(fitted.selected, fitted.features, graph, hunter="spa_current", estimator_seed=seed)
                for fitted in (first, second)]
    profiles = [recover_A_poisson(g.W, payload["train"], max_iter=500) for g in geometry]
    assert np.allclose(geometry[0].W, geometry[1].W, atol=2e-10, rtol=2e-10)
    assert np.allclose(profiles[0].A_hat, profiles[1].A_hat, atol=2e-10, rtol=2e-10)
    folded = [fold_in_fixed_A(profile.A_hat, payload["adapt"], max_iter=500) for profile in profiles]
    assert np.allclose(folded[0].W, folded[1].W, atol=2e-10, rtol=2e-10)
    return first, graph, {"scoring_count_mutation": True, "annotation_mutation": True,
                         "A_unchanged": True, "W_train_unchanged": True, "W_test_unchanged": True,
                         "features_and_weights_unchanged": True, "selected_lambda_unchanged": True,
                         "reported_profile_recovery_is_Poisson": True,
                         "all_five_folds": all(len(row["fold_scores"]) == 5 for row in first.metadata["cv"]["aggregate"])}


def run_smoke(root, config):
    from .geometry import fit_geometry, GeometryResourceBlocked
    from .competitors import fit_topicscore, fit_lda, fit_kl_nmf, fit_graph_kl_nmf, fit_spatial_lda
    root = Path(root)
    seed = config["seeds"]["estimator_seed"]
    started = time.perf_counter()
    from .cli import source_identity
    source = source_identity()
    identity = fingerprint({"kind": "synthetic_joint_smoke_v2", "source": source["source_tree_hash"],
                            "config": config, "spectral": asdict(smoke_spectral_config())})
    output = root / "results/joint_v2/smoke" / identity
    data_directory = root / "data/interim/joint_v2/smoke" / identity
    output.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    data = synthetic_fixture(config["seeds"]["molecule_seed"])
    for name in ("train", "score", "adapt", "test_score"):
        sparse.save_npz(data_directory / f"{name}.npz", data[name])
    atomic_npz(data_directory / "ids_and_truth.npz", {key: data[key] for key in
        ["coordinates", "strata", "biological_ids", "true_A", "true_W", "true_W_test", "annotations", "gene_ids"]})
    report = {"schema": "joint_v2_synthetic_smoke", "source": source, "config": config,
              "spectral_configuration": asdict(smoke_spectral_config()), "identity": identity,
              "synthetic_not_real_data": True, "training_shape": data["train"].shape,
              "training_biological_units": 3, "disconnected_graph_strata": 3,
              "intentional_budgets": {"lambda_grid_max_candidates": 2, "K": 3,
                                     "palm_iterations": 30, "poisson_iterations": 500,
                                     "foldin_iterations": 500, "lda_iterations": 3,
                                     "nmf_iterations": 20, "spatial_lda_outer_iterations": 1},
              "records": [], "substantive_failures": []}
    shapes = [data["train"].shape, data["train"].T.shape, (len(data["strata"]),) * 2]
    def evaluate(name, w, a, fit_metadata):
        w_original = w.copy()
        poisson = recover_A_poisson(w, data["train"], initial_A=None, max_iter=500)
        assert np.array_equal(w, w_original), "A pairing mutated shared W"
        recovered = {"A_full_Pois": poisson.A_hat}
        comparisons = {}
        for recovery, profile in recovered.items():
            folded = fold_in_fixed_A(profile, data["adapt"], max_iter=500)
            training = score_counts(w, profile, data["score"])
            transfer = score_counts(folded.W, profile, data["test_score"], inference_valid=folded.inference_valid)
            comparisons[recovery] = {"training": training["summary"], "transfer": transfer["summary"],
                                     "foldin_status_counts": {s: int(np.sum(folded.row_status == s)) for s in np.unique(folded.row_status)}}
        diagnostic = spatial_metrics(w, graph, data["coordinates"], data["strata"], pas_neighbors=neighbors)
        atomic_npz(output / f"{name}.npz", {"W": w, **recovered})
        report["records"].append({"method": name, "status": "completed", "fit": fit_metadata,
                                   "poisson_converged": poisson.converged, "poisson_status": poisson.status,
                                   "poisson_certificate": poisson.normalized_optimality_gap,
                                   "profile_recovery": "A_full_Pois", "native_A_reported": False,
                                   "A_current_used": False,
                                   "recovery_pairing_identical_W": True, "scores": comparisons,
                                   "spatial": diagnostic})
    with forbid_full_sparse_densification(shapes):
        try:
            p0, graph, leakage = check_leakage_sentinel(data, seed=seed)
            report["leakage_sentinel"] = leakage
        except Exception as exc:
            report["substantive_failures"].append({"stage": "leakage_sentinel", "error": str(exc), "traceback": traceback.format_exc()})
            graph, _ = build_graph(data["coordinates"], data["strata"])
            p0 = None
        neighbors = pas10_neighbors(data["coordinates"], data["strata"])
        for preprocessing in config["preprocessings"]:
            try:
                fitted = p0 if preprocessing == "P0_raw" and p0 is not None else fit_synthetic_training(data, preprocessing=preprocessing, seed=seed)[0]
                if fitted.selected is None:
                    raise AssertionError("all-five-fold smoke tuning has no converged candidate")
                atomic_json(output / f"cv_{preprocessing}.json", fitted.metadata)
            except Exception as exc:
                report["substantive_failures"].append({"stage": f"spectral/{preprocessing}", "error": str(exc)})
                continue
            for hunter in config["hunters"]:
                name = f"{preprocessing}__{hunter}"
                parameters = {"max_iterations": 30, "initialize_weights_iterations": 20,
                              "final_weight_refit_iterations": 20} if hunter in ("palm", "palm_accelerated") else None
                try:
                    geometry = fit_geometry(fitted.selected, fitted.features, graph, hunter=hunter,
                                            estimator_seed=seed, vertex_parameters=parameters)
                    evaluate(name, geometry.W, None, geometry.metadata)
                except GeometryResourceBlocked as exc:
                    report["records"].append({"method": name, "status": "explicit_resource_budget_block", "metadata": exc.metadata})
                except Exception as exc:
                    report["substantive_failures"].append({"stage": name, "error": str(exc), "traceback": traceback.format_exc()})
            if preprocessing == "P0_raw":
                try:
                    geometry = fit_geometry(fitted.selected, fitted.features, graph, hunter="spa_current", family="anchor", estimator_seed=seed)
                    evaluate("anchor_P0_spa_current", geometry.W, None, geometry.metadata)
                except Exception as exc:
                    report["substantive_failures"].append({"stage": "anchor_P0_spa_current", "error": str(exc), "traceback": traceback.format_exc()})
        calls = {"topicscore_raw": lambda: fit_topicscore(data["train"], 3, seed, max_iter=1000),
                 "lda": lambda: fit_lda(data["train"], 3, seed, max_iter=3),
                 "kl_nmf": lambda: fit_kl_nmf(data["train"], 3, seed, max_iter=20),
                 "graph_kl_nmf_legacy_0p25": lambda: fit_graph_kl_nmf(data["train"], graph, 3, seed, max_iter=20),
                 "spatial_lda_legacy_0p25": lambda: fit_spatial_lda(data["train"], data["coordinates"], data["strata"], 3, seed,
                                                                   outer_iterations=1, lda_iterations=2, admm_iterations=2)}
        if p0 is not None and p0.selected is not None:
            calls["topicscore_graph_denoised"] = lambda: fit_topicscore(data["train"], 3, seed,
                spectral={"U": p0.selected.U, "V": p0.selected.V, "singular_values": p0.selected.singular_values,
                          "weights": p0.features.weights, "retained_indices": p0.features.retained_indices}, max_iter=1000)
        for name, call in calls.items():
            try:
                fit = call()
                evaluate(name, fit.W, fit.A, {**fit.metadata, "status": fit.status, "converged": fit.converged})
            except Exception as exc:
                report["substantive_failures"].append({"stage": name, "error": str(exc), "traceback": traceback.format_exc()})
    report.update(status="passed" if not report["substantive_failures"] else "failed",
                  runtime_seconds=time.perf_counter() - started,
                  peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  full_sparse_count_densification_guard=True, output_path=str(output), data_path=str(data_directory))
    atomic_json(output / "smoke_report.json", report)
    atomic_json(root / "reports/joint_v2/SMOKE_STATUS.json", {"status": report["status"], "identity": identity,
                "report": str(output / "smoke_report.json"), "runtime_seconds": report["runtime_seconds"],
                "substantive_failures": report["substantive_failures"]})
    if report["substantive_failures"]:
        raise RuntimeError(f"synthetic smoke failed; see {output / 'smoke_report.json'}")
    return {"status": "passed", "report": str(output / "smoke_report.json"), "records": len(report["records"]),
            "identity": identity, "synthetic_not_full_size_pilot": True}
