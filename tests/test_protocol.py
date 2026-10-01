"""The paper protocol (docs/protocol.md): its configs and the metrics it adds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gplsi.pipeline import expand_tasks, load_config
from gplsi.pipeline.metrics import (
    spatial_domain_metrics,
    topic_overlap_metrics,
    topic_prevalence_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SEVEN = ["151507", "151508", "151509", "151510", "151673", "151674", "151675", "151676"]
BR5595 = ["151669", "151670", "151671", "151672"]


def test_production_configs_follow_the_paper_protocol() -> None:
    expected = {
        "crc": ([2, 3, 4, 5], 60),
        "spleen": ([3, 4, 5, 7, 10], 75),
        "cook": ([7], 15),
        "dlpfc": ([7], 180),
    }
    for dataset, (K, task_count) in expected.items():
        config = load_config(ROOT / "configs" / dataset / "production.json")
        assert config["grid"]["K"] == K
        assert len(config["grid"]["seed"]) == 5
        assert config["heldout_fraction"] == 0.2 and "subset" not in config
        spectral = config["spectral"]
        assert spectral["lambda_selection_mode"] == "cv_once"
        assert spectral["initialization"] == "weighted_debiased_mean_N_approx"
        assert spectral["nfolds"] == 5 and spectral["grid_len"] == 29
        assert config["gplsi"] == [
            {"geometry": "document_U", "preprocessings": ["P0_raw"], "vertex_hunters": ["svs_star"]}
        ]
        assert config["vertex_parameters"]["svs_star"]["L_mode"] == "svs_star_stability"
        assert config["A_recoveries"] == ["A_current", "A_full_L2", "A_full_Pois"]
        assert config["recovery"]["poisson_start"] == "pooled"
        assert config["baselines"] == ["plsi", "topicscore_raw", "topicscore_graph_denoised", "lda", "spatial_lda"]
        tasks = expand_tasks(config)
        assert len(tasks) == task_count
        # Within each task grid (DLPFC: each section group), every method runs in exactly one part.
        by_grid: dict[str, set[str]] = {}
        for part in config["parts"].values():
            methods = by_grid.setdefault(repr(part.get("grid")), set())
            names = part.get("baselines", [])
            assert methods.isdisjoint(names)
            methods.update(names)
        assert all(methods == set(config["baselines"]) for methods in by_grid.values())


def test_dlpfc_runs_K5_on_br5595_and_K7_elsewhere() -> None:
    for name in ("production.json", "production_tran.json"):
        tasks = expand_tasks(load_config(ROOT / "configs" / "dlpfc" / name))
        K_by_section = {}
        for task in tasks:
            K_by_section.setdefault(task["section"], set()).add(task["K"])
        assert {s: K for s, K in K_by_section.items()} == {
            **{s: {7} for s in SEVEN}, **{s: {5} for s in BR5595}
        }


def test_dlpfc_tran_config_matches_the_panel_config_except_the_vocabulary() -> None:
    from gplsi.pipeline.config import task_id

    panel = load_config(ROOT / "configs" / "dlpfc" / "production.json")
    tran = load_config(ROOT / "configs" / "dlpfc" / "production_tran.json")
    assert tran["dataset"]["vocabulary"] == "tran" and tran["dataset"]["tran_alpha"] == 0.1
    for key in ("gplsi", "baselines", "A_recoveries", "vertex_parameters", "spectral", "heldout_fraction"):
        assert tran[key] == panel[key]
    panel_tasks, tran_tasks = expand_tasks(panel), expand_tasks(tran)
    strip = lambda task: {k: v for k, v in task.items() if k != "panel_size"}  # noqa: E731
    assert [strip(t) for t in panel_tasks] == tran_tasks  # same seeds, sections, parts, order
    ids = {task_id(tran, t) for t in tran_tasks}
    assert all("__tran0.1__" in i for i in ids) and ids.isdisjoint(task_id(panel, t) for t in panel_tasks)


def test_sparse_tran_vocabulary_matches_the_paper_rule() -> None:
    from scipy.sparse import csr_matrix

    from gplsi.pipeline.panels import tran_vocabulary
    from gplsi.preprocessing import select_feature_columns

    rng = np.random.default_rng(3)
    counts = rng.poisson(rng.gamma(0.3, 2.0, size=60) * 0.5, size=(200, 60))
    counts[:, 0] += 1  # no empty rows
    lengths = counts.sum(axis=1)
    for alpha in (0.005, 0.1, 1.0):
        dense = select_feature_columns(counts / lengths[:, None], lengths, method="tran_paper_exact", alpha=alpha)
        sparse = tran_vocabulary(csr_matrix(counts), alpha=alpha)
        assert np.array_equal(sparse["indices"], dense.retained_indices)
        assert np.isclose(sparse["threshold"], dense.threshold_value)


def test_smoke_configs_inherit_the_protocol_in_one_job() -> None:
    for dataset in ("crc", "spleen", "cook", "dlpfc"):
        smoke = load_config(ROOT / "configs" / dataset / "smoke.json")
        production = load_config(ROOT / "configs" / dataset / "production.json")
        assert len(expand_tasks(smoke)) == 1 and "part" not in expand_tasks(smoke)[0]
        for key in ("gplsi", "baselines", "A_recoveries", "vertex_parameters", "dataset"):
            assert smoke[key] == production[key]


def test_pas_and_chaos_separate_coherent_from_scattered_labels() -> None:
    rng = np.random.default_rng(0)
    xx, yy = np.meshgrid(np.arange(30.0), np.arange(30.0))
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    blocks = (xy[:, 0] >= 15).astype(int) + 2 * (xy[:, 1] >= 15).astype(int)
    scattered = rng.integers(0, 4, size=len(xy))
    coherent = spatial_domain_metrics(blocks, xy)
    noisy = spatial_domain_metrics(scattered, xy)
    # Only points at block edges can be abnormal; most scattered points are.
    assert coherent["spatial_PAS"] < 0.05 < 0.5 < noisy["spatial_PAS"]
    # Every same-topic nearest neighbour is one grid step away within blocks.
    assert np.isclose(coherent["spatial_CHAOS"], 1.0)
    assert noisy["spatial_CHAOS"] > coherent["spatial_CHAOS"]


def test_spatial_metrics_never_pair_points_across_units() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 1, 1, 0])
    units = np.array(["a", "a", "b", "b"])
    # Pooled, each point has a same-topic twin at distance 0; per unit it has none.
    assert spatial_domain_metrics(labels, xy, units)["spatial_CHAOS_singleton_points"] == 4


def test_topic_overlap_and_diversity() -> None:
    distinct = np.eye(3, 6) * 0.9 + 0.1 / 6
    duplicated = np.vstack([distinct[0], distinct[0], distinct[1]])
    clean = topic_overlap_metrics(distinct, top=(1,))
    copy = topic_overlap_metrics(duplicated, top=(1,))
    assert np.isclose(copy["topic_cosine_max"], 1.0) and clean["topic_cosine_max"] < 0.2
    assert clean["topic_diversity_top1"] == 1.0
    assert np.isclose(copy["topic_diversity_top1"], 2 / 3)
    # t is capped at p: with t = p every topic lists every word.
    assert np.isclose(topic_overlap_metrics(distinct, top=(25,))["topic_diversity_top25"], 6 / 18)


def test_topic_prevalence() -> None:
    W = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.6, 0.4, 0.0], [0.5, 0.5, 0.0]])
    output = topic_prevalence_metrics(W)
    assert output["topic_prevalence_argmax"] == [0.75, 0.25, 0.0]  # tie goes to topic 0
    assert np.allclose(output["topic_prevalence_mean_W"], W.mean(axis=0))
    assert output["empty_topic_count"] == 1
    assert np.isclose(output["topic_effective_number"], np.exp(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25))))


def test_spatial_lda_accepts_a_graph_without_coordinates() -> None:
    from gplsi.baselines import fit_spatial_lda

    rng = np.random.default_rng(1)
    counts = rng.poisson(3.0, size=(30, 6)).astype(float) + 1
    edges = np.array([[i, i + 1] for i in range(29)])
    result = fit_spatial_lda(counts, 2, None, edges=edges, parameters={"max_lda_iter": 2, "n_iters": 1})
    assert result.W_hat.shape == (30, 2) and result.A_hat.shape == (2, 6)
    assert np.allclose(result.W_hat.sum(axis=1), 1) and np.allclose(result.A_hat.sum(axis=1), 1)
    assert result.metadata["spatial_regularization_scope"].startswith("given graph")


def test_smoothed_heldout_deviance_is_finite_with_zero_probabilities() -> None:
    from gplsi.pipeline.metrics import heldout_count_metrics

    W = np.array([[1.0, 0.0], [0.3, 0.7]])
    A = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
    test = np.array([[2, 1, 1], [0, 3, 2]])  # row 0 has a count on a zero-probability word
    output = heldout_count_metrics(W, A, test)
    assert output["heldout_poisson_deviance"] == float("inf")
    smoothed = [output[f"heldout_poisson_deviance_per_molecule_smoothed_{eps:.0e}"] for eps in (1e-4, 1e-3, 1e-2)]
    assert np.all(np.isfinite(smoothed)) and smoothed[0] > smoothed[1] > smoothed[2]
    # Without zeros, small smoothing barely moves the exact deviance.
    dense = heldout_count_metrics(W, np.array([[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]]), test)
    assert np.isclose(dense["heldout_poisson_deviance_per_molecule_smoothed_1e-04"],
                      dense["heldout_poisson_deviance_per_molecule"], rtol=1e-3)


def test_spatial_unit_configs_use_the_protocol_with_K12_and_the_wide_grid() -> None:
    base = load_config(ROOT / "configs" / "dlpfc" / "production.json")
    for dataset, units in (("merfish", 15), ("xenium", 25)):
        config = load_config(ROOT / "configs" / dataset / "production.json")
        assert config["grid"]["K"] == [12] and len(config["grid"]["seed"]) == 5
        assert len(config["grid"]["unit"]) == units
        assert config["spectral"]["grid_len"] == 50  # top 1e-4 * 1.2**49 ~ 0.76
        assert config["dataset"]["vocabulary"] == "all"
        for key in ("gplsi", "baselines", "A_recoveries", "vertex_parameters", "heldout_fraction"):
            assert config[key] == base[key]
        assert len(expand_tasks(config)) == units * 5 * 3


def test_consensus_alignment_recovers_permutations_of_one_profile_set() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    from shared import consensus_alignment

    rng = np.random.default_rng(5)
    truth = rng.dirichlet(np.full(30, 0.2), size=6)
    permutations = [rng.permutation(6) for _ in range(5)]
    noisy = [truth[p] + rng.uniform(0, 1e-3, size=truth.shape) for p in permutations]
    orders, consensus = consensus_alignment(noisy)
    reference = permutations[0]
    for permutation, order in zip(permutations, orders):
        # aligned topic k of every unit is the same true topic as unit 0's topic k
        assert np.array_equal(permutation[order], reference)
    assert np.allclose(consensus, truth[reference], atol=1e-2)


def test_cellular_neighborhoods_split_spatially_separated_cell_mixes() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_xenium", ROOT / "scripts" / "data" / "prepare_xenium.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    rng = np.random.default_rng(7)
    # Two cores; in each, a left half of type "a" cells and a right half mixing "b" and "c".
    xy, strata, types = [], [], []
    for core in ("core1", "core2"):
        points = rng.uniform(0, 100, size=(400, 2))
        xy.append(points)
        strata += [core] * 400
        types += ["a" if x < 50 else rng.choice(["b", "c"]) for x, _ in points]
    labels, composition = prepare.cellular_neighborhoods(np.vstack(xy), np.array(strata), np.array(types))
    assert labels.shape == (800,) and set(labels) <= {f"N{i}" for i in range(1, prepare.NEIGHBORHOOD_COUNT + 1)}
    sizes = pd.Series(labels).value_counts()
    assert list(sizes.index) == sorted(sizes.index, key=lambda n: -sizes[n])  # N1 is the largest
    assert np.allclose(composition.sum(axis=1), 1.0)
    # Deep inside the left halves every window is pure "a": all those cells share one neighbourhood.
    left = np.vstack(xy)[:, 0] < 40
    assert len(set(labels[left])) <= 2
    assert set(labels[left]).isdisjoint(set(labels[np.vstack(xy)[:, 0] > 60]))


def test_evaluation_labels_are_part_of_the_cache_key() -> None:
    from gplsi.pipeline.datasets import TaskData

    class Bundle:
        def hashes(self):
            return {"counts": "same"}

    first = TaskData(bundle=Bundle(), heldout=None, feature_names=np.array(["g"]),
                     labels=pd.DataFrame({"label": ["x", "y"]}))
    second = TaskData(bundle=Bundle(), heldout=None, feature_names=np.array(["g"]),
                      labels=pd.DataFrame({"label": ["x", "z"]}))
    assert first.hashes()["labels_sha256"] != second.hashes()["labels_sha256"]
    assert "labels_sha256" not in TaskData(bundle=Bundle(), heldout=None, feature_names=np.array(["g"])).hashes()
