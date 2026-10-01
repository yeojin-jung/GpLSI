from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from gplsi.real_data import DATA_ROOT, RealDataBundle, load_real_data, validate_frozen_contract
from gplsi.real_experiment import fit_geometry, fit_spectral_block
from gplsi.utils import get_folds_disconnected_G


ROOT = Path(__file__).resolve().parents[1]

def _load_available_real_data(dataset: str, **kwargs) -> RealDataBundle:
    """Skip only unavailable dataset files, while retaining contract failures."""
    relative_files = {
        "crc": ["crc/charville_labels.csv"],
        "spleen": [f"spleen/dataset/{name}.pkl" for name in ("merged_D", "merged_coord", "merged_data")],
        "cook": [f"cook/dataset/{name}" for name in ("train.json", "ingredient_mapping.pkl", "processed_edge_df.pkl")],
    }
    paths = [DATA_ROOT / relative for relative in relative_files[dataset]]
    if dataset == "crc":
        paths.extend((DATA_ROOT / "crc/output/output_3hop").glob("*.csv"))
    for path in paths:
        if not path.is_file():
            pytest.skip(f"Optional {dataset} dataset file is unavailable: {path}; set GPLSI_DATA_ROOT")
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                pytest.skip(f"Optional {dataset} dataset file is a Git LFS pointer: {path}; fetch its content")
    try:
        return load_real_data(dataset, **kwargs)
    except FileNotFoundError as error:
        pytest.skip(f"Optional {dataset} dataset file is unavailable: {error.filename}; set GPLSI_DATA_ROOT")



@pytest.mark.parametrize(
    ("dataset", "shape", "edge_count"),
    [
        ("crc", (113561, 8), 185922),
        ("spleen", (35271, 24), 105774),
        ("cook", (13597, 1019), 64708),
    ],
)
def test_canonical_real_data_loaders_match_frozen_contract(
    dataset: str, shape: tuple[int, int], edge_count: int
) -> None:
    bundle = _load_available_real_data(dataset)
    assert bundle.counts.shape == shape
    assert len(bundle.edge_df) == edge_count
    assert bundle.metadata["canonical_contract_validated"] is True
    assert np.allclose(bundle.frequencies.sum(axis=1), 1.0)
    subset = bundle.connected_subset(48, seed=71)
    assert subset.counts.shape == (48, shape[1])
    assert subset.edge_df[["src", "tgt"]].to_numpy().max() < 48
    assert subset.metadata["parent_n"] == shape[0]


def test_joint_spleen_loader_stacks_all_samples_with_block_diagonal_graph() -> None:
    bundle = _load_available_real_data("spleen", group="joint")
    assert bundle.counts.shape == (100840, 24)
    assert len(bundle.edge_df) == 302393
    assert bundle.metadata["joint_model"] is True
    assert bundle.metadata["joint_components"] == [
        "BALBc-1",
        "BALBc-2",
        "BALBc-3",
    ]
    assert pd.Series(bundle.group_ids).value_counts().to_dict() == {
        "BALBc-1": 35271,
        "BALBc-2": 33492,
        "BALBc-3": 32077,
    }
    endpoints = bundle.edge_df[["src", "tgt"]].to_numpy(dtype=int)
    assert np.all(bundle.group_ids[endpoints[:, 0]] == bundle.group_ids[endpoints[:, 1]])
    assert len(set(map(str, bundle.observation_ids))) == bundle.n
    assert bundle.metadata["canonical_contract_validated"] is True
    hashes = bundle.hashes()
    expected_hashes = {
        "D_sha256": "4856c96de463fde08f30d0a3373209562e296a993378f4f9daef01786b880fba",
        "X_sha256": "37ce8e4e163879d5f3e7672726bf7d3258cc4662c0fc02d937440fc850540e7c",
        "N_i_sha256": "6a322dce523ff1a64c7fbc32615632d4eda0ecb730046da35c6b96f2fc2444d0",
        "feature_ids_sha256": "09da04a6d56b0a5919ae0df92f0af339656d90d5c88319525f21c20ffa7ace3d",
        "observation_ids_sha256": "c55f9b6e8d53bc4f162c02f8660cc95aa4de2f69e09b503b1a8ecef09a8a25d6",
        "group_ids_sha256": "d2e95f71678473736669bf9c88a36d5ba4d8a9b7bf0c0fa26c13f754478bcc65",
        "edges_sha256": "34f81bbea3562b265df6a4172176cda7706ebe6f95f51e1587172ab8ad273d6e",
        "coordinates_sha256": "29e7399af57e19ca0590893244a4d3e7d649c2a067e1c703a5787cbdeef30cb3",
    }
    assert {key: hashes[key] for key in expected_hashes} == expected_hashes
    assert (
        bundle.metadata["canonical_contract_hashes"][
            "edge_weights_rounded_10_sha256"
        ]
        == "161087db751263fa33a73c467b20d37a1373738087af09c6e7be2d4136fc91f2"
    )
    assert validate_frozen_contract(bundle) is bundle
    subset = bundle.graph_stratified_subset(60, seed=71)
    assert set(map(str, subset.group_ids)) == {"BALBc-1", "BALBc-2", "BALBc-3"}
    subset_endpoints = subset.edge_df[["src", "tgt"]].to_numpy(dtype=int)
    assert np.all(
        subset.group_ids[subset_endpoints[:, 0]]
        == subset.group_ids[subset_endpoints[:, 1]]
    )


def test_seeded_graph_folds_are_repeatable() -> None:
    edge = pd.DataFrame(
        {"src": np.arange(19), "tgt": np.arange(1, 20), "weight": np.ones(19)}
    )
    first = get_folds_disconnected_G(edge, nfolds=5, rng=np.random.default_rng(17))[1]
    second = get_folds_disconnected_G(edge, nfolds=5, rng=np.random.default_rng(17))[1]
    assert first == second
    assert sorted(value for fold in first.values() for value in fold) == list(range(20))


def test_audited_graph_cv_scores_every_nonempty_fold() -> None:
    rng = np.random.default_rng(303)
    n, p, K = 24, 7, 3
    topics = rng.dirichlet(np.ones(p), size=K)
    proportions = rng.dirichlet(np.ones(K), size=n)
    lengths = np.full(n, 50.0)
    counts = np.vstack(
        [rng.multinomial(int(lengths[i]), proportions[i] @ topics) for i in range(n)]
    ).astype(np.int64)
    edge = pd.DataFrame(
        {"src": np.arange(n - 1), "tgt": np.arange(1, n), "weight": np.ones(n - 1)}
    )
    bundle = RealDataBundle(
        dataset="toy",
        counts=counts,
        frequencies=counts / lengths[:, None],
        document_lengths=lengths,
        feature_ids=np.asarray([f"f{value}" for value in range(p)], dtype=object),
        observation_ids=np.arange(n).astype(object),
        group_ids=np.asarray(["toy"] * n, dtype=object),
        edge_df=edge,
        weights=csr_matrix(
            (np.ones(n - 1), (np.arange(n - 1), np.arange(1, n))),
            shape=(n, n),
        ),
    ).validate()
    first = fit_spectral_block(
        bundle,
        K,
        "P3_tran_then_ke",
        seed=41,
        grid_len=1,
        maxiter=1,
        nfolds=5,
        n_jobs=1,
    )
    second = fit_spectral_block(
        bundle,
        K,
        "P3_tran_then_ke",
        seed=41,
        grid_len=1,
        maxiter=1,
        nfolds=5,
        n_jobs=1,
    )
    assert first.cv_errors["scoring_folds"] == [0, 1, 2, 3, 4]
    assert first.selected_rho == second.selected_rho
    np.testing.assert_allclose(
        first.cv_errors["summed_cv_errors"], second.cv_errors["summed_cv_errors"]
    )
    np.testing.assert_allclose(
        first.U_hat @ first.U_hat.T,
        second.U_hat @ second.U_hat.T,
        atol=1e-10,
    )


def test_count_thinning_is_exact_and_deterministic() -> None:
    bundle = _load_available_real_data("cook").graph_stratified_subset(80, seed=26090501)
    train_a, test_a = bundle.binomial_thinning(0.2, seed=991)
    train_b, test_b = bundle.binomial_thinning(0.2, seed=991)
    np.testing.assert_array_equal(train_a.counts, train_b.counts)
    np.testing.assert_array_equal(test_a, test_b)
    np.testing.assert_array_equal(train_a.counts + test_a, bundle.counts)
    assert np.all(train_a.document_lengths > 0)
    assert train_a.metadata["heldout_D_sha256"]


def test_handoff_configs_keep_the_2026_09_15_protocol() -> None:
    from gplsi.pipeline import load_config

    expected_K = {"crc": [1, 2, 3, 4, 5, 6], "spleen": [3, 4, 5, 7, 10], "cook_v2": [4, 5, 6, 7, 8, 10]}
    for relative in ("handoff/crc_production.json", "handoff/spleen_production_joint.json", "handoff/cook_production.json"):
        config = load_config(ROOT / "configs" / relative)
        assert config["grid"]["K"] == expected_K[config["dataset"]["name"]]
        assert config["heldout_fraction"] == 0.2 and "subset" not in config
        spectral = config["spectral"]
        assert spectral["grid_len"] == 29 and spectral["maxiter"] == 50
        assert spectral["lambda_selection_mode"] == "cv_once"
        assert spectral["initialization"] == "weighted_debiased_mean_N_approx"
        assert spectral["cv_fold_mode"] == "all"
        assert config["recovery"]["poisson_start"] == "pooled"
        assert config["A_recoveries"] == ["A_current", "A_full_Pois"]
        (block,) = config["gplsi"]
        assert block["vertex_hunters"] == ["spa_current", "svs", "svs_star", "palm", "palm_accelerated"]
        assert config["vertex_parameters"]["svs"]["L_mode"] == "mixedscore_adaptive"
        assert "pp_spa" not in config["vertex_parameters"]
        for method in ("palm", "palm_accelerated"):
            parameters = config["vertex_parameters"][method]
            assert parameters["lambda_"] == 1.0 and parameters["max_iterations"] == 300
            assert parameters["hull_reduction"] == "none"


def test_unsupervised_fit_interfaces_do_not_accept_downstream_labels() -> None:
    forbidden = {
        "outcomes",
        "outcome_labels",
        "survival_time",
        "survival_event",
        "anatomical_labels",
        "cuisine_labels",
        "anchor_feature_labels",
    }
    assert forbidden.isdisjoint(inspect.signature(fit_spectral_block).parameters)
    assert forbidden.isdisjoint(inspect.signature(fit_geometry).parameters)
