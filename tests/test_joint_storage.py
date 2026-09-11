"""Minimal persistence keeps the sufficient statistics and exact factor aliases."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse

from gplsi_joint_v2.artifacts import atomic_json, atomic_npz
from gplsi_joint_v2.config import default_config
from gplsi_joint_v2.metrics import score_counts
from gplsi_joint_v2.runner import _section_sufficient_statistics, _write_factors, _read_embedding


def test_section_sums_preserve_primary_scores_and_infinite_outcomes():
    W = np.array([[1., 0.], [1., 0.], [0., 1.]])
    A = np.array([[1., 0.], [.5, .5]])
    D = sparse.csr_matrix([[2, 0], [1, 1], [1, 3]])
    obs = pd.DataFrame({"bio_id": ["b1", "b1", "b2"], "section_id": ["s1", "s1", "s2"]})
    result = score_counts(W, A, D)
    compact = _section_sufficient_statistics(result["per_row"], obs)
    assert sum(r["scored_molecules"] for r in compact) == result["summary"]["scored_molecules"]
    assert np.isinf(compact[0]["deviance"])
    assert sum(r["support_violation_entries"] for r in compact) == 1
    assert sum(r["scored_observations"] for r in compact) == 3


def test_W_artifact_keeps_only_one_W_and_references_shared_row_contract(tmp_path):
    prepared = SimpleNamespace(split_metadata={"row_contract": "/project/data/shared.parquet",
                                "training_row_order_sha256": "fixed_rows"})
    W = np.array([[.2, .8], [.7, .3]])
    _write_factors(tmp_path, W, None, prepared, {"converged": True})
    with np.load(tmp_path / "factors.npz", allow_pickle=False) as result:
        assert result.files == ["W"]
        np.testing.assert_array_equal(result["W"], W)
    fit = json.loads((tmp_path / "fit.json").read_text())
    assert fit["row_contract"] == "/project/data/shared.parquet"
    assert fit["training_row_order_sha256"] == "fixed_rows"


def test_selected_zero_alias_roundtrips_without_duplicate_arrays_or_centers(tmp_path):
    U = np.eye(3)[:, :2]
    meta = {"lambda_value": 0., "converged": True, "iterations": 2, "metadata": {}}
    atomic_json(tmp_path / "spectral.json", {"blocks": {"selected": meta, "zero": meta},
                 "factor_aliases": {"selected": "zero"}, "features": {"preprocessing": "P0_raw"}})
    atomic_npz(tmp_path / "spectral.npz", {"zero_U": U, "zero_V": U, "zero_singular_values": np.ones(2),
                 "panel_indices": np.arange(3), "retained_indices": np.arange(3),
                 "feature_weights": np.ones(3), "eta": np.ones(3) / 3, "row_totals": np.ones(3)})
    selected, _, _ = _read_embedding(tmp_path, "selected")
    zero, _, _ = _read_embedding(tmp_path, "lambda_zero")
    np.testing.assert_array_equal(selected.U, zero.U)
    assert selected.U_bar is None
    assert zero.U_bar is None


def test_config_does_not_persist_transfer_memberships_or_cell_scores():
    storage = default_config()["storage"]
    assert storage["transfer_W"] == "transient_during_evaluation"
    assert storage["per_observation_scores"].startswith("transient")
    assert storage["legacy_result_deletion"] is False
