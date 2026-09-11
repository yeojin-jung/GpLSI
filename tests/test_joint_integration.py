import numpy as np

from gplsi_joint_v2.smoke import synthetic_fixture, check_leakage_sentinel, forbid_full_sparse_densification


def test_joint_training_heldout_and_annotation_leakage_with_all_five_graph_folds(monkeypatch):
    import gplsi_joint_v2.likelihood as likelihood
    def forbidden(*args, **kwargs):
        raise AssertionError("production leakage smoke must not fit A_current")
    monkeypatch.setattr(likelihood, "recover_A_current", forbidden)
    fixture = synthetic_fixture()
    n, p = fixture["train"].shape
    with forbid_full_sparse_densification([(n, p), (p, n), (n, n)]):
        fitted, graph, audit = check_leakage_sentinel(fixture)
    assert all(audit.values())
    assert audit["reported_profile_recovery_is_Poisson"] is True
    assert len(np.unique(fixture["biological_ids"])) == 3
    rows, columns = graph.nonzero()
    assert np.all(fixture["strata"][rows] == fixture["strata"][columns])
    assert fitted.selected.U.shape == (n, 3)
    assert set(fitted.metadata["cv"]["fold_curves"]) == {0.0, 1e-4}
    assert fitted.features.metadata["actual_panel"] == 24
