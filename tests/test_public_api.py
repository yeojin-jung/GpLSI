"""Regression checks for portable public imports and estimator controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import gplsi


def test_original_public_helpers_remain_importable() -> None:
    for name in (
        "GpLSI", "generate_data", "generate_weights_edge", "graphSVD",
        "_euclidean_proj_simplex", "get_component_mapping", "get_F_err",
        "get_l1_err", "get_accuracy", "moran", "get_PAS",
    ):
        assert callable(getattr(gplsi, name)), name
        assert name in gplsi.__all__


def test_estimator_uses_requested_graph_folds_on_default_preprocessing() -> None:
    rng = np.random.default_rng(149)
    n, p, K = 24, 7, 2
    X = rng.dirichlet(np.ones(p), size=n)
    edge = pd.DataFrame({"src": np.arange(n - 1), "tgt": np.arange(1, n), "weight": 1.0})
    weights = csr_matrix((edge.weight, (edge.src, edge.tgt)), shape=(n, n))
    model = gplsi.GpLSI(
        grid_len=1, maxiter=1, graph_nfolds=4,
        graph_cv_fold_mode="all", graph_n_jobs=1, random_state=17,
    ).fit(X, 50, K, edge, weights)
    assert model.preprocessing_result is None
    assert model.lambd_errs["scoring_folds"] == [0, 1, 2, 3]
    assert set(model.lambd_errs["fold_errors"]) == {0, 1, 2, 3}
    repeated = gplsi.GpLSI(
        grid_len=1, maxiter=1, graph_nfolds=4,
        graph_cv_fold_mode="all", graph_n_jobs=1, random_state=17,
    ).fit(X, 50, K, edge, weights)
    np.testing.assert_allclose(model.W_hat, repeated.W_hat, atol=1e-12)


def test_core_import_avoids_optional_experiment_dependencies() -> None:
    script = (
        "import sys; import gplsi; "
        "assert not any(name == 'rpy2' or name.startswith('rpy2.') for name in sys.modules); "
        "assert 'gplsi.simulation' not in sys.modules; "
        "assert 'gplsi.realdata_spleen' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_real_data_locations_allow_explicit_overrides(tmp_path: Path) -> None:
    data_path = tmp_path / "custom-data"
    contract_path = tmp_path / "custom-contract.json"
    environment = dict(os.environ, GPLSI_DATA_ROOT=str(data_path), GPLSI_AUDIT_CONTRACT=str(contract_path))
    script = (
        "import json; from gplsi.real_data import DATA_ROOT, AUDIT_CONTRACT; "
        "print(json.dumps([str(DATA_ROOT), str(AUDIT_CONTRACT)]))"
    )
    result = subprocess.run([sys.executable, "-c", script], env=environment, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [str(data_path), str(contract_path)]


def test_default_frozen_contract_is_available_with_the_package() -> None:
    environment = os.environ.copy()
    environment.pop("GPLSI_AUDIT_CONTRACT", None)
    script = (
        "import json; from gplsi.real_data import AUDIT_CONTRACT; "
        "assert AUDIT_CONTRACT.is_file(); "
        "assert AUDIT_CONTRACT.parent.name == 'contracts'; "
        "print(json.dumps(sorted(json.loads(AUDIT_CONTRACT.read_text())['datasets'])))"
    )
    result = subprocess.run([sys.executable, "-c", script], env=environment, check=True, capture_output=True, text=True)
    assert set(json.loads(result.stdout)) == {
        "stanford_crc_codex", "mouse_spleen_codex", "whats_cooking",
    }
