"""Validated loader for the v2 raw What's Cooking >=8/Jaccard contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from . import real_data


CANONICAL_DATASET = "whats_cooking_raw_min8_jaccard_v2"
DATASET_ALIASES = frozenset(
    {
        "cook_raw_min8_jaccard_v2",
        "whats_cooking_raw_min8_jaccard_v2",
    }
)
DEFAULT_CONTRACT_ROOT = (
    real_data.DATA_ROOT
    / "cook"
    / "dataset"
    / "raw_min8_jaccard_v2"
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "counts_csr.npz",
        "recipes.csv.gz",
        "feature_ids.json",
        "dropped_feature_ids.json",
        "feature_provenance.csv.gz",
        "jaccard_edges.csv.gz",
        "directed_jaccard_neighbors.csv.gz",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 2:
        raise real_data.RealDataContractError(
            "raw Cooking >=8/Jaccard v2 manifest requires schema_version 2"
        )
    if manifest.get("contract_version") != "raw_min8_jaccard_v2":
        raise real_data.RealDataContractError(
            "unexpected raw Cooking >=8/Jaccard contract version"
        )
    if manifest.get("dataset") != CANONICAL_DATASET:
        raise real_data.RealDataContractError(
            "unexpected raw Cooking >=8/Jaccard v2 dataset id"
        )
    row_filter = manifest.get("row_filter", {})
    column_filter = manifest.get("column_filter", {})
    matrix = manifest.get("matrix", {})
    if int(row_filter.get("minimum_inclusive", -1)) != 8:
        raise real_data.RealDataContractError(
            "raw Cooking v2 contract must use the inclusive >=8 recipe rule"
        )
    if column_filter.get("rule") != "corpus count among retained recipes > 0":
        raise real_data.RealDataContractError(
            "raw Cooking v2 contract is missing its positive-corpus column rule"
        )
    if column_filter.get("applied_after") != (
        "raw literal ingredient-count row total >= 8"
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 column filter was not declared after its row filter"
        )
    shape = matrix.get("shape", [])
    if len(shape) != 2 or min(map(int, shape)) <= 0:
        raise real_data.RealDataContractError("raw Cooking v2 matrix shape is invalid")
    if (
        int(row_filter.get("after_n", -1)) != int(shape[0])
        or int(column_filter.get("after_p", -1)) != int(shape[1])
        or int(matrix.get("n", -1)) != int(shape[0])
        or int(matrix.get("p", -1)) != int(shape[1])
        or int(column_filter.get("zero_count_after", -1)) != 0
        or int(matrix.get("zero_feature_count", -1)) != 0
        or int(matrix.get("positive_feature_count", -1)) != int(shape[1])
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 row/column filter dimensions are inconsistent"
        )
    artifacts = manifest.get("artifacts", {})
    missing = REQUIRED_ARTIFACTS.difference(artifacts)
    if missing:
        raise real_data.RealDataContractError(
            f"raw Cooking v2 manifest is missing artifacts {sorted(missing)}"
        )


def _validate_edges(
    edges: pd.DataFrame,
    directed: pd.DataFrame,
    *,
    n: int,
    top_k: int,
) -> None:
    edge_required = {
        "src",
        "tgt",
        "weight",
        "jaccard_similarity",
        "jaccard_distance",
    }
    if not edge_required.issubset(edges.columns):
        raise real_data.RealDataContractError(
            f"raw Cooking v2 edge table requires {sorted(edge_required)}"
        )
    directed_required = {
        "src",
        "tgt",
        "rank",
        "src_base_row",
        "tgt_base_row",
        "src_recipe_id",
        "tgt_recipe_id",
        "jaccard_similarity",
        "jaccard_distance",
    }
    if not directed_required.issubset(directed.columns):
        raise real_data.RealDataContractError(
            "raw Cooking v2 directed-neighbor audit columns are incomplete"
        )
    endpoints = edges[["src", "tgt"]].to_numpy(dtype=np.int64)
    if (
        len(edges) == 0
        or np.any(endpoints[:, 0] < 0)
        or np.any(endpoints[:, 1] >= n)
        or np.any(endpoints[:, 0] >= endpoints[:, 1])
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 graph is empty, out of range, or noncanonical"
        )
    if not np.all(edges["weight"].to_numpy(dtype=float) == 1.0):
        raise real_data.RealDataContractError(
            "raw Cooking v2 modeling graph must use unit weights"
        )
    similarity = edges["jaccard_similarity"].to_numpy(dtype=float)
    distance = edges["jaccard_distance"].to_numpy(dtype=float)
    if (
        np.any((similarity < 0) | (similarity > 1))
        or not np.allclose(distance, 1.0 - similarity, rtol=0.0, atol=1e-15)
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 Jaccard similarities or distances are invalid"
        )
    if len(directed) != n * top_k:
        raise real_data.RealDataContractError(
            "raw Cooking v2 directed-neighbor table has the wrong row count"
        )
    counts = directed.groupby("src", sort=False).size().reindex(range(n), fill_value=0)
    if np.any(counts.to_numpy() != top_k):
        raise real_data.RealDataContractError(
            "raw Cooking v2 directed-neighbor table is incomplete by source"
        )
    if np.any(directed["src"].to_numpy() == directed["tgt"].to_numpy()):
        raise real_data.RealDataContractError(
            "raw Cooking v2 directed-neighbor table contains a self choice"
        )


def load_contract(
    contract_root: Path | str = DEFAULT_CONTRACT_ROOT,
) -> real_data.RealDataBundle:
    """Load v2 and validate every frozen artifact and matrix invariant."""

    root = Path(contract_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise real_data.RealDataContractError(
            "raw Cooking >=8/Jaccard v2 contract is missing; run "
            "scripts/data/prepare_cook_v2.py"
        )
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)

    for name, record in manifest["artifacts"].items():
        path = root / name
        if not path.is_file():
            raise real_data.RealDataContractError(
                f"raw Cooking v2 artifact is missing: {name}"
            )
        observed = _sha256_file(path)
        expected = str(record["sha256"])
        if observed != expected:
            raise real_data.RealDataContractError(
                f"raw Cooking v2 artifact {name} changed: {observed} != {expected}"
            )

    counts_sparse = sparse.load_npz(root / "counts_csr.npz").tocsr()
    recipes = pd.read_csv(root / "recipes.csv.gz", keep_default_na=False)
    feature_ids = json.loads((root / "feature_ids.json").read_text())
    dropped_feature_ids = json.loads((root / "dropped_feature_ids.json").read_text())
    feature_provenance = pd.read_csv(
        root / "feature_provenance.csv.gz", keep_default_na=False
    )
    edges = pd.read_csv(root / "jaccard_edges.csv.gz")
    directed = pd.read_csv(root / "directed_jaccard_neighbors.csv.gz")

    expected_shape = tuple(map(int, manifest["matrix"]["shape"]))
    if counts_sparse.shape != expected_shape:
        raise real_data.RealDataContractError(
            f"raw Cooking v2 count shape {counts_sparse.shape} != {expected_shape}"
        )
    if len(recipes) != expected_shape[0] or len(feature_ids) != expected_shape[1]:
        raise real_data.RealDataContractError(
            "raw Cooking v2 row or feature metadata are misaligned"
        )
    if not {"id", "cuisine", "base_row_index", "raw_literal_ingredient_count"}.issubset(
        recipes.columns
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 recipe provenance columns are incomplete"
        )
    row_totals = np.asarray(counts_sparse.sum(axis=1)).reshape(-1).astype(np.int64)
    corpus_counts = np.asarray(counts_sparse.sum(axis=0)).reshape(-1).astype(np.int64)
    if np.any(row_totals < 8):
        raise real_data.RealDataContractError(
            "raw Cooking v2 contains a recipe with fewer than eight ingredients"
        )
    if not np.array_equal(
        row_totals,
        recipes["raw_literal_ingredient_count"].to_numpy(dtype=np.int64),
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 recipe lengths do not match count row totals"
        )
    if np.any(corpus_counts <= 0):
        raise real_data.RealDataContractError(
            "raw Cooking v2 contains a zero-corpus ingredient column"
        )
    required_provenance = {
        "retained_column_index",
        "base_column_index",
        "feature_id",
        "corpus_count",
    }
    if not required_provenance.issubset(feature_provenance.columns):
        raise real_data.RealDataContractError(
            "raw Cooking v2 feature provenance columns are incomplete"
        )
    if (
        len(feature_provenance) != expected_shape[1]
        or not np.array_equal(
            feature_provenance["retained_column_index"].to_numpy(dtype=np.int64),
            np.arange(expected_shape[1], dtype=np.int64),
        )
        or list(map(str, feature_provenance["feature_id"]))
        != list(map(str, feature_ids))
        or not np.array_equal(
            feature_provenance["corpus_count"].to_numpy(dtype=np.int64),
            corpus_counts,
        )
    ):
        raise real_data.RealDataContractError(
            "raw Cooking v2 feature provenance is not aligned to the count matrix"
        )
    column_filter = manifest["column_filter"]
    if len(dropped_feature_ids) != int(column_filter["removed_p"]):
        raise real_data.RealDataContractError(
            "raw Cooking v2 dropped-feature audit has the wrong length"
        )
    if int(column_filter["before_p"]) != len(feature_ids) + len(dropped_feature_ids):
        raise real_data.RealDataContractError(
            "raw Cooking v2 retained and dropped features do not reconstruct base p"
        )

    top_k = int(manifest["graph"]["top_k_directed"])
    _validate_edges(edges, directed, n=expected_shape[0], top_k=top_k)
    bundle = real_data._make_bundle(
        dataset=CANONICAL_DATASET,
        counts=counts_sparse.toarray(),
        feature_ids=list(map(str, feature_ids)),
        observation_ids=recipes["id"].to_numpy(),
        group_ids=recipes["cuisine"].astype(str).to_numpy(),
        edge_df=edges,
        coordinates=None,
        outcomes=pd.DataFrame(
            {"cuisine": recipes["cuisine"].astype(str).to_numpy()}
        ),
        metadata={
            "canonical_scope": manifest["canonical_scope"],
            "contract_version": manifest["contract_version"],
            "graph": manifest["graph"]["definition"],
            "graph_distance": "standard binary-set Jaccard distance",
            "graph_edge_weights": "unit weights after top-five neighbor selection",
            "independent_unit": "recipe; count thinning performed within recipe",
            "feature_thresholding": (
                "zero-corpus columns removed after the >=8 row filter; no fitted "
                "frequency threshold in the canonical data contract"
            ),
            "row_filter": manifest["row_filter"],
            "column_filter": manifest["column_filter"],
            "raw_min8_jaccard_v2_contract_manifest": str(manifest_path.resolve()),
            "raw_contract_hashes_expected": manifest["contract_hashes"],
        },
    )
    observed_hashes = bundle.hashes()
    for key, expected in manifest["contract_hashes"].items():
        if observed_hashes.get(key) != expected:
            raise real_data.RealDataContractError(
                f"raw Cooking v2 {key} changed: "
                f"{observed_hashes.get(key)} != {expected}"
            )
    bundle.metadata["canonical_contract_validated"] = True
    bundle.metadata["canonical_contract_hashes"] = observed_hashes
    bundle.metadata["canonical_contract_validation"] = (
        "standalone literal-count >=8 corpus, positive-corpus columns, and "
        "rebuilt binary-set Jaccard graph"
    )
    return bundle
