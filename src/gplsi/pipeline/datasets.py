"""Turn one task into training data, held-out counts, and evaluation labels.

Two families of data are supported:

* **canonical** datasets (``crc``, ``spleen``, ``cook``, ``cook_v2``) come from
  :func:`gplsi.real_data.load_real_data` with their frozen graph, followed by an
  optional connected subset, binomial count thinning, and (Cooking) removal of
  recipes left empty by the Tran feature threshold;
* **dlpfc** (Visium) sections come from the processed AnnData file.  Counts are
  thinned sparsely, a gene panel is chosen from training counts only, spots
  with no training counts on the reference panel are dropped, and a
  within-section kNN graph is built.

Everything depends only on the task's seed, so the same task always rebuilds
the same matrices (post-hoc refits rely on this).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..preprocessing import select_feature_columns
from ..real_data import REPO_ROOT, RealDataBundle, load_real_data
from ..real_experiment import PREPROCESSING_SPECS
from .graph import build_within_unit_knn_graph
from .panels import panel_indices, rank_features_by_dispersion, tran_vocabulary
from .splits import thin_and_split_sparse_counts


CANONICAL_DATASETS = ("crc", "spleen", "cook", "cook_v2")


@dataclass
class TaskData:
    """Training bundle plus everything needed to score a fit."""

    bundle: RealDataBundle
    heldout: np.ndarray | None
    feature_names: np.ndarray
    labels: pd.DataFrame | None = None
    reference_columns: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def hashes(self) -> dict[str, Any]:
        output = dict(self.bundle.hashes())
        output["heldout_sha256"] = None if self.heldout is None else sha256_array(self.heldout, "<i8")
        return output


def sha256_array(value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_task_data(config: dict[str, Any], task: dict[str, Any]) -> TaskData:
    name = str(config["dataset"]["name"])
    if name == "dlpfc":
        return _prepare_dlpfc(config, task)
    if name in CANONICAL_DATASETS:
        return _prepare_canonical(config, task)
    raise ValueError(f"unknown dataset {name!r}; expected dlpfc or one of {CANONICAL_DATASETS}")


# --------------------------------------------------------------------------
# CRC, spleen, What's Cooking
# --------------------------------------------------------------------------


def _prepare_canonical(config: dict[str, Any], task: dict[str, Any]) -> TaskData:
    dataset = config["dataset"]
    seed = int(task["seed"])
    canonical = load_real_data(str(dataset["name"]), group=str(dataset.get("group", "BALBc-1")))
    bundle = canonical
    subset = config.get("subset")
    if subset:
        strategy = subset.get("strategy", "connected")
        if strategy == "whole_groups":
            bundle = _whole_group_subset(bundle, int(subset["n"]), seed=seed)
        elif strategy == "graph_stratified":
            bundle = bundle.graph_stratified_subset(
                int(subset["n"]), seed=seed, max_groups=subset.get("max_groups")
            )
        else:
            bundle = bundle.connected_subset(int(subset["n"]), seed=seed)
    heldout = None
    if config.get("heldout_fraction") is not None:
        bundle, heldout = bundle.binomial_thinning(float(config["heldout_fraction"]), seed=seed)
    summary = {"canonical_n": canonical.n, "canonical_p": canonical.p}
    if config.get("post_tran_row_filter"):
        bundle, heldout, audit = _drop_rows_empty_after_tran(bundle, heldout, config, K=int(task["K"]))
        summary["post_tran_row_filter"] = {
            key: value for key, value in audit.items()
            if not isinstance(value, list) or key == "matching_preprocessings"
        }
    summary.update(n=bundle.n, p=bundle.p, N_mean=bundle.N_mean)
    return TaskData(
        bundle=bundle,
        heldout=heldout,
        feature_names=np.asarray(bundle.feature_ids).astype(str),
        summary=summary,
    )


def _whole_group_subset(bundle: RealDataBundle, n: int, *, seed: int) -> RealDataBundle:
    """Whole groups (e.g. CRC regions) in random order until at least ``n`` rows.

    Keeps every kept group's graph intact, so spatial methods that work per
    group (spatial LDA's Voronoi graph) never see a group of one or two cells.
    """

    groups = np.asarray(bundle.group_ids).astype(str)
    order = np.random.default_rng(seed).permutation(np.unique(groups))
    chosen, total = [], 0
    for group in order:
        if total >= n:
            break
        chosen.append(group)
        total += int(np.sum(groups == group))
    rows = np.flatnonzero(np.isin(groups, chosen))
    return bundle.induced_row_subset(
        rows,
        metadata_updates={"subset_strategy": "whole_groups", "subset_groups": sorted(chosen)},
    )


def _drop_rows_empty_after_tran(
    bundle: RealDataBundle,
    heldout: np.ndarray | None,
    config: dict[str, Any],
    *,
    K: int,
) -> tuple[RealDataBundle, np.ndarray | None, dict[str, Any]]:
    """Freeze Tran's training-column support, then remove rows it leaves empty.

    The support is computed once, before any row is removed, and stored in the
    bundle metadata so :func:`gplsi.real_experiment.fit_spectral_block` reuses
    it instead of recomputing the cutoff on the smaller row set.  All
    preprocessings of the run must share one Tran method and alpha.
    """

    names = [block_name for block in config.get("gplsi", []) for block_name in block["preprocessings"]]
    signatures = {
        (
            str(PREPROCESSING_SPECS[name].get("threshold_method", "none")),
            float(PREPROCESSING_SPECS[name].get("alpha", 0.005)),
        )
        for name in names
    }
    if len(signatures) != 1:
        raise ValueError("post_tran_row_filter needs all preprocessings to share one Tran threshold")
    (method, alpha), = signatures
    if not method.startswith("tran"):
        raise ValueError("post_tran_row_filter needs a Tran-threshold preprocessing")

    X = np.asarray(bundle.frequencies, dtype=float)
    positive = np.flatnonzero(X.mean(axis=0) > 0)
    threshold = select_feature_columns(
        X[:, positive], bundle.document_lengths, method=method, alpha=alpha, K=K
    )
    retained = positive[threshold.retained_indices]
    keep = np.flatnonzero(np.asarray(bundle.counts[:, retained].sum(axis=1)).reshape(-1) > 0)
    if keep.size == 0:
        raise ValueError("Tran support leaves no nonempty training rows")
    audit = {
        "rule": "drop rows with zero training-count mass on the frozen Tran support",
        "threshold_method": threshold.effective_method,
        "threshold_value": float(threshold.threshold_value),
        "alpha": float(alpha),
        "n_before": int(bundle.n),
        "n_after": int(keep.size),
        "removed_row_count": int(bundle.n - keep.size),
        "positive_canonical_indices": positive.tolist(),
        "retained_canonical_indices": retained.tolist(),
        "eta_hat_positive": threshold.eta_hat.tolist(),
        "N_used": float(threshold.N_used),
        "unequal_document_lengths": bool(threshold.unequal_document_lengths),
        "top_10_percent_fallback": bool(threshold.fallback_active),
        "threshold_warnings": [str(value) for value in threshold.warnings],
        "matching_preprocessings": names,
    }
    filtered = bundle.induced_row_subset(
        keep,
        metadata_updates={
            "post_tran_zero_training_mass_filter": audit,
            "frozen_tran_feature_support": audit,
        },
    )
    filtered_heldout = None if heldout is None else np.ascontiguousarray(heldout[keep])
    return filtered, filtered_heldout, audit


# --------------------------------------------------------------------------
# Visium DLPFC
# --------------------------------------------------------------------------


def load_dlpfc_section(processed_file: Path, section: str) -> dict[str, Any]:
    """Read one section's counts, coordinates, ids, and evaluation-only labels."""

    import anndata as ad

    backed = ad.read_h5ad(processed_file, backed="r")
    unit_column = str(backed.uns["benchmark_unit_column"])
    mask = np.asarray(backed.obs[unit_column].astype(str) == str(section))
    if not mask.any():
        raise ValueError(f"section {section!r} not found in {processed_file}")
    adata = backed[mask].to_memory()
    backed.file.close()
    forbidden = [str(column) for column in adata.uns["fit_forbidden_obs_columns"]]
    return {
        "counts": adata.X.tocsr().astype(np.int32),
        "coordinates": np.asarray(adata.obsm["spatial"], dtype=float),
        "observation_ids": adata.obs_names.astype(str).to_numpy(),
        "feature_ids": adata.var_names.astype(str).to_numpy(),
        "feature_symbols": (
            adata.var["symbol"].astype(str).to_numpy() if "symbol" in adata.var else None
        ),
        "graph_unit_ids": adata.obs[str(adata.uns.get("graph_unit_column", unit_column))]
        .astype(str)
        .to_numpy(),
        "labels": adata.obs[[c for c in forbidden if c in adata.obs]].reset_index(drop=True),
    }


def _prepare_dlpfc(config: dict[str, Any], task: dict[str, Any]) -> TaskData:
    """One section: split counts, choose the vocabulary on training counts only.

    ``dataset.vocabulary`` is ``"dispersion"`` (default: the top ``panel_size``
    genes by variance/mean; spots need a training count on the top
    ``reference_panel_size`` genes, which also give a common scoring
    vocabulary across panel sizes) or ``"tran"`` (every gene passing the Tran
    threshold at ``dataset.tran_alpha``; spots need a training count on it).
    """

    dataset = config["dataset"]
    seed = int(task["seed"])
    vocabulary = str(dataset.get("vocabulary", "dispersion"))
    processed_file = REPO_ROOT / dataset["file"]
    section = load_dlpfc_section(processed_file, str(task["section"]))
    split = thin_and_split_sparse_counts(
        section["counts"],
        retained_fraction=float(task.get("retained_fraction", 1.0)),
        test_fraction=float(config["heldout_fraction"]),
        seed=seed,
    )
    vocabulary_summary: dict[str, Any] = {"vocabulary": vocabulary}
    if vocabulary == "dispersion":
        panel_size = int(task["panel_size"])
        reference_size = int(dataset["reference_panel_size"])
        if reference_size > panel_size:
            raise ValueError("the reference panel must be no larger than the fitted panel")
        # Training-only panel, nested across sizes; the reference panel is a prefix.
        ranking = rank_features_by_dispersion(
            split.train, detection_fraction=float(dataset["detection_fraction"])
        )
        panel = panel_indices(ranking, panel_size)
        reference_columns: np.ndarray | None = np.arange(reference_size)
        vocabulary_summary.update(
            panel_size=panel_size,
            reference_panel_size=reference_size,
            eligible_genes=ranking["eligible_count"],
            detection_threshold=ranking["detection_threshold"],
        )
    elif vocabulary == "tran":
        selection = tran_vocabulary(split.train, alpha=float(dataset["tran_alpha"]))
        panel = selection["indices"]
        reference_columns = None
        vocabulary_summary.update(
            tran_alpha=selection["alpha"],
            tran_threshold=selection["threshold"],
            vocabulary_size=selection["kept_count"],
            vocabulary_training_mass=selection["kept_mass"],
        )
    else:
        raise ValueError(f"unknown DLPFC vocabulary {vocabulary!r}; expected 'dispersion' or 'tran'")
    train_panel = split.train[:, panel].toarray()
    required = reference_columns if reference_columns is not None else np.arange(train_panel.shape[1])
    keep = train_panel[:, required].sum(axis=1) > 0
    train = train_panel[keep].astype(np.int64)
    heldout = split.test[:, panel].toarray()[keep].astype(np.int64)

    coordinates = section["coordinates"][keep]
    graph_units = section["graph_unit_ids"][keep]
    graph = build_within_unit_knn_graph(coordinates, graph_units, k=int(dataset["graph_k"]))
    lengths = train.sum(axis=1).astype(float)
    feature_ids = section["feature_ids"][panel]
    symbols = section["feature_symbols"]
    bundle = RealDataBundle(
        dataset=f"dlpfc::{task['section']}",
        counts=train,
        frequencies=train / lengths[:, None],
        document_lengths=lengths,
        feature_ids=feature_ids,
        observation_ids=section["observation_ids"][keep],
        group_ids=graph_units,
        edge_df=graph.edges,
        weights=graph.adjacency,
        coordinates=coordinates,
        outcomes=None,
        metadata={"graph": graph.metadata, "split": split.metadata},
    ).validate()
    return TaskData(
        bundle=bundle,
        heldout=heldout,
        feature_names=(symbols[panel] if symbols is not None else feature_ids).astype(str),
        labels=section["labels"].loc[keep].reset_index(drop=True),
        reference_columns=reference_columns,
        summary={
            "processed_file": str(dataset["file"]),
            "section_spots": int(section["counts"].shape[0]),
            "kept_spots": int(keep.sum()),
            "dropped_spots_zero_reference_training": int((~keep).sum()),
            **vocabulary_summary,
            "train_molecules": int(train.sum()),
            "heldout_molecules": int(heldout.sum()),
            "graph": graph.metadata,
            "n": bundle.n,
            "p": bundle.p,
            "N_mean": bundle.N_mean,
        },
    )
