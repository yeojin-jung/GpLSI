"""Design-driven ablation runner for whole-transcriptome sections (Visium DLPFC).

Differences from :mod:`gplsi_spatial_benchmark.runner` (kept unchanged for the
Midway benchmark):

* counts are split sparsely, so all ~33k genes never need to be densified;
* the gene panel is chosen per task from *training* counts only, as a nested
  prefix of one variance-to-mean ranking (``panel_size`` task column);
* the row mask is shared across panel sizes: spots are kept when they have
  positive training counts on the smallest ("reference") panel;
* held-out scores are reported on the fitted panel and, for comparisons across
  panel sizes, on the common reference panel (conditional composition);
* the GpLSI grid, A recoveries, hunter parameters, penalty grid, and baselines
  come from a named design in the JSON config (``design`` task column).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from gplsi.real_data import RealDataBundle

from .graph import build_within_unit_knn_graph
from .methods import fit_method_suite
from .metrics import external_structure_metrics, heldout_count_metrics, spatial_metrics, topic_profile_metrics
from .panels import panel_indices, rank_features_by_dispersion
from .runner import _git_head, _json_safe, _sha256
from .splits import thin_and_split_sparse_counts


def load_section(processed_file: Path, unit_id: str):
    """Read one section's counts, coordinates, ids, and evaluation-only labels."""

    import anndata as ad

    backed = ad.read_h5ad(processed_file, backed="r")
    unit_column = str(backed.uns["benchmark_unit_column"])
    mask = np.asarray(backed.obs[unit_column].astype(str) == str(unit_id))
    if not mask.any():
        raise ValueError(f"unit {unit_id!r} not found in {processed_file}")
    section = backed[mask].to_memory()
    backed.file.close()
    forbidden = [str(c) for c in section.uns["fit_forbidden_obs_columns"]]
    return {
        "counts": section.X.tocsr().astype(np.int32),
        "coordinates": np.asarray(section.obsm["spatial"], dtype=float),
        "observation_ids": section.obs_names.astype(str).to_numpy(),
        "feature_ids": section.var_names.astype(str).to_numpy(),
        "feature_symbols": (
            section.var["symbol"].astype(str).to_numpy() if "symbol" in section.var else None
        ),
        "graph_unit_ids": section.obs[str(section.uns.get("graph_unit_column", unit_column))].astype(str).to_numpy(),
        "external": section.obs[[c for c in forbidden if c in section.obs]].reset_index(drop=True),
    }


def _task_value(task: dict, key: str, default=None):
    value = task.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "":
        return default
    return value


def task_identity(task: dict) -> str:
    return "__".join(
        [
            str(task["design"]),
            str(task["unit_id"]),
            f"K{int(task['K'])}",
            f"p{int(task['panel_size'])}",
            f"r{float(task['retained_fraction'])}",
            f"s{int(task['seed'])}",
        ]
    )


def prepare_task_data(config: dict, task: dict, repo_root: Path) -> dict:
    """Deterministically rebuild a task's split, panel, spot mask, graph, and bundle.

    Recovery-only refits (e.g. Poisson A from a saved W) call this again and get
    identical training/test matrices, because every step depends only on the
    task's seed and the training counts.
    """

    seed = int(task["seed"])
    panel_size = int(task["panel_size"])
    reference_size = int(config["panel"]["reference_size"])
    if reference_size > panel_size:
        raise ValueError("reference panel must be no larger than the fitted panel")
    processed_file = repo_root / config["processed_file"]
    section = load_section(processed_file, str(task["unit_id"]))
    split = thin_and_split_sparse_counts(
        section["counts"],
        retained_fraction=float(task["retained_fraction"]),
        test_fraction=float(config["heldout_test_fraction"]),
        seed=seed,
    )
    # Training-only panel, nested across sizes; the reference panel is a prefix.
    ranking = rank_features_by_dispersion(
        split.train, detection_fraction=float(config["panel"]["detection_fraction"])
    )
    panel = panel_indices(ranking, panel_size)
    reference_columns = np.arange(reference_size)  # positions within `panel`
    train_panel = split.train[:, panel].toarray()
    test_panel = split.test[:, panel].toarray()
    keep = train_panel[:, reference_columns].sum(axis=1) > 0
    train = train_panel[keep].astype(np.int64)
    test = test_panel[keep].astype(np.int64)

    coordinates = section["coordinates"][keep]
    graph_units = section["graph_unit_ids"][keep]
    external = section["external"].loc[keep].reset_index(drop=True)
    feature_ids = section["feature_ids"][panel]
    symbols = section["feature_symbols"][panel] if section["feature_symbols"] is not None else feature_ids
    graph = build_within_unit_knn_graph(coordinates, graph_units, k=int(config["graph"]["k"]))
    lengths = train.sum(axis=1).astype(float)
    bundle = RealDataBundle(
        dataset=f"{config['dataset']}::{task['unit_id']}",
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
        metadata={"external_columns_withheld": list(external.columns)},
    ).validate()
    return dict(
        bundle=bundle, test=test, external=external, graph=graph, coordinates=coordinates,
        feature_ids=feature_ids, symbols=symbols, reference_columns=reference_columns,
        observation_ids=section["observation_ids"][keep], split=split, ranking=ranking,
        processed_file=processed_file,
        data_summary={
            "processed_file": str(config["processed_file"]),
            "section_spots": int(section["counts"].shape[0]),
            "kept_spots": int(keep.sum()),
            "dropped_spots_zero_reference_training": int((~keep).sum()),
            "panel_size": panel_size,
            "reference_panel_size": reference_size,
            "eligible_genes": ranking["eligible_count"],
            "detection_threshold": ranking["detection_threshold"],
            "train_molecules": int(train.sum()),
            "test_molecules": int(test.sum()),
        },
    )


def score_fit(W: np.ndarray, A: np.ndarray, prepared: dict, seed: int) -> tuple[dict, list]:
    test, ref = prepared["test"], prepared["reference_columns"]
    reference = heldout_count_metrics(W, A[:, ref], test[:, ref])
    metrics = {
        **heldout_count_metrics(W, A, test),
        **{f"reference_panel__{k}": v for k, v in reference.items()},
        **external_structure_metrics(W, prepared["external"], seed),
        **spatial_metrics(W, prepared["graph"].edges),
    }
    profile_metrics, top_features = topic_profile_metrics(A, prepared["symbols"])
    metrics.update(profile_metrics)
    return metrics, top_features


def run_task(config_path: Path, task: dict, *, repo_root: Path | None = None) -> Path:
    started = perf_counter()
    repo_root = Path(repo_root or Path.cwd())
    config = json.loads(Path(config_path).read_text())
    design_name = str(task["design"])
    design = config["designs"][design_name]
    seed = int(task["seed"])
    K = int(task["K"])

    output_dir = repo_root / config["output_root"] / design_name
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = task_identity(task)
    json_path = output_dir / f"{identity}.json"
    npz_path = output_dir / f"{identity}.npz"

    prepared = prepare_task_data(config, task, repo_root)
    bundle, coordinates, graph = prepared["bundle"], prepared["coordinates"], prepared["graph"]
    split, ranking, processed_file = prepared["split"], prepared["ranking"], prepared["processed_file"]

    spectral_parameters = dict(config["spectral"])
    spectral_parameters.update(design.get("spectral_overrides", {}))
    if os.environ.get("GPLSI_GRAPH_N_JOBS"):
        spectral_parameters["n_jobs"] = int(os.environ["GPLSI_GRAPH_N_JOBS"])
    fits = fit_method_suite(
        bundle,
        coordinates,
        K,
        seed,
        document_preprocessings=tuple(design["document_preprocessings"]),
        document_vertex_hunters=tuple(design["document_vertex_hunters"]),
        anchor_preprocessings=tuple(design.get("anchor_preprocessings", [])),
        anchor_vertex_hunters=tuple(design.get("anchor_vertex_hunters", [])),
        initialization=str(config.get("spectral_initialization", "current")),
        spectral_parameters=spectral_parameters,
        A_recoveries=tuple(design["A_recoveries"]),
        vertex_parameters=config.get("vertex_parameters"),
        competitors=tuple(design.get("competitors", [])),
        include_spatial_lda="spatial_lda" in design.get("competitors", []),
        recovery_parameters=config.get("recovery"),
    )

    records, arrays = [], {}
    for index, fit in enumerate(fits):
        record = {
            "method": fit.method,
            "status": fit.status,
            "runtime_seconds": fit.runtime_seconds,
            "warnings": fit.warnings,
            "metadata": fit.metadata,
        }
        # Metrics are computed for every finite fit; `status` records whether the
        # optimizer certified convergence, so summaries can restrict to "ok".
        if fit.W is not None and fit.A is not None and np.isfinite(fit.W).all() and np.isfinite(fit.A).all():
            record["metrics"], record["top_features"] = score_fit(fit.W, fit.A, prepared, seed)
            arrays[f"W_{index}"] = fit.W.astype(np.float32)
            arrays[f"A_{index}"] = fit.A.astype(np.float32)
        records.append(record)
    arrays["observation_ids"] = prepared["observation_ids"]
    arrays["feature_ids"] = prepared["feature_ids"]
    arrays["feature_symbols"] = prepared["symbols"]
    arrays["coordinates"] = coordinates

    partial = npz_path.with_suffix(".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(npz_path)
    payload = {
        "schema_version": 1,
        "task": task,
        "identity": identity,
        "data": prepared["data_summary"],
        "split": split.metadata,
        "graph": graph.metadata,
        "array_index": {record["method"]: i for i, record in enumerate(records)},
        "results": records,
        "provenance": {
            "config_sha256": _sha256(Path(config_path)),
            "processed_h5ad_sha256": _sha256(processed_file),
            "python": platform.python_version(),
            "host": platform.node(),
            "gplsi_commit": _git_head(repo_root),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 * 1024 if sys.platform == "darwin" else 1024),
            "elapsed_seconds": perf_counter() - started,
            "spectral_parameters": spectral_parameters,
        },
    }
    partial_json = json_path.with_suffix(".json.partial")
    partial_json.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    partial_json.replace(json_path)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ablation task from a task manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.tasks, dtype={"unit_id": str})
    if not 0 <= args.index < len(frame):
        raise IndexError(f"task index {args.index} outside [0,{len(frame)})")
    print(run_task(args.config, frame.iloc[args.index].to_dict()))


if __name__ == "__main__":
    main()
