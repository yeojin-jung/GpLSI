"""Prespecified compact scientific figures from completed joint_v2 artifacts.

Map selection uses configuration and coordinates only. Every observation in
the selected original spatial stratum is plotted; no new per-cell data files
are persisted. W-parent identities prevent duplicate native/reference maps.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd

from .artifacts import atomic_json, compatible_completed
from .likelihood import simplex_matrix
from .metrics import topic_profile_metrics
from .splits import array_hash


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _number(value):
    return np.nan if value is None else float(value)


def _directory(root, task):
    prefix = "data/interim/joint_v2/stages" if task["stage"] == "prepare" else "results/joint_v2/stages"
    return Path(root) / prefix / task["stage"] / task["task_id"]


def plot_diagnostic_path(diagnostic, output_path, *, title="", task_id=None):
    """Keep zero lambda, negative Moran values, failed gaps, and undefined flags."""
    plt = _pyplot()
    spectral = sorted(diagnostic["spectral_path"], key=lambda r: r["lambda"])
    geometry = diagnostic["geometry_path"]
    x = np.array([r["lambda"] for r in spectral], float)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axes.ravel()
    for fold in range(5):
        values = [_number(r.get("CV_five_scores", [None] * 5)[fold])
                  if r.get("CV_five_scores") is not None and len(r["CV_five_scores"]) == 5 else np.nan for r in spectral]
        ax[0].plot(x, values, alpha=.5, linewidth=1, label=f"Fold {fold}")
    ax[0].plot(x, [_number(r.get("CV_aggregate_score")) for r in spectral], color="black", marker="o", label="Five-fold sum")
    ax[0].set_ylabel("Prespecified graph-CV loss")
    ax[0].legend(fontsize=7)
    ax[1].plot(x, [_number(r.get("optimized_unscaled_edge_group_norm")) for r in spectral], marker="o", label="Edge group norm")
    ax[1].plot(x, [_number(r.get("optimized_lambda_times_edge_group_norm")) for r in spectral], marker="s", label="Lambda × edge group norm")
    ax[1].set_ylabel("Actual optimized graph penalty")
    ax[1].legend(fontsize=7)
    quantities = [(2, "W_edge_squared_difference", "Final W weighted roughness"),
                  (3, "moran_I_equal_stratum_mean", "Moran's I (within-stratum centering)"),
                  (4, "one_minus_PAS_10", "1 − PAS₁₀"),
                  (5, "occupied_topics", "Occupied topics (hard memberships)")]
    audit = {"task_id": task_id, "lambda_includes_exact_zero": bool(np.any(x == 0)),
             "used_for_lambda_selection": False, "metric_coverage": {},
             "spectral_status_counts": pd.Series([r.get("status", "missing") for r in spectral]).value_counts().to_dict()}
    for hunter in sorted({r["hunter"] for r in geometry}):
        rows = sorted([r for r in geometry if r["hunter"] == hunter], key=lambda r: r["lambda"])
        for panel, field, ylabel in quantities:
            y = np.array([_number(r.get(field)) for r in rows])
            # NaN breaks the line at undefined or failed configurations.
            display = np.where(np.isfinite(y), y, np.nan)
            ax[panel].plot([r["lambda"] for r in rows], display, marker="o", markersize=3, label=hunter)
            ax[panel].set_ylabel(ylabel)
            audit["metric_coverage"][hunter + "/" + field] = {
                "expected_points": len(y), "finite_points": int(np.isfinite(y).sum()),
                "undefined_or_nonfinite_points": int((~np.isfinite(y)).sum()),
                "negative_points": int(np.sum(y < 0))}
    for panel in range(2, 6):
        ax[panel].legend(fontsize=6)
    positives = x[x > 0]
    for axis in ax:
        axis.set_xscale("symlog", linthresh=float(positives.min()) if len(positives) else 1e-6)
        axis.set_xlabel("Lambda (exact zero shown)")
        axis.grid(alpha=.15)
    ax[3].axhline(0, color="gray", linewidth=.6)
    ax[4].set_ylim(-.02, 1.02)
    undefined = sum(v["undefined_or_nonfinite_points"] for v in audit["metric_coverage"].values())
    fig.suptitle(title + f"\nUndefined/failed metric points: {undefined}; gaps are retained. Spatial metrics did not select lambda.", fontsize=10)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    atomic_json(output_path.with_suffix(".json"), audit)
    return {"figure": str(output_path), **audit}


def prespecified_map_sources(tasks, bases, config):
    """First frozen outer split, primary K/panel/seed, P0/SPA + anchor + competitors."""
    first = {dataset: min(row["outer_split_id"] for row in bases if row["dataset"] == dataset)
             for dataset in {row["dataset"] for row in bases}}
    selected = []
    for task in tasks:
        if task["stage"] not in ("geometry", "competitor"):
            continue
        spec = task["spec"]
        dataset = spec["dataset"]
        if (spec["outer_split_id"] != first[dataset] or spec["K"] != config["primary_K"][dataset]
            or spec["panel_requested"] != config["primary_panel"][dataset]
            or spec["estimator_seed"] != config["seeds"]["estimator_seed"] or spec["retention"] != 1):
            continue
        if task["stage"] == "geometry":
            if (spec.get("family") not in ("document", "anchor") or spec.get("preprocessing") != "P0_raw"
                or spec.get("hunter") != "spa_current" or spec.get("control") != "selected"):
                continue
        elif spec["method"] not in config["competitors"]:
            continue
        selected.append(task)
    return selected


def plot_training_map(W, observations, output_path, *, title="", expected_row_hash=None, W_parent=None):
    w = simplex_matrix(W, "W")
    required = {"obs_id", "graph_id", "x", "y"}
    if not required.issubset(observations) or len(observations) != len(w):
        raise ValueError("shared row contract and training W do not align")
    if observations.obs_id.duplicated().any():
        raise ValueError("duplicate observation IDs in shared row contract")
    if expected_row_hash is not None and array_hash(observations.obs_id.astype(str)) != expected_row_hash:
        raise ValueError("training row hash mismatch; refusing an incorrectly ordered map")
    stratum = sorted(observations.graph_id.astype(str).unique())[0]
    mask = observations.graph_id.astype(str).to_numpy() == stratum
    coordinates = observations.loc[mask, ["x", "y"]].to_numpy(float)
    if not np.isfinite(coordinates).all():
        raise ValueError("nonfinite map coordinates")
    local = w[mask]
    hard, confidence = local.argmax(axis=1), local.max(axis=1)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    points = axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=hard, cmap="tab20",
                             vmin=-.5, vmax=w.shape[1]-.5, s=3, linewidths=0, rasterized=True)
    fig.colorbar(points, ax=axes[0], ticks=np.arange(w.shape[1]), label="Method-specific topic index")
    certainty = axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=confidence, cmap="viridis",
                                vmin=1 / w.shape[1], vmax=1, s=3, linewidths=0, rasterized=True)
    fig.colorbar(certainty, ax=axes[1], label="Maximum topic membership")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set(xlabel="Source x coordinate", ylabel="Source y coordinate")
    fig.suptitle(title + f"\nFirst sorted training graph stratum; all {len(local):,} observations shown. Topic colors are not aligned across methods.", fontsize=9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    audit = {"W_parent": W_parent, "graph_stratum": stratum,
             "stratum_selection": "first_lexicographically_sorted_training_graph_id",
             "plotted_observations": int(mask.sum()), "eligible_stratum_observations": int(mask.sum()),
             "all_training_observations": len(w), "fitting_subsampling": False,
             "plotting_subsampling": False, "labels_used_for_selection": False,
             "training_row_order_sha256": array_hash(observations.obs_id.astype(str)),
             "hard_topic_counts": np.bincount(hard, minlength=w.shape[1]).tolist(),
             "no_per_cell_plot_data_persisted": True}
    atomic_json(output_path.with_suffix(".json"), audit)
    return {"figure": str(output_path), **audit}


@lru_cache(maxsize=6)
def available_symbols(root, dataset):
    """Use actual source symbol columns only; otherwise retain gene IDs alone."""
    root = Path(root)
    id_fields = ("feature_id", "gene_id", "gene_ids", "ensembl_id")
    symbol_fields = ("gene_symbol", "gene_symbols", "gene_name", "feature_name", "symbol")
    mapping, provenance = {}, []
    path = root / "data/interim" / dataset / "features.tsv.gz"
    if path.exists():
        frame = pd.read_csv(path, sep="\t")
        identifier = next((c for c in id_fields if c in frame), None)
        symbol = next((c for c in symbol_fields if c in frame), None)
        if identifier and symbol:
            for gene, name in zip(frame[identifier], frame[symbol]):
                if pd.notna(gene) and pd.notna(name) and str(name):
                    mapping[str(gene)] = str(name)
            provenance.append({"path": str(path), "ID_column": identifier, "symbol_column": symbol})
    contract_path = root / "data/processed/joint_v2" / dataset / "contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        for source in contract.get("sources", []):
            path = Path(source["path"])
            if path.suffix != ".h5ad" or not path.exists() or not path.resolve().is_relative_to((root / "data").resolve()):
                continue
            import anndata as ad
            object_ = ad.read_h5ad(path, backed="r")
            try:
                symbol = next((c for c in symbol_fields if c in object_.var), None)
                if symbol:
                    for gene, name in zip(object_.var_names, object_.var[symbol]):
                        if pd.notna(name) and str(name):
                            mapping[str(gene)] = str(name)
                    provenance.append({"path": str(path), "ID_column": "var_names", "symbol_column": symbol})
            finally:
                object_.file.close()
    return mapping, provenance


def top_profile_rows(A, feature_ids, *, symbols=None, top_n=20):
    a = simplex_matrix(A, "A")
    genes = np.asarray(feature_ids, str)
    if genes.shape != (a.shape[1],) or len(set(genes)) != len(genes):
        raise ValueError("unique feature IDs must match profile columns")
    symbols = symbols or {}
    metrics = topic_profile_metrics(a)
    rows = []
    for topic in range(a.shape[0]):
        columns = np.argsort(-a[topic], kind="stable")[:min(top_n, a.shape[1])]
        for rank, gene in enumerate(columns, 1):
            rows.append({"topic": topic, "rank": rank, "feature_id": genes[gene],
                         "gene_symbol": symbols.get(genes[gene]), "probability": float(a[topic, gene]),
                         "exclusivity": float(metrics["topic_feature_exclusivity"][topic, gene]),
                         "normalized_topic_entropy": float(metrics["normalized_entropy"][topic])})
    return rows


def plot_profile_stability(rows, output_path):
    usable = [r for r in rows if r.get("status") == "ok" and np.isfinite(_number(r.get("JSD_mean")))]
    if not usable:
        return None
    plt = _pyplot()
    groups = defaultdict(list)
    for row in usable:
        groups[(row["dataset"], row["comparison"])].append(_number(row["JSD_mean"]))
    keys = sorted(groups)
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * .8), 5), constrained_layout=True)
    for i, key in enumerate(keys):
        values = np.asarray(groups[key])
        # A deterministic strip displays all pair values; these are dependent
        # profile comparisons and receive no pair-as-replicate SE/error bar.
        offset = np.linspace(-.15, .15, len(values)) if len(values) > 1 else [0]
        ax.scatter(i + np.asarray(offset), values, s=8, alpha=.35)
        ax.plot([i - .2, i + .2], [np.median(values)] * 2, color="black", linewidth=2)
    ax.set_xticks(range(len(keys)), ["\n".join(key) for key in keys], rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="A-only matched base-2 JSD", title="Profile comparison values; black lines are descriptive medians, no pair-based SE")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {"figure": str(output_path), "profile_pairs": len(usable), "pairwise_SE_reported": False,
            "failed_or_undefined_pairs": len(rows) - len(usable), "labels_used_for_alignment": False}


def generate_figures(root, manifest, output_directory, *, tables_directory=None,
                     profile_rows=(), verify_hashes=False):
    root, directory = Path(root), Path(output_directory)
    tables_directory = (Path(tables_directory) if tables_directory is not None
                        else root / "reports/joint_v2" / directory.name / "scientific")
    config = manifest.metadata["config"]
    diagnostics, maps, tables, pending, failures = [], [], [], [], []
    for task in manifest.tasks("diagnostics"):
        source = _directory(root, task)
        if not compatible_completed(source, task["task_id"], verify_hashes=verify_hashes):
            continue
        try:
            diagnostics.append(plot_diagnostic_path(json.loads((source / "diagnostics.json").read_text()),
                directory / "lambda_paths" / (task["task_id"][:16] + ".png"),
                title=f"{task['dataset']} | {task['spec']['outer_split_id']} | {task['spec']['preprocessing']}", task_id=task["task_id"]))
        except Exception as exc:
            failures.append({"task_id": task["task_id"], "artifact": "diagnostic_plot", "error": str(exc)})
    sources = prespecified_map_sources(manifest.tasks(), manifest.metadata["bases"], config)
    selected_ids = {task["task_id"] for task in sources}
    for task in sources:
        source = _directory(root, task)
        if not compatible_completed(source, task["task_id"], verify_hashes=verify_hashes):
            pending.append({"task_id": task["task_id"], "method": task["spec"]["method"], "selection": "prespecified_map_parent_not_complete"})
            continue
        try:
            fit = json.loads((source / "fit.json").read_text())
            observations = pd.read_parquet(fit["row_contract"])
            with np.load(source / "factors.npz", allow_pickle=False) as saved:
                w = saved["W"]
            maps.append(plot_training_map(w, observations,
                directory / "maps" / f"{task['dataset']}_{task['task_id'][:16]}.png",
                title=f"{task['dataset']} | {task['spec']['method']} | K={task['spec']['K']}",
                expected_row_hash=fit["training_row_order_sha256"], W_parent=str(source)))
        except Exception as exc:
            failures.append({"task_id": task["task_id"], "artifact": "training_map", "error": str(exc)})
    # Competitor-native A participates only in its internal method fit. Every
    # displayed profile comes from the separately certified Poisson recovery.
    profile_sources = []
    for task in manifest.tasks():
        if (task["stage"] in ("recovery", "reference_recovery")
            and task["spec"].get("recovery") == "A_full_Pois" and selected_ids.intersection(task["parents"])):
            profile_sources.append(task)
    for task in profile_sources:
        source = _directory(root, task)
        if not compatible_completed(source, task["task_id"], verify_hashes=verify_hashes):
            continue
        try:
            path = source / "A.npz"
            with np.load(path, allow_pickle=False) as saved:
                a, genes = saved["A"], saved["feature_ids"]
            symbols, provenance = available_symbols(str(root), task["dataset"])
            rows = [{"task_id": task["task_id"], "dataset": task["dataset"], "method": task["spec"]["method"],
                     "recovery": task["spec"].get("recovery", "native"), "vocabulary": task["spec"].get("vocabulary", "native"),
                     **row} for row in top_profile_rows(a, genes, symbols=symbols)]
            table_path = tables_directory / "top_profiles" / f"{task['dataset']}_{task['task_id'][:16]}_top20.csv"
            table_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(table_path, index=False)
            tables.append({"path": str(table_path), "task_id": task["task_id"], "rows": len(rows),
                           "symbol_sources": provenance, "symbol_mapping_available": bool(symbols)})
        except Exception as exc:
            failures.append({"task_id": task["task_id"], "artifact": "top_profile_table", "error": str(exc)})
    stability = plot_profile_stability(profile_rows, directory / "profile_stability.png")
    result = {"diagnostic_figures": diagnostics, "prespecified_training_maps": maps,
              "profile_tables": tables, "profile_stability": stability, "pending_prespecified_maps": pending,
              "failures": failures, "W_parent_maps_deduplicated_across_A_recoveries": True,
              "no_new_per_cell_data_persisted": True,
              "map_selection": "first_frozen_outer_split; primary_K/panel/seed/full_counts; first_sorted_training_stratum"}
    atomic_json(directory / "FIGURE_INDEX.json", result)
    return result
