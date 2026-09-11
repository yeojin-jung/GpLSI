#!/usr/bin/env python3
"""Build compact, static data for the real-data results dashboard.

The ``select`` stage reads only completed fit tables and chooses full-coverage
configuration leaders plus representative fits. The ``materialize`` stage
loads just those selected estimate artifacts and canonical graph objects.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

METRIC = "heldout_metrics.heldout_multinomial_deviance"
COUNT = "heldout_metrics.heldout_count_total"
CONFIG_KEYS = [
    "dataset",
    "summary_group",
    "K",
    "estimator_family",
    "spectral_geometry",
    "vertex_hunter",
    "preprocessing",
    "A_recovery",
    "W_recovery_method",
]
PIPELINE_KEYS = [key for key in CONFIG_KEYS if key != "summary_group"]
GPLSI_ESTIMATORS = (
    (
        "document_gplsi",
        "document_U",
        "stable_document_vertex_solve",
    ),
    (
        "anchor_feature_gplsi",
        "word_Z",
        "Ghat_simplex_prevalence",
    ),
)
VERTEX_HUNTERS = ("spa_current", "svs", "svs_star", "palm", "palm_accelerated")
PREPROCESSINGS = (
    "P0_raw",
    "P1_tran_alpha_0p005",
    "P2_ke_weighted",
    "P3_tran_then_ke",
)
A_RECOVERIES = ("A_current", "A_full_Pois")
PLAN_CONFIGS = (
    REPO_ROOT / "configs/real_data_anchor_word_gplsi/crc/full.json",
    REPO_ROOT / "configs/real_data_anchor_word_gplsi/spleen/full_BALBc-1.json",
    REPO_ROOT / "configs/real_data_anchor_word_gplsi/spleen/full_BALBc-2.json",
    REPO_ROOT / "configs/real_data_anchor_word_gplsi/spleen/full_BALBc-3.json",
    REPO_ROOT / "configs/real_data_anchor_word_gplsi/spleen/full_joint.json",
    REPO_ROOT
    / "configs/real_data_anchor_word_gplsi/cook/full_manuscript_K5-7.json",
)
INDIVIDUAL_SPLEEN_GROUPS = frozenset({"BALBc-1", "BALBc-2", "BALBc-3"})
DATASET_LABELS = {
    "stanford_crc_codex": "Stanford CRC",
    "mouse_spleen_codex": "Mouse spleen",
    "whats_cooking": "What's Cooking",
}

ESTIMATOR_LABELS = {
    "document_gplsi": "GpLSI-U (observation simplex)",
    "anchor_feature_gplsi": "GpLSI-Z (feature-profile simplex)",
    "plsi": "pLSI (no graph)",
    "topicscore_raw": "Topic-SCORE (raw)",
    "topicscore_graph_denoised": "Topic-SCORE (graph-denoised)",
    "lda": "LDA (non-spatial)",
    "spatial_lda": "Spatial LDA (Calico)",
    "cell_type_baseline": "Observed 3-hop cell-type composition (8 fixed topics; no topic fit)",
    "published_baseline": "Published artifact — reference only",
}
METHOD_DESCRIPTIONS = {
    "document_gplsi": "Fits a simplex to the graph-denoised observation-side U cloud, recovers topic weights W by a stable vertex solve, and then estimates topic composition A.",
    "anchor_feature_gplsi": "Builds an original-scale normalized feature-profile Z cloud from graph-denoised factors, fits a simplex to that cloud, and recovers W through estimated topic prevalence without requiring pure observations.",
    "plsi": "Fits the graph-free truncated-SVD pLSI path to the same thinned training data; W is shared across its current-A and Poisson-A refits.",
    "topicscore_raw": "Runs the audited Tran Topic-SCORE port on canonical training frequencies with native eta^(-1/2) normalization and no graph denoising.",
    "topicscore_graph_denoised": "Applies Topic-SCORE ratio geometry to the P0 graph-aligned factors on the original unweighted feature scale; this is not anchor-feature GpLSI.",
    "lda": "Fits non-spatial latent Dirichlet allocation directly to the same thinned count matrix and uses its native W and A factors.",
    "spatial_lda": "Fits the vendored Calico spatial LDA to the same thinned counts using observation coordinates; unavailable when coordinates are absent.",
    "cell_type_baseline": "Uses the eight observed CRC cell-type proportions directly as fixed topics (A=I, W=X), then applies the selected composition transform and downstream classifier with the identical region aggregation and repeated folds used for learned topics.",
    "published_baseline": "Frozen historical output retained for provenance; it was not refit on the shared count-thinning split and has no comparable held-out score.",
}
VERTEX_LABELS = {
    "spa_current": "SPA",
    "svs": "SVS",
    "svs_star": "SVS*",
    "palm": "PALM",
    "palm_accelerated": "PALM-AA (accelerated)",
    "not_applicable": "—",
}
PREPROCESSING_LABELS = {
    "P0_raw": "no feature adjustment",
    "P1_tran_alpha_0p005": "Tran thresholding",
    "P2_ke_weighted": "Ke weighting",
    "P3_tran_then_ke": "Tran thresholding + Ke weighting",
    "native_baseline": "native baseline implementation",
}
A_LABELS = {
    "A_current": "least-squares A",
    "A_full_Pois": "Poisson-refit A (full counts)",
    "native_baseline": "native baseline implementation",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _load_completed(result_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    complete = [
        path
        for path in result_root.rglob("complete.json")
        if "/full_" in path.as_posix()
    ]
    if not complete:
        raise FileNotFoundError(f"no completed full tasks under {result_root}")
    fit_files = [path.parent / "fit_rows.csv" for path in complete]
    missing = [path for path in fit_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fit tables: {missing[:3]}")
    frame = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in fit_files],
        ignore_index=True,
    )
    return frame, complete


def _canonical_dataset(value: str) -> str:
    return {
        "crc": "stanford_crc_codex",
        "spleen": "mouse_spleen_codex",
        "cook": "whats_cooking",
    }[value]


def _load_plan_tasks() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in PLAN_CONFIGS:
        specification = json.loads(path.read_text())
        dataset = _canonical_dataset(str(specification["dataset"]))
        group = str(specification.get("group") or "all")
        configured_seeds = [int(seed) for seed in specification["seeds"]]
        default_seeds = {
            int(seed)
            for seed in specification.get("default_seeds", configured_seeds)
        }
        seeds_by_estimator = {
            str(estimator): {int(seed) for seed in seeds}
            for estimator, seeds in specification.get(
                "seeds_by_estimator_family", {}
            ).items()
        }
        default_required_shards = tuple(
            str(shard)
            for shard in (specification.get("execution_shards") or ["all"])
        )
        shards_by_K = specification.get("execution_shards_by_K", {})
        shards_by_seed = specification.get(
            "required_execution_shards_by_seed", {}
        )
        for K, seed in product(specification["K_values"], configured_seeds):
            planned_estimators = (
                ("*",)
                if int(seed) in default_seeds
                else tuple(
                    sorted(
                        estimator
                        for estimator, estimator_seeds in seeds_by_estimator.items()
                        if int(seed) in estimator_seeds
                    )
                )
            )
            if not planned_estimators:
                continue
            required_shards = tuple(
                str(shard)
                for shard in shards_by_seed.get(
                    str(seed),
                    shards_by_K.get(str(K), default_required_shards),
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "biological_group": group,
                    "summary_group": group,
                    "K": int(K),
                    "seed": int(seed),
                    "required_execution_shards": required_shards,
                    "planned_estimator_families": planned_estimators,
                    "planned_geometries": tuple(
                        str(geometry)
                        for geometry in specification.get(
                            "geometries", ["document_U", "word_Z"]
                        )
                    ),
                }
            )
    tasks = pd.DataFrame(rows)
    pooled = tasks[
        (tasks["dataset"] == "mouse_spleen_codex")
        & tasks["biological_group"].isin(INDIVIDUAL_SPLEEN_GROUPS)
    ].copy()
    pooled["summary_group"] = "pooled"
    return pd.concat([tasks, pooled], ignore_index=True)


def _variant_id(values: dict[str, Any] | pd.Series) -> str:
    return "__".join(str(values[key]) for key in PIPELINE_KEYS[2:])


def _pipeline_label(values: dict[str, Any] | pd.Series) -> str:
    estimator = ESTIMATOR_LABELS.get(
        str(values["estimator_family"]), str(values["estimator_family"])
    )
    if str(values["estimator_family"]) == "plsi":
        return " · ".join(
            [
                estimator,
                A_LABELS.get(str(values["A_recovery"]), str(values["A_recovery"])),
            ]
        )
    if str(values["estimator_family"]) not in {
        "document_gplsi",
        "anchor_feature_gplsi",
    }:
        return estimator
    return " · ".join(
        [
            estimator,
            VERTEX_LABELS.get(str(values["vertex_hunter"]), str(values["vertex_hunter"])),
            PREPROCESSING_LABELS.get(
                str(values["preprocessing"]), str(values["preprocessing"])
            ),
            A_LABELS.get(str(values["A_recovery"]), str(values["A_recovery"])),
        ]
    )


def _planned_configurations(tasks: pd.DataFrame) -> pd.DataFrame:
    baseline_specs = (
        ("plsi", "document_U", "not_applicable", "native_baseline", "A_current", "native_baseline"),
        ("plsi", "document_U", "not_applicable", "native_baseline", "A_full_Pois", "native_baseline"),
        ("topicscore_raw", "topicscore_ratio_raw", "not_applicable", "native_baseline", "native_baseline", "native_baseline"),
        ("topicscore_graph_denoised", "topicscore_ratio_graph", "not_applicable", "P0_raw", "native_baseline", "native_baseline"),
        ("lda", "not_applicable", "not_applicable", "native_baseline", "native_baseline", "native_baseline"),
        ("spatial_lda", "not_applicable", "not_applicable", "native_baseline", "native_baseline", "native_baseline"),
        ("published_baseline", "not_applicable", "not_applicable", "native_baseline", "native_baseline", "native_baseline"),
    )
    rows: list[dict[str, Any]] = []
    for (dataset, summary_group, K), task_group in tasks.groupby(
        ["dataset", "summary_group", "K"], dropna=False
    ):
        planned_families = (
            task_group["planned_estimator_families"]
            if "planned_estimator_families" in task_group
            else pd.Series([("*",)] * len(task_group), index=task_group.index)
        )

        planned_geometries = (
            task_group["planned_geometries"]
            if "planned_geometries" in task_group
            else pd.Series(
                [("document_U", "word_Z")] * len(task_group),
                index=task_group.index,
            )
        )

        def planned_n(estimator: str, geometry: str | None = None) -> int:
            return int(
                sum(
                    ("*" in families or estimator in families)
                    and (geometry is None or geometry in geometries)
                    for families, geometries in zip(
                        planned_families, planned_geometries
                    )
                )
            )

        base_prefix = {
            "dataset": dataset,
            "summary_group": summary_group,
            "K": int(K),
        }
        for estimator, geometry, W_recovery in GPLSI_ESTIMATORS:
            estimator_planned_n = planned_n(estimator, geometry)
            if estimator_planned_n == 0:
                continue
            prefix = {**base_prefix, "planned_n": estimator_planned_n}
            for hunter, preprocessing, A_recovery in product(
                VERTEX_HUNTERS, PREPROCESSINGS, A_RECOVERIES
            ):
                rows.append(
                    {
                        **prefix,
                        "estimator_family": estimator,
                        "spectral_geometry": geometry,
                        "vertex_hunter": hunter,
                        "preprocessing": preprocessing,
                        "A_recovery": A_recovery,
                        "W_recovery_method": W_recovery,
                        "graph_svd_version": "iterative_two_step",
                    }
                )
        for estimator, geometry, hunter, preprocessing, A_recovery, W_recovery in baseline_specs:
            prefix = {**base_prefix, "planned_n": planned_n(estimator)}
            rows.append(
                {
                    **prefix,
                    "estimator_family": estimator,
                    "spectral_geometry": geometry,
                    "vertex_hunter": hunter,
                    "preprocessing": preprocessing,
                    "A_recovery": A_recovery,
                    "W_recovery_method": W_recovery,
                    "graph_svd_version": (
                        "iterative_two_step"
                        if estimator == "topicscore_graph_denoised"
                        else "not_applicable"
                    ),
                }
            )
    planned = pd.DataFrame(rows)
    planned["variant_id"] = planned.apply(_variant_id, axis=1)
    planned["variant_label"] = planned.apply(_pipeline_label, axis=1)
    planned["method_description"] = planned["estimator_family"].map(
        METHOD_DESCRIPTIONS
    )
    return planned


def _with_summary_groups(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = frame.copy()
    expanded["summary_group"] = expanded["biological_group"].fillna("all").astype(str)
    pooled = expanded[
        (expanded["dataset"] == "mouse_spleen_codex")
        & expanded["summary_group"].isin(INDIVIDUAL_SPLEEN_GROUPS)
    ].copy()
    pooled["summary_group"] = "pooled"
    return pd.concat([expanded, pooled], ignore_index=True)


def _aggregate_configurations(frame: pd.DataFrame, planned: pd.DataFrame) -> pd.DataFrame:
    expanded = _with_summary_groups(frame)
    status_counts = (
        expanded.groupby(CONFIG_KEYS + ["status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    status_columns = [
        "ok",
        "failed",
        "not_available",
        "not_applicable",
        "published_artifact_only",
    ]
    for column in status_columns:
        if column not in status_counts:
            status_counts[column] = 0
    status_counts = status_counts.rename(
        columns={
            "ok": "n_ok",
            "failed": "n_failed",
            "not_available": "n_not_available",
            "not_applicable": "n_not_applicable",
            "published_artifact_only": "n_artifact_only",
        }
    )
    status_counts["n_complete"] = status_counts[
        [
            "n_ok",
            "n_failed",
            "n_not_available",
            "n_not_applicable",
            "n_artifact_only",
        ]
    ].sum(axis=1)

    ok = expanded[
        (expanded["status"] == "ok")
        & expanded[METRIC].notna()
        & expanded[COUNT].notna()
        & (expanded[COUNT] > 0)
    ].copy()
    ok["dev_per_count"] = ok[METRIC] / ok[COUNT]
    metric_summary = (
        ok.groupby(CONFIG_KEYS, dropna=False)["dev_per_count"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )
    failures = expanded[expanded["status"] == "failed"].copy()
    if failures.empty:
        failure_summary = pd.DataFrame(columns=CONFIG_KEYS + ["failure_reason"])
    else:
        failure_summary = (
            failures.groupby(CONFIG_KEYS, dropna=False)["failure_reason"]
            .agg(
                lambda values: " | ".join(
                    f"{reason} ({count})"
                    for reason, count in values.fillna("unspecified").value_counts().head(3).items()
                )
            )
            .reset_index()
        )

    summary = (
        planned.merge(status_counts, on=CONFIG_KEYS, how="left")
        .merge(metric_summary, on=CONFIG_KEYS, how="left")
        .merge(failure_summary, on=CONFIG_KEYS, how="left")
    )
    count_columns = [
        "n_ok",
        "n_failed",
        "n_not_available",
        "n_not_applicable",
        "n_artifact_only",
        "n_complete",
    ]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    summary["n_pending"] = (summary["planned_n"] - summary["n_complete"]).clip(lower=0)
    summary["coverage"] = summary["n_ok"] / summary["planned_n"]
    summary["task_coverage"] = summary["n_complete"] / summary["planned_n"]
    summary["full_coverage"] = summary["n_ok"] == summary["planned_n"]
    return summary.sort_values(
        ["dataset", "summary_group", "K", "estimator_family", "vertex_hunter", "preprocessing", "A_recovery"]
    )


def _matching_rows(frame: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key in PIPELINE_KEYS:
        value = config[key]
        if pd.isna(value):
            mask &= frame[key].isna().to_numpy()
        else:
            mask &= frame[key].eq(value).to_numpy()
    if str(config["dataset"]) == "mouse_spleen_codex":
        biological_groups = frame["biological_group"].fillna("all").astype(str)
        summary_group = str(config["summary_group"])
        if summary_group == "pooled":
            mask &= biological_groups.isin(INDIVIDUAL_SPLEEN_GROUPS).to_numpy()
        else:
            mask &= biological_groups.eq(summary_group).to_numpy()
    return frame.loc[mask]


def _parse_literal(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return default


def _selection_basis(estimator: str, summary_group: str) -> str:
    if estimator in {"document_gplsi", "anchor_feature_gplsi"}:
        if summary_group == "joint":
            return (
                "exact fitted configuration from the joint all-three-spleen run; "
                "representative seed nearest this configuration's median held-out "
                "deviance when repeated"
            )
        if summary_group == "pooled":
            return (
                "exact fitted configuration from the three separate spleen runs; "
                "within each biological replicate, representative seed nearest "
                "this configuration's median held-out deviance"
            )
        return (
            "exact fitted configuration; representative seed nearest this "
            "configuration's median held-out deviance"
        )

    paired = estimator == "plsi"
    representative = (
        "representative task shared across both pLSI A recoveries"
        if paired
        else "representative seed nearest the joint-fit median"
        if summary_group == "joint"
        else "representative seed nearest its biological-replicate median"
        if summary_group == "pooled"
        else "representative seed nearest the configuration median"
    )
    if summary_group == "joint":
        return (
            "fixed competitor configuration summarized across the joint spleen fit; "
            f"{representative}"
        )
    if summary_group == "pooled":
        return (
            "fixed competitor configuration summarized across all three spleen "
            f"replicates; {representative}"
        )
    return f"exact competitor configuration; {representative}"


def _select_representative_fits(
    frame: pd.DataFrame, summary: pd.DataFrame
) -> list[dict[str, Any]]:
    ok = frame[
        (frame["status"] == "ok")
        & frame[METRIC].notna()
        & frame[COUNT].notna()
        & (frame[COUNT] > 0)
        & frame["artifacts.estimate.path"].notna()
    ].copy()
    ok["dev_per_count"] = ok[METRIC] / ok[COUNT]
    selections: list[dict[str, Any]] = []
    summary_scope = (
        (
            (summary["dataset"] == "mouse_spleen_codex")
            & summary["summary_group"].isin(["pooled", "joint"])
        )
        | (
            (summary["dataset"] != "mouse_spleen_codex")
            & (summary["summary_group"] == "all")
        )
    )
    # Topic Explorer is an inventory, not a model-selection surface. Retain
    # one representative seed for every exact completed configuration instead
    # of only the best preprocessing/A-recovery in each GpLSI hunter branch.
    leaders = summary[
        (summary["estimator_family"] != "published_baseline")
        & (summary["n_ok"] > 0)
        & summary_scope
    ].copy()
    leaders = leaders.sort_values(
        ["dataset", "summary_group", "K", "estimator_family", "variant_id"]
    )
    plsi_representative_tasks: dict[tuple[str, int, str], str] = {}
    for _, leader in leaders.iterrows():
        matching = _matching_rows(ok, leader)
        groups = (
            sorted(matching["biological_group"].dropna().astype(str).unique())
            if leader["dataset"] == "mouse_spleen_codex"
            else [None]
        )
        for group in groups:
            group_rows = matching
            if group is not None:
                group_rows = group_rows[
                    group_rows["biological_group"].astype(str) == group
                ]
            if group_rows.empty:
                continue
            center = group_rows["dev_per_count"].median()
            plsi_key = (str(leader["dataset"]), int(leader["K"]), str(group or "all"))
            paired_task = plsi_representative_tasks.get(plsi_key)
            paired_rows = (
                group_rows[group_rows["task_config_hash"].astype(str) == paired_task]
                if str(leader["estimator_family"]) == "plsi" and paired_task
                else pd.DataFrame()
            )
            if not paired_rows.empty:
                chosen = paired_rows.iloc[0]
            else:
                chosen = group_rows.loc[
                    (group_rows["dev_per_count"] - center).abs().idxmin()
                ]
                if str(leader["estimator_family"]) == "plsi":
                    plsi_representative_tasks[plsi_key] = str(
                        chosen["task_config_hash"]
                    )
            artifact = str(chosen["artifacts.estimate.path"])
            marker = "results/real_data_anchor_word_gplsi/"
            if marker not in artifact:
                raise ValueError(f"unexpected estimate path: {artifact}")
            relative_artifact = marker + artifact.split(marker, 1)[1]
            variant_id = str(leader["variant_id"])
            selections.append(
                {
                    "id": "::".join(
                        [
                            "topic",
                            str(chosen["dataset"]),
                            str(int(chosen["K"])),
                            str(group or "all"),
                            variant_id,
                        ]
                    ),
                    "variant_id": variant_id,
                    "variant_label": str(leader["variant_label"]),
                    "method_description": METHOD_DESCRIPTIONS.get(
                        str(chosen["estimator_family"]), ""
                    ),
                    "is_competitor": str(chosen["estimator_family"])
                    not in {"document_gplsi", "anchor_feature_gplsi"},
                    "dataset": str(chosen["dataset"]),
                    "dataset_label": DATASET_LABELS[str(chosen["dataset"])],
                    "group": group,
                    "summary_group": str(leader["summary_group"]),
                    "K": int(chosen["K"]),
                    "seed": int(chosen["seed"]),
                    "fit_id": str(chosen["fit_id"]),
                    "estimator_family": str(chosen["estimator_family"]),
                    "spectral_geometry": str(chosen["spectral_geometry"]),
                    "vertex_hunter": str(chosen["vertex_hunter"]),
                    "preprocessing": str(chosen["preprocessing"]),
                    "A_recovery": str(chosen["A_recovery"]),
                    "W_recovery_method": str(chosen["W_recovery_method"]),
                    "dev_per_count": float(chosen["dev_per_count"]),
                    "configuration_mean_dev_per_count": float(leader["mean"]),
                    "configuration_median_dev_per_count": float(leader["median"]),
                    "configuration_n": int(leader["n_ok"]),
                    "configuration_planned_n": int(leader["planned_n"]),
                    "configuration_full_coverage": bool(leader["full_coverage"]),
                    "representative_group_mean_dev_per_count": float(
                        group_rows["dev_per_count"].mean()
                    ),
                    "representative_group_median_dev_per_count": float(center),
                    "representative_group_n": int(len(group_rows)),
                    "graph_svd_version": str(
                        chosen.get("graph_svd_version", "not_recorded")
                    ),
                    "selection_basis": _selection_basis(
                        str(chosen["estimator_family"]),
                        str(leader["summary_group"]),
                    ),
                    "anchor_feature_candidates": _parse_literal(
                        chosen.get("selected_anchor_feature_candidates"), []
                    ),
                    "selected_feature_names": _parse_literal(
                        chosen.get("selected_feature_names"), []
                    ),
                    "artifact": relative_artifact,
                }
            )
    return selections


def _logical_task_status(
    frame: pd.DataFrame, planned_tasks: pd.DataFrame
) -> pd.DataFrame:
    """Mark a planned K/seed task complete only after all of its shards finish."""

    task_keys = ["dataset", "biological_group", "K", "seed"]
    observed_columns = task_keys.copy()
    if "execution_shard" in frame.columns:
        observed_columns.append("execution_shard")
    observed = frame[observed_columns].copy()
    observed["biological_group"] = (
        observed["biological_group"].fillna("all").astype(str)
    )
    if "execution_shard" not in observed:
        observed["execution_shard"] = "all"
    else:
        observed["execution_shard"] = (
            observed["execution_shard"].fillna("all").astype(str)
        )
    shard_sets = (
        observed.drop_duplicates(task_keys + ["execution_shard"])
        .groupby(task_keys, dropna=False)["execution_shard"]
        .agg(frozenset)
        .to_dict()
    )

    tasks = planned_tasks.copy()
    tasks["biological_group"] = tasks["biological_group"].fillna("all").astype(str)
    if "required_execution_shards" not in tasks:
        tasks["required_execution_shards"] = [("all",)] * len(tasks)

    def task_is_complete(row: pd.Series) -> bool:
        key = tuple(row[column] for column in task_keys)
        configured = row["required_execution_shards"]
        required = frozenset([configured] if isinstance(configured, str) else configured)
        return required.issubset(shard_sets.get(key, frozenset()))

    tasks["task_complete"] = tasks.apply(task_is_complete, axis=1)
    return tasks


def _task_coverage(frame: pd.DataFrame, planned_tasks: pd.DataFrame) -> pd.DataFrame:
    logical_tasks = _logical_task_status(frame, planned_tasks)
    coverage = (
        logical_tasks.groupby(["dataset", "summary_group", "K"], dropna=False)
        .agg(
            planned_tasks=("seed", "size"),
            complete_tasks=("task_complete", "sum"),
        )
        .reset_index()
    )
    coverage["complete_tasks"] = coverage["complete_tasks"].astype(int)
    coverage["pending_tasks"] = (
        coverage["planned_tasks"] - coverage["complete_tasks"]
    ).clip(lower=0)
    coverage["state"] = np.select(
        [
            coverage["complete_tasks"] == 0,
            coverage["pending_tasks"] == 0,
        ],
        ["pending", "complete"],
        default="partial",
    )
    return coverage.sort_values(["dataset", "summary_group", "K"])


def _load_crc_outcome_summary(
    result_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(
        {
            *result_root.rglob("crc_outcome_rows*.csv"),
            *result_root.rglob("crc_cell_type_outcome_rows*.csv"),
        }
    )
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            outcome_frame = pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if not outcome_frame.empty:
            frames.append(outcome_frame)
    caveat = (
        "There are no ground-truth topic labels. These are downstream CRC "
        "outcome-prediction metrics, not topic-recovery accuracy; patient-level "
        "mapping is unavailable."
    )
    if not frames:
        return pd.DataFrame(), {
            "status": "pending",
            "unit": "CRC region",
            "evaluation_scope": "transductive unsupervised topic fit with region-level repeated stratified cross-validation",
            "caveat": caveat,
            "evaluated_tasks": 0,
            "planned_tasks": 60,
            "evaluated_W_fits": 0,
            "planned_W_fits": 2400,
            "evaluated_variant_cells": 0,
            "planned_variant_cells": 240,
            "A_recovery_role": "not used; A_current and A_full_Pois share W and are collapsed",
            "cell_type_baseline_status": "pending",
            "cell_type_baseline_K_values": [],
            "cell_type_baseline_predictor_dimension": None,
            "required_predictor_transforms": ["ilr", "log"],
            "predictor_transforms": [],
            "required_classifiers": ["ridge_logistic", "random_forest"],
            "classifiers": [],
            "status_by_transform": {"ilr": "pending", "log": "pending"},
            "evaluated_tasks_by_transform": {"ilr": 0, "log": 0},
            "evaluated_W_fits_by_transform": {"ilr": 0, "log": 0},
            "cell_type_baseline_status_by_transform": {
                "ilr": "pending",
                "log": "pending",
            },
            "cell_type_baseline_K_values_by_transform": {"ilr": [], "log": []},
            "cell_type_baseline_composition_dimension": None,
            "cell_type_baseline_predictor_dimension_by_transform": {},
            "status_by_evaluation": {
                "ilr::ridge_logistic": "pending",
                "ilr::random_forest": "pending",
                "log::ridge_logistic": "pending",
                "log::random_forest": "pending",
            },
            "evaluated_tasks_by_evaluation": {
                "ilr::ridge_logistic": 0,
                "ilr::random_forest": 0,
                "log::ridge_logistic": 0,
                "log::random_forest": 0,
            },
            "evaluated_W_fits_by_evaluation": {
                "ilr::ridge_logistic": 0,
                "ilr::random_forest": 0,
                "log::ridge_logistic": 0,
                "log::random_forest": 0,
            },
            "cell_type_baseline_status_by_evaluation": {
                "ilr::ridge_logistic": "pending",
                "ilr::random_forest": "pending",
                "log::ridge_logistic": "pending",
                "log::random_forest": "pending",
            },
        }

    outcomes = pd.concat(frames, ignore_index=True)
    inferred_transform = np.where(
        outcomes.get("evaluation_version", pd.Series("", index=outcomes.index))
        .astype(str)
        .str.contains("_log_"),
        "log",
        "ilr",
    )
    if "predictor_transform" not in outcomes:
        outcomes["predictor_transform"] = inferred_transform
    else:
        present = outcomes["predictor_transform"].notna() & outcomes[
            "predictor_transform"
        ].astype(str).ne("")
        outcomes["predictor_transform"] = outcomes["predictor_transform"].where(
            present, inferred_transform
        )
    inferred_classifier = np.where(
        outcomes.get("evaluation_version", pd.Series("", index=outcomes.index))
        .astype(str)
        .str.contains("_rf"),
        "random_forest",
        "ridge_logistic",
    )
    if "classifier" not in outcomes:
        outcomes["classifier"] = inferred_classifier
    else:
        present = outcomes["classifier"].notna() & outcomes["classifier"].astype(
            str
        ).ne("")
        outcomes["classifier"] = outcomes["classifier"].where(
            present, inferred_classifier
        )
    if "evaluated_at_utc" in outcomes:
        outcomes = outcomes.sort_values("evaluated_at_utc", kind="stable")
    outcomes = outcomes.drop_duplicates(
        [
            column
            for column in [
                "task_config_hash",
                "W_fit_id",
                "K",
                "target",
                "representation",
                "predictor_transform",
                "classifier",
            ]
            if column in outcomes
        ],
        keep="last",
    )
    outcome_keys = [
        "dataset",
        "K",
        "estimator_family",
        "spectral_geometry",
        "vertex_hunter",
        "preprocessing",
        "W_recovery_method",
        "target",
        "representation",
        "predictor_transform",
        "classifier",
    ]
    metrics = [
        "roc_auc_mean",
        "pr_auc_mean",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "f1_mean",
        "brier_mean",
    ]
    named_aggregations: dict[str, tuple[str, str]] = {
        "n_complete": ("task_config_hash", "nunique"),
        "region_count": ("region_count", "max"),
        "positive_count": ("positive_count", "max"),
    }
    if "predictor_dimension" in outcomes:
        named_aggregations["predictor_dimension"] = (
            "predictor_dimension",
            "max",
        )
    if "composition_dimension" in outcomes:
        named_aggregations["composition_dimension"] = (
            "composition_dimension",
            "max",
        )
    for metric in metrics:
        if metric in outcomes:
            named_aggregations[metric] = (metric, "mean")
            named_aggregations[f"{metric}_between_fit_std"] = (metric, "std")
    summary = (
        outcomes.groupby(outcome_keys, dropna=False)
        .agg(**named_aggregations)
        .reset_index()
    )
    summary["majority_accuracy"] = np.maximum(
        summary["positive_count"],
        summary["region_count"] - summary["positive_count"],
    ) / summary["region_count"]
    summary["A_recovery"] = "not_used_for_outcome"
    summary["planned_n"] = np.where(
        summary["estimator_family"] == "cell_type_baseline", 1, 10
    ).astype(int)
    summary["coverage"] = summary["n_complete"] / summary["planned_n"]
    summary["variant_id"] = summary.apply(
        lambda row: "__".join(
            str(row[key])
            for key in [
                "estimator_family",
                "spectral_geometry",
                "vertex_hunter",
                "preprocessing",
                "W_recovery_method",
            ]
        )
        + "__W_shared",
        axis=1,
    )
    def outcome_label(row: pd.Series) -> str:
        estimator = str(row["estimator_family"])
        estimator_label = ESTIMATOR_LABELS.get(estimator, estimator)
        if estimator == "cell_type_baseline":
            return estimator_label
        if estimator not in {"document_gplsi", "anchor_feature_gplsi"}:
            return f"{estimator_label} · shared W (A not used)"
        return " · ".join(
            [
                estimator_label,
                VERTEX_LABELS.get(
                    str(row["vertex_hunter"]), str(row["vertex_hunter"])
                ),
                PREPROCESSING_LABELS.get(
                    str(row["preprocessing"]), str(row["preprocessing"])
                ),
                "shared W (A not used)",
            ]
        )

    summary["variant_label"] = summary.apply(outcome_label, axis=1)
    learned_outcomes = outcomes[
        outcomes["estimator_family"] != "cell_type_baseline"
    ]
    required_transforms = ("ilr", "log")
    required_classifiers = ("ridge_logistic", "random_forest")
    evaluation_keys = [
        (transform, classifier)
        for transform in required_transforms
        for classifier in required_classifiers
    ]
    evaluated_tasks_by_evaluation = {
        f"{transform}::{classifier}": int(
            learned_outcomes.loc[
                (learned_outcomes["predictor_transform"] == transform)
                & (learned_outcomes["classifier"] == classifier),
                "task_config_hash",
            ].nunique()
        )
        for transform, classifier in evaluation_keys
    }
    status_by_evaluation = {
        key: (
            "complete"
            if count >= 60
            else "partial"
            if count > 0
            else "pending"
        )
        for key, count in evaluated_tasks_by_evaluation.items()
    }
    evaluated_tasks_by_transform = {
        transform: int(
            learned_outcomes.loc[
                learned_outcomes["predictor_transform"] == transform,
                "task_config_hash",
            ].nunique()
        )
        for transform in required_transforms
    }
    status_by_transform = {}
    for transform in required_transforms:
        states = [
            status_by_evaluation[f"{transform}::{classifier}"]
            for classifier in required_classifiers
        ]
        status_by_transform[transform] = (
            "complete"
            if all(state == "complete" for state in states)
            else "partial"
            if any(state != "pending" for state in states)
            else "pending"
        )
    evaluated_tasks = int(learned_outcomes["task_config_hash"].nunique())
    gplsi_outcomes = outcomes[
        outcomes["estimator_family"].isin(
            ["document_gplsi", "anchor_feature_gplsi"]
        )
    ]
    evaluated_W_fits = int(
        len(gplsi_outcomes[["task_config_hash", "W_fit_id"]].drop_duplicates())
    )
    evaluated_W_fits_by_transform = {
        transform: int(
            len(
                gplsi_outcomes.loc[
                    gplsi_outcomes["predictor_transform"] == transform,
                    ["task_config_hash", "W_fit_id"],
                ].drop_duplicates()
            )
        )
        for transform in required_transforms
    }
    evaluated_W_fits_by_evaluation = {
        f"{transform}::{classifier}": int(
            len(
                gplsi_outcomes.loc[
                    (gplsi_outcomes["predictor_transform"] == transform)
                    & (gplsi_outcomes["classifier"] == classifier),
                    ["task_config_hash", "W_fit_id"],
                ].drop_duplicates()
            )
        )
        for transform, classifier in evaluation_keys
    }
    status_by_evaluation = {
        key: (
            "complete"
            if evaluated_tasks_by_evaluation[key] >= 60 and count >= 2400
            else "partial"
            if evaluated_tasks_by_evaluation[key] > 0 or count > 0
            else "pending"
        )
        for key, count in evaluated_W_fits_by_evaluation.items()
    }
    for transform in required_transforms:
        states = [
            status_by_evaluation[f"{transform}::{classifier}"]
            for classifier in required_classifiers
        ]
        status_by_transform[transform] = (
            "complete"
            if all(state == "complete" for state in states)
            else "partial"
            if any(state != "pending" for state in states)
            else "pending"
        )
    gplsi_summary = summary[
        summary["estimator_family"].isin(
            ["document_gplsi", "anchor_feature_gplsi"]
        )
    ]
    evaluated_variant_cells = int(
        len(gplsi_summary[["K", "variant_id"]].drop_duplicates())
    )
    cell_type_outcomes = outcomes[
        outcomes["estimator_family"] == "cell_type_baseline"
    ]
    cell_type_K_values = sorted(
        map(int, cell_type_outcomes["K"].dropna().unique())
    )
    cell_type_K_values_by_transform = {
        transform: sorted(
            map(
                int,
                cell_type_outcomes.loc[
                    cell_type_outcomes["predictor_transform"] == transform, "K"
                ]
                .dropna()
                .unique(),
            )
        )
        for transform in required_transforms
    }
    cell_type_K_values_by_evaluation = {
        f"{transform}::{classifier}": sorted(
            map(
                int,
                cell_type_outcomes.loc[
                    (cell_type_outcomes["predictor_transform"] == transform)
                    & (cell_type_outcomes["classifier"] == classifier),
                    "K",
                ]
                .dropna()
                .unique(),
            )
        )
        for transform, classifier in evaluation_keys
    }
    cell_type_status_by_evaluation = {
        key: (
            "complete"
            if set(values) >= {1, 2, 3, 4, 5, 6}
            else "partial"
            if values
            else "pending"
        )
        for key, values in cell_type_K_values_by_evaluation.items()
    }
    cell_type_status_by_transform = {}
    for transform in required_transforms:
        states = [
            cell_type_status_by_evaluation[f"{transform}::{classifier}"]
            for classifier in required_classifiers
        ]
        cell_type_status_by_transform[transform] = (
            "complete"
            if all(state == "complete" for state in states)
            else "partial"
            if any(state != "pending" for state in states)
            else "pending"
        )
    cell_type_status = (
        "complete"
        if all(value == "complete" for value in cell_type_status_by_transform.values())
        else "partial"
        if any(value != "pending" for value in cell_type_status_by_transform.values())
        else "pending"
    )
    composition_dimensions = (
        pd.to_numeric(
            cell_type_outcomes.get("composition_dimension", pd.Series(dtype=float)),
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )
    predictor_dimension_by_transform: dict[str, int | None] = {}
    for transform in required_transforms:
        dimensions = (
            pd.to_numeric(
                cell_type_outcomes.loc[
                    cell_type_outcomes["predictor_transform"] == transform
                ].get("predictor_dimension", pd.Series(dtype=float)),
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )
        predictor_dimension_by_transform[transform] = (
            int(dimensions[0]) if len(dimensions) == 1 else None
        )
    composition_dimension = (
        int(composition_dimensions[0]) if len(composition_dimensions) == 1 else None
    )
    if composition_dimension is None:
        legacy_dimensions = [
            value
            for value in predictor_dimension_by_transform.values()
            if value is not None
        ]
        composition_dimension = max(legacy_dimensions, default=None)
    return summary, {
        "status": (
            "complete"
            if all(value == "complete" for value in status_by_evaluation.values())
            else "pending"
            if all(value == "pending" for value in status_by_evaluation.values())
            else "partial"
        ),
        "unit": "CRC region",
        "evaluation_scope": "transductive unsupervised topic fit with region-level repeated stratified 5-fold cross-validation (5 repeats)",
        "caveat": caveat,
        "evaluated_tasks": evaluated_tasks,
        "planned_tasks": 60,
        "evaluated_W_fits": evaluated_W_fits,
        "planned_W_fits": 2400,
        "evaluated_variant_cells": evaluated_variant_cells,
        "planned_variant_cells": 240,
        "A_recovery_role": "not used; A_current and A_full_Pois share W and are collapsed",
        "targets": ["primary_outcome", "recurrence"],
        "representations": ["soft_mean", "published_hard"],
        "required_predictor_transforms": list(required_transforms),
        "predictor_transforms": sorted(
            map(str, outcomes["predictor_transform"].dropna().unique())
        ),
        "transform_definitions": {
            "ilr": "ILR coordinates after a 1e-8 zero floor and re-closure",
            "log": "Elementwise natural log after a 1e-8 zero floor and re-closure; no log-ratio projection",
        },
        "required_classifiers": list(required_classifiers),
        "classifiers": sorted(map(str, outcomes["classifier"].dropna().unique())),
        "classifier_definitions": {
            "ridge_logistic": "Fold-standardized L2 logistic regression (C=1, liblinear)",
            "random_forest": "Untuned 100-tree random forest (Gini, sqrt features, unweighted classes, no scaling)",
        },
        "status_by_transform": status_by_transform,
        "evaluated_tasks_by_transform": evaluated_tasks_by_transform,
        "evaluated_W_fits_by_transform": evaluated_W_fits_by_transform,
        "status_by_evaluation": status_by_evaluation,
        "evaluated_tasks_by_evaluation": evaluated_tasks_by_evaluation,
        "evaluated_W_fits_by_evaluation": evaluated_W_fits_by_evaluation,
        "cell_type_baseline_status": cell_type_status,
        "cell_type_baseline_K_values": cell_type_K_values,
        "cell_type_baseline_predictor_dimension": composition_dimension,
        "cell_type_baseline_status_by_transform": cell_type_status_by_transform,
        "cell_type_baseline_K_values_by_transform": cell_type_K_values_by_transform,
        "cell_type_baseline_composition_dimension": composition_dimension,
        "cell_type_baseline_predictor_dimension_by_transform": predictor_dimension_by_transform,
        "cell_type_baseline_status_by_evaluation": cell_type_status_by_evaluation,
        "cell_type_baseline_K_values_by_evaluation": cell_type_K_values_by_evaluation,
    }


def select_data(result_root: Path, output: Path) -> None:
    frame, complete_execution_units = _load_completed(result_root)
    planned_tasks = _load_plan_tasks()
    planned = _planned_configurations(planned_tasks)
    ok = frame[
        (frame["status"] == "ok")
        & frame[METRIC].notna()
        & frame[COUNT].notna()
        & (frame[COUNT] > 0)
        & frame["artifacts.estimate.path"].notna()
    ].copy()
    aggregate = _aggregate_configurations(frame, planned)
    failures = (
        frame[frame["status"] == "failed"]
        .groupby(["dataset", "K", "failure_reason"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["dataset", "K", "rows"], ascending=[True, True, False])
    )
    status = (
        frame.groupby(["dataset", "status"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    task_coverage = _task_coverage(frame, planned_tasks)
    outcome_summary, outcome_contract = _load_crc_outcome_summary(result_root)
    expected_tasks = int(
        len(planned_tasks[planned_tasks["summary_group"] != "pooled"])
    )
    complete_tasks = int(
        task_coverage.loc[
            task_coverage["summary_group"] != "pooled", "complete_tasks"
        ].sum()
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": {
            "expected_tasks": expected_tasks,
            "complete_tasks": complete_tasks,
            "complete_execution_units": len(complete_execution_units),
            "fit_rows": len(frame),
            "successful_fit_rows": len(ok),
        },
        "experiment_contract": {
            "metric": "held-out multinomial deviance per held-out count",
            "metric_direction": "lower_is_better",
            "graph_svd_version": "iterative_two_step",
            "production_gplsi_variants_per_dataset_K": 80,
            "joint_spleen_gplsi_variants_per_K": 40,
            "gplsi_factorization": "Default scope: 2 estimator geometries × 5 vertex hunters × 4 preprocessing variants × 2 A recoveries = 80. Joint spleen scope: document-U geometry only × 5 vertex hunters × 4 preprocessing variants × 2 A recoveries = 40.",
            "joint_spleen_seed_policy": "One count-thinning/model seed for every method except ordinary LDA and spatial LDA, which use three seeds.",
            "excluded_vertex_hunters": ["pp-SPA"],
            "representative_fit_policy": "Topic Explorer includes every exact completed estimator × geometry × vertex-hunter × preprocessing × A-recovery configuration. For repeated fits, it displays the seed nearest that exact configuration's median held-out deviance; pLSI A recoveries share the same representative W task.",
            "logical_task_policy": "A logical K × seed task is complete only when every execution shard declared by its plan config is complete; unsharded legacy configs require the single 'all' shard.",
        },
        "task_coverage": _records(task_coverage),
        "status": _records(status),
        "failure_summary": _records(failures),
        "configuration_summary": _records(aggregate),
        "crc_outcome_contract": outcome_contract,
        "crc_outcome_summary": _records(outcome_summary),
        "selections": _select_representative_fits(frame, aggregate),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n"
    )


def _symmetric_edges(bundle: Any) -> np.ndarray:
    endpoints = bundle.edge_df[["src", "tgt"]].to_numpy(dtype=np.int64)
    endpoints = endpoints[endpoints[:, 0] != endpoints[:, 1]]
    endpoints = np.sort(endpoints, axis=1)
    return np.unique(endpoints, axis=0)


def _spectral_coordinates(bundle: Any) -> np.ndarray:
    adjacency = bundle.weights.maximum(bundle.weights.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=float)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = diags(inv_sqrt) @ adjacency @ diags(inv_sqrt)
    try:
        _, vectors = eigsh(normalized, k=3, which="LA", tol=1e-4)
        coordinates = vectors[:, :2]
    except Exception:
        angle = np.linspace(0.0, 2.0 * np.pi, bundle.n, endpoint=False)
        coordinates = np.column_stack((np.cos(angle), np.sin(angle)))
    return coordinates


def _normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    low = np.nanmin(coordinates, axis=0)
    span = np.nanmax(coordinates, axis=0) - low
    span[span <= np.finfo(float).eps] = 1.0
    return (coordinates - low) / span


def _graph_geometry(
    bundle: Any,
    *,
    key: str,
    max_nodes: int = 1800,
    max_edges: int = 4500,
) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(260908)
    allowed = np.arange(bundle.n, dtype=np.int64)
    display_group: str | None = None
    if bundle.dataset == "stanford_crc_codex":
        counts = pd.Series(bundle.group_ids.astype(str)).value_counts()
        display_group = str(counts.index[0])
        allowed = np.flatnonzero(bundle.group_ids.astype(str) == display_group)
    allowed_mask = np.zeros(bundle.n, dtype=bool)
    allowed_mask[allowed] = True
    endpoints = _symmetric_edges(bundle)
    endpoints = endpoints[
        allowed_mask[endpoints[:, 0]] & allowed_mask[endpoints[:, 1]]
    ]
    if allowed.size <= max_nodes:
        selected = np.sort(allowed)
    else:
        order = rng.permutation(len(endpoints))
        chosen: list[int] = []
        seen: set[int] = set()
        for edge_index in order:
            for node in endpoints[edge_index]:
                value = int(node)
                if value not in seen:
                    seen.add(value)
                    chosen.append(value)
                    if len(chosen) >= max_nodes:
                        break
            if len(chosen) >= max_nodes:
                break
        if len(chosen) < max_nodes:
            remaining = np.setdiff1d(allowed, np.asarray(chosen), assume_unique=False)
            fill = rng.choice(
                remaining, size=min(max_nodes - len(chosen), remaining.size), replace=False
            )
            chosen.extend(map(int, fill))
        selected = np.sort(np.asarray(chosen[:max_nodes], dtype=np.int64))

    remap = np.full(bundle.n, -1, dtype=np.int64)
    remap[selected] = np.arange(selected.size)
    kept_edges = endpoints[
        (remap[endpoints[:, 0]] >= 0) & (remap[endpoints[:, 1]] >= 0)
    ]
    if len(kept_edges) > max_edges:
        kept_edges = kept_edges[
            np.sort(rng.choice(len(kept_edges), max_edges, replace=False))
        ]
    local_edges = np.column_stack(
        (remap[kept_edges[:, 0]], remap[kept_edges[:, 1]])
    )
    coordinates = (
        bundle.coordinates
        if bundle.coordinates is not None
        else _spectral_coordinates(bundle)
    )
    coordinates = _normalize_coordinates(coordinates[selected])
    geometry = {
        "key": key,
        "dataset": bundle.dataset,
        "display_group": display_group,
        "coordinate_type": "spatial" if bundle.coordinates is not None else "spectral_graph",
        "sampled_nodes": int(selected.size),
        "total_nodes": int(allowed.size),
        "sampled_edges": int(len(local_edges)),
        "x": np.round(coordinates[:, 0], 5).tolist(),
        "y": np.round(coordinates[:, 1], 5).tolist(),
        "labels": [str(value) for value in bundle.group_ids[selected]],
        "edges": local_edges.tolist(),
    }
    return geometry, selected


def _artifact_path(artifact_root: Path, relative: str) -> Path:
    candidate = artifact_root / relative
    if candidate.is_file():
        return candidate
    candidate = artifact_root / Path(relative).relative_to("results")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"selected artifact is missing: {relative}")


def _w_entropy_values(W: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return row-wise Shannon entropy in nats and normalized by log(K)."""

    probabilities = np.asarray(W, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError(f"W must be a nonempty matrix, received {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("W contains nonfinite values")
    if float(np.min(probabilities)) < -1e-10:
        raise ValueError("W contains materially negative values")
    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("W contains a row with zero total topic weight")
    probabilities = probabilities / row_sums
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    entropy_nats = -np.sum(probabilities * logs, axis=1)
    entropy_nats = np.maximum(entropy_nats, 0.0)
    if probabilities.shape[1] == 1:
        return entropy_nats, None
    normalized = np.clip(entropy_nats / np.log(probabilities.shape[1]), 0.0, 1.0)
    return entropy_nats, normalized


def _w_entropy_summary(W: np.ndarray) -> dict[str, Any]:
    entropy_nats, normalized = _w_entropy_values(W)
    return {
        "definition": "equal-weight mean of row-wise Shannon entropies",
        "normalization": "H(W_i) / log(K); 0 is single-topic and 1 is uniform across K",
        "scope": "full saved W for this representative fit; not averaged across seeds",
        "n_observations": int(W.shape[0]),
        "mean_nats": round(float(np.mean(entropy_nats)), 7),
        "median_nats": round(float(np.median(entropy_nats)), 7),
        "mean_normalized": (
            None if normalized is None else round(float(np.mean(normalized)), 7)
        ),
        "median_normalized": (
            None if normalized is None else round(float(np.median(normalized)), 7)
        ),
        "q25_normalized": (
            None if normalized is None else round(float(np.quantile(normalized, 0.25)), 7)
        ),
        "q75_normalized": (
            None if normalized is None else round(float(np.quantile(normalized, 0.75)), 7)
        ),
    }


def _cuisine_similarity_order(
    names: list[str], profiles: np.ndarray
) -> tuple[np.ndarray, str]:
    """Order cuisine simplex profiles so adjacent rows have similar mean W."""

    alphabetical = np.arange(len(names), dtype=np.int64)
    if len(names) <= 2 or profiles.shape[1] <= 1:
        return alphabetical, "alphabetical_degenerate"
    probabilities = np.maximum(np.asarray(profiles, dtype=float), 0.0)
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probabilities)) or np.any(totals <= 0):
        return alphabetical, "alphabetical_degenerate"
    probabilities = probabilities / totals
    distances = pdist(probabilities, metric="jensenshannon")
    if (
        not np.all(np.isfinite(distances))
        or not distances.size
        or float(np.max(distances)) <= 1e-12
    ):
        return alphabetical, "alphabetical_degenerate"
    tree = linkage(distances, method="average", optimal_ordering=True)
    order = leaves_list(tree).astype(np.int64, copy=False)
    forward = tuple(names[index] for index in order)
    reverse = tuple(reversed(forward))
    if reverse < forward:
        order = order[::-1]
    return order, "jensen_shannon_average_optimal_leaf"


def _group_topic_composition(bundle: Any, W: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if bundle.dataset == "stanford_crc_codex" and bundle.outcomes is not None:
        regions = pd.Series(bundle.group_ids.astype(str), name="region")
        W_frame = pd.DataFrame(W)
        W_frame["region"] = regions.to_numpy()
        region_W = W_frame.groupby("region", sort=True).mean()
        outcome_frame = bundle.outcomes.copy()
        outcome_frame["region"] = regions.to_numpy()
        region_outcomes = outcome_frame.groupby("region", sort=True).first()
        for target in ("primary_outcome", "recurrence"):
            if target not in region_outcomes:
                continue
            labels = region_outcomes[target]
            for level in sorted(labels.dropna().unique()):
                selected = labels.index[labels == level]
                topics = region_W.loc[selected].mean(axis=0).to_numpy(dtype=float)
                rows.append(
                    {
                        "kind": target,
                        "group": f"{target.replace('_', ' ')} = {int(level)}",
                        "n": int(len(selected)),
                        "topics": np.round(topics, 5).tolist(),
                    }
                )
        return rows

    labels = pd.Series(bundle.group_ids.astype(str))
    levels = sorted(labels.unique())
    masks = [labels.eq(level).to_numpy() for level in levels]
    profiles = np.vstack([W[mask].mean(axis=0) for mask in masks])
    _, normalized_entropy = _w_entropy_values(W)
    if bundle.dataset == "whats_cooking":
        order, ordering = _cuisine_similarity_order(levels, profiles)
    else:
        order = np.arange(len(levels), dtype=np.int64)
        ordering = "alphabetical"
    for display_rank, level_index in enumerate(order):
        level = levels[int(level_index)]
        mask = masks[int(level_index)]
        topics = profiles[int(level_index)]
        rows.append(
            {
                "kind": (
                    "biological_replicate"
                    if bundle.dataset == "mouse_spleen_codex"
                    else "cuisine"
                ),
                "group": str(level),
                "display_rank": int(display_rank),
                "ordering": ordering,
                "n": int(mask.sum()),
                "topics": np.round(topics, 5).tolist(),
                "mean_normalized_w_entropy": (
                    None
                    if normalized_entropy is None
                    else round(float(np.mean(normalized_entropy[mask])), 7)
                ),
            }
        )
    return rows


def _feature_candidates(
    selection: dict[str, Any], bundle: Any, A: np.ndarray
) -> tuple[str, str, list[dict[str, Any]]]:
    """Return honest, method-aware feature summaries for the topic explorer.

    Only the feature-side GpLSI path records candidates tied to its fitted
    vertices.  The archived competitor artifacts contain W and A but do not
    contain algorithm-selected anchor indices, so those methods receive a
    clearly labelled post-hoc defining feature instead.
    """

    estimator = str(selection["estimator_family"])
    recorded = selection.get("anchor_feature_candidates") or []
    if estimator == "anchor_feature_gplsi" and recorded:
        candidates: list[dict[str, Any]] = []
        for candidate in recorded:
            normalized = dict(candidate)
            normalized["role"] = "nearest fitted-vertex feature"
            normalized["provenance"] = "recorded during the feature-side GpLSI fit"
            candidates.append(normalized)
        return (
            "vertex_nearest_anchor_candidate",
            "Each feature is the observed feature nearest a fitted feature-side vertex. These are candidate anchors, not verified biological anchors.",
            candidates,
        )

    topic_score = estimator in {
        "topicscore_raw",
        "topicscore_graph_denoised",
    }
    candidates = []
    for topic in range(A.shape[0]):
        index = int(np.argmax(A[topic]))
        column_total = float(A[:, index].sum())
        specificity = (
            float(A[topic, index]) / column_total if column_total > 0 else np.nan
        )
        other_max = float(np.max(np.delete(A[:, index], topic))) if A.shape[0] > 1 else 0.0
        exclusivity = (
            float(A[topic, index]) / (float(A[topic, index]) + other_max)
            if float(A[topic, index]) + other_max > 0
            else np.nan
        )
        candidates.append(
            {
                "topic": topic,
                "feature_name": str(bundle.feature_ids[index]),
                "topic_weight": round(float(A[topic, index]), 7),
                "topic_specificity": round(specificity, 7),
                "topic_exclusivity": round(exclusivity, 7),
                "role": (
                    "post-hoc Topic-SCORE defining feature"
                    if topic_score
                    else "top-probability defining feature"
                ),
                "provenance": "derived post hoc from the saved A matrix",
            }
        )
    if topic_score:
        return (
            "posthoc_topicscore_defining_feature",
            "The saved Topic-SCORE artifacts do not retain their internal selected-word indices. These are transparent post-hoc defining features (the largest A entry per topic), not algorithm-recorded anchors.",
            candidates,
        )
    return (
        "top_probability_defining_feature",
        "This method does not define feature-side anchors. The cards show the largest-probability feature in each saved topic A as a comparable defining-feature summary.",
        candidates,
    )


def _document_candidates(
    selection: dict[str, Any],
    bundle: Any,
    W: np.ndarray,
    arrays: Any,
    *,
    candidates_per_topic: int = 3,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Return method-aware anchor or representative documents.

    Observation-side SPA records exact selected rows.  SVS and PALM fit
    centroids/archetypes that need not be observed rows, so their nearest
    observed documents are derived in the same recovered U geometry.  Methods
    without document-side vertex hunting receive an explicitly post-hoc,
    highest-W representative list.
    """

    estimator = str(selection["estimator_family"])
    hunter = str(selection.get("vertex_hunter") or "not_applicable")
    W = np.asarray(W, dtype=float)
    K = int(selection["K"])
    if W.shape != (bundle.n, K):
        raise ValueError(
            f"unexpected W shape {W.shape} while deriving document candidates"
        )

    def optional_array(name: str) -> np.ndarray | None:
        if name not in arrays.files:
            return None
        value = np.asarray(arrays[name])
        return value if value.size else None

    vertex_distances: np.ndarray | None = None
    exact_indices: np.ndarray | None = None
    archetype_weights: np.ndarray | None = None
    if estimator == "document_gplsi":
        vertices = optional_array("vertices")
        W_raw = optional_array("W_raw")
        if (
            vertices is not None
            and W_raw is not None
            and vertices.ndim == 2
            and W_raw.shape == (bundle.n, K)
            and vertices.shape[0] == K
        ):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                recovered_embedding = W_raw @ vertices
                squared_distances = (
                    np.sum(recovered_embedding**2, axis=1)[:, None]
                    + np.sum(vertices**2, axis=1)[None, :]
                    - 2.0 * (recovered_embedding @ vertices.T)
                )
                vertex_distances = np.sqrt(np.maximum(squared_distances, 0.0))
            if np.all(np.any(np.isfinite(vertex_distances), axis=0)):
                vertex_distances[~np.isfinite(vertex_distances)] = np.inf
            else:
                vertex_distances = None
        selected = optional_array("selected_observation_indices")
        if (
            hunter == "spa_current"
            and selected is not None
            and selected.shape == (K,)
        ):
            exact_indices = selected.astype(np.int64, copy=False)
        archetypes = optional_array("archetype_to_data_weights")
        if archetypes is not None and archetypes.shape == (K, bundle.n):
            archetype_weights = np.asarray(archetypes, dtype=float)

    observation_kind = {
        "stanford_crc_codex": "Cell",
        "mouse_spleen_codex": "Cell",
        "whats_cooking": "Recipe",
    }.get(str(bundle.dataset), "Document")
    group_kind = {
        "stanford_crc_codex": "Region",
        "mouse_spleen_codex": "Spleen",
        "whats_cooking": "Cuisine",
    }.get(str(bundle.dataset), "Group")

    def stable_smallest(values: np.ndarray, count: int) -> np.ndarray:
        """Return the smallest values with canonical row index as tie-breaker."""

        count = min(len(values), count)
        if count == len(values):
            pool = np.arange(len(values), dtype=np.int64)
        else:
            provisional = np.argpartition(values, count - 1)[:count]
            cutoff = float(np.max(values[provisional]))
            strict = np.flatnonzero(values < cutoff)
            tied = np.flatnonzero(values == cutoff)
            pool = np.concatenate([strict, tied[: count - len(strict)]])
        return pool[np.lexsort((pool, values[pool]))]

    candidates: list[dict[str, Any]] = []
    hunter_label = VERTEX_LABELS.get(hunter, hunter)
    if hunter in {"svs", "svs_star"}:
        fitted_vertex_name = "fitted centroid vertex"
        fitted_vertex_plural = "centroid vertices"
    elif hunter in {"palm", "palm_accelerated"}:
        fitted_vertex_name = "fitted archetype"
        fitted_vertex_plural = "archetypes"
    else:
        fitted_vertex_name = "fitted vertex"
        fitted_vertex_plural = "vertices"
    for topic in range(K):
        ordered: list[int] = []
        exact_index: int | None = None
        if exact_indices is not None:
            proposed = int(exact_indices[topic])
            if 0 <= proposed < bundle.n:
                exact_index = proposed
                ordered.append(proposed)
        ranking_values = (
            vertex_distances[:, topic]
            if vertex_distances is not None
            else -W[:, topic]
        )
        candidate_pool_size = min(
            bundle.n, candidates_per_topic + (1 if exact_index is not None else 0)
        )
        ranked = stable_smallest(ranking_values, candidate_pool_size)
        for value in ranked:
            index = int(value)
            if index not in ordered:
                ordered.append(index)
            if len(ordered) >= candidates_per_topic:
                break

        for rank, index in enumerate(ordered, start=1):
            exact = index == exact_index
            if exact:
                role = "algorithm-selected anchor document"
                provenance = (
                    f"selected directly by {hunter_label} from the observation-side U cloud"
                )
            elif estimator == "document_gplsi" and vertex_distances is not None:
                role = f"nearest observed document to {fitted_vertex_name} (post hoc)"
                provenance = (
                    "derived from the saved W_raw and vertex matrices in the recovered observation-side U geometry"
                )
            else:
                role = "highest topic-membership representative (post hoc)"
                provenance = "derived post hoc from the saved W matrix"

            raw_observation_id = str(bundle.observation_ids[index])
            group = str(bundle.group_ids[index])
            if bundle.dataset == "stanford_crc_codex":
                observation_label = f"Cell {raw_observation_id.rsplit('::', 1)[-1]}"
                group_label = group
            elif bundle.dataset == "mouse_spleen_codex":
                numeric_parts = re.findall(r"\d+", raw_observation_id)
                node_label = numeric_parts[-1] if numeric_parts else raw_observation_id
                observation_label = f"Cell / spatial node {node_label}"
                group_label = group
            elif bundle.dataset == "whats_cooking":
                observation_label = f"Recipe {raw_observation_id}"
                group_label = group.replace("_", " ").title()
            else:
                observation_label = raw_observation_id
                group_label = group

            feature_row = np.asarray(bundle.frequencies[index], dtype=float)
            feature_order = np.argsort(-feature_row, kind="mergesort")
            top_features = [
                {
                    "feature": str(bundle.feature_ids[feature_index]),
                    "weight": round(float(feature_row[feature_index]), 7),
                }
                for feature_index in feature_order
                if feature_row[feature_index] > 0
            ][:6]
            record: dict[str, Any] = {
                "topic": topic,
                "rank": rank,
                "observation_index": index,
                "observation_id": raw_observation_id,
                "observation_label": observation_label,
                "observation_kind": observation_kind,
                "group": group,
                "group_label": group_label,
                "group_kind": group_kind,
                "topic_weight": round(float(W[index, topic]), 7),
                "purity": round(float(np.max(W[index])), 7),
                "topic_weights": np.round(W[index], 7).tolist(),
                "dominant_topic": int(np.argmax(W[index])) + 1,
                "document_length": int(round(float(bundle.document_lengths[index]))),
                "distance_to_fitted_vertex": (
                    None
                    if vertex_distances is None
                    else round(float(vertex_distances[index, topic]), 9)
                ),
                "is_algorithm_selected": exact,
                "role": role,
                "provenance": provenance,
                "top_features": top_features,
            }
            if archetype_weights is not None:
                record["archetype_contribution_weight"] = round(
                    float(archetype_weights[topic, index]), 9
                )
            candidates.append(record)

    if estimator == "document_gplsi" and exact_indices is not None:
        return (
            "exact_and_nearest_document_vertices",
            f"The first document for each topic is the exact observation selected by {hunter_label}; the next two are the nearest observed documents to that fitted vertex in the recovered U geometry.",
            candidates,
        )
    if estimator == "document_gplsi" and vertex_distances is not None:
        return (
            "nearest_document_vertex_representatives",
            f"{hunter_label} fits {fitted_vertex_plural} rather than selecting observed documents. The cards show the three observed documents nearest each fitted vertex in the recovered U geometry; they are representatives, not exact algorithm-selected anchors.",
            candidates,
        )
    if estimator == "document_gplsi":
        return (
            "highest_topic_membership_documents",
            "The saved fit does not contain usable document-side vertex geometry. The cards therefore show the three documents with the highest saved W weight for each topic as transparent post-hoc representatives.",
            candidates,
        )
    return (
        "highest_topic_membership_documents",
        "This method does not hunt observation-side vertices. The cards show the three documents with the highest saved W weight for each topic as transparent post-hoc representatives. If several documents tie—as they do at K=1—the canonical row index breaks the tie, so the representative is not unique.",
        candidates,
    )


def materialize_data(selection_path: Path, artifact_root: Path, output: Path) -> None:
    module_path = REPO_ROOT / "src/gplsi/real_data.py"
    spec = importlib.util.spec_from_file_location("gplsi_dashboard_real_data", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical data module from {module_path}")
    real_data = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = real_data
    spec.loader.exec_module(real_data)
    RealDataBundle = real_data.RealDataBundle
    load_real_data = real_data.load_real_data

    payload = json.loads(selection_path.read_text())
    geometries: dict[str, dict[str, Any]] = {}
    geometry_indices: dict[str, np.ndarray] = {}
    bundles: dict[str, RealDataBundle] = {}
    fit_index: list[dict[str, Any]] = []
    artifact_not_staged: list[dict[str, Any]] = []
    fit_output_dir = output.parent / "topic-fits"
    fit_output_dir.mkdir(parents=True, exist_ok=True)

    for selection in payload["selections"]:
        try:
            artifact = _artifact_path(artifact_root, selection["artifact"])
        except FileNotFoundError:
            artifact_not_staged.append(
                {
                    key: selection.get(key)
                    for key in (
                        "id",
                        "dataset",
                        "group",
                        "K",
                        "variant_id",
                        "variant_label",
                        "estimator_family",
                        "artifact",
                    )
                }
            )
            continue
        dataset = selection["dataset"]
        group = selection.get("group")
        bundle_key = f"{dataset}::{group or 'all'}"
        if bundle_key not in bundles:
            aliases = {
                "stanford_crc_codex": "crc",
                "mouse_spleen_codex": "spleen",
                "whats_cooking": "cook",
            }
            load_group = (
                str(group or "BALBc-1")
                if dataset == "mouse_spleen_codex"
                else "BALBc-1"
            )
            bundles[bundle_key] = load_real_data(aliases[dataset], group=load_group)
        bundle = bundles[bundle_key]
        if bundle_key not in geometries:
            geometry, indices = _graph_geometry(bundle, key=bundle_key)
            geometries[bundle_key] = geometry
            geometry_indices[bundle_key] = indices

        with np.load(artifact, allow_pickle=False) as arrays:
            W = np.asarray(arrays["W_hat"], dtype=float)
            A = np.asarray(arrays["A_hat"], dtype=float)
            (
                document_candidate_mode,
                document_candidate_note,
                document_candidates,
            ) = _document_candidates(selection, bundle, W, arrays)
        if W.shape != (bundle.n, selection["K"]):
            raise ValueError(f"unexpected W shape {W.shape} for {selection['id']}")
        if A.shape != (selection["K"], bundle.p):
            raise ValueError(f"unexpected A shape {A.shape} for {selection['id']}")
        indices = geometry_indices[bundle_key]
        top_features = []
        for topic in range(selection["K"]):
            top = np.argsort(A[topic])[-12:][::-1]
            top_features.append(
                [
                    {
                        "feature": str(bundle.feature_ids[index]),
                        "weight": round(float(A[topic, index]), 7),
                    }
                    for index in top
                ]
            )
        composition_indices = np.arange(A.shape[1], dtype=np.int64)
        candidate_mode, candidate_note, feature_candidates = _feature_candidates(
            selection, bundle, A
        )
        fit_payload = {
            **{key: value for key, value in selection.items() if key != "artifact"},
            "geometry_key": bundle_key,
            "W": np.round(W[indices], 5).tolist(),
            "topic_prevalence": np.round(W.mean(axis=0), 5).tolist(),
            "w_entropy_summary": _w_entropy_summary(W),
            "top_features": top_features,
            "composition_features": [
                str(bundle.feature_ids[index]) for index in composition_indices
            ],
            "composition": np.round(A[:, composition_indices], 7).tolist(),
            "composition_feature_count": int(len(composition_indices)),
            "composition_total_feature_count": int(bundle.p),
            "group_topic_composition": _group_topic_composition(bundle, W),
            "feature_candidate_mode": candidate_mode,
            "feature_candidate_note": candidate_note,
            "feature_candidates": feature_candidates,
            "document_candidate_mode": document_candidate_mode,
            "document_candidate_note": document_candidate_note,
            "document_candidates": document_candidates,
        }
        payload_identity = "::".join(
            [
                str(selection["id"]),
                str(selection["fit_id"]),
                str(selection["artifact"]),
            ]
        )
        payload_name = hashlib.sha256(payload_identity.encode()).hexdigest()[:24] + ".json"
        (fit_output_dir / payload_name).write_text(
            json.dumps(fit_payload, separators=(",", ":"), default=_json_default)
            + "\n"
        )
        fit_index.append(
            {
                **{
                    key: value
                    for key, value in selection.items()
                    if key != "artifact"
                },
                "payload_url": f"/data/topic-fits/{payload_name}",
            }
        )

    payload["geometries"] = geometries
    payload["fits"] = fit_index
    payload.pop("selections", None)
    payload["topic_explorer_availability"] = {
        "selected_fit_count": len(fit_index) + len(artifact_not_staged),
        "available_fit_count": len(fit_index),
        "artifact_not_staged_count": len(artifact_not_staged),
        "artifact_not_staged": artifact_not_staged,
        "delivery": "lazy_static_json_v1",
    }
    payload["dashboard_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n"
    )


def write_artifact_list(
    selection_path: Path, output: Path, dataset: str | None = None
) -> None:
    payload = json.loads(selection_path.read_text())
    selections = payload["selections"]
    if dataset is not None:
        selections = [item for item in selections if item["dataset"] == dataset]
    paths = sorted({item["artifact"] for item in selections})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(paths) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--result-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    artifact_list = subparsers.add_parser("artifact-list")
    artifact_list.add_argument("--selection", type=Path, required=True)
    artifact_list.add_argument("--output", type=Path, required=True)
    artifact_list.add_argument("--dataset")

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--selection", type=Path, required=True)
    materialize.add_argument("--artifact-root", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select":
        select_data(args.result_root, args.output)
    elif args.command == "artifact-list":
        write_artifact_list(args.selection, args.output, dataset=args.dataset)
    else:
        materialize_data(args.selection, args.artifact_root, args.output)


if __name__ == "__main__":
    main()
