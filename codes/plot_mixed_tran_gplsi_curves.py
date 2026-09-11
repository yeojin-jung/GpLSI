#!/usr/bin/env python3
"""Plot one-at-a-time curves for the mixed Tran-A/GpLSI-W screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
AXES = ["p", "N", "n", "a_zipf"]
AXIS_LABELS = {
    "p": "Vocabulary size p",
    "N": "Words per document N",
    "n": "Number of documents n",
    "a_zipf": "Zipf decay exponent",
}
AXIS_FILE_STEMS = {
    "p": "vocabulary_p",
    "N": "document_length_N",
    "n": "document_count_n",
    "a_zipf": "zipf_decay",
}
METHODS = ["spa_current", "svs_star", "pp_spa"]
METHOD_LABELS = {"spa_current": "SPA", "svs_star": "SVS*", "pp_spa": "pp-SPA"}
METHOD_COLORS = {"spa_current": "#4C78A8", "svs_star": "#F58518", "pp_spa": "#54A24B"}
METHOD_STYLES = {"spa_current": "-", "svs_star": "--", "pp_spa": ":"}
METHOD_MARKERS = {"spa_current": "o", "svs_star": "s", "pp_spa": "^"}
PREPS = ["P0", "P1", "P2", "P3"]
PREP_LABELS = {
    "P0": "P0: unprocessed",
    "P1": "P1: Tran threshold only",
    "P2": "P2: frequency weight only",
    "P3": "P3: threshold + weight",
}
PREP_COLORS = {"P0": "#4D4D4D", "P1": "#4C78A8", "P2": "#F58518", "P3": "#54A24B"}
PREP_MARKERS = {"P0": "o", "P1": "s", "P2": "^", "P3": "D"}
PREP_STYLES = {"P0": "--", "P1": "-.", "P2": ":", "P3": "-"}
FAMILY_LABELS = {
    "tran_exact_no_graph_benchmark": "Original Tran generator (ordinary pLSI; no graph)",
    "tran_A_gplsi_graph_W": "Mixed: Tran word frequencies + GpLSI graph W",
}


def _curve_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["A_mean_total_variation"] = frame.A_l1_error / (2.0 * frame.K)
    keys = [
        "experiment_family",
        "design_axis",
        "design_value",
        "preprocessing_variant",
        "vertex_hunter",
        "embedding_source",
    ]
    grouped = frame.groupby(keys, dropna=False)
    summary = grouped.agg(
        runs=("status", "size"),
        successful=("status", lambda x: int((x == "ok").sum())),
        unstable=("status", lambda x: int((x == "unstable").sum())),
        failed=("status", lambda x: int((x == "failed").sum())),
        W_rmse_median=("W_rmse", "median"),
        W_rmse_q25=("W_rmse", lambda x: x.quantile(0.25)),
        W_rmse_q75=("W_rmse", lambda x: x.quantile(0.75)),
        A_rmse_median=("A_rmse", "median"),
        A_rmse_q25=("A_rmse", lambda x: x.quantile(0.25)),
        A_rmse_q75=("A_rmse", lambda x: x.quantile(0.75)),
        A_mean_total_variation_median=("A_mean_total_variation", "median"),
        A_mean_total_variation_q25=("A_mean_total_variation", lambda x: x.quantile(0.25)),
        A_mean_total_variation_q75=("A_mean_total_variation", lambda x: x.quantile(0.75)),
        retained_feature_count_median=("retained_feature_count", "median"),
        observed_p_median=("p", "median"),
    ).reset_index()
    summary["failure_rate"] = (summary.failed + summary.unstable) / summary.runs
    return summary


def _line(axis, subset, metric: str, color: str, style: str) -> None:
    subset = subset.sort_values("design_value")
    subset = subset.loc[subset.successful >= np.ceil(subset.runs / 2)]
    x = subset.design_value.to_numpy(dtype=float)
    y = subset[f"{metric}_median"].to_numpy(dtype=float)
    low = subset[f"{metric}_q25"].to_numpy(dtype=float)
    high = subset[f"{metric}_q75"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return
    axis.plot(
        x[finite], y[finite], marker="o", markersize=4, linewidth=1.8,
        color=color, linestyle=style,
    )
    band = finite & np.isfinite(low) & np.isfinite(high)
    if band.any():
        axis.fill_between(x[band], low[band], high[band], color=color, alpha=0.10)


def _format_axis(axis, design_axis: str, metric: str) -> None:
    axis.set_xlabel(AXIS_LABELS[design_axis])
    axis.set_ylabel("W RMSE" if metric == "W_rmse" else "Mean topic TV error")
    axis.grid(alpha=0.25)
    if design_axis in {"p", "N", "n"}:
        axis.set_xscale("log")
        if axis.get_lines():
            axis.set_xticks(sorted(axis.get_lines()[0].get_xdata()))
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())


def _plot_family(summary: pd.DataFrame, family: str, output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(12.5, 14.5), squeeze=False)
    family_data = summary.loc[summary.experiment_family == family]
    for row, design_axis in enumerate(AXES):
        for col, metric in enumerate(["W_rmse", "A_mean_total_variation"]):
            axis = axes[row, col]
            for prep in ["P0", "P3"]:
                for method in METHODS:
                    subset = family_data.loc[
                        (family_data.design_axis == design_axis)
                        & (family_data.preprocessing_variant == prep)
                        & (family_data.vertex_hunter == method)
                        & (family_data.embedding_source == "U_hat")
                    ]
                    _line(axis, subset, metric, METHOD_COLORS[method], PREP_STYLES[prep])
            _format_axis(axis, design_axis, metric)
            if row == 0:
                axis.set_title("Document-topic recovery" if col == 0 else "Topic-word recovery")
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2, label=METHOD_LABELS[m])
        for m in METHODS
    ] + [
        Line2D([0], [0], color="#333333", linestyle=PREP_STYLES[p], linewidth=2, label=PREP_LABELS[p])
        for p in ["P0", "P3"]
    ]
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(FAMILY_LABELS[family], fontsize=15, y=0.995)
    figure.text(0.5, 0.975, "Medians/IQR across successful fits; points with fewer than 2/3 successes are omitted", ha="center", va="top", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_primary_comparison(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(12.5, 14.5), squeeze=False)
    data = summary.loc[summary.preprocessing_variant == "P3"]
    family_styles = {
        "tran_exact_no_graph_benchmark": "--",
        "tran_A_gplsi_graph_W": "-",
    }
    for row, design_axis in enumerate(AXES):
        for col, metric in enumerate(["W_rmse", "A_mean_total_variation"]):
            axis = axes[row, col]
            for family, style in family_styles.items():
                for method in METHODS:
                    subset = data.loc[
                        (data.experiment_family == family)
                        & (data.design_axis == design_axis)
                        & (data.vertex_hunter == method)
                        & (data.embedding_source == "U_hat")
                    ]
                    _line(axis, subset, metric, METHOD_COLORS[method], style)
            _format_axis(axis, design_axis, metric)
            if row == 0:
                axis.set_title("Document-topic recovery" if col == 0 else "Topic-word recovery")
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2, label=METHOD_LABELS[m])
        for m in METHODS
    ] + [
        Line2D(
            [0], [0], color="#333333", linestyle=family_styles[f], linewidth=2,
            label=("Original Tran/no graph" if f.startswith("tran_exact") else "Mixed Tran A + graph W"),
        )
        for f in family_styles
    ]
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle("Method curves with Tran thresholding + frequency weighting", fontsize=15, y=0.995)
    figure.text(0.5, 0.975, "Medians/IQR across successful fits; points with fewer than 2/3 successes are omitted", ha="center", va="top", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_experiment_panels(
    summary: pd.DataFrame,
    family: str,
    method: str,
    metric: str,
    output: Path,
) -> None:
    """Make four uncluttered panels, one for each varied parameter."""
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), squeeze=False)
    family_data = summary.loc[
        (summary.experiment_family == family)
        & (summary.vertex_hunter == method)
        & (summary.embedding_source == "U_hat")
    ]
    for axis, design_axis in zip(axes.ravel(), AXES):
        experiment_data = family_data.loc[family_data.design_axis == design_axis]
        all_x = np.sort(experiment_data.design_value.dropna().unique())
        for prep_index, prep in enumerate(PREPS):
            raw = experiment_data.loc[
                experiment_data.preprocessing_variant == prep
            ].sort_values("design_value")
            supported = raw.loc[raw.successful >= np.ceil(raw.runs / 2)]
            x = supported.design_value.to_numpy(dtype=float)
            if design_axis in {"p", "N", "n"}:
                x_plot = x * np.exp((prep_index - 1.5) * 0.015)
            else:
                span = float(np.ptp(all_x)) if len(all_x) > 1 else 1.0
                x_plot = x + (prep_index - 1.5) * 0.008 * span
            y = supported[f"{metric}_median"].to_numpy(dtype=float)
            low = supported[f"{metric}_q25"].to_numpy(dtype=float)
            high = supported[f"{metric}_q75"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                axis.plot(
                    x_plot[finite],
                    y[finite],
                    color=PREP_COLORS[prep],
                    linestyle=PREP_STYLES[prep],
                    marker=PREP_MARKERS[prep],
                    markersize=5,
                    linewidth=2,
                    label=PREP_LABELS[prep],
                )
                band = finite & np.isfinite(low) & np.isfinite(high)
                if band.any():
                    axis.fill_between(
                        x_plot[band], low[band], high[band],
                        color=PREP_COLORS[prep], alpha=0.10,
                    )
            unsupported = raw.loc[raw.successful < np.ceil(raw.runs / 2)]
            if not unsupported.empty:
                unsupported_x = unsupported.design_value.to_numpy(dtype=float)
                if design_axis in {"p", "N", "n"}:
                    unsupported_x *= np.exp((prep_index - 1.5) * 0.015)
                else:
                    span = float(np.ptp(all_x)) if len(all_x) > 1 else 1.0
                    unsupported_x += (prep_index - 1.5) * 0.008 * span
                axis.scatter(
                    unsupported_x,
                    np.full(len(unsupported), 0.025 + 0.025 * prep_index),
                    transform=axis.get_xaxis_transform(),
                    marker="x",
                    s=45,
                    linewidth=2,
                    color=PREP_COLORS[prep],
                    clip_on=False,
                )
        axis.set_title(AXIS_LABELS[design_axis])
        axis.set_xlabel(AXIS_LABELS[design_axis])
        axis.set_ylabel(
            "Document-topic W RMSE"
            if metric == "W_rmse"
            else "Mean topic total-variation error"
        )
        axis.grid(alpha=0.25)
        if design_axis in {"p", "N", "n"}:
            axis.set_xscale("log")
            axis.set_xticks(all_x)
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        else:
            axis.set_xticks(all_x)
    handles = [
        Line2D(
            [0], [0],
            color=PREP_COLORS[prep],
            linestyle=PREP_STYLES[prep],
            marker=PREP_MARKERS[prep],
            linewidth=2,
            label=PREP_LABELS[prep],
        )
        for prep in PREPS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.942),
    )
    metric_label = "W recovery" if metric == "W_rmse" else "A recovery"
    figure.suptitle(
        f"{FAMILY_LABELS[family]} — {METHOD_LABELS[method]} — {metric_label}",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.963,
        "Each panel is one experiment; curves are P0–P3. Medians/IQR across 3 seeds; curves are slightly offset so overlaps remain visible; × means fewer than 2 successes.",
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_vertex_comparison_by_preprocessing(
    summary: pd.DataFrame,
    family: str,
    design_axis: str,
    metric: str,
    output: Path,
) -> None:
    """Fix one experiment and compare all vertex hunters in P0--P3 panels."""
    figure, axes = plt.subplots(
        2, 2, figsize=(12.5, 9.2), squeeze=False, sharex=True, sharey=True
    )
    experiment_data = summary.loc[
        (summary.experiment_family == family)
        & (summary.design_axis == design_axis)
        & (summary.embedding_source == "U_hat")
    ]
    all_x = np.sort(experiment_data.design_value.dropna().unique())
    for axis, prep in zip(axes.ravel(), PREPS):
        prep_data = experiment_data.loc[
            experiment_data.preprocessing_variant == prep
        ]
        for method_index, method in enumerate(METHODS):
            raw = prep_data.loc[prep_data.vertex_hunter == method].sort_values(
                "design_value"
            )
            supported = raw.loc[raw.successful >= np.ceil(raw.runs / 2)]
            x = supported.design_value.to_numpy(dtype=float)
            if design_axis in {"p", "N", "n"}:
                x_plot = x * np.exp((method_index - 1) * 0.015)
            else:
                span = float(np.ptp(all_x)) if len(all_x) > 1 else 1.0
                x_plot = x + (method_index - 1) * 0.008 * span
            y = supported[f"{metric}_median"].to_numpy(dtype=float)
            low = supported[f"{metric}_q25"].to_numpy(dtype=float)
            high = supported[f"{metric}_q75"].to_numpy(dtype=float)
            finite = np.isfinite(x_plot) & np.isfinite(y)
            if finite.any():
                axis.plot(
                    x_plot[finite],
                    y[finite],
                    color=METHOD_COLORS[method],
                    linestyle=METHOD_STYLES[method],
                    marker=METHOD_MARKERS[method],
                    markersize=5,
                    linewidth=2,
                    label=METHOD_LABELS[method],
                )
                band = finite & np.isfinite(low) & np.isfinite(high)
                if band.any():
                    axis.fill_between(
                        x_plot[band],
                        low[band],
                        high[band],
                        color=METHOD_COLORS[method],
                        alpha=0.10,
                    )
            unsupported = raw.loc[raw.successful < np.ceil(raw.runs / 2)]
            if not unsupported.empty:
                unsupported_x = unsupported.design_value.to_numpy(dtype=float)
                if design_axis in {"p", "N", "n"}:
                    unsupported_x *= np.exp((method_index - 1) * 0.015)
                else:
                    span = float(np.ptp(all_x)) if len(all_x) > 1 else 1.0
                    unsupported_x += (method_index - 1) * 0.008 * span
                axis.scatter(
                    unsupported_x,
                    np.full(len(unsupported), 0.03 + 0.035 * method_index),
                    transform=axis.get_xaxis_transform(),
                    marker="x",
                    s=48,
                    linewidth=2,
                    color=METHOD_COLORS[method],
                    clip_on=False,
                )
        axis.set_title(PREP_LABELS[prep])
        axis.grid(alpha=0.25)
        if design_axis in {"p", "N", "n"}:
            axis.set_xscale("log")
            axis.set_xticks(all_x)
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        else:
            axis.set_xticks(all_x)
    for axis in axes[-1, :]:
        axis.set_xlabel(AXIS_LABELS[design_axis])
    y_label = (
        "Document-topic W RMSE"
        if metric == "W_rmse"
        else "Mean topic total-variation error"
    )
    for axis in axes[:, 0]:
        axis.set_ylabel(y_label)
    handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            marker=METHOD_MARKERS[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    metric_label = "W recovery" if metric == "W_rmse" else "A recovery"
    figure.suptitle(
        f"{FAMILY_LABELS[family]} — {AXIS_LABELS[design_axis]} — {metric_label}",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.963,
        "Subplots are P0–P3; curves compare all vertex-hunting algorithms. Medians/IQR across 3 seeds; × means fewer than 2 successes.",
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_all_algorithms_and_variants(
    summary: pd.DataFrame,
    family: str,
    metric: str,
    output: Path,
) -> None:
    """Put every vertex hunter and P0--P3 curve on each experiment subplot."""
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 9.8), squeeze=False)
    family_data = summary.loc[
        (summary.experiment_family == family)
        & (summary.embedding_source == "U_hat")
    ]
    for axis, design_axis in zip(axes.ravel(), AXES):
        experiment_data = family_data.loc[family_data.design_axis == design_axis]
        all_x = np.sort(experiment_data.design_value.dropna().unique())
        for prep_index, prep in enumerate(PREPS):
            for method_index, method in enumerate(METHODS):
                raw = experiment_data.loc[
                    (experiment_data.preprocessing_variant == prep)
                    & (experiment_data.vertex_hunter == method)
                ].sort_values("design_value")
                supported = raw.loc[raw.successful >= np.ceil(raw.runs / 2)]
                x = supported.design_value.to_numpy(dtype=float)
                display_offset = (prep_index - 1.5) * 0.012 + (method_index - 1) * 0.003
                if design_axis in {"p", "N", "n"}:
                    x_plot = x * np.exp(display_offset)
                else:
                    span = float(np.ptp(all_x)) if len(all_x) > 1 else 1.0
                    x_plot = x + display_offset * span
                y = supported[f"{metric}_median"].to_numpy(dtype=float)
                finite = np.isfinite(x_plot) & np.isfinite(y)
                if finite.any():
                    axis.plot(
                        x_plot[finite],
                        y[finite],
                        color=METHOD_COLORS[method],
                        linestyle=PREP_STYLES[prep],
                        marker=METHOD_MARKERS[method],
                        markersize=4.5,
                        linewidth=1.8,
                        alpha=0.90,
                    )
                unsupported = raw.loc[raw.successful < np.ceil(raw.runs / 2)]
                if not unsupported.empty:
                    unsupported_x = unsupported.design_value.to_numpy(dtype=float)
                    if design_axis in {"p", "N", "n"}:
                        unsupported_x *= np.exp(display_offset)
                    else:
                        span = float(np.ptp(all_x)) if len(all_x) > 1 else 1.0
                        unsupported_x += display_offset * span
                    axis.scatter(
                        unsupported_x,
                        np.full(
                            len(unsupported),
                            0.02 + 0.018 * prep_index + 0.006 * method_index,
                        ),
                        transform=axis.get_xaxis_transform(),
                        marker="x",
                        s=32,
                        linewidth=1.5,
                        color=METHOD_COLORS[method],
                        clip_on=False,
                    )
        axis.set_title(AXIS_LABELS[design_axis])
        axis.set_xlabel(AXIS_LABELS[design_axis])
        axis.set_ylabel(
            "Document-topic W RMSE"
            if metric == "W_rmse"
            else "Mean topic total-variation error"
        )
        axis.grid(alpha=0.22)
        if design_axis in {"p", "N", "n"}:
            axis.set_xscale("log")
            axis.set_xticks(all_x)
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        else:
            axis.set_xticks(all_x)
    method_handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[method],
            linestyle="-",
            marker=METHOD_MARKERS[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    prep_handles = [
        Line2D(
            [0], [0],
            color="#333333",
            linestyle=PREP_STYLES[prep],
            linewidth=2,
            label=prep,
        )
        for prep in PREPS
    ]
    figure.legend(
        handles=method_handles + prep_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    metric_label = "W recovery" if metric == "W_rmse" else "A recovery"
    figure.suptitle(
        f"{FAMILY_LABELS[family]} — all vertex hunters and P0–P3 — {metric_label}",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.963,
        "Each subplot is one parameter experiment. Color/marker = vertex hunter; line style = preprocessing variant; median across 3 seeds.",
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.015,
        "Curves have small horizontal display offsets to expose overlaps; × marks fewer than 2 successful fits. IQR values remain in the summary table.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.91), w_pad=4.0, h_pad=3.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output = args.output_directory if args.output_directory.is_absolute() else REPO_ROOT / args.output_directory
    frame = pd.read_csv(input_path)
    summary = _curve_summary(frame)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(input_path.parent / "mixed_tran_gplsi_curve_summary.csv", index=False)
    for family, family_stem in [
        ("tran_exact_no_graph_benchmark", "original_tran_plsi_no_graph"),
        ("tran_A_gplsi_graph_W", "mixed_gplsi_tran_frequencies"),
    ]:
        for method in METHODS:
            for metric, metric_stem in [
                ("W_rmse", "W_rmse"),
                ("A_mean_total_variation", "A_topic_TV"),
            ]:
                _plot_experiment_panels(
                    summary,
                    family,
                    method,
                    metric,
                    output / f"{family_stem}_{method}_{metric_stem}_P0_P1_P2_P3.png",
                )
        comparison_output = output / "vh_comparison_by_preprocessing_final"
        for design_axis in AXES:
            for metric, metric_stem in [
                ("W_rmse", "W_rmse"),
                ("A_mean_total_variation", "A_topic_TV"),
            ]:
                _plot_vertex_comparison_by_preprocessing(
                    summary,
                    family,
                    design_axis,
                    metric,
                    comparison_output
                    / f"{family_stem}_{AXIS_FILE_STEMS[design_axis]}_{metric_stem}_VH_by_P0_P1_P2_P3.png",
                )
        same_panel_output = output / "all_VH_and_P_same_experiment_panels_final"
        for metric, metric_stem in [
            ("W_rmse", "W_rmse"),
            ("A_mean_total_variation", "A_topic_TV"),
        ]:
            _plot_all_algorithms_and_variants(
                summary,
                family,
                metric,
                same_panel_output
                / f"{family_stem}_{metric_stem}_all_VH_P0_P1_P2_P3.png",
            )


if __name__ == "__main__":
    main()
