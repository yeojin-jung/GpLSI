#!/usr/bin/env python3
"""Focused pilot figures for document- and anchor-word GpLSI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {
    "spa_current": "#0072B2",
    "svs": "#009E73",
    "svs_star": "#D55E00",
    "pp_spa": "#CC79A7",
}
MARKERS = {"spa_current": "o", "svs": "s", "svs_star": "^", "pp_spa": "D"}
LINESTYLES = {
    "P0_raw": "-",
    "P1_threshold": "--",
    "P2_weight": "-.",
    "P3_threshold_weight": ":",
}
BASELINE_STYLES = {
    "spatial_lda": ("#000000", "P"),
    "topicscore_raw": ("#666666", "X"),
    "topicscore_graph_denoised": ("#E69F00", "v"),
    "lda": ("#56B4E9", "*"),
}


def _summarize(frame: pd.DataFrame, metric: str, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, dropna=False)[metric]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            mean="mean",
            std="std",
            count="count",
        )
        .reset_index()
    )


def _legend(fig: plt.Figure, include_baselines: bool = True) -> None:
    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=MARKERS[method],
            linewidth=2,
            label=method.replace("_current", "").replace("_", "-"),
        )
        for method, color in COLORS.items()
    ]
    preprocessing_handles = [
        Line2D([0], [0], color="#444444", linestyle=style, linewidth=2, label=name)
        for name, style in LINESTYLES.items()
    ]
    handles = algorithm_handles + preprocessing_handles
    if include_baselines:
        handles += [
            Line2D(
                [0], [0], color=color, marker=marker, linestyle="--", linewidth=1.8, label=method
            )
            for method, (color, marker) in BASELINE_STYLES.items()
        ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def _draw_curves(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    include_baselines: bool,
) -> None:
    gplsi = frame[
        frame.estimator_family.isin(["gplsi_document", "gplsi_anchor_word_profile"])
        & frame.status.isin(["ok", "unstable"])
    ]
    summary = _summarize(
        gplsi,
        metric,
        ["word_decay_parameter", "preprocessing_variant", "vertex_hunter"],
    )
    for (preprocessing, hunter), group in summary.groupby(
        ["preprocessing_variant", "vertex_hunter"], dropna=False
    ):
        group = group.sort_values("word_decay_parameter")
        color = COLORS.get(hunter, "#333333")
        ax.plot(
            group.word_decay_parameter,
            group["median"],
            color=color,
            marker=MARKERS.get(hunter, "o"),
            linestyle=LINESTYLES.get(preprocessing, "-"),
            linewidth=1.5,
            markersize=4,
            alpha=0.92,
        )
        if group["count"].min() >= 3:
            ax.fill_between(
                group.word_decay_parameter.to_numpy(float),
                group.q25.to_numpy(float),
                group.q75.to_numpy(float),
                color=color,
                alpha=0.035,
                linewidth=0,
            )
    if include_baselines:
        baselines = frame[
            frame.estimator_family.isin(BASELINE_STYLES)
            & frame.status.isin(["ok", "unstable"])
        ]
        baseline_summary = _summarize(
            baselines, metric, ["word_decay_parameter", "estimator_family"]
        )
        for method, group in baseline_summary.groupby("estimator_family"):
            group = group.sort_values("word_decay_parameter")
            color, marker = BASELINE_STYLES[method]
            ax.plot(
                group.word_decay_parameter,
                group["median"],
                color=color,
                marker=marker,
                linestyle="--",
                linewidth=1.7,
                markersize=5,
                alpha=0.95,
            )
    ax.set_xscale("log")
    ax.grid(alpha=0.22)
    ax.set_xlabel("Word-frequency decay exponent $a_{Zipf}$")


def plot_metric(
    data: pd.DataFrame,
    output: Path,
    metric: str,
    ylabel: str,
    stem: str,
    *,
    recovery_panels: bool,
) -> None:
    for design, design_data in data.groupby("design_variant"):
        if recovery_panels:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
            for row, side in enumerate(["document", "anchor_word"]):
                family = "gplsi_document" if side == "document" else "gplsi_anchor_word_profile"
                for column, recovery in enumerate(["current", "poisson_full"]):
                    panel = design_data[
                        ((design_data.estimator_family == family) & (design_data.A_recovery_method == recovery))
                        | design_data.estimator_family.isin(BASELINE_STYLES)
                    ]
                    _draw_curves(axes[row, column], panel, metric, include_baselines=True)
                    axes[row, column].set_title(f"{side.replace('_', ' ').title()} — {recovery}")
                    axes[row, column].set_ylabel(ylabel)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharex=True)
            for column, side in enumerate(["document", "anchor_word"]):
                family = "gplsi_document" if side == "document" else "gplsi_anchor_word_profile"
                panel = design_data[
                    ((design_data.estimator_family == family) & (design_data.A_recovery_method == "current"))
                    | design_data.estimator_family.isin(BASELINE_STYLES)
                ]
                _draw_curves(axes[column], panel, metric, include_baselines=True)
                axes[column].set_title(side.replace("_", " ").title())
                axes[column].set_ylabel(ylabel)
        fig.suptitle(f"{ylabel} vs word-frequency decay\n{design.replace('_', ' ')}", y=0.99)
        _legend(fig)
        fig.tight_layout(rect=(0, 0.12, 1, 0.96))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{design}_{stem}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_failure_and_condition(data: pd.DataFrame, output: Path) -> None:
    for design, design_data in data.groupby("design_variant"):
        gplsi = design_data[design_data.estimator_family.str.startswith("gplsi_")].copy()
        gplsi = gplsi[gplsi.A_recovery_method == "current"]
        gplsi["failed"] = gplsi.status.eq("failed").astype(float)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        for column, (side, family) in enumerate(
            [("document", "gplsi_document"), ("anchor_word", "gplsi_anchor_word_profile")]
        ):
            panel = gplsi[gplsi.estimator_family == family]
            failure = _summarize(
                panel,
                "failed",
                ["word_decay_parameter", "preprocessing_variant", "vertex_hunter"],
            )
            condition = _summarize(
                panel[panel.status.isin(["ok", "unstable"])],
                "vertex_condition_number",
                ["word_decay_parameter", "preprocessing_variant", "vertex_hunter"],
            )
            for summary, ax, ylabel in [
                (failure, axes[0, column], "Failure rate"),
                (condition, axes[1, column], "Vertex condition number"),
            ]:
                for (prep, hunter), group in summary.groupby(
                    ["preprocessing_variant", "vertex_hunter"]
                ):
                    group = group.sort_values("word_decay_parameter")
                    ax.plot(
                        group.word_decay_parameter,
                        group["mean"] if ylabel == "Failure rate" else group["median"],
                        color=COLORS[hunter],
                        marker=MARKERS[hunter],
                        linestyle=LINESTYLES[prep],
                        linewidth=1.5,
                        markersize=4,
                    )
                ax.set_xscale("log")
                if ylabel != "Failure rate":
                    ax.set_yscale("log")
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Word-frequency decay exponent $a_{Zipf}$")
                ax.grid(alpha=0.22)
                ax.set_title(side.replace("_", " ").title())
        fig.suptitle(f"Failure and conditioning\n{design.replace('_', ' ')}")
        _legend(fig, include_baselines=False)
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{design}_failure_and_vertex_condition.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_preprocessing(data: pd.DataFrame, output: Path) -> None:
    gplsi = data[
        data.estimator_family.eq("gplsi_anchor_word_profile")
        & data.vertex_hunter.eq("spa_current")
        & data.A_recovery_method.eq("current")
    ].copy()
    def minimum_anchor_count(value: str) -> float:
        counts = json.loads(value) if isinstance(value, str) and value else []
        return float(min(counts)) if counts else np.nan

    gplsi["minimum_retained_anchors"] = gplsi.retained_anchor_count_by_topic.fillna("[]").map(
        minimum_anchor_count
    )
    for design, panel in gplsi.groupby("design_variant"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
        for prep, group in panel.groupby("preprocessing_variant"):
            for metric, ax, ylabel in [
                ("retained_feature_fraction", axes[0], "Retained feature fraction"),
                ("minimum_retained_anchors", axes[1], "Minimum retained anchors/topic"),
            ]:
                summary = _summarize(group, metric, ["word_decay_parameter"])
                ax.plot(
                    summary.word_decay_parameter,
                    summary["median"],
                    color="#333333",
                    linestyle=LINESTYLES[prep],
                    marker="o",
                    label=prep,
                )
                ax.set_xscale("log")
                ax.set_xlabel("Word-frequency decay exponent $a_{Zipf}$")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.22)
        axes[0].legend(frameon=False)
        fig.suptitle(f"Preprocessing and anchor survival\n{design.replace('_', ' ')}")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{design}_retention_and_anchor_survival.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_vh_within_preprocessing(data: pd.DataFrame, output: Path) -> None:
    """Put every VH algorithm on the same P-specific subplot."""

    specifications = [
        ("W_rmse", "W entrywise RMSE", "W_error", ["current"]),
        (
            "A_mean_topic_TV",
            "A mean topic TV error",
            "A_error",
            ["current", "poisson_full"],
        ),
    ]
    family_by_side = {
        "document": "gplsi_document",
        "anchor_word": "gplsi_anchor_word_profile",
    }
    for design, design_data in data.groupby("design_variant"):
        for side, family in family_by_side.items():
            for metric, ylabel, stem, recoveries in specifications:
                for recovery in recoveries:
                    panel_data = design_data[
                        design_data.estimator_family.eq(family)
                        & design_data.A_recovery_method.eq(recovery)
                        & design_data.status.isin(["ok", "unstable"])
                    ]
                    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
                    for ax, preprocessing in zip(axes.flat, LINESTYLES):
                        subset = panel_data[
                            panel_data.preprocessing_variant.eq(preprocessing)
                        ]
                        summary = _summarize(
                            subset,
                            metric,
                            ["word_decay_parameter", "vertex_hunter"],
                        )
                        for hunter, group in summary.groupby("vertex_hunter"):
                            group = group.sort_values("word_decay_parameter")
                            ax.plot(
                                group.word_decay_parameter,
                                group["median"],
                                color=COLORS[hunter],
                                marker=MARKERS[hunter],
                                linewidth=2,
                                markersize=5,
                                label=hunter.replace("_current", "").replace("_", "-"),
                            )
                            if group["count"].min() >= 3:
                                ax.fill_between(
                                    group.word_decay_parameter.to_numpy(float),
                                    group.q25.to_numpy(float),
                                    group.q75.to_numpy(float),
                                    color=COLORS[hunter],
                                    alpha=0.08,
                                    linewidth=0,
                                )
                        ax.set_xscale("log")
                        ax.set_title(preprocessing)
                        ax.set_xlabel("Word-frequency decay exponent $a_{Zipf}$")
                        ax.set_ylabel(ylabel)
                        ax.grid(alpha=0.22)
                    handles = [
                        Line2D(
                            [0],
                            [0],
                            color=COLORS[hunter],
                            marker=MARKERS[hunter],
                            linewidth=2,
                            label=hunter.replace("_current", "").replace("_", "-"),
                        )
                        for hunter in COLORS
                    ]
                    fig.legend(
                        handles=handles,
                        loc="lower center",
                        ncol=4,
                        frameon=False,
                        bbox_to_anchor=(0.5, 0.01),
                    )
                    title_recovery = "" if metric == "W_rmse" else f" — {recovery} A"
                    fig.suptitle(
                        f"VH algorithms within each preprocessing variant\n"
                        f"{design.replace('_', ' ')} — {side.replace('_', ' ')}"
                        f"{title_recovery}"
                    )
                    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
                    filename = (
                        f"{design}_{side}_{stem}_{recovery}_VH_within_P"
                    )
                    for suffix in ("png", "pdf"):
                        fig.savefig(
                            output / f"{filename}.{suffix}",
                            dpi=220,
                            bbox_inches="tight",
                        )
                    plt.close(fig)


def paired_table_and_plot(data: pd.DataFrame, output: Path) -> None:
    paired = data[
        data.estimator_family.isin(["gplsi_document", "gplsi_anchor_word_profile"])
        & data.A_recovery_method.isin(["current", "poisson_full"])
        & data.status.isin(["ok", "unstable"])
    ].copy()
    id_columns = [
        "dataset",
        "seed",
        "word_decay_parameter",
        "simplex_side",
        "graph_svd_version",
        "preprocessing_variant",
        "vertex_hunter",
        "W_fit_id",
        "design_variant",
    ]
    value_columns = ["A_mean_topic_TV", "held_out_poisson_deviance"]
    wide = paired.pivot_table(
        index=id_columns,
        columns="A_recovery_method",
        values=value_columns,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{recovery}" for metric, recovery in wide.columns]
    wide = wide.reset_index()
    wide = wide.rename(
        columns={
            "simplex_side": "gplsi_side",
            "A_mean_topic_TV_current": "A_current_error",
            "A_mean_topic_TV_poisson_full": "A_poisson_error",
            "held_out_poisson_deviance_current": "A_current_deviance",
            "held_out_poisson_deviance_poisson_full": "A_poisson_deviance",
        }
    )
    wide["delta_error"] = wide.A_poisson_error - wide.A_current_error
    wide["delta_deviance"] = wide.A_poisson_deviance - wide.A_current_deviance
    wide.to_csv(output / "paired_current_vs_poisson_A_recovery.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for side, group in wide.groupby("gplsi_side"):
        axes[0].scatter(group.A_current_error, group.A_poisson_error, alpha=0.55, label=side)
        axes[1].scatter(group.A_current_deviance, group.A_poisson_deviance, alpha=0.55, label=side)
    for ax, x, y, title in [
        (axes[0], "A_current_error", "A_poisson_error", "A topic-TV error"),
        (axes[1], "A_current_deviance", "A_poisson_deviance", "Held-out deviance"),
    ]:
        values = np.concatenate([wide[x].to_numpy(float), wide[y].to_numpy(float)])
        low, high = np.nanmin(values), np.nanmax(values)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Current A recovery")
        ax.set_ylabel("Poisson-full A recovery")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Paired A recovery comparison (identical W within each point)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"paired_current_vs_poisson_A_recovery.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    run_directory = args.run_directory.resolve()
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run_directory / "figures"
    )
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(run_directory / "tidy_results.csv")
    plot_metric(data, output, "W_rmse", "W entrywise RMSE", "W_error_vs_decay", recovery_panels=False)
    plot_metric(data, output, "A_mean_topic_TV", "A mean topic TV error", "A_error_vs_decay", recovery_panels=True)
    plot_metric(data, output, "held_out_poisson_deviance", "Held-out Poisson deviance", "heldout_deviance_vs_decay", recovery_panels=True)
    plot_metric(data, output, "runtime_total", "Runtime (seconds)", "runtime_vs_decay", recovery_panels=True)
    plot_metric(data, output, "A_tail_word_rmse", "Tail-word A RMSE", "tail_A_error_vs_decay", recovery_panels=True)
    plot_failure_and_condition(data, output)
    plot_preprocessing(data, output)
    plot_vh_within_preprocessing(data, output)
    paired_table_and_plot(data, output)
    print(output)


if __name__ == "__main__":
    main()
