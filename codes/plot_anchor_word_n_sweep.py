#!/usr/bin/env python3
"""Plot VH, preprocessing, and A-recovery comparisons over one design axis."""

from __future__ import annotations

import argparse
import html
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
HUNTER_LABELS = {
    "spa_current": "SPA",
    "svs": "SVS",
    "svs_star": "SVS*",
    "pp_spa": "pp-SPA",
}
PREPROCESSING = ["P0_raw", "P1_threshold", "P2_weight", "P3_threshold_weight"]
PREPROCESSING_LABELS = {
    "P0_raw": "P0 · raw",
    "P1_threshold": "P1 · threshold",
    "P2_weight": "P2 · weighted",
    "P3_threshold_weight": "P3 · threshold + weighted",
}
FAMILIES = {
    "document": "gplsi_document",
    "anchor_word": "gplsi_anchor_word_profile",
}
DESIGN_LABELS = {
    "tran_mixed_word_decay_exact": "Original Tran · no graph · no anchors (stress test)",
    "tran_mixed_word_decay_graph_adapted": "Graph-adapted GpLSI · two anchors/topic",
}
RECOVERY_LABELS = {"current": "Regular A recovery", "poisson_full": "Poisson A recovery"}
AXIS_LABELS = {
    "n": "Number of documents n",
    "N": "Words per document N",
    "p": "Requested vocabulary size p",
}
AXIS_TITLES = {
    "n": "number of documents",
    "N": "document length",
    "p": "vocabulary size",
}
AXIS_COLUMNS = {"n": "n", "N": "N", "p": "requested_p"}


def _summarize(frame: pd.DataFrame, metric: str, axis: str) -> pd.DataFrame:
    return (
        frame.groupby([axis, "vertex_hunter"], dropna=False)[metric]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            count="count",
        )
        .reset_index()
    )


def _plot_hunters(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    axis: str,
    axis_values: list[float],
) -> None:
    summary = _summarize(frame, metric, axis)
    for hunter in COLORS:
        group = summary[summary.vertex_hunter.eq(hunter)].sort_values(axis)
        if group.empty:
            continue
        x = group[axis].to_numpy(float)
        ax.plot(
            x,
            group["median"].to_numpy(float),
            color=COLORS[hunter],
            marker=MARKERS[hunter],
            linewidth=2,
            markersize=5,
            label=HUNTER_LABELS[hunter],
        )
        if group["count"].min() >= 3:
            ax.fill_between(
                x,
                group.q25.to_numpy(float),
                group.q75.to_numpy(float),
                color=COLORS[hunter],
                alpha=0.10,
                linewidth=0,
            )
        incomplete = group["count"].to_numpy(int) < 12
        if incomplete.any():
            ax.scatter(
                x[incomplete],
                group["median"].to_numpy(float)[incomplete],
                marker=MARKERS[hunter],
                s=42,
                facecolor="white",
                edgecolor=COLORS[hunter],
                linewidth=1.5,
                zorder=5,
            )
    ax.set_xticks(axis_values)
    ax.grid(alpha=0.22)


def _legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[hunter],
            marker=MARKERS[hunter],
            linewidth=2,
            label=HUNTER_LABELS[hunter],
        )
        for hunter in COLORS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )


def _fixed_setting_text(data: pd.DataFrame, axis: str) -> str:
    specifications = [
        ("n", "n", "n"),
        ("N", "N", "N"),
        ("p", "requested_p", "p"),
        ("a_zipf", "word_decay_parameter", "a_{Zipf}"),
    ]
    parts: list[str] = []
    for logical_axis, column, label in specifications:
        if logical_axis == axis:
            continue
        values = data[column].dropna().unique()
        if len(values) == 1:
            value = float(values[0])
            value_label = f"{value:g}"
            parts.append(rf"${label}={value_label}$")
    return ", ".join(parts)


def plot_A_comparison(data: pd.DataFrame, output: Path, axis: str) -> list[Path]:
    paths: list[Path] = []
    valid = data[data.status.isin(["ok", "unstable"])]
    axis_column = AXIS_COLUMNS[axis]
    axis_values = sorted(valid[axis_column].dropna().unique().astype(float).tolist())
    fixed_text = _fixed_setting_text(valid, axis)
    for design, design_data in valid.groupby("design_variant"):
        for side, family in FAMILIES.items():
            panel_data = design_data[design_data.estimator_family.eq(family)]
            fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), sharex=True, sharey=True)
            for row, recovery in enumerate(["current", "poisson_full"]):
                for column, preprocessing in enumerate(PREPROCESSING):
                    panel = panel_data[
                        panel_data.A_recovery_method.eq(recovery)
                        & panel_data.preprocessing_variant.eq(preprocessing)
                    ]
                    ax = axes[row, column]
                    _plot_hunters(ax, panel, "A_mean_topic_TV", axis_column, axis_values)
                    if row == 0:
                        ax.set_title(PREPROCESSING_LABELS[preprocessing])
                    if column == 0:
                        ax.set_ylabel(f"{RECOVERY_LABELS[recovery]}\nA mean topic-TV error")
                    if row == 1:
                        ax.set_xlabel(AXIS_LABELS[axis])
            _legend(fig)
            fig.suptitle(
                f"A recovery versus {AXIS_TITLES[axis]}\n"
                f"{DESIGN_LABELS.get(design, design)} · {side.replace('_', '-')} simplex · "
                f"{fixed_text} · median/IQR over 12 seeds · hollow markers: incomplete fits",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0.07, 1, 0.93))
            stem = output / f"{design}_{side}_A_recovery_VH_P0_P1_P2_P3_vs_{axis}"
            for suffix in ("png", "pdf"):
                path = stem.with_suffix(f".{suffix}")
                fig.savefig(path, dpi=220, bbox_inches="tight")
                paths.append(path)
            plt.close(fig)
    return paths


def plot_W_comparison(data: pd.DataFrame, output: Path, axis: str) -> list[Path]:
    paths: list[Path] = []
    valid = data[
        data.status.isin(["ok", "unstable"])
        & data.A_recovery_method.eq("current")
    ]
    axis_column = AXIS_COLUMNS[axis]
    axis_values = sorted(valid[axis_column].dropna().unique().astype(float).tolist())
    fixed_text = _fixed_setting_text(valid, axis)
    for design, design_data in valid.groupby("design_variant"):
        for side, family in FAMILIES.items():
            panel_data = design_data[design_data.estimator_family.eq(family)]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
            for ax, preprocessing in zip(axes, PREPROCESSING):
                panel = panel_data[panel_data.preprocessing_variant.eq(preprocessing)]
                _plot_hunters(ax, panel, "W_rmse", axis_column, axis_values)
                ax.set_title(PREPROCESSING_LABELS[preprocessing])
                ax.set_xlabel(AXIS_LABELS[axis])
            axes[0].set_ylabel("W entrywise RMSE")
            _legend(fig)
            fig.suptitle(
                f"W recovery versus {AXIS_TITLES[axis]}\n"
                f"{DESIGN_LABELS.get(design, design)} · {side.replace('_', '-')} simplex · "
                f"{fixed_text} · median/IQR over 12 seeds · hollow markers: incomplete fits",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0.13, 1, 0.88))
            stem = output / f"{design}_{side}_W_recovery_VH_P0_P1_P2_P3_vs_{axis}"
            for suffix in ("png", "pdf"):
                path = stem.with_suffix(f".{suffix}")
                fig.savefig(path, dpi=220, bbox_inches="tight")
                paths.append(path)
            plt.close(fig)
    return paths


def write_summary(data: pd.DataFrame, output: Path, axis: str) -> Path:
    valid = data[
        data.estimator_family.isin(FAMILIES.values())
        & data.status.isin(["ok", "unstable"])
    ]
    groups = [
        "design_variant",
        "simplex_side",
        AXIS_COLUMNS[axis],
        "preprocessing_variant",
        "vertex_hunter",
        "A_recovery_method",
    ]
    summary = (
        valid.groupby(groups, dropna=False)[["W_rmse", "A_mean_topic_TV"]]
        .agg(["count", "median", lambda values: values.quantile(0.25), lambda values: values.quantile(0.75)])
    )
    summary.columns = [
        f"{metric}_{stat if isinstance(stat, str) else stat.__name__}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index().rename(columns={
        "W_rmse_<lambda_0>": "W_rmse_q25",
        "W_rmse_<lambda_1>": "W_rmse_q75",
        "A_mean_topic_TV_<lambda_0>": "A_mean_topic_TV_q25",
        "A_mean_topic_TV_<lambda_1>": "A_mean_topic_TV_q75",
    })
    path = output / f"{axis}_sweep_VH_P_A_recovery_summary.csv"
    summary.to_csv(path, index=False)
    return path


def build_gallery(output: Path, data: pd.DataFrame, axis: str) -> Path:
    suffix = f"vs_{axis}.png"
    ordered = [
        f"tran_mixed_word_decay_graph_adapted_document_A_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_graph_adapted_anchor_word_A_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_exact_document_A_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_exact_anchor_word_A_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_graph_adapted_document_W_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_graph_adapted_anchor_word_W_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_exact_document_W_recovery_VH_P0_P1_P2_P3_{suffix}",
        f"tran_mixed_word_decay_exact_anchor_word_W_recovery_VH_P0_P1_P2_P3_{suffix}",
    ]
    titles = {
        name: name.replace("tran_mixed_word_decay_", "").replace("_", " ").replace(".png", "")
        for name in ordered
    }
    cards = []
    for name in ordered:
        cards.append(
            f'<article><h2>{html.escape(titles[name])}</h2>'
            f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(titles[name])}"></a>'
            f'<p><a href="{html.escape(name)}">Full-resolution PNG</a> · '
            f'<a href="{html.escape(name.replace(".png", ".pdf"))}">PDF</a></p></article>'
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GpLSI n-sweep visual inspection</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #202124; background: #f7f8fa; }}
header {{ max-width: 1100px; margin: 0 auto 26px; }}
h1 {{ margin-bottom: 8px; }}
.note {{ line-height: 1.5; color: #454a50; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 28px; max-width: 1600px; margin: auto; }}
article {{ background: white; border: 1px solid #dfe2e6; border-radius: 12px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,.04); }}
h2 {{ font-size: 18px; margin: 0 0 12px; text-transform: capitalize; }}
img {{ width: 100%; height: auto; border: 1px solid #eee; }}
a {{ color: #075ea8; }}
</style></head><body><header>
<h1>GpLSI recovery as {html.escape(AXIS_TITLES[axis])} changes</h1>
<p class="note">Twelve seeds; {html.escape(AXIS_LABELS[axis])} = {html.escape(', '.join(f'{value:g}' for value in sorted(data[AXIS_COLUMNS[axis]].dropna().unique())))}; {html.escape(_fixed_setting_text(data, axis).replace('$', ''))}. Columns are P0–P3; colors are the four VH algorithms. In A figures, rows compare regular and Poisson recovery. Hollow markers indicate that fewer than 12 fits succeeded.</p>
</header><main class="grid">{''.join(cards)}</main></body></html>"""
    path = output / "visual_inspection_gallery.html"
    path.write_text(page)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--axis", choices=sorted(AXIS_LABELS), default="n")
    args = parser.parse_args()
    run_directory = args.run_directory.resolve()
    output = args.output_directory.resolve() if args.output_directory else run_directory / f"figures_{args.axis}_sweep"
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(run_directory / "tidy_results.csv")
    paths = plot_A_comparison(data, output, args.axis)
    paths += plot_W_comparison(data, output, args.axis)
    paths.append(write_summary(data, output, args.axis))
    paths.append(build_gallery(output, data, args.axis))
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
