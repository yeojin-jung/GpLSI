#!/usr/bin/env python3
"""DLPFC results of the paper protocol: layer recovery per section, and dominant-topic maps.

    python scripts/analysis/dlpfc/plot_production.py configs/dlpfc/production.json
    python scripts/analysis/dlpfc/plot_production.py results/dlpfc_production --sections 151673 151507

* ``layer_agreement.png`` / ``.csv``: ARI and NMI of argmax W against the
  manual layers (``layer_guess_reordered``, never used in fitting; computed
  by the pipeline on labelled spots), per method: each point is one section
  (mean over seeds), the bar the mean over sections. K = 7 sections and the
  K = 5 Br5595 sections are marked separately.
* ``maps_<section>.png``: the manual layers, then each method's argmax topic
  for one seed (smallest successful, or ``--seed``). A topic takes a layer's
  colour when the Hungarian overlap matches it to that layer (display only);
  unmatched topics are grey.

pLSI uses A_current and GpLSI A_full_Pois for labels; W, and so the
agreement and maps, do not depend on the A estimator. The ablation designs
(``configs/dlpfc/ablation/``) have their own scripts in this folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    hard_topics,
    load_selected_rows,
    matched_topic_colors,
    method_sort_key,
    representative_fits,
    run_directory,
    task_data,
)
from plot_spatial_maps import LAYER_COLORS, LAYERS, UNLABELLED, layer_labels  # noqa: E402

ARI = "metrics.external__layer_guess_reordered__ari"
NMI = "metrics.external__layer_guess_reordered__nmi"
BR5595 = {"151669", "151670", "151671", "151672"}


def agreement_figure(rows: pd.DataFrame, out: Path) -> None:
    rows = rows[rows["status"] == "ok"].copy()
    rows["section"] = rows["task.section"].astype(str)
    per_section = rows.groupby(["label", "section"])[[ARI, NMI]].mean().reset_index()
    per_section.to_csv(out / "layer_agreement.csv", index=False)
    labels = sorted(per_section["label"].unique(), key=lambda label: method_sort_key(rows[rows["label"] == label].iloc[0]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE)
    for ax, (metric, name) in zip(axes, ((ARI, "ARI"), (NMI, "NMI"))):
        for x, label in enumerate(labels):
            block = per_section[per_section["label"] == label]
            ax.bar(x, block[metric].mean(), width=0.6, color="#cde2fb")
            for marker, mask in (("o", ~block["section"].isin(BR5595)), ("^", block["section"].isin(BR5595))):
                values = block.loc[mask, metric]
                jitter = np.linspace(-0.18, 0.18, max(len(values), 1))[: len(values)]
                ax.scatter(x + jitter, values, s=14, marker=marker, color="#184f95", zorder=3)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"layer {name} (mean over seeds)", fontsize=8.5, color=INK2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color="#184f95", label="K = 7 section"),
               Line2D([], [], marker="^", ls="", color="#184f95", label="K = 5 section (Br5595)")]
    axes[1].legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "layer_agreement.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def map_figure(rows: pd.DataFrame, section: str, seed: int | None, out: Path) -> None:
    rows = rows[rows["task.section"].astype(str) == section]
    if rows.empty:
        return
    K = int(rows["K"].iloc[0])
    fits = representative_fits(rows, K, seed=seed)
    data = task_data(fits[0].task_dir)
    labels = layer_labels(data["observation_ids"])
    xy = data["coordinates"]
    layer_color = dict(zip(LAYERS, LAYER_COLORS))
    panels = [("Manual layers", [layer_color.get(label, UNLABELLED) for label in labels])]
    for fit in fits:
        hard = hard_topics(fit.W)
        colors = matched_topic_colors(hard, labels, layer_color, K)
        panels.append((f"{fit.label} (seed {int(fit.row['seed'])})", [colors[k] for k in hard]))
    columns = 4
    nrows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(3.6 * columns, 3.8 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, (title, colors) in zip(axes.flat, panels):
        ax.scatter(xy[:, 0], -xy[:, 1], c=colors, s=3, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Section {section}, K = {K}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"maps_{section}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="DLPFC protocol config or its results directory")
    parser.add_argument("--sections", nargs="+", help="sections to map (default: all)")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    rows = load_selected_rows(args.target)
    out = run_directory(args.target) / "figures" / "dlpfc"
    out.mkdir(parents=True, exist_ok=True)
    agreement_figure(rows, out)
    for section in args.sections or sorted(rows["task.section"].astype(str).unique()):
        map_figure(rows, section, args.seed, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
