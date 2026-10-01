#!/usr/bin/env python3
"""Held-out fit, spatial coherence and runtime of every method against K.

    python scripts/analysis/plot_metrics_vs_K.py configs/crc/production.json
    python scripts/analysis/plot_metrics_vs_K.py results/cook_production \\
        --hunters svs_star --preprocessings P0_raw P3_tran_then_ke

The protocol's per-fit metrics (docs/protocol.md), one panel each, skipping
any a dataset lacks (Cooking has no coordinates, so no PAS or CHAOS):

* fit: held-out Poisson deviance per held-out count, with predictions mixed
  with the uniform distribution at eps = 1e-3 so fits whose A has exact zeros
  stay finite; fitting time (log axis);
* W: PAS and CHAOS of hard topics (SpatialPCA definitions; lower = more
  coherent), mean Moran's I of the topic weights, W roughness (graph-weighted
  mean ``||W_i - W_j||^2``; lower is smoother, not necessarily better);
* A: max and mean pairwise cosine between topics, topic diversity (top 25).

Points are seed means; bars are ``sd / sqrt(n)``, drawn only with two or more
successful seeds. Seeds change the count split as well as the fit, so bars are
not sampling intervals. Methods are offset around the integer K. pLSI uses
A_current and GpLSI A_full_Pois (``--recovery`` changes that); A does not
affect the W metrics.

Writes ``<run>/figures/metrics_vs_K.{png,pdf}`` and the plotted values as CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    TOPIC_COLORS,
    load_selected_rows,
    method_sort_key,
    run_directory,
    seed_summary,
)

PANELS = {
    "heldout_metrics.heldout_poisson_deviance_per_molecule_smoothed_1e-03": ("Held-out deviance / count (eps = 1e-3)", False),
    "runtime.total": ("Fitting time (s)", True),
    "metrics.spatial_PAS": ("PAS (lower = smoother)", False),
    "metrics.spatial_CHAOS": ("CHAOS (lower = smoother)", False),
    "metrics.spatial_topic_moran_mean": ("Moran's I of W (mean over topics)", False),
    "diagnostics.graph_W_smoothness": ("W roughness", False),
    "metrics.topic_cosine_max": ("Max topic cosine (A)", False),
    "metrics.topic_cosine_mean": ("Mean topic cosine (A)", False),
    "metrics.topic_diversity_top25": ("Topic diversity, top 25 (A)", False),
}
COLUMNS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    parser.add_argument("--families", nargs="+", help="estimator families to show (default: all)")
    parser.add_argument("--hunters", nargs="+", help="GpLSI vertex hunters to show")
    parser.add_argument("--preprocessings", nargs="+", help="GpLSI preprocessings to show")
    parser.add_argument("--recovery", default="A_full_Pois", help="A recovery of the GpLSI rows")
    parser.add_argument("--name", default="metrics_vs_K", help="output file stem")
    args = parser.parse_args()

    rows = load_selected_rows(
        args.target,
        recoveries={"document_gplsi": args.recovery, "anchor_feature_gplsi": args.recovery},
        families=args.families, hunters=args.hunters, preprocessings=args.preprocessings,
    )
    rows = rows[rows["status"] == "ok"]
    metrics = [metric for metric in PANELS if metric in rows and rows[metric].notna().any()]
    summary = seed_summary(rows, ["label", "K"], metrics)
    labels = sorted(rows["label"].unique(), key=lambda label: method_sort_key(rows[rows["label"] == label].iloc[0]))
    offsets = dict(zip(labels, np.linspace(-0.25, 0.25, len(labels)) if len(labels) > 1 else [0.0]))

    nrows = int(np.ceil(len(metrics) / COLUMNS))
    fig, axes = plt.subplots(nrows, COLUMNS, figsize=(4.4 * COLUMNS, 3.3 * nrows + 0.8), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    for ax, metric in zip(axes.flat, metrics):
        title, log = PANELS[metric]
        for index, label in enumerate(labels):
            block = summary[(summary["label"] == label) & (summary["metric"] == metric)].sort_values("K")
            if block.empty:
                continue
            color = TOPIC_COLORS[index % len(TOPIC_COLORS)]
            x = block["K"].to_numpy(dtype=float) + offsets[label]
            ax.plot(x, block["mean"], marker="o", ms=4, lw=1.4, color=color, label=label)
            ax.errorbar(x, block["mean"], yerr=block["se"].fillna(0), fmt="none", ecolor=color, lw=1, capsize=2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(sorted(rows["K"].unique()))
        ax.set_xlabel("K", color=INK2)
        if log:
            ax.set_yscale("log")
        ax.tick_params(colors=INK2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 4), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, (0.6 + 0.25 * (len(names) // 4)) / (3.3 * nrows + 0.8), 1, 1))

    out = run_directory(args.target) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{args.name}.{suffix}", dpi=160, facecolor=SURFACE)
    summary.to_csv(out / f"{args.name}.csv", index=False)
    print(f"wrote {out / args.name}.png")


if __name__ == "__main__":
    main()
