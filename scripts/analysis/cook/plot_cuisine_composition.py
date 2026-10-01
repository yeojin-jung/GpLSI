#!/usr/bin/env python3
"""Average topic composition of each cuisine, per method and K (What's Cooking).

    python scripts/analysis/cook/plot_cuisine_composition.py configs/cook/production.json --K 4 5 6 7
    python scripts/analysis/cook/plot_cuisine_composition.py results/cook_production \\
        --hunters svs_star --preprocessings P0_raw P2_ke_weighted P3_tran_then_ke

For cuisine c and topic k the bar segment is the mean of row-normalised
``W[i, k]`` over the recipes of c (soft weights, recipes weighted equally; each
bar sums to one). It is the topic mix within a cuisine, not the cuisine mix
within a topic. Within each panel cuisines are ordered so that similar
profiles are adjacent (Jensen-Shannon distance, average linkage with optimal
leaf ordering), so rows can differ between panels. Topics are aligned to the
same-K LDA fit by full-A cosine (display only). One fit per method (smallest
successful seed, or ``--seed``). Cuisines come from the task's ``group_ids``,
which follow any rows a preprocessing dropped.

Top ingredients of the same aligned topics: ``plot_top_features.py``.
Writes ``<run>/figures/cook/cuisine_mean_W_K{K}.png`` and ``cuisine_mean_W.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    TOPIC_COLORS,
    align_to_reference,
    group_composition,
    load_selected_rows,
    representative_fits,
    run_directory,
    similarity_order,
    task_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="Cooking experiment config or its results directory")
    parser.add_argument("--K", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--hunters", nargs="+")
    parser.add_argument("--preprocessings", nargs="+")
    args = parser.parse_args()

    rows = load_selected_rows(args.target, families=args.families, hunters=args.hunters,
                              preprocessings=args.preprocessings)
    out = run_directory(args.target) / "figures" / "cook"
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for K in args.K or sorted(rows["K"].unique()):
        K = int(K)
        fits = representative_fits(rows, K, seed=args.seed)
        if not fits:
            continue
        align_to_reference(fits)
        columns = min(4, len(fits))
        nrows = int(np.ceil(len(fits) / columns))
        fig, axes = plt.subplots(nrows, columns, figsize=(3.6 * columns, 5.2 * nrows + 0.6), facecolor=SURFACE, squeeze=False)
        for ax, fit in zip(axes.flat, fits):
            cuisines = task_data(fit.task_dir)["group_ids"].astype(str)
            if len(cuisines) != len(fit.W):
                raise ValueError(f"{fit.label}: W rows do not match the task's cuisine labels")
            table = group_composition(fit.W[:, fit.order], cuisines)
            table = table[table["statistic"] == "mean_W"]
            wide = table.pivot(index="group", columns="topic", values="value")
            n = table.groupby("group")["n"].first()
            order = similarity_order(list(wide.index), wide.to_numpy())
            wide = wide.iloc[order]
            left = np.zeros(len(wide))
            for k in range(K):
                ax.barh(range(len(wide))[::-1], wide[k], left=left, height=0.75, color=TOPIC_COLORS[k % len(TOPIC_COLORS)])
                left += wide[k].to_numpy()
            ax.set_yticks(range(len(wide))[::-1], [f"{c} ({n[c]:,})" for c in wide.index], fontsize=7, color=INK2)
            ax.set_xlim(0, 1)
            ax.tick_params(axis="x", labelsize=7, colors=INK2)
            ax.set_title(fit.label, fontsize=8.5, color=INK, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            rank = {cuisine: position + 1 for position, cuisine in enumerate(wide.index)}
            tables.append(table.assign(K=K, method=fit.label, seed=int(fit.row["seed"]), topic=table["topic"] + 1,
                                       display_rank=table["group"].map(rank)).rename(columns={"group": "cuisine"}))
        for ax in list(axes.flat)[len(fits):]:
            ax.axis("off")
        handles = [Patch(color=TOPIC_COLORS[k % len(TOPIC_COLORS)], label=f"Topic {k + 1}") for k in range(K)]
        fig.legend(handles=handles, loc="lower center", ncol=K, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / f"cuisine_mean_W_K{K}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)
    pd.concat(tables).to_csv(out / "cuisine_mean_W.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
