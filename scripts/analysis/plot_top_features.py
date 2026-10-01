#!/usr/bin/env python3
"""Top features of every aligned topic, for large vocabularies (Cooking ingredients, genes).

    python scripts/analysis/plot_top_features.py configs/cook/production.json --K 4 6 7
    python scripts/analysis/plot_top_features.py results/cook_production --K 6 --top 10

For each K, methods are rows and LDA-aligned topics are columns (full-A cosine
Hungarian match; display only). A bar is the feature's probability in the
complete A row; the top ``--top`` are not renormalised, and the panel title
gives their total mass. Zero-probability slots are omitted (a sparse topic
states how many features are nonzero). Ties break by feature index. One fit per
method (smallest successful seed, or ``--seed``); pLSI uses A_current and
GpLSI A_full_Pois. For small vocabularies use ``plot_topic_composition.py``.

Writes ``<run>/figures/top_features_K{K}.png`` and ``top_features.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    TOPIC_COLORS,
    align_to_reference,
    load_selected_rows,
    representative_fits,
    run_directory,
    task_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    parser.add_argument("--K", type=int, nargs="+", help="K values (default: all)")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--hunters", nargs="+")
    parser.add_argument("--preprocessings", nargs="+")
    args = parser.parse_args()

    rows = load_selected_rows(args.target, families=args.families, hunters=args.hunters,
                              preprocessings=args.preprocessings)
    out = run_directory(args.target) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for K in args.K or sorted(rows["K"].unique()):
        fits = representative_fits(rows, int(K), seed=args.seed)
        if not fits:
            continue
        align_to_reference(fits)
        fig, axes = plt.subplots(len(fits), K, figsize=(2.6 * K, 0.22 * args.top * len(fits) + 0.6 * len(fits)),
                                 facecolor=SURFACE, squeeze=False)
        for r, fit in enumerate(fits):
            names = task_data(fit.task_dir)["feature_names"].astype(str)
            for k in range(K):
                profile = fit.A[fit.order[k]]
                top = np.lexsort((np.arange(len(profile)), -profile))[: args.top]
                shown = top[profile[top] > 0]
                ax = axes[r, k]
                ax.barh(range(len(shown))[::-1], profile[shown], color=TOPIC_COLORS[k % len(TOPIC_COLORS)], height=0.7)
                ax.set_yticks(range(len(shown))[::-1], [textwrap.shorten(names[j], 28) for j in shown], fontsize=6.5, color=INK2)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
                ax.tick_params(axis="x", labelsize=6, colors=INK2)
                note = f"top-{args.top} mass {100 * profile[top].sum():.0f}%"
                if len(shown) < args.top:
                    note = f"only {len(shown)} nonzero"
                ax.set_title(f"Topic {k + 1} · {note}", fontsize=7, color=INK, loc="left")
                if k == 0:
                    ax.set_ylabel(fit.label, fontsize=7.5, color=INK)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                records.extend(
                    dict(K=K, method=fit.label, seed=int(fit.row["seed"]), topic=k + 1,
                         original_topic=int(fit.order[k]) + 1, cosine_to_lda=float(fit.cosine[k]),
                         rank=rank + 1, feature=names[j], feature_index=int(j), probability=float(profile[j]))
                    for rank, j in enumerate(top)
                )
        fig.tight_layout()
        fig.savefig(out / f"top_features_K{K}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out / f'top_features_K{K}.png'}")
    pd.DataFrame(records).to_csv(out / "top_features.csv", index=False)


if __name__ == "__main__":
    main()
