#!/usr/bin/env python3
"""Near-pure document counts per topic, for every method and K.

    python scripts/analysis/near_pure_counts.py configs/spleen/production.json
    python scripts/analysis/near_pure_counts.py results/cook_production --threshold 0.9

For topic k the count is ``sum_i 1{W[i, k] >= threshold}`` after row
normalisation of W (default 0.95, so a document counts for at most one topic).
These are purity-defined documents, not the vertices a hunter selected, and a
larger count is a descriptive property, not a better model. Each count comes
from one fit (smallest successful seed, or ``--seed``), not a seed average.
Topic columns are aligned to the same-K LDA fit by full-A cosine (display only;
counts depend on W alone). Every GpLSI preprocessing and hunter in the run is kept.

Writes ``<run>/figures/near_pure_counts.csv`` (one row per fit and topic) and a
one-table-per-K PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    INK,
    PURITY_THRESHOLD,
    SURFACE,
    align_to_reference,
    load_selected_rows,
    near_pure_counts,
    representative_fits,
    run_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    parser.add_argument("--threshold", type=float, default=PURITY_THRESHOLD)
    parser.add_argument("--seed", type=int, help="preferred seed (default: smallest successful)")
    args = parser.parse_args()

    rows = load_selected_rows(args.target)
    records = []
    for K in sorted(rows["K"].unique()):
        fits = representative_fits(rows, int(K), seed=args.seed)
        if not fits:
            continue
        align_to_reference(fits)
        for fit in fits:
            counts = near_pure_counts(fit.W, args.threshold)[fit.order]
            records.extend(
                dict(K=int(K), method=fit.label, seed=int(fit.row["seed"]), n=len(fit.W), topic=k + 1,
                     count=int(counts[k]), share=float(counts[k] / len(fit.W)), cosine_to_lda=float(fit.cosine[k]))
                for k in range(fit.K)
            )
        failed = rows[(rows["K"] == K) & (rows["status"] != "ok")]
        records.extend(dict(K=int(K), method=label, count=None, status="failed")
                       for label in failed["label"].unique() if label not in {fit.label for fit in fits})
    table = pd.DataFrame(records)
    out = run_directory(args.target) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "near_pure_counts.csv", index=False)

    Ks = sorted(table["K"].unique())
    fig, axes = plt.subplots(len(Ks), 1, figsize=(9, 0.8 + sum(0.28 * table[table["K"] == K]["method"].nunique() + 0.6 for K in Ks)),
                             facecolor=SURFACE, squeeze=False)
    for ax, K in zip(axes[:, 0], Ks):
        block = table[table["K"] == K]
        wide = block.pivot_table(index="method", columns="topic", values="count", sort=False, dropna=False)
        wide = wide.reindex(block["method"].unique())
        cells = [["NA" if pd.isna(v) else f"{int(v):,}" for v in row] for row in wide.to_numpy()]
        ax.axis("off")
        ax.set_title(f"K = {K}: documents with W >= {args.threshold:g}", fontsize=9, color=INK, loc="left")
        if cells and cells[0]:
            table_artist = ax.table(cellText=cells, rowLabels=list(wide.index),
                                    colLabels=[f"Topic {int(c)}" for c in wide.columns], loc="center", cellLoc="right")
            table_artist.auto_set_font_size(False)
            table_artist.set_fontsize(7.5)
    fig.tight_layout()
    fig.savefig(out / "near_pure_counts.png", dpi=160, facecolor=SURFACE)
    print(f"wrote {out / 'near_pure_counts.csv'}")


if __name__ == "__main__":
    main()
