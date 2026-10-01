#!/usr/bin/env python3
"""Recovery of held-out labels by each method, per fitting unit (DLPFC, MERFISH, Xenium).

    python scripts/analysis/plot_label_recovery.py configs/xenium/production.json
    python scripts/analysis/plot_label_recovery.py configs/merfish/production.json --labels cell_type_coarse region_coarse

The pipeline scores every evaluation label of a fitting unit (labels never
enter a fit): ARI and NMI between the dominant topic and the label, and the
balanced accuracy of a cross-validated logistic classifier predicting the
label from W (``metrics.external__<label>__{ari,nmi,cv_balanced_accuracy}``;
for numeric labels such as plaque distance, ``__max_abs_spearman``). Here each
point is one unit (mean over seeds) and the bar is the mean over units; units,
not cells, are the replicates. W does not depend on the A estimator, so pLSI
and GpLSI show one row each.

Writes ``<run>/figures/label_recovery.{png,csv}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import INK, INK2, SURFACE, load_selected_rows, method_sort_key, run_directory  # noqa: E402

PATTERN = re.compile(r"^metrics\.external__(?P<label>.+)__(?P<metric>ari|nmi|cv_balanced_accuracy|max_abs_spearman)$")
METRIC_NAME = {"ari": "ARI", "nmi": "NMI", "cv_balanced_accuracy": "CV balanced accuracy",
               "max_abs_spearman": "max |Spearman| over topics"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="experiment config or its results directory")
    parser.add_argument("--labels", nargs="+", help="labels to show (default: all scored)")
    args = parser.parse_args()

    rows = load_selected_rows(args.target)
    rows = rows[rows["status"] == "ok"].copy()
    unit_column = next((c for c in ("task.unit", "task.section") if c in rows), None)
    rows["unit"] = rows[unit_column].astype(str) if unit_column else "all"
    columns = {c: PATTERN.match(c).groupdict() for c in rows.columns if PATTERN.match(c)}
    if args.labels:
        columns = {c: v for c, v in columns.items() if v["label"] in args.labels}
    if not columns:
        raise SystemExit("no scored labels in these rows")

    long = rows.melt(id_vars=["label", "unit", "seed"], value_vars=list(columns), var_name="column", value_name="value")
    long["annotation"] = long["column"].map(lambda c: columns[c]["label"])
    long["metric"] = long["column"].map(lambda c: columns[c]["metric"])
    per_unit = (long.dropna(subset=["value"])
                .groupby(["annotation", "metric", "label", "unit"])["value"].mean().reset_index())
    out = run_directory(args.target) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    per_unit.rename(columns={"label": "method"}).to_csv(out / "label_recovery.csv", index=False)

    methods = sorted(per_unit["label"].unique(), key=lambda m: method_sort_key(rows[rows["label"] == m].iloc[0]))
    panels = per_unit[["annotation", "metric"]].drop_duplicates().sort_values(["annotation", "metric"]).to_numpy()
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, (annotation, metric) in zip(axes.flat, panels):
        block = per_unit[(per_unit["annotation"] == annotation) & (per_unit["metric"] == metric)]
        for x, method in enumerate(methods):
            values = block.loc[block["label"] == method, "value"].to_numpy()
            if values.size == 0:
                continue
            ax.bar(x, values.mean(), width=0.6, color="#cde2fb")
            jitter = np.linspace(-0.2, 0.2, values.size) if values.size > 1 else np.zeros(1)
            ax.scatter(x + jitter, values, s=9, color="#184f95", zorder=3)
        ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"{annotation}: {METRIC_NAME[metric]}", fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="y", labelsize=7.5, colors=INK2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Held-out label recovery (points: units, mean over seeds; bars: mean over units)",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "label_recovery.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {out / 'label_recovery.png'}")


if __name__ == "__main__":
    main()
