#!/usr/bin/env python3
"""Spatial maps of each method's dominant topic next to a reference label, per fitting unit.

    python scripts/analysis/plot_unit_maps.py configs/merfish/production.json
    python scripts/analysis/plot_unit_maps.py configs/xenium/production.json --units HS44_PRE_VDZ_R --all-strata
    python scripts/analysis/plot_unit_maps.py configs/merfish/production.json --label cell_type_coarse

For DLPFC, MERFISH and Xenium (one model per unit). For each unit, one fit
per method (smallest successful seed, or ``--seed``); each cell is coloured by
its dominant topic (argmax W). A topic takes the colour of the reference class
it overlaps most under a one-to-one Hungarian matching on the unit's cells
(display only; unmatched topics are grey), so a method that recovers the
reference looks like the reference panel. Default references: MERFISH
``region_coarse`` (anatomical regions), Xenium ``neighborhood`` (cell-type
neighbourhoods built by ``prepare_xenium.py``), DLPFC
``layer_guess_reordered``. Coordinates are only comparable within a graph
stratum (section, slide/core), so each row of a figure is one stratum: the
unit's largest by default, all with ``--all-strata``.

Writes ``<run>/figures/maps/<unit>.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    INK,
    SURFACE,
    dataset_file,
    hard_topics,
    load_selected_rows,
    matched_topic_colors,
    representative_fits,
    run_directory,
    task_data,
)

DEFAULT_LABEL = {"merfish": "region_coarse", "xenium": "neighborhood", "dlpfc": "layer_guess_reordered"}
UNLABELLED = "#e4e3de"
REFERENCE_PALETTE = list(plt.get_cmap("tab20").colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="DLPFC, MERFISH or Xenium config or its results directory")
    parser.add_argument("--label", help="reference obs column (default: per dataset, see above)")
    parser.add_argument("--units", nargs="+", help="units to map (default: all)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-strata", action="store_true", help="one row per section / core instead of the largest")
    args = parser.parse_args()

    rows = load_selected_rows(args.target)
    rows = rows[rows["status"] == "ok"].copy()
    unit_column = "task.unit" if "task.unit" in rows else "task.section"
    rows["unit"] = rows[unit_column].astype(str)
    adata = ad.read_h5ad(dataset_file(args.target), backed="r")
    label = args.label or DEFAULT_LABEL[str(rows["dataset"].iloc[0])]
    reference = adata.obs[label].astype(str)
    classes = sorted(c for c in reference.unique() if c not in ("nan", "", "NoAnnotation"))
    palette = {c: REFERENCE_PALETTE[i % len(REFERENCE_PALETTE)] for i, c in enumerate(classes)}
    out = run_directory(args.target) / "figures" / "maps"
    out.mkdir(parents=True, exist_ok=True)

    for unit in args.units or sorted(rows["unit"].unique()):
        block = rows[rows["unit"] == unit]
        if block.empty:
            continue
        K = int(block["K"].iloc[0])
        fits = representative_fits(block, K, seed=args.seed)
        data = task_data(fits[0].task_dir)
        ids = data["observation_ids"].astype(str)
        labels = reference.loc[ids].to_numpy()
        strata = data["group_ids"].astype(str)
        sizes = pd.Series(strata).value_counts()
        shown = list(sizes.index) if args.all_strata else [sizes.index[0]]
        panels = [("Reference: " + label, [palette.get(v, UNLABELLED) for v in labels])]
        for fit in fits:
            hard = hard_topics(fit.W)
            colors = matched_topic_colors(hard, labels, palette, K)
            panels.append((f"{fit.label} (seed {int(fit.row['seed'])})", [colors[k] for k in hard]))
        xy = data["coordinates"]
        fig, axes = plt.subplots(len(shown), len(panels), figsize=(3.3 * len(panels), 3.4 * len(shown) + 0.9),
                                 facecolor=SURFACE, squeeze=False)
        for r, stratum in enumerate(shown):
            mask = strata == stratum
            size = max(0.3, min(4.0, 20000 / mask.sum()))
            for c, (title, colors) in enumerate(panels):
                ax = axes[r, c]
                ax.scatter(xy[mask, 0], xy[mask, 1], c=np.asarray(colors, dtype=object)[mask].tolist(),
                           s=size, linewidths=0, rasterized=True)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8, color=INK, loc="left")
                if c == 0:
                    ax.set_ylabel(f"{stratum}\n({mask.sum():,} cells)", fontsize=7.5, color=INK)
        present = [c for c in classes if np.any(labels == c)]
        fig.legend(handles=[Patch(color=palette[c], label=c) for c in present], loc="lower center",
                   ncol=min(len(present), 8), frameon=False, fontsize=7)
        fig.suptitle(f"{unit}, K = {K}: dominant topic, coloured by its Hungarian-matched {label} class (grey: unmatched)",
                     fontsize=9, x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0.06 + 0.02 * (len(present) // 8), 1, 0.96))
        fig.savefig(out / f"{unit}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out / f'{unit}.png'}")


if __name__ == "__main__":
    main()
