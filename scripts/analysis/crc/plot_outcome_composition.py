#!/usr/bin/env python3
"""CRC topic composition by outcome group, and example tissue maps.

    python scripts/analysis/crc/plot_outcome_composition.py configs/crc/production.json --K 4 5 6
    python scripts/analysis/crc/plot_outcome_composition.py results/crc_production --K 4 \\
        --hunters svs_star --preprocessings P0_raw

Bars (per K and endpoint): for each patient, all modeled cells of all their
regions are pooled; the patient's mean W and argmax-topic shares are computed;
each outcome group then averages its patients with equal weight. Descriptive,
in-sample compositions, not predictions or tests. Recurrence has 103 patients
and primary outcome 109 on the full data; 0/1 codes are kept as supplied.

Maps (per K and endpoint): two regions per outcome class, nearest the class's
median modeled-cell count, from distinct patients, chosen before looking at W.
The first row is the focal tumor-cell ``CELL_TYPE``; below it each method's
argmax topic, coloured by LDA-aligned topic (full-A cosine; display only). The
reference labels and the topics have separate legends: equal colours do not
assert equivalence.

One fit per method (smallest successful seed, or ``--seed``); pLSI uses
A_current and GpLSI A_full_Pois, which affect only the alignment. Writes
``<run>/figures/crc/``.
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
    hard_topics,
    load_selected_rows,
    representative_fits,
    run_directory,
    task_data,
)
from reference import TARGETS, example_regions, region_table, tumor_phenotypes  # noqa: E402

PHENOTYPE_COLORS = ["#332288", "#88CCEE", "#44AA99", "#117733", "#DDCC77", "#CC6677", "#AA4499"]


def composition_figure(fits, groups, patients, codes, target, K, out: Path) -> pd.DataFrame:
    observed = ~pd.isna(codes)
    tables = []
    fig, axes = plt.subplots(2, len(fits), figsize=(1.9 * len(fits) + 1.5, 5.4), facecolor=SURFACE, squeeze=False)
    for column, fit in enumerate(fits):
        W = fit.W[:, fit.order]
        table = group_composition(W[observed], codes[observed].astype(int).astype(str), parents=patients[observed])
        table.insert(0, "method", fit.label)
        tables.append(table)
        for row, statistic in enumerate(("mean_W", "argmax_share")):
            ax = axes[row, column]
            block = table[table["statistic"] == statistic]
            classes = sorted(block["group"].unique())
            bottom = np.zeros(len(classes))
            for k in range(K):
                values = np.array([block[(block["group"] == c) & (block["topic"] == k)]["value"].item() for c in classes])
                ax.bar(range(len(classes)), values, bottom=bottom, width=0.6, color=TOPIC_COLORS[k % len(TOPIC_COLORS)])
                bottom += values
            n = [block[block["group"] == c]["n"].iloc[0] for c in classes]
            ax.set_xticks(range(len(classes)), [f"{c}\n(n={m})" for c, m in zip(classes, n)], fontsize=7, color=INK2)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="y", labelsize=7, colors=INK2)
            if row == 0:
                ax.set_title(fit.label, fontsize=8, color=INK, loc="left")
            if column == 0:
                ax.set_ylabel({"mean_W": "patient mean W", "argmax_share": "patient argmax share"}[statistic], fontsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles = [Patch(color=TOPIC_COLORS[k % len(TOPIC_COLORS)], label=f"Topic {k + 1}") for k in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=K, frameon=False, fontsize=8)
    fig.suptitle(f"K = {K}, {target} (0/1 codes as supplied; patients weighted equally)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out / f"outcome_composition_K{K}_{target}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return pd.concat(tables).assign(K=K, target=target)


def map_figure(fits, data, phenotypes, target, K, out: Path) -> None:
    regions_by_class = example_regions(data["group_ids"], target)
    regions = [(code, region) for code, chosen in sorted(regions_by_class.items()) for region in chosen]
    groups = data["group_ids"].astype(str)
    xy = data["coordinates"]
    types = sorted(np.unique(phenotypes))
    type_color = dict(zip(types, PHENOTYPE_COLORS * 2))
    fig, axes = plt.subplots(len(fits) + 1, len(regions), figsize=(3.0 * len(regions) + 1.8, 2.8 * (len(fits) + 1)),
                             facecolor=SURFACE, squeeze=False)
    hards = [hard_topics(fit.W, fit.order) for fit in fits]
    for column, (code, region) in enumerate(regions):
        mask = groups == region
        panels = [("Tumor-cell type", [type_color[t] for t in phenotypes[mask]])]
        panels += [(fit.label, [TOPIC_COLORS[k % len(TOPIC_COLORS)] for k in hard[mask]]) for fit, hard in zip(fits, hards)]
        for row, (name, colors) in enumerate(panels):
            ax = axes[row, column]
            ax.scatter(xy[mask, 0], xy[mask, 1], c=colors, s=4, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{region.replace('Charville_', '')}\n{target} = {code}", fontsize=7.5, color=INK)
            if column == 0:
                ax.set_ylabel(name, fontsize=8, color=INK)
    type_handles = [Patch(color=type_color[t], label=t) for t in types]
    topic_handles = [Patch(color=TOPIC_COLORS[k % len(TOPIC_COLORS)], label=f"Topic {k + 1}") for k in range(K)]
    fig.legend(handles=type_handles, loc="lower left", ncol=4, frameon=False, fontsize=7, title="Focal tumor-cell type")
    fig.legend(handles=topic_handles, loc="lower right", ncol=min(K, 6), frameon=False, fontsize=7, title="Topic (LDA-aligned)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / f"tissue_maps_K{K}_{target}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="CRC experiment config or its results directory")
    parser.add_argument("--K", type=int, nargs="+", help="K values (default: all)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--hunters", nargs="+")
    parser.add_argument("--preprocessings", nargs="+")
    parser.add_argument("--no-maps", action="store_true")
    args = parser.parse_args()

    rows = load_selected_rows(args.target, families=args.families, hunters=args.hunters,
                              preprocessings=args.preprocessings)
    out = run_directory(args.target) / "figures" / "crc"
    out.mkdir(parents=True, exist_ok=True)
    regions = region_table()
    tables = []
    for K in args.K or sorted(rows["K"].unique()):
        fits = representative_fits(rows, int(K), seed=args.seed)
        if not fits:
            continue
        align_to_reference(fits)
        data = task_data(fits[0].task_dir)
        for fit in fits[1:]:
            if not np.array_equal(task_data(fit.task_dir)["observation_ids"], data["observation_ids"]):
                raise ValueError("fits at one K were made on different rows")
        groups = data["group_ids"].astype(str)
        patients = regions.loc[groups, "patient"].to_numpy()
        phenotypes = None if args.no_maps else tumor_phenotypes(data["observation_ids"])
        for target in TARGETS:
            codes = regions.loc[groups, target].to_numpy(dtype=float)
            tables.append(composition_figure(fits, groups, patients, codes, target, int(K), out))
            if phenotypes is not None:
                map_figure(fits, data, phenotypes, target, int(K), out)
    pd.concat(tables).to_csv(out / "outcome_composition.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
