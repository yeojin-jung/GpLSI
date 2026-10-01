#!/usr/bin/env python3
"""Spleen topics against the manual compartments (B-zone, marginal zone, PALS, red pulp).

    python scripts/analysis/spleen/compartment_comparison.py configs/spleen/production.json --K 4 5 7
    python scripts/analysis/spleen/compartment_comparison.py results/spleen_production \\
        --map-group BALBc-1 --hunters svs_star --preprocessings P0_raw

Labels are the CytoCommunity manual compartments frozen by
``scripts/data/prepare_spleen_compartment_annotations.py`` into
``data/spleen/dataset/compartments/<group>.csv.gz``, joined by original cell
id; they never enter a fit. Unannotated cells are excluded from every
denominator and drawn grey. Per K, with one fit per method (smallest
successful seed, or ``--seed``) and LDA-aligned topics (full-A cosine):

* ``compartment_agreement.csv``: ``evaluate_spleen_embedding`` on all labelled
  cells: ARI, AMI, the K = 4 global Hungarian exact-match accuracy (not
  cross-validated), per-spleen scores, and the leave-one-spleen-out ridge
  probe on sqrt(W) when the run has several spleens (transductive: W was fit
  without labels on all spleens);
* ``zone_shares_K{K}.png``: reference-compartment shares next to each method's
  argmax-topic shares on the labelled cells of ``--map-group``;
* ``spatial_K{K}.png``: the ``--map-group`` tissue, reference panel first, then
  each method's argmax topic. A topic takes a compartment's colour when the
  Hungarian overlap matches it to that compartment (display only; the match
  is not an identity); other topics are grey.

Writes ``<run>/figures/spleen/``.
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
    align_to_reference,
    hard_topics,
    load_selected_rows,
    matched_topic_colors,
    raw_cell_ids,
    representative_fits,
    run_directory,
    task_data,
)
from spleen_embedding_evaluation import COMPARTMENTS, UNLABELED, evaluate_spleen_embedding  # noqa: E402

from gplsi.real_data import DATA_ROOT  # noqa: E402

COMPARTMENT_ROOT = DATA_ROOT / "spleen" / "dataset" / "compartments"
# High-contrast reference palette of the handoff maps.
COMPARTMENT_COLORS = {"B-zone": "#E69F00", "marginal zone": "#CC3377", "PALS": "#78B500", "red pulp": "#0072B2"}
UNLABELED_COLOR = "#D0D0D0"


def compartments(observation_ids: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    ids = raw_cell_ids(observation_ids)
    groups = np.asarray(group_ids).astype(str)
    output = np.empty(len(ids), dtype=object)
    for group in pd.unique(groups):
        path = COMPARTMENT_ROOT / f"{group}.csv.gz"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/data/prepare_spleen_compartment_annotations.py")
        table = pd.read_csv(path).set_index("raw_cell_id")["compartment"]
        mask = groups == group
        output[mask] = table.loc[ids[mask]].to_numpy()
    return output.astype(str)


def flat(result: dict) -> dict:
    keep = {key: value for key, value in result.items() if not isinstance(value, (dict, list)) or key == "loso_probe"}
    probe = keep.pop("loso_probe", None) or {}
    keep.update({f"loso_{key}": value for key, value in probe.items() if not isinstance(value, (dict, list))})
    return keep


def zone_figure(fits, hards, labels, mask, K, group, out: Path) -> pd.DataFrame:
    labelled = mask & (labels != UNLABELED)
    reference = pd.Series(labels[labelled]).value_counts(normalize=True).reindex(COMPARTMENTS, fill_value=0)
    records = [dict(panel="Reference", category=c, share=float(reference[c])) for c in COMPARTMENTS]
    fig, ax = plt.subplots(figsize=(1.1 * (len(fits) + 1) + 2, 4), facecolor=SURFACE)
    bottom = 0.0
    for c in COMPARTMENTS:
        ax.bar(0, reference[c], bottom=bottom, color=COMPARTMENT_COLORS[c], width=0.6)
        bottom += reference[c]
    for column, (fit, hard) in enumerate(zip(fits, hards), start=1):
        colors = matched_topic_colors(hard[labelled], labels[labelled], COMPARTMENT_COLORS, K)
        shares = np.bincount(hard[labelled], minlength=K) / labelled.sum()
        bottom = 0.0
        for k in range(K):
            ax.bar(column, shares[k], bottom=bottom, color=colors[k], width=0.6, edgecolor=SURFACE, linewidth=0.5)
            if shares[k] >= 0.06:
                ax.text(column, bottom + shares[k] / 2, f"T{k + 1}", ha="center", va="center", fontsize=6.5, color="white")
            bottom += shares[k]
            records.append(dict(panel=fit.label, category=f"Topic {k + 1}", share=float(shares[k])))
    ax.set_xticks(range(len(fits) + 1), ["Reference"] + [fit.label for fit in fits], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(f"share of labelled {group} cells (argmax)", fontsize=8, color=INK2)
    ax.set_ylim(0, 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=v, label=k) for k, v in COMPARTMENT_COLORS.items()], frameon=False, fontsize=7,
              loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out / f"zone_shares_K{K}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(records).assign(K=K, group=group, n_labelled=int(labelled.sum()))


def spatial_figure(fits, hards, labels, mask, xy, K, group, out: Path) -> None:
    labelled = mask & (labels != UNLABELED)
    columns = min(4, len(fits) + 1)
    rows = int(np.ceil((len(fits) + 1) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.2 * rows + 0.6), facecolor=SURFACE, squeeze=False)
    panels = [("Reference compartments", [COMPARTMENT_COLORS.get(c, UNLABELED_COLOR) for c in labels[mask]])]
    for fit, hard in zip(fits, hards):
        colors = matched_topic_colors(hard[labelled], labels[labelled], COMPARTMENT_COLORS, K)
        panels.append((fit.label, [colors[k] for k in hard[mask]]))
    for ax, (title, colors) in zip(axes.flat, panels):
        ax.scatter(xy[mask, 0], xy[mask, 1], c=colors, s=0.8, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    handles = [Patch(color=v, label=k) for k, v in COMPARTMENT_COLORS.items()] + [Patch(color=UNLABELED_COLOR, label="unannotated")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8,
               title="Reference; a topic takes the colour of its Hungarian-matched compartment, others grey")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / f"spatial_K{K}.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="spleen experiment config or its results directory")
    parser.add_argument("--K", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--map-group", default="BALBc-1", help="spleen shown in the maps and zone bars")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--hunters", nargs="+")
    parser.add_argument("--preprocessings", nargs="+")
    args = parser.parse_args()

    rows = load_selected_rows(args.target, families=args.families, hunters=args.hunters,
                              preprocessings=args.preprocessings)
    out = run_directory(args.target) / "figures" / "spleen"
    out.mkdir(parents=True, exist_ok=True)
    agreement, shares = [], []
    for K in args.K or sorted(rows["K"].unique()):
        K = int(K)
        fits = representative_fits(rows, K, seed=args.seed)
        if not fits:
            continue
        align_to_reference(fits)
        data = task_data(fits[0].task_dir)
        groups = data["group_ids"].astype(str)
        labels = compartments(data["observation_ids"], groups)
        several = len(np.unique(groups)) > 1
        hards = [hard_topics(fit.W, fit.order) for fit in fits]
        for fit, hard in zip(fits, hards):
            arguments = dict(hard_topics=hard, hard_topic_count=K)
            try:
                result = evaluate_spleen_embedding(fit.W[:, fit.order], labels, groups, compute_loso=several, **arguments)
            except ValueError as error:  # a held-out spleen lacks a compartment (small subsets)
                result = evaluate_spleen_embedding(fit.W[:, fit.order], labels, groups, compute_loso=False, **arguments)
                result["loso_error"] = str(error)
            agreement.append(dict(K=K, method=fit.label, seed=int(fit.row["seed"]), **flat(result)))
        mask = groups == args.map_group
        if mask.any():
            shares.append(zone_figure(fits, hards, labels, mask, K, args.map_group, out))
            spatial_figure(fits, hards, labels, mask, data["coordinates"], K, args.map_group, out)
    table = pd.DataFrame(agreement)
    table.to_csv(out / "compartment_agreement.csv", index=False)
    if shares:
        pd.concat(shares).to_csv(out / "zone_shares.csv", index=False)
    columns = [c for c in ("K", "method", "ari", "ami",
                           "hungarian_exact_match_accuracy", "loso_balanced_accuracy") if c in table]
    print(table[columns].round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
