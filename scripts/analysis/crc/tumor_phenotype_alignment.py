#!/usr/bin/env python3
"""How CRC topics relate to the focal tumor-cell phenotypes (``CELL_TYPE``).

    python scripts/analysis/crc/tumor_phenotype_alignment.py configs/crc/production.json --K 4

Per method (one fit, smallest successful seed or ``--seed``; LDA-aligned topics):

* ``mean_W``: per phenotype, the mean W over its cells (columns sum to one);
* ``argmax``: per phenotype, the distribution of argmax topics over its cells;
* ``spearman``: across patients (all regions pooled, each patient once), the
  Spearman correlation between patient mean W of each topic and the patient's
  share of each phenotype among the same cells. Both sides are compositional;
  this is descriptive, not CCA or inference;
* agreement of argmax topics with the phenotypes: ARI, NMI, AMI and in-sample
  purity next to the majority-label share. No cell-level p-values are given:
  cells share patients and neighbourhoods.

Writes ``<run>/figures/crc/tumor_phenotypes_K{K}.png`` and CSV tables.
The CCA between topics and phenotypes is in ``evaluate_crc_patient_outcomes.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    align_to_reference,
    hard_topics,
    label_agreement,
    load_selected_rows,
    representative_fits,
    run_directory,
    task_data,
)
from reference import region_table, tumor_phenotypes  # noqa: E402


def heatmap(ax, matrix: np.ndarray, rows: list[str], columns: list[str], title: str, cmap: str, vmin, vmax) -> None:
    ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)), rows, fontsize=7, color=INK2)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=6.5, color=INK2)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.tick_params(length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="CRC experiment config or its results directory")
    parser.add_argument("--K", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--hunters", nargs="+")
    parser.add_argument("--preprocessings", nargs="+")
    args = parser.parse_args()

    rows = load_selected_rows(args.target, families=args.families, hunters=args.hunters,
                              preprocessings=args.preprocessings)
    out = run_directory(args.target) / "figures" / "crc"
    out.mkdir(parents=True, exist_ok=True)
    regions = region_table()
    agreement, tables = [], []
    for K in args.K or sorted(rows["K"].unique()):
        K = int(K)
        fits = representative_fits(rows, K, seed=args.seed)
        if not fits:
            continue
        align_to_reference(fits)
        data = task_data(fits[0].task_dir)
        phenotypes = tumor_phenotypes(data["observation_ids"])
        types = sorted(np.unique(phenotypes))
        patients = regions.loc[data["group_ids"].astype(str), "patient"].to_numpy()
        patient_types = pd.crosstab(patients, phenotypes, normalize="index")[types]
        topics = [f"T{k + 1}" for k in range(K)]

        fig, axes = plt.subplots(len(fits), 3, figsize=(15, 2.2 + 0.32 * max(K, len(types)) * len(fits)),
                                 facecolor=SURFACE, squeeze=False)
        for row, fit in enumerate(fits):
            W = fit.W[:, fit.order]
            hard = hard_topics(fit.W, fit.order)
            mean_W = pd.DataFrame(W).groupby(phenotypes).mean().loc[types].T.to_numpy()
            argmax = pd.crosstab(hard, phenotypes, normalize="columns").reindex(index=range(K), fill_value=0)[types].to_numpy()
            patient_W = pd.DataFrame(W).groupby(patients).mean().loc[patient_types.index].to_numpy()
            rho = np.array([[spearmanr(patient_W[:, k], patient_types[t]).statistic for t in types] for k in range(K)])
            heatmap(axes[row, 0], mean_W, topics, types, f"{fit.label}: mean W by phenotype", "Blues", 0, 1)
            heatmap(axes[row, 1], argmax, topics, types, "argmax topic share by phenotype", "Blues", 0, 1)
            heatmap(axes[row, 2], rho, topics, types, f"patient Spearman (n = {len(patient_types)})", "RdBu_r", -1, 1)
            scores = label_agreement(hard, phenotypes)
            agreement.append(dict(K=K, method=fit.label, seed=int(fit.row["seed"]), **scores))
            for name, matrix in (("mean_W", mean_W), ("argmax_share", argmax), ("patient_spearman", rho)):
                frame = pd.DataFrame(matrix, index=range(1, K + 1), columns=types).rename_axis("topic").reset_index()
                tables.append(frame.melt(id_vars="topic", var_name="phenotype").assign(K=K, method=fit.label, statistic=name))
        fig.tight_layout()
        fig.savefig(out / f"tumor_phenotypes_K{K}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)
    pd.DataFrame(agreement).to_csv(out / "tumor_phenotype_agreement.csv", index=False)
    pd.concat(tables).to_csv(out / "tumor_phenotype_tables.csv", index=False)
    print(pd.DataFrame(agreement).round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
