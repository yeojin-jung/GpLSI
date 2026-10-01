#!/usr/bin/env python3
"""Xenium: healthy control vs ulcerative colitis from each unit's topic composition.

    python scripts/analysis/xenium/disease_classification.py configs/xenium/production.json

Every patient-condition/timepoint unit is fitted separately, so topics are
first matched across all 25 units of the same method and seed
(``shared.consensus_alignment``: Hungarian cosine matching of A rows to an
iterated mean profile; the 290-gene panel is common). A unit's predictor is
its mean W over cells in that common topic order, in Helmert ILR coordinates
(K - 1 = 11). The task is the 9 healthy-control units (HC) against the 8
pre-treatment UC units (PRE_VDZ_R, PRE_VDZ_NR): one unit per patient, so
leave-one-unit-out is leave-one-patient-out. Classifier: training-fold
StandardScaler + L2 logistic regression (C = 1). Scores: ROC AUC and balanced
accuracy (threshold 0.5) of the 17 held-out probabilities, per seed; the
summary is the mean and SE over seeds. W was fitted without labels on all
cells (transductive), and with 17 units the AUC is exploratory.

References, with the same classifier: the units' annotated coarse cell-type
proportions (13, ILR; uses the labels, an upper reference) and their mean gene
frequencies (290, log; no model). Post-treatment units are left out: they
repeat patients and mix treatment with disease.

Writes ``<run>/figures/xenium/`` (``disease_classification.csv``,
``disease_classification_summary.csv``, ``.png``, ``unit_composition_seed<s>.png``).
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    TOPIC_COLORS,
    consensus_alignment,
    dataset_file,
    ilr,
    load_selected_rows,
    method_sort_key,
    normalize_rows,
    run_directory,
)

from gplsi.pipeline import load_arrays  # noqa: E402

HEALTHY, DISEASE = ["HC"], ["PRE_VDZ_R", "PRE_VDZ_NR"]
CONDITION_ORDER = ["HC", "PRE_VDZ_R", "PRE_VDZ_NR", "POST_VDZ_R", "POST_VDZ_NR"]


def leave_one_out(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    probability = np.empty(len(y))
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=2000))
        model.fit(X[train], y[train])
        probability[i] = model.predict_proba(X[i : i + 1])[0, 1]
    return probability


def scores(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probability = leave_one_out(X, y)
    return {"auc": float(roc_auc_score(y, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5))}


def unit_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adata = ad.read_h5ad(path)
    obs = adata.obs[["unit", "condition", "patient", "cell_type_coarse"]].astype(str)
    units = obs.groupby("unit")[["condition", "patient"]].first()
    cell_types = pd.crosstab(obs["unit"], obs["cell_type_coarse"], normalize="index")
    counts = adata.X.tocsr()
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    frequencies = counts.multiply(1.0 / np.maximum(lengths, 1)[:, None]).tocsr()
    mean_frequency = pd.DataFrame(
        np.vstack([np.asarray(frequencies[(obs["unit"] == u).to_numpy()].mean(axis=0)).ravel() for u in units.index]),
        index=units.index, columns=adata.var_names,
    )
    return units, cell_types, mean_frequency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="Xenium config or its results directory")
    args = parser.parse_args()

    rows = load_selected_rows(args.target)
    rows = rows[rows["status"] == "ok"]
    units, cell_types, mean_frequency = unit_tables(dataset_file(args.target))
    task_units = units[units["condition"].isin(HEALTHY + DISEASE)].index
    y = units.loc[task_units, "condition"].isin(DISEASE).to_numpy(dtype=int)
    out = run_directory(args.target) / "figures" / "xenium"
    out.mkdir(parents=True, exist_ok=True)

    records = [
        {"method": "Annotated cell types (reference)", "seed": None,
         **scores(ilr(cell_types.loc[task_units].to_numpy()), y)},
        {"method": "Mean gene frequencies (reference)", "seed": None,
         **scores(np.log(np.maximum(mean_frequency.loc[task_units].to_numpy(), 1e-8)), y)},
    ]
    compositions = []
    for (label, seed), group in rows.groupby(["label", "seed"]):
        fits = {row["task.unit"]: load_arrays(row) for _, row in group.iterrows()}
        if not set(task_units) <= set(fits):
            print(f"skip {label} seed {seed}: {len(set(task_units) - set(fits))} task units missing")
            continue
        names = sorted(fits)
        orders, _ = consensus_alignment([fits[u]["A_hat"] for u in names])
        composition = pd.DataFrame(
            [normalize_rows(fits[u]["W_hat"])[:, order].mean(axis=0) for u, order in zip(names, orders)],
            index=names,
        )
        compositions.append(composition.assign(method=label, seed=int(seed)))
        records.append({"method": label, "seed": int(seed), **scores(ilr(composition.loc[task_units].to_numpy()), y)})

    table = pd.DataFrame(records)
    table.to_csv(out / "disease_classification.csv", index=False)
    summary = (table.groupby("method", sort=False)[["auc", "balanced_accuracy"]]
               .agg(["mean", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, "count"]))
    summary.columns = [f"{m}_{s}" for m in ("auc", "balanced_accuracy") for s in ("mean", "se", "n_seeds")]
    summary.reset_index().to_csv(out / "disease_classification_summary.csv", index=False)
    print(summary.round(3).to_string())

    methods = [m for m in table["method"].unique() if not m.endswith("(reference)")]
    methods.sort(key=lambda m: method_sort_key(rows[rows["label"] == m].iloc[0]))
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 3, 3.8), facecolor=SURFACE)
    for x, method in enumerate(methods):
        values = table.loc[table["method"] == method, "auc"].to_numpy()
        ax.bar(x, values.mean(), width=0.6, color="#cde2fb")
        ax.scatter(x + np.linspace(-0.15, 0.15, len(values)), values, s=12, color="#184f95", zorder=3)
    for method, style in (("Annotated cell types (reference)", "--"), ("Mean gene frequencies (reference)", ":")):
        ax.axhline(table.loc[table["method"] == method, "auc"].item(), color=INK2, ls=style, lw=1, label=method)
    ax.axhline(0.5, color="#b8b7b0", lw=0.8)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("leave-one-patient-out AUC\n(HC 9 vs pre-treatment UC 8)", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "disease_classification.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    if compositions:
        composition = pd.concat(compositions)
        composition.to_csv(out / "unit_composition.csv")
        seed = int(composition["seed"].min())
        ordered = sorted(units.index, key=lambda u: (CONDITION_ORDER.index(units.loc[u, "condition"]), u))
        fig, axes = plt.subplots(len(methods), 1, figsize=(12, 1.9 * len(methods) + 1.6), facecolor=SURFACE,
                                 squeeze=False, sharex=True)
        for ax, method in zip(axes[:, 0], methods):
            block = composition[(composition["method"] == method) & (composition["seed"] == seed)].drop(columns=["method", "seed"])
            block = block.reindex(ordered)
            bottom = np.zeros(len(block))
            for k in block.columns:
                ax.bar(range(len(block)), block[k], bottom=bottom, color=TOPIC_COLORS[int(k) % len(TOPIC_COLORS)], width=0.8)
                bottom += block[k].fillna(0).to_numpy()
            ax.set_ylabel(method, fontsize=7.5, rotation=0, ha="right", va="center")
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.6, len(ordered) - 0.4)
        axes[-1, 0].set_xticks(range(len(ordered)), [f"{u}" for u in ordered], rotation=60, ha="right", fontsize=6.5)
        fig.legend(handles=[Patch(color=TOPIC_COLORS[k % len(TOPIC_COLORS)], label=f"T{k + 1}") for k in range(composition.shape[1] - 2)],
                   loc="lower center", ncol=12, frameon=False, fontsize=7)
        fig.suptitle(f"Mean W per unit (consensus-aligned topics, seed {seed}); units ordered by condition",
                     fontsize=9, x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(out / f"unit_composition_seed{seed}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
