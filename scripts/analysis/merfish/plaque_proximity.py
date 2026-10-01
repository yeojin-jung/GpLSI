#!/usr/bin/env python3
"""MERFISH: how well each cell's topic weights predict its distance to amyloid plaques.

    python scripts/analysis/merfish/plaque_proximity.py configs/merfish/production.json

Only animals carrying 5xFAD have plaques (8 of 15). Per fit (method, seed,
animal), the target is the cell's plaque distance as a within-animal rank in
[0, 1] (the source does not document its physical unit, so only ranks are
used). Predictor: the cell's topic weights W (K = 12), training-fold
standardized, ridge regression (alpha = 1). Cross-validation holds out spatial
blocks, not random cells, because neighbouring cells share their plaque
distance: each section is cut into 5 blocks by k-means on its coordinates,
and fold j holds out block j of every section. Scores on the held-out
predictions of all cells: Spearman correlation with the true rank, and AUC for
"near a plaque" (the 10% of the animal's cells closest to one) using the
prediction as the score.

References, with the same blocks and regression: one-hot annotated coarse cell
type (uses the labels) and all 300 genes' log-normalized expression (no
model). Points are animals (mean over seeds); the summary is mean and SE over
the 8 animals, which are the biological replicates.

Writes ``<run>/figures/merfish/plaque_proximity{.csv,_summary.csv,.png}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (  # noqa: E402
    INK,
    INK2,
    SURFACE,
    dataset_file,
    load_selected_rows,
    method_sort_key,
    normalize_rows,
    run_directory,
    task_data,
)

from gplsi.pipeline import load_arrays  # noqa: E402

FOLDS = 5
NEAR = 0.10


def spatial_folds(coordinates: np.ndarray, strata: np.ndarray, seed: int = 0) -> np.ndarray:
    folds = np.empty(len(strata), dtype=int)
    for stratum in np.unique(strata):
        mask = strata == stratum
        folds[mask] = KMeans(FOLDS, n_init=4, random_state=seed).fit_predict(coordinates[mask])
    return folds


def cross_validate(X: np.ndarray, target: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    prediction = np.empty(len(target))
    for fold in range(FOLDS):
        test = folds == fold
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[~test], target[~test])
        prediction[test] = model.predict(X[test])
    near = target <= NEAR
    return {"spearman": float(spearmanr(prediction, target).statistic),
            "near_plaque_auc": float(roc_auc_score(near, -prediction)),
            "n_cells": int(len(target))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="MERFISH config or its results directory")
    args = parser.parse_args()

    rows = load_selected_rows(args.target)
    rows = rows[rows["status"] == "ok"]
    adata = ad.read_h5ad(dataset_file(args.target))
    obs = adata.obs
    plaque_units = sorted(obs.loc[obs["genotype"].astype(str).str.contains("5xFAD"), "unit"].astype(str).unique())
    out = run_directory(args.target) / "figures" / "merfish"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    references_done: set[tuple[str, int]] = set()
    for _, row in rows[rows["task.unit"].astype(str).isin(plaque_units)].iterrows():
        unit, seed = str(row["task.unit"]), int(row["seed"])
        data = task_data(row["task_dir"])
        ids = data["observation_ids"].astype(str)
        distance = obs.loc[ids, "plaque_distance"].to_numpy(dtype=float)
        usable = np.isfinite(distance)
        target = np.empty(len(ids))
        target[usable] = (rankdata(distance[usable]) - 1) / max(usable.sum() - 1, 1)
        folds = spatial_folds(data["coordinates"], data["group_ids"].astype(str))
        W = normalize_rows(load_arrays(row)["W_hat"])
        records.append({"method": row["label"], "unit": unit, "seed": seed,
                        **cross_validate(W[usable], target[usable], folds[usable])})
        if (unit, seed) not in references_done:  # same cells and folds for every method of this task
            references_done.add((unit, seed))
            cell_types = pd.get_dummies(obs.loc[ids, "cell_type_coarse"].astype(str)).to_numpy(dtype=float)
            counts = adata[ids].X
            lengths = np.asarray(counts.sum(axis=1)).ravel()
            expression = np.log1p(1e4 * np.asarray(counts.multiply(1.0 / np.maximum(lengths, 1)[:, None]).todense()))
            for name, X in (("Annotated cell types (reference)", cell_types), ("All genes (reference)", expression)):
                records.append({"method": name, "unit": unit, "seed": seed,
                                **cross_validate(X[usable], target[usable], folds[usable])})

    table = pd.DataFrame(records)
    table.to_csv(out / "plaque_proximity.csv", index=False)
    per_animal = table.groupby(["method", "unit"])[["spearman", "near_plaque_auc"]].mean().reset_index()
    summary = per_animal.groupby("method")[["spearman", "near_plaque_auc"]].agg(
        ["mean", lambda v: v.std(ddof=1) / np.sqrt(len(v)), "count"])
    summary.columns = [f"{m}_{s}" for m in ("spearman", "near_plaque_auc") for s in ("mean", "se", "n_animals")]
    summary.reset_index().to_csv(out / "plaque_proximity_summary.csv", index=False)
    print(summary.round(3).to_string())

    methods = [m for m in per_animal["method"].unique() if not m.endswith("(reference)")]
    methods.sort(key=lambda m: method_sort_key(rows[rows["label"] == m].iloc[0]))
    fig, axes = plt.subplots(1, 2, figsize=(2 * (1.1 * len(methods) + 2.5), 3.8), facecolor=SURFACE)
    for ax, (metric, name) in zip(axes, (("spearman", "Spearman (predicted vs true rank)"),
                                          ("near_plaque_auc", f"AUC, nearest {int(NEAR * 100)}% of cells"))):
        for x, method in enumerate(methods):
            values = per_animal.loc[per_animal["method"] == method, metric].to_numpy()
            ax.bar(x, values.mean(), width=0.6, color="#cde2fb")
            ax.scatter(x + np.linspace(-0.18, 0.18, len(values)), values, s=10, color="#184f95", zorder=3)
        for method, style in (("Annotated cell types (reference)", "--"), ("All genes (reference)", ":")):
            values = per_animal.loc[per_animal["method"] == method, metric]
            if len(values):
                ax.axhline(values.mean(), color=INK2, ls=style, lw=1, label=method)
        ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].axhline(0.5, color="#b8b7b0", lw=0.8)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Plaque proximity from W, spatial-block CV within each 5xFAD animal (points: animals)",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "plaque_proximity.png", dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
