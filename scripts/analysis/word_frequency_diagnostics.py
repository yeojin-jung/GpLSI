#!/usr/bin/env python3
"""Word-frequency diagnostics of a dataset, to choose a threshold and judge heterogeneity.

    python scripts/analysis/word_frequency_diagnostics.py configs/cook/production.json
    python scripts/analysis/word_frequency_diagnostics.py configs/dlpfc/production.json
    python scripts/analysis/word_frequency_diagnostics.py configs/dlpfc/production.json --all-genes

Computed on the training counts of the config's first seed (after the 20%
count thinning), one curve per modeled unit (DLPFC section). By default the
vocabulary is the one the protocol fits; for DLPFC that is each section's
2,000-gene training panel. ``--all-genes`` (DLPFC) uses every gene instead,
and marks the genes in that panel.

Tran thresholding (``select_feature_columns``; ``P1_tran_alpha_0p005`` uses
alpha = 0.005). With D the n x p training counts, N_i = sum_j D_ij,
X = D / N (row frequencies) and mean length N_bar = (1/n) sum_i N_i:

    eta_j = (1/n) sum_i X_ij                       mean frequency of word j
    t(alpha) = alpha * sqrt( log(max(n, p)) / (n * N_bar) )
    keep word j  if  eta_j > t(alpha)

``tran_script_exact`` (the pipeline's P1/P3) uses the strict inequality and, if
fewer than 10% of words survive, keeps the top ceil(0.1 p) by eta instead;
rows are not renormalized.

Panels: rank-frequency of eta (log-log) with the cut t(alpha) (``--alpha``,
default 0.005); document frequency and overdispersion (variance / mean of
counts; 1 = Poisson), words kept at that alpha in colour; document lengths; share of words and of
count mass kept as alpha varies. ``summary.csv`` reports n, p, density, mean N,
heterogeneity of eta (max/min over positive words, Gini, mass in the top 1%
and 10% of words), t(0.005) and the words and mass kept at alpha = 0.005,
0.01, 0.1 (with ``--all-genes`` also the overlap with the panel).
Writes ``<run>/figures/word_frequency[_all_genes][_alpha<a>]/`` (suffix only
for a non-default alpha).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix, diags, issparse  # noqa: E402

from gplsi.pipeline import expand_tasks, load_config  # noqa: E402
from gplsi.pipeline.config import config_for_task  # noqa: E402
from gplsi.pipeline.datasets import load_dlpfc_section, prepare_task_data  # noqa: E402
from gplsi.pipeline.panels import panel_indices, rank_features_by_dispersion  # noqa: E402
from gplsi.pipeline.splits import thin_and_split_sparse_counts  # noqa: E402
from gplsi.real_data import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import INK, INK2, SURFACE, run_directory  # noqa: E402

ALPHAS = np.logspace(-4, 0.5, 60)
MARKED = (0.005, 0.01, 0.1)
KEPT, DROPPED, PANEL = "#2a78d6", "#b8b7b0", "#eb6834"


def gini(values: np.ndarray) -> float:
    values = np.sort(values)
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1) @ values / (n * values.sum()))


def tran_threshold(alpha, n: int, p: int, N_bar: float):
    return alpha * np.sqrt(np.log(max(n, p)) / (n * N_bar))


def unit_tasks(config: dict) -> list[dict]:
    """One task per modeled unit (DLPFC section, MERFISH/Xenium unit, else the whole dataset), first seed."""

    seen, tasks = set(), []
    for task in expand_tasks(config):
        key = task.get("unit", task.get("section"))
        if key not in seen:
            seen.add(key)
            tasks.append(task)
    return tasks


def diagnostics(counts, panel: np.ndarray | None = None, alpha: float = 0.005) -> dict:
    counts = csr_matrix(counts, dtype=float) if not issparse(counts) else counts.tocsr().astype(float)
    counts = counts[np.asarray(counts.sum(axis=1)).ravel() > 0]
    n, p = counts.shape
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    eta = np.asarray((diags(1.0 / lengths) @ counts).mean(axis=0)).ravel()
    mean = np.asarray(counts.mean(axis=0)).ravel()
    variance = np.asarray(counts.multiply(counts).mean(axis=0)).ravel() - mean**2
    total = np.asarray(counts.sum(axis=0)).ravel()
    cut = tran_threshold(alpha, n, p, lengths.mean())
    kept_curve = eta[None, :] > tran_threshold(ALPHAS, n, p, lengths.mean())[:, None]
    positive = eta[eta > 0]
    ordered = np.sort(eta)[::-1]
    summary = {
        "n": n, "p": p, "density": float(counts.nnz / (n * p)),
        "N_mean": float(lengths.mean()), "N_median": float(np.median(lengths)),
        "zero_words": int(np.sum(eta == 0)),
        "eta_max_over_min_positive": float(positive.max() / positive.min()),
        "eta_gini": gini(eta),
        "mass_top_1pct_words": float(ordered[: max(1, int(np.ceil(0.01 * p)))].sum() / ordered.sum()),
        "mass_top_10pct_words": float(ordered[: max(1, int(np.ceil(0.10 * p)))].sum() / ordered.sum()),
        f"tran_cut_alpha_{alpha:g}": float(cut),
    }
    for marked in sorted({*MARKED, alpha}):
        kept = eta > tran_threshold(marked, n, p, lengths.mean())
        summary[f"words_kept_alpha_{marked:g}"] = int(kept.sum())
        summary[f"mass_kept_alpha_{marked:g}"] = float(total[kept].sum() / total.sum())
    kept = eta > cut
    if panel is not None:
        in_panel = np.zeros(p, dtype=bool)
        in_panel[panel] = True
        summary["panel_size"] = int(in_panel.sum())
        summary[f"panel_genes_kept_alpha_{alpha:g}"] = int((in_panel & kept).sum())
        summary["panel_mass_share"] = float(total[in_panel].sum() / total.sum())
    return {
        "summary": summary, "eta": eta, "cut": cut, "kept": kept, "panel": panel,
        "document_frequency": np.asarray((counts > 0).mean(axis=0)).ravel(),
        "mean": mean,
        "dispersion": np.divide(variance, mean, out=np.full(p, np.nan), where=mean > 0),
        "lengths": lengths,
        "kept_words": kept_curve.mean(axis=1),
        "kept_mass": (kept_curve * total[None, :]).sum(axis=1) / total.sum(),
    }


def all_gene_counts(config: dict, task: dict):
    """A DLPFC section's full-gene training counts and its dispersion panel (indices)."""

    dataset = config["dataset"]
    section = load_dlpfc_section(REPO_ROOT / dataset["file"], str(task["section"]))
    split = thin_and_split_sparse_counts(
        section["counts"],
        retained_fraction=float(task.get("retained_fraction", 1.0)),
        test_fraction=float(config["heldout_fraction"]),
        seed=int(task["seed"]),
    )
    if "panel_size" not in task:  # Tran-vocabulary config: no dispersion panel to mark
        return split.train, None
    ranking = rank_features_by_dispersion(split.train, detection_fraction=float(dataset["detection_fraction"]))
    return split.train, panel_indices(ranking, int(task["panel_size"]))


def plot(results: dict, out: Path, all_genes: bool, alpha: float) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.4), facecolor=SURFACE)
    line_alpha = min(1.0, 3.0 / len(results) + 0.15)
    point_alpha = max(0.03, 0.4 / len(results))
    for name, result in results.items():
        eta, kept = result["eta"], result["kept"]
        order = np.argsort(-eta, kind="stable")
        positive = order[eta[order] > 0]
        ranks = np.arange(1, len(positive) + 1)
        (line,) = axes[0].loglog(ranks, eta[positive], lw=1, alpha=line_alpha, label=name)
        axes[0].axhline(result["cut"], color=line.get_color(), lw=0.6, ls="--", alpha=line_alpha)
        if all_genes and result["panel"] is not None:
            in_panel = np.isin(positive, result["panel"])
            axes[0].scatter(ranks[in_panel], eta[positive][in_panel], s=1, color=PANEL, alpha=point_alpha, zorder=3)
        colors = np.where(kept, KEPT, DROPPED)
        axes[1].scatter(eta, result["document_frequency"], s=2, c=colors, alpha=point_alpha, linewidths=0)
        axes[2].scatter(result["mean"], result["dispersion"], s=2, c=colors, alpha=point_alpha, linewidths=0)
        axes[3].hist(result["lengths"], bins=50, histtype="step", alpha=line_alpha)
        axes[4].semilogx(ALPHAS, result["kept_words"], lw=1.2, alpha=line_alpha)
        axes[4].semilogx(ALPHAS, result["kept_mass"], lw=1.2, ls="--", alpha=line_alpha)
    for ax in axes[1:3]:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[2].axhline(1, color=INK2, lw=0.8, ls=":")
    axes[4].axvline(alpha, color=INK2, lw=0.8, ls=":")
    axes[4].axhline(0.1, color=INK2, lw=0.6, ls=":")
    titles = [("rank", "mean frequency eta_j", f"Rank-frequency (dashed: Tran cut, alpha = {alpha:g})"),
              ("eta_j", "share of documents with the word", "Document frequency"),
              ("mean count", "variance / mean", "Overdispersion (1 = Poisson)"),
              ("document length N_i", "documents", "Document lengths"),
              ("Tran alpha", "share kept (solid: words, dashed: mass)", f"Tran threshold (dotted: alpha = {alpha:g})")]
    for ax, (xlabel, ylabel, title) in zip(axes, titles):
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=INK2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if 1 < len(results) <= 12:
        axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
    handles = [Line2D([], [], marker="o", ls="", color=KEPT, label=f"kept by Tran (alpha = {alpha:g})"),
               Line2D([], [], marker="o", ls="", color=DROPPED, label="dropped")]
    if all_genes:
        handles.append(Line2D([], [], marker="o", ls="", color=PANEL, label="in the 2,000-gene dispersion panel (rank-frequency)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "word_frequency.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment config")
    parser.add_argument("--all-genes", action="store_true",
                        help="DLPFC: every gene instead of the fitted panel (panel genes are marked)")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="Tran alpha whose cut is drawn and coloured (default 0.005, the protocol's P1)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.all_genes and config["dataset"]["name"] != "dlpfc":
        raise SystemExit("--all-genes applies to DLPFC only (the other datasets fit their full vocabulary)")
    results = {}
    for task in unit_tasks(config):
        name = str(task.get("unit", task.get("section", config["dataset"]["name"])))
        if args.all_genes:
            counts, panel = all_gene_counts(config, task)
            results[name] = diagnostics(counts, panel, args.alpha)
        else:
            data = prepare_task_data(config_for_task(config, task), task)
            results[name] = diagnostics(data.bundle.counts, alpha=args.alpha)
        print(f"{name}: n={results[name]['summary']['n']:,} p={results[name]['summary']['p']:,}")

    folder = "word_frequency_all_genes" if args.all_genes else "word_frequency"
    if args.alpha != 0.005:
        folder += f"_alpha{args.alpha:g}".replace(".", "p")
    out = run_directory(args.config) / "figures" / folder
    out.mkdir(parents=True, exist_ok=True)
    plot(results, out, args.all_genes, args.alpha)
    summary = pd.DataFrame([{"unit": name, **result["summary"]} for name, result in results.items()])
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame([
        {"unit": name, "alpha": alpha, "words_kept": words, "mass_kept": mass}
        for name, result in results.items()
        for alpha, words, mass in zip(ALPHAS, result["kept_words"], result["kept_mass"])
    ]).to_csv(out / "tran_curve.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
