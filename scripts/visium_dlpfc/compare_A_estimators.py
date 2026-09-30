"""Per-topic comparison of the three A estimators on one shared W (core design).

    python scripts/visium_dlpfc/compare_A_estimators.py [--geometry gplsi_document__P0_raw__svs]

A_current, A_full_L2 and A_full_Pois are all fitted from the same W, so topic k is the
same topic under every estimator and no matching is needed. For each of the 9 core tasks
and each topic this records:

* mass: the topic's share of total W mass (near-empty topics are < 1%);
* entropy: Shannon entropy of the topic's gene distribution divided by log(p) (the
  normalisation of topic_entropy_mean in metrics.py), and the effective number of genes
  exp(H);
* zeros: genes with exactly zero probability in the topic;
* JSD (base 2) and top-20-gene Jaccard overlap between each pair of estimators;
* between-topic similarity within each estimator: Pearson correlation of topic profiles on
  log2 enrichment log2(A_k / mean_k A), with a 1e-6 floor, over the 1,000 genes with the
  highest mean A_current.

Needs results/visium_dlpfc/summary/all_records.csv (summarize.py) and the core Poisson
refit. Writes CSVs and figures to results/visium_dlpfc/diagnostics/core/A_compare/.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ESTIMATORS = ["A_current", "A_full_L2", "A_full_Pois"]
COLORS = {"A_current": "#2a78d6", "A_full_L2": "#eb6834", "A_full_Pois": "#1baf7a"}
# A_full_L2 vs A_full_Pois is nearly identical to A_current vs A_full_Pois (A_current ~ A_full_L2 for
# good hunters), so the figures draw only the two contrasts against A_current.
PAIR_COLORS = {("A_current", "A_full_L2"): "#eb6834", ("A_current", "A_full_Pois"): "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLOOR = 1e-15
TOP_N = 20
N_CORR_GENES = 1000

plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def normalise(A: np.ndarray) -> np.ndarray:
    A = np.maximum(np.asarray(A, dtype=float), FLOOR)
    return A / A.sum(axis=1, keepdims=True)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def enrichment_corr(A: np.ndarray, genes: np.ndarray) -> np.ndarray:
    E = np.log2(np.maximum(A[:, genes], 1e-6) / np.maximum(A[:, genes].mean(axis=0, keepdims=True), 1e-6))
    return np.corrcoef(E)


def enriched_labels(A: np.ndarray, symbols: np.ndarray, n: int = 2) -> list[str]:
    """Top enriched genes per topic, log2(A_k / mean A) among the most expressed genes (as plot_topics.py)."""

    expressed = np.argsort(-A.mean(axis=0))[:N_CORR_GENES]
    E = np.log2(np.maximum(A[:, expressed], 1e-9) / A[:, expressed].mean(axis=0, keepdims=True))
    return [" ".join(symbols[expressed[np.argsort(-E[k])[:n]]]) for k in range(A.shape[0])]


def load_task(rows: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    """A per estimator (raw, as saved), W, and gene symbols for one task."""

    A = {}
    for _, row in rows.iterrows():
        with np.load(row.npz_path, allow_pickle=True) as z:
            A[row.A_recovery] = np.asarray(z[row.A_key], dtype=float)
    anchor = rows[rows.A_recovery.eq("A_current")].iloc[0]
    with np.load(anchor.feature_npz, allow_pickle=True) as z:
        W = np.asarray(z[f"W_{anchor.array_index}"], dtype=float)
        symbols = z["feature_symbols"].astype(str) if "feature_symbols" in z.files else z["feature_ids"].astype(str)
    return A, W, symbols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", default="gplsi_document__P0_raw__svs")
    parser.add_argument("--examples", nargs="+", default=["core__151673__K7__p2000__r1.0__s26090401",
                                                            "core__151507__K7__p2000__r1.0__s26090401"],
                        help="tasks drawn in the single-task figures")
    args = parser.parse_args()
    out = REPO / "results/visium_dlpfc/diagnostics/core/A_compare"
    out.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(REPO / "results/visium_dlpfc/summary/all_records.csv", dtype={"section": str})
    records = records[records.design.eq("core") & records.has_fit
                      & records.method.isin([f"{args.geometry}__{a}" for a in ESTIMATORS])]
    tag = args.geometry.removeprefix("gplsi_document__")

    topic_rows, pair_rows, corr_rows, examples = [], [], [], {}
    for identity, rows in records.groupby("identity"):
        if set(rows.A_recovery) != set(ESTIMATORS):
            print(f"skip {identity}: has {sorted(rows.A_recovery)}")
            continue
        raw, W, symbols = load_task(rows)
        A = {a: normalise(raw[a]) for a in ESTIMATORS}
        mass = W.sum(axis=0) / W.sum()
        K, p = A["A_current"].shape
        top = {a: np.argsort(-A[a], axis=1)[:, :TOP_N] for a in ESTIMATORS}
        corr_genes = np.argsort(-A["A_current"].mean(axis=0))[:N_CORR_GENES]
        labels = enriched_labels(A["A_current"], symbols)
        task = rows.iloc[0]
        for k in range(K):
            label = labels[k]
            base = dict(identity=identity, section=task.section, seed=task.seed, topic=k, mass=mass[k], label=label)
            for a in ESTIMATORS:
                H = -np.sum(A[a][k] * np.log(A[a][k]))
                topic_rows.append(dict(base, estimator=a, entropy=H / np.log(p), effective_genes=np.exp(H),
                                       zeros=int(np.sum(raw[a][k] <= FLOOR)),
                                       top1_share=float(A[a][k].max())))
            for a, b in combinations(ESTIMATORS, 2):
                overlap = len(set(top[a][k]) & set(top[b][k])) / len(set(top[a][k]) | set(top[b][k]))
                pair_rows.append(dict(base, pair=f"{a} vs {b}", a=a, b=b, jsd=jsd(A[a][k], A[b][k]),
                                      top20_jaccard=overlap))
        for a in ESTIMATORS:
            C = enrichment_corr(A[a], corr_genes)
            off = C[~np.eye(K, dtype=bool)]
            corr_rows.append(dict(identity=identity, section=task.section, seed=task.seed, estimator=a,
                                  mean_offdiag=float(off.mean()), max_offdiag=float(off.max()),
                                  mean_between_topic_jsd=float(np.mean(
                                      [jsd(A[a][i], A[a][j]) for i, j in combinations(range(K), 2)]))))
            if identity in args.examples:
                examples.setdefault(identity, {})[a] = C
        if identity in args.examples:
            examples[identity].update(raw=raw, mass=mass, labels=labels)

    topics, pairs, corrs = pd.DataFrame(topic_rows), pd.DataFrame(pair_rows), pd.DataFrame(corr_rows)
    topics.to_csv(out / f"{tag}__per_topic.csv", index=False)
    pairs.to_csv(out / f"{tag}__pairs.csv", index=False)
    corrs.to_csv(out / f"{tag}__between_topic.csv", index=False)

    near_empty = topics.drop_duplicates(["identity", "topic"]).mass < 0.01
    pd.set_option("display.width", 160)
    print(f"{tag}: {topics.identity.nunique()} tasks, {len(topics) // 3} topics "
          f"({int(near_empty.sum())} near-empty, < 1% mass)\n")
    pairs["near_empty"] = pairs.mass < 0.01
    print("JSD and top-20 Jaccard between estimators, per topic (median [max]):")
    print(pairs.groupby(["pair", "near_empty"]).agg(
        jsd_median=("jsd", "median"), jsd_max=("jsd", "max"),
        jaccard_median=("top20_jaccard", "median"), jaccard_min=("top20_jaccard", "min")).round(4).to_string())
    topics["near_empty"] = topics.mass < 0.01
    print("\nPer-topic entropy / effective genes / exact zeros (mean):")
    print(topics.groupby(["near_empty", "estimator"])[["entropy", "effective_genes", "zeros", "top1_share"]]
          .mean().round(3).to_string())
    print("\nBetween-topic similarity within each estimator (mean over tasks):")
    print(corrs.groupby("estimator")[["mean_offdiag", "max_offdiag", "mean_between_topic_jsd"]].mean().round(4)
          .to_string())

    # ---- Figure 1: per-topic change vs topic mass (all tasks) ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for (a, b), color in PAIR_COLORS.items():
        sub = pairs[pairs.a.eq(a) & pairs.b.eq(b)]
        axes[0].scatter(sub.mass, sub.jsd, s=16, color=color, alpha=0.8, label=f"{a} vs {b}", linewidths=0)
        axes[1].scatter(sub.mass, sub.top20_jaccard, s=16, color=color, alpha=0.8, linewidths=0)
    for ax in axes[:2]:
        ax.set_xscale("log"); ax.axvline(0.01, color=INK2, linestyle=":", linewidth=1)
        ax.set_xlabel("topic mass share (log; dotted = 1%)")
    axes[0].set_yscale("log"); axes[0].set_ylabel("JSD between estimators (bits)")
    axes[0].set_title("How much each topic's gene distribution changes"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("top-20 gene Jaccard overlap"); axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Do the top genes stay the same?")
    wide = topics.pivot_table(index=["identity", "topic", "mass"], columns="estimator", values="entropy").reset_index()
    for a in ["A_full_L2", "A_full_Pois"]:
        axes[2].scatter(wide.A_current, wide[a], s=16, color=COLORS[a], alpha=0.8, label=a, linewidths=0)
    lims = [wide[ESTIMATORS].min().min() - 0.01, wide[ESTIMATORS].max().max() + 0.01]
    axes[2].plot(lims, lims, color=INK2, linewidth=1)
    axes[2].set_xlabel("entropy / log p, A_current"); axes[2].set_ylabel("entropy / log p, other estimator")
    axes[2].set_title("Per-topic entropy vs A_current"); axes[2].legend(fontsize=7)
    fig.suptitle(f"{tag}: per-topic comparison of A estimators on the same W (9 core tasks × 7 topics)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / f"{tag}__per_topic_change.png", dpi=170); plt.close(fig)

    for example, ex in examples.items():
        short = example.removeprefix("core__").replace("__K7__p2000__r1.0__", "_")
        raw, mass, labels = ex["raw"], ex["mass"], ex["labels"]
        K = raw["A_current"].shape[0]
        # ---- Figure 2: gene-level scatter per topic ---------------------------------------
        fig, axes = plt.subplots(2, K, figsize=(2.3 * K, 4.9), squeeze=False)
        for k in range(K):
            x = np.maximum(raw["A_current"][k], 1e-9)
            for r, other in enumerate(["A_full_L2", "A_full_Pois"]):
                ax = axes[r, k]
                y = np.maximum(raw[other][k], 1e-9)
                ax.scatter(x, y, s=3, color=COLORS[other], alpha=0.5, linewidths=0, rasterized=True)
                ax.plot([1e-9, 1], [1e-9, 1], color=INK2, linewidth=0.8)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e-10, 0.2); ax.set_ylim(5e-10, 0.2)
                ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(f"T{k} · {mass[k]:.1%} mass\n{labels[k]}", fontsize=7.5)
                if k == 0:
                    ax.set_ylabel(f"{other}\n(zeros drawn at 1e-9)", fontsize=7.5)
                if r == 1:
                    ax.set_xlabel("A_current", fontsize=7.5)
        fig.suptitle(f"{example} · {tag}: gene probabilities per topic, other estimators vs A_current "
                     "(topic label = top enriched genes)", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(out / f"{tag}__gene_scatter_{short}.png", dpi=170); plt.close(fig)

        # ---- Figure 3: between-topic correlation per estimator + per-topic entropy --------
        fig = plt.figure(figsize=(16, 4.3), layout="constrained")
        grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.06, 1.25])
        ticks = [f"T{k} {labels[k].split()[0]} ({mass[k]:.0%})" for k in range(K)]
        for i, a in enumerate(ESTIMATORS):
            ax = fig.add_subplot(grid[0, i])
            C = ex[a]
            im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
            for (r, c), v in np.ndenumerate(C):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 else INK)
            ax.set_xticks(range(K), ticks, rotation=60, ha="right", fontsize=6.5)
            ax.set_yticks(range(K), ticks if i == 0 else [""] * K, fontsize=6.5)
            ax.set_title(a); ax.grid(False)
        fig.colorbar(im, cax=fig.add_subplot(grid[0, 3]), label="corr. of topic log2 enrichment")
        ax = fig.add_subplot(grid[0, 4])
        width = 0.26
        sub = topics[topics.identity.eq(example)]
        for i, a in enumerate(ESTIMATORS):
            H = sub[sub.estimator.eq(a)].sort_values("topic").entropy.to_numpy()
            ax.bar(np.arange(K) + (i - 1) * width, H, width=width, color=COLORS[a], label=a)
        ax.set_xticks(range(K), ticks, rotation=60, ha="right", fontsize=6.5)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel("entropy / log p"); ax.set_title("Per-topic entropy")
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        fig.suptitle(f"{example} · {tag}: between-topic similarity and entropy by estimator (label: top "
                     "enriched gene, topic mass)", fontsize=10, x=0.01, ha="left")
        fig.savefig(out / f"{tag}__between_topic_{short}.png", dpi=170); plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
