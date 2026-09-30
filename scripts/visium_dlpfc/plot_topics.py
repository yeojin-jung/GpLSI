"""Per-topic diagnostics: spatial map, layer composition, gene signature, top genes.

    python scripts/visium_dlpfc/plot_topics.py [--design core] [--seed 26090401] [--recovery A_current]

For each section (one seed) and a set of key methods, one figure:
  * one spatial map per topic (continuous weight W[:, k]) next to the manual layers;
  * layer composition: mean W[i, k] over the spots of each manual layer (columns sum to 1
    across topics), i.e. where each topic sits in the tissue;
  * gene signature: Pearson correlation, over the 1,000 most expressed panel genes, between
    the topic's log2 enrichment log2(A[k] / A_bar) and each layer's log2 enrichment
    log2(L_layer / L_all), where L are pooled count frequencies of the section's spots in
    that layer. A topic that matches one layer is "pure"; high correlation with two
    adjacent layers means a mixture;
  * the topic's top enriched genes, coloured by the layer where each gene is most enriched.

Topics are ordered by their Hungarian-matched layer (as in plot_spatial_maps.py).
Writes to results/visium_dlpfc/diagnostics/<design>/topics/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_spatial_maps as maps  # noqa: E402  (same directory)

REPO = maps.REPO
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c8480f", "#f0b9a1", "#f4f3ef", "#9ec5f4", "#184f95"])
METHODS = [
    ("gplsi_document__P0_raw__spa_current", "GpLSI P0 · spa_current (original)"),
    ("gplsi_document__P0_raw__svs", "GpLSI P0 · svs"),
    ("gplsi_document__P0_raw__svs_star", "GpLSI P0 · svs_star"),
    ("gplsi_document__P0_raw__pp_spa", "GpLSI P0 · pp_spa"),
    ("gplsi_document__P2_ke_weighted__svs_star", "GpLSI P2 · svs_star"),
    ("gplsi_anchor__P0_raw__spa_current", "GpLSI anchor · P0 SPA"),
    ("spatial_lda", "Spatial-LDA"),
    ("lda", "LDA"),
    ("graph_kl_nmf", "graph KL-NMF"),
    ("topicscore_graph_denoised", "TopicSCORE graph-denoised"),
]
# Follow-up designs fit only part of the core suite; methods without a fit are skipped.
DESIGN_METHODS = {
    "p2_wide": [(f"gplsi_document__P2_ke_weighted__{h}", f"GpLSI P2 (wide λ grid) · {h}")
                for h in ["spa_current", "svs", "svs_star", "pp_spa"]],
    "panel": [("gplsi_document__P0_raw__spa_current", "GpLSI P0 · spa_current (original)"),
              ("gplsi_document__P0_raw__svs_star", "GpLSI P0 · svs_star"),
              ("lda", "LDA")],
}
N_TOP_GENES = 8
N_SIGNATURE_GENES = 1000


def section_counts(ids: np.ndarray, genes: np.ndarray) -> np.ndarray:
    import anndata as ad

    adata = ad.read_h5ad(REPO / "data/processed/visium_dlpfc/visium_dlpfc.h5ad", backed="r")
    rows = adata.obs_names.get_indexer(ids)
    cols = adata.var_names.get_indexer(genes)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("spot or gene ids missing from the processed h5ad")
    order = np.argsort(rows)
    sub = adata.X[np.sort(rows)]
    sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    counts = np.empty_like(sub)
    counts[order] = sub
    return counts[:, cols].astype(float)


def layer_profiles(counts: np.ndarray, labels: np.ndarray):
    present = [layer for layer in maps.LAYERS if np.any(labels == layer)]
    labelled = np.isin(labels, present)
    overall = counts[labelled].sum(0)
    overall /= overall.sum()
    profiles = np.array([counts[labels == layer].sum(0) / counts[labels == layer].sum() for layer in present])
    return present, profiles, overall


def pick(payload, saved, method: str, recovery: str):
    """(W, A, ARI, name) for a method; GpLSI uses `recovery`, falling back to the other one."""

    names = [method] if not method.startswith("gplsi_") else [
        f"{method}__{recovery}", f"{method}__{'A_full_L2' if recovery == 'A_current' else 'A_current'}"]
    for name in names:
        index = payload["array_index"].get(name)
        if index is not None and f"W_{index}" in saved and f"A_{index}" in saved:
            record = payload["results"][index]
            return (saved[f"W_{index}"].astype(float), saved[f"A_{index}"].astype(float),
                    (record.get("metrics") or {}).get(maps.ARI), name)
    return None


def figure(payload, saved, labels, xy, counts, method, title, recovery, out: Path, design: str = "core") -> None:
    fitted = pick(payload, saved, method, recovery)
    if fitted is None:
        print("skip (no fit)", method)
        return
    W, A, ari, name = fitted
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-15)
    A = np.maximum(A, 0)
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-15)
    K = W.shape[1]
    hard = W.argmax(1)
    colors, matched = maps.topic_colors(hard, labels, K)
    rank = {layer: i for i, layer in enumerate(maps.LAYERS)}
    order = sorted(range(K), key=lambda k: (rank.get(matched.get(k), 99), k))
    tags = [f"T{k + 1}" + (f" ≈ {matched[k].replace('Layer', 'L')}" if k in matched else " (unmatched)") for k in order]

    present, profiles, overall = layer_profiles(counts, labels)
    mass = W.sum(0)
    A_bar = (mass[:, None] * A).sum(0) / mass.sum()
    top_expr = np.argsort(-overall)[:N_SIGNATURE_GENES]
    eps = 1e-7
    layer_enr = np.log2((profiles + eps) / (overall + eps))
    topic_enr = np.clip(np.log2((A + eps) / (A_bar + eps)), -6, 6)
    signature = np.array([[np.corrcoef(topic_enr[k, top_expr], layer_enr[l, top_expr])[0, 1]
                           for l in range(len(present))] for k in order])
    composition = np.array([[W[labels == layer, k].mean() for layer in present] for k in order])

    symbols = saved["feature_symbols"].astype(str) if "feature_symbols" in saved else saved["feature_ids"].astype(str)
    expressed = A > 1e-4
    gene_layer = np.argmax(layer_enr, axis=0)
    gene_strength = np.max(layer_enr, axis=0)

    fig = plt.figure(figsize=(max(2.05 * (K + 1), 17), 8.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.12)
    top_row = outer[0].subgridspec(1, K + 1, wspace=0.08)
    bottom = outer[1].subgridspec(1, 7, width_ratios=[1, 0.045, 0.2, 1, 0.045, 0.28, 1.75], wspace=0.06)
    ax = fig.add_subplot(top_row[0, 0])
    maps.layer_panel(ax, xy, labels, "manual layers", 3.0)
    for c, k in enumerate(order, start=1):
        ax = fig.add_subplot(top_row[0, c])
        top = max(float(np.quantile(W[:, k], 0.99)), 1e-6)
        ax.scatter(xy[:, 0], -xy[:, 1], c=W[:, k], cmap=SEQ, vmin=0, vmax=top, s=3.0, linewidths=0, rasterized=True)
        ax.set_title(f"{tags[c - 1]}\nmass {mass[k] / mass.sum():.0%}", fontsize=8, color=colors[k], fontweight="bold")
        ax.set_aspect("equal"); ax.axis("off")

    layer_ticks = [l.replace("Layer", "L") for l in present]
    ax = fig.add_subplot(bottom[0, 0])
    im = ax.imshow(composition, cmap=SEQ, vmin=0, vmax=max(float(composition.max()), 1e-6), aspect="auto")
    for (i, j), v in np.ndenumerate(composition):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.55 * composition.max() else maps.INK)
    ax.set_xticks(range(len(present)), layer_ticks); ax.set_yticks(range(K), tags, fontsize=8)
    ax.set_title("Where each topic sits\n(mean topic weight in each manual layer)", fontsize=9)
    fig.colorbar(im, cax=fig.add_subplot(bottom[0, 1]))

    ax = fig.add_subplot(bottom[0, 3])
    im = ax.imshow(signature, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    for (i, j), v in np.ndenumerate(signature):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if abs(v) > 0.6 else maps.INK)
    ax.set_xticks(range(len(present)), layer_ticks); ax.set_yticks(range(K), [""] * K)
    ax.set_title("Which layer's genes it carries\n(corr. of topic vs layer log2 enrichment)", fontsize=9)
    fig.colorbar(im, cax=fig.add_subplot(bottom[0, 4]))

    ax = fig.add_subplot(bottom[0, 6])
    ax.axis("off")
    ax.set_title(f"Top {N_TOP_GENES} enriched genes per topic\n(log2 vs average topic; colour = layer where the gene is most enriched)",
                 fontsize=9, loc="left")
    row_h = 1.0 / (K + 0.5)
    for i, k in enumerate(order):
        y = 1 - (i + 0.8) * row_h
        ax.text(0.0, y, tags[i], fontsize=8, fontweight="bold", color=colors[k], transform=ax.transAxes, va="center")
        candidates = np.where(expressed[k])[0]
        best = candidates[np.argsort(-topic_enr[k, candidates])][:N_TOP_GENES]
        x = 0.2
        for g in best:
            color = maps.LAYER_COLORS[maps.LAYERS.index(present[gene_layer[g]])] if gene_strength[g] > 0.5 else maps.INK2
            ax.text(x, y, symbols[g], fontsize=7.5, color=color, transform=ax.transAxes, va="center")
            x += 0.025 + 0.0125 * len(symbols[g])

    task = payload["task"]
    if design == "panel":
        title = f"{title} · {task['panel_size']:,} genes"
    fig.suptitle(f"{title} · section {task['unit_id']} · seed {task['seed']} · ARI {ari:.3f} · Â from {name.split('__')[-1] if name.startswith('gplsi_') else 'method'}",
                 fontsize=11, y=0.995)
    fig.patch.set_facecolor(maps.SURFACE)
    tag = method.replace("gplsi_document__", "gplsi_").replace("gplsi_anchor__", "anchor_")
    size = f"_p{task['panel_size']}" if design == "panel" else ""
    path = out / f"topics_{task['unit_id']}_s{task['seed']}{size}__{tag}.png"
    fig.savefig(path, dpi=140, facecolor=maps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="core")
    parser.add_argument("--seed", type=int, default=26090401)
    parser.add_argument("--recovery", default="A_current", choices=["A_current", "A_full_L2"])
    args = parser.parse_args()
    base = REPO / "results/visium_dlpfc" / args.design
    out = REPO / "results/visium_dlpfc/diagnostics" / args.design / "topics"
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(base.glob(f"*s{args.seed}.json")):
        payload = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=True) as z:
            saved = {k: z[k] for k in z.files}
        ids = saved["observation_ids"].astype(str)
        labels, xy = maps.layer_labels(ids), saved["coordinates"].astype(float)
        counts = section_counts(ids, saved["feature_ids"].astype(str))
        for method, title in DESIGN_METHODS.get(args.design, METHODS):
            figure(payload, saved, labels, xy, counts, method, title, args.recovery, out, args.design)


if __name__ == "__main__":
    main()
