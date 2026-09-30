"""Spatial maps of hard topic assignments next to the manual layers, per core task.

    python scripts/visium_dlpfc/plot_spatial_maps.py [--design core]

Each spot is coloured by its dominant topic (argmax of W). Topics are matched one-to-one
to manual layers by the Hungarian algorithm on the topic x layer overlap (labelled spots
only), and a matched topic is drawn in its layer's colour, so a map that recovers the
layers looks like the manual panel. Topics left unmatched (K = 7 exceeds the number of
labelled layers on Br5595) are drawn in greys. Writes to
results/visium_dlpfc/diagnostics/<design>/spatial/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAYERS = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]
# Reference categorical palette, fixed slot order: one hue per layer.
LAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
UNMATCHED = ["#52514e", "#8a8985", "#b8b7b0"]
UNLABELLED = "#e4e3de"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
HUNTERS = ["spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"]
BASELINES = ["spatial_lda", "lda", "graph_kl_nmf", "kl_nmf", "topicscore_graph_denoised", "topicscore_raw"]
ARI = "external__layer_guess_reordered__ari"


def layer_labels(ids: np.ndarray) -> np.ndarray:
    import anndata as ad

    obs = ad.read_h5ad(REPO / "data/processed/visium_dlpfc/visium_dlpfc.h5ad", backed="r").obs
    return obs.loc[ids, "layer_guess_reordered"].astype(str).to_numpy()


def topic_colors(hard: np.ndarray, labels: np.ndarray, K: int) -> tuple[list[str], dict[int, str]]:
    """Colour per topic: its Hungarian-matched layer colour, else a grey."""

    present = [layer for layer in LAYERS if np.any(labels == layer)]
    overlap = np.array([[np.sum((hard == k) & (labels == layer)) for layer in present] for k in range(K)])
    rows, cols = linear_sum_assignment(-overlap)
    colors, names, grey = [None] * K, {}, 0
    for k, c in zip(rows, cols):
        if overlap[k, c] > 0:
            colors[k] = LAYER_COLORS[LAYERS.index(present[c])]
            names[k] = present[c]
    for k in range(K):
        if colors[k] is None:
            colors[k] = UNMATCHED[min(grey, len(UNMATCHED) - 1)]
            grey += 1
    return colors, names


def draw(ax, xy, colors, title, size) -> None:
    ax.scatter(xy[:, 0], -xy[:, 1], c=colors, s=size, marker="o", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=7.5, color=INK, pad=3)
    ax.set_aspect("equal")
    ax.axis("off")


def method_W(payload: dict, saved, method: str):
    """W for a method; for GpLSI, any recovery of the geometry (all share W)."""

    candidates = [method]
    if method.startswith("gplsi_"):
        geometry = method.rsplit("__", 1)[0]
        candidates += [f"{geometry}__A_full_L2", f"{geometry}__A_current"]
    for name in candidates:
        index = payload["array_index"].get(name)
        if index is not None and f"W_{index}" in saved:
            record = payload["results"][index]
            return saved[f"W_{index}"].astype(float), (record.get("metrics") or {}).get(ARI)
    return None, None


def panel(ax, payload, saved, method, title, xy, labels, size) -> None:
    W, ari = method_W(payload, saved, method)
    if W is None:
        ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes, color=INK2)
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")
        return
    hard = W.argmax(1)
    colors, _ = topic_colors(hard, labels, W.shape[1])
    draw(ax, xy, [colors[k] for k in hard], f"{title}\nARI {ari:.3f}" if ari is not None else title, size)


def layer_panel(ax, xy, labels, title, size) -> None:
    colors = [LAYER_COLORS[LAYERS.index(v)] if v in LAYERS else UNLABELLED for v in labels]
    draw(ax, xy, colors, title, size)


def legend(fig, labels) -> None:
    present = [layer for layer in LAYERS if np.any(labels == layer)]
    handles = [Line2D([], [], marker="s", linestyle="", markersize=7, color=LAYER_COLORS[LAYERS.index(l)],
                      label=l.replace("Layer", "L")) for l in present]
    handles += [Line2D([], [], marker="s", linestyle="", markersize=7, color=UNMATCHED[1], label="unmatched topic"),
                Line2D([], [], marker="s", linestyle="", markersize=7, color=UNLABELLED, label="no manual label")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))


def task_figure(payload, saved, labels, xy, out: Path) -> None:
    task = payload["task"]
    layout = [
        [("__layers__", "manual layers")] + [(f"gplsi_document__P0_raw__{h}__A_current", f"GpLSI P0 · {h}") for h in HUNTERS],
        [("gplsi_anchor__P0_raw__spa_current__A_current", "GpLSI anchor · P0 SPA")]
        + [(f"gplsi_document__P2_ke_weighted__{h}__A_current", f"GpLSI P2 · {h}") for h in HUNTERS],
        [(m, m) for m in BASELINES] + [(None, None)],
    ]
    fig, axes = plt.subplots(3, 7, figsize=(16, 7.8))
    size = 4.2
    for r, row in enumerate(layout):
        for c, (method, title) in enumerate(row):
            ax = axes[r, c]
            if method is None:
                ax.axis("off")
            elif method == "__layers__":
                layer_panel(ax, xy, labels, title, size)
            else:
                panel(ax, payload, saved, method, title, xy, labels, size)
    legend(fig, labels)
    fig.suptitle(f"Section {task['unit_id']} · seed {task['seed']} · K={task['K']} · {task['panel_size']} genes "
                 f"(topics coloured by their Hungarian-matched layer; P1 = P0 and P3 = P2 here)", fontsize=10, y=0.995)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    name = f"map_{task['unit_id']}_s{task['seed']}.png"
    fig.savefig(out / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out / name)


def overview(tasks, out: Path) -> None:
    """One row per section (first seed): layers and the key methods."""

    methods = [
        ("__layers__", "manual layers"),
        ("gplsi_document__P0_raw__spa_current__A_current", "GpLSI original\n(P0, SPA)"),
        ("gplsi_document__P0_raw__svs__A_current", "GpLSI P0 · svs"),
        ("gplsi_document__P0_raw__svs_star__A_current", "GpLSI P0 · svs_star"),
        ("gplsi_document__P2_ke_weighted__svs_star__A_current", "GpLSI P2 · svs_star"),
        ("spatial_lda", "Spatial-LDA"),
        ("lda", "LDA"),
        ("graph_kl_nmf", "graph KL-NMF"),
    ]
    fig, axes = plt.subplots(len(tasks), len(methods), figsize=(2.25 * len(methods), 2.5 * len(tasks)), squeeze=False)
    all_labels = []
    for r, (payload, saved, labels, xy) in enumerate(tasks):
        all_labels.append(labels)
        for c, (method, title) in enumerate(methods):
            label = f"{payload['task']['unit_id']}\n{title}" if c == 0 else title
            if method == "__layers__":
                layer_panel(axes[r, c], xy, labels, label, 2.6)
            else:
                panel(axes[r, c], payload, saved, method, title, xy, labels, 2.6)
    legend(fig, np.concatenate(all_labels))
    fig.suptitle("Layer recovery maps, one seed per section (topics coloured by matched layer)", fontsize=10, y=0.995)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out / "overview_first_seed.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out / "overview_first_seed.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="core")
    args = parser.parse_args()
    base = REPO / "results/visium_dlpfc" / args.design
    out = REPO / "results/visium_dlpfc/diagnostics" / args.design / "spatial"
    out.mkdir(parents=True, exist_ok=True)
    first_seed = {}
    for path in sorted(base.glob("*.json")):
        payload = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=True) as z:
            saved = {k: z[k] for k in z.files}
        ids = saved["observation_ids"].astype(str)
        labels, xy = layer_labels(ids), saved["coordinates"].astype(float)
        task_figure(payload, saved, labels, xy, out)
        section = payload["task"]["unit_id"]
        if section not in first_seed or payload["task"]["seed"] < first_seed[section][0]["task"]["seed"]:
            first_seed[section] = (payload, saved, labels, xy)
    overview([first_seed[s] for s in sorted(first_seed)], out)


if __name__ == "__main__":
    main()
