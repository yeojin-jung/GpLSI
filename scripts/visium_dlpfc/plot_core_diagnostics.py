"""Diagnostic plots of the core-design result metrics.

    python scripts/visium_dlpfc/plot_core_diagnostics.py [--design core]

Reads results/visium_dlpfc/<design>/*.json directly (no npz, no h5ad) and writes PNGs
plus the tidy table they are drawn from to results/visium_dlpfc/diagnostics/<design>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAYER_ARI = "external__layer_guess_reordered__ari"

# Reference categorical palette, fixed slot order (dataviz skill references/palette.md).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

HUNTERS = ["spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"]
PRES = ["P0_raw", "P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke"]
PRE_SHORT = {"P0_raw": "P0 raw", "P1_tran_alpha_0p005": "P1 Tran", "P2_ke_weighted": "P2 KE",
             "P3_tran_then_ke": "P3 Tran+KE"}
BASELINES = ["spatial_lda", "lda", "topicscore_graph_denoised", "graph_kl_nmf", "topicscore_raw", "kl_nmf"]
GRID_TOP = 1e-4 * 1.2 ** 28


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def load(design: str) -> pd.DataFrame:
    rows = []
    for path in sorted((REPO / "results/visium_dlpfc" / design).glob("*.json")):
        payload = json.loads(path.read_text())
        task = payload["task"]
        for record in payload["results"]:
            method, meta, metrics = record["method"], record.get("metadata") or {}, record.get("metrics") or {}
            parts = method.split("__")
            gplsi = parts[0].startswith("gplsi_") and len(parts) == 4
            A = meta.get("A_recovery") or {}
            rows.append(dict(
                section=str(task["unit_id"]), seed=int(task["seed"]), method=method, status=record["status"],
                family=parts[0] if gplsi else "baseline",
                preprocessing=parts[1] if gplsi else "", hunter=parts[2] if gplsi else "",
                A_recovery=parts[3] if gplsi else "",
                runtime=record.get("runtime_seconds"), selected_rho=meta.get("selected_rho"),
                A_iterations=A.get("iterations"), A_pgn=A.get("projected_gradient_norm"),
                ari=metrics.get(LAYER_ARI), nmi=metrics.get("external__layer_guess_reordered__nmi"),
                moran=metrics.get("spatial_topic_moran_mean"),
                dev_floored=metrics.get("heldout_poisson_deviance_per_molecule_floored_1e-12"),
                zero_prob=metrics.get("heldout_zero_probability_molecules"),
                W_refit_converged=meta.get("W_refit_converged"),
            ))
    frame = pd.DataFrame(rows)
    for column in ["runtime", "selected_rho", "A_iterations", "A_pgn", "ari", "nmi", "moran", "dev_floored", "zero_prob"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def fig_ari_heatmaps(doc: pd.DataFrame, sections: list[str], out: Path) -> None:
    geo = doc[doc.A_recovery == "A_current"]
    table = geo.groupby(["section", "preprocessing", "hunter"]).ari.mean()
    panels = sections + ["mean"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=True)
    vmin, vmax = min(0.0, float(table.min())), float(table.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, panel in zip(axes, panels):
        grid = (table.groupby(level=[1, 2]).mean() if panel == "mean" else table.loc[panel]).unstack("hunter")
        grid = grid.reindex(index=PRES, columns=HUNTERS)
        im = ax.imshow(grid.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(grid.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if (v - vmin) / (vmax - vmin) > 0.55 else INK)
        ax.set_xticks(range(len(HUNTERS)), HUNTERS, rotation=35, ha="right")
        ax.set_yticks(range(len(PRES)), [PRE_SHORT[p] for p in PRES])
        ax.set_title(f"section {panel}" if panel != "mean" else "mean over sections")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="layer ARI (seed mean)")
    fig.suptitle("Layer ARI by preprocessing × vertex hunter (GpLSI document, core)", x=0.45, y=1.03)
    save(fig, out, "01_ari_heatmap_preprocessing_x_hunter.png")


def fig_ari_methods(frame: pd.DataFrame, sections: list[str], out: Path) -> None:
    doc = frame[(frame.family == "gplsi_document") & (frame.A_recovery == "A_current")
                & (frame.preprocessing == "P0_raw")].assign(label=lambda d: "GpLSI P0 " + d.hunter)
    anchor = frame[(frame.family == "gplsi_anchor") & (frame.A_recovery == "A_full_L2")].assign(label="GpLSI anchor P0 SPA")
    base = frame[frame.family == "baseline"].assign(label=lambda d: d.method)
    data = pd.concat([doc, anchor, base]).dropna(subset=["ari"])
    order = data.groupby("label").ari.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(order) + 1.2))
    offsets = np.linspace(-0.22, 0.22, len(sections))
    for s, (section, offset) in enumerate(zip(sections, offsets)):
        sub = data[data.section == section]
        y = sub.label.map({m: i for i, m in enumerate(order)}) + offset
        ax.scatter(sub.ari, y, s=26, color=SLOTS[s], marker=MARKERS[s], edgecolor=SURFACE, linewidth=0.8,
                   label=f"section {section}", zorder=3)
    means = data.groupby("label").ari.mean().reindex(order)
    ax.scatter(means.values, range(len(order)), marker="|", s=260, color=INK, linewidth=2, label="mean", zorder=4)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("layer ARI (each point = one seed)")
    ax.set_title("Layer ARI per method: 3 sections × 3 seeds (GpLSI on P0)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "02_ari_by_method_seed_points.png")


def fig_A_paired(doc: pd.DataFrame, out: Path) -> None:
    key = ["section", "seed", "preprocessing", "hunter"]
    wide = doc.pivot_table(index=key, columns="A_recovery", values=["dev_floored", "zero_prob"], aggfunc="first").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    diff = wide[("dev_floored", "A_full_L2")] - wide[("dev_floored", "A_current")]
    for h, hunter in enumerate(HUNTERS):
        values = diff[diff.index.get_level_values("hunter") == hunter]
        jitter = np.random.default_rng(h).uniform(-0.22, 0.22, len(values))
        ax.scatter(values, h + jitter, s=20, color=SLOTS[h], marker=MARKERS[h], edgecolor=SURFACE, linewidth=0.7, zorder=3)
        ax.scatter([values.median()], [h], marker="|", s=300, color=INK, linewidth=2, zorder=4)
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yticks(range(len(HUNTERS)), HUNTERS)
    ax.set_xlabel("A_full_L2 − A_current: floored deviance / molecule (symlog; < 0 favours A_full_L2)")
    ax.set_title(f"Deviance: A_full_L2 lower in {float((diff < 0).mean()):.0%} (bar = median)")
    for ax, metric, label, log in [(axes[1], "zero_prob", "zero-probability held-out molecules (+1)", True)]:
        x, y = wide[(metric, "A_current")], wide[(metric, "A_full_L2")]
        if log:
            x, y = x + 1, y + 1
        for h, hunter in enumerate(HUNTERS):
            mask = wide.index.get_level_values("hunter") == hunter
            ax.scatter(x[mask], y[mask], s=22, color=SLOTS[h], marker=MARKERS[h], edgecolor=SURFACE,
                       linewidth=0.7, label=hunter, zorder=3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle="--", zorder=2)
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"A_current: {label}"); ax.set_ylabel(f"A_full_L2: {label}")
        below = float((y < x).mean())
        ax.set_title(f"Zero-prob molecules: A_full_L2 lower in {below:.0%}")
    axes[1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1), title="hunter", title_fontsize=8)
    fig.suptitle("Dimension A: A_full_L2 vs A_current on the same Ŵ (one point per task × preprocessing × hunter)", y=1.02)
    fig.tight_layout()
    save(fig, out, "03_A_estimator_paired.png")


def fig_convergence(frame: pd.DataFrame, out: Path) -> None:
    l2 = frame[(frame.A_recovery == "A_full_L2") & frame.family.str.startswith("gplsi")].copy()
    l2["group"] = np.where(l2.family == "gplsi_anchor", "anchor SPA", l2.hunter)
    groups = HUNTERS + ["anchor SPA"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for g, group in enumerate(groups):
        sub = l2[l2.group == group]
        jitter = np.random.default_rng(g).uniform(-0.25, 0.25, len(sub))
        conv = sub.status.eq("ok").to_numpy()
        ax.scatter(sub.A_pgn[conv], g + jitter[conv], s=18, color=SLOTS[0], marker="o", edgecolor=SURFACE,
                   linewidth=0.6, label="converged" if g == 0 else None, zorder=3)
        ax.scatter(sub.A_pgn[~conv], g + jitter[~conv], s=22, color=SLOTS[1], marker="X", edgecolor=SURFACE,
                   linewidth=0.6, label="hit 2,000-iteration cap" if g == 0 else None, zorder=3)
    ax.set_xscale("log"); ax.set_yticks(range(len(groups)), groups)
    ax.set_xlabel("A_full_L2 projected-gradient norm at stop")
    ax.set_title("A_full_L2 stopping point by hunter")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2)

    ax = axes[1]
    gp = frame[frame.family.str.startswith("gplsi")].copy()
    gp["group"] = np.where(gp.family == "gplsi_anchor", "anchor", gp.hunter) + " · " + gp.A_recovery.str.replace("A_", "")
    order = [f"{h} · {a}" for h in HUNTERS + ["anchor"] for a in ["current", "full_L2"]]
    counts = gp.groupby(["group", "status"]).size().unstack(fill_value=0).reindex(order).fillna(0)
    left = np.zeros(len(order))
    for s, status in enumerate(["ok", "max_iter_reached", "failed"]):
        if status in counts:
            ax.barh(range(len(order)), counts[status], left=left, color=SLOTS[[0, 3, 7][s]], height=0.72,
                    edgecolor=SURFACE, linewidth=2, label=status)
            left += counts[status].to_numpy()
    ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
    ax.set_xlabel("records (9 tasks × 4 preprocessings; anchor: 9 tasks)")
    ax.set_title("Record status by hunter × A estimator")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3)
    fig.tight_layout()
    save(fig, out, "04_convergence_and_status.png")


def fig_lambda(doc: pd.DataFrame, sections: list[str], out: Path) -> None:
    rho = doc.drop_duplicates(["section", "seed", "preprocessing"])[["section", "seed", "preprocessing", "selected_rho"]]
    grid = 1e-4 * 1.2 ** np.arange(29)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for value in grid[-8:]:
        ax.axhline(value, color=GRID, linewidth=0.8, zorder=1)
    ax.axhline(GRID_TOP, color=SLOTS[7], linewidth=1.5, linestyle="--", zorder=2)
    ax.text(-0.45, GRID_TOP, "grid top 0.0165", va="bottom", ha="left", fontsize=8, color=INK2)
    offsets = np.linspace(-0.2, 0.2, len(sections))
    for s, (section, offset) in enumerate(zip(sections, offsets)):
        sub = rho[rho.section == section]
        x = sub.preprocessing.map({p: i for i, p in enumerate(PRES)}) + offset
        ax.scatter(x, sub.selected_rho, s=34, color=SLOTS[s], marker=MARKERS[s], edgecolor=SURFACE,
                   linewidth=0.8, label=f"section {section}", zorder=3)
    ax.set_xticks(range(len(PRES)), [PRE_SHORT[p] for p in PRES])
    ax.set_ylabel("selected graph penalty λ"); ax.set_ylim(grid[-5], GRID_TOP * 1.06)
    ax.set_title("Selected λ per task (faint lines = grid points): at or ≤2 steps below the grid top")
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    save(fig, out, "05_selected_lambda.png")


def fig_tradeoff(frame: pd.DataFrame, out: Path) -> None:
    doc = frame[(frame.family == "gplsi_document") & (frame.preprocessing.isin(["P0_raw", "P2_ke_weighted"]))]
    pts = []
    for (pre, hunter, a), g in doc.groupby(["preprocessing", "hunter", "A_recovery"]):
        pts.append(dict(kind="GpLSI", label=f"{pre[:2]} {hunter} {a.replace('A_', '')}", hunter=hunter, pre=pre,
                        ari=g.ari.mean(), dev=g.dev_floored.mean()))
    for method, g in frame[frame.family == "baseline"].groupby("method"):
        pts.append(dict(kind="baseline", label=method, hunter="", pre="", ari=g.ari.mean(), dev=g.dev_floored.mean()))
    pts = pd.DataFrame(pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, hunter in enumerate(HUNTERS):
        for pre, face in [("P0_raw", SLOTS[h]), ("P2_ke_weighted", SURFACE)]:
            sub = pts[(pts.hunter == hunter) & (pts.pre == pre)]
            ax.plot(sub.dev, sub.ari, color=SLOTS[h], linewidth=1, alpha=0.6, zorder=2)
            ax.scatter(sub.dev, sub.ari, s=38, marker=MARKERS[h], facecolor=face, edgecolor=SLOTS[h], linewidth=1.4,
                       label=f"{hunter} ({'P0' if pre == 'P0_raw' else 'P2'})", zorder=3)
    base = pts[pts.kind == "baseline"]
    ax.scatter(base.dev, base.ari, s=46, marker="*", color=INK, zorder=4, label="baselines")
    for _, row in base.iterrows():
        ax.annotate(row.label, (row.dev, row.ari), xytext=(5, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("floored held-out deviance / molecule (lower = better prediction) →")
    ax.set_ylabel("layer ARI (higher = better layers) →")
    ax.set_title("Layer recovery vs held-out prediction (means over 9 tasks)\nfilled = P0, hollow = P2; each joined pair = A_current / A_full_L2 on the same Ŵ")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    save(fig, out, "06_ari_vs_deviance_tradeoff.png")


def fig_runtime(frame: pd.DataFrame, out: Path) -> None:
    doc = frame[frame.family.str.startswith("gplsi")].assign(
        label=lambda d: np.where(d.family == "gplsi_anchor", "GpLSI anchor", "GpLSI document") + " · " + d.A_recovery)
    base = frame[frame.family == "baseline"].assign(label=lambda d: d.method)
    data = pd.concat([doc, base])
    data = data[data.status != "failed"]
    stats = data.groupby("label").runtime.agg(["median", "min", "max"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(stats) + 1.2))
    y = np.arange(len(stats))
    ax.hlines(y, stats["min"], stats["max"], color=SLOTS[0], linewidth=2, alpha=0.45)
    ax.scatter(stats["median"], y, s=36, color=SLOTS[0], zorder=3)
    for yi, (label, row) in zip(y, stats.iterrows()):
        ax.text(row["max"] * 1.08, yi, f"{row['median']:.0f} s", va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_yticks(y, stats.index)
    ax.set_xlabel("runtime per record, seconds (dot = median, line = range over tasks)")
    ax.set_title("Runtime per method, failed records excluded (GpLSI includes its share of the spectral step)")
    save(fig, out, "07_runtime.png")


def fig_ari_seed_spread(doc: pd.DataFrame, sections: list[str], out: Path) -> None:
    geo = doc[(doc.A_recovery == "A_current") & doc.preprocessing.isin(["P0_raw", "P2_ke_weighted"])]
    spread = geo.groupby(["preprocessing", "hunter", "section"]).ari.agg(lambda s: s.max() - s.min()).unstack("section")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, pre in zip(axes, ["P0_raw", "P2_ke_weighted"]):
        table = spread.loc[pre].reindex(HUNTERS)
        width = 0.26
        for s, section in enumerate(sections):
            ax.bar(np.arange(len(HUNTERS)) + (s - 1) * width, table[section], width=width - 0.03, color=SLOTS[s],
                   label=f"section {section}", edgecolor=SURFACE, linewidth=0)
        ax.set_xticks(range(len(HUNTERS)), HUNTERS, rotation=30, ha="right")
        ax.set_title(PRE_SHORT[pre])
    axes[0].set_ylabel("ARI range across 3 seeds (max − min)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Seed sensitivity of layer ARI", y=1.02)
    fig.tight_layout()
    save(fig, out, "08_ari_seed_spread.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="core")
    args = parser.parse_args()
    style()
    out = REPO / "results/visium_dlpfc/diagnostics" / args.design
    out.mkdir(parents=True, exist_ok=True)
    frame = load(args.design)
    frame.to_csv(out / "records.csv", index=False)
    sections = sorted(frame.section.unique())
    doc = frame[frame.family == "gplsi_document"]
    fig_ari_heatmaps(doc, sections, out)
    fig_ari_methods(frame, sections, out)
    fig_A_paired(doc, out)
    fig_convergence(frame, out)
    fig_lambda(doc, sections, out)
    fig_tradeoff(frame, out)
    fig_runtime(frame, out)
    fig_ari_seed_spread(doc, sections, out)


if __name__ == "__main__":
    main()
