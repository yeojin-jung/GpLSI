"""One-PDF visual report of the DLPFC ablation (figures for the results-doc tables).

    python scripts/analysis/dlpfc/make_report_pdf.py

Reads results/dlpfc/summary/{all_records,seed_stability}.csv (run summarize.py
first), the saved W of the core and p2_wide tasks (for topic usage), and two existing map
figures. Writes results/dlpfc/report/dlpfc_ablation_report.pdf. Aggregation follows
summarize.py: seeds are averaged within a section, then sections are averaged. Results
with no effect (thresholding P1 = P0; P0 on the wide penalty grid) are stated, not plotted.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from summarize import geometry_rows, seed_then_section  # noqa: E402  (same directory)

from records import DIAGNOSTICS as DIAG, RESULTS, SUMMARY  # noqa: E402

OUT = RESULTS / "report"

# Reference categorical slots 1-3 (validated all-pairs in light mode); ink and grid tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#ffffff"
PAGE = (11, 8.5)
HUNTERS = ["spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"]
HUNTER_LABEL = {"spa_current": "SPA (original)", "svs": "SVS", "svs_star": "SVS*", "pp_spa": "pp-SPA",
                "palm": "PALM", "palm_accelerated": "PALM accel."}
BASELINE_LABEL = {"spatial_lda": "Spatial-LDA", "lda": "LDA", "graph_kl_nmf": "graph KL-NMF", "kl_nmf": "KL-NMF",
                  "topicscore_graph_denoised": "TopicSCORE graph-den.", "topicscore_raw": "TopicSCORE raw"}
GRID_TOP = 1e-4 * 1.2 ** 28

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
})


def section_stats(frame: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    """Mean over sections of seed means, plus the per-section min and max."""

    per_section = seed_then_section(frame, keys, [metric])[0][metric].unstack("section")
    return pd.DataFrame({"mean": per_section.mean(axis=1), "lo": per_section.min(axis=1),
                         "hi": per_section.max(axis=1)})


def page(title: str, subtitle: str = "", figsize=PAGE):
    fig = plt.figure(figsize=figsize)
    fig.text(0.05, 0.955, title, fontsize=15, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.05, 0.915, subtitle, fontsize=9.5, color=INK2, va="top")
    return fig


def caption(fig, text: str, y: float = 0.13, width: int = 150) -> None:
    lines = []
    for para in text.strip().split("\n"):
        lines += textwrap.wrap(para.strip(), width) if para.strip() else [""]
    fig.text(0.05, y, "\n".join(lines), fontsize=8.8, color=INK, va="top", linespacing=1.45)


def text_page(pdf, title: str, blocks: list[tuple[str, list[str]]], subtitle: str = "") -> None:
    fig = page(title, subtitle)
    y = 0.86
    for heading, items in blocks:
        if heading:
            fig.text(0.05, y, heading, fontsize=11, fontweight="bold", color=INK, va="top")
            y -= 0.035
        for item in items:
            wrapped = textwrap.wrap(item, 135)
            fig.text(0.065, y, "•", fontsize=9.5, color=INK2, va="top")
            fig.text(0.08, y, "\n".join(wrapped), fontsize=9.5, color=INK, va="top", linespacing=1.4)
            y -= 0.028 * len(wrapped) + 0.012
        y -= 0.015
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, path: Path, title: str, subtitle: str, text: str, image_box=(0.03, 0.2, 0.94, 0.68)) -> None:
    fig = page(title, subtitle)
    ax = fig.add_axes(image_box)
    ax.imshow(plt.imread(path)); ax.axis("off")
    caption(fig, text, y=image_box[1] - 0.01)
    pdf.savefig(fig, dpi=220); plt.close(fig)


def direct_label(ax, x, y, text, color=INK2, **kw) -> None:
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, color=color, **kw)


def end_labels(ax, x, items, min_gap: float) -> None:
    """Label line ends at x, nudging labels apart vertically so they never overlap."""

    items = sorted(items, key=lambda item: item[0])
    placed = []
    for y, _, _ in items:
        placed.append(max(y, placed[-1] + min_gap) if placed else y)
    shift = (np.mean([i[0] for i in items]) - np.mean(placed))
    for (y, text, color), yl in zip(items, placed):
        ax.annotate(text, (x, y), xytext=(x * 1.12, yl + shift), textcoords="data", va="center", fontsize=7.5,
                    color=color, annotation_clip=False)


# ---- Figures ---------------------------------------------------------------------------

def fig_hunters(pdf, core, p2w) -> None:
    geo = geometry_rows(core)
    doc = geo[geo.family.eq("gplsi_document")]
    series = [
        ("P0 raw counts (= P1)", BLUE, doc[doc.preprocessing.eq("P0_raw")]),
        ("P2 KE weighting, core λ grid (= P3)", ORANGE, doc[doc.preprocessing.eq("P2_ke_weighted")]),
        ("P2 KE weighting, wide λ grid", AQUA, geometry_rows(p2w)),
    ]
    base = core[core.family.eq("baseline")]
    refs = section_stats(base, ["method"], "layer_ARI")["mean"]

    fig = page("1 · Vertex hunting is the dimension that matters",
               "Layer ARI of GpLSI by vertex hunter and preprocessing (K = 7; 3 sections × 3 seeds)")
    ax = fig.add_axes((0.2, 0.3, 0.55, 0.56))
    y = np.arange(len(HUNTERS))[::-1]
    offsets = [0.22, 0.0, -0.22]
    for (label, color, frame), dy in zip(series, offsets):
        s = section_stats(frame, ["hunter"], "layer_ARI").reindex(HUNTERS)
        ax.hlines(y + dy, s.lo, s.hi, color=color, linewidth=1.2, alpha=0.55)
        ax.scatter(s["mean"], y + dy, s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
    for (method, style), top in zip([("spatial_lda", "--"), ("lda", ":")], [len(HUNTERS) + 0.15, len(HUNTERS) - 0.2]):
        ax.axvline(refs[method], color=INK2, linestyle=style, linewidth=1)
        ax.text(refs[method] + 0.003, top, f"{BASELINE_LABEL[method]} {refs[method]:.3f}", fontsize=7.5, color=INK2,
                va="center")
    ax.set_yticks(y, [HUNTER_LABEL[h] for h in HUNTERS])
    ax.set_xlabel("layer ARI (dot = mean over sections; bar = range of section means)")
    ax.set_ylim(-0.6, len(HUNTERS) + 0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", visible=False)
    caption(fig, """
    SVS and SVS* more than double the original SPA hunter on raw counts (P0: 0.09 → 0.24), which ties Spatial-LDA (0.239) and beats LDA (0.228). SVS* is the most robust choice: at or near the top on every section and under both weightings. PALM and accelerated PALM add little over SPA.
    KE weighting (P2) lowers ARI for the good hunters. Refitting P2 on the wide λ grid (aqua) helps a little (mean over hunters 0.151 → 0.164; SVS* 0.205 → 0.223) but stays below P0, because three of seven vertices go to rare cell types (Fig. 8b).
    Not plotted: Tran thresholding (P1, P3) gives exactly the same results as P0 and P2 at 2,000 genes (ARI identical to 3 decimals).
    """, y=0.23)
    pdf.savefig(fig); plt.close(fig)


def fig_baselines(pdf, core) -> None:
    geo = geometry_rows(core)
    doc = core[core.family.eq("gplsi_document") & core.A_recovery.eq("A_current")]
    rows = []
    for hunter in ["svs", "svs_star", "pp_spa", "spa_current"]:
        rows.append((f"GpLSI P0 {HUNTER_LABEL[hunter]}", BLUE, doc[doc.preprocessing.eq("P0_raw") & doc.hunter.eq(hunter)]))
    rows.append(("GpLSI anchor (SPA)", BLUE, geo[geo.family.eq("gplsi_anchor")]))
    base = core[core.family.eq("baseline")]
    rows += [(label, ORANGE, base[base.method.eq(method)]) for method, label in BASELINE_LABEL.items()]
    anchor_runtime = core[core.family.eq("gplsi_anchor") & core.A_recovery.eq("A_full_L2")].runtime_seconds.median()
    stats = []
    for label, color, frame in rows:
        per_section = seed_then_section(frame.assign(k="x"), ["k"], ["layer_ARI"])[0]["layer_ARI"]
        seconds = anchor_runtime if "anchor" in label else frame.runtime_seconds.median()
        stats.append((label, color, per_section.droplevel(0), seconds))
    stats.sort(key=lambda item: item[2].mean())

    fig = page("2 · GpLSI vs baselines",
               "Layer ARI per method (core design, 2,000 genes, K = 7), with median runtime per fit")
    ax = fig.add_axes((0.2, 0.3, 0.5, 0.56))
    markers = {"151507": "o", "151669": "s", "151673": "^"}
    for yi, (label, color, per_section, seconds) in enumerate(stats):
        ax.hlines(yi, per_section.min(), per_section.max(), color=color, alpha=0.4, linewidth=1.2)
        for section, value in per_section.items():
            ax.scatter(value, yi, s=16, marker=markers[section], facecolor="none", edgecolor=color, linewidth=0.9,
                       zorder=3)
        ax.scatter(per_section.mean(), yi, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(1.03, yi, f"{seconds:,.0f} s", transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                color=INK2)
    ax.text(1.03, len(stats) - 0.3, "runtime", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2,
            fontweight="bold")
    ax.set_yticks(range(len(stats)), [item[0] for item in stats]); ax.set_ylim(-0.6, len(stats) - 0.1)
    ax.set_xlabel("layer ARI (large dot = mean of sections; small = each section's seed mean)")
    ax.grid(axis="y", visible=False)
    for section, marker in markers.items():
        ax.scatter([], [], marker=marker, facecolor="none", edgecolor=INK2, s=16, label=section)
    ax.scatter([], [], color=BLUE, s=40, label="GpLSI"); ax.scatter([], [], color=ORANGE, s=40, label="baselines")
    ax.legend(loc="upper left", bbox_to_anchor=(1.13, 1.0))
    gplsi_s = [item[3] for item in stats if item[0] == "GpLSI P0 SVS*"][0]
    slda_s = [item[3] for item in stats if item[0] == "Spatial-LDA"][0]
    caption(fig, f"""
    GpLSI with SVS/SVS* ties Spatial-LDA and LDA for layer recovery. ARI differences below about 0.03 are within seed noise, and each method's ranking changes between sections (small markers). It runs in about {gplsi_s:.0f} s per fit (including its share of the spectral step) against {slda_s:.0f} s for Spatial-LDA; LDA is about as fast as GpLSI.
    The original GpLSI (SPA) and the anchor variant are at the bottom with raw TopicSCORE and KL-NMF. ARI hides how the methods differ: LDA and KL-NMF predict held-out molecules best (Fig. 5), and Spatial-LDA's ARI comes mainly from a sharp white-matter topic (Fig. 3).
    """, y=0.23)
    pdf.savefig(fig); plt.close(fig)


def topic_usage(frame: pd.DataFrame) -> pd.DataFrame:
    """Topics used (dominant in >= 1% of spots) and near-empty topics (< 1% of mass), per fit."""

    fits = frame[frame.has_fit].copy()
    fits["geometry"] = np.where(fits.family.eq("baseline"), fits.method, fits.method.str.rsplit("__", n=1).str[0])
    fits = fits.drop_duplicates(["identity", "geometry"])
    out = []
    for _, row in fits.iterrows():
        with np.load(row.npz_path) as z:
            W = np.asarray(z["W_hat"], dtype=float)
        K = W.shape[1]
        share = np.bincount(W.argmax(1), minlength=K) / W.shape[0]
        mass = W.sum(0) / W.sum()
        out.append(dict(design=row.design, section=row.section, seed=row.seed, geometry=row.geometry,
                        used=int((share >= 0.01).sum()), near_empty=int((mass < 0.01).sum())))
    return pd.DataFrame(out)


def fig_topics(pdf, core, p2w, stability) -> None:
    usage = topic_usage(pd.concat([core, p2w]))
    usage.to_csv(OUT / "topic_usage.csv", index=False)
    rows = [(f"gplsi_document__P0_raw__{h}", f"GpLSI P0 {HUNTER_LABEL[h]}", "core", BLUE) for h in HUNTERS]
    rows += [("gplsi_anchor__P0_raw__spa_current", "GpLSI anchor (SPA)", "core", BLUE),
             ("gplsi_document__P2_ke_weighted__svs_star", "GpLSI P2 SVS*, core λ", "core", ORANGE),
             ("gplsi_document__P2_ke_weighted__svs_star", "GpLSI P2 SVS*, wide λ", "p2_wide", AQUA)]
    rows += [(m, label, "core", INK2) for m, label in BASELINE_LABEL.items()]

    fig = page("4 · Topic collapse and reproducibility",
               "How many of the K = 7 topics each method actually uses, and how stable its topic profiles are across seeds")
    ax = fig.add_axes((0.2, 0.3, 0.3, 0.56))
    y = np.arange(len(rows))[::-1]
    for yi, (method, label, design, color) in zip(y, rows):
        u = usage[usage.geometry.eq(method) & usage.design.eq(design)]
        ax.barh(yi, u.used.mean(), color=color, height=0.62)
        ax.text(u.used.mean() + 0.1, yi, f"{u.used.mean():.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(7, color=INK2, linestyle=":", linewidth=1)
    ax.set_yticks(y, [r[1] for r in rows]); ax.set_xlim(0, 7.8)
    ax.set_xlabel("topics used (dominant in ≥ 1% of spots), mean of 9 fits")
    ax.set_title("Topics used of K = 7"); ax.grid(axis="y", visible=False)

    ax2 = fig.add_axes((0.62, 0.3, 0.33, 0.56))
    stab = stability[stability.K.eq(7) & stability.panel_size.eq(2000)]
    stab = stab[stab.A_recovery.isna() | stab.A_recovery.eq("A_current")]
    stab = stab.assign(geometry=np.where(stab.family.eq("baseline"), stab.method,
                                         stab.method.str.rsplit("__", n=1).str[0]))
    jsd_rows = [r for r in rows if r[2] == "core"]
    yj = np.arange(len(jsd_rows))[::-1]
    for yi, (method, label, _, color) in zip(yj, jsd_rows):
        value = stab[stab.geometry.eq(method)].seed_JSD.mean()
        ax2.barh(yi, value, color=color, height=0.62)
        ax2.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=7.5, color=INK2)
    ax2.set_yticks(yj, [r[1] for r in jsd_rows]); ax2.set_xlim(0, 0.34)
    ax2.set_xlabel("matched JSD between seeds (lower = more stable)")
    ax2.set_title("Cross-seed stability of Â"); ax2.grid(axis="y", visible=False)
    caption(fig, """
    Many fits use far fewer than seven topics. LDA and Spatial-LDA use all of them, and GpLSI SVS/SVS* about six. The original SPA hunter, PALM and graph KL-NMF use about four, and KL-NMF about two. KE weighting (P2) spends about three vertices on near-empty rare-cell topics, whatever the penalty. Collapse, not noise, explains most of the ARI differences in Fig. 1.
    Stability (right; GpLSI with A_current, P0): pp-SPA gives the most reproducible GpLSI topics, SVS/SVS* are next, and SPA/PALM drift. Spatial-LDA and LDA are the most stable overall, and TopicSCORE the least.
    """, y=0.23)
    pdf.savefig(fig); plt.close(fig)


def fig_A(pdf, core) -> None:
    gp = core[core.family.eq("gplsi_document") & core.has_fit]
    base = core[core.family.eq("baseline")]
    refs = section_stats(base, ["method"], "deviance_floored")["mean"]
    fig = page("5 · Dimension A: topic–gene estimator (prediction only)",
               "Held-out Poisson deviance per molecule (floored at 1e-12), same Ŵ for all three estimators; lower is better")
    ax = fig.add_axes((0.17, 0.3, 0.36, 0.56))
    y = np.arange(len(HUNTERS))[::-1]
    for (A, color), dy in zip([("A_current", BLUE), ("A_full_L2", ORANGE), ("A_full_Pois", AQUA)], [0.22, 0, -0.22]):
        s = section_stats(gp[gp.preprocessing.eq("P0_raw") & gp.A_recovery.eq(A)], ["hunter"], "deviance_floored")
        s = s.reindex(HUNTERS)
        ax.scatter(s["mean"], y + dy, s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=A)
    ax.text(0.99, 0.02, f"for reference: LDA {refs['lda']:.4f}, KL-NMF {refs['kl_nmf']:.4f},\n"
            f"Spatial-LDA {refs['spatial_lda']:.4f} (left of this axis)", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=INK2)
    ax.set_yticks(y, [HUNTER_LABEL[h] for h in HUNTERS]); ax.set_ylim(-0.6, len(HUNTERS) - 0.2)
    ax.set_xlabel("floored deviance / molecule (P0, mean over sections)")
    ax.legend(loc="upper right"); ax.grid(axis="y", visible=False)
    ax.set_title("By hunter")

    gp = gp.assign(geometry=gp.method.str.rsplit("__", n=1).str[0])
    wide = gp.pivot_table(index=["identity", "geometry"], columns="A_recovery", values="deviance_floored",
                          aggfunc="first")
    ax2 = fig.add_axes((0.62, 0.3, 0.33, 0.56))
    contrasts = [("A_full_L2", "A_current"), ("A_full_Pois", "A_current"), ("A_full_Pois", "A_full_L2")]
    rng = np.random.default_rng(0)
    ticks = []
    for i, (b, a) in enumerate(contrasts):
        d = (wide[b] - wide[a]).dropna()
        ax2.scatter(d, np.full(len(d), -i) + rng.uniform(-0.18, 0.18, len(d)), s=7, color=BLUE, alpha=0.45,
                    linewidths=0)
        ticks.append(f"{b} − {a}\n{b} lower in {float((d < 0).mean()):.0%}")
    ax2.axvline(0, color=INK2, linewidth=1)
    ax2.set_yticks([0, -1, -2], ticks); ax2.set_ylim(-2.6, 0.6)
    ax2.set_xlabel("paired difference in floored deviance / molecule (< 0: first is better)")
    ax2.set_title("Paired over 216 core geometries"); ax2.grid(axis="y", visible=False)
    caption(fig, """
    The estimator does not affect layer recovery, since Ŵ is shared. For prediction the order is A_full_Pois > A_full_L2 > A_current: the Poisson MLE wins in 100% of paired geometries and removes most zero-probability held-out molecules (median 38 → 0).
    The gains are largest where the vertices are poor (PALM, SPA) and small for SVS/SVS*. A_full_Pois closes only 14–20% of the gap to LDA for the best-ARI hunters, so the remaining gap comes from Ŵ (smoothed, sparse spot mixtures), not from Â. It is also expensive: about 11–13 CPU-min per fit, against seconds for the other two.
    """, y=0.23)
    pdf.savefig(fig); plt.close(fig)


def fig_panel(pdf, panel) -> None:
    geo = geometry_rows(panel)
    geo = geo[geo.preprocessing.eq("P0_raw")]
    base = panel[panel.family.eq("baseline")]
    sizes = sorted(panel.panel_size.unique())
    fig = page("6 · Dimension C: gene panel size (500 → 5,000 genes)",
               "Panel design: 3 sections × 3 seeds × 4 nested panel sizes; P0 (thresholding P1 gives exactly the same results)")
    ax = fig.add_axes((0.07, 0.34, 0.25, 0.5))
    lines = [(geo[geo.hunter.eq("svs_star")], "GpLSI SVS*", BLUE), (base[base.method.eq("lda")], "LDA", ORANGE),
             (geo[geo.hunter.eq("spa_current")], "GpLSI SPA", AQUA),
             (geo[geo.hunter.eq("palm_accelerated")], "GpLSI PALM acc.", MUTED),
             (base[base.method.eq("topicscore_raw")], "TopicSCORE raw", MUTED),
             (base[base.method.eq("kl_nmf")], "KL-NMF", MUTED)]
    ends = []
    for frame, label, color in lines:
        s = section_stats(frame, ["panel_size"], "layer_ARI")["mean"].reindex(sizes)
        ax.plot(sizes, s.values, color=color, marker="o", markersize=5, linewidth=1.6 if color == MUTED else 2)
        ends.append((s.values[-1], label, INK if color != MUTED else INK2))
    end_labels(ax, sizes[-1], ends, min_gap=0.011)
    ax.set_xscale("log"); ax.set_xticks(sizes, [f"{p:,}" for p in sizes]); ax.minorticks_off()
    ax.set_xlabel("genes in panel"); ax.set_title("Layer ARI"); ax.set_xlim(400, 16000)

    ax2 = fig.add_axes((0.42, 0.34, 0.22, 0.5))
    gp = panel[panel.family.eq("gplsi_document") & panel.preprocessing.eq("P0_raw") & panel.has_fit]
    for (frame, label, color) in [(gp[gp.A_recovery.eq("A_current")], "GpLSI A_current", BLUE),
                                  (gp[gp.A_recovery.eq("A_full_Pois")], "GpLSI A_full_Pois", AQUA),
                                  (base[base.method.eq("lda")], "LDA", ORANGE)]:
        s = section_stats(frame, ["panel_size"], "ref500_deviance_floored")["mean"].reindex(sizes)
        ax2.plot(sizes, s.values, color=color, marker="o", markersize=5)
        end_labels(ax2, sizes[-1], [(s.values[-1], label.replace("GpLSI ", "GpLSI\n"), INK)], min_gap=0)
    ax2.set_xscale("log"); ax2.set_xticks(sizes, [f"{p:,}" for p in sizes]); ax2.minorticks_off()
    ax2.set_xlabel("genes in panel"); ax2.set_xlim(400, 20000)
    ax2.set_title("Deviance, shared 500 genes")

    ax3 = fig.add_axes((0.78, 0.34, 0.18, 0.5))
    rho = geo.groupby(["panel_size", "section"]).selected_rho.median().unstack("section").reindex(sizes)
    ax3.plot(sizes, rho.median(axis=1).values, color=BLUE, marker="o", markersize=5)
    ax3.fill_between(sizes, rho.min(axis=1), rho.max(axis=1), color=BLUE, alpha=0.15, linewidth=0)
    ax3.axhline(GRID_TOP, color=INK2, linestyle=":", linewidth=1)
    ax3.text(sizes[0], GRID_TOP * 0.985, "core grid top", ha="left", va="top", fontsize=7.5, color=INK2)
    ax3.set_xscale("log"); ax3.set_xticks(sizes, [f"{p:,}" for p in sizes]); ax3.minorticks_off()
    ax3.set_xlabel("genes in panel"); ax3.set_title("Selected λ (P0)")
    caption(fig, """
    Panel size barely matters for GpLSI SVS*: its ARI stays at 0.23–0.25 from 500 to 5,000 genes with the same laminar topics, and the extra genes mainly add seed stability and sharpen the deep layers. SPA and PALM change their whole partition with panel size, the same fragility seen across seeds. LDA gains a little from more genes.
    Held-out prediction of the 500 genes shared by every panel is flat for every method, and the GpLSI-to-LDA gap is the same at every size. Small panels want more smoothing: at ≤ 1,000 genes the selected λ sits at the top of the core grid, so extended runs need the 50-point grid there.
    Not plotted: Tran thresholding (α = 0.005) removes 0–85 genes, and P1 − P0 ARI is exactly 0 in 318 of 324 paired records. The thresholding axis is empty on this dataset.
    """, y=0.24)
    pdf.savefig(fig); plt.close(fig)


def fig_p2_wide(pdf, core, p2w) -> None:
    core_geo = geometry_rows(core)
    p0 = core_geo[core_geo.family.eq("gplsi_document") & core_geo.preprocessing.eq("P0_raw")
                  & core_geo.hunter.eq("svs_star")]
    core_p2 = core_geo[core_geo.family.eq("gplsi_document") & core_geo.preprocessing.eq("P2_ke_weighted")]
    wide = geometry_rows(p2w)
    fig = page("7 · KE weighting (P2) on the right penalty grid",
               "P2 GpLSI refit on the 50-point λ grid (top ≈ 0.75) vs the core grid (top ≈ 0.0165); same 9 tasks")
    metrics = [("layer_ARI", "Layer ARI"), ("layer_NMI", "Layer NMI"), ("moran_I", "Moran's I (spatial smoothness)")]
    y = np.arange(len(HUNTERS))[::-1]
    for i, (metric, title) in enumerate(metrics):
        ax = fig.add_axes((0.14 + i * 0.29, 0.34, 0.22, 0.5))
        a = section_stats(core_p2, ["hunter"], metric)["mean"].reindex(HUNTERS)
        b = section_stats(wide, ["hunter"], metric)["mean"].reindex(HUNTERS)
        ax.hlines(y, a, b, color=MUTED, linewidth=1.5)
        ax.scatter(a, y, s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="core λ grid")
        ax.scatter(b, y, s=42, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="wide λ grid")
        ref = section_stats(p0.assign(k="x"), ["k"], metric)["mean"].iloc[0]
        ax.axvline(ref, color=BLUE, linestyle="--", linewidth=1.2)
        ax.text(ref, len(HUNTERS) - 0.35, "P0 SVS*", color=INK, fontsize=7.5, ha="right")
        ax.set_yticks(y, [HUNTER_LABEL[h] for h in HUNTERS] if i == 0 else [""] * len(HUNTERS))
        ax.set_ylim(-0.6, len(HUNTERS) - 0.2); ax.set_title(title); ax.grid(axis="y", visible=False)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=2)
    caption(fig, """
    The core grid under-smoothed KE weighting: its cross-validated λ is 0.10–0.21, 7–13× above the core grid top, while P0's optimum lies inside the core grid (the wide grid picks the same λ with identical results, so it is not plotted). With the right λ every P2 hunter gets smoother maps (Moran's I +0.12 to +0.21) and better soft assignments (NMI +0.05 to +0.10); P2 SVS*'s NMI slightly exceeds P0's.
    ARI does not catch up with P0, because the failure is structural: the vertex hunter picks the over-dispersed rare-cell programs (interneurons, plasma cells, red blood cells) as extreme points at any λ, leaving about four topics for the layers (Fig. 8b).
    """, y=0.24)
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(SUMMARY / "all_records.csv", dtype={"section": str})
    stability = pd.read_csv(SUMMARY / "seed_stability.csv", dtype={"section": str})
    core, panel, p2w = (records[records.design.eq(d)] for d in ["core", "panel", "p2_wide"])
    path = OUT / "dlpfc_ablation_report.pdf"
    with PdfPages(path) as pdf:
        text_page(pdf, "GpLSI ablation on Visium DLPFC: results", [
            ("Setup", [
                "Three donors (sections 151507, 151669, 151673) × 3 seeds, K = 7, 20% of molecules held out, 6-NN spatial "
                "graph. 60 tasks and 1,617 fitted records; the only failures are 4 anchor-GpLSI fits with a rank-deficient Ŵ.",
                "Three dimensions: (A) topic–gene estimator A_current / A_full_L2 / A_full_Pois; (B) vertex hunter SPA, SVS, "
                "SVS*, pp-SPA, PALM, accelerated PALM; (C) preprocessing: raw counts P0, Tran thresholding P1, KE weighting "
                "P2, both P3; and panel size 500–5,000 genes.",
            ]),
            ("Main findings", [
                "Vertex hunting matters most (Figs. 1–4, 8). SVS/SVS* raise GpLSI's layer ARI from 0.09 (SPA) to 0.24, tying "
                "Spatial-LDA at about 1/10 of its runtime, with coherent laminar topics carrying textbook markers. SVS* is the "
                "most robust hunter.",
                "The topic–gene estimator does not affect layer recovery; for prediction A_full_Pois > A_full_L2 > A_current, "
                "but the gap to LDA lives in Ŵ, not in Â (Fig. 5).",
                "Gene thresholding has no effect at any panel size, and panel size barely changes the SVS* results (Fig. 6).",
                "KE weighting is not a good default even at its optimal λ: it spends 3 of 7 vertices on rare cell types "
                "(Figs. 7, 8b).",
                "Penalty grid: P0 at 2,000 genes selects an interior λ; P2/P3 and panels of ≤ 1,000 genes need the wide grid "
                "(Figs. 6, 7).",
            ]),
            ("Caveats", [
                "3 donors only; ARI differences below about 0.03 are within seed noise. Br5595 (151669) has the largest "
                "seed spread; K = 5 does not fix it, partly because its annotation labels the superficial L2/3 cortex "
                "'L3' (results doc §3.15).",
            ]),
        ], subtitle="Generated by scripts/analysis/dlpfc/make_report_pdf.py · details in "
                    "docs/visium_dlpfc_results.md")
        fig_hunters(pdf, core, p2w)
        fig_baselines(pdf, core)
        image_page(pdf, DIAG / "core/spatial/overview_first_seed.png", "3 · Maps: what the layer ARI numbers look like",
                   "Dominant topic per spot, one seed per section; topics coloured by their Hungarian-matched manual layer",
                   """
    GpLSI SVS/SVS* give coherent laminar bands (L1, L2/3, L4/5, deep, WM); the original SPA puts most of the cortex in one topic. Spatial-LDA and LDA get the white matter sharply, but their cortical topics overlap and are speckled. P2 SVS* on the core grid is noisy (Fig. 7 fixes the penalty).
    Colours come from a forced one-to-one matching, so read topic identity from the topic figures (Fig. 8), not the colour.
    """)
        fig_topics(pdf, core, p2w, stability)
        fig_A(pdf, core)
        fig_panel(pdf, panel)
        fig_p2_wide(pdf, core, p2w)
        image_page(pdf, DIAG / "core/topics/topics_151673_s26090401__gplsi_P0_raw__svs_star.png",
                   "8a · Topic anatomy: GpLSI P0 SVS* (151673, ARI 0.28)",
                   "Top: topic weight maps. Bottom: mean weight per layer, correlation with each layer's gene signature, top genes",
                   """
    The topics are real laminar programs with textbook markers: L1 (RELN), L2/3 (CALB1, CARTPT, CUX2, HPCAL1; signature 0.96 with L3), L4/5 (RORB, PVALB, NEFH; 0.85 with L4), an oligodendrocyte L6/WM mixture (MBP, MOBP) and WM (MOG, CLDN11; 0.98). The last topic is a vascular/blood compartment (HBB, C1QC).
    ARI 0.28 understates the biology: the errors are merging adjacent layers and spending topics on non-laminar compartments.
    """, image_box=(0.03, 0.22, 0.94, 0.66))
        p2_topics = DIAG / "p2_wide/topics/topics_151673_s26090401__gplsi_P2_ke_weighted__svs_star.png"
        image_page(pdf, p2_topics, "8b · Topic anatomy: GpLSI P2 SVS*, wide λ grid (151673)",
                   "Same layout; KE weighting at its cross-validated penalty",
                   """
    The maps are smooth, but three of seven topics are near-empty rare-cell programs: interneurons (NPY, SST, CORT, CRHBP), plasma cells (IGHG, IGKC, JCHAIN) and red blood cells (HBA1/2, HBB). One topic takes 58% of the mass and spans L3–L6.
    KE weighting up-weights over-dispersed genes, which makes these rare cells the extreme points the hunter picks. Keeping P2 would need a larger K or a hunter that down-weights isolated, low-mass vertices.
    """, image_box=(0.03, 0.22, 0.94, 0.66))
        text_page(pdf, "Decisions to take", [
            ("For the extended grid (360 tasks)", [
                "Thresholding axis: empty at α = 0.005 at every panel size. Drop it, or try a much stronger α?",
                "KE weighting (P2/P3): drop it, or keep it with the 50-point λ grid and a larger K so rare-cell topics do not "
                "starve the layers?",
                "Hunters: SVS and SVS* are nearly redundant; PALM adds little over SPA. A reduced set could be SPA (original), "
                "SVS* and pp-SPA.",
                "A_full_Pois: about 11–13 CPU-min per geometry. Run it only for the kept hunters, and add a small pseudocount "
                "to remove the remaining zero-probability molecules?",
            ]),
            ("Optional follow-ups", [
                "A fixed-λ sweep (a true λ ablation) needs a small ablation_runner.py change.",
            ]),
        ])
    print("wrote", path)


if __name__ == "__main__":
    main()
