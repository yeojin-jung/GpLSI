"""Summarize DLPFC ablation results into tables and figures.

    python scripts/analysis/dlpfc/summarize.py [--designs core panel lambda_wide lambda_wide_tsgd p2_wide k5_br5595]

Aggregation: seeds are averaged within a section first; sections (one per
donor in the pre-meeting tier) are then averaged, and per-section values are
kept so the three donors can be read separately. W-only metrics (layer ARI/NMI,
spatial) are identical across A recoveries of the same geometry by
construction, so they are read from one row per geometry. Paired A contrasts
use only tasks where both recoveries have finite scores.
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
from scipy.optimize import linear_sum_assignment  # noqa: E402

from records import H5AD, SUMMARY, load_task, load_tasks  # noqa: E402
DONOR = {s: d for d, ss in {
    "Br5292": ["151507", "151508", "151509", "151510"],
    "Br5595": ["151669", "151670", "151671", "151672"],
    "Br8100": ["151673", "151674", "151675", "151676"],
}.items() for s in ss}
LAYER = "layer_guess_reordered"
HUNTER_ORDER = ["spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"]
PRE_ORDER = ["P0_raw", "P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke"]
A_ORDER = ["A_current", "A_full_L2", "A_full_Pois"]
METRICS = {
    f"external__{LAYER}__ari": "layer_ARI",
    f"external__{LAYER}__nmi": "layer_NMI",
    f"external__{LAYER}__cv_balanced_accuracy": "layer_balanced_acc",
    "heldout_poisson_deviance_per_molecule": "deviance_per_molecule",
    "heldout_log_likelihood_per_molecule": "loglik_per_molecule",
    "heldout_zero_probability_molecules": "zero_prob_molecules",
    "heldout_poisson_deviance_per_molecule_floored_1e-12": "deviance_floored",
    "reference_panel__heldout_poisson_deviance_per_molecule": "ref500_deviance_per_molecule",
    "reference_panel__heldout_poisson_deviance_per_molecule_floored_1e-12": "ref500_deviance_floored",
    "reference_panel__heldout_zero_probability_molecules": "ref500_zero_prob_molecules",
    "spatial_topic_moran_mean": "moran_I",
    "spatial_hard_topic_neighbor_agreement": "neighbor_agreement",
    "topic_entropy_mean": "topic_entropy",
    "top_gene_exclusivity_mean": "top_gene_exclusivity",
}
W_METRICS = ["layer_ARI", "layer_NMI", "layer_balanced_acc", "moran_I", "neighbor_agreement"]


def parse_method(method: str) -> dict:
    parts = method.split("__")
    if parts[0] in {"gplsi_document", "gplsi_anchor"} and len(parts) == 4:
        return dict(family=parts[0], preprocessing=parts[1], hunter=parts[2], A_recovery=parts[3])
    return dict(family="baseline", preprocessing="", hunter="", A_recovery="")


def load_records(designs: list[str]) -> tuple[pd.DataFrame, dict]:
    rows, tasks = [], {}
    for design in designs:
        for payload, _ in load_tasks(design):
            task = payload["task"]
            tasks[payload["identity"]] = payload
            for index, record in enumerate(payload["results"]):
                meta = record.get("metadata", {})
                A_meta = meta.get("A_recovery") or {}
                row = dict(
                    identity=payload["identity"], design=design, section=task["unit_id"],
                    donor=DONOR.get(task["unit_id"], ""), K=task["K"],
                    panel_size=task["panel_size"], retained_fraction=task["retained_fraction"],
                    seed=task["seed"], method=record["method"], array_index=index,
                    npz_path=record["npz_path"], A_key="A_hat" if "metrics" in record else "",
                    feature_npz=str(Path(payload["task_dir"]) / "data.npz"),
                    status=record["status"], runtime_seconds=record.get("runtime_seconds"),
                    selected_rho=meta.get("selected_rho"),
                    spectral_iterations=meta.get("spectral_iterations"),
                    retained_feature_count=meta.get("retained_feature_count"),
                    A_converged=A_meta.get("converged"), A_status=A_meta.get("status"),
                    A_iterations=A_meta.get("iterations"),
                    A_gap=A_meta.get("normalized_optimality_gap"),
                    W_refit_converged=meta.get("W_refit_converged"),
                    hunter_converged=(meta.get("vertex_hunting") or {}).get("optimizer_converged"),
                    has_fit="metrics" in record,
                    error=meta.get("exception_type", "") + (": " + meta.get("exception", "")[:160] if meta.get("exception") else ""),
                    **parse_method(record["method"]),
                )
                for key, name in METRICS.items():
                    row[name] = (record.get("metrics") or {}).get(key, np.nan)
                rows.append(row)
    frame = pd.DataFrame(rows)
    for column in METRICS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, tasks


def seed_then_section(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Average over seeds within section, then report per-section and overall means."""

    per_section = frame.groupby(keys + ["section"], dropna=False)[metrics].mean()
    overall = per_section.groupby(level=list(range(len(keys))), dropna=False).mean()
    return per_section, overall


def geometry_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per GpLSI geometry (W-only metrics), from any recovery with a fit."""

    g = frame[frame.family != "baseline"].copy()
    g["geometry"] = g.method.str.rsplit("__", n=1).str[0]
    g = g[g.has_fit].sort_values("A_recovery")
    return g.drop_duplicates(["identity", "geometry"])


def fmt_table(table: pd.DataFrame, digits: int = 3) -> str:
    return table.to_string(float_format=lambda v: f"{v:.{digits}f}")


def topic_alignment_jsd(A1, g1, A2, g2) -> float:
    """Mean matched Jensen-Shannon divergence (base 2) over shared genes."""

    shared, i1, i2 = np.intersect1d(g1, g2, return_indices=True)
    P = np.maximum(A1[:, i1], 1e-15); P /= P.sum(1, keepdims=True)
    Q = np.maximum(A2[:, i2], 1e-15); Q /= Q.sum(1, keepdims=True)
    M = 0.5 * (P[:, None, :] + Q[None, :, :])
    kl = lambda X, Y: np.sum(X * np.log2(X / Y), axis=-1)  # noqa: E731
    cost = 0.5 * kl(P[:, None, :], M) + 0.5 * kl(Q[None, :, :], M)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def seed_stability(frame: pd.DataFrame, design: str) -> pd.DataFrame:
    """Cross-seed topic-profile reproducibility (matched JSD, lower is better)."""

    out = []
    sub = frame[(frame.design == design) & frame.has_fit]
    for (section, K, panel, method), group in sub.groupby(["section", "K", "panel_size", "method"]):
        if group.seed.nunique() < 2:
            continue
        loaded = []
        for _, row in group.iterrows():
            with np.load(row.feature_npz, allow_pickle=True) as z:
                genes = z["feature_ids"].astype(str)
            with np.load(row.npz_path, allow_pickle=True) as z:
                loaded.append((z[row.A_key].astype(float), genes))
        values = [topic_alignment_jsd(*loaded[i], *loaded[j]) for i, j in combinations(range(len(loaded)), 2)]
        out.append(dict(section=section, K=K, panel_size=panel, method=method, seed_JSD=float(np.mean(values))))
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", nargs="+", default=["core", "panel", "lambda_wide", "lambda_wide_tsgd", "p2_wide", "k5_br5595"])
    parser.add_argument("--no-stability", action="store_true")
    args = parser.parse_args()
    out = SUMMARY
    out.mkdir(parents=True, exist_ok=True)
    frame, tasks = load_records(args.designs)
    if frame.empty:
        raise SystemExit("no results found")
    frame.to_csv(out / "all_records.csv", index=False)
    report: list[str] = [f"# DLPFC ablation summary\n\nTasks: {frame.identity.nunique()}  records: {len(frame)}\n"]

    def section(title, body):
        report.append(f"\n## {title}\n\n```\n{body}\n```\n")

    # ---- Failures / convergence --------------------------------------------------
    status = frame.assign(ok=frame.status.eq("ok"), fit=frame.has_fit).groupby(
        ["design", "family", "A_recovery"], dropna=False)[["ok", "fit"]].mean()
    section("Share of records with status ok / with a finite fit", fmt_table(status))
    failed = frame[~frame.has_fit].groupby(["design", "method"]).error.first()
    if len(failed):
        section("Failed records (first error message)", failed.to_string())

    core = frame[frame.design == "core"]
    if not core.empty:
        geo = geometry_rows(core)
        doc = geo[geo.family == "gplsi_document"]
        per_sec, overall = seed_then_section(doc, ["preprocessing", "hunter"], W_METRICS)
        ari = per_sec["layer_ARI"].unstack("section")
        ari["mean"] = ari.mean(axis=1)
        section("Axes 1+2: layer ARI by preprocessing x hunter (seed-averaged; columns = sections)", fmt_table(ari))
        section("Axes 1+2: W-only metrics, mean over sections", fmt_table(overall))
        hunter_marginal = seed_then_section(doc, ["hunter"], W_METRICS)[1]
        pre_marginal = seed_then_section(doc, ["preprocessing"], W_METRICS)[1]
        section("Axis 1 (vertex hunting) marginal over preprocessings", fmt_table(hunter_marginal))
        section("Axis 2 (preprocessing) marginal over hunters", fmt_table(pre_marginal))
        retained = doc.groupby("preprocessing").retained_feature_count.agg(["min", "median", "max"])
        section("Genes retained by the spectral step (of 2,000)", fmt_table(retained, 0))
        rho = doc.groupby("preprocessing").selected_rho.agg(["min", "median", "max"])
        section("Selected graph penalty (grid max ~0.0165 in core)", fmt_table(rho, 5))

        # ---- Axis 3: A recoveries, paired on common support ---------------------
        gp = core[core.family == "gplsi_document"].copy()
        gp["geometry"] = gp.method.str.rsplit("__", n=1).str[0]
        def paired(value):
            return gp[gp.has_fit].pivot_table(index=["identity", "section", "geometry"], columns="A_recovery",
                                              values=value, aggfunc="first")

        wide, floored, zero = paired("deviance_per_molecule"), paired("deviance_floored"), paired("zero_prob_molecules")
        lines = []
        for a, b in [("A_current", "A_full_Pois"), ("A_current", "A_full_L2"), ("A_full_L2", "A_full_Pois")]:
            for label, table in [("deviance/molecule (both finite)", wide), ("floored deviance/molecule", floored),
                                 ("zero-prob molecules", zero)]:
                if a in table and b in table:
                    pair = table[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
                    if pair.empty:
                        lines.append(f"{b} - {a}: {label}: no geometry has both scores")
                        continue
                    lines.append(f"{b} - {a}: {label}: mean difference {float((pair[b] - pair[a]).mean()):+.4f}, "
                                 f"median {float((pair[b] - pair[a]).median()):+.4f} "
                                 f"(n={len(pair)}; {b} lower in {float((pair[b] < pair[a]).mean()):.0%}, "
                                 f"tied in {float((pair[b] == pair[a]).mean()):.0%})")
        for a in A_ORDER:
            if a in zero:
                z = zero[a].dropna()
                lines.append(f"{a}: geometries with any zero-probability held-out molecule {float((z > 0).mean()):.0%}; "
                             f"median zero-prob molecules {float(z.median()):.0f}")
        conv = gp.groupby("A_recovery").A_converged.apply(lambda s: s.dropna().astype(bool).mean())
        lines.append("A-recovery convergence rate: " + ", ".join(f"{k}={v:.0%}" for k, v in conv.items()))
        pois = gp[(gp.A_recovery == "A_full_Pois") & gp.has_fit]
        if not pois.empty:
            gap = pd.to_numeric(pois.A_gap, errors="coerce").dropna()
            lines.append(
                f"A_full_Pois: {int(pois.A_converged.fillna(False).astype(bool).sum())}/{len(pois)} reached the 1e-8 tolerance; "
                f"the rest are near-converged at the iteration cap with normalized gap "
                f"median {gap.median():.2e}, max {gap.max():.2e} (n={len(gap)})"
                if len(gap) else "A_full_Pois: no normalized gap recorded")
        failed_A = gp[~gp.has_fit].groupby("A_recovery").size()
        if len(failed_A):
            lines.append("Failed A recoveries (no fit): " + ", ".join(f"{k}={v}" for k, v in failed_A.items()))
        section("Axis 3 (A recovery): paired contrasts on identical W", "\n".join(lines))
        scored = gp[gp.has_fit].replace([np.inf, -np.inf], np.nan)
        scored = scored.assign(finite_deviance_share=scored.deviance_per_molecule.notna().astype(float))
        deviance = seed_then_section(scored, ["A_recovery"],
                                     ["finite_deviance_share", "deviance_per_molecule", "deviance_floored",
                                      "zero_prob_molecules", "topic_entropy", "top_gene_exclusivity"])[1]
        section("Axis 3: prediction / profile metrics by A recovery "
                "(deviance_per_molecule averages only the finite share, so compare deviance_floored across rows)",
                fmt_table(deviance, 4))

        # ---- Baselines vs GpLSI ---------------------------------------------------
        base = core[core.family == "baseline"]
        best = doc.groupby(["preprocessing", "hunter"]).layer_ARI.mean().idxmax()
        chosen = pd.concat([
            base,
            doc[(doc.preprocessing == best[0]) & (doc.hunter == best[1])].assign(method=f"GpLSI best ({best[0]}, {best[1]})"),
            doc[(doc.preprocessing == "P0_raw") & (doc.hunter == "spa_current")].assign(method="GpLSI original (P0, SPA)"),
            geo[geo.family == "gplsi_anchor"].assign(method="GpLSI anchor (P0, SPA)"),
        ])
        comp = seed_then_section(chosen, ["method"], ["layer_ARI", "layer_NMI", "layer_balanced_acc", "moran_I"])[0]
        comp = comp["layer_ARI"].unstack("section")
        comp["mean"] = comp.mean(axis=1)
        section("GpLSI vs baselines: layer ARI (columns = sections)", fmt_table(comp.sort_values("mean", ascending=False)))
        refit = base.dropna(subset=["W_refit_converged"])
        if not refit.empty:
            flag = refit.groupby("method").W_refit_converged.agg(
                converged=lambda s: int(s.astype(bool).sum()), records="size")
            section("Baselines whose W refit did not converge (status still 'ok'; read with care)", flag.to_string())
        base_dev = seed_then_section(base, ["method"], ["deviance_floored", "zero_prob_molecules"])[1]
        section("Baselines: floored held-out deviance / molecule and zero-probability molecules", fmt_table(base_dev, 4))
        runtime = core.groupby(["family"]).runtime_seconds.median()
        section("Median runtime per record (s)", fmt_table(runtime, 1))

        # Figure: ARI heatmap preprocessing x hunter.
        mean_ari = doc.groupby(["preprocessing", "hunter"]).layer_ARI.mean().unstack("hunter")
        mean_ari = mean_ari.reindex(index=[p for p in PRE_ORDER if p in mean_ari.index],
                                    columns=[h for h in HUNTER_ORDER if h in mean_ari.columns])
        fig, ax = plt.subplots(figsize=(8, 3.6))
        im = ax.imshow(mean_ari.values, cmap="viridis")
        ax.set_xticks(range(mean_ari.shape[1]), mean_ari.columns, rotation=30, ha="right")
        ax.set_yticks(range(mean_ari.shape[0]), mean_ari.index)
        for (i, j), v in np.ndenumerate(mean_ari.values):
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}", ha="center", va="center",
                    color="white" if v < np.nanmean(mean_ari.values) else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="layer ARI")
        ax.set_title("DLPFC layer ARI (mean over sections and seeds)")
        fig.tight_layout(); fig.savefig(out / "fig_ari_preprocessing_by_hunter.png", dpi=180); plt.close(fig)

        # Figure: spatial maps for one seed of each section.
        plot_spatial_maps(core, doc, best, tasks, out)

        if not args.no_stability:
            stab = seed_stability(core, "core")
            if not stab.empty:
                stab = stab.assign(**stab.method.apply(parse_method).apply(pd.Series))
                table = stab[stab.family == "gplsi_document"].groupby(["preprocessing", "hunter", "A_recovery"]).seed_JSD.mean()
                section("Cross-seed topic reproducibility (matched JSD, lower = more stable)", fmt_table(table.unstack("A_recovery"), 4))
                section("Cross-seed reproducibility, baselines", fmt_table(stab[stab.family == "baseline"].groupby("method").seed_JSD.mean(), 4))
                stab.to_csv(out / "seed_stability.csv", index=False)

    panel = frame[frame.design == "panel"]
    if not panel.empty:
        geo = geometry_rows(panel)
        w = seed_then_section(geo, ["preprocessing", "hunter", "panel_size"], ["layer_ARI", "layer_NMI"])[1]
        section("Axis 2b (panel size): layer ARI/NMI", fmt_table(w["layer_ARI"].unstack("panel_size")))
        # Floored: the unfloored reference deviance is infinite for most GpLSI fits, so a
        # finite-only average would compare different subsets of fits across panel sizes.
        gp = panel[panel.family == "gplsi_document"]
        ref = seed_then_section(gp, ["preprocessing", "hunter", "A_recovery", "panel_size"], ["ref500_deviance_floored"])[1]
        ref = ref.groupby(level=["preprocessing", "A_recovery", "panel_size"]).mean()
        section("Axis 2b: floored held-out deviance on the common 500-gene reference panel (lower is better; mean over hunters)",
                fmt_table(ref["ref500_deviance_floored"].unstack("panel_size"), 4))
        zero = seed_then_section(gp, ["preprocessing", "A_recovery", "panel_size"], ["ref500_zero_prob_molecules"])[1]
        section("Axis 2b: zero-probability held-out molecules on the reference panel",
                fmt_table(zero["ref500_zero_prob_molecules"].unstack("panel_size"), 1))
        kept = geo.groupby(["preprocessing", "panel_size"]).retained_feature_count.median().unstack("panel_size")
        section("Axis 2b: genes retained by the spectral step", fmt_table(kept, 0))
        base = panel[panel.family == "baseline"]
        if not base.empty:
            b = seed_then_section(base, ["method", "panel_size"], ["layer_ARI"])[1]["layer_ARI"].unstack("panel_size")
            section("Axis 2b: baselines layer ARI by panel size", fmt_table(b))
            bd = seed_then_section(base, ["method", "panel_size"], ["ref500_deviance_floored"])[1]
            section("Axis 2b: baselines floored reference-panel deviance by panel size",
                    fmt_table(bd["ref500_deviance_floored"].unstack("panel_size"), 4))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for (pre, hunter), g in w["layer_ARI"].groupby(level=[0, 1]):
            axes[0].plot(g.index.get_level_values("panel_size"), g.values, marker="o", label=f"{pre[:2]} {hunter}")
        for (pre, a), g in ref["ref500_deviance_floored"].groupby(level=[0, 1]):
            axes[1].plot(g.index.get_level_values("panel_size"), g.values, marker="o", label=f"{pre[:2]} {a}")
        for ax, label in zip(axes, ["layer ARI", "held-out deviance / molecule (ref. 500 genes)"]):
            ax.set_xscale("log"); ax.set_xlabel("genes in panel (p)"); ax.set_ylabel(label); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig_panel_size.png", dpi=180); plt.close(fig)

    lam = frame[frame.design == "lambda_wide"]
    if not lam.empty:
        geo = geometry_rows(lam)
        compare = pd.concat([geo, geometry_rows(core[core.seed.isin(lam.seed.unique())]).assign(design="core")])
        # Anchor GpLSI is also P0/spa_current; keep only the document geometries lambda_wide reruns.
        compare = compare[compare.family.eq("gplsi_document") & compare.hunter.eq("spa_current")
                          & compare.preprocessing.isin(lam.preprocessing.unique())]
        table = compare.groupby(["preprocessing", "design", "section"])[["selected_rho", "layer_ARI", "moran_I"]].mean()
        section("Penalty-grid sensitivity (core grid max ~0.0165 vs wide grid max ~0.75)", fmt_table(table, 4))

    p2w = frame[frame.design == "p2_wide"]
    if not p2w.empty:
        compare = pd.concat([geometry_rows(p2w), geometry_rows(core[core.preprocessing.eq("P2_ke_weighted")]).assign(design="core")])
        compare = compare[compare.family.eq("gplsi_document")]
        section("P2 on the wide penalty grid (p2_wide) vs core grid: selected penalty",
                fmt_table(compare.groupby(["design", "section"]).selected_rho.agg(["min", "median", "max"]), 4))
        w = seed_then_section(compare, ["hunter", "design"], ["layer_ARI"])
        section("P2 wide vs core grid: layer ARI by hunter (columns = sections)",
                fmt_table(w[0]["layer_ARI"].unstack("section").assign(mean=w[1]["layer_ARI"])))
        section("P2 wide vs core grid: W-only metrics, mean over sections",
                fmt_table(seed_then_section(compare, ["hunter", "design"], W_METRICS)[1]))
        gp = pd.concat([p2w, core[core.preprocessing.eq("P2_ke_weighted")].assign(design="core")])
        gp = gp[gp.family.eq("gplsi_document") & gp.has_fit]
        section("P2 wide vs core grid: prediction by A recovery",
                fmt_table(seed_then_section(gp, ["A_recovery", "design"], ["deviance_floored", "zero_prob_molecules"])[1], 4))

    k5 = frame[frame.design == "k5_br5595"]
    if not k5.empty:
        # K = 7 references on the same section and seeds: core for P0 and baselines, p2_wide
        # for P2 (core's P2 used the short penalty grid).
        same = frame.section.isin(k5.section.unique()) & frame.seed.isin(k5.seed.unique())
        ref = pd.concat([
            frame[same & frame.design.eq("core") & ~frame.preprocessing.isin(["P1_tran_alpha_0p005", "P2_ke_weighted", "P3_tran_then_ke"])],
            frame[same & frame.design.eq("p2_wide")],
        ])
        both = pd.concat([ref, k5])
        geo = geometry_rows(both)
        geo = geo[geo.family.eq("gplsi_document")]
        w = geo.groupby(["preprocessing", "hunter", "K"])[["layer_ARI", "layer_NMI", "moran_I"]].agg(["mean", "min", "max"])
        section("K = 5 vs K = 7 on Br5595: GpLSI W-only metrics over seeds (mean, min, max)", fmt_table(w))
        base = both[both.family.eq("baseline") & both.has_fit]
        b = base.groupby(["method", "K"])[["layer_ARI", "layer_NMI", "deviance_floored"]].mean()
        section("K = 5 vs K = 7 on Br5595: baselines (seed mean)", fmt_table(b, 4))
        gp = both[both.family.eq("gplsi_document") & both.has_fit]
        d = gp.groupby(["preprocessing", "A_recovery", "K"])[["deviance_floored", "zero_prob_molecules"]].mean()
        section("K = 5 vs K = 7 on Br5595: GpLSI prediction by A recovery (mean over hunters and seeds)", fmt_table(d, 4))
        rho = geo.groupby(["preprocessing", "K"]).selected_rho.agg(["min", "median", "max"])
        section("K = 5 vs K = 7 on Br5595: selected penalty", fmt_table(rho, 4))
        if not args.no_stability:
            # Matched JSD averages over K matched pairs, so it is comparable across K only roughly.
            sections_k5 = set(k5.section)
            stab = pd.concat([seed_stability(frame, "k5_br5595"), seed_stability(frame[same], "core")])
            if not stab.empty:
                stab = stab[stab.section.isin(sections_k5) & ~stab.method.str.contains("P1_tran|P2_ke_weighted|P3_tran_then_ke|A_full_Pois")]
                section("K = 5 vs K = 7 on Br5595: cross-seed matched JSD (P0 and baselines)",
                        fmt_table(stab.pivot_table(index="method", columns="K", values="seed_JSD"), 4))

    tsgd = frame[frame.design == "lambda_wide_tsgd"]
    if not tsgd.empty:
        # Each graph-denoised TopicSCORE row records the penalty of the P0
        # spectral block it reuses.
        rows = frame[frame.method.eq("topicscore_graph_denoised") & frame.design.isin(["core", "lambda_wide_tsgd"])
                     & frame.seed.isin(tsgd.seed.unique())].copy()
        table = rows.groupby(["design", "section"])[["selected_rho", "layer_ARI", "deviance_floored"]].mean()
        section("Graph-denoised TopicSCORE: penalty-grid sensitivity (shares the P0 spectral step)", fmt_table(table, 4))

    (out / "report.md").write_text("".join(report))
    print("".join(report))
    print(f"\nWrote {out}")


def plot_spatial_maps(core, doc, best, tasks, out) -> None:
    """Hard-topic maps next to manual layers for the best GpLSI and key baselines."""

    import anndata as ad

    obs = ad.read_h5ad(H5AD, backed="r").obs[[LAYER]]
    methods = {
        "GpLSI best": f"gplsi_document__{best[0]}__{best[1]}__A_current",
        "GpLSI original": "gplsi_document__P0_raw__spa_current__A_current",
        "LDA": "lda", "Spatial-LDA": "spatial_lda", "graph KL-NMF": "graph_kl_nmf",
    }
    first_seed = core.seed.min()
    sections = sorted(core.section.unique())
    fig, axes = plt.subplots(len(sections), len(methods) + 1, figsize=(2.6 * (len(methods) + 1), 2.8 * len(sections)),
                             squeeze=False)
    for r, section_id in enumerate(sections):
        ident = core[(core.section == section_id) & (core.seed == first_seed)].identity.iloc[0]
        payload = tasks[ident]
        _, z = load_task(Path(payload["task_dir"]))
        xy = z["coordinates"]; ids = z["observation_ids"].astype(str)
        labels = obs.loc[ids, LAYER].astype(str).to_numpy()
        layer_codes = pd.Categorical(labels, categories=sorted(set(labels) - {""})).codes
        panels = [("manual layers", np.where(layer_codes < 0, np.nan, layer_codes))]
        for title, method in methods.items():
            index = payload["array_index"].get(method)
            key = f"W_{index}"
            panels.append((title, z[key].argmax(1) if index is not None and key in z else None))
        for c, (title, values) in enumerate(panels):
            ax = axes[r, c]
            if values is not None:
                ax.scatter(xy[:, 0], -xy[:, 1], c=values, s=2, cmap="tab10", linewidths=0)
            ax.set_title(f"{section_id} {title}" if c == 0 else title, fontsize=8)
            ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "fig_spatial_maps.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
