"""Summarize DLPFC ablation results into tables and figures.

    python scripts/visium_dlpfc/summarize.py [--designs core panel lambda_wide]

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
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
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
        base = REPO / "results/visium_dlpfc" / design
        sources = [(path, path.with_suffix(".npz"), False) for path in sorted(base.glob("*.json"))]
        sources += [(path, path.with_suffix(".npz"), True) for path in sorted((base / "poisson").glob("*.json"))]
        for path, npz_path, is_refit in sources:
            payload = json.loads(path.read_text())
            task = payload["task"]
            if not is_refit:
                tasks[payload["identity"]] = payload
            array_position = -1
            for index, record in enumerate(payload["results"]):
                if "metrics" in record:
                    array_position += 1
                A_key = f"A_{array_position if is_refit else index}"
                meta = record.get("metadata", {})
                A_meta = meta.get("A_recovery", {})
                row = dict(
                    identity=payload["identity"], design=design, section=str(task["unit_id"]),
                    donor=DONOR.get(str(task["unit_id"]), ""), K=int(task["K"]),
                    panel_size=int(task["panel_size"]), retained_fraction=float(task["retained_fraction"]),
                    seed=int(task["seed"]), method=record["method"], array_index=index,
                    npz_path=str(npz_path), A_key=A_key if "metrics" in record else "",
                    feature_npz=str(base / f"{payload['identity']}.npz"),
                    status=record["status"], runtime_seconds=record.get("runtime_seconds"),
                    selected_rho=meta.get("selected_rho"),
                    spectral_iterations=meta.get("spectral_iterations"),
                    retained_feature_count=meta.get("retained_feature_count"),
                    A_converged=A_meta.get("converged"), A_status=A_meta.get("status"),
                    A_iterations=A_meta.get("iterations"),
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
    parser.add_argument("--designs", nargs="+", default=["core", "panel", "lambda_wide"])
    parser.add_argument("--no-stability", action="store_true")
    args = parser.parse_args()
    out = REPO / "results/visium_dlpfc/summary"
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
        wide = gp.pivot_table(index=["identity", "section", "geometry"], columns="A_recovery",
                              values="deviance_per_molecule", aggfunc="first")
        zero = gp.pivot_table(index=["identity", "section", "geometry"], columns="A_recovery",
                              values="zero_prob_molecules", aggfunc="first")
        lines = []
        for a, b in [("A_current", "A_full_Pois"), ("A_current", "A_full_L2"), ("A_full_L2", "A_full_Pois")]:
            if a in wide and b in wide:
                pair = wide[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
                lines.append(f"{b} - {a}: mean deviance/molecule difference {float((pair[b] - pair[a]).mean()):+.4f} "
                             f"(n={len(pair)} geometries with both finite; {b} better in {float((pair[b] < pair[a]).mean()):.0%})")
        for a in A_ORDER:
            if a in zero:
                z = zero[a].dropna()
                lines.append(f"{a}: geometries with any zero-probability held-out molecule {float((z > 0).mean()):.0%}; "
                             f"median zero-prob molecules {float(z.median()):.0f}")
        conv = gp.groupby("A_recovery").A_converged.apply(lambda s: s.dropna().astype(bool).mean())
        lines.append("A-recovery convergence rate: " + ", ".join(f"{k}={v:.0%}" for k, v in conv.items()))
        section("Axis 3 (A recovery): paired contrasts on identical W", "\n".join(lines))
        deviance = seed_then_section(gp.replace([np.inf, -np.inf], np.nan), ["A_recovery"],
                                     ["deviance_per_molecule", "deviance_floored", "topic_entropy", "top_gene_exclusivity"])[1]
        section("Axis 3: prediction / profile metrics by A recovery (finite scores only)", fmt_table(deviance, 4))

        # ---- Baselines vs GpLSI ---------------------------------------------------
        base = core[core.family == "baseline"]
        best = doc.groupby(["preprocessing", "hunter"]).layer_ARI.mean().idxmax()
        chosen = pd.concat([
            base,
            doc[(doc.preprocessing == best[0]) & (doc.hunter == best[1])].assign(method=f"GpLSI best ({best[0]}, {best[1]})"),
            doc[(doc.preprocessing == "P0_raw") & (doc.hunter == "spa_current")].assign(method="GpLSI original (P0, SPA)"),
        ])
        comp = seed_then_section(chosen, ["method"], ["layer_ARI", "layer_NMI", "layer_balanced_acc", "moran_I"])[0]
        comp = comp["layer_ARI"].unstack("section")
        comp["mean"] = comp.mean(axis=1)
        section("GpLSI vs baselines: layer ARI (columns = sections)", fmt_table(comp.sort_values("mean", ascending=False)))
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
        gp = panel[panel.family == "gplsi_document"].replace([np.inf, -np.inf], np.nan)
        ref = seed_then_section(gp, ["preprocessing", "A_recovery", "panel_size"], ["ref500_deviance_per_molecule"])[1]
        section("Axis 2b: held-out deviance on the common 500-gene reference panel (lower is better)",
                fmt_table(ref["ref500_deviance_per_molecule"].unstack("panel_size"), 4))
        kept = geo.groupby(["preprocessing", "panel_size"]).retained_feature_count.median().unstack("panel_size")
        section("Axis 2b: genes retained by the spectral step", fmt_table(kept, 0))
        base = panel[panel.family == "baseline"]
        if not base.empty:
            b = seed_then_section(base, ["method", "panel_size"], ["layer_ARI"])[1]["layer_ARI"].unstack("panel_size")
            section("Axis 2b: baselines layer ARI by panel size", fmt_table(b))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for (pre, hunter), g in w["layer_ARI"].groupby(level=[0, 1]):
            axes[0].plot(g.index.get_level_values("panel_size"), g.values, marker="o", label=f"{pre[:2]} {hunter}")
        for (pre, a), g in ref["ref500_deviance_per_molecule"].groupby(level=[0, 1]):
            axes[1].plot(g.index.get_level_values("panel_size"), g.values, marker="o", label=f"{pre[:2]} {a}")
        for ax, label in zip(axes, ["layer ARI", "held-out deviance / molecule (ref. 500 genes)"]):
            ax.set_xscale("log"); ax.set_xlabel("genes in panel (p)"); ax.set_ylabel(label); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig_panel_size.png", dpi=180); plt.close(fig)

    lam = frame[frame.design == "lambda_wide"]
    if not lam.empty:
        geo = geometry_rows(lam)
        compare = pd.concat([geo, geometry_rows(core[core.seed.isin(lam.seed.unique())]).assign(design="core")])
        compare = compare[compare.hunter.eq("spa_current") & compare.preprocessing.isin(lam.preprocessing.unique())]
        table = compare.groupby(["preprocessing", "design", "section"])[["selected_rho", "layer_ARI", "moran_I"]].mean()
        section("Penalty-grid sensitivity (core grid max ~0.0165 vs wide grid max ~0.75)", fmt_table(table, 4))

    (out / "report.md").write_text("".join(report))
    print("".join(report))
    print(f"\nWrote {out}")


def plot_spatial_maps(core, doc, best, tasks, out) -> None:
    """Hard-topic maps next to manual layers for the best GpLSI and key baselines."""

    import anndata as ad

    obs = ad.read_h5ad(REPO / "data/processed/visium_dlpfc/visium_dlpfc.h5ad", backed="r").obs[[LAYER]]
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
        with np.load(REPO / "results/visium_dlpfc/core" / f"{ident}.npz", allow_pickle=True) as z:
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
