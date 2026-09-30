# Visium DLPFC ablation: results and experiment status

This is a living document: the single place that records what has run, what the results
are, and what is left. The design, methods, formulas and metrics are defined in
[`visium_dlpfc_ablation.md`](visium_dlpfc_ablation.md), and the smoke-test report is in
[`visium_dlpfc_smoke_test.md`](visium_dlpfc_smoke_test.md).
**What each metric means and how to read it: [Appendix A](#appendix-a-metrics-guide).**

**Last updated:** 2026-09-28. Status: **every planned pre-meeting design is done**:
core, lambda_wide, lambda_wide_tsgd, panel, the Poisson refits (core, panel, lambda_wide,
p2_wide), the p2_wide rerun (§3.14) and K = 5 on Br5595 (§3.15). The tables come from
`summarize.py` over all designs (`results/visium_dlpfc/summary/report.md`: 63 tasks,
1,767 records).
**Start with [§0 Summary](#0-summary-for-the-advisor-meeting).** A figure-only version of
the main tables (11 pages) is `results/visium_dlpfc/report/dlpfc_ablation_report.pdf`, made by
`scripts/visium_dlpfc/make_report_pdf.py` (rerun it after `summarize.py`).

Branch `yeojin-exp` (uncommitted working changes, see §6). Cluster: DSI, partition
`general`, account `general_group`, project directory `/net/projects2/mercury/yeojin/GpLSI`.
There is a user limit of 10 concurrent jobs.

---

## 0. Summary for the advisor meeting

Three donors (sections 151507, 151669, 151673) × 3 seeds, K = 7, held-out 20% of
molecules. Across 60 tasks and 1,617 fitted records, the only failures are 4 anchor-GpLSI
fits whose Ŵ is rank-deficient. Figures are under `results/visium_dlpfc/diagnostics/`.

**Main findings, in order of importance:**

1. **Vertex hunting is the dimension that matters (§3.2, §3.11).** Replacing the original
   SPA hunter with SVS/SVS* raises GpLSI's layer ARI from 0.09 to 0.24 (P0). That ties
   Spatial-LDA (0.24) and LDA (0.23) at a small fraction of Spatial-LDA's cost (about 30 s vs 430 s per fit), and gives more
   coherent laminar maps than either. The topics are real programs with textbook markers:
   L1 (RELN), L2/3 (CARTPT, HPCAL1, CUX2), L4/5 (RORB, PVALB, NEFH) and WM (MOG, CLDN11).
   SPA/PALM collapse to about 4 used topics and are fragile across seeds and panel sizes.
   **svs_star is the most robust hunter**: it is at or near the top on every section, under
   both weightings and at every panel size.
2. **The topic–gene estimator (dimension A) does not affect layer recovery, and ranks
   A_full_Pois > A_full_L2 > A_current for prediction (§3.12).** A_full_Pois improves
   held-out deviance in 100% of 216 paired geometries and removes most
   zero-probability molecules (median 38 → 0). Its gains are small for good hunters,
   though: it closes only 14–20% of the deviance gap to LDA, so **that gap lives in Ŵ
   (smoothed, sparse spot mixtures), not in Â.** It is also expensive (about 13 CPU-min
   per geometry, against seconds).
3. **Gene thresholding (Tran α = 0.005) has no effect at any panel size (§3.13)**: P1 = P0
   exactly in 318 of 324 records. Panel size (500–5,000 genes) barely changes svs_star (ARI 0.234 →
   0.248, the same topics, better seed stability), and held-out prediction of the common 500 genes is
   flat for every method.
4. **KE weighting (P2/P3) is not a good default, even at its optimal λ (§3.10, §3.14).**
   The core grid under-smoothed it (its CV optimum is 7–13× above the grid). At the right
   λ its maps become clean and NMI rises, but ARI stays below P0, because 3 of 7 vertices
   go to rare cell types (interneurons, plasma cells, red blood cells) whatever the
   smoothing.
5. **Penalty selection:** P0 at 2,000 genes selects an interior λ, but P2/P3 and panels
   of ≤ 1,000 genes need the wide grid (`grid_len` 50).

**Caveats:** 3 donors only; ARI differences below about 0.03 are within seed noise.
Br5595 (151669) has the largest seed spread throughout. K = 5 (its number of labelled
layers) does not fix this (§3.15): part of the problem is the annotation itself, which
labels the upper cortex "L3", while every good fit finds a separate superficial L2/3
program there.

**Decisions to take** are listed in §7.

---

## 1. Experiment status

| Design | What it tests | Tasks | Slurm job | State | Wall time per job | Output |
|---|---|---|---|---|---|---|
| smoke | Wiring check: full method suite with a 3-point λ grid | 1 | 1917480 | ✅ done | 21.6 min | `results/visium_dlpfc/smoke/` |
| **core** | All 3 dimensions on the 2,000-gene panel; all baselines | 9 | 1935143 (array 0–8) | ✅ done, all COMPLETED | 13–28 min | `results/visium_dlpfc/core/` |
| lambda_wide | Penalty-grid sensitivity for GpLSI (grid to λ ≈ 0.75) | 3 | 1937108 (array 0–2) | ✅ done, all COMPLETED (see §3.10) | 33–38 min | `results/visium_dlpfc/lambda_wide/` |
| lambda_wide_tsgd | Same wide grid for graph-denoised TopicSCORE, which reuses the P0 spectral step | 3 | 1937153 (array 0–2) | ✅ done, all COMPLETED (see §3.9) | 16–30 min | `results/visium_dlpfc/lambda_wide_tsgd/` |
| A_full_Pois refit (core) | Poisson maximum-likelihood A from each saved core Ŵ | 9 tasks × 25 geometries | 1937260 (single job, 9 workers) | ✅ done, 225/225 fits, 0 failed; 4 reached 1e-8, the rest near-converged (§3.12) | 5 h 24 min | `results/visium_dlpfc/core/poisson/` |
| panel | Panel size 500/1,000/2,000/5,000 × thresholding on/off | 36 | 1937268 (array 0–2, `TASK_STRIDE=3`: each job runs 12 tasks, one seed) | ✅ done, 36/36; 540 records: 422 ok, 118 max_iter_reached, 0 failed | 64–97 min per job; 2.6 / 4.2 / 6.9 / 14.2 min per task at p = 500 / 1,000 / 2,000 / 5,000 | `results/visium_dlpfc/panel/` |
| A_full_Pois refit (panel + lambda_wide) | Same, for the 36 panel and 3 lambda_wide tasks | 39 tasks × 6 or 2 geometries | 1938320 (single job, 9 workers) | ✅ done, 222/222 fits, 0 failed; near-converged except 12 panel fits that reached 1e-8 | 5 h 18 min | `results/visium_dlpfc/{panel,lambda_wide}/poisson/` |
| **p2_wide** (new) | Core P2 (KE weighting) × 6 hunters × A_current/A_full_L2 on the 50-point λ grid, to test whether P2's losses were a grid artefact | 9 | 1939519 (array 0–4, `TASK_STRIDE=5`) | ✅ done, 9/9; 108 records: 80 ok, 28 max_iter_reached (A_full_L2), 0 failed (§3.14) | 18–33 min per job (2 tasks per job except one); about 9–16 min per task | `results/visium_dlpfc/p2_wide/` |
| A_full_Pois refit (p2_wide) | Same, for the 9 p2_wide tasks | 9 tasks × 6 geometries | 1939617 (single job, 9 workers) | ✅ done, 54/54 fits, 0 failed; 4 reached 1e-8, the rest near-converged (gap median 1.4e-6, max 2.0e-4) | 45 min | `results/visium_dlpfc/p2_wide/poisson/` |
| **k5_br5595** (new) | K = 5 on Br5595 (151669): P0 and P2 × 6 hunters × A_current/A_full_L2, anchor, all 6 baselines, 50-point λ grid | 3 | 1939616 (array 0–2) | ✅ done, 3/3; 96 records: 83 ok, 13 max_iter_reached (A_full_L2), 0 failed (§3.15). No Poisson refit | 35–75 min | `results/visium_dlpfc/k5_br5595/` |
| extended | K sweep and count thinning on all 12 sections | 360 | — | not planned before the meeting | — | — |

Common settings: K = 7; seeds 26090401–03; sections **151507** (Br5292), **151669**
(Br5595, layers L3–L6 + WM only) and **151673** (Br8100); 20% held-out molecules; symmetric
6-NN spatial graph.

Resources per job: 5 CPUs, 8 GB (16 GB for panel), BLAS single-threaded. Peak memory was
at most 2.8 GB per task.

---

## 2. What each record contains

Each task writes `<identity>.json` (status, metrics, runtime and diagnostics for every
method, plus provenance: commit, config hash, data hash, host) and `<identity>.npz` (Ŵ and
Â for every fitted method, spot and gene IDs, coordinates). Nothing is written until a task
ends. Failed methods are kept, with their exception and traceback.

A core task has 56 records:
- 48 document-GpLSI: 4 preprocessings × 6 hunters × 2 A estimators.
- 2 anchor-GpLSI: P0/SPA × 2 A estimators.
- 6 baselines.

The Poisson refit adds 25 records per task (one per distinct Ŵ).

---

## 3. Results (§3.1–3.11 core, 9 tasks / 504 records; §3.12–3.14 refits, panel, p2_wide)

### 3.1 Run health

| Status | Records | Notes |
|---|---:|---|
| ok | 389 | |
| max_iter_reached | 111 | All are `A_full_L2` at its 2,000-iteration cap. By hunter: spa_current 36 (every one), palm 30, palm_accelerated 30, anchor 9, svs_star 4, svs 2, pp_spa 0. A_full_L2 converged in 114 of 225 fits; A_current always "converges" because it is closed-form. |
| failed | 4 | Always `gplsi_anchor__P0_raw__spa_current__A_current`: `RecoveryError: historical current A recovery is singular`. 2 of 3 seeds on 151507 and 2 of 3 on 151673. |

**Why the anchor fit fails.** In these tasks the anchor Ŵ has rank 5 or 6 instead of 7
(duplicate topics), so ŴᵀŴ has condition number 1e17 to 1e19 and A_current's
(ŴᵀŴ)⁻¹ does not exist. Even in the tasks that did not fail, ŴᵀŴ is badly conditioned
(condition number 1e5 to 1e7), and one topic can carry almost no mass (column sum 0.05).
This is a property of the anchor-feature geometry at K = 7, not a code bug. A_full_L2
still fits these Ŵ, and the Poisson refit now does too (§5, fix 1).

### 3.2 Dimensions B × C: layer ARI by preprocessing × vertex hunter

Seed-averaged ARI against `layer_guess_reordered`. A_current and A_full_L2 share Ŵ, so ARI
is identical for both.

| Hunter | P0/P1: 151507 | 151669 | 151673 | **mean** | P2/P3: 151507 | 151669 | 151673 | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spa_current (original) | 0.125 | −0.042 | 0.194 | **0.092** | 0.212 | 0.031 | 0.143 | **0.128** |
| svs | 0.271 | 0.193 | 0.258 | **0.241** | 0.218 | 0.121 | 0.174 | **0.171** |
| svs_star | 0.258 | 0.167 | 0.276 | **0.234** | 0.218 | 0.222 | 0.174 | **0.205** |
| pp_spa | 0.189 | 0.154 | 0.254 | **0.199** | 0.185 | 0.130 | 0.170 | **0.162** |
| palm | 0.094 | −0.007 | 0.238 | **0.108** | 0.205 | 0.031 | 0.143 | **0.126** |
| palm_accelerated | 0.129 | −0.012 | 0.268 | **0.128** | 0.204 | 0.007 | 0.136 | **0.116** |

P0 and P1 are identical to 3 decimals, and so are P2 and P3, so the table merges them
(see §4, issue on thresholding).

Marginal means:

| Marginal | Value | layer ARI | layer NMI | Moran's I | neighbor agreement |
|---|---|---:|---:|---:|---:|
| Hunter (over preprocessings) | svs_star | **0.219** | 0.351 | 0.614 | 0.822 |
| | svs | 0.206 | 0.345 | 0.600 | 0.820 |
| | pp_spa | 0.180 | 0.318 | 0.696 | 0.806 |
| | palm_accelerated | 0.122 | 0.296 | 0.697 | 0.868 |
| | palm | 0.117 | 0.271 | 0.697 | 0.877 |
| | spa_current | 0.110 | 0.290 | 0.664 | 0.884 |
| Preprocessing (over hunters) | P0_raw = P1_tran | **0.167** | 0.334 | 0.827 | 0.905 |
| | P2_ke = P3_tran_then_ke | 0.151 | 0.290 | 0.496 | 0.787 |

Genes retained by the spectral step: P0/P2 keep 2,000 of 2,000; P1/P3 keep 1,995–2,000
(median 1,997).

### 3.3 Selected graph penalty

| Preprocessing | Selected λ (min / median / max over tasks) |
|---|---|
| P0, P1 | 0.01145 / 0.01374 / 0.01374 (top of grid is 0.01648) |
| P2, P3 | **0.01648 in every task (the top of the grid)** |

With P2 and P3 the cross-validation always picks the largest penalty on offer, and with
P0 and P1 it picks one or two steps below the top. The optimum is probably above the
grid, which is why lambda_wide and lambda_wide_tsgd were launched.

### 3.4 Dimension A: A estimators on the same Ŵ (A_current vs A_full_L2; A_full_Pois in §3.12)

The comparison is paired over 216 document-GpLSI geometries (task × preprocessing ×
hunter).

| Contrast (A_full_L2 − A_current) | Mean | Median | A_full_L2 lower in |
|---|---:|---:|---:|
| Floored held-out deviance / molecule | −0.0071 | −0.0006 | 68% |
| Zero-probability held-out molecules | −16.7 | +1 | 30% (tied in 18%) |

| | A_current | A_full_L2 | A_full_Pois |
|---|---:|---:|---:|
| Geometries with ≥1 zero-probability held-out molecule | 100% | 99% | 40% |
| Median zero-probability molecules | 38 | 33 | 0 |
| Share of fits with finite (unfloored) deviance | 0% | 1% | 56% |
| Floored deviance / molecule (mean) | 2.0225 | 2.0155 | **2.0124** |
| Topic entropy | 0.758 | 0.762 | 0.779 |
| Top-gene exclusivity | 0.355 | 0.353 | 0.352 |
| Converged | 100% (closed form) | 53% | 2% at 1e-8; rest near-converged (median gap 5.7e-6) |

The unfloored deviance is infinite for nearly every GpLSI fit, because at least one
held-out molecule gets probability 0. The floored deviance (1e-12 floor) is therefore
the comparable number.

The floor adds only a little. Each zero-probability molecule costs about 40 deviance
units, but they are rare: across all core fits the floor adds a median of **0.0009** per
molecule (maximum 0.019). So differences in floored deviance mostly reflect how well
the fitted profiles predict all held-out counts, not the zero-probability molecules.
(Corrected 2026-09-27: an earlier version said the floored deviance was dominated by
them.)

### 3.5 GpLSI vs baselines (layer ARI, seed-averaged)

| Method | 151507 | 151669 | 151673 | **mean** | Floored deviance / molecule | Zero-prob molecules |
|---|---:|---:|---:|---:|---:|---:|
| GpLSI best (P0, svs) | 0.271 | 0.193 | 0.258 | **0.241** | see §3.4 | see §3.4 |
| spatial_lda | 0.279 | 0.195 | 0.241 | **0.239** | 2.0081 | 0 |
| lda | 0.321 | 0.158 | 0.206 | **0.228** | 2.0040 | 0 |
| topicscore_graph_denoised | 0.183 | 0.114 | 0.151 | 0.149 | 2.0249 | 74.9 |
| graph_kl_nmf | 0.100 | 0.167 | 0.156 | 0.141 | 2.0052 | 0 |
| GpLSI original (P0, SPA) | 0.125 | −0.042 | 0.194 | 0.092 | see §3.4 | see §3.4 |
| topicscore_raw | 0.128 | 0.062 | 0.082 | 0.091 | 2.0274 | 102.8 |
| GpLSI anchor (P0, SPA) | 0.071 | 0.015 | 0.129 | 0.072 | see §3.4 | see §3.4 |
| kl_nmf | 0.030 | 0.005 | 0.101 | 0.046 | 2.0022 | 0 |

**Seed range (min to max ARI over 3 seeds):**

| Method | 151507 | 151669 | 151673 |
|---|---|---|---|
| GpLSI P0/svs | 0.260–0.283 | 0.067–0.271 | 0.247–0.273 |
| GpLSI P0/spa_current | 0.039–0.209 | −0.060 to −0.009 | 0.165–0.236 |
| spatial_lda | 0.270–0.288 | 0.148–0.288 | 0.228–0.259 |
| lda | 0.287–0.353 | 0.114–0.214 | 0.158–0.256 |

**TopicSCORE W refit:** it converged in only 2 of 9 tasks (graph-denoised) and 1 of 9
(raw), yet those records are labelled `ok`. The summary now flags this.

### 3.6 Cross-seed stability of topic profiles (matched Jensen–Shannon divergence; lower is more stable)

| Group | Method | JSD |
|---|---|---:|
| GpLSI hunters, P0 | pp_spa | 0.022 (most stable) |
| | svs_star | 0.043 |
| | svs | 0.046 |
| | palm_accelerated | 0.063 |
| | palm | 0.071 |
| | spa_current | 0.090 |
| Baselines | spatial_lda | 0.007 |
| | lda | 0.014 |
| | kl_nmf | 0.062 |
| | graph_kl_nmf | 0.086 |
| | topicscore_graph_denoised | 0.201 |
| | topicscore_raw | 0.226 |

A_current and A_full_L2 are equally stable, with differences of at most 0.004.

### 3.7 Runtime (median per record)

| Family | Median per record | Notes |
|---|---|---|
| Baselines | 99 s | spatial_lda 315–644 s per task; topicscore_raw about 130–280 s; graph-denoised TopicSCORE about 160 s |
| Document GpLSI | 30 s | includes its share of the spectral step |
| Anchor GpLSI | 34 s | |

The full-grid spectral step is only 11–31 s per preprocessing, because the graph SVD
stops after 2 outer iterations.

### 3.8 Diagnostic plots

Generated by `python scripts/visium_dlpfc/plot_core_diagnostics.py`. It reads only the
9 core JSONs and takes a few seconds. The figures and their source table (`records.csv`)
are in `results/visium_dlpfc/diagnostics/core/`, which is gitignored, so regenerate them
with the script.

Colors follow a fixed categorical order; sections and hunters also get distinct marker
shapes.

**Fig. 1: `01_ari_heatmap_preprocessing_x_hunter.png`**, layer ARI per section,
preprocessing × hunter.
- The P0 = P1 and P2 = P3 row pairs are identical in every section, so thresholding has
  no effect at p = 2,000.
- The best hunter changes with the section:
  - svs is best on 151507 (0.27);
  - svs, svs_star and pp_spa lead on 151669, and the SPA and PALM hunters fall to ≈ 0
    there;
  - on 151673 every hunter does reasonably (0.19–0.28), with palm_accelerated and
    svs_star highest.
- KE weighting (P2) evens the hunters out. On 151507 it pulls every hunter to 0.18–0.22,
  and it improves SPA/PALM on 151507 and 151669, but it costs svs and pp_spa on every
  section.
- svs_star is the only hunter that is at or near the top in every section under both
  weightings. That makes it the most robust choice.

**Fig. 2: `02_ari_by_method_seed_points.png`**, every seed as a point.
- The top group (GpLSI P0 svs 0.241, spatial_lda 0.239, svs_star 0.234, lda 0.228) is
  separated by less than the seed scatter, so these four are statistically tied on 3
  donors.
- GpLSI svs's mean is pulled down by a single 151669 seed at 0.067. Its other 8 runs are
  0.24–0.28, the tightest high-ARI cluster in the plot.
- LDA's lead on 151507 (0.29–0.35) is offset by weak 151669 runs.
- The original GpLSI (P0 SPA), PALM, anchor GpLSI and graph KL-NMF all have
  151669 runs at or below 0. Only svs/svs_star/pp_spa among GpLSI stay clearly positive
  on every run.

**Fig. 3: `03_A_estimator_paired.png`**, A_full_L2 vs A_current on the same Ŵ.
- **Deviance (left):** the benefit of A_full_L2 depends entirely on the hunter.
  - For spa_current, palm and palm_accelerated, A_full_L2 lowers floored deviance by
    about 1e-2 per molecule in almost every geometry.
  - For svs and svs_star the two estimators are practically identical (differences
    below 1e-3, slightly favouring A_current).
  - pp_spa is split around 0.
  - Interpretation: when the hunter already finds good vertices, A_current's closed-form
    projection is nearly optimal; when the vertices are poor, the full least-squares
    refit repairs part of the damage.
- **Zero-probability molecules (right):**
  - For svs and svs_star the points lie on the diagonal, so the estimator doesn't matter.
    pp_spa sits slightly above it.
  - A_full_L2 cuts them by 10–100× for palm_accelerated (and many palm fits).
  - It **increases** them for spa_current (points above the diagonal).
  - Neither estimator removes zero-probability molecules for any hunter. That is still
    left to A_full_Pois.

**Fig. 4: `04_convergence_and_status.png`**
- A_full_L2 convergence splits cleanly by hunter:
  - svs, svs_star and pp_spa converge (projected-gradient norm about 1e-5; pp_spa 36/36);
  - spa_current (36/36) and anchor SPA (9/9) never converge;
  - palm and palm_accelerated converge in only 6 of 36.
- The unconverged fits stop at projected-gradient norms of 1e-4 to 1e-2, with anchor up
  to 1e-1. So "max_iter_reached" is far from optimal for anchor and moderately so for
  SPA/PALM.
- The same hunters that give poor ARI give ill-conditioned least-squares problems. This
  points to near-degenerate vertex sets (nearly collinear topics), consistent with the
  rank-deficient anchor Ŵ in §3.1.
- Raising the iteration cap would change A_full_L2 only for the hunters we would not
  recommend anyway.

**Fig. 5: `05_selected_lambda.png`**
- With P2/P3 every task selects the grid top (0.0165).
- With P0/P1, 151507 and 151669 select 0.0137 (one step below the top) and 151673
  selects 0.0114 (two steps below).
- Selection is perfectly consistent across the three seeds (points overlap), so the
  choice is driven by the section, not the split.
- The KE weighting evidently wants more smoothing than the grid allows. lambda_wide
  will show whether its optimum and ARI move once the ceiling is lifted.

**Fig. 6: `06_ari_vs_deviance_tradeoff.png`**, layer ARI vs floored deviance (means over
9 tasks).
- LDA and Spatial-LDA sit in the "good on both" corner: highest ARI and near-best
  deviance.
- GpLSI svs/svs_star (P0) match them on ARI but are about 0.013–0.016 per molecule worse
  on deviance. Zero-probability molecules explain almost none of that gap: removing the
  floor's contribution leaves GpLSI svs 0.018 behind LDA. It is a genuine prediction gap
  in the Euclidean (L2-type) topic–gene estimates. A_full_Pois, which maximizes the same
  likelihood the deviance measures, is the test of whether a likelihood-based Â closes it.
- KL-NMF has the best deviance and the worst ARI, so prediction and layer recovery rank
  methods differently.
- The horizontal segments (A_current ↔ A_full_L2) are long for PALM and SPA and
  negligible for svs/svs_star/pp_spa, the same message as Fig. 3.
- P2 (hollow) shifts every GpLSI hunter left (better deviance by about 0.01). It lowers
  ARI for svs, svs_star and pp_spa, and raises it for SPA and PALM.

**Fig. 7: `07_runtime.png`**
- Spatial-LDA is the costliest method (median 431 s, up to about 640 s), then the two
  TopicSCOREs (about 190–230 s). The TopicSCOREs cost that much because of their
  5,000-iteration W refit.
- GpLSI runs in about 30 s per geometry, including its share of the spectral step,
  comparable to LDA and graph KL-NMF.
- A_full_L2 adds almost nothing over A_current in wall time (median 30 vs 29 s), even
  when it runs to the cap.
- The failed anchor A_current records are excluded.

**Fig. 8: `08_ari_seed_spread.png`**, ARI range across the 3 seeds.
- On P0, seed sensitivity is large for SPA/PALM on 151507 (0.13–0.17), and for svs on
  151669 (0.20, the single outlying seed from Fig. 2).
- P2 KE weighting almost removes seed sensitivity on 151507 and 151673 (≤ 0.04), but not
  on 151669. There every hunter except svs still varies by 0.07–0.15.
- 151669 (Br5595, 5 labelled layers at K = 7) is the unstable section throughout. Its
  results should be read with that in mind, and it is a candidate for a K = 5
  sensitivity run in the extended grid.

### 3.9 Penalty-grid sensitivity: graph-denoised TopicSCORE (lambda_wide_tsgd)

The 50-point grid runs to λ ≈ 0.75. Seed 26090401 is compared against the same core task.

| Section | Selected λ (P0), core → wide | GpLSI P0/SPA ARI | graph-denoised TopicSCORE ARI | its floored deviance | its W refit converged |
|---|---|---|---|---|---|
| 151507 | 0.01374 → 0.01374 | 0.039 → 0.039 | 0.194 → 0.194 | 2.3447 → 2.3447 | no |
| 151669 | 0.01374 → 0.01374 | −0.058 → −0.058 | 0.172 → 0.172 | 1.8941 → 1.8941 | no |
| 151673 | 0.01145 → 0.01145 | 0.179 → 0.179 | 0.149 → 0.149 | 1.8265 → 1.8265 | yes |

- **For P0 the optimum is genuinely inside the core grid.** Given 21 larger penalties,
  cross-validation picks exactly the same λ in every section, so every core result that
  depends on the P0 spectral step is unaffected by the grid ceiling. That includes
  graph-denoised TopicSCORE.
- **The runs are reproducible.** Every metric matches core bit for bit, which confirms
  that the task pipeline (split, panel, graph, CV folds) is deterministic.
- **The wide grid is slower.** These tasks took 16–30 min for only two methods, which is
  longer than a full core task, because the wide grid has 50 × 5 CV fits. That matters
  for sizing any wide-grid extended runs.
- **P2 is different** (see §3.10).

### 3.10 Penalty-grid sensitivity: GpLSI (lambda_wide)

This design reruns P0 and P2 × spa_current × both A estimators on a 50-point grid (up to
λ ≈ 0.75), for seed 26090401. Each value is shown as core → wide. The grid index is in
brackets; the core grid top is #28, 0.0165.

| Section | Preprocessing | Selected λ | Layer ARI | Moran's I | Floored deviance (A_current) | Zero-prob molecules A_current / A_full_L2 |
|---|---|---|---|---|---|---|
| 151507 | P0 | 0.0137 (#27) → same | 0.039 → 0.039 | 0.802 → 0.802 | 2.3555 → 2.3555 | unchanged |
| 151507 | **P2** | 0.0165 (#28) → **0.122 (#39)** | 0.207 → 0.208 | 0.539 → 0.621 | 2.3201 → 2.3230 | 29 → 23 / 110 → 35 |
| 151669 | P0 | 0.0137 (#27) → same | −0.058 → −0.058 | 0.807 → 0.807 | 1.8935 → 1.8935 | unchanged |
| 151669 | **P2** | 0.0165 (#28) → **0.176 (#41)** | −0.037 → −0.042 | 0.430 → 0.628 | 1.8873 → 1.8942 | 102 → 109 / 106 → 108 |
| 151673 | P0 | 0.0114 (#26) → same | 0.179 → 0.179 | 0.833 → 0.833 | 1.8340 → 1.8340 | unchanged |
| 151673 | **P2** | 0.0165 (#28) → **0.212 (#42)** | 0.154 → **0.191** | 0.516 → 0.612 | 1.8199 → 1.8316 | 18 → 19 / 122 → 19 |

All 12 records completed. The A_full_L2 fits hit the iteration cap in both runs, as
they did in core.

- **P0: the core grid is adequate.** The selected λ and every metric are identical.
- **P2: the core grid was too narrow.** Its cross-validation optimum is 11–14 grid steps
  above the old ceiling (λ ≈ 0.12–0.21, about 7–13× larger), and the graph SVD runs
  longer (up to 6 iterations instead of 2–3). This is consistent with KE weighting
  rescaling the data so that a larger penalty is needed for the same smoothing.
- **What the extra smoothing does:**
  - Spatial coherence rises on every section (Moran's I +0.08 to +0.20), recovering a third
    to a half of the gap to P0.
  - ARI gains on 151673 (+0.037), is flat on 151507 and slightly lower on 151669.
  - Floored deviance gets slightly worse (+0.003 to +0.012), the usual smoothness–fit
    trade-off.
  - A_full_L2 zero-probability molecules fall sharply on 151507 and 151673.
- **Consequence for the core conclusions:**
  - The P2/P3 rows in §3.2 were run with an under-smoothed penalty, so they
    **understate** P2/P3 on spatial metrics and possibly on ARI.
  - The "KE weighting lowers Moran's I" finding in §4 is partly an artefact of the grid.
  - Only spa_current was checked. Whether svs/svs_star on P2 improve with the larger λ is
    unknown.
- **Recommended fix:** for extended runs (and ideally a core rerun of P2/P3), set
  `grid_len` 50 for P2/P3. It costs about 2× in spectral time, since these tasks took
  33–38 min against about 18 min for a core task.

### 3.11 Spatial maps and topic anatomy (what the topics actually are)

ARI compresses a fit to one number. These figures show what each method's topics are:
where they sit in the tissue, and which layer's genes they carry.

**Scripts:**
- `scripts/visium_dlpfc/plot_spatial_maps.py`, output in
  `results/visium_dlpfc/diagnostics/core/spatial/`:
  - `overview_first_seed.png`: 3 sections × 8 key methods;
  - `map_<section>_s<seed>.png`: all 9 tasks, each with every GpLSI geometry (P0 and P2
    × 6 hunters, plus anchor) and all 6 baselines.
- `scripts/visium_dlpfc/plot_topics.py`, output in
  `results/visium_dlpfc/diagnostics/core/topics/`:
  - `topics_<section>_s<seed>__<method>.png`: 10 key methods × 3 sections (seed 26090401).

Both scripts need the processed h5ad, so run them on a compute node.

**How to read the maps:**
- Each spot is coloured by its dominant topic.
- To make maps comparable with the annotation, topics are matched one-to-one to manual
  layers (Hungarian matching on the overlap of argmax topic and layer) and drawn in the
  matched layer's colour; unmatched topics are grey.
- The matching forces a one-to-one assignment. A topic labelled "≈ L3" may in fact span
  L2–L5, which is exactly what the topic figures reveal.

**How to read the topic figures:**

| Panel | Shows |
|---|---|
| Top row | Each topic's continuous weight Ŵ·ₖ over the tissue, with its share of total topic mass |
| Bottom left: "where it sits" | Mean topic weight within each manual layer |
| Bottom middle: "which layer's genes it carries" | Correlation, over the 1,000 most expressed panel genes, between the topic's gene enrichment log2(Âₖ / Ā) and each layer's enrichment log2(L_layer / L_all). L are pooled count frequencies of the section's spots in that layer. One high cell means a layer-specific topic; several adjacent high cells mean a mixture of layers. |
| Right | The topic's top 8 enriched genes, each coloured by the layer where it is most enriched (grey if not layer-specific) |

For GpLSI, Â is from A_current.

**Findings:**

1. **The best GpLSI fits recover real laminar programs, with textbook markers.** On 151673,
   GpLSI P0 svs_star (ARI 0.28) has:

   | Topic | Top markers | Layer signature correlation |
   |---|---|---|
   | L1 | RELN, CXCL14 | 0.79 with L1 |
   | L2/3 | CALB1, CARTPT, CUX2, HPCAL1, HOPX | 0.96 with L3 |
   | L4/5 | RORB, PVALB, NEFH, NEFM | 0.85 with L4 |
   | Oligodendrocyte L6/WM mixture | MBP, MOBP, AQP1 | 0.87 with WM |
   | WM | MOG, CLDN11, GJB1 | 0.98 with WM |

   The two remaining topics are not layers but tissue compartments:
   - a vascular/meningeal topic (ACTA2, MYL9, TAGLN, CALD1, IGHM);
   - a blood/immune topic (HBB, C1QC, IGHM).

   So "ARI 0.28" understates the biology. The method finds L1, superficial (L2/3), middle
   (L4/5), deep and WM programs; its errors are merging adjacent layers and spending
   topics on non-laminar compartments.

   On 151507, GpLSI P0 svs has a clean L1 topic (RELN, AQP4, GFAP, correlation 0.97) and
   an L4 program (PVALB, NEFH, correlation 0.78 with L4), but only 5 of 7 topics carry
   mass (finding 4).
2. **The original GpLSI (P0 SPA) wastes most of its topics.** On 151673, one topic holds
   42% of the mass and spans L2–L6. Another (19% of mass) is a
   mitochondrial/interferon/quality program (MT-ATP8, COX6C, IFI27, HLA-C) spread
   diffusely over the tissue, and a third holds 1%. Only L1 and WM come out clean, which
   is why SPA's ARI is low.
3. **Spatial-LDA's ARI edge comes from WM and L1, not from resolving the middle layers.**
   - Its WM topic is the sharpest of any method (mean weight 0.84 in WM; signature 0.99).
   - Three of its cortical topics carry nearly the same L3–L5 signature (0.6–0.84) and
     are spatially speckled. The cortex is split into overlapping, non-laminar programs.
   - It also has a mitochondrial topic (MT-ATP8, MT-ND5) and an immunoglobulin topic.

   Its maps are visibly noisier than GpLSI svs/svs_star, which have spatially coherent
   bands. Neither shows up in ARI.
4. **Topic collapse: many fits use far fewer than K = 7 topics.** This is a new diagnostic,
   counted over all 9 core tasks from Ŵ (A-independent).

   "Near-empty topics" are topics with less than 1% of total mass. "Topics used" counts
   topics that are the dominant topic of at least 1% of spots.

   | Method | Near-empty topics (mean / max) | Topics used (mean) |
   |---|---|---|
   | LDA | 0 / 0 | 7.0 |
   | Spatial-LDA | 0 / 0 | 6.9 |
   | GpLSI P0 pp_spa | 0 / 0 | 6.6 |
   | GpLSI P0 svs, svs_star | 0.9 / 2 | 6.1 |
   | GpLSI P0 palm_accelerated | 0 / 0 | 5.8 |
   | TopicSCORE graph-denoised | 0.9 / 2 | 5.0 |
   | GpLSI P0 palm | 0 / 0 | 4.2 |
   | graph KL-NMF | 0.8 / 3 | 4.1 |
   | GpLSI P0 spa_current (original) | 0.8 / 2 | 4.0 |
   | GpLSI anchor P0 SPA (the 5 tasks where A_current succeeded) | 1.8 / 3 | 4.0 |
   | GpLSI P2 svs, svs_star | **2.7 / 3** | 4.0 |
   | GpLSI P2 spa_current | 2.3 / 3 | 3.6 |
   | TopicSCORE raw | 1.6 / 3 | 3.4 |
   | KL-NMF | 0 / 0 | **2.3** (one topic dominates almost every spot) |

   P1 = P0 and P3 = P2 throughout.
5. **Why KE weighting (P2/P3) loses ARI: its vertices go to rare cell types.** In the P2
   svs_star fit on 151673, the near-empty topics are sharp programs of scattered cells:
   - interneurons (NPY, SST, CORT, CRHBP);
   - red blood cells (HBA1/2, HBB);
   - plasma cells (IGHG1/3/4, IGKC, JCHAIN).

   KE weighting up-weights high-dispersion genes, and these rare-cell genes are the most
   over-dispersed, so they become the extreme points the vertex hunter picks. The layers
   are then left with about 4 topics: a superficial L1/L2 topic, one L2–L6 topic holding
   48% of the mass, a vascular topic and WM.

   This is a mechanism, not a tuning issue. The larger λ found by lambda_wide (§3.10) may
   smooth these single-spot programs away and partly relieve it. That should be checked
   with the P2 wide-grid rerun.
6. **Hungarian labels are a colouring device, not an identification.** For example, the
   151507 svs topic drawn as "L5" has mean weight 0.86 in L2 and carries L2/3 markers
   (C1QL2, CALB1, HPCAL1, CNKSR2). Always read identity from the signature heatmap and
   the genes.

### 3.12 Dimension A completed: A_full_Pois (Poisson maximum likelihood from the same Ŵ)

The Poisson refit ran on every core geometry (216 document + 9 anchor), every panel
geometry and lambda_wide. **No fit failed.** As decided, fits start from pooled
frequencies and stop at 1,500 iterations: 4/225 core fits (and 12/216 panel fits) reach
the 1e-8 tolerance; the rest are **near-converged**, with normalized optimality gap
median 5.7e-6 and max 4.2e-4. The gap depends on the hunter: svs/svs_star about 6e-8,
pp_spa 2.6e-6, spa_current 2.1e-5, PALM 6e-5 to 1.3e-4. The poorly conditioned vertex sets
converge slowest, as with A_full_L2.

**Paired contrasts on identical Ŵ (216 core geometries):**

| Contrast | Floored deviance / molecule: mean (median) | Lower in | Zero-prob molecules: mean (median) | Lower in |
|---|---:|---:|---:|---:|
| A_full_Pois − A_current | −0.0101 (−0.0037) | **100%** | −67 (−33) | 98% |
| A_full_Pois − A_full_L2 | −0.0031 (−0.0027) | **100%** | −50 (−27) | 97% |
| A_full_L2 − A_current | −0.0071 (−0.0006) | 68% | −17 (+1) | 30% |

**Floored deviance by hunter (core, mean of 9 tasks) and distance to LDA (2.0040):**

| Preprocessing | Hunter | A_current | A_full_L2 | A_full_Pois | Pois gap to LDA |
|---|---|---:|---:|---:|---:|
| P0 | spa_current | 2.0251 | 2.0198 | 2.0158 | +0.012 |
| P0 | svs | 2.0237 | 2.0239 | 2.0202 | +0.016 |
| P0 | svs_star | 2.0211 | 2.0211 | 2.0187 | +0.015 |
| P0 | pp_spa | 2.0202 | 2.0195 | 2.0169 | +0.013 |
| P0 | palm | 2.0469 | 2.0207 | 2.0154 | +0.011 |
| P0 | palm_accelerated | 2.0497 | 2.0167 | 2.0133 | +0.009 |
| P2 | spa_current | 2.0115 | 2.0097 | 2.0067 | +0.003 |
| P2 | svs_star | 2.0128 | 2.0128 | 2.0119 | +0.008 |
| P2 | palm | 2.0161 | 2.0087 | **2.0050** | +0.001 |

Baselines for reference: KL-NMF 2.0022, LDA 2.0040, graph KL-NMF 2.0052, Spatial-LDA
2.0081, TopicSCORE 2.025–2.027.

**What this shows:**

1. **A_full_Pois is the best A estimator for prediction, uniformly.** It beats A_current and
   A_full_L2 in every one of the 216 paired geometries. The gain is largest where the
   vertices are poor (PALM: −0.03 vs A_current) and small for svs/svs_star (−0.002 to
   −0.004), which repeats the §3.8 Fig. 3 message: good vertices leave little for the A
   step to repair.
2. **It closes only 14–20% of the gap to LDA for the best-ARI hunters** (svs, svs_star,
   pp_spa). For P0 svs/svs_star
   the gap to LDA falls from about 0.017–0.020 to 0.015–0.016 per molecule. So the remaining
   gap is not an A-estimation problem: it comes from **Ŵ**. GpLSI's Ŵ is smoothed and
   sparse (svs/svs_star set about 60% of spot–topic weights to exactly 0; PALM about 4%),
   while LDA's per-spot mixtures are fitted to each spot's own counts. With KE weighting
   (P2), whose Ŵ is less smooth (§3.10), Pois-refit GpLSI reaches LDA-level deviance
   (P2 palm 2.0050, P2 SPA 2.0067). *Superseded: that Ŵ was under-smoothed by the core
   grid. At P2's selected λ the gap is back to 0.011–0.016 (§3.14, p2_wide refit).*
3. **Zero-probability molecules mostly disappear, but not entirely.** 56% of Pois fits now
   have a finite unfloored deviance (0–1% for the others). The remaining zeros are
   concentrated in the hunters with sparse Ŵ: P0 svs/svs_star (median 2, max 21 molecules),
   P2 svs/svs_star (median 12–13, max 134); PALM has essentially none. The mechanism: the
   unpenalized MLE correctly assigns probability 0 to a gene never observed, in training,
   in the spots a topic covers; when a test spot's Ŵ has zeros on every other topic, a
   held-out molecule of that gene is "impossible". LDA avoids this with its Dirichlet
   prior on topic–gene profiles. **A tiny pseudocount in the Poisson refit (or flooring
   Ŵ) would remove the remaining zeros**; this is a one-line option worth adding.
4. **Profiles become slightly less sparse** (topic entropy 0.758 → 0.779), with unchanged
   top-gene exclusivity (0.35), so the marker genes are the same; the MLE mainly spreads a
   little mass to low-expression genes.
5. **A_full_Pois is less reproducible across seeds for svs/svs_star** (matched JSD 0.084 vs
   0.046 for A_current on P0; pp_spa unchanged at 0.022). The likely cause is the
   near-empty topics of those hunters (§3.11 finding 4): a topic with under 1% of the mass
   has an MLE profile that is barely identified, so it drifts between seeds, whereas the
   projection-based estimators pin it near the vertex.
6. **Cost:** 5.3 h of wall time on 9 workers for 225 geometries, about 13 CPU-min per
   geometry, against seconds for A_current/A_full_L2. It is by far the most expensive
   part of GpLSI in this pipeline.

**Bottom line for dimension A:** for layer recovery the A estimator is irrelevant (Ŵ is
shared). For prediction, A_full_Pois > A_full_L2 > A_current, consistently but by small
amounts (≤ 0.01 per molecule except PALM). A_full_Pois is the only estimator that makes
most held-out likelihoods finite, which matters if GpLSI is reported next to
likelihood-based methods.

### 3.13 Dimension C: panel size (500 / 1,000 / 2,000 / 5,000 genes) × thresholding

36 tasks (3 sections × 3 seeds × 4 panel sizes), each with P0 and P1 × three hunters
(spa_current, svs_star, palm_accelerated) × three A estimators, plus LDA, KL-NMF and raw
TopicSCORE. All 540 records plus 216 Poisson refits completed; none failed.

**Maps first.** `results/visium_dlpfc/diagnostics/panel/spatial/panel_<section>_s<seed>.png`
(made by the new `plot_design_maps.py --design panel`) put the four panel sizes in rows
and the methods in columns; the per-topic figures are in `.../panel/topics/`
(`topics_<section>_s26090401_p<size>__<method>.png`).

- **GpLSI svs_star is stable across panel sizes.** On 151673 and 151507 the same bands
  appear at 500 and 5,000 genes (L1/meninges, L2/3, L4/5, deep/L6, WM); only boundaries
  shift. The topic figures confirm the same programs: CARTPT/HPCAL1/HOPX (L2/3, signature
  0.95 with L3 at both sizes), NEFH/PVALB/NEFM (L4/5, 0.80), MOG/CLDN11 (WM, 0.96–0.99).
  At 5,000 genes lower-expression markers enter (KRT17 and SCGB1D2 in the L6 topic,
  CBLN4 and PCDH8 in L2/3), and the 500-gene fit spends one topic on a blood program
  (HBB, HBA1/2) that at 5,000 genes is replaced by a genuine L6/WM-border topic. Coarse
  structure needs few genes; extra genes sharpen the deep layers.
- **SPA and PALM change their whole partition with panel size.** On 151507, spa_current
  (seed 1) goes 0.22 → 0.04 → 0.04 → 0.17 across sizes, and palm_accelerated 0.22 → 0.21 →
  0.13 → 0.06: the map switches between "two bands + WM" and "one topic covering all of
  cortex". This is the same vertex fragility seen across seeds in core, now seen across
  panels.
- **LDA's maps stay speckled at every size**, and it gains more from genes than GpLSI does
  on 151507 (0.27 → 0.35 for seed 1).

**Layer ARI (seed-averaged, mean of 3 sections):**

| Method | 500 | 1,000 | 2,000 | 5,000 |
|---|---:|---:|---:|---:|
| GpLSI P0 svs_star | 0.234 | 0.237 | 0.234 | **0.248** |
| GpLSI P0 palm_accelerated | 0.131 | 0.128 | 0.128 | 0.122 |
| GpLSI P0 spa_current | 0.138 | 0.106 | 0.092 | 0.115 |
| LDA | 0.206 | 0.229 | 0.228 | 0.247 |
| TopicSCORE raw | 0.092 | 0.091 | 0.091 | 0.126 |
| KL-NMF | 0.050 | 0.040 | 0.046 | 0.054 |

svs_star's seed range shrinks as genes are added on 151669 (0.16 at 500 → 0.07 at 5,000),
so more genes buy stability even when the mean barely moves.

**Thresholding (P1 vs P0) does nothing at any panel size.** Tran α = 0.005 removes 0, 0–1,
0–5 and 4–85 genes (median 50 at p = 5,000), and the ARI difference between P1 and P0 is
exactly 0 in 318 of 324 paired records (max 0.0003). Floored deviances differ by at most
1.2e-5. The
thresholding axis is therefore **empty on this dataset**: panel selection by dispersion
already removes the genes the threshold targets. A stronger α or a different
threshold would be needed to make this dimension informative.

**Held-out prediction on the common 500-gene reference panel (floored deviance, lower is
better; GpLSI averaged over the three hunters):**

| | 500 | 1,000 | 2,000 | 5,000 |
|---|---:|---:|---:|---:|
| GpLSI A_current | 1.1695 | 1.1676 | 1.1679 | 1.1672 |
| GpLSI A_full_L2 | 1.1609 | 1.1610 | 1.1610 | 1.1612 |
| GpLSI A_full_Pois | 1.1584 | 1.1586 | 1.1585 | 1.1580 |
| KL-NMF | 1.1439 | 1.1436 | 1.1437 | 1.1440 |
| LDA | 1.1458 | 1.1468 | 1.1460 | 1.1469 |
| TopicSCORE raw | 1.1633 | 1.1654 | 1.1668 | 1.1668 |

- **Adding genes does not improve prediction of the 500 core genes for any method**
  (all rows flat within 0.002). The extra genes carry little information about the
  abundant ones.
- The A-estimator ordering (Pois < L2 < current) holds at every size, and Pois removes
  nearly all reference-panel zero-probability molecules (mean 0.1–0.2 vs 11–36).
- The GpLSI-to-LDA gap (about 0.012) is the same at every size, consistent with it coming
  from Ŵ (§3.12).

**Selected λ moves with panel size:** median 0.0165 (the core grid top) at 500 and 1,000
genes, 0.0137 at 2,000 and 0.0095–0.0114 at 5,000. Smaller panels have noisier
per-gene frequencies and want more smoothing, so **the core grid is also too narrow for
p ≤ 1,000**. Any extended run with small panels should use the wide grid.

**Cost:** a panel task takes 2.6 / 4.2 / 6.9 / 14.2 min at p = 500 / 1,000 / 2,000 / 5,000
(median per GpLSI record 42 → 76 s).

**Topic usage** (from Ŵ; mean topics used / near-empty): svs_star 6.1 → 5.8 and about 1
near-empty topic at every size; spa_current about 4 topics used at every size; LDA 6.7–7.
Panel size does not fix topic collapse.

### 3.14 P2 (KE weighting) rerun on the wide penalty grid (p2_wide, new 2026-09-28)

**Why:** lambda_wide (§3.10) showed that P2's cross-validated λ lies 7–13× above the core
grid's top, but it only reran spa_current on one seed. So the question was open whether
KE weighting's losses in core (ARI, smoothness, topics wasted on rare cells) were a grid
artefact. p2_wide reruns all six hunters × A_current/A_full_L2 on P2 with the 50-point
grid, for the same 9 section × seed tasks as core. P3 is omitted because P3 = P2 at
p = 2,000. (No Poisson refit was run for p2_wide.)

**Selected λ:** 0.10–0.12 on 151507, 0.15–0.18 on 151669 (the seeds disagree there), and
0.21 on 151673, against 0.0165 (the grid top) in core. The spa_current / seed-1 records
reproduce lambda_wide exactly.

**Maps first** (`results/visium_dlpfc/diagnostics/p2_wide/spatial/p2_wide_<section>_s<seed>.png`:
top row core grid, bottom row wide grid, one column per hunter; per-topic figures in
`.../p2_wide/topics/`):

- **The larger λ turns P2's speckled maps into clean regions.** On the core grid P2
  svs/svs_star maps are salt-and-pepper mixes of two or three topics over the cortex; on
  the wide grid every hunter gives smooth, contiguous domains.
- **But the domains are coarse: KE-weighted GpLSI still resolves only about 4 tissue
  programs.** The P2 svs_star topic figure for 151673 (seed 1, ARI 0.206) shows:

  | Topic | Mass | What it is | Top markers |
  |---|---:|---|---|
  | "≈ L1" | 7% | meninges / vasculature | MYH11, ACTA2, TAGLN, COL1A1 |
  | "≈ L3" | 20% | superficial L1–L3 | RELN, CALB2, CARTPT, CUX2 |
  | "≈ L5" | **58%** | all of L3–L6 in one topic (mean weight 0.61–0.94 in L3–L6) | PCP4, SMYD2, TBR1, RORB |
  | "≈ WM" | 14% | white matter | MOG, MOBP, KLK6 (signature 0.99) |
  | "≈ L4" | 0% | interneurons (scattered spots) | NPY, SST, CRHBP, CORT |
  | "≈ L6" | 0% | plasma cells | IGHG1/3/4, IGKC, JCHAIN |
  | unmatched | 0% | red blood cells | HBA1/2, HBB |

  These are **the same three rare-cell vertices as on the core grid** (§3.11 finding 5).
  Smoothing cannot remove them because they are chosen by the vertex hunter on the
  KE-weighted spectral embedding: the over-dispersed rare-cell genes create extreme points
  whatever λ is. With 3 of 7 vertices spent on them, the four remaining topics cannot
  separate L3, L4, L5 and L6.

**Numbers (seed-averaged, mean over 3 sections):**

| P2 hunter | ARI core → wide | NMI core → wide | Balanced acc. core → wide | Moran's I core → wide | Near-empty topics core → wide |
|---|---|---|---|---|---|
| spa_current | 0.128 → 0.134 | 0.293 → 0.352 | 0.561 → 0.709 | 0.506 → 0.625 | 2.3 → 2.4 |
| svs | 0.171 → 0.174 | 0.302 → 0.389 | 0.493 → 0.595 | 0.449 → 0.612 | 2.7 → 2.7 |
| **svs_star** | 0.205 → **0.223** | 0.310 → **0.407** | 0.503 → 0.630 | 0.461 → 0.619 | 2.7 → 2.7 |
| pp_spa | 0.162 → 0.191 | 0.268 → 0.372 | 0.550 → 0.701 | 0.521 → 0.729 | 0.7 → 1.4 |
| palm | 0.126 → 0.120 | 0.284 → 0.333 | 0.578 → 0.721 | 0.521 → 0.650 | 1.2 → 1.9 |
| palm_accelerated | 0.116 → 0.144 | 0.282 → 0.350 | 0.579 → 0.734 | 0.515 → 0.650 | 0.7 → 1.7 |
| *P0 svs_star (reference)* | *0.234* | *0.392* | *0.691* | *0.766* | *0.9* |

Per section, the gains are on 151673 (every hunter +0.03 to +0.08) and none on 151507
(+0.00 to +0.03). On 151669 they are mixed, with very large seed spread: svs_star's
seeds range over 0.21 and pp_spa's over 0.41.

Held-out floored deviance gets slightly worse with the extra smoothing (A_current
2.014 → 2.024, A_full_L2 2.011 → 2.019), the usual smoothness–fit trade-off, while
zero-probability molecules halve (74 → 37 for A_current).

**What this settles:**
1. **The core grid did under-smooth P2**, and fixing it helps every spatial and
   soft-assignment metric: NMI +0.05 to +0.10, balanced accuracy +0.10 to +0.16, Moran's I
   +0.12 to +0.21. P2 svs_star's NMI (0.407) now slightly exceeds P0's (0.392).
2. **It does not rescue KE weighting on ARI.** The best P2 hunter (svs_star 0.223) is still
   below P0 svs (0.241) and svs_star (0.234), and the mean over hunters moves only
   0.151 → 0.164.
3. **Its main failure is structural, not tuning:** three of seven vertices go to rare
   cell types at any λ, and larger λ does not reduce near-empty topics (they rise
   slightly, 1.7 → 2.1 per fit). If KE weighting is to be kept, it needs either a larger K
   or a hunter step that down-weights isolated, low-mass vertices. Otherwise P0 is the
   right default.
4. **For the extended grid:** P2/P3 must use `grid_len` 50. That costs about 9–16 min per
   task for the P2 spectral step plus 12 recoveries.

**A_full_Pois for p2_wide (added 2026-09-28, job 1939617).** All 54 geometries were refit
(0 failed; 4 at 1e-8, the rest near-converged with gap median 1.4e-6). Floored deviance,
mean over hunters: A_current 2.0240, A_full_L2 2.0192, **A_full_Pois 2.0167**; zero-probability
molecules 37 → 13. The ordering Pois < L2 < current holds again, with the largest gains for
PALM (2.035 → 2.015). **Correction to §3.12, finding 2:** there, Pois-refit P2 looked
LDA-level (P2 palm 2.0050, P2 SPA 2.0067), but that was with the under-smoothed core-grid Ŵ.
At P2's selected λ the Pois deviance is 2.015–2.020, a gap to LDA (2.0040) of 0.011–0.016,
the same as P0. So KE weighting has no prediction advantage either once λ is right.

---

### 3.15 K = 5 on Br5595 (k5_br5595, new 2026-09-28)

**Why:** section 151669 (donor Br5595) is labelled with only five layers (L3, L4, L5, L6,
WM), so K = 7 might over-split it. It was also the section with the largest seed spread
throughout. The k5_br5595 design reruns it at K = 5 with the same seeds, splits and panels as
core (P0 and P2 × 6 hunters × A_current/A_full_L2, anchor, all 6 baselines; 50-point λ
grid). The K = 7 references are core (P0, baselines) and p2_wide (P2, same λ grid). No
Poisson refit was run.

**Maps first** (`results/visium_dlpfc/diagnostics/k5_br5595/spatial/k5_151669_s<seed>.png`:
top row K = 7, bottom row K = 5; topic figures in `.../k5_br5595/topics/`):

- **Fewer topics do not make GpLSI find the five labelled layers.** At K = 5, SVS/SVS* still
  spend two topics on things the annotation does not separate. The SVS topic figure for
  seed 1 shows:

  | Topic | Mass | Where | Top enriched genes | What it is |
  |---|---:|---|---|---|
  | "≈ L6" | 20% | the top edge of the section | CALB1, CARTPT, HPCAL1, LAMP5 | **superficial L2/3** (labelled "L3") |
  | "≈ L3" | 25% | upper-middle cortex | CPB1, PVALB, VAMP1, SCN1B | middle cortex (L3/L4) |
  | "≈ L5" | 45% | all of L4–L6 (mean weight 0.69–0.94) | SMYD2, PCP4, NEUROD6, NPY | deep cortex, one topic |
  | "≈ WM" | 6% | bottom | MOBP, MBP, MYRF, PIP | white matter |
  | unmatched | 4% | a patch on the right edge | RELN, ADIPOQ, CIDEC, FABP4, SAA2 | meninges / adipose tissue |

- **The annotation limits ARI on this section.** Every good fit (at K = 5 and K = 7)
  finds a separate superficial L2/3 program along the top edge, and the manual labels call
  that whole region "L3". This is real biology that ARI counts as an error.
- **Which structure a seed gets varies a lot.** Seed 3 SVS (ARI 0.408, the highest of any
  method on this section) puts the top-edge topic in the WM colour and separates an L5 band
  from WM at the bottom. Seed 1 SVS (0.073) merges L4–L6 and splits the upper cortex in two.
  The same topics appear; the seeds differ in which boundaries they draw.
- **LDA and Spatial-LDA look the same at both K**: speckled, with the WM and the right-edge
  patch clearest.

**Numbers (151669, mean over 3 seeds; min–max in brackets):**

| Method | ARI K = 7 | ARI K = 5 | NMI K = 7 → 5 | Moran's I K = 7 → 5 | Seed JSD K = 7 → 5 |
|---|---|---|---|---|---|
| GpLSI P0 svs | 0.193 (0.07–0.27) | **0.236 (0.07–0.41)** | 0.369 → 0.393 | 0.685 → 0.805 | 0.048 → 0.075 |
| GpLSI P0 svs_star | 0.167 (0.13–0.21) | 0.198 (0.06–0.28) | 0.389 → 0.390 | 0.723 → 0.813 | 0.043 → 0.074 |
| GpLSI P0 pp_spa | 0.154 (0.12–0.20) | 0.156 (0.06–0.26) | 0.346 → 0.301 | 0.866 → 0.929 | 0.020 → 0.015 |
| GpLSI P0 spa_current | −0.042 | −0.093 | 0.158 → 0.183 | 0.823 → 0.858 | 0.062 → 0.044 |
| GpLSI P0 palm / palm_acc. | −0.007 / −0.012 | −0.046 / −0.017 | 0.142 → 0.187 / 0.198 → 0.266 | 0.835 → 0.904 / 0.848 → 0.913 | 0.060 → 0.049 / 0.054 → 0.035 |
| GpLSI P2 svs_star (wide λ) | 0.234 (0.11–0.32) | 0.087 (0.08–0.10) | 0.342 → 0.256 | 0.654 → 0.582 | — |
| LDA | 0.158 | 0.189 | 0.193 → 0.167 | — | 0.013 → 0.011 |
| Spatial-LDA | 0.195 | 0.179 | 0.265 → 0.213 | — | 0.008 → 0.006 |
| graph KL-NMF | 0.167 | 0.104 | 0.149 → 0.040 | — | 0.143 → 0.013 |

(P1 = P0 at K = 7, and P1/P3 were not rerun. Seed JSD uses A_current for GpLSI. It averages
over the K matched topics, so it compares only roughly across K.)

**What this shows:**
1. **K = 5 helps the best P0 hunters a little on average** (SVS +0.04, SVS* +0.03) and
   makes their maps smoother (Moran's I +0.09 to +0.12). But **the seed spread gets
   wider, not narrower** (SVS 0.07–0.41), and the topic profiles are *less* reproducible
   across seeds (JSD 0.048 → 0.075). With 5 vertices, which extreme points SVS picks
   depends more on the seed.
2. **SPA and PALM are no better at K = 5** (all ≤ 0 ARI): their failure on this section is
   not caused by too many topics.
3. **KE weighting gets much worse at K = 5** (P2 SVS* 0.234 → 0.087). Its rare-cell
   vertices take a larger share of a smaller K, the opposite of what a larger K was meant to
   relieve (§3.14).
4. **Held-out prediction is slightly worse at K = 5** (P0 A_current 1.9012 → 1.9024; LDA
   1.8862 → 1.8885), as expected with fewer topics.
5. **P0's λ is unchanged at K = 5** (0.0137, interior). P2's rises to 0.21–0.30.
6. **Conclusion:** K = 5 is not a fix for Br5595. For the extended grid, keep K = 7 as the
   main setting and read the K sweep (5/7/9) per donor. For this donor, the annotation's
   single "L3" label for the superficial cortex caps what ARI can show; the maps and topic
   figures are the better read.

---

## 4. Interpretation (3 donors, 3 seeds; core + panel + λ designs + Poisson refits)

1. **Vertex hunting matters most.** Replacing the original SPA hunter with SVS or SVS*
   more than doubles layer ARI on P0 (0.092 → 0.24). That puts GpLSI level with
   Spatial-LDA (0.239) and ahead of LDA (0.228) on average, at a fraction of Spatial-LDA's
   cost. pp-SPA is a solid middle choice (0.199) with the most reproducible topics across
   seeds. PALM and accelerated PALM are only marginally better than SPA on average (0.117
   and 0.122 vs 0.110).
2. **Original SPA is fragile on 151669.** Its ARI is negative there, and its seed spread
   is the largest of all hunters on 151507 and 151673. 151669 (Br5595) has only 5
   labelled layer classes while K = 7, so every method is penalized there, but SPA and
   PALM break down completely.
3. **Gene thresholding (Tran α = 0.005) does nothing at any panel size.** It removes at
   most 5 genes at p = 2,000 and at most 85 at p = 5,000, and P1 matches P0 exactly in 318
   of 324 panel records (§3.13). **Panel size matters little for svs_star** (ARI 0.234 →
   0.248 from 500 to 5,000 genes; same topics, better seed stability) and erratically for
   SPA/PALM. Held-out prediction of the common 500 genes is flat in panel size for every
   method.
4. **KE weighting (P2/P3) helps weak hunters and hurts strong ones, even at its own
   optimal λ.** On the core grid it raises SPA (0.092 → 0.128) but lowers SVS
   (0.241 → 0.171). Rerun at its cross-validated λ (p2_wide, §3.14) its maps become
   smooth (Moran's I 0.50 → 0.65) and NMI/balanced accuracy improve, but ARI only moves
   0.151 → 0.164, and the best P2 fit (svs_star 0.223) stays below P0 svs/svs_star. The
   cause is structural: KE weighting makes rare cell types (interneurons, plasma cells,
   red blood cells) extreme points, so 3 of 7 vertices go to near-empty topics at any λ.
5. **The A estimator does not change layer recovery, and ranks A_full_Pois > A_full_L2 >
   A_current for prediction** (§3.12). A_full_Pois wins in 100% of paired geometries and
   removes most zero-probability molecules (median 38 → 0), making 56% of fits'
   likelihoods finite. But it closes only 14–20% of the deviance gap to LDA for the good
   hunters. **The remaining gap is in Ŵ, not Â**: GpLSI's spot mixtures are smoothed and
   sparse. The leftover zeros come from that sparsity, and a small pseudocount in the
   refit would remove them. A_full_Pois costs about 13 CPU-min per geometry.
6. **Likelihood baselines lead on held-out deviance** (KL-NMF 2.002, LDA 2.004), but
   KL-NMF has the worst layer ARI. Prediction and layer recovery rank methods differently,
   so both must be reported.
7. **The penalty grid was too narrow for P2/P3 and for small panels.** For P0 at
   p = 2,000 the wide grid picks the same interior λ (§3.9, §3.10), so P0 conclusions
   stand. For P2 the optimum is λ ≈ 0.10–0.21; using it improves spatial and soft metrics
   substantially but ARI only slightly (§3.14). Panels of 500–1,000 genes also select the
   core grid's top (§3.13).

Caveats: only 3 donors (sections and seeds are not independent replicates); K = 7 exceeds
the 5 labelled layer classes on Br5595; ARI differences of a few hundredths are within
seed noise (see the ranges in §3.5).

---

## 5. Problems found and fixes made

| # | Problem | Status |
|---|---|---|
| 1 | `refit_poisson.py` took Ŵ only from the A_current record, so the 4 failed anchor A_current fits (no Ŵ saved at that index) would have become "failed" Poisson refits even though A_full_L2 saved the same Ŵ. | **Fixed:** new `geometry_sources()` picks Ŵ from any recovery of a geometry. Unit test `test_poisson_refit_takes_W_from_any_recovery_of_a_geometry` passes, and a 5-iteration real-data test refit 25/25 geometries in 52 s (anchor included). |
| 2 | `summarize.py` compared A estimators only on unfloored deviance, which is infinite for nearly all GpLSI fits, so almost nothing was compared. | **Fixed:** paired contrasts now also use floored deviance and zero-probability counts, and report ties. |
| 3 | `summarize.py` never read the Poisson optimality gap, so it could not report "near-converged". | **Fixed:** reads `normalized_optimality_gap` and reports how many fits reached 1e-8 plus the median and max gap of the rest. |
| 4 | The TopicSCORE W refit is unconverged but labelled `ok`. | **Fixed in reporting:** a new table in the summary. The TopicSCORE code is unchanged. |
| 5 | Anchor GpLSI was missing from the GpLSI-vs-baselines table. | **Fixed.** |
| 6 | The by-A-estimator table averaged unfloored deviance over finite values only, which is misleading. | **Fixed:** adds `finite_deviance_share` and zero-probability counts. |
| 7 | `summarize.py` did not know the new `lambda_wide_tsgd` design. | **Fixed:** new section and default design list. |
| 8 | Panel's 36 tasks do not fit under the 10-job limit. | **Fixed:** `slurm_array.sh` accepts `TASK_STRIDE`, so array element *i* runs tasks *i*, *i*+S, …; one failed task does not stop the others. |
| 9 | scikit-sparse 0.5 removed `cholesky_AAt` (breaks `pycvxcluster`). | **Fixed:** pinned `<0.5` in `environment.yaml`. |
| 10 | Anchor GpLSI Ŵ is rank-deficient at K = 7, so A_current is singular. | **Reported, not fixed.** A property of the method; kept as failures. |
| 11 | A_full_L2 hits its 2,000-iteration cap in 49% of fits (all SPA-hunter fits). | **Reported.** The solver (`recovery.py`) is untouched per the constraints. A_full_Pois beats it on every geometry anyway (§3.12), so raising the cap is low priority. |
| 12 | λ is selected at or near the grid top. | **Resolved:** P0 at p = 2,000 is interior (lambda_wide); P2 needs the wide grid (p2_wide, §3.14); p ≤ 1,000 also selects the grid top (§3.13). |
| 13 | `summarize.py`'s penalty-grid table mixed the anchor-GpLSI record (also P0/spa_current) into the "core" rows, so core and lambda_wide appeared to differ for P0 (e.g. 0.029 vs 0.039 on 151507). | **Fixed:** restricted to document GpLSI; the rows now match bit for bit. |
| 14 | `summarize.py`'s panel reference-panel deviance averaged the **unfloored** deviance over finite fits only, so each cell averaged a different subset (A_current at p = 500 was NaN; A_full_Pois looked worse than A_current). | **Fixed:** uses the floored reference-panel deviance, averaged over hunters, and adds reference zero-probability counts and baselines. |
| 15 | A_full_Pois leaves a few zero-probability molecules for sparse-Ŵ hunters (svs/svs_star). | **Reported.** A property of the unpenalized MLE; a pseudocount option would fix it (§3.12). |

---

## 6. Code and config changes (all uncommitted on `yeojin-exp`)

- `environment.yaml`: scikit-sparse `<0.5`.
- `scripts/visium_dlpfc/slurm_array.sh`: DSI partition/account, conda activation, logs in
  `logs/visium_dlpfc/`, and the `TASK_STRIDE` loop.
- `scripts/visium_dlpfc/slurm_refit.sh` (new): Slurm job for the Poisson refit.
- `configs/visium_dlpfc/ablation.json`: `poisson_refit` block (pooled start,
  interior_mass 1e-6, max_iter 1500, tolerance 1e-8), the `lambda_wide_tsgd` design, and
  (2026-09-28) the `p2_wide` design.
- `scripts/visium_dlpfc/make_tasks.py`, `configs/visium_dlpfc/tasks_lambda_wide_tsgd.csv`
  and `configs/visium_dlpfc/tasks_p2_wide.csv`: the new manifests. The other manifests are
  unchanged (regenerated byte for byte).
- `scripts/visium_dlpfc/refit_poisson.py`: fix 1.
- `scripts/visium_dlpfc/summarize.py`: fixes 2–7, 13, 14, and a `p2_wide` section.
- `scripts/visium_dlpfc/plot_core_diagnostics.py` (new): the §3.8 diagnostic figures.
- `scripts/visium_dlpfc/plot_spatial_maps.py`, `scripts/visium_dlpfc/plot_topics.py` (new):
  the §3.11 maps and topic figures.
- `scripts/visium_dlpfc/plot_design_maps.py` (new): side-by-side maps for panel (rows =
  panel sizes) and p2_wide (rows = core vs wide grid). `plot_topics.py` now takes
  `--design panel|p2_wide` with a matching method list and panel-size file names.
- `tests/test_dlpfc_ablation.py`: a new test for fix 1. **Full suite on a compute node
  (2026-09-28): 306 passed, 15 subtests passed, 1 min 51 s.**
- `docs/visium_dlpfc_smoke_test.md`, `docs/visium_dlpfc_results.md` (this file).
- Added later on 2026-09-28:
  - `configs/visium_dlpfc/ablation.json` and `make_tasks.py`: the `k5_br5595` design and
    `tasks_k5_br5595.csv`.
  - `summarize.py`: a K = 5 vs K = 7 section (K = 7 references: core for P0 and
    baselines, p2_wide for P2).
  - `plot_design_maps.py --design k5_br5595` (rows K = 7 / K = 5).
  - `scripts/visium_dlpfc/make_report_pdf.py` (new): the one-PDF figure report.
  - `scripts/visium_dlpfc/compare_A_estimators.py` (new): per-topic A-estimator comparison
    on a shared Ŵ.

Untouched, per the constraints: `src/gplsi_spatial_benchmark/runner.py` and the Poisson
solver in `src/gplsi/recovery.py`.

---

## 7. What is left

Done on 2026-09-28: the Poisson refits (core, panel, lambda_wide); the panel analysis and
figures; p2_wide (the P2 wide-grid rerun) with figures; `summarize.py` on all designs, with
fixes 13 and 14; the full `pytest` (306 passed); the ablation doc §9/§12. Later the same
day: the Poisson refit for p2_wide (§3.14), K = 5 on Br5595 (§3.15), the PDF report
(`results/visium_dlpfc/report/`), and a per-topic A-estimator comparison on a shared Ŵ
(`scripts/visium_dlpfc/compare_A_estimators.py`, output in
`results/visium_dlpfc/diagnostics/core/A_compare/`: for P0 SVS, A_current ≈ A_full_L2 per
topic; A_full_Pois keeps every real topic's top genes, entropy and between-topic structure
and only fills in the low-probability tail; large changes occur only in near-empty topics).

1. **Commit to `yeojin-exp`** when you ask (no push without approval). §6 lists the files.
2. **Decisions for the meeting:**
   - **Thresholding axis:** it is empty at α = 0.005 at every panel size. Drop it, or try a
     much stronger α?
   - **KE weighting (P2/P3):** keep it (with `grid_len` 50 and perhaps K = 9, so rare-cell
     topics don't starve the layers) or drop it from the extended grid? K = 5 on Br5595
     made it much worse (§3.15), and it has no prediction advantage at its right λ (§3.14).
   - **Hunters:** svs vs svs_star are nearly redundant on P0. svs_star is the more robust
     (best or near-best on every section under both weightings, best P2). PALM variants add
     little over SPA. The extended grid could keep spa_current (original), svs_star and
     pp_spa.
   - **A_full_Pois in the extended grid:** it costs about 13 CPU-min per geometry. Run it
     only for the kept hunters, and add a small pseudocount option to remove the remaining
     zero-probability molecules (§3.12)?
3. **Optional follow-ups** (not run):
   - A fixed-λ sweep (a real λ ablation; needs a small `ablation_runner.py` change).
   - Poisson refit for k5_br5595 (about 45 min on one job), if K = 5 prediction matters.
   - `compare_A_estimators.py --geometry gplsi_document__P0_raw__palm`, to see the
     per-topic comparison where A_full_L2 does differ from A_current.
4. **Extended 360-task grid after the meeting:** pack with `TASK_STRIDE` (10-job limit),
   and use `grid_len` 50 for P2/P3 and for panels with p ≤ 1,000 (§3.13).

---

## Appendix A. Metrics guide

This appendix gives the plain-language reading of every number in this document. The
formal definitions are in [`visium_dlpfc_ablation.md` §8](visium_dlpfc_ablation.md), and
the code is `src/gplsi_spatial_benchmark/metrics.py` (called from `score_fit` in
`ablation_runner.py`).

**Notation:**
- Ŵ (spots × K) is each spot's topic mixture; its rows sum to 1.
- Â (K × genes) is each topic's gene profile; its rows sum to 1.
- P = ŴÂ is the predicted gene distribution of each spot.
- Y is the held-out test counts (20% of molecules), and mᵢ is spot i's number of test
  molecules.

Every metric is computed on the same task (section, seed, split, panel), so methods are
compared on identical data.

**What each metric depends on:**

| Metric group | Depends on |
|---|---|
| Layer ARI, layer NMI, layer balanced accuracy, Moran's I, neighbour agreement | Ŵ only, so identical for A_current / A_full_L2 / A_full_Pois of one geometry |
| Deviance, log score, zero-probability molecules | both Ŵ and Â, so they separate the A estimators |
| Topic entropy, top-gene exclusivity | Â only |

### A.1 Layer recovery: does Ŵ find the cortical layers?

These use the manual layer annotation (`layer_guess_reordered`: L1–L6 and white matter),
which the methods never see. Only labelled spots are scored.

#### Layer ARI (Adjusted Rand Index)

`external__layer_guess_reordered__ari`; summary column `layer_ARI`.

- **What:** each spot goes to its dominant topic (argmax of Ŵᵢ), giving a clustering.
  ARI asks, over all pairs of spots, how often the clustering and the manual layers agree
  on whether the pair belongs together, corrected so that a random labelling with the
  same group sizes scores 0.
- **Range:** −0.5 to 1, higher is better. 1 is a perfect match up to renaming; 0 is
  chance; below 0 is worse than chance.
- **Our values:** 0.05–0.25 for method means; best single runs about 0.35 (LDA on
  151507).
- **Caveats:**
  - The hard argmax discards the mixture, so boundary spots are forced into one topic.
  - A topic model isn't trained to find layers: some topics track cell types or
    technical effects.
  - K = 7 exceeds the 5 labelled classes on 151669 (Br5595), so the extra topics must
    split real layers and every method's ARI is capped there.
  - ARI is sensitive to large groups (e.g. white matter).
  - Read differences against the seed spread (§3.5, Fig. 8): a few hundredths is often
    noise.

#### Layer NMI (Normalized Mutual Information)

`…__nmi`; summary column `layer_NMI`.

- **What:** the same hard clustering. It measures how much knowing a spot's topic tells
  you about its layer (mutual information divided by the average of the two entropies).
- **Range:** 0 to 1, higher is better. It is not adjusted for chance, so random
  labellings score slightly above 0.
- **Our values:** 0.26–0.39 for GpLSI.
- **Caveats:** NMI is more forgiving than ARI when one layer is split across two
  topics, because the split still carries information. That is why NMI and ARI can
  rank methods differently.

#### Layer balanced accuracy

`…__cv_balanced_accuracy`; summary column `layer_balanced_acc`.

- **What:** uses the whole soft mixture Ŵᵢ, not the argmax. A class-weighted multinomial
  logistic regression predicts the layer from Ŵᵢ, with 5-fold stratified
  cross-validation. It reports the mean per-class recall, so small layers count as much
  as large ones.
- **Range:** 0 to 1, higher is better. Chance is 1/(number of layers): about 0.14 with 7
  classes, 0.2 on 151669.
- **Our values:** 0.49–0.78.
- **Caveats:** it answers "is layer information linearly present in Ŵ?", not "do the
  topics line up with layers". It can be high when ARI is low, e.g. the PALM hunters
  (balanced accuracy 0.77, ARI 0.11 on P0).

### A.2 Held-out prediction: does ŴÂ predict unseen molecules?

Twenty percent of each spot's molecules are held out before fitting. These metrics ask
how well P = ŴÂ predicts which genes those molecules belong to. The spot's test total mᵢ
is taken as given, so they score how molecules are allocated across genes, not
sequencing depth.

#### Held-out Poisson deviance per molecule

`heldout_poisson_deviance_per_molecule`; summary column `deviance_per_molecule`.

- **What:** Dev = 2 Σᵢⱼ [Yᵢⱼ log(Yᵢⱼ / (mᵢPᵢⱼ)) − (Yᵢⱼ − mᵢPᵢⱼ)], divided by the total
  number of test molecules. It is 0 for a perfect prediction and grows as the predicted
  gene distribution departs from the observed counts.
- **Range:** ≥ 0, lower is better.
- **Our values:** about 1.8–2.4. The level differs by section (151507 is about 0.4 higher),
  so compare methods within a section or in paired differences.
- **Caveats:**
  - A single held-out molecule in a gene with Pᵢⱼ = 0 makes it **infinite**. That is the
    case for almost every GpLSI and TopicSCORE fit, so the unfloored value is usually
    missing (`None`/NaN) and averages of it cover only the finite subset.
  - It isn't comparable across panel sizes, because more genes means a harder
    prediction problem. Use the reference-panel version (A.2, reference-panel deviance)
    instead.

#### Floored deviance per molecule

`heldout_poisson_deviance_per_molecule_floored_1e-12`; summary column `deviance_floored`.

- **What:** the same deviance after raising every Pᵢⱼ to at least 10⁻¹² and
  renormalizing, so it is always finite. This is the number used to compare A
  estimators and methods.
- **Range:** ≥ 0, lower is better.
- **How much the floor adds:** each zero-probability molecule costs about 40 deviance
  units (about 2·log(1/(mᵢ·10⁻¹²))), around 20× a typical molecule. But they are rare,
  so in core the floor adds a median of 0.0009 per molecule (maximum 0.019). Differences
  in floored deviance therefore mostly reflect genuine prediction quality.
- **Scale of differences:** in core, method differences are about 0.005–0.05; A_full_L2
  vs A_current differs by a median of 0.0006, and by up to about 0.02 for PALM/SPA.

#### Held-out log score per molecule

`heldout_log_likelihood_per_molecule`; summary column `loglik_per_molecule`.

- **What:** Σ Yᵢⱼ log Pᵢⱼ / Σ Y, the average log-probability assigned to each held-out
  molecule. It is the deviance's likelihood counterpart, without the saturated-model
  constant.
- **Range:** ≤ 0, higher (closer to 0) is better.
- **Caveats:** it is −∞ whenever the deviance is infinite. A floored variant is also
  stored.

#### Zero-probability molecules

`heldout_zero_probability_molecules`; summary column `zero_prob_molecules`. Also stored:
`_entries` (spot–gene cells) and `_rows` (spots affected).

- **What:** the number of held-out molecules that fall in a gene the model gives
  probability exactly 0 in that spot. It happens when Â has exact zeros, which the
  simplex projections in A_current, A_full_L2 and TopicSCORE produce, and Ŵ doesn't mix
  in another topic that covers the gene.
- **Range:** ≥ 0, lower is better; 0 is required for a finite deviance.
- **Our values:** GpLSI median about 35 per fit (up to about 1,200), TopicSCORE 75–100,
  and exactly 0 for LDA, Spatial-LDA and both KL-NMFs (their gene profiles are strictly
  positive).
- **Caveats:** this is a model-validity flag (the model calls observed data impossible)
  more than an accuracy measure. It is tiny relative to the about 2.3M test molecules
  per task.

#### Reference-panel deviance and zero-probability molecules

`reference_panel__heldout_poisson_deviance_per_molecule` and
`reference_panel__heldout_zero_probability_molecules`; summary columns `ref500_*`.

- **What:** the same metrics after restricting P and Y to a fixed set of 500 genes (the
  top of the training-only dispersion ranking) that is in every panel, with P
  renormalized over those genes.
- **Why:** the only deviance comparable across panel sizes (500–5,000 genes). It is the
  main prediction metric for the panel design.
- **Range:** ≥ 0, lower is better. Values are about half the full-panel values (about
  1.0 on 151673), because 500 genes is an easier prediction problem.

### A.3 Spatial structure of Ŵ (diagnostics, not accuracy)

These are computed on the symmetric 6-nearest-neighbour spatial graph, with Gaussian
edge weights. **They are not accuracy metrics:** any method can raise them by
over-smoothing. Read them as "how smooth is the map", alongside ARI.

#### Moran's I

`spatial_topic_moran_mean`; summary column `moran_I`.

- **What:** for each topic, the spatial autocorrelation of its weight Ŵ·ₖ across graph
  neighbours, averaged over topics:
  I = (n / Σw) · Σ₍ᵢ,ⱼ₎ wᵢⱼ zᵢ zⱼ / Σᵢ zᵢ², where z is the centred topic weight.
- **Range:** about −1 to 1. Near 1 means neighbouring spots have very similar topic
  weights, 0 means no spatial pattern, and < 0 is a checkerboard.
- **Our values:** GpLSI P0 about 0.75–0.88; P2 about 0.45–0.52 on the core grid and about
  0.61–0.63 at its own optimal λ (§3.10).
- **Caveats:** it goes up mechanically with the graph penalty λ, so compare it only at
  comparable smoothing, or read it together with the selected λ.

#### Neighbour agreement

`spatial_hard_topic_neighbor_agreement`; summary column `neighbor_agreement`.

- **What:** the edge-weighted fraction of neighbouring spot pairs whose dominant topics
  (argmax) are the same.
- **Range:** 0 to 1, higher is smoother. About 1/K would be expected for random
  assignment of equally sized topics.
- **Our values:** GpLSI about 0.74–0.94.
- **Caveats:** the same as Moran's I. Real layer boundaries cap the achievable value
  below 1.

### A.4 Topic–gene profiles Â (interpretability)

#### Topic entropy

`topic_entropy_mean`; summary column `topic_entropy`.

- **What:** for each topic, the entropy of its gene distribution divided by log(number of
  genes), averaged over topics.
- **Range:** 0 to 1. Low means the topic puts its mass on few genes (sharp, marker-like);
  1 means uniform over genes.
- **Our values:** about 0.76 for GpLSI.
- **Caveats:** there is no "better" direction by itself. Very low entropy can mean a
  topic dominated by one or two highly expressed genes (e.g. mitochondrial or
  haemoglobin). Use it to compare A estimators on the same Ŵ.

#### Top-gene exclusivity

`top_gene_exclusivity_mean`; summary column `top_gene_exclusivity`.

- **What:** for each gene, the share of its total topic mass held by its most-loaded
  topic, max_k Âₖⱼ / Σₗ Âₗⱼ. It is averaged over each topic's 20 highest-probability
  genes, then over topics.
- **Range:** 1/K (about 0.14) to 1, higher is more distinctive. At 1, each topic's top
  genes are unique to it.
- **Our values:** about 0.35 for GpLSI, so a topic's top genes are shared with other
  topics. That is expected, because highly expressed housekeeping genes top several
  topics.
- **Caveats:** it measures only the top 20 genes per topic. The top-20 symbols themselves
  are stored per record in `top_features` for marker checks.

### A.5 Reproducibility across seeds

#### Seed stability (matched Jensen–Shannon divergence)

`seed_JSD`, computed in `summarize.py`; written to `seed_stability.csv`.

- **What:** for each section and method, compare the Â from every pair of seeds.
  - The seeds' panels can differ slightly, so only shared genes are used.
  - Topics are matched between the two runs by the Hungarian algorithm.
  - The Jensen–Shannon divergence (base 2) is averaged over matched topics, then over
    seed pairs.
- **Range:** 0 to 1, lower is more reproducible. 0 means identical topic profiles across
  seeds.
- **Our values:** Spatial-LDA 0.007 and LDA 0.014 (most stable); GpLSI 0.02–0.09 by hunter,
  with pp_spa most stable and spa_current least; TopicSCORE 0.20–0.23.
- **Caveats:** the seed changes the train/test split (and the method's own randomness),
  so this mixes data and algorithm variability.

#### Topic usage / collapse (§3.11)

This is not stored per record; it is computed from the saved Ŵ.
- **Near-empty topics:** topics whose share of total topic mass Σᵢ Ŵᵢₖ / n is below 1%.
- **Topics used:** topics that are the argmax of at least 1% of spots.
- **How to read it:** K = 7 was requested. A fit that uses 4 topics has effectively chosen
  a smaller K, usually by spending vertices on rare cell programs (P2/P3) or by one topic
  absorbing most spots (KL-NMF, SPA). Low usage caps achievable ARI. It is a property of
  Ŵ, so it is identical across A estimators.

#### ARI seed range

This is not stored. It is computed in the §3.5 tables and Fig. 8: the max − min layer ARI
over the 3 seeds of one section. It is the quickest guide to whether an ARI difference
is real.

### A.6 Paired contrasts (dimension A)

Every A estimator of one geometry shares the same Ŵ (same task × preprocessing ×
hunter). The comparison is therefore paired: take B − A per geometry, then summarize
the paired differences (mean, median, share with B lower, share tied).

- This removes all variation due to Ŵ, section and seed, so small effects (about 10⁻³)
  are detectable.
- Paired contrasts use only geometries where both estimators have a finite value for
  that metric.

### A.7 Run diagnostics (per record)

| Diagnostic | Field | How to read |
|---|---|---|
| Status | `status` | `ok`; `max_iter_reached` (a fit exists but the optimizer hit its cap; metrics still computed); `failed` (no fit, exception kept). Failed records are never dropped. |
| A_full_L2 convergence | `metadata.A_recovery.converged`, `iterations`, `projected_gradient_norm` | Converged when the projected-gradient norm falls below tolerance (about 1e-5 in practice) within 2,000 iterations. Norms of 1e-4 to 1e-1 at the cap mean the fit is progressively further from optimal (Fig. 4). |
| A_full_Pois gap | `metadata.A_recovery.normalized_optimality_gap` | Frank–Wolfe/KKT gap divided by total training counts. The tolerance is 1e-8; fits stopping at 1,500 iterations above it are labelled **near-converged**, with the gap reported (about 1e-5 in diagnostics). |
| PALM convergence | `metadata.vertex_hunting.optimizer_converged` | The PALM hunter can return vertices without converging; the result is still used. |
| TopicSCORE W refit | `metadata.W_refit_converged` | Converged only in 1–2 of 9 core tasks, while the status still says `ok`. |
| Selected λ | `metadata.selected_rho` | The graph penalty chosen by 5-fold CV. Selection at the grid top means the grid is too narrow (§3.3, §3.10). |
| Genes retained | `metadata.retained_feature_count` | How many panel genes survive the P1/P3 threshold. |
| Runtime | `runtime_seconds` | Wall time for the record, including its share of the spectral step, so summing across records double-counts it. |
