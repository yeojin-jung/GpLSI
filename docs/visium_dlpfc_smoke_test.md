# Visium DLPFC ablation — smoke test report

> Recorded on branch `yeojin-exp` before the merge with the handoff code. Paths refer to that
> layout: `scripts/visium_dlpfc/` is now `scripts/analysis/dlpfc/` (plus `scripts/run_experiment.py`
> and `scripts/refit_A.py`), `configs/visium_dlpfc/` is `configs/dlpfc/ablation/`, and `results/visium_dlpfc/`
> is `results/dlpfc/`. See `docs/visium_dlpfc_ablation.md` §10–11.

Date: 2026-09-27 · Branch: `yeojin-exp` · Commit at run time: `9f4eb12` (plus uncommitted
DSI setup changes to `environment.yaml`, `scripts/visium_dlpfc/slurm_array.sh`,
`configs/visium_dlpfc/ablation.json`).

**Verdict: pass.** All 56 method records ran end to end on real data. Every record has a
status and metrics, and there were no exceptions. The run is a wiring check: the ARI and
deviance numbers below are **not scientific results**, because the spectral step was
deliberately truncated (see §1).

## 1. What was run

| Item | Value |
|---|---|
| Slurm job | `1917480` (1-element array), partition `general`, account `general_group`, node `g009` |
| Resources requested | 5 CPUs, 32 GB, 12 h; BLAS threads pinned to 1; `GPLSI_GRAPH_N_JOBS=5` |
| Task manifest | `configs/visium_dlpfc/tasks_smoke.csv` |
| Task | design `smoke`, section **151673** (Br8100, 3,639 spots, 7 layers), K = 7, 2,000-gene training-only panel, `retained_fraction = 1.0` (no thinning), seed 26090401 |
| Data | 9,125,011 train / 2,280,982 test molecules (20% held out); 13,368 eligible genes (detection threshold 36 spots); 500-gene reference panel; 0 spots dropped |
| Graph | symmetric 6-NN, 11,160 edges, Gaussian weights scaled by the within-section median kNN distance |
| Spectral step (smoke override) | `grid_len = 3`, `maxiter = 3`, `nfolds = 3` (core uses 29 / 50 / 5) |
| Output | `results/visium_dlpfc/smoke/smoke__151673__K7__p2000__r1.0__s26090401.{json,npz}` |
| Logs | `logs/visium_dlpfc/slurm-1917480_0.{out,err}` |

Method suite (same width as `core`):

- **Document-side GpLSI**: 4 preprocessings (P0_raw, P1_tran_alpha_0p005, P2_ke_weighted,
  P3_tran_then_ke) × 6 vertex hunters (spa_current, svs, svs_star, pp_spa, palm,
  palm_accelerated) × 2 A recoveries (A_current, A_full_L2) = 48 records.
- **Anchor GpLSI**: P0_raw / spa_current × 2 A recoveries = 2 records.
- **Baselines**: topicscore_graph_denoised, topicscore_raw, lda, kl_nmf, graph_kl_nmf,
  spatial_lda = 6 records.
- A_full_Pois is **not** part of the runner. It is produced afterwards by
  `scripts/visium_dlpfc/refit_poisson.py` from the saved Ŵ, and was not exercised here.

## 2. Acceptance checks

| Check | Result |
|---|---|
| Every record carries a status | ✅ 56/56 — 49 `ok`, 7 `max_iter_reached`, 0 `failed` |
| Non-failed records have metrics | ✅ all 56 have layer ARI, held-out deviance (floored), and zero-probability counts |
| No unexpected exceptions | ✅ none; stderr contains only one sklearn NMF `ConvergenceWarning` (a 1-iteration initializer) |
| Job health | `COMPLETED`, exit 0:0 |

## 3. Running time and memory

- **Wall clock: 21 m 37 s** (runner-reported 1,290 s).
- **Peak memory: ~1.5 GiB** (runner) and 2.0 GB MaxRSS (Slurm), against 32 GB requested.

Per-method runtime (`runtime_seconds`):

| Method | Seconds | Share of wall |
|---|---:|---:|
| spatial_lda | 601.2 | 47% |
| topicscore_raw | 282.3 | 22% |
| topicscore_graph_denoised | 174.0 | 13% |
| lda | 57.5 | 4% |
| graph_kl_nmf | 56.0 | 4% |
| kl_nmf | 13.6 | 1% |
| each GpLSI record | 4.9 – 10.1 | — |

The GpLSI runtimes include the spectral step that the records of one preprocessing
share (`shared_spectral_runtime_seconds` = 5–9 s per preprocessing). Summing
`runtime_seconds` therefore double-counts that step (1,576 s summed vs 1,290 s wall). With
the smoke spectral settings, GpLSI in total costs about 2 minutes, and **the baselines use
about 92% of the wall time**. Vertex hunting takes about 1 ms for SPA-type hunters and
about 1.1 s for PALM.

## 4. Results (smoke settings — for wiring only)

### Layer ARI (`external__layer_guess_reordered__ari`), document GpLSI

A_current and A_full_L2 give identical ARI, as expected, because ARI depends only on Ŵ.

| Hunter | P0_raw | P1_tran | P2_ke | P3_tran_then_ke |
|---|---:|---:|---:|---:|
| spa_current | 0.143 | 0.143 | 0.146 | 0.146 |
| svs | 0.102 | 0.102 | 0.167 | 0.167 |
| svs_star | 0.102 | 0.102 | 0.167 | 0.167 |
| pp_spa | 0.119 | 0.119 | 0.161 | 0.161 |
| palm | 0.150 | 0.150 | 0.142 | 0.142 |
| palm_accelerated | 0.157 | 0.157 | 0.127 | 0.127 |

Anchor GpLSI (P0/SPA): 0.071.

Baselines:

| Baseline | ARI |
|---|---:|
| spatial_lda | 0.259 |
| lda | 0.203 |
| graph_kl_nmf | 0.179 |
| kl_nmf | 0.102 |
| topicscore_graph_denoised | 0.094 |
| topicscore_raw | 0.064 |

### Held-out deviance and zero-probability molecules

| Method (selection) | dev/molecule (floored 1e-12) | zero-prob molecules (main / reference panel) |
|---|---:|---:|
| kl_nmf | 1.8109 | 0 / 0 |
| graph_kl_nmf | 1.8124 | 0 / 0 |
| lda | 1.8158 | 0 / 0 |
| spatial_lda | 1.8197 | 0 / 0 |
| GpLSI P2 palm, A_current → A_full_L2 | 1.8189 → 1.8183 | 22 → 90 |
| GpLSI P0 spa_current, A_current → A_full_L2 | 1.8291 → 1.8277 | 18 → 31 |
| GpLSI P0 palm, A_current → A_full_L2 | 1.8813 → 1.8395 | 401 → 7 |
| GpLSI P0 pp_spa, A_current → A_full_L2 | 1.8640 → 1.8733 | 673 → 1,233 |
| topicscore_raw | 1.8307 | 172 / 160 |
| topicscore_graph_denoised | 1.8341 | 148 / 126 |

Every GpLSI and TopicSCORE fit assigns zero probability to at least one held-out molecule
(range 7–1,233). The likelihood baselines never do.

## 5. Issues found and how they will be resolved

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 1 | **Unfloored deviance is `None`** for all 50 GpLSI and both TopicSCORE records, because Â gives zero probability to held-out molecules, which makes the deviance infinite. This is intended behavior, not a bug, but a naive summary would silently drop these methods from deviance comparisons. | Medium (reporting) | `summarize.py` will report the floored deviance **together with** the zero-probability molecule count, and will never average over `None`. The main fix for these methods is the A_full_Pois refit (`refit_poisson.py`), which produces strictly positive Â. The ad-hoc inspection script's "non-finite deviance" flag was a false alarm and will be corrected. |
| 2 | **7 A_full_L2 fits hit `max_iter_reached`** (2,000 iterations): anchor P0/SPA, and P2 and P3 × {spa_current, palm, palm_accelerated}. Projected-gradient norms are 1e-4 to 3e-4, except the anchor fit at 3e-2. | Medium | These are kept and reported as non-converged, never dropped. After core, compare them with their converged counterparts. If they matter, raise the A_full_L2 iteration cap in a config override. The solver in `src/gplsi/recovery.py` itself will not be modified. |
| 3 | **PALM hunters report `optimizer_converged: False`** on P2 and P3. The vertex-hunting status is still `ok`. | Low–medium | Surface `vertex_hunting.optimizer_converged` as a column in the summary tables. Watch whether it persists at full spectral settings. |
| 4 | **TopicSCORE W refit is not converged** (`W_refit_converged: False` after 5,000 iterations), yet the record status is `ok`. The status understates the problem. | Medium (reporting) | `summarize.py` will read `W_refit_converged` and flag the record. Whether to raise the TopicSCORE refit cap will be decided after core. |
| 5 | **Selected λ is at the edge of the grid.** Every preprocessing selected ρ = 0.000144, the largest value on the 3-point smoke grid. This is expected with a 3-point grid, but the same edge effect in core would mean the 29-point grid is too narrow. | Watch | Check `selected_rho` in the core results. The `lambda_wide` design (grid extended to about 0.75) tests exactly this. |
| 6 | **P0 ≈ P1 and P2 ≈ P3.** The Tran threshold (α = 0.005) removes only 5 of 2,000 genes on this panel, so ARI is identical and deviance nearly so. | Design | Not a bug. On a training-only 2,000-gene panel, thresholding has almost nothing left to remove. The `panel` design (500–5,000 genes) is where thresholding can matter. This will be noted when interpreting the preprocessing contrast. |
| 7 | **svs ≈ svs_star.** ARI is identical in all four preprocessings. Â differs slightly only in P3 (zero-prob 86 vs 128). | Low | Keep both in core. If core confirms they are redundant, drop svs_star from later designs to save cost. |
| 8 | **Direction of A_full_L2 vs A_current is mixed.** A_full_L2 increases zero-probability molecules for SPA/pp_spa hunters but sharply reduces them for PALM (401 → 7). | Scientific | This is what the paired A-estimator contrast will quantify on core. It is not interpretable from one smoke task. |
| 9 | **spatial_lda cost** is 10 minutes per task, about half the wall time. | Budget | It stays in core, as decided. For `panel`, it is already excluded. |
| 10 | **Environment:** scikit-sparse 0.5.0 removed `cholesky_AAt`, which `pycvxcluster.algos.admm` imports. | Fixed | Pinned `scikit-sparse>=0.4.12,<0.5` in `environment.yaml`. |

## 6. Implications for sizing the real runs

- The smoke spectral step (3 λ × 3 folds × ≤3 iterations) is tiny. Core runs 29 λ × 5 folds
  with up to 50 iterations, so the spectral cost may grow by up to about 270× in the worst
  case. That is roughly 35 minutes per preprocessing, or about 2.5–3 h per core task
  including baselines. This is an extrapolation, not a measurement.
- Memory is small (under 2 GB), so core requests **8 GB**.
- **Submitted:** core, 9 tasks (3 sections × 3 seeds), as array **`1935143`**, with 5 CPUs,
  8 GB and 12 h each, capped at 9 concurrent jobs (`%9`). The 9 jobs respect the 10-job
  limit, and each job runs its ~56 fits sequentially.
- **Measured (core task `1935143_8`, 151673 / seed 26090403, node p001):**
  - 12 m 55 s runner wall time (13 m 57 s in Slurm), with 2.8 GB MaxRSS. The extrapolation
    above was far too pessimistic: the graph SVD stops after 2 outer iterations, so the full
    spectral step takes 11–31 s per preprocessing.
  - spatial_lda (315 s) still dominates.
  - 56 records: 46 `ok`, 9 `max_iter_reached` (all A_full_L2), and 1 `failed`. The failure
    is `gplsi_anchor__P0_raw__spa_current__A_current`, with `RecoveryError: historical
    current A recovery is singular`. It is recorded, not dropped.
  - **Selected λ is at or near the top of the 29-point grid:**
    - P2 and P3 chose 0.01648, the top value.
    - P0 and P1 chose 0.01145, one step below.

    This confirms issue 5, and makes `lambda_wide` necessary.
- `lambda_wide` (3 tasks) and `panel` (36 tasks, which will need batching under the job
  limit) wait until the first core task reports a measured runtime.

## 7. Next steps

1. Monitor `1935143`. When the first task finishes, check its status coverage, metrics,
   `selected_rho` and measured spectral runtime, and update §6.
2. Commit the DSI setup changes and this report to `yeojin-exp` (no push without approval).
3. After core: run `refit_poisson.py` (A_full_Pois) as a Slurm job, then `summarize.py`.
   Fix both scripts against real output, adding tests, including the reporting points in
   issues 1, 3 and 4.
4. Then submit `lambda_wide`, then `panel` in batches of ≤10 jobs.
