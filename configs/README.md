# Experiment configs

One JSON file = one experiment, run by `scripts/run_experiment.py`. Results
go to `<output_root>/<name>/<task id>/`.

| Folder | Experiments |
|---|---|
| `base.json` | The paper protocol shared by all four datasets ([docs/protocol.md](../docs/protocol.md)) |
| `crc/`, `spleen/`, `cook/`, `dlpfc/` | `production.json` (the protocol's full pass on that dataset) and `smoke.json` (the same methods on a tiny subset, one job); DLPFC also `production_tran.json` / `smoke_tran.json` (Tran α = 0.1 vocabulary instead of the 2,000-gene dispersion panel) |
| `dlpfc/ablation/` | Visium DLPFC ablation designs and their own `base.json`: `core`, `panel`, `lambda_wide`, `lambda_wide_tsgd`, `p2_wide`, `k5_br5595`, `extended`, `smoke` (see `docs/visium_dlpfc_ablation.md`) |
| `handoff/` | Claire Donnat's 2026-09-15 production designs (CRC, spleen, Cooking, Cooking Tran-threshold sensitivity and SVS\* stability-L runs), to reproduce the handoff reports; results go to `results/handoff_*` |
| `simulations/` | Configs of the synthetic experiments in `scripts/simulations/` (their own format) |

## Fields

```jsonc
{
  "extends": "../base.json",        // optional; dicts are merged, lists and "parts" replaced
  "name": "crc_production",         // results directory (default: file name)
  "description": "...",
  "output_root": "results",
  "dataset": {"name": "crc"},       // crc | spleen (+ "group": "BALBc-1".."3" or "joint") | cook | cook_v2 | dlpfc
                                    // dlpfc: "vocabulary": "dispersion" (top panel_size genes) or "tran" (+ "tran_alpha")
  "grid": {"K": [5, 6], "seed": [1, 2]},   // one task per combination;
                                           // dlpfc adds "section", "panel_size", "retained_fraction"
  "subset": {"n": 80, "strategy": "graph_stratified"},   // optional: connected, graph_stratified, or whole_groups
  "heldout_fraction": 0.2,          // binomial count thinning: train / held-out molecules
  "post_tran_row_filter": false,    // Cooking alpha runs: drop recipes empty after the Tran threshold

  "spectral": {                     // graph-aligned SVD, one fit per preprocessing
    "lamb_start": 1e-4, "step_size": 1.2, "grid_len": 29, "maxiter": 50, "eps": 1e-5,
    "nfolds": 5, "n_jobs": 1, "cv_fold_mode": "all",
    "lambda_selection_mode": "cv_once",      // or "cv_each_iteration"
    "initialization": "weighted_debiased_mean_N_approx"   // or "current"
  },
  "gplsi": [                        // GpLSI variants: geometry x preprocessings x hunters
    {"geometry": "document_U", "preprocessings": ["P0_raw", "P2_ke_weighted"],
     "vertex_hunters": ["spa_current", "svs", "svs_star", "pp_spa", "palm", "palm_accelerated"]},
    {"geometry": "word_Z", "preprocessings": ["P0_raw"], "vertex_hunters": ["spa_current"]}
  ],
  "A_recoveries": ["A_current", "A_full_L2", "A_full_Pois", "A_full_Pois_SQUAREM"],
  "recovery": {"poisson_max_iter": 2000, "poisson_tolerance": 1e-8,
               "poisson_start": "pooled"},  // or "A_current" (warm start from the paired A_current)
  "vertex_parameters": {"svs": {"L_mode": "mixedscore_adaptive"}},
  "unavailable": [{"K": [8], "vertex_hunters": ["svs"], "reason": "..."}],  // recorded, not fitted
  "baselines": ["plsi", "topicscore_raw", "topicscore_graph_denoised", "lda",
                "spatial_lda", "kl_nmf", "graph_kl_nmf"],
  "topicscore_graph_preprocessing": "P0_raw",   // spectral block reused by graph-denoised TopicSCORE
  "spatial_lda_parameters": {},
  "posthoc_refit": {"A_recovery": "A_full_Pois", "poisson_start": "pooled"},  // scripts/refit_A.py defaults

  "parts": {                        // optional: split each task into separately scheduled jobs
    "P0_raw__fast": {"gplsi": [...], "baselines": []},
    "baseline_seeds": {"grid": {"seed": [2, 3]}, "gplsi": [], "baselines": ["lda"]}
  }
}
```

Preprocessings: `P0_raw` (none), `P1_tran_alpha_0p005` (Tran threshold),
`P2_ke_weighted` (inverse-root-frequency weights), `P3_tran_then_ke` (both), and the
Cooking sensitivity variants `P1_tran_alpha_0p01_drop_zero_rows`, ... (full list in
`gplsi.real_experiment.PREPROCESSING_SPECS`).

Every fit's row records a cache key built from the data hashes, these settings
(excluding the method lists, so parts share rows) and the source code; a rerun
skips rows whose key is unchanged.
