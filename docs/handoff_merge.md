# Merging the 2026-09-29 real-data handoff (branch `yeojin-merge`)

The handoff (`GpLSI_handoff/`, CRC, joint spleen and What's Cooking) and this
repository both descend from commit `4a46188`. This branch keeps one copy of
the source and one experiment pipeline for all four datasets.

## Source changes taken from the handoff

| File | Change |
|---|---|
| `graphSVD.py` | `lambda_selection_mode="cv_once"`: select the penalty once on the initial V, then hold it fixed; `select_lambda_by_cv` / `update_U_tilde_fixed_lambda` split out |
| `vertex_hunting.py` | SVS\* `L_mode="svs_star_stability"` (native, scalable L selection); exact polynomial MixedSCORE stability (sorted row-sum matching replaces K! permutations); SLSQP hull projection |
| `recovery.py` | Poisson statistics backends (`row_tiled`, `csr_streamed`), chunked Poisson gradient, `refit_A_full_poisson_squarem` |
| `topicscore.py`, `preprocessing.py`, `baselines.py` | sparse/large-matrix paths (truncated SVD above 10M entries, skipped spectrum diagnostic, sparse LDA input); unchanged on small inputs |
| `real_data.py`, `cook_min8_jaccard_v2.py` | What's Cooking v2 corpus (`dataset: cook_v2`) |
| `real_experiment.py` | P1/P3 presets at alpha 0.01 / 0.1, frozen Tran support after row filtering, `lambda_selection_mode` |

## Kept from this repository (the handoff copy lacked or reverted them)

* `GpLSI` forwards `random_state`, graph folds, fold mode and workers on the
  default (P0) path; its defaults stay `legacy_first_three` / 3 workers, so the
  frozen regression fixture still reproduces.
* pp-SPA uses the economy SVD (`full_matrices=False`); the full SVD allocates an
  unused n x n matrix.
* `recover_A` keeps the `A_current` warm start as an option and adds
  `poisson_start="pooled"` (the handoff's behaviour) and `A_full_Pois_SQUAREM`.
* `project_rows_simplex` is bitwise identical to the per-row loop (processed in
  row blocks, as in the handoff); `refit_A_full_l2` keeps its Gram form.
* `real_experiment.fit_spectral_block` keeps the `initialization` argument (the
  handoff hard-coded `weighted_debiased_mean_N_approx`).

Fixed while merging: SVS\* no longer receives SVS's adaptive L selection when it
runs with a fixed L (the handoff runner failed every such fit), and the
`csr_streamed` Poisson backend no longer requires the acceleration package.

## Verification

* The loaded CRC (113,561 x 8), joint spleen (100,840 x 24) and Cooking v2
  (19,017 x 4,911) data have the same hashes as `GpLSI_handoff/canonical_metadata`.
* Same configs run through the handoff runner and the new pipeline (CRC,
  300 rows, 4 preprocessings x 6 hunters x 2 geometries x 2 A recoveries + baselines;
  Cooking v2 alpha 0.01 with the post-Tran row filter): all W and A agree to
  <= 1.8e-15.
* DLPFC section 151673 through the pre-merge ablation runner and the new
  pipeline: W and A agree to float32 storage precision (3e-8) and all
  float64 metrics are identical, except exact-zero-probability counts of the
  two TopicSCORE baselines (the handoff's faster TopicSCORE W refit changes
  values at 1e-16, which moves entries across exactly zero).
* `pytest`: 115 passed.

## Protocol differences (settled 2026-09-30)

The handoff's production runs (now `configs/handoff/base.json`) select the
penalty once (`cv_once`) and use the `weighted_debiased_mean_N_approx`
initialization and a pooled Poisson start; the DLPFC ablation
(`configs/dlpfc/ablation/base.json`) re-selects the penalty at every iteration,
uses the `current` initialization and fixed-L SVS. The paper protocol
(`configs/base.json`, `docs/protocol.md`) follows the handoff on all three
points for every dataset, DLPFC included, and selects SVS\*'s L by its own
stability rule. The ablation configs are unchanged, so their results stay
reproducible.

## Not merged

* The acceleration package (`gplsi_joint_v2.acceleration`: component SSNAL,
  PDHG, disk-backed counts, CUDA) is documented in `GpLSI_handoff/acceleration/`
  but its code is not in the handoff; it lives on Claire's branch
  `codex/gplsi-large-scale-acceleration`.
* The joint-cohort benchmark (`gplsi_joint_v2`), the Midway independent-fit
  benchmark (`gplsi_spatial_benchmark`), the original real-data scripts
  (`codes/run_{crc,spleen,cook}.py`), dashboard builders, one-off audits, and
  submission ledgers remain on `main` / `yeojin-exp` and in the handoff folder.
