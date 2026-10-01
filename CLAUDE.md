# GpLSI — working notes for Claude (resume point)

Owner: Yeo Jin Jung (advisor: Claire Donnat). Branch: `yeojin-merge` (cleaned, merged
codebase; the pre-merge state is backed up on `yeojin-exp`). This file is only the
resume checklist.

## Goal
One working folder for all GpLSI real-data experiments: CRC, spleen, What's Cooking
(Claire's handoff, `../GpLSI_handoff/`) and Visium DLPFC. The paper's first full pass
follows **`docs/protocol.md`** (fixed 2026-09-30). The DLPFC ablation (design in
`docs/visium_dlpfc_ablation.md`) is a separate, earlier study.

## Layout (README "Repository layout"; details in configs/README.md, data/README.md)
- `src/gplsi/` estimator; `src/gplsi/pipeline/` the single experiment pipeline.
- `scripts/run_experiment.py CONFIG [--task i]`, `scripts/refit_A.py`, `scripts/summarize.py`;
  `scripts/{data,analysis,simulations,slurm}/`.
- `configs/base.json` = paper protocol; `configs/<dataset>/production.json` + `smoke.json`
  (DLPFC twice: `production.json` = 2,000-gene dispersion panel, `production_tran.json` = Tran α = 0.1;
  MERFISH and Xenium: one model per animal / patient-timepoint, K = 12, wide λ grid;
  data and experiment notes in `docs/merfish_xenium.md`);
  `configs/dlpfc/ablation/` = DLPFC ablation; `configs/handoff/` = Claire's 2026-09-15 designs.
- Analyses: `scripts/analysis/` (shared helpers `shared.py`; per dataset subfolders).
- `data/{crc,spleen,cook,dlpfc}/`; dlpfc h5ad and Cooking v2 are built, not tracked.

## Status as of 2026-09-30
- Merge + protocol committed and pushed to `origin/yeojin-merge` (2026-09-30), MERFISH +
  Xenium added 2026-10-01 (data prep, configs, cell-type neighbourhoods for Xenium, maps,
  plaque and disease tasks); not yet merged into `main`. Ask before committing; never push
  without asking. `docs/GPLSI_experiments.pdf` stays untracked (user's choice).
- `data/spleen/dataset/compartments/` is deliberately untracked (user undecided);
  build it on each machine (see Cluster setup).
- Verified: data hashes match the handoff; new pipeline = handoff runner (W/A ≤ 1.8e-15)
  and = pre-merge DLPFC runner (float32 storage precision); pytest 115 passed.
  Record of what was ported/fixed: `docs/handoff_merge.md`.
- Protocol configs written and smoke-tested locally on all four datasets (2026-09-30);
  pytest 130 passed. **The user submits production runs on the cluster themselves — do
  not submit jobs.** Word-frequency diagnostics (full data) are in
  `results/<dataset>_production/figures/word_frequency/`.
- Existing DLPFC ablation results (on the cluster, `results/visium_dlpfc/`) are in the old
  format; the ablation scripts read the new format (`scripts/analysis/dlpfc/records.py`).

## Pending decisions (ask the user)
1. After the first pass: thresholding/weighting (P1–P3), mainly for Cooking (Tran keeps
   1,333/4,911 at α = 0.005); spleen and DLPFC are heterogeneous but Tran keeps ~all.
   Held-out deviance: protocol uses the ε = 1e-3 uniform-smoothed deviance (exact one is
   infinite when A has exact zeros); ε = 1e-4/1e-2 recorded as sensitivity.
2. Claire's acceleration code (`gplsi_joint_v2.acceleration`) is not in the handoff; it is
   on her branch `codex/gplsi-large-scale-acceleration`. Integrate later?
3. DLPFC items from `docs/visium_dlpfc_results.md` §7 (thresholding axis, KE weighting,
   hunter subset for `configs/dlpfc/ablation/extended.json`, fixed-λ sweep).

## Cluster setup (DSI: `ssh login.ds`, project dir `/net/projects2/mercury/yeojin`)
```sh
git clone -b yeojin-merge <repo-url> GpLSI && cd GpLSI
conda env create -f environment.yaml && conda activate gplsi-env   # needs suitesparse/scikit-sparse
python -m pip install -e '.[dev,benchmark]'
python scripts/data/prepare_dlpfc.py --download      # 47,681 spots x 33,538 genes
python scripts/data/prepare_cook_v2.py               # What's Cooking v2 corpus
python scripts/data/prepare_merfish.py --download   # 5.5 GB source; 8 parallel ranges (slow server)
python scripts/data/prepare_xenium.py --download    # 1.5 GB source
curl -sLO https://github.com/huBioinfo/CytoCommunity/raw/main/CODEX_SpleenDataset.zip
python scripts/data/prepare_spleen_compartment_annotations.py --archive CODEX_SpleenDataset.zip  # spleen labels
n=$(python scripts/run_experiment.py configs/dlpfc/smoke.json --list | wc -l)
sbatch --array=0-$((n-1)) --export=ALL,CONFIG=configs/dlpfc/smoke.json scripts/slurm/run_experiment.sh
# full pass: configs/<dataset>/production.json; run the spatial_lda part one task per job
```
`scripts/slurm/*.sh` are set up for DSI (partition `general`, account `general_group`,
conda `gplsi-env`, logs in `logs/`). **User job limit: 10 concurrent jobs.** Pack tasks
with `--export=...,TASK_STRIDE=S --array=0-$((S-1))`.

## Gotchas
- Scripts that use multiprocessing need an `if __name__ == "__main__"` guard (macOS
  spawn re-imports the script; graph-SVD CV uses a Pool).
- Keep BLAS single-threaded per task (launchers set `OMP/OPENBLAS/MKL_NUM_THREADS=1`).
- Result rows are cached by data + settings + source-code hash: any edit under
  `src/gplsi/` makes reruns recompute.
- `GpLSI` class defaults stay `legacy_first_three` folds / 3 workers (frozen regression
  fixture); the pipeline sets `cv_fold_mode="all"` explicitly.
- Default penalty grid top is 1e-4·1.2^28 ≈ 0.0165. On DLPFC, P0's λ is interior
  (0.011–0.014) at p = 2,000; P2/P3 and panels with p ≤ 1,000 hit the top (optimum
  λ ≈ 0.10–0.21, see `lambda_wide`, `p2_wide`).
- Br5595 sections (151669–151672) have only L3–L6 + WM.
- Adaptive (MixedSCORE) SVS is unavailable at K ≥ 8; use SVS* `svs_star_stability`.
- `results/`, `logs/`, `figures/`, `data/dlpfc/*`, `data/cook/dataset/raw_*/` are gitignored.
