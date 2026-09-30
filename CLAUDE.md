# GpLSI — working notes for Claude (resume point)

Owner: Yeo Jin Jung (advisor: Claire Donnat). Branch: `yeojin-exp`. Read
`docs/visium_dlpfc_ablation.md` first — it is the full design, method/formula/code
map, metrics, diagnostics, and plan. This file is only the resume checklist.

## Goal
Ablate GpLSI on the Visium DLPFC dataset for transcriptomics practitioners, along
three dimensions: (A) topic-gene estimator `A_current` / `A_full_L2` / `A_full_Pois`,
(B) vertex hunting (SPA, SVS, SVS*, pp-SPA, PALM, accel. PALM), (C) gene
thresholding (panel size 500–5000; P0–P3 preprocessing). Advisor meeting ~2026-09-28/29.

## Status as of 2026-09-28 (end of third session)
**Resume here:** read `docs/visium_dlpfc_results.md` §0 (summary) and §7 (what is left),
then `docs/visium_dlpfc_session_log.md`.
- **All pre-meeting designs are done**, with no jobs running: smoke, core, lambda_wide,
  lambda_wide_tsgd, panel, p2_wide (P2 on the wide λ grid), k5_br5595 (K = 5 on 151669,
  job 1939616, §3.15), and the Poisson refits for core, panel, lambda_wide and p2_wide.
  k5_br5595 has no Poisson refit.
- `summarize.py` has been run on all designs (63 tasks, 1,767 records); its output is in
  `results/visium_dlpfc/summary/`. Full pytest: 306 passed (session 2; only
  `tests/test_dlpfc_ablation.py` was rerun in session 3: 6 passed).
- Figures are in `results/visium_dlpfc/diagnostics/{core,panel,p2_wide,k5_br5595}/`, made by
  `plot_core_diagnostics.py`, `plot_spatial_maps.py`, `plot_design_maps.py` (panel,
  p2_wide, k5_br5595) and `plot_topics.py --design ...`. The last three need the h5ad, so
  run them via `srun`.
- PDF figure report: `make_report_pdf.py` → `results/visium_dlpfc/report/` (rerun after
  `summarize.py`). Per-topic A-estimator comparison on a shared W:
  `compare_A_estimators.py [--geometry ...]` → `diagnostics/core/A_compare/`.
- All changes are **uncommitted** on `yeojin-exp`. Ask before committing; never push
  without asking.

## Decisions made
1. Spatial-LDA stays in `core`.
2. Poisson refit: pooled start, max_iter 1500, tolerance 1e-8; report the gap as
   "near-converged".
3. The P2 wide-grid rerun was done (p2_wide): KE weighting stays below P0 even at its
   optimal λ.

## Pending decisions (for the advisor meeting; see results doc §7)
Thresholding axis (empty at α = 0.005); keep or drop KE weighting; hunter subset for
the extended grid; A_full_Pois cost and pseudocount; fixed-λ sweep. (K = 5 for Br5595
was run: not a fix, see results doc §3.15.)

## Next steps
1. Commit when the user asks.
2. After the meeting: `tasks_extended.csv` (360 tasks; `TASK_STRIDE` packing; `grid_len`
   50 for P2/P3 and for p ≤ 1,000).

## Cluster setup (DSI: `ssh login.ds`, project dir `/net/projects2/mercury/yeojin`)
```sh
git clone -b yeojin-exp <repo-url> GpLSI && cd GpLSI
conda env create -f environment.yaml && conda activate gplsi-env   # needs suitesparse/scikit-sparse
python -m pip install -e '.[dev,benchmark]'
python scripts/visium_dlpfc/prepare_data.py --download   # 47,681 spots x 33,538 genes
python scripts/visium_dlpfc/make_tasks.py
mkdir -p logs/visium_dlpfc
sbatch --array=0-0 --export=ALL,TASKS=configs/visium_dlpfc/tasks_smoke.csv scripts/visium_dlpfc/slurm_array.sh
```
`slurm_array.sh` is set up for DSI (partition `general`, account `general_group`, conda
`gplsi-env`, logs in `logs/visium_dlpfc/`). **User job limit: 10 concurrent jobs.** Pack
tasks with `--export=...,TASK_STRIDE=S --array=0-$((S-1))`. The Poisson refit job script
is `scripts/visium_dlpfc/slurm_refit.sh`.

## Gotchas
- Scripts that use multiprocessing need an `if __name__ == "__main__"` guard (macOS
  spawn re-imports the script; graph-SVD CV uses a Pool).
- Keep BLAS single-threaded per task (launchers set `OMP/OPENBLAS/MKL_NUM_THREADS=1`).
- Do not modify `src/gplsi_spatial_benchmark/runner.py` (Claire's Midway pipeline);
  DLPFC work uses `ablation_runner.py`.
- `src/gplsi/recovery.py` has two speed-only changes (vectorized
  `project_rows_simplex`; Gram-form `refit_A_full_l2`); Poisson solver unchanged.
  Tell Claire.
- Default penalty grid top is 1e-4·1.2^28 ≈ 0.0165. P0's λ is interior (0.011–0.014) at
  p = 2,000. P2/P3 hit the top, and their true optimum is λ ≈ 0.10–0.21 (lambda_wide,
  p2_wide). Panels with p ≤ 1,000 also hit the top.
- Br5595 sections (151669–151672) have only L3–L6 + WM.
- `results/`, `logs/`, `data/processed/`, `data/interim/` are gitignored.
