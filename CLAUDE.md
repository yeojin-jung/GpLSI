# GpLSI — working notes for Claude (resume point)

Owner: Yeo Jin Jung (advisor: Claire Donnat). Branch: `yeojin-exp`. Read
`docs/visium_dlpfc_ablation.md` first — it is the full design, method/formula/code
map, metrics, diagnostics, and plan. This file is only the resume checklist.

## Goal
Ablate GpLSI on the Visium DLPFC dataset for transcriptomics practitioners, along
three dimensions: (A) topic-gene estimator `A_current` / `A_full_L2` / `A_full_Pois`,
(B) vertex hunting (SPA, SVS, SVS*, pp-SPA, PALM, accel. PALM), (C) gene
thresholding (panel size 500–5000; P0–P3 preprocessing). Advisor meeting ~2026-09-28/29.

## Status as of 2026-09-27
- Done: data prep, ablation runner, configs, task manifests, local + Slurm launchers,
  summarizer, Poisson post-hoc refit script, tests (305 pass), diagnostics on 151673.
- **No ablation results yet.** Nothing launched beyond diagnostics.
- Untested on real output: `scripts/visium_dlpfc/summarize.py`,
  `scripts/visium_dlpfc/refit_poisson.py` (its `poisson_refit` settings block —
  `initial`, `interior_mass`, `max_iter`, `tolerance` — is not yet in the config).

## Pending decisions (ask the user before launching)
1. Drop `spatial_lda` from the `core` design? (unmeasured; >15 min per fit)
2. Poisson refit settings. Default EM stalls (gap 2.3e-3 after 2000 its vs tol 1e-8).
   Proposal: start from pooled frequencies (`initial: "pooled"`) or `A_current` with
   `interior_mass: 0.01`, `max_iter: 1500`, report gap as "near-converged".

## Next steps
1. Cluster setup (below), then `pytest -q` on a compute node.
2. Smoke: `tasks_smoke.csv` (1 task) via Slurm; check `results/visium_dlpfc/smoke/`.
3. core (9) → lambda_wide (3) → panel (36); then `refit_poisson.py`; then `summarize.py`.
4. After the meeting: `tasks_extended.csv` (360).

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
`slurm_array.sh` has no partition/account and no conda activation yet — set them for
the DSI cluster (check `sinfo`, ask the user which partition/account).

## Gotchas
- Scripts that use multiprocessing need an `if __name__ == "__main__"` guard (macOS
  spawn re-imports the script; graph-SVD CV uses a Pool).
- Keep BLAS single-threaded per task (launchers set `OMP/OPENBLAS/MKL_NUM_THREADS=1`).
- Do not modify `src/gplsi_spatial_benchmark/runner.py` (Claire's Midway pipeline);
  DLPFC work uses `ablation_runner.py`.
- `src/gplsi/recovery.py` has two speed-only changes (vectorized
  `project_rows_simplex`; Gram-form `refit_A_full_l2`); Poisson solver unchanged.
  Tell Claire.
- Default penalty grid top is 1e-4·1.2^28 ≈ 0.0165 (not 0.024). On 151673/P0 the chosen
  λ was 0.0114 (interior).
- Br5595 sections (151669–151672) have only L3–L6 + WM.
- `results/`, `logs/`, `data/processed/`, `data/interim/` are gitignored.
