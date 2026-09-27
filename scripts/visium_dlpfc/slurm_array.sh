#!/bin/bash
# Slurm array over one DLPFC task manifest. Set account/partition for your cluster:
#   sbatch --array=0-$(( $(wc -l < configs/visium_dlpfc/tasks_core.csv) - 2 )) \
#          --export=ALL,TASKS=configs/visium_dlpfc/tasks_core.csv scripts/visium_dlpfc/slurm_array.sh
#SBATCH --job-name=gplsi-dlpfc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --output=logs/visium_dlpfc/slurm-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GPLSI_GRAPH_N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
python -m gplsi_spatial_benchmark.ablation_runner \
  --config "${CONFIG:-configs/visium_dlpfc/ablation.json}" --tasks "${TASKS}" --index "${SLURM_ARRAY_TASK_ID}"
