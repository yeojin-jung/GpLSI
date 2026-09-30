#!/bin/bash
# Slurm array over one DLPFC task manifest (DSI cluster: partition `general`,
# account `general_group`; 12h partition cap).
#   sbatch --array=0-$(( $(wc -l < configs/visium_dlpfc/tasks_core.csv) - 2 )) \
#          --export=ALL,TASKS=configs/visium_dlpfc/tasks_core.csv scripts/visium_dlpfc/slurm_array.sh
#SBATCH --job-name=gplsi-dlpfc
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/visium_dlpfc/slurm-%A_%a.out
#SBATCH --error=logs/visium_dlpfc/slurm-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/visium_dlpfc

source /opt/conda/etc/profile.d/conda.sh
conda activate gplsi-env

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# One BLAS thread per task; the allocated CPUs go to the graph-SVD CV fold pool.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export GPLSI_GRAPH_N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

# TASK_STRIDE=S (optional) makes array element i run tasks i, i+S, i+2S, ... sequentially,
# so a manifest larger than the job limit fits in S jobs (--array=0-$((S-1))).
n_tasks=$(( $(wc -l < "${TASKS}") - 1 ))
stride="${TASK_STRIDE:-$n_tasks}"
status=0
for (( index=SLURM_ARRAY_TASK_ID; index<n_tasks; index+=stride )); do
  echo "=== task ${index} of ${TASKS} ($(date '+%F %T'))"
  python -m gplsi_spatial_benchmark.ablation_runner \
    --config "${CONFIG:-configs/visium_dlpfc/ablation.json}" --tasks "${TASKS}" --index "${index}" || {
      echo "!!! task ${index} failed with exit $?" >&2; status=1; }
done
exit "${status}"
