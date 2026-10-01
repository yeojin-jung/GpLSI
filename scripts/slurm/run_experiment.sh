#!/bin/bash
# Slurm array over the tasks of one experiment config (DSI cluster defaults:
# partition `general`, account `general_group`, 12h cap, 10 concurrent jobs).
#
#   n=$(python scripts/run_experiment.py configs/crc/production.json --list | wc -l)
#   sbatch --array=0-$((n-1)) --export=ALL,CONFIG=configs/crc/production.json scripts/slurm/run_experiment.sh
#
# With TASK_STRIDE=S, array element i runs tasks i, i+S, i+2S, ... in turn, so
# `--array=0-$((S-1))` covers any number of tasks within the job limit. Tasks
# are listed part by part (--list); TASK_FIRST and TASK_LAST (inclusive) limit a
# submission to one index range, e.g. one part packed into 10 jobs:
#   sbatch --array=0-9 --export=ALL,CONFIG=...,TASK_FIRST=40,TASK_LAST=79,TASK_STRIDE=10 ...
# Keep the slow spatial_lda part to about one task per job (docs/protocol.md, 1.8).
#SBATCH --job-name=gplsi
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /opt/conda/etc/profile.d/conda.sh
conda activate gplsi-env

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# One BLAS thread per task; the allocated CPUs go to the graph-CV fold pool.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export GPLSI_GRAPH_N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

SCRIPT="${SCRIPT:-scripts/run_experiment.py}"
n_tasks=$(python "${SCRIPT}" "${CONFIG}" --list | wc -l)
first="${TASK_FIRST:-0}"
last="${TASK_LAST:-$((n_tasks-1))}"
(( last > n_tasks-1 )) && last=$((n_tasks-1))
stride="${TASK_STRIDE:-${n_tasks}}"
status=0
for (( index=first+SLURM_ARRAY_TASK_ID; index<=last; index+=stride )); do
  echo "=== task ${index} of ${CONFIG} ($(date '+%F %T'))"
  python "${SCRIPT}" "${CONFIG}" --task "${index}" ${EXTRA_ARGS:-} || {
    echo "!!! task ${index} failed with exit $?" >&2; status=1; }
done
exit "${status}"
