#!/bin/bash
# Post-hoc A refit from saved W, one array element per task (see scripts/refit_A.py).
#
#   n=$(python scripts/run_experiment.py configs/dlpfc/ablation/core.json --list | wc -l)
#   sbatch --array=0-$((n-1)) --export=ALL,CONFIG=configs/dlpfc/ablation/core.json scripts/slurm/refit_A.sh
#
# Extra flags (e.g. EXTRA_ARGS="--recovery A_full_Pois_SQUAREM") are passed through.
# For configs split into parts, --list overcounts; surplus elements exit at once.
#SBATCH --job-name=gplsi-refit
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/refit-%A_%a.out
#SBATCH --error=logs/refit-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /opt/conda/etc/profile.d/conda.sh
conda activate gplsi-env

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python scripts/refit_A.py "${CONFIG}" --task "${SLURM_ARRAY_TASK_ID}" ${TASK_STRIDE:+--stride ${TASK_STRIDE}} ${EXTRA_ARGS:-}
