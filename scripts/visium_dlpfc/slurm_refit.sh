#!/bin/bash
# A_full_Pois refit from saved W, one worker per task JSON (DSI cluster).
#   sbatch --export=ALL,DESIGNS="core" scripts/visium_dlpfc/slurm_refit.sh
#SBATCH --job-name=gplsi-refit
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/visium_dlpfc/refit-%j.out
#SBATCH --error=logs/visium_dlpfc/refit-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /opt/conda/etc/profile.d/conda.sh
conda activate gplsi-env
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/visium_dlpfc/refit_poisson.py --designs ${DESIGNS:-core} --workers "${SLURM_CPUS_PER_TASK:-1}"
