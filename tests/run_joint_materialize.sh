#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${GPLSI_BENCHMARK_ROOT:-$SOURCE}"
PYTHON="${PYTHON:-python}"
export GPLSI_BENCHMARK_ROOT="$ROOT"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$ROOT/.cache/joint_v2"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
DATASET_INDEX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
if [[ ! "$DATASET_INDEX" =~ ^[0-2]$ ]]; then
    echo "Dataset index must be 0 (Visium), 1 (MERFISH), or 2 (Xenium)." >&2
    exit 2
fi
export TMPDIR="$ROOT/data/interim/joint_v2/materialize_tmp_${SLURM_ARRAY_JOB_ID:-manual}_${DATASET_INDEX}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
datasets=(visium_dlpfc merfish_trem2_5xfad xenium_uc)
"$PYTHON" -m gplsi_joint_v2.cli materialize --root "$ROOT" --dataset "${datasets[$DATASET_INDEX]}"
