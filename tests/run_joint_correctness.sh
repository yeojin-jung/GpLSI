#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${GPLSI_BENCHMARK_ROOT:-$SOURCE}"
PYTHON="${PYTHON:-python}"
export GPLSI_BENCHMARK_ROOT="$ROOT"
export XDG_CACHE_HOME="$ROOT/.cache/joint_v2"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export R_USER_CACHE_DIR="$XDG_CACHE_HOME/R"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TMPDIR="$ROOT/data/interim/joint_v2/test_tmp_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$R_USER_CACHE_DIR" "$PIP_CACHE_DIR" "$ROOT/reports/joint_v2"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$SOURCE"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
from gplsi_joint_v2.artifacts import atomic_json, sha256_file
from gplsi_joint_v2.cli import source_identity
from gplsi_joint_v2.config import default_config, fingerprint
root = Path(os.environ["GPLSI_BENCHMARK_ROOT"])
job = os.environ.get("SLURM_JOB_ID", "manual")
atomic_json(root / "reports/joint_v2" / f"correctness_{job}_source.json", {
    "source": source_identity(),
    "config_hash": fingerprint(default_config()),
    "job_id": job,
    "tests": {str(path): sha256_file(path) for path in sorted(Path("tests").glob("test_joint_*.py"))},
    "poisson_repair_tests_sha256": sha256_file("tests/test_poisson_recovery.py"),
})
PY
"$PYTHON" -m pytest tests/test_poisson_recovery.py tests/test_joint_*.py -q --junitxml="$ROOT/reports/joint_v2/correctness_${SLURM_JOB_ID:-manual}.xml"
