#!/bin/bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ACTION [--root PATH] [arguments...]" >&2
  exit 2
fi
GPLSI_ACTION="$1"
shift
GPLSI_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      if [[ $# -lt 2 ]]; then echo "--root requires a path" >&2; exit 2; fi
      export GPLSI_BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --root=*) export GPLSI_BENCHMARK_ROOT="${1#--root=}"; shift ;;
    *) GPLSI_ARGS+=("$1"); shift ;;
  esac
done
source "$(dirname "${BASH_SOURCE[0]}")/job_environment.sh"
GPLSI_PYTHON="${PYTHON:-python}"
case "$GPLSI_ACTION" in
  preflight|controller|launch)
    exec "$GPLSI_PYTHON" -m "gplsi_joint_v2.$GPLSI_ACTION" --root "$GPLSI_BENCHMARK_ROOT" "${GPLSI_ARGS[@]}" ;;
  array)
    exec "$GPLSI_PYTHON" -m gplsi_joint_v2.scheduler run-array --root "$GPLSI_BENCHMARK_ROOT" "${GPLSI_ARGS[@]}" ;;
  *)
    exec "$GPLSI_PYTHON" -m gplsi_joint_v2.cli "$GPLSI_ACTION" --root "$GPLSI_BENCHMARK_ROOT" "${GPLSI_ARGS[@]}" ;;
esac
