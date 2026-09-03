#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <parent-iteration-k> [extra run_current_iteration.py options...]" >&2
  exit 2
fi

K="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FATHI_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"

export FATHI_BENCHMARK_ROOT="$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

RUN="fathi_s43_repro_p20_t052"
CHILD=$((K + 1))
TRANS="$(printf 'iter_%03d_to_iter_%03d' "$K" "$CHILD")"
LOG="$REPO/results/$RUN/$TRANS/current_iteration_driver.log"
mkdir -p "$(dirname "$LOG")"

printf 'RUN=%s\nPARENT_ITERATION=%s\nTRANSITION=%s\nLOG=%s\n' "$RUN" "$K" "$TRANS" "$LOG"

python -m scripts.fathi_benchmark.run_current_iteration \
  --repo "$REPO" \
  --parent-iteration "$K" \
  "$@" 2>&1 | tee -a "$LOG"

exit "${PIPESTATUS[0]}"
