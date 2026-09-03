#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 PARENT_ITERATION" >&2
    exit 2
fi

K="$1"

REPO="${FATHI_BENCHMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

exec python \
    "$REPO/scripts/fathi_benchmark/audit_current_iteration.py" \
    --repo "$REPO" \
    --run "fathi_s43_repro_p20_t052" \
    --parent-iteration "$K"
