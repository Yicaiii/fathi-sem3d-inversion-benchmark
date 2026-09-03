#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-${FATHI_PRODUCTION_GIT_WORKTREE:-$HOME/fathi-benchmark-current-upload-20260902}}"
OUTPUT="${2:?usage: $0 [REPO] OUTPUT_MANIFEST}"

exec python "$REPO/scripts/fathi_benchmark/build_production_source_manifest.py" \
  --repo "$REPO" \
  --output "$OUTPUT"
