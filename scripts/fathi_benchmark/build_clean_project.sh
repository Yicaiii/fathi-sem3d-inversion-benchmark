#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 MANIFEST OUTPUT_DIR [ARCHIVE.tar.gz]" >&2
  exit 2
fi

MANIFEST="$1"
OUTPUT="$2"
ARCHIVE="${3:-}"
SOURCE_REPO="${FATHI_PRODUCTION_GIT_WORKTREE:-$HOME/fathi-benchmark-current-upload-20260902}"

ARGS=(
  --source-repo "$SOURCE_REPO"
  --manifest "$MANIFEST"
  --output "$OUTPUT"
)

if [ -n "$ARCHIVE" ]; then
  ARGS+=(--archive "$ARCHIVE")
fi

exec python "$SOURCE_REPO/scripts/fathi_benchmark/build_clean_project.py" "${ARGS[@]}"
