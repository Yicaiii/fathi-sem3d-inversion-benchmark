#!/usr/bin/env bash
set -euo pipefail

ROOT="${FATHI_BENCHMARK_ROOT:-$HOME/sem3d_fathi_clean}"
cd "$ROOT"

ITER_K=""
EXECUTE=false
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iter-k)
      ITER_K="$2"
      shift 2
      ;;
    --execute)
      EXECUTE=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --config)
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "RESULT = FAIL_BAD_ARGUMENT"
      exit 1
      ;;
  esac
done

if [[ "$ITER_K" != "7" ]]; then
  echo "Executable task: residual_generation legacy adapter"
  echo "=================================================="
  echo
  echo "This adapter currently delegates to validated iter_007_to_iter_008 scripts only."
  echo "ITER_K = $ITER_K"
  echo
  echo "RESULT = FAIL_UNSUPPORTED_ITER"
  exit 1
fi

TRANSITION="iter_007_to_iter_008"
RESIDUAL_DIR="$ROOT/results/fathi_loop_v2/$TRANSITION/residual_sources"
SUMMARY_TXT="$RESIDUAL_DIR/454B_strict_residual_timeseries_summary.txt"
H5_OUT="$RESIDUAL_DIR/454B_strict_residual_timeseries.h5"

SCRIPT_A="scripts/fathi_loop_v2/454A_compute_strict_forward_residual_manifest.py"
SCRIPT_B="scripts/fathi_loop_v2/454B_build_strict_residual_timeseries_h5.py"

REPORT_DIR="$ROOT/benchmark_fathi_strict/reports/executable_tasks"
mkdir -p "$REPORT_DIR"
REPORT_TXT="$REPORT_DIR/${TRANSITION}_residual_generation_task.txt"

{
  echo "Executable task: residual_generation"
  echo "===================================="
  echo
  echo "wrapper_mode = legacy_adapter_to_validated_454A_454B"
  echo "transition = $TRANSITION"
  echo "execute = $EXECUTE"
  echo "force = $FORCE"
  echo
  echo "old_scripts:"
  echo "  $SCRIPT_A"
  echo "  $SCRIPT_B"
  echo
  echo "h5_out = $H5_OUT"
  echo "summary_txt = $SUMMARY_TXT"
  echo

  if [[ -f "$H5_OUT" && -f "$SUMMARY_TXT" && "$FORCE" == "false" ]]; then
    if grep -q "RESULT = PASS" "$SUMMARY_TXT"; then
      echo "Existing residual output is already PASS."
      grep -E "global_J|total_J|common_position_count|ok_residual_count|ok_receiver_count|bad_count|RESULT" "$SUMMARY_TXT" || true
      echo
      echo "RESULT = PASS_ALREADY_EXISTS"
      exit 0
    fi
  fi

  if [[ "$EXECUTE" == "false" ]]; then
    echo "Plan only. Would run:"
    echo "  python3 $SCRIPT_A"
    echo "  python3 $SCRIPT_B"
    echo
    echo "RESULT = PASS_PLAN_ONLY"
    exit 0
  fi

  echo "Running validated old residual scripts..."
  echo
  echo ">>> python3 $SCRIPT_A"
  python3 "$SCRIPT_A"
  echo
  echo ">>> python3 $SCRIPT_B"
  python3 "$SCRIPT_B"
  echo

  if [[ -f "$H5_OUT" && -f "$SUMMARY_TXT" ]] && grep -q "RESULT = PASS" "$SUMMARY_TXT"; then
    echo "Residual generation completed using validated old scripts."
    grep -E "global_J|total_J|common_position_count|ok_residual_count|ok_receiver_count|bad_count|RESULT" "$SUMMARY_TXT" || true
    echo
    echo "RESULT = PASS_EXECUTED"
    exit 0
  fi

  echo "Residual scripts ran, but final summary is not PASS."
  echo
  echo "RESULT = FAIL_OLD_RESIDUAL_OUTPUT_NOT_PASS"
  exit 1
} | tee "$REPORT_TXT"
