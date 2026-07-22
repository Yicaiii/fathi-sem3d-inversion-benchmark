from pathlib import Path
import os
from datetime import datetime
import json

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

OUT = ROOT / "benchmark_fathi_strict/reports/dashboard"
OUT.mkdir(parents=True, exist_ok=True)

def read_tail(path, n=80):
    if not path.exists():
        return [f"MISSING: {path}"]
    return path.read_text(errors="ignore").splitlines()[-n:]

files = {
    "official_status": ROOT / "benchmark_fathi_strict/reports/official_benchmark_status.txt",
    "inventory": ROOT / "benchmark_fathi_strict/reports/inventory/benchmark_inventory_fast_summary.txt",
    "forward": ROOT / "benchmark_fathi_strict/reports/generic_runs/iter_007_to_iter_008_forward_run_iteration.txt",
    "residual": ROOT / "benchmark_fathi_strict/reports/generic_runs/iter_007_to_iter_008_residual_run_iteration.txt",
    "prepare_adjoint": ROOT / "benchmark_fathi_strict/reports/generic_runs/iter_007_to_iter_008_prepare_adjoint_run_iteration.txt",
    "run_adjoint": ROOT / "benchmark_fathi_strict/reports/generic_runs/iter_007_to_iter_008_run_adjoint_run_iteration.txt",
    "resume": ROOT / "benchmark_fathi_strict/reports/resume/iter_007_to_iter_008_resume_plan.txt",
}

created = datetime.now().isoformat()

txt_out = OUT / "benchmark_dashboard.txt"
json_out = OUT / "benchmark_dashboard.json"

payload = {
    "created": created,
    "benchmark_name": "Fathi layered elastic inversion benchmark",
    "current_completed_state": "iter_007",
    "current_active_transition": "iter_007_to_iter_008",
    "current_stage": "S03-S05 adjoint execution",
    "adjoint_progress": "1/30",
    "next_heavy_command": "python3 scripts/fathi_loop_v2/456A_run_strict_adjoint_batch.py --component x --batch batch_001 --execute",
    "generic_engine_status": {
        "forward": "connected",
        "residual": "connected",
        "prepare_adjoint": "connected",
        "run_adjoint": "connected status-only",
        "rhs": "not connected",
        "mtilde": "not connected",
        "candidates": "not connected",
        "accept": "not connected",
    },
    "files": {k: str(v) for k, v in files.items()},
}

json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("FATHI BENCHMARK DASHBOARD")
lines.append("=========================")
lines.append("")
lines.append(f"created = {created}")
lines.append("")
lines.append("Benchmark identity:")
lines.append("  This is the full Fathi layered elastic inversion benchmark.")
lines.append("  The unit of automation is a generic strict iteration engine:")
lines.append("    state_k -> forward -> residual -> adjoint -> RHS -> Mtilde -> candidates -> accept -> state_{k+1}")
lines.append("")
lines.append("Current official state:")
lines.append("  last completed reusable state = iter_007")
lines.append("  current active transition = iter_007_to_iter_008")
lines.append("  current transition status = IN_PROGRESS_STRICT")
lines.append("  current resume point = S03-S05 adjoint execution")
lines.append("  adjoint progress = 1/30")
lines.append("")
lines.append("Generic engine connection status:")
for k, v in payload["generic_engine_status"].items():
    lines.append(f"  {k}: {v}")
lines.append("")
lines.append("Next heavy command to continue later:")
lines.append(f"  {payload['next_heavy_command']}")
lines.append("")
lines.append("Important rules:")
lines.append("  Do not create new one-off code for each iteration.")
lines.append("  New iteration code must accept --iter-k and derive paths automatically.")
lines.append("  Existing heavy outputs are protected and should not be moved manually.")
lines.append("  Only accept stage is allowed to create state_{k+1}.")
lines.append("")
lines.append("Protected current data:")
lines.append("  results/fathi_loop_v2/states_corrected/iter_007_state_v2_corrected.npz")
lines.append("  data/inversion_linear/iter_007/accepted")
lines.append("  data/inversion_linear/iter_008/forward_dudx_mgcap_full_batches/strict_full_forward_000/traces")
lines.append("  results/fathi_loop_v2/iter_007_to_iter_008/residual_sources")
lines.append("  data/inversion_linear/iter_008/adjoint_full_grid_batches")
lines.append("  results/fathi_loop_v2/iter_007_to_iter_008/adjoint_runs")
lines.append("")
lines.append("Summary tails:")
for name, path in files.items():
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{name}: {path}")
    lines.append("-" * 80)
    lines.extend(read_tail(path, n=50))
lines.append("")
lines.append("RESULT = PASS")

txt_out.write_text("\n".join(lines), encoding="utf-8")

print(f"txt = {txt_out}")
print(f"json = {json_out}")
print("RESULT = PASS")
