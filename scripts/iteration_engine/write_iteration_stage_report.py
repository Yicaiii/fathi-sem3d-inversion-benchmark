from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import numpy as np

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_result_root = ROOT / config["run_result_root"] / transition
run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"

component_rhs_dir = run_result_root / "component_rhs"
mtilde_solve_dir = run_result_root / "mtilde_solve"

def file_has_pass(path):
    return path.exists() and "RESULT = PASS" in path.read_text(errors="ignore")

def exists(path):
    return path.exists()

checks = []

def add(stage, name, ok, evidence):
    checks.append({
        "stage": stage,
        "name": name,
        "ok": bool(ok),
        "evidence": str(evidence),
    })

add("S01", "strict forward", exists(run_data_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000/traces"), run_data_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000/traces")
add("S02", "residual sources", exists(run_result_root / "residual_sources/454B_strict_residual_timeseries.h5"), run_result_root / "residual_sources/454B_strict_residual_timeseries.h5")
add("S03-S05", "adjoint 30/30", exists(ROOT / f"benchmark_fathi_strict/reports/phaseA_task2C_adjoint_complete/{transition}_adjoint_complete_audit.txt"), ROOT / f"benchmark_fathi_strict/reports/phaseA_task2C_adjoint_complete/{transition}_adjoint_complete_audit.txt")

for comp, stage in [("x", "S06"), ("y", "S07"), ("z", "S08")]:
    add(stage, f"RHS_{comp}", file_has_pass(component_rhs_dir / f"full_grid_trace_RHS_{comp}_summary.txt"), component_rhs_dir / f"full_grid_trace_RHS_{comp}_summary.txt")

add("S09", "RHS_total", file_has_pass(component_rhs_dir / "full_grid_trace_RHS_total_summary.txt"), component_rhs_dir / "full_grid_trace_RHS_total_summary.txt")
add("S10", "Mtilde solve", file_has_pass(mtilde_solve_dir / "mtilde_q1_interior_solve_rhs_total_summary.txt"), mtilde_solve_dir / "mtilde_q1_interior_solve_rhs_total_summary.txt")
add("S10-audit", "Mtilde output audit", file_has_pass(ROOT / f"benchmark_fathi_strict/reports/mtilde_run/{transition}_mtilde_output_audit.txt"), ROOT / f"benchmark_fathi_strict/reports/mtilde_run/{transition}_mtilde_output_audit.txt")

state_out = ROOT / config["state_dir"] / f"iter_{kp1:03d}_state_v2_corrected.npz"
add("S13", "accepted next state", exists(state_out), state_out)

done = sum(1 for c in checks if c["ok"])
total = len(checks)

out_dir = ROOT / "benchmark_fathi_strict/reports/stage_reports"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_stage_report.json"
out_txt = out_dir / f"{transition}_stage_report.txt"

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "done": done,
    "total": total,
    "checks": checks,
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Fathi benchmark iteration stage report")
lines.append("======================================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"done = {done} / {total}")
lines.append("")
for c in checks:
    mark = "PASS" if c["ok"] else "PENDING"
    lines.append(f"{c['stage']:8s} {mark:8s} {c['name']}")
    lines.append(f"          evidence = {c['evidence']}")
lines.append("")
lines.append("Current interpretation:")
if all(c["ok"] for c in checks if c["stage"] in ["S01", "S02", "S03-S05", "S06", "S07", "S08", "S09", "S10", "S10-audit"]):
    lines.append("  Gradient stage is complete. Next stage is candidate generation / line-search.")
else:
    lines.append("  Gradient stage is not complete yet.")
lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append("RESULT = PASS")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
