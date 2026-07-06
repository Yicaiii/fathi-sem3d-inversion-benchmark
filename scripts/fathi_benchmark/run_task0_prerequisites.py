from pathlib import Path
from datetime import datetime
import argparse
import json
import sys
import os

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument(
    "--stage",
    choices=[
        "plan",
        "strict_forward",
        "residual",
        "prepare_adjoint",
        "adjoint",
        "audit_adjoint",
        "all",
    ],
    default="plan",
)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"
run_result_root = ROOT / config["run_result_root"] / transition

strict_forward_dir = run_data_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000"
strict_forward_traces = strict_forward_dir / "traces"

residual_h5 = run_result_root / "residual_sources/454B_strict_residual_timeseries.h5"
residual_summary_txt = run_result_root / "residual_sources/454B_strict_residual_timeseries_summary.txt"

adjoint_base = run_data_root / "adjoint_full_grid_batches"
adjoint_audit = ROOT / f"benchmark_fathi_strict/reports/phaseA_task2C_adjoint_complete/{transition}_adjoint_complete_audit.txt"

def count_capteurs(trace_dir):
    if not trace_dir.exists():
        return 0
    return len(list(trace_dir.glob("capteurs.*.h5")))

def check_strict_forward():
    n = count_capteurs(strict_forward_traces)
    return {
        "stage": "strict_forward",
        "ok": n > 0,
        "status": "PASS_ALREADY_EXISTS" if n > 0 else "MISSING",
        "evidence": str(strict_forward_traces),
        "capteurs_file_count": n,
    }

def check_residual():
    ok = residual_h5.exists()
    return {
        "stage": "residual",
        "ok": ok,
        "status": "PASS_ALREADY_EXISTS" if ok else "MISSING",
        "evidence": str(residual_h5),
        "summary": str(residual_summary_txt),
    }

def check_prepare_adjoint():
    records = []
    ok = True
    for comp in ["x", "y", "z"]:
        for i in range(10):
            batch = f"batch_{i:03d}"
            d = adjoint_base / comp / batch
            has_dir = d.exists()
            has_input = (d / "input.spec").exists()
            has_mat = (d / "mat/h5/Mat_0_Kappa.h5").exists()
            r_ok = has_dir and has_input and has_mat
            ok = ok and r_ok
            records.append({
                "component": comp,
                "batch": batch,
                "workspace": str(d),
                "has_dir": has_dir,
                "has_input_spec": has_input,
                "has_material_h5": has_mat,
                "ok": r_ok,
            })

    return {
        "stage": "prepare_adjoint",
        "ok": ok,
        "status": "PASS_ALREADY_EXISTS" if ok else "MISSING",
        "batch_count": len(records),
        "ok_batch_count": sum(1 for r in records if r["ok"]),
        "records_preview": records[:5],
    }

def check_adjoint():
    records = []
    ok = True
    for comp in ["x", "y", "z"]:
        for i in range(10):
            batch = f"batch_{i:03d}"
            traces = adjoint_base / comp / batch / "traces"
            n = count_capteurs(traces)
            r_ok = n > 0
            ok = ok and r_ok
            records.append({
                "component": comp,
                "batch": batch,
                "traces": str(traces),
                "capteurs_file_count": n,
                "ok": r_ok,
            })

    return {
        "stage": "adjoint",
        "ok": ok,
        "status": "PASS_ALREADY_EXISTS" if ok else "MISSING",
        "batch_count": len(records),
        "ok_batch_count": sum(1 for r in records if r["ok"]),
        "records_preview": records[:5],
    }

def check_audit_adjoint():
    ok = adjoint_audit.exists() and "RESULT = PASS" in adjoint_audit.read_text(errors="ignore")
    return {
        "stage": "audit_adjoint",
        "ok": ok,
        "status": "PASS_ALREADY_EXISTS" if ok else "MISSING",
        "evidence": str(adjoint_audit),
    }

CHECKS = {
    "strict_forward": check_strict_forward,
    "residual": check_residual,
    "prepare_adjoint": check_prepare_adjoint,
    "adjoint": check_adjoint,
    "audit_adjoint": check_audit_adjoint,
}

sequence = [
    "strict_forward",
    "residual",
    "prepare_adjoint",
    "adjoint",
    "audit_adjoint",
]

created = datetime.now().isoformat()

report_dir = ROOT / "benchmark_fathi_strict/reports/task_wrappers"
report_dir.mkdir(parents=True, exist_ok=True)

if args.stage == "plan":
    records = []
else:
    stages = sequence if args.stage == "all" else [args.stage]
    records = [CHECKS[s]() for s in stages]

payload = {
    "created": created,
    "transition": transition,
    "stage": args.stage,
    "execute": args.execute,
    "scope": "prerequisite checker for already-produced strict_forward/residual/adjoint outputs",
    "records": records,
}

if args.stage == "plan":
    result = "PASS_PLAN"
else:
    all_ok = all(r["ok"] for r in records)
    result = "PASS_ALREADY_EXISTS" if all_ok else "FAIL_MISSING_PREREQUISITES"

payload["result"] = result

out_json = report_dir / f"{transition}_task0_prerequisites_{args.stage}.json"
out_txt = report_dir / f"{transition}_task0_prerequisites_{args.stage}.txt"

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Task 0 prerequisite wrapper")
lines.append("===========================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"stage = {args.stage}")
lines.append(f"execute = {args.execute}")
lines.append("")
lines.append("Scope:")
lines.append("  This wrapper checks prerequisite outputs for a resumed iteration.")
lines.append("  It does not yet execute strict_forward/residual/adjoint from scratch.")
lines.append("")
lines.append("Stages:")
for s in sequence:
    lines.append(f"  {s}")
lines.append("")

if records:
    for r in records:
        lines.append("-" * 100)
        lines.append(f"stage = {r['stage']}")
        lines.append(f"status = {r['status']}")
        lines.append(f"ok = {r['ok']}")
        for key in ["evidence", "summary", "capteurs_file_count", "batch_count", "ok_batch_count"]:
            if key in r:
                lines.append(f"{key} = {r[key]}")

lines.append("")
lines.append("Important:")
lines.append("  If this returns FAIL_MISSING_PREREQUISITES, the current engine cannot continue from scratch yet.")
lines.append("  The missing stage must be wrapped with a real executor before full standalone mode.")
lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if result == "FAIL_MISSING_PREREQUISITES":
    sys.exit(2)
