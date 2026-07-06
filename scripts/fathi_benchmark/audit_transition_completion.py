from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import re
import hashlib

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

ctx_path = ROOT / "results/fathi_loop_v2" / transition / f"{transition}_iteration_context.json"

def load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

ctx = load_json(ctx_path) if ctx_path.exists() else {}

def rel_or_abs(x):
    if not x:
        return None
    p = Path(x)
    return p if p.is_absolute() else ROOT / p

def exists(path):
    return path is not None and path.exists()

def count(pattern_root, globpat):
    if pattern_root is None or not pattern_root.exists():
        return 0
    return len(list(pattern_root.glob(globpat)))

def text_has(path, pattern):
    if not exists(path):
        return False
    return pattern in path.read_text(errors="ignore")

def last_result(path):
    if not exists(path):
        return "MISSING"
    txt = path.read_text(errors="ignore")
    m = re.findall(r"RESULT = ([A-Z0-9_]+)", txt)
    return m[-1] if m else "NO_RESULT"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

strict_forward_traces = rel_or_abs(ctx.get("strict_forward_traces"))
residual_h5 = rel_or_abs(ctx.get("residual_h5"))
residual_summary = rel_or_abs(ctx.get("residual_summary_txt"))
adjoint_base = rel_or_abs(ctx.get("output_adjoint_batches_dir") or ctx.get("adjoint_batches_dir"))
mtilde_dir = rel_or_abs(ctx.get("mtilde_dir") or ctx.get("mtilde_solve_dir"))
candidates_dir = rel_or_abs(ctx.get("candidates_dir"))
accepted_dir = rel_or_abs(ctx.get("accepted_dir") or ctx.get("output_accepted_dir"))
next_state = rel_or_abs(ctx.get("next_state"))
parent_state = rel_or_abs(ctx.get("parent_state"))

reports = {
    "prepare_strict_forward": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/{transition}_prepare_strict_forward_task.txt",
    "strict_forward": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/{transition}_strict_forward_task.txt",
    "residual_generation": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/{transition}_residual_generation_task.txt",
    "prepare_adjoint": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/{transition}_prepare_adjoint_task.txt",
    "run_full_all_existing": ROOT / f"benchmark_fathi_strict/reports/run_iteration_full/{transition}_run_iteration_full_context_all_existing.txt",
    "run_full_plan": ROOT / f"benchmark_fathi_strict/reports/run_iteration_full/{transition}_run_iteration_full_context_plan.txt",
}

adjoint_prepared = 0
adjoint_with_traces = 0
adjoint_missing_preview = []

if adjoint_base:
    for comp in ["x", "y", "z"]:
        for i in range(10):
            b = f"batch_{i:03d}"
            d = adjoint_base / comp / b
            prepared = (
                (d / "input.spec").exists()
                and (d / "mesh.input").exists()
                and (d / "material.input").exists()
                and (d / "material.spec").exists()
                and (d / "mat/h5/Mat_0_Kappa.h5").exists()
                and (d / "mat/h5/Mat_0_Mu.h5").exists()
                and (d / "mat/h5/Mat_0_Density.h5").exists()
            )
            traces = count(d / "traces", "capteurs.*.h5")
            if prepared:
                adjoint_prepared += 1
            if traces > 0:
                adjoint_with_traces += 1
            if len(adjoint_missing_preview) < 10 and not prepared:
                adjoint_missing_preview.append(f"{comp}/{b}")

accepted_h5 = accepted_dir / "mat/h5" if accepted_dir else None
accepted_material_ok = (
    exists(accepted_h5 / "Mat_0_Kappa.h5")
    and exists(accepted_h5 / "Mat_0_Mu.h5")
    and exists(accepted_h5 / "Mat_0_Density.h5")
) if accepted_h5 else False

checks = {
    "context_exists": ctx_path.exists(),
    "parent_state_exists": exists(parent_state),
    "next_state_exists": exists(next_state),

    "strict_forward_traces_exist": count(strict_forward_traces, "capteurs.*.h5") > 0,
    "strict_forward_trace_count": count(strict_forward_traces, "capteurs.*.h5"),

    "residual_h5_exists": exists(residual_h5),
    "residual_summary_pass": text_has(residual_summary, "RESULT = PASS"),

    "adjoint_prepared_30": adjoint_prepared == 30,
    "adjoint_prepared_count": adjoint_prepared,
    "adjoint_with_traces_30": adjoint_with_traces == 30,
    "adjoint_with_traces_count": adjoint_with_traces,

    "mtilde_dir_exists": exists(mtilde_dir),
    "mtilde_file_count": count(mtilde_dir, "*") if mtilde_dir else 0,

    "candidates_dir_exists": exists(candidates_dir),
    "candidate_file_count": count(candidates_dir, "**/*") if candidates_dir else 0,

    "accepted_dir_exists": exists(accepted_dir),
    "accepted_material_ok": accepted_material_ok,

    "prepare_strict_forward_report_result": last_result(reports["prepare_strict_forward"]),
    "strict_forward_report_result": last_result(reports["strict_forward"]),
    "residual_generation_report_result": last_result(reports["residual_generation"]),
    "prepare_adjoint_report_result": last_result(reports["prepare_adjoint"]),
    "all_existing_report_result": last_result(reports["run_full_all_existing"]),
}

core_complete = (
    checks["context_exists"]
    and checks["parent_state_exists"]
    and checks["next_state_exists"]
    and checks["strict_forward_traces_exist"]
    and checks["residual_h5_exists"]
    and checks["residual_summary_pass"]
    and checks["adjoint_prepared_30"]
    and checks["adjoint_with_traces_30"]
    and checks["mtilde_dir_exists"]
    and checks["candidates_dir_exists"]
    and checks["accepted_dir_exists"]
    and checks["accepted_material_ok"]
)

out_dir = ROOT / "benchmark_fathi_strict/reports/audit"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_completion_audit.json"
out_txt = out_dir / f"{transition}_completion_audit.txt"

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "context": str(ctx_path),
    "checks": checks,
    "adjoint_missing_preview": adjoint_missing_preview,
    "paths": {
        "strict_forward_traces": str(strict_forward_traces),
        "residual_h5": str(residual_h5),
        "residual_summary": str(residual_summary),
        "adjoint_base": str(adjoint_base),
        "mtilde_dir": str(mtilde_dir),
        "candidates_dir": str(candidates_dir),
        "accepted_dir": str(accepted_dir),
        "next_state": str(next_state),
    },
    "result": "PASS_COMPLETE" if core_complete else "CHECK_INCOMPLETE",
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Transition completion audit")
lines.append("===========================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"context = {ctx_path}")
lines.append("")
lines.append("Checks:")
for key, value in checks.items():
    lines.append(f"  {key} = {value}")

lines.append("")
lines.append("Important paths:")
for key, value in payload["paths"].items():
    lines.append(f"  {key} = {value}")

lines.append("")
lines.append("Adjoint missing preview:")
if adjoint_missing_preview:
    for x in adjoint_missing_preview:
        lines.append(f"  {x}")
else:
    lines.append("  none")

lines.append("")
lines.append("Interpretation:")
if core_complete:
    lines.append(f"  {transition} looks complete enough to be used as a trusted parent for the next iteration.")
else:
    lines.append(f"  {transition} is not fully complete. Resume the missing stages before starting the next iteration.")

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
