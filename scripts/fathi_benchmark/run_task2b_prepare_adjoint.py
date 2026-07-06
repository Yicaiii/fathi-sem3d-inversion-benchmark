from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess
import sys
import os

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--context", required=True)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

ctx_path = Path(args.context)
if not ctx_path.is_absolute():
    ctx_path = ROOT / ctx_path

ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
transition = ctx["transition"]

adj_base = Path(ctx["output_adjoint_batches_dir"])
if not adj_base.is_absolute():
    adj_base = ROOT / adj_base

report_dir = ROOT / "benchmark_fathi_strict/reports/executable_tasks"
report_dir.mkdir(parents=True, exist_ok=True)

out_txt = report_dir / f"{transition}_prepare_adjoint_task.txt"
out_json = report_dir / f"{transition}_prepare_adjoint_task.json"

scripts = [
    "scripts/fathi_benchmark/generic_from_legacy/455A_extract_old_adjoint_source_format_generic.py",
    "scripts/fathi_benchmark/generic_from_legacy/455B_prepare_strict_adjoint_batches_from_residual_generic.py",
    "scripts/fathi_benchmark/generic_from_legacy/455C_audit_strict_adjoint_batches_generic.py",
]

def inspect_batches():
    records = []
    for comp in ["x", "y", "z"]:
        for i in range(10):
            batch = f"batch_{i:03d}"
            d = adj_base / comp / batch
            rec = {
                "component": comp,
                "batch": batch,
                "dir": str(d),
                "has_dir": d.exists(),
                "has_input_spec": (d / "input.spec").exists(),
                "has_material_h5": (d / "mat/h5/Mat_0_Kappa.h5").exists(),
                "has_traces": (d / "traces").exists() and len(list((d / "traces").glob("capteurs.*.h5"))) > 0,
            }
            rec["ok_prepared"] = rec["has_dir"] and rec["has_input_spec"] and rec["has_material_h5"]
            records.append(rec)
    return records

created = datetime.now().isoformat()
records = inspect_batches()
prepared_count = sum(1 for r in records if r["ok_prepared"])
trace_count_ready = sum(1 for r in records if r["has_traces"])

payload = {
    "created": created,
    "transition": transition,
    "task_type": "prepare_adjoint",
    "context": str(ctx_path),
    "adjoint_batches_dir": str(adj_base),
    "execute": args.execute,
    "force": args.force,
    "prepared_count_before": prepared_count,
    "trace_count_ready_before": trace_count_ready,
    "result": None,
}

lines = []
lines.append("Executable task: prepare_adjoint")
lines.append("================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"context = {ctx_path}")
lines.append(f"adjoint_batches_dir = {adj_base}")
lines.append(f"execute = {args.execute}")
lines.append(f"force = {args.force}")
lines.append("")
lines.append(f"prepared_count_before = {prepared_count} / 30")
lines.append(f"trace_count_ready_before = {trace_count_ready} / 30")
lines.append("")

if prepared_count == 30 and not args.force:
    payload["result"] = "PASS_ALREADY_EXISTS"
    lines.append("Adjoint batch workspaces are already prepared.")
    lines.append("No 455A/455B/455C execution was launched.")

elif not args.execute:
    payload["result"] = "PASS_PLAN_ONLY"
    lines.append("Plan only. Would run:")
    for s in scripts:
        lines.append(f"  python3 {s} --context {ctx_path.relative_to(ROOT)}")

else:
    run_records = []
    ok = True

    for s in scripts:
        cmd = [sys.executable, s, "--context", str(ctx_path.relative_to(ROOT))]
        lines.append("")
        lines.append("Running:")
        lines.append("  " + " ".join(cmd))

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout_tail = proc.stdout.splitlines()[-80:]
        stderr_tail = proc.stderr.splitlines()[-80:]

        child_result = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("RESULT ="):
                child_result = line.split("=", 1)[1].strip()
                break

        rec = {
            "script": s,
            "returncode": proc.returncode,
            "child_result": child_result,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        run_records.append(rec)

        lines.append(f"returncode = {proc.returncode}")
        lines.append(f"child_result = {child_result}")
        lines.append("stdout tail:")
        for x in stdout_tail:
            lines.append("  " + x)

        if stderr_tail:
            lines.append("stderr tail:")
            for x in stderr_tail:
                lines.append("  " + x)

        if proc.returncode != 0 or child_result != "PASS":
            ok = False
            break

    records_after = inspect_batches()
    prepared_count_after = sum(1 for r in records_after if r["ok_prepared"])

    payload["script_runs"] = run_records
    payload["prepared_count_after"] = prepared_count_after
    payload["result"] = "PASS_EXECUTED" if ok and prepared_count_after == 30 else "FAIL_PREPARE_ADJOINT"

    lines.append("")
    lines.append(f"prepared_count_after = {prepared_count_after} / 30")

payload["records_preview"] = records[:5]

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if payload["result"].startswith("FAIL"):
    sys.exit(1)
