from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess
import sys
import os

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--context", default=None)
parser.add_argument("--iter-k", type=int, default=None)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

def rel_or_abs(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

def load_context():
    if args.context:
        ctx_path = rel_or_abs(args.context)
    else:
        if args.iter_k is None:
            raise SystemExit("Need either --context or --iter-k")
        k = args.iter_k
        kp1 = k + 1
        transition = f"iter_{k:03d}_to_iter_{kp1:03d}"
        ctx_path = ROOT / "results/fathi_loop_v2" / transition / f"{transition}_iteration_context.json"

    if not ctx_path.exists():
        raise SystemExit(f"Missing context: {ctx_path}")

    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    return ctx_path, ctx

ctx_path, ctx = load_context()
transition = ctx["transition"]

residual_dir = rel_or_abs(ctx.get("residual_dir", Path(ctx["work_root"]) / "residual_sources"))
h5_out = rel_or_abs(ctx.get("residual_h5", residual_dir / "454B_strict_residual_timeseries.h5"))
summary_txt = rel_or_abs(ctx.get("residual_summary_txt", residual_dir / "454B_strict_residual_timeseries_summary.txt"))

script_a = ROOT / "scripts/fathi_benchmark/generic_from_legacy/454A_compute_strict_forward_residual_manifest_generic.py"
script_b = ROOT / "scripts/fathi_benchmark/generic_from_legacy/454B_build_strict_residual_timeseries_h5_generic.py"

report_dir = ROOT / "benchmark_fathi_strict/reports/executable_tasks"
report_dir.mkdir(parents=True, exist_ok=True)

out_json = report_dir / f"{transition}_residual_generation_task.json"
out_txt = report_dir / f"{transition}_residual_generation_task.txt"

created = datetime.now().isoformat()

def summary_pass():
    if not h5_out.exists() or not summary_txt.exists():
        return False
    return "RESULT = PASS" in summary_txt.read_text(errors="ignore")

record = {
    "created": created,
    "transition": transition,
    "task_type": "residual_generation",
    "context": str(ctx_path),
    "h5_out": str(h5_out),
    "summary_txt": str(summary_txt),
    "execute": args.execute,
    "force": args.force,
    "result": None,
}

lines = []
lines.append("Executable task: residual_generation")
lines.append("====================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"context = {ctx_path}")
lines.append(f"h5_out = {h5_out}")
lines.append(f"summary_txt = {summary_txt}")
lines.append(f"execute = {args.execute}")
lines.append(f"force = {args.force}")
lines.append("")

if summary_pass() and not args.force:
    record["result"] = "PASS_ALREADY_EXISTS"
    lines.append("Residual output already exists and summary is PASS.")
    lines.append("No residual regeneration was launched.")

elif not args.execute:
    record["result"] = "PASS_PLAN_ONLY"
    lines.append("Plan only. Would run:")
    lines.append(f"  python3 {script_a.relative_to(ROOT)} --context {ctx_path.relative_to(ROOT)}")
    lines.append(f"  python3 {script_b.relative_to(ROOT)} --context {ctx_path.relative_to(ROOT)}")

else:
    run_records = []
    ok = True

    for script in [script_a, script_b]:
        cmd = [sys.executable, str(script.relative_to(ROOT)), "--context", str(ctx_path.relative_to(ROOT))]
        lines.append("")
        lines.append("Running:")
        lines.append("  " + " ".join(cmd))

        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        child_result = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("RESULT ="):
                child_result = line.split("=", 1)[1].strip()
                break

        rec = {
            "script": str(script.relative_to(ROOT)),
            "returncode": proc.returncode,
            "child_result": child_result,
            "stdout_tail": proc.stdout.splitlines()[-100:],
            "stderr_tail": proc.stderr.splitlines()[-100:],
        }
        run_records.append(rec)

        lines.append(f"returncode = {proc.returncode}")
        lines.append(f"child_result = {child_result}")
        lines.append("stdout tail:")
        for x in rec["stdout_tail"]:
            lines.append("  " + x)
        if rec["stderr_tail"]:
            lines.append("stderr tail:")
            for x in rec["stderr_tail"]:
                lines.append("  " + x)

        if proc.returncode != 0 or child_result != "PASS":
            ok = False
            break

    record["script_runs"] = run_records
    record["result"] = "PASS_EXECUTED" if ok and summary_pass() else "FAIL_RESIDUAL_GENERATION"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {record['result']}")

out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if record["result"].startswith("FAIL"):
    sys.exit(1)
