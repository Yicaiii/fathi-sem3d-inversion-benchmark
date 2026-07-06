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

def rel_or_abs(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

workspace = rel_or_abs(ctx["strict_forward_workspace"])
traces_dir = rel_or_abs(ctx["strict_forward_traces"])
parent_accepted_dir = rel_or_abs(ctx["input_accepted_dir"])

report_dir = ROOT / "benchmark_fathi_strict/reports/executable_tasks"
report_dir.mkdir(parents=True, exist_ok=True)

out_txt = report_dir / f"{transition}_prepare_strict_forward_task.txt"
out_json = report_dir / f"{transition}_prepare_strict_forward_task.json"

script_450b = "scripts/fathi_benchmark/generic_from_legacy/450B_select_strict_forward_full_template_generic.py"
script_450c = "scripts/fathi_benchmark/generic_from_legacy/450C_prepare_strict_full_forward_run_generic.py"

def count_capteurs():
    if not traces_dir.exists():
        return 0
    return len(list(traces_dir.glob("capteurs.*.h5")))

def inspect_workspace():
    required = {
        "workspace": workspace.exists(),
        "input_spec": (workspace / "input.spec").exists(),
        "mesh_input": (workspace / "mesh.input").exists(),
        "material_input": (workspace / "material.input").exists(),
        "material_spec": (workspace / "material.spec").exists(),
        "kappa_h5": (workspace / "mat/h5/Mat_0_Kappa.h5").exists(),
        "mu_h5": (workspace / "mat/h5/Mat_0_Mu.h5").exists(),
        "density_h5": (workspace / "mat/h5/Mat_0_Density.h5").exists(),
    }
    ok = all(required.values())
    return ok, required

created = datetime.now().isoformat()
ok_before, required_before = inspect_workspace()
trace_count_before = count_capteurs()

payload = {
    "created": created,
    "transition": transition,
    "task_type": "prepare_strict_forward",
    "context": str(ctx_path),
    "workspace": str(workspace),
    "traces_dir": str(traces_dir),
    "parent_accepted_dir": str(parent_accepted_dir),
    "execute": args.execute,
    "force": args.force,
    "workspace_ok_before": ok_before,
    "required_before": required_before,
    "trace_count_before": trace_count_before,
    "result": None,
}

lines = []
lines.append("Executable task: prepare_strict_forward")
lines.append("=======================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"context = {ctx_path}")
lines.append(f"workspace = {workspace}")
lines.append(f"traces_dir = {traces_dir}")
lines.append(f"parent_accepted_dir = {parent_accepted_dir}")
lines.append(f"execute = {args.execute}")
lines.append(f"force = {args.force}")
lines.append("")
lines.append(f"workspace_ok_before = {ok_before}")
lines.append(f"trace_count_before = {trace_count_before}")
lines.append("")
lines.append("required_before:")
for k, v in required_before.items():
    lines.append(f"  {k} = {v}")
lines.append("")

if ok_before and not args.force:
    payload["result"] = "PASS_ALREADY_EXISTS"
    lines.append("Strict forward workspace is already prepared.")
    lines.append("No 450C execution was launched.")

elif not args.execute:
    payload["result"] = "PASS_PLAN_ONLY"
    lines.append("Plan only. Would run:")
    lines.append(f"  python3 {script_450b} --context {ctx_path.relative_to(ROOT)}")
    lines.append(f"  python3 {script_450c} --context {ctx_path.relative_to(ROOT)}")

else:
    script_runs = []
    all_child_ok = True

    for script in [script_450b, script_450c]:
        cmd = [
            sys.executable,
            script,
            "--context",
            str(ctx_path.relative_to(ROOT)),
        ]

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

        stdout_tail = proc.stdout.splitlines()[-100:]
        stderr_tail = proc.stderr.splitlines()[-100:]

        child_result = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("RESULT ="):
                child_result = line.split("=", 1)[1].strip()
                break

        script_runs.append({
            "script": script,
            "returncode": proc.returncode,
            "child_result": child_result,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        })

        lines.append(f"returncode = {proc.returncode}")
        lines.append(f"child_result = {child_result}")
        lines.append("stdout tail:")
        for x in stdout_tail:
            lines.append("  " + x)

        if stderr_tail:
            lines.append("")
            lines.append("stderr tail:")
            for x in stderr_tail:
                lines.append("  " + x)

        if proc.returncode != 0 or child_result != "PASS":
            all_child_ok = False
            break

    ok_after, required_after = inspect_workspace()
    trace_count_after = count_capteurs()

    payload["script_runs"] = script_runs
    payload["workspace_ok_after"] = ok_after
    payload["required_after"] = required_after
    payload["trace_count_after"] = trace_count_after

    lines.append("")
    lines.append(f"workspace_ok_after = {ok_after}")
    lines.append(f"trace_count_after = {trace_count_after}")
    lines.append("")
    lines.append("required_after:")
    for k, v in required_after.items():
        lines.append(f"  {k} = {v}")

    if all_child_ok and ok_after:
        payload["result"] = "PASS_EXECUTED"
    elif ok_after:
        payload["result"] = "PASS_EXECUTED_WITH_CHILD_RESULT_CHECK"
    else:
        payload["result"] = "FAIL_PREPARE_STRICT_FORWARD"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if payload["result"].startswith("FAIL"):
    sys.exit(1)
