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
parser.add_argument("--np", type=int, default=None)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
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

transition = ctx.get("transition")
if not transition:
    k = int(ctx["iter_k"])
    kp1 = int(ctx["iter_kp1"])
    transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

iter_k = int(ctx.get("iter_k", args.iter_k if args.iter_k is not None else -1))
iter_kp1 = int(ctx.get("iter_kp1", iter_k + 1))

np_count = args.np or int(ctx.get("mpi_cores", 12))

sem3d_exe = Path(ctx.get("sem3d_exe", str(Path.home() / "SEM/build/SEM3D/sem3d.exe")))
if not sem3d_exe.is_absolute():
    sem3d_exe = ROOT / sem3d_exe

workspace_raw = ctx.get("strict_forward_workspace")
traces_raw = ctx.get("strict_forward_traces")

if not workspace_raw:
    raise SystemExit("Context missing strict_forward_workspace")
if not traces_raw:
    raise SystemExit("Context missing strict_forward_traces")

workspace = rel_or_abs(workspace_raw)
traces_dir = rel_or_abs(traces_raw)

run_root = rel_or_abs(ctx.get("transition_result_root", f"results/fathi_loop_v2/{transition}"))
log_dir = run_root / "strict_forward_run/logs"
log_dir.mkdir(parents=True, exist_ok=True)

report_dir = ROOT / "benchmark_fathi_strict/reports/executable_tasks"
report_dir.mkdir(parents=True, exist_ok=True)

stdout_log = log_dir / "strict_forward_stdout.log"
stderr_log = log_dir / "strict_forward_stderr.log"

out_json = report_dir / f"{transition}_strict_forward_task.json"
out_txt = report_dir / f"{transition}_strict_forward_task.txt"

created = datetime.now().isoformat()

def count_capteurs():
    if not traces_dir.exists():
        return 0
    return len(list(traces_dir.glob("capteurs.*.h5")))

required = [
    sem3d_exe,
    workspace,
    workspace / "input.spec",
    workspace / "mesh.input",
    workspace / "material.input",
    workspace / "material.spec",
    workspace / "mat/h5/Mat_0_Kappa.h5",
    workspace / "mat/h5/Mat_0_Mu.h5",
    workspace / "mat/h5/Mat_0_Density.h5",
]

missing = [p for p in required if not p.exists()]
existing_trace_count = count_capteurs()

record = {
    "created": created,
    "transition": transition,
    "task_type": "strict_forward",
    "context": str(ctx_path),
    "iter_k": iter_k,
    "iter_kp1": iter_kp1,
    "workspace": str(workspace),
    "traces_dir": str(traces_dir),
    "sem3d_exe": str(sem3d_exe),
    "np": np_count,
    "execute": args.execute,
    "force": args.force,
    "missing": [str(p) for p in missing],
    "existing_trace_count_before": existing_trace_count,
    "result": None,
}

lines = []
lines.append("Executable task: strict_forward")
lines.append("===============================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"context = {ctx_path}")
lines.append(f"workspace = {workspace}")
lines.append(f"traces_dir = {traces_dir}")
lines.append(f"existing_trace_count_before = {existing_trace_count}")
lines.append(f"execute = {args.execute}")
lines.append(f"force = {args.force}")
lines.append("")

if missing:
    record["result"] = "FAIL_MISSING_INPUTS"
    lines.append("Missing required inputs:")
    for p in missing:
        lines.append(f"  {p}")

elif existing_trace_count > 0 and not args.force:
    record["result"] = "PASS_ALREADY_EXISTS"
    lines.append("Strict forward traces already exist.")
    lines.append("No SEM3D run was launched.")

elif not args.execute:
    record["result"] = "PASS_PLAN_ONLY"
    cmd = ["mpirun", "-np", str(np_count), str(sem3d_exe)]
    record["cmd"] = cmd
    lines.append("Plan only. SEM3D was not launched.")
    lines.append("Command would be:")
    lines.append("  " + " ".join(cmd))

else:
    cmd = ["mpirun", "-np", str(np_count), str(sem3d_exe)]
    record["cmd"] = cmd

    lines.append("Running SEM3D strict forward:")
    lines.append("  " + " ".join(cmd))
    lines.append("")

    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=workspace, stdout=out, stderr=err, text=True)

    after_count = count_capteurs()

    record["returncode"] = proc.returncode
    record["trace_count_after"] = after_count
    record["stdout_log"] = str(stdout_log)
    record["stderr_log"] = str(stderr_log)

    lines.append(f"returncode = {proc.returncode}")
    lines.append(f"trace_count_after = {after_count}")
    lines.append(f"stdout_log = {stdout_log}")
    lines.append(f"stderr_log = {stderr_log}")

    if proc.returncode == 0 and after_count > 0:
        record["result"] = "PASS_EXECUTED"
    else:
        record["result"] = "FAIL_EXECUTION"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {record['result']}")

out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if record["result"].startswith("FAIL"):
    sys.exit(1)
