from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess
import sys
import os

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument("--np", type=int, default=12)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

sem3d_exe = Path(config.get("sem3d_exe", "/home/crellamaybe/SEM/build/SEM3D/sem3d.exe"))
if not sem3d_exe.is_absolute():
    sem3d_exe = ROOT / sem3d_exe

run_result_root = ROOT / config["run_result_root"] / transition
run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"

candidate_root = run_result_root / "candidates"
candidate_dir = candidate_root / args.candidate

workspace = run_data_root / "candidate_forward_workspaces" / args.candidate

report_dir = ROOT / "benchmark_fathi_strict/reports/candidate_forward"
log_dir = run_result_root / "candidate_forward_runs/logs"
report_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

stdout_log = log_dir / f"{args.candidate}_stdout.log"
stderr_log = log_dir / f"{args.candidate}_stderr.log"
summary_path = run_result_root / "candidate_forward_runs" / f"{args.candidate}_forward_summary.txt"
summary_path.parent.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

required = [
    sem3d_exe,
    candidate_dir,
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

traces_dir = workspace / "traces"
existing_trace_files = sorted(traces_dir.glob("capteurs.*.h5")) if traces_dir.exists() else []

record = {
    "created": created,
    "transition": transition,
    "candidate": args.candidate,
    "execute": args.execute,
    "force": args.force,
    "np": args.np,
    "workspace": str(workspace),
    "sem3d_exe": str(sem3d_exe),
    "stdout_log": str(stdout_log),
    "stderr_log": str(stderr_log),
    "missing": [str(p) for p in missing],
    "existing_trace_count_before": len(existing_trace_files),
    "result": None,
}

lines = []
lines.append("Task 5A candidate forward runner")
lines.append("================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"execute = {args.execute}")
lines.append(f"force = {args.force}")
lines.append(f"np = {args.np}")
lines.append("")
lines.append(f"workspace = {workspace}")
lines.append(f"sem3d_exe = {sem3d_exe}")
lines.append(f"stdout_log = {stdout_log}")
lines.append(f"stderr_log = {stderr_log}")
lines.append("")

if missing:
    lines.append("Missing required inputs:")
    for p in missing:
        lines.append(f"  {p}")
    record["result"] = "FAIL_MISSING_INPUTS"

elif existing_trace_files and not args.force:
    lines.append("Existing candidate traces found.")
    lines.append(f"existing_trace_count_before = {len(existing_trace_files)}")
    lines.append("Will not rerun SEM3D unless --force is provided.")
    lines.append("")
    lines.append("RESULT = PASS_ALREADY_EXISTS")
    record["result"] = "PASS_ALREADY_EXISTS"

elif not args.execute:
    cmd = ["mpirun", "-np", str(args.np), str(sem3d_exe)]
    lines.append("DRY RUN ONLY. SEM3D was not launched.")
    lines.append("Command would be:")
    lines.append("  " + " ".join(cmd))
    lines.append("")
    lines.append("To execute:")
    lines.append(f"  python3 scripts/iteration_engine/run_candidate_forward.py --iter-k {k} --candidate {args.candidate} --execute")
    record["result"] = "PASS_DRYRUN"

else:
    cmd = ["mpirun", "-np", str(args.np), str(sem3d_exe)]
    record["cmd"] = cmd

    lines.append("Running SEM3D candidate forward:")
    lines.append("  " + " ".join(cmd))
    lines.append("")

    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            stdout=out,
            stderr=err,
            text=True,
        )

    record["returncode"] = proc.returncode

    trace_files_after = sorted((workspace / "traces").glob("capteurs.*.h5")) if (workspace / "traces").exists() else []
    record["trace_count_after"] = len(trace_files_after)

    stdout_tail = stdout_log.read_text(errors="ignore").splitlines()[-80:] if stdout_log.exists() else []
    stderr_tail = stderr_log.read_text(errors="ignore").splitlines()[-80:] if stderr_log.exists() else []

    record["stdout_tail"] = stdout_tail
    record["stderr_tail"] = stderr_tail

    lines.append(f"returncode = {proc.returncode}")
    lines.append(f"trace_count_after = {len(trace_files_after)}")
    lines.append("")
    lines.append("stdout tail:")
    for x in stdout_tail:
        lines.append("  " + x)

    if stderr_tail:
        lines.append("")
        lines.append("stderr tail:")
        for x in stderr_tail:
            lines.append("  " + x)

    if proc.returncode == 0 and len(trace_files_after) > 0:
        record["result"] = "PASS"
    else:
        record["result"] = "FAIL"

json_path = report_dir / f"{transition}_{args.candidate}_forward_run.json"
txt_path = report_dir / f"{transition}_{args.candidate}_forward_run.txt"

lines.append("")
lines.append(f"json = {json_path}")
lines.append(f"summary = {summary_path}")
lines.append("")
lines.append(f"RESULT = {record['result']}")

json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
txt_path.write_text("\n".join(lines), encoding="utf-8")
summary_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if record["result"] not in ["PASS", "PASS_DRYRUN", "PASS_ALREADY_EXISTS"]:
    sys.exit(1)
