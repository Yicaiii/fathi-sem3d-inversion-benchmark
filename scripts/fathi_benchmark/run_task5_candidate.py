from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import subprocess
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument(
    "--stage",
    choices=["plan", "forward", "misfit", "accept", "all"],
    default="plan",
)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force-forward", action="store_true")
parser.add_argument("--np", type=int, default=12)
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

report_dir = ROOT / "benchmark_fathi_strict/reports/task5_wrappers"
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

def run_cmd(cmd, name):
    print("")
    print("=" * 100)
    print(f"RUNNING {name}")
    print("=" * 100)
    print(" ".join(cmd))
    print("")

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
    )

    if proc.returncode != 0:
        print("")
        print(f"FAILED: {name}")
        print(f"returncode = {proc.returncode}")
        sys.exit(proc.returncode)

def cmd_forward():
    cmd = [
        sys.executable,
        "scripts/iteration_engine/run_candidate_forward.py",
        "--iter-k", str(k),
        "--candidate", args.candidate,
        "--np", str(args.np),
    ]
    if args.execute:
        cmd.append("--execute")
    if args.force_forward:
        cmd.append("--force")
    return cmd

def cmd_misfit_v2():
    return [
        sys.executable,
        "scripts/iteration_engine/compute_candidate_misfit_v2.py",
        "--iter-k", str(k),
        "--candidate", args.candidate,
    ]

def cmd_accept_v2():
    return [
        sys.executable,
        "scripts/iteration_engine/accept_candidate_if_descent_v2.py",
        "--iter-k", str(k),
        "--candidate", args.candidate,
    ]

plan = {
    "created": created,
    "transition": transition,
    "candidate": args.candidate,
    "stage": args.stage,
    "execute": args.execute,
    "force_forward": args.force_forward,
    "np": args.np,
    "canonical_task5": {
        "forward": "scripts/iteration_engine/run_candidate_forward.py",
        "misfit": "scripts/iteration_engine/compute_candidate_misfit_v2.py",
        "accept": "scripts/iteration_engine/accept_candidate_if_descent_v2.py",
    },
    "deprecated_do_not_use": {
        "misfit": "scripts/iteration_engine/compute_candidate_misfit.py",
        "accept": "scripts/iteration_engine/accept_candidate_if_descent.py",
    },
}

plan_json = report_dir / f"{transition}_{args.candidate}_task5_plan.json"
plan_txt = report_dir / f"{transition}_{args.candidate}_task5_plan.txt"

lines = []
lines.append("Canonical Task 5 candidate runner")
lines.append("=================================")
lines.append("")
lines.append(f" Task 5 candidate runner")
lines.append("=================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"stage = {args.stage}")
lines.append(f"execute = {args.execute}")
lines.append("")
lines.append("Canonical scripts:")
lines.append("  5A forward = scripts/iteration_engine/run_candidate_forward.py")
lines.append("  5B misfit  = scripts/iteration_engine/compute_candidate_misfit_v2.py")
lines.append("  5C accept  = scripts/iteration_engine/accept_candidate_if_descent_v2.py")
lines.append("")
lines.append("Deprecated scripts, do not use:")
lines.append("  scripts/iteration_engine/compute_candidate_misfit.py")
lines.append("  scripts/iteration_engine/accept_candidate_if_descent.py")
lines.append("")
lines.append("Commands:")
lines.append("  forward:")
lines.append("    " + " ".join(cmd_forward()))
lines.append("  misfit:")
lines.append("    " + " ".join(cmd_misfit_v2()))
lines.append("  accept:")
lines.append("    " + " ".join(cmd_accept_v2()))
lines.append("")

plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
plan_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if args.stage == "plan":
    print("RESULT = PASS_PLAN")
    sys.exit(0)

if args.stage in ["forward", "all"]:
    run_cmd(cmd_forward(), "Task 5A candidate forward")

if args.stage in ["misfit", "all"]:
    run_cmd(cmd_misfit_v2(), "Task 5B candidate misfit v2")

if args.stage in ["accept", "all"]:
    run_cmd(cmd_accept_v2(), "Task 5C accept candidate v2")

print("")
print("RESULT = PASS")
