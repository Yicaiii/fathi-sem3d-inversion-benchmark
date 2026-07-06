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
parser.add_argument(
    "--stage",
    choices=[
        "plan",
        "status",
        "prerequisites",
        "gradient",
        "candidates",
        "task5",
        "all",
    ],
    default="plan",
)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force-forward", action="store_true")
parser.add_argument("--np", type=int, default=12)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

report_dir = ROOT / "benchmark_fathi_strict/reports/run_iteration"
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

def run_cmd(name, cmd):
    print("")
    print("=" * 100)
    print(f"RUN_ITERATION: {name}")
    print("=" * 100)
    print(" ".join(cmd))
    print("")

    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if proc.returncode != 0:
        print("")
        print(f"RUN_ITERATION FAILED AT: {name}")
        print(f"returncode = {proc.returncode}")
        sys.exit(proc.returncode)

def cmd_status():
    p = ROOT / "scripts/iteration_engine/write_iteration_stage_report_v2.py"
    if p.exists():
        return [
            sys.executable,
            "scripts/iteration_engine/write_iteration_stage_report_v2.py",
            "--iter-k", str(k),
        ]
    return [
        sys.executable,
        "scripts/iteration_engine/refresh_benchmark_status.py",
        "--iter-k", str(k),
    ]

def cmd_prerequisites():
    cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_task0_prerequisites.py",
        "--iter-k", str(k),
        "--stage", "all",
    ]
    if args.execute:
        cmd.append("--execute")
    return cmd

def cmd_gradient():
    cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_task3_gradient.py",
        "--iter-k", str(k),
        "--stage", "all",
    ]
    if args.execute:
        cmd.append("--execute")
    return cmd

def cmd_candidates():
    cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_task4_candidates.py",
        "--iter-k", str(k),
        "--stage", "all",
    ]
    if args.execute:
        cmd.append("--execute")
    return cmd

def cmd_task5():
    cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_task5_candidate.py",
        "--iter-k", str(k),
        "--candidate", args.candidate,
        "--stage", "all",
        "--np", str(args.np),
    ]
    if args.execute:
        cmd.append("--execute")
    if args.force_forward:
        cmd.append("--force-forward")
    return cmd

plan = {
    "created": created,
    "transition": transition,
    "stage": args.stage,
    "candidate": args.candidate,
    "execute": args.execute,
    "force_forward": args.force_forward,
    "np": args.np,
    "scope": {
        "current_engine_type": "resumed_iteration_orchestrator",
        "all_sequence": [
            "prerequisites_check",
            "gradient",
            "candidates",
            "task5",
            "status"
        ],
        "not_yet_full_standalone": True,
        "missing_real_executors_for": [
            "strict_forward",
            "residual_generation",
            "prepare_adjoint",
            "run_adjoint_batches"
        ],
    },
    "commands": {
        "status": cmd_status(),
        "prerequisites": cmd_prerequisites(),
        "gradient": cmd_gradient(),
        "candidates": cmd_candidates(),
        "task5": cmd_task5(),
    },
}

plan_json = report_dir / f"{transition}_run_iteration_plan.json"
plan_txt = report_dir / f"{transition}_run_iteration_plan.txt"

lines = []
lines.append("Fathi benchmark run_iteration orchestrator")
lines.append("==========================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"stage = {args.stage}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"execute = {args.execute}")
lines.append(f"force_forward = {args.force_forward}")
lines.append(f"np = {args.np}")
lines.append("")
lines.append("Scope warning:")
lines.append("  This is currently a resumed iteration orchestrator.")
lines.append("  --stage all means:")
lines.append("    prerequisites_check -> gradient -> candidates -> task5 -> status")
lines.append("  The prerequisites_check verifies that strict forward, residual, and adjoint outputs already exist.")
lines.append("  It is not yet a fully standalone from-scratch Fathi iteration runner.")
lines.append("")
lines.append("Canonical stages:")
lines.append("  prerequisites -> Task 0: check strict_forward/residual/adjoint outputs")
lines.append("  gradient      -> Task 3: RHS + Mtilde")
lines.append("  candidates    -> Task 4: candidate materials + workspaces")
lines.append("  task5         -> Task 5: candidate forward + misfit_v2 + accept_v2")
lines.append("")
lines.append("Commands:")
for name in ["prerequisites", "status", "gradient", "candidates", "task5"]:
    lines.append("-" * 80)
    lines.append(name)
    lines.append("  " + " ".join(plan["commands"][name]))
lines.append("")
lines.append("RESULT = PASS_PLAN")

plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
plan_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if args.stage == "plan":
    sys.exit(0)

if args.stage == "status":
    run_cmd("status", cmd_status())

elif args.stage == "prerequisites":
    run_cmd("prerequisites", cmd_prerequisites())

elif args.stage == "gradient":
    run_cmd("gradient", cmd_gradient())

elif args.stage == "candidates":
    run_cmd("candidates", cmd_candidates())

elif args.stage == "task5":
    run_cmd("task5", cmd_task5())

elif args.stage == "all":
    run_cmd("prerequisites", cmd_prerequisites())
    run_cmd("gradient", cmd_gradient())
    run_cmd("candidates", cmd_candidates())
    run_cmd("task5", cmd_task5())
    run_cmd("status", cmd_status())

print("")
print("RESULT = PASS")
