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
parser.add_argument(
    "--stage",
    choices=[
        "plan",
        "manifests",
        "rhs_x",
        "rhs_y",
        "rhs_z",
        "rhs_total",
        "mtilde",
        "audit",
        "all",
    ],
    default="plan",
)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

config = json.loads((ROOT / args.config).read_text())

run_result_root = ROOT / config["run_result_root"] / transition
manifest_dir = run_result_root / "rhs_manifests"
component_rhs_dir = run_result_root / "component_rhs"

report_dir = ROOT / "benchmark_fathi_strict/reports/task_wrappers"
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

def cmd_manifests():
    return [
        sys.executable,
        "scripts/iteration_engine/build_rhs_manifests_generic_v2.py",
        "--iter-k", str(k),
    ]

def cmd_rhs(comp):
    return [
        sys.executable,
        "scripts/longterm/424B_compute_rhs_component_from_traces.py",
        "--component", comp,
        "--forward-manifest",
        str((manifest_dir / "forward_full_grid_trace_manifest.csv").relative_to(ROOT)),
        "--adjoint-manifest",
        str((manifest_dir / f"adjoint_{comp}_full_grid_trace_manifest.csv").relative_to(ROOT)),
        "--out-dir",
        str(component_rhs_dir.relative_to(ROOT)),
        "--label",
        "full_grid_trace",
    ]

def cmd_rhs_total():
    return [
        sys.executable,
        "scripts/iteration_engine/assemble_rhs_total_generic.py",
        "--iter-k", str(k),
    ]

def cmd_mtilde():
    return [
        sys.executable,
        "scripts/iteration_engine/solve_mtilde_generic.py",
        "--iter-k", str(k),
        "--execute",
    ]

def cmd_audit():
    return [
        sys.executable,
        "scripts/iteration_engine/audit_mtilde_outputs_generic.py",
        "--iter-k", str(k),
    ]

command_map = {
    "manifests": cmd_manifests(),
    "rhs_x": cmd_rhs("x"),
    "rhs_y": cmd_rhs("y"),
    "rhs_z": cmd_rhs("z"),
    "rhs_total": cmd_rhs_total(),
    "mtilde": cmd_mtilde(),
    "audit": cmd_audit(),
}

all_sequence = [
    "manifests",
    "rhs_x",
    "rhs_y",
    "rhs_z",
    "rhs_total",
    "mtilde",
    "audit",
]

def run_cmd(name, cmd):
    print("")
    print("=" * 100)
    print(f"TASK 3 RUNNING: {name}")
    print("=" * 100)
    print(" ".join(cmd))
    print("")

    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if proc.returncode != 0:
        print("")
        print(f"TASK 3 FAILED: {name}")
        print(f"returncode = {proc.returncode}")
        sys.exit(proc.returncode)

plan = {
    "created": created,
    "transition": transition,
    "stage": args.stage,
    "execute": args.execute,
    "commands": command_map,
    "all_sequence": all_sequence,
    "outputs": {
        "component_rhs_dir": str(component_rhs_dir),
        "mtilde_solve_dir": str(run_result_root / "mtilde_solve"),
    },
}

plan_json = report_dir / f"{transition}_task3_gradient_plan.json"
plan_txt = report_dir / f"{transition}_task3_gradient_plan.txt"

lines = []
lines.append("Canonical Task 3 gradient runner")
lines.append("================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"stage = {args.stage}")
lines.append(f"execute = {args.execute}")
lines.append("")
lines.append("Task 3 sequence:")
for name in all_sequence:
    lines.append(f"  {name}")
lines.append("")
lines.append("Commands:")
for name in all_sequence:
    lines.append("-" * 80)
    lines.append(name)
    lines.append("  " + " ".join(command_map[name]))
lines.append("")
lines.append("Outputs:")
lines.append(f"  component_rhs_dir = {component_rhs_dir}")
lines.append(f"  mtilde_solve_dir = {run_result_root / 'mtilde_solve'}")
lines.append("")
lines.append("RESULT = PASS_PLAN")

plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
plan_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if args.stage == "plan":
    sys.exit(0)

if not args.execute:
    print("")
    print("DRY WRAPPER ONLY.")
    print("No command was executed because --execute was not provided.")
    print("RESULT = PASS_DRYRUN")
    sys.exit(0)

if args.stage == "all":
    for name in all_sequence:
        run_cmd(name, command_map[name])
else:
    run_cmd(args.stage, command_map[args.stage])

print("")
print("RESULT = PASS")
