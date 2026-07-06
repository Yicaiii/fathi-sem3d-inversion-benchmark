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
parser.add_argument(
    "--stage",
    choices=[
        "plan",
        "preflight",
        "prepare_strict_forward",
        "strict_forward",
        "residual_generation",
        "prepare_adjoint",
        "adjoint_sample",
        "adjoint_all",
        "gradient",
        "candidates",
        "task5",
        "all_existing",
        "all_full",
    ],
    default="plan",
)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument("--component", default="x", choices=["x", "y", "z"])
parser.add_argument("--batch", default="batch_000")
parser.add_argument("--np", type=int, default=None)
parser.add_argument("--execute-heavy", action="store_true")
parser.add_argument("--allow-mutate", action="store_true")
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
iter_k = int(ctx["iter_k"])
iter_kp1 = int(ctx["iter_kp1"])
np_count = args.np or int(ctx.get("mpi_cores", 12))

report_dir = ROOT / "benchmark_fathi_strict/reports/run_iteration_full"
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()
out_json = report_dir / f"{transition}_run_iteration_full_context_{args.stage}.json"
out_txt = report_dir / f"{transition}_run_iteration_full_context_{args.stage}.txt"

ctx_rel = str(ctx_path.relative_to(ROOT))

def cmd_prepare_strict_forward(execute=True):
    cmd = [sys.executable, "scripts/fathi_benchmark/run_task1b_prepare_strict_forward.py", "--context", ctx_rel]
    if execute:
        cmd.append("--execute")
    if args.force:
        cmd.append("--force")
    return cmd

def cmd_strict_forward(execute=False):
    cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_task1_strict_forward.py",
        "--context", ctx_rel,
        "--np", str(np_count),
    ]
    if execute:
        cmd.append("--execute")
    if args.force:
        cmd.append("--force")
    return cmd

def cmd_residual_generation(execute=True):
    cmd = [sys.executable, "scripts/fathi_benchmark/run_task2_residual_generation.py", "--context", ctx_rel]
    if execute:
        cmd.append("--execute")
    if args.force:
        cmd.append("--force")
    return cmd

def cmd_prepare_adjoint(execute=True):
    cmd = [sys.executable, "scripts/fathi_benchmark/run_task2b_prepare_adjoint.py", "--context", ctx_rel]
    if execute:
        cmd.append("--execute")
    if args.force:
        cmd.append("--force")
    return cmd

def cmd_adjoint_batch(component, batch, execute=False):
    cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_task2c_adjoint_batch.py",
        "--context", ctx_rel,
        "--component", component,
        "--batch", batch,
        "--np", str(np_count),
    ]
    if execute:
        cmd.append("--execute")
    if args.force:
        cmd.append("--force")
    return cmd

def cmd_gradient():
    return [
        sys.executable,
        "scripts/fathi_benchmark/run_task3_gradient.py",
        "--iter-k", str(iter_k),
        "--stage", "all",
        "--execute",
    ]

def cmd_candidates():
    return [
        sys.executable,
        "scripts/fathi_benchmark/run_task4_candidates.py",
        "--iter-k", str(iter_k),
        "--stage", "all",
        "--execute",
    ]

def cmd_task5():
    return [
        sys.executable,
        "scripts/fathi_benchmark/run_task5_candidate.py",
        "--iter-k", str(iter_k),
        "--candidate", args.candidate,
        "--stage", "all",
        "--np", str(np_count),
        "--execute",
    ]

def extract_result(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("RESULT ="):
            return line.split("=", 1)[1].strip()
    return None

def run_cmd(label, cmd):
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child_result = extract_result(proc.stdout)

    base_ok = proc.returncode == 0 and child_result is not None and child_result.startswith("PASS")

    expected_missing_ok = (
        args.stage == "preflight"
        and label == "adjoint_sample_plan"
        and child_result == "FAIL_MISSING_INPUTS"
    )

    ok = base_ok or expected_missing_ok

    return {
        "label": label,
        "cmd": cmd,
        "returncode": proc.returncode,
        "child_result": child_result,
        "base_ok": base_ok,
        "expected_missing_ok": expected_missing_ok,
        "ok": ok,
        "stdout_tail": proc.stdout.splitlines()[-120:],
        "stderr_tail": proc.stderr.splitlines()[-120:],
    }

def build_stages():
    if args.stage == "plan":
        return [
            ("prepare_strict_forward", cmd_prepare_strict_forward(execute=True)),
            ("strict_forward", cmd_strict_forward(execute=args.execute_heavy)),
            ("residual_generation", cmd_residual_generation(execute=True)),
            ("prepare_adjoint", cmd_prepare_adjoint(execute=True)),
            ("adjoint_sample", cmd_adjoint_batch(args.component, args.batch, execute=args.execute_heavy)),
            ("gradient", cmd_gradient()),
            ("candidates", cmd_candidates()),
        ]

    if args.stage == "preflight":
        return [
            ("prepare_strict_forward_plan", cmd_prepare_strict_forward(execute=False)),
            ("strict_forward_plan", cmd_strict_forward(execute=False)),
            ("residual_generation_plan", cmd_residual_generation(execute=False)),
            ("prepare_adjoint_plan", cmd_prepare_adjoint(execute=False)),
            ("adjoint_sample_plan", cmd_adjoint_batch(args.component, args.batch, execute=False)),
        ]

    if args.stage == "prepare_strict_forward":
        return [("prepare_strict_forward", cmd_prepare_strict_forward(execute=True))]

    if args.stage == "strict_forward":
        return [("strict_forward", cmd_strict_forward(execute=args.execute_heavy))]

    if args.stage == "residual_generation":
        return [("residual_generation", cmd_residual_generation(execute=True))]

    if args.stage == "prepare_adjoint":
        return [("prepare_adjoint", cmd_prepare_adjoint(execute=True))]

    if args.stage == "adjoint_sample":
        return [("adjoint_sample", cmd_adjoint_batch(args.component, args.batch, execute=args.execute_heavy))]

    if args.stage == "adjoint_all":
        stages = []
        for comp in ["x", "y", "z"]:
            for i in range(10):
                stages.append((f"adjoint_{comp}_batch_{i:03d}", cmd_adjoint_batch(comp, f"batch_{i:03d}", execute=args.execute_heavy)))
        return stages

    if args.stage == "gradient":
        return [("gradient", cmd_gradient())]

    if args.stage == "candidates":
        return [("candidates", cmd_candidates())]

    if args.stage == "task5":
        if not args.execute_heavy:
            raise SystemExit("stage task5 requires --execute-heavy")
        if not args.allow_mutate:
            raise SystemExit("stage task5 requires --allow-mutate")
        return [("task5", cmd_task5())]

    if args.stage == "all_existing":
        stages = [
            ("prepare_strict_forward", cmd_prepare_strict_forward(execute=True)),
            ("strict_forward_existing_or_plan", cmd_strict_forward(execute=False)),
            ("residual_generation", cmd_residual_generation(execute=True)),
            ("prepare_adjoint", cmd_prepare_adjoint(execute=True)),
        ]
        for comp in ["x", "y", "z"]:
            for i in range(10):
                stages.append((f"adjoint_{comp}_batch_{i:03d}_existing_or_plan", cmd_adjoint_batch(comp, f"batch_{i:03d}", execute=False)))
        stages.extend([
            ("gradient", cmd_gradient()),
            ("candidates", cmd_candidates()),
        ])
        return stages

    if args.stage == "all_full":
        if not args.execute_heavy:
            raise SystemExit("all_full requires --execute-heavy")
        if not args.allow_mutate:
            raise SystemExit("all_full requires --allow-mutate")
        stages = [
            ("prepare_strict_forward", cmd_prepare_strict_forward(execute=True)),
            ("strict_forward", cmd_strict_forward(execute=True)),
            ("residual_generation", cmd_residual_generation(execute=True)),
            ("prepare_adjoint", cmd_prepare_adjoint(execute=True)),
        ]
        for comp in ["x", "y", "z"]:
            for i in range(10):
                stages.append((f"adjoint_{comp}_batch_{i:03d}", cmd_adjoint_batch(comp, f"batch_{i:03d}", execute=True)))
        stages.extend([
            ("gradient", cmd_gradient()),
            ("candidates", cmd_candidates()),
            ("task5", cmd_task5()),
        ])
        return stages

    raise SystemExit(f"Unhandled stage: {args.stage}")

stages = build_stages()

record = {
    "created": created,
    "transition": transition,
    "iter_k": iter_k,
    "iter_kp1": iter_kp1,
    "context": str(ctx_path),
    "stage": args.stage,
    "np": np_count,
    "execute_heavy": args.execute_heavy,
    "allow_mutate": args.allow_mutate,
    "force": args.force,
    "stage_count": len(stages),
    "runs": [],
    "result": None,
}

lines = []
lines.append("Fathi full context iteration orchestrator")
lines.append("========================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"iter_k = {iter_k}")
lines.append(f"iter_kp1 = {iter_kp1}")
lines.append(f"context = {ctx_path}")
lines.append(f"stage = {args.stage}")
lines.append(f"np = {np_count}")
lines.append(f"execute_heavy = {args.execute_heavy}")
lines.append(f"allow_mutate = {args.allow_mutate}")
lines.append(f"force = {args.force}")
lines.append("")
lines.append("Safety:")
lines.append("  SEM3D heavy execution only happens when --execute-heavy is provided.")
lines.append("  task5/all_full requires --allow-mutate.")
lines.append("")
lines.append("Planned commands:")
for label, cmd in stages:
    lines.append("-" * 100)
    lines.append(label)
    lines.append("  " + " ".join(cmd))

if args.stage == "plan":
    record["result"] = "PASS_PLAN"
    lines.append("")
    lines.append("Plan only. No command was executed.")
else:
    all_ok = True
    for label, cmd in stages:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"RUN {label}")
        lines.append("=" * 100)
        lines.append("  " + " ".join(cmd))

        run = run_cmd(label, cmd)
        record["runs"].append(run)

        lines.append(f"returncode = {run['returncode']}")
        lines.append(f"child_result = {run['child_result']}")
        lines.append(f"base_ok = {run.get('base_ok')}")
        lines.append(f"expected_missing_ok = {run.get('expected_missing_ok')}")
        lines.append(f"ok = {run['ok']}")
        lines.append("stdout tail:")
        for x in run["stdout_tail"]:
            lines.append("  " + x)
        if run["stderr_tail"]:
            lines.append("stderr tail:")
            for x in run["stderr_tail"]:
                lines.append("  " + x)

        if not run["ok"]:
            all_ok = False
            lines.append("")
            lines.append("STOP_ON_FAIL")
            break

    record["result"] = "PASS" if all_ok else "FAIL"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {record['result']}")

out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if record["result"] == "FAIL":
    sys.exit(1)
