from datetime import datetime
import argparse
import json
import subprocess
import sys

from scripts.fathi_benchmark.runtime_paths import (
    repository_root,
    resolve_path,
)


ROOT = repository_root()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument(
    "--stage",
    choices=[
        "plan",
        "status",
        "prerequisites",
        "gradient",
        "regularization",
        "tv_candidates",
        "tv_acceptance_plan",
        "all_light_tv",
        "candidates",
        "task5",
        "all",
    ],
    default="plan",
)
parser.add_argument(
    "--candidate",
    default="line_search_neg_mtilde_1p00MPa",
)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--force-forward", action="store_true")
parser.add_argument(
    "--allow-non-descent",
    action="store_true",
)
parser.add_argument(
    "--overwrite-existing",
    action="store_true",
)
parser.add_argument("--np", type=int, default=12)
parser.add_argument(
    "--config",
    default="benchmark_fathi_strict/config/benchmark_config.json",
)

parser.add_argument("--context", default=None)
parser.add_argument("--tv-config", default=None)
parser.add_argument("--tv-label", default=None)
parser.add_argument("--alpha-lambda", type=float, default=0.0)
parser.add_argument("--alpha-mu", type=float, default=0.0)
parser.add_argument("--candidate-state", default=None)
parser.add_argument("--parent-j-data", type=float, default=None)
parser.add_argument("--candidate-j-data", type=float, default=None)
parser.add_argument("--parent-tv-json", default=None)
parser.add_argument("--candidate-tv-json", default=None)
parser.add_argument(
    "--steps-mpa",
    nargs="+",
    type=float,
    default=[0.10, 0.25, 0.50, 1.00],
)

args = parser.parse_args()

config_path = resolve_path(
    args.config,
    base=ROOT,
)

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

report_dir = resolve_path(
    "benchmark_fathi_strict/reports/run_iteration",
    base=ROOT,
)
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()


def run_cmd(name, cmd):
    print("")
    print("=" * 100)
    print(f"RUN_ITERATION: {name}")
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
        print(f"RUN_ITERATION FAILED AT: {name}")
        print(f"returncode = {proc.returncode}")
        sys.exit(proc.returncode)


def cmd_status():
    return [
        sys.executable,
        "-m",
        (
            "scripts.iteration_engine."
            "write_iteration_stage_report_v2"
        ),
        "--iter-k",
        str(k),
        "--config",
        str(config_path),
    ]


def cmd_prerequisites():
    cmd = [
        sys.executable,
        "-m",
        (
            "scripts.fathi_benchmark."
            "run_task0_prerequisites"
        ),
        "--iter-k",
        str(k),
        "--stage",
        "all",
        "--config",
        str(config_path),
    ]

    if args.execute:
        cmd.append("--execute")

    return cmd


def cmd_gradient():
    cmd = [
        sys.executable,
        "-m",
        (
            "scripts.fathi_benchmark."
            "run_task3_gradient"
        ),
        "--iter-k",
        str(k),
        "--stage",
        "all",
        "--config",
        str(config_path),
    ]

    if args.execute:
        cmd.append("--execute")

    return cmd


def cmd_candidates():
    cmd = [
        sys.executable,
        "-m",
        (
            "scripts.fathi_benchmark."
            "run_task4_candidates"
        ),
        "--iter-k",
        str(k),
        "--stage",
        "all",
        "--config",
        str(config_path),
    ]

    if args.execute:
        cmd.append("--execute")

    return cmd


def cmd_task5():
    cmd = [
        sys.executable,
        "-m",
        (
            "scripts.fathi_benchmark."
            "run_task5_candidate"
        ),
        "--iter-k",
        str(k),
        "--candidate",
        args.candidate,
        "--stage",
        "all",
        "--np",
        str(args.np),
        "--config",
        str(config_path),
    ]

    if args.execute:
        cmd.append("--execute")

    if args.force_forward:
        cmd.append("--force-forward")

    if args.allow_non_descent:
        cmd.append("--allow-non-descent")

    if args.overwrite_existing:
        cmd.append("--overwrite-existing")

    return cmd


def cmd_tv_iteration_stage(stage):
    cmd = [
        sys.executable,
        "-m",
        "scripts.regularization.run_tv_iteration_stage",
        "--iter-k",
        str(k),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--alpha-lambda",
        str(args.alpha_lambda),
        "--alpha-mu",
        str(args.alpha_mu),
        "--steps-mpa",
        *[str(value) for value in args.steps_mpa],
    ]

    if args.context:
        cmd.extend(
            [
                "--context",
                args.context,
            ]
        )

    if args.tv_config:
        cmd.extend(
            [
                "--tv-config",
                args.tv_config,
            ]
        )

    if args.tv_label:
        cmd.extend(
            [
                "--label",
                args.tv_label,
            ]
        )

    if args.candidate_state:
        cmd.extend(
            [
                "--candidate-state",
                args.candidate_state,
            ]
        )

    if args.parent_j_data is not None:
        cmd.extend(
            [
                "--parent-j-data",
                str(args.parent_j_data),
            ]
        )

    if args.candidate_j_data is not None:
        cmd.extend(
            [
                "--candidate-j-data",
                str(args.candidate_j_data),
            ]
        )

    if args.parent_tv_json:
        cmd.extend(
            [
                "--parent-tv-json",
                args.parent_tv_json,
            ]
        )

    if args.candidate_tv_json:
        cmd.extend(
            [
                "--candidate-tv-json",
                args.candidate_tv_json,
            ]
        )

    if args.execute:
        cmd.append("--execute")

    return cmd


plan = {
    "created": created,
    "transition": transition,
    "stage": args.stage,
    "candidate": args.candidate,
    "execute": args.execute,
    "force_forward": args.force_forward,
    "allow_non_descent": args.allow_non_descent,
    "overwrite_existing": args.overwrite_existing,
    "np": args.np,
    "config": str(config_path),
    "scope": {
        "current_engine_type": (
            "resumed_iteration_orchestrator"
        ),
        "all_sequence": [
            "prerequisites_check",
            "gradient",
            "candidates",
            "task5",
            "status",
        ],
        "optional_tv_sequence": [
            "data_rhs",
            "regularization",
            "total_rhs",
            "total_mtilde_gradient",
            "tv_candidates",
            "candidate_forward",
            "total_objective",
            "acceptance",
        ],
        "tv_integration": {
            "inside_iteration": True,
            "position": (
                "after_data_rhs_before_mtilde"
            ),
            "light_stages_launch_sem3d": False,
            "light_stages_mutate_accepted_state": False,
            "physical_descent_requires_candidate_forward": True,
        },
        "not_yet_full_standalone": True,
        "missing_real_executors_for": [
            "strict_forward",
            "residual_generation",
            "prepare_adjoint",
            "run_adjoint_batches",
        ],
    },
    "commands": {
        "status": cmd_status(),
        "prerequisites": cmd_prerequisites(),
        "gradient": cmd_gradient(),
        "regularization": (
            cmd_tv_iteration_stage("regularization")
        ),
        "tv_candidates": (
            cmd_tv_iteration_stage("candidates")
        ),
        "tv_acceptance_plan": (
            cmd_tv_iteration_stage("acceptance-plan")
        ),
        "all_light_tv": (
            cmd_tv_iteration_stage("all-light")
        ),
        "candidates": cmd_candidates(),
        "task5": cmd_task5(),
    },
}

plan_json = (
    report_dir
    / f"{transition}_run_iteration_plan.json"
)

plan_txt = (
    report_dir
    / f"{transition}_run_iteration_plan.txt"
)

lines = []

lines.append(
    "Fathi benchmark run_iteration orchestrator"
)
lines.append(
    "=========================================="
)
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"stage = {args.stage}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"execute = {args.execute}")
lines.append(
    f"force_forward = {args.force_forward}"
)
lines.append(
    f"allow_non_descent = "
    f"{args.allow_non_descent}"
)
lines.append(
    f"overwrite_existing = "
    f"{args.overwrite_existing}"
)
lines.append(f"np = {args.np}")
lines.append(f"config = {config_path}")
lines.append(
    f"alpha_lambda = {args.alpha_lambda}"
)
lines.append(
    f"alpha_mu = {args.alpha_mu}"
)
lines.append("")

lines.append("Scope warning:")
lines.append(
    "  This is currently a resumed iteration orchestrator."
)
lines.append("  --stage all means:")
lines.append(
    "    prerequisites_check -> gradient "
    "-> candidates -> task5 -> status"
)
lines.append(
    "  Optional TV is inserted after data RHS "
    "and before the Mtilde total-gradient solve."
)
lines.append(
    "  TV light stages do not launch SEM3D "
    "and do not mutate the accepted state."
)
lines.append(
    "  Physical descent for a nonzero-TV candidate "
    "still requires a later candidate forward."
)
lines.append("")

lines.append("Canonical stages:")
lines.append(
    "  prerequisites       -> check existing "
    "forward/residual/adjoint outputs"
)
lines.append(
    "  gradient            -> data RHS "
    "+ data-only Mtilde"
)
lines.append(
    "  regularization      -> TV RHS "
    "+ total RHS + total Mtilde"
)
lines.append(
    "  tv_candidates       -> candidates "
    "from the total gradient"
)
lines.append(
    "  tv_acceptance_plan  -> J_data "
    "+ alpha * TV comparison"
)
lines.append(
    "  all_light_tv        -> all TV stages "
    "without SEM3D"
)
lines.append(
    "  candidates          -> data-only "
    "candidate materials"
)
lines.append(
    "  task5               -> candidate forward "
    "+ misfit + acceptance"
)
lines.append("")

lines.append("Commands:")

for name in [
    "prerequisites",
    "status",
    "gradient",
    "regularization",
    "tv_candidates",
    "tv_acceptance_plan",
    "all_light_tv",
    "candidates",
    "task5",
]:
    lines.append("-" * 80)
    lines.append(name)
    lines.append(
        "  " + " ".join(
            plan["commands"][name]
        )
    )

lines.append("")
lines.append("RESULT = PASS_PLAN")

plan_json.write_text(
    json.dumps(plan, indent=2),
    encoding="utf-8",
)

plan_txt.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n".join(lines))

if args.stage == "plan":
    sys.exit(0)

if args.stage == "status":
    run_cmd(
        "status",
        cmd_status(),
    )

elif args.stage == "prerequisites":
    run_cmd(
        "prerequisites",
        cmd_prerequisites(),
    )

elif args.stage == "gradient":
    run_cmd(
        "gradient",
        cmd_gradient(),
    )

elif args.stage == "regularization":
    run_cmd(
        "regularization",
        cmd_tv_iteration_stage(
            "regularization"
        ),
    )

elif args.stage == "tv_candidates":
    run_cmd(
        "tv_candidates",
        cmd_tv_iteration_stage(
            "candidates"
        ),
    )

elif args.stage == "tv_acceptance_plan":
    run_cmd(
        "tv_acceptance_plan",
        cmd_tv_iteration_stage(
            "acceptance-plan"
        ),
    )

elif args.stage == "all_light_tv":
    run_cmd(
        "all_light_tv",
        cmd_tv_iteration_stage(
            "all-light"
        ),
    )

elif args.stage == "candidates":
    run_cmd(
        "candidates",
        cmd_candidates(),
    )

elif args.stage == "task5":
    run_cmd(
        "task5",
        cmd_task5(),
    )

elif args.stage == "all":
    run_cmd(
        "prerequisites",
        cmd_prerequisites(),
    )
    run_cmd(
        "gradient",
        cmd_gradient(),
    )
    run_cmd(
        "candidates",
        cmd_candidates(),
    )
    run_cmd(
        "task5",
        cmd_task5(),
    )
    run_cmd(
        "status",
        cmd_status(),
    )

print("")
print("RESULT = PASS")
