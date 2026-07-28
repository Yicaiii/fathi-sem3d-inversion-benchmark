from datetime import datetime
from importlib.util import find_spec
import argparse
import json
import subprocess
import sys

from scripts.fathi_benchmark.runtime_paths import repository_root, resolve_path


SOLVER_MODULE = (
    "scripts.longterm."
    "426C_solve_mtilde_interior_rhs_total"
)

DEFAULT_MATRIX_PATH = (
    "results/audit_teacher_feedback/"
    "iter005_mass_matrix/"
    "Mtilde_q1_consistent_sparse.npz"
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--iter-k",
    type=int,
    required=True,
)
parser.add_argument(
    "--execute",
    action="store_true",
)
parser.add_argument(
    "--config",
    default=(
        "benchmark_fathi_strict/config/"
        "benchmark_config.json"
    ),
)
args = parser.parse_args()


ROOT = repository_root()

config_path = resolve_path(
    args.config,
    base=ROOT,
)

if not config_path.exists():
    print(f"Missing config: {config_path}")
    sys.exit(1)

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_result_root = (
    resolve_path(
        config["run_result_root"],
        base=ROOT,
    )
    / transition
)

component_rhs_dir = (
    run_result_root
    / "component_rhs"
)

mtilde_solve_dir = (
    run_result_root
    / "mtilde_solve"
)

report_dir = resolve_path(
    "benchmark_fathi_strict/reports/mtilde_run",
    base=ROOT,
)

matrix_path = resolve_path(
    config.get(
        "mtilde_matrix_path",
        DEFAULT_MATRIX_PATH,
    ),
    base=ROOT,
)

mtilde_solve_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_dir.mkdir(
    parents=True,
    exist_ok=True,
)

required_inputs = [
    (
        component_rhs_dir
        / "full_grid_trace_RHS_total_lambda.npy"
    ),
    (
        component_rhs_dir
        / "full_grid_trace_RHS_total_mu.npy"
    ),
    (
        component_rhs_dir
        / "full_grid_trace_RHS_total_coords.npy"
    ),
    (
        component_rhs_dir
        / "full_grid_trace_RHS_total_summary.txt"
    ),
    matrix_path,
]

missing = [
    path
    for path in required_inputs
    if not path.exists()
]

solver_available = (
    find_spec(SOLVER_MODULE)
    is not None
)

created = datetime.now().isoformat()

record = {
    "created": created,
    "transition": transition,
    "execute": args.execute,
    "component_rhs_dir": str(
        component_rhs_dir
    ),
    "mtilde_solve_dir": str(
        mtilde_solve_dir
    ),
    "matrix_path": str(matrix_path),
    "solver_module": SOLVER_MODULE,
    "solver_available": solver_available,
    "missing": [
        str(path)
        for path in missing
    ],
    "result": None,
}

lines = [
    "Task 3J generic Mtilde solve",
    "============================",
    "",
    f"created = {created}",
    f"transition = {transition}",
    f"execute = {args.execute}",
    "",
    (
        "component_rhs_dir = "
        f"{component_rhs_dir}"
    ),
    (
        "mtilde_solve_dir = "
        f"{mtilde_solve_dir}"
    ),
    f"matrix_path = {matrix_path}",
    f"solver_module = {SOLVER_MODULE}",
    (
        "solver_available = "
        f"{solver_available}"
    ),
    "",
]

if not solver_available:
    lines.extend([
        (
            "Solver module was not found: "
            f"{SOLVER_MODULE}"
        ),
        "",
        "RESULT = FAIL_MISSING_SOLVER_MODULE",
    ])

    record["result"] = (
        "FAIL_MISSING_SOLVER_MODULE"
    )

elif missing:
    lines.append(
        "Missing required inputs:"
    )

    for path in missing:
        lines.append(f"  {path}")

    lines.extend([
        "",
        "RESULT = FAIL_MISSING_INPUTS",
    ])

    record["result"] = (
        "FAIL_MISSING_INPUTS"
    )

else:
    rhs_summary = (
        component_rhs_dir
        / "full_grid_trace_RHS_total_summary.txt"
    )

    rhs_summary_text = rhs_summary.read_text(
        errors="ignore",
    )

    if "RESULT = PASS" not in rhs_summary_text:
        lines.extend([
            (
                "RHS_total summary does not "
                "contain RESULT = PASS:"
            ),
            f"  {rhs_summary}",
            "",
            "RESULT = FAIL_BAD_RHS_TOTAL",
        ])

        record["result"] = (
            "FAIL_BAD_RHS_TOTAL"
        )

    else:
        cmd = [
            sys.executable,
            "-m",
            SOLVER_MODULE,
            "--matrix",
            str(matrix_path),
            "--rhs-dir",
            str(component_rhs_dir),
            "--out-dir",
            str(mtilde_solve_dir),
        ]

        record["command"] = cmd

        lines.extend([
            "Solver command:",
            "  " + " ".join(cmd),
            "",
        ])

        if not args.execute:
            lines.extend([
                "DRY RUN ONLY.",
                "No Mtilde solve was launched.",
                "",
                "RESULT = PASS_DRYRUN",
            ])

            record["result"] = (
                "PASS_DRYRUN"
            )

        else:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            record["returncode"] = (
                proc.returncode
            )
            record["stdout_tail"] = (
                proc.stdout.splitlines()[-120:]
            )
            record["stderr_tail"] = (
                proc.stderr.splitlines()[-120:]
            )

            lines.append("stdout tail:")
            lines.extend(
                "  " + line
                for line in record["stdout_tail"]
            )
            lines.append("")

            lines.append("stderr tail:")
            lines.extend(
                "  " + line
                for line in record["stderr_tail"]
            )
            lines.append("")

            summary = (
                mtilde_solve_dir
                / (
                    "mtilde_q1_interior_solve_"
                    "rhs_total_summary.txt"
                )
            )

            outputs = [
                (
                    mtilde_solve_dir
                    / (
                        "Mtilde_q1_consistent_"
                        "interior_38440_sparse.npz"
                    )
                ),
                (
                    mtilde_solve_dir
                    / (
                        "Mtilde_q1_consistent_"
                        "interior_38440_indices.npy"
                    )
                ),
                (
                    mtilde_solve_dir
                    / (
                        "Mtilde_q1_consistent_"
                        "interior_38440_coords.npy"
                    )
                ),
                (
                    mtilde_solve_dir
                    / (
                        "g_lambda_mtilde_q1_"
                        "interior_solve_rhs_total.npy"
                    )
                ),
                (
                    mtilde_solve_dir
                    / (
                        "g_mu_mtilde_q1_"
                        "interior_solve_rhs_total.npy"
                    )
                ),
                (
                    mtilde_solve_dir
                    / (
                        "g_mtilde_q1_interior_"
                        "solve_rhs_total_coords.npy"
                    )
                ),
                summary,
            ]

            output_missing = [
                path
                for path in outputs
                if not path.exists()
            ]

            record["output_missing"] = [
                str(path)
                for path in output_missing
            ]

            summary_ok = False

            if summary.exists():
                summary_text = summary.read_text(
                    errors="ignore",
                )
                summary_ok = (
                    "RESULT = PASS"
                    in summary_text
                )

            record["summary_ok"] = (
                summary_ok
            )

            if (
                proc.returncode == 0
                and not output_missing
                and summary_ok
            ):
                record["result"] = "PASS"
            else:
                record["result"] = "FAIL"

            lines.extend([
                (
                    "returncode = "
                    f"{proc.returncode}"
                ),
                (
                    "output_missing = "
                    f"{len(output_missing)}"
                ),
                (
                    "summary_ok = "
                    f"{summary_ok}"
                ),
                "",
                (
                    "RESULT = "
                    f"{record['result']}"
                ),
            ])

out_json = (
    report_dir
    / f"{transition}_solve_mtilde_generic.json"
)

out_txt = (
    report_dir
    / f"{transition}_solve_mtilde_generic.txt"
)

out_json.write_text(
    json.dumps(
        record,
        indent=2,
    ),
    encoding="utf-8",
)

out_txt.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))

if record["result"] not in [
    "PASS",
    "PASS_DRYRUN",
]:
    sys.exit(1)
