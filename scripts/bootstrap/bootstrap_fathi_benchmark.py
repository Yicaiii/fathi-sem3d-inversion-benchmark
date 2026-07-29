#!/usr/bin/env python3
"""Bootstrap the reduced Fathi SEM3D benchmark from committed configuration.

This is the public from-scratch bootstrap entry point for the benchmark assets.
It orchestrates the three lower-level tools in this order for the selected
material models:

1. generate_sem3d_workspace.py
2. run_sem3d_mesher.py
3. run_sem3d_solver.py
4. read-only mesher and solver audits

The command is plan-only unless ``--execute`` is supplied. A smoke run can be
requested with ``--smoke-seconds``; the lower-level solver runner temporarily
modifies ``input.spec`` and restores it after execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_DIRECTORY = {
    "true_layered": "true_layered",
    "initial_homogeneous": "initial_homogeneous",
}


def repository_root() -> Path:
    default_root = Path(__file__).resolve().parents[2]
    return Path(
        os.environ.get("FATHI_BENCHMARK_ROOT", str(default_root))
    ).expanduser().resolve()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be an object: {path}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def selected_models(value: str) -> list[str]:
    if value == "both":
        return ["true_layered", "initial_homogeneous"]
    return [value]


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def run_command(command: list[str], *, cwd: Path) -> tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, check=False)
    return int(completed.returncode), time.perf_counter() - started


def script_paths(root: Path) -> dict[str, Path]:
    paths = {
        "generator": root / "scripts" / "bootstrap" / "generate_sem3d_workspace.py",
        "mesher": root / "scripts" / "bootstrap" / "run_sem3d_mesher.py",
        "solver": root / "scripts" / "bootstrap" / "run_sem3d_solver.py",
    }
    for label, path in paths.items():
        require(path.is_file(), f"Bootstrap {label} script not found: {path}")
    return paths


def build_step_commands(
    *,
    root: Path,
    config: Path,
    output_root: Path,
    models: list[str],
    smoke_seconds: float | None,
    timeout_seconds: float,
    process_count: int | None,
    overwrite: bool,
    audit_only: bool,
) -> list[dict[str, Any]]:
    scripts = script_paths(root)
    python = sys.executable
    steps: list[dict[str, Any]] = []

    for model in models:
        workspace = output_root / MODEL_DIRECTORY[model]

        if audit_only:
            mesher_audit = [
                python,
                str(scripts["mesher"]),
                "--config",
                str(config),
                "--workspace",
                str(workspace),
                "--audit-only",
            ]
            solver_audit = [
                python,
                str(scripts["solver"]),
                "--config",
                str(config),
                "--workspace",
                str(workspace),
                "--audit-only",
            ]
            steps.extend(
                [
                    {
                        "model": model,
                        "stage": "mesher_audit",
                        "workspace": str(workspace),
                        "command": mesher_audit,
                    },
                    {
                        "model": model,
                        "stage": "solver_audit",
                        "workspace": str(workspace),
                        "command": solver_audit,
                    },
                ]
            )
            continue

        generator = [
            python,
            str(scripts["generator"]),
            "--config",
            str(config),
            "--model",
            model,
            "--output",
            str(workspace),
            "--write",
        ]
        mesher = [
            python,
            str(scripts["mesher"]),
            "--config",
            str(config),
            "--workspace",
            str(workspace),
            "--execute",
        ]
        solver = [
            python,
            str(scripts["solver"]),
            "--config",
            str(config),
            "--workspace",
            str(workspace),
            "--timeout-seconds",
            f"{timeout_seconds:.17g}",
            "--execute",
        ]
        if smoke_seconds is not None:
            solver.extend(["--smoke-seconds", f"{smoke_seconds:.17g}"])
        if process_count is not None:
            solver.extend(["--np", str(process_count)])
        if overwrite:
            mesher.append("--overwrite")
            solver.append("--overwrite")

        mesher_audit = [
            python,
            str(scripts["mesher"]),
            "--config",
            str(config),
            "--workspace",
            str(workspace),
            "--audit-only",
        ]
        solver_audit = [
            python,
            str(scripts["solver"]),
            "--config",
            str(config),
            "--workspace",
            str(workspace),
            "--audit-only",
        ]

        for stage, command in (
            ("generate", generator),
            ("mesh", mesher),
            ("solve", solver),
            ("mesher_audit", mesher_audit),
            ("solver_audit", solver_audit),
        ):
            steps.append(
                {
                    "model": model,
                    "stage": stage,
                    "workspace": str(workspace),
                    "command": command,
                }
            )

    return steps


def write_manifest(
    output_root: Path,
    *,
    config: Path,
    profile_name: str,
    models: list[str],
    smoke_seconds: float | None,
    timeout_seconds: float,
    process_count: int | None,
    audit_only: bool,
    status: str,
    steps: list[dict[str, Any]],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "bootstrap_manifest.json"
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "config": str(config),
        "output_root": str(output_root),
        "models": models,
        "smoke_seconds": smoke_seconds,
        "timeout_seconds": timeout_seconds,
        "process_count_override": process_count,
        "audit_only": audit_only,
        "status": status,
        "steps": steps,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def print_plan(
    *,
    root: Path,
    config: Path,
    output_root: Path,
    profile_name: str,
    models: list[str],
    smoke_seconds: float | None,
    timeout_seconds: float,
    process_count: int | None,
    audit_only: bool,
    execute: bool,
    overwrite: bool,
    steps: list[dict[str, Any]],
) -> None:
    print("FATHI SEM3D BENCHMARK BOOTSTRAP")
    print("================================")
    print()
    print(f"root = {root}")
    print(f"profile = {profile_name}")
    print(f"config = {config}")
    print(f"output root = {output_root}")
    print(f"models = {', '.join(models)}")
    print(f"smoke seconds = {smoke_seconds if smoke_seconds is not None else 'disabled'}")
    print(f"timeout seconds = {timeout_seconds:g}")
    print(f"MPI process override = {process_count if process_count is not None else 'from config'}")
    print(f"audit only = {audit_only}")
    print(f"overwrite = {overwrite}")
    print(f"mode = {'EXECUTE' if execute else 'PLAN ONLY'}")
    print()
    print("planned steps:")
    for index, step in enumerate(steps, start=1):
        print(
            f"  {index:02d}. {step['model']:20s} "
            f"{step['stage']:14s} {command_text(step['command'])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fathi_reduced_3x3_12p5.json",
        help="Committed benchmark JSON specification.",
    )
    parser.add_argument(
        "--output-root",
        default="data/bootstrap/fathi_reduced_3x3_12p5",
        help="Root directory containing true_layered and initial_homogeneous.",
    )
    parser.add_argument(
        "--model",
        choices=("both", "true_layered", "initial_homogeneous"),
        default="both",
        help="Bootstrap both models or only one selected model.",
    )
    parser.add_argument(
        "--smoke-seconds",
        type=float,
        default=None,
        help="Run both solvers with a temporary short simulation and traces enabled.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum runtime for each SEM3D solver execution.",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=None,
        help="Optional MPI process-count override; otherwise use the config value.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the complete bootstrap. Without this flag, print the plan only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before a fresh execution.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run read-only mesher and solver audits on existing workspaces.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    config = resolve_path(root, args.config)
    output_root = resolve_path(root, args.output_root)
    spec = load_json(config)
    profile_name = str(spec.get("name", config.stem))
    models = selected_models(args.model)

    require(args.timeout_seconds > 0, "--timeout-seconds must be positive")
    if args.smoke_seconds is not None:
        require(args.smoke_seconds > 0, "--smoke-seconds must be positive")
    if args.np is not None:
        require(args.np > 0, "--np must be positive")
    require(
        not (args.audit_only and args.execute),
        "--audit-only and --execute are mutually exclusive",
    )
    require(
        not (args.audit_only and args.smoke_seconds is not None),
        "--smoke-seconds cannot be used with --audit-only",
    )
    require(
        not (args.audit_only and args.overwrite),
        "--overwrite cannot be used with --audit-only",
    )

    steps = build_step_commands(
        root=root,
        config=config,
        output_root=output_root,
        models=models,
        smoke_seconds=args.smoke_seconds,
        timeout_seconds=args.timeout_seconds,
        process_count=args.np,
        overwrite=args.overwrite,
        audit_only=args.audit_only,
    )
    print_plan(
        root=root,
        config=config,
        output_root=output_root,
        profile_name=profile_name,
        models=models,
        smoke_seconds=args.smoke_seconds,
        timeout_seconds=args.timeout_seconds,
        process_count=args.np,
        audit_only=args.audit_only,
        execute=args.execute,
        overwrite=args.overwrite,
        steps=steps,
    )

    if args.audit_only:
        require(output_root.is_dir(), f"Bootstrap output root not found: {output_root}")
    elif not args.execute:
        print()
        print("PLAN ONLY: no files were changed. Add --execute to run the bootstrap.")
        print("RESULT = PASS_FATHI_BOOTSTRAP_PLAN")
        return 0
    else:
        if output_root.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Bootstrap output root already exists: {output_root}\n"
                    "Use --overwrite only after checking the existing outputs."
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=False)

    executed_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        print()
        print("=" * 96)
        print(
            f"STEP {index}/{len(steps)}: "
            f"{step['model']} / {step['stage']}"
        )
        print("=" * 96)
        print(command_text(step["command"]))

        return_code, elapsed = run_command(step["command"], cwd=root)
        record = {
            "model": step["model"],
            "stage": step["stage"],
            "workspace": step["workspace"],
            "command": step["command"],
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "passed": return_code == 0,
        }
        executed_steps.append(record)
        print(f"bootstrap_step_return_code = {return_code}")
        print(f"bootstrap_step_elapsed_seconds = {elapsed:.3f}")

        if return_code != 0:
            manifest = write_manifest(
                output_root,
                config=config,
                profile_name=profile_name,
                models=models,
                smoke_seconds=args.smoke_seconds,
                timeout_seconds=args.timeout_seconds,
                process_count=args.np,
                audit_only=args.audit_only,
                status="failed",
                steps=executed_steps,
            )
            print(f"bootstrap_manifest = {manifest}")
            print("RESULT = FAIL_FATHI_BOOTSTRAP")
            return return_code if 0 < return_code < 256 else 1

    manifest = write_manifest(
        output_root,
        config=config,
        profile_name=profile_name,
        models=models,
        smoke_seconds=args.smoke_seconds,
        timeout_seconds=args.timeout_seconds,
        process_count=args.np,
        audit_only=args.audit_only,
        status="passed",
        steps=executed_steps,
    )

    print()
    print("FATHI BOOTSTRAP SUMMARY")
    print("=======================")
    print(f"models completed = {', '.join(models)}")
    print(f"step count = {len(executed_steps)}")
    print(f"bootstrap_manifest = {manifest}")
    if args.audit_only:
        print("RESULT = PASS_FATHI_BOOTSTRAP_AUDIT")
    else:
        print("RESULT = PASS_FATHI_BOOTSTRAP_EXECUTION_AND_AUDIT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
