#!/usr/bin/env python3
"""Run and validate the SEM3D forward solver for a generated workspace.

The command is plan-only unless ``--execute`` is supplied. For a fast
end-to-end validation, ``--smoke-seconds`` temporarily shortens ``sim_time``
and enables trace output. The original ``input.spec`` is restored after the
run, including when SEM3D fails or times out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_WORKSPACE_FILES = (
    "input.spec",
    "material.input",
    "material.spec",
    "stations.txt",
    "gaussian_stf.txt",
    "mat/h5/Mat_0_Kappa.h5",
    "mat/h5/Mat_0_Mu.h5",
    "mat/h5/Mat_0_Density.h5",
)
CONTROLLED_OUTPUT_NAMES = ("traces", "snapshots", "res", "prot", "fin_sem")
COMPLETION_MARKER = "fin du calcul sur processeurs"


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


def resolve_executable(value: str, label: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or len(expanded.parts) > 1:
        path = expanded.resolve()
    else:
        found = shutil.which(value)
        if not found:
            raise FileNotFoundError(f"{label} executable not found on PATH: {value}")
        path = Path(found).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} executable not found: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"{label} is not executable: {path}")
    return path


def default_solver_value() -> str:
    return os.environ.get(
        "SEM3D_EXE",
        str(Path.home() / "SEM" / "build" / "SEM3D" / "sem3d.exe"),
    )


def default_mpirun_value() -> str:
    return os.environ.get("MPIEXEC", "mpirun")


def scalar(text: str, key: str) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*=\s*([^;]+);",
        text,
    )
    if not match:
        raise ValueError(f"input.spec field not found: {key}")
    return match.group(1).strip().strip('"')


def parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"input.spec field {key} must be true or false, got: {value}")


def parse_input_settings(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "sim_time_s": float(scalar(text, "sim_time")),
        "save_traces": parse_bool(scalar(text, "save_traces"), "save_traces"),
        "mesh_file": scalar(text, "mesh_file"),
        "mat_file": scalar(text, "mat_file"),
    }


def replace_scalar(text: str, key: str, value: str) -> str:
    updated, count = re.subn(
        rf"(?m)^(\s*{re.escape(key)}\s*=\s*)[^;]+;",
        rf"\g<1>{value};",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"input.spec field was not found exactly once: {key}")
    return updated


def patch_input_for_smoke(path: Path, smoke_seconds: float) -> tuple[bytes, dict[str, Any]]:
    original = path.read_bytes()
    text = original.decode("utf-8")
    text = replace_scalar(text, "sim_time", f"{smoke_seconds:.17g}")
    text = replace_scalar(text, "save_traces", "true")
    path.write_text(text, encoding="utf-8")
    return original, parse_input_settings(path)


def validate_workspace_inputs(
    workspace: Path,
    *,
    mesh_directory: Path,
    mesh_stem: str,
    partition_count: int,
) -> dict[str, Any]:
    require(workspace.is_dir(), f"Workspace directory not found: {workspace}")

    file_sizes: dict[str, int] = {}
    for relative in REQUIRED_WORKSPACE_FILES:
        path = workspace / relative
        require(path.is_file(), f"Required solver input not found: {path}")
        size = path.stat().st_size
        require(size > 0, f"Required solver input is empty: {path}")
        file_sizes[relative] = size

    require(mesh_directory.is_dir(), f"Solver mesh directory not found: {mesh_directory}")
    expected = [
        mesh_directory / f"{mesh_stem}.{index:04d}.h5"
        for index in range(partition_count)
    ]
    missing = [path for path in expected if not path.is_file()]
    empty = [path for path in expected if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        details = [path.name for path in missing + empty]
        raise ValueError(
            "SEM3D mesh partition audit failed before solver execution: "
            + ", ".join(details)
        )

    settings = parse_input_settings(workspace / "input.spec")
    require(
        settings["mesh_file"] == mesh_stem,
        f"input.spec mesh_file={settings['mesh_file']!r} does not match {mesh_stem!r}",
    )

    return {
        "file_sizes": file_sizes,
        "mesh_directory": str(mesh_directory),
        "partition_count": partition_count,
        "input_settings": settings,
    }


def existing_solver_outputs(workspace: Path) -> list[Path]:
    return [workspace / name for name in CONTROLLED_OUTPUT_NAMES if (workspace / name).exists()]


def remove_solver_outputs(workspace: Path) -> list[Path]:
    removed = existing_solver_outputs(workspace)
    for path in removed:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return removed


def trace_files(workspace: Path) -> list[Path]:
    traces = workspace / "traces"
    if not traces.is_dir():
        return []
    return sorted(traces.glob("capteurs*.h5"))


def audit_solver_outputs(
    workspace: Path,
    *,
    expect_traces: bool,
    return_code: int | None,
    timed_out: bool,
) -> dict[str, Any]:
    fin_sem = workspace / "fin_sem"
    fin_sem_text = fin_sem.read_text(encoding="utf-8", errors="replace").strip() if fin_sem.is_file() else ""
    traces = trace_files(workspace)
    empty_traces = [path for path in traces if path.stat().st_size == 0]
    stdout_path = workspace / "logs" / "solver.stdout"
    stdout_text = (
        stdout_path.read_text(encoding="utf-8", errors="replace")
        if stdout_path.is_file()
        else ""
    )
    completion_marker_found = COMPLETION_MARKER in stdout_text.lower()

    execution_ok = return_code in (None, 0) and not timed_out
    traces_ok = (not expect_traces) or (bool(traces) and not empty_traces)
    fin_sem_ok = fin_sem.is_file() and fin_sem_text == "1"

    return {
        "return_code": return_code,
        "timed_out": timed_out,
        "expect_traces": expect_traces,
        "trace_count": len(traces),
        "trace_files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in traces
        ],
        "empty_trace_files": [path.name for path in empty_traces],
        "fin_sem": {
            "exists": fin_sem.is_file(),
            "bytes": fin_sem.stat().st_size if fin_sem.is_file() else 0,
            "value": fin_sem_text,
        },
        "completion_marker_found": completion_marker_found,
        "passed": execution_ok and traces_ok and fin_sem_ok,
    }


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_solver(
    *,
    command: list[str],
    workspace: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> tuple[int, bool, float]:
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            return_code = int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = 124
    return return_code, timed_out, time.perf_counter() - started


def write_manifest(
    workspace: Path,
    *,
    config: Path,
    solver: Path,
    mpirun: Path,
    command: list[str],
    process_count: int,
    timeout_seconds: float,
    smoke_seconds: float | None,
    original_input: dict[str, Any],
    effective_input: dict[str, Any],
    input_audit: dict[str, Any],
    removed_outputs: list[Path],
    return_code: int,
    timed_out: bool,
    elapsed_seconds: float,
    output_audit: dict[str, Any],
) -> Path:
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = logs / "solver_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "config": str(config),
        "solver_executable": str(solver),
        "mpirun_executable": str(mpirun),
        "command": command,
        "working_directory": str(workspace),
        "process_count": process_count,
        "timeout_seconds": timeout_seconds,
        "smoke_seconds": smoke_seconds,
        "input_spec_restored_after_run": True,
        "original_input_settings": original_input,
        "effective_input_settings": effective_input,
        "input_audit": input_audit,
        "removed_stale_outputs": [str(path.relative_to(workspace)) for path in removed_outputs],
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_seconds,
        "stdout": "logs/solver.stdout",
        "stderr": "logs/solver.stderr",
        "output_audit": output_audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def expected_traces_for_audit(workspace: Path, current_setting: bool) -> bool:
    manifest_path = workspace / "logs" / "solver_manifest.json"
    if not manifest_path.is_file():
        return current_setting
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return bool(manifest["effective_input_settings"]["save_traces"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return current_setting


def print_plan(
    *,
    root: Path,
    config: Path,
    workspace: Path,
    solver: Path | str,
    mpirun: Path | str,
    process_count: int,
    timeout_seconds: float,
    smoke_seconds: float | None,
    execute: bool,
    overwrite: bool,
) -> None:
    print("SEM3D SOLVER RUNNER")
    print("===================")
    print()
    print(f"root = {root}")
    print(f"config = {config}")
    print(f"workspace = {workspace}")
    print(f"solver = {solver}")
    print(f"mpirun = {mpirun}")
    print(f"MPI processes = {process_count}")
    print(f"timeout seconds = {timeout_seconds:g}")
    print(f"smoke seconds = {smoke_seconds if smoke_seconds is not None else 'disabled'}")
    print(f"overwrite stale outputs = {overwrite}")
    print(f"mode = {'EXECUTE' if execute else 'PLAN ONLY'}")
    print()
    print("effective invocation:")
    print(f"  cd {workspace}")
    print(f"  {mpirun} -np {process_count} {solver}")
    if smoke_seconds is not None:
        print(f"  temporary input.spec override: sim_time={smoke_seconds:g}, save_traces=true")
        print("  restore original input.spec after execution")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fathi_reduced_3x3_12p5.json",
        help="Benchmark JSON specification, relative to repository root by default.",
    )
    parser.add_argument("--workspace", required=True, help="Meshed SEM3D workspace.")
    parser.add_argument(
        "--solver",
        default=None,
        help="SEM3D executable. Defaults to SEM3D_EXE or ~/SEM/build/SEM3D/sem3d.exe.",
    )
    parser.add_argument(
        "--mpirun",
        default=None,
        help="MPI launcher. Defaults to MPIEXEC or mpirun on PATH.",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=None,
        help="MPI process count. Defaults to sem3d_mesh.partition_count.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum solver runtime. Default: 900 seconds.",
    )
    parser.add_argument(
        "--smoke-seconds",
        type=float,
        default=None,
        help=(
            "Temporarily set sim_time to this value and enable traces. "
            "The original input.spec is restored after the run."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Run SEM3D.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing solver outputs before execution.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit existing solver outputs without running SEM3D.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    config_path = resolve_path(root, args.config)
    workspace = resolve_path(root, args.workspace)
    spec = load_json(config_path)

    mesh_cfg = spec["sem3d_mesh"]
    mesh_stem = str(mesh_cfg["mesh_file_stem"])
    partition_count = int(mesh_cfg["partition_count"])
    mesh_directory = workspace / str(mesh_cfg.get("output_directory", "sem"))
    process_count = args.np if args.np is not None else partition_count

    solver_value = args.solver or default_solver_value()
    mpirun_value = args.mpirun or default_mpirun_value()

    require(process_count > 0, "--np must be positive")
    require(args.timeout_seconds > 0, "--timeout-seconds must be positive")
    if args.smoke_seconds is not None:
        require(args.smoke_seconds > 0, "--smoke-seconds must be positive")
    require(
        not (args.execute and args.audit_only),
        "--execute and --audit-only are mutually exclusive",
    )
    require(
        not (args.audit_only and args.smoke_seconds is not None),
        "--smoke-seconds cannot be used with --audit-only",
    )

    print_plan(
        root=root,
        config=config_path,
        workspace=workspace,
        solver=solver_value,
        mpirun=mpirun_value,
        process_count=process_count,
        timeout_seconds=args.timeout_seconds,
        smoke_seconds=args.smoke_seconds,
        execute=args.execute,
        overwrite=args.overwrite,
    )

    input_audit = validate_workspace_inputs(
        workspace,
        mesh_directory=mesh_directory,
        mesh_stem=mesh_stem,
        partition_count=partition_count,
    )
    original_input = dict(input_audit["input_settings"])

    print()
    print("validated solver inputs:")
    print(f"  mesh directory = {mesh_directory}")
    print(f"  partitions = {input_audit['partition_count']}")
    print(f"  sim_time = {original_input['sim_time_s']}")
    print(f"  save_traces = {original_input['save_traces']}")

    if args.audit_only:
        expect_traces = expected_traces_for_audit(
            workspace,
            original_input["save_traces"],
        )
        audit = audit_solver_outputs(
            workspace,
            expect_traces=expect_traces,
            return_code=None,
            timed_out=False,
        )
        print()
        print(f"expect_traces = {expect_traces}")
        print(f"trace_count = {audit['trace_count']}")
        print(f"fin_sem = {audit['fin_sem']}")
        print(f"audit_passed = {audit['passed']}")
        if audit["passed"]:
            print("RESULT = PASS_SEM3D_SOLVER_OUTPUT_AUDIT")
            return 0
        print(json.dumps(audit, indent=2, sort_keys=True))
        print("RESULT = FAIL_SEM3D_SOLVER_OUTPUT_AUDIT")
        return 1

    if not args.execute:
        print()
        print("No files were changed. Add --execute to run SEM3D.")
        print("RESULT = PASS_SEM3D_SOLVER_PLAN")
        return 0

    solver = resolve_executable(str(solver_value), "SEM3D solver")
    mpirun = resolve_executable(str(mpirun_value), "MPI launcher")
    command = [str(mpirun), "-np", str(process_count), str(solver)]

    stale = existing_solver_outputs(workspace)
    if stale and not args.overwrite:
        names = ", ".join(path.name for path in stale)
        raise FileExistsError(
            "Solver outputs already exist. Refusing to mix stale and fresh files. "
            f"Use --overwrite to remove them first. Existing: {names}"
        )
    removed_outputs = remove_solver_outputs(workspace) if stale else []

    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "solver.stdout"
    stderr_path = logs / "solver.stderr"

    original_bytes: bytes | None = None
    effective_input = original_input
    return_code = -1
    timed_out = False
    elapsed = 0.0

    print()
    print("Running SEM3D solver...")
    try:
        if args.smoke_seconds is not None:
            original_bytes, effective_input = patch_input_for_smoke(
                workspace / "input.spec",
                args.smoke_seconds,
            )
        return_code, timed_out, elapsed = run_solver(
            command=command,
            workspace=workspace,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=args.timeout_seconds,
        )
    finally:
        if original_bytes is not None:
            (workspace / "input.spec").write_bytes(original_bytes)

    output_audit = audit_solver_outputs(
        workspace,
        expect_traces=bool(effective_input["save_traces"]),
        return_code=return_code,
        timed_out=timed_out,
    )
    manifest_path = write_manifest(
        workspace,
        config=config_path,
        solver=solver,
        mpirun=mpirun,
        command=command,
        process_count=process_count,
        timeout_seconds=args.timeout_seconds,
        smoke_seconds=args.smoke_seconds,
        original_input=original_input,
        effective_input=effective_input,
        input_audit=input_audit,
        removed_outputs=removed_outputs,
        return_code=return_code,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
        output_audit=output_audit,
    )

    print(f"return_code = {return_code}")
    print(f"timed_out = {timed_out}")
    print(f"elapsed_seconds = {elapsed:.3f}")
    print(f"stdout = {stdout_path}")
    print(f"stderr = {stderr_path}")
    print(f"manifest = {manifest_path}")
    print(f"input_spec_restored = {parse_input_settings(workspace / 'input.spec') == original_input}")
    print(f"trace_count = {output_audit['trace_count']}")
    print(f"fin_sem = {output_audit['fin_sem']}")

    if timed_out:
        print("RESULT = FAIL_SEM3D_SOLVER_TIMEOUT")
        return 124
    if return_code != 0:
        print("RESULT = FAIL_SEM3D_SOLVER_EXECUTION")
        return return_code if 0 < return_code < 256 else 1
    if not output_audit["passed"]:
        print(json.dumps(output_audit, indent=2, sort_keys=True))
        print("RESULT = FAIL_SEM3D_SOLVER_OUTPUT_AUDIT")
        return 1

    print("RESULT = PASS_SEM3D_SOLVER_EXECUTION_AND_AUDIT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
