#!/usr/bin/env python3
"""Run, organize, and validate the SEM3D mesher for a benchmark workspace.

The command is plan-only unless ``--execute`` is supplied. The mesher reads
``mesh.input`` from standard input and reads ``mat.dat`` and ``mater.in`` from
the workspace working directory. SEM3D's mesher writes its files initially in
the workspace root; this runner then moves the complete mesh product into the
configured solver mesh directory (``sem/`` for this benchmark).
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


REQUIRED_INPUTS = ("mesh.input", "mat.dat", "mater.in")
MASTER_XMF_FILES = (
    "mesh4spec.elems.xmf",
    "mesh4spec.faces.xmf",
    "mesh4spec.edges.xmf",
    "mesh4spec.mirror.xmf",
    "mesh4spec.comms.faces.xmf",
    "mesh4spec.comms.edges.xmf",
)


def repository_root() -> Path:
    """Return the repository root, honoring ``FATHI_BENCHMARK_ROOT``."""
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


def default_mesher_path() -> Path:
    configured = os.environ.get("SEM3D_MESHER")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "SEM" / "build" / "MESH" / "mesher").resolve()


def resolve_mesh_output_directory(workspace: Path, configured: str) -> Path:
    relative = Path(configured)
    require(not relative.is_absolute(), "sem3d_mesh.output_directory must be relative")
    require(".." not in relative.parts, "sem3d_mesh.output_directory may not contain '..'")
    require(relative.parts, "sem3d_mesh.output_directory may not be empty")
    return workspace / relative


def expected_partition_paths(
    output_directory: Path,
    stem: str,
    partition_count: int,
) -> list[Path]:
    return [
        output_directory / f"{stem}.{index:04d}.h5"
        for index in range(partition_count)
    ]


def controlled_outputs(directory: Path, stem: str) -> list[Path]:
    candidates: set[Path] = set()
    candidates.update(directory.glob(f"{stem}*.h5"))
    candidates.update(directory.glob(f"{stem}*.xmf"))
    candidates.add(directory / "domains.txt")
    return sorted(path for path in candidates if path.exists())


def existing_mesher_outputs(
    workspace: Path,
    output_directory: Path,
    stem: str,
) -> list[Path]:
    candidates = set(controlled_outputs(workspace, stem))
    if output_directory != workspace:
        candidates.update(controlled_outputs(output_directory, stem))
    return sorted(candidates)


def remove_mesher_outputs(
    workspace: Path,
    output_directory: Path,
    stem: str,
) -> list[Path]:
    removed = existing_mesher_outputs(workspace, output_directory, stem)
    for path in removed:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if output_directory != workspace and output_directory.is_dir():
        try:
            output_directory.rmdir()
        except OSError:
            pass
    return removed


def relocate_mesher_outputs(
    workspace: Path,
    output_directory: Path,
    stem: str,
) -> list[Path]:
    """Move root-level mesher products into the SEM3D solver mesh directory."""
    if output_directory == workspace:
        return []

    root_outputs = controlled_outputs(workspace, stem)
    require(root_outputs, "Mesher returned success but produced no controlled outputs")
    output_directory.mkdir(parents=True, exist_ok=True)

    relocated: list[Path] = []
    for source in root_outputs:
        destination = output_directory / source.name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing organized mesh output: {destination}"
            )
        shutil.move(str(source), str(destination))
        relocated.append(destination)
    return relocated


def validate_required_inputs(workspace: Path) -> dict[str, int]:
    require(workspace.is_dir(), f"Workspace directory not found: {workspace}")
    sizes: dict[str, int] = {}
    for name in REQUIRED_INPUTS:
        path = workspace / name
        require(path.is_file(), f"Required mesher input not found: {path}")
        size = path.stat().st_size
        require(size > 0, f"Required mesher input is empty: {path}")
        sizes[name] = size
    return sizes


def audit_mesher_outputs(
    workspace: Path,
    output_directory: Path,
    stem: str,
    partition_count: int,
) -> dict[str, Any]:
    expected = expected_partition_paths(output_directory, stem, partition_count)
    missing = [path for path in expected if not path.is_file()]
    empty = [path for path in expected if path.is_file() and path.stat().st_size == 0]

    actual = sorted(output_directory.glob(f"{stem}.[0-9][0-9][0-9][0-9].h5"))
    expected_names = {path.name for path in expected}
    unexpected = [path for path in actual if path.name not in expected_names]

    domains = output_directory / "domains.txt"
    missing_master_xmf = [
        output_directory / name
        for name in MASTER_XMF_FILES
        if not (output_directory / name).is_file()
        or (output_directory / name).stat().st_size == 0
    ]
    stray_root_outputs = (
        controlled_outputs(workspace, stem)
        if output_directory != workspace
        else []
    )

    return {
        "output_directory": str(output_directory),
        "partition_count_expected": partition_count,
        "partition_count_actual": len(actual),
        "partition_files": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in actual
        ],
        "missing_partitions": [path.name for path in missing],
        "empty_partitions": [path.name for path in empty],
        "unexpected_partitions": [path.name for path in unexpected],
        "domains_txt": {
            "exists": domains.is_file(),
            "bytes": domains.stat().st_size if domains.is_file() else 0,
        },
        "missing_or_empty_master_xmf": [path.name for path in missing_master_xmf],
        "stray_root_outputs": [path.name for path in stray_root_outputs],
        "passed": (
            output_directory.is_dir()
            and len(actual) == partition_count
            and not missing
            and not empty
            and not unexpected
            and domains.is_file()
            and domains.stat().st_size > 0
            and not missing_master_xmf
            and not stray_root_outputs
        ),
    }


def write_manifest(
    workspace: Path,
    *,
    config: Path,
    mesher: Path,
    output_directory: Path,
    timeout_seconds: float,
    return_code: int,
    elapsed_seconds: float,
    input_sizes: dict[str, int],
    removed_outputs: list[Path],
    relocated_outputs: list[Path],
    audit: dict[str, Any],
) -> Path:
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = logs / "mesher_manifest.json"
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "config": str(config),
        "mesher_executable": str(mesher),
        "command": [str(mesher)],
        "stdin": "mesh.input",
        "working_directory": str(workspace),
        "solver_mesh_directory": str(output_directory),
        "timeout_seconds": timeout_seconds,
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "input_sizes": input_sizes,
        "removed_stale_outputs": [str(path.relative_to(workspace)) for path in removed_outputs],
        "relocated_outputs": [str(path.relative_to(workspace)) for path in relocated_outputs],
        "audit": audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def print_plan(
    *,
    root: Path,
    config: Path,
    workspace: Path,
    mesher: Path,
    output_directory: Path,
    stem: str,
    partition_count: int,
    timeout_seconds: float,
    execute: bool,
    overwrite: bool,
) -> None:
    print("SEM3D MESHER RUNNER")
    print("===================")
    print()
    print(f"root = {root}")
    print(f"config = {config}")
    print(f"workspace = {workspace}")
    print(f"mesher = {mesher}")
    print(f"mesh stem = {stem}")
    print(f"solver mesh directory = {output_directory}")
    print(f"expected partitions = {partition_count}")
    print(f"timeout seconds = {timeout_seconds:g}")
    print(f"overwrite stale outputs = {overwrite}")
    print(f"mode = {'EXECUTE' if execute else 'PLAN ONLY'}")
    print()
    print("effective invocation:")
    print(f"  cd {workspace}")
    print(f"  {mesher} < mesh.input")
    if output_directory != workspace:
        print(f"  organize generated mesh products under {output_directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fathi_reduced_3x3_12p5.json",
        help="Benchmark JSON specification, relative to repository root by default.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Generated SEM3D workspace containing mesh.input, mat.dat and mater.in.",
    )
    parser.add_argument(
        "--mesher",
        default=None,
        help=(
            "SEM3D mesher executable. Defaults to SEM3D_MESHER or "
            "~/SEM/build/MESH/mesher."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum mesher runtime. Default: 600 seconds.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the mesher. Without this flag the command is plan-only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing controlled mesh outputs before execution.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit an already meshed workspace without running the mesher.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    config_path = resolve_path(root, args.config)
    workspace = resolve_path(root, args.workspace)
    mesher = (
        Path(args.mesher).expanduser().resolve()
        if args.mesher
        else default_mesher_path()
    )

    spec = load_json(config_path)
    mesh_cfg = spec["sem3d_mesh"]
    stem = str(mesh_cfg["mesh_file_stem"])
    partition_count = int(mesh_cfg["partition_count"])
    output_directory = resolve_mesh_output_directory(
        workspace,
        str(mesh_cfg.get("output_directory", "sem")),
    )
    require(partition_count > 0, "partition_count must be positive")
    require(args.timeout_seconds > 0, "timeout-seconds must be positive")
    require(not (args.execute and args.audit_only), "--execute and --audit-only are mutually exclusive")

    print_plan(
        root=root,
        config=config_path,
        workspace=workspace,
        mesher=mesher,
        output_directory=output_directory,
        stem=stem,
        partition_count=partition_count,
        timeout_seconds=args.timeout_seconds,
        execute=args.execute,
        overwrite=args.overwrite,
    )

    input_sizes = validate_required_inputs(workspace)
    print()
    print("validated inputs:")
    for name, size in input_sizes.items():
        print(f"  {name:12s} {size:8d} bytes")

    if args.audit_only:
        audit = audit_mesher_outputs(
            workspace,
            output_directory,
            stem,
            partition_count,
        )
        print()
        print(f"solver_mesh_directory = {output_directory}")
        print(f"partition_count = {audit['partition_count_actual']}")
        print(f"domains.txt = {audit['domains_txt']}")
        print(f"stray_root_outputs = {audit['stray_root_outputs']}")
        print(f"audit_passed = {audit['passed']}")
        if audit["passed"]:
            print("RESULT = PASS_SEM3D_MESHER_OUTPUT_AUDIT")
            return 0
        print(json.dumps(audit, indent=2, sort_keys=True))
        print("RESULT = FAIL_SEM3D_MESHER_OUTPUT_AUDIT")
        return 1

    if not args.execute:
        print()
        print("No files were changed. Add --execute to run the mesher.")
        print("RESULT = PASS_SEM3D_MESHER_PLAN")
        return 0

    require(mesher.is_file(), f"Mesher executable not found: {mesher}")
    require(os.access(mesher, os.X_OK), f"Mesher is not executable: {mesher}")

    stale = existing_mesher_outputs(workspace, output_directory, stem)
    if stale and not args.overwrite:
        names = ", ".join(str(path.relative_to(workspace)) for path in stale[:12])
        if len(stale) > 12:
            names += f", ... ({len(stale)} total)"
        raise FileExistsError(
            "Mesher outputs already exist. Refusing to mix stale and fresh files. "
            f"Use --overwrite to remove them first. Existing: {names}"
        )

    removed_outputs: list[Path] = []
    if stale:
        removed_outputs = remove_mesher_outputs(workspace, output_directory, stem)

    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "mesher.stdout"
    stderr_path = logs / "mesher.stderr"

    print()
    print("Running mesher...")
    started = time.perf_counter()
    return_code = -1
    timed_out = False

    try:
        with (
            (workspace / "mesh.input").open("rb") as stdin_handle,
            stdout_path.open("wb") as stdout_handle,
            stderr_path.open("wb") as stderr_handle,
        ):
            completed = subprocess.run(
                [str(mesher)],
                cwd=workspace,
                stdin=stdin_handle,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=args.timeout_seconds,
                check=False,
            )
            return_code = int(completed.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 124

    elapsed = time.perf_counter() - started
    relocated_outputs: list[Path] = []
    if not timed_out and return_code == 0:
        relocated_outputs = relocate_mesher_outputs(
            workspace,
            output_directory,
            stem,
        )

    audit = audit_mesher_outputs(
        workspace,
        output_directory,
        stem,
        partition_count,
    )
    manifest_path = write_manifest(
        workspace,
        config=config_path,
        mesher=mesher,
        output_directory=output_directory,
        timeout_seconds=args.timeout_seconds,
        return_code=return_code,
        elapsed_seconds=elapsed,
        input_sizes=input_sizes,
        removed_outputs=removed_outputs,
        relocated_outputs=relocated_outputs,
        audit=audit,
    )

    print(f"return_code = {return_code}")
    print(f"timed_out = {timed_out}")
    print(f"elapsed_seconds = {elapsed:.3f}")
    print(f"stdout = {stdout_path}")
    print(f"stderr = {stderr_path}")
    print(f"manifest = {manifest_path}")
    print(f"solver_mesh_directory = {output_directory}")
    print(f"relocated_output_count = {len(relocated_outputs)}")
    print(f"partition_count = {audit['partition_count_actual']}")
    print(f"domains.txt = {audit['domains_txt']}")
    print(f"stray_root_outputs = {audit['stray_root_outputs']}")

    if timed_out:
        print("RESULT = FAIL_SEM3D_MESHER_TIMEOUT")
        return 124
    if return_code != 0:
        print("RESULT = FAIL_SEM3D_MESHER_EXECUTION")
        return return_code if 0 < return_code < 256 else 1
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        print("RESULT = FAIL_SEM3D_MESHER_OUTPUT_AUDIT")
        return 1

    print("RESULT = PASS_SEM3D_MESHER_EXECUTION_AND_AUDIT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
