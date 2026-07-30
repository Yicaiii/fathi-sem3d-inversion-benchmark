#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import h5py

from .common import (
    copy_static_workspace,
    find_sem3d,
    iteration_context,
    load_config,
    output_paths,
    remove_runtime_outputs,
    replace_run_name,
    require,
    resolve,
    set_dudx,
    sha256,
    trace_position_map,
    write_json,
)


def count_stations(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def dudx_ok(trace_dir: Path) -> bool:
    files = sorted(trace_dir.glob("capteurs.*.h5"))
    if not files:
        return False
    with h5py.File(files[0], "r") as handle:
        if "Variables" not in handle:
            return False
        values = [
            item.decode(errors="ignore").strip()
            if isinstance(item, bytes)
            else str(item).strip()
            for item in handle["Variables"][()]
        ]
    return all(
        any(re.fullmatch(rf"DUDX[\s_]*{index}", value) for value in values)
        for index in range(1, 10)
    )


def audit(workspace: Path, expected: int) -> dict:
    fin = workspace / "fin_sem"
    trace_dir = workspace / "traces"
    fin_value = fin.read_text(errors="ignore").strip() if fin.is_file() else ""
    files = sorted(trace_dir.glob("capteurs.*.h5")) if trace_dir.is_dir() else []
    positions = 0
    error = ""
    if files:
        try:
            positions = len(trace_position_map(trace_dir))
        except Exception as exc:
            error = repr(exc)
    has_dudx = dudx_ok(trace_dir) if files else False
    complete = (
        fin_value == "1"
        and bool(files)
        and positions == expected
        and has_dudx
    )
    return {
        "workspace": str(workspace),
        "fin_sem": fin_value,
        "trace_file_count": len(files),
        "control_position_count": positions,
        "expected_control_position_count": expected,
        "dudx_1_to_9": has_dudx,
        "position_error": error,
        "complete": complete,
    }


def prepare(root: Path, cfg: dict, workspace: Path, force: bool) -> dict:
    context = iteration_context(cfg)
    parent = resolve(root, cfg["parent_accepted_dir"])
    template_value = cfg.get("strict_forward_template_workspace")
    require(
        bool(template_value),
        "strict_forward_template_workspace is required",
    )
    template = resolve(root, template_value)
    expected = int(cfg["control_region"]["expected_count"])

    required = [
        parent / "input.spec",
        parent / "mat/h5/Mat_0_Kappa.h5",
        parent / "mat/h5/Mat_0_Mu.h5",
        parent / "mat/h5/Mat_0_Density.h5",
        parent / "sem",
        template / "stations.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing next-forward inputs:\n" + "\n".join(missing))
    require(
        count_stations(template / "stations.txt") == expected,
        "Strict-forward template station count mismatch",
    )

    if workspace.exists() or workspace.is_symlink():
        current = audit(workspace, expected)
        if current["complete"] and not force:
            return {
                "created": datetime.now().isoformat(),
                "parent_accepted": str(parent),
                "strict_forward_template": str(template),
                **current,
                "skipped": True,
                "result": "PASS",
            }
        if not force:
            marker = workspace / "MINI_NEXT_FORWARD_PREPARATION.json"
            runtime_names = ("traces", "logs", "res", "prot", "mirror", "fin_sem")
            runtime_present = any(
                (workspace / name).exists() or (workspace / name).is_symlink()
                for name in runtime_names
            )
            if marker.is_file() and not runtime_present:
                return json.loads(marker.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"Workspace exists but is not a clean prepared workspace: "
                f"{workspace}. Inspect or quarantine it first."
            )

    if force and (workspace.exists() or workspace.is_symlink()):
        if workspace.is_symlink() or workspace.is_file():
            workspace.unlink()
        else:
            shutil.rmtree(workspace)

    copy_static_workspace(parent, workspace, symlink_sem=True)

    for stale in workspace.glob("MINI_*.json"):
        if stale.is_file():
            stale.unlink()

    shutil.copy2(template / "stations.txt", workspace / "stations.txt")

    input_text = (parent / "input.spec").read_text(encoding="utf-8")
    input_text = replace_run_name(
        input_text,
        f"mini_{context['parent_tag']}_to_{context['next_tag']}_strict_forward",
    )
    input_text = re.sub(
        r"save_snap\s*=\s*(?:true|false)\s*;",
        "save_snap = false;",
        input_text,
        count=1,
    )
    input_text = set_dudx(input_text, True)
    (workspace / "input.spec").write_text(input_text, encoding="utf-8")

    payload = {
        "created": datetime.now().isoformat(),
        "parent_iteration": context["parent_iteration"],
        "next_iteration": context["next_iteration"],
        "parent_accepted": str(parent),
        "workspace": str(workspace),
        "strict_forward_template": str(template),
        "control_station_count": expected,
        "station_sha256": sha256(workspace / "stations.txt"),
        "material_sha256": {
            name: sha256(parent / "mat/h5" / name)
            for name in (
                "Mat_0_Kappa.h5",
                "Mat_0_Mu.h5",
                "Mat_0_Density.h5",
            )
        },
        "dudx": True,
        "sem_directory_is_symlink": (workspace / "sem").is_symlink(),
        "skipped": False,
        "result": "PASS",
    }
    write_json(workspace / "MINI_NEXT_FORWARD_PREPARATION.json", payload)
    return payload


def run_sem3d(root: Path, cfg: dict, workspace: Path, np_count: int, force: bool) -> dict:
    paths = output_paths(root, cfg)
    context = iteration_context(cfg)
    expected = int(cfg["control_region"]["expected_count"])
    current = audit(workspace, expected)

    if current["complete"] and not force:
        record = {
            "created": datetime.now().isoformat(),
            **current,
            "skipped": True,
            "result": "PASS",
        }
        write_json(paths["report_dir"] / "next_forward_run.json", record)
        return record

    if force:
        remove_runtime_outputs(workspace)
    else:
        runtime_names = ("traces", "logs", "res", "prot", "mirror", "fin_sem")
        runtime_present = any(
            (workspace / name).exists() or (workspace / name).is_symlink()
            for name in runtime_names
        )
        require(
            not runtime_present,
            "Partial runtime outputs exist. Quarantine them before rerunning.",
        )

    sem3d = find_sem3d(root, cfg)
    require(shutil.which("mpirun") is not None, "mpirun not found")
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "sem3d_stdout.log"
    stderr_path = logs / "sem3d_stderr.log"
    cmd = ["mpirun", "-np", str(np_count), str(sem3d)]

    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout,          stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    elapsed = time.time() - started
    checked = audit(workspace, expected)
    ok = proc.returncode == 0 and checked["complete"]
    record = {
        "created": datetime.now().isoformat(),
        "parent_iteration": context["parent_iteration"],
        "next_iteration": context["next_iteration"],
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        **checked,
        "skipped": False,
        "result": "PASS" if ok else "FAIL",
    }
    write_json(paths["report_dir"] / "next_forward_run.json", record)
    return record


def show(cfg: dict, record: dict) -> None:
    context = iteration_context(cfg)
    print("MINI ITERATION FORWARD HANDOFF")
    print("==============================")
    print(f"transition = {context['parent_tag']} -> {context['next_tag']}")
    for key in (
        "workspace",
        "fin_sem",
        "trace_file_count",
        "control_position_count",
        "expected_control_position_count",
        "dudx_1_to_9",
        "complete",
        "result",
    ):
        if key in record:
            print(f"{key} = {record[key]}")
    if record.get("complete") and record.get("result") == "PASS":
        print(f"RESULT = {context['forward_handoff_marker']}")
    else:
        print("RESULT = MINI_NEXT_FORWARD_INCOMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fathi_mini_e2e_iter001_to_iter002.json",
    )
    parser.add_argument("--np", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    root, cfg, _ = load_config(args.config)
    context = iteration_context(cfg)
    paths = output_paths(root, cfg)
    workspace = resolve(root, cfg["strict_forward_workspace"])
    expected = int(cfg["control_region"]["expected_count"])

    if args.status_only:
        report = paths["report_dir"] / "next_forward_run.json"
        if report.is_file():
            record = json.loads(report.read_text(encoding="utf-8"))
        else:
            checked = audit(workspace, expected)
            record = {
                **checked,
                "result": "PASS" if checked["complete"] else "IN_PROGRESS",
            }
        show(cfg, record)
        return

    prepared = prepare(root, cfg, workspace, args.force)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    write_json(paths["report_dir"] / "next_forward_preparation.json", prepared)

    if args.prepare_only:
        print("MINI NEXT FORWARD PREPARATION")
        print("=============================")
        print(f"transition = {context['parent_tag']} -> {context['next_tag']}")
        print(f"parent_accepted = {prepared['parent_accepted']}")
        print(f"workspace = {prepared['workspace']}")
        print(f"control_station_count = {prepared['control_station_count']}")
        print("dudx = True")
        print(f"RESULT = {context['forward_preparation_marker']}")
        return

    np_count = args.np or int(cfg["mpi_cores"])
    print(
        f"Starting strict forward handoff "
        f"{context['parent_tag']} -> {context['next_tag']}",
        flush=True,
    )
    record = run_sem3d(root, cfg, workspace, np_count, args.force)
    show(cfg, record)
    if record["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
