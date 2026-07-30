#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import find_sem3d, load_config, output_paths, require, write_json


def _write_context(root: Path, cfg: dict, paths: dict[str, Path], sem3d: Path, np_count: int) -> Path:
    context_path = paths["report_dir"] / "mini_iteration_context_for_existing_runner.json"
    payload = {
        "created": datetime.now().isoformat(),
        "role": "mini_iteration_context_adapter_for_existing_adjoint_runner",
        "transition": cfg["transition"],
        "iter_k": 0,
        "iter_kp1": 1,
        "output_adjoint_batches_dir": str(paths["adjoint_root"]),
        "adjoint_batches_dir": str(paths["adjoint_root"]),
        "transition_result_root": str(paths["transition_root"]),
        "sem3d_exe": str(sem3d),
        "mpi_cores": int(np_count),
        "source_config": "configs/fathi_mini_e2e_3600.json",
    }
    write_json(context_path, payload)
    return context_path


def _safe_link(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.symlink_to(target.resolve())


def _mini_report(
    *,
    component: str,
    workspace: Path,
    delegated_command: list[str],
    delegated_report: Path,
    delegated_record: dict,
    elapsed_seconds: float,
    plan_only: bool,
) -> dict:
    fin_path = workspace / "fin_sem"
    trace_dir = workspace / "traces"
    fin_value = fin_path.read_text(encoding="utf-8", errors="ignore").strip() if fin_path.is_file() else ""
    trace_files = sorted(trace_dir.glob("capteurs.*.h5")) if trace_dir.is_dir() else []

    if plan_only:
        result = "PASS_PLAN_ONLY"
    else:
        ok = (
            delegated_record.get("result") in {"PASS_EXECUTED", "PASS_ALREADY_EXISTS"}
            and delegated_record.get("returncode", 0) == 0
            and fin_value == "1"
            and bool(trace_files)
        )
        result = "PASS" if ok else "FAIL"

    return {
        "created": datetime.now().isoformat(),
        "component": component,
        "workspace": str(workspace),
        "delegated_to": "scripts/fathi_benchmark/run_task2c_adjoint_batch.py",
        "delegated_command": delegated_command,
        "delegated_report": str(delegated_report),
        "delegated_result": delegated_record.get("result"),
        "returncode": delegated_record.get("returncode", 0 if plan_only else None),
        "elapsed_seconds": elapsed_seconds,
        "fin_sem": fin_value,
        "trace_file_count": len(trace_files),
        "stdout": delegated_record.get("stdout_log"),
        "stderr": delegated_record.get("stderr_log"),
        "plan_only": plan_only,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mini adjoint adapter that delegates SEM3D execution to the repository's "
            "existing run_task2c_adjoint_batch.py runner."
        )
    )
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--component", choices=["x", "y", "z", "all"], default="all")
    parser.add_argument("--np", type=int)
    parser.add_argument("--plan", action="store_true", help="Delegate in plan-only mode; do not launch SEM3D")
    parser.add_argument("--force", action="store_true", help="Rejected intentionally; quarantine partial outputs instead")
    args = parser.parse_args()

    if args.force:
        raise SystemExit(
            "--force is disabled in the mini adapter. Preserve partial outputs and quarantine them before any rerun."
        )

    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    sem3d = find_sem3d(root, cfg)
    np_count = args.np or int(cfg["mpi_cores"])

    require(shutil.which("mpirun") is not None, "mpirun not found in PATH")

    delegated_runner = root / "scripts/fathi_benchmark/run_task2c_adjoint_batch.py"
    require(delegated_runner.is_file(), f"Missing existing adjoint runner: {delegated_runner}")

    context_path = _write_context(root, cfg, paths, sem3d, np_count)
    components = cfg["adjoint"]["components"] if args.component == "all" else [args.component]

    records: list[dict] = []
    env = os.environ.copy()
    env["FATHI_BENCHMARK_ROOT"] = str(root)

    for component in components:
        workspace = paths["adjoint_root"] / component / "batch_000"
        require((workspace / "input.spec").is_file(), f"Adjoint workspace not prepared: {workspace}")

        fin_path = workspace / "fin_sem"
        trace_dir = workspace / "traces"
        existing_traces = sorted(trace_dir.glob("capteurs.*.h5")) if trace_dir.is_dir() else []
        fin_value = fin_path.read_text(encoding="utf-8", errors="ignore").strip() if fin_path.is_file() else ""

        if not args.plan and (existing_traces or fin_path.exists()):
            if fin_value == "1" and existing_traces:
                print(f"component={component}: already complete; preserving existing output", flush=True)
            else:
                raise SystemExit(
                    f"STOP: partial runtime output exists for {component}: {workspace}. "
                    "Do not use --force; audit or quarantine it first."
                )

        cmd = [
            sys.executable,
            str(delegated_runner),
            "--context",
            str(context_path),
            "--component",
            component,
            "--batch",
            "batch_000",
            "--np",
            str(np_count),
        ]
        if not args.plan:
            cmd.append("--execute")

        print("=" * 100)
        print("DELEGATING MINI ADJOINT TO EXISTING REPOSITORY RUNNER")
        print(f"component = {component}")
        print(f"plan_only = {args.plan}")
        print("command = " + " ".join(cmd))
        print("=" * 100, flush=True)

        started = time.time()
        proc = subprocess.run(cmd, cwd=root, env=env)
        elapsed = time.time() - started
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        delegated_report = (
            root
            / "benchmark_fathi_strict/reports/executable_tasks/adjoint_batches"
            / f"{cfg['transition']}_adjoint_{component}_batch_000_task.json"
        )
        require(delegated_report.is_file(), f"Missing delegated report: {delegated_report}")
        delegated_record = json.loads(delegated_report.read_text(encoding="utf-8"))

        if not args.plan:
            stdout_value = delegated_record.get("stdout_log")
            stderr_value = delegated_record.get("stderr_log")
            if stdout_value:
                _safe_link(workspace / "logs/sem3d_stdout.log", Path(stdout_value))
            if stderr_value:
                _safe_link(workspace / "logs/sem3d_stderr.log", Path(stderr_value))

        record = _mini_report(
            component=component,
            workspace=workspace,
            delegated_command=cmd,
            delegated_report=delegated_report,
            delegated_record=delegated_record,
            elapsed_seconds=elapsed,
            plan_only=args.plan,
        )
        write_json(paths["report_dir"] / f"adjoint_run_{component}.json", record)
        records.append(record)

        print(
            f"component={component} delegated_result={record['delegated_result']} "
            f"mini_result={record['result']} elapsed={record['elapsed_seconds']:.3f}s",
            flush=True,
        )

        expected = "PASS_PLAN_ONLY" if args.plan else "PASS"
        if record["result"] != expected:
            raise SystemExit(1)

    summary_result = "PASS_PLAN_ONLY" if args.plan else "PASS"
    write_json(
        paths["report_dir"] / "adjoint_runs_summary.json",
        {
            "created": datetime.now().isoformat(),
            "delegated_to": "scripts/fathi_benchmark/run_task2c_adjoint_batch.py",
            "records": records,
            "result": summary_result,
        },
    )

    print(
        "RESULT = PASS_MINI_ADJOINT_DELEGATION_PLAN"
        if args.plan
        else "RESULT = PASS_MINI_ADJOINT_RUNS"
    )


if __name__ == "__main__":
    main()
