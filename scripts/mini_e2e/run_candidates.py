#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import shutil
import subprocess
import time

from .common import find_sem3d, load_config, output_paths, require, write_json


def run_one(workspace, candidate, sem3d, np_count, force):
    fin = workspace / "fin_sem"
    traces = workspace / "traces"
    if fin.is_file() and fin.read_text(errors="ignore").strip() == "1" and list(traces.glob("capteurs.*.h5")) and not force:
        return {"candidate": candidate, "workspace": str(workspace), "skipped": True, "result": "PASS"}
    if force:
        for name in ("traces", "logs", "res", "prot", "mirror", "fin_sem"):
            path = workspace / name
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "sem3d_stdout.log"
    stderr_path = logs / "sem3d_stderr.log"
    cmd = ["mpirun", "-np", str(np_count), str(sem3d)]
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=workspace, stdout=stdout, stderr=stderr, text=True)
    elapsed = time.time() - started
    fin_value = fin.read_text(errors="ignore").strip() if fin.exists() else ""
    trace_files = sorted(traces.glob("capteurs.*.h5")) if traces.exists() else []
    ok = proc.returncode == 0 and fin_value == "1" and bool(trace_files)
    return {
        "created": datetime.now().isoformat(),
        "candidate": candidate,
        "workspace": str(workspace),
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "fin_sem": fin_value,
        "trace_file_count": len(trace_files),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "skipped": False,
        "result": "PASS" if ok else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fathi_mini_e2e_3600.json")
    parser.add_argument("--candidate")
    parser.add_argument("--np", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, cfg, _ = load_config(args.config)
    paths = output_paths(root, cfg)
    sem3d = find_sem3d(root, cfg)
    np_count = args.np or int(cfg["mpi_cores"])
    require(shutil.which("mpirun") is not None, "mpirun not found")
    workspaces = sorted(paths["candidate_ws_root"].glob("mini_line_search_pos_mtilde_*"))
    if args.candidate:
        workspaces = [path for path in workspaces if path.name == args.candidate]
    require(bool(workspaces), "No candidate workspaces found")
    records = []
    for workspace in workspaces:
        print(f"Starting candidate {workspace.name}", flush=True)
        record = run_one(workspace, workspace.name, sem3d, np_count, args.force)
        records.append(record)
        write_json(paths["report_dir"] / f"candidate_run_{workspace.name}.json", record)
        print(f"candidate={workspace.name} result={record['result']} elapsed={record.get('elapsed_seconds')}", flush=True)
        if record["result"] != "PASS":
            raise SystemExit(1)
    write_json(paths["report_dir"] / "candidate_runs_summary.json", {"records": records, "result": "PASS"})
    print("RESULT = PASS_MINI_CANDIDATE_RUNS")


if __name__ == "__main__":
    main()
