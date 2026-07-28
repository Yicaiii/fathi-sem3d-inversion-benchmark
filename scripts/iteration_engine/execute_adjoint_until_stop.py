from pathlib import Path
import os
from datetime import datetime
import argparse
import subprocess
import json
import sys
import time

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--max-batches", type=int, default=3)
parser.add_argument("--sleep-seconds", type=int, default=5)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

out_dir = ROOT / "benchmark_fathi_strict/reports/execute_adjoint_until_stop"
out_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

out_txt = out_dir / f"{transition}_auto_adjoint_{run_id}.txt"
out_json = out_dir / f"{transition}_auto_adjoint_{run_id}.json"

records = []

def append(lines):
    with out_txt.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print("\n".join(lines), flush=True)

append([
    "Task 2B execute adjoint until stop",
    "===================================",
    "",
    f"created = {created}",
    f"transition = {transition}",
    f"execute = {args.execute}",
    f"max_batches = {args.max_batches}",
    "",
])

if not args.execute:
    append([
        "DRY RUN ONLY.",
        "This script would repeatedly call:",
        f"  python3 scripts/iteration_engine/execute_next_adjoint_batch.py --iter-k {k} --execute",
        "",
        "To really run:",
        f"  python3 scripts/iteration_engine/execute_adjoint_until_stop.py --iter-k {k} --max-batches {args.max_batches} --execute",
        "",
        "RESULT = PASS_DRYRUN",
    ])
    sys.exit(0)

for i in range(args.max_batches):
    step_no = i + 1

    append([
        "",
        "-" * 80,
        f"auto step {step_no}/{args.max_batches}",
        "-" * 80,
    ])

    cmd = [
        sys.executable,
        "scripts/iteration_engine/execute_next_adjoint_batch.py",
        "--iter-k",
        str(k),
        "--execute",
    ]

    append([
        "Running command:",
        "  " + " ".join(cmd),
        "",
    ])

    started = datetime.now().isoformat()

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ended = datetime.now().isoformat()

    stdout_tail = proc.stdout.splitlines()[-80:]
    stderr_tail = proc.stderr.splitlines()[-80:]

    ok = proc.returncode == 0 and "RESULT = PASS" in proc.stdout

    records.append({
        "step": step_no,
        "started": started,
        "ended": ended,
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": ok,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    })

    append([
        f"started = {started}",
        f"ended = {ended}",
        f"returncode = {proc.returncode}",
        f"ok = {ok}",
        "",
        "stdout tail:",
    ] + ["  " + x for x in stdout_tail])

    if stderr_tail:
        append(["", "stderr tail:"] + ["  " + x for x in stderr_tail])

    if not ok:
        append([
            "",
            "STOPPING because this batch did not PASS.",
            "Check the logs above before continuing.",
            "",
            "RESULT = FAIL",
        ])
        out_json.write_text(json.dumps({
            "created": created,
            "transition": transition,
            "execute": args.execute,
            "max_batches": args.max_batches,
            "records": records,
            "result": "FAIL",
        }, indent=2), encoding="utf-8")
        sys.exit(1)

    # Refresh global status after each successful batch.
    status_cmd = [
        sys.executable,
        "scripts/fathi_benchmark/run_iteration.py",
        "--iter-k",
        str(k),
        "--stage",
        "status",
    ]

    append([
        "",
        "Refreshing benchmark status:",
        "  " + " ".join(status_cmd),
    ])

    status_proc = subprocess.run(
        status_cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    status_ok = status_proc.returncode == 0 and "RESULT = PASS" in status_proc.stdout

    append([
        f"status_refresh_returncode = {status_proc.returncode}",
        f"status_refresh_ok = {status_ok}",
    ])

    if not status_ok:
        append([
            "",
            "STOPPING because status refresh failed.",
            "",
            "RESULT = FAIL",
        ])
        out_json.write_text(json.dumps({
            "created": created,
            "transition": transition,
            "execute": args.execute,
            "max_batches": args.max_batches,
            "records": records,
            "result": "FAIL_STATUS_REFRESH",
        }, indent=2), encoding="utf-8")
        sys.exit(1)

    # Check whether adjoint is complete.
    status_json = (
        ROOT
        / "benchmark_fathi_strict/reports/phaseA_task1E_run_adjoint"
        / f"{transition}_run_adjoint_generic_status.json"
    )

    if status_json.exists():
        status = json.loads(status_json.read_text())
        pass_count = status.get("pass_count")
        stage_state = status.get("stage_state")
        next_item = status.get("next_item")

        append([
            f"current stage_state = {stage_state}",
            f"current adjoint progress = {pass_count}/30",
        ])

        if next_item:
            append([
                f"next item = {next_item.get('component')} {next_item.get('batch')}",
            ])
        else:
            append([
                "next item = NONE",
            ])

        if stage_state == "COMPLETE" or pass_count == 30:
            append([
                "",
                "All adjoint batches are complete.",
                "",
                "RESULT = PASS",
            ])
            out_json.write_text(json.dumps({
                "created": created,
                "transition": transition,
                "execute": args.execute,
                "max_batches": args.max_batches,
                "records": records,
                "result": "PASS_COMPLETE",
            }, indent=2), encoding="utf-8")
            sys.exit(0)

    time.sleep(args.sleep_seconds)

append([
    "",
    f"Reached max_batches = {args.max_batches}.",
    "This is normal. Run the same command again to continue.",
    "",
    f"log = {out_txt}",
    "",
    "RESULT = PASS",
])

out_json.write_text(json.dumps({
    "created": created,
    "transition": transition,
    "execute": args.execute,
    "max_batches": args.max_batches,
    "records": records,
    "result": "PASS",
}, indent=2), encoding="utf-8")
