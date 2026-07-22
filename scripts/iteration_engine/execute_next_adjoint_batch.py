from pathlib import Path
import os
from datetime import datetime
import argparse
import subprocess
import json
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--np", type=int, default=12)
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

out_dir = ROOT / "benchmark_fathi_strict/reports/execute_next_adjoint"
out_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

# 1. Refresh run_adjoint status first
status_cmd = [
    sys.executable,
    "scripts/iteration_engine/run_adjoint_generic.py",
    "--iter-k",
    str(k),
    "--mode",
    "status",
]

status_proc = subprocess.run(
    status_cmd,
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

if status_proc.returncode != 0:
    print(status_proc.stdout)
    print(status_proc.stderr)
    sys.exit("Failed to refresh run_adjoint status.")

status_json = (
    ROOT
    / "benchmark_fathi_strict/reports/phaseA_task1E_run_adjoint"
    / f"{transition}_run_adjoint_generic_status.json"
)

if not status_json.exists():
    sys.exit(f"Missing status json: {status_json}")

status = json.loads(status_json.read_text())

next_item = status.get("next_item")
pass_count_before = status.get("pass_count")
stage_state_before = status.get("stage_state")

record = {
    "created": created,
    "transition": transition,
    "execute": args.execute,
    "np": args.np,
    "stage_state_before": stage_state_before,
    "pass_count_before": pass_count_before,
    "next_item": next_item,
    "action": None,
    "cmd": None,
    "returncode": None,
    "result": None,
}

lines = []
lines.append("Task 2A execute next adjoint batch")
lines.append("==================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"execute = {args.execute}")
lines.append(f"stage_state_before = {stage_state_before}")
lines.append(f"pass_count_before = {pass_count_before}/30")
lines.append("")

if not next_item:
    record["action"] = "nothing_to_run"
    record["result"] = "PASS"
    lines.append("All adjoint batches appear complete.")
    lines.append("No next batch to run.")
else:
    comp = next_item["component"]
    batch = next_item["batch"]

    cmd = [
        sys.executable,
        "scripts/fathi_loop_v2/456A_run_strict_adjoint_batch.py",
        "--component",
        comp,
        "--batch",
        batch,
        "--np",
        str(args.np),
    ]

    if args.execute:
        cmd.append("--execute")

    record["cmd"] = cmd

    lines.append("Next batch:")
    lines.append(f"  component = {comp}")
    lines.append(f"  batch = {batch}")
    lines.append("")
    lines.append("Command:")
    lines.append("  " + " ".join(cmd))
    lines.append("")

    if not args.execute:
        record["action"] = "dryrun_only"
        record["result"] = "PASS_DRYRUN"
        lines.append("DRY_RUN_ONLY: SEM3D was not launched.")
        lines.append("")
        lines.append("To execute exactly this next batch:")
        lines.append(f"  python3 scripts/iteration_engine/execute_next_adjoint_batch.py --iter-k {k} --execute")
    else:
        record["action"] = "executed_next_batch"

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        record["returncode"] = proc.returncode
        record["stdout_tail"] = proc.stdout.splitlines()[-80:]
        record["stderr_tail"] = proc.stderr.splitlines()[-80:]

        lines.append("Execution stdout tail:")
        for x in record["stdout_tail"]:
            lines.append("  " + x)
        if record["stderr_tail"]:
            lines.append("")
            lines.append("Execution stderr tail:")
            for x in record["stderr_tail"]:
                lines.append("  " + x)

        if proc.returncode == 0 and "RESULT = PASS" in proc.stdout:
            record["result"] = "PASS"
        else:
            record["result"] = "FAIL"

        # Refresh status after execution
        status_proc2 = subprocess.run(
            status_cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if status_proc2.returncode == 0 and status_json.exists():
            status2 = json.loads(status_json.read_text())
            record["pass_count_after"] = status2.get("pass_count")
            record["stage_state_after"] = status2.get("stage_state")
            record["next_item_after"] = status2.get("next_item")

            lines.append("")
            lines.append("Status after execution:")
            lines.append(f"  stage_state_after = {record['stage_state_after']}")
            lines.append(f"  pass_count_after = {record['pass_count_after']}/30")
            if record["next_item_after"]:
                lines.append(
                    "  next_item_after = "
                    + record["next_item_after"]["component"]
                    + " "
                    + record["next_item_after"]["batch"]
                )
            else:
                lines.append("  next_item_after = NONE")

json_path = out_dir / f"{transition}_execute_next_adjoint_batch.json"
txt_path = out_dir / f"{transition}_execute_next_adjoint_batch.txt"

json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

lines.append("")
lines.append(f"json = {json_path}")
lines.append("")
lines.append(f"RESULT = {record['result']}")

txt_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if record["result"] not in ["PASS", "PASS_DRYRUN"]:
    sys.exit(1)
