from pathlib import Path
import os
from datetime import datetime
import json
import sys
import argparse

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

status_json = ROOT / "benchmark_fathi_strict/reports/phaseA_task1E_run_adjoint" / f"{transition}_run_adjoint_generic_status.json"

if not status_json.exists():
    print(f"Missing run_adjoint status json: {status_json}")
    print("Run this first:")
    print(f"  python3 scripts/iteration_engine/run_adjoint_generic.py --iter-k {k} --mode status")
    sys.exit(1)

status = json.loads(status_json.read_text())

out_dir = ROOT / "benchmark_fathi_strict/reports/resume"
out_dir.mkdir(parents=True, exist_ok=True)

out_txt = out_dir / f"{transition}_resume_plan.txt"
out_json = out_dir / f"{transition}_resume_plan.json"

next_item = status.get("next_item")
resume_command = status.get("resume_command", "")

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "stage_state": status.get("stage_state"),
    "pass_count": status.get("pass_count"),
    "expected_total": status.get("expected_total"),
    "next_item": next_item,
    "resume_command": resume_command,
    "next_after_adjoint": [
        "audit adjoint traces",
        "assemble RHS x/y/z",
        "assemble RHS_total",
        "solve Mtilde",
        "generate candidates",
        "candidate forward/misfit",
        "accept state_{k+1}"
    ],
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Fathi benchmark resume plan")
lines.append("===========================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"stage_state = {payload['stage_state']}")
lines.append(f"adjoint progress = {payload['pass_count']}/{payload['expected_total']}")
lines.append("")
if next_item:
    lines.append("Next heavy command to continue:")
    lines.append(f"  {resume_command}")
else:
    lines.append("Adjoint execution is complete.")
    lines.append("Next stage: audit adjoint traces / assemble RHS.")
lines.append("")
lines.append("Important:")
lines.append("  This resume plan does not run anything.")
lines.append("  It is only a saved instruction for continuing later.")
lines.append("  Before running heavy commands, check disk space with: df -h ~ .")
lines.append("")
lines.append("After all adjoint batches complete:")
for x in payload["next_after_adjoint"]:
    lines.append(f"  - {x}")
lines.append("")
lines.append("RESULT = PASS")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
