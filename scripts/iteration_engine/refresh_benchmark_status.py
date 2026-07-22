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
args = parser.parse_args()

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

steps = [
    {
        "name": "inventory_fast",
        "cmd": ["python3", "scripts/iteration_engine/audit_benchmark_inventory_fast.py"],
    },
    {
        "name": "official_status",
        "cmd": ["python3", "scripts/iteration_engine/write_official_benchmark_status.py"],
    },
    {
        "name": "forward_status",
        "cmd": ["python3", "scripts/iteration_engine/forward_batch_generic.py", "--iter-k", str(k), "--mode", "status"],
    },
    {
        "name": "residual_status",
        "cmd": ["python3", "scripts/iteration_engine/residual_generic.py", "--iter-k", str(k), "--mode", "status"],
    },
    {
        "name": "prepare_adjoint_status",
        "cmd": ["python3", "scripts/iteration_engine/prepare_adjoint_generic.py", "--iter-k", str(k), "--mode", "status"],
    },
    {
        "name": "run_adjoint_status",
        "cmd": ["python3", "scripts/iteration_engine/run_adjoint_generic.py", "--iter-k", str(k), "--mode", "status"],
    },
    {
        "name": "resume_plan",
        "cmd": ["python3", "scripts/iteration_engine/write_resume_plan.py", "--iter-k", str(k)],
    },
    {
        "name": "dashboard",
        "cmd": ["python3", "scripts/iteration_engine/write_benchmark_dashboard.py"],
    },
]

out_dir = ROOT / "benchmark_fathi_strict/reports/status_refresh"
out_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()
records = []

for step in steps:
    print(f"Running {step['name']}...", flush=True)
    proc = subprocess.run(
        step["cmd"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ok = proc.returncode == 0 and "RESULT = PASS" in proc.stdout

    records.append({
        "name": step["name"],
        "cmd": step["cmd"],
        "returncode": proc.returncode,
        "ok": ok,
        "stdout_tail": proc.stdout.splitlines()[-40:],
        "stderr_tail": proc.stderr.splitlines()[-40:],
    })

    if not ok:
        print(f"FAILED at step: {step['name']}", flush=True)
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], flush=True)
        break

overall_ok = all(r["ok"] for r in records)

payload = {
    "created": created,
    "transition": transition,
    "overall_ok": overall_ok,
    "records": records,
}

json_path = out_dir / f"{transition}_status_refresh.json"
txt_path = out_dir / f"{transition}_status_refresh.txt"

json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Fathi benchmark status refresh")
lines.append("==============================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"overall_ok = {overall_ok}")
lines.append("")
lines.append("Steps:")
for r in records:
    lines.append("-" * 80)
    lines.append(f"name = {r['name']}")
    lines.append(f"cmd = {' '.join(r['cmd'])}")
    lines.append(f"returncode = {r['returncode']}")
    lines.append(f"ok = {r['ok']}")
    if r["stderr_tail"]:
        lines.append("stderr_tail:")
        for x in r["stderr_tail"]:
            lines.append(f"  {x}")

lines.append("")
lines.append("Key files refreshed:")
lines.append("  benchmark_fathi_strict/reports/official_benchmark_status.txt")
lines.append("  benchmark_fathi_strict/reports/dashboard/benchmark_dashboard.txt")
lines.append(f"  benchmark_fathi_strict/reports/resume/{transition}_resume_plan.txt")
lines.append("")
lines.append("RESULT = PASS" if overall_ok else "RESULT = FAIL")

txt_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if not overall_ok:
    sys.exit(1)
