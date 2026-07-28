from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

def read_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")

def has_pass(path: Path):
    txt = read_text(path)
    return "RESULT = PASS" in txt or "RESULT=PASS" in txt

def first_capteurs_file(batch_dir: Path):
    # Fast presence check; do not count huge directories.
    direct_traces = batch_dir / "traces"
    if direct_traces.exists():
        for p in direct_traces.glob("capteurs*.h5"):
            return p

    prot = batch_dir / "prot"
    if prot.exists():
        for cap_dir in prot.glob("Protection_*/Capteurs"):
            for p in cap_dir.glob("capteurs*.h5"):
                return p

    # fallback: only if standard locations did not work
    res = batch_dir / "res"
    if res.exists():
        for p in res.glob("capteurs*.h5"):
            return p

    return None

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config_path = ROOT / args.config
if not config_path.exists():
    print(f"Missing config: {config_path}")
    sys.exit(1)

config = json.loads(config_path.read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_result_root = ROOT / config["run_result_root"] / transition
run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"

adj_base = run_data_root / "adjoint_full_grid_batches"
adjoint_runs = run_result_root / "adjoint_runs"
logs_dir = adjoint_runs / "logs"

status_json = (
    ROOT
    / "benchmark_fathi_strict/reports/phaseA_task1E_run_adjoint"
    / f"{transition}_run_adjoint_generic_status.json"
)

status = {}
if status_json.exists():
    status = json.loads(status_json.read_text())

components = ["x", "y", "z"]
batches = [f"batch_{i:03d}" for i in range(10)]

records = []
pass_summary_count = 0
trace_present_count = 0
missing_or_bad = []

for comp in components:
    for batch in batches:
        batch_dir = adj_base / comp / batch
        summary = adjoint_runs / f"456A_{comp}_{batch}_run_summary.txt"
        stdout_log = logs_dir / f"{comp}_{batch}_stdout.log"
        stderr_log = logs_dir / f"{comp}_{batch}_stderr.log"

        summary_pass = has_pass(summary)
        if summary_pass:
            pass_summary_count += 1

        trace_sample = first_capteurs_file(batch_dir)
        trace_present = trace_sample is not None
        if trace_present:
            trace_present_count += 1

        record = {
            "component": comp,
            "batch": batch,
            "batch_dir": str(batch_dir),
            "batch_dir_exists": batch_dir.exists(),
            "summary": str(summary),
            "summary_exists": summary.exists(),
            "summary_pass": summary_pass,
            "stdout_log": str(stdout_log),
            "stdout_log_exists": stdout_log.exists(),
            "stderr_log": str(stderr_log),
            "stderr_log_exists": stderr_log.exists(),
            "trace_present": trace_present,
            "trace_sample": str(trace_sample) if trace_sample else "",
        }
        records.append(record)

        if not summary_pass or not trace_present:
            missing_or_bad.append(record)

out_dir = ROOT / "benchmark_fathi_strict/reports/phaseA_task2C_adjoint_complete"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_adjoint_complete_audit.json"
out_txt = out_dir / f"{transition}_adjoint_complete_audit.txt"

status_pass_count = status.get("pass_count", None)
status_stage_state = status.get("stage_state", "")

summary_all_ok = pass_summary_count == 30
trace_all_ok = trace_present_count == 30
status_ok = status_pass_count == 30 or status_stage_state == "COMPLETE"

overall_ok = summary_all_ok and trace_all_ok and status_ok

payload = {
    "created": datetime.now().isoformat(),
    "task": "Task 2C adjoint complete audit",
    "transition": transition,
    "status_json": str(status_json),
    "status_pass_count": status_pass_count,
    "status_stage_state": status_stage_state,
    "pass_summary_count": pass_summary_count,
    "trace_present_count": trace_present_count,
    "summary_all_ok": summary_all_ok,
    "trace_all_ok": trace_all_ok,
    "status_ok": status_ok,
    "overall_ok": overall_ok,
    "records": records,
    "missing_or_bad": missing_or_bad,
    "result": "PASS" if overall_ok else "CHECK_NEEDED",
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Task 2C adjoint complete audit")
lines.append("==============================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append("")
lines.append("Status from run_adjoint_generic:")
lines.append(f"  status_stage_state = {status_stage_state}")
lines.append(f"  status_pass_count = {status_pass_count}/30")
lines.append("")
lines.append("Audit counts:")
lines.append(f"  pass_summary_count = {pass_summary_count}/30")
lines.append(f"  trace_present_count = {trace_present_count}/30")
lines.append(f"  summary_all_ok = {summary_all_ok}")
lines.append(f"  trace_all_ok = {trace_all_ok}")
lines.append(f"  status_ok = {status_ok}")
lines.append("")
lines.append("Per-batch summary:")
for comp in components:
    lines.append("-" * 80)
    lines.append(f"component = {comp}")
    for r in [x for x in records if x["component"] == comp]:
        lines.append(
            f"  {r['batch']}: "
            f"summary_pass={r['summary_pass']} "
            f"trace_present={r['trace_present']} "
            f"stdout_log={r['stdout_log_exists']} "
            f"stderr_log={r['stderr_log_exists']}"
        )

if missing_or_bad:
    lines.append("")
    lines.append("Missing or suspicious batches:")
    for r in missing_or_bad:
        lines.append(
            f"  {r['component']} {r['batch']}: "
            f"summary_pass={r['summary_pass']} "
            f"trace_present={r['trace_present']} "
            f"batch_dir={r['batch_dir']}"
        )
else:
    lines.append("")
    lines.append("Missing or suspicious batches: NONE")

lines.append("")
lines.append("Interpretation:")
lines.append("  PASS means the adjoint execution stage is formally closed.")
lines.append("  After this, the next benchmark stage is RHS assembly.")
lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if payload["result"] != "PASS":
    sys.exit(2)
