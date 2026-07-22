from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

def has_pass(path: Path):
    if not path.exists():
        return False
    txt = path.read_text(errors="ignore")
    return "RESULT = PASS" in txt or "RESULT=PASS" in txt

def read_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
parser.add_argument("--mode", choices=["status", "dryrun"], default="status")
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
adjoint_runs_dir = run_result_root / "adjoint_runs"
legacy_adjoint_runs_dir = ROOT / "results/fathi_loop_v2/iter_007_to_iter_008/adjoint_runs"

# For current layout, adjoint run summaries are here:
summary_root_candidates = [
    adjoint_runs_dir,
    legacy_adjoint_runs_dir,
]

prepare_summary = run_result_root / "residual_sources/455C_audit_strict_adjoint_batches_summary.txt"
prepare_ok =_summary = run_result_root / "residual_sources/455C_audit_strict_adjoint_batches_summary.txt"
prepare_ok = has_pass(prepare_summary)

components = ["x", "y", "z"]
expected_batches = [f"batch_{i:03d}" for i in range(10)]

records = []
pass_count = 0
executed_count = 0
not_run_count = 0
bad_count = 0

for comp in components:
    for batch in expected_batches:
        batch_dir = adj_base / comp / batch
        source_files = sorted(batch_dir.glob(f"s*{comp}.txt")) if batch_dir.exists() else []

        # summary name from 456A script:
        # 456A_x_batch_000_run_summary.txt
        summary_candidates = [
            root / f"456A_{comp}_{batch}_run_summary.txt"
            for root in summary_root_candidates
        ]

        summary_path = None
        summary_text = ""
        for cand in summary_candidates:
            if cand.exists():
                summary_path = cand
                summary_text = read_text(cand)
                break

        executed = "execute = True" in summary_text
        pass_run = executed and ("RESULT = PASS" in summary_text or "RESULT=PASS" in summary_text)
        fail_run = executed and ("RESULT = FAIL" in summary_text or "RESULT=FAIL" in summary_text)

        if pass_run:
            status = "PASS"
            pass_count += 1
            executed_count += 1
        elif fail_run:
            status = "FAIL"
            bad_count += 1
            executed_count += 1
        elif summary_path is not None and "PASS_DRYRUN" in summary_text:
            status = "DRYRUN_ONLY"
        else:
            status = "NOT_RUN"
            not_run_count += 1

        source_ok = batch_dir.exists() and len(source_files) == 225

        records.append({
            "component": comp,
            "batch": batch,
            "batch_dir": str(batch_dir),
            "batch_dir_exists": batch_dir.exists(),
            "source_count": len(source_files),
            "source_ok": source_ok,
            "summary_path": str(summary_path) if summary_path else "",
            "summary_exists": summary_path is not None,
            "executed": executed,
            "status": status,
        })

next_item = None
for r in records:
    if r["status"] in ["NOT_RUN", "DRYRUN_ONLY"]:
        next_item = r
        break

all_sources_ok = all(r["source_ok"] for r in records)
all_done = pass_count == 30
overall_ok = prepare_ok and all_sources_ok and bad_count == 0

if all_done:
    stage_state = "COMPLETE"
elif overall_ok:
    stage_state = "IN_PROGRESS"
else:
    stage_state = "CHECK_NEEDED"

out_dir = ROOT / "benchmark_fathi_strict/reports/phaseA_task1E_run_adjoint"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_run_adjoint_generic_status.json"
out_txt = out_dir / f"{transition}_run_adjoint_generic_status.txt"

resume_command = ""
if next_item:
    resume_command = (
        "python3 scripts/fathi_loop_v2/456A_run_strict_adjoint_batch.py "
        f"--component {next_item['component']} "
        f"--batch {next_item['batch']} "
        "--execute"
    )

payload = {
    "created": datetime.now().isoformat(),
    "task": "Task 1E run_adjoint_generic",
    "iter_k": k,
    "iter_kp1": kp1,
    "transition": transition,
    "run_result_root": str(run_result_root),
    "run_data_root": str(run_data_root),
    "adj_base": str(adj_base),
    "prepare_summary": str(prepare_summary),
    "prepare_ok": prepare_ok,
    "expected_total": 30,
    "pass_count": pass_count,
    "executed_count": executed_count,
    "not_run_count": not_run_count,
    "bad_count": bad_count,
    "all_sources_ok": all_sources_ok,
    "all_done": all_done,
    "stage_state": stage_state,
    "next_item": next_item,
    "resume_command": resume_command,
    "records": records,
    "result": "PASS" if overall_ok else "CHECK_NEEDED",
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Task 1E run_adjoint_generic status")
lines.append("===================================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"stage_state = {stage_state}")
lines.append("")
lines.append("Summary:")
lines.append(f"  prepare_adjoint_PASS = {prepare_ok}")
lines.append(f"  expected_total = 30")
lines.append(f"  pass_count = {pass_count}/30")
lines.append(f"  executed_count = {executed_count}/30")
lines.append(f"  not_run_count = {not_run_count}/30")
lines.append(f"  bad_count = {bad_count}")
lines.append(f"  all_sources_ok = {all_sources_ok}")
lines.append("")
lines.append("Batch status:")
for comp in components:
    lines.append("-" * 60)
    lines.append(f"component = {comp}")
    for r in [x for x in records if x["component"] == comp]:
        lines.append(
            f"  {r['batch']}: status={r['status']} "
            f"source_count={r['source_count']} source_ok={r['source_ok']}"
        )
lines.append("")
if next_item:
    lines.append("Next batch to resume:")
    lines.append(f"  component = {next_item['component']}")
    lines.append(f"  batch = {next_item['batch']}")
    lines.append("")
    lines.append("Resume command:")
    lines.append(f"  {resume_command}")
else:
    lines.append("Next batch to resume:")
    lines.append("  NONE, all 30 adjoint batches are complete.")
lines.append("")
lines.append("Interpretation:")
lines.append("  This does not run SEM3D.")
lines.append("  It only detects adjoint execution progress and the next safe resume point.")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if payload["result"] not in ["PASS"]:
    sys.exit(2)
