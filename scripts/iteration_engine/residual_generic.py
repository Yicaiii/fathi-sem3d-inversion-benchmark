from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import subprocess
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

def human_size(path: Path):
    if not path.exists() and not path.is_symlink():
        return "MISSING"
    try:
        out = subprocess.check_output(
            ["du", "-sh", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.split()[0]
    except Exception:
        return "unknown"

def tail_text(path: Path, n=40):
    if not path.exists():
        return [f"MISSING: {path}"]
    return path.read_text(errors="ignore").splitlines()[-n:]

def has_pass(path: Path):
    if not path.exists():
        return False
    txt = path.read_text(errors="ignore")
    return "RESULT = PASS" in txt or "RESULT=PASS" in txt

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

forward_traces = run_data_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000/traces"
residual_dir = run_result_root / "residual_sources"

manifest_summary = residual_dir / "454A_strict_forward_residual_manifest_summary.txt"
manifest_csv = residual_dir / "454A_strict_forward_residual_manifest.csv"
manifest_json = residual_dir / "454A_strict_forward_residual_manifest.json"

residual_summary = residual_dir / "454B_strict_residual_timeseries_summary.txt"
residual_h5 = residual_dir / "454B_strict_residual_timeseries.h5"
residual_json = residual_dir / "454B_strict_residual_timeseries.json"

out_dir = ROOT / "benchmark_fathi_strict/reports/phaseA_task1C_residual"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_residual_generic_status.json"
out_txt = out_dir / f"{transition}_residual_generic_status.txt"

required = {
    "config": config_path,
    "true_observed_traces_dir": ROOT / config["true_observed_traces_dir"],
    "forward_traces": forward_traces,
    "residual_dir": residual_dir,
    "454A_manifest_summary": manifest_summary,
    "454A_manifest_csv": manifest_csv,
    "454A_manifest_json": manifest_json,
    "454B_residual_summary": residual_summary,
    "454B_residual_h5": residual_h5,
    "454B_residual_json": residual_json,
}

required_status = {
    name: {
        "path": str(path),
        "exists": path.exists(),
        "size": human_size(path),
    }
    for name, path in required.items()
}

ok_required = all(v["exists"] for v in required_status.values())
ok_454A = has_pass(manifest_summary)
ok_454B = has_pass(residual_summary)
overall_ok = ok_required and ok_454A and ok_454B

record = {
    "created": datetime.now().isoformat(),
    "task": "Task 1C residual_generic",
    "mode": args.mode,
    "iter_k": k,
    "iter_kp1": kp1,
    "transition": transition,
    "run_result_root": str(run_result_root),
    "run_data_root": str(run_data_root),
    "residual_dir": str(residual_dir),
    "required_status": required_status,
    "ok_required": ok_required,
    "ok_454A_manifest_PASS": ok_454A,
    "ok_454B_residual_h5_PASS": ok_454B,
    "overall_ok": overall_ok,
    "result": "PASS" if overall_ok else "CHECK_NEEDED",
    "meaning": "This generic wrapper can identify and validate the residual stage for any iter_k without hard-coded transition paths.",
}

out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

lines = []
lines.append("Task 1C residual_generic status")
lines.append("================================")
lines.append("")
lines.append(f"created = {record['created']}")
lines.append(f"transition = {transition}")
lines.append(f"iter_k = {k}")
lines.append(f"iter_kp1 = {kp1}")
lines.append("")
lines.append("Derived paths:")
lines.append(f"  run_result_root = {run_result_root}")
lines.append(f"  run_data_root = {run_data_root}")
lines.append(f"  forward_traces = {forward_traces}")
lines.append(f"  residual_dir = {residual_dir}")
lines.append("")
lines.append("Required checks:")
for name, info in required_status.items():
    lines.append(f"  {name}: exists={info['exists']} size={info['size']} path={info['path']}")
lines.append("")
lines.append("PASS checks:")
lines.append(f"  454A manifest RESULT PASS = {ok_454A}")
lines.append(f"  454B residual timeseries RESULT PASS = {ok_454B}")
lines.append("")
if manifest_summary.exists():
    lines.append("454A tail:")
    lines.extend("  " + x for x in tail_text(manifest_summary, n=20))
    lines.append("")
if residual_summary.exists():
    lines.append("454B tail:")
    lines.extend("  " + x for x in tail_text(residual_summary, n=20))
    lines.append("")
lines.append("Interpretation:")
lines.append("  This is not a new residual computation.")
lines.append("  This is Task 1C wrapper unification: residual paths are derived from iter_k and config.")
lines.append("  Next step after PASS: connect this wrapper to run_iteration.py --stage residual.")
lines.append("")
lines.append(f"RESULT = {record['result']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if record["result"] != "PASS":
    sys.exit(2)
