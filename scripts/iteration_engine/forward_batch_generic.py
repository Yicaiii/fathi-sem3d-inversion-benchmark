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

def tail_text(path: Path, n=60):
    if not path.exists():
        return [f"MISSING: {path}"]
    return path.read_text(errors="ignore").splitlines()[-n:]

def contains_result_pass(path: Path):
    if not path.exists():
        return False
    txt = path.read_text(errors="ignore")
    return "RESULT = PASS" in txt or "RESULT=PASS" in txt

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
parser.add_argument("--batch", default="strict_full_forward_000")
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

state_dir = ROOT / config["state_dir"]
run_data_root = ROOT / config["run_data_root"]
run_result_root = ROOT / config["run_result_root"]

state_in = state_dir / f"iter_{k:03d}_state_v2_corrected.npz"
state_out = state_dir / f"iter_{kp1:03d}_state_v2_corrected.npz"

forward_dir = run_data_root / f"iter_{kp1:03d}" / "forward_dudx_mgcap_full_batches" / args.batch
traces_dir = forward_dir / "traces"

strict_forward_report_dir = run_result_root / transition / "strict_forward"
possible_audit = strict_forward_report_dir / f"449E_{args.batch}_strict_forward_pilot_audit_summary.txt"

out_dir = ROOT / "benchmark_fathi_strict/reports/phaseA_task1A_forward"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_{args.batch}_forward_generic_status.json"
out_txt = out_dir / f"{transition}_{args.batch}_forward_generic_status.txt"

required = {
    "config": config_path,
    "state_in": state_in,
    "sem3d_exe": Path(config["sem3d_exe"]),
    "true_observed_traces_dir": ROOT / config["true_observed_traces_dir"],
    "forward_dir": forward_dir,
    "input.spec": forward_dir / "input.spec",
    "material.spec": forward_dir / "material.spec",
    "material.input": forward_dir / "material.input",
    "mesh.input": forward_dir / "mesh.input",
    "Mat_0_Kappa.h5": forward_dir / "mat/h5/Mat_0_Kappa.h5",
    "Mat_0_Mu.h5": forward_dir / "mat/h5/Mat_0_Mu.h5",
    "Mat_0_Density.h5": forward_dir / "mat/h5/Mat_0_Density.h5",
}

trace_files = sorted(traces_dir.glob("capteurs.*.h5")) if traces_dir.exists() else []

required_status = {
    name: {
        "path": str(path),
        "exists": path.exists(),
        "size": human_size(path),
    }
    for name, path in required.items()
}

ok_required = all(v["exists"] for v in required_status.values())
ok_traces = traces_dir.exists() and len(trace_files) > 0
ok_audit = contains_result_pass(possible_audit)

overall_ok = ok_required and ok_traces and ok_audit

record = {
    "created": datetime.now().isoformat(),
    "task": "Task 1A forward_batch_generic",
    "mode": args.mode,
    "iter_k": k,
    "iter_kp1": kp1,
    "transition": transition,
    "batch": args.batch,
    "state_in": str(state_in),
    "state_out": str(state_out),
    "forward_dir": str(forward_dir),
    "traces_dir": str(traces_dir),
    "trace_file_count": len(trace_files),
    "trace_files_preview": [str(p) for p in trace_files[:5]],
    "trace_total_size": human_size(traces_dir),
    "possible_audit_summary": str(possible_audit),
    "possible_audit_exists": possible_audit.exists(),
    "possible_audit_has_RESULT_PASS": ok_audit,
    "required_status": required_status,
    "ok_required": ok_required,
    "ok_traces": ok_traces,
    "ok_audit": ok_audit,
    "overall_ok": overall_ok,
    "result": "PASS" if overall_ok else "CHECK_NEEDED",
    "meaning": "This generic wrapper can identify and validate the forward stage for any iter_k without hard-coded transition paths.",
}

out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

lines = []
lines.append("Task 1A forward_batch_generic status")
lines.append("====================================")
lines.append("")
lines.append(f"created = {record['created']}")
lines.append(f"transition = {transition}")
lines.append(f"iter_k = {k}")
lines.append(f"iter_kp1 = {kp1}")
lines.append(f"batch = {args.batch}")
lines.append("")
lines.append("Derived paths:")
lines.append(f"  state_in = {state_in}")
lines.append(f"  state_out = {state_out}")
lines.append(f"  forward_dir = {forward_dir}")
lines.append(f"  traces_dir = {traces_dir}")
lines.append("")
lines.append("Required checks:")
for name, info in required_status.items():
    lines.append(f"  {name}: exists={info['exists']} size={info['size']} path={info['path']}")
lines.append("")
lines.append(f"trace_file_count = {len(trace_files)}")
lines.append(f"trace_total_size = {human_size(traces_dir)}")
for p in trace_files[:10]:
    lines.append(f"  trace_preview = {p.name} size={human_size(p)}")
lines.append("")
lines.append("Audit summary:")
lines.append(f"  possible_audit = {possible_audit}")
lines.append(f"  exists = {possible_audit.exists()}")
lines.append(f"  has_RESULT_PASS = {ok_audit}")
lines.append("")
if possible_audit.exists():
    lines.append("Audit tail:")
    lines.extend("  " + x for x in tail_text(possible_audit, n=25))
    lines.append("")
lines.append("Interpretation:")
lines.append("  This is not a new SEM3D run.")
lines.append("  This is Task 1A wrapper unification: forward paths are derived from iter_k and config.")
lines.append("  Next step after PASS: connect this wrapper to run_iteration.py --stage forward.")
lines.append("")
lines.append(f"RESULT = {record['result']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if record["result"] != "PASS":
    sys.exit(2)
