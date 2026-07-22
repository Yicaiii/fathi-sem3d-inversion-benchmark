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

def tail_text(path: Path, n=30):
    if not path.exists():
        return [f"MISSING: {path}"]
    return path.read_text(errors="ignore").splitlines()[-n:]

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

residual_dir = run_result_root / "residual_sources"
adj_base = run_data_root / "adjoint_full_grid_batches"

summary_455B = residual_dir / "455B_prepare_strict_adjoint_batches_from_residual_summary.txt"
summary_455C = residual_dir / "455C_audit_strict_adjoint_batches_summary.txt"

out_dir = ROOT / "benchmark_fathi_strict/reports/phaseA_task1D_prepare_adjoint"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_prepare_adjoint_generic_status.json"
out_txt = out_dir / f"{transition}_prepare_adjoint_generic_status.txt"

components = ["x", "y", "z"]
component_records = []
overall_batches_ok = True

for comp in components:
    comp_dir = adj_base / comp
    batch_dirs = sorted([p for p in comp_dir.glob("batch_*") if p.is_dir()]) if comp_dir.exists() else []

    comp_ok = comp_dir.exists() and len(batch_dirs) == 10
    batch_records = []

    for b in batch_dirs:
        source_files = sorted(b.glob(f"s*{comp}.txt"))

        required = {
            "input.spec": b / "input.spec",
            "material.spec": b / "material.spec",
            "material.input": b / "material.input",
            "mesh.input": b / "mesh.input",
            "Mat_0_Kappa.h5": b / "mat/h5/Mat_0_Kappa.h5",
            "Mat_0_Mu.h5": b / "mat/h5/Mat_0_Mu.h5",
            "Mat_0_Density.h5": b / "mat/h5/Mat_0_Density.h5",
        }

        required_ok = all(p.exists() for p in required.values())
        source_ok = len(source_files) == 225
        batch_ok = required_ok and source_ok

        if not batch_ok:
            comp_ok = False

        batch_records.append({
            "batch": b.name,
            "path": str(b),
            "required_ok": required_ok,
            "source_count": len(source_files),
            "source_ok": source_ok,
            "ok": batch_ok,
        })

    if not comp_ok:
        overall_batches_ok = False

    component_records.append({
        "component": comp,
        "component_dir": str(comp_dir),
        "component_dir_exists": comp_dir.exists(),
        "batch_count": len(batch_dirs),
        "ok": comp_ok,
        "batches": batch_records,
    })

ok_455B = has_pass(summary_455B)
ok_455C = has_pass(summary_455C)
overall_ok = ok_455B and ok_455C and overall_batches_ok

record = {
    "created": datetime.now().isoformat(),
    "task": "Task 1D prepare_adjoint_generic",
    "mode": args.mode,
    "iter_k": k,
    "iter_kp1": kp1,
    "transition": transition,
    "run_result_root": str(run_result_root),
    "run_data_root": str(run_data_root),
    "adj_base": str(adj_base),
    "summary_455B": str(summary_455B),
    "summary_455C": str(summary_455C),
    "ok_455B_PASS": ok_455B,
    "ok_455C_PASS": ok_455C,
    "overall_batches_ok": overall_batches_ok,
    "components": component_records,
    "overall_ok": overall_ok,
    "result": "PASS" if overall_ok else "CHECK_NEEDED",
    "meaning": "This generic wrapper can identify and validate prepared adjoint batches for any iter_k.",
}

out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

lines = []
lines.append("Task 1D prepare_adjoint_generic status")
lines.append("======================================")
lines.append("")
lines.append(f"created = {record['created']}")
lines.append(f"transition = {transition}")
lines.append(f"iter_k = {k}")
lines.append(f"iter_kp1 = {kp1}")
lines.append("")
lines.append("Derived paths:")
lines.append(f"  run_result_root = {run_result_root}")
lines.append(f"  run_data_root = {run_data_root}")
lines.append(f"  residual_dir = {residual_dir}")
lines.append(f"  adj_base = {adj_base}")
lines.append("")
lines.append("PASS summaries:")
lines.append(f"  455B exists = {summary_455B.exists()}")
lines.append(f"  455B RESULT PASS = {ok_455B}")
lines.append(f"  455C exists = {summary_455C.exists()}")
lines.append(f"  455C RESULT PASS = {ok_455C}")
lines.append("")
lines.append("Component checks:")
for cr in component_records:
    lines.append("-" * 60)
    lines.append(f"component = {cr['component']}")
    lines.append(f"component_dir_exists = {cr['component_dir_exists']}")
    lines.append(f"batch_count = {cr['batch_count']}")
    lines.append(f"ok = {cr['ok']}")
    for br in cr["batches"][:10]:
        lines.append(f"  {br['batch']}: required_ok={br['required_ok']} source_count={br['source_count']} source_ok={br['source_ok']} ok={br['ok']}")
lines.append("")
if summary_455B.exists():
    lines.append("455B tail:")
    lines.extend("  " + x for x in tail_text(summary_455B, 20))
    lines.append("")
if summary_455C.exists():
    lines.append("455C tail:")
    lines.extend("  " + x for x in tail_text(summary_455C, 20))
    lines.append("")
lines.append("Interpretation:")
lines.append("  This is not an adjoint SEM3D run.")
lines.append("  This only verifies that x/y/z adjoint batches are prepared and auditable from iter_k.")
lines.append("")
lines.append(f"RESULT = {record['result']}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if record["result"] != "PASS":
    sys.exit(2)
