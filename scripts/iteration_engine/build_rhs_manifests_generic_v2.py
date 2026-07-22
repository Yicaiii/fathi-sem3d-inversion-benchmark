from pathlib import Path
import os
from datetime import datetime
import argparse
import csv
import json
import re
import sys

import h5py
import numpy as np

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()
UU_RE = re.compile(r"^UU_\d+$")

def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))

def sorted_uu_keys(h):
    keys = [k for k in h.keys() if UU_RE.match(k)]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))

def batch_name_from_trace_file(p: Path) -> str:
    # .../batch_000/traces/capteurs.0000.h5 -> batch_000
    # .../strict_full_forward_000/traces/capteurs.0000.h5 -> strict_full_forward_000
    if p.parent.name == "traces":
        return p.parent.parent.name
    return p.parent.name

def scan_trace_files(trace_dir: Path):
    if not trace_dir.exists():
        return []
    return sorted(trace_dir.glob("capteurs.*.h5"))

def write_manifest_from_trace_files(trace_files, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    finite_pos_count = 0
    bad_rows = []

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "trace_file",
                "receiver_key",
                "x",
                "y",
                "z",
                "nsteps",
                "ncols",
                "t0",
                "t1",
            ],
        )
        w.writeheader()

        for trace_file in trace_files:
            batch = batch_name_from_trace_file(trace_file)
            trace_rel = rel(trace_file)

            try:
                with h5py.File(trace_file, "r") as h:
                    for key in sorted_uu_keys(h):
                        arr = h[key]
                        if len(arr.shape) != 2:
                            bad_rows.append({
                                "trace_file": trace_rel,
                                "receiver_key": key,
                                "reason": f"unexpected_shape_{arr.shape}",
                            })
                            continue

                        nsteps = int(arr.shape[0])
                        ncols = int(arr.shape[1])

                        pos_key = key + "_pos"
                        if pos_key in h:
                            pos = np.asarray(h[pos_key]).reshape(-1)
                            if pos.size >= 3:
                                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                            else:
                                x = y = z = float("nan")
                        else:
                            x = y = z = float("nan")

                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            finite_pos_count += 1

                        try:
                            t0 = float(arr[0, 0])
                            t1 = float(arr[nsteps - 1, 0])
                        except Exception:
                            t0 = float("nan")
                            t1 = float("nan")

                        w.writerow({
                            "batch": batch,
                            "trace_file": trace_rel,
                            "receiver_key": key,
                            "x": x,
                            "y": y,
                            "z": z,
                            "nsteps": nsteps,
                            "ncols": ncols,
                            "t0": t0,
                            "t1": t1,
                        })
                        row_count += 1

            except Exception as e:
                bad_rows.append({
                    "trace_file": trace_rel,
                    "receiver_key": "",
                    "reason": repr(e),
                })

    return {
        "csv": str(out_csv),
        "trace_file_count": len(trace_files),
        "row_count": row_count,
        "finite_pos_count": finite_pos_count,
        "bad_rows": bad_rows[:50],
        "bad_row_count": len(bad_rows),
    }

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
expected_n = int(config.get("interior_gradient_size", 38440))

run_result_root = ROOT / config["run_result_root"] / transition
run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"

forward_trace_dir = run_data_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000/traces"
adj_base = run_data_root / "adjoint_full_grid_batches"

out_dir = run_result_root / "rhs_manifests"
out_dir.mkdir(parents=True, exist_ok=True)

report_dir = ROOT / "benchmark_fathi_strict/reports/rhs_discovery"
report_dir.mkdir(parents=True, exist_ok=True)

forward_files = scan_trace_files(forward_trace_dir)
forward_result = write_manifest_from_trace_files(
    forward_files,
    out_dir / "forward_full_grid_trace_manifest.csv",
)

component_results = {}
for comp in ["x", "y", "z"]:
    trace_files = []
    for i in range(10):
        trace_dir = adj_base / comp / f"batch_{i:03d}" / "traces"
        trace_files.extend(scan_trace_files(trace_dir))

    component_results[comp] = write_manifest_from_trace_files(
        trace_files,
        out_dir / f"adjoint_{comp}_full_grid_trace_manifest.csv",
    )

all_counts = [forward_result["row_count"]] + [
    component_results[c]["row_count"] for c in ["x", "y", "z"]
]
all_finite = [forward_result["finite_pos_count"]] + [
    component_results[c]["finite_pos_count"] for c in ["x", "y", "z"]
]
all_bad = [forward_result["bad_row_count"]] + [
    component_results[c]["bad_row_count"] for c in ["x", "y", "z"]
]

overall_ok = (
    all(x == expected_n for x in all_counts)
    and all(x == expected_n for x in all_finite)
    and all(x == 0 for x in all_bad)
)

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "expected_n": expected_n,
    "forward_trace_dir": str(forward_trace_dir),
    "adj_base": str(adj_base),
    "out_dir": str(out_dir),
    "forward": forward_result,
    "components": component_results,
    "overall_ok": overall_ok,
    "result": "PASS" if overall_ok else "CHECK_NEEDED",
}

json_out = report_dir / f"{transition}_rhs_manifest_build_v2.json"
txt_out = report_dir / f"{transition}_rhs_manifest_build_v2.txt"
json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("RHS manifest build generic v2")
lines.append("=============================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"expected_n = {expected_n}")
lines.append("")
lines.append("Forward:")
lines.append(f"  trace_file_count = {forward_result['trace_file_count']}")
lines.append(f"  row_count = {forward_result['row_count']}")
lines.append(f"  finite_pos_count = {forward_result['finite_pos_count']}")
lines.append(f"  bad_row_count = {forward_result['bad_row_count']}")
lines.append(f"  csv = {forward_result['csv']}")
lines.append("")
lines.append("Adjoint:")
for comp in ["x", "y", "z"]:
    r = component_results[comp]
    lines.append(f"  component = {comp}")
    lines.append(f"    trace_file_count = {r['trace_file_count']}")
    lines.append(f"    row_count = {r['row_count']}")
    lines.append(f"    finite_pos_count = {r['finite_pos_count']}")
    lines.append(f"    bad_row_count = {r['bad_row_count']}")
    lines.append(f"    csv = {r['csv']}")
lines.append("")
lines.append("Expected:")
lines.append("  each manifest should have 38440 data rows")
lines.append("  wc -l should show 38441 including header")
lines.append("")
lines.append(f"json = {json_out}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

txt_out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if not overall_ok:
    sys.exit(2)
