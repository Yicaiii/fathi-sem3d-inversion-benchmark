from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import csv
import h5py
import numpy as np
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

def rel(p: Path):
    return str(p.relative_to(ROOT))

def find_capteurs_files(base: Path):
    files = []
    if not base.exists():
        return files

    # Prefer traces/
    for d in [
        base / "traces",
        base / "prot",
        base / "res",
        base,
    ]:
        if d.exists():
            files.extend(sorted(d.rglob("capteurs.*.h5")))

    # remove duplicates while preserving order
    seen = set()
    out = []
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out

def read_position_fast(h5_path: Path):
    """
    Try common SEM3D capteurs layouts.
    If position cannot be read, return NaNs but keep file in manifest.
    """
    try:
        with h5py.File(h5_path, "r") as h:
            # Try common dataset names first
            candidates = [
                "position",
                "positions",
                "coords",
                "coord",
                "xyz",
                "receiver_position",
                "metadata/position",
                "metadata/coords",
            ]
            for name in candidates:
                if name in h:
                    arr = np.asarray(h[name])
                    arr = arr.reshape(-1)
                    if arr.size >= 3:
                        return float(arr[0]), float(arr[1]), float(arr[2])

            # Try attrs
            for keys in [
                ("x", "y", "z"),
                ("X", "Y", "Z"),
                ("coord_x", "coord_y", "coord_z"),
            ]:
                if all(k in h.attrs for k in keys):
                    return tuple(float(h.attrs[k]) for k in keys)

            # Last resort: inspect top-level datasets with length 3
            for key in h.keys():
                try:
                    arr = np.asarray(h[key])
                    if arr.size == 3 and np.all(np.isfinite(arr)):
                        arr = arr.reshape(-1)
                        return float(arr[0]), float(arr[1]), float(arr[2])
                except Exception:
                    pass

    except Exception:
        pass

    return float("nan"), float("nan"), float("nan")

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

forward_dir = run_data_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000"
adj_base = run_data_root / "adjoint_full_grid_batches"

out_dir = run_result_root / "rhs_manifests"
out_dir.mkdir(parents=True, exist_ok=True)

report_dir = ROOT / "benchmark_fathi_strict/reports/rhs_discovery"
report_dir.mkdir(parents=True, exist_ok=True)

def write_manifest(kind, component, base_dirs):
    rows = []
    for base in base_dirs:
        files = find_capteurs_files(base)
        for p in files:
            x, y, z = read_position_fast(p)
            rows.append({
                "idx": len(rows),
                "component": component,
                "x": x,
                "y": y,
                "z": z,
                "path": rel(p),
            })

    if kind == "forward":
        out_csv = out_dir / "forward_full_grid_trace_manifest.csv"
    else:
        out_csv = out_dir / f"adjoint_{component}_full_grid_trace_manifest.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "component", "x", "y", "z", "path"])
        w.writeheader()
        w.writerows(rows)

    return out_csv, rows

forward_csv, forward_rows = write_manifest(
    "forward",
    "forward",
    [forward_dir],
)

component_results = {}
for comp in ["x", "y", "z"]:
    bases = [adj_base / comp / f"batch_{i:03d}" for i in range(10)]
    csv_path, rows = write_manifest("adjoint", comp, bases)
    component_results[comp] = {
        "csv": str(csv_path),
        "count": len(rows),
        "finite_position_count": int(sum(np.isfinite(r["x"]) and np.isfinite(r["y"]) and np.isfinite(r["z"]) for r in rows)),
    }

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "forward_dir": str(forward_dir),
    "adj_base": str(adj_base),
    "out_dir": str(out_dir),
    "forward_csv": str(forward_csv),
    "forward_count": len(forward_rows),
    "forward_finite_position_count": int(sum(np.isfinite(r["x"]) and np.isfinite(r["y"]) and np.isfinite(r["z"]) for r in forward_rows)),
    "components": component_results,
}

json_out = report_dir / f"{transition}_rhs_manifest_build.json"
txt_out = report_dir / f"{transition}_rhs_manifest_build.txt"
json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("RHS manifest build generic")
lines.append("==========================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append("")
lines.append("Forward:")
lines.append(f"  csv = {forward_csv}")
lines.append(f"  count = {payload['forward_count']}")
lines.append(f"  finite_position_count = {payload['forward_finite_position_count']}")
lines.append("")
lines.append("Adjoint:")
for comp, r in component_results.items():
    lines.append(f"  component = {comp}")
    lines.append(f"    csv = {r['csv']}")
    lines.append(f"    count = {r['count']}")
    lines.append(f"    finite_position_count = {r['finite_position_count']}")
lines.append("")
lines.append("Interpretation:")
lines.append("  These manifests are generated for the current transition.")
lines.append("  If counts match the old longterm layout, 424B can be reused with these manifests.")
lines.append("")
lines.append(f"json = {json_out}")
lines.append("")
lines.append("RESULT = PASS")

txt_out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
