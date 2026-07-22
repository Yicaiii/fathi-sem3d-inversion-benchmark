from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import numpy as np
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

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
component_rhs_dir = run_result_root / "component_rhs"
component_rhs_dir.mkdir(parents=True, exist_ok=True)

def load_component(comp):
    lam_path = component_rhs_dir / f"full_grid_trace_RHS_{comp}_lambda.npy"
    mu_path = component_rhs_dir / f"full_grid_trace_RHS_{comp}_mu.npy"
    coords_path = component_rhs_dir / f"full_grid_trace_RHS_{comp}_coords.npy"
    summary_path = component_rhs_dir / f"full_grid_trace_RHS_{comp}_summary.txt"

    missing = [p for p in [lam_path, mu_path, coords_path, summary_path] if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing files for component {comp}: " + ", ".join(str(p) for p in missing))

    if "RESULT = PASS" not in summary_path.read_text(errors="ignore"):
        raise RuntimeError(f"Component {comp} summary does not contain RESULT = PASS: {summary_path}")

    lam = np.load(lam_path)
    mu = np.load(mu_path)
    coords = np.load(coords_path)

    return {
        "component": comp,
        "lambda": lam,
        "mu": mu,
        "coords": coords,
        "paths": {
            "lambda": str(lam_path),
            "mu": str(mu_path),
            "coords": str(coords_path),
            "summary": str(summary_path),
        },
    }

components = {c: load_component(c) for c in ["x", "y", "z"]}

# Shape checks
for c, d in components.items():
    if d["lambda"].shape != (expected_n,):
        raise RuntimeError(f"Unexpected lambda shape for {c}: {d['lambda'].shape}")
    if d["mu"].shape != (expected_n,):
        raise RuntimeError(f"Unexpected mu shape for {c}: {d['mu'].shape}")
    if d["coords"].shape != (expected_n, 3):
        raise RuntimeError(f"Unexpected coords shape for {c}: {d['coords'].shape}")

    if not np.all(np.isfinite(d["lambda"])):
        raise RuntimeError(f"Non-finite lambda RHS for {c}")
    if not np.all(np.isfinite(d["mu"])):
        raise RuntimeError(f"Non-finite mu RHS for {c}")
    if not np.all(np.isfinite(d["coords"])):
        raise RuntimeError(f"Non-finite coords for {c}")

# Coordinate consistency checks
coords_x = components["x"]["coords"]
coord_diff_y = float(np.max(np.abs(coords_x - components["y"]["coords"])))
coord_diff_z = float(np.max(np.abs(coords_x - components["z"]["coords"])))

if coord_diff_y > 1e-12:
    raise RuntimeError(f"Coordinates mismatch between x and y: max diff = {coord_diff_y}")
if coord_diff_z > 1e-12:
    raise RuntimeError(f"Coordinates mismatch between x and z: max diff = {coord_diff_z}")

rhs_lambda_total = (
    components["x"]["lambda"]
    + components["y"]["lambda"]
    + components["z"]["lambda"]
)
rhs_mu_total = (
    components["x"]["mu"]
    + components["y"]["mu"]
    + components["z"]["mu"]
)
coords_total = coords_x.copy()

out_lam = component_rhs_dir / "full_grid_trace_RHS_total_lambda.npy"
out_mu = component_rhs_dir / "full_grid_trace_RHS_total_mu.npy"
out_coords = component_rhs_dir / "full_grid_trace_RHS_total_coords.npy"
out_txt = component_rhs_dir / "full_grid_trace_RHS_total_summary.txt"

np.save(out_lam, rhs_lambda_total)
np.save(out_mu, rhs_mu_total)
np.save(out_coords, coords_total)

def stats(name, arr):
    return [
        f"{name} finite = {np.count_nonzero(np.isfinite(arr))} / {arr.size}",
        f"{name} min = {float(np.min(arr)):.16e}",
        f"{name} max = {float(np.max(arr)):.16e}",
        f"{name} maxabs = {float(np.max(np.abs(arr))):.16e}",
        f"{name} l2 = {float(np.sqrt(np.sum(arr * arr))):.16e}",
    ]

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "component_rhs_dir": str(component_rhs_dir),
    "expected_n": expected_n,
    "coord_diff_y": coord_diff_y,
    "coord_diff_z": coord_diff_z,
    "outputs": {
        "lambda_total": str(out_lam),
        "mu_total": str(out_mu),
        "coords_total": str(out_coords),
        "summary": str(out_txt),
    },
    "lambda_total_shape": rhs_lambda_total.shape,
    "mu_total_shape": rhs_mu_total.shape,
    "coords_total_shape": coords_total.shape,
    "result": "PASS",
}

json_out = ROOT / "benchmark_fathi_strict/reports/rhs_run" / f"{transition}_RHS_total_assemble.json"
json_out.parent.mkdir(parents=True, exist_ok=True)
json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

lines = []
lines.append("RHS TOTAL SUMMARY")
lines.append("=================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"component_rhs_dir = {component_rhs_dir}")
lines.append("")
lines.append("Input components:")
for c in ["x", "y", "z"]:
    lines.append(f"  component = {c}")
    lines.append(f"    lambda = {components[c]['paths']['lambda']}")
    lines.append(f"    mu     = {components[c]['paths']['mu']}")
    lines.append(f"    coords = {components[c]['paths']['coords']}")
lines.append("")
lines.append("Coordinate consistency:")
lines.append(f"  max |coords_x - coords_y| = {coord_diff_y:.16e}")
lines.append(f"  max |coords_x - coords_z| = {coord_diff_z:.16e}")
lines.append("")
lines.append("Total RHS shapes:")
lines.append(f"  lambda_total shape = {rhs_lambda_total.shape}")
lines.append(f"  mu_total shape = {rhs_mu_total.shape}")
lines.append(f"  coords_total shape = {coords_total.shape}")
lines.append("")
lines.append("Total RHS stats:")
lines.extend("  " + x for x in stats("rhs_lambda_total", rhs_lambda_total))
lines.extend("  " + x for x in stats("rhs_mu_total", rhs_mu_total))
lines.append("")
lines.append("Outputs:")
lines.append(f"  {out_lam}")
lines.append(f"  {out_mu}")
lines.append(f"  {out_coords}")
lines.append("")
lines.append("Interpretation:")
lines.append("  RHS_total is complete.")
lines.append("  This is still the RHS of the discrete control equation, not yet the final Mtilde gradient.")
lines.append("  Next step: solve Mtilde g = RHS_total.")
lines.append("")
lines.append(f"json = {json_out}")
lines.append("")
lines.append("RESULT = PASS")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
