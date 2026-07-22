from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import numpy as np
import h5py
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"
shape = tuple(config.get("material_shape", [41, 33, 33]))

run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"
run_result_root = ROOT / config["run_result_root"] / transition
state_path = ROOT / config["state_dir"] / f"iter_{kp1:03d}_state_v2_corrected.npz"
accepted_dir = run_data_root / "accepted"
accepted_summary = accepted_dir / "accepted_summary.txt"
misfit_summary = run_result_root / "candidate_misfits/line_search_neg_mtilde_1p00MPa_misfit_summary_v2.json"
acceptance_json = ROOT / f"benchmark_fathi_strict/reports/acceptance/{transition}_line_search_neg_mtilde_1p00MPa_acceptance_v2.json"

paths = {
    "state_path": state_path,
    "accepted_dir": accepted_dir,
    "accepted_summary": accepted_summary,
    "misfit_summary": misfit_summary,
    "acceptance_json": acceptance_json,
    "kappa_h5": accepted_dir / "mat/h5/Mat_0_Kappa.h5",
    "mu_h5": accepted_dir / "mat/h5/Mat_0_Mu.h5",
    "density_h5": accepted_dir / "mat/h5/Mat_0_Density.h5",
}

def find_dataset(path):
    matches = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and tuple(obj.shape) == shape:
                matches.append(name)
        h.visititems(visit)
    if not matches:
        raise RuntimeError(f"No dataset with shape {shape} in {path}")
    return matches[0]

def read_h5(path):
    ds = find_dataset(path)
    with h5py.File(path, "r") as h:
        arr = np.asarray(h[ds], dtype=np.float64)
    return arr, ds

def stats(a):
    return {
        "shape": list(a.shape),
        "finite": int(np.count_nonzero(np.isfinite(a))),
        "size": int(a.size),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "maxabs": float(np.nanmax(np.abs(a))),
        "l2": float(np.sqrt(np.nansum(a * a))),
    }

created = datetime.now().isoformat()
missing = [name for name, p in paths.items() if not p.exists()]

report_dir = ROOT / "benchmark_fathi_strict/reports/acceptance"
report_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "created": created,
    "transition": transition,
    "paths": {k2: str(v) for k2, v in paths.items()},
    "missing": missing,
}

lines = []
lines.append("Task 6A accepted state audit")
lines.append("============================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append("")

if missing:
    lines.append("Missing files:")
    for name in missing:
        lines.append(f"  {name}: {paths[name]}")
    result = "FAIL_MISSING"
else:
    state = np.load(state_path)
    acceptance = json.loads(acceptance_json.read_text())
    misfit = json.loads(misfit_summary.read_text())

    kappa_h5, kappa_ds = read_h5(paths["kappa_h5"])
    mu_h5, mu_ds = read_h5(paths["mu_h5"])
    density_h5, density_ds = read_h5(paths["density_h5"])
    lambda_h5 = kappa_h5 - (2.0 / 3.0) * mu_h5

    state_lambda = state["lambda"]
    state_mu = state["mu"]
    state_kappa = state["kappa"]
    state_density = state["density"]

    checks = {}
    checks["state_has_J"] = "J" in state.files
    checks["state_lambda_shape_ok"] = state_lambda.shape == shape
    checks["state_mu_shape_ok"] = state_mu.shape == shape
    checks["state_kappa_shape_ok"] = state_kappa.shape == shape
    checks["state_density_shape_ok"] = state_density.shape == shape

    checks["state_lambda_finite"] = bool(np.all(np.isfinite(state_lambda)))
    checks["state_mu_finite"] = bool(np.all(np.isfinite(state_mu)))
    checks["state_kappa_finite"] = bool(np.all(np.isfinite(state_kappa)))
    checks["state_density_finite"] = bool(np.all(np.isfinite(state_density)))

    checks["state_lambda_positive"] = bool(np.min(state_lambda) > 0)
    checks["state_mu_positive"] = bool(np.min(state_mu) > 0)
    checks["state_kappa_positive"] = bool(np.min(state_kappa) > 0)
    checks["state_density_positive"] = bool(np.min(state_density) > 0)

    checks["h5_state_lambda_match"] = float(np.max(np.abs(lambda_h5 - state_lambda))) < 1e-8
    checks["h5_state_mu_match"] = float(np.max(np.abs(mu_h5 - state_mu))) < 1e-8
    checks["h5_state_kappa_match"] = float(np.max(np.abs(kappa_h5 - state_kappa))) < 1e-8
    checks["h5_state_density_match"] = float(np.max(np.abs(density_h5 - state_density))) < 1e-8

    parent_J = float(acceptance["parent_J"])
    candidate_J = float(acceptance["candidate_J"])
    delta_J = float(acceptance["delta_J"])
    descent = bool(acceptance["descent"])

    checks["descent_true"] = descent
    checks["candidate_J_less_parent_J"] = candidate_J < parent_J
    checks["state_J_matches_candidate_J"] = abs(float(state["J"]) - candidate_J) < 1e-40
    checks["misfit_J_matches_candidate_J"] = abs(float(misfit["total_J"]) - candidate_J) < 1e-40

    payload.update({
        "parent_J": parent_J,
        "candidate_J": candidate_J,
        "delta_J": delta_J,
        "descent": descent,
        "state_keys": list(state.files),
        "datasets": {
            "kappa": kappa_ds,
            "mu": mu_ds,
            "density": density_ds,
        },
        "stats": {
            "lambda": stats(state_lambda),
            "mu": stats(state_mu),
            "kappa": stats(state_kappa),
            "density": stats(state_density),
        },
        "checks": checks,
    })

    result = "PASS" if all(checks.values()) else "CHECK_NEEDED"

    lines.append("Misfit:")
    lines.append(f"  parent_J = {parent_J:.16e}")
    lines.append(f"  candidate_J = {candidate_J:.16e}")
    lines.append(f"  delta_J = {delta_J:.16e}")
    lines.append(f"  descent = {descent}")
    lines.append("")
    lines.append("State:")
    lines.append(f"  state_path = {state_path}")
    lines.append(f"  accepted_dir = {accepted_dir}")
    lines.append("")
    lines.append("Material stats:")
    for name in ["lambda", "mu", "kappa", "density"]:
        s = payload["stats"][name]
        lines.append(f"  {name}: shape={s['shape']} min={s['min']:.16e} max={s['max']:.16e} finite={s['finite']}/{s['size']}")
    lines.append("")
    lines.append("Checks:")
    for k2, v in checks.items():
        lines.append(f"  {k2} = {v}")

payload["result"] = result

out_json = report_dir / f"{transition}_accepted_state_audit.json"
out_txt = report_dir / f"{transition}_accepted_state_audit.txt"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
