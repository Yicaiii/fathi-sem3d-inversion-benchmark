from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--steps-mpa", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.00])
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"
shape = tuple(config.get("material_shape", [41, 33, 33]))

state_in = ROOT / config["state_dir"] / f"iter_{k:03d}_state_v2_corrected.npz"
run_result_root = ROOT / config["run_result_root"] / transition
mtilde_dir = run_result_root / "mtilde_solve"
candidate_root = run_result_root / "candidates"
candidate_root.mkdir(parents=True, exist_ok=True)

parent_accepted = ROOT / config["run_data_root"] / f"iter_{k:03d}" / "accepted"
parent_h5_dir = parent_accepted / "mat/h5"

gL_path = mtilde_dir / "g_lambda_mtilde_q1_interior_solve_rhs_total.npy"
gM_path = mtilde_dir / "g_mu_mtilde_q1_interior_solve_rhs_total.npy"
idx_path = mtilde_dir / "Mtilde_q1_consistent_interior_38440_indices.npy"

required = [
    state_in,
    gL_path,
    gM_path,
    idx_path,
    parent_h5_dir / "Mat_0_Kappa.h5",
    parent_h5_dir / "Mat_0_Mu.h5",
    parent_h5_dir / "Mat_0_Density.h5",
]

missing = [p for p in required if not p.exists()]
if missing:
    print("Missing inputs:")
    for p in missing:
        print(" ", p)
    sys.exit(1)

def find_dataset_name(path: Path, expected_shape):
    matches = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if tuple(obj.shape) == tuple(expected_shape):
                    matches.append(name)
        h.visititems(visit)
    if not matches:
        raise RuntimeError(f"No dataset with shape {expected_shape} in {path}")
    return matches[0]

def read_h5_field(path: Path, expected_shape):
    name = find_dataset_name(path, expected_shape)
    with h5py.File(path, "r") as h:
        arr = np.asarray(h[name], dtype=np.float64)
    return arr, name

def overwrite_h5_field(path: Path, data: np.ndarray, dataset_name: str):
    with h5py.File(path, "r+") as h:
        if dataset_name not in h:
            raise RuntimeError(f"Dataset {dataset_name} not found in {path}")
        if tuple(h[dataset_name].shape) != tuple(data.shape):
            raise RuntimeError(f"Shape mismatch for {path}:{dataset_name}: {h[dataset_name].shape} vs {data.shape}")
        h[dataset_name][...] = data

def pick_state_array(state, names):
    for n in names:
        if n in state.files:
            arr = state[n]
            if tuple(arr.shape) == shape:
                return arr.astype(np.float64), n
    return None, None

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

state = np.load(state_in)

kappa_parent_h5, kappa_dataset = read_h5_field(parent_h5_dir / "Mat_0_Kappa.h5", shape)
mu_parent_h5, mu_dataset = read_h5_field(parent_h5_dir / "Mat_0_Mu.h5", shape)
density_parent_h5, density_dataset = read_h5_field(parent_h5_dir / "Mat_0_Density.h5", shape)

mu_parent, mu_source = pick_state_array(state, ["mu", "mu_field", "mu_grid", "Mu", "MU"])
if mu_parent is None:
    mu_parent = mu_parent_h5.copy()
    mu_source = "parent_h5_Mu"

kappa_parent, kappa_source = pick_state_array(state, ["kappa", "kappa_field", "kappa_grid", "Kappa", "KAPPA"])
if kappa_parent is None:
    kappa_parent = kappa_parent_h5.copy()
    kappa_source = "parent_h5_Kappa"

lambda_parent, lambda_source = pick_state_array(state, ["lambda", "lambda_field", "lambda_grid", "lam", "Lambda", "LAMBDA"])
if lambda_parent is None:
    lambda_parent = kappa_parent - (2.0 / 3.0) * mu_parent
    lambda_source = "computed_from_kappa_minus_2over3_mu"

density_parent, density_source = pick_state_array(state, ["density", "rho", "Density", "RHO"])
if density_parent is None:
    density_parent = density_parent_h5.copy()
    density_source = "parent_h5_Density"

gL = np.load(gL_path).astype(np.float64)
gM = np.load(gM_path).astype(np.float64)
idx = np.load(idx_path).astype(np.int64)

if gL.shape != (38440,) or gM.shape != (38440,) or idx.shape != (38440,):
    raise RuntimeError(f"Unexpected gradient/index shapes: gL={gL.shape}, gM={gM.shape}, idx={idx.shape}")

if not np.all(np.isfinite(gL)) or not np.all(np.isfinite(gM)):
    raise RuntimeError("Non-finite gradient values.")

scaleL = float(np.max(np.abs(gL)))
scaleM = float(np.max(np.abs(gM)))
if scaleL <= 0 or scaleM <= 0:
    raise RuntimeError(f"Bad gradient scale: scaleL={scaleL}, scaleM={scaleM}")

n_total = int(np.prod(shape))
if int(idx.min()) < 0 or int(idx.max()) >= n_total:
    raise RuntimeError(f"indices out of range for shape {shape}: min={idx.min()} max={idx.max()}")

flat_lambda_parent = lambda_parent.reshape(-1)
flat_mu_parent = mu_parent.reshape(-1)

records = []
created = datetime.now().isoformat()

for step_mpa in args.steps_mpa:
    step_pa = float(step_mpa) * 1e6
    label = f"{step_mpa:.2f}".replace(".", "p") + "MPa"
    cand_name = f"line_search_neg_mtilde_{label}"
    cand_dir = candidate_root / cand_name
    cand_h5_dir = cand_dir / "mat/h5"
    cand_dir.mkdir(parents=True, exist_ok=True)

    if cand_h5_dir.exists():
        shutil.rmtree(cand_h5_dir)
    shutil.copytree(parent_h5_dir, cand_h5_dir)

    flat_lambda = flat_lambda_parent.copy()
    flat_mu = flat_mu_parent.copy()

    flat_lambda[idx] = flat_lambda[idx] - step_pa * (gL / scaleL)
    flat_mu[idx] = flat_mu[idx] - step_pa * (gM / scaleM)

    lambda_new = flat_lambda.reshape(shape)
    mu_new = flat_mu.reshape(shape)
    kappa_new = lambda_new + (2.0 / 3.0) * mu_new
    density_new = density_parent.copy()

    finite_ok = (
        np.all(np.isfinite(lambda_new))
        and np.all(np.isfinite(mu_new))
        and np.all(np.isfinite(kappa_new))
        and np.all(np.isfinite(density_new))
    )
    positive_ok = (
        np.min(lambda_new) > 0
        and np.min(mu_new) > 0
        and np.min(kappa_new) > 0
        and np.min(density_new) > 0
    )

    overwrite_h5_field(cand_h5_dir / "Mat_0_Kappa.h5", kappa_new, kappa_dataset)
    overwrite_h5_field(cand_h5_dir / "Mat_0_Mu.h5", mu_new, mu_dataset)
    overwrite_h5_field(cand_h5_dir / "Mat_0_Density.h5", density_new, density_dataset)

    state_npz = cand_dir / f"{cand_name}_state_candidate.npz"
    np.savez_compressed(
        state_npz,
        lambda_field=lambda_new,
        mu=mu_new,
        kappa=kappa_new,
        density=density_new,
        parent_state=str(state_in),
        iter_k=k,
        iter_kp1=kp1,
        transition=transition,
        step_mpa=step_mpa,
        step_pa=step_pa,
        gradient_lambda=str(gL_path),
        gradient_mu=str(gM_path),
        gradient_indices=str(idx_path),
        direction="negative_mtilde_gradient_maxabs_normalized",
    )

    rec = {
        "candidate": cand_name,
        "label": label,
        "step_mpa": step_mpa,
        "step_pa": step_pa,
        "candidate_dir": str(cand_dir),
        "mat_h5_dir": str(cand_h5_dir),
        "state_npz": str(state_npz),
        "finite_ok": bool(finite_ok),
        "positive_ok": bool(positive_ok),
        "lambda": stats(lambda_new),
        "mu": stats(mu_new),
        "kappa": stats(kappa_new),
        "density": stats(density_new),
        "max_abs_delta_lambda": float(np.max(np.abs(lambda_new - lambda_parent))),
        "max_abs_delta_mu": float(np.max(np.abs(mu_new - mu_parent))),
    }
    records.append(rec)

report_dir = ROOT / "benchmark_fathi_strict/reports/candidate_generation"
report_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "created": created,
    "transition": transition,
    "shape": list(shape),
    "state_in": str(state_in),
    "lambda_source": lambda_source,
    "mu_source": mu_source,
    "kappa_source": kappa_source,
    "density_source": density_source,
    "gradient_scale_lambda": scaleL,
    "gradient_scale_mu": scaleM,
    "candidate_root": str(candidate_root),
    "records": records,
}

all_ok = all(r["finite_ok"] and r["positive_ok"] for r in records)
payload["result"] = "PASS" if all_ok else "CHECK_NEEDED"

json_out = report_dir / f"{transition}_candidate_generation.json"
txt_out = report_dir / f"{transition}_candidate_generation.txt"

json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Task 4B candidate generation from Mtilde gradient")
lines.append("================================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"state_in = {state_in}")
lines.append(f"candidate_root = {candidate_root}")
lines.append("")
lines.append("Sources:")
lines.append(f"  lambda_source = {lambda_source}")
lines.append(f"  mu_source = {mu_source}")
lines.append(f"  kappa_source = {kappa_source}")
lines.append(f"  density_source = {density_source}")
lines.append("")
lines.append("Gradient normalization:")
lines.append(f"  scale_lambda = {scaleL:.16e}")
lines.append(f"  scale_mu = {scaleM:.16e}")
lines.append("")
lines.append("Candidates:")
for r in records:
    lines.append("-" * 80)
    lines.append(f"candidate = {r['candidate']}")
    lines.append(f"  step_mpa = {r['step_mpa']}")
    lines.append(f"  candidate_dir = {r['candidate_dir']}")
    lines.append(f"  state_npz = {r['state_npz']}")
    lines.append(f"  finite_ok = {r['finite_ok']}")
    lines.append(f"  positive_ok = {r['positive_ok']}")
    lines.append(f"  lambda min/max = {r['lambda']['min']:.16e} {r['lambda']['max']:.16e}")
    lines.append(f"  mu min/max     = {r['mu']['min']:.16e} {r['mu']['max']:.16e}")
    lines.append(f"  kappa min/max  = {r['kappa']['min']:.16e} {r['kappa']['max']:.16e}")
    lines.append(f"  max_abs_delta_lambda = {r['max_abs_delta_lambda']:.16e}")
    lines.append(f"  max_abs_delta_mu     = {r['max_abs_delta_mu']:.16e}")

lines.append("")
lines.append(f"json = {json_out}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

txt_out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if payload["result"] != "PASS":
    sys.exit(2)
