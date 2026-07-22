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

state_in = ROOT / config["state_dir"] / f"iter_{k:03d}_state_v2_corrected.npz"
run_result_root = ROOT / config["run_result_root"] / transition
mtilde_dir = run_result_root / "mtilde_solve"
parent_accepted = ROOT / config["run_data_root"] / f"iter_{k:03d}" / "accepted"

paths = {
    "state_in": state_in,
    "g_lambda": mtilde_dir / "g_lambda_mtilde_q1_interior_solve_rhs_total.npy",
    "g_mu": mtilde_dir / "g_mu_mtilde_q1_interior_solve_rhs_total.npy",
    "g_coords": mtilde_dir / "g_mtilde_q1_interior_solve_rhs_total_coords.npy",
    "indices": mtilde_dir / "Mtilde_q1_consistent_interior_38440_indices.npy",
    "parent_kappa_h5": parent_accepted / "mat/h5/Mat_0_Kappa.h5",
    "parent_mu_h5": parent_accepted / "mat/h5/Mat_0_Mu.h5",
    "parent_density_h5": parent_accepted / "mat/h5/Mat_0_Density.h5",
}

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

def h5_datasets(path):
    out = []
    if not path.exists():
        return out
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out.append({
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                })
        h.visititems(visit)
    return out

out_dir = ROOT / "benchmark_fathi_strict/reports/candidate_generation"
out_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()
payload = {
    "created": created,
    "transition": transition,
    "paths": {k2: str(v) for k2, v in paths.items()},
    "exists": {k2: v.exists() for k2, v in paths.items()},
}

lines = []
lines.append("Task 4A candidate input audit")
lines.append("=============================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append("")

missing = [k2 for k2, v in paths.items() if not v.exists()]
lines.append("Required files:")
for k2, p in paths.items():
    lines.append(f"  {k2}: exists={p.exists()} path={p}")

if missing:
    result = "FAIL_MISSING"
else:
    state = np.load(state_in)
    payload["state_keys"] = list(state.files)

    lines.append("")
    lines.append("State keys:")
    for key in state.files:
        a = state[key]
        if hasattr(a, "shape"):
            lines.append(f"  {key}: shape={a.shape} dtype={a.dtype}")
        else:
            lines.append(f"  {key}: {a}")

    gL = np.load(paths["g_lambda"])
    gM = np.load(paths["g_mu"])
    gC = np.load(paths["g_coords"])
    idx = np.load(paths["indices"])

    payload["gradient_stats"] = {
        "g_lambda": stats(gL),
        "g_mu": stats(gM),
        "g_coords": stats(gC),
        "indices_shape": list(idx.shape),
        "indices_min": int(np.min(idx)),
        "indices_max": int(np.max(idx)),
        "indices_unique": int(np.unique(idx).size),
    }

    lines.append("")
    lines.append("Gradient stats:")
    lines.append(f"  g_lambda shape = {gL.shape}, maxabs = {payload['gradient_stats']['g_lambda']['maxabs']:.16e}")
    lines.append(f"  g_mu     shape = {gM.shape}, maxabs = {payload['gradient_stats']['g_mu']['maxabs']:.16e}")
    lines.append(f"  g_coords shape = {gC.shape}")
    lines.append(f"  indices shape = {idx.shape}, min = {int(np.min(idx))}, max = {int(np.max(idx))}, unique = {int(np.unique(idx).size)}")

    payload["h5_datasets"] = {
        "kappa": h5_datasets(paths["parent_kappa_h5"]),
        "mu": h5_datasets(paths["parent_mu_h5"]),
        "density": h5_datasets(paths["parent_density_h5"]),
    }

    lines.append("")
    lines.append("Parent H5 datasets:")
    for name in ["kappa", "mu", "density"]:
        lines.append(f"  {name}:")
        for d in payload["h5_datasets"][name]:
            lines.append(f"    {d['name']} shape={d['shape']} dtype={d['dtype']}")

    ok = (
        gL.shape == (38440,)
        and gM.shape == (38440,)
        and gC.shape == (38440, 3)
        and idx.shape == (38440,)
        and np.unique(idx).size == 38440
        and np.all(np.isfinite(gL))
        and np.all(np.isfinite(gM))
    )

    result = "PASS" if ok else "CHECK_NEEDED"

payload["result"] = result

out_json = out_dir / f"{transition}_candidate_input_audit.json"
out_txt = out_dir / f"{transition}_candidate_input_audit.txt"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
