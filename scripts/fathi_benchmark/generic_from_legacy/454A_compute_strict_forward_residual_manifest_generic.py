from pathlib import Path
import json
import argparse
import os
import csv
import re
import numpy as np
import h5py
from datetime import datetime

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

ap = argparse.ArgumentParser()
ap.add_argument("--context", required=True)
args = ap.parse_args()

CTX = Path(args.context)
if not CTX.is_absolute():
    CTX = ROOT / CTX
ctx = json.loads(CTX.read_text())

true_dir = Path(ctx["true_observed_traces"])
sim_dir = Path(ctx["output_forward_batches_dir"]) / "strict_full_forward_000" / "traces"

outdir = Path(ctx["work_root"]) / "residual_sources"
outdir.mkdir(parents=True, exist_ok=True)

def is_uu_name(name):
    base = name.split("/")[-1]
    return re.match(r"^UU_\d+$", base) is not None

def pos_key(pos, ndigits=8):
    return tuple(round(float(x), ndigits) for x in pos)

def collect_trace_index(trace_dir):
    index = {}
    files = sorted(trace_dir.glob("capteurs.*.h5"))
    for fp in files:
        with h5py.File(fp, "r") as f:
            uu_names = []
            f.visititems(lambda name, obj: uu_names.append(name) if isinstance(obj, h5py.Dataset) and is_uu_name(name) else None)
            for name in uu_names:
                pos_name = name + "_pos"
                if pos_name not in f:
                    continue
                pos = np.asarray(f[pos_name][()], dtype=np.float64)
                key = pos_key(pos)
                index[key] = {
                    "file": str(fp),
                    "dataset": name,
                    "pos": [float(x) for x in pos],
                }
    return index

def load_uu(file_path, dataset):
    with h5py.File(file_path, "r") as f:
        arr = np.asarray(f[dataset][()], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise RuntimeError(f"Bad UU shape: {file_path} {dataset} {arr.shape}")
    t = arr[:, 0]
    u = arr[:, 1:4]
    return t, u

def strict_increasing(t):
    return np.all(np.isfinite(t)) and np.all(np.diff(t) > 0)

true_index = collect_trace_index(true_dir)
sim_index = collect_trace_index(sim_dir)

common_keys = sorted(set(true_index.keys()).intersection(sim_index.keys()))

rows = []
total_J = 0.0
total_n = 0
bad = []

for i, key in enumerate(common_keys):
    tr = true_index[key]
    sr = sim_index[key]

    try:
        tt, ut = load_uu(tr["file"], tr["dataset"])
        ts, us = load_uu(sr["file"], sr["dataset"])

        if not strict_increasing(tt):
            raise RuntimeError("true time is not strictly increasing")
        if not strict_increasing(ts):
            raise RuntimeError("sim time is not strictly increasing")
        if not np.all(np.isfinite(ut)):
            raise RuntimeError("true displacement nonfinite")
        if not np.all(np.isfinite(us)):
            raise RuntimeError("sim displacement nonfinite")

        t0 = max(float(tt[0]), float(ts[0]))
        t1 = min(float(tt[-1]), float(ts[-1]))

        mask = (tt >= t0) & (tt <= t1)
        t_eval = tt[mask]
        u_true = ut[mask, :]

        if len(t_eval) < 2:
            raise RuntimeError("insufficient time overlap")

        u_sim_interp = np.empty_like(u_true)
        for c in range(3):
            u_sim_interp[:, c] = np.interp(t_eval, ts, us[:, c])

        residual = u_sim_interp - u_true
        integrand = np.sum(residual * residual, axis=1)
        local_J = 0.5 * float(np.trapezoid(integrand, t_eval))

        total_J += local_J
        total_n += int(residual.size)

        rows.append({
            "key_x": key[0],
            "key_y": key[1],
            "key_z": key[2],
            "true_file": tr["file"],
            "true_dataset": tr["dataset"],
            "sim_file": sr["file"],
            "sim_dataset": sr["dataset"],
            "n_time": int(len(t_eval)),
            "n_values": int(residual.size),
            "local_J": local_J,
            "max_abs_residual": float(np.max(np.abs(residual))),
            "rms_residual": float(np.sqrt(np.mean(residual * residual))),
        })

    except Exception as e:
        bad.append({
            "key": key,
            "true": tr,
            "sim": sr,
            "error": str(e),
        })

csv_path = outdir / "454A_strict_forward_residual_manifest.csv"
json_path = outdir / "454A_strict_forward_residual_manifest.json"
txt_path = outdir / "454A_strict_forward_residual_manifest_summary.txt"

with csv_path.open("w", newline="") as f:
    fieldnames = [
        "key_x", "key_y", "key_z",
        "true_file", "true_dataset",
        "sim_file", "sim_dataset",
        "n_time", "n_values",
        "local_J", "max_abs_residual", "rms_residual",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

json_path.write_text(json.dumps({
    "created": datetime.now().isoformat(),
    "true_dir": str(true_dir),
    "sim_dir": str(sim_dir),
    "true_index_count": len(true_index),
    "sim_index_count": len(sim_index),
    "common_position_count": len(common_keys),
    "ok_residual_count": len(rows),
    "bad_count": len(bad),
    "total_J": total_J,
    "total_n_values": total_n,
    "bad_preview": bad[:20],
}, indent=2))

lines = []
lines.append("454A strict forward residual manifest")
lines.append("=====================================")
lines.append("")
lines.append(f"created = {datetime.now().isoformat()}")
lines.append(f"true_dir = {true_dir}")
lines.append(f"sim_dir = {sim_dir}")
lines.append("")
lines.append(f"true_index_count = {len(true_index)}")
lines.append(f"sim_index_count = {len(sim_index)}")
lines.append(f"common_position_count = {len(common_keys)}")
lines.append(f"ok_residual_count = {len(rows)}")
lines.append(f"bad_count = {len(bad)}")
lines.append("")
lines.append(f"total_J = {total_J:.16e}")
lines.append(f"total_n_values = {total_n}")
lines.append("")
lines.append(f"csv = {csv_path}")
lines.append(f"json = {json_path}")
lines.append("")
lines.append("Meaning:")
lines.append("  Residual is computed by matching capteurs by physical position.")
lines.append("  Simulated displacement is interpolated onto the true/observed time grid.")
lines.append("  This is the correct strict input for adjoint source construction.")
lines.append("")
lines.append("RESULT = PASS" if len(rows) > 0 and len(bad) == 0 else "RESULT = CHECK")

txt_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
