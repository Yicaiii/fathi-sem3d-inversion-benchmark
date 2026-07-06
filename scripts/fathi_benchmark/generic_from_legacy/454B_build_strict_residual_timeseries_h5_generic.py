from pathlib import Path
import json
import argparse
import os
import csv
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

manifest_csv = Path(ctx["work_root"]) / "residual_sources/454A_strict_forward_residual_manifest.csv"
outdir = Path(ctx["work_root"]) / "residual_sources"
outdir.mkdir(parents=True, exist_ok=True)

out_h5 = outdir / "454B_strict_residual_timeseries.h5"
out_json = outdir / "454B_strict_residual_timeseries.json"
out_txt = outdir / "454B_strict_residual_timeseries_summary.txt"

if not manifest_csv.exists():
    raise RuntimeError(f"Missing manifest CSV: {manifest_csv}")

def load_uu(file_path, dataset):
    with h5py.File(file_path, "r") as f:
        arr = np.asarray(f[dataset][()], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise RuntimeError(f"Bad UU shape: {file_path} {dataset} {arr.shape}")
    t = arr[:, 0]
    u = arr[:, 1:4]
    return t, u

rows = []
with manifest_csv.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    raise RuntimeError("Residual manifest is empty.")

station_records = []
global_J = 0.0
global_n_values = 0
lengths = []

with h5py.File(out_h5, "w") as h5:
    h5.attrs["created"] = datetime.now().isoformat()
    h5.attrs["context"] = str(CTX)
    h5.attrs["source_manifest"] = str(manifest_csv)
    h5.attrs["definition"] = "residual = simulated_displacement_interpolated_on_true_time_grid - true_observed_displacement"
    h5.attrs["time_reversal_note"] = "residual_time_reversed is stored for SEM3D forward-in-time adjoint source construction, but final source sign must be checked against the existing adjoint builder."

    for i, r in enumerate(rows):
        pos = np.array([
            float(r["key_x"]),
            float(r["key_y"]),
            float(r["key_z"]),
        ], dtype=np.float64)

        tt, ut = load_uu(r["true_file"], r["true_dataset"])
        ts, us = load_uu(r["sim_file"], r["sim_dataset"])

        t0 = max(float(tt[0]), float(ts[0]))
        t1 = min(float(tt[-1]), float(ts[-1]))

        mask = (tt >= t0) & (tt <= t1)
        t_eval = tt[mask]
        u_true = ut[mask, :]

        if len(t_eval) < 2:
            raise RuntimeError(f"Insufficient overlap for station {i}")

        u_sim = np.empty_like(u_true)
        for c in range(3):
            u_sim[:, c] = np.interp(t_eval, ts, us[:, c])

        residual = u_sim - u_true
        residual_rev = residual[::-1, :].copy()

        integrand = np.sum(residual * residual, axis=1)
        local_J = 0.5 * float(np.trapezoid(integrand, t_eval))

        g = h5.create_group(f"station_{i:04d}")
        g.attrs["station_index"] = i
        g.attrs["true_file"] = r["true_file"]
        g.attrs["true_dataset"] = r["true_dataset"]
        g.attrs["sim_file"] = r["sim_file"]
        g.attrs["sim_dataset"] = r["sim_dataset"]
        g.attrs["local_J"] = local_J
        g.attrs["n_time"] = len(t_eval)

        g.create_dataset("position", data=pos)
        g.create_dataset("time_true_grid", data=t_eval)
        g.create_dataset("residual_forward_time_xyz", data=residual)
        g.create_dataset("residual_time_reversed_xyz", data=residual_rev)

        # 这两个先都存下来，后面根据旧 adjoint builder 决定用 plus 还是 minus。
        g.create_dataset("source_plus_time_reversed_xyz", data=residual_rev)
        g.create_dataset("source_minus_time_reversed_xyz", data=-residual_rev)

        station_records.append({
            "station_index": i,
            "position": [float(x) for x in pos],
            "true_file": r["true_file"],
            "true_dataset": r["true_dataset"],
            "sim_file": r["sim_file"],
            "sim_dataset": r["sim_dataset"],
            "n_time": int(len(t_eval)),
            "local_J": local_J,
            "max_abs_residual": float(np.max(np.abs(residual))),
            "rms_residual": float(np.sqrt(np.mean(residual * residual))),
        })

        global_J += local_J
        global_n_values += int(residual.size)
        lengths.append(int(len(t_eval)))

    h5.create_dataset("station_positions", data=np.array([x["position"] for x in station_records], dtype=np.float64))
    h5.create_dataset("station_n_time", data=np.array(lengths, dtype=np.int64))

summary = {
    "created": datetime.now().isoformat(),
    "context": str(CTX),
    "manifest_csv": str(manifest_csv),
    "out_h5": str(out_h5),
    "station_count": len(station_records),
    "global_J": global_J,
    "global_n_values": global_n_values,
    "n_time_min": int(min(lengths)),
    "n_time_max": int(max(lengths)),
    "all_same_length": bool(min(lengths) == max(lengths)),
    "records_preview": station_records[:5],
}

out_json.write_text(json.dumps(summary, indent=2))

lines = []
lines.append("454B strict residual timeseries H5")
lines.append("==================================")
lines.append("")
lines.append(f"created = {summary['created']}")
lines.append(f"manifest_csv = {manifest_csv}")
lines.append(f"out_h5 = {out_h5}")
lines.append("")
lines.append(f"station_count = {summary['station_count']}")
lines.append(f"global_J = {summary['global_J']:.16e}")
lines.append(f"global_n_values = {summary['global_n_values']}")
lines.append(f"n_time_min = {summary['n_time_min']}")
lines.append(f"n_time_max = {summary['n_time_max']}")
lines.append(f"all_same_length = {summary['all_same_length']}")
lines.append("")
lines.append("Stored per station:")
lines.append("  position")
lines.append("  time_true_grid")
lines.append("  residual_forward_time_xyz")
lines.append("  residual_time_reversed_xyz")
lines.append("  source_plus_time_reversed_xyz")
lines.append("  source_minus_time_reversed_xyz")
lines.append("")
lines.append("Meaning:")
lines.append("  This file is the strict residual source database for x/y/z adjoint construction.")
lines.append("  We still need to inspect the old adjoint source builder before choosing source sign and file format.")
lines.append("")
lines.append("RESULT = PASS")

out_txt.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
