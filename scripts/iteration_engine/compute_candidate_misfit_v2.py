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

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--candidate", default="line_search_neg_mtilde_1p00MPa")
parser.add_argument("--round-decimals", type=int, default=8)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text())

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_result_root = ROOT / config["run_result_root"] / transition
run_data_root = ROOT / config["run_data_root"] / f"iter_{kp1:03d}"

candidate_workspace = run_data_root / "candidate_forward_workspaces" / args.candidate
candidate_trace_dir = candidate_workspace / "traces"
true_trace_dir = ROOT / config["true_observed_traces_dir"]

out_dir = run_result_root / "candidate_misfits"
out_dir.mkdir(parents=True, exist_ok=True)

report_dir = ROOT / "benchmark_fathi_strict/reports/candidate_misfit"
report_dir.mkdir(parents=True, exist_ok=True)

def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def pos_key(pos):
    return tuple(round(float(x), args.round_decimals) for x in pos)

def sorted_uu_keys(h):
    keys = [k for k in h.keys() if UU_RE.match(k)]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))

def build_position_map(trace_dir: Path):
    mp = {}
    dup = 0
    files = sorted(trace_dir.glob("capteurs.*.h5"))

    for f in files:
        with h5py.File(f, "r") as h:
            for key in sorted_uu_keys(h):
                pos_name = key + "_pos"
                if pos_name not in h:
                    continue

                pos = np.asarray(h[pos_name], dtype=np.float64).reshape(-1)
                if pos.size < 3:
                    continue

                kpos = pos_key(pos[:3])
                if kpos in mp:
                    dup += 1
                    continue

                shape = tuple(h[key].shape)
                mp[kpos] = {
                    "trace_file": f,
                    "receiver_key": key,
                    "position": np.asarray(pos[:3], dtype=np.float64),
                    "shape": shape,
                }

    return mp, dup, len(files)

def read_time_and_disp(entry):
    f = entry["trace_file"]
    key = entry["receiver_key"]

    with h5py.File(f, "r") as h:
        arr = np.asarray(h[key], dtype=np.float64)

    if arr.ndim != 2:
        raise RuntimeError(f"Trace is not 2D: file={f}, key={key}, shape={arr.shape}")

    if arr.shape[1] < 4:
        raise RuntimeError(f"Trace has fewer than 4 columns: file={f}, key={key}, shape={arr.shape}")

    t = arr[:, 0].astype(np.float64)
    u = arr[:, 1:4].astype(np.float64)

    if t.ndim != 1:
        raise RuntimeError(f"Bad time ndim: {t.ndim}")

    if u.ndim != 2 or u.shape[1] != 3:
        raise RuntimeError(f"Bad displacement shape: {u.shape}")

    if len(t) != u.shape[0]:
        raise RuntimeError(f"Length mismatch: len(t)={len(t)}, u.shape={u.shape}")

    if not np.all(np.isfinite(t)):
        raise RuntimeError("Non-finite time values")

    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite displacement values")

    dt = np.diff(t)
    if len(dt) == 0:
        raise RuntimeError("Empty time vector")

    if np.any(dt <= 0):
        bad_dt = dt[dt <= 0][:10]
        raise RuntimeError(f"Time is not strictly increasing. bad_dt_preview={bad_dt}")

    return t, u

def interp_synthetic_to_true(t_true, t_syn, u_syn):
    out = np.empty((len(t_true), 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(t_true, t_syn, u_syn[:, j])
    return out

def read_parent_J():
    candidates = [
        run_result_root / "residual_sources/454B_strict_residual_timeseries_summary.json",
        run_result_root / "residual_sources/454A_strict_forward_residual_manifest_summary.json",
        run_result_root / "residual_sources/454B_strict_residual_timeseries_summary.txt",
        run_result_root / "residual_sources/454A_strict_forward_residual_manifest_summary.txt",
    ]

    for p in candidates:
        if not p.exists():
            continue

        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text())
                for key in ["global_J", "total_J", "J"]:
                    if key in data:
                        return float(data[key]), str(p), key
            except Exception:
                pass

        txt = p.read_text(errors="ignore")
        for pat in [
            r"global_J\s*=\s*([0-9eE+\-.]+)",
            r"total_J\s*=\s*([0-9eE+\-.]+)",
            r"\bJ\s*=\s*([0-9eE+\-.]+)",
        ]:
            m = re.search(pat, txt)
            if m:
                return float(m.group(1)), str(p), pat

    return None, None, None

created = datetime.now().isoformat()

required = [
    candidate_trace_dir,
    true_trace_dir,
]
missing = [p for p in required if not p.exists()]

payload = {
    "created": created,
    "transition": transition,
    "candidate": args.candidate,
    "candidate_trace_dir": str(candidate_trace_dir),
    "true_trace_dir": str(true_trace_dir),
    "missing": [str(p) for p in missing],
}

lines = []
lines.append("Task 5B candidate misfit v2")
lines.append("===========================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"candidate_trace_dir = {candidate_trace_dir}")
lines.append(f"true_trace_dir = {true_trace_dir}")
lines.append("")

if missing:
    result = "FAIL_MISSING_INPUTS"
    lines.append("Missing required dirs:")
    for p in missing:
        lines.append(f"  {p}")

else:
    true_map, true_dup, true_file_count = build_position_map(true_trace_dir)
    syn_map, syn_dup, syn_file_count = build_position_map(candidate_trace_dir)

    common = sorted(set(true_map.keys()) & set(syn_map.keys()))

    rows = []
    bad = []
    total_J = 0.0
    total_n_values = 0

    for kpos in common:
        te = true_map[kpos]
        se = syn_map[kpos]

        try:
            t_true, u_true = read_time_and_disp(te)
            t_syn, u_syn = read_time_and_disp(se)

            u_syn_interp = interp_synthetic_to_true(t_true, t_syn, u_syn)
            residual = u_syn_interp - u_true

            integrand = np.sum(residual * residual, axis=1)
            J_receiver = 0.5 * float(integrate_trapezoid(integrand, t_true))

            if not np.isfinite(J_receiver):
                raise RuntimeError(f"Non-finite J_receiver: {J_receiver}")

            rows.append({
                "x": kpos[0],
                "y": kpos[1],
                "z": kpos[2],
                "J": J_receiver,
                "n_true": len(t_true),
                "n_syn": len(t_syn),
                "t_true_0": float(t_true[0]),
                "t_true_1": float(t_true[-1]),
                "t_syn_0": float(t_syn[0]),
                "t_syn_1": float(t_syn[-1]),
                "true_file": str(te["trace_file"].relative_to(ROOT)),
                "true_key": te["receiver_key"],
                "synthetic_file": str(se["trace_file"].relative_to(ROOT)),
                "synthetic_key": se["receiver_key"],
            })

            total_J += J_receiver
            total_n_values += int(residual.size)

        except Exception as e:
            bad.append({
                "position": kpos,
                "true_file": str(te["trace_file"]),
                "true_key": te["receiver_key"],
                "true_shape": te["shape"],
                "synthetic_file": str(se["trace_file"]),
                "synthetic_key": se["receiver_key"],
                "synthetic_shape": se["shape"],
                "error": repr(e),
            })

    csv_path = out_dir / f"{args.candidate}_misfit_by_receiver_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "x", "y", "z", "J",
            "n_true", "n_syn",
            "t_true_0", "t_true_1",
            "t_syn_0", "t_syn_1",
            "true_file", "true_key",
            "synthetic_file", "synthetic_key",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    parent_J, parent_J_source, parent_J_key = read_parent_J()
    descent = None
    delta_J = None
    if parent_J is not None and np.isfinite(total_J):
        delta_J = total_J - parent_J
        descent = total_J < parent_J

    ok = (
        len(common) == 225
        and len(rows) == 225
        and len(bad) == 0
        and np.isfinite(total_J)
        and total_n_values > 0
    )

    result = "PASS" if ok else "CHECK_NEEDED"

    payload.update({
        "true_file_count": true_file_count,
        "synthetic_file_count": syn_file_count,
        "true_count": len(true_map),
        "synthetic_count": len(syn_map),
        "common_position_count": len(common),
        "ok_receiver_count": len(rows),
        "bad_count": len(bad),
        "bad_preview": bad[:20],
        "true_duplicate_positions": true_dup,
        "synthetic_duplicate_positions": syn_dup,
        "total_J": total_J,
        "total_n_values": total_n_values,
        "csv": str(csv_path),
        "parent_J": parent_J,
        "parent_J_source": parent_J_source,
        "parent_J_key": parent_J_key,
        "delta_J": delta_J,
        "descent": descent,
    })

    lines.append("Counts:")
    lines.append(f"  true_file_count = {true_file_count}")
    lines.append(f"  synthetic_file_count = {syn_file_count}")
    lines.append(f"  true_count = {len(true_map)}")
    lines.append(f"  synthetic_count = {len(syn_map)}")
    lines.append(f"  common_position_count = {len(common)}")
    lines.append(f"  ok_receiver_count = {len(rows)}")
    lines.append(f"  bad_count = {len(bad)}")
    lines.append(f"  true_duplicate_positions = {true_dup}")
    lines.append(f"  synthetic_duplicate_positions = {syn_dup}")
    lines.append("")
    lines.append("Misfit:")
    lines.append(f"  total_J = {total_J:.16e}")
    lines.append(f"  total_n_values = {total_n_values}")
    lines.append(f"  csv = {csv_path}")
    lines.append("")
    lines.append("Parent comparison:")
    lines.append(f"  parent_J = {parent_J}")
    lines.append(f"  parent_J_source = {parent_J_source}")
    lines.append(f"  delta_J = {delta_J}")
    lines.append(f"  descent = {descent}")

    if bad:
        lines.append("")
        lines.append("Bad preview:")
        for b in bad[:5]:
            lines.append("  " + repr(b))

payload["result"] = result

json_path = out_dir / f"{args.candidate}_misfit_summary_v2.json"
txt_path = out_dir / f"{args.candidate}_misfit_summary_v2.txt"
report_json = report_dir / f"{transition}_{args.candidate}_misfit_v2.json"
report_txt = report_dir / f"{transition}_{args.candidate}_misfit_v2.txt"

lines.append("")
lines.append(f"json = {json_path}")
lines.append("")
lines.append(f"RESULT = {result}")

json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
txt_path.write_text("\n".join(lines), encoding="utf-8")
report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
report_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
