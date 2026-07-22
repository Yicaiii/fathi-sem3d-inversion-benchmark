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

def pos_key(pos):
    return tuple(round(float(x), args.round_decimals) for x in pos)

def sorted_uu_keys(h):
    keys = [k for k in h.keys() if UU_RE.match(k)]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))

def iter_trace_entries(trace_dir: Path):
    files = sorted(trace_dir.glob("capteurs.*.h5"))
    for f in files:
        with h5py.File(f, "r") as h:
            for key in sorted_uu_keys(h):
                pos_name = key + "_pos"
                if pos_name not in h:
                    continue
                pos = np.asarray(h[pos_name]).reshape(-1)
                if pos.size < 3:
                    continue
                yield {
                    "trace_file": f,
                    "receiver_key": key,
                    "position": np.asarray(pos[:3], dtype=np.float64),
                    "shape": tuple(h[key].shape),
                }

def build_position_map(trace_dir: Path):
    mp = {}
    dup = 0
    for e in iter_trace_entries(trace_dir):
        kpos = pos_key(e["position"])
        if kpos in mp:
            dup += 1
            # keep first occurrence
            continue
        mp[kpos] = e
    return mp, dup

def read_trace(entry):
    with h5py.File(entry["trace_file"], "r") as h:
        arr = np.asarray(h[entry["receiver_key"]], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise RuntimeError(f"Bad trace shape {arr.shape} for {entry}")
    t = arr[:, 0]
    u = arr[:, 1:4]
    return t, u

def interp_synth_to_true(t_true, t_syn, u_syn):
    out = np.empty((len(t_true), 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(t_true, t_syn, u_syn[:, j])
    return out

created = datetime.now().isoformat()

required = [
    candidate_trace_dir,
    true_trace_dir,
]
missing = [p for p in required if not p.exists()]

lines = []
payload = {
    "created": created,
    "transition": transition,
    "candidate": args.candidate,
    "candidate_trace_dir": str(candidate_trace_dir),
    "true_trace_dir": str(true_trace_dir),
    "missing": [str(p) for p in missing],
}

lines.append("Task 5B candidate misfit")
lines.append("========================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"candidate = {args.candidate}")
lines.append(f"candidate_trace_dir = {candidate_trace_dir}")
lines.append(f"true_trace_dir = {true_trace_dir}")
lines.append("")

if missing:
    lines.append("Missing required dirs:")
    for p in missing:
        lines.append(f"  {p}")
    result = "FAIL_MISSING_INPUTS"

else:
    true_map, true_dup = build_position_map(true_trace_dir)
    syn_map, syn_dup = build_position_map(candidate_trace_dir)

    common = sorted(set(true_map.keys()) & set(syn_map.keys()))

    rows = []
    total_J = 0.0
    total_n_values = 0
    bad = []

    for kpos in common:
        te = true_map[kpos]
        se = syn_map[kpos]

        try:
            t_true, u_true = read_trace(te)
            t_syn, u_syn = read_trace(se)

            u_syn_i = interp_synth_to_true(t_true, t_syn, u_syn)
            residual = u_syn_i - u_true

            integrand = np.sum(residual * residual, axis=1)
            J = 0.5 * float(np.trapz(integrand, t_true))

            rows.append({
                "x": kpos[0],
                "y": kpos[1],
                "z": kpos[2],
                "J": J,
                "n_true": len(t_true),
                "n_syn": len(t_syn),
                "true_file": str(te["trace_file"].relative_to(ROOT)),
                "true_key": te["receiver_key"],
                "synthetic_file": str(se["trace_file"].relative_to(ROOT)),
                "synthetic_key": se["receiver_key"],
            })

            total_J += J
            total_n_values += int(residual.size)

        except Exception as e:
            bad.append({
                "position": kpos,
                "error": repr(e),
            })

    csv_path = out_dir / f"{args.candidate}_misfit_by_receiver.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "x", "y", "z", "J", "n_true", "n_syn",
            "true_file", "true_key", "synthetic_file", "synthetic_key"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    payload.update({
        "true_count": len(true_map),
        "synthetic_count": len(syn_map),
        "true_duplicate_positions": true_dup,
        "synthetic_duplicate_positions": syn_dup,
        "common_position_count": len(common),
        "bad_count": len(bad),
        "bad_preview": bad[:20],
        "total_J": total_J,
        "total_n_values": total_n_values,
        "csv": str(csv_path),
    })

    ok = len(common) > 0 and len(bad) == 0 and np.isfinite(total_J)
    result = "PASS" if ok else "CHECK_NEEDED"

    lines.append("Counts:")
    lines.append(f"  true_count = {len(true_map)}")
    lines.append(f"  synthetic_count = {len(syn_map)}")
    lines.append(f"  common_position_count = {len(common)}")
    lines.append(f"  true_duplicate_positions = {true_dup}")
    lines.append(f"  synthetic_duplicate_positions = {syn_dup}")
    lines.append(f"  bad_count = {len(bad)}")
    lines.append("")
    lines.append("Misfit:")
    lines.append(f"  total_J = {total_J:.16e}")
    lines.append(f"  total_n_values = {total_n_values}")
    lines.append(f"  csv = {csv_path}")

payload["result"] = result

json_path = out_dir / f"{args.candidate}_misfit_summary.json"
txt_path = out_dir / f"{args.candidate}_misfit_summary.txt"

report_json = report_dir / f"{transition}_{args.candidate}_misfit.json"
report_txt = report_dir / f"{transition}_{args.candidate}_misfit.txt"

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
