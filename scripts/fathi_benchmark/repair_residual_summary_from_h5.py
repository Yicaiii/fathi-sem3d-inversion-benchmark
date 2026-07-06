from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import h5py
import numpy as np
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
args = parser.parse_args()

config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))

k = args.iter_k
kp1 = k + 1
transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_result_root = ROOT / config["run_result_root"] / transition
out_dir = run_result_root / "residual_sources"

h5_path = out_dir / "454B_strict_residual_timeseries.h5"
summary_json = out_dir / "454B_strict_residual_timeseries_summary.json"
summary_txt = out_dir / "454B_strict_residual_timeseries_summary.txt"

report_dir = ROOT / "benchmark_fathi_strict/reports/executable_tasks"
report_dir.mkdir(parents=True, exist_ok=True)

created = datetime.now().isoformat()

if not h5_path.exists():
    raise SystemExit(f"Missing residual H5: {h5_path}")

with h5py.File(h5_path, "r") as h:
    attrs = dict(h.attrs)
    receiver_count = len(h["receivers"].keys()) if "receivers" in h else None

station_count = int(attrs.get("station_count", receiver_count if receiver_count is not None else -1))
bad_count = int(attrs.get("bad_count", 0))
global_J = float(attrs.get("global_J", np.nan))
total_n_values = int(attrs.get("total_n_values", -1))

ok = (
    receiver_count == 225
    and station_count == 225
    and bad_count == 0
    and np.isfinite(global_J)
    and global_J > 0
)

payload = {
    "created": created,
    "transition": transition,
    "source": "repair_residual_summary_from_h5",
    "h5": str(h5_path),
    "receiver_count": receiver_count,
    "station_count": station_count,
    "bad_count": bad_count,
    "global_J": global_J,
    "total_n_values": total_n_values,
    "result": "PASS" if ok else "CHECK_NEEDED",
}

summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Strict residual generation summary")
lines.append("==================================")
lines.append("")
lines.append(f"created = {created}")
lines.append(f"transition = {transition}")
lines.append(f"h5 = {h5_path}")
lines.append(f"receiver_count = {receiver_count}")
lines.append(f"station_count = {station_count}")
lines.append(f"global_J = {global_J:.16e}")
lines.append(f"total_n_values = {total_n_values}")
lines.append(f"bad_count = {bad_count}")
lines.append("")
lines.append(f"json = {summary_json}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

summary_txt.write_text("\n".join(lines), encoding="utf-8")

report_json = report_dir / f"{transition}_residual_summary_repair.json"
report_txt = report_dir / f"{transition}_residual_summary_repair.txt"
report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
report_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if payload["result"] != "PASS":
    sys.exit(2)
