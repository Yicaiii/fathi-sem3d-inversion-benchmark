from pathlib import Path
import os
from datetime import datetime
import argparse
import json
import h5py
import numpy as np
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

run_result_root = ROOT / config["run_result_root"] / transition
candidate_root = run_result_root / "candidates"

def read_field(path):
    datasets = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, tuple(obj.shape)))
        h.visititems(visit)

        matches = [name for name, shp in datasets if shp == shape]
        if not matches:
            raise RuntimeError(f"No dataset with shape {shape} in {path}. Datasets={datasets}")
        arr = np.asarray(h[matches[0]], dtype=np.float64)
        return arr, matches[0]

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

records = []
for cand_dir in sorted(candidate_root.glob("line_search_neg_mtilde_*")):
    h5_dir = cand_dir / "mat/h5"
    state_files = sorted(cand_dir.glob("*_state_candidate.npz"))

    rec = {
        "candidate": cand_dir.name,
        "candidate_dir": str(cand_dir),
        "h5_dir_exists": h5_dir.exists(),
        "state_files": [str(p) for p in state_files],
        "ok": False,
    }

    try:
        kappa, kappa_ds = read_field(h5_dir / "Mat_0_Kappa.h5")
        mu, mu_ds = read_field(h5_dir / "Mat_0_Mu.h5")
        density, density_ds = read_field(h5_dir / "Mat_0_Density.h5")
        lam = kappa - (2.0 / 3.0) * mu

        checks = {
            "state_exists": len(state_files) == 1,
            "kappa_finite": bool(np.all(np.isfinite(kappa))),
            "mu_finite": bool(np.all(np.isfinite(mu))),
            "density_finite": bool(np.all(np.isfinite(density))),
            "lambda_positive": bool(np.min(lam) > 0),
            "mu_positive": bool(np.min(mu) > 0),
            "kappa_positive": bool(np.min(kappa) > 0),
            "density_positive": bool(np.min(density) > 0),
        }

        rec.update({
            "datasets": {
                "kappa": kappa_ds,
                "mu": mu_ds,
                "density": density_ds,
            },
            "checks": checks,
            "lambda": stats(lam),
            "mu": stats(mu),
            "kappa": stats(kappa),
            "density": stats(density),
            "ok": all(checks.values()),
        })

    except Exception as e:
        rec["error"] = repr(e)

    records.append(rec)

all_ok = len(records) > 0 and all(r["ok"] for r in records)

report_dir = ROOT / "benchmark_fathi_strict/reports/candidate_generation"
report_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "candidate_root": str(candidate_root),
    "candidate_count": len(records),
    "records": records,
    "result": "PASS" if all_ok else "CHECK_NEEDED",
}

json_out = report_dir / f"{transition}_candidate_audit.json"
txt_out = report_dir / f"{transition}_candidate_audit.txt"

json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("Task 4C candidate audit")
lines.append("=======================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append(f"candidate_root = {candidate_root}")
lines.append(f"candidate_count = {len(records)}")
lines.append("")
for r in records:
    lines.append("-" * 80)
    lines.append(f"candidate = {r['candidate']}")
    lines.append(f"ok = {r['ok']}")
    if "error" in r:
        lines.append(f"error = {r['error']}")
    else:
        lines.append(f"lambda min/max = {r['lambda']['min']:.16e} {r['lambda']['max']:.16e}")
        lines.append(f"mu min/max     = {r['mu']['min']:.16e} {r['mu']['max']:.16e}")
        lines.append(f"kappa min/max  = {r['kappa']['min']:.16e} {r['kappa']['max']:.16e}")
        lines.append("checks:")
        for k2, v in r["checks"].items():
            lines.append(f"  {k2} = {v}")

lines.append("")
lines.append(f"json = {json_out}")
lines.append("")
lines.append(f"RESULT = {payload['result']}")

txt_out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if payload["result"] != "PASS":
    sys.exit(2)
