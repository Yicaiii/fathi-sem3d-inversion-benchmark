from pathlib import Path
import json
import sys
import os
import argparse
import re
import numpy as np
from datetime import datetime

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

_context_parser = argparse.ArgumentParser(add_help=False)
_context_parser.add_argument("--context", required=True)
_context_args, _remaining_argv = _context_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

CTX = Path(_context_args.context)
if not CTX.is_absolute():
    CTX = ROOT / CTX
ctx = json.loads(CTX.read_text())

iter_root = Path(ctx.get("output_iter_root", ROOT / "data/inversion_linear/iter_008"))
adj_base = Path(ctx.get("output_adjoint_batches_dir", iter_root / "adjoint_full_grid_batches"))

outdir = Path(ctx["work_root"]) / "residual_sources"
outdir.mkdir(parents=True, exist_ok=True)

source_block_re = re.compile(r"source\s*\{(.*?)\};", re.S)
time_file_re = re.compile(r'time_file\s*=\s*"([^"]+)"')
dir_re = re.compile(r"dir\s*=\s*([^;]+);")

def parse_source_files_from_input(input_spec):
    text = input_spec.read_text(errors="ignore")
    blocks = source_block_re.findall(text)
    rows = []
    for i, b in enumerate(blocks):
        tm = time_file_re.search(b)
        dm = dir_re.search(b)
        if tm:
            rows.append({
                "block_id": i,
                "time_file": tm.group(1),
                "dir": dm.group(1).strip() if dm else None,
            })
    return rows

def inspect_source_file(p):
    arr = np.loadtxt(p)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return {
        "path": str(p),
        "shape": list(arr.shape),
        "finite": bool(np.all(np.isfinite(arr))),
        "time_increasing": bool(np.all(np.diff(arr[:, 0]) > 0)),
        "time_min": float(np.min(arr[:, 0])),
        "time_max": float(np.max(arr[:, 0])),
        "value_min": float(np.min(arr[:, 1])),
        "value_max": float(np.max(arr[:, 1])),
        "value_l2": float(np.linalg.norm(arr[:, 1])),
    }

records = []
overall_ok = True

for comp in ["x", "y", "z"]:
    comp_dir = adj_base / comp
    for batch in sorted(comp_dir.glob("batch_*")):
        input_spec = batch / "input.spec"
        material_spec = batch / "material.spec"
        kappa = batch / "mat/h5/Mat_0_Kappa.h5"
        mu = batch / "mat/h5/Mat_0_Mu.h5"
        density = batch / "mat/h5/Mat_0_Density.h5"

        source_rows = parse_source_files_from_input(input_spec) if input_spec.exists() else []
        missing = []
        bad = []
        inspected = []

        for r in source_rows:
            sf = batch / r["time_file"]
            if not sf.exists():
                missing.append(str(sf))
                continue
            try:
                info = inspect_source_file(sf)
                inspected.append(info)
                if not info["finite"] or not info["time_increasing"]:
                    bad.append(info)
            except Exception as e:
                bad.append({"path": str(sf), "error": str(e)})

        required_ok = all(p.exists() for p in [input_spec, material_spec, kappa, mu, density])
        batch_ok = required_ok and len(source_rows) > 0 and len(missing) == 0 and len(bad) == 0
        if not batch_ok:
            overall_ok = False

        records.append({
            "component": comp,
            "batch": batch.name,
            "batch_dir": str(batch),
            "required_ok": required_ok,
            "input_spec_exists": input_spec.exists(),
            "material_spec_exists": material_spec.exists(),
            "kappa_exists": kappa.exists(),
            "mu_exists": mu.exists(),
            "density_exists": density.exists(),
            "source_block_count": len(source_rows),
            "missing_source_count": len(missing),
            "bad_source_count": len(bad),
            "source_preview": inspected[:5],
            "missing_preview": missing[:10],
            "bad_preview": bad[:10],
            "ok": batch_ok,
        })

summary = {
    "created": datetime.now().isoformat(),
    "context": str(CTX),
    "adj_base": str(adj_base),
    "batch_count": len(records),
    "overall_ok": overall_ok,
    "records": records,
}

json_path = outdir / "455C_audit_strict_adjoint_batches.json"
txt_path = outdir / "455C_audit_strict_adjoint_batches_summary.txt"
json_path.write_text(json.dumps(summary, indent=2))

lines = []
lines.append("455C audit strict adjoint batches")
lines.append("=================================")
lines.append("")
lines.append(f"created = {summary['created']}")
lines.append(f"adj_base = {adj_base}")
lines.append(f"batch_count = {len(records)}")
lines.append(f"overall_ok = {overall_ok}")
lines.append("")
for r in records:
    lines.append("------------------------------------------------------------")
    lines.append(f"component = {r['component']}")
    lines.append(f"batch = {r['batch']}")
    lines.append(f"batch_dir = {r['batch_dir']}")
    lines.append(f"required_ok = {r['required_ok']}")
    lines.append(f"source_block_count = {r['source_block_count']}")
    lines.append(f"missing_source_count = {r['missing_source_count']}")
    lines.append(f"bad_source_count = {r['bad_source_count']}")
    lines.append(f"ok = {r['ok']}")
    lines.append(f"source_preview = {r['source_preview'][:2]}")
lines.append("")
lines.append("RESULT = PASS" if overall_ok else "RESULT = CHECK")

txt_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
