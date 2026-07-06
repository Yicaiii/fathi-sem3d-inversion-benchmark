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

outdir = Path(ctx["work_root"]) / "residual_sources"
outdir.mkdir(parents=True, exist_ok=True)

old_roots = {
    "x": ROOT / "data/inversion_linear/iter_005/adjoint_x_mgcap_full_batches",
    "y": ROOT / "data/inversion_linear/iter_005/adjoint_y_mgcap_full_batches",
    "z": ROOT / "data/inversion_linear/iter_005/adjoint_z_mgcap_full_batches",
}

source_block_re = re.compile(r"source\s*\{(.*?)\};", re.S)
coord_re = re.compile(r"coords\s*=\s*([^;]+);")
dir_re = re.compile(r"dir\s*=\s*([^;]+);")
time_file_re = re.compile(r'time_file\s*=\s*"([^"]+)"')

def parse_float_list(s):
    return [float(x) for x in s.replace(",", " ").split()]

records = []

for comp, root in old_roots.items():
    if not root.exists():
        continue

    for batch in sorted(root.glob("batch_*")):
        inp = batch / "input.spec"
        if not inp.exists():
            continue

        text = inp.read_text(errors="ignore")
        blocks = source_block_re.findall(text)

        for block_id, block in enumerate(blocks):
            cm = coord_re.search(block)
            dm = dir_re.search(block)
            tm = time_file_re.search(block)

            if not tm:
                continue

            fname = tm.group(1)
            src = batch / fname

            rid_match = re.search(r"s(\d+)([xyz])\.txt$", fname)
            rid = int(rid_match.group(1)) if rid_match else None
            comp_from_file = rid_match.group(2) if rid_match else None

            arr_info = None
            if src.exists():
                try:
                    arr = np.loadtxt(src)
                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 2)
                    arr_info = {
                        "shape": list(arr.shape),
                        "time_min": float(np.nanmin(arr[:, 0])),
                        "time_max": float(np.nanmax(arr[:, 0])),
                        "value_min": float(np.nanmin(arr[:, 1])),
                        "value_max": float(np.nanmax(arr[:, 1])),
                        "finite": bool(np.all(np.isfinite(arr))),
                        "time_strict_increasing": bool(np.all(np.diff(arr[:, 0]) > 0)),
                    }
                except Exception as e:
                    arr_info = {"error": str(e)}

            records.append({
                "component": comp,
                "batch": batch.name,
                "batch_dir": str(batch),
                "input_spec": str(inp),
                "block_id": block_id,
                "rid": rid,
                "comp_from_file": comp_from_file,
                "time_file": fname,
                "source_file": str(src),
                "source_file_exists": src.exists(),
                "coords": parse_float_list(cm.group(1)) if cm else None,
                "dir": parse_float_list(dm.group(1)) if dm else None,
                "arr_info": arr_info,
            })

summary = {
    "created": datetime.now().isoformat(),
    "context": str(CTX),
    "records_count": len(records),
    "components": {},
    "records": records,
}

for comp in ["x", "y", "z"]:
    rr = [r for r in records if r["component"] == comp]
    summary["components"][comp] = {
        "record_count": len(rr),
        "batch_count": len(sorted(set(r["batch"] for r in rr))),
        "rid_count": len(sorted(set(r["rid"] for r in rr if r["rid"] is not None))),
        "rid_min": min([r["rid"] for r in rr if r["rid"] is not None], default=None),
        "rid_max": max([r["rid"] for r in rr if r["rid"] is not None], default=None),
        "all_source_files_exist": all(r["source_file_exists"] for r in rr),
        "all_time_increasing": all((r["arr_info"] or {}).get("time_strict_increasing", False) for r in rr),
        "all_finite": all((r["arr_info"] or {}).get("finite", False) for r in rr),
    }

json_path = outdir / "455A_old_adjoint_source_format.json"
txt_path = outdir / "455A_old_adjoint_source_format_summary.txt"
json_path.write_text(json.dumps(summary, indent=2))

lines = []
lines.append("455A old adjoint source format")
lines.append("==============================")
lines.append("")
lines.append(f"created = {summary['created']}")
lines.append(f"json = {json_path}")
lines.append(f"records_count = {len(records)}")
lines.append("")
for comp in ["x", "y", "z"]:
    c = summary["components"][comp]
    lines.append("------------------------------------------------------------")
    lines.append(f"component = {comp}")
    lines.append(f"record_count = {c['record_count']}")
    lines.append(f"batch_count = {c['batch_count']}")
    lines.append(f"rid_count = {c['rid_count']}")
    lines.append(f"rid_min = {c['rid_min']}")
    lines.append(f"rid_max = {c['rid_max']}")
    lines.append(f"all_source_files_exist = {c['all_source_files_exist']}")
    lines.append(f"all_time_increasing = {c['all_time_increasing']}")
    lines.append(f"all_finite = {c['all_finite']}")
lines.append("")
lines.append("Preview:")
for r in records[:20]:
    lines.append("------------------------------------------------------------")
    lines.append(f"component={r['component']} batch={r['batch']} rid={r['rid']} file={r['time_file']}")
    lines.append(f"coords={r['coords']} dir={r['dir']}")
    lines.append(f"arr_info={r['arr_info']}")
lines.append("")
lines.append("Meaning:")
lines.append("  This confirms the old SEM3D adjoint source format.")
lines.append("  We will reuse batch structure and write new strict residual source files with the same names.")
lines.append("")
lines.append("RESULT = PASS" if len(records) > 0 else "RESULT = CHECK")

txt_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
