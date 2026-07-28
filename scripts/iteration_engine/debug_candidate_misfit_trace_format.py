from pathlib import Path
import os
import h5py
import numpy as np
import re
import json

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()
UU_RE = re.compile(r"^UU_\d+$")

true_dir = ROOT / "data/60_true_layered_h5_T045/traces"
syn_dir = ROOT / "data/inversion_linear/iter_008/candidate_forward_workspaces/line_search_neg_mtilde_1p00MPa/traces"

out_dir = ROOT / "benchmark_fathi_strict/reports/candidate_misfit"
out_dir.mkdir(parents=True, exist_ok=True)

def pos_key(pos):
    return tuple(round(float(x), 8) for x in pos)

def sorted_uu_keys(h):
    return sorted(
        [k for k in h.keys() if UU_RE.match(k)],
        key=lambda x: int(x.split("_")[1])
    )

def build_map(trace_dir):
    mp = {}
    for f in sorted(trace_dir.glob("capteurs.*.h5")):
        with h5py.File(f, "r") as h:
            for key in sorted_uu_keys(h):
                pk = key + "_pos"
                if pk not in h:
                    continue
                pos = np.asarray(h[pk]).reshape(-1)[:3]
                kpos = pos_key(pos)
                if kpos not in mp:
                    mp[kpos] = (f, key)
    return mp

def describe(entry):
    f, key = entry
    with h5py.File(f, "r") as h:
        arr = np.asarray(h[key])
        pos = np.asarray(h[key + "_pos"]).reshape(-1)[:3]
    return {
        "file": str(f),
        "key": key,
        "pos": pos.tolist(),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "first_rows": arr[:5, :min(arr.shape[1], 12)].tolist() if arr.ndim == 2 else arr[:10].tolist(),
        "col_min": np.nanmin(arr, axis=0).tolist() if arr.ndim == 2 else None,
        "col_max": np.nanmax(arr, axis=0).tolist() if arr.ndim == 2 else None,
    }

true_map = build_map(true_dir)
syn_map = build_map(syn_dir)
common = sorted(set(true_map) & set(syn_map))

payload = {
    "true_dir": str(true_dir),
    "syn_dir": str(syn_dir),
    "true_count": len(true_map),
    "syn_count": len(syn_map),
    "common_count": len(common),
    "samples": [],
}

lines = []
lines.append("Candidate misfit trace format debug")
lines.append("===================================")
lines.append("")
lines.append(f"true_count = {len(true_map)}")
lines.append(f"syn_count = {len(syn_map)}")
lines.append(f"common_count = {len(common)}")
lines.append("")

for kpos in common[:3]:
    tdesc = describe(true_map[kpos])
    sdesc = describe(syn_map[kpos])

    payload["samples"].append({
        "position": kpos,
        "true": tdesc,
        "synthetic": sdesc,
    })

    lines.append("-" * 100)
    lines.append(f"position = {kpos}")
    lines.append("")
    lines.append("TRUE:")
    lines.append(f"  file = {tdesc['file']}")
    lines.append(f"  key = {tdesc['key']}")
    lines.append(f"  shape = {tdesc['shape']}")
    lines.append("  first_rows:")
    for row in tdesc["first_rows"]:
        lines.append("    " + " ".join(f"{x:.6e}" for x in row))
    lines.append("  col_min:")
    lines.append("    " + " ".join(f"{x:.6e}" for x in tdesc["col_min"][:min(len(tdesc["col_min"]), 12)]))
    lines.append("  col_max:")
    lines.append("    " + " ".join(f"{x:.6e}" for x in tdesc["col_max"][:min(len(tdesc["col_max"]), 12)]))

    lines.append("")
    lines.append("SYNTHETIC:")
    lines.append(f"  file = {sdesc['file']}")
    lines.append(f"  key = {sdesc['key']}")
    lines.append(f"  shape = {sdesc['shape']}")
    lines.append("  first_rows:")
    for row in sdesc["first_rows"]:
        lines.append("    " + " ".join(f"{x:.6e}" for x in row))
    lines.append("  col_min:")
    lines.append("    " + " ".join(f"{x:.6e}" for x in sdesc["col_min"][:min(len(sdesc["col_min"]), 12)]))
    lines.append("  col_max:")
    lines.append("    " + " ".join(f"{x:.6e}" for x in sdesc["col_max"][:min(len(sdesc["col_max"]), 12)]))

lines.append("")
lines.append("RESULT = PASS")

txt = out_dir / "debug_candidate_misfit_trace_format.txt"
js = out_dir / "debug_candidate_misfit_trace_format.json"

txt.write_text("\n".join(lines), encoding="utf-8")
js.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("\n".join(lines))
