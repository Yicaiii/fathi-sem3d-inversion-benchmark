from pathlib import Path
import json
import argparse
import sys
import os
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

work = Path(ctx["work_root"])
outdir = work / "strict_forward"
outdir.mkdir(parents=True, exist_ok=True)

search_roots = [
    ROOT / "data/inversion_linear/iter_005",
    ROOT / "data/inversion_linear/iter_006",
    ROOT / "data/inversion_linear/iter_007",
]

bad_name_tokens = [
    "adjoint",
    "source_x",
    "source_y",
    "source_z",
    "line_search",
    "trial_alpha",
    "smoke_forward_neg_g",
    "candidate",
    "accepted",
]

good_name_tokens = [
    "forward_dudx_mgcap_full",
    "forward_dudx_mgcap",
    "forward_dudx",
]

def has_required_files(d):
    required = [
        d / "input.spec",
        d / "material.spec",
        d / "mat/h5/Mat_0_Kappa.h5",
        d / "mat/h5/Mat_0_Mu.h5",
        d / "mat/h5/Mat_0_Density.h5",
    ]
    return all(p.exists() for p in required)

def hardcoded_hits(d):
    hits = []
    for fname in ["input.spec", "material.spec", "material.input", "mesh.input"]:
        p = d / fname
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        for token in [
            "iter_005",
            "iter_006",
            "iter_007",
            "longterm_capteurs_material_grid",
        ]:
            if token in text:
                hits.append({"file": str(p), "token": token})
    return hits

def trace_info(d):
    traces = sorted((d / "traces").glob("capteurs.*.h5")) if (d / "traces").exists() else []
    total = 0
    for p in traces:
        try:
            total += p.stat().st_size
        except Exception:
            pass
    return len(traces), total

def score_dir(d):
    name = d.name.lower()
    full = str(d).lower()

    if any(tok in name for tok in bad_name_tokens):
        return None

    if not any(tok in name for tok in good_name_tokens):
        return None

    if not has_required_files(d):
        return None

    score = 0
    if "forward_dudx_mgcap_full" in name:
        score += 200
    if "forward_dudx_mgcap" in name:
        score += 120
    if "forward_dudx" in name:
        score += 80
    if "full" in name:
        score += 40
    if "pilot" in name:
        score -= 60

    trace_count, trace_bytes = trace_info(d)
    score += min(trace_count, 20)

    # Existing huge traces suggest this was a real full-grid run, but not required.
    if trace_bytes > 500 * 1024**2:
        score += 30
    if trace_bytes > 2 * 1024**3:
        score += 50

    return score

records = []

for root in search_roots:
    if not root.exists():
        continue

    # Top-level dirs are usually enough here.
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        score = score_dir(d)
        if score is None:
            continue

        tc, tb = trace_info(d)

        records.append({
            "path": str(d),
            "name": d.name,
            "score": score,
            "trace_count": tc,
            "trace_total_bytes": tb,
            "trace_total_gb": tb / 1024**3,
            "hardcoded_hits": hardcoded_hits(d),
            "input_spec": str(d / "input.spec"),
            "material_spec": str(d / "material.spec"),
            "kappa": str(d / "mat/h5/Mat_0_Kappa.h5"),
            "mu": str(d / "mat/h5/Mat_0_Mu.h5"),
            "density": str(d / "mat/h5/Mat_0_Density.h5"),
        })

records = sorted(records, key=lambda r: r["score"], reverse=True)
best = records[0] if records else None

result = {
    "created": datetime.now().isoformat(),
    "context": str(CTX),
    "records": records,
    "best": best,
    "status": "PASS" if best else "CHECK",
    "rule": "Only dirs whose names look like forward_dudx/mgcap are allowed; adjoint, line_search, accepted, trial, smoke are excluded.",
}

(outdir / "450B_strict_forward_full_template_selection.json").write_text(json.dumps(result, indent=2))

lines = []
lines.append("450B select strict forward full template")
lines.append("=======================================")
lines.append("")
lines.append(f"created = {result['created']}")
lines.append(f"context = {CTX}")
lines.append(f"candidate_count = {len(records)}")
lines.append("")
lines.append("Best:")
lines.append(str(best))
lines.append("")
lines.append("Candidates:")
for i, r in enumerate(records, start=1):
    lines.append("------------------------------------------------------------")
    lines.append(f"{i}. score={r['score']}")
    lines.append(f"path = {r['path']}")
    lines.append(f"trace_count = {r['trace_count']}")
    lines.append(f"trace_total_gb = {r['trace_total_gb']:.3f}")
    lines.append(f"hardcoded_hits = {r['hardcoded_hits']}")
lines.append("")
lines.append("RESULT = PASS" if best else "RESULT = CHECK")

txt = "\n".join(lines) + "\n"
(outdir / "450B_strict_forward_full_template_selection_summary.txt").write_text(txt)
print(txt)
