from pathlib import Path
import argparse
import json
import sys
import os
import shutil
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
selection_json = outdir / "450B_strict_forward_full_template_selection.json"

if not selection_json.exists():
    raise RuntimeError("Missing 450B selection. Run 450B first.")

selection = json.loads(selection_json.read_text())
best = selection.get("best")
if not best:
    raise RuntimeError("450B did not select a valid full forward template.")

runtime_names = {
    "traces",
    "prot",
    "res",
    "OUTPUT_FILES",
    "mirror",
    "fin_sem",
    "DEALLOCATING",
    "logs",
    "stat.log",
}

def ignore_func(dirpath, names):
    ignored = []
    for n in names:
        if n in runtime_names:
            ignored.append(n)
        if n.endswith(".o") or n.endswith(".e"):
            ignored.append(n)
    return ignored

def copy_current_material(dst):
    src_h5 = Path(ctx["input_accepted_dir"]) / "mat/h5"
    dst_h5 = dst / "mat/h5"
    dst_h5.mkdir(parents=True, exist_ok=True)

    for name in ["Mat_0_Kappa.h5", "Mat_0_Mu.h5", "Mat_0_Density.h5"]:
        src = src_h5 / name
        if not src.exists():
            raise RuntimeError(f"Missing current accepted material: {src}")
        shutil.copy2(src, dst_h5 / name)

def hardcoded_hits(d):
    hits = []
    for fname in ["input.spec", "material.spec", "material.input", "mesh.input"]:
        p = d / fname
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        for token in [
            "/home/crellamaybe/sem3d_fathi_clean/data/inversion_linear/iter_005",
            "/home/crellamaybe/sem3d_fathi_clean/data/inversion_linear/iter_006",
            "longterm_capteurs_material_grid",
        ]:
            if token in text:
                hits.append({"file": str(p), "token": token})
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(best["path"])
    dst = Path(ctx["output_forward_batches_dir"]) / "strict_full_forward_000"

    if not src.exists():
        raise RuntimeError(f"Selected source missing: {src}")

    if dst.exists():
        if not args.allow_overwrite:
            raise RuntimeError(f"Destination exists: {dst}. Use --allow-overwrite if intentional.")
        trash = dst.parent / f"_old_strict_full_forward_000_moved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(dst), str(trash))

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=ignore_func, symlinks=True)

    copy_current_material(dst)

    marker = {
        "created": datetime.now().isoformat(),
        "purpose": "generic strict full-grid forward run",
        "source_template": str(src),
        "destination": str(dst),
        "current_model_source": ctx["input_accepted_dir"],
        "runtime_outputs_excluded": sorted(runtime_names),
        "scientific_status": "STRICT_FORWARD_PREPARED: material is copied from the current accepted model.",
    }
    (dst / "STRICT_ITER007_TO_ITER008_FULL_FORWARD_MARKER.json").write_text(json.dumps(marker, indent=2))

    required = {
        "input.spec": dst / "input.spec",
        "material.spec": dst / "material.spec",
        "material.input": dst / "material.input",
        "mesh.input": dst / "mesh.input",
        "kappa": dst / "mat/h5/Mat_0_Kappa.h5",
        "mu": dst / "mat/h5/Mat_0_Mu.h5",
        "density": dst / "mat/h5/Mat_0_Density.h5",
    }

    checks = {k: p.exists() for k, p in required.items()}
    hits = hardcoded_hits(dst)

    ok = all(checks.values()) and len(hits) == 0

    result = {
        "created": datetime.now().isoformat(),
        "source": str(src),
        "destination": str(dst),
        "checks": {k: {"path": str(p), "exists": p.exists()} for k, p in required.items()},
        "hardcoded_hits": hits,
        "ok": ok,
    }

    (outdir / "450C_prepare_strict_full_forward_run.json").write_text(json.dumps(result, indent=2))

    lines = []
    lines.append("450C prepare strict full forward run")
    lines.append("====================================")
    lines.append("")
    lines.append(f"created = {result['created']}")
    lines.append(f"source = {src}")
    lines.append(f"destination = {dst}")
    lines.append("")
    lines.append("Required checks:")
    for k, p in required.items():
        lines.append(f"  {k}: exists={p.exists()} path={p}")
    lines.append("")
    lines.append("Hardcoded scan:")
    if hits:
        for h in hits:
            lines.append(f"  FOUND: {h}")
    else:
        lines.append("  none")
    lines.append("")
    lines.append("Safety:")
    lines.append("  Runtime traces/prot/res were excluded.")
    lines.append("  Material HDF5 was replaced by the current accepted material.")
    lines.append("  SEM3D was not run.")
    lines.append("")
    lines.append("RESULT = PASS" if ok else "RESULT = CHECK")

    txt = "\n".join(lines) + "\n"
    (outdir / "450C_prepare_strict_full_forward_run_summary.txt").write_text(txt)
    print(txt)

if __name__ == "__main__":
    main()
