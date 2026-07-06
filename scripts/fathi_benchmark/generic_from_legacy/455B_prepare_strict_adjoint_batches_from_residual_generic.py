from pathlib import Path
import argparse
import json
import sys
import os
import shutil
import re
import numpy as np
import h5py
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

residual_h5 = Path(ctx["work_root"]) / "residual_sources/454B_strict_residual_timeseries.h5"
format_json = Path(ctx["work_root"]) / "residual_sources/455A_old_adjoint_source_format.json"

if not residual_h5.exists():
    raise RuntimeError(f"Missing residual H5. Run 454B first: {residual_h5}")
if not format_json.exists():
    raise RuntimeError(f"Missing 455A format json. Run 455A first: {format_json}")

old_roots = {
    "x": ROOT / "data/inversion_linear/iter_005/adjoint_x_mgcap_full_batches",
    "y": ROOT / "data/inversion_linear/iter_005/adjoint_y_mgcap_full_batches",
    "z": ROOT / "data/inversion_linear/iter_005/adjoint_z_mgcap_full_batches",
}

iter_root = Path(ctx.get("output_iter_root", ROOT / "data/inversion_linear/iter_008"))
out_base = Path(ctx.get("output_adjoint_batches_dir", iter_root / "adjoint_full_grid_batches"))

outdir = Path(ctx["work_root"]) / "residual_sources"
outdir.mkdir(parents=True, exist_ok=True)

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
            raise RuntimeError(f"Missing iter007 accepted material: {src}")
        shutil.copy2(src, dst_h5 / name)

def decode_attr(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)

def build_residual_by_rid():
    out = {}
    skipped = []

    with h5py.File(residual_h5, "r") as f:
        for name in sorted(f.keys()):
            obj = f[name]

            # Only real station groups are valid.
            # Skip root-level datasets like station_positions and station_n_time.
            if not isinstance(obj, h5py.Group):
                continue
            if re.fullmatch(r"station_\d{4}", name) is None:
                continue

            g = obj

            if "true_dataset" not in g.attrs:
                skipped.append({"name": name, "reason": "missing true_dataset attr"})
                continue

            true_dataset = decode_attr(g.attrs["true_dataset"])
            m = re.search(r"UU_(\d+)$", true_dataset)
            if not m:
                skipped.append({"name": name, "reason": f"cannot parse rid from true_dataset={true_dataset}"})
                continue

            rid = int(m.group(1))

            t = np.asarray(g["time_true_grid"][()], dtype=np.float64)
            t_adj = t - t[0]

            if "source_plus_time_reversed_xyz" in g:
                src = np.asarray(g["source_plus_time_reversed_xyz"][()], dtype=np.float64)
            else:
                src = np.asarray(g["residual_time_reversed_xyz"][()], dtype=np.float64)

            if src.ndim != 2 or src.shape[1] != 3:
                raise RuntimeError(f"Bad residual source shape for {name}: {src.shape}")
            if len(t_adj) != src.shape[0]:
                raise RuntimeError(f"Time/source length mismatch for {name}: time={len(t_adj)} source={src.shape}")

            out[rid] = {
                "time": t_adj,
                "xyz": src,
                "position": np.asarray(g["position"][()], dtype=np.float64),
                "station_group": name,
                "true_dataset": true_dataset,
            }

    return out, skipped

def write_sources_for_batch(batch_dir, comp, residual_by_rid):
    comp_index = {"x": 0, "y": 1, "z": 2}[comp]
    source_files = sorted(batch_dir.glob(f"s*{comp}.txt"))

    written = []
    missing_rids = []

    for sf in source_files:
        m = re.search(r"s(\d+)([xyz])\.txt$", sf.name)
        if not m:
            continue

        rid = int(m.group(1))
        cc = m.group(2)

        if cc != comp:
            continue

        if rid not in residual_by_rid:
            missing_rids.append(rid)
            continue

        rec = residual_by_rid[rid]
        arr = np.column_stack([rec["time"], rec["xyz"][:, comp_index]])

        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Nonfinite source array for {sf}")

        if not np.all(np.diff(arr[:, 0]) > 0):
            raise RuntimeError(f"Time column is not strictly increasing for {sf}")

        np.savetxt(sf, arr, fmt="%.16e")

        written.append({
            "file": str(sf),
            "rid": rid,
            "component": comp,
            "shape": list(arr.shape),
            "time_min": float(arr[:, 0].min()),
            "time_max": float(arr[:, 0].max()),
            "value_min": float(arr[:, 1].min()),
            "value_max": float(arr[:, 1].max()),
        })

    return written, missing_rids

def hardcoded_scan(dst):
    hits = []
    for fname in ["input.spec", "material.spec", "material.input", "mesh.input"]:
        p = dst / fname
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

    residual_by_rid, skipped = build_residual_by_rid()

    if not residual_by_rid:
        raise RuntimeError("No residual stations were loaded from residual H5.")

    result = {
        "created": datetime.now().isoformat(),
        "context": str(CTX),
        "residual_h5": str(residual_h5),
        "out_base": str(out_base),
        "residual_rid_count": len(residual_by_rid),
        "skipped_station_groups": skipped,
        "components": {},
        "ok": True,
    }

    for comp, old_root in old_roots.items():
        if not old_root.exists():
            raise RuntimeError(f"Missing old adjoint template root for {comp}: {old_root}")

        comp_out = out_base / comp

        if comp_out.exists():
            if not args.allow_overwrite:
                raise RuntimeError(f"Destination exists: {comp_out}. Use --allow-overwrite if intentional.")
            trash = comp_out.parent / f"_old_{comp}_moved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(comp_out), str(trash))

        comp_out.mkdir(parents=True, exist_ok=True)

        comp_records = []
        total_written = 0
        all_missing = []
        all_hardcoded = []

        for old_batch in sorted(old_root.glob("batch_*")):
            dst = comp_out / old_batch.name
            shutil.copytree(old_batch, dst, ignore=ignore_func, symlinks=True)

            copy_current_material(dst)

            written, missing_rids = write_sources_for_batch(dst, comp, residual_by_rid)
            hardcoded = hardcoded_scan(dst)

            marker = {
                "created": datetime.now().isoformat(),
                "purpose": f"strict iter007->iter008 adjoint {comp} batch",
                "source_template": str(old_batch),
                "destination": str(dst),
                "current_model_source": ctx["input_accepted_dir"],
                "residual_source": str(residual_h5),
                "component": comp,
                "written_source_count": len(written),
                "missing_rids": missing_rids,
                "source_sign": "plus: residual = simulated - true, source = residual reversed in time",
            }

            (dst / f"STRICT_ITER007_TO_ITER008_ADJOINT_{comp.upper()}_MARKER.json").write_text(json.dumps(marker, indent=2))

            total_written += len(written)
            all_missing.extend(missing_rids)
            all_hardcoded.extend(hardcoded)

            comp_records.append({
                "batch": old_batch.name,
                "old_batch": str(old_batch),
                "new_batch": str(dst),
                "written_source_count": len(written),
                "missing_rids": missing_rids,
                "hardcoded_hits": hardcoded,
                "written_preview": written[:3],
            })

        comp_ok = (
            total_written > 0
            and len(all_missing) == 0
            and len(all_hardcoded) == 0
        )

        result["components"][comp] = {
            "old_root": str(old_root),
            "new_root": str(comp_out),
            "batch_count": len(comp_records),
            "total_written_source_count": total_written,
            "missing_rids": sorted(set(all_missing)),
            "hardcoded_hits": all_hardcoded,
            "ok": comp_ok,
            "records": comp_records,
        }

        if not comp_ok:
            result["ok"] = False

    if skipped:
        result["ok"] = False

    json_path = outdir / "455B_prepare_strict_adjoint_batches_from_residual.json"
    txt_path = outdir / "455B_prepare_strict_adjoint_batches_from_residual_summary.txt"
    json_path.write_text(json.dumps(result, indent=2))

    lines = []
    lines.append("455B prepare strict adjoint batches from residual")
    lines.append("=================================================")
    lines.append("")
    lines.append(f"created = {result['created']}")
    lines.append(f"residual_h5 = {residual_h5}")
    lines.append(f"out_base = {out_base}")
    lines.append(f"residual_rid_count = {len(residual_by_rid)}")
    lines.append(f"skipped_station_groups = {skipped}")
    lines.append("")

    for comp in ["x", "y", "z"]:
        c = result["components"][comp]
        lines.append("------------------------------------------------------------")
        lines.append(f"component = {comp}")
        lines.append(f"old_root = {c['old_root']}")
        lines.append(f"new_root = {c['new_root']}")
        lines.append(f"batch_count = {c['batch_count']}")
        lines.append(f"total_written_source_count = {c['total_written_source_count']}")
        lines.append(f"missing_rids = {c['missing_rids']}")
        lines.append(f"hardcoded_hits_count = {len(c['hardcoded_hits'])}")
        lines.append(f"ok = {c['ok']}")
        lines.append("batch preview:")
        for r in c["records"][:3]:
            lines.append(f"  {r['batch']}: written={r['written_source_count']} missing={r['missing_rids']} hardcoded={r['hardcoded_hits']}")
            lines.append(f"    preview={r['written_preview']}")

    lines.append("")
    lines.append("Safety:")
    lines.append("  SEM3D was not run.")
    lines.append("  Old runtime traces/prot/res/logs were excluded.")
    lines.append("  Material HDF5 files were replaced by iter007 accepted material.")
    lines.append("  Source files were regenerated from strict residual H5.")
    lines.append("")
    lines.append("RESULT = PASS" if result["ok"] else "RESULT = CHECK")

    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
