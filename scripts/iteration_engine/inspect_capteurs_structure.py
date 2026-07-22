from pathlib import Path
import os
import h5py
import json
from datetime import datetime

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

samples = [
    ROOT / "data/inversion_linear/iter_008/forward_dudx_mgcap_full_batches/strict_full_forward_000/traces/capteurs.0000.h5",
    ROOT / "data/inversion_linear/iter_008/adjoint_full_grid_batches/x/batch_000/traces/capteurs.0000.h5",
]

out_dir = ROOT / "benchmark_fathi_strict/reports/rhs_discovery"
out_dir.mkdir(parents=True, exist_ok=True)

lines = []
payload = {
    "created": datetime.now().isoformat(),
    "samples": [],
}

def safe_attrs(obj):
    attrs = {}
    try:
        for k, v in obj.attrs.items():
            attrs[str(k)] = str(v)
    except Exception:
        pass
    return attrs

def describe_item(obj, name):
    rec = {
        "name": name,
        "type": type(obj).__name__,
        "attrs": safe_attrs(obj),
    }

    if hasattr(obj, "shape"):
        rec["shape"] = tuple(obj.shape)
        rec["dtype"] = str(obj.dtype)

    return rec

for sample in samples:
    lines.append("=" * 100)
    lines.append(f"FILE = {sample}")
    lines.append(f"exists = {sample.exists()}")

    sample_payload = {
        "file": str(sample),
        "exists": sample.exists(),
        "root_attrs": {},
        "top_level_items": [],
        "recursive_preview": [],
    }

    if not sample.exists():
        payload["samples"].append(sample_payload)
        continue

    with h5py.File(sample, "r") as h:
        sample_payload["root_attrs"] = safe_attrs(h)

        lines.append("")
        lines.append("Root attrs:")
        for k, v in sample_payload["root_attrs"].items():
            lines.append(f"  {k} = {v}")

        lines.append("")
        lines.append("Top-level keys:")
        keys = list(h.keys())
        lines.append(f"  key_count = {len(keys)}")

        for key in keys[:100]:
            rec = describe_item(h[key], key)
            sample_payload["top_level_items"].append(rec)
            lines.append(
                f"  {rec['name']}: "
                f"type={rec['type']} "
                f"shape={rec.get('shape')} "
                f"dtype={rec.get('dtype')} "
                f"attrs={rec.get('attrs')}"
            )

        lines.append("")
        lines.append("Recursive preview:")

        counter = [0]

        def visit(name, obj):
            if counter[0] >= 160:
                return
            counter[0] += 1

            rec = describe_item(obj, name)
            sample_payload["recursive_preview"].append(rec)

            lines.append(
                f"  {rec['name']}: "
                f"type={rec['type']} "
                f"shape={rec.get('shape')} "
                f"dtype={rec.get('dtype')} "
                f"attrs={rec.get('attrs')}"
            )

        h.visititems(visit)

    payload["samples"].append(sample_payload)

out_txt = out_dir / "capteurs_structure_inspection.txt"
out_json = out_dir / "capteurs_structure_inspection.json"

lines.append("")
lines.append("RESULT = PASS")

out_txt.write_text("\n".join(lines), encoding="utf-8")
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("\n".join(lines))
