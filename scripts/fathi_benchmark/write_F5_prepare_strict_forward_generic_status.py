from pathlib import Path
from datetime import datetime
import json
import os
import re
import sys
import hashlib

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

transition_ref = "iter_007_to_iter_008"
transition_next = "iter_008_to_iter_009"

paths = {
    "generic_450b": ROOT / "scripts/fathi_benchmark/generic_from_legacy/450B_select_strict_forward_full_template_generic.py",
    "generic_450c": ROOT / "scripts/fathi_benchmark/generic_from_legacy/450C_prepare_strict_full_forward_run_generic.py",
    "wrapper": ROOT / "scripts/fathi_benchmark/run_task1b_prepare_strict_forward.py",
    "direct_ref_report": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/{transition_ref}_prepare_strict_forward_task.txt",
    "direct_next_report": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/{transition_next}_prepare_strict_forward_task.txt",
    "payload_ref_report": ROOT / f"benchmark_fathi_strict/armonik/task_outputs_v2/{transition_ref}_prepare_strict_forward_execute_runner_output.txt",
    "iter008_accepted_h5": ROOT / "data/inversion_linear/iter_008/accepted/mat/h5",
    "iter009_workspace_h5": ROOT / "data/inversion_linear/iter_009/forward_dudx_mgcap_full_batches/strict_full_forward_000/mat/h5",
    "iter009_workspace": ROOT / "data/inversion_linear/iter_009/forward_dudx_mgcap_full_batches/strict_full_forward_000",
}

def read(path):
    return path.read_text(errors="ignore") if path.exists() else ""

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def same_file(a, b):
    return a.exists() and b.exists() and sha256(a) == sha256(b)

generic_450b_text = read(paths["generic_450b"])
generic_450c_text = read(paths["generic_450c"])
wrapper_text = read(paths["wrapper"])
direct_ref_text = read(paths["direct_ref_report"])
direct_next_text = read(paths["direct_next_report"])
payload_ref_text = read(paths["payload_ref_report"])

payload_files = sorted(
    (ROOT / "benchmark_fathi_strict/armonik/payloads_v2").glob("*prepare_strict_forward*execute.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

latest_payload = payload_files[0] if payload_files else None
latest_payload_data = {}
if latest_payload:
    try:
        latest_payload_data = json.loads(latest_payload.read_text())
    except Exception:
        latest_payload_data = {}

material_names = [
    "Mat_0_Kappa.h5",
    "Mat_0_Mu.h5",
    "Mat_0_Density.h5",
]

material_audit = {}
for name in material_names:
    src = paths["iter008_accepted_h5"] / name
    dst = paths["iter009_workspace_h5"] / name
    material_audit[name] = {
        "src": str(src),
        "dst": str(dst),
        "src_exists": src.exists(),
        "dst_exists": dst.exists(),
        "same_sha256": same_file(src, dst),
    }

required_workspace_files = {
    "input.spec": paths["iter009_workspace"] / "input.spec",
    "mesh.input": paths["iter009_workspace"] / "mesh.input",
    "material.input": paths["iter009_workspace"] / "material.input",
    "material.spec": paths["iter009_workspace"] / "material.spec",
    "Mat_0_Kappa.h5": paths["iter009_workspace_h5"] / "Mat_0_Kappa.h5",
    "Mat_0_Mu.h5": paths["iter009_workspace_h5"] / "Mat_0_Mu.h5",
    "Mat_0_Density.h5": paths["iter009_workspace_h5"] / "Mat_0_Density.h5",
}

workspace_audit = {k: v.exists() for k, v in required_workspace_files.items()}

checks = {
    "generic_450b_exists": paths["generic_450b"].exists(),
    "generic_450b_context_driven": "--context" in generic_450b_text and "iter_007_to_iter_008_iteration_context.json" not in generic_450b_text,

    "generic_450c_exists": paths["generic_450c"].exists(),
    "generic_450c_context_driven": "--context" in generic_450c_text and "iter_007_to_iter_008_iteration_context.json" not in generic_450c_text,

    "wrapper_exists": paths["wrapper"].exists(),
    "wrapper_runs_450b_then_450c": "script_450b" in wrapper_text and "script_450c" in wrapper_text,
    "wrapper_uses_context_workspace": "strict_forward_workspace" in wrapper_text,
    "wrapper_uses_parent_accepted_dir": "input_accepted_dir" in wrapper_text,

    "iter007_reference_report_pass": ("RESULT = PASS_ALREADY_EXISTS" in direct_ref_text) or ("RESULT = PASS_EXECUTED" in direct_ref_text),
    "iter008_to_iter009_prepare_pass": ("RESULT = PASS_EXECUTED" in direct_next_text) or ("RESULT = PASS_ALREADY_EXISTS" in direct_next_text),
    "iter009_workspace_required_files_exist": all(workspace_audit.values()),

    "payload_report_pass": re.search(r"^RESULT = PASS$", payload_ref_text, re.M) is not None,
    "payload_child_pass": ("RESULT = PASS_ALREADY_EXISTS" in payload_ref_text) or ("RESULT = PASS_EXECUTED" in payload_ref_text),
    "latest_payload_heavy_false": latest_payload_data.get("task_type") == "prepare_strict_forward" and latest_payload_data.get("heavy") is False,
    "latest_payload_non_mutating": latest_payload_data.get("task_type") == "prepare_strict_forward" and latest_payload_data.get("mutates_state") is False,

    "material_from_iter008_accepted": all(v["same_sha256"] for v in material_audit.values()),
}

result = "PASS" if all(checks.values()) else "CHECK_NEEDED"

out_dir = ROOT / "benchmark_fathi_strict/reports/genericization"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / "F5_prepare_strict_forward_generic_status.json"
out_txt = out_dir / "F5_prepare_strict_forward_generic_status.txt"

payload = {
    "created": datetime.now().isoformat(),
    "phase": "F5_prepare_strict_forward_generic",
    "transition_reference": transition_ref,
    "transition_next": transition_next,
    "checks": checks,
    "workspace_audit": workspace_audit,
    "material_audit": material_audit,
    "latest_payload": str(latest_payload) if latest_payload else None,
    "latest_payload_data": latest_payload_data,
    "paths": {k: str(v) for k, v in paths.items()},
    "result": result,
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("F5 prepare_strict_forward generic status")
lines.append("========================================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"reference_transition = {transition_ref}")
lines.append(f"next_transition = {transition_next}")
lines.append("")
lines.append("Checks:")
for k, v in checks.items():
    lines.append(f"  {k} = {v}")

lines.append("")
lines.append("Iter009 workspace audit:")
for k, v in workspace_audit.items():
    lines.append(f"  {k} = {v}")

lines.append("")
lines.append("Material source audit: iter_008/accepted -> iter_009 strict forward workspace")
for name, info in material_audit.items():
    lines.append(f"  {name}:")
    lines.append(f"    src_exists = {info['src_exists']}")
    lines.append(f"    dst_exists = {info['dst_exists']}")
    lines.append(f"    same_sha256 = {info['same_sha256']}")

lines.append("")
lines.append("Latest prepare_strict_forward payload:")
lines.append(f"  path = {latest_payload}")
lines.append(f"  heavy = {latest_payload_data.get('heavy')}")
lines.append(f"  mutates_state = {latest_payload_data.get('mutates_state')}")

lines.append("")
lines.append("Interpretation:")
lines.append("  F5 prepares the strict forward SEM3D workspace from the current accepted model.")
lines.append("  It does not run SEM3D.")
lines.append("  For iter_008 -> iter_009, the workspace exists and its material HDF5 files match iter_008/accepted.")
lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
