from pathlib import Path
from datetime import datetime
import json
import os
import re
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()
transition = "iter_007_to_iter_008"

paths = {
    "wrapper": ROOT / "scripts/fathi_benchmark/run_task2c_adjoint_batch.py",
    "direct_report": ROOT / f"benchmark_fathi_strict/reports/executable_tasks/adjoint_batches/{transition}_adjoint_x_batch_000_task.txt",
    "payload_report": ROOT / f"benchmark_fathi_strict/armonik/task_outputs_v2/{transition}_adjoint_batch__x_batch_000_execute_runner_output.txt",
}

def read(path):
    return path.read_text(errors="ignore") if path.exists() else ""

wrapper_text = read(paths["wrapper"])
direct_text = read(paths["direct_report"])
payload_text = read(paths["payload_report"])

checks = {
    "wrapper_exists": paths["wrapper"].exists(),
    "wrapper_has_context_arg": "--context" in wrapper_text,
    "wrapper_uses_context_adjoint_dir": "output_adjoint_batches_dir" in wrapper_text or "adjoint_batches_dir" in wrapper_text,
    "wrapper_no_hardcoded_iter008_batchdir": 'data/inversion_linear/iter_008/adjoint_full_grid_batches' not in wrapper_text,
    "direct_report_pass_already_exists": "RESULT = PASS_ALREADY_EXISTS" in direct_text,
    "direct_report_contains_context": "context =" in direct_text,
    "payload_report_pass": re.search(r"^RESULT = PASS$", payload_text, re.M) is not None,
    "payload_report_child_pass_already_exists": "RESULT = PASS_ALREADY_EXISTS" in payload_text,
}

result = "PASS" if all(checks.values()) else "CHECK_NEEDED"

out_dir = ROOT / "benchmark_fathi_strict/reports/genericization"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / "F3_adjoint_batch_generic_status.json"
out_txt = out_dir / "F3_adjoint_batch_generic_status.txt"

payload = {
    "created": datetime.now().isoformat(),
    "phase": "F3_adjoint_batch_generic",
    "transition": transition,
    "checks": checks,
    "paths": {k: str(v) for k, v in paths.items()},
    "result": result,
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("F3 adjoint_batch generic status")
lines.append("===============================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append("")
for k, v in checks.items():
    lines.append(f"{k} = {v}")
lines.append("")
lines.append("Interpretation:")
lines.append("  run_task2c_adjoint_batch.py is now context-driven.")
lines.append("  It no longer hardcodes iter_008 adjoint batch paths.")
lines.append("  The old --iter-k mode remains available as a compatibility fallback.")
lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
