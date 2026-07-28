from pathlib import Path
import os
from datetime import datetime
import json
import re

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

scripts = [
    "scripts/longterm/424B_compute_rhs_component_from_traces.py",
    "scripts/longterm/425A_sum_rhs_xyz_components.py",
    "scripts/longterm/426A_audit_mtilde_vs_rhs_total.py",
    "scripts/longterm/426C_solve_mtilde_interior_rhs_total.py",
    "scripts/fathi_loop_v2/wrappers/wrap_assemble_rhs_total.py",
    "scripts/fathi_loop_v2/wrappers/wrap_solve_mtilde.py",
    "scripts/armonik/tasks/task_reduce_rhs_component.py",
    "scripts/armonik/tasks/task_assemble_rhs_total.py",
    "scripts/armonik/tasks/task_solve_mtilde.py",
]

out_dir = ROOT / "benchmark_fathi_strict/reports/rhs_discovery"
out_dir.mkdir(parents=True, exist_ok=True)

out_txt = out_dir / "rhs_source_scripts_audit.txt"
out_json = out_dir / "rhs_source_scripts_audit.json"

patterns = [
    "ROOT",
    "OUT",
    "DIR",
    "Path",
    "np.save",
    "np.load",
    "capteurs",
    "RHS",
    "rhs",
    "Mtilde",
    "mtilde",
    "g_lambda",
    "g_mu",
    "full_grid_trace",
    "component_rhs",
    "mtilde_solve",
]

records = []

lines = []
lines.append("RHS / Mtilde source scripts audit")
lines.append("=================================")
lines.append("")
lines.append(f"created = {datetime.now().isoformat()}")
lines.append("")

for rel in scripts:
    p = ROOT / rel
    rec = {
        "script": rel,
        "exists": p.exists(),
        "hits": [],
    }

    lines.append("-" * 100)
    lines.append(f"script = {rel}")
    lines.append(f"exists = {p.exists()}")

    if not p.exists():
        records.append(rec)
        continue

    text = p.read_text(errors="ignore")
    text_lines = text.splitlines()

    for i, line in enumerate(text_lines, start=1):
        if any(pattern in line for pattern in patterns):
            hit = {
                "line": i,
                "text": line.rstrip(),
            }
            rec["hits"].append(hit)

    lines.append("")
    lines.append("Important lines:")
    for hit in rec["hits"][:250]:
        lines.append(f"  L{hit['line']:04d}: {hit['text']}")

    records.append(rec)

payload = {
    "created": datetime.now().isoformat(),
    "records": records,
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append("RESULT = PASS")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
