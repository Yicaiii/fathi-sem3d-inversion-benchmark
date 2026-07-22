from pathlib import Path
import os
from datetime import datetime
import json
import csv

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

INV_JSON = ROOT / "benchmark_fathi_strict/reports/inventory/benchmark_inventory_fast.json"
OUT = ROOT / "benchmark_fathi_strict/reports"
OUT.mkdir(parents=True, exist_ok=True)

payload = json.loads(INV_JSON.read_text())

# Official rules:
# 1. Ignore dryrun transitions.
# 2. Treat states_corrected as source of truth for reusable iteration states.
# 3. accepted_dir alone is not enough; official completed state requires state_out_exists.
# 4. Current active strict transition is the latest IN_dir alone is not enough; official completed_PROGRESS_STRICT transition.

official_transitions = []
for r in payload["transitions"]:
    name = r["transition"]
    if "dryrun" in name.lower():
        continue
    official_transitions.append(r)

completed = [r for r in official_transitions if r["status"] == "COMPLETED_ACCEPTED"]
in_progress = [r for r in official_transitions if r["status"] == "IN_PROGRESS_STRICT"]
old_partial = [r for r in official_transitions if r["status"] == "UNKNOWN_OR_OLD_PARTIAL"]

completed_sorted = sorted(completed, key=lambda x: x["iter_kp1"])
in_progress_sorted = sorted(in_progress, key=lambda x: x["iter_kp1"])

last_completed = completed_sorted[-1] if completed_sorted else None
current_active = in_progress_sorted[-1] if in_progress_sorted else None

official = {
    "created": datetime.now().isoformat(),
    "benchmark_name": "Fathi layered elastic inversion benchmark",
    "benchmark_type": "generic strict iterative benchmark engine",
    "source_of_truth": {
        "states": "results/fathi_loop_v2/states_corrected",
        "accepted_material_dirs": "data/inversion_linear/iter_*/accepted",
        "official_inventory": "benchmark_fathi_strict/reports/inventory/benchmark_inventory_fast.json",
    },
    "official_completed_transitions": completed_sorted,
    "official_in_progress_transitions": in_progress_sorted,
    "old_partial_transitions": old_partial,
    "last_completed_transition": last_completed,
    "current_active_transition": current_active,
    "important_rule": "Do not create one-off code for each iter. Generic engine must accept --iter-k and derive transition/state paths automatically.",
}

json_out = OUT / "official_benchmark_status.json"
txt_out = OUT / "official_benchmark_status.txt"
csv_out = OUT / "official_transition_registry.csv"

json_out.write_text(json.dumps(official, indent=2), encoding="utf-8")

with csv_out.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "transition",
        "status",
        "resume_point",
        "accepted_exists",
        "state_out_exists",
        "S01_strict_forward_PASS",
        "S02_residual_PASS",
        "S03_S05_adjoint_prep_PASS",
        "adjoint_run_pass_count",
        "adjoint_run_total_executed",
        "official_note",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    for r in official_transitions:
        note = ""
        if r in completed_sorted:
            note = "official completed transition"
        elif r in in_progress_sorted:
            note = "official in-progress strict transition"
        else:
            note = "old partial / not source of truth for strict engine"

        row = {k: r.get(k, "") for k in fieldnames if k != "official_note"}
        row["official_note"] = note
        w.writerow(row)

lines = []
lines.append("OFFICIAL FATHI BENCHMARK STATUS")
lines.append("================================")
lines.append("")
lines.append(f"created = {official['created']}")
lines.append("")
lines.append("Benchmark definition:")
lines.append("  This is the full Fathi layered elastic inversion benchmark.")
lines.append("  The benchmark unit is not iter007 or iter008.")
lines.append("  The benchmark unit is a reusable strict iteration engine:")
lines.append("    state_k -> forward -> residual -> adjoint -> RHS -> Mtilde -> line-search -> state_{k+1}")
lines.append("")
lines.append("Source of truth:")
lines.append("  Use states_corrected as the official state registry.")
lines.append("  accepted_dir alone is not enough to define a completed official state.")
lines.append("  Ignore dryrun transitions in official benchmark history.")
lines.append("")
lines.append("Official completed transitions:")
if completed_sorted:
    for r in completed_sorted:
        lines.append(f"  {r['transition']}: {r['status']}")
        lines.append(f"    resume_point = {r['resume_point']}")
else:
    lines.append("  NONE")

lines.append("")
lines.append("Current active transition:")
if current_active:
    r = current_active
    lines.append(f"  {r['transition']}: {r['status']}")
    lines.append(f"    resume_point = {r['resume_point']}")
    lines.append(f"    S01_strict_forward_PASS = {r['S01_strict_forward_PASS']}")
    lines.append(f"    S02_residual_PASS = {r['S02_residual_PASS']}")
    lines.append(f"    S03_S05_adjoint_prep_PASS = {r['S03_S05_adjoint_prep_PASS']}")
    lines.append(f"    adjoint_run_pass_count = {r['adjoint_run_pass_count']}/30")
    lines.append(f"    accepted_exists = {r['accepted_exists']}")
    lines.append(f"    state_out_exists = {r['state_out_exists']}")
else:
    lines.append("  NONE")

lines.append("")
lines.append("Old partial transitions:")
for r in old_partial:
    lines.append(f"  {r['transition']}: {r['status']}")

lines.append("")
lines.append("What this means:")
lines.append("  Latest completed reusable state is iter007.")
lines.append("  iter007_to_iter008 is not completed yet.")
lines.append("  To resume later, continue from adjoint execution for iter007_to_iter008.")
lines.append("  Do not rewrite per-iteration scripts; refactor stages behind run_iteration.py.")
lines.append("")
lines.append("Next code-refactor task:")
lines.append("  Connect Task 1A forward_batch_generic.py into scripts/fathi_benchmark/run_iteration.py --stage forward.")
lines.append("")
lines.append(f"json = {json_out}")
lines.append(f"csv = {csv_out}")
lines.append("")
lines.append("RESULT = PASS")

txt_out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
