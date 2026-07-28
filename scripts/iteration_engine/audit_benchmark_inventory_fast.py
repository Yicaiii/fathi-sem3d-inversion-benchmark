from pathlib import Path
import os
from datetime import datetime
import json
import csv
import re

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()
OUT = ROOT / "benchmark_fathi_strict/reports/inventory"
OUT.mkdir(parents=True, exist_ok=True)

STATE_DIR = ROOT / "results/fathi_loop_v2/states_corrected"
DATA_ROOT = ROOT / "data/inversion_linear"
RESULT_ROOT = ROOT / "results/fathi_loop_v2"

def has_pass(p: Path):
    if not p.exists():
        return False
    txt = p.read_text(errors="ignore")
    return "RESULT = PASS" in txt or "RESULT=PASS" in txt

def exists_rel(p: Path):
    return str(p.relative_to(ROOT)) if p.exists() and str(p).startswith(str(ROOT)) else str(p)

def state_iter_from_name(p: Path):
    m = re.search(r"iter_(\d+)_state", p.name)
    return int(m.group(1)) if m else None

print("Scanning states...", flush=True)

states = []
for p in sorted(STATE_DIR.glob("iter_*_state*.npz")):
    states.append({
        "iter": state_iter_from_name(p),
        "state_path": exists_rel(p),
        "exists": p.exists(),
    })

print("Scanning transitions...", flush=True)

transition_names = set()

for p in RESULT_ROOT.glob("iter_*_to_iter_*"):
    if p.is_dir():
        transition_names.add(p.name)

for p in DATA_ROOT.glob("iter_*"):
    if p.is_dir():
        m = re.search(r"iter_(\d+)$", p.name)
        if m:
            kp1 = int(m.group(1))
            if kp1 > 0:
                transition_names.add(f"iter_{kp1-1:03d}_to_iter_{kp1:03d}")

transitions = []

for name in sorted(transition_names):
    print(f"  checking {name}", flush=True)

    m = re.search(r"iter_(\d+)_to_iter_(\d+)", name)
    if not m:
        continue

    k = int(m.group(1))
    kp1 = int(m.group(2))

    result_dir = RESULT_ROOT / name
    data_dir = DATA_ROOT / f"iter_{kp1:03d}"
    accepted_dir = data_dir / "accepted"
    state_out = STATE_DIR / f"iter_{kp1:03d}_state_v2_corrected.npz"

    # Engineering completion evidence
    engineering_reports = [
        result_dir / "reports/446F_iter006_to_iter007_engineering_loop_report.txt",
        result_dir / "446F_iter006_to_iter007_engineering_loop_report.txt",
    ]
    engineering_accept_pass = any(has_pass(p) for p in engineering_reports)

    # Strict stage evidence
    strict_forward_audit = result_dir / "strict_forward/449E_strict_full_forward_000_strict_forward_pilot_audit_summary.txt"
    residual_454A = result_dir / "residual_sources/454A_strict_forward_residual_manifest_summary.txt"
    residual_454B = result_dir / "residual_sources/454B_strict_residual_timeseries_summary.txt"
    adjoint_prep_455C = result_dir / "residual_sources/455C_audit_strict_adjoint_batches_summary.txt"

    adjoint_runs_dir = result_dir / "adjoint_runs"
    adjoint_run_pass_count = 0
    adjoint_run_total = 0
    if adjoint_runs_dir.exists():
        for p in adjoint_runs_dir.glob("456A_*_run_summary.txt"):
            txt = p.read_text(errors="ignore")
            if "execute = True" in txt:
                adjoint_run_total += 1
                if "RESULT = PASS" in txt:
                    adjoint_run_pass_count += 1

    has_accepted = accepted_dir.exists()
    has_state_out = state_out.exists()

    s01 = has_pass(strict_forward_audit)
    s02 = has_pass(residual_454A) and has_pass(residual_454B)
    s03prep = has_pass(adjoint_prep_455C)

    if has_accepted and has_state_out:
        status = "COMPLETED_ACCEPTED"
    elif engineering_accept_pass:
        status = "COMPLETED_ENGINEERING_EVIDENCE"
    elif s01 or s02 or s03prep or adjoint_run_total > 0:
        status = "IN_PROGRESS_STRICT"
    else:
        status = "UNKNOWN_OR_OLD_PARTIAL"

    if status.startswith("COMPLETED"):
        resume_point = f"next iteration can start from state iter_{kp1:03d}"
    else:
        if not s01:
            resume_point = "S01 strict forward"
        elif not s02:
            resume_point = "S02 residual source generation"
        elif not s03prep:
            resume_point = "S03-S05 adjoint batch preparation"
        elif adjoint_run_pass_count < 30:
            resume_point = f"S03-S05 adjoint execution, passed {adjoint_run_pass_count}/30"
        else:
            resume_point = "S06-S09 RHS assembly"

    transitions.append({
        "transition": name,
        "iter_k": k,
        "iter_kp1": kp1,
        "status": status,
        "resume_point": resume_point,
        "result_dir_exists": result_dir.exists(),
        "data_dir_exists": data_dir.exists(),
        "accepted_exists": has_accepted,
        "state_out_exists": has_state_out,
        "S01_strict_forward_PASS": s01,
        "S02_residual_PASS": s02,
        "S03_S05_adjoint_prep_PASS": s03prep,
        "adjoint_run_pass_count": adjoint_run_pass_count,
        "adjoint_run_total_executed": adjoint_run_total,
    })

payload = {
    "created": datetime.now().isoformat(),
    "states": states,
    "transitions": transitions,
}

json_path = OUT / "benchmark_inventory_fast.json"
csv_path = OUT / "transition_inventory_fast.csv"
txt_path = OUT / "benchmark_inventory_fast_summary.txt"

json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

with csv_path.open("w", newline="", encoding="utf-8") as f:
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
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in transitions:
        w.writerow({k: r.get(k, "") for k in fieldnames})

lines = []
lines.append("Fathi benchmark inventory FAST")
lines.append("==============================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append("")
lines.append("States found:")
for s in states:
    lines.append(f"  iter={s['iter']} exists={s['exists']} path={s['state_path']}")

lines.append("")
lines.append("Transitions:")
for r in transitions:
    lines.append("-" * 80)
    lines.append(f"transition = {r['transition']}")
    lines.append(f"status = {r['status']}")
    lines.append(f"resume_point = {r['resume_point']}")
    lines.append(f"accepted_exists = {r['accepted_exists']}")
    lines.append(f"state_out_exists = {r['state_out_exists']}")
    lines.append(f"S01_strict_forward_PASS = {r['S01_strict_forward_PASS']}")
    lines.append(f"S02_residual_PASS = {r['S02_residual_PASS']}")
    lines.append(f"S03_S05_adjoint_prep_PASS = {r['S03_S05_adjoint_prep_PASS']}")
    lines.append(f"adjoint_run_pass_count = {r['adjoint_run_pass_count']}/30")

lines.append("")
lines.append(f"json = {json_path}")
lines.append(f"csv = {csv_path}")
lines.append("")
lines.append("RESULT = PASS")

txt_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines), flush=True)
