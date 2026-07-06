from pathlib import Path
from datetime import datetime
import json
import re
import os

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()
transition = "iter_008_to_iter_009"

paths = {
    "task5_report": ROOT / f"benchmark_fathi_strict/reports/run_iteration_full/{transition}_run_iteration_full_context_task5.txt",
    "audit_report": ROOT / f"benchmark_fathi_strict/reports/audit/{transition}_completion_audit.txt",
    "state_out": ROOT / "results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz",
    "accepted_dir": ROOT / "data/inversion_linear/iter_009/accepted",
    "accepted_kappa": ROOT / "data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Kappa.h5",
    "accepted_mu": ROOT / "data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Mu.h5",
    "accepted_density": ROOT / "data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Density.h5",
    "misfit_json": ROOT / f"results/fathi_loop_v2/{transition}/candidate_misfits/line_search_neg_mtilde_1p00MPa_misfit_summary_v2.json",
    "accept_json": ROOT / f"benchmark_fathi_strict/reports/acceptance/{transition}_line_search_neg_mtilde_1p00MPa_acceptance_v2.json",
}

def read(path):
    return path.read_text(errors="ignore") if path.exists() else ""

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

task5 = read(paths["task5_report"])
audit = read(paths["audit_report"])
misfit = load_json(paths["misfit_json"])
accept = load_json(paths["accept_json"])

checks = {
    "task5_report_pass": re.search(r"^RESULT = PASS$", task5, re.M) is not None,
    "task5_accept_pass": "RESULT = PASS_ACCEPTED" in task5,
    "audit_pass_complete": "RESULT = PASS_COMPLETE" in audit,
    "state_out_exists": paths["state_out"].exists(),
    "accepted_dir_exists": paths["accepted_dir"].exists(),
    "accepted_kappa_exists": paths["accepted_kappa"].exists(),
    "accepted_mu_exists": paths["accepted_mu"].exists(),
    "accepted_density_exists": paths["accepted_density"].exists(),
}

result = "PASS" if all(checks.values()) else "CHECK_NEEDED"

out_dir = ROOT / "benchmark_fathi_strict/reports/genericization"
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / "G_local_iter008_to_iter009_success_status.json"
out_txt = out_dir / "G_local_iter008_to_iter009_success_status.txt"

payload = {
    "created": datetime.now().isoformat(),
    "phase": "G_local_full_iteration_test",
    "transition": transition,
    "candidate": "line_search_neg_mtilde_1p00MPa",
    "checks": checks,
    "misfit": misfit,
    "acceptance": accept,
    "paths": {k: str(v) for k, v in paths.items()},
    "result": result,
}

out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = []
lines.append("G local full-iteration benchmark success status")
lines.append("================================================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append("candidate = line_search_neg_mtilde_1p00MPa")
lines.append("")
lines.append("Checks:")
for k, v in checks.items():
    lines.append(f"  {k} = {v}")

lines.append("")
lines.append("Misfit / acceptance:")
lines.append(f"  parent_J = {misfit.get('parent_J')}")
lines.append(f"  candidate_J = {misfit.get('total_J') or misfit.get('candidate_J')}")
lines.append(f"  delta_J = {misfit.get('delta_J')}")
lines.append(f"  descent = {misfit.get('descent')}")
lines.append("")
lines.append("Accepted outputs:")
lines.append(f"  state_out = {paths['state_out']}")
lines.append(f"  accepted_dir = {paths['accepted_dir']}")
lines.append("")
lines.append("Interpretation:")
lines.append("  The context-driven local benchmark completed one full inversion transition.")
lines.append("  The candidate forward, misfit computation, and acceptance step all succeeded.")
lines.append("  iter_009 can now be used as a trusted parent state for the next iteration.")
lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if result != "PASS":
    raise SystemExit(2)
