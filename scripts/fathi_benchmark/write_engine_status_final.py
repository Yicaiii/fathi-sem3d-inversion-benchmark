from pathlib import Path
import os
from datetime import datetime
import json
import numpy as np
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

config_path = ROOT / "benchmark_fathi_strict/config/benchmark_config.json"
config = json.loads(config_path.read_text())

transition = "iter_007_to_iter_008"

paths = {
    "config": config_path,
    "task3_wrapper": ROOT / "scripts/fathi_benchmark/run_task3_gradient.py",
    "task4_wrapper": ROOT / "scripts/fathi_benchmark/run_task4_candidates.py",
    "task5_wrapper": ROOT / "scripts/fathi_benchmark/run_task5_candidate.py",
    "run_iteration": ROOT / "scripts/fathi_benchmark/run_iteration.py",
    "deprecated_doc": ROOT / "benchmark_fathi_strict/docs/deprecated_scripts_do_not_use.txt",
    "consolidation_plan": ROOT / "benchmark_fathi_strict/docs/benchmark_engine_consolidation_plan.txt",
    "iter008_state": ROOT / "results/fathi_loop_v2/states_corrected/iter_008_state_v2_corrected.npz",
    "accepted_summary": ROOT / "data/inversion_linear/iter_008/accepted/accepted_summary.txt",
    "stage_report": ROOT / "benchmark_fathi_strict/reports/stage_reports/iter_007_to_iter_008_stage_report_v2.txt",
    "task3_plan": ROOT / "benchmark_fathi_strict/reports/task_wrappers/iter_007_to_iter_008_task3_gradient_plan.txt",
    "task4_plan": ROOT / "benchmark_fathi_strict/reports/task_wrappers/iter_007_to_iter_008_task4_candidates_plan.txt",
    "task5_plan": ROOT / "benchmark_fathi_strict/reports/task5_wrappers/iter_007_to_iter_008_line_search_neg_mtilde_1p00MPa_task5_plan.txt",
    "run_iteration_plan": ROOT / "benchmark_fathi_strict/reports/run_iteration/iter_007_to_iter_008_run_iteration_plan.txt",
    "misfit_v2": ROOT / "results/fathi_loop_v2/iter_007_to_iter_008/candidate_misfits/line_search_neg_mtilde_1p00MPa_misfit_summary_v2.json",
    "acceptance_v2": ROOT / "benchmark_fathi_strict/reports/acceptance/iter_007_to_iter_008_line_search_neg_mtilde_1p00MPa_acceptance_v2.json",
}

exists = {k: p.exists() for k, p in paths.items()}

missing = [k for k, ok in exists.items() if not ok]

payload = {
    "created": datetime.now().isoformat(),
    "transition": transition,
    "paths": {k: str(p) for k, p in paths.items()},
    "exists": exists,
    "missing": missing,
}

lines = []
lines.append("Fathi benchmark engine final status")
lines.append("===================================")
lines.append("")
lines.append(f"created = {payload['created']}")
lines.append(f"transition = {transition}")
lines.append("")

if missing:
    lines.append("Missing expected files:")
    for k in missing:
        lines.append(f"  {k}: {paths[k]}")
    result = "CHECK_NEEDED"
else:
    state = np.load(paths["iter008_state"])
    acceptance = json.loads(paths["acceptance_v2"].read_text())
    misfit = json.loads(paths["misfit_v2"].read_text())

    parent_J = float(acceptance["parent_J"])
    candidate_J = float(acceptance["candidate_J"])
    delta_J = float(acceptance["delta_J"])
    descent = bool(acceptance["descent"])

    state_J = float(state["J"])
    misfit_J = float(misfit["total_J"])

    checks = {
        "parent_J_matches_expected": abs(parent_J - 3.8268653135568962e-19) < 1e-30,
        "candidate_J_matches_expected": abs(candidate_J - 3.8263972312235541e-19) < 1e-30,
        "delta_negative": delta_J < 0,
        "descent_true": descent,
        "state_J_matches_candidate_J": abs(state_J - candidate_J) < 1e-40,
        "misfit_J_matches_candidate_J": abs(misfit_J - candidate_J) < 1e-40,
        "iter008_state_has_lambda": "lambda" in state.files,
        "iter008_state_has_mu": "mu" in state.files,
        "iter008_state_has_kappa": "kappa" in state.files,
        "iter008_state_has_density": "density" in state.files,
    }

    payload.update({
        "parent_J": parent_J,
        "candidate_J": candidate_J,
        "delta_J": delta_J,
        "descent": descent,
        "state_keys": list(state.files),
        "checks": checks,
    })

    result = "PASS" if all(checks.values()) else "CHECK_NEEDED"

    lines.append("Validated result:")
    lines.append(f"  parent_J    = {parent_J:.16e}")
    lines.append(f"  candidate_J = {candidate_J:.16e}")
    lines.append(f"  delta_J     = {delta_J:.16e}")
    lines.append(f"  descent     = {descent}")
    lines.append("")
    lines.append("Accepted state:")
    lines.append(f"  {paths['iter008_state']}")
    lines.append("")
    lines.append("Canonical wrappers:")
    lines.append(f"  Task 3 gradient   = {paths['task3_wrapper']}")
    lines.append(f"  Task 4 candidates = {paths['task4_wrapper']}")
    lines.append(f"  Task 5 candidate  = {paths['task5_wrapper']}")
    lines.append(f"  run_iteration     = {paths['run_iteration']}")
    lines.append("")
    lines.append("Canonical Task 5 note:")
    lines.append("  Task 5 must use compute_candidate_misfit_v2.py and accept_candidate_if_descent_v2.py.")
    lines.append("  The old candidate misfit script is deprecated.")
    lines.append("")
    lines.append("Checks:")
    for k, v in checks.items():
        lines.append(f"  {k} = {v}")

payload["result"] = result

out_txt = ROOT / "benchmark_fathi_strict/ENGINE_STATUS_FINAL.txt"
out_json = ROOT / "benchmark_fathi_strict/ENGINE_STATUS_FINAL.json"

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append(f"RESULT = {result}")

out_txt.write_text("\n".join(lines), encoding="utf-8")
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("\n".join(lines))

if result != "PASS":
    sys.exit(2)
