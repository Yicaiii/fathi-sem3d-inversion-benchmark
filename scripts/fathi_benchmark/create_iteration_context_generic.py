from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import sys

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--iter-k", type=int, required=True)
parser.add_argument("--config", default="benchmark_fathi_strict/config/benchmark_config.json")
parser.add_argument("--write", action="store_true")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

cfg_path = ROOT / args.config
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

k = args.iter_k
kp1 = k + 1

transition = f"iter_{k:03d}_to_iter_{kp1:03d}"

run_data_root = ROOT / cfg["run_data_root"]
run_result_root = ROOT / cfg["run_result_root"]
state_dir = ROOT / cfg["state_dir"]

parent_state = state_dir / f"iter_{k:03d}_state_v2_corrected.npz"
next_state = state_dir / f"iter_{kp1:03d}_state_v2_corrected.npz"

output_iter_root = run_data_root / f"iter_{kp1:03d}"
transition_result_root = run_result_root / transition

accepted_parent_dir = run_data_root / f"iter_{k:03d}" / "accepted"
accepted_next_dir = output_iter_root / "accepted"

context = {
    "created": datetime.now().isoformat(),
    "project_root": str(ROOT),
    "config_path": str(cfg_path.relative_to(ROOT)),
    "iter_k": k,
    "iter_kp1": kp1,
    "transition": transition,

    "parent_state": str(parent_state.relative_to(ROOT)),
    "next_state": str(next_state.relative_to(ROOT)),

    "parent_accepted_dir": str(accepted_parent_dir.relative_to(ROOT)),
    "accepted_dir": str(accepted_next_dir.relative_to(ROOT)),

    "true_observed_traces_dir": cfg["true_observed_traces_dir"],
    "true_material_dir": cfg.get("true_material_dir", None),

    "output_iter_root": str(output_iter_root.relative_to(ROOT)),
    "transition_result_root": str(transition_result_root.relative_to(ROOT)),

    "strict_forward_workspace": str((output_iter_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000").relative_to(ROOT)),
    "strict_forward_traces": str((output_iter_root / "forward_dudx_mgcap_full_batches/strict_full_forward_000/traces").relative_to(ROOT)),

    "residual_dir": str((transition_result_root / "residual_sources").relative_to(ROOT)),
    "residual_h5": str((transition_result_root / "residual_sources/454B_strict_residual_timeseries.h5").relative_to(ROOT)),
    "residual_summary_txt": str((transition_result_root / "residual_sources/454B_strict_residual_timeseries_summary.txt").relative_to(ROOT)),

    "adjoint_batches_dir": str((output_iter_root / "adjoint_full_grid_batches").relative_to(ROOT)),

    "rhs_manifest_dir": str((transition_result_root / "rhs_manifests").relative_to(ROOT)),
    "component_rhs_dir": str((transition_result_root / "component_rhs").relative_to(ROOT)),
    "mtilde_dir": str((transition_result_root / "mtilde_solve").relative_to(ROOT)),

    "candidates_dir": str((transition_result_root / "candidates").relative_to(ROOT)),
    "candidate_forward_workspaces_dir": str((output_iter_root / "candidate_forward_workspaces").relative_to(ROOT)),
    "candidate_misfit_dir": str((transition_result_root / "candidate_misfits").relative_to(ROOT)),

    "sem3d_exe": cfg.get("sem3d_exe", str(Path.home() / "SEM/build/SEM3D/sem3d.exe")),
    "mpi_cores": cfg.get("mpi_cores", 12),

    "line_search_candidates": cfg.get("line_search", {}).get("amplitudes_MPa", [0.10, 0.25, 0.50, 1.00]),

    "role": "generic_iteration_context",
    "observed_data_rule": "use receiver observed traces only; true model is not used for update",

    # Legacy aliases for validated iter_007 -> iter_008 scripts.
    # These make old 454A/454B/455B/455C logic reusable with minimal changes.
    "root": str(ROOT),
    "work_root": str(transition_result_root),
    "true_observed_traces": str((ROOT / cfg["true_observed_traces_dir"]).resolve()),
    "input_state": str(parent_state),
    "output_state": str(next_state),
    "input_accepted_dir": str(accepted_parent_dir),
    "output_accepted_dir": str(accepted_next_dir),
    "output_forward_batches_dir": str((output_iter_root / "forward_dudx_mgcap_full_batches").resolve()),
    "output_adjoint_batches_dir": str((output_iter_root / "adjoint_full_grid_batches").resolve()),
    "output_candidate_runs_dir": str((output_iter_root / "candidate_forward_workspaces").resolve()),
    "line_search_dir": str((transition_result_root / "candidate_misfits").resolve()),
    "mtilde_solve_dir": str((transition_result_root / "mtilde_solve").resolve()),
    "component_rhs_dir": str((transition_result_root / "component_rhs").resolve()),
    "from_tag": f"iter_{k:03d}",
    "to_tag": f"iter_{kp1:03d}",
}

out_dir = transition_result_root
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / f"{transition}_iteration_context.json"
out_txt = out_dir / f"{transition}_iteration_context.txt"

checks = {
    "parent_state_exists": parent_state.exists(),
    "parent_accepted_dir_exists": accepted_parent_dir.exists(),
    "true_observed_traces_dir_exists": (ROOT / cfg["true_observed_traces_dir"]).exists(),
}

context["preflight_checks"] = checks
context["preflight_ok"] = all(checks.values())

lines = []
lines.append("Generic Fathi iteration context")
lines.append("===============================")
lines.append("")
lines.append(f"transition = {transition}")
lines.append(f"iter_k = {k}")
lines.append(f"iter_kp1 = {kp1}")
lines.append("")
lines.append("Core rule:")
lines.append("  The update uses observed receiver traces and synthetic receiver traces.")
lines.append("  The true model is not used in the update; it is only a synthetic benchmark reference.")
lines.append("")
lines.append("Paths:")
for key in [
    "parent_state",
    "next_state",
    "parent_accepted_dir",
    "accepted_dir",
    "true_observed_traces_dir",
    "strict_forward_workspace",
    "strict_forward_traces",
    "residual_h5",
    "adjoint_batches_dir",
    "component_rhs_dir",
    "mtilde_dir",
    "candidates_dir",
    "candidate_forward_workspaces_dir",
]:
    lines.append(f"  {key} = {context.get(key)}")

lines.append("")
lines.append("Preflight checks:")
for key, value in checks.items():
    lines.append(f"  {key} = {value}")

lines.append("")
lines.append(f"json = {out_json}")
lines.append("")
lines.append("RESULT = PASS" if context["preflight_ok"] else "RESULT = CHECK_NEEDED")

if args.write:
    if out_json.exists() and not args.overwrite:
        backup_note = []
        backup_note.append("")
        backup_note.append("WRITE_SKIPPED:")
        backup_note.append("  Context already exists.")
        backup_note.append("  Use --overwrite only if you intentionally want to replace it.")
        backup_note.append(f"  existing = {out_json}")
        lines.extend(backup_note)
    else:
        out_json.write_text(json.dumps(context, indent=2), encoding="utf-8")
        out_txt.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

if not context["preflight_ok"]:
    sys.exit(2)
