from pathlib import Path
from datetime import datetime
import json
import shutil
import os

ROOT = Path(os.environ.get("FATHI_BENCHMARK_ROOT", str(Path.home() / "sem3d_fathi_clean"))).expanduser().resolve()
config_path = ROOT / "benchmark_fathi_strict/config/benchmark_config.json"

cfg = json.loads(config_path.read_text(encoding="utf-8"))

backup = config_path.with_suffix(".json.backup_after_iter008_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
shutil.copy2(config_path, backup)

old_checkpoint = cfg.get("current_verified_checkpoint", None)

cfg["engine_scope"] = {
    "package_type": "resumed_engine_light",
    "automation_scope": "post_adjoint_resumed_iteration",
    "current_run_iteration_all_means": "gradient -> candidates -> task5 -> status",
    "assumes_existing_outputs": [
        "strict_forward_traces",
        "residual_sources",
        "adjoint_x_y_z_traces"
    ],
    "not_yet_full_standalone": True,
    "missing_standalone_stages": [
        "strict_forward",
        "residual_generation",
        "prepare_adjoint",
        "run_adjoint_batches",
        "audit_adjoint"
    ]
}

cfg["current_verified_checkpoint"] = {
    "last_completed_state": 8,
    "last_completed_transition": "iter_007_to_iter_008",
    "last_completed_status": "PASS_ACCEPTED",
    "accepted_candidate": "line_search_neg_mtilde_1p00MPa",
    "parent_J": 3.8268653135568962e-19,
    "candidate_J": 3.8263972312235541e-19,
    "delta_J": -4.6808233334203494e-23,
    "descent": True,
    "state_path": "results/fathi_loop_v2/states_corrected/iter_008_state_v2_corrected.npz",
    "accepted_dir": "data/inversion_linear/iter_008/accepted",
    "validated_stages_local": [
        "S01 strict forward",
        "S02 residual sources",
        "S03-S05 adjoint 30/30",
        "S06-S08 RHS_x_y_z",
        "S09 RHS_total",
        "S10 Mtilde solve",
        "S11 candidates",
        "S12 candidate forward and misfit",
        "S13 accept"
    ],
    "canonical_engine_wrappers": {
        "task3_gradient": "scripts/fathi_benchmark/run_task3_gradient.py",
        "task4_candidates": "scripts/fathi_benchmark/run_task4_candidates.py",
        "task5_candidate": "scripts/fathi_benchmark/run_task5_candidate.py",
        "run_iteration": "scripts/fathi_benchmark/run_iteration.py"
    }
}

cfg["active_transition"] = {
    "next_transition": "iter_008_to_iter_009",
    "status": "not_started",
    "parent_state": "results/fathi_loop_v2/states_corrected/iter_008_state_v2_corrected.npz"
}

cfg["armonik_integration"] = {
    "status": "local_payload_prototype_passed",
    "payload_policy": "small_json_parameters_only",
    "large_data_policy": "shared_filesystem_or_object_storage_paths_only",
    "root_env_variable": "FATHI_BENCHMARK_ROOT",
    "validated_payload_tasks": [
        "candidate_misfit"
    ],
    "next_payload_tasks": [
        "candidate_forward",
        "rhs_component",
        "adjoint_batch",
        "strict_forward",
        "residual_generation"
    ]
}

history = cfg.get("checkpoint_history", [])
history.append({
    "created": datetime.now().isoformat(),
    "event": "config_updated_after_iter008_acceptance",
    "previous_current_verified_checkpoint": old_checkpoint,
})
cfg["checkpoint_history"] = history

config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

print("Config updated after iter008 acceptance")
print("=======================================")
print(f"config = {config_path}")
print(f"backup = {backup}")
print("")
print("current_verified_checkpoint:")
print(json.dumps(cfg["current_verified_checkpoint"], indent=2))
print("")
print("RESULT = PASS")
