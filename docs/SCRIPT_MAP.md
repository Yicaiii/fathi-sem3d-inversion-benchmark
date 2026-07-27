# Script Map

## Purpose

This document distinguishes the public entry points, internal workflow stages,
validation utilities, reporting tools, experimental regularization modules, and
historical scripts contained in the repository.

The objective is to expose a small and stable user interface while preserving
the existing numerical implementation during the refactoring phase.

---

## 1. Public entry points

These are the scripts that a user should normally execute directly.

| Script | Role | Current status |
|---|---|---|
| `scripts/fathi_benchmark/create_iteration_context_generic.py` | Creates one iteration context from a configuration | Active public entry point |
| `scripts/fathi_benchmark/run_iteration_full_context.py` | Runs the workflow stages from a complete iteration context | Active public entry point |
| `scripts/fathi_benchmark/run_iteration.py` | Earlier iteration runner | Compatibility entry point; review before removal |

The target architecture is to keep only one or two public commands.

---

## 2. Active workflow stage wrappers

These scripts currently coordinate the major stages of one inversion iteration.

| Script | Stage |
|---|---|
| `run_task0_prerequisites.py` | Preflight and prerequisite checks |
| `run_task1b_prepare_strict_forward.py` | Strict forward preparation |
| `run_task1_strict_forward.py` | SEM3D forward execution |
| `run_task2_residual_generation.py` | Residual generation |
| `run_task2b_prepare_adjoint.py` | Adjoint workspace preparation |
| `run_task2c_adjoint_batch.py` | Adjoint batch execution |
| `run_task3_gradient.py` | RHS and material-gradient stage |
| `run_task4_candidates.py` | Candidate generation |
| `run_task5_candidate.py` | Candidate evaluation |

During refactoring, these scripts should progressively become thin wrappers
calling reusable functions from a Python package.

---

## 3. Active numerical modules

The following files implement reusable workflow operations.

### Forward, residual and adjoint

- `scripts/iteration_engine/forward_batch_generic.py`
- `scripts/iteration_engine/residual_generic.py`
- `scripts/iteration_engine/prepare_adjoint_generic.py`
- `scripts/iteration_engine/run_adjoint_generic.py`
- `scripts/iteration_engine/execute_next_adjoint_batch.py`
- `scripts/iteration_engine/execute_adjoint_until_stop.py`

### RHS and Mtilde

- `scripts/iteration_engine/build_rhs_manifests_generic_v2.py`
- `scripts/iteration_engine/assemble_rhs_total_generic.py`
- `scripts/iteration_engine/solve_mtilde_generic.py`
- `scripts/iteration_engine/generate_candidates_from_mtilde_gradient.py`

### Candidate evaluation

- `scripts/iteration_engine/prepare_candidate_forward_workspaces.py`
- `scripts/iteration_engine/run_candidate_forward.py`
- `scripts/iteration_engine/compute_candidate_misfit_v2.py`
- `scripts/iteration_engine/accept_candidate_if_descent_v2.py`

These modules are active, but they should eventually be moved from executable
scripts into importable library functions.

---

## 4. Scientific guards and fail-fast validation

These checks are scientifically important and should be retained.

- `scripts/fathi_benchmark/enforce_forward_operator.py`
- `scripts/fathi_benchmark/audit_transition_completion.py`
- `scripts/iteration_engine/audit_accepted_state.py`
- `scripts/iteration_engine/audit_adjoint_complete.py`
- `scripts/iteration_engine/audit_benchmark_inventory_fast.py`
- `scripts/iteration_engine/audit_candidate_inputs.py`
- `scripts/iteration_engine/audit_candidates_generic.py`
- `scripts/iteration_engine/audit_mtilde_outputs_generic.py`
- `scripts/iteration_engine/audit_rhs_source_scripts.py`
- `scripts/iteration_engine/inspect_capteurs_h5_structure.py`
- `scripts/iteration_engine/inspect_capteurs_structure.py`

Target design:

1. validate the configuration before an expensive SEM3D run;
2. stop immediately when a critical inconsistency is detected;
3. preserve lightweight validation evidence in each execution stage.

---

## 5. TV regularization modules

The TV implementation is stored in `scripts/regularization/`.

| Step | Script | Role |
|---|---|---|
| 1 | `01_audit_q1_material_mesh.py` | Audit the material-grid ordering |
| 2 | `02_compute_tv_q1_full_grid.py` | Compute the full-grid TV contribution |
| 3 | `03_test_tv_q1.py` | Validate the TV discretization |
| 4 | `04_restrict_tv_rhs_to_active.py` | Restrict the TV RHS to active indices |
| 5 | `05_assemble_data_tv_rhs.py` | Assemble data and TV contributions |
| 6 | `06_solve_mtilde_data_tv_rhs.py` | Solve the combined control problem |
| 7 | `07_diagnose_tv_weight.py` | Diagnose regularization scaling |
| 8 | `08_generate_tv_candidates_from_mtilde_gradient.py` | Generate regularized candidates |

Status: implemented at the RHS, Mtilde and candidate-generation levels.
The complete nine-source end-to-end numerical validation is still pending.

---

## 6. Reporting and status generation

These scripts generate reports or dashboards and are not numerical entry points.

- `write_benchmark_dashboard.py`
- `write_iteration_stage_report.py`
- `write_iteration_stage_report_v2.py`
- `write_official_benchmark_status.py`
- `write_resume_plan.py`
- `refresh_benchmark_status.py`
- `write_engine_status_final.py`
- `write_F3_adjoint_batch_generic_status.py`
- `write_F4_strict_forward_generic_status.py`
- `write_F5_prepare_strict_forward_generic_status.py`
- `write_G_local_iteration_success_status.py`

These should later be consolidated into one reporting module.

---

## 7. Candidate legacy or duplicated files

The following files are not deleted yet. They require comparison with the
current active implementation.

### Generated from historical scripts

- `scripts/fathi_benchmark/generic_from_legacy/450B_select_strict_forward_full_template_generic.py`
- `scripts/fathi_benchmark/generic_from_legacy/450C_prepare_strict_full_forward_run_generic.py`
- `scripts/fathi_benchmark/generic_from_legacy/454A_compute_strict_forward_residual_manifest_generic.py`
- `scripts/fathi_benchmark/generic_from_legacy/454B_build_strict_residual_timeseries_h5_generic.py`
- `scripts/fathi_benchmark/generic_from_legacy/455A_extract_old_adjoint_source_format_generic.py`
- `scripts/fathi_benchmark/generic_from_legacy/455B_prepare_strict_adjoint_batches_from_residual_generic.py`
- `scripts/fathi_benchmark/generic_from_legacy/455C_audit_strict_adjoint_batches_generic.py`

### Versioned duplicates requiring review

- `build_rhs_manifests_generic.py`
- `build_rhs_manifests_generic_v2.py`
- `compute_candidate_misfit.py`
- `compute_candidate_misfit_v2.py`
- `write_iteration_stage_report.py`
- `write_iteration_stage_report_v2.py`

### Iteration-specific or one-off utilities

- `scripts/fathi_benchmark/update_config_after_iter008.py`
- `scripts/fathi_benchmark/patch_root_portability.py`
- `scripts/fathi_benchmark/repair_residual_summary_from_h5.py`
- `scripts/longterm/424B_compute_rhs_component_from_traces.py`

No file in this section should be removed until its replacement and call sites
have been verified.

---

## 8. Target repository interface

The final user-facing interface should be approximately:

```text
scripts/
├── run_benchmark.py
├── run_stage.py
└── clean_runtime.py

```

All numerical operations should be exposed as reusable functions in a package,
while validation should be called automatically by the workflow.
