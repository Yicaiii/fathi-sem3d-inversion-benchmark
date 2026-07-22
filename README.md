# Fathi SEM3D Inversion Benchmark and PyMoniK Integration



---

## 1. Project Overview

This repository contains a reusable benchmark workflow for SEM3D-based elastic parameter inversion.

The benchmark is designed around a generic iteration pattern:

```text
state_k
  -> run inversion transition k -> k+1
  -> state_{k+1}
```

Each iteration takes the accepted material model from the previous iteration, runs a forward simulation, compares synthetic receiver traces with observed receiver traces, prepares adjoint sources, runs adjoint simulations, computes the gradient-like control update through an Mtilde solve, generates candidate material models, evaluates candidate misfit, and accepts the candidate only if the receiver misfit decreases.

The current validated transition is:

```text
iter_008 -> iter_009
```

The accepted output is:

```text
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
data/inversion_linear/iter_009/accepted
```

---

## 2. Scientific Logic

The inversion follows the structure of the Fathi-style PDE-constrained optimization workflow.

At a high level:

```text
minimize receiver displacement misfit
subject to SEM3D forward elastic wave equation
```

The key idea is not to directly compare full-grid fields. Instead, the workflow compares receiver traces:

```text
synthetic receiver traces - observed receiver traces
```

This residual is then used to build adjoint sources. The adjoint fields are used together with the forward fields to assemble RHS terms for the material update. The update is obtained through an Mtilde / mass-like control solve, not by simply dividing by a scalar mass.

The control parameters updated in each accepted iteration are mainly:

```text
lambda
mu
kappa
```

The density is currently kept fixed in the tested workflow:

```text
density = 2000
```

In the current implementation, `kappa` is derived consistently from the elastic parameters. The accepted candidate writes new material HDF5 files:

```text
Mat_0_Kappa.h5
Mat_0_Mu.h5
Mat_0_Density.h5
```

---

## 3. Validated Result

The local full iteration test has successfully completed:

```text
transition = iter_008_to_iter_009
candidate  = line_search_neg_mtilde_1p00MPa
```

Misfit comparison:

```text
parent_J    = 3.8263972312235541e-19
candidate_J = 3.8259162917013906e-19
delta_J     = -4.8093952216354196e-23
descent     = True
accepted    = True
```

Final audit:

```text
RESULT = PASS_COMPLETE
```

Important accepted outputs:

```text
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Kappa.h5
data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Mu.h5
data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Density.h5
```

---

## 4. Repository Layout

Recommended repository structure:

```text
sem3d_fathi_clean/
├── scripts/
│   ├── fathi_benchmark/
│   │   ├── create_iteration_context_generic.py
│   │   ├── run_iteration_full_context.py
│   │   ├── run_task1b_prepare_strict_forward.py
│   │   ├── run_task1_strict_forward.py
│   │   ├── run_task2_residual_generation.py
│   │   ├── run_task2b_prepare_adjoint.py
│   │   ├── run_task2c_adjoint_batch.py
│   │   ├── run_task3_gradient.py
│   │   ├── run_task4_candidates.py
│   │   ├── run_task5_candidate.py
│   │   └── audit_transition_completion.py
│   │
│   ├── fathi_benchmark/generic_from_legacy/
│   │   ├── 454A_compute_strict_forward_residual_manifest_generic.py
│   │   ├── 454B_build_strict_residual_timeseries_h5_generic.py
│   │   ├── 455A_extract_old_adjoint_source_format_generic.py
│   │   ├── 455B_prepare_strict_adjoint_batches_from_residual_generic.py
│   │   ├── 455C_audit_strict_adjoint_batches_generic.py
│   │   └── 450B_select_strict_forward_full_template_generic.py
│   │
│   ├── iteration_engine/
│   │   ├── build_rhs_manifests_generic_v2.py
│   │   ├── assemble_rhs_total_generic.py
│   │   ├── solve_mtilde_generic.py
│   │   ├── audit_candidate_inputs.py
│   │   ├── generate_candidates_from_mtilde_gradient.py
│   │   ├── audit_candidates_generic.py
│   │   ├── prepare_candidate_forward_workspaces.py
│   │   ├── run_candidate_forward.py
│   │   ├── compute_candidate_misfit_v2.py
│   │   └── accept_candidate_if_descent_v2.py
│   │
│   └── longterm/
│       └── 424B_compute_rhs_component_from_traces.py
│
├── results/
│   └── fathi_loop_v2/
│       ├── states_corrected/
│       ├── iter_007_to_iter_008/
│       ├── iter_008_to_iter_009/
│       └── iter_009_to_iter_010/
│
├── data/
│   ├── 60_true_layered_h5_T045/
│   │   └── traces/
│   └── inversion_linear/
│       ├── iter_008/
│       └── iter_009/
│
└── benchmark_fathi_strict/
    └── reports/
```

Important note:

```text
Large runtime data should usually NOT be committed to GitHub.
```

Do not commit huge folders such as:

```text
data/inversion_linear/*/adjoint_full_grid_batches/*/*/traces
data/inversion_linear/*/forward_dudx_mgcap_full_batches/*/traces
data/inversion_linear/*/candidate_forward_workspaces/*/traces
results/fathi_loop_v2/*/residual_sources/*.h5
```

Commit scripts, configs, small summaries, reports, and README files instead.

---

## 5. Main Inputs and Outputs

### 5.1 Inputs

#### Accepted parent state

For transition `iter_k -> iter_{k+1}`:

```text
results/fathi_loop_v2/states_corrected/iter_{k:03d}_state_v2_corrected.npz
data/inversion_linear/iter_{k:03d}/accepted
```

For the validated transition:

```text
results/fathi_loop_v2/states_corrected/iter_008_state_v2_corrected.npz
data/inversion_linear/iter_008/accepted
```

#### Observed receiver traces

```text
data/60_true_layered_h5_T045/traces
```

These traces are treated as the observed data. The true model is used only for generating observed traces and validation, not for directly updating the inversion parameters.

#### SEM3D executable

```text
/home/crellamaybe/SEM/build/SEM3D/sem3d.exe
```

#### Iteration context

For `iter_008 -> iter_009`:

```text
results/fathi_loop_v2/iter_008_to_iter_009/iter_008_to_iter_009_iteration_context.json
```

The context file stores the paths and transition metadata needed by every task.

---

### 5.2 Outputs

#### Strict forward traces

```text
data/inversion_linear/iter_009/forward_dudx_mgcap_full_batches/strict_full_forward_000/traces
```

#### Residual time series

```text
results/fathi_loop_v2/iter_008_to_iter_009/residual_sources/454B_strict_residual_timeseries.h5
results/fathi_loop_v2/iter_008_to_iter_009/residual_sources/454B_strict_residual_timeseries_summary.txt
```

#### Adjoint batches

```text
data/inversion_linear/iter_009/adjoint_full_grid_batches/x/batch_000
...
data/inversion_linear/iter_009/adjoint_full_grid_batches/z/batch_009
```

There are 30 adjoint batches in total:

```text
x: batch_000 ... batch_009
y: batch_000 ... batch_009
z: batch_000 ... batch_009
```

#### RHS and Mtilde solve

```text
results/fathi_loop_v2/iter_008_to_iter_009/component_rhs
results/fathi_loop_v2/iter_008_to_iter_009/mtilde_solve
```

Important Mtilde outputs:

```text
g_lambda_mtilde_q1_interior_solve_rhs_total.npy
g_mu_mtilde_q1_interior_solve_rhs_total.npy
g_mtilde_q1_interior_solve_rhs_total_coords.npy
Mtilde_q1_consistent_interior_38440_indices.npy
```

#### Candidates

```text
results/fathi_loop_v2/iter_008_to_iter_009/candidates
data/inversion_linear/iter_009/candidate_forward_workspaces
```

Validated candidate:

```text
line_search_neg_mtilde_1p00MPa
```

Other generated candidates:

```text
line_search_neg_mtilde_0p10MPa
line_search_neg_mtilde_0p25MPa
line_search_neg_mtilde_0p50MPa
line_search_neg_mtilde_1p00MPa
```

#### Accepted output

```text
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
data/inversion_linear/iter_009/accepted
```

---

## 6. One Full Iteration Workflow

The full local workflow is:

```text
create context
  -> prepare strict forward
  -> run strict forward
  -> generate residual
  -> prepare adjoint batches
  -> run 30 adjoint batches
  -> compute gradient / Mtilde solve
  -> generate candidates
  -> run candidate forward + misfit + accept
  -> audit transition
```

### Step 0. Activate environment

```bash
cd ~/sem3d_fathi_clean
source .venv/bin/activate
```

### Step 1. Create context

For `iter_008 -> iter_009`:

```bash
python3 scripts/fathi_benchmark/create_iteration_context_generic.py \
  --iter-k 8 \
  --write
```

Set context variable:

```bash
CTX8=results/fathi_loop_v2/iter_008_to_iter_009/iter_008_to_iter_009_iteration_context.json
```

### Step 2. Prepare strict forward workspace

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage prepare_strict_forward
```

Output:

```text
data/inversion_linear/iter_009/forward_dudx_mgcap_full_batches/strict_full_forward_000
```

### Step 3. Run strict forward SEM3D

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage strict_forward \
  --execute-heavy
```

Expected output:

```text
data/inversion_linear/iter_009/forward_dudx_mgcap_full_batches/strict_full_forward_000/traces
```

### Step 4. Generate residual

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage residual_generation
```

Expected output:

```text
results/fathi_loop_v2/iter_008_to_iter_009/residual_sources/454B_strict_residual_timeseries.h5
```

### Step 5. Prepare adjoint batches

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage prepare_adjoint
```

Expected output:

```text
30 adjoint workspaces
```

### Step 6. Run all adjoint batches

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage adjoint_all \
  --execute-heavy
```

This runs:

```text
x/batch_000 ... x/batch_009
y/batch_000 ... y/batch_009
z/batch_000 ... z/batch_009
```

Expected check:

```text
ready_batches = 30 / 30
missing_batches = 0 / 30
```

### Step 7. Compute gradient and Mtilde update

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage gradient
```

This step computes RHS terms and solves Mtilde.

Expected output:

```text
results/fathi_loop_v2/iter_008_to_iter_009/mtilde_solve
```

### Step 8. Generate candidates

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage candidates
```

Expected output:

```text
results/fathi_loop_v2/iter_008_to_iter_009/candidates
data/inversion_linear/iter_009/candidate_forward_workspaces
```

### Step 9. Run Task5 candidate forward / misfit / accept

```bash
time python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX8" \
  --stage task5 \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute-heavy \
  --allow-mutate
```

This step does:

```text
Task 5A: candidate forward SEM3D
Task 5B: candidate misfit
Task 5C: accept if descent
```

Expected accepted output:

```text
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
data/inversion_linear/iter_009/accepted
```

### Step 10. Audit transition

```bash
python3 scripts/fathi_benchmark/audit_transition_completion.py --iter-k 8

cat benchmark_fathi_strict/reports/audit/iter_008_to_iter_009_completion_audit.txt
```

Expected result:

```text
RESULT = PASS_COMPLETE
```

---

## 8. Does Each Iteration Update Mu and Other Parameters?

Yes.

Each accepted iteration updates the elastic material model.

The main updated arrays are:

```text
lambda
mu
kappa
```

The accepted state is written to:

```text
results/fathi_loop_v2/states_corrected/iter_{k+1:03d}_state_v2_corrected.npz
```

The accepted SEM3D material files are written to:

```text
data/inversion_linear/iter_{k+1:03d}/accepted/mat/h5/Mat_0_Kappa.h5
data/inversion_linear/iter_{k+1:03d}/accepted/mat/h5/Mat_0_Mu.h5
data/inversion_linear/iter_{k+1:03d}/accepted/mat/h5/Mat_0_Density.h5
```

In the current tested workflow:

```text
mu changes
lambda changes
kappa changes
density remains fixed
```

The update is accepted only when:

```text
J_candidate < J_parent
```

---

## 9. PyMoniK / ArmoniK Integration Plan

The local benchmark is already decomposed into task-like stages. Therefore, the PyMoniK / ArmoniK integration should not rewrite the numerical algorithm. It should only replace local sequential execution with scheduled task execution.

### 9.1 DAG Structure

The task graph is:

```text
prepare_strict_forward
  -> strict_forward
    -> residual_generation
      -> prepare_adjoint
        -> adjoint_x_batch_000
        -> adjoint_x_batch_001
        -> ...
        -> adjoint_z_batch_009
          -> gradient
            -> candidates
              -> task5_candidate
                -> audit_transition
```

The most useful parallel part is:

```text
30 adjoint batches
```

Later, multiple candidate forward runs can also be parallelized:

```text
line_search_neg_mtilde_0p10MPa
line_search_neg_mtilde_0p25MPa
line_search_neg_mtilde_0p50MPa
line_search_neg_mtilde_1p00MPa
```

### 9.2 Payload Design

PyMoniK tasks should pass small JSON payloads only.

Example payload for one adjoint batch:

```json
{
  "task_type": "adjoint_batch",
  "iter_k": 9,
  "context": "results/fathi_loop_v2/iter_009_to_iter_010/iter_009_to_iter_010_iteration_context.json",
  "component": "x",
  "batch": "batch_000",
  "np": 12,
  "execute": true
}
```

Example payload for candidate stage:

```json
{
  "task_type": "task5_candidate",
  "iter_k": 9,
  "context": "results/fathi_loop_v2/iter_009_to_iter_010/iter_009_to_iter_010_iteration_context.json",
  "candidate": "line_search_neg_mtilde_1p00MPa",
  "np": 12,
  "execute_heavy": true,
  "allow_mutate": true
}
```

### 9.3 Shared Filesystem Requirement

Large SEM3D files should not be passed through PyMoniK payloads.

They should remain on a shared filesystem:

```text
~/sem3d_fathi_clean/data
~/sem3d_fathi_clean/results
~/sem3d_fathi_clean/benchmark_fathi_strict/reports
```

The payload only tells the worker where to find the context and what task to run.

### 9.4 Worker Entry Logic

A PyMoniK worker can call the same local scripts:

```bash
python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage adjoint_sample \
  --component x \
  --batch batch_000 \
  --execute-heavy
```

or directly:

```bash
python3 scripts/fathi_benchmark/run_task2c_adjoint_batch.py \
  --context "$CTX" \
  --component x \
  --batch batch_000 \
  --np 12 \
  --execute
```

### 9.5 Recommended Integration Steps

#### Phase P0. Local frozen benchmark

Already completed:

```text
iter_008 -> iter_009 = PASS_COMPLETE
```

#### Phase P1. Create next context

```bash
python3 scripts/fathi_benchmark/create_iteration_context_generic.py \
  --iter-k 9 \
  --write
```

#### Phase P2. Dry-run DAG

Create PyMoniK tasks but do not execute heavy SEM3D yet.

Expected result:

```text
all payloads valid
all dependencies valid
all planned commands correct
```

#### Phase P3. Execute non-heavy stages

Run:

```text
prepare_strict_forward
residual_generation
prepare_adjoint
gradient
candidates
audit
```

only when their dependencies are satisfied.

#### Phase P4. Execute heavy stages

Run SEM3D tasks:

```text
strict_forward
30 adjoint_batch tasks
candidate_forward tasks
```

#### Phase P5. Acceptance and audit

Run:

```text
candidate_misfit
accept_candidate_if_descent
audit_transition_completion
```

---

## 10. Safety Rules

### Do not run mutation stages without explicit confirmation

The acceptance stage changes the trusted next state:

```text
results/fathi_loop_v2/states_corrected/iter_{k+1}_state_v2_corrected.npz
data/inversion_linear/iter_{k+1}/accepted
```

Therefore, local execution requires:

```bash
--allow-mutate
```

### Do not run SEM3D accidentally

Heavy SEM3D execution requires:

```bash
--execute-heavy
```

### Do not delete current accepted results

Do not delete:

```text
data/inversion_linear/iter_009/accepted
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
results/fathi_loop_v2/iter_008_to_iter_009
```

These are the proof that the benchmark completed successfully.

---

## 11. Suggested .gitignore

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
env/

# Logs
*.log
nohup.out

# Large SEM3D runtime outputs
data/inversion_linear/*/forward_dudx_mgcap_full_batches/*/traces/
data/inversion_linear/*/adjoint_full_grid_batches/*/*/traces/
data/inversion_linear/*/candidate_forward_workspaces/*/traces/
data/inversion_linear/*/*/prot/
data/inversion_linear/*/*/Protection_*/

# Large HDF5 / NPZ runtime files
*.h5
*.hdf5
*.npz
*.npy

# Keep small summaries and configs if needed
!**/*summary*.txt
!**/*context*.json
!**/*context*.txt
!README.md
!README*.txt
```

If you need to publish small example files, place them under:

```text
examples/
```

and explicitly unignore them.

---

## 12. Minimal Reproduction Command List

For a new transition `iter_k -> iter_{k+1}`:

```bash
cd ~/sem3d_fathi_clean
source .venv/bin/activate

python3 scripts/fathi_benchmark/create_iteration_context_generic.py \
  --iter-k 9 \
  --write

CTX=results/fathi_loop_v2/iter_009_to_iter_010/iter_009_to_iter_010_iteration_context.json

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage prepare_strict_forward

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage strict_forward \
  --execute-heavy

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage residual_generation

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage prepare_adjoint

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage adjoint_all \
  --execute-heavy

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage gradient

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage candidates

python3 scripts/fathi_benchmark/run_iteration_full_context.py \
  --context "$CTX" \
  --stage task5 \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute-heavy \
  --allow-mutate

python3 scripts/fathi_benchmark/audit_transition_completion.py \
  --iter-k 9
```

---

## 13. Current Status

```text
Local benchmark transition tested:
  iter_008 -> iter_009

Status:
  PASS_COMPLETE

Candidate:
  line_search_neg_mtilde_1p00MPa

Result:
  candidate accepted
  misfit decreased
  iter_009 accepted state generated

Next engineering goal:
  connect the validated context-driven task graph to PyMoniK / ArmoniK
```

---

## Canonical Fathi 80 MPa initialization

The strict Fathi validation starts from a homogeneous model with
lambda = mu = 80 MPa, Kappa = 133.333333 MPa and density = 2000
kg/m3.

The canonical forward operator contains nine vertical impulse sources
on a 3 x 3 surface grid. The observed objective uses 225 physical
receivers, while the forward/adjoint gradient workflow uses 38,440
full-grid stations.

Strict-forward preparation automatically preserves the source
operator from the accepted parent model. A full-grid receiver
template is never allowed to replace the nine-source operator.

See:

- `docs/FATHI80_INITIAL.md`
- `docs/TV_REGULARIZATION_WORKFLOW.md`

Large SEM3D traces, snapshots and inversion workspaces are local
runtime data and are not stored in Git.
