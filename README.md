# Fathi SEM3D Inversion Benchmark and PyMoniK Integration



---

## Start here

This repository now exposes two different levels of execution. They are not two
inversion algorithms.

### A. From-scratch benchmark bootstrap

Use the bootstrap entry point on a new machine or in a clean output directory:

```bash
python -m scripts.bootstrap.bootstrap_fathi_benchmark \
  --output-root /path/to/runtime/fathi_reduced_3x3_12p5
```

The command is plan-only by default. It prints the full workflow without creating
files or launching SEM3D.

A short real smoke bootstrap is:

```bash
export FATHI_BENCHMARK_ROOT=/path/to/fathi-sem3d-inversion-benchmark
export SEM3D_MESHER=/path/to/SEM/build/MESH/mesher
export SEM3D_EXE=/path/to/SEM/build/SEM3D/sem3d.exe

cd "$FATHI_BENCHMARK_ROOT"
source .venv/bin/activate

python -m scripts.bootstrap.bootstrap_fathi_benchmark \
  --output-root /tmp/fathi_reduced_3x3_12p5_smoke \
  --smoke-seconds 0.012 \
  --timeout-seconds 900 \
  --execute
```

A full `0.1 s` bootstrap uses the same command without `--smoke-seconds`:

```bash
python -m scripts.bootstrap.bootstrap_fathi_benchmark \
  --output-root /path/to/runtime/fathi_reduced_3x3_12p5 \
  --timeout-seconds 900 \
  --execute
```

The bootstrap builds and audits both benchmark workspaces:

```text
true_layered
  -> generate SEM3D inputs
  -> run mesher
  -> validate 12 mesh partitions
  -> run solver
  -> validate observed traces

initial_homogeneous
  -> generate SEM3D inputs
  -> run mesher
  -> validate 12 mesh partitions
  -> run solver
  -> validate predicted traces
```

The machine-readable benchmark profile is:

```text
configs/fathi_reduced_3x3_12p5.json
```

It defines the reduced Fathi-style operator used here: 9 vertical point sources,
225 physical receivers, 12 mesh partitions, a layered synthetic truth, and a
homogeneous 80 MPa initial model.

### B. Iteration workflow interfaces

The repository currently contains two iteration orchestrators with different scopes.

#### Resumed public workflow

```bash
python -m scripts.fathi_benchmark.run_iteration \
  --iter-k K \
  --stage plan
```

`run_iteration.py` is the canonical resumed-workflow interface. It assumes that the
strict forward, residual, and adjoint prerequisites already exist. Its `all` stage
covers:

```text
prerequisites
  -> gradient and Mtilde solve
  -> candidate generation
  -> candidate forward, misfit, and acceptance
  -> status
```

#### Full local transition workflow

```bash
python scripts/fathi_benchmark/run_iteration_full_context.py \
  --iter-k K \
  --stage all_full \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute-heavy \
  --allow-mutate
```

`run_iteration_full_context.py` is the current complete local transition runner. It
covers:

```text
accepted state_k
  -> strict forward
  -> residual
  -> 30 adjoint batches
  -> gradient and Mtilde solve
  -> candidates
  -> candidate forward and acceptance
  -> accepted state_{k+1}
```

The long-term public API is intended to remain `run_iteration.py`, with the
full-context runner retained as an internal implementation layer. That interface
consolidation is not yet complete.

All runtime roots may be supplied through configuration and
`FATHI_BENCHMARK_ROOT`. Large generated workspaces and traces are runtime data and
must not be committed to Git.

## C. Validated Mini End-to-End Transition: `iter_000 -> iter_001`

The repository also provides a compact, fully validated mini transition for demonstrating the complete inversion logic without launching the full 38,440-control-point and 30-adjoint-batch workflow.

This mini workflow is not a different inversion algorithm. It is a reduced execution profile of the same sequence:

```text
accepted iter_000 model
  -> predicted-versus-observed receiver residual
  -> adjoint x / y / z
  -> forward-adjoint RHS assembly
  -> Mtilde control solve
  -> candidate material models
  -> candidate forward simulations
  -> receiver-misfit comparison
  -> accepted iter_001 model
```

### Mini profile

The validated mini profile uses:

```text
physical receivers used for residual and misfit = 49
control stations used for forward/adjoint DUDX = 3,600
control grid = 15 x 15 x 16
adjoint executions = 3
  x: batch_000
  y: batch_000
  z: batch_000
candidate step sizes = 0.05 MPa and 0.10 MPa
MPI ranks per SEM3D execution = 12
```

The two main configuration files are:

```text
configs/fathi_mini_e2e_3600.json
configs/fathi_mini_strict_15x15x16.json
```

The mini transition reuses the bootstrap trace datasets:

```text
observed traces:
data/bootstrap/fathi_reduced_3x3_12p5/true_layered/traces

parent predicted traces:
data/bootstrap/fathi_reduced_3x3_12p5/initial_homogeneous/traces
```

The 49 physical receivers are used to define the receiver residual and objective function. The 3,600 control stations are a separate operator used to store forward and adjoint `DUDX_1 ... DUDX_9` fields for material-gradient assembly.

### Environment

```bash
cd /path/to/fathi-sem3d-inversion-benchmark
source /path/to/venv/bin/activate

export FATHI_BENCHMARK_ROOT="$PWD"
export SEM3D_BIN=/path/to/SEM/build/SEM3D/sem3d.exe
```

A local MPI runtime must also be available:

```bash
command -v mpirun
test -x "$SEM3D_BIN"
```

### Regression tests

Before running the transition:

```bash
python -m pytest -q
```

Validated result:

```text
40 passed, 4 subtests passed
```

### Stage 1. Preflight

The preflight checks the accepted parent model, predicted and observed trace directories, strict mini forward data, the 3,600 control stations, the 49 selected physical receivers, the SEM3D executable, and the available Mtilde matrix.

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage preflight
```

Expected marker:

```text
RESULT = PASS_MINI_E2E_PREFLIGHT
```

### Stage 2. Receiver residual

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage residual
```

This stage computes:

```text
residual = parent predicted displacement - observed displacement
```

It writes a machine-readable HDF5 residual package for 49 physical receivers, including the forward-time residual and time-reversed adjoint source signals.

Validated parent objective:

```text
parent_J = 9.2275437349432100e-22
```

Expected marker:

```text
RESULT = PASS
```

### Stage 3. Prepare adjoint workspaces

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage prepare-adjoint
```

This creates:

```text
data/inversion_tv_fathi80_mini3600/iter_001/
  adjoint_full_grid_batches/
    x/batch_000
    y/batch_000
    z/batch_000
```

Each workspace contains:

```text
49 non-zero adjoint source files
3,600 control stations
SEM3D mesh link
material inputs
input.spec
```

Expected marker:

```text
RESULT = PASS_MINI_ADJOINT_PREPARATION
```

### Stage 4. Run adjoint x, y, and z

These are heavy SEM3D stages.

The mini adapter delegates execution to the repository's existing adjoint runner:

```text
scripts/fathi_benchmark/run_task2c_adjoint_batch.py
```

Run all three components:

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage adjoint-all \
  --np 12
```

Equivalent component-level commands are:

```bash
python -m scripts.mini_e2e.run_adjoint \
  --config configs/fathi_mini_e2e_3600.json \
  --component x \
  --np 12

python -m scripts.mini_e2e.run_adjoint \
  --config configs/fathi_mini_e2e_3600.json \
  --component y \
  --np 12

python -m scripts.mini_e2e.run_adjoint \
  --config configs/fathi_mini_e2e_3600.json \
  --component z \
  --np 12
```

Each completed component must satisfy:

```text
returncode = 0
fin_sem = 1
trace files = 6
unique control positions = 3,600
DUDX_1 ... DUDX_9 available
```

Expected marker:

```text
RESULT = PASS_MINI_ADJOINT_RUNS
```

### Stage 5. Assemble the material RHS and solve Mtilde

This stage does not launch SEM3D.

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage gradient
```

It performs:

```text
build forward and adjoint control manifests
  -> compute RHS_x
  -> compute RHS_y
  -> compute RHS_z
  -> assemble RHS_total
  -> solve the 3,600 x 3,600 Mtilde subset system
```

Validated checks:

```text
forward manifest rows = 3,600
adjoint x manifest rows = 3,600
adjoint y manifest rows = 3,600
adjoint z manifest rows = 3,600

relative Mtilde residual for lambda = 2.9216733343895482e-16
relative Mtilde residual for mu     = 2.8947877553402663e-16
```

Expected markers:

```text
RESULT = PASS_MINI_TRACE_MANIFESTS
RESULT = PASS_MINI_RHS_COMPONENT
RESULT = PASS_MINI_RHS_TOTAL
RESULT = PASS_MINI_MTILDE_SOLVE
```

### Stage 6. Generate candidate material models

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage candidates
```

The validated candidate names are:

```text
mini_line_search_pos_mtilde_0p05MPa
mini_line_search_pos_mtilde_0p10MPa
```

Under the residual and adjoint sign convention used by this mini implementation, the tested descent direction is the positive normalized Mtilde control direction.

This sign convention is recorded explicitly in the candidate metadata and should not be inferred only from the candidate directory name.

Expected markers:

```text
RESULT = PASS_MINI_CANDIDATE_GENERATION
RESULT = PASS_MINI_CANDIDATE_WORKSPACES
```

### Stage 7. Run candidate forward simulations

This stage launches two heavy SEM3D forward simulations, one for each candidate.

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage candidate-forward \
  --np 12
```

Each candidate forward uses the same 49 physical receivers as the parent objective.

Expected marker:

```text
RESULT = PASS_MINI_CANDIDATE_RUNS
```

### Stage 8. Misfit comparison and acceptance

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage accept
```

Validated result:

```text
parent_J = 9.2275437349432100e-22

mini_line_search_pos_mtilde_0p05MPa
  J       = 9.2275429553166404e-22
  delta_J = -7.7962656954727976e-29
  descent = True

mini_line_search_pos_mtilde_0p10MPa
  J       = 9.2275421771626136e-22
  delta_J = -1.5577805963808890e-28
  descent = True

best candidate = mini_line_search_pos_mtilde_0p10MPa
accepted       = True
```

The acceptance rule remains:

```text
J_candidate < J_parent
```

Expected marker:

```text
RESULT = PASS_ACCEPTED
```

Do not use `--allow-non-descent` for a formal validated transition.

### Stage 9. Final status

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage status
```

The accepted output is written under:

```text
data/inversion_tv_fathi80_mini3600/iter_001/accepted
```

The corrected accepted state is written to:

```text
results/fathi_loop_tv_fathi80_mini3600/
  states_corrected/
  iter_001_state_v2_corrected.npz
```

The final machine-readable status can be saved with:

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --stage status \
  > reports/mini_iter000_to_iter001_final_status.txt
```

Final validated marker:

```text
RESULT = COMPLETE_MINI_ITER000_TO_ITER001
```

### Compact execution summary

After the required bootstrap and strict mini forward prerequisites exist, the transition can be executed stage by stage with:

```bash
python -m scripts.mini_e2e.run_mini_e2e --stage preflight
python -m scripts.mini_e2e.run_mini_e2e --stage residual
python -m scripts.mini_e2e.run_mini_e2e --stage prepare-adjoint
python -m scripts.mini_e2e.run_mini_e2e --stage adjoint-all --np 12
python -m scripts.mini_e2e.run_mini_e2e --stage gradient
python -m scripts.mini_e2e.run_mini_e2e --stage candidates
python -m scripts.mini_e2e.run_mini_e2e --stage candidate-forward --np 12
python -m scripts.mini_e2e.run_mini_e2e --stage accept
python -m scripts.mini_e2e.run_mini_e2e --stage status
```

### Heavy and non-heavy stages

```text
No SEM3D:
  preflight
  residual
  prepare-adjoint
  gradient
  candidates
  accept
  status

SEM3D:
  adjoint-x
  adjoint-y
  adjoint-z
  candidate forward: 0.05 MPa
  candidate forward: 0.10 MPa
```

The strict mini forward that supplies the 3,600 forward-control traces is also a heavy SEM3D prerequisite.

### Runtime data policy

Do not commit generated runtime data such as:

```text
data/inversion_tv_fathi80_mini3600/
results/fathi_loop_tv_fathi80_mini3600/
_quarantine/
traces/
prot/
res/
mirror/
large HDF5 and NPZ runtime outputs
```

Commit only the reproducible code, configuration, tests, small status summaries, and documentation required to recreate and audit the workflow.

## 1. Project Overview

This repository contains a reusable benchmark workflow for SEM3D-based elastic parameter inversion.

The benchmark is designed around a generic iteration pattern:

```text
state_k
  -> run inversion transition k -> k+1
  -> state_{k+1}
```

Each iteration takes the accepted material model from the previous iteration, runs a forward simulation, compares synthetic receiver traces with observed receiver traces, prepares adjoint sources, runs adjoint simulations, computes the gradient-like control update through an Mtilde solve, generates candidate material models, evaluates candidate misfit, and accepts the candidate only if the receiver misfit decreases.

Iteration numbering is generic: every transition is expressed as
`iter_k -> iter_{k+1}`.

The repository also preserves one historical full-transition validation:

```text
iter_008 -> iter_009 = PASS_COMPLETE
```

That transition is evidence for the tested local workflow, not a hard-coded runtime
requirement. New runs should use `--iter-k K` and the generic `iter_{k:03d}` path
convention.

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

## 3. Validated Results

### 3.1 Fresh bootstrap validation

The profile `fathi_reduced_3x3_12p5` has been validated from machine-readable
configuration through real SEM3D execution for both models:

```text
true_layered
  generator = PASS
  mesher = PASS
  mesh partitions = 12
  solver smoke run = PASS
  traces written = PASS

initial_homogeneous
  generator = PASS
  mesher = PASS
  mesh partitions = 12
  solver smoke run = PASS
  traces written = PASS
```

The complete bootstrap is controlled by:

```text
scripts/bootstrap/bootstrap_fathi_benchmark.py
```

Each workspace also contains machine-readable manifests under `logs/` for mesher
and solver execution. The output root contains the overall bootstrap manifest.

### 3.2 Historical inversion-transition validation

The local full iteration test previously completed:

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

Historical accepted outputs:

```text
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Kappa.h5
data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Mu.h5
data/inversion_linear/iter_009/accepted/mat/h5/Mat_0_Density.h5
```

## 4. Repository Layout

Key repository components:

```text
fathi-sem3d-inversion-benchmark/
├── configs/
│   └── fathi_reduced_3x3_12p5.json
│
├── scripts/
│   ├── bootstrap/
│   │   ├── generate_sem3d_workspace.py
│   │   ├── run_sem3d_mesher.py
│   │   ├── run_sem3d_solver.py
│   │   └── bootstrap_fathi_benchmark.py
│   │
│   ├── fathi_benchmark/
│   │   ├── run_iteration.py
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
│   ├── iteration_engine/
│   │   ├── assemble_rhs_total_generic.py
│   │   ├── solve_mtilde_generic.py
│   │   ├── generate_candidates_from_mtilde_gradient.py
│   │   ├── run_candidate_forward.py
│   │   ├── compute_candidate_misfit_v2.py
│   │   └── accept_candidate_if_descent_v2.py
│   │
│   └── regularization/
│       ├── 01_audit_q1_material_mesh.py
│       ├── 02_compute_tv_q1_full_grid.py
│       ├── 03_test_tv_q1.py
│       ├── 04_restrict_tv_rhs_to_active.py
│       ├── 05_assemble_data_tv_rhs.py
│       ├── 06_solve_mtilde_data_tv_rhs.py
│       ├── 07_diagnose_tv_weight.py
│       └── 08_generate_tv_candidates_from_mtilde_gradient.py
│
├── tests/
│   └── bootstrap/
│       ├── test_generate_sem3d_workspace.py
│       ├── test_run_sem3d_mesher.py
│       ├── test_run_sem3d_solver.py
│       └── test_bootstrap_fathi_benchmark.py
│
├── benchmark_fathi_strict/
├── benchmark_fathi_tv/
├── docs/
├── reports/
└── README.md
```

Generated bootstrap workspaces have this runtime structure:

```text
<output-root>/
├── bootstrap_manifest.json
├── true_layered/
│   ├── input.spec
│   ├── material.input
│   ├── mat/h5/
│   ├── sem/mesh4spec.0000.h5 ... mesh4spec.0011.h5
│   ├── traces/
│   └── logs/
└── initial_homogeneous/
    ├── input.spec
    ├── material.input
    ├── mat/h5/
    ├── sem/mesh4spec.0000.h5 ... mesh4spec.0011.h5
    ├── traces/
    └── logs/
```

Large runtime data should usually not be committed to GitHub. Commit scripts,
configuration, tests, small manifests, summaries, and documentation instead.

## 5. Main Inputs and Outputs

### 5.1 Bootstrap inputs

#### Benchmark profile

```text
configs/fathi_reduced_3x3_12p5.json
```

The profile defines geometry, material grids, true and initial material models,
source and receiver operators, STF parameters, partition count, and expected output
semantics.

#### External executables

```bash
export SEM3D_MESHER=/path/to/SEM/build/MESH/mesher
export SEM3D_EXE=/path/to/SEM/build/SEM3D/sem3d.exe
```

The runtime also requires `mpirun`, Python, NumPy, and h5py.

A minimal preflight is:

```bash
python --version
python -c "import numpy, h5py"
command -v mpirun
test -x "$SEM3D_MESHER"
test -x "$SEM3D_EXE"
```

### 5.2 Bootstrap outputs

For an output root `<bootstrap-root>`:

```text
<bootstrap-root>/true_layered
<bootstrap-root>/initial_homogeneous
<bootstrap-root>/bootstrap_manifest.json
```

The true-model traces are synthetic observed data. The initial-model traces are the
prediction from the homogeneous starting model. The true material field is used only
to generate and validate synthetic observations; it is not passed directly into the
inversion update.

### 5.3 Iteration inputs

For transition `iter_k -> iter_{k+1}`:

```text
results/fathi_loop_v2/states_corrected/iter_{k:03d}_state_v2_corrected.npz
data/inversion_linear/iter_{k:03d}/accepted
results/fathi_loop_v2/iter_{k:03d}_to_iter_{k+1:03d}/
  iter_{k:03d}_to_iter_{k+1:03d}_iteration_context.json
```

Observed receiver traces must be referenced through configuration or context. A
fresh bootstrap can provide the true-model trace directory.

### 5.4 Iteration outputs

```text
data/inversion_linear/iter_{k+1:03d}/forward_dudx_mgcap_full_batches/
results/fathi_loop_v2/iter_{k:03d}_to_iter_{k+1:03d}/residual_sources/
data/inversion_linear/iter_{k+1:03d}/adjoint_full_grid_batches/
results/fathi_loop_v2/iter_{k:03d}_to_iter_{k+1:03d}/component_rhs/
results/fathi_loop_v2/iter_{k:03d}_to_iter_{k+1:03d}/mtilde_solve/
results/fathi_loop_v2/iter_{k:03d}_to_iter_{k+1:03d}/candidates/
data/inversion_linear/iter_{k+1:03d}/candidate_forward_workspaces/
```

There are 30 adjoint batches:

```text
x: batch_000 ... batch_009
y: batch_000 ... batch_009
z: batch_000 ... batch_009
```

Accepted outputs follow the generic convention:

```text
results/fathi_loop_v2/states_corrected/iter_{k+1:03d}_state_v2_corrected.npz
data/inversion_linear/iter_{k+1:03d}/accepted
```

## 6. One Full Iteration Workflow

The current complete local transition is context-driven:

```text
accepted state_k
  -> create/read iteration context
  -> prepare strict forward
  -> run strict forward
  -> generate receiver residual
  -> prepare 30 adjoint workspaces
  -> run 30 adjoint simulations
  -> assemble RHS and solve Mtilde
  -> generate candidate models
  -> run candidate forward and compute misfit
  -> accept only if the configured descent rule passes
  -> audit transition
```

### Step 0. Activate the environment

```bash
export FATHI_BENCHMARK_ROOT=/path/to/fathi-sem3d-inversion-benchmark
export SEM3D_EXE=/path/to/SEM/build/SEM3D/sem3d.exe

cd "$FATHI_BENCHMARK_ROOT"
source .venv/bin/activate
```

### Step 1. Create a generic context

```bash
K=9

python scripts/fathi_benchmark/create_iteration_context_generic.py \
  --iter-k "$K" \
  --write
```

The context path follows:

```text
results/fathi_loop_v2/
  iter_{k:03d}_to_iter_{k+1:03d}/
  iter_{k:03d}_to_iter_{k+1:03d}_iteration_context.json
```

### Step 2. Plan the complete transition

```bash
python scripts/fathi_benchmark/run_iteration_full_context.py \
  --iter-k "$K" \
  --stage preflight
```

### Step 3. Execute the complete transition

```bash
python scripts/fathi_benchmark/run_iteration_full_context.py \
  --iter-k "$K" \
  --stage all_full \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute-heavy \
  --allow-mutate
```

This is the current full local runner. Heavy SEM3D stages are not launched unless
`--execute-heavy` is supplied. The accepted state is not changed unless
`--allow-mutate` is supplied.

### Resuming after prerequisites already exist

Use the resumed interface only when strict forward, residual, and adjoint outputs
are already complete:

```bash
python -m scripts.fathi_benchmark.run_iteration \
  --iter-k "$K" \
  --stage all \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute
```

### Audit

```bash
python scripts/fathi_benchmark/audit_transition_completion.py \
  --iter-k "$K"
```

The historical `iter_008 -> iter_009` commands and reports remain useful regression
evidence, but they are not the only supported iteration numbers.

---

## 7. Tests

Run the complete bootstrap regression suite with:

```bash
python -m unittest discover \
  -s tests/bootstrap \
  -p 'test_*.py' \
  -v
```

The suite covers:

```text
workspace generation
mesher execution and 12-partition audit
solver execution, smoke overrides, trace checks, and input restoration
complete two-model bootstrap planning and orchestration
```

A real SEM3D smoke bootstrap is still required to validate the local SEM3D build,
MPI runtime, filesystem, and external binary compatibility.

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
$FATHI_BENCHMARK_ROOT/data
$FATHI_BENCHMARK_ROOT/results
$FATHI_BENCHMARK_ROOT/benchmark_fathi_strict/reports
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

### Bootstrap is plan-only unless execution is explicit

```bash
python -m scripts.bootstrap.bootstrap_fathi_benchmark \
  --output-root /path/to/output
```

The command above does not create files. Add `--execute` only after reviewing the
plan. Use `--overwrite` only for a verified disposable or intentionally replaceable
output root.

### Heavy iteration stages require explicit execution

The full transition runner requires:

```text
--execute-heavy
```

before launching SEM3D stages.

### Accepted-state mutation requires explicit permission

The acceptance stage changes the trusted next state:

```text
results/fathi_loop_v2/states_corrected/iter_{k+1:03d}_state_v2_corrected.npz
data/inversion_linear/iter_{k+1:03d}/accepted
```

Local execution therefore requires:

```text
--allow-mutate
```

### Do not treat output presence as proof of success

Audit solver status using all of the following:

```text
process return code = 0
timed_out = false
fin_sem = 1
expected non-empty traces exist
solver manifest reports audit_passed = true
```

A `fin_sem` file alone is not sufficient because failed starts may still create a
small completion file.

### Preserve historical validated results

Do not delete the historical accepted state and reports used as regression evidence:

```text
data/inversion_linear/iter_009/accepted
results/fathi_loop_v2/states_corrected/iter_009_state_v2_corrected.npz
results/fathi_loop_v2/iter_008_to_iter_009
```

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

### Fresh bootstrap smoke run

```bash
export FATHI_BENCHMARK_ROOT=/path/to/fathi-sem3d-inversion-benchmark
export SEM3D_MESHER=/path/to/SEM/build/MESH/mesher
export SEM3D_EXE=/path/to/SEM/build/SEM3D/sem3d.exe

cd "$FATHI_BENCHMARK_ROOT"
source .venv/bin/activate

python -m scripts.bootstrap.bootstrap_fathi_benchmark \
  --output-root /tmp/fathi_reduced_3x3_12p5_smoke \
  --smoke-seconds 0.012 \
  --timeout-seconds 900 \
  --execute
```

### Generic full transition

```bash
K=9

python scripts/fathi_benchmark/create_iteration_context_generic.py \
  --iter-k "$K" \
  --write

python scripts/fathi_benchmark/run_iteration_full_context.py \
  --iter-k "$K" \
  --stage all_full \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute-heavy \
  --allow-mutate

python scripts/fathi_benchmark/audit_transition_completion.py \
  --iter-k "$K"
```

### Resumed transition after prerequisites exist

```bash
python -m scripts.fathi_benchmark.run_iteration \
  --iter-k "$K" \
  --stage all \
  --candidate line_search_neg_mtilde_1p00MPa \
  --execute
```

## 13. Current Status

```text
Fresh reproducibility:
  machine-readable profile = implemented
  standalone workspace generator = implemented and tested
  SEM3D mesher runner = implemented and tested
  12-partition audit = implemented and tested
  SEM3D solver runner = implemented and tested
  true_layered real smoke run = PASS
  initial_homogeneous real smoke run = PASS
  two-model bootstrap entry = implemented and tested

Iteration orchestration:
  run_iteration.py = resumed public workflow
  run_iteration_full_context.py = current full local transition workflow
  single unified iteration interface = not yet completed

Historical inversion validation:
  iter_008 -> iter_009 = PASS_COMPLETE
  accepted candidate = line_search_neg_mtilde_1p00MPa

ArmoniK:
  task graph and payload design = planned
  production integration and performance benchmark = not yet completed
```

## 14. Canonical Fathi 80 MPa Initialization

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

### Reusable iteration handoff

The mini implementation is iteration-generic. `parent_iteration` and
`next_iteration` in the JSON configuration determine the transition paths,
accepted workspace and state filename.

The accepted `iter_001` model can start the next strict-DUDX forward with:

```bash
python -m scripts.mini_e2e.run_mini_e2e \
  --config configs/fathi_mini_e2e_iter001_to_iter002.json \
  --stage next-forward \
  --np 12
```

Expected marker:

```text
RESULT = PASS_MINI_ITER001_TO_ITER002_FORWARD_HANDOFF
```

See `docs/MINI_ITERATION_HANDOFF.md`.

<!-- TV_FINAL_SUMMARY_V1 -->

## TV regularization in the generic iteration

TV regularization is integrated as an optional stage inside each inversion
transition:

```text
data RHS
→ TV RHS
→ total RHS
→ Mtilde total gradient
→ TV candidates
→ candidate forward
→ total objective
→ acceptance
```

Canonical stages:

```text
regularization
tv_candidates
tv_acceptance_plan
all_light_tv
```

Lightweight validation:

```bash
python -m scripts.regularization.validate_tv_lightweight
python -m pytest -q
```

Validated boundary:

```text
TV mathematics and workflow integration: PASS
SEM3D launched by lightweight tests: False
accepted state mutated: False
new full-scale non-zero-TV physical descent: Not claimed
```

Detailed documentation:

```text
docs/SEM3D_ALGORITHM_EXPLAINED_CN.md
docs/TV_IN_ITERATION_WORKFLOW.md
docs/FINAL_REPORT_SCOPE_CN.md
```

## Unified iteration entry point

The canonical public command is:

```bash
python -m scripts.fathi_benchmark.run_iteration \
  --config benchmark_fathi_strict/config/benchmark_config.json \
  --iter-k 0 \
  --stage plan
```

Both public and compatibility entry points delegate to:

```text
scripts/fathi_benchmark/iteration_pipeline.py
```

`run_iteration_full_context.py` no longer contains a second implementation.

## Mtilde artifact acquisition

Mtilde is generated, registered, extracted, validated, and attached to an
iteration context through one command:

```bash
python -m scripts.mtilde.ensure_mtilde   --config <benchmark-config.json>   --context <iteration-context.json>   --execute
```

The manager prefers an existing validated full matrix when configured. If the
full matrix is absent, it deterministically generates the structured Q1
consistent mass matrix from `mtilde_artifact.full_grid`. It then maps the
current forward control stations to full-grid indices and writes the active
principal submatrix bundle:

```text
artifacts/mtilde/<artifact-id>/
├── Mtilde.npz
├── coords.npy
├── active_indices.npy
└── manifest.json
```

The matrix, coordinates, indices, hashes, ordering contract, and parent
artifact are recorded in the manifest. The active artifact paths are written
to both the iteration context and the benchmark configuration.

The gradient task enforces Mtilde as a prerequisite.
`scripts/fathi_benchmark/run_task3_gradient.py` first runs
`scripts.mtilde.ensure_mtilde` for the current iteration context and only then
delegates to `run_task3_gradient_core.py`. This applies to every caller,
including the canonical iteration runner and direct task invocation.
