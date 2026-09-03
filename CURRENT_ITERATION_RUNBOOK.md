# CURRENT Fathi §4.3 Iteration Runbook

## Frozen production status

Run:

`fathi_s43_repro_p20_t052`

The CURRENT production engine is frozen after the successful real
end-to-end transition:

`iter_001 -> iter_002`

Final runtime result:

`PASS_CURRENT_ITERATION_001_TO_002_CLOSED`

The generic CURRENT runner is intended for accepted-parent iterations
`k >= 1`.

## Environment

```bash
cd "$HOME/fathi-sem3d-gpu-inversion-benchmark"
source "$HOME/sem3d_fathi_clean/.venv/bin/activate"

export FATHI_BENCHMARK_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Run one inversion iteration

For `iter_k -> iter_{k+1}`:

```bash
K=2
bash scripts/fathi_benchmark/run_current_iteration.sh "$K"
```

For the following iteration, only change `K`:

```bash
K=3
bash scripts/fathi_benchmark/run_current_iteration.sh "$K"
```

The runner derives parent, child and transition paths dynamically.

## Monitor

For `K=2`:

```bash
tail -F "$HOME/fathi-sem3d-gpu-inversion-benchmark/results/fathi_s43_repro_p20_t052/iter_002_to_iter_003/current_iteration_driver.log"
```

## Resume policy

If an execution is interrupted:

1. Do not delete the transition directory.
2. Do not delete candidates or checkpoints.
3. Do not regenerate already certified stages.
4. Run the same `K` command again.

The runner reuses or resumes durable artifacts.

## CURRENT production sequence

```text
accepted m_k
    |
parent external forward
    |
objective / residual
    |
exact discrete reverse
    |
material covector
    |
control-space bridge
    |
Mtilde solve
    |
registered physical gradient g_k
    |
L-BFGS history
    |
physical-space L-BFGS
    |
Fathi Eq.25 lambda bias
    |
candidate = parent + alpha * direction
    |
external candidate forward
    |
Armijo decision
    |
accepted-child promotion
    |
accepted m_{k+1}
```

## Frozen mathematical contract

- Exact discrete reverse is the algebraic transpose of the certified
  external discrete forward route.
- Mtilde defines the physical/control-space metric.
- L-BFGS vectors use physical Pa units.
- L-BFGS history memory target: 15.
- Fathi Eq.25 uses the Euclidean L2 norm.
- `W(k) = max(1 - k/50, 0)`.
- Armijo parameters:
  - `alpha0 = 1`
  - `c1 = 1e-4`
  - `rho = 0.5`
- Candidate update:
  `m_candidate = m_parent + alpha * p_parent`.
- No max-absolute normalization is used.

## Superseded production routes

The CURRENT iteration route must not fall back to:

- `bridge_stage5o_certified_gradient.py`
- `run_current_t052_*`
- `finalize_current_t052_*`
- `424B_compute_rhs_component_from_traces.py`
- `compute_search_direction.py`
- `prepare_gpu_adjoint_full.py`
- `run_gpu_adjoint_task.py`
- `solve_gpu_mtilde_gradient.py`

Historical assets may only be reused where explicitly certified as
immutable operator assets.

## Git policy

Generated numerical data are not source code and must not be committed:

- `results/`
- `data/reproduction/`
- `*.h5`
- `*.hdf5`
- `*.npy`
- `*.npz`
- checkpoints
- replay caches
- numerical logs
