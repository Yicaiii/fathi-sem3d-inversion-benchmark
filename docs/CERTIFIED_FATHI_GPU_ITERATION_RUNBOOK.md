# Certified Fathi GPU inversion iteration runbook

This runbook describes the reusable certified path. Historical Stage5N,
Stage5N-R, and Stage5O artifacts remain immutable certification evidence.

## One-time prerequisites

- A GPU SEM3D executable and MPI launcher for workflows that explicitly need
  ordinary SEM3D execution.
- The benchmark Python environment and its NumPy/SciPy/HDF5 dependencies.
- The bootstrap/current accepted model under `data/<run>/iter_K/accepted`.
- `results/<run>/certified_external_reference.json`, which freezes the external
  topology, PML coefficients, coupled mass, GLL operator, physical receiver
  operator, true external receiver array, timestep, and STF contract.
- `results/<run>/certified_gradient_registry.json` for immutable L-BFGS
  gradient history.

Set the environment once:

```bash
cd "$HOME/fathi-sem3d-gpu-inversion-benchmark"
source "$HOME/sem3d_fathi_clean/.venv/bin/activate"
export FATHI_BENCHMARK_ROOT="$PWD"
```

## Iteration k

Only `--iter-k K` changes between iterations:

```bash
python -m scripts.fathi_benchmark.run_certified_iteration \
  --config configs/<run>.json --iter-k K --stage status
```

The certified sequence is:

1. `context` — create the certified-external iteration context; ordinary
   Capteur traces are not required.
2. `parent-forward --run-expensive` — evaluate the accepted parent material
   through the frozen external recurrence and retain primal checkpoints.
3. `reverse-preflight`, `prepare-replay --run-expensive`, then
   `reverse --run-expensive` — apply the previously certified exact adjoint.
4. `gradient-bridge` — transpose solid/PML gradients to the full H5 control,
   restrict with coordinate-derived H5 indices, and solve Mtilde.
5. `search-direction` — compute the physical lambda/mu optimizer direction.
6. `register-gradient` — register the certified optimizer gradient for future
   L-BFGS history.
7. `line-search-next` — cheaply reuse evaluated objectives, prepare and audit
   the next backtracking candidate, and stop before its external forward.
8. `line-search --run-expensive` — resume at the first unevaluated physical
   step, evaluate one candidate at a time, and stop immediately on acceptance.
   Lower-level `candidate-generate --step-mpa X`, `candidate-audit --step-mpa
   X`, `candidate-preflight --step-mpa X`, and `candidate-forward --step-mpa X
   --run-expensive` stages remain available.
9. `promotion-audit`, then `promote` — apply the strict
   `J_candidate < J_parent` gate without overwriting unrelated accepted state.
10. Continue with `K+1`.

There is deliberately no blind `all` stage. Expensive or resumable stages
require `--run-expensive`.

## Optimizer history

At iteration zero, no curvature pair exists:

```text
p0 = -g0
method = lbfgs_initialization
```

At iteration one and later, the registry supplies certified historical
gradients. The first pair is:

```text
s0 = m1 - m0
y0 = g1 - g0
```

The standard curvature gate decides whether the pair enters the two-loop
recursion. A reported `lbfgs_restart` is valid when the gate rejects every
available pair; the mathematics must not be altered merely to force acceptance.

## Backtracking and acceptance policy

The benchmark default remains `strict_descent`:

```text
accept iff J_candidate < J_parent
```

The explicitly selectable `armijo` policy follows the Fathi Algorithm 1 form
with configurable `c1` and `rho`:

```text
accept iff J_candidate
          < J_parent + c1 * (step_pa / direction_scale) * g_dot_p
```

The factor `step_pa / direction_scale` is required by the candidate generator's
actual update:

```text
m_candidate = m_parent + step_pa * p / joint_maxabs(p_lambda, p_mu)
```

Consequently, a physical step such as `0.1 MPa` is not an L-BFGS alpha of
`0.1`. It means a `100000 Pa` joint maximum material update; its multiplier on
the raw L-BFGS direction is `100000 / direction_scale`.

The default backtracking sequence begins at `0.1 MPa` and uses `rho=0.5`, so
rejections produce `0.05`, `0.025`, `0.0125`, and smaller physical steps. The
line-search summary is resumable and reuses an existing certified PASS
candidate objective without rerunning it. Line search never promotes a model;
promotion remains an explicit later stage.

## Objective and SEM3D boundary

Ordinary GPU `capteurs.*.h5` traces are not the certified optimization
objective. The frozen objective uses the physical receiver operator:

```text
residual = current_external - true_external
J = 0.5 * sum(w[t] * residual[t,r,c]^2)
```

SEM3D remains the GPU solver used to establish and validate the benchmark
state/operator evidence. The production certified objective and reverse use
the benchmark-side external recurrence previously validated against SEM3D
solver state and snapshots. SEM3D itself does not perform L-BFGS.
