# TV inside one generic iteration

## Canonical position

```text
strict forward
→ residual
→ adjoint
→ data RHS
→ TV regularization
→ total RHS
→ Mtilde total gradient
→ TV candidates
→ candidate forward
→ total objective
→ acceptance
```

TV is integrated after data-RHS assembly and before the Mtilde solve.

## Canonical runner stages

```text
gradient
regularization
tv_candidates
tv_acceptance_plan
all_light_tv
```

### gradient

Produces the data-only right-hand sides and data-only gradient artifacts.

### regularization

Computes:

```text
parent material
→ TV value
→ TV derivative
→ active restriction
→ data RHS + alpha × TV RHS
→ total Mtilde gradient
```

This stage does not launch SEM3D.

### tv_candidates

Generates candidate materials from the total gradient.

This stage does not launch SEM3D.

### tv_acceptance_plan

Computes the total-objective comparison interface:

\[
J_{\mathrm{total}}
=
J_{\mathrm{data}}
+
lpha_\lambda R_{\mathrm{TV}}(\widehat\lambda)
+
lpha_\mu R_{\mathrm{TV}}(\widehat\mu).
\]

This stage does not launch SEM3D.

### all_light_tv

Runs all TV-side mathematical stages that do not require a new wave simulation.

## Lightweight validation

The lightweight test verifies:

```text
constant-field TV derivative
directional derivative
alpha=0 regression
non-zero alpha synthetic field
candidate generation
total-objective acceptance
```

The expected safety flags are:

```text
sem3d_launched = false
accepted_state_mutated = false
```

## Physical-validation boundary

A real non-zero-TV descent claim requires a candidate forward because
\(J_{\mathrm{data,candidate}}\) must be calculated from new SEM3D traces.

Therefore:

```text
workflow integration: validated
TV mathematics: validated
synthetic candidate generation: validated
new full-scale physical descent: not claimed
```
