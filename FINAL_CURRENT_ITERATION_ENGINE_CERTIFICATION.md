# FINAL CURRENT ITERATION ENGINE CERTIFICATION

## Scope

Run:

`fathi_s43_repro_p20_t052`

Certified real production transition:

`iter_001 -> iter_002`

This certification freezes the CURRENT production iteration route after
a real end-to-end numerical execution.

## Real production result

```text
RESULT = PASS_CURRENT_ITERATION_001_TO_002_CLOSED

ARMIJO_RESULT =
PASS_ITER001_EXTERNAL_ARMIJO_ACCEPTED

PROMOTION_RESULT =
PASS_ITER001_TO_ITER002_PROMOTED

CHILD_RESULT =
PASS_CURRENT_T052_ITER002_ACCEPTED_MODEL
```

Accepted line-search step:

```text
alpha = 1.0
```

Objectives:

```text
J_parent   = 3.7881758065829525e-08
J_accepted = 2.1527753044112665e-08
```

The accepted candidate external forward completed all 1040 time samples
before promotion.

## Certified interface chain

```text
C01 CURRENT config routing                  PASS
C02 certified reference routing             PASS
C03 parent forward / durable reuse           PASS
C04 exact reverse / durable reuse            PASS
C05 reverse -> gradient bridge               PASS
C06 Mtilde physical-gradient contract        PASS
C07 registered-gradient contract             PASS
C08 curvature-history contract               PASS
C09 physical-space L-BFGS                    PASS
C10 Fathi Eq.25 lambda-bias contract         PASS
C11 candidate-generation contract            PASS
C12 candidate -> external driver contract    PASS
C13 real 1040-step external forward          PASS
C14 objective evaluation                     PASS
C15 Armijo trial -> summary contract         PASS
C16 accepted trial -> promotion contract     PASS
C17 iter002 accepted-model contract          PASS
C18 iter002 durable-state contract            PASS
```

Known CURRENT interface blockers at this frozen transition:

```text
0
```

## Frozen final artifact hashes

Armijo summary:

`966d8cbc8687169d352bef567e1fabe26f647e1cd8638d7e29c2b94acf86bccf`

`iter_002/accepted/accepted_summary.json`:

`6518cc2a5e756bd035eed4eed767e42b66ceea885257e859e7032b5ae1a82ff2`

`iter_002_state.npz`:

`a988e11d207cb6fa51514f4340338856d7a185a16303fb77477bd417c5f7032d`

Accepted iter002 material:

```text
Density:
27a25652cabd9a39ccf9cf122748fb0199a4e95f2da97703eb2fe2863f10d079

Kappa:
1aff50683475d179c53e0706ef73dcb36ee1ea4646cb4c510cfc7505f6519a43

Mu:
d8fab8ed867a760d96fcea43ed56f3c391bb413b467a4046efe35bff75b1bb1a
```

## Mathematical freeze

The successful production transition does not alter the previously
certified mathematics:

```text
EXACT_REVERSE_MATH_CHANGED = false
MTILDE_MATH_CHANGED = false
LBFGS_MATH_CHANGED = false
EQ25_MATH_CHANGED = false
ARMIJO_MATH_CHANGED = false
```

Recent changes are production integration changes:

- CURRENT runtime/reference configuration routing;
- durable CURRENT certified-reference generation;
- generic exact-reverse routing;
- candidate-forward `primal` receiver-label alignment;
- unified `K`-driven runner.

## Numerical rerun policy

Previously certified expensive stages must not be rerun merely because
of metadata, routing, documentation, plotting, or Git operations.

A numerical stage may be rerun only when its mathematical/numerical
inputs have genuinely changed or its numerical result is invalidated.

## Final result

```text
RESULT = PASS_FINAL_CURRENT_ITERATION_ENGINE_CERTIFICATION
```
