# Fathi §4.3 layered-medium reproduction contract

Status: **STATIC ACQUISITION PREFLIGHT PASS — no forward or reverse job was launched**
Reference run: `fathi_s43_gpu_p40_np2_apow13p8155_pml6p25_t052`
Paper: A. Fathi et al., *Full-waveform inversion in three-dimensional PML-truncated elastic media*, CMAME 296 (2015), 39–72 ([author PDF](https://www.caee.utexas.edu/prof/kallivokas/publications/pubs/CMAME-2015.pdf)).

This contract defines a new paper-fidelity lineage. It does not replace the certified physical-step lineage, modify its accepted material, or alter the Stage 5O external forward/adjoint reference. The separate design configuration is `configs/fathi_s43_reproduction_policy.json`; it is deliberately inactive until the gates at the end of this document pass.

## Classification legend

- **MATCH** — the paper statement and current implementation agree at the audited level.
- **DIFFERENT** — an observed implementation choice differs from the paper.
- **PAPER_UNSPECIFIED** — the paper does not provide the value needed for an exact reproduction.
- **PAPER_INFERRED** — the value follows from paper statements and published numerical counts, but is not quoted directly.
- **PAPER_IMPLEMENTATION_ASSUMPTION** — a necessary implementation choice consistent with the paper but not stated explicitly.
- **NEEDS_AUDIT** — the necessary evidence or unit convention is not yet fixed; execution must not assume a value.

## Paper/implementation alignment matrix

| Quantity | Paper contract | Current certified p40 lineage | Class | Consequence |
|---|---|---|---|---|
| Physical domain | 40 m × 40 m × 45 m (§4.3, p. 54) | x,y = −20…20 m; z = −45…0 m | **MATCH** | Reuse physical geometry. |
| PML thickness | 6.25 m on truncation boundaries (§4.3) | 6.25 m on four sides and bottom | **MATCH** | Geometry can be reused. |
| PML attenuation law | The paper says “throughout” α₀=5, β₀=400 s⁻¹ and quadratic profile m=2 (p. 52) | SEM3D `npow=2`, `Apow=13.815510558`; no demonstrated α₀/β₀ equivalence | **DIFFERENT** | Preserve certified operator, but do not call its damping parameters paper-identical. |
| Density | 2000 kg/m³ (§4.3) | 2000 kg/m³ | **MATCH** | Reuse. |
| Initial λ and μ | Homogeneous 80 MPa (§4.3, p. 56) | Homogeneous 80 MPa each | **MATCH** | Reuse. |
| Target λ and μ | 80 MPa for −12≤z≤0; 101.25 MPa for −27≤z<−12; 125 MPa below −27 m (Eq. 30) | Same values and interfaces over the 45 m physical depth | **MATCH** | Reuse target field. |
| Element family/order | Quadratic hexahedral spectral elements, 27-node bricks | `ngll=3`, polynomial order 2 | **MATCH** | Reuse mesh topology. |
| Element size | 1.25 m (§4.3) | 1.25 m; 32×32×36 physical elements | **MATCH** | Reuse mesh. |
| Time integrator | Explicit RK4 (p. 44) | Certified SEM3D velocity/Newmark-style discrete recurrence | **DIFFERENT** | A pragmatic reproduction may retain the certified recurrence, but must label the difference. |
| Time step | Δt = 10⁻³ s (§4.3) | fixed certified Δt = 3.608439182435162×10⁻⁴ s | **DIFFERENT** | Do not silently force 1 ms into the certified recurrence; choose and certify the p20 time contract at Gate B. |
| Source spatial loading | Vertical stress loading over surface square −17.5≤x,y≤17.5 m (§4.3); Fig. 6 labels the green square “loaded region” | Nine simultaneous vertical `impulse` point sources on a 3×3 grid at x,y∈{−12.5,0,12.5} m | **DIFFERENT** | The reproduction uses one uniform distributed vertical traction over the complete square. Uniformity is a **PAPER_IMPLEMENTATION_ASSUMPTION**, not a quoted paper value. |
| Source load-case interpretation | The prose and Fig. 6 define a spatial loaded region but do not enumerate independent shots | Nine simultaneous point sources | **PAPER_IMPLEMENTATION_ASSUMPTION** | The reproduction treats the complete patch as one distributed load field; it does not reuse the old nine forces. |
| Source amplitude | Gaussian load amplitude 1 kPa (p. 50) | STF peak 1 multiplied by SEM3D default source amplitude 1, then applied as a point-force vector | **DIFFERENT** | Current loading is not 1000 Pa traction. |
| Point-force / traction conversion | Paper weak form uses surface traction; exact discrete quadrature is not reported | No surface-area integration: the source lands on a GLL node and injects the unit temporal amplitude directly into the nodal force | **DIFFERENT** | The reproduction now has a Q2/LGL consistent surface operator using the physical face Jacobian; its static force-conservation preflight passes. |
| Receiver region | Surface square −17.5≤x,y≤17.5 m (§4.3) | Same region, z=0 | **MATCH** | Region can be reused. |
| Receiver sampling | “at every grid point” (§4.3); exact interpretation/count is not enumerated | 15×15 = 225 points at 2.5 m spacing | **DIFFERENT** | Current grid is sparser than either the 1.25 m element grid or quadratic GLL grid. |
| Reproduction receiver count/spacing | Quadratic 1.25 m discretization, published 616,850 λ+μ parameters, 65×65×73 Q2 material grid, “every grid point”, and the 35 m interval imply 57×57 nodes at 0.625 m spacing | 225 / 2.5 m | **PAPER_INFERRED** | The runtime count is derived from actual mesh coordinates; the paper is not claimed to state 3,249 explicitly. |
| Residual quantity | Vector displacement difference at receivers (Eq. 7) | Three-component displacement, `current_external - true_external` | **MATCH** | Sign does not affect J; keep the certified sign for the adjoint. |
| Data objective | ½ Σreceivers ∫‖u−uᵐ‖² dt (Eq. 7) | Same data-term structure | **MATCH** | Mathematical data term agrees. |
| Objective content used by line search | Eq. 7 is data misfit plus regularization | Current certified p40 line search uses data-only external J | **DIFFERENT** | Paper-fidelity policy must use `J_total` consistently. |
| Discrete time quadrature | Not stated | Native fixed-Δt trapezoidal weights | **PAPER_UNSPECIFIED** | Trapezoid is a documented implementation choice, not a claimed paper match. |
| Objective normalization / plotted units | Eq. 7 shows no normalization; Fig. 21 supplies no discrete scaling or units | No receiver, sample, energy, or initial-misfit normalization | **PAPER_UNSPECIFIED** | Fig. 21’s absolute ordinate cannot be reconstructed from the paper alone. |
| TV functional | Smoothed isotropic TV, Eq. 9 | TV is absent from the active certified p40 context; an optional normalized-Q1 kernel exists separately | **DIFFERENT** | Enable only in the new lineage and certify it. |
| TV epsilon | ε = 0.01 (§4.3) | Existing optional kernel defaults to dimensionless ε=10⁻³ and computes `sqrt(norm(grad(m_hat))²+epsilon²)` | **DIFFERENT** | Paper Eq. 9 adds ε, not ε². Conversion depends on the control scaling. |
| Units/scaling of ε | Paper does not state the internal units or nondimensionalization of λ, μ and ε | Existing TV kernel normalizes by median parent material | **NEEDS_AUDIT** | Do not set the existing kernel to 0.01 and call it a match. |
| Regularization factors | Eq. 24: `R = P * norm(g_mis) / norm(g_reg)`; §3.3 suggests P in 0.5–0.3 | No active regularization policy in current p40 context | **DIFFERENT** | Add a configuration-driven policy. |
| Exact §4.3 P values/schedule | Not given | None | **PAPER_UNSPECIFIED** | Do not fabricate a 310/860-stage regularization schedule. |
| λ search-direction bias | Eq. 25; W=1 initially, decreases linearly to zero around k=50, then remains zero | No λ bias in certified physical-step policy | **DIFFERENT** | New policy uses `W(k)=max(1−k/50,0)` with global k. |
| L-BFGS memory | m=15 (footnote on p. 47) | m=5 | **DIFFERENT** | New policy uses 15. |
| L-BFGS initial inverse-Hessian scaling / safeguards | Not reported | Current implementation has its own restart/tolerance rules | **PAPER_UNSPECIFIED** | Must be made explicit and regression-tested before a long run. |
| Material coordinate units used by optimizer | Paper plots MPa but does not state internal L-BFGS coordinate scaling | Certified bridge stores Pa and then max-abs normalizes directions | **NEEDS_AUDIT** | Raw α=1 is meaningless until the new optimizer coordinate convention is frozen. |
| Update equation | `lambda_(k+1)=lambda_k+alpha_lambda*s_lambda` and `mu_(k+1)=mu_k+alpha_mu*s_mu` (Eq. 19) | Joint-maxabs direction scaled to a physical MPa step | **DIFFERENT** | New policy uses raw L-BFGS directions and no MPa normalization. |
| Acceptance equation | Armijo Algorithm 1 on J | Strict descent on data-only J | **DIFFERENT** | New policy uses Armijo on the same `J_total` used by its gradient. |
| Armijo initial α | αλ=αμ=1 (Algorithm 1 example used by requested contract) | Current policy starts at a 0.1 MPa normalized step | **DIFFERENT** | New policy starts both raw alphas at 1. |
| Armijo c₁ | 10⁻⁴ | Config value 10⁻⁴, inactive under strict descent | **MATCH** value / **DIFFERENT** use | Activate only in new policy. |
| Armijo ρ | 0.5 | 0.5 | **MATCH** | Reuse value. |
| p20 pulse | fmax=20 Hz, μ̄=0.11 s, σ̄=0.0014 s², tend=0.20 s (Table 1) | No p20 stage | **DIFFERENT** | Add stage. |
| p30 pulse | fmax=30 Hz, μ̄=0.08 s, σ̄=0.0007 s², tend=0.15 s | No p30 stage | **DIFFERENT** | Add stage. |
| p40 pulse | fmax=40 Hz, μ̄=0.06 s, σ̄=0.0004 s², tend=0.12 s | Center 0.06 s and implementation width 0.02 s, whose square is 0.0004 s²; file continues a tiny Gaussian tail beyond 0.12 s | **MATCH** shape parameters / **DIFFERENT** truncation | New generator must encode Table 1 directly and enforce active interval. |
| Source temporal formula | `exp(-(t-μ̄)²/σ̄)` (p. 50) | `exp(-((t-center)/sigma_s)²)` | **MATCH** for p40 because `sigma_s²=σ̄` | Preserve the parameter-convention distinction. |
| Simulation T by frequency | p20 0.45 s; p30 0.40 s; p40 0.40 s (§4.3) | p40-only T=0.52 s | **DIFFERENT** | New stages use paper T. |
| Frequency transitions | Text: p20 for 310 iterations, p30 through iteration 860, p40 to convergence at 1112 | No continuation; current run starts at p40 | **DIFFERENT** | New text-based schedule is 0→310→860→1112. |
| Paper-internal frequency labels | Table 1/text call p20 20 Hz, but Fig. 18 labels its first panels 10 Hz; Fig. 21’s first red marker appears near 400 although text says 310 | N/A | **NEEDS_AUDIT** | Use Table 1 and §4.3 prose as primary; record the figure conflicts rather than hiding them. |

## Static source and receiver operator contract

The reproduction source is frozen as
`uniform_distributed_vertical_surface_traction` over
`[-17.5,17.5] × [-17.5,17.5]` at `z=0`, with positive vertical peak
traction 1000 Pa. For every complete surface element in that region, the
operator assembles

```text
F_i(t) = integral_Gamma N_i(x,y) traction(x,y,t) dGamma
```

using the actual Q2 connectivity, the existing three-point LGL weights in
each surface coordinate, and the physical isoparametric surface Jacobian.
The region boundary coincides with element boundaries; a future region that
cuts an element is rejected unless an explicit cut-cell quadrature is added.
The static preflight obtains 1225 m² and a peak vertical resultant of
1,225,000 N, with zero horizontal resultant, centered support, exact x/y
symmetry, and zero RHS outside the region.

Receiver rows are constructed by selecting every actual solid Q2/LGL node at
`z=0` inside the same inclusive bounds, sorting with y outer and x inner, and
using a single unit weight at each collocated node. The measured mesh yields
57×57 = 3,249 rows and 0.625 m spacing. Those values are outputs of mesh
inspection, not hard-coded operator inputs.

The p20/p30/p40 temporal loads are constructed analytically from their stage
configs and are exactly zero after their configured active end. The static
p20 grid includes μ̄=0.11 s and therefore reaches exactly 1000 Pa. No legacy
p40 STF file is read or reused.

## Objective-scale audit

### Current source chain

1. `gaussian_stf.txt` peaks at 1.0. Its p40 time shape is the Table 1 Gaussian after converting σ̄=0.0004 s² to an implementation width `sqrt(σ̄)=0.02 s`.
2. Each `source` block omits `amplitude`; the SEM3D parser initializes `source->amplitude = 1`.
3. `type = impulse` builds a Lagrange-interpolated collocated point-force vector. For these source coordinates the interpolation is exactly a single GLL node.
4. The SEM3D step adds `STF × point-force-vector`; the certified external recurrence equivalently applies `inverse_mass × STF × direction` at each source node.
5. There is no multiplication by 1000 Pa, no loaded-surface area, and no surface quadrature/Jacobian conversion to nodal force.

Therefore the current source is a unit-amplitude discrete point-force experiment, not the paper’s 1 kPa vertical surface-stress experiment.

### Current receiver/objective chain

- The certified physical receiver operator returns displacement in the simulation’s SI length coordinate system at 225 surface points.
- The residual has displacement units and sign `simulated - measured`.
- The certified data objective is exactly

  ```text
  J_data = 0.5 * sum(time_weight[t] * residual[t,r,c]^2)
  ```

  over 1441 samples, 225 receivers and three components, using fixed-Δt trapezoidal weights.
- It applies no division by receiver count, sample count, true-data energy, initial objective, source energy, or component count.

The frozen lineage’s objective is about 4.44×10⁻²⁰. Fig. 21 begins near 10⁻² and ends near 10⁻⁸, but the paper does not specify a discrete quadrature, exact receiver count, source-to-nodal-force operator, or plotted scaling. More importantly, the current physical loading and acquisition differ. **The two absolute objective values are not directly comparable.** A normalized trend such as `J_total(k)/J_total(0)` may be compared qualitatively only after the p20 acquisition and total-objective contracts are fixed; it must not be presented as the paper’s raw ordinate.

## Separate paper-fidelity optimizer policy

The existing certified policy remains unchanged. The new inactive policy is:

```text
optimizer_policy = fathi_raw_lbfgs_armijo

lambda_candidate = lambda_parent + alpha_lambda * s_lambda
mu_candidate     = mu_parent     + alpha_mu     * s_mu
```

There is no joint-maxabs normalization and no MPa step cap. L-BFGS stores 15 vector pairs. Algorithm 1 starts `alpha_lambda = alpha_mu = 1`, uses `c1 = 1e-4`, and halves both alphas with `rho = 0.5` until:

```text
J_total(candidate) < J_total(parent)
  + c1 * (alpha_lambda * dot(g_lambda, s_lambda)
        + alpha_mu     * dot(g_mu,     s_mu))
```

The λ bias is applied to the L-BFGS search directions using paper Eq. 25 and `W(k)=max(1-k/50,0)` for global iteration k. The paper defines the norm notation used on the same page as the Euclidean norm, so Eq. 25 uses Euclidean L2 normalization. Zero-norm handling remains fail-closed: the implementation does not invent a normalized vector.

For the current-T052 physical optimizer gate, the following is explicitly a **reproduction assumption, not a fact reported by Fathi**. Optimizer vectors remain in physical Pa coordinates and all optimizer pairings use `<a,b>_M=a^T Mtilde b`. Freeze `m_ref_pa=80,000,000` and `J_ref=3.78991998304714295e-08` at iteration 0, set `gamma0=m_ref_pa^2/J_ref`, and use `H0_phys=gamma0 I` when no history exists. Future history stores physical `s` and physical Riesz-gradient `y`, uses `gamma_k=sMy/yMy`, and skips pairs that fail a positive relative Mtilde-curvature safeguard. No damping rule is assumed.

## TV objective/gradient consistency

The paper-fidelity branch must use one mathematical object throughout:

```text
J_total = J_data + R_TV(lambda, mu)
Mtilde * g = regularization_gradient + misfit_gradient
```

The candidate and parent use the same λ/μ regularization factors during a single line search. Factors may be recomputed only after an accepted iteration. This makes the directional derivative on the Armijo right-hand side consistent with the objective on the left.

Section 3.3 gives `R = P ||g_mis||/||g_reg||` and suggests reducing P from 0.5 toward 0.3. Section 4.3 does not report the exact P values or schedule. The design therefore records the policy and range but leaves the schedule unset. The zero-`g_reg` case is also unresolved.

Paper Eq. 9 uses `sqrt(|grad m|² + epsilon)` with ε=0.01. The existing optional Q1 code uses normalized material and `sqrt(|grad m_hat|² + epsilon_dimensionless²)`. These are not identical parameter conventions. The adapter value must be derived after the optimizer/TV material scaling is frozen; it must not be guessed.

## Frequency continuation contract

The configuration follows Table 1 and the §4.3 prose:

| Stage | Pulse | Gaussian parameters | T | Paper-text global interval |
|---|---|---|---|---|
| p20 | 20 Hz | μ̄=0.11 s, σ̄=0.0014 s², tend=0.20 s | 0.45 s | start → 310 |
| p30 | 30 Hz | μ̄=0.08 s, σ̄=0.0007 s², tend=0.15 s | 0.40 s | 310 → 860 |
| p40 | 40 Hz | μ̄=0.06 s, σ̄=0.0004 s², tend=0.12 s | 0.40 s | 860 → convergence near 1112 |

No stage is authorized to run by this document. The exact zero/one-based directory mapping is frozen at Gate A.

## Efficiency contract without scientific changes

### Checkpoint and reuse

Every candidate cache key must include candidate-material, operator, source, STF, receiver, objective-policy, and regularization-factor hashes. A resumable forward records the last completed transition and the accumulated objective lower bound. An incomplete candidate is never promoted to a complete objective.

### Mathematically safe Armijo early rejection

Let `J_data_partial(n)` be the sum of the already completed, nonnegative, final-quadrature-weighted residual terms, and let `R_TV_candidate` be computed before the forward. Let `A` be the fixed Armijo right-hand side for this candidate. Because all remaining data terms are nonnegative:

```text
if J_data_partial(n) + R_TV_candidate >= A:
    reject candidate and stop its forward
```

This is a rigorous lower-bound rejection, not a heuristic. It is valid only when the candidate regularization factors are frozen for that line search and the partial sum uses the same final objective weights. A negative Armijo threshold is immediately impossible because `J_total >= 0`.

### GPU and orchestration

Port the already-certified external recurrence to a GPU backend without changing operator arrays, update equations, receiver interpolation, source operator, or reduction definitions. Regress CPU versus GPU for receiver traces, `J_data`, `J_total`, tangent action, and λ/μ adjoint projections. Keep CPU as the numerical oracle. Add restartable per-iteration manifests before introducing ArmoniK. ArmoniK tasks must be content-addressed and idempotent; orchestration is deferred until the local ten-iteration gate passes.

## Reproduction gates

- **Gate A — paper/config alignment:** all `NEEDS_AUDIT` execution blockers are resolved; surface traction, receiver interpretation, optimizer units, TV convention, iteration indexing and accepted pragmatic deviations are frozen in hashes.
- **Gate B — p20 forward/reference:** paper-fidelity p20 source/receiver/T contract passes a one-time CPU reference forward and checkpoint/restart identity test.
- **Gate C — p20 adjoint:** p20 tangent/FD refinement and one exact reverse close for both λ and μ against the new total-objective contract.
- **Gate D — optimizer:** raw L-BFGS direction, λ bias, total-gradient dot products, and Armijo decision pass deterministic regression tests with no maxabs normalization.
- **Gate E — three iterations:** the first three accepted `J_total` values decrease and every candidate is provenance-complete.
- **Gate F — ten iterations:** `J_total(k)/J_total(0)` shows a finite, sensible trend; material remains finite/positive; CPU/GPU spot checks remain within certified tolerances.

Only Gate F authorizes the first 310-iteration p20 run.

## Frozen-lineage decision

- Stage 5O and `certified_external_reference.json` remain unchanged.
- The 0.00625 MPa backtracking experiment is not required and must not resume.
- Existing p40 candidate objectives remain historical certified artifacts; they are neither deleted nor reused as paper-fidelity p20 objectives.
- No SEM3D, external forward, tangent, or reverse job was launched for this audit.
- No Git commit or push is authorized.
