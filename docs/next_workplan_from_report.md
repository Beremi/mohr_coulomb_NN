# Next Workplan From `report.md`

This plan combines:

- the literature guidance summarized in [report.md](../report.md)
- what has actually worked in this repository so far
- the current bottlenecks observed on real sampled slope-stability data

The goal is not to restart from theory. The goal is to turn the paper ideas into the shortest path toward a solver-usable constitutive surrogate.

## Current Position

What we already know from our own runs:

- naive synthetic-only training was not enough
- training on real sampled constitutive states was necessary
- for the `cover_layer`, heavy epoch-refresh training with a fitted synthetic generator now works well
- the current best cover-layer model is [best.pt](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/best.pt)
- that model reached real-test stress MAE `1.4951`, RMSE `9.7174`, branch accuracy `0.9577`
- the remaining weak point is not average error; it is tail behavior and solver robustness

What the literature adds:

- do not rely on plain black-box `6 -> 6` stress regression as the final design
- keep as much constitutive structure as possible
- branch awareness matters
- exact elastic handling is low-risk and high-value
- return-mapping-assisted learning is the strongest next route when solver robustness matters more than pointwise regression

## Main Decision

We should not jump directly to PINNs, RNNs, or a full thermodynamic redesign.

The next steps should be:

1. keep the current best heavy fitted-refresh models as baselines
2. move toward a hybrid constitutive surrogate that preserves exact structure where possible
3. only then try the more research-heavy variants

So the near-term direction is:

- exact elastic branch
- branch-aware plastic surrogate
- reduced constitutive parameters as first-class inputs
- principal / invariant / trial-state features
- solver-facing evaluation, not only pointwise error

## Work Plan

### Phase 0 — Freeze and Benchmark What Already Works

Purpose:
- avoid losing the current best result while we change model structure

Tasks:

1. Freeze the current best cover-layer checkpoint and report as the baseline winner.
2. Record a stable benchmark set of metrics:
   - real test MAE / RMSE / max abs
   - per-branch MAE
   - branch confusion
   - history plot
3. Add one script that evaluates any checkpoint on the same fixed real and synthetic holdouts so comparisons stay consistent.

Deliverables:

- stable baseline report for cover layer
- one canonical evaluation command

Exit criterion:

- every future experiment can be compared against the same cover-layer baseline in one table

### Phase 1 — Upgrade the Dataset Schema

Purpose:
- align the stored training data with what the literature says the model should see

Tasks:

1. Extend dataset creation to save, for each sample:
   - engineering strain `E`
   - exact stress `S`
   - exact branch label
   - reduced parameters `c_bar, sin_phi, G, K, lambda`
   - trial stress
   - ordered principal strains
   - ordered principal trial stresses
   - invariants such as `p`, `q`, and a Lode-angle-like coordinate
   - optional tangent if available
   - optional yield value / branch-threshold-related scalars
2. Keep raw material parameters too, but treat them as bookkeeping rather than the main learning representation.
3. Create fixed validation subsets for:
   - broad random states
   - near-yield states
   - hard branches `left_edge`, `right_edge`, `apex`
   - FE-trace states from real runs

Why this phase matters:

- it enables branch-aware principal/invariant models
- it also enables return-mapping-assisted models later without rebuilding the entire dataset again

Exit criterion:

- one HDF5 schema supports both supervised stress regression and hybrid constitutive experiments

### Phase 2 — Build the Best Practical Baseline

Purpose:
- produce the best solver-oriented baseline before trying more research-heavy ideas

Model:

- exact elastic branch retained
- learned plastic branch only
- branch-aware feedforward model for the plastic part
- inputs:
  - reduced parameters
  - trial stress / trial invariants
  - ordered principal trial quantities
  - spectral-gap indicators
- outputs:
  - branch logits for plastic branches
  - plastic stress correction in principal or invariant space

Training recipe:

- heavy long-horizon training, like the successful `cover_layer` run
- branch-balanced or hard-branch-upweighted batches
- Huber or MAE stress loss first
- branch cross-entropy always on
- add yield residual only after the base model is stable

Why this is next:

- it is the lowest-risk way to import the report’s main advice
- it preserves the exact part of the constitutive law that is easiest and safest to keep analytic
- it targets our actual failure mode: hard plastic branches and solver-critical neighborhoods

Experiments:

1. `exact_elastic + learned_plastic_raw_branch`
2. `exact_elastic + learned_plastic_principal_branch`
3. same models with auxiliary invariant loss

Exit criterion:

- beats the current heavy `cover_layer` baseline on real test
- especially on `left_edge` and `right_edge`

### Phase 3 — Expand From Cover Layer to All Real Material Families

Purpose:
- stop optimizing one material in isolation and test whether the same approach generalizes across the actual problem

Tasks:

1. Repeat Phase 2 for:
   - `general_foundation`
   - `weak_foundation`
   - `general_slope`
   - `cover_layer`
2. Train per-material models first.
3. Compare against one shared model with a material-id or reduced-parameter conditioning input.
4. Keep the same reporting format for all four.

Decision rule:

- if per-material models win clearly, deploy per-material
- if one shared model stays close, prefer it for implementation simplicity

Exit criterion:

- one clear decision on per-material vs shared deployment

### Phase 4 — Solver Integration Benchmark

Purpose:
- answer the real project question: does the surrogate preserve the limit-analysis result well enough

Tasks:

1. Insert the best checkpoint into the constitutive call wrapper.
2. Start with one simple 3D case first.
3. Compare against the exact constitutive law on:
   - final SRF / factor of safety
   - global convergence behavior
   - local constitutive failures
   - iteration counts
   - runtime
   - displacement / stress field differences if available

This phase is mandatory. Pointwise stress quality alone is not enough.

Exit criterion:

- either the surrogate is solver-acceptable on the benchmark, or we have a precise failure mode to target next

### Phase 5 — Return-Mapping-Assisted Learning

Purpose:
- move from better regression to better local constitutive robustness

This is the strongest next method from the paper review for our codebase.

Model family:

- `NN_f(x)` predicts yield-related scalar information
- `NN_n(x)` predicts correction direction / gradient information
- a local closest-point or cutting-plane-like corrector uses those learned quantities inside a classical solve

Why this phase comes after Phase 4:

- if the direct hybrid surrogate already works in the solver, this phase may not be needed immediately
- if solver robustness is still the bottleneck, this is the most justified upgrade

Exit criterion:

- improved solver behavior near branch transitions without giving up stress accuracy

### Phase 6 — Research-Grade Extensions

Only after the above is stable.

Candidates:

1. EPNN-like decomposition:
   - output physically meaningful correction pieces instead of one monolithic stress head
2. si-PiNet-style constrained stress integration:
   - treat the update as admissible stress search
3. thermodynamic / dual-potential models:
   - only if we need a second-generation architecture

These are promising, but they are not the shortest path to the immediate project goal.

## What We Should Not Do Next

Based on both the literature and our own results, the next step should not be:

- another blind width/depth sweep with the same plain output head
- RNNs for the current elastic-perfectly-plastic local map
- PDE-level PINNs
- single-integration-point `B`-copula work without stronger FE compatibility data

## Priority Order

If we want the highest expected payoff per unit effort, the order should be:

1. dataset schema upgrade with trial / invariant / branch labels
2. exact elastic + learned plastic branch-aware model
3. per-material rollout on all real families
4. in-solver benchmark
5. return-mapping-assisted surrogate

## Success Criteria

We should call the next version successful only if it satisfies both:

Pointwise:

- lower real-test MAE / RMSE than the current best baseline
- materially lower error on `left_edge` and `right_edge`
- low yield-violation or admissibility-error rate once that metric is added

Solver-level:

- same or very close SRF / factor-of-safety result
- no major increase in solver failures or nonlinear iterations
- actual runtime benefit or at least a credible path to it

## Would Real `U` and `B` Data Help?

Yes. It would help a lot.

Not because we need it for the current direct real-data training baseline. We do not. The current best cover-layer model already proves we can learn useful surrogates from real `E -> S` data alone.

It would help because it unlocks three things we cannot do well today:

1. **FE-compatible synthetic augmentation**
   Right now our synthetic `E` generators are still approximations of the real strain distribution.
   With real nodal displacements `U` and local strain-displacement operators `B`, we could generate synthetic strains through the actual FE map:
   `E = B U`
   instead of sampling directly in strain space.

2. **Element-level, not pointwise, kinematic structure**
   The current failed single-IP copula attempt showed exactly why this matters.
   One integration point alone under-identifies the local displacement pattern.
   Real `U` and `B` data would let us fit or replay whole compatible element deformation patterns.

3. **Better solver-facing validation**
   With `U` and `B`, we can test not only whether stress is correct for isolated `E`, but whether the surrogate behaves correctly on actual deformation states produced by the FE discretization.

### What Exact `U` / `B` Data Would Be Most Valuable

Best case:

- per element
- local nodal displacement vector `U`
- integration-point `B` blocks
- element geometry / Jacobian information
- material id
- integration-point strain `E`
- exact constitutive outputs:
  - stress
  - branch
  - tangent if available

Second-best:

- enough information to reconstruct `E = B U` for whole element blocks across all integration points

Less useful but still helpful:

- only isolated `U` and `E` without clear element/block identity

### What `U` / `B` Data Would Change in the Plan

If we get good real `U` / `B` data, I would insert a new track between Phase 1 and Phase 2:

- build an FE-compatible displacement generator
- generate synthetic but kinematically valid `E`
- use that for augmentation around real element deformation patterns

That would likely improve the hard-branch generalization more than another plain network-size sweep.

## Recommended Immediate Next Move

The best next concrete step is:

1. upgrade the dataset schema with trial / invariant / branch labels
2. implement `exact elastic + learned plastic` as the next model family
3. keep the current heavy cover-layer run as the baseline to beat

If real `U` and `B` data are available soon, we should collect them now and fold them into the dataset upgrade instead of waiting until later.
