# Cover Layer Single-Material Surrogate Workplan

This document is the concrete next-step plan for replicating the constitutive surrogate for **one fixed material only**, namely the `cover_layer`, treating the constitutive operator as:

- input: engineering strain `E in R^6`
- output: stress `S in R^6`

The scope here is intentionally narrow:

- one material only
- first target is the operator `E -> S`
- we use everything we have now:
  - the current best cover-layer baseline
  - the new full export with `U` and `B`
  - the paper-guided ideas from [report.md](/home/beremi/repos/mohr_coulomb_NN/report.md)

This plan is designed to be executable, not aspirational.

## Fixed Material Definition

Cover-layer raw material parameters in the repository are:

- `c0 = 15`
- `phi = 30 deg`
- `psi = 0 deg`
- `young = 10000`
- `poisson = 0.33`

Reference:
- [real_materials.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/real_materials.py)

In the heterogeneous mesh export, the cover layer corresponds to:

- `material id = 0`

Reference:
- [full_export_inspection.md](/home/beremi/repos/mohr_coulomb_NN/docs/full_export_inspection.md)

## Current Baseline To Beat

Current best cover-layer checkpoint:

- [best.pt](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/best.pt)

Current best real-test metrics:

- stress MAE: `1.4951`
- stress RMSE: `9.7174`
- stress max abs: `526.0352`
- branch accuracy: `0.9577`

Reference:
- [cover_layer_fitted_refresh_heavy_w1024_d6.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_fitted_refresh_heavy_w1024_d6.md)

Any new experiment should be compared against this, even if the new model itself does not explicitly output branch.

## Main Strategy

The literature and our own results point to this progression:

1. keep a strong direct `E -> S` baseline
2. improve the structure of the operator, not only width/depth
3. use the new `U/B` export to generate FE-compatible training states
4. only then move toward more hybrid return-mapping-assisted models

So the single-material plan is:

- first stabilize the best possible direct cover-layer operator
- then inject structure:
  - exact elastic branch
  - plastic correction target
  - branch awareness
  - principal/invariant representation
- then use `U/B` data for FE-compatible augmentation

## Dataset Tracks We Should Maintain

We should keep four dataset tracks for the cover layer.

### Track A — Real exact-domain data

This is the strongest fixed reference set:

- real cover-layer `E`
- exact `S` recomputed from the Python constitutive law

Use:

- validation
- testing
- sanity benchmark

Expected role:

- stable anchor dataset
- best way to compare model families fairly

### Track B — Real FE-trace data from full export

From the new full export:

- cover-layer integration-point `E`
- corresponding `U`
- global `B`
- exact `S`
- optional `DS` for subset of calls

Use:

- FE-compatible data analysis
- `U/B`-driven augmentation
- solver-trace validation

Expected role:

- bridge between isolated constitutive regression and solver-compatible deformation patterns

### Track C — Synthetic holdout data

Fixed synthetic test set generated once and never trained on.

Use:

- track whether the model is learning the intended synthetic operator family
- detect when a model is over-specialized to real traces only

Expected role:

- secondary test, not the main success criterion

### Track D — Refreshed training stream

The train split that changes during training.

Candidates:

- direct strain-space fitted generator
- `U/B`-pushforward generator
- hybrid mixture of real replay + synthetic augmentation

Expected role:

- make training broad enough to generalize but still centered on real cover-layer states

## Experiment Ladder

Each experiment has:

- purpose
- implementation
- success target
- what to do immediately after it finishes

### Experiment 0 — Build the cover-layer full-export dataset

Purpose:

- make the new full export usable for single-material experiments
- ensure we can isolate the cover-layer states consistently

Implementation:

1. Extract all cover-layer integration-point states from [constitutive_problem_3D_full.h5](/home/beremi/repos/mohr_coulomb_NN/constitutive_problem_3D_full.h5).
2. Store:
   - `E`
   - `S`
   - `U`
   - mapping from integration points to elements
   - `B` access metadata
   - `DS` where available
   - reduced constitutive parameters
3. Recompute branch labels and save them.
4. Build fixed train/val/test splits by call, not just by point.

Expected result:

- one clean cover-layer full dataset
- exact consistency between exported `E/S` and recomputed constitutive labels

Go / no-go:

- proceed only if call-level split and branch recomputation are consistent

Next step if successful:

- run Experiment 1

Next step if not successful:

- fix extraction/indexing before any new training

### Experiment 1 — Reproduce the current best direct baseline on the new split

Purpose:

- establish a fresh baseline on the new full-export-derived cover-layer split

Implementation:

- same model family as the current winner:
  - raw branch-aware feedforward model
  - heavy long training
  - refreshed synthetic training stream
- evaluate on the new fixed cover-layer real split

Expected result:

- similar order of accuracy to the current best:
  - MAE roughly `1.5 - 2.0`
  - RMSE roughly `9 - 11`
  - branch accuracy above `0.94`

Next step if successful:

- freeze this as the new single-material baseline for all following experiments

Next step if worse than expected:

- inspect split mismatch before changing architecture

### Experiment 2 — Exact elastic + learned plastic, still in raw space

Purpose:

- reduce the learning burden without changing too many things at once

Implementation:

1. Compute elastic trial stress exactly.
2. If the sample is elastic:
   - return exact elastic stress directly
3. If the sample is plastic:
   - train the model only on the plastic correction
   - output either:
     - `Delta S = S - S_trial`
     - or normalized `Delta S`

Loss:

- Huber or MAE on plastic samples
- branch CE on plastic branch subclasses

Expected result:

- lower elastic error immediately
- better hard-branch stability than the plain full-stress model
- small but real gain in overall RMSE

Target:

- beat the baseline on real RMSE
- reduce `left_edge` and `right_edge` MAE

Next step if successful:

- keep exact elastic handling for all later experiments

Next step if only elastic improves but hard branches do not:

- move to principal/invariant outputs in Experiment 3

### Experiment 3 — Exact elastic + learned plastic in principal space

Purpose:

- align the surrogate with the constitutive geometry

Implementation:

Inputs:

- ordered principal strains
- trial principal stresses
- invariants such as `p`, `q`, and a Lode-angle-like coordinate
- spectral gaps

Outputs:

- plastic correction in ordered principal stresses
- branch logits

Then:

- reconstruct the global stress tensor analytically

Expected result:

- better edge/apex behavior than raw-space plastic correction
- improved tail behavior
- lower per-branch MAE on `left_edge` and `right_edge`

Target:

- beat Experiment 2 on hard branches
- maintain or improve total MAE and RMSE

Next step if successful:

- declare this the main structured baseline

Next step if not successful:

- inspect whether the failure comes from representation or from the training distribution

### Experiment 4 — Add yield / invariant auxiliary losses

Purpose:

- stabilize constitutive geometry without committing to a full hybrid return-mapping model yet

Implementation:

Keep the Experiment 3 model, and add:

- invariant loss on predicted `p/q/theta`
- light yield-residual penalty

Important:

- use these as small auxiliary terms only
- do not let them dominate stress fitting initially

Expected result:

- smaller admissibility drift
- smoother behavior near branch boundaries
- possibly similar MAE but better tail and solver-facing behavior

Target:

- same or better MAE/RMSE
- lower hard-branch max error
- lower yield-violation rate once tracked

Next step if successful:

- keep auxiliary losses in the default recipe

Next step if metrics degrade:

- roll back to Experiment 3 and keep the losses out of the production baseline

### Experiment 5 — `U/B`-pushforward augmentation

Purpose:

- replace strain-space-only augmentation with FE-compatible deformation augmentation

Implementation:

1. Fit a distribution over cover-layer displacement patterns `U` from the full export.
2. Generate new synthetic displacements in `U` space.
3. Push them through the real `B` operator:
   - `E = B U`
4. Filter:
   - element inversion / negative volume
   - extreme outliers
   - optionally branch quotas
5. Label with the exact constitutive operator.

Training setup:

- mix real replay and `U/B`-generated synthetic training states
- keep fixed real test

Expected result:

- improved generalization on real FE-trace data
- smaller gap between synthetic and real test behavior
- better coverage of physically compatible strain states

Target:

- improve real test more than synthetic test
- reduce hard-branch errors without inflating the average elastic-region error

Next step if successful:

- make `U/B` augmentation the default training stream

Next step if unsuccessful:

- inspect the fitted `U` distribution before changing the network again

### Experiment 6 — Tangent-aware auxiliary training on the `DS` subset

Purpose:

- make the operator more solver-usable, even if we still optimize stress first

Implementation:

Train on the subset of calls with `DS`, using:

- stress loss on all samples
- tangent auxiliary loss on `DS`-available samples only

Do not require the network to predict a fully production-ready tangent yet.
At first, it is enough to regularize the local derivative behavior.

Expected result:

- modest change in pointwise stress metrics
- potentially better local smoothness and solver-facing robustness

Target:

- no material regression in stress MAE/RMSE
- improved derivative consistency on tangent subset

Next step if successful:

- keep tangent auxiliary supervision for solver-facing models

Next step if unsuccessful:

- drop tangent loss from the direct `E -> S` baseline and revisit later in a hybrid model

### Experiment 7 — Hold-out FE-trace generalization benchmark

Purpose:

- test whether the model is learning the operator on states that matter to the FE solve

Implementation:

Build a test split consisting of whole unseen calls from the full export.

Measure:

- stress MAE / RMSE / max abs
- per-branch error
- error vs `||U||`
- error vs stress magnitude
- error vs branch
- error vs near-yield indicator

Expected result:

- the best model should retain the improvements seen on the pointwise cover-layer test set
- if not, the issue is still FE-state mismatch, not pure constitutive approximation

Next step if successful:

- proceed to cover-layer-only solver insertion

Next step if unsuccessful:

- prioritize better `U/B` augmentation over further network changes

### Experiment 8 — Cover-layer-only solver shadow test

Purpose:

- check whether the improved operator survives contact with the actual code path

Implementation:

In the constitutive wrapper:

- replace only the cover-layer material with the surrogate
- keep the other materials exact

Compare:

- convergence behavior
- local constitutive failures
- SRF / factor of safety change
- runtime

Expected result:

- little or no deterioration in convergence
- SRF drift small enough to justify broader rollout

Target:

- no obvious solver instability
- SRF difference within an acceptable tolerance defined before the run

Next step if successful:

- replicate the same path for the other materials

Next step if unsuccessful:

- move to the hybrid return-mapping-assisted route rather than just fitting the direct operator harder

## Recommended Order

If we want the shortest high-probability path, the order should be:

1. Experiment 0
2. Experiment 1
3. Experiment 2
4. Experiment 3
5. Experiment 5
6. Experiment 7
7. Experiment 8

Experiments 4 and 6 are important, but they are second-wave refinements:

- Experiment 4 if we need constitutive-geometry regularization
- Experiment 6 if we need solver-facing derivative control

## Expected Best Outcome

If this plan works well, the likely best practical single-material model is:

- exact elastic branch
- learned plastic correction
- principal/invariant-space branch-aware head
- trained with heavy long-horizon schedule
- mixed real replay + `U/B`-pushforward augmentation

That is the most plausible route to beating the current direct baseline while staying close to the actual constitutive structure.

## What To Avoid

For the single-material cover-layer path, we should not spend more time on:

- another plain raw full-stress MLP with only width/depth changes
- single-integration-point copula augmentation without element-level `U/B` structure
- RNNs
- PDE-level PINNs

Those are lower-value than the structured operator path above.

## Success Criteria

We should call the single-material plan successful only if all three happen:

1. real test improves over the current baseline
   - especially on `left_edge` and `right_edge`
2. FE-trace holdout performance improves, not only synthetic holdout performance
3. the cover-layer-only solver shadow test remains stable

That is the bar for declaring the cover-layer surrogate ready to move from “good regression result” to “serious constitutive replacement candidate.”
