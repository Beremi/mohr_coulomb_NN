# Cover Layer Domain-Coverage Brief for External Review

## Purpose of This Brief

This document is meant to be passed to an external expert who can help us reason about **domain coverage** for the cover-layer constitutive branch-prediction problem.

The current bottleneck is no longer simple optimization. The branch predictors can now learn the synthetic task well. The main remaining question is whether our **synthetic element-state generator** is covering the right part of the real deformation manifold, especially around the `smooth / right_edge` region.

## The Exact Model We Are Trying To Approximate

For the current branch-only work, the target is the **branch decision** of the exact 3D Mohr-Coulomb constitutive update at each integration point.

Current branch set:

- `elastic`
- `smooth`
- `left_edge`
- `right_edge`
- `apex`

At the exact constitutive level, the branch is determined by:

1. local engineering strain `E` at one integration point
2. reduced constitutive parameters
   - `c_bar`
   - `sin_phi`
   - `shear = G`
   - `bulk = K`
   - `lame = lambda`

These come from the exact constitutive operator in [mohr_coulomb.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/mohr_coulomb.py), which mirrors the MATLAB/C Mohr-Coulomb implementation.

So the branch map we are approximating is effectively:

`(E, reduced_material) -> branch`

For the synthetic generation path, the strains themselves come from FE-compatible element states:

`(local_coords, local_displacements, reduced_material) -> B(coords) * u -> E_11 -> exact_branch_11`

## What “One Material” Means Here

The **raw material family** is fixed to `cover_layer`.

Raw cover-layer material in the repo:

- `c0 = 15`
- `phi = 30 deg`
- `psi = 0 deg`
- `young = 10000`
- `poisson = 0.33`

But the actual constitutive calls in the exported data use **reduced parameters along the strength-reduction path**, not a single fixed reduced state.

That means:

- raw material family is fixed
- reduced constitutive state is **not** fixed across calls

This turned out to matter a lot. Earlier branch predictors that ignored reduced material and used only `E` were structurally ambiguous.

## Real Data We Have

Primary source:

- [constitutive_problem_3D_full.h5](/home/beremi/repos/mohr_coulomb_NN/constitutive_problem_3D_full.h5)

Inspection summary:

- [full_export_inspection.md](/home/beremi/repos/mohr_coulomb_NN/docs/full_export_inspection.md)

### Full-export contents

The full export contains:

- `554` captured constitutive calls
- full displacement vector `U` for each call
- full sparse global `B` operator
- mesh coordinates and connectivity
- per-integration-point strains `E`
- per-integration-point stresses `S`
- reduced constitutive parameters per integration point
- `DS` tangents for `157 / 554` calls

Key mesh facts:

- `18419` P2 tetrahedra
- `10` nodes per tetrahedron
- `11` integration points per tetrahedron
- `27605` nodes total
- `82815` displacement DOFs total

The export is self-consistent:

- `E = B @ U` reconstructs exactly

### Cover-layer subset

Cover layer is one material family in the full mesh.

Counts:

- cover-layer elements per call: `5124`
- cover-layer integration-point rows per call: `5124 * 11 = 56364`

Element-level grouping is available from the full export through:

- [full_export.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/full_export.py)

Each grouped cover-layer element block contains:

- `local_coords`: `(10, 3)`
- `local_displacements`: `(10, 3)`
- `strain_eng`: `(11, 6)`
- `stress`: `(11, 6)`
- `material_reduced`: one reduced-material row per element
- `branch_id`: `(11,)`, recomputed from the exact constitutive operator

### Real split used in the branch-predictor work

Frozen call split:

- [call_splits.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json)

Counts:

- `generator_fit`: `332` calls
- `real_val`: `111` calls
- `real_test`: `111` calls

Important detail:

- the current branch-predictor experiments do **not** use all cover-layer elements from those calls at once
- for tractability, they sample up to `128` elements per selected call

In the current best branch-classifier experiments:

- synthetic train seed bank uses `24` calls from `generator_fit`
- synthetic eval seed bank uses `8` disjoint calls from `generator_fit`
- real diagnostics use `4` representative calls from `real_val`
- real diagnostics use `4` representative calls from `real_test`

So the current real diagnostic set is:

- `4` calls
- `128` elements per call
- `512` elements total
- `512 * 11 = 5632` integration-point labels

## What Synthetic Data We Generate

Current synthetic-generation code:

- [cover_branch_generation.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/cover_branch_generation.py)

Current trainer:

- [train_cover_layer_strain_branch_predictor_synth_only.py](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/train_cover_layer_strain_branch_predictor_synth_only.py)

### Current synthetic generator family

The current best synthetic generator is:

- `seeded_local_noise_branch_balanced`

This is **not** a global probabilistic model over all element states.
It is a **real-seed local perturbation generator**.

### Step-by-step generator pipeline

#### 1. Build a real seed bank

From selected real cover-layer calls, collect element-level blocks:

- local coordinates
- local nodal displacements
- exact branch patterns
- reduced material rows

This is done by `collect_blocks(...)`.

#### 2. Canonicalize each P2 element state

Canonicalization is done in [full_export.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/full_export.py) via `canonicalize_p2_element_states(...)` and removes nuisance variation:

- translate element to a local origin
- build a local orthonormal basis from the 4 corner nodes
- scale by a characteristic element length
- express nodal displacements in that local basis

Stored per seed:

- canonicalized local displacements
- local basis
- characteristic length

#### 3. Assign seed weights

The generator stores two seed-selection distributions:

- `uniform`
- `branch_balanced`

`branch_balanced` is built by giving more weight to elements whose 11-point branch pattern contains rarer branches.

#### 4. Sample seed elements

For each synthetic element:

- draw one real seed element from the seed bank
- selection is either `uniform` or `branch_balanced`

#### 5. Add local noise in canonical displacement space

For each seed element, perturb the canonicalized nodal displacement field.

The local perturbation scale is:

- `max(0.10 * global_disp_std, 0.20 * abs(seed_disp))`

Then Gaussian noise is applied with an outer `noise_scale`.

Typical noise scales used by the current trainer:

- easy cycle: `0.10`
- match cycle: `0.20`
- coverage cycle: mixture of `0.20` and `0.25`
- hard cycle: mixture of `0.20` and `0.30`

#### 6. Map back to physical element displacement

The perturbed canonical displacement is converted back using:

- stored element basis
- stored characteristic length

This gives a synthetic physical nodal displacement field for that element.

#### 7. Reject obviously invalid deformations

Current realism filter:

- positive corner-volume check using [positive_corner_volume_mask](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/fe_p2_tetra.py)
- minimum corner-volume ratio: `0.01`

What this rejects:

- inverted corner tetrahedra
- near-collapsed corner tetrahedra

What it does **not** yet reject:

- all possible higher-order P2 distortions that might still be unrealistic
- all possible Jacobian pathologies at every point inside the curved element

#### 8. Compute exact strains from FE kinematics

For accepted synthetic elements:

- build local `B` blocks from the real local coordinates
- compute `E = B * u` at all `11` integration points

So the synthetic strains are FE-compatible with that local element state.

#### 9. Exact-label the branches

The exact constitutive operator is called with:

- synthetic `E`
- the synthetic element’s reduced material row, repeated across the `11` integration points

This produces exact labels:

- `elastic / smooth / left_edge / right_edge / apex`

#### 10. Flatten to pointwise branch-classification samples

For the current branch-predictor family, each synthetic element contributes:

- `11` pointwise samples

The current best classifier is still **pointwise**, not element-contextual.

## What the Current Best Branch Predictor Actually Uses

Current best model card:

- [cover_layer_branch_predictor_trial_raw_material.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_trial_raw_material.md)

Current best checkpoint summary:

- [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_trial_raw_material_20260315/summary.json)

### Input representation

Current winner does **not** use raw element coords/displacements directly.

It uses per-integration-point structured features:

- `asinh(E)` in engineering-strain Voigt form
- `asinh(sigma_trial / c_bar)`
- reduced material features:
  - `log(c_bar)`
  - `atanh(sin_phi)`
  - `log(G)`
  - `log(K)`
  - `log(lambda)`

So the current best classifier approximates:

`structured_trial_features(E, reduced_material) -> branch`

not

`(coords, disp) -> branch`

### Current best classifier performance

Winner metrics:

- synthetic test accuracy / macro recall: `0.9845 / 0.9875`
- real test accuracy / macro recall: `0.6861 / 0.6601`

Real per-branch recall:

- elastic: `0.9603`
- smooth: `0.1959`
- left_edge: `0.7540`
- right_edge: `0.6667`
- apex: `0.7236`

So `smooth` is the main unresolved branch.

## Current Synthetic Benchmark Sizes

For the current best branch-predictor runs:

- synthetic train: regenerated every epoch, `16384` elements
- synthetic val: fixed, `16384` elements
- synthetic test: fixed, `16384` elements

Since each element has `11` integration points:

- synthetic val/test each contain `180224` pointwise branch labels

## Strongest Current Evidence of Domain Mismatch

### 1. Branch-frequency mismatch is still large

From [coverage_summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/coverage_summary.json):

Real branch fractions on the checked slice:

- elastic: `0.2163`
- smooth: `0.1815`
- left_edge: `0.2420`
- right_edge: `0.1623`
- apex: `0.1980`

Current synthetic generator branch fractions:

- elastic: `0.0310`
- smooth: `0.2938`
- left_edge: `0.2724`
- right_edge: `0.0812`
- apex: `0.3216`

This is one of the biggest remaining mismatches:

- synthetic **underproduces elastic**
- synthetic **underproduces right_edge**
- synthetic **overproduces apex**
- synthetic **overproduces smooth**

### 2. Upper strain tail is still undercovered

From the same coverage summary:

- real strain-norm p95: `24.4033`
- synthetic strain-norm p95: `19.6969`

So even after generator correction, the synthetic generator is still short in the upper tail.

### 3. Failed real elements are often outside the generator’s practical support

From [cover_layer_failed_element_support_report.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_failed_element_support_report.md):

Compared with successful real elements, failed real elements are much more likely to lie outside the generator’s practical seed-neighborhood support in canonical displacement space.

Key numbers:

- failed elements outside synthetic seed-neighborhood p95: `0.7733`
- successful elements outside synthetic seed-neighborhood p95: `0.1314`

For `smooth` failures specifically:

- outside synthetic seed-neighborhood p95: `0.7121`

Interpretation:

- many of the hard real elements are not near the actual local displacement neighborhoods the generator is sampling

### 4. But some failed real elements are still near the synthetic strain cloud

Also from the failed-support analysis:

- failed elements outside synthetic strain-NN p95: `0.1227`

So many failed real elements are still near **some** synthetic strain state.

Interpretation:

- the problem is not only “we never sample that region at all”
- it is also “the sampled local branch geometry is wrong there”

### 5. Historical evidence: real `smooth` often looks like synthetic `right_edge`

From the earlier failure analysis on the older strain-only branch classifier:

- true real `smooth` points were predicted as `right_edge` about `81%` of the time

This older result should not be used as a metric for the current best model, but it is still a useful geometric clue:

- the synthetic generator has likely been getting the `smooth / right_edge` neighborhood wrong for some time

### 6. Reduced-material omission was a real mismatch source

Earlier strain-only classifiers ignored reduced material completely.

That turned out to be a major problem because:

- raw family is fixed
- reduced state is not fixed

Adding reduced material plus trial stress immediately fixed:

- elastic recall, from `0.0000` to `0.9603`

So one class of domain mismatch was not purely geometric; it was also **missing constitutive-state information**.

## What We Think The Main Domain-Coverage Problems Are Now

These are the current best hypotheses.

### A. Synthetic generator mass is still in the wrong neighborhoods

Because the generator is a local-noise perturbation around seed elements, it is only strong where:

- the seed bank contains the right real neighborhoods
- the local noise scale is large enough
- the validity filter does not reject too much of the needed region

The failed-support analysis suggests many hard real elements are still outside those practical neighborhoods.

### B. The `smooth / right_edge` boundary geometry is still wrong

This is probably the single most important issue now.

Evidence:

- `smooth` remains the weakest real branch in the current best model
- earlier analyses showed real `smooth` mapping into `right_edge`
- broad strain coverage alone is not enough

So the missing piece is probably not just more samples, but better samples in the **correct local branch geometry**.

### C. Branch balancing is not the same as difficulty balancing

Current seed weighting balances branches by frequency, but it does not explicitly target:

- near-boundary states
- hard `smooth`
- `smooth/right_edge` transitions
- hard upper-tail regimes

So the generator can be branch-balanced and still badly miss the cases that matter most.

### D. The validity filter is still fairly weak

Current realism filter only checks corner-volume positivity with a minimum ratio.

Possible issue:

- some accepted synthetic deformations may be “valid enough” geometrically but still not representative of real solver states
- some important hard real states may require a generator that respects stronger FE realism constraints

### E. Current branch classifier is pointwise

The current best model predicts branch one integration point at a time.

So it does **not** use:

- the 11-point element pattern
- cross-IP context
- local branch consistency across one element

This may matter especially for ambiguous `smooth` states, although the current evidence says generator/domain mismatch is still the larger issue.

## What We Need Help With From an External Expert

The most useful expert help would be on these questions:

1. Given the current real export, what is the best way to define a **realistic local deformation distribution** for cover-layer P2 tetrahedra?
2. Is the current canonical local-noise generator too weak because it is only a **seed-neighborhood perturbation model**?
3. How should we target the `smooth / right_edge` region explicitly?
4. Should realism checking use stronger FE constraints than corner-volume positivity?
5. Would a better generator be:
   - call-regime-conditioned
   - branch-margin-conditioned
   - element-pattern-conditioned
   - or based on direct modeling of canonical `U` distributions rather than local Gaussian noise?

## Key Files for the Expert

Real data and FE reconstruction:

- [constitutive_problem_3D_full.h5](/home/beremi/repos/mohr_coulomb_NN/constitutive_problem_3D_full.h5)
- [full_export.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/full_export.py)
- [full_export_inspection.md](/home/beremi/repos/mohr_coulomb_NN/docs/full_export_inspection.md)

Synthetic generation:

- [cover_branch_generation.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/cover_branch_generation.py)

Exact constitutive operator:

- [mohr_coulomb.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/mohr_coulomb.py)

Current best branch predictor:

- [train_cover_layer_strain_branch_predictor_synth_only.py](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/train_cover_layer_strain_branch_predictor_synth_only.py)
- [cover_layer_branch_predictor_trial_raw_material.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_trial_raw_material.md)
- [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_trial_raw_material_20260315/summary.json)

Mismatch evidence:

- [cover_layer_branch_predictor_failure_analysis.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_failure_analysis.md)
- [cover_layer_failed_element_support_report.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_failed_element_support_report.md)
- [cover_layer_branch_predictor_report_driven_iteration.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_report_driven_iteration.md)

## Bottom Line

The current best branch predictor is now good enough to show that:

- the task is learnable
- constitutive structure in the input matters
- reduced material state matters

But the synthetic generator is still not matching the real domain well enough, especially for:

- the upper strain tail
- the branch-frequency mix
- the `smooth / right_edge` local geometry

That is where we need the next help.
