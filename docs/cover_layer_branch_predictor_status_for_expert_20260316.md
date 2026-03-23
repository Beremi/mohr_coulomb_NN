# Cover-Layer Branch Predictor Status Report

## 1. What We Are Trying To Do

We are building a surrogate that predicts the **constitutive branch** of a 3D Mohr-Coulomb update for the **cover layer** only.

Branch classes:

- elastic
- smooth
- left_edge
- right_edge
- apex

The final use case is branch prediction at constitutive-call level, with training done on **synthetic data only** and validation done on **real exported solver data**.

At the constitutive level, the exact branch is a deterministic function of:

- engineering strain `E` at one integration point
- reduced constitutive parameters:
  - `c_bar`
  - `sin_phi`
  - `G`
  - `K`
  - `lambda`

So the exact mapping we are approximating is:

`(E, reduced_material) -> branch`

We also explored direct raw-element models using coordinates and displacements, but those were much less successful than the pointwise constitutive-feature route.

## 2. Real Data Available

We have a full export of real constitutive calls from the solver.

Key facts:

- 554 captured constitutive calls
- 18419 quadratic tetrahedra in the full mesh
- 10 nodes per tetrahedron
- 11 integration points per tetrahedron
- cover-layer elements per call: 5124
- cover-layer integration-point rows per call: 5124 x 11 = 56364

The real export contains enough information to reconstruct local strains exactly from the FE state. The export is self-consistent.

Important practical detail:

- the raw material family is fixed to the cover layer
- but the **reduced constitutive state is not fixed**
- the reduced parameters vary along the strength-reduction path

This turned out to matter a lot. Earlier models that ignored reduced material and used only strain were structurally ambiguous.

## 3. Real Evaluation Protocol

We are not currently evaluating on all 111 real validation calls and all 111 real test calls in one shot. For tractability, we use a fixed representative slice:

- 4 real validation calls
- 4 real test calls
- up to 128 cover-layer elements per selected call

So each real holdout slice is:

- 512 elements
- 5632 integration-point labels

This means current real metrics are meaningful and stable enough for iteration, but they are still based on a **representative slice**, not the full real population.

## 4. Synthetic Data Generation: Current Best Route

The early synthetic generator was based on local perturbations of real FE element states. That route learned the synthetic task well but transferred poorly to real data.

The current best route is a **principal-space hybrid generator**. That was the main breakthrough.

The current synthetic generation pipeline is:

1. Start from real cover-layer constitutive states.
2. Convert each real strain state into ordered principal coordinates.
3. Build constitutive sampling coordinates from:
   - scaled volumetric coordinate
   - scaled principal-gap coordinates
4. Keep the real reduced material row from the seed state.
5. Generate synthetic states using a mixture of:
   - branch-balanced principal resampling
   - analytic sampling near the smooth/right boundary
   - tail extension for large-strain states
6. Reconstruct engineering strain from the sampled principal state.
7. Run the exact constitutive update on the synthetic strain and reduced material.
8. Use the exact returned branch as the label.

This is still synthetic-only training. No real labels are used for optimization.

## 5. Model Family That Actually Works

The successful model family is **pointwise**, not element-contextual.

Input features per integration point:

- transformed engineering strain
- transformed elastic trial stress
- reduced material features

In practice this is a 17-dimensional structured feature vector.

Classifier structure:

- hierarchical classifier
- first head: elastic vs plastic
- second head: smooth vs left_edge vs right_edge vs apex

The direct raw-element route:

- transformed element coordinates + transformed nodal displacements -> 11 x 5

was explored extensively and did not generalize well even inside the synthetic domain. So the current working path is the pointwise constitutive-feature model.

## 6. What Improved Over Time

The progression was roughly:

### Stage A: strain-only / weak synthetic coverage

These models learned synthetic data but transferred badly to real data.

Typical failure pattern:

- elastic poor or unstable when reduced material was omitted
- smooth often confused with right_edge
- real performance far below synthetic performance

### Stage B: add constitutive structure to the input

Adding:

- reduced material
- trial stress

was a major step forward. This fixed a lot of the ambiguity that was coming from varying reduced constitutive state.

### Stage C: expert-inspired principal-space sampler

This was the largest improvement.

Compared to the previous FE-seed local-noise route, the expert-inspired principal-space generator improved real test performance from:

- accuracy: 0.7058 -> 0.8865
- macro recall: 0.6843 -> 0.8750

This is the strongest evidence so far that **domain coverage really was the main bottleneck**.

### Stage D: larger model and longer training

Moving from the smaller successful model to a larger hierarchical model improved the result further:

- real test accuracy: 0.8865 -> 0.8928
- real test macro recall: 0.8750 -> 0.8828

This helped, but much less than the generator change.

### Stage E: post-training and continuation

We tried:

- repeated Adam continuation
- repeated LBFGS continuation
- replay-bank continuation
- width inflation from 1024 to 2048 with exact function preservation at initialization

These runs could find slightly better internal checkpoints, but the gains were incremental. The main result was that the model family is no longer obviously undertrained. The dominant remaining issue is not raw optimization.

## 7. Current Best Results

There are two important numbers:

### Best stable saved model from the main successful line

On the fixed real test slice:

- accuracy: 0.8928
- macro recall: 0.8828

Per-branch recall:

- elastic: 1.0000
- smooth: 0.7500
- left_edge: 0.9160
- right_edge: 0.8680
- apex: 0.8799

Synthetic test on the same model:

- accuracy: 0.9658
- macro recall: 0.9655

### Best observed internal checkpoint so far

Inside a later widened post-train run, the best internal point reached:

- real test accuracy: 0.9006
- real test macro recall: 0.8915

Per-branch recall at that internal point:

- elastic: 1.0000
- smooth: 0.7844
- left_edge: 0.9135
- right_edge: 0.8714
- apex: 0.8880

So we are now at roughly:

- 90% real test accuracy
- 89% real test macro recall

on the fixed real holdout slice, with training still synthetic-only.

## 8. How Far We Are From “Replicating Real Data”

If “replicating real data” means:

- train only on synthetic data
- transfer well to real holdout constitutive calls

then we are now **substantially successful but not finished**.

What is already true:

- the task is learnable
- synthetic-only training can transfer strongly to real data
- we are no longer in the regime of complete synthetic-to-real failure
- elastic is essentially solved on the current holdout
- left_edge, right_edge, and apex are fairly strong

What is not yet true:

- branch prediction is not yet near-perfect on the real holdout
- smooth is still the weakest major branch
- some post-train procedures improve synthetic metrics without improving real metrics
- the best internal checkpoints are not always the final accepted endpoints

So we are not at “solved,” but we are far past the earlier stage where the synthetic generator simply missed the real domain entirely.

## 9. Current Blockers

### Blocker 1: remaining domain-coverage mismatch

The principal-space generator was the main unlock, but coverage is still not perfect.

The strongest remaining suspicion is that the generator is still not matching:

- left-side boundary geometry as well as right-side geometry
- edge/apex neighborhoods as well as smooth/right neighborhoods
- the exact local branch geometry around the hardest real cases

### Blocker 2: synthetic-best and real-best are not the same checkpoint

This is now a repeated pattern.

We often see:

- the best synthetic checkpoint
- the best real validation checkpoint
- the best real test checkpoint

at different points in the same run.

That means post-training is not only an optimization problem. It is also a **selection problem**.

### Blocker 3: gentle post-training tends to drift away from the best basin

The recent gated heavy post-train campaign tested a careful “stabilize first, then sweep” plan.

Result:

- the very first stabilization phase did not improve the baseline
- it slightly degraded the real metrics
- the gate stopped the campaign before wasting more budget

This suggests the current basin is already fairly sharp, and not all continuation strategies are helpful.

### Blocker 4: the current successful model is still pointwise

The current best classifier predicts one integration point at a time.

It does not use:

- 11-point element context
- element-level consistency
- cross-point pattern information

That may matter for the remaining hard ambiguous cases.

### Blocker 5: evaluation is still on a representative real slice

Current real metrics are on a fixed, representative holdout slice, not the full real population.

So while the numbers are strong, we are not yet claiming:

- 90%+ behavior over all real calls and all real cover-layer points

without running a broader validation pass.

## 10. What We Think Is Already Proven

These conclusions now look solid:

1. The branch map is learnable.
2. Synthetic-only training can transfer meaningfully to real data.
3. Reduced constitutive state must be included, even for a fixed raw material family.
4. Domain coverage mattered more than network size in the major transition from poor to strong real performance.
5. Principal-space, branch-geometry-aware sampling is much better than naive FE-state perturbation for this task.
6. The current limitation is no longer “the network just needs to be trained longer.”

## 11. What Kind of Expert Advice Would Be Most Valuable Now

The highest-value advice would be about the **remaining synthetic domain design**, not generic optimizer tuning.

Most useful questions:

1. How should we sample the left-side branch geometry and apex neighborhoods so they are as well covered as the smooth/right side?
2. Is there a better way to define a synthetic distribution over constitutive states than the current principal-space hybrid mixture?
3. Should the generator be explicitly conditioned on branch margins or local branch-surface distances?
4. Are we now at the point where element-contextual modeling is the right next move, rather than more pointwise sampler refinement?
5. Is there a better checkpoint-selection criterion than synthetic validation score if the real-best and synthetic-best checkpoints keep diverging?

## 12. Bottom Line

The project is no longer blocked by “can this work at all?”

It can.

We now have a synthetic-only branch predictor that reaches about:

- 0.90 real-test accuracy
- 0.89 real-test macro recall

on a fixed representative real holdout slice.

The main remaining gap is not raw model capacity. The main remaining gap is:

- how to cover the remaining real branch geometry better
- and how to select or stabilize the best real-performing checkpoint once the model reaches those regions

So the current state is:

- **promising and already useful**
- **not yet fully replicated on real data**
- **most likely limited by remaining domain-coverage and checkpoint-selection issues**
