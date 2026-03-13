# Cover Layer B-Copula Notes

This note records the first FE-local displacement-generator attempt based on the upstream P2 tetrahedral `B` operator.

## What Was Implemented

- local P2 tetrahedral FE helper in `src/mc_surrogate/fe_p2_tetra.py`
- integration-point geometry pool for the cover layer from `SSR_hetero_ada_L1.h5`
- local right-inverse maps from engineering strain to a minimum-norm nodal displacement realization
- Gaussian-copula displacement samplers in `scripts/experiments/cover_layer_b_copula_refresh.py`
- negative/near-degenerate corner-volume rejection
- fixed real validation/test and fixed synthetic holdout evaluation

The current distribution-screen report is:

- `docs/cover_layer_b_copula_screen.md`

## Key Result

The FE-local `B` construction is correct, but the first integration-point-level copula generators are not yet good enough for training.

Best screened distribution mode:

- `branch_raw`

Why it looked best in screening:

- branch occupancy was close to the real test split
- it was better than the global copula variants in total screen score

Why it is still not good enough:

- its stress-space distribution is still too distorted
- training on it did not come close to the earlier exact-domain / fitted-strain baselines

## Training Attempt

Run:

- `experiment_runs/real_sim/cover_layer_b_copula_20260312/cover_raw_branch_w384_d6_branch_raw`

The run was stopped early because the real validation error was still far too large after the first small-batch stage.

Best saved checkpoint:

- `experiment_runs/real_sim/cover_layer_b_copula_20260312/cover_raw_branch_w384_d6_branch_raw/best.pt`

Evaluation of that checkpoint:

- real test stress MAE: `48.42`
- real test stress RMSE: `88.71`
- real test max abs: `870.14`
- real test branch accuracy: `0.440`
- synthetic test stress MAE: `39.49`
- synthetic test stress RMSE: `69.03`
- synthetic test max abs: `618.65`
- synthetic test branch accuracy: `0.916`

This is much worse than the previous cover-layer baselines, so this generator should not be used for further training in its current form.

## Main Insight

The weak point is not the FE `B` operator itself. The weak point is the level at which the distribution is being fitted.

The current B-copula generator still fits displacements at a single integration point. That is too underdetermined and does not preserve the element-level compatibility that the FE formulation actually imposes.

The raw export gives a stronger route:

- each captured call has shape `202609 = 18419 * 11`
- this matches the full integration-point field of the 3D mesh exactly
- cover-layer states can therefore be grouped into `5124` cover-layer elements with `11` integration points each

That means the next version should fit **one nodal displacement vector per tetrahedron**, then derive all `11` strains from that single element displacement field.

## Element-Level Prototype Insight

A quick prototype confirmed the export is structured enough for that stronger fit:

- cover-layer element blocks can be reconstructed from the raw export
- one element-level displacement vector can be recovered by least squares from the stacked `66 x 30` map

But a first **global** element-level Gaussian copula still collapsed most generated states to the apex branch. That indicates the element-level route is correct, but it still needs a better conditional fit than a single global copula.

## Next Recommended Version

The next serious attempt should be:

1. recover element-level cover-layer strain blocks from the raw export
2. fit displacement distributions on the stacked `11`-point element operator, not pointwise `6 x 30` blocks
3. condition the fit on at least one of:
   - material state / SSR factor
   - dominant element branch pattern
   - empirical neighborhood blending around recovered element displacements
4. then retrain against the same fixed real cover-layer test split
