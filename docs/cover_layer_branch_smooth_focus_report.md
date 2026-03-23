# Cover Layer Smooth-Focus Generator Report

## Goal

After finding that many failed real elements were outside the practical support of the current synthetic generator, we tried the obvious correction:

- anchor the synthetic generator on **real validation elements where `smooth` is misclassified**
- regenerate synthetic data around those neighborhoods
- retrain the same staged branch predictor

The point of this phase was to test whether the remaining gap is mainly a **missing-domain** problem or whether the representation/classifier itself is now the next bottleneck.

## Experiments

All runs used the same staged branch-predictor trainer and the same real validation/test splits.

Compared generators:

1. `seeded_local_noise_branch_balanced`
   - current best pre-focus baseline
2. `seeded_smooth_fail_mixture`, moderate
   - `focus_mix_fraction = 0.35`
   - `focus_noise_multiplier = 1.15`
3. `seeded_smooth_fail_mixture`, tight
   - `focus_mix_fraction = 0.70`
   - `focus_noise_multiplier = 0.35`

The focus anchors were built from **real validation** elements containing `smooth` points that the bootstrap model misclassified. This avoids using the real test split to define the generator.

## Comparison

![Smooth Focus Comparison](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/smooth_focus_compare.png)

| Run | Real test acc | Real test macro | Real `smooth` recall | Real `left_edge` recall | Real `right_edge` recall | Real `apex` recall | Synthetic acc | Synthetic macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seeded balanced | `0.6000` | `0.5741` | `0.0051` | `0.5275` | `0.8223` | `0.5829` | `0.9763` | `0.9827` |
| smooth-focus mix `0.35/1.15` | `0.5891` | `0.5659` | `0.0041` | `0.4906` | `0.9260` | `0.4765` | `0.9730` | `0.9808` |
| smooth-focus tight `0.70/0.35` | `0.5447` | `0.5268` | `0.0072` | `0.3738` | `0.7859` | `0.5923` | `0.9493` | `0.9612` |

## What Happened

### Moderate smooth-focus mixture

This did **not** improve the overall model.

Effects:

- real test accuracy dropped slightly: `0.6000 -> 0.5891`
- real test macro recall dropped slightly: `0.5741 -> 0.5659`
- real `smooth` recall stayed effectively zero: `0.0051 -> 0.0041`
- real `right_edge` improved: `0.8223 -> 0.9260`
- real `left_edge` and `apex` got worse

So the added failure anchors changed the branch tradeoff, but they did not recover the missing real `smooth` regime.

### Tight smooth-focus mixture

This was worse overall.

Effects:

- real test accuracy dropped materially: `0.6000 -> 0.5447`
- real test macro recall dropped materially: `0.5741 -> 0.5268`
- real `smooth` recall still stayed effectively zero: `0.0072`

So tightening the focus around the failed `smooth` anchors did not fix the issue either.

## Interpretation

This result matters because it rules out a simple explanation.

If the problem were only:

- “the missing real neighborhoods are not in the synthetic generator”

then adding real validation failure neighborhoods as synthetic anchors should have improved real `smooth` recall clearly.

It did not.

That means the remaining problem is likely stronger than just missing gross support.

## What This Suggests

At this point, the evidence points to one of these:

1. the current synthetic generator still does not reproduce the **local branch geometry** of real `smooth` points, even when anchored nearby
2. the current **pointwise feature representation** is not sufficient to separate real `smooth` from real `right_edge`
3. the branch labels near the real `smooth/right_edge` transition are inherently harder than the current pointwise classifier can express

The key point is:

- generator correction helped a lot earlier
- but **smooth-focused generator correction alone is no longer enough**

## Recommendation

Do not keep tuning the same generator knob blindly.

The next higher-value step is:

1. keep the seeded branch-balanced generator as the best current baseline
2. keep the failure-anchor analysis as evidence
3. change the model structure from **pointwise** to **element-contextual**

Specifically:

- input all `11` integration-point feature vectors together
- predict all `11` branches together
- let the model use within-element context to separate `smooth` from nearby `right_edge` / `left_edge` / `apex`

That is now the most defensible next experiment.

## Artifacts

- moderate focus run: [summary.json](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_smooth_focus_20260314/summary.json)
- tight focus run: [summary.json](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_smooth_focus_tight_20260314/summary.json)
- baseline run: [summary.json](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314/summary.json)
