# High-Capacity Constitutive Surrogate Study

## Setup

- Dataset: `experiment_runs/high_capacity/dataset.h5`
- Study root: `experiment_runs/high_capacity/study_pilot`
- Shared dataset size: `7933` test points out of the generated split
- Selected pilot by score: `pilot_02_w512_d8`
- Repeat runs: `12`

The study used a larger principal-stress network, cosine learning-rate scheduling, and an LBFGS fine-tune stage after AdamW. A composite dataset was generated to widen coverage: broad branch-balanced samples, focused branch-balanced samples, and benchmark path augmentation.

## Pilot Comparison

![Pilot comparison](../experiment_runs/high_capacity/study_pilot/report/pilot_comparison.png)

| Pilot | Width | Depth | Adam epochs | LBFGS | Test stress MAE | Test stress RMSE | Test max abs | Benchmark mean principal MAE | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_01_w512_d6 | 512.0000 | 6.0000 | 240.0000 | 6.0000 | 0.9972 | 1.3763 | 12.9758 | 1.5312 | 2.5284 |
| pilot_02_w512_d8 | 512.0000 | 8.0000 | 280.0000 | 8.0000 | 0.9964 | 1.3842 | 27.7645 | 1.5274 | 2.5238 |
| pilot_03_w768_d8 | 768.0000 | 8.0000 | 320.0000 | 10.0000 | 0.9978 | 1.3764 | 11.6846 | 1.5282 | 2.5259 |

## Repeat Stability

![Repeat distributions](../experiment_runs/high_capacity/study_pilot/report/repeat_distributions.png)

![Repeat histories](../experiment_runs/high_capacity/study_pilot/report/repeat_history_overlay.png)

| Metric | Mean | Std | Best | Worst |
| --- | --- | --- | --- | --- |
| test stress MAE | 0.9961 | 0.0007 | 0.9949 | 0.9974 |
| test stress RMSE | 1.3797 | 0.0028 | 1.3744 | 1.3842 |
| benchmark mean principal MAE | 1.5261 | 0.0006 | 1.5252 | 1.5274 |

Best / median / worst repeats by test stress MAE:

| Run | Seed | Test stress MAE | Test stress RMSE | Test max abs | Benchmark mean principal MAE | Benchmark worst principal MAE |
| --- | --- | --- | --- | --- | --- | --- |
| seed_1012 | 1012.0000 | 0.9949 | 1.3744 | 10.1284 | 1.5252 | 2.7035 |
| seed_1010 | 1010.0000 | 0.9961 | 1.3766 | 9.4221 | 1.5260 | 2.7087 |
| seed_1011 | 1011.0000 | 0.9974 | 1.3814 | 20.9569 | 1.5274 | 2.7126 |

## Best Run Diagnostics

![Best history](../experiment_runs/high_capacity/study_pilot/report/best_history.png)

![Best parity](../experiment_runs/high_capacity/study_pilot/report/best_parity_stress.png)

![Best error histogram](../experiment_runs/high_capacity/study_pilot/report/best_stress_error_hist.png)

![Best branch confusion](../experiment_runs/high_capacity/study_pilot/report/best_branch_confusion.png)

- Best repeat: `seed_1012`
- Test stress MAE: `0.994877`
- Test stress RMSE: `1.374361`
- Test stress max abs: `10.128434`
- Relative MAE against mean absolute test stress: `17.1111%`
- Branch accuracy: `99.7101%`

Worst benchmark path cases for the best repeat:

| Material | Path | SRF | Stress MAE | Stress max abs | Principal MAE | Principal max abs |
| --- | --- | --- | --- | --- | --- | --- |
| weak_foundation | left_edge | 1.0000 | 1.3518 | 6.8171 | 2.7035 | 6.8171 |
| weak_foundation | left_edge | 1.2500 | 1.3274 | 6.4226 | 2.6548 | 6.4226 |
| weak_foundation | left_edge | 1.5000 | 1.2549 | 5.6720 | 2.5098 | 5.6720 |
| general_foundation | left_edge | 1.2500 | 1.1511 | 5.9206 | 2.3021 | 5.9206 |
| general_foundation | left_edge | 1.0000 | 1.1320 | 5.9777 | 2.2640 | 5.9777 |
| general_foundation | left_edge | 1.5000 | 1.1312 | 5.5327 | 2.2625 | 5.5327 |
| weak_foundation | left_edge | 2.0000 | 1.0302 | 5.3737 | 2.0605 | 5.3737 |
| general_foundation | left_edge | 2.0000 | 0.9884 | 5.3927 | 1.9768 | 5.3927 |

## Insights

- The larger network plus cosine scheduling and LBFGS made the training stable, but the improvement over the earlier tuned demo saturated quickly.
- Across the repeated runs, randomness had very little effect on the final test MAE. The configuration appears highly repeatable.
- The broad high-coverage dataset is materially harder than the earlier focused demo dataset, which explains why the absolute error is higher than in the narrower notebook run.
- Benchmark path errors remain largest on left-edge cases and on weaker materials, which is consistent with the branch geometry being hardest there.

## Replacement Assessment

This study gives a stronger local approximation than the earlier notebook, and it is stable across repeated runs. However, based on these results alone, it is **not yet justified to claim that the surrogate can replace the constitutive relationship in production code while preserving the same limit-analysis result**.

Reasons:

- Even the best repeated run still has local test stress MAE around `0.995` and worst benchmark principal MAE around `2.704`.
- The study validates the local constitutive map only. It does not insert the surrogate into the FE limit-analysis loop and measure the resulting factor of safety, collapse multiplier, plastic-zone shape, or Newton/continuation behavior.
- Limit analysis can be sensitive to local errors near branch transitions and in the active plastic zone, so local surrogate accuracy alone is not sufficient to guarantee the same global result.

Current judgment:

- Good enough for continued surrogate research and for controlled FE integration tests.
- Not enough evidence yet for a drop-in replacement when the requirement is to keep the final limit-analysis result the same.

Recommended next step:

Run the surrogate inside the actual limit-analysis code on a small benchmark set and compare factor of safety, load multiplier, iteration counts, and plastic-zone localization against the exact constitutive call.
