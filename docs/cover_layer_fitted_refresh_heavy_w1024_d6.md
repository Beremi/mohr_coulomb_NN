# Cover Layer Heavy Fitted-Refresh Run

This report covers the long `w1024 d6` cover-layer run trained with:

- fresh synthetic `E -> S` training data regenerated every epoch
- fixed real validation/test split
- fixed synthetic holdout test
- fitted `local_noise` strain generator
- long staged schedule across `64 -> 128 -> 256 -> 512 -> 1024 -> 2048`

The goal was simple: keep the fitted-distribution idea, but remove the too-short training horizon from the earlier attempts.

## Result

This run is the current best cover-layer model in the repository.

- checkpoint: `experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/best.pt`
- runtime: `03:19:52`
- best epoch: `5482`
- best real validation stress MSE: `72.5715`

Real test metrics:

- stress MAE: `1.4951`
- stress RMSE: `9.7174`
- stress max abs: `526.0352`
- branch accuracy: `0.9577`

Synthetic holdout metrics:

- stress MAE: `0.5861`
- stress RMSE: `1.3813`
- stress max abs: `51.0301`
- branch accuracy: `0.9988`

## Comparison To Earlier Cover-Layer Runs

| Run | Real test MAE | Real test RMSE | Max abs | Branch acc |
|---|---:|---:|---:|---:|
| cyclic `384x6` | `4.7124` | `13.1843` | `520.7037` | `0.9195` |
| cyclic `512x6` | `4.8037` | `12.9621` | `497.6740` | `0.9195` |
| adaptive `1024x6` | `5.6846` | `21.0031` | `757.3899` | `0.9007` |
| fitted refresh focused `384x6` | `4.0053` | `14.5005` | `788.4071` | `0.8816` |
| heavy fitted refresh `384x6` | `1.8579` | `10.3853` | `533.2682` | `0.9498` |
| heavy fitted refresh `1024x6` | `1.4951` | `9.7174` | `526.0352` | `0.9577` |

Compared with the previous best exact-domain cyclic baseline (`384x6`), this run cuts:

- MAE by about `68%`
- RMSE by about `26%`
- left-edge branch MAE from `7.2815` to `2.3914`
- right-edge branch MAE from `7.7393` to `2.8533`

![comparison real metrics](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/comparison_real_metrics.png)

![comparison hard branches](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/comparison_hard_branches.png)

## Distribution Fit

The synthetic training stream used the fitted `local_noise` generator, which was already the strongest option from the fitted-refresh screening. That fit still is not perfect, but it is good enough that longer training now pays off instead of collapsing out of domain.

- real test `max |principal strain| q95 / q995`: `14.5484 / 38.2638`
- synthetic holdout `max |principal strain| q95 / q995`: `9.3004 / 25.8713`
- real test stress magnitude `q95 / q995`: `306.9618 / 749.3344`
- synthetic holdout stress magnitude `q95 / q995`: `671.8993 / 788.9667`
- branch TV distance: `0.0115`
- fitted sampler score: `1.6853`

![distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/distribution_fit_local_noise.png)

## Training Curves

The long schedule matters here. The run improved steadily over hours, not minutes, and the strongest gains came after multiple learning-rate drops and later batch stages.

![history log](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/history_log.png)

![branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/branch_accuracy.png)

Key observations from the history:

- the run did not peak early; it kept improving deep into later stages
- real and synthetic held-out losses dropped together, which is a good sign for domain alignment
- the synthetic holdout stays easier than the real test, but the gap is much smaller than in the earlier failed refresh runs
- the late-stage floor is stable rather than erratic

## Real Test Visuals

![real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/eval_real/parity_stress.png)

![real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/eval_real/stress_error_hist.png)

![real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/eval_real/branch_confusion.png)

Real per-branch stress MAE:

| Branch | Stress MAE |
|---|---:|
| elastic | `0.5675` |
| smooth | `1.0963` |
| left_edge | `2.3914` |
| right_edge | `2.8533` |
| apex | `0.2908` |

The hardest branches are still `left_edge` and `right_edge`, but they are no longer catastrophically worse than the rest.

## Synthetic Holdout Visuals

![synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/eval_synth/parity_stress.png)

![synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/eval_synth/stress_error_hist.png)

![synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/eval_synth/branch_confusion.png)

Synthetic per-branch stress MAE:

| Branch | Stress MAE |
|---|---:|
| elastic | `0.5927` |
| smooth | `0.3698` |
| left_edge | `0.9014` |
| right_edge | `1.0058` |
| apex | `0.0493` |

## Takeaways

This run changes the conclusion for the cover layer.

- The fitted-refresh idea does work if we train it long enough.
- The `w1024 d6` model is materially better than the earlier `384x6` and `512x6` cover-layer baselines.
- The remaining weak point is still the tail max error, not the average fit.
- The real/synthetic gap is now small enough that this is worth testing as a serious constitutive replacement candidate for the cover layer.

What I would do next:

1. Freeze this checkpoint as the current cover-layer winner.
2. Add the same heavy fitted-refresh treatment to the other real material families.
3. Then test either per-material deployment or a mixture/expert wrapper in the solver.
