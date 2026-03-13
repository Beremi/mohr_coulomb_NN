# Cover Layer Fitted-Distribution Epoch Refresh

This study replaces the old branch-targeted synthetic sampler with strain samplers fitted directly to the real cover-layer `E` distribution.
Each run keeps validation and testing fixed on the real cover-layer dataset, while regenerating new exact-labeled synthetic training data every epoch.

## Real-Domain Fit

- real primary dataset: `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5`
- train pool size: `63343`
- fitted principal cap: `38.232278`
- empirical train branch probabilities: `{'elastic': 0.14729330786353662, 'smooth': 0.26023396428966106, 'left_edge': 0.24394171415941776, 'right_edge': 0.17301043524935666, 'apex': 0.17552057843802787}`

| Real test | N | max |eps_principal| q95 | q995 | stress | q95 | q995 |
|---|---:|---:|---:|---:|---:|
| real test | 7967 | 14.5484 | 38.2638 | 306.9618 | 749.3344 |

## Synthetic Holdout Distribution Fit

| Sampler | max |eps_principal| q95 | q995 | stress | q95 | q995 | branch TV | fit score |
|---|---:|---:|---:|---:|---:|---:|
| `local_noise` | 9.3004 | 25.8713 | 671.8993 | 788.9667 | 0.0115 | 1.6853 |

## Training Results

| Sampler | Runtime | Best epoch | Real test MAE | RMSE | Max abs | Branch acc | Synthetic test MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_noise` | 01:44:00 | 2295 | 1.8579 | 10.3853 | 533.2682 | 0.9498 | 0.9281 | 1.9315 |

Best real-test sampler in this run: `local_noise`.

Reference fixed-dataset exact-domain baseline (`cover_raw_branch_w384_d6`):

- real test MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`

## local_noise

- distribution fit score: `1.6853`
- real test MAE / RMSE / max abs: `1.8579` / `10.3853` / `533.2682`
- synthetic test MAE / RMSE / max abs: `0.9281` / `1.9315` / `94.9504`

![local_noise distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/distribution_fit_local_noise.png)

![local_noise history](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/history_log.png)

![local_noise branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/branch_accuracy.png)

![local_noise real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/eval_real/parity_stress.png)

![local_noise real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/eval_real/stress_error_hist.png)

![local_noise real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/eval_real/branch_confusion.png)

![local_noise synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/eval_synth/parity_stress.png)

![local_noise synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/eval_synth/stress_error_hist.png)

![local_noise synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_20260312/cover_raw_branch_w384_d6_local_noise/eval_synth/branch_confusion.png)

