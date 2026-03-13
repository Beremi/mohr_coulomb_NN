# Cover Layer Fitted-Distribution Epoch Refresh

This study replaces the old branch-targeted synthetic sampler with strain samplers fitted directly to the real cover-layer `E` distribution.
Each run keeps validation and testing fixed on the real cover-layer dataset, while regenerating new exact-labeled synthetic training data every epoch.

## Real-Domain Fit

- real primary dataset: `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5`
- train pool size: `63343`
- fitted principal cap: `48.966018`
- empirical train branch probabilities: `{'elastic': 0.14729330786353662, 'smooth': 0.26023396428966106, 'left_edge': 0.24394171415941776, 'right_edge': 0.17301043524935666, 'apex': 0.17552057843802787}`

| Real test | N | max |eps_principal| q95 | q995 | stress | q95 | q995 |
|---|---:|---:|---:|---:|---:|
| real test | 7967 | 14.5484 | 38.2638 | 306.9618 | 749.3344 |

## Synthetic Holdout Distribution Fit

| Sampler | max |eps_principal| q95 | q995 | stress | q95 | q995 | branch TV | fit score |
|---|---:|---:|---:|---:|---:|---:|
| `bootstrap_jitter` | 13.7562 | 37.6034 | 16463.8379 | 50892.4336 | 0.2333 | 8.5071 |
| `interp_local` | 12.6739 | 27.8382 | 9148.7646 | 27046.7617 | 0.2741 | 7.7109 |
| `tail_mix` | 12.7236 | 31.0700 | 15576.7812 | 53466.6328 | 0.2352 | 8.7719 |

## Training Results

| Sampler | Runtime | Best epoch | Real test MAE | RMSE | Max abs | Branch acc | Synthetic test MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bootstrap_jitter` | 00:00:02 | 1 | 50.8660 | 78.7471 | 1026.1788 | 0.2360 | 1009.6646 | 3341.0332 |
| `interp_local` | 00:00:01 | 1 | 44.7444 | 71.6738 | 956.1057 | 0.2854 | 610.0937 | 1923.9508 |
| `tail_mix` | 00:00:01 | 1 | 42.9225 | 69.9641 | 941.0675 | 0.2879 | 976.3802 | 3243.5708 |

Best real-test sampler in this run: `tail_mix`.

Reference fixed-dataset exact-domain baseline (`cover_raw_branch_w384_d6`):

- real test MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`

## bootstrap_jitter

- distribution fit score: `8.5071`
- real test MAE / RMSE / max abs: `50.8660` / `78.7471` / `1026.1788`
- synthetic test MAE / RMSE / max abs: `1009.6646` / `3341.0332` / `53059.6562`

![bootstrap_jitter distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/distribution_fit_bootstrap_jitter.png)

![bootstrap_jitter history](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/history_log.png)

![bootstrap_jitter branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/branch_accuracy.png)

![bootstrap_jitter real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/eval_real/parity_stress.png)

![bootstrap_jitter real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/eval_real/stress_error_hist.png)

![bootstrap_jitter real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/eval_real/branch_confusion.png)

![bootstrap_jitter synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/eval_synth/parity_stress.png)

![bootstrap_jitter synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/eval_synth/stress_error_hist.png)

![bootstrap_jitter synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_bootstrap_jitter/eval_synth/branch_confusion.png)

## interp_local

- distribution fit score: `7.7109`
- real test MAE / RMSE / max abs: `44.7444` / `71.6738` / `956.1057`
- synthetic test MAE / RMSE / max abs: `610.0937` / `1923.9508` / `55366.8672`

![interp_local distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/distribution_fit_interp_local.png)

![interp_local history](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/history_log.png)

![interp_local branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/branch_accuracy.png)

![interp_local real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/eval_real/parity_stress.png)

![interp_local real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/eval_real/stress_error_hist.png)

![interp_local real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/eval_real/branch_confusion.png)

![interp_local synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/eval_synth/parity_stress.png)

![interp_local synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/eval_synth/stress_error_hist.png)

![interp_local synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_interp_local/eval_synth/branch_confusion.png)

## tail_mix

- distribution fit score: `8.7719`
- real test MAE / RMSE / max abs: `42.9225` / `69.9641` / `941.0675`
- synthetic test MAE / RMSE / max abs: `976.3802` / `3243.5708` / `65285.9023`

![tail_mix distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/distribution_fit_tail_mix.png)

![tail_mix history](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/history_log.png)

![tail_mix branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/branch_accuracy.png)

![tail_mix real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/eval_real/parity_stress.png)

![tail_mix real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/eval_real/stress_error_hist.png)

![tail_mix real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/eval_real/branch_confusion.png)

![tail_mix synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/eval_synth/parity_stress.png)

![tail_mix synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/eval_synth/stress_error_hist.png)

![tail_mix synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke/cover_raw_branch_w64_d2_tail_mix/eval_synth/branch_confusion.png)

