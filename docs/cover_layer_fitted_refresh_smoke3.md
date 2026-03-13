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
| `local_noise` | 8.0332 | 26.2509 | 662.3775 | 791.8551 | 0.0115 | 1.8065 |
| `local_interp` | 8.6914 | 21.1434 | 612.1282 | 783.6120 | 0.0115 | 1.8547 |
| `tail_local` | 8.6941 | 24.3880 | 674.8459 | 786.7102 | 0.0115 | 1.8132 |

## Training Results

| Sampler | Runtime | Best epoch | Real test MAE | RMSE | Max abs | Branch acc | Synthetic test MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_noise` | 00:00:02 | 1 | 41.0915 | 67.9178 | 1024.8890 | 0.2361 | 53.2411 | 90.4649 |
| `local_interp` | 00:00:02 | 4 | 36.7757 | 62.0169 | 956.7203 | 0.3836 | 47.2008 | 82.2626 |
| `tail_local` | 00:00:01 | 1 | 39.3557 | 65.0558 | 952.7665 | 0.2989 | 54.8974 | 94.0546 |

Best real-test sampler in this run: `local_interp`.

Reference fixed-dataset exact-domain baseline (`cover_raw_branch_w384_d6`):

- real test MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`

## local_noise

- distribution fit score: `1.8065`
- real test MAE / RMSE / max abs: `41.0915` / `67.9178` / `1024.8890`
- synthetic test MAE / RMSE / max abs: `53.2411` / `90.4649` / `516.9683`

![local_noise distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/distribution_fit_local_noise.png)

![local_noise history](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/history_log.png)

![local_noise branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/branch_accuracy.png)

![local_noise real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/eval_real/parity_stress.png)

![local_noise real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/eval_real/stress_error_hist.png)

![local_noise real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/eval_real/branch_confusion.png)

![local_noise synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/eval_synth/parity_stress.png)

![local_noise synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/eval_synth/stress_error_hist.png)

![local_noise synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_noise/eval_synth/branch_confusion.png)

## local_interp

- distribution fit score: `1.8547`
- real test MAE / RMSE / max abs: `36.7757` / `62.0169` / `956.7203`
- synthetic test MAE / RMSE / max abs: `47.2008` / `82.2626` / `554.7511`

![local_interp distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/distribution_fit_local_interp.png)

![local_interp history](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/history_log.png)

![local_interp branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/branch_accuracy.png)

![local_interp real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/eval_real/parity_stress.png)

![local_interp real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/eval_real/stress_error_hist.png)

![local_interp real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/eval_real/branch_confusion.png)

![local_interp synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/eval_synth/parity_stress.png)

![local_interp synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/eval_synth/stress_error_hist.png)

![local_interp synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_local_interp/eval_synth/branch_confusion.png)

## tail_local

- distribution fit score: `1.8132`
- real test MAE / RMSE / max abs: `39.3557` / `65.0558` / `952.7665`
- synthetic test MAE / RMSE / max abs: `54.8974` / `94.0546` / `560.0723`

![tail_local distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/distribution_fit_tail_local.png)

![tail_local history](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/history_log.png)

![tail_local branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/branch_accuracy.png)

![tail_local real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/eval_real/parity_stress.png)

![tail_local real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/eval_real/stress_error_hist.png)

![tail_local real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/eval_real/branch_confusion.png)

![tail_local synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/eval_synth/parity_stress.png)

![tail_local synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/eval_synth/stress_error_hist.png)

![tail_local synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/cover_raw_branch_w64_d2_tail_local/eval_synth/branch_confusion.png)

