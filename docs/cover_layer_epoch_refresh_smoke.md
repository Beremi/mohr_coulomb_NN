# Cover Layer Epoch-Refresh Training

This run keeps the cover-layer material family fixed, but regenerates a new synthetic training set every epoch.
The held-out evaluations are:
- real cover-layer test split from `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5`
- independent synthetic holdout test from `experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/synthetic_test.h5`

## Setup

- model: `cover_raw_branch_w64_d2_epoch_refresh` (`width=64`, `depth=2`)
- runtime: `00:00:00`
- best epoch by real-val stress MSE: `1`
- best real-val stress MSE: `3984.316406`
- real material pool rows used for synthetic generation: `63343`
- unique reduced material rows in that pool: `337`
- synthetic train samples regenerated each epoch: `2048`
- synthetic test samples: `1024`
- principal-strain cap: `4.005286e+01`
- branch fractions per regenerated epoch: `[0.2, 0.2, 0.2, 0.2, 0.2]`

## Distribution Check

| Dataset | N | max |eps_principal| q95 | q995 | stress | q95 | q995 |
|---|---:|---:|---:|---:|---:|
| real test | 7967 | 1.454836e+01 | 3.826379e+01 | 306.9618 | 749.3344 |
| synthetic holdout | 1024 | 3.093817e-03 | 3.479920e-03 | 45.0000 | 45.0000 |

## Results

| Test set | Stress MAE | Stress RMSE | Stress Max Abs | Branch Acc |
|---|---:|---:|---:|---:|
| real | 31.3151 | 64.9649 | 1039.4233 | 0.1784 |
| synthetic | 4.2903 | 6.2507 | 35.8949 | 0.3330 |

Comparison to the previous fixed-dataset cover-layer best (`w384 d6` on exact-domain training file):

- previous real test MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`
- refreshed-data real test MAE / RMSE / max abs: `31.3151` / `64.9649` / `1039.4233`

## History

![epoch refresh history](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/history_log.png)

![epoch refresh branch accuracy](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/branch_accuracy.png)

## Real Test Plots

![real parity](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/eval_real/parity_stress.png)

![real error histogram](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/eval_real/stress_error_hist.png)

![real branch confusion](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/eval_real/branch_confusion.png)

## Synthetic Holdout Plots

![synthetic parity](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/eval_synth/parity_stress.png)

![synthetic error histogram](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/eval_synth/stress_error_hist.png)

![synthetic branch confusion](../experiment_runs/real_sim/cover_layer_epoch_refresh_smoke/cover_raw_branch_w64_d2_epoch_refresh/eval_synth/branch_confusion.png)

