# Cover Layer Epoch-Refresh Training

This run keeps the cover-layer material family fixed, but regenerates a new synthetic training set every epoch.
The held-out evaluations are:
- real cover-layer test split from `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5`
- independent synthetic holdout test from `experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/synthetic_test.h5`

## Setup

- model: `cover_raw_branch_w384_d6_epoch_refresh` (`width=384`, `depth=6`)
- runtime: `00:06:25`
- best epoch by real-val stress MSE: `198`
- best real-val stress MSE: `2698.890137`
- real material pool rows used for synthetic generation: `63343`
- unique reduced material rows in that pool: `337`
- synthetic train samples regenerated each epoch: `65536`
- synthetic test samples: `16384`
- principal-strain cap: `4.005286e+01`
- branch fractions per regenerated epoch: `[0.2, 0.2, 0.2, 0.2, 0.2]`

## Distribution Check

| Dataset | N | max |eps_principal| q95 | q995 | stress | q95 | q995 |
|---|---:|---:|---:|---:|---:|
| real test | 7967 | 1.454836e+01 | 3.826379e+01 | 306.9618 | 749.3344 |
| synthetic holdout | 16384 | 2.995578e-03 | 3.496501e-03 | 45.0000 | 45.0000 |

## Results

| Test set | Stress MAE | Stress RMSE | Stress Max Abs | Branch Acc |
|---|---:|---:|---:|---:|
| real | 23.1082 | 53.8699 | 985.5306 | 0.3983 |
| synthetic | 0.0206 | 0.0408 | 2.9746 | 1.0000 |

Comparison to the previous fixed-dataset cover-layer best (`w384 d6` on exact-domain training file):

- previous real test MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`
- refreshed-data real test MAE / RMSE / max abs: `23.1082` / `53.8699` / `985.5306`

## History

![epoch refresh history](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/history_log.png)

![epoch refresh branch accuracy](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/branch_accuracy.png)

## Real Test Plots

![real parity](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/eval_real/parity_stress.png)

![real error histogram](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/eval_real/stress_error_hist.png)

![real branch confusion](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/eval_real/branch_confusion.png)

## Synthetic Holdout Plots

![synthetic parity](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/eval_synth/parity_stress.png)

![synthetic error histogram](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/eval_synth/stress_error_hist.png)

![synthetic branch confusion](../experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/cover_raw_branch_w384_d6_epoch_refresh/eval_synth/branch_confusion.png)

