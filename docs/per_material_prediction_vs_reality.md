# Per-Material Prediction vs Reality

This report shows the prediction-vs-reality behavior on the primary real test split for the best
per-material checkpoint available for each material family:
- `general_foundation`: empirical exact relabel model
- `weak_foundation`: empirical exact relabel model
- `general_slope`: hybrid hardcase synthetic model
- `cover_layer`: hybrid hardcase synthetic model

The comparison target in the tables is the current global real-trained baseline.

## Summary

![Primary MAE by material](../experiment_runs/real_sim/per_material_prediction_report_20260312/primary_mae_by_material.png)

| Material | Selected MAE | Baseline MAE | Selected RMSE | Baseline RMSE | Selected Branch Acc | Baseline Branch Acc |
|---|---:|---:|---:|---:|---:|---:|
| general_foundation | 0.9729 | 1.8580 | 1.6418 | 3.2566 | 0.9911 | 0.9585 |
| weak_foundation | 0.9444 | 3.6945 | 1.7276 | 8.3429 | 0.9824 | 0.9539 |
| general_slope | 13.6957 | 5.1771 | 27.0371 | 19.9995 | 0.9319 | 0.9564 |
| cover_layer | 11.3344 | 5.6771 | 21.5272 | 19.0208 | 0.9028 | 0.9072 |

## general_foundation

- checkpoint: `experiment_runs/real_sim/per_material_synth_to_real_20260312/runs/general_foundation/best.pt`
- primary test dataset: `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/general_foundation_primary.h5`
- stress MAE: `0.9729`
- stress RMSE: `1.6418`
- stress max abs: `24.7492`
- branch accuracy: `0.9911`
- sample L2 test loss mean / p90 / p99: `3.1796` / `5.7501` / `12.7616`
- relative sample test loss mean / p90 / p99: `0.0042` / `0.0051` / `0.0107`

Training history:

![general_foundation history log](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/history_log.png)

Prediction vs reality parity on the primary real test split:

![general_foundation parity](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/parity_stress.png)

Test-loss histogram over samples:

![general_foundation sample loss histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/sample_l2_hist.png)

Test loss versus true stress magnitude:

![general_foundation loss vs true norm](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/sample_l2_vs_true_norm.png)

Relative test-loss CDF:

![general_foundation relative loss cdf](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/relative_loss_cdf.png)

Componentwise stress-error histogram:

![general_foundation stress error histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/stress_error_hist.png)

Branch confusion on the same test split:

![general_foundation branch confusion](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_foundation/branch_confusion.png)

## weak_foundation

- checkpoint: `experiment_runs/real_sim/per_material_synth_to_real_20260312/runs/weak_foundation/best.pt`
- primary test dataset: `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/weak_foundation_primary.h5`
- stress MAE: `0.9444`
- stress RMSE: `1.7276`
- stress max abs: `51.2800`
- branch accuracy: `0.9824`
- sample L2 test loss mean / p90 / p99: `3.0056` / `5.6301` / `13.8914`
- relative sample test loss mean / p90 / p99: `0.0150` / `0.0232` / `0.1501`

Training history:

![weak_foundation history log](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/history_log.png)

Prediction vs reality parity on the primary real test split:

![weak_foundation parity](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/parity_stress.png)

Test-loss histogram over samples:

![weak_foundation sample loss histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/sample_l2_hist.png)

Test loss versus true stress magnitude:

![weak_foundation loss vs true norm](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/sample_l2_vs_true_norm.png)

Relative test-loss CDF:

![weak_foundation relative loss cdf](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/relative_loss_cdf.png)

Componentwise stress-error histogram:

![weak_foundation stress error histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/stress_error_hist.png)

Branch confusion on the same test split:

![weak_foundation branch confusion](../experiment_runs/real_sim/per_material_prediction_report_20260312/weak_foundation/branch_confusion.png)

## general_slope

- checkpoint: `experiment_runs/real_sim/per_material_hybrid_hardcases_20260312/runs/general_slope/best.pt`
- primary test dataset: `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/general_slope_primary.h5`
- stress MAE: `13.6957`
- stress RMSE: `27.0371`
- stress max abs: `677.1581`
- branch accuracy: `0.9319`
- sample L2 test loss mean / p90 / p99: `44.1891` / `88.4151` / `243.1853`
- relative sample test loss mean / p90 / p99: `0.4034` / `0.7780` / `3.8451`

Training history:

![general_slope history log](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/history_log.png)

Prediction vs reality parity on the primary real test split:

![general_slope parity](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/parity_stress.png)

Test-loss histogram over samples:

![general_slope sample loss histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/sample_l2_hist.png)

Test loss versus true stress magnitude:

![general_slope loss vs true norm](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/sample_l2_vs_true_norm.png)

Relative test-loss CDF:

![general_slope relative loss cdf](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/relative_loss_cdf.png)

Componentwise stress-error histogram:

![general_slope stress error histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/stress_error_hist.png)

Branch confusion on the same test split:

![general_slope branch confusion](../experiment_runs/real_sim/per_material_prediction_report_20260312/general_slope/branch_confusion.png)

## cover_layer

- checkpoint: `experiment_runs/real_sim/per_material_hybrid_hardcases_20260312/runs/cover_layer/best.pt`
- primary test dataset: `experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5`
- stress MAE: `11.3344`
- stress RMSE: `21.5272`
- stress max abs: `483.0251`
- branch accuracy: `0.9028`
- sample L2 test loss mean / p90 / p99: `36.4380` / `66.7153` / `181.5102`
- relative sample test loss mean / p90 / p99: `0.7593` / `1.5573` / `4.8656`

Training history:

![cover_layer history log](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/history_log.png)

Prediction vs reality parity on the primary real test split:

![cover_layer parity](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/parity_stress.png)

Test-loss histogram over samples:

![cover_layer sample loss histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/sample_l2_hist.png)

Test loss versus true stress magnitude:

![cover_layer loss vs true norm](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/sample_l2_vs_true_norm.png)

Relative test-loss CDF:

![cover_layer relative loss cdf](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/relative_loss_cdf.png)

Componentwise stress-error histogram:

![cover_layer stress error histogram](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/stress_error_hist.png)

Branch confusion on the same test split:

![cover_layer branch confusion](../experiment_runs/real_sim/per_material_prediction_report_20260312/cover_layer/branch_confusion.png)
