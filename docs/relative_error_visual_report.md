# Relative Error Visual Report

Checkpoint: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt`

This report focuses on:

- staged test-loss evolution
- robust relative error on held-out samples
- random-sample relative differences between predicted and true stresses

Relative metrics in this report use robust denominators:

- sample relative error: `||pred - true||_2 / max(||true||_2, 1)`
- signed componentwise relative difference: `(pred - true) / max(|true|, 1)`

## Test Loss Curves

![Test loss log](../experiment_runs/real_sim/staged_20260312/relative_error_report/test_loss_log.png)

![Test stress mse log](../experiment_runs/real_sim/staged_20260312/relative_error_report/test_stress_mse_log.png)

## Primary Test Split `d512`

- `stress_mae = 4.8612`
- `stress_rmse = 17.7217`
- `stress_max_abs = 936.4613`
- `branch_accuracy = 0.9424`
- `mean relative sample error = 0.1689`
- `median relative sample error = 0.0357`
- `p90 relative sample error = 0.3447`
- `p99 relative sample error = 2.2254`

![Primary relative histogram](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/relative_hist.png)

![Primary relative CDF](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/relative_cdf.png)

![Primary random relative scatter](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/random_relative_scatter.png)

![Primary random component heatmap](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/random_component_relative_heatmap.png)

## Cross-Check Test Split `d256`

- `stress_mae = 4.7536`
- `stress_rmse = 17.4137`
- `stress_max_abs = 759.1741`
- `branch_accuracy = 0.9432`
- `mean relative sample error = 0.1549`
- `median relative sample error = 0.0353`
- `p90 relative sample error = 0.3420`
- `p99 relative sample error = 2.0378`

![Cross relative histogram](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/relative_hist.png)

![Cross relative CDF](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/relative_cdf.png)

![Cross random relative scatter](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/random_relative_scatter.png)

![Cross random component heatmap](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/random_component_relative_heatmap.png)

