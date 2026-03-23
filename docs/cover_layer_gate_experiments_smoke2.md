# Cover-Layer Gate Experiments

This report trains dedicated branch gates for the already-learned plastic experts and tests whether routing quality is now enough to beat the direct baseline.

## Gate Training

| Gate | Best Val Macro Recall | Best Epoch |
|---|---:|---:|
| gate_raw | 0.5297 | 8 |
| gate_trial | 0.5734 | 8 |

## Ensemble Results

| Mode | Real MAE | Real RMSE | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |
|---|---:|---:|---:|---:|---:|---:|
| baseline_reference | 8.2331 | 22.9041 | 0.8255 | 19.4116 | 156.1580 | 0.7536 |
| oracle_reference | 7.3426 | 31.0570 | 1.0000 | 14.5764 | 152.9653 | 1.0000 |
| gate_raw_hard | 21.9042 | 65.1851 | 0.4928 | 25.3986 | 161.8699 | 0.5786 |
| gate_raw_soft | 420.7067 | 1444.8049 | 0.4928 | 43.5754 | 159.0564 | 0.5786 |
| gate_raw_threshold_t0.75 | 8.1595 | 23.3004 | 0.4928 | 19.0026 | 156.2805 | 0.5786 |
| gate_trial_hard | 17.2237 | 54.9632 | 0.5593 | 9.3304 | 73.8741 | 0.7081 |
| gate_trial_soft | 419.0840 | 1283.1072 | 0.5593 | 15.1679 | 85.2309 | 0.7081 |
| gate_trial_threshold_t0.65 | 6.9866 | 22.5554 | 0.5593 | 9.7655 | 75.4300 | 0.7081 |

Best real-holdout mode: `gate_trial_threshold_t0.65`

![MAE comparison](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/compare_mae.png)

![RMSE comparison](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/compare_rmse.png)

## Gate Histories

### gate_raw

![history](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw/history.png)

### gate_trial

![history](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial/history.png)

## Mode Figures

### baseline_reference

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/baseline_reference/synthetic/branch_confusion.png)

### oracle_reference

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/oracle_reference/synthetic/branch_confusion.png)

### gate_raw_hard

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_hard/synthetic/branch_confusion.png)

### gate_raw_soft

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_soft/synthetic/branch_confusion.png)

### gate_raw_threshold_t0.75

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_raw_threshold_t0.75/synthetic/branch_confusion.png)

### gate_trial_hard

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_hard/synthetic/branch_confusion.png)

### gate_trial_soft

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_soft/synthetic/branch_confusion.png)

### gate_trial_threshold_t0.65

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_smoke2_20260313/gate_trial_threshold_t0.65/synthetic/branch_confusion.png)

## Best Real-Holdout Dissection

- mean relative sample error: `0.3678`
- median relative sample error: `0.1644`
- p90 relative sample error: `0.7895`
- p99 relative sample error: `3.4730`

Interpretation:

- If a dedicated gate beats the baseline in deployable hard or soft routing, the branch-expert path is now validated end to end.
- If soft routing helps but hard routing still fails, the gate probabilities are useful but not sharp enough for top-1 dispatch.
- If nothing beats the baseline, then even with dedicated gates the branch-expert route is still not enough.

