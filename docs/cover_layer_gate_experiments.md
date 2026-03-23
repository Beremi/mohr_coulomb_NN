# Cover-Layer Gate Experiments

This report trains dedicated branch gates for the already-learned plastic experts and tests whether routing quality is now enough to beat the direct baseline.

## Gate Training

| Gate | Best Val Macro Recall | Best Epoch |
|---|---:|---:|
| gate_raw | 0.9590 | 1181 |
| gate_trial | 0.9444 | 1126 |

## Ensemble Results

| Mode | Real MAE | Real RMSE | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |
|---|---:|---:|---:|---:|---:|---:|
| baseline_reference | 8.2331 | 22.9041 | 0.8255 | 19.4116 | 156.1580 | 0.7536 |
| oracle_reference | 7.3426 | 31.0570 | 1.0000 | 14.5764 | 152.9653 | 1.0000 |
| gate_raw_hard | 7.3752 | 31.2995 | 0.9559 | 11.5130 | 111.0791 | 0.9396 |
| gate_raw_soft | 10.4225 | 172.3640 | 0.9559 | 11.2719 | 113.0077 | 0.9396 |
| gate_raw_threshold_t0.65 | 7.3324 | 30.9220 | 0.9559 | 14.4552 | 151.4229 | 0.9396 |
| gate_trial_hard | 7.4910 | 31.2505 | 0.9433 | 14.8467 | 154.7130 | 0.9586 |
| gate_trial_soft | 9.0482 | 48.9804 | 0.9433 | 14.8237 | 154.7335 | 0.9586 |
| gate_trial_threshold_t0.85 | 7.1258 | 29.1589 | 0.9433 | 14.9925 | 155.1699 | 0.9586 |

Best real-holdout mode: `gate_trial_threshold_t0.85`

![MAE comparison](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/compare_mae.png)

![RMSE comparison](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/compare_rmse.png)

## Gate Histories

### gate_raw

![history](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw/history.png)

### gate_trial

![history](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial/history.png)

## Mode Figures

### baseline_reference

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/baseline_reference/synthetic/branch_confusion.png)

### oracle_reference

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/oracle_reference/synthetic/branch_confusion.png)

### gate_raw_hard

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_hard/synthetic/branch_confusion.png)

### gate_raw_soft

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_soft/synthetic/branch_confusion.png)

### gate_raw_threshold_t0.65

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw_threshold_t0.65/synthetic/branch_confusion.png)

### gate_trial_hard

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_hard/synthetic/branch_confusion.png)

### gate_trial_soft

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_soft/synthetic/branch_confusion.png)

### gate_trial_threshold_t0.85

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_trial_threshold_t0.85/synthetic/branch_confusion.png)

## Best Real-Holdout Dissection

- mean relative sample error: `0.3344`
- median relative sample error: `0.0487`
- p90 relative sample error: `0.7114`
- p99 relative sample error: `4.7094`

Interpretation:

- If a dedicated gate beats the baseline in deployable hard or soft routing, the branch-expert path is now validated end to end.
- If soft routing helps but hard routing still fails, the gate probabilities are useful but not sharp enough for top-1 dispatch.
- If nothing beats the baseline, then even with dedicated gates the branch-expert route is still not enough.

