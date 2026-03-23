# Cover-Layer Branch Experts

This report tests a branch-specialized plastic surrogate on top of the existing strong baseline branch gate.

Setup:

- branch gate: existing `baseline_raw_branch` checkpoint
- elastic branch: exact elastic trial stress
- plastic branches: separate raw stress experts for `smooth`, `left_edge`, `right_edge`, `apex`
- deployable ensemble: predicted-branch routing
- optimistic ceiling: oracle routing with true branch labels
- robust variant: confidence-threshold routing with baseline fallback

## Expert Training Sets

| Expert | Train | Val | Test |
|---|---:|---:|---:|
| smooth | 25415 | 5351 | 5900 |
| left_edge | 24285 | 5398 | 4979 |
| right_edge | 17293 | 3843 | 3595 |
| apex | 17787 | 3581 | 3897 |

## Ensemble Results

| Mode | Real MAE | Real RMSE | Real Max Abs | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_reference | 8.2331 | 22.9041 | 695.4951 | 0.8255 | 19.4116 | 156.1580 | 0.7536 |
| oracle_branch_experts | 7.3426 | 31.0570 | 923.3083 | 1.0000 | 14.5764 | 152.9653 | 1.0000 |
| predicted_branch_experts | 8.7657 | 33.3613 | 923.3083 | 0.8255 | 16.8918 | 153.8132 | 0.7536 |
| threshold_branch_experts_t0.85 | 8.4466 | 28.5546 | 923.3083 | 0.8255 | 18.8781 | 156.5044 | 0.7536 |

Best real-holdout mode: `oracle_branch_experts`

![MAE comparison](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/compare_mae.png)

![RMSE comparison](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/compare_rmse.png)

## Per-Expert Histories

### smooth

![history](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/expert_smooth/history_log.png)

### left_edge

![history](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/expert_left_edge/history_log.png)

### right_edge

![history](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/expert_right_edge/history_log.png)

### apex

![history](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/expert_apex/history_log.png)

## Ensemble Figures

### baseline_reference

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/baseline_reference/synthetic/branch_confusion.png)

### oracle_branch_experts

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/oracle_branch_experts/synthetic/branch_confusion.png)

### predicted_branch_experts

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/predicted_branch_experts/synthetic/branch_confusion.png)

### threshold_branch_experts_t0.85

- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/real/parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/real/relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/real/error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/real/branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/synthetic/parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/synthetic/relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/synthetic/error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_branch_experts_20260313/threshold_branch_experts_t0.85/synthetic/branch_confusion.png)

## Best Real-Holdout Dissection

- mean relative sample error: `0.3703`
- median relative sample error: `0.0408`
- p90 relative sample error: `0.6806`
- p99 relative sample error: `5.6485`

Interpretation:

- The branch experts themselves are useful.
  Oracle routing improves real MAE from `8.2331` to `7.3426`, which is a real gain on the hardest benchmark we have.
- The current gate is the bottleneck.
  Predicted routing is worse than the baseline (`8.7657` vs `8.2331` real MAE), so the plastic experts are not being selected reliably enough.
- Confidence fallback helps, but not enough.
  Threshold routing reduces the damage relative to plain predicted routing, but still does not beat the baseline.
- This is the clearest positive result we have had since the baseline:
  the local plastic maps are learnable branch-by-branch, and the next high-value move is to improve the branch gate rather than changing the experts again immediately.

- If oracle routing improves a lot but predicted routing does not, the branch experts are good and the gate is the bottleneck.
- If predicted routing plus baseline fallback beats the baseline, this is immediately useful.
- If even oracle routing barely helps, branch-specialized raw experts are not enough and we should change the plastic target again.
