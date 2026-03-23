# Cover-Layer Principal Correction

This report tests the corrected structured next step after the single-material execution study:

- principal-space plastic correction
- exact elastic gating through the branch head
- plastic-only regression loss
- checkpoint selection by real validation stress MAE

Reference control from the previous study:
- baseline checkpoint: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/best.pt`

## Result Table

| Experiment | Real MAE | Real RMSE | Real Max Abs | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_raw_branch_reference | 8.2331 | 22.9041 | 695.4951 | 0.8255 | 19.4116 | 156.1580 | 0.7536 |
| principal_trial_branch_residual_real | 146.7714 | 262.0139 | 17380.0312 | 0.7207 | 113.7457 | 237.0713 | 0.8787 |
| principal_trial_branch_residual_hybrid | 127.8092 | 223.6647 | 7806.4771 | 0.7118 | 80.4896 | 155.8859 | 0.8817 |

![Metric comparison](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/comparison_metrics.png)

![Per-branch MAE](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/comparison_per_branch.png)

## Figures

### baseline_raw_branch_reference

- history: ![history](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/history_log.png)
- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_branch_confusion.png)

### principal_trial_branch_residual_real

- history: ![history](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/history_log.png)
- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/real_parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/real_relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/real_error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/real_branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/synthetic_parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/synthetic_relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/synthetic_error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_real/eval/synthetic_branch_confusion.png)

### principal_trial_branch_residual_hybrid

- history: ![history](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/history_log.png)
- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/real_parity.png)
- real relative error: ![real rel](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/real_relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/real_error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/real_branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/synthetic_parity.png)
- synthetic relative error: ![synth rel](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/synthetic_relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/synthetic_error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_principal_correction_20260313/principal_trial_branch_residual_hybrid/eval/synthetic_branch_confusion.png)

## Best-Model Real Holdout Dissection

- mean relative sample error: `0.4528`
- median relative sample error: `0.2549`
- p90 relative sample error: `0.8746`
- p99 relative sample error: `3.7585`

### Per-Branch Mean Relative Error

- `elastic`: `0.4951`
- `smooth`: `0.3074`
- `left_edge`: `0.5620`
- `right_edge`: `0.6476`
- `apex`: `0.3226`

## Discussion

This experiment is successful only if the corrected principal-space model closes the gap to the old direct baseline on real stress metrics, not just on normalized training loss.

What happened in practice:

- The corrected principal-space models improved materially over the earlier broken residual formulation.
- The hybrid corrected model was better than the real-only corrected model on both real and synthetic holdouts.
- But neither corrected model came remotely close to the old direct baseline.

The most important diagnostic detail is branch-specific:

- elastic branch error is essentially solved by the corrected structured models
- the remaining failure is plastic, especially `smooth`, `left_edge`, `right_edge`, and `apex`
- so the exact-elastic-plus-plastic-correction idea is directionally right, but the current plastic target is still not a good enough representation

This narrows the next step a lot:

- do not abandon the direct baseline; it remains the only production-grade cover-layer model we have
- do not spend more time tuning this exact principal residual family
- move to a different plastic target or expert structure, now that we know elastic handling is not the blocker

Interpretation:

- If the corrected real-only model beats the old baseline, the core issue was target/metric design, not the structured idea itself.
- If the hybrid corrected model then improves further, `U/B` augmentation becomes worth keeping.
- If both remain behind the old baseline, the next move should be a different structured target altogether rather than more tuning of this residual family.
