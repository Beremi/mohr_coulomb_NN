# Cover-Layer Single-Material Execution

This report executes the main cover-layer single-material workplan against the new full export and compares long-horizon models on a fixed real holdout split plus a fixed synthetic `U/B` holdout.

## Datasets

- Real exact-domain dataset: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_single_material_20260313/cover_layer_full_real_exact_256.h5`
  - total samples: `141824`
  - split counts: `{'train': 99328, 'val': 21248, 'test': 21248}`
  - branch counts: `{'elastic': 20500, 'smooth': 36666, 'left_edge': 34662, 'right_edge': 24731, 'apex': 25265}`
- Synthetic holdout dataset:
  - path: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_single_material_20260313/cover_layer_full_synthetic_holdout.h5`
  - total samples: `21248`
  - branch counts: `{'elastic': 7362, 'smooth': 10387, 'left_edge': 2065, 'right_edge': 1352, 'apex': 82}`
- Hybrid training dataset:
  - synthetic train rows added: `99328`
  - split counts: `{'train': 198656, 'val': 21248, 'test': 21248}`

## Experiments

Three long-horizon experiments were run:

1. `baseline_raw_branch`: direct raw-space stress regression with auxiliary branch head on the real exact-domain split.
2. `structured_trial_raw_branch_residual`: exact elastic trial features with a learned stress residual and branch head on the same real split.
3. `hybrid_trial_raw_branch_residual`: the same structured model trained on a hybrid dataset that mixes the real train split with `U/B`-pushforward synthetic augmentation.

All runs used long plateau training with the same width/depth and an LBFGS tail refinement.

Common training recipe:

- width/depth: `1024 x 6`
- optimizer: `AdamW`
- initial LR: `3e-4`
- scheduler: plateau, factor `0.5`, floor `1e-6`
- patience: `700`
- plateau patience: `120`
- batch size: `4096`
- LBFGS tail: `8` epochs

What differed between runs:

- `baseline_raw_branch` changed only the network head and target formulation: direct stress in raw feature space.
- `structured_trial_raw_branch_residual` changed the target to a trial-stress residual with the same cover-layer real split.
- `hybrid_trial_raw_branch_residual` kept that structured residual model, but doubled the train split with `99,328` additional `U/B`-generated synthetic rows.

## Result Table

| Experiment | Real MAE | Real RMSE | Real Max Abs | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_raw_branch | 8.2331 | 22.9041 | 695.4951 | 0.8255 | 19.4116 | 156.1580 | 0.7536 |
| structured_trial_raw_branch_residual | 550.1094 | 887.6166 | 26850.1445 | 0.9088 | 333.1902 | 638.0900 | 0.9523 |
| hybrid_trial_raw_branch_residual | 447.6032 | 920.7029 | 33470.1289 | 0.8536 | 172.5689 | 255.6980 | 0.9472 |

Best real-test model in this study: `baseline_raw_branch`.

![Metric comparison](../experiment_runs/real_sim/cover_layer_single_material_20260313/comparison_metrics.png)

![Per-branch MAE](../experiment_runs/real_sim/cover_layer_single_material_20260313/comparison_per_branch.png)

## Per-Experiment Figures

### baseline_raw_branch

- training history: ![history](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/history_log.png)
- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_parity.png)
- real relative error CDF: ![real rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/real_branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_parity.png)
- synthetic relative error CDF: ![synth rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/eval/synthetic_branch_confusion.png)

### structured_trial_raw_branch_residual

- training history: ![history](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/history_log.png)
- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/real_parity.png)
- real relative error CDF: ![real rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/real_relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/real_error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/real_branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/synthetic_parity.png)
- synthetic relative error CDF: ![synth rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/synthetic_relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/synthetic_error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/structured_trial_raw_branch_residual/eval/synthetic_branch_confusion.png)

### hybrid_trial_raw_branch_residual

- training history: ![history](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/history_log.png)
- real parity: ![real parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/real_parity.png)
- real relative error CDF: ![real rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/real_relative_error_cdf.png)
- real error vs magnitude: ![real mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/real_error_vs_magnitude.png)
- real branch confusion: ![real branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/real_branch_confusion.png)
- synthetic parity: ![synth parity](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/synthetic_parity.png)
- synthetic relative error CDF: ![synth rel](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/synthetic_relative_error_cdf.png)
- synthetic error vs magnitude: ![synth mag](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/synthetic_error_vs_magnitude.png)
- synthetic branch confusion: ![synth branch](../experiment_runs/real_sim/cover_layer_single_material_20260313/hybrid_trial_raw_branch_residual/eval/synthetic_branch_confusion.png)

## Real-Test Dissection Of The Best Model

Best model: `baseline_raw_branch`

- mean relative sample error: `0.4528`
- median relative sample error: `0.2549`
- p90 relative sample error: `0.8746`
- p99 relative sample error: `3.7585`

### Error By Stress-Magnitude Bin

| Stress-Magnitude Bin | N | Sample MAE | Component MAE | Mean Relative |
|---|---:|---:|---:|---:|
| 6.01 to 37.16 | 5312 | 17.9474 | 5.7143 | 0.9181 |
| 37.16 to 45.00 | 4887 | 15.7926 | 4.9193 | 0.3641 |
| 45.00 to 80.90 | 5737 | 16.5599 | 5.2510 | 0.2839 |
| 80.90 to 174.58 | 3187 | 28.9796 | 9.2980 | 0.2605 |
| 174.58 to 1473.49 | 2125 | 89.7537 | 28.6046 | 0.2384 |

### Per-Branch Mean Relative Error

- `elastic`: `0.4951`
- `smooth`: `0.3074`
- `left_edge`: `0.5620`
- `right_edge`: `0.6476`
- `apex`: `0.3226`

### Worst Real Holdout Calls

| Call | N | Component MAE | Sample MAE | Mean Relative |
|---|---:|---:|---:|---:|
| call_000510 | 256 | 20.0337 | 62.8385 | 0.8129 |
| call_000538 | 256 | 18.7084 | 58.4296 | 0.6625 |
| call_000534 | 256 | 18.5626 | 58.4609 | 0.6960 |
| call_000546 | 256 | 18.4907 | 58.5933 | 0.8521 |
| call_000531 | 256 | 18.3468 | 57.5307 | 0.7003 |
| call_000471 | 256 | 17.9876 | 56.7111 | 0.7234 |
| call_000494 | 256 | 17.1550 | 54.3173 | 0.5754 |
| call_000464 | 256 | 16.4394 | 52.4566 | 0.5569 |
| call_000473 | 256 | 16.3183 | 51.4827 | 0.6775 |
| call_000472 | 256 | 16.1340 | 50.4877 | 0.7164 |

## Discussion

The important comparison in this study is not only real-test error, but the gap between the real holdout and the synthetic `U/B` holdout.

- If a model is good on synthetic but poor on real, the training distribution is still off.
- If both improve together, the structure and the augmentation are helping in the right direction.
- The hybrid result is the key test of whether the new full export is already useful for FE-compatible augmentation.

What actually happened here:

- The direct raw baseline was the only model that stayed numerically stable in stress space on both holdouts.
- The residual models achieved much lower normalized validation loss during training, but that did **not** translate into correct stress prediction.
- So for this setup, the residual formulation is currently optimizing the wrong practical objective even though its training curves look better.
- The `U/B` augmentation did improve the structured model relative to the non-hybrid structured run, but it was nowhere near enough to beat the direct baseline.

This means the current bottleneck is not just “more FE-compatible synthetic data.” It is the combination of:

- target parameterization
- checkpoint-selection metric
- and possibly the feature scaling for residual stress

The real-test dissection supports that reading:

- worst errors are concentrated in later calls with larger deformation/stress states
- relative error is highest in the lowest-stress bin, but the absolute damage is in the high-stress tail
- `left_edge` and `right_edge` remain the hardest constitutive regions for the winning baseline

The best next move after this study should follow the winner:

- If the structured residual model wins without augmentation, keep the exact-elastic-plus-plastic-correction structure and improve the representation next.
- If the hybrid model wins, make `U/B` augmentation part of the default cover-layer training recipe.
- If neither beats the older heavy fitted-refresh baseline convincingly, the next bottleneck is still the synthetic state generator rather than the network itself.

For the next iteration, the most defensible move is:

1. keep `baseline_raw_branch` as the control
2. redesign the residual model selection metric so checkpoints are chosen by true stress error, not only normalized residual loss
3. only then retry `U/B` augmentation on that corrected structured model
