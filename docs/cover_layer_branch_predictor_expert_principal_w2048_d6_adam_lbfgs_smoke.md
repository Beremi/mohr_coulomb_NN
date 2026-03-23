# Cover Layer Branch Predictor Inflated Adam Then LBFGS Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/best.pt`
- width inflation: `1024 -> 2048`
- inflation noise scale: `1.0e-09`
- Adam phase: `1` datasets, lr `1.0e-06`, batch `8192`, train points `8192`
- LBFGS phase: `1` datasets, `1` steps each, lr `1.0e-02`

## Inflation Check

- synthetic test accuracy / macro recall: `0.9659` / `0.9647` -> `0.9659` / `0.9647`
- real test accuracy / macro recall: `0.8972` / `0.8878` -> `0.8972` / `0.8878`
- prediction change rate on synthetic val / synthetic test / real val / real test: `0.000000` / `0.000000` / `0.000000` / `0.000000`

## Phase Results

- after Adam synthetic test accuracy / macro recall: `0.9658` / `0.9647`
- after Adam real test accuracy / macro recall: `0.8974` / `0.8879`
- after LBFGS synthetic test accuracy / macro recall: `0.9660` / `0.9647`
- after LBFGS real test accuracy / macro recall: `0.8979` / `0.8885`

## Checkpoints

- inflated checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs_smoke_20260316/inflated_init.pt`
- after Adam: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs_smoke_20260316/after_adam.pt`
- after LBFGS: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs_smoke_20260316/after_lbfgs.pt`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs_smoke_20260316/training_history.png)
