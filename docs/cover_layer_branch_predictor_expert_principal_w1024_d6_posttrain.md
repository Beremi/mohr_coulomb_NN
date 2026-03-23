# Cover Layer Branch Predictor Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/best.pt`
- architecture: `w1024 d6`
- feature set: `trial_raw_material`
- generator: `principal_hybrid` / `expert_principal`
- accepted post-train loops: `0` out of `1` attempted

## Baseline vs Post-Train

- baseline synthetic test accuracy / macro recall: `0.9658` / `0.9655`
- post-train synthetic test accuracy / macro recall: `0.9658` / `0.9655`
- baseline real test accuracy / macro recall: `0.8928` / `0.8828`
- post-train real test accuracy / macro recall: `0.8928` / `0.8828`

## Loop Results

- loop `1` accepted: `False`
  start score: macro `0.9660`, acc `0.9660`
  end score: macro `0.9660`, acc `0.9660`
  epochs added: `120`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_posttrain_20260315/training_history.png)

## Confusions

![Confusions](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_posttrain_20260315/confusions.png)
