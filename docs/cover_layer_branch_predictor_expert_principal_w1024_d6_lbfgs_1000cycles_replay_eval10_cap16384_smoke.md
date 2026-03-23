# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training dataset per resample: `81920` pointwise samples
- training batch size: `8192` pointwise samples
- replay mode: `misprediction_bank`
- replay cap per cycle: `16384`
- final replay bank size: `1191`
- optimizer: `lbfgs`
- fixed LR: `1.0e-01`
- per-loop patience before resampling: `1`
- eval/checkpoint interval: every `10` loops
- acceptance mode: `improve_only`
- attempted loops: `2`
- accepted loops: `0`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9659` / `0.9647`
- final synthetic test accuracy / macro recall: `0.9659` / `0.9647`
- baseline real test accuracy / macro recall: `0.8972` / `0.8878`
- final real test accuracy / macro recall: `0.8972` / `0.8878`

## Loop Results

- loop `2` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_1000cycles_replay_eval10_cap16384_20260315_smoke/training_history.png)
