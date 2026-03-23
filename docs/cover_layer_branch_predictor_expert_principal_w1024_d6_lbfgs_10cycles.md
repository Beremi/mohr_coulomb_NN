# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training dataset per resample: `81920` pointwise samples
- training batch size: `8192` pointwise samples
- optimizer: `lbfgs`
- fixed LR: `1.0e-01`
- per-loop patience before resampling: `1`
- acceptance mode: `improve_only`
- attempted loops: `10`
- accepted loops: `2`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9659` / `0.9647`
- final synthetic test accuracy / macro recall: `0.9660` / `0.9650`
- baseline real test accuracy / macro recall: `0.8972` / `0.8878`
- final real test accuracy / macro recall: `0.8963` / `0.8863`

## Loop Results

- loop `1` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `2` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `3` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `4` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `5` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `6` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `7` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `8` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `9` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `10` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_10cycles_20260315/training_history.png)
