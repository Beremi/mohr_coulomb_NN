# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_resample_v2_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training dataset per resample: `81920` pointwise samples
- training batch size: `8192` pointwise samples
- fixed LR: `1.0e-06`
- per-loop patience before resampling: `100`
- attempted loops: `10`
- accepted loops: `2`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9663` / `0.9657`
- final synthetic test accuracy / macro recall: `0.9663` / `0.9656`
- baseline real test accuracy / macro recall: `0.8931` / `0.8832`
- final real test accuracy / macro recall: `0.8956` / `0.8859`

## Loop Results

- loop `1` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8931`, macro `0.8832`
  epochs run before resample/stop: `100`
- loop `2` accepted: `True`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8935`, macro `0.8835`
  epochs run before resample/stop: `101`
- loop `3` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8935`, macro `0.8835`
  epochs run before resample/stop: `100`
- loop `4` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8935`, macro `0.8835`
  epochs run before resample/stop: `100`
- loop `5` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8935`, macro `0.8835`
  epochs run before resample/stop: `100`
- loop `6` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8935`, macro `0.8835`
  epochs run before resample/stop: `100`
- loop `7` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9664`
  best loop real test: acc `0.8935`, macro `0.8835`
  epochs run before resample/stop: `100`
- loop `8` accepted: `True`
  best synthetic val score: macro `0.9662`, acc `0.9666`
  best loop real test: acc `0.8956`, macro `0.8859`
  epochs run before resample/stop: `115`
- loop `9` accepted: `False`
  best synthetic val score: macro `0.9662`, acc `0.9666`
  best loop real test: acc `0.8956`, macro `0.8859`
  epochs run before resample/stop: `100`
- loop `10` accepted: `False`
  best synthetic val score: macro `0.9662`, acc `0.9666`
  best loop real test: acc `0.8956`, macro `0.8859`
  epochs run before resample/stop: `100`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_resample_10xdata_lr1e6_20260315/training_history.png)
