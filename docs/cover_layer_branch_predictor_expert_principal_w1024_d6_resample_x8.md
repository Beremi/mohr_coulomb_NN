# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training batch size: `65536` pointwise samples
- fixed LR: `1.0e-05`
- per-loop patience before resampling: `20`
- attempted loops: `10`
- accepted loops: `1`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9658` / `0.9655`
- final synthetic test accuracy / macro recall: `0.9660` / `0.9656`
- baseline real test accuracy / macro recall: `0.8928` / `0.8828`
- final real test accuracy / macro recall: `0.8926` / `0.8826`

## Loop Results

- loop `1` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run before resample/stop: `20`
- loop `2` accepted: `True`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `22`
- loop `3` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `4` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `5` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `6` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `7` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `8` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `9` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`
- loop `10` accepted: `False`
  best synthetic val score: macro `0.9661`, acc `0.9661`
  best loop real test: acc `0.8926`, macro `0.8826`
  epochs run before resample/stop: `20`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_resample_x8_20260315/training_history.png)
