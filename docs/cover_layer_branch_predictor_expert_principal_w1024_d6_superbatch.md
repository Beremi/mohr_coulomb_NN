# Cover Layer Branch Predictor Superbatch Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- superbatch size: `8192` pointwise samples
- attempted loops: `10`
- accepted loops: `0`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9658` / `0.9655`
- final synthetic test accuracy / macro recall: `0.9658` / `0.9655`
- baseline real test accuracy / macro recall: `0.8928` / `0.8828`
- final real test accuracy / macro recall: `0.8928` / `0.8828`

## Loop Results

- loop `1` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `2` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `3` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `4` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `5` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `6` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `7` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `8` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `9` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`
- loop `10` accepted: `False`
  best synthetic val score: macro `0.9660`, acc `0.9660`
  best loop real test: acc `0.8928`, macro `0.8828`
  epochs run: `40`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_superbatch_20260315/training_history.png)
