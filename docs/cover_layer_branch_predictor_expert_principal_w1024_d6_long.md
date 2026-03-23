# Cover Layer Branch Predictor Model Card

## Summary

This card describes the current **synthetic-only branch** classifier for `cover_layer`.

- task: `E -> branch`
- feature set: `trial_raw_material`
- model type: `hierarchical`
- generator recipe: `expert_principal`
- input: engineering strain, elastic trial stress, and reduced material features
- output: one of `elastic / smooth / left_edge / right_edge / apex`
- detail: includes `asinh(E)`, `asinh(sigma_trial / c_bar)`, and reduced material logs

## Architecture

- model type: `hierarchical`
- input dimension: `17`
- hidden width: `1024`
- depth: `6`
- activation: `GELU`
- output classes: `5`
- plastic loss weight: `1.0`

## Synthetic Benchmark

- synthetic train seed calls: `24`
- synthetic eval seed calls: `8`
- synthetic val elements: `16384`
- synthetic test elements: `16384`
- generator: `principal_hybrid`

![Branch frequencies](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/benchmark_branch_frequencies.png)

## Training

- cycles: `4`
- batch-size schedule: `[64, 128, 256, 512, 1024]`
- base learning rates by cycle: `[0.001, 0.000215443469003, 4.64158883361e-05, 1e-05]`
- stage max epochs: `80`
- stage patience: `30`
- plateau patience/factor: `25 / 0.5`
- LBFGS tail per cycle: `5` epochs

Training regenerates synthetic data every epoch and checkpoints only on the fixed synthetic validation split.

## Convergence

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/training_history.png)

## Metrics

- synthetic val accuracy / macro recall: `0.9660` / `0.9660`
- synthetic test accuracy / macro recall: `0.9658` / `0.9655`
- real val accuracy / macro recall: `0.8787` / `0.8745`
- real test accuracy / macro recall: `0.8928` / `0.8828`

## Confusions

![Confusions](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/confusions.png)

## Assessment

This run should first be judged by synthetic validation/test quality, because the whole purpose was to prove the synthetic-domain training loop itself. Real validation/test are diagnostic only.
