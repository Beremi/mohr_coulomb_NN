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
- hidden width: `256`
- depth: `3`
- activation: `GELU`
- output classes: `5`
- plastic loss weight: `1.0`

## Synthetic Benchmark

- synthetic train seed calls: `24`
- synthetic eval seed calls: `8`
- synthetic val elements: `16384`
- synthetic test elements: `16384`
- generator: `principal_hybrid`

![Branch frequencies](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_20260315/benchmark_branch_frequencies.png)

## Training

- cycles: `4`
- batch-size schedule: `[64, 128, 256, 512, 1024]`
- base learning rates by cycle: `[0.001, 0.000215443469003, 4.64158883361e-05, 1e-05]`
- stage max epochs: `60`
- stage patience: `20`
- plateau patience/factor: `20 / 0.5`
- LBFGS tail per cycle: `3` epochs

Training regenerates synthetic data every epoch and checkpoints only on the fixed synthetic validation split.

## Convergence

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_20260315/training_history.png)

## Metrics

- synthetic val accuracy / macro recall: `0.9575` / `0.9582`
- synthetic test accuracy / macro recall: `0.9572` / `0.9576`
- real val accuracy / macro recall: `0.8693` / `0.8646`
- real test accuracy / macro recall: `0.8865` / `0.8750`

## Confusions

![Confusions](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_20260315/confusions.png)

## Assessment

This run should first be judged by synthetic validation/test quality, because the whole purpose was to prove the synthetic-domain training loop itself. Real validation/test are diagnostic only.
