# Cover Layer Strain-Only Branch Predictor Model Card

## Summary

This card describes the new **synthetic-only strain-to-branch** classifier for `cover_layer`.

- task: `E -> branch`
- input: transformed engineering strain only (`asinh(E)` followed by z-score)
- output: one of `elastic / smooth / left_edge / right_edge / apex`
- no material or trial-stress features

## Architecture

- model type: shared pointwise MLP
- input dimension: `6`
- hidden width: `1024`
- depth: `6`
- activation: `GELU`
- output classes: `5`

## Synthetic Benchmark

- synthetic train seed calls: `24`
- synthetic eval seed calls: `8`
- synthetic val elements: `4096`
- synthetic test elements: `16384`
- generator: `seeded_local_noise_branch_balanced`

![Branch frequencies](../experiment_runs/real_sim/cover_layer_strain_branch_predictor_synth_only_20260314/benchmark_branch_frequencies.png)

## Training

- cycles: `4`
- batch-size schedule: `[64, 128, 256, 512, 1024]`
- base learning rates by cycle: `[0.001, 0.0005, 0.00025, 0.0001]`
- stage max epochs: `60`
- stage patience: `20`
- plateau patience/factor: `6 / 0.5`
- LBFGS tail per cycle: `3` epochs

Training regenerates synthetic data every epoch and checkpoints only on the fixed synthetic validation split.

## Convergence

![Training history](../experiment_runs/real_sim/cover_layer_strain_branch_predictor_synth_only_20260314/training_history.png)

## Metrics

- synthetic val accuracy / macro recall: `0.9186` / `0.9354`
- synthetic test accuracy / macro recall: `0.9186` / `0.9360`
- real val accuracy / macro recall: `0.4421` / `0.4639`
- real test accuracy / macro recall: `0.4382` / `0.4725`

## Confusions

![Confusions](../experiment_runs/real_sim/cover_layer_strain_branch_predictor_synth_only_20260314/confusions.png)

## Assessment

This run should first be judged by synthetic validation/test quality, because the whole purpose was to prove the synthetic-domain training loop itself. Real validation/test are diagnostic only.
