# Cover Layer Branch Predictor Model Card (Staged, Seeded Branch-Balanced Generator)

## Summary

This card describes the current **synthetic-only** cover-layer branch predictor run.

- checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314/best.pt`
- material scope: `cover_layer` only
- task: predict one of `elastic / smooth / left_edge / right_edge / apex`
- prediction granularity: pointwise, one prediction per integration point
- deployment intent: use exact FE kinematics to compute `E`, then run the classifier

## Architecture

- model type: shared pointwise MLP
- input dimension: `17`
- hidden width: `512`
- depth: `6`
- output classes: `5`
- activation: `GELU`
- parameter count: `1,325,061`

Input features are built with `build_trial_features(...)`:

- `asinh(strain_eng)` -> `6`
- `asinh(trial_stress / c_bar)` -> `6`
- reduced material features -> `5`

Total: `17` features per integration point.

## Training Setup

- training data: synthetic only
- synthetic generator mode: `seeded_local_noise_branch_balanced`
- synthetic elements per epoch: `2500`
- synthetic holdout elements: `1500`
- real fit calls used to fit the synthetic generator: `12`
- real validation calls: `4`
- real test calls: `4`
- max elements per call in this run: `128`
- optimizer: `AdamW`
- base learning rate: `0.001`
- weight decay: `0.0001`
- cycles: `3`
- batch-size schedule per cycle: `[64, 128, 256, 512, 1024]`
- stage max epochs: `45`
- plateau patience/factor: `5` / `0.5`
- best epoch: `303`

This run regenerates synthetic training data throughout training and never uses real labels for optimization.

## Convergence

![Training History](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_model_card_20260314/training_history.png)

What to look at:

- training loss should keep decreasing without the early collapse we saw in the short baseline
- real validation, real test, and synthetic holdout curves are plotted together
- if stage markers are present, they show batch-size transitions across cycles

## Accuracy

| Split | Accuracy | Macro Recall |
|---|---:|---:|
| Real val | 0.5639 | 0.5624 |
| Real test | 0.6000 | 0.5741 |
| Synthetic holdout | 0.9763 | 0.9827 |

So the direct answer to the synthetic-test question is:

- synthetic holdout accuracy: `0.9763`
- synthetic holdout macro recall: `0.9827`

## Per-Branch Recall

![Per-Branch Recall](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_model_card_20260314/recalls.png)

| Branch | Real val | Real test | Synthetic holdout |
|---|---:|---:|---:|
| elastic | 0.9574 | 0.9324 | 1.0000 |
| smooth | 0.0026 | 0.0051 | 0.9471 |
| left_edge | 0.5425 | 0.5275 | 0.9847 |
| right_edge | 0.8065 | 0.8223 | 0.9926 |
| apex | 0.5032 | 0.5829 | 0.9891 |

## Confusion Structure

![Confusion Matrices](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_model_card_20260314/confusions.png)

Key interpretation:

- if real and synthetic confusion look similar, the bottleneck is likely still the synthetic generator
- if synthetic is good but real is poor, the bottleneck is transfer / coverage mismatch
- for this task, `smooth`, `left_edge`, and `right_edge` are the branches to watch

## Current Assessment

This model should be judged by hard-branch recall and macro recall, not overall accuracy alone.
