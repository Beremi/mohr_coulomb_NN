# Cover Layer Branch Predictor Model Card

## Summary

This card describes the current **synthetic-only** cover-layer branch predictor baseline.

- checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_baseline_20260314/best.pt`
- material scope: `cover_layer` only
- task: predict one of `elastic / smooth / left_edge / right_edge / apex`
- prediction granularity: pointwise, one prediction per integration point
- deployment intent: use exact FE kinematics to compute `E`, then run the classifier

## Architecture

- model type: shared pointwise MLP
- input dimension: `17`
- hidden width: `256`
- depth: `4`
- output classes: `5`
- activation: `GELU`
- parameter count: `203,269`

Input features are built with `build_trial_features(...)`:

- `asinh(strain_eng)` -> `6`
- `asinh(trial_stress / c_bar)` -> `6`
- reduced material features -> `5`

Total: `17` features per integration point.

## Training Setup

- training data: synthetic only
- synthetic generator mode: `empirical_local_noise`
- synthetic elements per epoch: `300`
- synthetic holdout elements: `500`
- real fit calls used to fit the synthetic generator: `4`
- real validation calls: `2`
- real test calls: `2`
- max elements per call in this baseline run: `64`
- optimizer: `AdamW`
- learning rate: `0.001`
- weight decay: `0.0001`
- epoch budget: `60`
- early-stop patience: `15`
- best epoch: `13`

This run regenerates synthetic training data every epoch, but **never trains on real labels**.

## Convergence

![Training History](../experiment_runs/real_sim/cover_layer_branch_predictor_model_card_20260314/training_history.png)

What to look at:

- the train loss decreases cleanly
- real validation and real test accuracy rise quickly, then plateau early
- the plateau happens while macro recall stays poor, which is a sign of class collapse rather than simple undertraining

## Accuracy

| Split | Accuracy | Macro Recall |
|---|---:|---:|
| Real val | 0.6236 | 0.4011 |
| Real test | 0.7031 | 0.4005 |
| Synthetic holdout | 0.6260 | 0.5019 |

So the direct answer to the synthetic-test question is:

- synthetic holdout accuracy: `0.6260`
- synthetic holdout macro recall: `0.5019`

## Per-Branch Recall

![Per-Branch Recall](../experiment_runs/real_sim/cover_layer_branch_predictor_model_card_20260314/recalls.png)

| Branch | Real val | Real test | Synthetic holdout |
|---|---:|---:|---:|
| elastic | 1.0000 | 0.9986 | 0.9020 |
| smooth | 0.0057 | 0.0000 | 0.4271 |
| left_edge | 0.0000 | 0.0041 | 0.0542 |
| right_edge | 0.0000 | 0.0000 | 0.1261 |
| apex | 1.0000 | 1.0000 | 1.0000 |

## Confusion Structure

![Confusion Matrices](../experiment_runs/real_sim/cover_layer_branch_predictor_model_card_20260314/confusions.png)

Key interpretation:

- `elastic` and `apex` are learned strongly
- `smooth`, `left_edge`, and `right_edge` remain weak
- the same qualitative failure appears on both real and synthetic holdout data
- that points back to the synthetic generator / class coverage rather than a purely real-vs-synthetic gap

## Current Assessment

This model is **not good enough** yet for the intended branch-prediction role.

What it proves:

- the synthetic-only training loop works
- the model can learn stable easy branches
- the current synthetic generator is informative enough to give nontrivial real accuracy

What it does not prove:

- reliable hard-branch prediction
- near-perfect real fit
- suitability for a sharp gate

## Recommended Next Step

Do not scale this classifier first.

The next move should stay on the generator side:

- improve synthetic coverage of `smooth`, `left_edge`, and `right_edge`
- make the generator more branch-balanced in the hard regions
- then rerun this same classifier as the control
