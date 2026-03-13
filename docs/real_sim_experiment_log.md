# Real Simulation Experiment Log

This is the running log for experiments aimed at improving fit on constitutive states sampled from `constitutive_problem_3D.h5`.

The baseline reports remain:

- `docs/real_sim_validation.md`
- `docs/real_sim_retraining.md`

This file starts from those results and records what was tried next, what changed, and what to test after that.

Update:

- the later long-horizon sweep in `docs/real_sim_long_sweep.md` supersedes the checkpoint ranking in this file
- the staged trainer study in `docs/staged_real_sim_training.md` supersedes both of those rankings
- the current best local checkpoint is now:
  - `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt`

## Baseline Before This Iteration

Best completed checkpoint at the start of this log:

- `experiment_runs/real_sim/sweep_256/raw_w512_d6/best.pt`

Reference numbers:

| Model | Training data | Test data | Stress MAE | Stress RMSE | Max abs | Branch acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Synthetic baseline `principal` | synthetic | real sampled `256` | 75.03 | 164.39 | 2570.13 | 0.434 |
| Real `principal_w512_d6` | real sampled `256` | real sampled `256` | 20.70 | 47.61 | 1503.24 | 0.738 |
| Real `raw_w512_d6` | real sampled `256` | real sampled `256` | 19.03 | 42.13 | 1003.42 | n/a |
| `raw_w512_d6` cross-eval | real sampled `256` | real sampled `512` | 19.11 | 42.41 | 1096.72 | n/a |

## Working Hypothesis

The best current read of the problem is:

- raw stress prediction is the right base representation for the real export
- the remaining error is concentrated in branch-dependent plastic tails
- the next likely improvements are branch-aware supervision and residual-to-trial targets

## Planned Iteration Queue

### Stage 1: branch awareness

Goal:

- test whether the raw model needs explicit regime separation

Planned run:

- `raw_branch_w512_d6`

### Stage 2: residual target

Goal:

- test whether predicting plastic correction is easier than predicting full stress

Planned runs:

- `trial_raw_residual_w512_d6`
- `trial_raw_branch_residual_w512_d6`

### Stage 3: tail emphasis

Goal:

- if tail errors remain dominant, explicitly optimize for them

Planned runs:

- weighted-loss version of the best Stage 1 or 2 model
- optional hard-example fine-tune

## Current Status

Code changes completed before running the next experiments:

- added `raw_branch`
- added `trial_raw_branch`
- added `trial_raw_residual`
- added `trial_raw_branch_residual`
- wired these model kinds through training, inference, CLI, and tests

Validation:

- `11 passed` on the focused test suite after those changes

## Results From This Iteration

### Completed Follow-Up Runs

All follow-up runs in this section used:

- training dataset: `experiment_runs/real_sim/baseline_sample_256.h5`
- cross-check dataset: `experiment_runs/real_sim/train_sample_512.h5`
- optimizer schedule: plateau-reduced Adam plus LBFGS finish
- device: CUDA

#### New runs

| Run | Primary split stress MAE | Cross split stress MAE | Cross split stress RMSE | Cross split max abs | Branch acc | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `raw_branch_w512_d6` | 9.20 | 9.27 | 25.31 | 1047.76 | 0.907 | winner |
| `trial_raw_residual_w512_d6` | 68.36 | 66.48 | 201.80 | 15086.38 | n/a | failed badly in actual stress space |
| `trial_raw_branch_residual_w512_d6` | 398.71 | 392.84 | 819.14 | 29306.14 | 0.873 | failed badly |
| `raw_branch_tail_w512_d6_a05` | 9.63 | 9.82 | 25.15 | 1129.04 | 0.907 | slightly better primary max-abs, worse overall |
| `raw_branch_w768_d8` | 9.22 | 9.31 | 26.82 | 1346.04 | 0.909 | no overall win over `w512_d6` |

#### Strongest comparison against the old best raw baseline

Old best raw baseline on the broader cross-check split:

- `stress_mae = 19.11`
- `stress_rmse = 42.41`
- highest stress quartile MAE: `34.44`

New winning branch-aware raw model on the same cross-check split:

- `stress_mae = 9.27`
- `stress_rmse = 25.31`
- highest stress quartile MAE: `15.21`
- branch accuracy: `0.907`

That is the central result of this iteration.

### Diagnostics for the winning run

Winning checkpoint:

- `experiment_runs/real_sim/iter_20260311/raw_branch_w512_d6/best.pt`

Saved evaluation outputs:

- primary split plots: `experiment_runs/real_sim/iter_20260311/raw_branch_w512_d6/eval_primary`
- cross split plots: `experiment_runs/real_sim/iter_20260311/raw_branch_w512_d6/eval_cross`
- consolidated follow-up metrics: `experiment_runs/real_sim/iter_20260311/followup_metrics_summary.json`

Key branch-wise MAE on the broader cross-check split:

- `elastic`: `9.42`
- `smooth`: `7.54`
- `left_edge`: `12.03`
- `right_edge`: `14.78`
- `apex`: `3.61`

### What We Learned

1. The raw representation was already correct; it just needed explicit regime supervision.
2. The auxiliary branch head was the highest-value change in the whole real-data study.
3. Residual-to-trial targets were misleading in normalized-loss space and poor in actual stress space.
4. Mild tail weighting did not improve the broad cross-check and is not the right next lever.
5. More width and depth did not beat the smaller `raw_branch_w512_d6` model where it matters.

### Decision

I am stopping the local experiment loop here.

Reason:

- `raw_branch_w512_d6` is clearly better than every other run in this iteration on the broad cross-check dataset
- the remaining variants did not produce a cleaner overall trade-off
- the next meaningful step is solver integration, not another blind local sweep

## Stop Condition

This log should stop only when one of these is true:

1. a new model is clearly better than the previous best raw checkpoint on real-data validation and looks strong enough for in-solver replacement testing
2. repeated targeted experiments fail to improve the current best raw model, which would mean the next bottleneck is solver integration rather than local surrogate fit

This stop condition is now satisfied by case `2`.
