# Real Simulation Retraining

## Scope

This report follows the baseline validation in `docs/real_sim_validation.md` and focuses on what changed after moving from synthetic-only training to real sampled constitutive states exported from the slope-stability solve.

The main changes in this phase were:

- corrected the Python constitutive operator to match the upstream MATLAB/Octave `constitutive_problem_3D` formulation used by the captured export
- added a real-export sampling path that converts `constitutive_problem_3D.h5` into framework HDF5 datasets with exact branch labels
- updated training/evaluation to support batched checkpoint inference on large datasets and to tolerate datasets without branch labels
- tested three real-data surrogate families:
  - `principal` on sampled real data
  - `raw` on sampled real data
  - `trial_raw`, a raw-stress model with signed `asinh` strain features and signed `asinh` elastic trial-stress features

## Data Used

- Real sampled dataset for sweep: `experiment_runs/real_sim/baseline_sample_256.h5`
  - `141824` samples
  - `256` states per captured constitutive call
- Broader real sampled dataset for stress-test / cross-check: `experiment_runs/real_sim/train_sample_512.h5`
  - `283648` samples
  - `512` states per captured constitutive call

The export/exact consistency stayed at machine precision:

- `exact_match_mae ~= 4.6e-13`
- `exact_match_rmse ~= 3.0e-12`
- `exact_match_max_abs ~= 2.5e-10`

So the remaining error is learned-model error, not label corruption.

## Main Comparison

### Baseline vs Real-Data Retrains on the `256`-per-call Real Test Split

| Model | Training source | Stress MAE | Stress RMSE | Stress max abs | Branch accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic baseline `principal` | synthetic only | 75.0274 | 164.3916 | 2570.1333 | 0.4343 |
| Real-trained `principal_w512_d6` | `baseline_sample_256.h5` | 20.7010 | 47.6118 | 1503.2391 | 0.7379 |
| Real-trained `raw_w512_d6` | `baseline_sample_256.h5` | 19.0299 | 42.1296 | 1003.4167 | n/a |

Key point:

- moving to real sampled states reduced stress MAE by about `74.6%` for the best model (`75.03 -> 19.03`)
- the best completed model in this study is the plain `raw` model retrained on the real sampled dataset

### Generalization Check on the Broader `512`-per-call Real Test Split

| Model | Checkpoint source | Test dataset | Stress MAE | Stress RMSE | Stress max abs |
| --- | --- | --- | ---: | ---: | ---: |
| `raw_256_on_512` | `sweep_256/raw_w512_d6/best.pt` | `train_sample_512.h5` | 19.1084 | 42.4101 | 1096.7228 |
| `principal_256_on_512` | `sweep_256/principal_w512_d6/best.pt` | `train_sample_512.h5` | 20.6967 | 47.4673 | 1663.1201 |
| `trial_raw_512` | `final_trial_raw_512/best.pt` | `train_sample_512.h5` | 21.0530 | 56.3501 | 2068.5413 |

This is the most important table in the report.

The winner remains the `raw` model trained on the `256`-per-call real sample, even when evaluated on the broader `512`-per-call real sample drawn with a different seed.

## What Was Tested

### `principal_w512_d6` on real sampled data

- run dir: `experiment_runs/real_sim/sweep_256/principal_w512_d6`
- best epoch: `170`
- best validation loss: `0.184576`
- best validation stress MSE: `2470.03`
- training history: `experiment_runs/real_sim/sweep_256/principal_w512_d6/training_history.png`
- eval plots:
  - `experiment_runs/real_sim/sweep_256/principal_w512_d6/eval_plots/parity_stress.png`
  - `experiment_runs/real_sim/sweep_256/principal_w512_d6/eval_plots/parity_principal.png`
  - `experiment_runs/real_sim/sweep_256/principal_w512_d6/eval_plots/stress_error_hist.png`
  - `experiment_runs/real_sim/sweep_256/principal_w512_d6/eval_plots/branch_confusion.png`

Strengths:

- best branch-aware model
- much stronger branch classification than the synthetic baseline
- elastic and smooth branches improved strongly

Weaknesses:

- higher stress RMSE and max absolute error than the best `raw` model
- edge and apex behavior remained harder than for the `raw` winner

### `raw_w512_d6` on real sampled data

- run dir: `experiment_runs/real_sim/sweep_256/raw_w512_d6`
- best epoch: `141`
- best validation loss: `0.147042`
- best validation stress MSE: `1826.78`
- training history: `experiment_runs/real_sim/sweep_256/raw_w512_d6/training_history.png`
- eval plots on the `256`-per-call real test split:
  - `experiment_runs/real_sim/sweep_256/raw_w512_d6/eval_plots/parity_stress.png`
  - `experiment_runs/real_sim/sweep_256/raw_w512_d6/eval_plots/stress_error_hist.png`
- cross-eval plots on the broader `512`-per-call real test split:
  - `experiment_runs/real_sim/cross_eval/raw_256_on_512/parity_stress.png`
  - `experiment_runs/real_sim/cross_eval/raw_256_on_512/stress_error_hist.png`

Strengths:

- best stress MAE, RMSE, and max-abs error among the completed and cross-checked runs
- generalized well from the `256` sample to the broader `512` sample
- much better apex and edge errors than the principal model on the same broad sample

Weaknesses:

- no auxiliary branch head, so it gives up direct branch classification
- elastic branch error is higher than the principal model’s elastic-branch error

### `trial_raw` with signed `asinh` trial-stress features

- run dir: `experiment_runs/real_sim/final_trial_raw_512`
- best epoch: `91`
- best validation loss: `0.269803`
- best validation stress MSE: `3396.54`
- training history: `experiment_runs/real_sim/final_trial_raw_512/training_history.png`
- eval plots:
  - `experiment_runs/real_sim/final_trial_raw_512/eval_plots/parity_stress.png`
  - `experiment_runs/real_sim/final_trial_raw_512/eval_plots/stress_error_hist.png`

Takeaway:

- the signed transform idea was reasonable and did not fail outright
- on the broader `512`-sample test set it was competitive in MAE, but it did not beat the simpler `raw_256_on_512` checkpoint
- it also carried worse RMSE and max-abs error than the winning `raw` model

So the transform is interesting, but not the current best choice.

## Insights

### 1. The original bottleneck was not just “network capacity”

The biggest gains came from:

- fixing the exact constitutive target to match the upstream MATLAB code
- training on real sampled states that actually occur in the slope-stability solve

That mattered more than simply making the network deeper or training longer on the synthetic generator.

### 2. Real-data alignment beat broader-but-harder retraining

The best checkpoint in this phase was not one of the larger fresh retrains on the `512`-per-call dataset.

The strongest result came from the `raw` model trained on the smaller but still representative `256`-per-call real sample, and it still generalized well to the broader `512`-per-call test split.

That suggests:

- better sampling alignment matters
- optimization stability matters
- more data alone is not enough unless the schedule and weighting also match the harder sample distribution

### 3. Direct raw-stress prediction is currently the better fit for the real export

On the real sampled data, the direct `raw` model beat the principal-stress model on:

- stress MAE
- stress RMSE
- max absolute error

This is consistent with the earlier diagnosis that principal reconstruction is not obviously the best inductive bias once the target is the actual exported constitutive response rather than the older synthetic operator.

### 4. The branch head is useful diagnostically, but not yet enough to justify the principal model as the final choice

The principal model still has value:

- branch accuracy rose from `0.4343` to `0.7379`
- it remains the best branch-aware model in the repo

But for pure stress prediction on the real sampled states, the raw model is stronger right now.

## Current Recommendation

If we want the best current surrogate for real constitutive states from this slope-stability workflow, the recommended checkpoint is:

- `experiment_runs/real_sim/sweep_256/raw_w512_d6/best.pt`

Supporting reason:

- it is the best performer on the original real sampled test split
- it remains the best performer when cross-checked on the broader `512`-per-call real test split

## Is It Good Enough To Replace The Constitutive Law In Limit Analysis?

Not yet, not with the standard “keep the result the same” bar.

Why not:

- even the best current checkpoint still has stress MAE around `19.1` and RMSE around `42.4` on the broader real sampled test split
- max absolute stress error is still around `1.10e3`
- the model still predicts stress only; it does not provide the consistent tangent operator
- we still have not inserted the surrogate into the actual limit-analysis solve and compared:
  - factor of safety / collapse multiplier
  - convergence behavior of the nonlinear solve
  - plastic-zone localization
  - final displacement / stress fields

So the correct conclusion is:

- this is now a much stronger local constitutive surrogate than the synthetic-only baseline
- it is good enough to justify an in-solver replacement experiment
- it is not yet proven good enough to replace the constitutive relationship in production and preserve the same limit-analysis result

## Recommended Next Step

The next technical step is no longer another blind architecture sweep.

It is:

1. plug `raw_w512_d6/best.pt` into the actual constitutive call site
2. run the same slope-stability case side-by-side with the exact constitutive law
3. compare solver convergence and final safety-factor outputs directly

That experiment will answer the real question this project cares about.
