# Real Simulation Long Sweep

This note records the longer-horizon retraining pass requested after the earlier real-data study looked undertrained.

Update:

- the later staged-training study in `docs/staged_real_sim_training.md` supersedes the checkpoint ranking in this file

The previous best real-data checkpoint before this sweep was:

- `experiment_runs/real_sim/iter_20260311/raw_branch_w512_d6/best.pt`

Its key broad-split metrics were:

- `stress_mae = 9.27`
- `stress_rmse = 25.31`
- `stress_max_abs = 1047.76`
- `branch_accuracy = 0.907`

## Goal

Test whether the earlier real-data runs were stopped too early.

The focus in this sweep was deliberately narrow:

- keep the winning `raw_branch` model family
- use much longer plateau training
- keep LBFGS finishing
- compare training on both the `256`-sample and `512`-sample real datasets
- prepare larger-capacity runs, but stop early if schedule length clearly dominated capacity

## Sweep Setup

Driver script:

- `scripts/experiments/long_real_sim_sweep.py`

Datasets:

- `experiment_runs/real_sim/baseline_sample_256.h5`
- `experiment_runs/real_sim/train_sample_512.h5`

Common schedule choices:

- `scheduler_kind = plateau`
- `plateau_factor = 0.5`
- `min_lr = 1e-7`
- Adam with much larger patience than before
- LBFGS finish enabled in the scripted sweep

## Runs Actually Carried Far Enough To Judge

### 1. `rb_d256_w512_d6_long`

Training dataset:

- `baseline_sample_256.h5`

Training config:

- width `512`
- depth `6`
- batch size `4096`
- lr `1e-3`
- epochs budget `900`
- early-stop patience `220`
- plateau patience `30`
- LBFGS epochs `12`

Outcome:

- completed cleanly
- best epoch: `860`
- best validation loss: `0.149319`
- best checkpoint: `experiment_runs/real_sim/long_sweep_20260311/rb_d256_w512_d6_long/best.pt`

Evaluation:

| Test split | Stress MAE | Stress RMSE | Max abs | Branch acc |
| --- | ---: | ---: | ---: | ---: |
| `d256` | `8.36` | `30.45` | `1078.90` | `0.912` |
| `d512` | `8.39` | `31.10` | `1248.76` | `0.911` |

Interpretation:

- average fit improved strongly over the old best model
- top-decile error improved strongly
- but RMSE and max-abs got worse
- the model over-improved easy elastic/smooth states and still left ugly edge outliers

This run was an important diagnostic result, but not the final winner.

### 2. `rb_d512_w512_d6_long`

Training dataset:

- `train_sample_512.h5`

Training config:

- width `512`
- depth `6`
- batch size `4096`
- lr `1e-3`
- epochs budget `900`
- early-stop patience `220`
- plateau patience `30`
- LBFGS epochs `12`

Outcome:

- training was manually stopped after the curve had flattened and a clearly better checkpoint had already been saved
- the best checkpoint remained available at:
  - `experiment_runs/real_sim/long_sweep_20260311/rb_d512_w512_d6_long/best.pt`
- best history row before interruption:
  - epoch `340`
  - validation loss `0.0969998`
  - validation stress MSE `565.84`
  - validation branch accuracy `0.927`

Saved evaluation outputs:

- `experiment_runs/real_sim/long_sweep_20260311/rb_d512_w512_d6_long/eval_primary_interim`
- `experiment_runs/real_sim/long_sweep_20260311/rb_d512_w512_d6_long/eval_cross_interim`

Evaluation of the saved best checkpoint:

| Test split | Stress MAE | Stress RMSE | Max abs | Branch acc |
| --- | ---: | ---: | ---: | ---: |
| `d256` | `6.65` | `23.90` | `831.65` | `0.928` |
| `d512` | `6.52` | `23.13` | `1121.65` | `0.928` |

This is the new best local surrogate from the real-data studies.

## Comparison Against The Previous Best

Previous best checkpoint:

- `experiment_runs/real_sim/iter_20260311/raw_branch_w512_d6/best.pt`

New best checkpoint:

- `experiment_runs/real_sim/long_sweep_20260311/rb_d512_w512_d6_long/best.pt`

### On the broader `d512` test split

| Model | Stress MAE | Stress RMSE | Max abs | Branch acc | Top-decile MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous best | `9.27` | `25.31` | `1047.76` | `0.907` | `15.85` |
| new long-run best | `6.52` | `23.13` | `1121.65` | `0.928` | `8.98` |

### On the `d256` test split

| Model | Stress MAE | Stress RMSE | Max abs | Branch acc | Top-decile MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous best | `9.20` | `24.51` | `987.06` | `0.907` | `16.36` |
| new long-run best | `6.65` | `23.90` | `831.65` | `0.928` | `10.51` |

## Most Important Finding

The earlier criticism was correct:

- the previous real-data sweep was undertrained

Longer plateau schedules on the same `raw_branch` architecture produced the largest single improvement since moving to real-data training.

The broad-data long run improved:

- average stress error
- RMSE
- branch accuracy
- top-decile error
- almost every branch-wise MAE

The only notable metric that did not improve on the `d512` split was `max_abs`, which stayed large and even rose slightly relative to the previous best.

## What This Means

This new checkpoint is much more credible as a candidate constitutive surrogate than the earlier `~9.3` MAE model.

But I still would not call it replacement-grade yet for the exact limit-analysis use case.

Why:

- `stress_max_abs` is still around `1.12e3` on the broader split
- the model still predicts stress only, not a consistent tangent
- local fit still is not the same thing as preserving the global factor-of-safety / collapse solution

So the updated conclusion is:

- local fit is now substantially better than before
- the replacement argument is no longer “obviously too bad”
- but the real bar is still an in-solver comparison, not more local metrics alone

## Next Local Step If We Keep Optimizing

The next best local experiment is no longer “train longer.”

That part worked.

If we continue the local study, the most justified next step is:

1. resume the scripted larger-capacity runs only if we want to test whether the new long schedule has already saturated `512x6`
2. otherwise switch to hard-example mining or edge-focused weighting, since the remaining weakness is concentrated in the worst branch outliers

## Recommended Next Overall Step

Try the new checkpoint inside the actual constitutive call path:

- `experiment_runs/real_sim/long_sweep_20260311/rb_d512_w512_d6_long/best.pt`

That is the first checkpoint in this project that is strong enough to make that solver-level experiment worthwhile.
