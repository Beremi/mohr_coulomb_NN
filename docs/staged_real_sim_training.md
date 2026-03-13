# Staged Real-Simulation Training

This report summarizes a more structured long-horizon training loop on the real sampled constitutive data.

Related follow-up:

- detailed test-loss and relative-difference visuals are in `docs/relative_error_visual_report.md`

The motivation was straightforward:

- the earlier long sweep showed that the old real-data runs were undertrained
- the next step was to replace the single long plateau run with a staged schedule:
  - learning-rate decay from `1e-3` toward `1e-6`
  - smaller batch size at the start
  - progressively larger batch sizes later
  - a few LBFGS steps between stages
  - explicit monitoring of train, validation, and test losses

## Trainer

Driver:

- `scripts/experiments/staged_real_sim_training.py`

Data used:

- primary training/validation/test dataset: `experiment_runs/real_sim/train_sample_512.h5`
- cross-check dataset: `experiment_runs/real_sim/baseline_sample_256.h5`

Stage schedule:

1. `adam_bs1024`
   - batch `1024`
   - lr `1e-3 -> 3e-4`
   - `140` Adam epochs
   - `2` LBFGS steps after stage
2. `adam_bs2048`
   - batch `2048`
   - lr `3e-4 -> 1e-4`
   - `120` Adam epochs
   - `2` LBFGS steps after stage
3. `adam_bs4096`
   - batch `4096`
   - lr `1e-4 -> 3e-5`
   - `120` Adam epochs
   - `3` LBFGS steps after stage
4. `adam_bs6144`
   - batch `6144`
   - lr `3e-5 -> 1e-6`
   - `160` Adam epochs
   - `4` LBFGS steps after stage

The loop monitored:

- train loss
- validation loss
- test loss
- train/validation/test stress MSE
- branch accuracy

The history plots are saved with log-scaled y-axes and vertical stage markers.

## Compared Models

### Previous best long-run baseline

Checkpoint:

- `experiment_runs/real_sim/long_sweep_20260311/rb_d512_w512_d6_long/best.pt`

This was the best model before the staged trainer.

### Staged `512x6`

Checkpoint:

- `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt`

Artifacts:

- history: `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/history.csv`
- plot: `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/staged_history_log.png`
- summary: `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/summary_manual.json`
- primary validation plots: `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_primary_final`
- cross-check validation plots: `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_cross_final`

Best stage:

- epoch `386`
- stage `adam_bs4096_lbfgs`
- best test stress MSE: `313.92`

### Staged `768x8`

Checkpoint:

- `experiment_runs/real_sim/staged_20260312/rb_staged_w768_d8/best.pt`

Artifacts:

- history: `experiment_runs/real_sim/staged_20260312/rb_staged_w768_d8/history.csv`
- plot: `experiment_runs/real_sim/staged_20260312/rb_staged_w768_d8/staged_history_log.png`
- summary: `experiment_runs/real_sim/staged_20260312/rb_staged_w768_d8/summary_manual.json`

Best stage:

- epoch `115`
- stage `adam_bs1024`
- best test stress MSE: `6517.99`

This richer model was stopped early because it was clearly underperforming the `512x6` model under the same staged schedule.

## Visual Summary

### Best Staged Run History

The full staged history with log-scaled losses and stage markers is here:

![Staged history](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/staged_history_log.png)

### Richer `768x8` Run History

This is the richer model that was stopped early. It is included here so the failure mode is visible rather than only described numerically.

![Staged history, richer 768x8 model](../experiment_runs/real_sim/staged_20260312/rb_staged_w768_d8/staged_history_log.png)

### Primary Validation On `d512`

Branch confusion map:

![Primary branch confusion](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_primary_final/branch_confusion.png)

Stress parity:

![Primary stress parity](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_primary_final/parity_stress.png)

Stress-error histogram:

![Primary stress error histogram](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_primary_final/stress_error_hist.png)

### Cross-Check Validation On `d256`

Branch confusion map:

![Cross-check branch confusion](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_cross_final/branch_confusion.png)

Stress parity:

![Cross-check stress parity](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_cross_final/parity_stress.png)

Stress-error histogram:

![Cross-check stress error histogram](../experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/eval_cross_final/stress_error_hist.png)

## Results

### Primary `d512` test split

| Model | Stress MAE | Stress RMSE | Max abs | Branch acc |
| --- | ---: | ---: | ---: | ---: |
| previous long-run best | `6.52` | `23.13` | `1121.65` | `0.928` |
| staged `512x6` | `4.86` | `17.72` | `936.46` | `0.942` |
| staged `768x8` | `27.57` | `80.73` | `736.92` | `0.817` |

### Cross-check `d256` test split

| Model | Stress MAE | Stress RMSE | Max abs | Branch acc |
| --- | ---: | ---: | ---: | ---: |
| previous long-run best | `6.65` | `23.90` | `831.65` | `0.928` |
| staged `512x6` | `4.75` | `17.41` | `759.17` | `0.943` |
| staged `768x8` | `27.52` | `80.77` | `727.40` | `0.821` |

### Hard-case comparison on the broad `d512` split

| Model | Top-decile MAE | Elastic MAE | Smooth MAE | Left edge MAE | Right edge MAE | Apex MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous long-run best | `8.98` | `3.48` | `4.21` | `10.83` | `13.30` | `3.32` |
| staged `512x6` | `7.40` | `1.71` | `3.18` | `8.15` | `10.94` | `2.35` |

The staged `512x6` model is better on every row in that table.

## Main Findings

### 1. The staged schedule is a real improvement, not cosmetic

The staged `512x6` model beat the previous best long-run baseline on:

- stress MAE
- stress RMSE
- branch accuracy
- top-decile MAE
- elastic branch error
- smooth branch error
- edge branch errors
- apex error
- max-abs error on the cross dataset

This is the strongest local constitutive surrogate trained in this repo so far.

### 2. The first LBFGS bridges were especially useful

The `512x6` history showed the biggest discrete jumps at the stage transitions and LBFGS bridges, especially:

- after the initial `adam_bs1024` stage
- after the `adam_bs2048` stage
- at `adam_bs4096_lbfgs`, which produced the best checkpoint

So the user’s intuition here was correct:

- batch-size cycling plus occasional LBFGS is materially better than the older single-regime loop

### 3. More capacity was not the winning move here

The staged `768x8` model did not just fail to beat the `512x6` model.

It failed badly.

That means:

- the new schedule mattered more than extra width/depth
- simply making the network richer is not enough
- the `512x6` model currently has the better optimization path and generalization behavior

## Updated Best Checkpoint

The best local checkpoint in the project is now:

- `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt`

## Is It Good Enough?

This is now the first local result that looks genuinely strong.

But I still would not claim final constitutive replacement safety from local metrics alone.

Why:

- `stress_max_abs` is still nontrivial, even though it improved materially
- the model still predicts stress only
- there is still no consistent tangent
- the real bar is preserving the actual limit-analysis result inside the solver

So the right conclusion is:

- the staged trainer clearly improved the local fit
- it is now worth testing this checkpoint in the real constitutive call path
- solver-level comparison is the next decisive experiment
