# Cover Layer Cyclic Sweep

This study focuses only on the `cover_layer` material family.

Training setup:
- train on exact constitutive labels only, using the real cover-layer input domain with no augmentation,
- monitor synthetic train loss plus real validation/test losses,
- use repeated batch-size cycles `64 -> 128 -> 256 -> 512 -> 1024 -> 2048` and then repeat with lower learning rates,
- compare several `raw_branch` network sizes.

Exact-domain dataset: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_layer_exact_domain.h5`

## Coverage Validation

The key check here was whether the old synthetic sets were missing the real input domain. They were not.
The issue was different: the old augmented sets inflated the stress/target tails far beyond the real cover-layer distribution.

- exact-domain relabel vs stored real MAE: `1.426771e-04`
- exact-domain relabel vs stored real RMSE: `5.994247e-04`
- exact-domain relabel vs stored real max abs: `1.925659e-02`

| Dataset | Train N | Out-of-box sample rate vs real test | NN z mean | NN z p90 | Stress q95 | Stress q995 |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_train | 206686 | 0.0000 | 0.1554 | 0.4623 | 2107.4209 | 18705.4355 |
| exact_domain_train | 63343 | 0.0000 | 0.1273 | 0.3563 | 306.5076 | 764.4935 |

Real cover-layer test stress magnitude q95 / q995: `306.9618` / `749.3344`

Interpretation:
- `hybrid_train` covered the input box, but its stress tail was far too wide.
- `exact_domain_train` matches the real distribution by construction and does not miss any real test input state.

## MEX Cross-Check

The upstream C/MEX kernel from the Octave repo was compared numerically against the Python constitutive operator
on `200` cover-layer test states.

- samples: `200`
- C-vs-Python MAE: `7.542811e-14`
- C-vs-Python RMSE: `5.309594e-13`
- C-vs-Python max abs: `7.954526e-12`

This is effectively machine precision agreement, so the dataset conventions are consistent with the MEX kernel.

## Sweep Results

| Spec | Primary MAE | Primary RMSE | Primary Max Abs | Primary Branch Acc | Cross MAE | Cross RMSE |
|---|---:|---:|---:|---:|---:|---:|
| cover_raw_branch_w256_d5 | 4.9602 | 13.4734 | 432.1105 | 0.9209 | 4.8423 | 13.5662 |
| cover_raw_branch_w384_d6 | 4.7124 | 13.1843 | 520.7037 | 0.9195 | 4.3785 | 11.7293 |
| cover_raw_branch_w512_d6 | 4.8037 | 12.9621 | 497.6740 | 0.9195 | 4.5254 | 11.6177 |

Best spec by primary real-test MAE: `cover_raw_branch_w384_d6`

## cover_raw_branch_w256_d5

- history: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/history.csv`
- checkpoint: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/best.pt`
- best real-val stress MSE: `219.878693`
- primary test MAE / RMSE / max abs: `4.9602` / `13.4734` / `432.1105`
- cross test MAE / RMSE / max abs: `4.8423` / `13.5662` / `372.5598`

Training/test history:

![cover_raw_branch_w256_d5 history](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/history_log.png)

Primary real test parity and error plots:

![cover_raw_branch_w256_d5 primary parity](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/eval_primary/parity_stress.png)

![cover_raw_branch_w256_d5 primary error hist](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/eval_primary/stress_error_hist.png)

Cross real test parity and error plots:

![cover_raw_branch_w256_d5 cross parity](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/eval_cross/parity_stress.png)

![cover_raw_branch_w256_d5 cross error hist](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w256_d5/eval_cross/stress_error_hist.png)

## cover_raw_branch_w384_d6

- history: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/history.csv`
- checkpoint: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/best.pt`
- best real-val stress MSE: `200.624329`
- primary test MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`
- cross test MAE / RMSE / max abs: `4.3785` / `11.7293` / `308.3859`

Training/test history:

![cover_raw_branch_w384_d6 history](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/history_log.png)

Primary real test parity and error plots:

![cover_raw_branch_w384_d6 primary parity](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/eval_primary/parity_stress.png)

![cover_raw_branch_w384_d6 primary error hist](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/eval_primary/stress_error_hist.png)

Cross real test parity and error plots:

![cover_raw_branch_w384_d6 cross parity](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/eval_cross/parity_stress.png)

![cover_raw_branch_w384_d6 cross error hist](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w384_d6/eval_cross/stress_error_hist.png)

## cover_raw_branch_w512_d6

- history: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/history.csv`
- checkpoint: `experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/best.pt`
- best real-val stress MSE: `188.614380`
- primary test MAE / RMSE / max abs: `4.8037` / `12.9621` / `497.6740`
- cross test MAE / RMSE / max abs: `4.5254` / `11.6177` / `291.2180`

Training/test history:

![cover_raw_branch_w512_d6 history](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/history_log.png)

Primary real test parity and error plots:

![cover_raw_branch_w512_d6 primary parity](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/eval_primary/parity_stress.png)

![cover_raw_branch_w512_d6 primary error hist](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/eval_primary/stress_error_hist.png)

Cross real test parity and error plots:

![cover_raw_branch_w512_d6 cross parity](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/eval_cross/parity_stress.png)

![cover_raw_branch_w512_d6 cross error hist](../experiment_runs/real_sim/cover_layer_cyclic_20260312/cover_raw_branch_w512_d6/eval_cross/stress_error_hist.png)

