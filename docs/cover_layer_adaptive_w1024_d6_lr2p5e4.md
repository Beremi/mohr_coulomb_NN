# Cover Layer Adaptive 1024x6 Run

This run keeps the cover-layer-only exact-domain setup from the previous sweep,
but replaces the short fixed stage schedule with a longer adaptive cycle schedule.

## Configuration

- network: `cover_raw_branch_w1024_d6_adaptive` (`width=1024`, `depth=6`)
- cycles: `4`
- batch sizes: `[64, 128, 256, 512, 1024, 2048]`
- stage initial LR formula: `base_lr * cycle_lr_decay^(cycle-1) * sqrt(64 / batch_size)`
- base LR: `2.500e-04`
- cycle LR decay: `0.500`
- min LR: `1.000e-06`
- LR reduction rule: halve LR after `5` bad epochs on train loss
- stage advance rule: move to the next batch size after `20` bad epochs on train loss
- runtime: `interrupted at 00:25:07`
- train log: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/train.log`

Exact-domain dataset: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_layer_exact_domain.h5`

Training was interrupted after the later stages stopped improving the best saved checkpoint.

## Coverage Validation

The same cover-layer domain check still holds: the training inputs do cover the real cover-layer test inputs.
The main mismatch in the older synthetic set was target-tail inflation, not input-domain miss.

- exact-domain relabel vs stored real MAE: `1.426771e-04`
- exact-domain relabel vs stored real RMSE: `5.994247e-04`
- exact-domain relabel vs stored real max abs: `1.925659e-02`

| Dataset | Train N | Out-of-box sample rate vs real test | NN z mean | Stress q95 | Stress q995 |
|---|---:|---:|---:|---:|---:|
| hybrid_train | 206686 | 0.0000 | 0.1554 | 2107.4209 | 18705.4355 |
| exact_domain_train | 63343 | 0.0000 | 0.1273 | 306.5076 | 764.4935 |

## MEX Cross-Check

- samples: `200`
- C-vs-Python MAE: `7.542811e-14`
- C-vs-Python RMSE: `5.309594e-13`
- C-vs-Python max abs: `7.954526e-12`

## Results

| Split | Stress MAE | Stress RMSE | Stress Max Abs | Branch Acc |
|---|---:|---:|---:|---:|
| primary | 5.6846 | 21.0031 | 757.3899 | 0.9007 |
| cross | 5.3361 | 19.6284 | 591.1774 | 0.8985 |

- best epoch by real-val stress MSE: `282`
- best real-val stress MSE: `454.663177`

Primary-split comparison to the previous cover-layer sweep best-by-MAE (`w384 d6`):

- previous primary MAE / RMSE / max abs: `4.7124` / `13.1843` / `520.7037`
- current primary MAE / RMSE / max abs: `5.6846` / `21.0031` / `757.3899`

Cross-split comparison to the previous cover-layer sweep best-by-RMSE (`w512 d6`):

- previous cross MAE / RMSE / max abs: `4.5254` / `11.6177` / `291.2180`
- current cross MAE / RMSE / max abs: `5.3361` / `19.6284` / `591.1774`

## History

![adaptive training history](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/history_log.png)

![adaptive branch accuracy](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/branch_accuracy.png)

## Validation Plots

Primary real test:

![primary parity](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/eval_primary/parity_stress.png)

![primary error histogram](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/eval_primary/stress_error_hist.png)

![primary branch confusion](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/eval_primary/branch_confusion.png)

Cross real test:

![cross parity](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/eval_cross/parity_stress.png)

![cross error histogram](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/eval_cross/stress_error_hist.png)

![cross branch confusion](../experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/eval_cross/branch_confusion.png)

