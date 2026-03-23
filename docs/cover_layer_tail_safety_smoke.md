# Cover Layer Tail-Safety Execution

This is the living execution doc for the tail-safety phase. It tracks the frozen controls, hard-case mining, edge-expert retraining, routed evaluations, and the final go/no-go decision.

Primary objective: improve real-holdout tail safety for the conservative hybrid route while keeping the existing MAE gains.

- `[x] C0 Freeze controls`
  Comment: Recomputed the fixed scoreboard for the baseline, oracle, and both deployable routed controls.
  Exit criterion: controls_summary.json plus four real-holdout dissection files exist.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/controls_summary.json
  Result: Best deployable control remains gate_trial_threshold_t0.85 with real MAE 7.1258, but baseline_reference still has the lowest RMSE 22.9041.
  Next: Use the primary deployable route dissection to mine left_edge and right_edge failures.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/controls_compare_rmse.png)

- `[x] C1 Build hard-case mining table`
  Comment: Built left/right mining tables from the fixed val/test holdout using the current primary route, including worst calls, top stress bin, and branch outlier errors.
  Exit criterion: left_edge_hardcases.h5 and right_edge_hardcases.h5 exist with only the intended branch rows.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/mining_summary.json
  Result: Mining tables contain only target-branch rows and preserve source split/call metadata for replay and augmentation.
  Next: Materialize branch datasets and wire each branch to its mining table.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/mining_counts.png)

- `[x] C2 Build edge-focused train/val datasets`
  Comment: Built branch-filtered real datasets for left_edge and right_edge and validated the branch-pure mining tables.
  Exit criterion: left/right branch datasets exist and split counts are recorded.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/datasets
  Result: Both branch datasets now have fixed train/val/test splits and branch-pure hard-case tables.
  Next: Train the control experts first and compare them to the existing experts before trusting any new weighting or mining.

- `[ ] C3 Train control edge experts`
  Comment: Retrain control left/right experts and compare them to the published experts.
  Exit criterion: control experts reproduce the old branch validation MAE within 3% and the deployable real-holdout MAE within 5%.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/training_control_summary.json

- `[ ] C4 Train tail-weighted edge experts`
  Comment: Retrain left/right experts with tail-weighted loss only.
  Exit criterion: tail-weighted route improves at least one of edge MAE, top-bin sample MAE, or p99 relative error by the requested margin.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/training_tail_weighted_summary.json

- `[ ] C5 Train hard-mined edge experts`
  Comment: Retrain left/right experts with hard-mined replay plus local augmentation.
  Exit criterion: hard-mined primary route beats control and tail-weighted on RMSE and p99 without materially worse MAE.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/training_hard_mined_summary.json

- `[ ] C6 Re-evaluate deployable routing`
  Comment: Evaluate the conservative hybrid routes with the updated edge experts.
  Exit criterion: a routed expert ensemble beats baseline on RMSE, p99, and edge combined MAE while keeping MAE within +5%.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/routes_summary.json

- `[ ] C7 Decide go/no-go for solver-facing shadow test`
  Comment: Make a solver-shadow go/no-go decision from the final scoreboard.
  Exit criterion: decision text written with the winning route and the caveat list.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_tail_safety_smoke.md

## C0 Frozen Controls

| Mode | Real MAE | Real RMSE | Real p99 | Real Edge MAE | Synthetic MAE | Synthetic RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline_reference | 8.2331 | 22.9041 | 3.7585 | 11.9159 | 19.4116 | 156.1580 |
| oracle_reference | 7.3426 | 31.0570 | 5.6485 | 14.2041 | 14.5764 | 152.9653 |
| gate_trial_threshold_t0.85 | 7.1258 | 29.1589 | 4.7094 | 13.4129 | 14.9925 | 155.1699 |
| gate_raw_threshold_t0.65 | 7.3324 | 30.9220 | 5.2683 | 13.8961 | 14.4552 | 151.4229 |

![Frozen control MAE](../experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/controls_compare_mae.png)

![Frozen control RMSE](../experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/controls_compare_rmse.png)

![Frozen control p99](../experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/controls_compare_p99.png)

## C1 Mining Summary

| Branch | Rows | Val | Test | Rel Threshold | Worst-Call Count |
|---|---:|---:|---:|---:|---:|
| left_edge | 3618 | 1817 | 1801 | 0.8549 | 20 |
| right_edge | 2566 | 1276 | 1290 | 0.9996 | 20 |

![Mining counts](../experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/mining_counts.png)

## C2 Edge Datasets

| Branch | Train | Val | Test | Dataset | Mining Table |
|---|---:|---:|---:|---|---|
| left_edge | 24285 | 5398 | 4979 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/datasets/left_edge_dataset.h5` | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/mining/left_edge_hardcases.h5` |
| right_edge | 17293 | 3843 | 3595 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/datasets/right_edge_dataset.h5` | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_smoke_20260313/mining/right_edge_hardcases.h5` |

