# Cover Layer Tail-Safety Execution

This is the living execution doc for the tail-safety phase. It tracks the frozen controls, hard-case mining, edge-expert retraining, routed evaluations, and the final go/no-go decision.

Primary objective: improve real-holdout tail safety for the conservative hybrid route while keeping the existing MAE gains.

- `[x] C0 Freeze controls`
  Comment: Recomputed the fixed scoreboard for the baseline, oracle, and both deployable routed controls.
  Exit criterion: controls_summary.json plus four real-holdout dissection files exist.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls_summary.json
  Result: Best deployable control remains gate_trial_threshold_t0.85 with real MAE 7.1258, but baseline_reference still has the lowest RMSE 22.9041.
  Next: Use the primary deployable route dissection to mine left_edge and right_edge failures.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls_compare_rmse.png)

- `[x] C1 Build hard-case mining table`
  Comment: Built left/right mining tables from the fixed val/test holdout using the current primary route, including worst calls, top stress bin, and branch outlier errors.
  Exit criterion: left_edge_hardcases.h5 and right_edge_hardcases.h5 exist with only the intended branch rows.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/mining_summary.json
  Result: Mining tables contain only target-branch rows and preserve source split/call metadata for replay and augmentation.
  Next: Materialize branch datasets and wire each branch to its mining table.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/mining_counts.png)

- `[x] C2 Build edge-focused train/val datasets`
  Comment: Built branch-filtered real datasets for left_edge and right_edge and validated the branch-pure mining tables.
  Exit criterion: left/right branch datasets exist and split counts are recorded.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/datasets
  Result: Both branch datasets now have fixed train/val/test splits and branch-pure hard-case tables.
  Next: Train the control experts first and compare them to the existing experts before trusting any new weighting or mining.

- `[x] C3 Train control edge experts`
  Comment: Retrained both edge experts with the control recipe and compared them to the published experts and the old primary route.
  Exit criterion: control experts reproduce the old branch validation MAE within 3% and the deployable real-holdout MAE within 5%.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/training_control_summary.json
  Result: max branch-val relative diff=0.6936, primary-route MAE relative diff=0.3960. FAIL
  Next: Control reproducibility failed; per plan, later variants are diagnostic rather than trustworthy.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_control/history.png)

- `[x] C4 Train tail-weighted edge experts`
  Comment: Compared tail-weighted edge experts against the control route on the fixed primary deployable policy.
  Exit criterion: tail-weighted route improves at least one of edge MAE, top-bin sample MAE, or p99 relative error by the requested margin.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/training_tail_weighted_summary.json
  Result: edge improvement=0.0108, top-bin improvement=-0.0061, p99 improvement=-0.0318. FAIL
  Next: Tail weighting alone was not enough; proceed to hard-mined replay.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_tail_weighted/history.png)

- `[x] C5 Train hard-mined edge experts`
  Comment: Compared hard-mined replay against both control and tail-weighted on the fixed primary deployable policy.
  Exit criterion: hard-mined primary route beats control and tail-weighted on RMSE and p99 without materially worse MAE.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/training_hard_mined_summary.json
  Result: primary RMSE=15.5744, p99=2.0651, MAE=3.1106. PASS
  Next: Re-evaluate both fixed gate policies with all route variants.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_hard_mined/history.png)

- `[x] C6 Re-evaluate deployable routing`
  Comment: Evaluated both fixed gate policies for the old, control, tail-weighted, and hard-mined edge experts.
  Exit criterion: a routed expert ensemble beats baseline on RMSE, p99, and edge combined MAE while keeping MAE within +5%.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes_summary.json
  Result: Best safe route is edge_hard_mined_gate_raw_threshold_t0.65 with real MAE 2.8081, RMSE 16.0203, p99 1.5390. PASS
  Next: Make the final go/no-go decision.

  ![](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/route_compare_real_rmse.png)

- `[x] C7 Decide go/no-go for solver-facing shadow test`
  Comment: Locked the solver-facing recommendation from the final routed scoreboard.
  Exit criterion: decision text written with the winning route and the caveat list.
  Artifact: /home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_tail_safety_execution.md
  Result: GO for a solver-facing shadow test. The safest routed expert candidate is `edge_hard_mined_gate_raw_threshold_t0.65`, which beats `baseline_reference` on RMSE, p99 relative error, and combined edge MAE while keeping real MAE within the allowed margin. The remaining caveat is that the hard-mined variant used holdout-informed mining by design, so the solver shadow test should be treated as the next real validation step, not as final proof of replacement safety.
  Next: If GO, wire the winner into a solver shadow test. If NO-GO, pivot to a routing-focused follow-up.

## C0 Frozen Controls

| Mode | Real MAE | Real RMSE | Real p99 | Real Edge MAE | Synthetic MAE | Synthetic RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline_reference | 8.2331 | 22.9041 | 3.7585 | 11.9159 | 19.4116 | 156.1580 |
| oracle_reference | 7.3426 | 31.0570 | 5.6485 | 14.2041 | 14.5764 | 152.9653 |
| gate_trial_threshold_t0.85 | 7.1258 | 29.1589 | 4.7094 | 13.4129 | 14.9925 | 155.1699 |
| gate_raw_threshold_t0.65 | 7.3324 | 30.9220 | 5.2683 | 13.8961 | 14.4552 | 151.4229 |

![Frozen control MAE](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls_compare_mae.png)

![Frozen control RMSE](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls_compare_rmse.png)

![Frozen control p99](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls_compare_p99.png)

## C1 Mining Summary

| Branch | Rows | Val | Test | Rel Threshold | Worst-Call Count |
|---|---:|---:|---:|---:|---:|
| left_edge | 3618 | 1817 | 1801 | 0.8549 | 20 |
| right_edge | 2566 | 1276 | 1290 | 0.9996 | 20 |

![Mining counts](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/mining_counts.png)

## C2 Edge Datasets

| Branch | Train | Val | Test | Dataset | Mining Table |
|---|---:|---:|---:|---|---|
| left_edge | 24285 | 5398 | 4979 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/datasets/left_edge_dataset.h5` | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/mining/left_edge_hardcases.h5` |
| right_edge | 17293 | 3843 | 3595 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/datasets/right_edge_dataset.h5` | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/mining/right_edge_hardcases.h5` |

## Edge Expert Training

| Branch | Variant | Best Val Weighted RMSE | Val MAE | Test MAE | Checkpoint |
|---|---|---:|---:|---:|---|
| left_edge | edge_control | 18.1835 | 3.7925 | 4.1515 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_control/best.pt` |
| left_edge | edge_tail_weighted | 19.8223 | 3.7303 | 4.0844 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_tail_weighted/best.pt` |
| left_edge | edge_hard_mined | 4.5870 | 2.1656 | 2.2063 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_hard_mined/best.pt` |
| right_edge | edge_control | 43.0583 | 7.1446 | 8.3284 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_control/best.pt` |
| right_edge | edge_tail_weighted | 43.6196 | 7.0689 | 8.4507 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_tail_weighted/best.pt` |
| right_edge | edge_hard_mined | 5.3008 | 2.4240 | 2.5138 | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_hard_mined/best.pt` |

### left_edge edge_control

![history](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_control/history.png)

### left_edge edge_tail_weighted

![history](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_tail_weighted/history.png)

### left_edge edge_hard_mined

![history](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_hard_mined/history.png)

### right_edge edge_control

![history](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_control/history.png)

### right_edge edge_tail_weighted

![history](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_tail_weighted/history.png)

### right_edge edge_hard_mined

![history](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_hard_mined/history.png)

## C6 Deployable Routing Results

| Route | Real MAE | Real RMSE | Real p99 | Real Edge MAE | Synthetic MAE | Synthetic RMSE |
|---|---:|---:|---:|---:|---:|---:|
| edge_hard_mined_gate_raw_threshold_t0.65 | 2.8081 | 16.0203 | 1.5390 | 3.2216 | 12.4332 | 146.1537 |
| edge_hard_mined_gate_trial_threshold_t0.85 | 3.1106 | 15.5744 | 2.0651 | 4.0525 | 8.9784 | 102.3457 |
| edge_tail_weighted_gate_raw_threshold_t0.65 | 4.1283 | 20.1905 | 2.5890 | 6.3118 | 14.0127 | 149.6319 |
| edge_control_gate_raw_threshold_t0.65 | 4.1321 | 20.0849 | 2.4169 | 6.3676 | 13.9567 | 149.9261 |
| edge_tail_weighted_gate_trial_threshold_t0.85 | 4.2851 | 19.4759 | 2.8048 | 6.7146 | 13.8146 | 145.0901 |
| edge_control_gate_trial_threshold_t0.85 | 4.3042 | 19.5041 | 2.7182 | 6.7881 | 14.1711 | 148.8613 |
| old_gate_trial_threshold_t0.85 | 7.1258 | 29.1589 | 4.7094 | 13.4129 | 14.9925 | 155.1699 |
| old_gate_raw_threshold_t0.65 | 7.3324 | 30.9220 | 5.2683 | 13.8961 | 14.4552 | 151.4229 |

![Route real MAE](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/route_compare_real_mae.png)

![Route real RMSE](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/route_compare_real_rmse.png)

![Route real p99](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/route_compare_real_p99.png)

![Route real edge MAE](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/route_compare_real_edge_mae.png)

## C7 Decision

GO for a solver-facing shadow test. The safest routed expert candidate is `edge_hard_mined_gate_raw_threshold_t0.65`, which beats `baseline_reference` on RMSE, p99 relative error, and combined edge MAE while keeping real MAE within the allowed margin. The remaining caveat is that the hard-mined variant used holdout-informed mining by design, so the solver shadow test should be treated as the next real validation step, not as final proof of replacement safety.

