# Cover Layer Branch Predictor Margin Cycle

## Summary

- model family: `hierarchical trial_raw_material w1024 d6`
- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_100cycles_20260315/loop_17_accepted.pt`
- selector: `0.5 * broad_macro + 0.3 * hard_macro + 0.2 * smooth_hard`
- baseline selector score: `0.8819`
- final selector score: `0.8819`
- phase1 success gate passed: `False`

## Baseline

- current 4-call slice accuracy / macro: `0.8825` / `0.8787`
- large real-val accuracy / macro: `0.9045` / `0.9082`
- hard-panel accuracy / macro / smooth: `0.8763` / `0.8943` / `0.7976`
- real test accuracy / macro: `0.8981` / `0.8905`
- large real-val harmful fail rate: `0.0880`
- phase0 dispatch check max stress error: `0.0000e+00`

## Phase 1

- best selector checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_100cycles_20260315/loop_17_accepted.pt`
- best selector large real-val accuracy / macro: `0.9045` / `0.9082`
- best selector hard-panel accuracy / macro / smooth: `0.8763` / `0.8943` / `0.7976`
- best selector real test accuracy / macro: `0.8981` / `0.8905`
- best selector large real-val harmful fail rate: `0.0880`
- best selector large real-val harmful adjacent fail rate: `0.0812`

## Full Real-Val Finalists

- best full real-val checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_margin_cycle_smoke_20260316/checkpoints/phase1_best_real_val_full.pt`
- full real-val macro recall: `0.9049`
- full real-val harmful fail rate: `0.0932`

## Stress Usefulness

- broad real-val wrong-branch mean / p95 full-stress error: `0.7024` / `2.7755`
- broad real-val harmful-fail mean / p95 full-stress error: `0.7619` / `2.9171`
- hard-panel harmful-fail mean / p95 full-stress error: `0.7817` / `2.9652`
- real test overall mean / p95 full-stress error: `0.0848` / `0.2738`

## Final Decision

- branch-only line did not clear the planned success gate; next move should pivot

## Artifacts

- baseline harm-confusion table: `baseline_harm_confusions.json`
- final harm-confusion table: `final_harm_confusions.json`
- benchmark summary: `benchmark_summary.json`
- training history: `history.csv` and `training_history.png`
- final confusions: `final_confusions.png`
