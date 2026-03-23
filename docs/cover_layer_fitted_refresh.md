# Cover Layer Fitted-Distribution Refresh

This report documents the second pass on regenerated synthetic cover-layer training after the earlier epoch-refresh run showed that the old branch-targeted sampler was badly off-domain.

## Outcome

- winning sampler: `local_noise`
- winning checkpoint: `experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/best.pt`
- model: `raw_branch`, `width=384`, `depth=6`
- best checkpoint came from `cycle1_bs64`, at epoch `112`

Final evaluation of that checkpoint:

| Test set | Stress MAE | Stress RMSE | Stress Max Abs | Branch Acc |
|---|---:|---:|---:|---:|
| real cover-layer test | 4.0053 | 14.5005 | 788.4071 | 0.8816 |
| independent synthetic holdout | 2.6013 | 5.9819 | 292.4198 | 0.9767 |

Reference points:

- previous wrong-domain epoch-refresh run: real MAE `23.1082`
- previous fixed exact-domain cover-layer baseline: real MAE `4.7124`, RMSE `13.1843`, max abs `520.7037`

So this corrected fitted-distribution training clearly fixed the failure mode. It improved real-test MAE over the fixed exact-domain baseline, but RMSE and worst-case error are still worse, so it is not a clean universal win yet.

## What Changed

The first fitted sampler family was rejected. It matched principal-strain quantiles, but after exact relabeling it produced absurd stress magnitudes, which meant the fit was wrong in operator space, not just strain space.

The replacement sampler family is engineering-strain anchored:

- `local_noise`: sample a real train state, add small branch-aware local noise in engineering-strain space
- `local_interp`: interpolate between two real states from the same branch, then add small local noise
- `tail_local`: same idea as `local_noise`, but tail-biased toward high-stress states

All three use exact constitutive relabeling plus acceptance checks:

- realized branch must match the source branch
- realized stress magnitude must stay inside fitted real-data caps
- realized principal-strain magnitude must stay inside fitted real-data caps

That is what finally made “new synthetic data every epoch” usable instead of destructive.

## Distribution Screening

The useful screening run is in [cover_layer_fitted_refresh_smoke3.md](../docs/cover_layer_fitted_refresh_smoke3.md).

Small-model screening on the corrected sampler family:

| Sampler | Fit Score | Real MAE | Real RMSE | Synthetic MAE |
|---|---:|---:|---:|---:|
| `local_noise` | 1.8065 | 41.0915 | 67.9178 | 53.2411 |
| `local_interp` | 1.8547 | 36.7757 | 62.0169 | 47.2008 |
| `tail_local` | 1.8132 | 39.3557 | 65.0558 | 54.8974 |

`local_interp` looked slightly better on the tiny screen, but the longer full-capacity runs showed `local_noise` trained much more cleanly. Once the `w384 d6` run started, `local_noise` was the only variant that consistently drove real validation loss into the right range, so it was promoted and the others were not pushed through the full long run.

The final `local_noise` synthetic holdout distribution is still not perfect:

- real test `max |eps_principal| q95 / q995`: `14.55 / 38.26`
- synthetic holdout `max |eps_principal| q95 / q995`: `9.30 / 25.87`
- real test `|stress| q95 / q995`: `306.96 / 749.33`
- synthetic holdout `|stress| q95 / q995`: `671.90 / 788.97`

So the generator is still too concentrated in stress magnitude even though the training outcome is now strong.

## Final Validation

Per-branch real-test stress MAE for the winning checkpoint:

- `elastic`: `1.6897`
- `smooth`: `3.2262`
- `left_edge`: `5.6940`
- `right_edge`: `6.6887`
- `apex`: `2.1219`

The worst remaining local weakness is still the right/left edge region, not the elastic or apex behavior.

![local_noise screening distribution fit](../experiment_runs/real_sim/cover_layer_fitted_refresh_smoke3/distribution_fit_local_noise.png)

![local_noise training history](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/history_log_manual.png)

![local_noise branch accuracy](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/branch_accuracy_manual.png)

![real parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/manual_eval/real/parity_stress.png)

![real error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/manual_eval/real/stress_error_hist.png)

![real branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/manual_eval/real/branch_confusion.png)

![synthetic parity](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/manual_eval/synth/parity_stress.png)

![synthetic error histogram](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/manual_eval/synth/stress_error_hist.png)

![synthetic branch confusion](../experiment_runs/real_sim/cover_layer_fitted_refresh_focused_20260312/cover_raw_branch_w384_d6_local_noise/manual_eval/synth/branch_confusion.png)

## Assessment

This is the first regenerated-synthetic cover-layer result that I would call genuinely good.

- It fixed the catastrophic off-domain failure of the old epoch-refresh sampler.
- It beat the previous fixed exact-domain baseline on real-test MAE.
- It still has a heavier tail than the fixed exact-domain baseline.

My read is:

- for cover layer alone, this fitted local-noise sampler is now a serious option
- for solver replacement, I still would not claim “same result guaranteed” because the RMSE and max-abs tail are not yet as good as the fixed exact-domain baseline

The most justified next step is no longer “invent a new generic sampler.” It is one of these:

- use this exact `local_noise` recipe on the other material families
- narrow the `local_noise` stress distribution so synthetic q95 is closer to the real q95
- test this checkpoint in the actual constitutive call path and compare the limit-analysis result directly
