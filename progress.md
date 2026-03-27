# Progress

This is the compact handoff state for future agents working in `/home/beremi/repos/mohr_coulomb_NN`.

Read this first.

## Canonical Report Set

- `docs/repository_general_report_20260326.md`
- `docs/repository_general_report_metric_dictionary_20260326.md`
- `docs/cover_layer_comprehensive_report_20260326.md`
- `docs/repository_general_report_appendix_20260326.md`
- `docs/repository_general_report_processed_manifest_20260326.json`

## Current Status

- Date: `2026-03-27`
- Canonical closeout for the absorbed historical report stack: `docs/repository_general_report_20260326.md`
- Canonical closeout for the local cover-layer prehistory: `docs/cover_layer_comprehensive_report_20260326.md`
- Last executed program: `projection-student preservation/compression packet`
- Current program state: `continue_projection_student`
- Reason: the exact projected teacher remains solver-credible on forward stress, the March 27 same-capacity preservation control materially beat the March 26 `medium_exact` student on validation, but the preserved student still missed the Tier 1 `hard_p95 <= 180` gate, so the route stays alive only in a narrowed preservation-first mode and compression remains closed
- `DS` status: still blocked; the projected-teacher probe was mixed and the preserved student remains well above the reopen bar

## Projection-Student Outcome

The March 26 strategic override in `tasks1.md` reopened the retired direct-replacement line only along the new `projection-student` route:

- accurate unconstrained stress predictor first
- exact admissibility second
- no packet5
- no new atlas / routing / exact-latent families

Implemented artifacts live under:

- `experiment_runs/real_sim/projection_student_20260326/`
- `docs/tasks/projection_student_work_packet_20260326.md`
- `docs/executions/projection_student_work_packet_20260326.md`

Projection operator delivered:

- exact Euclidean projection of ordered principal stress onto the convex admissible Mohr-Coulomb set
- five candidates only: `pass_through`, `smooth`, `left_edge`, `right_edge`, `apex`
- deterministic tie-breaking
- exact and `softmin` modes
- reusable NumPy and Torch APIs in `src/mc_surrogate/principal_projection.py`

Phase 0 outcome with the frozen March 24 raw teacher:

- teacher checkpoint: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt`
- validation raw teacher broad / hard / p95 / yield: `5.771003 / 6.949572 / 76.398941 / 1.001325e-01`
- validation projected teacher broad / hard / p95 / yield: `5.877489 / 7.062693 / 76.398941 / 2.199150e-08`
- test raw teacher broad / hard / p95 / yield: `5.981674 / 7.241915 / 79.455521 / 1.061293e-01`
- test projected teacher broad / hard / p95 / yield: `6.083902 / 7.344495 / 79.955505 / 2.252151e-08`
- Phase 0 decision: `pass`

Phase 2 bounded projected-student outcome:

- size bands executed exactly once each: `small 48x2`, `medium 80x2`, `large 96x2`
- preferred projection mode remained `exact`; no softmin fallback was needed
- winner: `medium_exact`, `29847` params
- validation winner broad / hard / p95 / yield: `27.308128 / 29.599710 / 250.701462 / 2.270879e-08`
- test winner broad / hard / p95 / yield: `28.099649 / 30.332142 / 259.966797 / 2.252200e-08`
- vs packet2 validation (`32.458569 / 37.824432 / 328.641602 / 6.173982e-08`): better on all three stress metrics while preserving numerical-zero yield
- vs packet3 oracle validation (`29.512514 / 32.682789 / 302.360168 / 5.414859e-08`): better on all three stress metrics while preserving numerical-zero yield
- stronger `DS` reopen bar `<= 8 / <= 10 / <= 100 / <= 1e-6`: `False`

Current decision after this reopen:

- continue the `projection-student` line
- keep `DS` blocked for now
- do not reopen packet5 / new routing / new atlas / exact-latent follow-ons from this state

## Projection-Student Preservation / Compression Outcome

The March 27 strategic brief tightened the live direct-replacement route from “project a student” to “preserve the projected teacher first, compress second”.

Implemented artifacts live under:

- `experiment_runs/real_sim/projection_student_preservation_compression_20260327/`
- `docs/tasks/projection_student_preservation_compression_work_packet_20260327.md`
- `docs/executions/projection_student_preservation_compression_work_packet_20260327.md`

Phase A0 candidate-zero freeze:

- full-row teacher projection cache built at `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp0_teacher_cache/teacher_projection_preservation_cache.h5`
- validation projected teacher broad / hard / p95 / yield: `5.877489 / 7.062693 / 76.398941 / 2.199150e-08`
- test projected teacher broad / hard / p95 / yield: `6.083902 / 7.344495 / 79.955505 / 2.252151e-08`
- cache includes exact stress, projected / dispatched teacher stress, projection displacement, selected candidate, split ids, hard masks, `ds_valid` masks, branch ids, and boundary-nearness masks for all `283648` rows

Phase A1 projected-teacher `DS` probe:

- directional finite-difference probe executed on canonical `ds_valid_val` / `ds_valid_test` using `3` random unit directions per row and both `exact` and `softmin(tau=0.05)` projection modes
- overall packet probe status: `mixed`
- validation / test probe status: `blocked` / `mixed`
- validation softmin forward stress-component MAE vs exact projection: `1.012578`
- test softmin forward stress-component MAE vs exact projection: `0.969067`
- candidate-switch rates remained very small, but directional tangent relative-error tails stayed too large for a `DS` reopen

Phase B preservation path:

- preservation dataset built at `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp2_preservation_dataset/projection_student_preservation_plastic.h5`
- preservation dataset rows: `236700`
- split counts: `166254 / 34343 / 36103`
- dataset now includes projected-teacher provisional / projected / delta principal targets, candidate ids, displacement norms, `ds_valid_mask`, and `sampling_weight`

Phase B1 same-capacity preservation control:

- model: `trial_principal_geom_projected_student`, `256x4`, `540935` params, exact projection, warm-started from `candidate_b`
- validation broad / hard / p95 / yield: `16.891312 / 19.413279 / 196.215149 / 2.329866e-08`
- test broad / hard / p95 / yield: `17.608580 / 20.012241 / 201.945511 / 2.342785e-08`
- vs March 26 `medium_exact` validation (`27.308128 / 29.599710 / 250.701462 / 2.270879e-08`): materially better on all three stress metrics while preserving numerical-zero yield
- Tier 1 opening gate `<= 18 / <= 20 / <= 180 / <= 1e-6`: `False` because `hard_p95_principal` remained at `196.215149`

Phase C compression decision:

- Tier 1 compression: `skipped`
- Tier 2 compression: `skipped`
- reason: the same-capacity preservation control materially improved the line but did not clear the explicit Tier 1 opening gate, so bounded compression is not yet justified

Current decision after the March 27 preservation/compression packet:

- continue the `projection-student` route only in narrowed preservation-first mode
- keep `DS` blocked
- do not open Tier 1 / Tier 2 compression yet
- focus any immediate follow-on on closing the remaining hard-tail (`p95`) gap to the projected teacher before reopening compression or broader tangent work

## Final Program X Outcome

Program X executed the historical `report11.md` bounded study on the frozen March 24 validation-first split:

- `X0` baseline freeze: completed
- `X1` exact latent extraction: defined cleanly without breaking the default constitutive API
- `X2` exact-latent dataset: built under `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/`
- `X3` Stage 0 exactness: passed after preserving exact branch latents / trial principal values at full precision in the derived dataset and auditing against a fresh exact recomputation
- `X4-X5` teacher-forced branchwise oracle: executed once across the bounded branch architecture list
- `X6` final decision: `stop_direct_replacement`

Branchwise latent schema:

- `smooth`: one scalar `plastic_multiplier`
- `left_edge`: one scalar `plastic_multiplier`
- `right_edge`: one scalar `plastic_multiplier`
- `apex`: analytic-only, zero learned latent dimension

Measured Stage 0 exactness:

- round-trip mean abs: `3.961551e-13`
- round-trip max abs: `2.328306e-10`
- yield violation p95: `4.662968e-14`
- branch agreement: `1.0`

Measured validation oracle:

- broad plastic MAE: `250.392441`
- hard plastic MAE: `328.857300`
- hard p95 principal: `2331.702637`
- hard relative p95 principal: `24.178808`
- yield violation p95: `1.258780e+00`

Measured held-out test oracle:

- broad plastic MAE: `285.859314`
- hard plastic MAE: `378.356659`
- hard p95 principal: `2632.823242`
- hard relative p95 principal: `27.600845`
- yield violation p95: `1.260843e+00`

Program X comparison on validation:

- vs packet2 deployed winner (`32.458569 / 37.824432 / 328.641602`): much worse
- vs packet3 oracle winner (`29.512514 / 32.682789 / 302.360168`): much worse
- vs packet4 deployed winner (`41.599422 / 46.363712 / 374.894531`): much worse
- minimum oracle bar `<= 25 / <= 28 / <= 250 / <= 1e-6`: `False`
- stronger bar for any future `DS` reopening `<= 20 / <= 24 / <= 200 / <= 1e-6`: `False`

Branchwise validation winners:

- `smooth`: `smooth_w96_d3`, `20737` params, val stress MAE `275.604462`, val principal p95 `956.649841`, val yield p95 `1.278197`
- `left_edge`: `left_edge_w64_d3`, `9729` params, val stress MAE `274.984619`, val principal p95 `1876.733887`, val yield p95 `1.056348`
- `right_edge`: `right_edge_w64_d3`, `9729` params, val stress MAE `337.012878`, val principal p95 `3318.269531`, val yield p95 `1.320152`

Program X decision:

- Do not open a routing study.
- Do not reopen the packet2 / packet3 / packet4 admissible-surface family.
- Do not open an implicit-layer fallback by default from this brief.
- Treat the direct local NN replacement line as stopped unless a new explicit strategic brief says otherwise.

## Final Packet4 Outcome

Packet4 was the final bounded direct surface-family packet:

- exact elastic dispatch
- plastic-only learned rows
- smooth / left / right / apex soft admissible atlas
- post-hoc validation temperature sweep over `T in {0.60, 0.80, 1.00, 1.25, 1.50, 2.00}`
- yield treated as a stability constraint, not a leaderboard metric

Measured validation winner:

- architecture: `film_w128_d3`
- checkpoint: `best.pt`
- selected temperature: `1.25`
- broad plastic MAE: `41.599422`
- hard plastic MAE: `46.363712`
- hard p95 principal: `374.894531`
- yield violation p95: `6.008882e-08`

Measured one-shot held-out test:

- broad plastic MAE: `43.238693`
- hard plastic MAE: `48.038338`
- hard p95 principal: `388.482666`
- yield violation p95: `5.825863e-08`

Decision:

- packet4 beat packet2 on deployed validation core stress metrics: `False`
- packet4 reached the `<= 25 / <= 28 / <= 250 / <= 1e-6` continue bar: `False`
- packet4 reached the `DS` bar `<= 15 / <= 18 / <= 150 / <= 1e-6`: `False`
- final decision for the direct line: `stop`

## Important Packet4 Notes

- The optional edge-anchor micro-audit was checked before the sweep and did not justify changing the anchors; keep the current `b_tr` / `a_tr` anchors.
- Stage 0 passed on the frozen benchmark after fixing the packet4 route-target adjacency bug.
- Packet4 required two implementation fixes during execution:
  - route targets for edge rows were corrected to avoid forbidden direct smooth↔apex mixing
  - the soft-atlas decode had a `lambda_tr` broadcasting bug; fixing it required rerunning packet4 from scratch
- The packet4 oracle diagnostic ended up worse than deployed routing on all three stress metrics, so it does **not** count as routing-gap success.

## Recommended Read Order

1. `progress.md`
2. `docs/repository_general_report_20260326.md`
3. `docs/repository_general_report_metric_dictionary_20260326.md`
4. `docs/cover_layer_comprehensive_report_20260326.md`
5. `docs/executions/projection_student_work_packet_20260326.md`
6. `docs/tasks/projection_student_work_packet_20260326.md`
7. `docs/repository_general_report_processed_manifest_20260326.json`
8. `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/decision_logic_summary.json`
9. `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/decision_logic_summary.json`

## Frozen Benchmark

The canonical benchmark remains the March 24 validation-first split:

- source H5: `constitutive_problem_3D_full.h5`
- grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`
- split seed: `20260324`
- split unit: constitutive call
- split fractions: `70 / 15 / 15`
- samples per call: `512`

Canonical sizes:

- `train`: `198656`
- `val`: `42496`
- `test`: `42496`
- `broad_val`: `42496`
- `broad_test`: `42496`
- `hard_val`: `21198`
- `hard_test`: `22705`
- `ds_valid`: `73261`
- `ds_valid_val`: `11224`
- `ds_valid_test`: `9839`

## Reference Surface Baselines

Packet2 remains the best deployed admissible-surface baseline:

- validation broad plastic MAE: `32.458569`
- validation hard plastic MAE: `37.824432`
- validation hard p95 principal: `328.641602`
- validation yield violation p95: `6.173982e-08`

Packet3 established that branch-local signal exists but hard routing fails in deployment:

- predicted validation broad / hard plastic MAE: `38.164696 / 43.860016`
- predicted validation hard p95 principal: `399.468353`
- oracle validation broad / hard plastic MAE: `29.512514 / 32.682789`
- oracle validation hard p95 principal: `302.360168`

Packet4 final validation winner:

- broad / hard plastic MAE: `41.599422 / 46.363712`
- hard p95 principal: `374.894531`
- yield violation p95: `6.008882e-08`

## Current Artifacts

Projection-student root:

- `experiment_runs/real_sim/projection_student_20260326/`
- `experiment_runs/real_sim/projection_student_preservation_compression_20260327/`

Key projection-student outputs:

- Phase 0 summary: `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_summary.json`
- Phase 0 comparison tables: `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_comparison_val.csv`, `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_comparison_test.csv`
- Phase 1 dataset summary: `experiment_runs/real_sim/projection_student_20260326/exp1_student_dataset/dataset_summary.json`
- Phase 1 dataset: `experiment_runs/real_sim/projection_student_20260326/exp1_student_dataset/projection_student_plastic_only.h5`
- Phase 2 architecture summary: `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/projected_student_architecture_summary.csv`
- Phase 2 summary: `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/phase2_summary.json`
- Winner checkpoint: `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/best.pt`
- Winner validation/test metrics: `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/projected_student_eval_val.json`, `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/projected_student_eval_test.json`
- Phase 3 status: `experiment_runs/real_sim/projection_student_20260326/exp3_ds_probe/ds_probe_summary.json`
- March 27 teacher cache summary: `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp0_teacher_cache/teacher_projection_preservation_summary.json`
- March 27 teacher `DS` probe summary: `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp1_teacher_ds_probe/teacher_ds_probe_summary.json`
- March 27 preservation dataset summary: `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp2_preservation_dataset/preservation_dataset_summary.json`
- March 27 preservation control summary: `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/phase3_summary.json`
- March 27 compression decisions: `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp4_tier1_compression/phase4_summary.json`, `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp5_tier2_compression/phase5_summary.json`

Program X root:

- `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/`

Key Program X outputs:

- Baseline freeze manifest: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp0_baseline_freeze/baseline_manifest.json`
- Latent dataset summary: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp1_latent_dataset/stage0_dataset_summary.json`
- Latent schema: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp1_latent_dataset/latent_schema_summary.json`
- Stage 0 exactness summary: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp2_stage0_exactness/stage0_exactness_summary.json`
- Branch winner summary: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp3_branchwise_oracle/winner_selection.json`
- Oracle validation summary: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/oracle_val_summary.json`
- Oracle test summary: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/oracle_test_summary.json`
- Decision logic: `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/decision_logic_summary.json`

Packet4 root:

- `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp0_soft_atlas_dataset/`
- `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/`

Key packet4 outputs:

- Stage 0 summary: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp0_soft_atlas_dataset/stage0_summary.json`
- Stage 0 route audit: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp0_soft_atlas_dataset/stage0_route_target_summary.json`
- Stage 0 feature audit: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp0_soft_atlas_dataset/stage0_feature_stats.json`
- Validation temperature sweep: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/temperature_sweep_val.csv`
- Validation temperature sweep JSON: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/temperature_sweep_val.json`
- Architecture summary: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/architecture_summary_val.csv`
- Checkpoint selection: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/checkpoint_selection.json`
- Decision logic: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/decision_logic_summary.json`
- Winner validation summary: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_val_predicted_summary.json`
- Winner test summary: `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_test_summary.json`

## What Not To Do Next

- Do not reopen the direct replacement line by default after Program X.
- Do not open a routing follow-on from Program X; the true-branch oracle result does not justify it.
- Do not open a packet5 redesign by default.
- Do not widen the packet4 architecture family.
- Do not start `DS` work for this direct line.
- Do not reinterpret the packet4 oracle underperformance as a rescue signal.

If the user wants to continue this topic anyway, treat it as a new explicit decision to reopen a stopped line, not as the default next packet.
