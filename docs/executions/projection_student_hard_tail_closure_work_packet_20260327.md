# Projection-Student Hard-Tail Closure Work Packet

Execution closeout for the March 27, 2026 hard-tail closure packet.

Paired task memo: [`docs/tasks/projection_student_hard_tail_closure_work_packet_20260327.md`](/home/beremi/repos/mohr_coulomb_NN/docs/tasks/projection_student_hard_tail_closure_work_packet_20260327.md)

Repository: `/home/beremi/repos/mohr_coulomb_NN`
Output root: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327`

## Outcome

- overall best packet run: `combined_focus`
- family: same-capacity `trial_principal_geom_projected_student`, `256x4`, exact projection, warm-start from the March 27 control-zero checkpoint
- validation broad / hard / p95 / yield: `16.687582` / `18.913580` / `176.145065` / `2.310791e-08`
- test broad / hard / p95 / yield: `17.594461` / `19.734207` / `183.963257` / `2.290212e-08`
- explicit same-capacity gate `<= 18 / <= 20 / <= 180 / <= 1e-6`: `passed`
- compression reopen bar for this packet `<= 15 / <= 18 / <= 160 / <= 1e-6`: `not passed`
- `DS` reopen bar `<= 10 / <= 12 / <= 120 / <= 1e-6`: `not passed`

Key artifacts:

- control-zero phase summary: [`exp0_control_zero_slices/phase0_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp0_control_zero_slices/phase0_summary.json)
- same-capacity phase summary: [`exp1_tail_closure_controls/phase1_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp1_tail_closure_controls/phase1_summary.json)
- same-capacity sweep table: [`exp1_tail_closure_controls/same_capacity_summary.csv`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp1_tail_closure_controls/same_capacity_summary.csv)
- best same-capacity validation rowwise H5: [`exp1_tail_closure_controls/best_same_capacity_val_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp1_tail_closure_controls/best_same_capacity_val_rowwise.h5)
- best same-capacity test rowwise H5: [`exp1_tail_closure_controls/best_same_capacity_test_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp1_tail_closure_controls/best_same_capacity_test_rowwise.h5)
- near-same-capacity phase summary: [`exp2_near_same_capacity_followon/phase2_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp2_near_same_capacity_followon/phase2_summary.json)

## Phase A

Control-zero artifacts were frozen before any retraining:

- validation rowwise H5: [`exp0_control_zero_slices/control_zero_val_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp0_control_zero_slices/control_zero_val_rowwise.h5)
- test rowwise H5: [`exp0_control_zero_slices/control_zero_test_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp0_control_zero_slices/control_zero_test_rowwise.h5)
- validation slice table: [`exp0_control_zero_slices/control_zero_val_slice_summary.csv`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp0_control_zero_slices/control_zero_val_slice_summary.csv)
- validation call table: [`exp0_control_zero_slices/control_zero_val_call_concentration.csv`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp0_control_zero_slices/control_zero_val_call_concentration.csv)

Validation control-zero localization:

- hard-panel mean / p95 / p99 max principal abs error: `47.573296` / `196.215146` / `466.679443`
- `any_boundary_mask` contained `17011` hard rows and `536 / 1060 = 50.57%` of validation hard-top5 rows, so boundary focus was enabled by the packet rule
- high displacement stayed below the trigger even though it was real: teacher-projection displacement decile 10 held `377 / 1060 = 35.57%` of validation hard-top5 rows on only `11.03%` hard-base share and had `96.705052` mean max-abs error versus the `47.573296` hard-panel mean
- no dominant candidate or branch passed the stricter trigger:
  `right_edge` true branch contributed `430 / 1060 = 40.57%` of validation hard-top5 rows on `15.91%` hard-base share,
  `left_edge` contributed `39.15%` on `21.61%` base share,
  and teacher `smooth` candidate contributed `73.21%` on `60.01%` base share
- the sharpest boundary slices were apex-facing, not yield-facing:
  `near_left_apex_mask` contributed `31.89%` of validation hard-top5 with slice p95 `238.998`,
  `near_right_apex_mask` contributed `30.94%` with slice p95 `236.284`,
  while `near_yield_mask` contributed `0` validation hard-top5 rows
- call concentration was not tight enough to justify call-specific mechanics:
  the top five validation calls contributed `273 / 1060 = 25.75%` of validation hard-top5 rows

Interpretation:

- the remaining tail was real but only partially localized
- boundary-aware weighting was justified
- a dedicated high-displacement-only run was not justified
- candidate-specific or branch-specific focus was not justified by the packet trigger

## Phase B/C

The preservation dataset was rebuilt with boundary masks and `any_boundary_mask` under:
[`exp1_tail_closure_controls/base_preservation_dataset.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/exp1_tail_closure_controls/base_preservation_dataset.h5)

Same-capacity outcomes:

- `weighted_focus` failed decisively:
  validation `24.283125 / 26.305031 / 224.304886 / 1.965997e-08`,
  test `24.997667 / 27.010593 / 231.277725 / 1.991480e-08`
- `tail_loss_focus` materially helped:
  validation `16.417603 / 18.928406 / 186.317963 / 2.339881e-08`,
  test `17.286566 / 19.767761 / 193.175446 / 2.354289e-08`
- `combined_focus` was the packet winner:
  validation `16.687582 / 18.913580 / 176.145065 / 2.310791e-08`,
  test `17.594461 / 19.734207 / 183.963257 / 2.290212e-08`
- `dominant_slice_focus` was correctly skipped because Phase A never produced a dominant candidate / branch trigger

What helped:

- pure sampling reweighting did not help
- the route improved once the evidence-backed boundary weighting was paired with hard-row quantile tail loss and stronger teacher/projection-delta alignment on the focused rows
- relative to the frozen control-zero checkpoint, `combined_focus` improved validation `hard_p95_principal` by `20.070084` and test `hard_p95_principal` by `17.982254` while also improving validation broad and hard MAE

What did not disappear:

- the winning run still had strong edge-branch concentration
- on validation, `right_edge` still contributed `466 / 1060 = 43.96%` of hard-top5 rows and `left_edge` still contributed `36.60%`
- apex-adjacent boundary masks remained the sharpest boundary slices even after the win
- the packet closed the explicit gate, but it did not produce a diffuse or fully “solved” tail

## Phase D

Phase D was opened only because `combined_focus` cleared the explicit same-capacity gate. Both cold-start follow-ons regressed:

- `w288_d4`:
  validation `23.803234 / 23.876484 / 201.930908 / 2.347200e-08`,
  test `24.466400 / 24.781445 / 216.831635 / 2.355032e-08`
- `w256_d5`:
  validation `24.007879 / 23.714962 / 195.058914 / 2.332956e-08`,
  test `24.335867 / 24.154778 / 201.856415 / 2.327210e-08`

Decision for this phase:

- stop near-same-capacity escalation in this packet
- the same tail configuration did not survive a cold-start width/depth increase
- the overall packet anchor remains the same-capacity `combined_focus` run

## Decision

- where the control-zero hard tail was concentrated:
  moderately boundary-heavy by absolute share, strongly apex-adjacent inside the boundary family, strongly edge-branch-heavy, high-displacement-skewed but below the explicit high-displacement trigger, and not dominated by a tiny set of constitutive calls
- which changes helped:
  boundary-aware focus plus hard-row quantile tail loss and stronger teacher alignment
- which changes did not help:
  sampling-only weighting and cold-start near-same-capacity scaling
- whether the route cleared `hard_p95 <= 180`:
  `yes`, via `combined_focus` at validation `176.145065`
- whether compression remains closed:
  `yes`, because the packet’s reopen bar `<= 15 / <= 18 / <= 160 / <= 1e-6` was not reached
- whether `DS` is still blocked:
  `yes`
- continuation state:
  `continue_projection_student`, but only in the same-capacity projection-student preservation family
- stop rule reached for this sub-branch:
  do not continue cold-start near-same-capacity widening from this anchor

Bottom line:

This packet did what it was supposed to do. It localized the hard tail first, converted that localization into bounded in-family training mechanics, and showed that the same-capacity projected student can clear the explicit `hard_p95 <= 180` gate. The route survives, but compression stays closed, `DS` stays blocked, and the failed cold-start width/depth follow-ons mean the credible continuation state is the same-capacity `combined_focus` anchor rather than a broader model-size search.
