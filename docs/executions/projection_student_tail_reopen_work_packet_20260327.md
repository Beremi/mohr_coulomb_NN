# Projection-Student Tail Reopen Work Packet

Execution report for the March 27, 2026 projection-student tail-reopen packet.

Paired task memo:
[`docs/tasks/projection_student_tail_reopen_work_packet_20260327.md`](/home/beremi/repos/mohr_coulomb_NN/docs/tasks/projection_student_tail_reopen_work_packet_20260327.md)

Repository:
`/home/beremi/repos/mohr_coulomb_NN`

Output root:
`/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327`

## Outcome

- reopen bar `<= 15 / <= 18 / <= 160 / <= 1e-6`: `False`
- best same-capacity packet run: `anchor_ema_restart`
- best validation broad / hard / p95 / yield: `15.899072 / 18.059217 / 170.881073 / 2.263277e-08`
- best test broad / hard / p95 / yield: `16.647007 / 18.692650 / 176.318420 / 2.301265e-08`
- improvement over the frozen `combined_focus` anchor on validation: about `0.789 / 0.854 / 5.264`
- Phase 4 full-real follow-on: `not opened`
- compression remains closed: `True`
- student `DS` remains blocked: `True`
- route decision: `continue narrowly`, not `stop`

The packet materially improved the same-capacity anchor but still missed the reopen target, especially on `hard_p95_principal`.

## Phase 0

- frozen anchor checkpoint:
  [`exp0_anchor_freeze/anchor_checkpoint/best.pt`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp0_anchor_freeze/anchor_checkpoint/best.pt)
- anchor freeze summary:
  [`exp0_anchor_freeze/phase0_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp0_anchor_freeze/phase0_summary.json)
- train rowwise:
  [`exp0_anchor_freeze/anchor_rowwise_train.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp0_anchor_freeze/anchor_rowwise_train.h5)
- val rowwise:
  [`exp0_anchor_freeze/anchor_rowwise_val.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp0_anchor_freeze/anchor_rowwise_val.h5)
- test rowwise:
  [`exp0_anchor_freeze/anchor_rowwise_test.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp0_anchor_freeze/anchor_rowwise_test.h5)

Reproducibility passed exactly enough to continue:

- validation `16.687582 / 18.913580 / 176.145065 / 2.310791e-08`
- test `17.594461 / 19.734207 / 183.963257 / 2.290212e-08`

## Phase 1

- teacher-gap audit summary:
  [`exp1_teacher_gap_audit/phase1_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp1_teacher_gap_audit/phase1_summary.json)
- train rowwise audit:
  [`exp1_teacher_gap_audit/teacher_gap_train_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp1_teacher_gap_audit/teacher_gap_train_rowwise.h5)
- weighting bands:
  [`exp1_teacher_gap_audit/teacher_gap_weighting_bands.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp1_teacher_gap_audit/teacher_gap_weighting_bands.json)

Train-side focus thresholds:

- `gap_q90 = 95.042069`
- `gap_q95 = 141.095757`
- `delta_q90 = 373.594583`
- `teacher_disp_q90 = 7.308768`

Interpretation:

- the remaining student-teacher gap was concentrated enough to justify bounded teacher-gap weighting
- edge and apex-adjacent rows stayed important
- the signal was not concentrated enough to justify reopening any retired route

## Phase 2

- projected-teacher DS probe summary:
  [`exp2_projected_teacher_ds_probe/phase2_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp2_projected_teacher_ds_probe/phase2_summary.json)
- projected-teacher DS probe CSV:
  [`exp2_projected_teacher_ds_probe/teacher_ds_probe_summary.csv`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp2_projected_teacher_ds_probe/teacher_ds_probe_summary.csv)

Exact mode remained the best tangent diagnostic.

Plastic directional probe summary:

- `ds_valid_val`, exact: relative-error p95 `1.045769`, cosine mean `0.897726`, finite rate `1.0`, switch rate `6.15e-04`
- `ds_valid_test`, exact: relative-error p95 `1.043742`, cosine mean `0.900727`, finite rate `1.0`, switch rate `8.50e-04`

Softmin at `tau = 0.25 / 0.50 / 1.00` made the tangent metrics worse rather than better. The projected teacher therefore stays diagnostically alive for tangents, but it does not reopen `DS`.

## Phase 3

- refinement summary:
  [`exp3_same_capacity_refinement/phase3_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp3_same_capacity_refinement/phase3_summary.json)
- refinement CSV:
  [`exp3_same_capacity_refinement/same_capacity_refinement_summary.csv`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp3_same_capacity_refinement/same_capacity_refinement_summary.csv)
- best val rowwise:
  [`exp3_same_capacity_refinement/best_phase3_val_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp3_same_capacity_refinement/best_phase3_val_rowwise.h5)
- best test rowwise:
  [`exp3_same_capacity_refinement/best_phase3_test_rowwise.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp3_same_capacity_refinement/best_phase3_test_rowwise.h5)

Validation results:

- `anchor_ema_restart`: `15.899072 / 18.059217 / 170.881073 / 2.263277e-08`
- `teacher_gap_cvar`: `16.619057 / 18.870619 / 174.028900 / 2.127750e-08`
- `edge_apex_delta_focus`: `18.217728 / 21.011776 / 192.699768 / 2.204672e-08`
- `full_combo_ema`: `16.981205 / 19.520977 / 182.852615 / 2.222299e-08`

What helped:

- low-LR warm restart from the anchor plus EMA

What did not help:

- the stronger train-gap CVaR-style weighting, which stayed worse than the anchor and missed the packet material-improvement bar
- the edge/apex plus delta-focused continuation
- the full combined recipe

Stop-rule status:

- the packet stop rule was **not** triggered because `anchor_ema_restart` improved validation broad / hard / p95 enough to clear the `0.5 / 0.3 / 5.0` improvement bar

## Phase 4

- summary:
  [`exp4_full_real_distill_followon/phase4_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_tail_reopen_20260327/exp4_full_real_distill_followon/phase4_summary.json)
- status: `skipped_not_justified`

Reason:

- the best Phase 3 run was better than the anchor but did not reach the narrow Phase 4 opening neighborhood `<= 15.8 / <= 18.4 / <= 168 / <= 1e-6`

## Decision

- whether the reopen bar was cleared: `no`
- which same-capacity change helped: `anchor_ema_restart`
- which same-capacity changes did not help: `teacher_gap_cvar`, `edge_apex_delta_focus`, `full_combo_ema`
- whether the projected teacher is tangent-viable enough to keep diagnostically alive: `yes, diagnostically alive only`
- whether Phase 4 was justified: `no`
- whether compression remains closed: `yes`
- whether `DS` is still blocked: `yes`
- whether the route should continue or stop: `continue narrowly`

Bottom line:

This packet made real progress, but not enough progress to reopen compression. The same-capacity line is still alive because EMA warm restart moved the anchor materially in the right direction, but the remaining tail did not close enough to justify a broader reopen, a full-real follow-on, or any student `DS` work.
