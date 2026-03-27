# Projection-Student Hard-Tail Closure Work Packet

Date: `2026-03-27`

Repository: `/home/beremi/repos/mohr_coulomb_NN`

Source note:
This document is the normalized follow-on task memo after the executed March 27 projection-student preservation/compression packet. It is a forward execution brief for the next live packet, not an execution report.

Relationship to the previous packet:

- previous executed task memo:
  [`docs/tasks/projection_student_preservation_compression_work_packet_20260327.md`](/home/beremi/repos/mohr_coulomb_NN/docs/tasks/projection_student_preservation_compression_work_packet_20260327.md)
- previous execution report:
  [`docs/executions/projection_student_preservation_compression_work_packet_20260327.md`](/home/beremi/repos/mohr_coulomb_NN/docs/executions/projection_student_preservation_compression_work_packet_20260327.md)

The March 27 packet changed the state of the route in a very specific way.

The projected teacher remains excellent on forward stress and admissibility.
The same-capacity preservation control materially improved over the March 26 `medium_exact` student.
But the route is still blocked on a concentrated hard-tail problem:

- projected teacher validation: `5.877489 / 7.062693 / 76.398941 / 2.199150e-08`
- best preserved student validation: `16.891312 / 19.413279 / 196.215149 / 2.329866e-08`

That means the live bottleneck is no longer generic “compression.”
The live bottleneck is:

> **close the remaining hard-tail (`hard_p95_principal`) gap to the projected teacher while preserving the current broad / hard gains and keeping numerical-zero yield.**

This packet is therefore a **hard-tail closure packet**, not a compression packet and not a `DS` packet.

## 0. Strategic state

### 0.1 Closed lines

The following lines remain closed and should not be reopened inside this packet:

- packet2 / packet3 / packet4 as active admissible-surface families
- Program X follow-ons
- packet5
- atlas variants
- routing redesign
- exact-latent follow-ons
- separate `DS` heads
- compression sweeps below the current best preserved student

### 0.2 Only live direct-replacement route

The only live route is:

- projection-student
- projected-teacher preservation first
- hard-tail closure second
- compression only after hard-tail closure succeeds
- broader `DS` work only after hard-tail closure succeeds

### 0.3 What the March 27 packet proved

The last packet supports four strong conclusions:

1. The exact projection layer is no longer the bottleneck.
2. The teacher-preservation path is real; a `256x4` projected student materially improved on the March 26 bounded student.
3. The route is still not ready for compression because the preserved student missed the explicit `hard_p95 <= 180` opening gate.
4. The route is still not ready for `DS` because both the projected-teacher probe and the best preserved student remain above the tangent-open bar.

### 0.4 New scientific question

The live question is now:

> **Can the repository close the remaining hard-tail gap with same-capacity or near-same-capacity projected students by using better error localization, better tail-focused preservation losses, and better targeted sampling, without reopening compression or `DS` too early?**

## 1. Benchmark and immutable references

Keep the canonical March 24 benchmark frozen exactly.

### 1.1 Frozen benchmark

- grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`
- panel sidecar: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5`
- split seed: `20260324`
- split unit: constitutive call
- split fractions: `70 / 15 / 15`
- samples per call: `512`

### 1.2 Canonical sizes

| panel         |   rows |
| ------------- | -----: |
| train         | 198656 |
| val           |  42496 |
| test          |  42496 |
| hard_val      |  21198 |
| hard_test     |  22705 |
| ds_valid      |  73261 |
| ds_valid_val  |  11224 |
| ds_valid_test |   9839 |

Use the full-panel `ds_valid_val/test` counts above, not smaller plastic-only reinterpretations.

### 1.3 Current live anchors

#### Projected teacher

Validation:

- broad plastic MAE: `5.877489`
- hard plastic MAE: `7.062693`
- hard p95 principal: `76.398941`
- yield p95: `2.199150e-08`

Test:

- broad plastic MAE: `6.083902`
- hard plastic MAE: `7.344495`
- hard p95 principal: `79.955505`
- yield p95: `2.252151e-08`

#### March 26 bounded student winner

Validation:

- broad plastic MAE: `27.308128`
- hard plastic MAE: `29.599710`
- hard p95 principal: `250.701462`
- yield p95: `2.270879e-08`

#### March 27 preservation control

Validation:

- broad plastic MAE: `16.891312`
- hard plastic MAE: `19.413279`
- hard p95 principal: `196.215149`
- yield p95: `2.329866e-08`

Test:

- broad plastic MAE: `17.608580`
- hard plastic MAE: `20.012241`
- hard p95 principal: `201.945511`
- yield p95: `2.342785e-08`

#### Projected-teacher `DS` probe state

- overall status: `mixed`
- validation status: `blocked`
- test status: `mixed`
- validation softmin forward stress-component MAE vs exact projection: `1.012578`
- test softmin forward stress-component MAE vs exact projection: `0.969067`

## 2. Packet objective

The objective of this packet is to determine, with maximal practical progress, whether the route can close the remaining hard-tail gap by staying within the projection-student preservation family.

By the end of the packet you should ideally have:

1. a control-zero hard-tail localization report
2. reusable rowwise slice artifacts for the current best preserved student
3. targeted tail-closure training mechanics in the projected-student path
4. a bounded same-capacity tail-closure sweep
5. a bounded near-same-capacity follow-on only if justified
6. a decision-grade execution report and a `progress.md` update if state changes materially

## 3. Success bars for this packet

### 3.1 Minimum packet success

Treat the packet as a strong success if at least one validation run reaches:

- broad plastic MAE `<= 18`
- hard plastic MAE `<= 20`
- hard p95 principal `<= 180`
- yield p95 `<= 1e-6`

That is the explicit bar that would have opened Tier 1 compression in the previous packet.

### 3.2 Strong packet success

Treat the packet as very strong if at least one validation run reaches:

- broad plastic MAE `<= 15`
- hard plastic MAE `<= 18`
- hard p95 principal `<= 160`
- yield p95 `<= 1e-6`

That is the neighborhood that would make compression plausible again.

### 3.3 `DS` remains closed unless this bar is met

Do not open broader student-level `DS` work unless a student reaches about:

- broad plastic MAE `<= 10`
- hard plastic MAE `<= 12`
- hard p95 principal `<= 120`
- yield p95 `<= 1e-6`

### 3.4 Stop bar

Stop the packet's bounded escalation if either of the following becomes clear:

- same-capacity variants cannot materially beat the current `196.215149` validation `hard_p95_principal`
- any p95 improvement comes only by giving back too much broad / hard MAE or by harming numerical-zero yield

## 4. Output root and deliverables

Recommended output root:

```text
experiment_runs/real_sim/projection_student_hard_tail_closure_20260327/
```

Recommended substructure:

```text
exp0_control_zero_slices/
exp1_tail_closure_controls/
exp2_near_same_capacity_followon/
```

Required deliverables:

- rowwise error H5 or CSV artifacts for the current best preserved student
- slice summary JSON/CSV tables
- code changes for any tail-focused training mechanics
- bounded training artifacts for same-capacity and any justified near-same-capacity controls
- execution report:
  `docs/executions/projection_student_hard_tail_closure_work_packet_20260327.md`
- `progress.md` update if state changes materially

## 5. Phase A — control-zero hard-tail localization

The first task is not to train a new model. The first task is to localize where the current hard tail actually lives.

### A0. Freeze control zero

Treat the March 27 preservation control as control zero:

- checkpoint:
  `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/control_exact_w256_d4/best.pt`
- validation summary:
  `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/control_exact_w256_d4/projected_student_eval_val.json`
- test summary:
  `experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/control_exact_w256_d4/projected_student_eval_test.json`

### A1. Build rowwise control-zero error slices

For validation and test, cache at minimum:

- row ids: `split_id`, `source_call_id`, `source_row_in_call`
- true branch id
- hard mask
- `ds_valid` mask
- all saved `near_*` boundary masks
- teacher projection candidate id
- teacher projection displacement norm
- exact principal stress
- projected teacher principal stress
- control-zero projected student principal stress
- control-zero absolute principal error
- control-zero principal max-abs error

### A2. Produce concentration summaries

Summarize the current `hard_p95` concentration by:

- true plastic branch
- teacher projection candidate
- hard panel vs non-hard
- `near_*` boundary masks
- teacher projection displacement decile
- top 1 percent, 5 percent, and 10 percent error rows
- constitutive call grouping, if certain calls dominate the tail

The goal is to answer:

- Is the tail mostly edge-driven?
- Is it mostly boundary-driven?
- Is it concentrated in high-displacement rows?
- Is it concentrated in a small subset of calls or panels?

### A3. Make the next-phase weighting evidence-based

Do not guess the next weighting scheme. Use the slice analysis to justify it.

## 6. Phase B — implement tail-closure mechanics

This packet may extend the training path, but only in ways that stay inside the projected-student preservation route.

### B0. Allowed training-path changes

Allowed and likely useful changes:

- configurable hard-row weighting
- configurable boundary-row weighting
- configurable high-displacement weighting
- configurable candidate-specific weighting
- optional branch-specific weighting if the tail is clearly concentrated by true branch
- optional top-k, top-quantile, or clipped-tail principal loss on hard rows
- optional stronger weighting on projected-teacher preservation for the rows that dominate the tail

### B1. Still keep these constraints

Keep these constraints:

- exact projection remains the deployment map
- no separate `DS` head
- no new 5-way projection-candidate classifier head unless a very strong artifact-backed reason emerges
- keep backward compatibility for existing projected-student datasets and runs

### B2. Tail-focused loss intent

Preserve the existing preservation objective, but allow bounded additions that directly target the hard tail.

Examples that are in-family:

- stronger weighting on hard / boundary / displacement-heavy rows
- a capped tail loss on the largest hard-row principal errors
- stronger teacher-projected alignment on the rows where the current model misses the projected teacher most badly

## 7. Phase C — bounded same-capacity tail-closure sweep

Run a bounded same-capacity sweep before any capacity increase.

### C0. Required baseline

Always keep one exact comparison to control zero:

- `trial_principal_geom_projected_student`
- width `256`
- depth `4`
- exact projection
- warm start from the March 27 preservation control or from `candidate_b`, whichever is empirically stronger in the current code path

### C1. Suggested bounded variants

Run a small bounded set only. Good default list:

1. stronger hard / boundary / displacement weighting
2. tail loss on hard rows
3. combined weighting + tail loss
4. one evidence-backed branch or candidate concentration variant if the slice analysis identifies a clear dominant failure mode

Do not turn this into an open-ended hyperparameter sweep.

### C2. Required evaluation

Every run must report at minimum:

- broad plastic MAE
- hard plastic MAE
- hard p95 principal
- yield p95
- rowwise or sliced hard-tail summaries consistent with Phase A

## 8. Phase D — near-same-capacity follow-on only if justified

Only if the same-capacity sweep shows the route is clearly alive, you may open a tiny near-same-capacity follow-on.

Allowed examples:

- `288x4`
- `256x5`

Only open this phase if at least one same-capacity run does one of:

- clears the `<= 180` validation `hard_p95` gate
- or reaches roughly `<= 185` with corroborated test improvement and no regression on broad / hard / yield

Do not open a wide architecture search.
Do not reopen compression tiers here.

## 9. Explicit non-goals

Do not spend time on:

- packet2 / packet3 / packet4 / Program X reopen work
- any new routing study
- any new atlas family
- any packet5-like admissible representation search
- compression below the current best preserved student unless the strong bar is reached
- student-level `DS` supervision
- FE harness construction
- open-ended model-size searches

## 10. Decision rules

### Continue narrowly if at least one is true

- same-capacity or near-same-capacity tail-closure materially improves the current `196.215149` validation `hard_p95_principal`
- a run clears the explicit `<= 180` gate while preserving `<= 18 / <= 20 / <= 1e-6`
- slice analysis reveals a very concentrated and actionable failure mode even if the first sweep is only partially successful

### Reopen compression later only if this becomes true

- a run reaches roughly `<= 15 / <= 18 / <= 160 / <= 1e-6`

### Keep `DS` blocked unless this becomes true

- a run reaches roughly `<= 10 / <= 12 / <= 120 / <= 1e-6`

### Stop the near-same-capacity escalation if this becomes true

- same-capacity variants fail to materially improve the hard tail
- or p95 improves only by sacrificing too much broad / hard accuracy

## 11. Required read order for the executing agent

1. `progress.md`
2. `docs/executions/projection_student_preservation_compression_work_packet_20260327.md`
3. `docs/tasks/projection_student_preservation_compression_work_packet_20260327.md`
4. `docs/executions/projection_student_work_packet_20260326.md`
5. `docs/repository_general_report_20260326.md`
6. `docs/repository_general_report_metric_dictionary_20260326.md`
7. `docs/mohr_coulomb.md`

Then inspect these implementation entrypoints:

- `src/mc_surrogate/projection_student_preservation.py`
- `src/mc_surrogate/training.py`
- `scripts/experiments/run_projection_student_preservation_compression.py`
- `src/mc_surrogate/principal_projection.py`
- `tests/test_projection_student_preservation.py`
- `tests/test_training.py`

## 12. Desired end state

The best end state of this packet is not another high-level memo.

The best end state is that the repo actually:

- localizes where the current hard tail lives
- implements evidence-based tail-closure mechanics
- shows whether same-capacity or near-same-capacity projected students can clear the `hard_p95 <= 180` gate
- leaves behind a much clearer next move than the current “narrowed preservation-first” state

If the route succeeds, leave a credible pre-compression continuation state.

If it fails, fail decisively and document exactly why the hard-tail closure route stalled.
