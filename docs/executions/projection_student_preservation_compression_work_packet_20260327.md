# Projection-Student Preservation Compression Work Packet

Execution report for the March 27, 2026 `projection-student` preservation/compression packet.

Paired task memo:
[`docs/tasks/projection_student_preservation_compression_work_packet_20260327.md`](/home/beremi/repos/mohr_coulomb_NN/docs/tasks/projection_student_preservation_compression_work_packet_20260327.md)

Repository:
`/home/beremi/repos/mohr_coulomb_NN`

Output root:
`/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327`

## Strategic Setup

This packet continued the only still-live direct-replacement route after the March 26 projection audit:

- preserve the projected teacher first
- compress only after preservation works
- keep `projection-student` as the only active direct-replacement family

What remained closed during this packet:

- packet2 / packet3 / packet4 as active research families
- Program X follow-ons
- packet5
- new atlas variants
- routing redesign
- exact-latent follow-ons
- separate `DS` heads

Controlling scientific questions:

- Is the frozen projected teacher numerically stable enough to justify later tangent work?
- Can a same-capacity projected student preserve the projected teacher materially better than the March 26 `medium_exact` baseline?
- Does the route clear the explicit gate for bounded compression, or should compression stay closed?

## Benchmark and Anchors

Canonical benchmark:

- grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`
- panel sidecar: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5`
- split seed: `20260324`
- split unit: constitutive call
- split fractions: `70 / 15 / 15`
- samples per call: `512`

Canonical sizes:

- `train`: `198656`
- `val`: `42496`
- `test`: `42496`
- `hard_val`: `21198`
- `hard_test`: `22705`
- `ds_valid`: `73261`
- `ds_valid_val`: `11224`
- `ds_valid_test`: `9839`

Teacher used:

- checkpoint: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt`
- summary: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/summary.json`

March 26 continuation anchors:

- projected teacher validation broad / hard / p95 / yield: `5.877489` / `7.062693` / `76.398941` / `2.199150e-08`
- projected teacher test broad / hard / p95 / yield: `6.083902` / `7.344495` / `79.955505` / `2.252151e-08`
- March 26 `medium_exact` validation broad / hard / p95 / yield: `27.308128` / `29.599710` / `250.701462` / `2.270879e-08`

Compression and `DS` gates carried into this packet:

- Tier 1 open bar: `broad <= 18`, `hard <= 20`, `hard_p95 <= 180`, `yield <= 1e-6`
- broader `DS` reopen bar: `broad <= 10`, `hard <= 12`, `hard_p95 <= 120`, `yield <= 1e-6`

## Implementation

Primary code paths used or extended in this packet:

- [`scripts/experiments/run_projection_student_preservation_compression.py`](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/run_projection_student_preservation_compression.py)
- [`src/mc_surrogate/projection_student_preservation.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/projection_student_preservation.py)
- [`src/mc_surrogate/training.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/training.py)
- [`src/mc_surrogate/principal_projection.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/principal_projection.py)
- [`tests/test_projection_student_preservation.py`](/home/beremi/repos/mohr_coulomb_NN/tests/test_projection_student_preservation.py)
- [`tests/test_training.py`](/home/beremi/repos/mohr_coulomb_NN/tests/test_training.py)

### Phase A0: candidate-zero teacher cache

Implemented a full-row cache that joins the frozen teacher checkpoint outputs with the canonical grouped dataset and panel sidecar. The cache stores:

- exact stress and exact principal stress
- exact trial principal stress
- teacher checkpoint stress before elastic dispatch
- elastic-dispatched teacher stress
- exact-projected teacher stress
- projection displacement vector and norm
- selected projection candidate, full candidate tensor, squared distances, and feasible-mask audit
- split ids, call-local row ids, branch ids, hard masks, `ds_valid` masks, and boundary-nearness masks

### Phase A1: projected-teacher `DS` probe

Implemented a finite-difference directional tangent probe on canonical `ds_valid_val` and `ds_valid_test` rows:

- centered finite differences with `h = 1e-6 * max(max(|strain|), 1)`
- `3` fixed random unit directions per row from seed `20260327`
- both `exact` and `softmin(tau=0.05)` projection modes
- per-row outputs for `||Jv_pred - DS_true v||`, relative error, cosine similarity, base / plus / minus candidate ids, and candidate-switch events
- explicit boundary-instability summaries over the saved `near_*` masks

Implementation note:

- the probe path was tightened to reuse a loaded teacher checkpoint and batch `+h/-h` perturbations together, so the packet can run the full probe without repeatedly reloading the same checkpoint

### Phase B0 and B1: preservation dataset and same-capacity control

Extended the projected-student training path to consume preservation targets and sample weights:

- optional dataset keys: `teacher_provisional_stress_principal`, `teacher_projected_stress_principal`, `teacher_projection_delta_principal`, `teacher_projection_candidate_id`, `teacher_projection_disp_norm`, `ds_valid_mask`, `sampling_weight`, and `trial_principal_geom_feature_f1`
- backward compatibility: legacy `teacher_stress_principal` still works as the provisional-teacher alias
- training now auto-uses `WeightedRandomSampler` when `sampling_weight` is present
- projected-student loss now follows the preservation objective: exact-principal fit, projected-teacher preservation, provisional-teacher preservation, projection-delta preservation, and branch CE over the existing 4-way plastic head

Preservation dataset weighting rule used in this packet:

- base `1.0`
- `x1.5` for `hard_mask`
- `x1.5` for teacher candidate in `{left_edge, right_edge}`
- `x1.5` for `teacher_projection_disp_norm >= train p90`

Same-capacity control configuration:

- model: `trial_principal_geom_projected_student`
- width / depth: `256 / 4`
- params: `540935`
- projection mode: `exact`
- batch size: `4096`
- lr: `3e-4`
- weight decay: `1e-5`
- patience: `12`
- epochs: `60`
- seed: `20260327`
- warm start: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/candidate_b/finetune/best.pt`

### Phase C: bounded compression gating

Compression logic stayed strict in this packet:

- open Tier 1 only if the same-capacity preservation control beats `medium_exact` on all three primary validation stress metrics, preserves numerical-zero yield, and clears the explicit `18 / 20 / 180 / 1e-6` gate
- open Tier 2 only if a Tier 1 winner clears the stricter `15 / 18 / 160 / 1e-6` gate
- if the same-capacity control misses the Tier 1 bar, compression stops rather than expanding the search

## Results

### Phase A0: candidate-zero teacher cache

Validation:

| Evaluation | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Dispatch teacher | 5.771003 | 6.949572 | 76.398941 | 1.001325e-01 |
| Projected teacher | 5.877489 | 7.062693 | 76.398941 | 2.199150e-08 |

Test:

| Evaluation | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Dispatch teacher | 5.981674 | 7.241915 | 79.455521 | 1.061293e-01 |
| Projected teacher | 6.083902 | 7.344495 | 79.955505 | 2.252151e-08 |

Projection displacement summary on plastic rows:

- validation: `mean 2.858072`, `p50 1.005166`, `p95 11.713972`, `max 123.145325`
- test: `mean 3.011442`, `p50 1.097234`, `p95 12.012154`, `max 251.903427`
- validation candidate counts: `{'pass_through': 10129, 'smooth': 20464, 'left_edge': 1572, 'right_edge': 266, 'apex': 1912}`
- test candidate counts: `{'pass_through': 10215, 'smooth': 21541, 'left_edge': 1876, 'right_edge': 291, 'apex': 2180}`

Interpretation:

- candidate zero remained scientifically strong: projection drove yield back to numerical zero while preserving the frozen teacher's forward-stress quality close to the March 26 audit

### Phase A1: projected-teacher `DS` probe

| Split | Rows | Plastic rows | Exact plastic rel-error p95 | Exact plastic cosine mean | Exact boundary switch | Softmin plastic rel-error p95 | Softmin plastic cosine mean | Softmin boundary switch | Softmin forward stress MAE vs exact | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| val | 11224 | 8947 | 1.039073 | 0.900533 | 0.000624 | 1.087665 | 0.895754 | 0.000624 | 1.012578 | `blocked` |
| test | 9839 | 7650 | 1.037820 | 0.899618 | 0.001081 | 1.084557 | 0.894981 | 0.001081 | 0.969067 | `mixed` |

Boundary observations:

- validation exact boundary switch rate: `0.000624`
- validation softmin boundary switch rate: `0.000624`
- test exact boundary switch rate: `0.001081`
- test softmin boundary switch rate: `0.001081`

Interpretation:

- overall probe status: `mixed`
- validation / test status: `blocked` / `mixed`
- candidate-switch rates were very small, so the projection map was not thrashing across candidate boundaries
- the remaining problem was tangent quality: directional relative-error tails stayed too large, especially on plastic and boundary-near rows
- softmin did not create a clean forward-equivalent tangent path in validation because its forward stress delta versus exact stayed around one stress unit on average

### Phase A2: FE shadow

- status: `skipped_no_reusable_harness`
- reason: `No in-repo FE shadow harness is available for the canonical benchmark.`
- no new FE harness was built in this packet, per the task stop rule

### Phase B0: preservation dataset

- dataset path: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp2_preservation_dataset/projection_student_preservation_plastic.h5`
- rows: `236700`
- split counts: `{'train': 166254, 'val': 34343, 'test': 36103}`
- hard counts: `{'train': 102607, 'val': 21198, 'test': 22705}`
- `ds_valid` counts: `{'train': 39946, 'val': 8947, 'test': 7650}`
- sampling-weight summary: `mean 1.416002`, `p50 1.500000`, `p95 2.250000`, `max 3.375000`
- teacher projection-displacement summary: `mean 2.912682`, `p50 1.073631`, `p95 11.572837`, `max 255.412903`

Interpretation:

- the preservation dataset captured the projected teacher as a reusable supervision target rather than forcing every later run to rebuild projections from scratch
- weighting biased sampling toward hard rows, edge candidates, and large teacher-displacement rows exactly as scoped

### Phase B1: same-capacity preservation control

Validation comparison:

| Model | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Projected teacher | 5.877489 | 7.062693 | 76.398941 | 2.199150e-08 |
| March 26 `medium_exact` | 27.308128 | 29.599710 | 250.701462 | 2.270879e-08 |
| March 27 preservation control | 16.891312 | 19.413279 | 196.215149 | 2.329866e-08 |

Held-out test comparison:

| Model | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Projected teacher | 6.083902 | 7.344495 | 79.955505 | 2.252151e-08 |
| March 27 preservation control | 17.608580 | 20.012241 | 201.945511 | 2.342785e-08 |

Validation improvement versus March 26 `medium_exact`:

- broad plastic MAE improvement: `10.416817`
- hard plastic MAE improvement: `10.186432`
- hard p95 principal improvement: `54.486313`
- materially beats `medium_exact` on validation: `True`

Tier 1 opening gate check:

| Gate | Threshold | Control value | Pass |
| --- | ---: | ---: | --- |
| broad plastic MAE | 18.000000 | 16.891312 | `True` |
| hard plastic MAE | 20.000000 | 19.413279 | `True` |
| hard p95 principal | 180.000000 | 196.215149 | `False` |
| yield p95 | 1.000000e-06 | 2.329866e-08 | `True` |

Interpretation:

- this was the main success of the packet: a same-capacity preservation control materially closed the gap to the projected teacher relative to the March 26 student
- the remaining blocker was hard-tail quality, not admissibility or average stress accuracy
- Tier 1 open decision: `False`

### Phase C: bounded compression decision

- Tier 1 status: `skipped`
- Tier 2 status: `skipped`
- Tier 1 reason: `Preservation control did not clear the Tier 1 opening gate.`
- Tier 2 reason: `Tier 1 did not clear the Tier 2 opening gate.`

Interpretation:

- compression did not open because the packet stop rule was triggered before the sweep, not because an opened compression sweep failed
- in practical terms, Tier 1 looked dead for this packet because the preservation control was still too tail-heavy to justify spending budget on smaller models

## Decision

- projected-teacher `DS` viability: `mixed`
- same-capacity preservation materially closed the gap: `True`
- Tier 1 compression verdict: `dead`
- `DS` still blocked: `True`
- route decision: `narrow`

Decision reasoning:

- candidate zero remains strong enough that the route is worth preserving
- the teacher `DS` probe did not justify opening broader tangent work yet
- the same-capacity preservation control clearly improved on the March 26 student, so the route is not dead
- the remaining gap is concentrated in hard-tail principal-stress error, so the next packet should narrow onto that preservation problem before reopening compression

Bottom line:

The projection-student line should continue only in narrowed preservation-first mode. `DS` remains blocked, and bounded compression should remain closed until the preserved student closes the remaining `hard_p95` gap to the projected teacher.

## Artifact Map

Primary packet artifacts:

- teacher cache summary: [`exp0_teacher_cache/teacher_projection_preservation_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp0_teacher_cache/teacher_projection_preservation_summary.json)
- teacher cache H5: [`exp0_teacher_cache/teacher_projection_preservation_cache.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp0_teacher_cache/teacher_projection_preservation_cache.h5)
- teacher `DS` probe summary: [`exp1_teacher_ds_probe/teacher_ds_probe_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp1_teacher_ds_probe/teacher_ds_probe_summary.json)
- teacher `DS` probe CSV: [`exp1_teacher_ds_probe/teacher_ds_probe_summary.csv`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp1_teacher_ds_probe/teacher_ds_probe_summary.csv)
- teacher `DS` rowwise val H5: [`exp1_teacher_ds_probe/teacher_ds_probe_val.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp1_teacher_ds_probe/teacher_ds_probe_val.h5)
- teacher `DS` rowwise test H5: [`exp1_teacher_ds_probe/teacher_ds_probe_test.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp1_teacher_ds_probe/teacher_ds_probe_test.h5)
- preservation dataset summary: [`exp2_preservation_dataset/preservation_dataset_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp2_preservation_dataset/preservation_dataset_summary.json)
- preservation dataset H5: [`exp2_preservation_dataset/projection_student_preservation_plastic.h5`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp2_preservation_dataset/projection_student_preservation_plastic.h5)
- preservation control summary: [`exp3_preservation_control/phase3_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/phase3_summary.json)
- preservation control checkpoint: [`exp3_preservation_control/control_exact_w256_d4/best.pt`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/control_exact_w256_d4/best.pt)
- preservation control validation metrics: [`exp3_preservation_control/control_exact_w256_d4/projected_student_eval_val.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/control_exact_w256_d4/projected_student_eval_val.json)
- preservation control test metrics: [`exp3_preservation_control/control_exact_w256_d4/projected_student_eval_test.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp3_preservation_control/control_exact_w256_d4/projected_student_eval_test.json)
- Tier 1 compression status: [`exp4_tier1_compression/phase4_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp4_tier1_compression/phase4_summary.json)
- Tier 2 compression status: [`exp5_tier2_compression/phase5_summary.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/exp5_tier2_compression/phase5_summary.json)
- FE shadow status: [`fe_shadow_status.json`](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/projection_student_preservation_compression_20260327/fe_shadow_status.json)

