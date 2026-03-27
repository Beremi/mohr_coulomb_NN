# Projection-Student Work Packet

Execution report for the March 26, 2026 `projection-student` reopen route.

Paired task memo:
[`docs/tasks/projection_student_work_packet_20260326.md`](/home/beremi/repos/mohr_coulomb_NN/docs/tasks/projection_student_work_packet_20260326.md)

Repository:
`/home/beremi/repos/mohr_coulomb_NN`

## Strategic Setup

This work packet reopened the previously retired direct-replacement line only along one new route:

- accurate unconstrained stress predictor first
- exact admissibility second
- projection-student only

What remained closed:

- packet2 / packet3 / packet4 as active research families
- Program X follow-ons
- packet5
- new atlas variants
- new routing families
- more exact-latent studies
- separate `DS` heads

The controlling scientific question was whether a good stress predictor could be preserved by moving admissibility enforcement out of the network output parameterization and into an exact ordered-principal projection layer.

## Benchmark and Teacher

Canonical benchmark:

- grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`
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

Canonical raw teacher anchor:

- validation `broad_plastic_mae / hard_plastic_mae / hard_p95_principal / yield_violation_p95`
  = `5.771003 / 6.949572 / 76.398941 / 1.001325e-01`
- test `broad_plastic_mae / hard_plastic_mae / hard_p95_principal / yield_violation_p95`
  = `5.981674 / 7.241915 / 79.455521 / 1.061293e-01`

Comparison baselines:

- packet2 deployed validation:
  `32.458569 / 37.824432 / 328.641602 / 6.173982e-08`
- packet3 oracle validation:
  `29.512514 / 32.682789 / 302.360168 / 5.414859e-08`
- packet4 deployed validation:
  `41.599422 / 46.363712 / 374.894531 / 6.008882e-08`
- Program X oracle validation:
  `250.392441 / 328.857300 / 2331.702637 / 1.258780e+00`

## Implementation

### Exact MC projection operator

Implemented operator:

- exact Euclidean projection of ordered principal stress onto the convex admissible Mohr-Coulomb set
- elastic pass-through when already admissible
- plastic candidate set restricted to exactly five candidates:
  `pass_through`, `smooth`, `left_edge`, `right_edge`, `apex`
- deterministic tie-breaking
- `exact` and `softmin` modes

Candidate definitions:

- smooth-face projection onto `(1 + sin_phi) * s1 - (1 - sin_phi) * s3 = c_bar`
- left-edge KKT projection under `s1 = s2` and the yield plane
- right-edge KKT projection under `s2 = s3` and the yield plane
- apex projection to `s1 = s2 = s3 = c_bar / (2 * sin_phi)`

Main code paths:

- [`src/mc_surrogate/principal_projection.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/principal_projection.py)
- [`src/mc_surrogate/models.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/models.py)
- [`src/mc_surrogate/training.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/training.py)
- [`scripts/experiments/run_projection_student_cycle.py`](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/run_projection_student_cycle.py)
- [`tests/test_principal_projection.py`](/home/beremi/repos/mohr_coulomb_NN/tests/test_principal_projection.py)
- [`tests/test_training.py`](/home/beremi/repos/mohr_coulomb_NN/tests/test_training.py)

### Projected student path

Added model kind:

- `trial_principal_geom_projected_student`

Forward path:

- predict provisional ordered principal stress residuals
- add them to exact trial principal stress
- keep elastic rows on exact elastic dispatch
- project plastic rows through the ordered-principal MC projector
- reconstruct final Voigt stress only after projection

Bounded Phase 2 sweep:

- small `48x2`, `11767` params
- medium `80x2`, `29847` params
- large `96x2`, `41959` params
- `60` epochs, batch size `4096`, lr `3e-4`, weight decay `1e-5`, patience `12`
- exact projection remained stable, so no softmin fallback was needed

## Results

### Phase 0: raw teacher vs projected teacher

Validation:

| Evaluation | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Raw teacher | 5.771003 | 6.949572 | 76.398941 | 1.001325e-01 |
| Projected teacher | 5.877489 | 7.062693 | 76.398941 | 2.199150e-08 |

Test:

| Evaluation | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Raw teacher | 5.981674 | 7.241915 | 79.455521 | 1.061293e-01 |
| Projected teacher | 6.083902 | 7.344495 | 79.955505 | 2.252151e-08 |

Projection displacement summary:

- validation displacement `p50 / p95 / max`: `1.005166 / 11.713972 / 123.145325`
- test displacement `p50 / p95 / max`: `1.097234 / 12.012154 / 251.903427`

Validation branchwise displacement by true branch:

- smooth: `disp_mean 1.755613`, `disp_p95 7.440804`, `post_mae 3.542079`
- left_edge: `disp_mean 5.890051`, `disp_p95 21.946810`, `post_mae 9.658993`
- right_edge: `disp_mean 2.627641`, `disp_p95 9.042906`, `post_mae 11.216394`
- apex: `disp_mean 1.725519`, `disp_p95 5.811047`, `post_mae 1.808879`

Phase 0 decision:

- success: `pass`
- interpretation: the projection layer drove yield to numerical zero while preserving teacher stress quality well inside the continue bar

### Phase 2: bounded projected-student sweep

Validation:

| Run | Params | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small exact | 11767 | 29.981367 | 33.359127 | 295.993927 | 2.321422e-08 |
| medium exact | 29847 | 27.308128 | 29.599710 | 250.701462 | 2.270879e-08 |
| large exact | 41959 | 29.941349 | 31.859295 | 274.728699 | 2.344744e-08 |

Held-out test:

| Run | Params | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small exact | 11767 | 30.568085 | 33.636116 | 306.701538 | 2.287059e-08 |
| medium exact | 29847 | 28.099649 | 30.332142 | 259.966797 | 2.252200e-08 |
| large exact | 41959 | 30.192339 | 32.157185 | 284.643707 | 2.317175e-08 |

Winner:

- `medium_exact`
- checkpoint:
  `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/best.pt`
- it won because ranking was by validation `hard_p95_principal`, then `hard_plastic_mae`, then `broad_plastic_mae`, under the hard yield gate `yield_violation_p95 <= 1e-6`

### Comparison against key baselines

Validation comparison:

| Model or route | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: |
| Packet2 deployed | 32.458569 | 37.824432 | 328.641602 | 6.173982e-08 |
| Packet3 oracle | 29.512514 | 32.682789 | 302.360168 | 5.414859e-08 |
| Packet4 deployed | 41.599422 | 46.363712 | 374.894531 | 6.008882e-08 |
| Program X oracle | 250.392441 | 328.857300 | 2331.702637 | 1.258780e+00 |
| Projection-student Phase 2 winner | 27.308128 | 29.599710 | 250.701462 | 2.270879e-08 |

Interpretation:

- Phase 0 succeeded decisively and established that exact projection was scientifically viable for this benchmark
- Phase 2 materially beat packet2, packet3 oracle, and packet4 on the validation stress metrics while preserving numerical-zero yield
- Phase 2 remained far worse than the raw projected teacher, so the student did not yet close the gap needed for a credible final deployed replacement

## Decision

- Phase 0 succeeded: `True`
- Phase 2 materially beat packet2 while preserving admissibility: `True`
- `DS` still blocked: `True`
- continuation state: `continue_projection_student`

Reasoning:

- the route cleared the explicit continuation rule `<= 10 / <= 12 / <= 120 / <= 1e-6` in Phase 0
- the bounded student also beat the best deployed admissible-surface baseline while keeping yield at numerical zero
- the stronger `DS` reopen bar `<= 8 / <= 10 / <= 100 / <= 1e-6` was not reached, so Phase 3 was not opened in earnest

Bottom line:

The projection-student line should continue as the only live direct-replacement route, but only as a bounded continuation from this exact projection-based setup. `DS` remains blocked until forward stress quality moves much closer to the raw projected teacher.

## Artifact Map

Output root:

- `experiment_runs/real_sim/projection_student_20260326/`

Phase 0 artifacts:

- summary: `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_summary.json`
- cache: `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/teacher_projection_cache.h5`
- validation comparison table:
  `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_comparison_val.csv`
- test comparison table:
  `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_comparison_test.csv`

Phase 1 artifact:

- plastic-only derived dataset:
  `experiment_runs/real_sim/projection_student_20260326/exp1_student_dataset/projection_student_plastic_only.h5`

Phase 2 artifacts:

- phase summary:
  `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/phase2_summary.json`
- architecture summary:
  `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/projected_student_architecture_summary.csv`
- winner checkpoint:
  `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/best.pt`
- winner validation metrics:
  `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/projected_student_eval_val.json`
- winner test metrics:
  `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/medium_exact/projected_student_eval_test.json`

Phase 3 artifact:

- status note:
  `experiment_runs/real_sim/projection_student_20260326/exp3_ds_probe/ds_probe_summary.json`

Primary comparison anchors:

- packet2 winner:
  `experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp1_branchless_surface/`
- packet3 winner summaries:
  `experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/`
- packet4 winner summaries:
  `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/`
- Program X oracle summaries:
  `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/`
