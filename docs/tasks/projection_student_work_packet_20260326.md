# Projection-Student Work Packet

Date: `2026-03-26`

Repository: `/home/beremi/repos/mohr_coulomb_NN`

Source note:
This document is the normalized archival assignment copy of root [`tasks1.md`](/home/beremi/repos/mohr_coulomb_NN/tasks1.md). The root file remains the original strategic brief. This copy exists so the work packet has a stable task memo paired with its execution report.

## Strategic Override

Before this work packet, the controlling state was:

- packet4 closed
- Program X exact-latent study executed
- Program X Stage 0 exactness passed
- Program X teacher-forced true-branch oracle failed badly
- direct replacement retired
- `DS` blocked

This work packet reopens the previously retired direct-replacement line only along one scientifically distinct route:

- accurate unconstrained stress predictor first
- exact admissibility second
- via a `projection-student` program

The old packet2 / packet3 / packet4 / Program X line remains closed as a family. This packet does not authorize:

- `packet5`
- new atlas variants
- new routing families
- more exact-latent studies
- separate `DS` heads

## Primary Goal

Determine whether the repository can recover a credible deployed Mohr-Coulomb surrogate by:

1. starting from the best stress-accurate raw teacher on the canonical March 24 benchmark
2. projecting predicted plastic principal stresses onto the admissible MC set
3. turning that into a differentiable projection layer
4. training a small projected student around that layer
5. only then probing whether an autodiff path toward `DS` becomes realistic

Desired deliverables:

1. a Phase 0 projection-only audit
2. reusable code for an exact or near-exact MC projection layer in ordered principal space
3. a bounded projected-student training program
4. decision-grade reports and summaries
5. a `progress.md` update if the new work materially changes repository state

If the line fails early, it should fail decisively and be documented clearly.

## Required Read Order

Read these first:

1. [`tasks1.md`](/home/beremi/repos/mohr_coulomb_NN/tasks1.md)
2. [`progress.md`](/home/beremi/repos/mohr_coulomb_NN/progress.md)
3. [`docs/repository_general_report_20260326.md`](/home/beremi/repos/mohr_coulomb_NN/docs/repository_general_report_20260326.md)
4. [`docs/repository_general_report_metric_dictionary_20260326.md`](/home/beremi/repos/mohr_coulomb_NN/docs/repository_general_report_metric_dictionary_20260326.md)
5. [`docs/mohr_coulomb.md`](/home/beremi/repos/mohr_coulomb_NN/docs/mohr_coulomb.md)

Likely implementation entrypoints:

- [`src/mc_surrogate/models.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/models.py)
- [`tests/test_nn_replacement_surface.py`](/home/beremi/repos/mohr_coulomb_NN/tests/test_nn_replacement_surface.py)
- [`scripts/experiments/run_hybrid_gate_redesign_cycle.py`](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/run_hybrid_gate_redesign_cycle.py)
- [`scripts/experiments/run_safe_hybrid_pivot.py`](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/run_safe_hybrid_pivot.py)
- [`scripts/experiments/run_hybrid_campaign.py`](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/run_hybrid_campaign.py)

## Canonical Benchmark

Use the March 24 validation-first split unless a stronger artifact-backed reason appears:

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

Trust raw artifact files over old prose if numbers disagree.

## Teacher and Comparison Baselines

Default teacher candidate:

- checkpoint: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt`
- summary: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/summary.json`

Historical raw-stress context:

- `experiment_runs/real_sim/baseline_validation/real_metrics.json`
- `experiment_runs/real_sim/staged_20260312/staged_results_manual.json`

Comparison anchors:

- packet2 deployed winner:
  `experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp1_branchless_surface/`
- packet3 predicted and oracle winners:
  `experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/`
- packet4 deployed winner:
  `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/`
- Program X oracle:
  `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/`

## Reuse Bias

Do not start from scratch unless reuse truly blocks progress.

Prefer reusing:

- exact piecewise formulas already present in the repo
- branch-specialized admissible decode and projection helpers in [`src/mc_surrogate/models.py`](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/models.py)
- exact MC formulas and ordered-principal notes in [`docs/mohr_coulomb.md`](/home/beremi/repos/mohr_coulomb_NN/docs/mohr_coulomb.md)
- branch-surface and round-trip tests in [`tests/test_nn_replacement_surface.py`](/home/beremi/repos/mohr_coulomb_NN/tests/test_nn_replacement_surface.py)
- benchmark plumbing in the existing experiment scripts

The intended bias is:

- reuse exact piecewise formulas
- wrap them cleanly as a reusable projection operator
- add tests before trusting the new layer

## Program Structure

Use this output root unless a nearby variant is required:

- `experiment_runs/real_sim/projection_student_20260326/`

### Phase 0: Projection Audit of the Current Best Raw Baseline

Run the best validated raw teacher on the canonical benchmark.

Rules:

- keep exact elastic dispatch for elastic rows
- for plastic rows, post-process predicted ordered principal stresses with an exact admissibility projection onto the MC set

Record at minimum:

- broad plastic MAE
- hard plastic MAE
- hard p95 principal
- hard relative p95 principal if available
- yield violation p95
- yield violation max
- projection displacement norms
- projection displacement summary by true branch
- validation and test where feasible

Required comparison:

- raw teacher before projection
- projection-only audit result
- packet2 deployed winner
- packet3 predicted winner
- packet3 oracle winner
- packet4 deployed winner

If Phase 0 already blows up badly, do not hide it.

### Phase 1: Differentiable MC Projection Layer

Build a reusable spectral forward path:

- input: provisional ordered principal stress `sigma_tilde`, reduced material row
- output: admissible principal stress `Pi_MC(sigma_tilde, m)`

Required behavior:

- exact elastic pass-through when admissible
- plastic projection to admissible candidates
- support for smooth / left / right / apex regimes as needed
- exact candidate selection at evaluation time
- optional smoothed or soft-min relaxation only if training needs it

Required validation:

- unit tests for admissibility
- branch and candidate consistency tests
- stability tests on edge and apex neighborhoods
- no silent yield leakage

### Phase 2: Projected Student Training

Run a bounded training program only.

Recommended target:

- provisional ordered principal stress
- optional auxiliary branch logits

Recommended loss structure:

- post-projection stress loss against exact stress
- pre-projection distillation loss toward the teacher
- projection displacement penalty
- optional branch auxiliary loss

Run only three size bands unless a compelling reason appears:

- small: `10k-15k`
- medium: `20k-30k`
- large: `40k-60k`

Allowed bounded variants include:

- exact vs softened projection during training
- reduced material feature variants
- branch-logit auxiliary head on or off
- different displacement-penalty weights
- different teacher targets or teacher checkpoints on the same canonical benchmark

Do not turn this into a broad architecture search.

### Phase 3: `DS` Probe Only After Stress Quality Is Good Enough

Only if Phase 2 reaches the reopening band:

- use autodiff through the full projected forward path
- compare `J_pred v` against `DS_true v`
- use `2` to `4` random strain directions per sample

Do not build a separate `36`-output tangent head.

If the forward model never reaches the stated bars, do not spend time on full `DS`.

## Decision Rules

Continue the projection-student line only if Phase 0 or Phase 2 gets roughly:

- broad plastic MAE `<= 10`
- hard plastic MAE `<= 12`
- hard p95 principal `<= 120`
- yield violation p95 `<= 1e-6`

Only start real `DS` work if the route reaches something like:

- broad plastic MAE `<= 8`
- hard plastic MAE `<= 10`
- hard p95 principal `<= 100`
- yield violation p95 `<= 1e-6`

Retire the direct-replacement line again if either is true:

- projection-only audit already blows up badly
- projected student cannot stay clearly better than packet2 while driving yield to numerical zero

## Explicit Non-Goals

Do not spend time on:

- `packet5`
- another atlas variant
- another hard-routing family
- more exact-latent studies
- separate `DS` heads
- more gating or rejector redesign
- open-ended architecture sweeps

If the projection-student route fails decisively, do not start a full generalized-standard-material or implicit-potential program in the same pass. At most, leave a short strategic note describing that as the next scientifically distinct route.

## Required Deliverables

Aim to leave behind:

1. code implementing the projection operator and any new projected-student model path
2. tests for projection correctness and admissibility
3. scripts or commands needed to rerun the audit and the bounded training program
4. result artifacts under `experiment_runs/real_sim/projection_student_20260326/`
5. decision-grade documentation
6. a `progress.md` update if repository state materially changes

The final writeup must state:

- what teacher was used
- what exact projection operator was implemented
- whether Phase 0 succeeded or failed
- whether Phase 2 materially beat packet2 while preserving admissibility
- whether `DS` is still blocked
- whether the projection-student line should continue or stop
