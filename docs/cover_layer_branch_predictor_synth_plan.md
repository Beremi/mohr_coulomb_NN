# Cover Layer Synthetic-Only Branch Predictor Plan

## Goal

Train a **branch predictor for the fixed `cover_layer` material using only synthetic training data**, then validate it on real cover-layer data from the full constitutive export.

The target is not a partial improvement. The target is:

- near-perfect branch prediction on real data
- especially strong `left_edge` / `right_edge` recall
- exact 11-point element branch-pattern consistency

## Frozen Problem Definition

We work on one P2 tetrahedral element state at a time.

Inputs available from the real export:

- element coordinates `X` for the 10 P2 nodes
- nodal displacements `U`
- 11 integration-point strains `E`
- exact constitutive branch labels from the exact constitutive operator

For the branch predictor itself, the preferred decomposition is:

1. model a realistic synthetic distribution in canonicalized element-state space `(X, U)`
2. compute `E` exactly via FE kinematics
3. predict the 11 integration-point branches from those exact `E`

This keeps FE kinematics exact and only learns the constitutive classification step.

## Success Metrics

The branch predictor is considered good enough only if all of these are true on **real held-out calls**:

- per-integration-point accuracy `>= 99.5%`
- stable-region accuracy `>= 99.9%`
- `left_edge` recall `>= 99.0%`
- `right_edge` recall `>= 99.0%`
- exact 11-point branch-pattern accuracy `>= 97.0%`
- no systematic confusion outside ambiguity bands

If we miss these, we iterate. The plan below defines where to return.

## Iteration Loop

Always follow this loop:

1. real benchmark freeze
2. element-block extraction and verification
3. synthetic generator fit
4. synthetic-vs-real deformation diagnostics
5. synthetic branch-label generation
6. branch predictor training
7. real validation and error dissection
8. return to the earliest failing phase

## Checkpoint Plan

- `[x] P0 Confirm real full-export element organization`
  Comment: verify that cover-layer real data can be grouped into element blocks with 11 integration points.
  Exit criterion: exact `E = B @ U` consistency and stable cover-layer element indexing.
  If this fails: stop and fix indexing / grouping before any ML work.
  Result: the full export groups cleanly into `5124` cover-layer elements per call with `11` integration points each; exact-vs-export stress mismatch on the first smoke pass stayed at numerical-noise scale (`mae ~= 2e-7`, `max_abs ~= 7.6e-6`).
  Next: freeze the real call split and keep building canonicalized element-state blocks.

- `[x] P1 Freeze the real benchmark split`
  Comment: split by whole constitutive calls, not random rows.
  Exit criterion: three frozen splits exist:
  - `generator_fit`
  - `real_val`
  - `real_test`
  If this fails: do not train anything; fix split leakage first.
  Result: a deterministic call split with seed `0` is now frozen at `332 / 111 / 111` calls for `generator_fit / real_val / real_test`.
  Next: build branch-labeled canonicalized cover-layer element blocks against this split.

- `[ ] P2 Build canonicalized cover-layer element blocks`
  Comment: for each real element state, extract `(X, U, E_11, branch_11)` and canonicalize `(X, U)`.
  Exit criterion: we can reconstruct and store canonicalized element states without geometry corruption.
  If this fails: return to `P0`.

- `[ ] P3 Fit the first synthetic element-state generator`
  Comment: start simple, not with a large learned generative model.
  Candidate order:
  1. PCA + Gaussian mixture on reduced displacement coordinates
  2. Gaussian copula on reduced coordinates
  3. branch-conditioned mixture if needed
  Exit criterion: synthetic samples are valid and pass deformation sanity checks.
  If this fails: keep the benchmark frozen and improve only the generator, not the classifier.

- `[ ] P4 Validate synthetic deformation coverage against real data`
  Comment: this is the most important gate before classifier training.
  Compare synthetic vs real on:
  - `E` marginals
  - principal strains
  - `E_11` covariance
  - branch frequencies after exact relabeling
  - 11-point branch-pattern frequencies
  Exit criterion: synthetic coverage is close enough to real that real validation failures are likely classifier failures, not generator failures.
  If this fails: return to `P3`.

- `[ ] P5 Build exact synthetic branch datasets`
  Comment: generate synthetic `(X, U)`, compute exact `E_11`, compute exact branch labels.
  Exit criterion: one synthetic training set and one untouched synthetic holdout set exist.
  If this fails: return to `P3` or `P4` depending on whether the issue is generation or filtering.

- `[ ] P6 Train baseline branch predictors`
  Comment: train synthetic-only, validate on real.
  Model order:
  1. shared pointwise classifier on each of the 11 integration points
  2. element-context classifier over all 11 points
  3. hierarchical elastic/plastic then plastic-subclass classifier
  Exit criterion: at least one model reaches strong real validation accuracy without obvious class collapse.
  If this fails:
  - if errors are global: return to `P4`
  - if errors are mostly rare branches: return to `P5`
  - if errors are mostly boundary confusion: move to `P7`

- `[ ] P7 Add boundary-focused synthetic curriculum`
  Comment: oversample hard synthetic states near elastic/plastic, smooth/edge, and edge/apex transitions.
  Exit criterion: real `left_edge` / `right_edge` recall improves materially.
  If this fails: return to `P4` and reconsider the generator rather than pushing longer training.

- `[ ] P8 Real validation and error dissection`
  Comment: evaluate on real held-out calls only after a serious synthetic-only training run.
  Required reports:
  - per-IP confusion
  - per-branch precision/recall
  - exact 11-point pattern accuracy
  - worst-call table
  - accuracy vs branch margin
  - accuracy vs strain magnitude
  Exit criterion: one model meets the success metrics above.
  If this fails:
  - errors cluster in a few calls -> return to `P3/P4`
  - errors cluster near boundaries -> return to `P7`
  - errors are broad and pattern-level -> strengthen the model in `P6`

## Phase-By-Phase Implementation Tasks

### Phase A — Data Foundation

1. Add element-block extraction from the full export for cover-layer calls.
2. Add canonicalization for local element coordinates and displacements.
3. Add a smoke script that summarizes:
   - element counts
   - deformation validity
   - branch-pattern counts
   - canonicalized coordinate / displacement ranges
4. Freeze real call-level splits.

Return point:
- if grouping or canonicalization looks wrong, stop here and fix it before fitting any generator.

### Phase B — Generator

1. Reduce canonicalized displacement states to a compact latent space.
2. Fit a simple probabilistic generator.
3. Add realism filters:
   - positive volume
   - positive Jacobian / non-collapse
   - real-envelope displacement magnitude
   - real-envelope strain magnitude
4. Compare generated `E_11` against real `E_11`.

Return point:
- if synthetic coverage is off, do not start the classifier; improve the generator only.

### Phase C — Branch Labels

1. Generate synthetic `(X, U)` states.
2. Compute exact `E_11`.
3. Compute exact branch labels from the constitutive operator.
4. Build a synthetic train / synthetic holdout split.

Return point:
- if branch frequencies or branch patterns are unrealistic, go back to Phase B.

### Phase D — Classifier

1. Train a shared pointwise classifier.
2. Train an element-context classifier.
3. If needed, train a hierarchical classifier.
4. Always select models on **real validation** only.

Return point:
- if the classifier misses easy states, go back to Phase B
- if it only misses boundaries, go to Phase E

### Phase E — Boundary Curriculum

1. Generate synthetic hard cases near constitutive boundaries.
2. Add curriculum / class rebalance.
3. Retrain from the best Phase D model family.

Return point:
- if boundary errors remain high, revisit generator coverage before adding more model complexity.

## Immediate Next Actions

- `[x] A1` Write the first real-data preparation utilities.
- `[x] A2` Run a smoke summary on a few real cover-layer calls.
- `[x] A3` Freeze the real benchmark split.
- `[x] A4` Implement the first generator-fit baseline.

Current artifacts:

- extraction / canonicalization utilities:
  - [full_export.py](../src/mc_surrogate/full_export.py)
- smoke summary script:
  - [prepare_cover_layer_branch_blocks.py](../scripts/experiments/prepare_cover_layer_branch_blocks.py)
- first smoke output:
  - `../experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/summary.json`
- frozen call split:
  - `../experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json`
- first generator baseline smoke:
  - [fit_cover_layer_branch_generator_baseline.py](../scripts/experiments/fit_cover_layer_branch_generator_baseline.py)
  - `../experiment_runs/real_sim/cover_layer_branch_generator_baseline_20260314/summary.json`

Early generator note:

- the first PCA + Gaussian displacement baseline is intentionally simple and currently **off-domain** on the smoke subset
- in the first smoke run, the synthetic branch histogram over-produced `apex` and under-produced realistic strain magnitudes
- this is acceptable at this stage because it gives us a clean baseline to improve in `P3/P4`
- the first multi-mode generator screen improved this materially when call selection was spread across deformation regimes:
  - best smoke mode so far: `empirical_local_noise`
  - artifact: `../experiment_runs/real_sim/cover_layer_branch_generator_screen_20260314/summary.json`
- the first synthetic-only pointwise branch-classifier smoke run is not good enough yet:
  - strong collapse to `elastic` / `apex`
  - very poor `smooth`, `left_edge`, `right_edge` recall on real data
  - artifact: `../experiment_runs/real_sim/cover_layer_branch_predictor_baseline_20260314/summary.json`

Interpretation right now:

- the FE-compatible synthetic path is working
- regime-aware fit-call selection matters
- the current generator is still not good enough to support a high-quality real branch classifier
- next work should stay focused on `P3/P4`, not on making the classifier larger

## Notes

- The classifier should learn constitutive branching, not FE kinematics. We already know FE kinematics exactly.
- The best early classifier input will probably be exact per-IP strain features and possibly exact elastic trial-state features.
- The hard part is expected to be synthetic deformation coverage, not the final classifier architecture.
