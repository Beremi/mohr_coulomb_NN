# Expert Briefing: Mohr-Coulomb Surrogate Status

Prepared on 2026-03-23.

This note is meant to be self-contained. An external expert should be able to read only this document and understand the problem setting, the main experiments already performed, the current evidence, and where guidance is needed next.

## Executive Summary

We are building a local neural surrogate for the 3D Mohr-Coulomb constitutive update used at finite-element integration points in slope-stability analysis.

The main result so far is mixed:

- the problem is clearly learnable
- constitutive structure matters much more than raw model size
- synthetic-to-real transfer improved dramatically once we moved to trial-state and reduced-material features plus a stronger synthetic generator
- but the current branch predictor is still not trustworthy enough for solver-facing use

The later March 16, 2026 reports are especially important here: they show that the branch-only line did not clear its usefulness gate and should pivot rather than continue blind polishing.

## 1. Exact Problem Setting

The target is not a PDE solution operator. It is the local constitutive map evaluated independently at a large number of integration points:

`(engineering strain, reduced constitutive parameters) -> stress, branch`

The branch set is:

- `elastic`
- `smooth`
- `left_edge`
- `right_edge`
- `apex`

The exact implementation is in [`src/mc_surrogate/mohr_coulomb.py`](../src/mc_surrogate/mohr_coulomb.py), with supporting material reduction in [`src/mc_surrogate/materials.py`](../src/mc_surrogate/materials.py). The repo-level overview is in [README.md](../README.md), and the constitutive background is in [mohr_coulomb.md](mohr_coulomb.md).

Why the task is difficult:

- the law is piecewise, not globally smooth
- edge and apex neighborhoods are the hardest geometric regions
- the update is spectral and coaxial, so representation matters
- real solver states do not match naive synthetic strain sampling
- branch imbalance and near-interface states dominate the practical failures

## 2. Available Data and Why It Matters

The repo works with two main data regimes.

### Synthetic constitutive data

Synthetic HDF5 datasets are produced by the exact constitutive operator and usually store:

- `strain_eng`
- `stress`
- `strain_principal`
- `stress_principal`
- `eigvecs`
- `branch_id`
- `plastic_multiplier`
- `f_trial`
- `material_raw`
- `material_reduced`

See:

- [README.md](../README.md)
- [dataset_creation.md](dataset_creation.md)
- [`src/mc_surrogate/data.py`](../src/mc_surrogate/data.py)

### Real exported solver data

The repo also has a full FE export described in [full_export_inspection.md](full_export_inspection.md).

Key facts from that export:

- `554` captured constitutive calls
- `18419` quadratic tetrahedra
- `11` integration points per element
- sparse global `B`
- full displacement vectors `U`
- exact strains `E`
- exact stresses `S`
- tangents `DS` for a substantial subset of calls

This matters because it allows FE-compatible strain reconstruction and makes synthetic-to-real validation much more meaningful than purely random constitutive sampling.

## 3. Experiment Progression

### Stage A: naive or weakly structured synthetic-only classifiers

The early synthetic route relied on local perturbations of real FE states or weak synthetic coverage.

Result:

- synthetic metrics could look good
- real transfer stayed poor
- size alone did not solve the gap

This is visible in [cover_layer_branch_predictor_report_driven_iteration.md](cover_layer_branch_predictor_report_driven_iteration.md), where a larger `strain_only` model still performed badly on real data.

### Stage B: add constitutive structure to the inputs

The next meaningful improvement came from adding:

- reduced constitutive parameters
- elastic trial-stress features

This was a major correction. The task had been structurally ambiguous when reduced material variation along the strength-reduction path was ignored.

See:

- [cover_layer_branch_predictor_report_driven_iteration.md](cover_layer_branch_predictor_report_driven_iteration.md)
- [cover_layer_branch_predictor_status_for_expert_20260316.md](cover_layer_branch_predictor_status_for_expert_20260316.md)

### Stage C: principal-space hybrid synthetic generator

This was the biggest synthetic-to-real breakthrough.

The current best generator:

1. starts from real cover-layer constitutive states
2. converts strains to ordered principal coordinates
3. samples in a constitutive coordinate system tied to volumetric and principal-gap structure
4. keeps the real reduced-material row
5. mixes branch-balanced sampling, boundary-focused sampling, and tail extension
6. relabels using the exact constitutive operator

The strongest conclusion from this stage is that domain coverage was a bigger bottleneck than network capacity.

### Stage D: hierarchical branch classifiers and continuation

The successful branch-predictor family became:

- pointwise
- hierarchical
- built on structured `trial_raw_material`-style inputs

We then tried:

- longer training
- repeated Adam continuation
- repeated LBFGS continuation
- replay-bank continuation
- width inflation

These produced only incremental gains. By March 16, 2026, the evidence suggested the family was no longer obviously undertrained.

### Stage E: branch-only usefulness analysis

The later March 16 work asked a more important question than raw slice accuracy:

- are the remaining mistakes actually benign near boundaries, or solver-harmful?
- does the branch-only route clear a practical usefulness gate?

The answer was negative.

See:

- [cover_layer_branch_predictor_best_model_soft_hard_fails.md](cover_layer_branch_predictor_best_model_soft_hard_fails.md)
- [cover_layer_branch_predictor_margin_cycle.md](cover_layer_branch_predictor_margin_cycle.md)

## 4. What Worked

These findings look robust across the reports:

1. Constitutive structure helps much more than brute-force model size.
2. Reduced-material and trial-state features are essential.
3. Principal-space hybrid generation was the largest synthetic-to-real improvement.
4. Pointwise constitutive features beat raw element-context models for this task.
5. Branch-aware supervision helps both classification and stress prediction.

This is consistent with the high-level literature review in [report.md](../report.md).

## 5. What Failed or Saturated

These findings also look robust:

1. Bigger strain-only MLPs did not fix real transfer.
2. Residual-to-trial stress targets looked attractive in normalized-loss space but failed badly in actual stress space.
3. Raw element-context models were not competitive.
4. More optimizer polishing produced little practical gain.
5. Synthetic-best checkpoints and real-best checkpoints are not reliably the same model.
6. The remaining errors are concentrated exactly where the constitutive geometry is hardest.

## 6. Best Results We Actually Reached

### Branch predictor line

From [cover_layer_branch_predictor_status_for_expert_20260316.md](cover_layer_branch_predictor_status_for_expert_20260316.md):

Best stable saved model on the fixed real test slice:

- accuracy: `0.8928`
- macro recall: `0.8828`

Per-branch recall:

- elastic: `1.0000`
- smooth: `0.7500`
- left_edge: `0.9160`
- right_edge: `0.8680`
- apex: `0.8799`

Best observed internal checkpoint:

- accuracy: `0.9006`
- macro recall: `0.8915`

This means the branch task is no longer failing catastrophically. But it is not the same as saying it is solver-usable.

### Branch harm analysis

From [cover_layer_branch_predictor_best_model_soft_hard_fails.md](cover_layer_branch_predictor_best_model_soft_hard_fails.md):

- `real_val_large` accuracy / macro: `0.9045 / 0.9082`
- `real_test` accuracy / macro: `0.8981 / 0.8905`
- `real_val_large` hard-fail rate: `0.0880`
- `real_test` hard-fail rate: `0.0964`

Among wrong predictions, the large majority are not soft boundary slips:

- `real_val_large`: hard among wrong = `0.9215`
- `real_test`: hard among wrong = `0.9460`

This is one of the strongest reasons the current predictor is still unusable.

### Margin-cycle decision

From [cover_layer_branch_predictor_margin_cycle.md](cover_layer_branch_predictor_margin_cycle.md):

- selector score did not improve over baseline
- phase 1 success gate: `False`
- final decision: branch-only line did not clear the planned success gate and should pivot

### Branch experts result

From [cover_layer_branch_experts.md](cover_layer_branch_experts.md):

- oracle branch routing improved real MAE from `8.2331` to `7.3426`
- predicted routing worsened it to `8.7657`
- confidence-threshold fallback still did not beat baseline

Interpretation:

- the branch-specialized experts themselves are promising
- the branch gate remains the bottleneck

### Real-data stress regression line

From [real_sim_experiment_log.md](real_sim_experiment_log.md):

The best branch-aware raw stress model on the broad cross-check split reached:

- stress MAE: `9.27`
- stress RMSE: `25.31`
- branch accuracy: `0.907`

This was a large improvement over the earlier raw baseline:

- stress MAE: `19.11`
- stress RMSE: `42.41`

So the repo has had real progress on pointwise stress prediction, but the current branch-predictor route is still not deployment-ready.

## 7. Why The Predictor Is Still Unusable

The short answer is that good slice metrics are not enough.

The practical reasons are:

1. much of the branch-predictor story was established on a representative slice rather than the full real population
2. `smooth` remains the weakest important branch
3. the failure mass is still concentrated in hard geometric regions
4. most wrong predictions are harmful, not benign-adjacent
5. predicted branch routing is still not good enough to unlock the branch-expert upside
6. the March 16 margin-cycle report explicitly says the branch-only line should pivot

That combination makes the predictor promising but still unusable as a solver-facing decision mechanism.

## 8. Current State-of-the-Art Framing

The most relevant literature synthesis for this repo is in [report.md](../report.md). Its practical conclusion is that the right baseline is not a generic black-box regressor.

The strongest directions for this problem family are:

- invariant-aware or principal-stress feedforward surrogates
- branch-aware and return-mapping-inspired models
- architectures that preserve exact constitutive structure where possible
- thermodynamics-encoded constitutive networks
- reduced-coordinate formulations such as Haigh-Westergaard-style surrogates

The tactical memo in [report2.md](../report2.md) narrows this further. Its recommendation is:

- stop doing blind optimizer or post-train sweeps
- select checkpoints on broad plus hard sets
- target margin balance rather than only branch balance
- spend less synthetic mass in deep elastic interior
- measure induced stress error from predicted branches
- stop polishing branch-only models if the hard cases do not materially improve

This matches the March 16 branch-usefulness reports.

## 9. Questions Where Expert Guidance Is Needed

We would especially value guidance on these decisions:

1. Should we keep any branch-only line alive, or pivot immediately to a hybrid constitutive surrogate?
2. Is the right next design an exact elastic branch plus learned plastic correction, rather than a standalone branch predictor?
3. How should we define solver-facing usefulness: branch accuracy, induced stress error, tangent error, or a weighted solver-impact metric?
4. What evaluation protocol would you trust for deciding whether a model is ready for solver integration?
5. Is there a principled way to encode branch-margin geometry without overfitting to synthetic boundary tubes?
6. Would you recommend a soft mixture-of-experts, exact branch dispatch with learned gating, or a return-mapping-assisted learning route next?
7. Given the current evidence, what is the shortest credible path to a solver-usable surrogate?

## 10. Most Relevant Repo References

- [README.md](../README.md)
- [report.md](../report.md)
- [report2.md](../report2.md)
- [mohr_coulomb.md](mohr_coulomb.md)
- [full_export_inspection.md](full_export_inspection.md)
- [real_sim_experiment_log.md](real_sim_experiment_log.md)
- [cover_layer_branch_predictor_report_driven_iteration.md](cover_layer_branch_predictor_report_driven_iteration.md)
- [cover_layer_branch_predictor_status_for_expert_20260316.md](cover_layer_branch_predictor_status_for_expert_20260316.md)
- [cover_layer_branch_predictor_best_model_soft_hard_fails.md](cover_layer_branch_predictor_best_model_soft_hard_fails.md)
- [cover_layer_branch_predictor_margin_cycle.md](cover_layer_branch_predictor_margin_cycle.md)
- [cover_layer_branch_experts.md](cover_layer_branch_experts.md)
