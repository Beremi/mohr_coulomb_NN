# Cover Layer Branch Predictor: Report-Driven Iteration

## Goal

Revisit the synthetic-only `E -> branch` cover-layer classifier using ideas pulled directly from [report.md](/home/beremi/repos/mohr_coulomb_NN/report.md), then test whether better constitutive structure helps more than simply making the MLP larger.

The report’s most relevant guidance for this branch task was:

- use **trial-state / invariant-aware** inputs instead of raw Voigt strain alone
- include the **reduced material path**, because the constitutive operator moves along the strength-reduction manifold
- do not expect size alone to fix a structurally ambiguous input representation

## What Was Tested

### 1. Larger Strain-Only Baseline

Reference run:
- model card: [cover_layer_strain_branch_predictor_model_card_w2048_d6_long.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_strain_branch_predictor_model_card_w2048_d6_long.md)
- summary: [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_strain_branch_predictor_synth_only_w2048_d6_long_20260314/summary.json)

Setup:
- feature set: `strain_only`
- width/depth: `2048 x 6`
- longer schedule than before
- regenerated synthetic train set every epoch
- fixed synthetic val/test: `16384 / 16384`
- fixed real diagnostics

Result:
- synthetic test accuracy / macro recall: `0.9289 / 0.9369`
- real test accuracy / macro recall: `0.4341 / 0.4644`
- real elastic recall: `0.0000`

Conclusion:
- larger capacity improved synthetic fit only slightly
- real behavior stayed poor
- width did **not** solve the real bottleneck

### 2. Structured Trial + Material Features

Winner:
- model card: [cover_layer_branch_predictor_trial_raw_material.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_trial_raw_material.md)
- summary: [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_trial_raw_material_20260315/summary.json)

Setup:
- feature set: `trial_raw_material`
- features:
  - `asinh(E)`
  - `asinh(sigma_trial / c_bar)`
  - reduced material features `log(c_bar), atanh(sin_phi), log(G), log(K), log(lambda)`
- width/depth: `256 x 3`
- same staged LR-cycle schedule as the best recent strain-only run
- regenerated synthetic train set every epoch

Result:
- synthetic val accuracy / macro recall: `0.9850 / 0.9879`
- synthetic test accuracy / macro recall: `0.9845 / 0.9875`
- real val accuracy / macro recall: `0.6387 / 0.6369`
- real test accuracy / macro recall: `0.6861 / 0.6601`
- real elastic recall: `0.9603`
- real smooth recall: `0.1959`

Conclusion:
- this is a **major improvement**
- the missing reduced-material/trial-state information was a real problem
- the report’s guidance was right: constitutive structure helped far more than size

### 3. Principal / Invariant + Trial-Yield Variant

Tested variant:
- feature set: `trial_principal_yield_material`
- features:
  - ordered principal strains
  - trial principal stresses
  - invariant-like scalars from the trial/principal state
  - reduced material features
  - normalized trial-yield scalar

Observed behavior:
- synthetic validation plateaued around `0.79`
- synthetic test tracked it closely
- real test stayed around `0.54-0.56`

This run was stopped early because it was clearly dominated by `trial_raw_material` on both synthetic-domain mastery and real diagnostics.

Conclusion:
- this more compressed/principalized feature set was **too lossy or too hard** in the current classifier/training form
- for branch classification, the simpler `trial_raw_material` representation worked better

## Comparison

| Variant | Input Dim | Synthetic Test Acc | Synthetic Test Macro | Real Test Acc | Real Test Macro | Real Elastic Recall | Real Smooth Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strain_only` `256x3` LR-cycle | `6` | `0.9264` | `0.9346` | `0.4672` | `0.5019` | `0.0000` | `0.4173` |
| `strain_only` `2048x6` long | `6` | `0.9289` | `0.9369` | `0.4341` | `0.4644` | `0.0000` | `0.2569` |
| `trial_raw_material` `256x3` | `17` | `0.9845` | `0.9875` | `0.6861` | `0.6601` | `0.9603` | `0.1959` |

## Visuals

Winner history:

![trial raw material history](../experiment_runs/real_sim/cover_layer_branch_predictor_trial_raw_material_20260315/training_history.png)

Winner confusions:

![trial raw material confusions](../experiment_runs/real_sim/cover_layer_branch_predictor_trial_raw_material_20260315/confusions.png)

Strain-only baseline history:

![strain-only history](../experiment_runs/real_sim/cover_layer_strain_branch_predictor_synth_only_w256_d3_lrcycle_20260314/training_history.png)

## What We Learned

1. The earlier “one material only means no material features” simplification was too aggressive for this dataset.
   The raw material family is fixed to cover layer, but the **reduced constitutive state varies along the strength-reduction path**. That matters for branch prediction.

2. The branch map is not just a function of raw strain magnitude in practice.
   Adding exact elastic trial-stress features and reduced material features made the classifier much more identifiable.

3. Bigger MLPs were not the answer.
   `2048 x 6` did not fix the real branch problem.

4. The remaining weak branch is now `smooth`.
   The winner fixed elastic almost completely and improved edge/apex substantially, but `smooth` is still too low on real data.

## Recommended Next Step

Stay with the winning representation:
- `trial_raw_material`

Do **not** go back to raw strain-only size sweeps.

The next experiment should target the remaining `smooth` weakness directly:

1. keep the `trial_raw_material` features
2. add a **two-stage hierarchical classifier**
   - stage A: `elastic` vs `plastic`
   - stage B: `smooth / left_edge / right_edge / apex` on plastic points
3. oversample `smooth` and `smooth/right_edge` boundary regions in the synthetic generator
4. keep the same LR-cycle staged training recipe

That is the cleanest continuation from this iteration.
