# Cover Layer Sampling Expert-Opinion Test

## Question

Does the new expert opinion in [sampling.md](/home/beremi/repos/mohr_coulomb_NN/sampling.md) help the cover-layer branch predictor if we keep the current best classifier setup and change only the synthetic sampling strategy?

## Short Answer

Yes. It helped a lot.

Using the same best classifier configuration:

- feature set: `trial_raw_material`
- model: hierarchical branch classifier
- width/depth: `256 x 3`
- same LR cycles, batch ramp, and LBFGS tail

switching from the old FE-seed local-noise generator to an expert-inspired principal-space hybrid generator improved real diagnostics dramatically:

- real test accuracy: `0.7058 -> 0.8865`
- real test macro recall: `0.6843 -> 0.8750`
- real `smooth` recall: `0.2392 -> 0.7490`

This is the strongest evidence so far that **domain coverage really was the main bottleneck**.

## What From `sampling.md` Was Implemented

The expert opinion recommended moving the generator from FE perturbations toward **constitutive principal coordinates** and explicitly targeting the dangerous `smooth / right_edge` geometry.

Implemented subset:

1. **Principal-space real-like resampler**
   - built from real cover-layer pointwise states
   - sampled in constitutive coordinates:
     - `(K I1) / c_bar`
     - `log(G Δ12 / c_bar)`
     - `log(G Δ23 / c_bar)`
   - kept the real reduced material row
   - kept a real eigenvector frame from the seed state
   - reconstructed full engineering strain after sampling

2. **Analytic `smooth / right_edge` boundary tube**
   - used the exact principal-space branch geometry
   - solved analytically for the `smooth_right` surface scale
   - jittered around that surface to create a tube across both sides

3. **Tail extension**
   - started from real high-tail states
   - scaled hydrostatic and deviatoric parts separately in principal space

4. **Same classifier and training loop**
   - kept the current best branch model and training recipe structure
   - only changed the sampler

What was **not** implemented from the expert note:

- full “use all `generator_fit` calls” support model
- symmetric `smooth_left` / apex tube family
- FE realism filters beyond the current pointwise use case
- principal-stress regression surrogate

So this was a **partial but faithful first test**, not the full proposal.

## Files

New run:

- model card: [cover_layer_branch_predictor_expert_principal.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_expert_principal.md)
- summary: [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_20260315/summary.json)
- benchmark summary: [benchmark_summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_20260315/benchmark_summary.json)

Previous best baseline:

- model card: [cover_layer_branch_predictor_hierarchical_smooth_focus.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_hierarchical_smooth_focus.md)
- summary: [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_hierarchical_smooth_focus_20260315/summary.json)

Implementation:

- trainer: [train_cover_layer_strain_branch_predictor_synth_only.py](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/train_cover_layer_strain_branch_predictor_synth_only.py)
- principal geometry helpers: [branch_geometry.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/branch_geometry.py)
- expert-inspired sampler: [principal_branch_generation.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/principal_branch_generation.py)

## Experimental Setup

### Baseline being challenged

Baseline model:

- feature set: `trial_raw_material`
- model type: `hierarchical`
- recipe: `smooth_focus`
- generator: FE-seed local noise

Baseline real test:

- accuracy: `0.7058`
- macro recall: `0.6843`

### New generator recipe

Generator kind:

- `principal_hybrid`

Cycle mixture:

- easy:
  - `70%` branch-balanced principal resampling
  - `30%` smooth-focused principal resampling
- match:
  - `60%` branch-balanced principal resampling
  - `25%` analytic `smooth_right` boundary tube
  - `15%` tail extension
- coverage:
  - `50%` branch-balanced principal resampling
  - `30%` analytic `smooth_right` boundary tube
  - `20%` tail extension
- hard:
  - `35%` smooth-edge principal resampling
  - `40%` analytic `smooth_right` boundary tube
  - `25%` tail extension

Everything else was kept aligned with the previous best run.

## Results

### Synthetic benchmark

Old best:

- synthetic test accuracy: `0.9832`
- synthetic test macro recall: `0.9870`

New expert-inspired generator:

- synthetic test accuracy: `0.9572`
- synthetic test macro recall: `0.9576`

Interpretation:

- the synthetic benchmark became harder
- synthetic accuracy dropped
- but the model still learned the new synthetic domain well enough

This is important because the real metrics went up sharply while the synthetic benchmark became more demanding.

### Real diagnostics

Old best:

- real test accuracy: `0.7058`
- real test macro recall: `0.6843`

New expert-inspired generator:

- real test accuracy: `0.8865`
- real test macro recall: `0.8750`

Absolute gain:

- accuracy: `+0.1808`
- macro recall: `+0.1907`

### Real per-branch recall

Old best:

- elastic: `0.9603`
- smooth: `0.2392`
- left_edge: `0.6870`
- right_edge: `0.7327`
- apex: `0.8022`

New expert-inspired generator:

- elastic: `1.0000`
- smooth: `0.7490`
- left_edge: `0.9101`
- right_edge: `0.8289`
- apex: `0.8871`

Absolute gain:

- elastic: `+0.0397`
- smooth: `+0.5098`
- left_edge: `+0.2231`
- right_edge: `+0.0962`
- apex: `+0.0849`

This is the most important table in the report. The generator change did not just improve one branch. It improved **all** real branches, and it massively improved `smooth`.

## Coverage Comparison

### Branch-frequency mismatch

Real test branch fractions:

- elastic: `0.2550`
- smooth: `0.1804`
- left_edge: `0.2093`
- right_edge: `0.1587`
- apex: `0.1966`

Old generator synthetic test:

- elastic: `0.0320`
- smooth: `0.3049`
- left_edge: `0.2721`
- right_edge: `0.0815`
- apex: `0.3096`

New generator synthetic test:

- elastic: `0.0997`
- smooth: `0.2930`
- left_edge: `0.1088`
- right_edge: `0.1642`
- apex: `0.3344`

What improved:

- `elastic` moved closer to real
- `right_edge` moved much closer to real
- total branch-frequency L1 gap improved from `0.6005` to `0.5117`

What is still off:

- `left_edge` is now too low
- `apex` is still too high
- `smooth` is still too high

So the new generator is **not** “perfectly matched” in raw branch fractions.

### Strain-tail coverage

Real test strain-norm p95:

- `23.1378`

Old generator synthetic test p95:

- `14.8182`

New generator synthetic test p95:

- `21.8003`

This is a strong improvement.

Gap to real:

- old: `8.3196`
- new: `1.3375`

So the expert-inspired generator fixed a major part of the upper-tail undercoverage.

### Smooth/right boundary emphasis

Real test fraction with `|m_s_right| < 0.05`:

- `0.1740`

Old generator:

- `0.0908`

New generator:

- `0.3545`

Interpretation:

- the old generator undercovered the `smooth/right` boundary tube
- the new generator deliberately overcovered it

That overcoverage appears to have been useful. It is likely one of the main reasons `smooth` recall jumped so much.

## Interpretation

The expert note absolutely helped.

The strongest conclusions are:

1. **Moving the generator into constitutive principal coordinates was the right move.**
   The result improved sharply without changing the branch model architecture.

2. **Targeting the `smooth/right` geometry explicitly was useful.**
   The biggest recall gain was exactly on `smooth`, which was our hardest branch.

3. **Upper-tail coverage mattered.**
   The new generator matches the real strain tail much better, and that coincides with the much stronger real performance.

4. **Raw branch-frequency matching is not the whole story.**
   The new synthetic branch fractions are still imperfect, yet real performance improved massively.
   That means the previous main issue was not only “wrong branch counts.” It was also “wrong local constitutive geometry.”

## What This Means For Next Iteration

This generator family is clearly worth keeping.

The next step should be a controlled refinement, not a reset:

- keep the principal-space hybrid generator
- keep the current hierarchical `trial_raw_material` classifier
- add a **left-side corrective component** to the generator:
  - `smooth_left` tube
  - maybe `left_edge`-focused real-like resampling
- then retest whether we can close the remaining branch-frequency mismatch without giving back the `smooth` gains

## Bottom Line

Yes, the expert opinion in [sampling.md](/home/beremi/repos/mohr_coulomb_NN/sampling.md) helped in a materially important way.

The first partial implementation already moved the model from:

- good but still limited real performance

to:

- very strong real branch prediction on the current real diagnostic split

So this is now the best direction we have for generator design.
