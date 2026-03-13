# Real Simulation Fit Ideas

This note is the working hypothesis list for improving fit on constitutive states sampled from `constitutive_problem_3D.h5`.

It is intentionally opinionated. The goal is not to enumerate every possible model idea, but to rank the ones most likely to improve the real-data fit based on the completed experiments in:

- `docs/real_sim_validation.md`
- `docs/real_sim_retraining.md`

## Current Position

Best completed checkpoint after the latest iteration:

- `experiment_runs/real_sim/iter_20260311/raw_branch_w512_d6/best.pt`

Best measured stress accuracy so far:

- real sampled `256` test split: `stress_mae = 9.20`
- broader real sampled `512` test split: `stress_mae = 9.27`

What is already clear:

- synthetic-only training is not good enough for the real slope-simulation states
- real-data retraining matters more than extra synthetic coverage
- direct raw-stress models currently beat principal-stress and trial-stress-residual models on the real export
- the hardest remaining errors are concentrated in high-compression `smooth`, `left_edge`, and `right_edge` states
- those failures are usually stress-magnitude underprediction, not random noise

What the latest iteration established:

- adding an auxiliary branch head to the raw model was the highest-value change by far
- the branch-aware raw model roughly halved the real-data MAE (`~19.1 -> ~9.27`) on the broader cross-check dataset
- simple tail weighting did not beat the unweighted branch-aware raw model
- a larger branch-aware model (`768x8`) did not improve the overall fit enough to justify replacing the smaller winner
- the residual-to-trial target looked promising in normalized loss but failed badly in actual stress metrics, so it is not the right target in the current formulation

## Most Likely Winning Direction

### 1. Raw model with auxiliary branch head

Priority: completed and validated

Why:

- the plain raw model is already the best stress regressor
- the remaining errors are branch-structured rather than isotropic
- adding a branch head should help the network separate `smooth` versus edge-return regimes without forcing a principal-stress output parameterization

Observed effect:

- this was the winning intervention
- cross-dataset stress MAE improved from `19.11` to `9.27`
- highest stress quartile MAE improved from `34.44` to `15.21`
- branch accuracy reached `0.907`

Winning variant:

- `raw_branch_w512_d6`
- same optimizer schedule as the current best raw model

### 2. Trial-stress residual target

Priority: tested, not recommended in the current form

Why:

- the constitutive law is naturally an elastic trial state plus plastic correction
- the current worst misses are large plastic corrections on top of large trial stresses
- asking the net to predict the full stress may be wasting capacity on an easy elastic component

Preferred formulation:

- inputs: current `trial_raw` features
- target: `sigma - sigma_trial`
- first without any extra transform
- if needed, then test `asinh((sigma - sigma_trial) / s)`

Observed effect:

- normalized target loss looked excellent
- actual stress metrics were much worse than the direct raw-branch model
- this confirms that low residual-space loss was not aligned with the actual stress objective here

### 3. Combine both: branch-aware residual model

Priority: tested, not recommended

Why:

- if the raw model needs regime separation and the residual target reduces target scale, these two ideas should reinforce each other

Observed effect:

- the auxiliary branch head helped the direct raw-stress model
- it did not help the residual target model
- the combined model peaked early and stayed far behind the direct raw-branch winner

## Likely Secondary Improvements

### 4. Tail-weighted stress loss

Priority: tested once, currently lower priority

Why:

- current global MAE hides the fact that the worst errors are concentrated in the top stress quartile
- the end use cares more about those hard plastic states than about reducing already-small elastic errors

Candidate weighting:

- per-sample weight from `1 + alpha * log1p(||sigma|| / s)`
- tune `alpha` mildly rather than aggressively

Observed effect:

- the weighted run lowered the primary-split max-abs error (`987 -> 828`)
- but it worsened average MAE and worsened the high-stress tail on the broader cross-check split
- this does not look like the right next move without a more careful weighting design

### 5. Hard-example mining from the real export

Priority: best remaining local-fit idea if we keep pushing

Why:

- we already know the approximate shape of the bad region
- reusing the same worst states in a second-stage fine-tune is cheaper than regenerating entirely new synthetic distributions

Candidate strategy:

- collect top-error real samples from the best current run
- oversample them 2x to 4x in a fine-tuning dataset

Expected effect:

- reduced extreme outliers
- modest RMSE and max-abs improvement

## Ideas That Look Less Promising Right Now

### Principal-only architectures

Why lower priority:

- on the real export they consistently trail the best raw model on stress accuracy

### Custom sign-aware output layers

Why lower priority:

- `asinh` already gives the right signed compression behavior if we need it
- custom output nonlinearities are more invasive and less interpretable than target transforms

### Simply making the network larger

Why lower priority:

- the latest `raw_branch_w768_d8` run did not improve the winning `raw_branch_w512_d6` checkpoint on the broad cross-check split
- the current evidence points to targeted hard-case handling rather than generic extra capacity

## Good-Enough Bar

I do not consider the fit good enough until all of the following are true:

1. local real-data metrics materially beat the current `~19` MAE baseline
2. high-stress-tail error drops clearly, not just average error
3. max-abs error shrinks enough that the extreme states are no longer obviously unreliable
4. the surrogate behaves well when inserted into the actual limit-analysis solve
5. the final safety-factor / collapse result stays acceptably close to the exact constitutive law

## Recommended Next Step

At this point I would stop the purely local architecture sweep and do one of these instead:

1. integrate `raw_branch_w512_d6` into the actual solver path and measure whether the limit-analysis result stays close enough
2. if more local improvement is still needed before solver integration, try hard-example mining around the worst `left_edge` and `right_edge` states rather than larger generic models
