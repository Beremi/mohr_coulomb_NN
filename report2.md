Yes. The next move is **not** another optimizer/post-train sweep. It is one bounded cycle that tightens evaluation, fixes the remaining generator geometry, and then decides whether branch-only work is done.

1. **Replace slice-based checkpoint selection with a real broad-set + hard-set selector.**
   Your current fixed 4-call slice was good enough to iterate, but it is now too small for trustworthy selection. Keep the real test set frozen, and evaluate checkpoints on the **full real validation population** or at least a much larger stable panel. Then split that validation into:

* a **broad** set: all normal real-val points
* a **hard** set: near branch surfaces, near repeated principal values, and upper-tail states

The constitutive docs already define the branch geometry through the ordered-principal thresholds `γ_sl, γ_sr, γ_la, γ_ra` and the candidate plastic multipliers that decide admissibility, so you can score points by **distance to decision surfaces**, not just branch ID. That is the right selection basis now. ([GitHub][1])

A good checkpoint score would be something like

`score = 0.5 * broad_macro + 0.3 * hard_macro + 0.2 * smooth_hard`

where `smooth_hard` is recall on the hard subset for true `smooth`.

2. **Change the generator target from branch balance to margin balance.**
   Branch balancing got you past the big failure mode. The remaining gap is almost certainly about **where inside each branch** you sample, especially near interfaces. The docs show that left/right and edge/apex geometry are paired analytically, so the next generator should build explicit tubes around **all** relevant surfaces, not just the smooth/right one. ([GitHub][1])

I would make the next generator mixture:

* `35%` real-like principal-space resampling
* `35%` explicit boundary tubes

  * `8%` smooth/right
  * `8%` smooth/left
  * `7%` right/apex
  * `7%` left/apex
  * `5%` near-yield
* `15%` upper-tail extension
* `10%` repeated-eigenvalue / small-gap states
* `5%` canonical loading paths

And I would fit generator hyperparameters against **generator_fit** distributions of:
`m_yield, m_smooth_left, m_smooth_right, m_left_apex, m_right_apex, I1, Δ12, Δ23`, plus your tail norm.
That is a much better calibration target than raw branch frequencies.

3. **Stop spending synthetic mass on deep elastic interior.**
   On your fixed slice, elastic is already effectively solved. The exact constitutive law has a clean elastic condition `f_tr <= 0`, and your own literature summary explicitly recommends keeping the elastic branch exact to reduce learning burden. ([GitHub][1])

So for the next cycle:

* keep a **thin yield tube**
* keep some elastic near-boundary points
* drastically reduce deep elastic interior mass
* or just make elastic exact in downstream use

4. **Do not jump to raw element-context models yet.**
   Everything in your results says the winning abstraction is still the **pointwise constitutive representation**. The repo README frames the target as a local pointwise map, and the report’s recommended baseline is a principal/invariant, branch-aware feedforward model rather than a raw-element or recurrent formulation. ([GitHub][2])

If you want to test context, do the cheap version first:
for each IP, concatenate **element-level summaries across the 11 points**, such as min/mean/max of
`f_tr, I1, Δ12, Δ23, m_smooth_left, m_smooth_right, m_left_apex, m_right_apex`.

That gives you element pattern information **without** throwing away the successful constitutive-feature formulation.

5. **Run the most informative probe before doing more branch-model work.**
   Use the predicted branch to dispatch into the **exact closed-form branch formula** and compute stress from the exact branch expression. The docs make clear that the update is piecewise with explicit formulas on elastic, smooth, left edge, right edge, and apex. ([GitHub][1])

This probe tells you something extremely valuable:

* if branch-predicted + exact-formula stress is already robust, branch-only work is close to useful
* if it is still brittle, the remaining issue is not “continuous within-branch regression,” it is still branch mistakes near critical surfaces

That is the cleanest way to measure how much value remains in the branch project.

6. **Put a hard stop on branch-only polishing.**
   I would give the branch predictor **one more serious cycle** with:

* full real-val selection
* margin-conditioned generator v2
* optional 11-IP summary context

If that does **not** improve full real-val macro recall by about 1 point, or does not materially lift the hard `smooth`/edge/apex buckets, stop polishing branch-only and pivot.

The pivot should be to the model family your repo/report already point toward:

* **exact elastic branch**
* **principal-space plastic surrogate**
* **auxiliary branch head**
* output = updated principal stresses or principal stress correction
* reconstruct global stress analytically
  That matches both the README recommendation and the literature-summary baseline much better than more branch-only tuning. ([GitHub][2])

So the immediate work package I would choose is:

1. build geometry-aware real validation panels and a real-val checkpoint selector
2. build generator v2 with symmetric left/right and edge/apex boundary tubes
3. run branch-predicted + exact-formula stress as the decision experiment

For the branch-surface margins and analytic surface samplers, use the helper here: [branch_geometry_prototype.py](sandbox:/mnt/data/branch_geometry_prototype.py)

[1]: https://raw.githubusercontent.com/Beremi/mohr_coulomb_NN/main/docs/mohr_coulomb.md "raw.githubusercontent.com"
[2]: https://github.com/Beremi/mohr_coulomb_NN "GitHub - Beremi/mohr_coulomb_NN · GitHub"


Yes — **provided you still have the underlying** `(E, reduced_material)` for those failed points.

For your Mohr–Coulomb update, a wrong **adjacent** branch can easily give almost the same `S`, because the local constitutive map is **piecewise-defined but stress-continuous across the branch interfaces**. Your implementation switches branches using `f_trial`, `γ_sl`, `γ_sr`, `γ_la`, `γ_ra`, and the candidate plastic multipliers `λ_s`, `λ_l`, `λ_r`, `λ_a`; the neighboring branch formulas meet on those interfaces. ([GitHub][1])

The important adjacent pairs are:

* `elastic ↔ smooth`: at `f_trial = 0`, the smooth plastic multiplier is zero, so the smooth formula reduces exactly to the elastic trial stress. ([GitHub][1])
* `smooth ↔ left_edge`: on `λ_s = γ_sl`, the smooth return collapses to `σ1 = σ2`, which is the left-edge condition. ([GitHub][1])
* `smooth ↔ right_edge`: on `λ_s = γ_sr`, the smooth return collapses to `σ2 = σ3`, which is the right-edge condition. ([GitHub][1])
* `left_edge ↔ apex` and `right_edge ↔ apex`: on `λ_l = γ_la` or `λ_r = γ_ra`, the edge pair collapses to triple equality, and the common stress is the apex value `c_bar / (2 sin φ)`. ([GitHub][1])

So for **stress** `S`, some branch failures are genuinely benign. But for the **tangent** `DS`, they may still be harmful, because the docs explicitly describe the operator as piecewise smooth with kinks at the elastic/plastic, smooth/edge, and edge/apex transitions, and the tangent is branch-dependent. In other words: **small stress error does not imply small tangent error**. ([GitHub][1])

## How to check this on your current fails

Do **not** use branch accuracy alone. Add a second evaluation layer: **induced stress error from the predicted branch**.

### 1. For each failed point, compute “forced-branch” candidate stresses

Given one sample `x = (E, c_bar, sin_phi, G, K, lambda)`:

* compute the exact result `(σ*, b*)` with the normal constitutive update
* also compute the closed-form stress for **each branch formula**

  * `σ_el(x)`
  * `σ_sm(x)`
  * `σ_le(x)`
  * `σ_re(x)`
  * `σ_ap(x)`

Then if the classifier predicts `b_hat`, define

`e_sigma_argmax(x) = ||σ^(b_hat)(x) - σ*(x)||`

That is the quantity that tells you whether the branch fail is harmless or not.

Because your update is coaxial in principal space, do this **first in ordered principal stresses**, not in full Voigt stress. That is more stable near `left_edge / right_edge / apex`, where repeated principal values make eigenvector directions non-unique. The docs explicitly note the coaxial principal-space structure and branch formulas in principal form. ([GitHub][1])

A good normalized version is:

`rel_e_sigma = ||σ_pr^(b_hat) - σ_pr^*||_2 / (||σ_pr^*||_2 + c_bar + 1e-12)`

Then call a fail:

* **benign** if `b_hat != b*` and `rel_e_sigma <= τ`
* **harmful** if `b_hat != b*` and `rel_e_sigma > τ`

Start with `τ = 1e-2` and then tune it to what actually matters in your solver.

### 2. Tag each fail by the relevant interface margin

For each sample, compute a signed distance-to-interface surrogate:

* `elastic ↔ smooth`: `m_y = |f_trial|`
* `smooth ↔ left_edge`: `m_sl = |γ_sl - λ_s|`
* `smooth ↔ right_edge`: `m_sr = |γ_sr - λ_s|`
* `left_edge ↔ apex`: `m_la = |γ_la - λ_l|`
* `right_edge ↔ apex`: `m_ra = |γ_ra - λ_r|`

Better still, normalize them:

`m_norm(a,b) = |a-b| / (|a| + |b| + eps)`

Then for each fail pair, use the corresponding margin:

* true/pred in `{smooth,left_edge}` → use `m_sl`
* true/pred in `{smooth,right_edge}` → use `m_sr`
* true/pred in `{left_edge,apex}` → use `m_la`
* true/pred in `{right_edge,apex}` → use `m_ra`
* true/pred in `{elastic,smooth}` → use normalized `|f_trial|`

If many fails have **tiny margin** and **tiny induced stress error**, then your branch metric is overstating the practical damage. If they have tiny margin but still sizable stress error, something is off in the local geometry or in how you are forcing the wrong branch.

### 3. Replace the plain confusion matrix with a confusion-by-harm table

For each `(true_branch, predicted_branch)` pair, report:

* count
* median `rel_e_sigma`
* p95 `rel_e_sigma`
* median relevant margin

This will tell you very quickly whether, say, `smooth -> right_edge` is mostly a harmless boundary slip or a real constitutive miss.

A very useful summary is:

* overall branch macro recall
* benign fail rate
* harmful fail rate
* harmful adjacent fail rate
* harmful non-adjacent fail rate

That split is much more meaningful than raw accuracy once you care about `S`.

## How to use this during training

If the deployment target is **stress**, not branch purity, then the training objective should reflect that.

### Best practical change: add a state-dependent branch cost

For each training sample, build a 5-way cost vector

`C_b(x) = rel_err(σ_pr^(b)(x), σ_pr^*(x))`

Then train with

```python
p = softmax(logits)
loss = CE(logits, true_branch) + alpha * sum_b p[b] * clip(C_b, 0, Cmax)
```

This does two useful things:

* it still teaches the exact branch
* but it penalizes **harmful** wrong branches more than **benign** adjacent ones

That is much better than treating all branch mistakes as equally bad.

### Also change checkpoint selection

Instead of selecting by synthetic branch score alone, monitor on validation:

* branch macro recall
* induced stress error from argmax branch
* harmful fail rate

A checkpoint with slightly worse branch accuracy but much lower harmful fail rate is often the better model for `S`.

## One diagnostic that is especially informative

Compute both:

* `argmax-dispatched` stress error
  `σ_argmax = σ^(argmax p)(x)`
* `soft branch-mixture` stress error
  `σ_soft = sum_b p_b σ^(b)(x)`

in **principal stress space**.

If `σ_soft` is much better than `σ_argmax`, your model already “knows” it is near a boundary and the problem is the hard argmax decision, not the representation. That is a strong sign that many branch fails are benign-adjacent.

## What this means for data generation

If your real goal is `S`, then **exactly on the interface** is not the most informative place to oversample, because adjacent branches coincide there in stress.

The informative region for stress is a **thin two-sided annulus around the interface**:

* close enough that branch confusion is plausible
* far enough that the wrong adjacent branch produces a nontrivial stress gap

So for `S` training, target small-but-nonzero margin bands such as:

* `0.01 <= m <= 0.05`
* `0.05 <= m <= 0.10`

For `DS`, the opposite is true: the kink neighborhood itself matters more.

## Bottom line

Yes: **some of your branch fails can absolutely be adjacent-branch mistakes with almost identical stress**.

The right way to investigate is:

1. compute candidate stress for **every branch formula**
2. measure **induced stress error** for the predicted branch
3. stratify by **adjacent pair** and **distance to the shared interface**
4. use that harm signal in both **training loss** and **checkpoint selection**

If you do only one thing next, make it a `candidate_principal_stresses_3d(...)` helper beside `constitutive_update_3d(...)` and start reporting **harmful fail rate**, not just branch recall.

[1]: https://raw.githubusercontent.com/Beremi/mohr_coulomb_NN/main/docs/mohr_coulomb.md "raw.githubusercontent.com"
