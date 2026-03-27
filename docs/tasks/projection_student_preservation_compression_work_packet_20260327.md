# Projection-Student Preservation And Compression Work Packet

Date: `2026-03-27`

Repository: `/home/beremi/repos/mohr_coulomb_NN`

Source note:
This document is the normalized follow-on task memo created from the March 27 continuation brief that followed the executed March 26 projection-student reopen packet. It is a planning and execution brief for the next live packet, not an execution report.

Relationship to the previous packet:

- previous executed task memo:
  [`docs/tasks/projection_student_work_packet_20260326.md`](/home/beremi/repos/mohr_coulomb_NN/docs/tasks/projection_student_work_packet_20260326.md)
- previous execution report:
  [`docs/executions/projection_student_work_packet_20260326.md`](/home/beremi/repos/mohr_coulomb_NN/docs/executions/projection_student_work_packet_20260326.md)

This last packet changes the picture a lot.

You are still **not close to a small, fully learned, exact-admissible MC surrogate**, but you are now **much closer to a solver-credible projected surrogate** than the packet-family closeout made it seem.

The key result in your report is Phase 0, not Phase 2. In Phase 0, the exact projection layer took the strong March 24 raw teacher from validation `5.771 / 6.950 / 76.399 / 1.00e-01` to `5.877 / 7.063 / 76.399 / 2.20e-08`, and similarly on test from `5.982 / 7.242 / 79.456 / 1.06e-01` to `6.084 / 7.344 / 79.956 / 2.25e-08`. That is a tiny stress penalty for a massive admissibility gain. So the scientific bottleneck is no longer “how do we encode admissibility in the output space?” It is “how do we preserve a very good projected teacher while compressing it into a smaller student?” That is a completely different problem.

That is also the direction most consistent with recent constitutive-ML work. The field has been moving away from unconstrained black-box stress maps and toward architectures with more embedded structure, reduced coordinates, implicit/projection layers, and automatic differentiation for tangents. Reviews now frame “integrated physical knowledge” as the main frontier rather than pure approximation capacity; reduced Haigh–Westergaard-style coordinates have improved interpolation/extrapolation relative to plain principal-stress training; implicit-layer elastoplastic models were introduced specifically to cope with the non-smooth yield transition; and AD is now viewed as a practical route for constitutive tangents and code maintainability. ([Springer Link][1])

So my advice is to **reframe the live line immediately**:

## New state of the program

The packet2/3/4 and Program X family should remain closed.

The **projection-student route is now the only scientifically live direct-replacement route**.

But even inside that route, the right framing is not “find a new constitutive representation.” It is:

**preserve the projected teacher first, compress second, open `DS` only after preservation is demonstrated.**

That is the main shift I would make.

## What the newest results actually mean

Your report supports four conclusions.

First, the **projection layer itself worked**. That is no longer hypothetical. It is exact enough numerically, drives yield violation to machine-noise scale, and preserves forward stress very well.

Second, the current student did **not** fail because the projection idea is wrong. It failed because the current student is too aggressive a compression and too weakly teacher-aligned. A `29.85k` student getting to `27.31 / 29.60 / 250.70` while staying admissible is not a dead route; it is an underpowered compression baseline.

Third, the current student already **beats all prior admissible packet families** on forward stress while keeping yield near zero. That matters. The line is now better than packet2, packet3 oracle, and packet4, and incomparably better than Program X, all under the same canonical benchmark.

Fourth, the report’s decision “continue_projection_student” is correct, but the next phase should be much more specific than “keep training students.” It should be a **teacher-preservation and compression program**.

## What I would do next

I would split the work into two tracks that use the same projection operator.

## Track A: promote the projected teacher as the current shadow candidate

Do not wait for a small student before learning whether the projection route is solver-credible.

The projected teacher already has forward metrics in the regime that used to define your credible bar. So treat it as **candidate zero** and test the two things that matter next:

1. **autodiff tangent viability**
2. **FE shadow behavior**, if you have even a limited harness

This is the fastest way to learn whether the route is worth serious continuation.

### A1. Add a projected-teacher `DS` probe

Use the current teacher checkpoint plus the exact projection layer. Do not retrain anything yet.

Evaluate on `ds_valid_val` and `ds_valid_test`:

* directional tangent error `||Jv_pred - DS_true v||`
* cosine similarity between `Jv_pred` and `DS_true v`
* distribution split by true plastic branch
* instability rate near branch boundaries
* exact-projection mode versus softmin-projection mode

The reason to include both exact and softmin modes is not forward stress. It is tangent smoothness. Exact mode is the truthful deployment map; softmin mode may produce better-behaved gradients near candidate switches. Implicit/projection-style constitutive layers are attractive precisely because they provide a differentiable structure around difficult elastoplastic transitions. ([SSRN][2])

### Expected result

I would expect:

* exact mode to preserve the excellent forward stress numbers you already measured
* directional `Jv` to be usable on most rows but noisy near candidate boundaries
* softmin mode to smooth `Jv` noticeably, with a small forward-stress penalty

If softmin raises validation broad/hard plastic MAE by less than about `0.3–0.5` while materially improving tangent stability, that is a strong sign that `DS` through the projection route is viable.

### A2. Run a tiny FE shadow test if you can

Even a very small shadow harness is worth it now.

Use the projected teacher only, not a student, and compare against the exact constitutive routine on a fixed benchmark set:

* Newton iteration count
* convergence failures
* load-step drift
* displacement/stress field differences
* whether projection-mode choice changes robustness

### Expected result

Because the projected teacher forward metrics are already so close to the raw teacher while being fully admissible, I would expect the FE shadow to behave much more like the raw baseline than like packet2/4. If it does, that immediately promotes the projection route from “interesting reopen” to “credible solver-facing path.”

## Track B: treat the student as a compression problem, not a representation problem

This is the part I think your current packet is under-optimizing.

Right now the student is learning from exact stress through a projection layer. That is not enough. The student should be trained to mimic a **specific good preimage** of the projection map: the teacher.

There are many provisional stresses that project to a similar admissible final stress. The raw teacher gives you one especially useful preimage, because you already know it projects well.

So the next student phase should become **projection distillation**.

## B1. Build a teacher-projection cache

For every training row, cache:

* exact trial principal stress
* raw teacher predicted principal stress before projection
* projected teacher principal stress
* exact stress
* projection displacement vector
* selected projection candidate (`pass_through`, `smooth`, `left`, `right`, `apex`)
* teacher plastic branch probabilities if available

If the larger full export still exists beyond the `512`-per-call grouped dataset, run the teacher plus projection on **much more real data** and use that as extra pseudo-labeled compression data. This is one of the highest-value moves available to you right now, because teacher inference is cheap and the task is now compression, not discovery.

## B2. Run the critical control: same-capacity teacher-preservation model

Before trying to make the model smaller, verify that the projection-student codepath can preserve the teacher at all.

Take a model with the **same feature richness and roughly the same capacity class as the raw teacher**, append the projection layer, and train it to preserve the projected-teacher mapping.

Do not optimize for small size in this control. Optimize for fidelity.

### Loss I would use

Let

* `σ̃_T` = raw teacher principal stress before projection
* `σ̂_T = Π(σ̃_T)` = projected teacher principal stress
* `σ*` = exact principal stress
* `σ̃_S` = student principal stress before projection
* `σ̂_S = Π(σ̃_S)` = projected student principal stress

Then use

```text
L = 1.0 * Huber(σ̂_S, σ*)
  + 0.75 * Huber(σ̂_S, σ̂_T)
  + 0.50 * Huber(σ̃_S, σ̃_T)
  + 0.25 * Huber((σ̂_S - σ̃_S), (σ̂_T - σ̃_T))
  + 0.10 * CE(candidate_S, candidate_T)
```

with extra weighting on hard rows and high-displacement rows.

This is the most important change in the whole next phase. The current student is being asked to solve the forward problem too directly. It should first be taught the teacher’s provisional stress geometry.

### Expected result

This control should get much closer to the projected teacher than your current `30k–40k` students. If it does not, then the problem is not compression; it is a codepath/feature mismatch in the student formulation.

That would be a critical diagnostic.

## B3. Only after that, do structured compression

Once the same-capacity projected student can reproduce the projected teacher well, compress in stages.

I would use three compression tiers:

* **Tier 1**: `100k–150k` params
* **Tier 2**: `50k–100k` params
* **Tier 3**: `20k–50k` params

Do not go back to `10k–40k` only until Tier 1 works. Your current medium student already showed that the route is alive, but it also showed that compressing too hard, too early, leaves a huge gap.

### Sampling strategy

Weight the training distribution toward the rows that matter most for projection and tails:

* left/right edge rows
* top displacement quantiles from the projected teacher audit
* hard-panel plastic rows

Your own Phase 0 displacement summary already identifies where the projection is doing the most work: left and right edges are hardest, with the largest displacements and highest post-projection branchwise MAE.

### Expected result

If this route is genuinely viable, Tier 1 should be able to get validation into something like:

* broad plastic MAE `<= 12–15`
* hard plastic MAE `<= 14–18`
* hard p95 principal `<= 120–160`
* yield violation p95 `<= 1e-6`

If Tier 1 cannot reach that neighborhood, the compression route is much weaker than Phase 0 made it look.

## B4. Only then compress further with pruning / quantization

Once a teacher-like projected student exists, then compress it.

Do not use architecture search as the first compression mechanism. Use:

* structured pruning
* weight clustering
* quantization
* possibly shallow distillation into a smaller student

That sequence is much more likely to preserve the forward map than trying to discover a tiny model from scratch. It also lines up with the broader mechanics-informed trend: first get the physically integrated architecture to work, then compress it. ([Springer Link][1])

## Where `DS` should actually open

I would no longer tie `DS` only to the tiny student.

There are now two relevant `DS` gates.

### Gate 1: projected teacher `DS` probe

Open immediately. This is a diagnostic gate, not a deployment gate.

### Gate 2: projected student `DS` development

Open only if one projected student reaches about:

* broad plastic MAE `<= 10`
* hard plastic MAE `<= 12`
* hard p95 principal `<= 120`
* yield violation p95 `<= 1e-6`

and the projected-teacher `Jv` probe looks numerically stable enough to justify the tangent path.

Then add only a **directional tangent loss**, not a separate `DS` head. Recent AD-based constitutive implementation work strongly supports this style of workflow: use autodiff to simplify tangent development and debugging instead of hand-building a separate tangent model. ([ojs.cvut.cz][3])

## What I would stop doing

I would keep all of these closed:

* packet5 or any packet-family continuation
* atlas variants
* routing redesign
* Program X follow-ons
* exact-latent branchwise targets
* separate `DS` heads

Those lines have already told you what they can tell you.

## My actual interpretation of “how close are we?”

Closer than it looks.

You are **not** close to a small all-neural exact-admissible MC replacement.

But you are **close to a projected NN MC surrogate** that is admissible and forward-accurate enough to deserve a real tangent and shadow-solver probe.

That is a much stronger position than where packet4 left you.

The real next question is no longer “can a neural model represent MC?” Phase 0 already answered that in the practical sense.

The real next question is:

**can you preserve the projected teacher while compressing it enough to make deployment worthwhile?**

That is the next experiment family I would run.

The cleanest next deliverable would be a bounded continuation memo with exactly these four items:

1. projected-teacher `DS` probe
2. same-capacity projected-teacher preservation control
3. Tier-1 distillation/compression sweep
4. stop rule if Tier 1 fails to approach the projected teacher materially

[1]: https://link.springer.com/article/10.1007/s11831-023-10009-y "Neural Networks for Constitutive Modeling: From Universal Function Approximators to Advanced Models and the Integration of Physics | Archives of Computational Methods in Engineering | Springer Nature Link"
[2]: https://papers.ssrn.com/sol3/Delivery.cfm/4ec8d221-f567-4489-a329-38f90982f5ec-MECA.pdf?abstractid=5210734&mirid=1&utm_source=chatgpt.com "Learning elastoplasticity with implicit layers"
[3]: https://ojs.cvut.cz/ojs/index.php/ap/article/view/10646 "
		Robust implementation of elastoplastic constitutive models using automatic differentiation in PyTorch
							\| Acta Polytechnica
			"
