Yes. For Mohr–Coulomb `(\varepsilon, material) -> S`, I would change the generator at the **constitutive-coordinate level**, not just tune the current FE-noise model.

The exact operator you are replacing is the **reduced local constitutive map** `(\varepsilon, \bar c, \sin\phi, G, K, \lambda) -> \sigma`, with five branches, and the corrected stress is coaxial with the trial strain. Your repo already treats this as a pointwise problem and recommends a **principal-stress surrogate** for exactly that reason. ([GitHub][1])

The most useful reparameterization for the generator is not raw `E` and not canonicalized nodal `u`, but ordered principal coordinates
`I1 = ε1 + ε2 + ε3`, `Δ12 = ε1 - ε2 >= 0`, `Δ23 = ε2 - ε3 >= 0`.

Then
`ε1 = (I1 + 2Δ12 + Δ23)/3`,
`ε2 = (I1 - Δ12 + Δ23)/3`,
`ε3 = (I1 - Δ12 - 2Δ23)/3`.

In these coordinates the branch geometry becomes much cleaner:
`γ_sl = Δ12 / (1 + sinψ)`,
`γ_sr = Δ23 / (1 - sinψ)`,
`γ_la = (Δ12 + 2Δ23) / (3 - sinψ)`,
`γ_ra = (2Δ12 + Δ23) / (3 + sinψ)`.
For the right side of the pyramid you are simply in the regime `γ_sl > γ_sr`; the **smooth/right** interface is where `λ_s = γ_sr`; and `right_edge` is the admissible region with `γ_sl > γ_sr` and `λ_r` between `γ_sr` and `γ_ra`. That is the region I would target explicitly rather than hoping local displacement noise lands there. ([GitHub][1])

One subtlety is worth making explicit to any outside reviewer: the local update you are approximating is the **associated reduced problem**. Raw `(c0, φ, ψ, E, ν, SRF)` are first Davis-reduced to `(\bar c, \sin\phi, G, K, \lambda)`, and then the constitutive update uses `sinψ = sinφ`. So the branch geometry is not governed directly by the raw `ψ = 0°` cover-layer input. ([GitHub][1])

My recommendation is a **hybrid generator** with one primary component and two corrective components:

1. **Real-like principal-space resampler**
   Use **all** `generator_fit` cover-layer IP states to fit the support model, not just a small seed bank. Sample in a whitened space such as
   `[I1, log(Δ12+eps), log(Δ23+eps), log(c̄), atanh(sinφ), log(G), log(K), log(λ)]`
   or, even better, a dimensionless version like
   `[(K I1)/c̄, (G Δ12)/c̄, (G Δ23)/c̄, atanh(sinφ)]`.
   Then do local perturbation there. This matches the actual constitutive manifold much more directly than perturbing canonical nodal displacements.

2. **Analytic boundary-tube sampler**
   For each sampled reduced material row, choose a principal direction `d = [d1,d2,d3]` from real data, solve the exact scale `α` that lands on a named surface, then jitter `α` slightly to sample both sides of that surface.
   The important formulas are:

   * yield surface: `α_y = c̄ / A(d)`
   * smooth/right surface: `α_sr = c̄ / (A(d) - D_s * (d2-d3)/(1-s))`
     where
     `A(d) = 2G((1+s)d1 - (1-s)d3) + 2λ s (d1+d2+d3)`, `s = sinφ`,
     `D_s = 4 λ s^2 + 4 G (1 + s^2)`.
     The other surfaces are analogous:
   * smooth/left: `λ_s = γ_sl`
   * left/apex: `λ_l = γ_la`
   * right/apex: `λ_r = γ_ra`
     This lets you hit the dangerous interfaces **exactly**, then thicken them into tubes with a small signed jitter.

3. **Tail extension anchored on real hard states**
   Start from real high-tail states and separately scale hydrostatic and deviatoric parts. For example:

   * keep Lode sector fixed,
   * scale `I1` by `β_v`,
   * scale `(Δ12, Δ23)` by `β_d`,
     with `β_v, β_d` in something like `[1.0, 1.35]`.
     That is a better way to extend the upper strain tail than isotropic Gaussian noise in `u`.

4. **Optional FE-compatible element sampler**
   Keep the current element-level generator, but only as a realism component or for future element-context models. For a pure constitutive surrogate it should not be the main coverage mechanism.

A starting mixture that makes sense for `S` would be:

* `60%` real-like resampled states,
* `25%` analytic boundary tubes,
* `10%` tail extension,
* `5%` broad interior coverage.

Inside that `25%`, I would overweight the surfaces as:

* `40%` smooth/right,
* `20%` smooth/left,
* `15%` yield,
* `12.5%` left/apex,
* `12.5%` right/apex.

That is not branch balancing. It is **difficulty balancing**.

If you keep any element-space generator, corner-volume positivity is too weak on its own. I would add:

* `det(J) > 0` at all 11 quadrature points and a few extra interior barycentric samples,
* a lower bound on the minimum singular value of `J`,
* an upper bound on `cond(J)`,
* a midside-node distortion measure relative to the affine corner tetrahedron,
* an empirical support gate in canonical `U` space, such as PCA/Mahalanobis distance against real cover-layer elements.

What I would store in the dataset, beyond what you already save, is:

* `gamma_sl, gamma_sr, gamma_la, gamma_ra`
* `lambda_s, lambda_l, lambda_r, lambda_a`
* normalized margins like
  `m_s_right = (gamma_sr - lambda_s) / (|gamma_sr| + |lambda_s| + eps)`
* `m_left_vs_right = (gamma_sl - gamma_sr) / (|gamma_sl| + |gamma_sr| + eps)`
* `source_type` = `{real_like, boundary_tube, tail, fe_seed}`
* `surface_id` for boundary samples
* `regime_id` or `source_call_id`

That gives you the diagnostics you actually need: not only overall stress MAE, but stress error as a function of **distance to the smooth/right boundary**.

For training, I would keep the surrogate aligned with the operator:

* predict **ordered principal stresses** and rotate back to global `S`,
* add branch logits as an auxiliary head,
* validate on one **real-prior** split and one **boundary challenge** split,
* if you later supervise `DS`, exclude a tiny tube around exact interfaces, because `S` is the continuous object here but `DS` is the discontinuous one. The repo already stores principal quantities, branch labels, and optional tangent support in that direction. ([GitHub][2])

The strongest signal in the brief you shared is this: some failed real states are outside the seed-neighborhood support, but many are still near the synthetic strain cloud. That means the next gain is unlikely to come from “more FE-valid noise” alone. It is more likely to come from getting the **local constitutive geometry** right in the neighborhood of `smooth/right_edge`.

I also put together a small prototype helper module with the exact principal-space branch metrics and analytic surface samplers: [branch_geometry_prototype.py](sandbox:/mnt/data/branch_geometry_prototype.py)

The first experiment I would run is simple: train the principal-stress surrogate on a mixture of real-like resampling plus a heavy smooth/right boundary tube, then report real-test stress error in bins of `|m_s_right|`. That will tell you very quickly whether the generator change is attacking the actual bottleneck.

[1]: https://raw.githubusercontent.com/Beremi/mohr_coulomb_NN/main/docs/mohr_coulomb.md "raw.githubusercontent.com"
[2]: https://github.com/Beremi/mohr_coulomb_NN "GitHub - Beremi/mohr_coulomb_NN · GitHub"
