# Dataset creation rationale

## Goal

The dataset generator should not merely draw random strains. It should produce a set of material-point states that is useful for the **actual deployment regime**: repeated constitutive calls inside slope-stability finite elements with strength reduction.

That means the dataset must cover:

- elastic states,
- smooth plastic return states,
- edge returns,
- apex returns,
- different stiffness scales,
- different strength scales,
- different friction levels,
- arbitrary tensor orientations.

## Material parameters sampled

The generator samples raw geotechnical inputs:

- effective cohesion \(c_0\),
- friction angle \(\phi\),
- dilatancy angle \(\psi\),
- Young's modulus \(E\),
- Poisson ratio \(\nu\),
- strength-reduction factor,
- Davis reduction rule A/B/C.

Then it converts them to the reduced constitutive parameters actually used by the constitutive law:

- \(\bar c\),
- \(\sin\phi\),
- \(G\),
- \(K\),
- \(\lambda\).

## Why the ranges are chosen this way

The original 3D slope-stability examples in the upstream repository use values such as:

- \(c_0\) around 6, 10, 15, 18,
- \(\phi\) around 30° to 45°,
- \(E\) around 10,000 to 50,000,
- \(\nu\) around 0.30 to 0.33.

The generator widens those ranges rather than copying them exactly. That is deliberate.

The surrogate should see enough variation to interpolate safely across:

- stiffer and softer zones,
- stronger and weaker layers,
- early and late strength-reduction stages.

The widened default ranges are still geotechnically plausible, but they are broader than the benchmark scripts so the model is not tied to one mesh or one case.

## Why branch balancing matters

If you sample strain states naively, the dataset will usually be dominated by elastic points and smooth returns.

That is bad for surrogate training because the hard parts of the constitutive law are exactly the branch transitions and corner returns.

So the generator uses **branch-targeted rejection sampling**:

1. propose a material state,
2. propose a principal-strain direction designed to favor a target branch,
3. scale it relative to an estimated yield level,
4. evaluate the exact constitutive law,
5. keep the sample only if the realized branch matches the target branch.

This gives much better coverage of left-edge, right-edge, and apex states.

## Principal-strain proposals

The generator uses different templates depending on the target branch:

- **elastic**: sub-yield states,
- **smooth**: clearly separated principal strains, slightly above yield,
- **left edge**: \(\varepsilon_1 \approx \varepsilon_2 > \varepsilon_3\),
- **right edge**: \(\varepsilon_1 > \varepsilon_2 \approx \varepsilon_3\),
- **apex**: near-hydrostatic compression.

This mirrors the geometry of the Mohr-Coulomb surface in principal space.

## Why yield-relative scaling is useful

The generator does not choose strain magnitudes completely blindly.

For a sampled principal-strain direction \(d\), it estimates the scale that would place the trial state near the yield surface by solving for a scalar \(\alpha\) in

\[
f^{\mathrm{tr}}(\alpha d) \approx 0.
\]

Because the trial yield value is linear in the principal strains, this scaling estimate is cheap and effective.

Then the actual sample is obtained by multiplying that estimated yield scale by a branch-specific factor:

- less than one for elastic states,
- slightly above one for smooth returns,
- further above one for edge and apex states.

This keeps the dataset concentrated in the parts of state space that matter most.

## Rotation augmentation

The branch logic depends on principal values, but the FE code will call the surrogate in arbitrary global tensor orientations.

Therefore, after a principal-strain state is generated, the code draws a random 3D rotation and maps it back to a full strain tensor:

\[
\varepsilon = Q\,\mathrm{diag}(\varepsilon_1,\varepsilon_2,\varepsilon_3)\,Q^{\mathsf T}.
\]

This is essential. Without random rotations, a raw Cartesian network would overfit to a preferred basis.

## Why this covers slope-stability use better than naive random sampling

In a real slope-stability solve, the constitutive operator sees a mixture of:

- mostly elastic points,
- plastic zones concentrated near slip surfaces,
- branch changes near corners of the Mohr-Coulomb surface,
- material heterogeneity,
- changing strength reduction over the SSR continuation.

The generator is designed to mimic that broad envelope, not to mimic one single benchmark mesh exactly.

## HDF5 layout

Each dataset stores:

- `strain_eng` — engineering strain vector,
- `stress` — exact stress vector,
- `strain_principal` — ordered principal strains,
- `stress_principal` — ordered principal stresses,
- `eigvecs` — trial-strain eigenvectors,
- `branch_id` — branch label,
- `plastic_multiplier` — plastic multiplier used by the realized branch,
- `f_trial` — trial yield value,
- `material_raw` — original sampled material parameters,
- `material_reduced` — reduced constitutive parameters,
- `tangent` — optional numerical tangent,
- `split_id` — train/val/test split.

This layout supports both:

- exact-law verification,
- multiple surrogate designs from one data file.

## Practical advice

A good workflow is:

1. generate a **balanced** dataset first,
2. train the principal-stress surrogate,
3. inspect errors by branch,
4. if needed, generate a second dataset with a more realistic branch distribution for fine-tuning.

That usually works better than starting from a dataset that simply mirrors the natural dominance of elastic points.
