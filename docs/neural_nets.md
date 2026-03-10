# Neural-network surrogate design

## What is being learned

The local surrogate approximates the constitutive map

\[
(\varepsilon,\bar c,\sin\phi,G,K,\lambda) \mapsto \sigma.
\]

This is a **pointwise constitutive surrogate**, not a global field-to-field operator over meshes.

That distinction matters. A model such as DeepONet or FNO is designed to learn mappings between function spaces, for example from an input field or PDE coefficient field to an output solution field. Here the object to replace is much smaller: one material-point operator.

## Recommended architecture: principal-stress MLP

The default model in this repository is `model_kind=principal`.

### Inputs

The recommended features are:

- ordered principal strains \((\varepsilon_1,\varepsilon_2,\varepsilon_3)\),
- trace \(I_1\),
- a deviatoric magnitude,
- a Lode-angle surrogate \(\cos(3\theta)\),
- transformed reduced material parameters:
  - \(\log \bar c\),
  - \(\operatorname{atanh}(\sin\phi)\),
  - \(\log G\),
  - \(\log K\),
  - \(\log \lambda\).

So the full feature vector has 11 entries.

### Outputs

- ordered principal stresses \((\sigma_1,\sigma_2,\sigma_3)\),
- auxiliary branch logits for
  - elastic,
  - smooth,
  - left edge,
  - right edge,
  - apex.

### Why this is the default

This architecture encodes the most valuable a priori structure:

1. **isotropy**: the constitutive law does not depend on the arbitrary global coordinate frame,
2. **coaxiality**: for this return mapping, corrected stress and trial strain share principal directions,
3. **low intrinsic dimensionality**: once the rotation is factored out, the constitutive map is much simpler,
4. **piecewise geometry**: the branch head tells the network that the target map is not globally smooth.

In practice this means the network does not waste capacity learning rotational invariance from raw Cartesian components.

## Baseline architecture: raw Voigt MLP

The baseline `model_kind=raw` uses:

- input: raw engineering strain components + transformed material features,
- output: all six stress Voigt components.

This baseline is useful for ablation and sanity checking, but it is intentionally not the main recommendation.

## Network core

Both models use the same residual MLP backbone:

- linear input projection,
- repeated residual blocks,
- GELU activations,
- optional dropout,
- linear regression head.

The principal model also adds a branch-classification head.

This is a good fit because the input dimension is tiny and the constitutive map is piecewise smooth but not sequence-like or spatially local.

## Normalization

Training is carried out on standardized features and targets:

\[
\hat x = \frac{x-\mu_x}{\sigma_x},
\qquad
\hat y = \frac{y-\mu_y}{\sigma_y}.
\]

This keeps the optimizer stable despite the large variation in:

- cohesion,
- elastic moduli,
- stress magnitudes,
- branch-dependent response scales.

## Loss and optimization

The default training loop uses:

- mean-squared error on the regression target,
- auxiliary cross-entropy on the branch class when available,
- AdamW optimizer,
- ReduceLROnPlateau scheduler,
- gradient clipping,
- early stopping.

The total loss for the principal model is

\[
\mathcal L
=
\mathcal L_{\text{reg}}
+
\alpha \mathcal L_{\text{branch}},
\]

with \(\alpha\) controlled by `branch_loss_weight`.

## Why not start with DeepONet or FNO here?

Those are excellent models for learning operators over function spaces, but this task is different.

- The constitutive update is evaluated at one integration point at a time.
- There is no input field discretization or output field discretization to exploit.
- The relevant structure is tensor invariance and branch geometry, not nonlocal PDE coupling.

For that reason an invariant-aware pointwise MLP is the stronger first model.

## Why the principal model is the best first network

For this specific task, the "best" network is the one that exploits the constitutive structure that is already known analytically.

A good first surrogate therefore should:

- remove arbitrary rotations from the learning problem,
- preserve principal ordering,
- emphasize the branch structure,
- stay small enough to train quickly on tabular HDF5 data.

The principal-stress residual MLP with branch head satisfies all four requirements.

It is not necessarily the final best possible model, but it is the best **practical starting point**.

## Good next extensions

If you want to push beyond the current implementation, the most promising upgrades are:

### 1. Mixture of experts

Use a gating network plus branch-specialized experts for

- elastic,
- smooth,
- left edge,
- right edge,
- apex.

That matches the actual constitutive geometry very closely.

### 2. Stronger multi-task training

Predict both

- stresses,
- branch class,
- possibly the plastic multiplier,
- possibly the trial yield value.

The extra tasks may help around kinks.

### 3. Tangent supervision

If you generate tangent data, add a loss term on

\[
\frac{\partial \sigma}{\partial \varepsilon}.
\]

That is particularly useful if the final deployment target is a Newton-type finite element solve.

### 4. Hard ordering constraints

Instead of sorting the three predicted principal stresses, predict a base value plus positive increments so that

\[
\sigma_1 \ge \sigma_2 \ge \sigma_3
\]

is built into the output parameterization.

### 5. Physics-informed calibration losses

You can penalize violations of:

- symmetry,
- branch consistency,
- ordering,
- basic yield-surface structure.

## Deployment note

If the surrogate is ultimately inserted into a global slope-stability solve, the most relevant validation criterion is not only pointwise MAE on random samples. It is also:

- stability near branch transitions,
- pathwise behavior,
- robustness over the material range,
- behavior in the plastic zones that dominate failure.

That is why this repository includes both scalar metrics and path-comparison plots.
