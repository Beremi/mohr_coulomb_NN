# 3D Mohr-Coulomb constitutive relationship

## Scope

This document explains the 3D constitutive law implemented in `src/mc_surrogate/mohr_coulomb.py`, how it relates to the original Octave/MATLAB implementation, and why the branch structure matters for surrogate learning.

The target operator is the local map

\[
\varepsilon \mapsto \sigma
\]

for an **elastic-perfectly-plastic Mohr-Coulomb material** with **associated flow** in the reduced constitutive problem used inside the slope-stability code.

## What the original slope-stability code actually evaluates

In the original repository, the constitutive function used by the 3D mechanics code is

\[
(\varepsilon,\bar c,\sin\phi,G,K,\lambda) \mapsto \sigma,
\]

where

- \(\varepsilon\) is the engineering-strain vector in 3D Voigt form,
- \(\bar c\) and \(\sin\phi\) are the **reduced** Mohr-Coulomb parameters after Davis reduction,
- \(G\) is the shear modulus,
- \(K\) is the bulk modulus,
- \(\lambda\) is the Lamé coefficient.

So the constitutive law itself does **not** directly take the raw geotechnical inputs \((c_0,\phi,\psi,E,\nu,\text{SRF})\). The raw parameters are first reduced, then the reduced operator is applied.

This is an important design detail for the surrogate. You can place the surrogate at two different levels:

1. surrogate for the **reduced local constitutive operator** only,
2. surrogate for the larger map from raw material inputs plus strain to stress.

This repository focuses on the first placement, because it is the cleanest drop-in replacement for the local constitutive call.

## Strain and stress conventions

The code uses 3D Voigt ordering

\[
[\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},\gamma_{12},\gamma_{13},\gamma_{23}],
\]

where the shear entries are **engineering shears**:

\[
\gamma_{ij} = 2\varepsilon_{ij}.
\]

The Cauchy stress output is stored as

\[
[\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{12},\sigma_{13},\sigma_{23}].
\]

That is why the original MATLAB code multiplies the strain vector by

\[
\mathrm{IDENT}=\mathrm{diag}(1,1,1,1/2,1/2,1/2)
\]

before computing invariants and elastic stresses.

## Elastic law

For isotropic linear elasticity,

\[
\sigma = \lambda \,\mathrm{tr}(\varepsilon)\,I + 2G\,\varepsilon.
\]

In principal form, if \(\varepsilon_1 \ge \varepsilon_2 \ge \varepsilon_3\), then the trial principal stresses are

\[
\sigma_i^{\mathrm{tr}} = \lambda I_1 + 2G\varepsilon_i,
\qquad
I_1 = \varepsilon_1+\varepsilon_2+\varepsilon_3.
\]

## Mohr-Coulomb yield function in principal stress space

With ordered principal stresses \(\sigma_1 \ge \sigma_2 \ge \sigma_3\), the Mohr-Coulomb surface can be written as

\[
f(\sigma)
=
(1+\sin\phi)\sigma_1
-
(1-\sin\phi)\sigma_3
-
2c\cos\phi.
\]

In the reduced constitutive problem used by the code, this becomes

\[
f(\sigma)
=
(1+\sin\phi)\sigma_1
-
(1-\sin\phi)\sigma_3
-
\bar c,
\]

where

\[
\bar c = 2c_\lambda \cos(\phi_\lambda).
\]

The trial yield value therefore becomes

\[
f^{\mathrm{tr}}
=
2G\left((1+\sin\phi)\varepsilon_1-(1-\sin\phi)\varepsilon_3\right)
+
2\lambda\sin\phi\,I_1
-
\bar c.
\]

If \(f^{\mathrm{tr}}\le 0\), the point is elastic.

## Davis reduction

The slope-stability solver uses Davis A/B/C reduction strategies to map raw parameters to reduced constitutive parameters.

The output used by the constitutive law is

- \(\bar c\),
- \(\sin\phi\).

The formulas implemented in `materials.py` are direct translations of the original `reduction.m`.

## Associated flow and branch structure

For the associated case used here, the dilatancy satisfies

\[
\sin\psi = \sin\phi.
\]

The return mapping is **piecewise**. Depending on the location of the trial state in principal space, the corrected stress falls onto one of five branches:

1. elastic,
2. smooth portion of the yield surface,
3. left edge,
4. right edge,
5. apex.

That branch structure is the central reason why this constitutive operator is easy for an exact solver but awkward for a naive neural network.

## Branch criteria

Let the ordered principal strains satisfy

\[
\varepsilon_1 \ge \varepsilon_2 \ge \varepsilon_3.
\]

The original implementation computes the branch-decision thresholds

\[
\gamma_{sl}=\frac{\varepsilon_1-\varepsilon_2}{1+\sin\psi},
\qquad
\gamma_{sr}=\frac{\varepsilon_2-\varepsilon_3}{1-\sin\psi},
\]

\[
\gamma_{la}=\frac{\varepsilon_1+\varepsilon_2-2\varepsilon_3}{3-\sin\psi},
\qquad
\gamma_{ra}=\frac{2\varepsilon_1-\varepsilon_2-\varepsilon_3}{3+\sin\psi}.
\]

It also computes candidate plastic multipliers for the smooth, left-edge, right-edge, and apex returns.

The realized branch is then chosen by testing the inequalities that define which candidate is admissible.

## Closed-form stress update on each branch

### Elastic

\[
\sigma_i = \lambda I_1 + 2G\varepsilon_i.
\]

### Smooth return

\[
\sigma_1
=
\lambda I_1 + 2G\varepsilon_1
-
\Delta\gamma_s\left(2\lambda\sin\psi + 2G(1+\sin\psi)\right),
\]

\[
\sigma_2
=
\lambda I_1 + 2G\varepsilon_2
-
\Delta\gamma_s\left(2\lambda\sin\psi\right),
\]

\[
\sigma_3
=
\lambda I_1 + 2G\varepsilon_3
-
\Delta\gamma_s\left(2\lambda\sin\psi - 2G(1-\sin\psi)\right).
\]

### Left edge

Here \(\sigma_1=\sigma_2\). The corrected principal stresses are

\[
\sigma_{12}
=
\lambda I_1 + G(\varepsilon_1+\varepsilon_2)
-
\Delta\gamma_l\left(2\lambda\sin\psi + G(1+\sin\psi)\right),
\]

\[
\sigma_3
=
\lambda I_1 + 2G\varepsilon_3
-
\Delta\gamma_l\left(2\lambda\sin\psi - 2G(1-\sin\psi)\right).
\]

### Right edge

Here \(\sigma_2=\sigma_3\). The corrected principal stresses are

\[
\sigma_1
=
\lambda I_1 + 2G\varepsilon_1
-
\Delta\gamma_r\left(2\lambda\sin\psi + 2G(1+\sin\psi)\right),
\]

\[
\sigma_{23}
=
\lambda I_1 + G(\varepsilon_2+\varepsilon_3)
-
\Delta\gamma_r\left(2\lambda\sin\psi - G(1-\sin\psi)\right).
\]

### Apex

At the apex,

\[
\sigma_1=\sigma_2=\sigma_3=\frac{\bar c}{2\sin\phi}.
\]

## Why the stress and strain remain coaxial

For this isotropic return mapping, the corrected stress and the trial strain share principal directions. That is why the original code works in principal space and then reconstructs the stress tensor using eigenprojections.

This coaxiality is also the key reason why the recommended surrogate predicts **principal stresses** rather than the full six stress components directly.

## How the original MATLAB/Octave implementation is made efficient

The original `constitutive_problem_3D.m` is written in a very vectorized style.

### 1. Vectorized invariants

Instead of looping over integration points, it computes arrays of

- \(I_1\),
- \(I_2\),
- \(I_3\),
- Lode-angle-related quantities,
- ordered eigenvalues,

all at once.

### 2. Eigenprojections from invariants

For distinct eigenvalues, the code reconstructs spectral projectors through polynomial expressions in the trial strain tensor, instead of calling an eigensolver point by point.

That is one of the main technical tricks in the MATLAB version. It avoids repeated small-matrix eigendecompositions and stays highly vectorized.

### 3. Branch masks

Each return branch is activated by a Boolean mask. The code then evaluates only the corresponding formulas on the masked subset.

### 4. Consistent tangent in the original code

The original implementation also includes analytic consistent-tangent expressions. Those are long and branch-dependent.

This Python repository does **not** hand-port the full analytic tangent. Instead it provides an optional centered finite-difference tangent. That is enough for diagnostics, dataset enrichment, and tangent-supervised experiments, but it is not the same as the original exact consistent tangent.

## How the Python implementation differs

The Python implementation keeps the **same branch formulas** and the same reduced-input interface, but it uses a batched dense eigendecomposition from NumPy rather than reproducing every spectral-projection trick from the MATLAB code.

That is a deliberate trade-off:

- the formulas stay faithful,
- the code is shorter and easier to audit,
- the implementation is still vectorized over batches,
- the optional tangent stays simple.

For dataset generation and surrogate work, that is usually the right compromise.

## Why this constitutive law is hard for a neural surrogate

The map is not globally smooth. It is piecewise smooth with kinks at

- the elastic/plastic interface,
- the smooth/edge transitions,
- the edge/apex transitions.

Those are exactly the places where a naive network tends to perform worst.

That is why the repository does three things:

1. samples branches explicitly,
2. uses a spectral/invariant-aware architecture,
3. includes an auxiliary branch-classification loss in the recommended model.

## References

- Original repository: `sysala/slope_stability`
- Constitutive files of interest:
  - `slope_stability/+CONSTITUTIVE_PROBLEM/constitutive_problem_3D.m`
  - `slope_stability/+CONSTITUTIVE_PROBLEM/reduction.m`
  - `slope_stability/+CONSTITUTIVE_PROBLEM/potential_3D.m`
- Sysala, Čermák, Ligurský: *Subdifferential-based implicit return-mapping operators in Mohr-Coulomb plasticity*, ZAMM, 2017.
