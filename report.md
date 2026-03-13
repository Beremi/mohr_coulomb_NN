
# State of the art for neural-network surrogates of integration-point constitutive updates
## With emphasis on the 3D Mohr–Coulomb operator in `Beremi/slope_stability_octave`

Prepared on 2026-03-12.

---

## 1. Scope and main conclusion

This report focuses on **material-point / integration-point constitutive surrogates**: neural-network models that replace, accelerate, or restructure the local constitutive update performed at each Gauss point, rather than replacing the full PDE solver. The target use case is the 3D Mohr–Coulomb constitutive operator in the repository:

- `https://github.com/Beremi/slope_stability_octave`

The high-level conclusion is:

1. **The closest state-of-the-art methods to your problem are not generic black-box stress regressors.**
   The most relevant recent work either
   - preserves the **stress-integration / return-mapping structure** and lets the network learn only the difficult parts, or
   - encodes **thermodynamics / plasticity structure** directly in the architecture or loss.

2. **For the specific `slope_stability_octave` Mohr–Coulomb implementation, a recurrent model is *not* the best first choice.**
   The local law in that repository is an **elastic-perfectly plastic**, reduced-parameter, associated-flow Mohr–Coulomb update with explicit branch formulas. For fixed reduced material parameters, it is effectively a **static local map** from total strain to stress/tangent, not a generic hidden-history constitutive process.  
   Therefore the strongest practical baseline is a:
   - **principal-stress / invariant-aware feedforward network**, ideally
   - **branch-aware** (elastic / smooth / edge / apex) and
   - possibly embedded inside a **return-mapping-inspired wrapper**.

3. **Your training difficulties are most likely caused by the geometry of the constitutive law, not by lack of network size.**
   The 3D Mohr–Coulomb update is:
   - **piecewise smooth**
   - **nonsmooth at edges and the apex**
   - **rotation-equivariant / coaxial**
   - strongly affected by **branch imbalance**
   - sensitive near repeated principal values and near the elastic–plastic boundary.  
   These properties make raw `6 -> 6` Voigt regression a poor default.

---

## 2. What exactly the repository is doing at the constitutive level

### 2.1 Repository context

The repository is a slope-stability code for 2D/3D lower-bound limit analysis with strength reduction. The README states that the `+CONSTITUTIVE_PROBLEM` part contains the material model and yield criteria, and that a C/MEX implementation is used for the constitutive routines.

### 2.2 Material parameters used in the 3D example

In the 3D homogeneous slope example, the material data are assigned as

```matlab
mat_props = [6, 45, 0, 40000, 0.3, 20, 20]
```

with the interpretation

```text
[c0, phi, psi, young, poisson, gamma_sat, gamma_unsat]
```

and `Davis_type = 'B'`.

The heterogeneous-material routine converts angles to radians and computes the elastic moduli

\[
G = \frac{E}{2(1+\nu)}, \qquad
K = \frac{E}{3(1-2\nu)}, \qquad
\lambda = K - \frac{2}{3}G .
\]

### 2.3 Davis reduction in the repository

The constitutive object stores both the original Mohr–Coulomb parameters and the reduced parameters used by the actual local integration routine:

- `c0`, `phi`, `psi`
- `c_bar`
- `sin_phi`
- `shear`, `bulk`, `lame`

The reduction routine applies Davis procedure A/B/C.  
For the commonly used **Davis type B** branch in the repository:

\[
c_{01} = \frac{c_0}{\lambda_\mathrm{SRF}}, \qquad
\phi_1 = \arctan\!\left(\frac{\tan\phi}{\lambda_\mathrm{SRF}}\right), \qquad
\psi_1 = \arctan\!\left(\frac{\tan\psi}{\lambda_\mathrm{SRF}}\right),
\]

\[
\beta = \frac{\cos\phi_1 \cos\psi_1}{1-\sin\phi_1 \sin\psi_1},
\qquad
c_{0,\lambda} = \beta\, c_{01},
\qquad
\phi_\lambda = \arctan\!\bigl(\beta \tan\phi_1\bigr),
\]

and the constitutive update uses

\[
\bar c = 2\, c_{0,\lambda}\cos\phi_\lambda,
\qquad
\sin\bar\phi = \sin\phi_\lambda.
\]

This is important for machine learning: the actual integration-point map in the code is driven by **reduced** quantities, not by the original \((c_0,\phi,\psi)\) alone.

### 2.4 The 3D constitutive update

The 3D kernel computes stress and a consistent tangent for an **elastic-perfectly plastic Mohr–Coulomb model with an associated flow rule**. In the kernel the line

```c
sin_psi = sin_phi;
```

makes the local integration routine associated after the Davis reduction.

The algorithm:

1. Builds elastic operators.
2. Computes the trial elastic state from strain.
3. Extracts principal values of the trial strain / stress-like quantities.
4. Evaluates the trial yield quantity.
5. Chooses one of several branches:
   - elastic
   - smooth return to the yield surface
   - left edge
   - right edge
   - apex
6. Reconstructs stress and tangent.

A representative trial-yield expression in the repository is

\[
f_\mathrm{tr}
=
2G\Bigl[(1+\sin\bar\phi)\,\varepsilon_1^\star
-
(1-\sin\bar\phi)\,\varepsilon_3^\star\Bigr]
+
2\lambda\,\sin\bar\phi\, I_1
-
\bar c,
\]

where the update is expressed in terms of the ordered principal trial quantities and the volumetric part \(I_1\).

### 2.5 Why this matters for learning

This local constitutive law has four properties that dominate the machine-learning design:

1. **Coaxiality / spectral structure**  
   The stress update is reconstructed in the principal basis of the trial state. A surrogate should exploit this rather than learn arbitrary coordinate-frame behavior.

2. **Piecewise-defined branches**  
   The mapping is not globally smooth. Smooth-region and edge/apex formulas are different. A single unconstrained MLP has to approximate several qualitatively different submaps at once.

3. **Apex and edge singular geometry**  
   Near corners and the apex, the update becomes difficult for both classical algorithms and neural surrogates. This is where training often fails first.

4. **Strength reduction changes the parameter manifold**  
   The slope-stability analysis repeatedly queries the constitutive operator along a family of reduced materials. If the dataset does not cover this reduction path well, FE performance deteriorates quickly.

---

## 3. Taxonomy of the literature most relevant to your use case

The papers I found fall into five groups.

### Group A — Direct local constitutive surrogates with plasticity structure
These are the closest to your target.

- **EPNN** (Eghbalian, Pouragha, Wan): architecture embeds additive strain decomposition and nonlinear incremental elasticity.
- **Return-mapping-assisted learning** (Meng et al., 2025 preprint): separate networks for yield function and stress gradient, used inside cutting-plane / closest-point return mapping.
- **Haigh–Westergaard coordinate surrogate** (2025): learns elastoplastic updates in reduced invariant coordinates.
- **si-PiNet / PiNet**: neural stress integration with elastoplastic prior information.

### Group B — Thermodynamics-based constitutive networks
These are more physics-encoded and often more robust, especially for generalization.

- **TANN**
- **CANN / iCANN**
- generalized dual-potential iCANN extension for pressure-sensitive inelasticity

### Group C — Sequence models for path-dependent materials
Useful for hardening, cyclic loading, or experimental sequence data, but less natural for your current Mohr–Coulomb kernel.

- RNN / GRU / LSTM constitutive models
- self-consistent recurrent architectures
- recursive Bayesian models for uncertainty
- transfer-learning RNN models

### Group D — Soil / geomaterial constitutive networks with embedded physics
Very relevant conceptually.

- non-associative Drucker–Prager PINN
- tensor-based physics-encoded NN for soil (TB-PeNN)
- generalized-plasticity-encoded NN (GPM-PeNN)

### Group E — PDE-level PINNs for elastoplasticity / Mohr–Coulomb
Relevant for adjacent future work, but not a replacement for your local constitutive operator.

- PINN for Mohr–Coulomb plasticity in geomechanics
- field-solver elastoplastic PINNs

---

## 4. Most relevant papers, ranked for your exact problem

### Tier 1 — Directly useful for your Mohr–Coulomb surrogate plan

#### 4.1 si-PiNet: a novel stress integration method for elastoplastic models (2025)

**Why it matters**  
This is one of the closest matches to your problem statement because it targets **stress integration itself**, not just constitutive regression. The method is explicitly reported on **von Mises, Mohr–Coulomb, and Modified Cam-clay**.

**Main idea**  
Use a prior-information neural network to search for the correct stress update under a strain increment while constraining the search with elastoplastic theory. The paper reports lower sensitivity to strain-increment magnitude than conventional stress integration algorithms.

**Relevance to your repository**  
Very high. If you want a next-step method beyond plain supervised regression, this is arguably the most on-target recent reference.

**Reimplementation notes**
- Preserve your existing input physics:
  - trial stress / trial strain information
  - reduced Mohr–Coulomb parameters
  - branch / consistency information
- Keep the constitutive update framed as **finding the admissible corrected stress**, not directly learning all outputs from scratch.
- Evaluate not just stress error, but:
  - yield-surface residual
  - branch consistency
  - sensitivity to strain-increment size
  - Newton robustness inside FE

**What to borrow**
- “stress integration as constrained search” viewpoint
- prior-information encoding from plasticity
- robustness tests versus increment-size changes

#### 4.2 Numerical Implementation of Deep Learning-Based Constitutive Model for Geomaterials Using Return Mapping Algorithms (Meng et al., 2025 preprint)

**Why it matters**  
This is probably the closest *implementation pattern* to your current code. The framework uses:
- one fully connected network for the **yield function**
- another for the **stress gradient**
- then inserts them into classic return-mapping algorithms such as
  - **cutting-plane algorithm (CPA)**
  - **closest-point projection method (CPPM)**

**Why this is powerful**  
It does **not** discard the return-mapping structure that your repository already relies on conceptually.  
Instead, it replaces the analytic constitutive ingredients inside the classical algorithm.

**Relevance to Mohr–Coulomb**
Very high, especially because the preprint explicitly discusses comprehensive datasets, targeted data augmentation, Sobolev training, and implementation in a UMAT-like setting.

**Reimplementation notes**
- Use your exact Python Mohr–Coulomb reimplementation to generate labels for:
  - yield value \(f\)
  - stress-gradient / flow-direction information
  - corrected stress
  - branch label
- Train two networks:
  1. \( \hat f(x) \)
  2. \( \widehat{\nabla f}(x) \) or branchwise gradient surrogate
- Insert these into a Python return-mapping wrapper.
- Compare with:
  - direct stress regression
  - exact analytic kernel
- This route is especially promising if your final goal is FEM robustness.

**Key caution**
For Mohr–Coulomb the exact yield surface is nonsmooth. Gradient-learning near edges and apex is delicate. A practical workaround is:
- smooth surrogate for the smooth region,
- explicit handling of edges/apex,
- or a classifier + branchwise models.

#### 4.3 EPNN: A physics-informed deep neural network for surrogate modeling in classical elasto-plasticity (2022/2023)

**Why it matters**  
EPNN is one of the clearest examples where embedding constitutive structure in the network gives better data efficiency and extrapolation than a plain ANN.

**Core idea**
The network architecture hardwires:
- additive decomposition of strain into elastic and plastic parts
- nonlinear incremental elasticity

The authors show:
- less training data needed
- better extrapolation to unseen loading paths
- better robustness than regular ANNs

**Relevance to your case**
High, but slightly less direct than si-PiNet / return-mapping-assisted learning because EPNN was demonstrated on sands with synthetic data from a different constitutive model. Still, the design principle is highly applicable.

**Useful training details reported by the authors**
- MAE worked better than MSE in their setup.
- Data were normalized to \([-1,1]\).
- A 60/20/20 train/validation/test split was used.
- ADAM optimizer was used.
- The best architecture in their case used subnetworks with roughly 60–75 neurons per hidden layer, with different depths for different outputs.

**What to borrow**
- architecture-level physics encoding
- multi-output decomposition instead of one giant raw stress network
- MAE / scale-robust training losses
- explicit testing on unseen loading paths

#### 4.4 A Neural Network-integrated Elastoplastic Constitutive Model using Haigh–Westergaard Coordinates and Data Augmentation (2025)

**Why it matters**  
Mohr–Coulomb is naturally expressed in principal / invariant coordinates. A 2025 paper showed that learning elastoplastic constitutive updates in **Haigh–Westergaard coordinates** can improve out-of-distribution generalization compared with training directly in principal-stress space.

**Main lesson**
The coordinate choice is not cosmetic.  
For pressure-sensitive, invariant-driven plasticity, choosing coordinates aligned with the constitutive geometry can materially improve generalization.

**Direct translation to your case**
For the Beremi kernel, a very strong baseline is to learn in one of these spaces:
- ordered principal stresses / strains
- stress invariants plus Lode-angle-like coordinate
- Haigh–Westergaard coordinates

This is likely better than raw 6-component Voigt inputs.

---

### Tier 2 — Strong conceptual matches for pressure-sensitive soils

#### 4.5 Physics-infused DNN for non-associative Drucker–Prager elastoplasticity (Roy et al., 2024)

**Why it matters**  
Drucker–Prager is a smooth, pressure-sensitive cousin of Mohr–Coulomb. This paper constructs a multi-objective physics-infused loss with:
- constitutive law terms
- yield criterion
- non-associative flow rule
- Kuhn–Tucker conditions
- boundary conditions
- transfer learning

**What to borrow**
Even if you do not train a PDE PINN, the loss design is relevant:
- stress residual
- yield residual
- complementarity / KKT residual
- optional transfer-learning schedule

**Practical use**
This is a good “bridge paper” if your Mohr–Coulomb surrogate becomes unstable and you want to first prove ideas on a smoother pressure-sensitive model.

#### 4.6 Tensor-based physics-encoded neural networks for modeling constitutive behavior of soil (TB-PeNN, 2024)

**Why it matters**
This model:
- uses **stress tensor invariants** and soil state variables as inputs,
- outputs coefficients for tensorial constitutive relations,
- is designed to satisfy physical laws by construction,
- and is intended for numerical-software integration.

**Main lesson for your work**
Do not learn arbitrary tensor outputs when a structured tensor form is available.  
In your case, the equivalent statement is:
- learn the **principal / invariant scalar mapping**
- reconstruct the stress tensor exactly.

#### 4.7 Generalized plasticity model encoded physics-encoded neural network (GPM-PeNN, 2025)

**Why it matters**
This is a recent geomechanics line aimed at improving stress-path generalization and extrapolation by encoding generalized-plasticity structure as prior information.

**Main lesson**
For geomaterials, stress-path coverage and encoded priors matter more than brute-force data volume.

---

### Tier 3 — Excellent for general constitutive learning, but not my first choice for this repository

#### 4.8 TANN / CANN / iCANN family

These papers are foundational for thermodynamically consistent constitutive learning. Their strengths are:
- objectivity
- symmetry preservation
- thermodynamic consistency
- strong extrapolation potential

The newer iCANN dual-potential extension is especially interesting because it is formulated in terms of **stress invariants** and is explicitly intended to capture **pressure-sensitive inelasticity**.

**Why not first choice here?**
Your immediate task is narrower and simpler:
- fixed small-strain law
- local static mapping after Davis reduction
- explicit branch formulas already available

So while TANN/iCANN are intellectually strong, they are probably a **second-generation** direction after you establish a good branch-aware Mohr–Coulomb baseline.

#### 4.9 RNN-based constitutive surrogates

RNN / GRU / LSTM models are powerful for path-dependent plasticity. Important recent papers also show that naive recurrent models can fail because they are not **self-consistent** with respect to increment subdivision, and they may destabilize FE Newton iterations.

**Why not first choice for your current problem?**
Your repository’s local Mohr–Coulomb operator is elastic-perfectly plastic with no hidden hardening variable and is called as a local algebraic update for reduced material parameters.  
That means the most natural first surrogate is **not recurrent**.

**When they become relevant**
- cyclic loading data
- experimental training without complete internal-state labels
- hardening / softening models
- sequence-to-sequence constitutive discovery

---

## 5. The single most important design decision for your repository

### Do not learn raw `strain_voigt -> stress_voigt` as the main model.

For the Beremi constitutive operator, the best first surrogate is:

1. **Preprocess the input exactly as the constitutive law sees it**
   - compute ordered principal trial quantities or invariant coordinates
   - use reduced parameters \((\bar c,\sin\bar\phi,G,K,\lambda)\)
   - optionally include trial yield value \(f_\mathrm{tr}\)

2. **Predict in principal / invariant space**
   - updated principal stresses
   - or stress correction in principal space
   - or branch plus branchwise scalar outputs

3. **Rotate / reconstruct back to tensor form analytically**

This is the strongest change you can make if training is currently unstable.

---

## 6. Recommended surrogate formulations for the Beremi Mohr–Coulomb operator

Below are the formulations I would test in this order.

### Model A — Branch-aware principal-space supervised surrogate (recommended first baseline)

#### Inputs
For each sample:
- ordered principal trial strains or trial stresses
- \(I_1\), \(J_2^{1/2}\), Lode angle / Haigh–Westergaard coordinates
- reduced parameters:
  - \(\bar c\)
  - \(\sin\bar\phi\)
  - \(G\), \(K\), \(\lambda\)
- optional indicators:
  - trial yield function \(f_\mathrm{tr}\)
  - spectral gaps \(\varepsilon_1-\varepsilon_2\), \(\varepsilon_2-\varepsilon_3\)

#### Outputs
- branch label in \{elastic, smooth, left-edge, right-edge, apex\}
- updated principal stresses \((\sigma_1,\sigma_2,\sigma_3)\)

Optional extra outputs:
- plastic multiplier
- signed distance / yield residual
- tangent components in principal basis

#### Architecture
- shared trunk MLP
- branch-classification head
- stress-regression head
- or five branch-specific experts gated by the classifier

#### Why it fits the repository
It directly mirrors the exact algorithmic structure of the kernel.

### Model B — Hybrid exact-elastic + learned-plastic surrogate

Keep the elastic branch exact:
- if \(f_\mathrm{tr} \le 0\), return exact elastic stress
- otherwise call the NN only for plastic branches

This reduces the learning burden substantially and is often the easiest way to stabilize training.

### Model C — Return-mapping-assisted surrogate

Learn:
- yield function surrogate
- gradient / flow-direction surrogate
- maybe hard branch logic separately

Then embed the learned pieces in a local return-mapping solver.

This is the closest to the 2025 return-mapping preprint and likely the best route if your final bottleneck is **FEM robustness**, not just pointwise stress accuracy.

### Model D — PiNet / si-PiNet-style neural stress integrator

Treat the update as a constrained stress-search problem with encoded elastoplastic priors.

This is the strongest research direction if you want a publishable extension, but not the easiest first implementation.

---

## 7. Dataset design for Mohr–Coulomb in this repository

### 7.1 What the dataset must cover

Your dataset should not only be “large”; it must cover the **geometric events** that matter to the constitutive operator:

1. Elastic interior
2. Near-yield boundary
3. Smooth plastic return
4. Left edge
5. Right edge
6. Apex
7. Near-equal principal values
8. Pure volumetric states
9. Deviatoric-dominated states
10. Unloading / reloading if present in use

### 7.2 Why naive random sampling fails

Uniform random sampling in raw strain space usually under-samples:
- branch-transition neighborhoods
- corners and apex
- low-measure but high-importance regions

This is one of the main reasons networks look good on average error but fail in FE use.

### 7.3 Recommended sampling strategy

Use a **mixture sampler**:

#### (a) Broad background sampling
Sample:
- \(E\), \(\nu\), \(c_0\), \(\phi\), \(\psi\), SRF range
- strain states covering expected slope-stability magnitudes

#### (b) Branch-targeted sampling
Use the exact constitutive code to identify samples from each branch and actively balance:
- elastic
- smooth
- left edge
- right edge
- apex

#### (c) Boundary-focused refinement
Oversample near:
- \(f_\mathrm{tr} \approx 0\)
- branch decision thresholds
- repeated-eigenvalue regions
- high curvature in invariant space

#### (d) Physics-shaped loading paths
Include canonical paths:
- hydrostatic compression / extension
- triaxial compression
- triaxial extension
- pure shear
- mixed nonproportional paths
- principal-axis permutations

### 7.4 Parameterization choice

For training the local constitutive operator, it is usually better to learn with the **effective reduced parameters**:
- \(\bar c\)
- \(\sin\bar\phi\)
- elastic moduli

rather than the raw \((c_0,\phi,\psi,\lambda_\mathrm{SRF})\) only.

You can still store both.  
A good compromise is:

- raw inputs for bookkeeping and FE coupling:
  - \(c_0,\phi,\psi,\lambda_\mathrm{SRF}, E,\nu\)
- reduced inputs for learning:
  - \(\bar c,\sin\bar\phi,G,K,\lambda\)

### 7.5 Labels to save

For each point save:

- raw strain tensor / Voigt strain
- ordered principal trial quantities
- invariants / Haigh–Westergaard coordinates
- material parameters (raw and reduced)
- exact branch label
- exact updated stress
- exact updated principal stresses
- optional tangent
- optional plastic multiplier(s)
- optional yield value and branch-threshold values

This makes multiple model families possible from the same HDF5 file.

---

## 8. Training recipe I would actually use

### 8.1 Input / output representation

**Preferred**
- principal-space or invariant-space inputs
- dimensionless or normalized outputs

Examples:
- normalize stresses by \(\bar c + \epsilon\) or by a characteristic modulus
- normalize strains by a characteristic elastic strain scale
- normalize scalar inputs to \([-1,1]\) or zero mean / unit variance

### 8.2 Loss design

Use a weighted multi-term loss:

\[
\mathcal L
=
w_\sigma \mathcal L_\sigma
+
w_b \mathcal L_\text{branch}
+
w_f \mathcal L_f
+
w_J \mathcal L_\text{invariant}
+
w_T \mathcal L_\text{tangent}
\]

where:
- \( \mathcal L_\sigma \): stress loss (Huber or MAE preferred at first)
- \( \mathcal L_\text{branch} \): cross-entropy for branch label
- \( \mathcal L_f \): yield / admissibility residual
- \( \mathcal L_\text{invariant} \): error on \(p, q,\theta\) or principal values
- \( \mathcal L_\text{tangent} \): optional tangent penalty if available

**Recommendation**
Start with:
- branch CE + Huber stress loss
- then add yield residual only after the baseline is stable

### 8.3 Batching

Use **stratified branch-balanced batches**.  
A batch should contain representative samples from all branches, or at least heavily upweight rare branches.

### 8.4 Curriculum

A very effective curriculum for this problem is:

1. train on elastic + smooth branch
2. add edge branches
3. add apex
4. fine-tune on full balanced data with boundary oversampling

This is often easier than full all-at-once training.

### 8.5 Optimizer / schedule

Good default:
- AdamW
- cosine decay or ReduceLROnPlateau
- early stopping on validation branch-balanced error

### 8.6 Metrics that matter

Do not stop at global MAE. Track:

- total stress MAE / RMSE
- per-branch stress error
- branch classification accuracy / F1
- invariant errors \(p,q,\theta\)
- yield violation rate
- sign errors on admissibility
- FE-level Newton iterations
- FE-level convergence failures
- error in factor of safety (FoS / SRF at collapse)

---

## 9. Why your current training is probably struggling

This diagnosis is specific to the Beremi kernel.

### 9.1 Rotational symmetry is being violated
If you train on raw Voigt components without spectral / invariant preprocessing, the model must rediscover frame-indifference and coaxiality from data. That is unnecessarily hard.

### 9.2 The network is fitting several different maps at once
Elastic, smooth plastic, edges, and apex are qualitatively different regimes.

### 9.3 Rare but critical regions are underrepresented
Edge and apex samples may be numerically rare but dominate failure cases in FE.

### 9.4 Loss scaling is poor
Stress components, invariants, and branch logic live on different scales. Pure MSE often over-focuses large-magnitude outputs and can neglect regime classification.

### 9.5 The dataset may be parameterized in the wrong variables
If the actual constitutive operator uses reduced \((\bar c,\sin\bar\phi)\) but the network only sees raw \((c_0,\phi,\psi,\lambda_\mathrm{SRF})\), you are asking it to learn both reduction and return mapping at once.

### 9.6 Newton robustness is not equivalent to pointwise stress accuracy
A network with low mean stress error can still produce:
- wrong tangent trends
- branch misclassifications near the yield surface
- inadmissible stresses
which then destroy FE convergence.

---

## 10. Reimplementation blueprints from the literature

### 10.1 EPNN-style blueprint

**Goal**  
Improve data efficiency and extrapolation without fully re-deriving plasticity.

**Minimum experiment**
- inputs: invariant/principal trial state + reduced material parameters
- outputs: corrected stress and plastic correction features
- architecture:
  - shared trunk
  - subnets for stress correction and auxiliary variables
- losses:
  - MAE or Huber
  - optional auxiliary plasticity consistency term
- tests:
  - unseen loading paths
  - branch-balanced validation
  - FE insertion in one small 3D slope benchmark

### 10.2 Return-mapping-assisted blueprint

**Goal**  
Keep the classical structure; learn only difficult functions.

**Minimum experiment**
- train `NN_f` for yield
- train `NN_g` for gradient / stress normal / correction direction
- build CPA/closest-point local solver using these surrogates
- compare with exact kernel on:
  - pointwise stress error
  - local admissibility
  - FE convergence

### 10.3 si-PiNet-style blueprint

**Goal**  
Learn stress integration as constrained search.

**Minimum experiment**
- start from the exact trial state
- predict admissible corrected stress subject to prior elastoplastic structure
- include consistency and increment-size robustness tests
- benchmark especially near branch transitions

### 10.4 TANN / iCANN-style blueprint

**Goal**  
Guarantee thermodynamic structure and better extrapolation.

**Minimum experiment**
- choose free-energy / dissipation-based outputs
- derive stresses and internal variables by automatic differentiation
- start on a smoother benchmark first if implementation time is limited
- migrate to Mohr–Coulomb-inspired pressure-sensitive formulation afterward

---

## 11. Suggested experimental roadmap for your project

### Phase 1 — Get a trustworthy baseline
- exact Python port of the local constitutive update
- balanced HDF5 dataset
- branch-aware principal-space MLP
- exact elastic branch retained
- stress + branch loss only

### Phase 2 — Make it FE-usable
- integrate the surrogate into a local constitutive wrapper
- test on one homogeneous 3D slope benchmark
- compare Newton iterations, residuals, and failure SRF

### Phase 3 — Add physics
Pick one of:
- return-mapping-assisted learning
- EPNN-like decomposition
- yield residual / admissibility penalties

### Phase 4 — Research-grade extension
Pick one of:
- si-PiNet-style neural stress integrator
- TANN/iCANN pressure-sensitive constitutive model
- smoother Mohr–Coulomb approximation with explicit branch handling

---


## 11A. Detailed reimplementation notes extracted from the literature

This section isolates implementation details that are concrete enough to reuse.

### 11A.1 EPNN (Eghbalian et al.)

The accessible paper text reports the following details.

**Data and split**
- Synthetic data were generated from material-point simulations of a dilatancy-based constitutive model for sands.
- Data were split into **60% train / 20% validation / 20% test**.

**Preprocessing**
- Inputs and outputs were normalized to **\([-1,1]\)**.

**Loss**
- The authors report that **MAE** was more stable than MSE in their setup and better avoided bias toward outputs of larger magnitude.

**Optimizer**
- **ADAM** was used.
- The accessible text reports learning rates of the order of:
  - \(3\times10^{-4}\) for stress-related subnetworks
  - \(10^{-3}\) for plastic-strain / void-ratio subnetworks

**Architecture**
- The architecture is decomposed into subnetworks rather than one monolithic network.
- In the reported sand case, good-performing subnetworks had on the order of:
  - 3 hidden layers × 60 neurons for some outputs
  - 4 hidden layers × 75 neurons for plastic-strain-related outputs

**Why this is useful for you**
Even though the underlying constitutive model is not Mohr–Coulomb, the transferable lessons are:
- normalize aggressively,
- use MAE/Huber before MSE,
- decompose outputs into physically meaningful subnetworks,
- benchmark on **unseen loading paths**, not only random held-out points.

### 11A.2 He & Semnani (2023) — path-dependent materials for FE analysis

**Problem addressed**
RNN constitutive surrogates can depend strongly on the size of strain increments, which causes FE convergence problems.

**Two key ideas**
1. new model architectures
2. **random walk-based training data generation**

**What to borrow**
If you later move beyond the current static Mohr–Coulomb map to path-dependent soil models, do not generate only neat proportional loading paths. Random-walk loading trajectories are a much better stress test for increment-robustness.

**Why not first choice now**
The current Beremi Mohr–Coulomb kernel is not the best target for a recurrent surrogate because it does not expose a rich hidden-history state like hardening plasticity does.

### 11A.3 Roy et al. (2024) — non-associative Drucker–Prager

**Loss ingredients explicitly emphasized by the paper**
- constitutive-law terms
- Drucker–Prager yield criterion
- non-associative flow rule
- Kuhn–Tucker consistency conditions
- boundary conditions
- transfer learning

**What to borrow**
For your local surrogate, a simplified version of this idea is:
- supervised stress loss,
- yield residual penalty,
- branch/admissibility penalty,
- optional transfer-learning schedule:
  1. pretrain on smooth Drucker–Prager-like data or only smooth Mohr–Coulomb branch,
  2. fine-tune on full nonsmooth Mohr–Coulomb data.

### 11A.4 TB-PeNN (2024)

**Key structural decision**
The network outputs **coefficients of a tensorial constitutive relation**, not the full stress tensor directly.

**What to borrow for Beremi Mohr–Coulomb**
The equivalent choice for your kernel is:
- do not predict six arbitrary stress components first;
- predict low-dimensional scalar quantities in principal/invariant space;
- reconstruct the tensor analytically.

### 11A.5 si-PiNet / PiNet

The accessible metadata and abstracts emphasize:
- solving stress integration as a constrained search problem,
- encoding elastoplastic prior information,
- testing on **Mohr–Coulomb**,
- improved robustness with respect to strain-increment size.

**Minimal reimplementation target**
A pragmatic reproduction in your Python code base does not need the entire original PiNet stack. Start with:
1. compute exact trial state from strain,
2. use a network to predict a correction in principal space,
3. project / constrain the correction with a differentiable admissibility penalty,
4. compare against exact branch formulas.

**This is the closest “research upgrade” over a plain supervised branch-aware MLP.**

### 11A.6 Meng et al. (2025 preprint) — learning the yield and stress gradient

This preprint is especially important because it points to a hybrid approach that remains compatible with traditional FE implementation.

**Reimplementation template**
- Dataset:
  - input state \(x\)
  - exact yield value \(f(x)\)
  - exact stress-gradient / normal information
  - exact corrected stress
- Network 1:
  - \(NN_f(x)\rightarrow \hat f\)
- Network 2:
  - \(NN_n(x)\rightarrow \widehat{\partial f/\partial \sigma}\) or an equivalent correction direction
- Local solver:
  - use \(NN_f\) and \(NN_n\) inside a classical closest-point or cutting-plane loop

**Where it is especially attractive**
If your main pain point is not pointwise error but **Newton breakdown inside the slope solver**, this hybrid route is more promising than direct one-shot stress regression.

---

## 11B. A paper-compatible experiment matrix for your project

Below is a concrete set of experiments that would let you reproduce the spirit of the literature on your exact Mohr–Coulomb operator.

### Experiment 0 — Sanity baseline
**Purpose:** establish a truthful baseline and detect data bugs.

- exact Python constitutive update
- no learning
- verify branch frequencies, stress ranges, and tangent ranges
- plot branch occupancy in invariant space

### Experiment 1 — Raw black-box MLP
**Purpose:** demonstrate why the naive approach struggles.

- input: raw 6 strain components + raw material parameters
- output: raw 6 stress components
- loss: MSE
- expected outcome: okay average fit, poor FE robustness

This experiment is worth keeping because it creates a clean negative baseline.

### Experiment 2 — Principal-space MLP
**Purpose:** isolate the effect of coordinate choice.

- input: ordered principal trial quantities + reduced parameters
- output: updated principal stresses
- loss: MAE or Huber
- reconstruct tensor analytically

Compare directly with Experiment 1.

### Experiment 3 — Branch-aware principal-space MLP
**Purpose:** isolate the effect of branch structure.

- same as Experiment 2
- add branch classifier head
- use branch-balanced batches
- optionally use expert heads per branch

This should be your main baseline.

### Experiment 4 — Exact elastic + learned plastic
**Purpose:** reduce learning burden.

- exact elastic branch
- neural surrogate only for plastic samples
- compare FE convergence against Experiment 3

### Experiment 5 — EPNN-inspired decomposition
**Purpose:** test whether architecture-level plasticity decomposition helps beyond branch awareness.

- decompose outputs into physically meaningful pieces
- preserve branch classifier
- use MAE/Huber
- test on held-out loading paths

### Experiment 6 — Return-mapping-assisted learning
**Purpose:** move toward FE robustness.

- train yield surrogate and correction-direction surrogate
- insert into local iterative corrector
- compare with exact constitutive update under:
  - near-yield states
  - edge/apex states
  - FE Newton iterations

### Experiment 7 — si-PiNet-style constrained stress integration
**Purpose:** research-grade extension.

- formulate corrected stress as constrained neural search
- benchmark sensitivity to strain-increment magnitude
- benchmark Mohr–Coulomb edge/apex robustness

### Experiment 8 — FE benchmark on the 3D slope
**Purpose:** prove practical value.

- choose the homogeneous 3D slope example first
- compare:
  - SRF / factor-of-safety estimate
  - displacement field
  - plastic-zone pattern if available
  - nonlinear iterations and failures
  - wall-clock cost

---

## 11C. Metrics and plots that should appear in your report or paper

The literature often reports only stress errors. For your use case, that is not enough.

### Pointwise constitutive metrics
- MAE / RMSE of stress components
- MAE / RMSE of principal stresses
- error in \(p\), \(q\), and Lode angle
- branch accuracy / confusion matrix
- yield violation rate
- per-branch maximum error
- error concentrated near \(f_\mathrm{tr}\approx 0\)

### Geometry-aware plots
- samples colored by branch in Haigh–Westergaard coordinates
- stress error versus trial-yield value
- stress error versus spectral gap
- edge/apex zoom plots
- parity plots per branch
- histogram of signed yield residual

### FE-level metrics
- number of local constitutive failures
- number of global Newton iterations
- number of line-search activations or continuation failures
- final SRF / FoS error
- displacement norm error
- runtime speedup vs exact constitutive update

---

## 11D. Practical hyperparameter defaults for your first successful run

These are not taken from one single paper; they are a synthesis adapted to your Mohr–Coulomb problem.

### Data
- 1–5 million broad synthetic points if generation is cheap
- but maintain a **balanced training subset** for optimization
- keep dedicated validation sets for:
  - random broad samples
  - near-yield samples
  - edge/apex samples
  - FE-trace samples from actual slope simulations

### Model
- trunk width: 128–256
- depth: 4–6 hidden layers
- GELU / SiLU / LeakyReLU
- branch head + stress head
- optional expert heads per branch

### Optimization
- AdamW
- learning rate \(10^{-3}\) to start
- cosine decay or plateau scheduler
- gradient clipping if using mixed losses
- early stopping on weighted per-branch validation score

### Loss
- Huber for stress
- cross-entropy for branch
- optional yield residual with low weight initially

### Batch construction
- 20–30% elastic
- 20–30% smooth
- 10–20% left edge
- 10–20% right edge
- 10–20% apex
- 10–20% boundary-focused hard samples

The exact percentages can be adjusted to your observed branch frequencies.


## 12. What I would recommend as the best next move for *your* repository

If I had to choose **one** model family to try next for the Beremi code, it would be:

### Recommended first research-quality baseline
**A branch-aware principal-space feedforward surrogate, trained on exact labels from the Python Mohr–Coulomb port, with exact elastic branch retained and strong oversampling near branch transitions.**

Why this first:
- it matches the existing algorithmic geometry,
- it is much easier to debug than RNNs or PINNs,
- it directly addresses the likely cause of your current training problems,
- and it gives a clean baseline for later comparison with EPNN or si-PiNet-like methods.

### Recommended second model
**A return-mapping-assisted surrogate** in the style of Meng et al. (yield network + gradient network inside local stress integration).

Why second:
- it is the most natural bridge from your current exact kernel to a learnable operator,
- it preserves local-solver structure,
- and it is more likely to remain stable in FE than plain direct regression.

### Recommended third model
**si-PiNet / PiNet-style neural stress integration** if your goal is publishable novelty.

---

## 13. Concrete checklist for reproducing literature-style experiments on your Mohr–Coulomb operator

### Data generation
- [ ] use exact Python constitutive port as label generator
- [ ] save raw + reduced material parameters
- [ ] save branch label
- [ ] save trial-state invariants and final-state invariants
- [ ] oversample elastic–plastic boundary
- [ ] oversample edge and apex returns
- [ ] include permutation and near-degenerate principal-value cases

### Modeling
- [ ] begin in principal / invariant coordinates
- [ ] use branch classifier
- [ ] use branch-balanced loss
- [ ] keep elastic branch exact at first
- [ ] reconstruct tensor output analytically

### Training
- [ ] use MAE/Huber instead of plain MSE first
- [ ] balanced mini-batches
- [ ] curriculum by branch complexity
- [ ] monitor per-branch metrics
- [ ] test on unseen parameter combinations and loading paths

### Evaluation
- [ ] pointwise stress error
- [ ] branch confusion matrix
- [ ] yield violation rate
- [ ] invariant errors
- [ ] FE Newton convergence
- [ ] factor-of-safety error in slope benchmark

---

## 14. Paper-by-paper quick notes

### 14.1 Foundational and review papers
- **Dornheim et al.** — broad review of neural-network constitutive modeling; good taxonomy.
- **Fuhg et al.** — review of data-driven constitutive laws for solids.
- **Lefik & Schrefler (2003)** — early finite-element implementation of ANN constitutive model.
- **Ghaboussi et al.** — historical starting point for ANN constitutive surrogates.

### 14.2 Strong local constitutive-surrogate papers
- **Eghbalian et al. (EPNN)** — constitutive structure inside the architecture.
- **Meng et al. (2025 preprint)** — learned yield + gradient + return mapping.
- **Haigh–Westergaard 2025 paper** — coordinate system matters for elastoplastic learning.
- **si-PiNet (2025)** — neural stress integration with prior information.
- **PiNet 2026** — neural numerical integration without labeled data/Jacobian for some models.

### 14.3 Strong physics-encoded constitutive papers
- **TANN (2021)** — thermodynamic consistency.
- **iCANN plasticity extension (2025/2026 line)** — dual potentials, pressure sensitivity, finite-strain direction.
- **TB-PeNN (2024)** — tensor-based physics encoding for soil.
- **GPM-PeNN (2025)** — generalized-plasticity encoded priors.

### 14.4 Sequence-model papers
- **He & Semnani (2023)** — RNN constitutive models for J2 and Drucker–Prager with random-walk training data.
- **Bonatti & Mohr (2022)** — self-consistency issue in RNN constitutive models.
- **Borkowski et al.** — regularized GRU constitutive modeling.
- **Heidenreich & Mohr (2024)** — multi-task RNN constitutive transfer learning.
- **Recursive Bayesian NN for sands (2025)** — sequence learning + uncertainty.

### 14.5 Adjacent PDE-level papers
- **Mohr–Coulomb PINN (2026)** — useful if you later want end-to-end geomechanics surrogates, but not a replacement for the local operator.
- **Drucker–Prager PINN (2024)** — useful bridge for pressure-sensitive constitutive physics.

---

## 15. Final recommendation

For the Mohr–Coulomb constitutive law in `slope_stability_octave`, I recommend this exact progression:

1. **Do not start with RNNs.**
2. Build a **branch-aware principal/invariant-space feedforward surrogate**.
3. Keep the **elastic branch exact**.
4. Use **reduced Davis parameters** as learning inputs.
5. Save **branch labels** and oversample branch transitions.
6. Evaluate in the FE loop, not only at the material point.
7. Once that baseline works, move to either:
   - **return-mapping-assisted learning**, or
   - **si-PiNet-style neural stress integration**.

That is the most defensible path technically and the one most aligned with the current state of the art.

---

## 16. Reference notes and links

This report was prepared from the following web sources and primary papers/repositories that were accessible during the search:

### The target repository and related Mohr–Coulomb implementation
- Beremi/slope_stability_octave repository
- raw files for:
  - 3D homogeneous example
  - material-property assignment
  - constitutive class
  - reduction routine
  - 3D constitutive kernel
- Sysala et al. paper on subdifferential-based return mapping in Mohr–Coulomb plasticity

### Reviews
- Dornheim et al., *Neural Networks for Constitutive Modeling*
- Fuhg et al., review on data-driven constitutive laws for solids
- Wang et al., review of ML-aided granular-material modeling

### Direct constitutive / integration-point surrogate papers
- Eghbalian et al., *A physics-informed deep neural network for surrogate modeling in classical elasto-plasticity*
- Meng et al., *Numerical Implementation of Deep Learning-Based Constitutive Model for Geomaterials Using Return Mapping Algorithms*
- Zhang et al., *si-PiNet: a novel stress integration method for elastoplastic models*
- Zhang et al., *Neural network based numerical integration for elastoplastic constitutive relations*
- Haigh–Westergaard coordinate elastoplastic NN paper (2025)

### Physics-encoded constitutive modeling
- Masi et al., *Thermodynamics-based Artificial Neural Networks for constitutive modeling*
- generalized dual-potential iCANN paper
- Roy et al., *Physics-infused DNN for non-associative Drucker–Prager*
- Wang et al., *Tensor-based physics-encoded neural networks for modeling constitutive behavior of soil*
- GPM-PeNN paper

### Sequence-model papers
- He & Semnani, *Machine learning based modeling of path-dependent materials for finite element analysis*
- Bonatti & Mohr, self-consistency in recurrent constitutive models
- related GRU / LSTM / rBNN soil papers

### Adjacent Mohr–Coulomb PINN
- Yuan et al., *A physics-informed machine learning computational framework for solving Mohr–Coulomb plasticity in geomechanics*

