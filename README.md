# Mohr-Coulomb surrogate repository

This repository provides a Python workflow for replacing the local 3D Mohr-Coulomb constitutive evaluation used in slope-stability finite elements with a neural surrogate.

It contains four main pieces:

1. a Python implementation of the **3D elastic-perfectly-plastic Mohr-Coulomb constitutive update** with the same branch structure as the Octave/MATLAB repository (`elastic`, `smooth`, `left edge`, `right edge`, `apex`),
2. a **training-data generator** that draws material states and strain states, evaluates the exact constitutive law, and saves the result to **HDF5**,
3. a **surrogate training pipeline** built around PyTorch,
4. visualization and evaluation scripts for checking whether the network learned the constitutive map well enough to be used as a drop-in local operator.

## Why this repository is structured this way

The constitutive relation in a finite element slope-stability solve is a **pointwise nonlinear map**

\[
(\varepsilon, \text{material}) \mapsto \sigma
\]

applied independently at a very large number of integration points. That makes it a good candidate for a learned local surrogate, but it is **not** the same problem as learning a global PDE solution operator. For that reason this repository uses an **invariant-aware pointwise MLP** as the main surrogate, while also including a raw-input baseline.

## Repository layout

```text
src/mc_surrogate/
  materials.py          # elastic moduli + Davis reduction
  voigt.py              # Voigt / tensor conversions
  mohr_coulomb.py       # 3D constitutive update
  sampling.py           # dataset generation logic
  data.py               # HDF5 I/O
  models.py             # surrogate architectures + feature builders
  training.py           # training / checkpointing / inference helpers
  inference.py          # constitutive-style wrapper around a checkpoint
  viz.py                # plotting helpers
scripts/
  generate_dataset.py
  train_surrogate.py
  evaluate_model.py
  plot_training_history.py
  plot_path_comparison.py
  inspect_dataset.py
  compare_with_octave.py
docs/
  mohr_coulomb.md
  neural_nets.md
  dataset_creation.md
tests/
configs/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or, without installation:

```bash
PYTHONPATH=src python scripts/inspect_dataset.py path/to/file.h5
```

## Quick start

### 1) Generate a dataset

```bash
python scripts/generate_dataset.py \
  --output data/mc_train.h5 \
  --n-samples 50000 \
  --seed 7
```

### 2) Train the recommended surrogate

```bash
python scripts/train_surrogate.py \
  --dataset data/mc_train.h5 \
  --run-dir runs/principal_v1 \
  --model-kind principal \
  --epochs 150 \
  --batch-size 2048
```

### 3) Evaluate it

```bash
python scripts/evaluate_model.py \
  --dataset data/mc_train.h5 \
  --checkpoint runs/principal_v1/best.pt \
  --output-dir runs/principal_v1/eval
```

### 4) Plot the training curves

```bash
python scripts/plot_training_history.py \
  --history runs/principal_v1/history.csv \
  --output runs/principal_v1/history.png
```

### 5) Compare a loading path

```bash
python scripts/plot_path_comparison.py \
  --checkpoint runs/principal_v1/best.pt \
  --output runs/principal_v1/path.png
```

## Recommended surrogate

The default recommendation in this repository is the **principal-stress network**:

- input: ordered principal strains plus reduced material parameters,
- output: ordered principal stresses,
- reconstruction: rotate the principal stresses back to the global tensor basis using the strain eigenvectors.

That choice bakes in the two most important structural facts of the constitutive law:

- the law is **isotropic**, so the response depends on invariants / principal values rather than on coordinate-frame-specific components,
- for this return mapping the corrected stress is **coaxial** with the trial strain tensor.

The raw-input network is included as a baseline, but the principal network is the one to try first.

## Exact constitutive implementation

`src/mc_surrogate/mohr_coulomb.py` implements the closed-form branch logic used by the original constitutive routine:

- elastic,
- smooth return to the Mohr-Coulomb surface,
- left edge,
- right edge,
- apex.

The stress update follows the ordered-principal-value formulation. A numerical tangent by centered finite differences is available when needed for diagnostics or tangent-supervised experiments.

## HDF5 dataset contents

The generator writes:

- `strain_eng` — engineering strain vectors,
- `stress` — exact stress vectors,
- `strain_principal` — principal strains,
- `stress_principal` — principal stresses of the exact response,
- `eigvecs` — principal directions of the trial strain tensor,
- `branch_id` — constitutive branch label,
- `plastic_multiplier` — branch-specific plastic multiplier candidate realized by the return mapping,
- `f_trial` — trial yield value,
- `material_raw` — unreduced geotechnical material parameters,
- `material_reduced` — reduced constitutive parameters actually used by the constitutive law,
- `tangent` — optional numerical tangent.

## Notes on validation against Octave

If you have a local clone of the original repository and Octave installed, you can compare this Python implementation against the original code:

```bash
python scripts/compare_with_octave.py \
  --octave-repo /path/to/slope_stability
```

If you also have a generated HDF5 dataset, you can compare on a random subset from that file:

```bash
python scripts/compare_with_octave.py \
  --octave-repo /path/to/slope_stability \
  --dataset data/mc_train.h5 \
  --max-points 1000
```

## Documentation

- [Mohr-Coulomb constitutive law](docs/mohr_coulomb.md)
- [Neural-network surrogate design](docs/neural_nets.md)
- [Dataset generation rationale](docs/dataset_creation.md)
- [Repository entry point and maintenance guide](docs/repo_entrypoint.md)
- [Expert briefing on current surrogate status (2026-03-23)](docs/expert_briefing_20260323.md)

## Reference implementation and theory

The implementation here is informed by:

- the constitutive files in the `sysala/slope_stability` repository, especially `constitutive_problem_3D.m`, `reduction.m`, and `potential_3D.m`,
- the paper *Subdifferential-based implicit return-mapping operators in Mohr-Coulomb plasticity* by Sysala, Čermák, and Ligurský,
- the benchmark parameter ranges used in the 3D homogeneous and heterogeneous slope-stability scripts in the same repository.

For the local surrogate design, the main practical conclusion is:

- use a **pointwise invariant-aware network** first,
- do **not** start with a field-to-field neural operator unless the problem changes from a local constitutive map to a global PDE map.

## Current scope and limitations

This repository is designed to accelerate the **local** constitutive evaluation. It does **not** replace the full FE solve, global equilibrium iterations, or continuation logic.

Also, the tangent in this repository is numerical finite difference by default. That is sufficient for dataset diagnostics and tangent-supervised experiments, but it is not a hand-ported analytic consistent tangent from the original code.
