# Cover-Layer Best-Candidate Model Card

This model card documents the current best deployable cover-layer surrogate route:
`edge_hard_mined_gate_raw_threshold_t0.65`.

It is intended to be readable on its own and sharable with collaborators who were not involved in the training loop.

## 1. At A Glance

- **Target operator:** fixed-material cover-layer constitutive surrogate from engineering strain `E in R^6` to stress `S in R^6`
- **Winning route:** raw gate with confidence threshold `0.65`, baseline fallback, exact elastic trial for branch `elastic`, frozen `smooth/apex` experts, hard-mined `left_edge/right_edge` experts
- **Primary validation target:** real exact holdout from the exported slope-stability constitutive calls
- **Status:** best local candidate so far; strong enough for a solver shadow test, not yet proven safe for constitutive replacement

## 2. Material Scope

This card is for the **cover-layer** material family only, using the raw material family recovered from the slope model:

| Material | c0 | phi [deg] | psi [deg] | Young | Poisson |
|---|---:|---:|---:|---:|---:|
| cover_layer | 15.0 | 30.0 | 0.0 | 10000.0 | 0.33 |

Important nuance: the model family is fixed to this raw material family, but the reduced material inputs still vary with the strength-reduction factor captured in the real export. So the last five features are **not** constant over the dataset.

## 3. System Architecture

![Architecture overview](../experiment_runs/real_sim/cover_layer_model_card_20260314/architecture_overview.png)

### 3.1 Inputs

Every neural component in the winning route uses the same raw feature vector of length `11`:

| Index | Feature | Notes |
|---|---|---|
| 1-6 | `[e11, e22, e33, g12, g13, g23]` | engineering strain in Voigt form |
| 7 | `log(c_bar)` | reduced cohesion |
| 8 | `atanh(sin(phi_bar))` | reduced friction term |
| 9 | `log(G)` | shear modulus |
| 10 | `log(K)` | bulk modulus |
| 11 | `log(lambda)` | Lamé parameter |

### 3.2 Neural Components

| Component | Role | Input Dim | Hidden Width | Residual Blocks | Output | Activation | Parameters |
|---|---|---:|---:|---:|---|---|---:|
| `baseline_raw_branch` | fallback stress model + auxiliary branch head | 11 | 1024 | 6 | stress(6) + branch logits(5) | GELU | 12,643,339 |
| `gate_raw` | route selector | 11 | 512 | 4 | branch logits(5) | GELU | 2,118,149 |
| `smooth` expert | plastic expert for `smooth` | 11 | 512 | 6 | stress(6) | GELU | 3,173,382 |
| `apex` expert | plastic expert for `apex` | 11 | 512 | 6 | stress(6) | GELU | 3,173,382 |
| `left_edge` expert | plastic expert for `left_edge` | 11 | 512 | 6 | stress(6) | GELU | 3,173,382 |
| `right_edge` expert | plastic expert for `right_edge` | 11 | 512 | 6 | stress(6) | GELU | 3,173,382 |

### 3.3 Residual Block

All MLP components use the same tabular residual block:

`LayerNorm(width) -> Linear(width,width) -> GELU -> Dropout(0) -> Linear(width,width) -> residual add -> LayerNorm(width) -> GELU`

### 3.4 Final Routing Logic

For each sample:

1. Compute raw features.
2. Run `baseline_raw_branch` and `gate_raw`.
3. If `max softmax(gate_raw) < 0.65`, return **baseline stress**.
4. Else, route by the predicted branch:
   - `elastic`: return **exact elastic trial stress**
   - `smooth`: return **smooth expert**
   - `left_edge`: return **left_edge hard-mined expert**
   - `right_edge`: return **right_edge hard-mined expert**
   - `apex`: return **apex expert**

Total parameters stored in the current deployment bundle: **27,455,016**.

## 4. How It Was Trained

This final route is not one monolithic end-to-end training run. It is a staged system assembled from separately trained components.

### 4.1 Datasets

| Dataset | Path | Total | Train | Val | Test | Purpose |
|---|---|---:|---:|---:|---:|---|
| real exact cover-layer | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_single_material_20260313/cover_layer_full_real_exact_256.h5` | 141,824 | 99,328 | 21,248 | 21,248 | main real supervision |
| synthetic U/B holdout | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_single_material_20260313/cover_layer_full_synthetic_holdout.h5` | 21,248 | 0 | 0 | 21,248 | auxiliary generalization check |
| left_edge branch dataset | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/datasets/left_edge_dataset.h5` | 34,662 | 24,285 | 5,398 | 4,979 | edge-expert retraining |
| right_edge branch dataset | `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/datasets/right_edge_dataset.h5` | 24,731 | 17,293 | 3,843 | 3,595 | edge-expert retraining |

Training supervision was **predominantly real exact data**, not synthetic data:

- `baseline_raw_branch`: trained on the real exact cover-layer dataset
- `gate_raw`: trained on the same real exact cover-layer dataset
- `smooth` and `apex` experts: trained on branch-filtered real exact datasets
- `left_edge` and `right_edge` hard-mined experts: trained on branch-filtered real exact datasets plus mined replay / local augmentation

The synthetic `U/B` dataset was **not** used as the main supervision source for the winning route. It is kept as an auxiliary test of domain coverage.

### 4.2 Training Stages

| Stage | Component(s) | Data | Optimizer / Schedule | Notes |
|---|---|---|---|---|
| A | `baseline_raw_branch` | real exact cover-layer | AdamW, plateau LR `3e-4 -> 1e-6`, LBFGS tail | direct stress + branch auxiliary head |
| B | `smooth`, `left_edge`, `right_edge`, `apex` experts | branch-filtered real exact | AdamW, plateau LR `3e-4 -> 1e-6`, LBFGS tail | first branch-specialized experts |
| C | `gate_raw` | real exact cover-layer | CE training on raw features | threshold later selected by validation MAE |
| D | `left_edge` + `right_edge` hard-mined experts | branch-filtered real exact + mined replay | AdamW, 3 cycles over batch `64->128->256->512->1024`, plateau drops by `0.5`, floor `1e-6` | this is the tail-safety phase |

### 4.3 Baseline Training Configuration

`baseline_raw_branch` config:

- model kind: `raw_branch`
- width/depth: `1024 x 6`
- epochs cap: `3000`
- batch size: `4096`
- initial LR: `0.0003`
- scheduler: `plateau`
- plateau factor / patience: `0.5` / `120`
- patience: `700`
- min LR: `1e-06`
- weight decay: `1e-05`
- LBFGS tail: `8` epochs at LR `0.25`
- branch loss weight: `0.1`

### 4.4 Gate Training Configuration

`gate_raw` config:

- feature kind: `raw`
- width/depth: `512 x 4`
- input dim: `11`
- objective: branch classification with macro-recall-driven checkpointing
- selected threshold: `0.65` from validation stress MAE

### 4.5 Hard-Mined Edge Retraining

`left_edge` hard-mined config:

- width/depth: `512 x 6`
- cycles: `3`
- batch schedule: `[64, 128, 256, 512, 1024]`
- base LR / floor: `0.001` / `1e-06`
- plateau patience: `20`
- stage patience: `80`
- replay ratio: `0.35`
- local noise scale: `0.05`

`right_edge` hard-mined config is the same structure with a different seed (`2024`).

Mining table sizes used in the tail-safety phase:

| Mining Table | Rows | Val Rows | Test Rows | Relative-Error Threshold |
|---|---:|---:|---:|---:|
| left_edge | 3,618 | 1,817 | 1,801 | 0.8549 |
| right_edge | 2,566 | 1,276 | 1,290 | 0.9996 |

Important caveat: these mining tables were intentionally built from the fixed validation/test holdout of the current route. That made the tail-safety experiment effective, but it also means the resulting gains are **holdout-informed** and should be treated as optimistic until they are confirmed in a solver shadow test or on a fresh untouched split.

## 5. Training Convergence

![Component convergence](../experiment_runs/real_sim/cover_layer_model_card_20260314/component_convergence.png)

How to read this figure:

- **Top-left:** the baseline’s validation stress MSE falls steadily while branch accuracy rises. This is the original direct model that all routed variants fall back to.
- **Top-right:** the raw gate keeps improving macro recall while validation CE loss drops. This is why the final route uses `gate_raw` rather than `gate_trial`.
- **Bottom-left / bottom-right:** the tail-safety result is visible directly. The `edge_hard_mined` curves collapse the validation weighted RMSE for both edge branches, while `edge_control` and `edge_tail_weighted` plateau much higher.

What matters most in this convergence plot is not just that the losses go down, but **which curves separate**:

- the baseline converges to a strong general fallback
- the gate converges to a reliable branch selector
- the hard-mined edge experts are the only variant that materially changes the edge-tail regime

## 6. Validation Summary

### 6.1 Real Holdout: Winner vs Baseline

![Real breakdown](../experiment_runs/real_sim/cover_layer_model_card_20260314/real_breakdown.png)

| Metric | baseline_reference | winner route | Relative improvement |
|---|---:|---:|---:|
| stress MAE | 8.2331 | 2.8081 | 65.9% |
| stress RMSE | 22.9041 | 16.0203 | 30.1% |
| p90 relative error | 0.8746 | 0.3590 | 59.0% |
| p99 relative error | 3.7585 | 1.5390 | 59.1% |
| edge combined MAE | 11.9159 | 3.2216 | 73.0% |
| branch accuracy | 0.8255 | 0.9559 | 13.0 pts |

### 6.2 Synthetic Holdout: Winner vs Baseline

![Synthetic breakdown](../experiment_runs/real_sim/cover_layer_model_card_20260314/synthetic_breakdown.png)

| Metric | baseline_reference | winner route | Relative improvement |
|---|---:|---:|---:|
| stress MAE | 19.4116 | 12.4332 | 35.9% |
| stress RMSE | 156.1580 | 146.1537 | 6.4% |
| branch accuracy | 0.7536 | 0.9396 | 18.6 pts |

Interpretation:

- The real holdout is the primary target, and the winner is dramatically better there.
- The synthetic holdout also improves, but much less cleanly, especially for `left_edge`. This says the route is strongly tuned to the real exported cover-layer distribution, not universally solved across all synthetic regimes.

## 7. Detailed Real Validation

### 7.1 Winner Route Figures

- parity: ![winner real parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/parity.png)
- relative-error CDF: ![winner real cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/relative_error_cdf.png)
- error vs stress magnitude: ![winner real mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/error_vs_magnitude.png)
- branch confusion: ![winner real branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/branch_confusion.png)

### 7.2 Baseline Reference Figures

- parity: ![baseline real parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/parity.png)
- relative-error CDF: ![baseline real cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/relative_error_cdf.png)
- error vs stress magnitude: ![baseline real mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/error_vs_magnitude.png)
- branch confusion: ![baseline real branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/branch_confusion.png)

### 7.3 Real Holdout Error Tables

#### Per-Branch Stress MAE

| Branch | baseline_reference | winner route |
|---|---:|---:|
| elastic | 6.3507 | 0.1591 |
| smooth | 6.2732 | 5.0757 |
| left_edge | 11.7174 | 2.9032 |
| right_edge | 12.1908 | 3.6626 |
| apex | 4.4876 | 0.4209 |

#### Stress-Magnitude-Bin Sample MAE

| Stress bin | baseline_reference | winner route |
|---|---:|---:|
| 6 to 37 | 17.9474 | 6.6302 |
| 37 to 45 | 15.7926 | 2.6528 |
| 45 to 81 | 16.5599 | 4.4908 |
| 81 to 175 | 28.9796 | 11.5347 |
| 175 to 1473 | 89.7537 | 36.3578 |


#### Per-Branch Mean Relative Error

| Branch | baseline_reference | winner route |
|---|---:|---:|
| elastic | 0.4951 | 0.0090 |
| smooth | 0.3074 | 0.1652 |
| left_edge | 0.5620 | 0.1977 |
| right_edge | 0.6476 | 0.2852 |
| apex | 0.3226 | 0.0304 |


#### Worst Real Holdout Calls For The Winner

| Call | N | Component MAE | Sample MAE | Mean Relative |
|---|---:|---:|---:|---:|
| call_000531 | 256 | 7.3776 | 22.8088 | 0.2448 |
| call_000510 | 256 | 6.8128 | 21.1056 | 0.1700 |
| call_000464 | 256 | 6.5531 | 20.5764 | 0.1640 |
| call_000537 | 256 | 5.9801 | 18.7235 | 0.2244 |
| call_000404 | 256 | 5.6824 | 18.0349 | 0.1643 |
| call_000450 | 256 | 5.6216 | 17.7110 | 0.1939 |
| call_000534 | 256 | 5.5686 | 17.4182 | 0.1868 |
| call_000538 | 256 | 5.2287 | 16.4679 | 0.1820 |
| call_000449 | 256 | 5.2022 | 16.4725 | 0.2669 |
| call_000471 | 256 | 5.1485 | 15.9177 | 0.1861 |


## 8. Detailed Synthetic Validation

### 8.1 Winner Route Figures

- parity: ![winner synth parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/parity.png)
- relative-error CDF: ![winner synth cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/relative_error_cdf.png)
- error vs stress magnitude: ![winner synth mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/error_vs_magnitude.png)
- branch confusion: ![winner synth branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/branch_confusion.png)

### 8.2 Baseline Reference Figures

- parity: ![baseline synth parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/parity.png)
- relative-error CDF: ![baseline synth cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/relative_error_cdf.png)
- error vs stress magnitude: ![baseline synth mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/error_vs_magnitude.png)
- branch confusion: ![baseline synth branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/branch_confusion.png)

### 8.3 Synthetic Holdout Per-Branch Stress MAE

| Branch | baseline_reference | winner route |
|---|---:|---:|
| elastic | 7.1941 | 0.2190 |
| smooth | 10.6609 | 4.9318 |
| left_edge | 117.1670 | 99.6440 |
| right_edge | 4.7829 | 4.0987 |
| apex | 4.1992 | 0.4189 |

## 9. What To Tell Others

If you need a short summary for collaborators:

- The current best cover-layer surrogate is **not a single network**. It is a routed system composed of a strong baseline model, a dedicated raw gate, and branch-specific experts.
- It was trained mostly on **real exact relabeled constitutive-call data**, not on purely synthetic samples.
- The biggest gain came from **hard-mined retraining of the `left_edge` and `right_edge` experts**.
- On the real holdout, it improves over the direct baseline from MAE/RMSE `8.2331/22.9041` to `2.8081/16.0203`.
- The route is strong enough for a **solver shadow test**, but not yet proven safe for constitutive replacement because the hard-mining phase was holdout-informed.

## 10. Caveats

1. The strongest edge-expert gains came from holdout-informed mining. This makes the result useful, but optimistic.
2. The real holdout is where the route clearly wins. The synthetic holdout still shows large `left_edge` errors, so the route is not a universal fix over all synthetic regimes.
3. The current evaluation script computes all experts in batch and then routes for convenience. A deployment implementation should route more efficiently.
4. This card documents the **best local candidate**, not a solver-validated constitutive replacement.

## 11. Key Artifact Paths

- winning route metrics: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes_summary.json`
- tail-safety execution report: `/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_tail_safety_execution.md`
- baseline checkpoint: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_single_material_20260313/baseline_raw_branch/best.pt`
- gate checkpoint: `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_gate_experiments_20260313/gate_raw/best.pt`
- left/right hard-mined checkpoints:
  - `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/left_edge_edge_hard_mined/best.pt`
  - `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_tail_safety_20260313/experts/right_edge_edge_hard_mined/best.pt`
- frozen smooth/apex checkpoints:
  - `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_experts_20260313/expert_smooth/best.pt`
  - `/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_experts_20260313/expert_apex/best.pt`
