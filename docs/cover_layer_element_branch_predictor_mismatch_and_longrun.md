# Cover Layer Direct Element Model: Mismatch Check and Longer Training

## Goal

This note answers two concrete questions about the direct raw-element branch model:

1. is there a likely normalization / input mismatch bug?
2. does simply training longer help?

Model family under discussion:

\[
(\text{coords}_{10 \times 3}^{std}, \text{disp}_{10 \times 3}^{std}) \rightarrow \text{branches}_{11 \times 5}
\]

## 1. Normalization / Input Mismatch Check

The input diagnostics are in [input_diagnostics.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/input_diagnostics.json).

### Main result

There is **some** real-vs-synthetic shift, but it does **not** look like a catastrophic normalization bug.

Selected numbers:

| Split | mean `|z|` | p95 `|z|` | p99 `|z|` | frac `|z|>5` |
|---|---:|---:|---:|---:|
| synthetic calibration | `0.6276` | `2.0182` | `3.8312` | `0.0026` |
| real val | `0.7195` | `2.4924` | `4.6229` | `0.0072` |
| real test | `0.7492` | `2.5634` | `5.0027` | `0.0100` |
| synthetic holdout | `0.6202` | `2.0111` | `3.8519` | `0.0034` |

Interpretation:

- real element inputs are somewhat shifted relative to synthetic calibration
- but the shift is moderate, not explosive
- almost no features fall into extreme z-score ranges

So a simple broken scaling path is **not** the best explanation.

## 2. Synthetic Pattern Complexity Check

I also checked how diverse the element-level label patterns are.

Using the current seeded branch-balanced generator:

- unique real test branch patterns: `335`
- unique synthetic branch patterns: `2749`

That matters a lot.

The direct raw-element model is trying to learn a much richer structured output space than the old pointwise model, and the synthetic distribution is highly diverse at the full 11-point pattern level.

This does **not** prove a bug, but it explains why the direct model is a much harder learning problem.

## 3. Tiny Synthetic Overfit Check

I trained the direct raw-element MLP on a tiny synthetic sample of `256` elements and evaluated it on the same sample.

### `128 x 4`

- epoch 1: accuracy `0.2895`, pattern accuracy `0.0000`
- epoch 100: accuracy `0.7462`, pattern accuracy `0.3252`
- epoch 200: accuracy `0.8222`, pattern accuracy `0.4660`
- epoch 400: accuracy `0.8795`, pattern accuracy `0.6068`

### `1024 x 8`

- epoch 1: accuracy `0.3208`, pattern accuracy `0.0000`
- epoch 100: accuracy `0.6567`, pattern accuracy `0.1748`
- epoch 200: accuracy `0.7361`, pattern accuracy `0.4175`
- epoch 400: accuracy `0.7718`, pattern accuracy `0.4709`

Interpretation:

- the direct model path is **not completely broken**
- it can learn a small synthetic sample
- but it is still much harder to optimize than the pointwise branch predictor

So this points away from a total label/input mismatch bug and more toward:

- hard optimization
- weak inductive bias
- generator difficulty at the full element-pattern level

## 4. Longer Training Run

I also ran a much longer `128 x 4` direct model:

- output: [summary.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_w128_d4_long_20260314/summary.json)
- curve comparison: [long_vs_short.png](../experiment_runs/real_sim/cover_layer_element_branch_predictor_w128_d4_long_20260314/long_vs_short.png)

Compared to the shorter `128 x 4` run:

| Metric | short `128x4` | long `128x4` |
|---|---:|---:|
| real val accuracy | `0.3638` | `0.3638` |
| real val macro recall | `0.3519` | `0.3519` |
| real val pattern accuracy | `0.1738` | `0.1738` |
| real test accuracy | `0.3944` | `0.3944` |
| real test macro recall | `0.3505` | `0.3505` |
| real test pattern accuracy | `0.2305` | `0.2305` |
| synthetic accuracy | `0.0952` | `0.0952` |
| synthetic macro recall | `0.2876` | `0.2876` |
| synthetic pattern accuracy | `0.0000` | `0.0000` |

The best checkpoint stayed the same early one:

- `best_epoch = 12`

So simply training longer did **not** help.

## Conclusion

At this point, the evidence says:

1. there is **not** a strong sign of a basic normalization/input bug
2. the direct raw-element model can learn in principle, because it overfits a tiny synthetic sample
3. the current full training setup does **not** improve with more epochs
4. the main issue is likely structural:
   - the direct MLP has weak inductive bias for this task
   - the synthetic full-pattern space is very diverse
   - the current generator still mismatches real branch occupancy

So the next move should probably **not** be another plain-MLP long run.

The next useful move for this direct-family idea is a more structured model, not just more time.

## Artifacts

- normalization diagnostics: [input_diagnostics.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/input_diagnostics.json)
- initial direct run report: [cover_layer_element_branch_predictor_initial.md](./cover_layer_element_branch_predictor_initial.md)
- short `128x4` run: [summary.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_w128_d4_20260314/summary.json)
- long `128x4` run: [summary.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_w128_d4_long_20260314/summary.json)
