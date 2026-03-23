# Cover Layer Element-Branch Predictor Diagnostics

## Question

For the direct raw-element model

\[
(\text{coords}_{10 \times 3}^{std}, \text{disp}_{10 \times 3}^{std}) \rightarrow \text{branches}_{11 \times 5}
\]

is the failure likely caused by:

- broken normalization / input scaling
- too-large network size
- or something more structural?

## Input / Normalization Check

Diagnostics are stored in [input_diagnostics.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/input_diagnostics.json).

Key findings:

### Real inputs are not wildly off the calibration scale

| Split | mean `|z|` | p95 `|z|` | p99 `|z|` | frac `|z| > 5` |
|---|---:|---:|---:|---:|
| synthetic calibration | `0.6276` | `2.0182` | `3.8312` | `0.0026` |
| real val | `0.7195` | `2.4924` | `4.6229` | `0.0072` |
| real test | `0.7492` | `2.5634` | `5.0027` | `0.0100` |
| synthetic holdout | `0.6202` | `2.0111` | `3.8519` | `0.0034` |

Interpretation:

- real inputs are shifted relative to synthetic calibration, but not catastrophically
- there are very few extreme normalized outliers
- so the direct model failure does **not** look like a simple normalization bug

### Coordinate features look well behaved

- canonical coordinate range in calibration: about `[-1.03, 1.21]`
- canonical displacement range in calibration: about `[-48.9, 47.3]`

So the inputs are not numerically exploding.

## But the Branch Distribution Is Still Mismatched

From the same diagnostics:

- real test branch fractions:
  - elastic `0.2653`
  - smooth `0.1729`
  - left_edge `0.2356`
  - right_edge `0.1559`
  - apex `0.1703`
- synthetic holdout branch fractions:
  - elastic `0.0291`
  - smooth `0.2899`
  - left_edge `0.2727`
  - right_edge `0.0834`
  - apex `0.3249`

So even for the direct model, the synthetic generator is still not matching the real branch occupancy well.

## Smaller Network Check

I also ran the same direct element model with:

- width `128`
- depth `4`

Run:

- [summary.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_w128_d4_20260314/summary.json)

Comparison plot:

![Size Comparison](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/size_compare.png)

## Size Comparison

| Metric | `w1024 d8` | `w128 d4` |
|---|---:|---:|
| real val accuracy | `0.3597` | `0.3638` |
| real val macro recall | `0.3550` | `0.3519` |
| real val pattern accuracy | `0.1660` | `0.1738` |
| real test accuracy | `0.3828` | `0.3944` |
| real test macro recall | `0.3565` | `0.3505` |
| real test pattern accuracy | `0.2188` | `0.2305` |
| synthetic accuracy | `0.0818` | `0.0952` |
| synthetic macro recall | `0.3018` | `0.2876` |
| synthetic pattern accuracy | `0.0000` | `0.0000` |

## Interpretation

The smaller model is **not materially better**.

It moves a few metrics slightly:

- slightly higher real test accuracy
- slightly lower macro recall
- still essentially zero synthetic pattern accuracy

So the problem is **not** that the original direct model was simply too large.

## Bottom Line

The evidence so far says:

1. **normalization/input scaling is not the main failure**
2. **network size is not the main failure**
3. the direct raw-element MLP is failing more structurally:
   - it does not fit the synthetic task well
   - and the synthetic branch distribution is still mismatched to real data

So the next useful improvement for this direct family is probably not:

- “normalize differently”
- or “make it smaller”

It is more likely one of:

- a more structured raw-element architecture
- or a better synthetic generator for the direct model family

## Artifacts

- direct model report: [cover_layer_element_branch_predictor_initial.md](./cover_layer_element_branch_predictor_initial.md)
- normalization diagnostics: [input_diagnostics.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/input_diagnostics.json)
- size comparison plot: [size_compare.png](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/size_compare.png)
