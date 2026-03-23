# Cover Layer Element-Branch Predictor: Initial Run

## Goal

This is the first run of the **direct model you originally asked for**:

- fixed material: `cover_layer` only
- training data: synthetic only
- validation/test: real held-out calls
- model input: canonicalized element node coordinates + canonicalized nodal displacements
- model output: `11 x 5` branch logits for the 11 integration points

So this model is:

\[
(\text{coords}_{10 \times 3}^{std}, \text{disp}_{10 \times 3}^{std}) \rightarrow \text{branches}_{11 \times 5}
\]

and does **not** use material values or per-integration-point trial features.

## Architecture

Initial baseline:

- model: `ElementBranchMLP`
- input:
  - canonicalized coords: `10 x 3`
  - canonicalized displacements: `10 x 3`
  - flattened total input dimension: `60`
- output:
  - `11 x 5` logits
- width: `1024`
- depth: `8`
- activation: `GELU`

This is the simplest direct raw-element baseline for your idea.

## Training Setup

- script: [train_cover_layer_element_branch_predictor.py](../scripts/experiments/train_cover_layer_element_branch_predictor.py)
- output dir: `../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314`
- synthetic generator: `seeded_local_noise_branch_balanced`
- synthetic elements per epoch: `2500`
- synthetic holdout elements: `1500`
- fit calls: `12`
- real val calls: `4`
- real test calls: `4`
- staged schedule:
  - cycles: `3`
  - batch sizes: `32 -> 64 -> 128 -> 256 -> 512`
  - plateau factor: `0.5`
  - min LR: `1e-6`

## Convergence

![Training History](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/training_history.png)

What to notice:

- the model does **not** fit the synthetic holdout well
- because of that, the problem is not synthetic-to-real transfer alone
- the current direct raw-element baseline is underfitting or poorly structured for this task

## Results

Best checkpoint:

- real val accuracy: `0.3597`
- real val macro recall: `0.3550`
- real val exact element pattern accuracy: `0.1660`
- real test accuracy: `0.3828`
- real test macro recall: `0.3565`
- real test exact element pattern accuracy: `0.2188`
- synthetic holdout accuracy: `0.0818`
- synthetic holdout macro recall: `0.3018`
- synthetic holdout exact element pattern accuracy: `0.0000`

Per-branch real test recall:

- `elastic`: `0.9572`
- `smooth`: `0.0000`
- `left_edge`: `0.0023`
- `right_edge`: `0.8178`
- `apex`: `0.0052`

## Comparison To The Pointwise Branch Predictor

Current best pointwise branch predictor for comparison:

- real test accuracy: `0.6000`
- real test macro recall: `0.5741`
- synthetic holdout accuracy: `0.9763`
- synthetic holdout macro recall: `0.9827`

So this first direct raw-element baseline is **much worse** than the pointwise branch predictor, even on synthetic data.

## Interpretation

This result is still useful.

It tells us:

1. the raw `coords + disp -> 11 x 5` formulation is now implemented correctly
2. but the **first plain MLP baseline is not sufficient**
3. the bottleneck is not only domain transfer, because the model also fails badly on synthetic holdout

That means the next step for this model family should not be “same architecture, just a little longer.”

The likely issue is inductive bias:

- this direct model is asking one MLP to learn FE kinematics and constitutive branch structure at once
- that is much harder than the pointwise feature-based classifier

## Practical Conclusion

We have now finally tested the direct model family you originally wanted.

The first baseline answer is:

- **the idea is implemented**
- **the plain direct MLP baseline is not good enough**

So if we continue with this family, the next move should be a more structured element model, for example:

- node-wise encoder + element-context head
- or an FE-inspired architecture that keeps raw coords/displacements as input but uses stronger structure internally

## Artifacts

- summary: [summary.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/summary.json)
- training history: [training_history.png](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/training_history.png)
- comparison snapshot: [comparison.json](../experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314/comparison.json)
