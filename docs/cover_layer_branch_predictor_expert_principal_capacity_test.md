# Cover Layer Expert-Principal Capacity Test

## Purpose

After the expert-inspired principal-space sampler worked, the next question was whether a larger model with a longer schedule could push the result further.

This report compares:

1. the first successful expert-principal run
2. a larger, longer-trained follow-up on the exact same sampler

## Compared Runs

Smaller expert-principal baseline:

- [cover_layer_branch_predictor_expert_principal.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_expert_principal.md)
- [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_20260315/summary.json)

Larger/longer follow-up:

- [cover_layer_branch_predictor_expert_principal_w1024_d6_long.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_expert_principal_w1024_d6_long.md)
- [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/summary.json)

## Setup

Both runs use:

- feature set: `trial_raw_material`
- model type: `hierarchical`
- generator: `principal_hybrid`
- recipe: `expert_principal`

Differences:

Smaller run:

- width/depth: `256 x 3`
- stage max epochs: `60`
- stage patience: `20`
- plateau patience: `20`
- LBFGS tail: `3`
- elements per cycle: `16384`

Larger run:

- width/depth: `1024 x 6`
- stage max epochs: `80`
- stage patience: `30`
- plateau patience: `25`
- LBFGS tail: `5`
- elements per cycle: `24576`

## Results

### Synthetic test

Smaller run:

- accuracy: `0.9572`
- macro recall: `0.9576`

Larger run:

- accuracy: `0.9658`
- macro recall: `0.9655`

Change:

- accuracy: `+0.0086`
- macro recall: `+0.0079`

### Real test

Smaller run:

- accuracy: `0.8865`
- macro recall: `0.8750`

Larger run:

- accuracy: `0.8928`
- macro recall: `0.8828`

Change:

- accuracy: `+0.0062`
- macro recall: `+0.0078`

### Real per-branch recall

Smaller run:

- elastic: `1.0000`
- smooth: `0.7490`
- left_edge: `0.9101`
- right_edge: `0.8289`
- apex: `0.8871`

Larger run:

- elastic: `1.0000`
- smooth: `0.7500`
- left_edge: `0.9160`
- right_edge: `0.8680`
- apex: `0.8799`

Change:

- elastic: `+0.0000`
- smooth: `+0.0010`
- left_edge: `+0.0059`
- right_edge: `+0.0391`
- apex: `-0.0072`

## Interpretation

Yes, larger and more intense training helped.

But the size of the gain matters:

- the sampler change gave the big breakthrough
- the larger model gave a smaller follow-up gain

So the current picture is:

1. **sampling was the main unlock**
2. **capacity is now a secondary improvement lever**

The most meaningful gain from the larger run was:

- better `right_edge`

The larger run did not materially move `smooth` further:

- `0.7490 -> 0.7500`

So if the next goal is another substantial jump, it is more likely to come from refining the principal-space generator than from simply scaling the MLP again.

## Recommendation

Keep the larger run as the current best classifier checkpoint:

- [cover_layer_branch_predictor_expert_principal_w1024_d6_long.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_expert_principal_w1024_d6_long.md)

But for the next experiment, focus on generator geometry again:

- add a symmetric `smooth_left` / `left_edge` corrective component
- reduce `apex` overproduction
- keep the new upper-tail coverage

That is likely the highest-value next move.
