# Cover Layer Branch Predictor Hierarchical Iteration

## Purpose

This experiment takes the current best pointwise branch predictor and changes only two things:

- classifier head: flat `5`-way classifier -> hierarchical `elastic vs plastic` + `plastic branch`
- synthetic sampling recipe: default branch-balanced -> smooth-focused recipe with extra `smooth` and `smooth/right_edge` emphasis

Everything else stays aligned with the current winner so the comparison is fair:

- feature set: `trial_raw_material`
- hidden width / depth: `256 x 3`
- synthetic-only training
- regenerated synthetic training data every epoch
- fixed synthetic validation/test
- same four LR cycles, batch ramp, and LBFGS tail

Baseline for comparison:

- [cover_layer_branch_predictor_trial_raw_material.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_trial_raw_material.md)

New run:

- [cover_layer_branch_predictor_hierarchical_smooth_focus.md](/home/beremi/repos/mohr_coulomb_NN/docs/cover_layer_branch_predictor_hierarchical_smooth_focus.md)

## Run Artifacts

- output directory: [cover_layer_branch_predictor_hierarchical_smooth_focus_20260315](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_hierarchical_smooth_focus_20260315)
- summary: [summary.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_hierarchical_smooth_focus_20260315/summary.json)
- training log: [train.log](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_hierarchical_smooth_focus_20260315/train.log)
- history: [history.json](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/real_sim/cover_layer_branch_predictor_hierarchical_smooth_focus_20260315/history.json)

## Results

### Synthetic benchmark

Baseline flat model:

- synthetic test accuracy: `0.9845`
- synthetic test macro recall: `0.9875`

Hierarchical smooth-focus model:

- synthetic test accuracy: `0.9832`
- synthetic test macro recall: `0.9870`

Interpretation:

- synthetic-domain quality stayed essentially unchanged
- the new head/recipe did not buy synthetic accuracy, but it also did not materially damage it

### Real diagnostics

Baseline flat model:

- real test accuracy: `0.6861`
- real test macro recall: `0.6601`

Hierarchical smooth-focus model:

- real test accuracy: `0.7058`
- real test macro recall: `0.6843`

Absolute change:

- real test accuracy: `+0.0197`
- real test macro recall: `+0.0242`

### Real per-branch recall

Baseline:

- elastic: `0.9603`
- smooth: `0.1959`
- left_edge: `0.7540`
- right_edge: `0.6667`
- apex: `0.7236`

Hierarchical smooth-focus:

- elastic: `0.9603`
- smooth: `0.2392`
- left_edge: `0.6870`
- right_edge: `0.7327`
- apex: `0.8022`

Change:

- elastic: `+0.0000`
- smooth: `+0.0433`
- left_edge: `-0.0670`
- right_edge: `+0.0660`
- apex: `+0.0786`

## Interpretation

This is a real improvement.

The key point is that the new run improved the branch we were explicitly targeting:

- `smooth` recall improved from `0.1959` to `0.2392`

At the same time:

- `elastic` stayed solved
- `right_edge` improved materially
- `apex` improved materially

The tradeoff is:

- `left_edge` got worse

So the hierarchical head plus smooth-focused generator moved the decision geometry in a useful direction, but not uniformly across all plastic branches.

## What This Suggests

The result supports two conclusions:

1. The current bottleneck is still not raw model capacity.
   The same `256 x 3` backbone improved once the output structure and synthetic recipe changed.

2. Generator targeting matters.
   Extra mass around `smooth` and `smooth/right_edge` helped the exact branch we expected, which is a good sign that domain coverage is still the main lever.

## Recommended Next Step

Do one more controlled iteration, not a broad sweep:

- keep the hierarchical head
- keep `trial_raw_material`
- modify the synthetic recipe so it preserves the current `smooth/right_edge` gains while restoring `left_edge`

Concretely, the next recipe should:

- keep `smooth_focus`
- keep `smooth_edge`
- add a symmetric `left_edge` emphasis mode in the seed bank
- compare:
  - `smooth/right only`
  - `smooth + left/right symmetric`

That is the most direct next test if the goal is to improve `smooth` without paying for it by degrading `left_edge`.
