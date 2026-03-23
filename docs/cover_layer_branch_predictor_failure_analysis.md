# Cover Layer Branch Predictor Failure Analysis

## Short Answer

The branch predictor is still way off on real data because the corrected synthetic generator is **still generating the wrong kind of `smooth` states**.

This is not a simple undertraining problem.

The staged model learns the synthetic task very well:

- synthetic holdout accuracy: `0.9763`
- synthetic holdout macro recall: `0.9827`

But it is still poor on real data:

- real test accuracy: `0.6000`
- real test macro recall: `0.5741`

That means the dominant issue is still **synthetic-to-real transfer**.

## Main Evidence

### 1. Real `smooth` recall is essentially zero

From [summary.json](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314/summary.json):

- real test `smooth` recall: `0.0051`
- synthetic holdout `smooth` recall: `0.9471`

So the classifier clearly knows how to classify the **synthetic** `smooth` branch, but it does not recognize the **real** `smooth` branch at all.

### 2. Real `smooth` points are being mapped to `right_edge`

From [why_off_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/why_off_analysis.json), the prediction distribution for **true real `smooth`** points is:

- `smooth -> smooth`: `0.0051`
- `smooth -> right_edge`: `0.8111`
- `smooth -> left_edge`: `0.1170`
- `smooth -> apex`: `0.0452`

So the model is not uncertain. It is mostly deciding that real `smooth` points look like `right_edge`.

### 3. The model is confidently wrong on real `smooth`

From [confidence_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/confidence_analysis.json):

- for **real `smooth`**:
  - mean probability of the true class: `0.0538`
  - mean top-class probability: `0.6037`
- for **synthetic `smooth`**:
  - mean probability of the true class: `0.9404`
  - mean top-class probability: `0.9705`

This is the clearest sign that the problem is not lack of optimization. The model is confidently correct on synthetic `smooth`, and confidently wrong on real `smooth`.

## Coverage Findings

From [coverage_summary.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/coverage_summary.json):

### The corrected generator did improve some things

- branch TV: `0.3009 -> 0.2663`
- mean strain relative error: `0.1572 -> 0.0102`

So the generator correction was real and helpful.

### But it still misses the upper strain tail

- real strain-norm p95: `24.40`
- synthetic strain-norm p95: `19.70`
- p95 relative error: `0.1929`

That means the generator still undercovers the harder tail states.

## Why This Likely Happens

The current seeded generator is better than the old empirical noise model, but it still samples from the wrong geometry of cases:

1. It branch-balances by seed elements, not by **branch difficulty** or **margin to branch boundaries**.
2. It produces synthetic `smooth` states that are easy and internally consistent for the classifier.
3. The real `smooth` states appear to lie much closer to the `right_edge` decision region.

So the generator is not just missing counts. It is missing the **right local branch geometry**.

## What Is Probably *Not* The Main Problem

### Not simple undertraining

The synthetic metrics are too high for that explanation.

### Probably not just lack of element context

From [context_pattern_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/context_pattern_analysis.json):

- real `smooth` elements have mean `2.90` distinct branches
- synthetic `smooth` elements have mean `3.38` distinct branches

So the synthetic elements are not simpler in a naive way. The problem is not just “real elements are mixed and synthetic elements are pure.”

## Practical Conclusion

The failure mode is now fairly specific:

- `elastic` is basically solved
- `right_edge` is decent
- `left_edge` is partially learned
- `apex` improved a lot
- `smooth` is still being generated in the wrong region and gets collapsed into `right_edge` on real data

So the next step should not be a wider network or longer training loop.

The next step should be:

1. make the generator target **real `smooth` states specifically**
2. oversample hard `smooth` points and near-boundary `smooth/right_edge` transitions
3. only then retrain the same staged classifier
