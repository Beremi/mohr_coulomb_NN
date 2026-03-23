# Cover Layer Branch Generator Correction Report

## Summary

We revisited the synthetic-data generator for the **cover-layer synthetic-only branch predictor** because the previous staged model learned the synthetic holdout well but transferred poorly to real data.

This pass compared:

- old generator: `empirical_local_noise`
- corrected generator: `seeded_local_noise_branch_balanced`

Then we retrained the same staged branch predictor on the corrected generator and compared the resulting synthetic and real performance.

## What Changed

The old generator perturbed canonicalized displacements empirically, but it was still too loose and branch-misaligned. The corrected generator instead:

- anchors each synthetic sample to a real cover-layer seed element
- perturbs the seed in canonicalized displacement space with **local** noise
- branch-balances the seed selection
- keeps the same exact FE `B` mapping and exact constitutive branch labeling

So the correction was not a new classifier. It was a tighter synthetic deformation generator.

## Coverage Check

Coverage was re-evaluated against the same real validation calls used by the staged training run.

![Coverage Comparison](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/coverage_compare.png)

| Generator | Branch TV | Mean Strain Rel. Err. | P95 Strain Rel. Err. |
|---|---:|---:|---:|
| `empirical_local_noise` | `0.3009` | `0.1572` | `0.0527` |
| `seeded_local_noise_branch_balanced` | `0.2663` | `0.0102` | `0.1929` |

Interpretation:

- The corrected generator improved **branch histogram coverage**.
- It improved **mean strain magnitude coverage** dramatically.
- It still undercovers the **upper strain tail**.

That last point matters because the remaining real-data gap is likely concentrated in exactly those hard tail states.

The raw coverage summary is stored in [coverage_summary.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/coverage_summary.json).

## Retraining Result

We retrained the same staged branch predictor with:

- identical architecture
- identical staged schedule
- identical real validation/test splits
- only the synthetic generator changed

![Accuracy Comparison](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/accuracy_compare.png)

| Metric | Old `empirical_local_noise` | New `seeded_local_noise_branch_balanced` |
|---|---:|---:|
| real val accuracy | `0.5069` | `0.5639` |
| real val macro recall | `0.5031` | `0.5624` |
| real test accuracy | `0.5328` | `0.6000` |
| real test macro recall | `0.5049` | `0.5741` |
| synthetic holdout accuracy | `0.9586` | `0.9763` |
| synthetic holdout macro recall | `0.9682` | `0.9827` |

This is a real improvement on both synthetic and real data.

## Per-Branch Change

Real test recall changed like this:

| Branch | Old | New |
|---|---:|---:|
| `elastic` | `0.9324` | `0.9324` |
| `smooth` | `0.0236` | `0.0051` |
| `left_edge` | `0.3911` | `0.5275` |
| `right_edge` | `0.7790` | `0.8223` |
| `apex` | `0.3983` | `0.5829` |

Interpretation:

- `left_edge`, `right_edge`, and `apex` improved materially.
- `elastic` stayed effectively saturated.
- `smooth` remained the clear failure branch.

So the corrected generator is closer to the real deformation manifold, but it still does not populate the real `smooth` regime properly.

## Best Current Run

The best corrected run is:

- checkpoint: [best.pt](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314/best.pt)
- summary: [summary.json](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314/summary.json)
- model card: [cover_layer_branch_predictor_staged_seeded_balanced_model_card.md](./cover_layer_branch_predictor_staged_seeded_balanced_model_card.md)

Headline numbers:

- real val accuracy: `0.5639`
- real val macro recall: `0.5624`
- real test accuracy: `0.6000`
- real test macro recall: `0.5741`
- synthetic holdout accuracy: `0.9763`
- synthetic holdout macro recall: `0.9827`

## Conclusion

The diagnosis was correct:

- the earlier synthetic generator was not covering the real deformation distribution tightly enough
- correcting the generator improved real transfer materially

But this is not solved yet.

The remaining issue is not “the network cannot fit the synthetic task.” It can.
The remaining issue is that the corrected synthetic generator still misses an important part of the real branch geometry, especially the `smooth` regime and the upper strain tail.

So the right next direction is:

1. keep `seeded_local_noise_branch_balanced` as the new baseline generator
2. specifically target the missing `smooth` regime in the generator
3. rebalance or condition the generator by branch-pattern or call-regime, not just pointwise branch counts
4. retrain the same staged classifier again before changing the classifier architecture
