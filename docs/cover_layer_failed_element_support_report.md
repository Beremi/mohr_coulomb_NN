# Cover Layer Failed Element Support Report

## Question

Are the **real test elements where branch prediction fails** actually inside the practical sample space of the current synthetic generator, or are they mostly outside it?

This report answers that for the current best synthetic-only branch predictor:

- run: [cover_layer_branch_predictor_staged_seeded_balanced_20260314](../experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314)
- generator: `seeded_local_noise_branch_balanced`

## How “Possible Sample Space” Was Defined

Because the generator uses Gaussian local noise, the literal support is unbounded, so “possible” has to mean **practical support actually used by this generator**.

I checked that in two complementary ways:

1. **Seed-neighborhood support in canonical displacement space**
   - Each real element was canonicalized.
   - Its canonical displacement was compared to the nearest fit seed.
   - Distance was normalized by the exact local noise scale used by the generator.
   - This answers: is the element inside the generator’s actual seed-based deformation neighborhoods?

2. **Nearest-neighbor support in full element strain space**
   - A larger synthetic cloud was generated from the same generator.
   - Each real element was compared to its nearest synthetic neighbor in flattened `11 x 6` strain space.
   - This answers: is the element at least near some generated synthetic strain state?

So this report distinguishes between:

- **outside the generator’s practical deformation neighborhoods**
- **outside the synthetic strain cloud entirely**

## Headline Result

Yes: the failed real elements are **mostly outside the practical seed-neighborhood sample space** of the current synthetic generator.

But they are **not all outside the broad synthetic strain cloud**.

That is the important nuance.

It means:

- the generator is not placing enough mass near the real failed elements in element-deformation space
- but some failed elements still land near synthetic strain states, which suggests the remaining problem is also about **wrong branch geometry / wrong local labeling structure**, not only global coverage

## Main Comparison

![Failed Support Compare](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/failed_support_compare.png)

![Failed Support Distance Compare](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/failed_support_distance_compare.png)

## Quantitative Summary

Reference thresholds from the synthetic cloud:

- synthetic seed-neighborhood `z_rms` p95: `1.1904`
- synthetic seed-neighborhood `z_rms` p99: `1.2895`
- synthetic strain-NN p95: `7.6794`
- synthetic strain-NN p99: `12.6009`

### All failed vs all successful real test elements

| Metric | Failed | Successful |
|---|---:|---:|
| count | `375` | `137` |
| median nearest-seed `z_rms` | `4.3868` | `0.1436` |
| frac outside synthetic seed p95 | `0.7733` | `0.1314` |
| frac outside synthetic seed p99 | `0.7547` | `0.1241` |
| median strain NN distance | `1.2585` | `0.8281` |
| frac outside synthetic strain NN p95 | `0.1227` | `0.0073` |
| frac outside synthetic strain NN p99 | `0.0453` | `0.0000` |

This is the clearest answer:

- in **canonical displacement / generator-neighborhood space**, failed elements are usually outside practical support
- in **strain-space NN distance**, most failed elements are still not extremely far from the synthetic cloud

## Smooth-Failure Elements

This matters most because `smooth` is the branch that still collapses badly.

| Metric | Smooth-fail |
|---|---:|
| count | `264` |
| median nearest-seed `z_rms` | `2.2092` |
| frac outside synthetic seed p95 | `0.7121` |
| frac outside synthetic seed p99 | `0.6856` |
| median strain NN distance | `0.9323` |
| frac outside synthetic strain NN p95 | `0.0871` |
| frac outside synthetic strain NN p99 | `0.0379` |

Interpretation:

- most smooth-fail elements are **outside the generator’s practical local displacement support**
- but many are still **near some synthetic strain state**

That strongly suggests the generator is missing the **right branch-conditioned local geometry**, especially around `smooth/right_edge`, not just missing the whole region in a gross sense.

## Branch-Wise Notes

Selected branch-wise results:

- `elastic` failures are mostly inside support and small; that branch is basically solved.
- `smooth` failures are the cleanest sign of generator misspecification.
- `right_edge` and `apex` failures also skew outside the seed-neighborhood support, but transfer is much better there than for `smooth`.

The full branch-wise table is stored in [failed_element_support_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/failed_element_support_analysis.json).

## Worst Failed Examples

The most off-support failed elements are genuinely extreme relative to the current generator.

Examples from [failed_element_support_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/failed_element_support_analysis.json):

- element `449`
  - nearest-seed `z_rms = 16.17`
  - strain NN distance `56.97`
  - true pattern: `2-2-2-2-2-2-4-2-2-2-2`
  - predicted pattern: all `apex`

- element `507`
  - nearest-seed `z_rms = 16.95`
  - strain NN distance `46.81`
  - true pattern: `1-2-1-3-1-2-1-2-3-3-3`
  - predicted pattern: all `elastic`

Those are not subtle misses. They are far away from what the current generator really samples.

## Conclusion

So the answer is:

- **Yes**, many of the failed real elements are outside the practical sample space of the current synthetic generator, especially in canonical displacement / seed-neighborhood space.
- **No**, they are not all completely outside the broader synthetic strain cloud.

That means the remaining problem is two-part:

1. the generator still does not cover enough of the real deformation manifold in the neighborhoods that matter
2. even where broad strain coverage exists, the synthetic branch geometry is still wrong, especially for `smooth`

## What This Means For The Next Iteration

The next generator improvement should be:

1. target **smooth-fail** elements specifically
2. generate around those local deformation neighborhoods
3. emphasize the `smooth/right_edge` transition region
4. keep the same staged classifier first, so we isolate generator improvement before changing architecture

## Artifacts

- support analysis JSON: [failed_element_support_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/failed_element_support_analysis.json)
- failure-mode JSON: [why_off_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/why_off_analysis.json)
- confidence JSON: [confidence_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/confidence_analysis.json)
- context-pattern JSON: [context_pattern_analysis.json](../experiment_runs/real_sim/cover_layer_branch_generator_correction_20260314/context_pattern_analysis.json)
