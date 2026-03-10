# Demo Training Experiments

## Goal

The original demo notebook ran end to end, but the fit was unusable. This document records the tuning pass that was used to produce the final notebook configuration.

The experiment driver is:

- [scripts/experiments/sweep_demo_training.py](/home/beremi/repos/mohr_coulomb_NN/scripts/experiments/sweep_demo_training.py)

The key implementation change that enabled stable demo datasets is the optional principal-strain cap added to:

- [sampling.py](/home/beremi/repos/mohr_coulomb_NN/src/mc_surrogate/sampling.py)

## Root cause of the bad first fit

The first notebook used a small uncapped dataset over the full default material range. That occasionally produced pathological elastic samples with enormous strain and stress magnitudes.

Observed failure mode:

- dataset max stress reached about `2.79e17`
- dataset `99th` percentile of max sample stress was only about `65.9`
- almost all normal samples were in the `O(1)` to `O(10^2)` range

That single scale outlier contaminated target standardization and pushed the trained network toward absurd predictions even on a clean test split.

## What was tested

| Experiment | Dataset | Model / training | Stress MAE | Stress RMSE | Max abs err | Branch acc. | Triaxial principal MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_uncapped_small` | 1000 samples, default ranges, no strain cap | width `128`, depth `3`, `12` epochs | `1.868e14` | `2.714e14` | `1.321e15` | `0.58` | `4.100e14` |
| `wide_capped_medium` | 3000 samples, default ranges, cap `5e-3` | width `128`, depth `3`, `40` epochs | `0.899` | `1.228` | `10.692` | `0.963` | `1.646` |
| `wide_capped_large` | 5000 samples, default ranges, cap `5e-3` | width `256`, depth `4`, `60` epochs | `0.803` | `1.078` | `9.746` | `0.996` | `1.386` |
| `focused_capped_large` | 5000 samples, focused ranges, cap `2e-3` | width `256`, depth `4`, `60` epochs | `0.695` | `0.926` | `3.483` | `0.998` | `1.331` |
| `focused_capped_xlarge` | 7000 samples, focused ranges, cap `2e-3` | width `384`, depth `5`, `80` epochs | `0.735` | `0.996` | `10.493` | `0.991` | `1.322` |

Aggregate CSV:

- [results.csv](/home/beremi/repos/mohr_coulomb_NN/experiment_runs/demo_sweep/results.csv)

## Main findings

- The dominant issue was data scale, not network capacity.
- Adding `max_abs_principal_strain` immediately removed the catastrophic failure mode.
- Once the data was clean, even the smaller capped model became reasonable.
- Moving from `128x3` to `256x4` helped materially.
- Increasing further to `384x5` with more data and epochs did not improve the demo metric set; it slightly worsened stress MAE and max error.
- A focused material window was better for a notebook demo because it kept the stress scale compact while still covering all constitutive branches.

## Final notebook choice

The final notebook uses the `focused_capped_large` configuration:

- `5000` generated samples
- branch-balanced sampling
- focused material ranges:
  - cohesion `5` to `25`
  - friction angle `25` to `45` degrees
  - dilatancy `0` to `10` degrees
  - Young's modulus `1e4` to `6e4`
  - Poisson ratio `0.25` to `0.38`
  - strength reduction `0.9` to `2.0`
- principal-strain cap `2e-3`
- principal model with width `256`, depth `4`
- `60` epochs, batch size `256`, AdamW, branch loss weight `0.1`

This is the configuration transferred into:

- [mohr_coulomb_demo.ipynb](/home/beremi/repos/mohr_coulomb_NN/notebooks/mohr_coulomb_demo.ipynb)

## Practical takeaways

- If you want a broader operating range, regenerate the dataset with wider material ranges, but keep the strain cap.
- If a specific branch or parameter corner matters more, add points by increasing `n_samples` or changing the sampling ranges rather than only making the network larger.
- If you want to stay with the wider default material ranges, the `wide_capped_large` setup is the safer starting point than the focused one.
- If future work needs very wide strain scales, the training pipeline will likely need a more robust target treatment than plain standardization.
