# Real Simulation Validation

## Setup

- Real dataset: `experiment_runs/real_sim/baseline_sample_256.h5`
- Synthetic reference dataset: `experiment_runs/high_capacity/dataset.h5`
- Checkpoint: `experiment_runs/high_capacity/study_pilot/repeats/seed_1012/best.pt`
- Device: `cuda`

## Exact Export Compatibility

- Sampled real export was converted with exact branch labels using the updated Python constitutive operator.
- Export/exact stress agreement in preprocessing: MAE `4.680e-13`, RMSE `3.020e-12`, max abs `2.506e-10`.

## Distribution Shift

| Quantity | Synthetic q50 | Synthetic q90 | Synthetic q99 | Synthetic max | Real q50 | Real q90 | Real q99 | Real max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abs strain component | 0.000129081 | 0.00102694 | 0.0026114 | 0.00813037 | 0.0200475 | 2.54753 | 13.2399 | 67.756 |
| abs stress component | 2.22524 | 16.8031 | 42.5522 | 219.391 | 25.9808 | 278.086 | 833.684 | 2973.88 |

- Real branch fractions: `{"0": 0.1654656475631769, "1": 0.3922396773465704, "2": 0.20995741200361012, "3": 0.12647365749097472, "4": 0.10586360559566788}`
- Synthetic branch fractions: `{"0": 0.2134681323114159, "1": 0.15331282775312627, "2": 0.1996520774505849, "3": 0.23530153287615974, "4": 0.19826542960871318}`

Plots:
- `experiment_runs/real_sim/baseline_validation/strain_abs_distribution.png`
- `experiment_runs/real_sim/baseline_validation/stress_abs_distribution.png`
- `experiment_runs/real_sim/baseline_validation/branch_distribution.png`

## Baseline Surrogate

### Synthetic Test Split

| Metric | Value |
| --- | ---: |
| Samples | 7933 |
| Stress MAE | 0.994877 |
| Stress RMSE | 1.374361 |
| Stress max abs | 10.128434 |
| Principal MAE | 1.782598 |
| Principal RMSE | 2.300003 |
| Branch accuracy | 0.997101 |

### Real Test Split

| Metric | Value |
| --- | ---: |
| Samples | 14183 |
| Stress MAE | 75.027443 |
| Stress RMSE | 164.391632 |
| Stress max abs | 2570.133301 |
| Principal MAE | 138.479324 |
| Principal RMSE | 233.799149 |
| Branch accuracy | 0.434323 |

### Real Stress Error by Strain Magnitude Quartile

| Strain magnitude bin | Samples | Stress MAE | Stress RMSE |
| --- | ---: | ---: | ---: |
| [0, 0.0115726] | 3546 | 62.441124 | 138.093933 |
| [0.0115726, 0.0824286] | 3545 | 115.559448 | 233.189224 |
| [0.0824286, 2.12957] | 3546 | 51.262005 | 105.557854 |
| [2.12957, 65.4352] | 3546 | 70.858620 | 153.350983 |

Plots:
- `experiment_runs/real_sim/baseline_validation/real_parity_stress.png`
- `experiment_runs/real_sim/baseline_validation/real_parity_principal.png`
- `experiment_runs/real_sim/baseline_validation/real_stress_error_hist.png`
- `experiment_runs/real_sim/baseline_validation/real_error_vs_strain.png`

## Findings

- The current surrogate is accurate on the synthetic reference test split but degrades strongly on the real sampled states.
- The dominant issue is distribution shift: real strain magnitudes are orders of magnitude larger than the synthetic training distribution used so far.
- Branch classification is also weak on real data, which suggests the current feature/architecture choice is not robust enough for the actual state occupancy of the slope-stability solve.
- The preprocessing step confirms the real export itself is consistent with the updated exact constitutive operator, so the remaining gap is in the learned model and training data, not in the sampled real labels.
