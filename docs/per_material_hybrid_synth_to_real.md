# Per-Material Hybrid Hardcase Study

This follow-up keeps the good per-material empirical-exact models for the two foundation families,
and improves the two hard plastic families with extra branch-balanced fixed-material synthetic data.

Base experiment: `experiment_runs/real_sim/per_material_synth_to_real_20260312`
Hybrid output root: `experiment_runs/real_sim/per_material_hybrid_hardcases_20260312`

## Hardcase Results

| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| general_slope | 14325 | 13.6957 | 27.0371 | 677.1581 | 0.9319 |
| cover_layer | 7967 | 11.3344 | 21.5272 | 483.0251 | 0.9028 |
| hardcase aggregate | 22292 | 12.8518 | 25.2066 | 677.1581 | 0.9215 |

Hardcase comparison against the earlier per-material empirical-only models:

| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| general_slope | 14325 | 19.4763 | 33.8573 | 704.9789 | 0.9228 |
| cover_layer | 7967 | 12.9654 | 24.1759 | 503.0659 | 0.8637 |
| hardcase aggregate | 22292 | 17.1494 | 30.7493 | 704.9789 | 0.9017 |

## Mixed Best Aggregate

This aggregate uses the per-material empirical models for `general_foundation` and `weak_foundation`,
and the new hybrid models for the hard materials.

| Split | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| primary mixed-best | 28365 | 10.3046 | 22.3597 | 677.1581 | 0.9352 |
| cross mixed-best | 14183 | 10.4333 | 23.2323 | 731.7447 | 0.9372 |
| primary baseline | 28365 | 4.8612 | 17.7217 | 936.4612 | 0.9424 |
| cross baseline | 14183 | 4.7536 | 17.4137 | 759.1741 | 0.9432 |

## Notes

- If the hardcase hybrid models improve materially, the problem is not that synthetic training is impossible; it is that empirical exact relabeling alone undersupplies hard plastic branches.
- If the mixed-best aggregate still loses to the global real-trained baseline, the current synthetic-only route is still not competitive enough.
