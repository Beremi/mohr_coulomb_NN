# Cover Layer B-Copula Refresh

This study replaces direct strain-space perturbation with a local FE-driven generator.
Synthetic samples are produced by sampling local P2 tetra nodal displacements, filtering invalid corner-volume changes, and then mapping them through the local `B` operator to engineering strains.

## FE Geometry Pool

- mesh: `/tmp/slope_stability_octave/slope_stability/meshes/SSR_hetero_ada_L1.h5`
- local integration-point pool size: `56364`
- element count in pool: `5124`
- quadrature points per element: `11`
- mean characteristic length: `2.9006`

## Distribution Fits

| mode | total score | principal q95 logerr | principal q995 logerr | stress q95 logerr | stress q995 logerr | branch TV | synthetic test |
|---|---:|---:|---:|---:|---:|---:|---|
| `branch_raw` | `2.2297` | `0.5816` | `0.5864` | `0.9448` | `0.1060` | `0.0110` | `experiment_runs/real_sim/cover_layer_b_copula_smoke4/synthetic_test_branch_raw.h5` |
| `branch_raw_blend` | `3.0885` | `1.2080` | `0.8436` | `0.9205` | `0.1054` | `0.0110` | `experiment_runs/real_sim/cover_layer_b_copula_smoke4/synthetic_test_branch_raw_blend.h5` |

### branch_raw

![distribution fit branch_raw](../experiment_runs/real_sim/cover_layer_b_copula_smoke4/distribution_fit_branch_raw.png)

### branch_raw_blend

![distribution fit branch_raw_blend](../experiment_runs/real_sim/cover_layer_b_copula_smoke4/distribution_fit_branch_raw_blend.png)

