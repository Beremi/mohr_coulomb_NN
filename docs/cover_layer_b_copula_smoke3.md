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
| `branch_raw` | `2.1485` | `0.5479` | `0.5502` | `0.9357` | `0.1037` | `0.0110` | `experiment_runs/real_sim/cover_layer_b_copula_smoke3/synthetic_test_branch_raw.h5` |
| `branch_pca` | `2.4035` | `0.6334` | `0.7149` | `0.9397` | `0.1045` | `0.0110` | `experiment_runs/real_sim/cover_layer_b_copula_smoke3/synthetic_test_branch_pca.h5` |

### branch_raw

![distribution fit branch_raw](../experiment_runs/real_sim/cover_layer_b_copula_smoke3/distribution_fit_branch_raw.png)

### branch_pca

![distribution fit branch_pca](../experiment_runs/real_sim/cover_layer_b_copula_smoke3/distribution_fit_branch_pca.png)

