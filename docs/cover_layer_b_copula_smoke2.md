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
| `raw` | `3.2998` | `0.1674` | `0.3355` | `1.9201` | `0.1075` | `0.7694` | `experiment_runs/real_sim/cover_layer_b_copula_smoke2/synthetic_test_raw.h5` |
| `pca` | `3.3559` | `0.2219` | `0.4231` | `1.9201` | `0.0214` | `0.7694` | `experiment_runs/real_sim/cover_layer_b_copula_smoke2/synthetic_test_pca.h5` |
| `branch_raw` | `2.2280` | `0.5372` | `0.6384` | `0.9329` | `0.1085` | `0.0110` | `experiment_runs/real_sim/cover_layer_b_copula_smoke2/synthetic_test_branch_raw.h5` |
| `branch_pca` | `2.3069` | `0.6507` | `0.6069` | `0.9303` | `0.1081` | `0.0110` | `experiment_runs/real_sim/cover_layer_b_copula_smoke2/synthetic_test_branch_pca.h5` |

### raw

![distribution fit raw](../experiment_runs/real_sim/cover_layer_b_copula_smoke2/distribution_fit_raw.png)

### pca

![distribution fit pca](../experiment_runs/real_sim/cover_layer_b_copula_smoke2/distribution_fit_pca.png)

### branch_raw

![distribution fit branch_raw](../experiment_runs/real_sim/cover_layer_b_copula_smoke2/distribution_fit_branch_raw.png)

### branch_pca

![distribution fit branch_pca](../experiment_runs/real_sim/cover_layer_b_copula_smoke2/distribution_fit_branch_pca.png)

