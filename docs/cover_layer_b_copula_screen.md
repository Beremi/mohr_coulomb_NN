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
| `raw` | `2.9746` | `0.2190` | `0.5132` | `1.4813` | `0.0004` | `0.7606` | `experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/synthetic_test_raw.h5` |
| `pca` | `3.4403` | `0.2356` | `0.5115` | `1.9201` | `0.0026` | `0.7706` | `experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/synthetic_test_pca.h5` |
| `branch_raw` | `2.2721` | `0.5718` | `0.6448` | `0.9407` | `0.1035` | `0.0112` | `experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/synthetic_test_branch_raw.h5` |
| `branch_pca` | `2.3877` | `0.5810` | `0.7484` | `0.9411` | `0.1060` | `0.0112` | `experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/synthetic_test_branch_pca.h5` |
| `branch_raw_blend` | `3.1212` | `1.2412` | `0.8591` | `0.9059` | `0.1039` | `0.0112` | `experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/synthetic_test_branch_raw_blend.h5` |

### raw

![distribution fit raw](../experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/distribution_fit_raw.png)

### pca

![distribution fit pca](../experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/distribution_fit_pca.png)

### branch_raw

![distribution fit branch_raw](../experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/distribution_fit_branch_raw.png)

### branch_pca

![distribution fit branch_pca](../experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/distribution_fit_branch_pca.png)

### branch_raw_blend

![distribution fit branch_raw_blend](../experiment_runs/real_sim/cover_layer_b_copula_screen_20260312/distribution_fit_branch_raw_blend.png)

