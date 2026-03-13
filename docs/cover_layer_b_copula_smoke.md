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
| `raw` | `2.4930` | `0.2375` | `0.4658` | `0.9766` | `0.0530` | `0.7601` | `experiment_runs/real_sim/cover_layer_b_copula_smoke/synthetic_test_raw.h5` |
| `pca` | `3.3622` | `0.2290` | `0.4334` | `1.9201` | `0.0176` | `0.7621` | `experiment_runs/real_sim/cover_layer_b_copula_smoke/synthetic_test_pca.h5` |

