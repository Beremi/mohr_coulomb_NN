import numpy as np

from mc_surrogate.real_materials import (
    assign_material_families,
    default_slope_material_specs,
    estimate_strength_reduction,
    reduced_from_spec,
)


def test_assign_material_families_at_unit_strength_reduction():
    specs = default_slope_material_specs()
    rows = np.vstack([reduced_from_spec(spec, [1.0]) for spec in specs])
    assigned = assign_material_families(rows, specs=specs)
    assert assigned["material_names"] == [spec.name for spec in specs]
    assert np.array_equal(assigned["material_id"], np.arange(len(specs)))
    assert np.allclose(assigned["estimated_strength_reduction"], 1.0, atol=1.0e-5)
    assert np.all(assigned["fit_error"] < 1.0e-6)


def test_estimate_strength_reduction_matches_reduced_material_curve():
    specs = default_slope_material_specs()
    strength_reduction = np.array([0.92, 1.0, 1.35, 1.85], dtype=float)
    reduced = reduced_from_spec(specs[1], strength_reduction)
    estimated, error = estimate_strength_reduction(reduced, specs[1])
    assert np.allclose(estimated, strength_reduction, atol=5.0e-4)
    assert np.all(error < 1.0e-5)
