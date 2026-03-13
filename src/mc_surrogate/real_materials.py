"""Material-family utilities for the captured slope-stability export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import build_reduced_material_from_raw, isotropic_moduli_from_young_poisson


@dataclass(frozen=True)
class RawMaterialSpec:
    """One raw material family used in the slope-stability problem."""

    name: str
    c0: float
    phi_deg: float
    psi_deg: float
    young: float
    poisson: float
    davis_type: str = "C"


def default_slope_material_specs() -> tuple[RawMaterialSpec, ...]:
    """Return the raw material families used by the real simulation export."""
    return (
        RawMaterialSpec("general_foundation", c0=15.0, phi_deg=38.0, psi_deg=0.0, young=50000.0, poisson=0.30),
        RawMaterialSpec("weak_foundation", c0=10.0, phi_deg=35.0, psi_deg=0.0, young=50000.0, poisson=0.30),
        RawMaterialSpec("general_slope", c0=18.0, phi_deg=32.0, psi_deg=0.0, young=20000.0, poisson=0.33),
        RawMaterialSpec("cover_layer", c0=15.0, phi_deg=30.0, psi_deg=0.0, young=10000.0, poisson=0.33),
    )


def elastic_signature(spec: RawMaterialSpec) -> np.ndarray:
    """Return the fixed elastic signature [G, K, lambda] for a raw material."""
    shear, bulk, lame = isotropic_moduli_from_young_poisson(spec.young, spec.poisson)
    return np.array([float(shear[0]), float(bulk[0]), float(lame[0])], dtype=float)


def reduced_from_spec(
    spec: RawMaterialSpec,
    strength_reduction: np.ndarray | float,
) -> np.ndarray:
    """Return reduced material rows [c_bar, sin_phi, G, K, lambda]."""
    lam = np.asarray(strength_reduction, dtype=float).reshape(-1)
    n = lam.shape[0]
    reduced = build_reduced_material_from_raw(
        c0=np.full(n, spec.c0, dtype=float),
        phi_rad=np.full(n, np.deg2rad(spec.phi_deg), dtype=float),
        psi_rad=np.full(n, np.deg2rad(spec.psi_deg), dtype=float),
        young=np.full(n, spec.young, dtype=float),
        poisson=np.full(n, spec.poisson, dtype=float),
        strength_reduction=lam,
        davis_type=np.array([spec.davis_type] * n, dtype=object),
    )
    return np.column_stack([reduced.c_bar, reduced.sin_phi, reduced.shear, reduced.bulk, reduced.lame])


def estimate_strength_reduction(
    material_reduced: np.ndarray,
    spec: RawMaterialSpec,
    *,
    lower: float = 0.75,
    upper: float = 3.5,
    iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate strength-reduction factors for reduced-material states from one raw family.

    The inversion uses the monotone relation between `c_bar` and the strength reduction
    factor, then scores the recovered state with both `c_bar` and `sin_phi`.
    """
    mat = np.asarray(material_reduced, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != 5:
        raise ValueError("material_reduced must have shape (n, 5).")

    n = mat.shape[0]
    c_obs = mat[:, 0]
    lo = np.full(n, lower, dtype=float)
    hi = np.full(n, upper, dtype=float)

    # c_bar decreases with increasing strength reduction for these materials.
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        c_mid = reduced_from_spec(spec, mid)[:, 0]
        increase_lower = c_mid > c_obs
        lo[increase_lower] = mid[increase_lower]
        hi[~increase_lower] = mid[~increase_lower]

    lam = 0.5 * (lo + hi)
    pred = reduced_from_spec(spec, lam)
    denom = np.maximum(np.abs(mat[:, :2]), np.array([1.0, 1.0e-3], dtype=float))
    fit_error = np.sqrt(np.mean(((pred[:, :2] - mat[:, :2]) / denom) ** 2, axis=1))
    return lam.astype(np.float32), fit_error.astype(np.float32)


def assign_material_families(
    material_reduced: np.ndarray,
    *,
    specs: tuple[RawMaterialSpec, ...] | None = None,
) -> dict[str, np.ndarray | list[str]]:
    """
    Assign each reduced-material row to the closest raw material family.

    Returns the chosen material id, an estimated strength-reduction factor, and the
    normalized family-fit error.
    """
    mat = np.asarray(material_reduced, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != 5:
        raise ValueError("material_reduced must have shape (n, 5).")

    if specs is None:
        specs = default_slope_material_specs()
    names = [spec.name for spec in specs]

    elastic = np.stack([elastic_signature(spec) for spec in specs], axis=0)
    elastic_scale = np.maximum(np.abs(elastic), 1.0)
    elastic_rel_err = np.max(np.abs(mat[:, None, 2:5] - elastic[None, :, :]) / elastic_scale[None, :, :], axis=2)

    # Separate unique elastic families first; the two foundation materials share elasticity.
    candidate_groups = (
        (0, 1),
        (2,),
        (3,),
    )

    group_elastic_error = np.stack(
        [np.min(elastic_rel_err[:, group], axis=1) for group in candidate_groups],
        axis=1,
    )
    group_id = np.argmin(group_elastic_error, axis=1)

    material_id = np.full(mat.shape[0], -1, dtype=np.int16)
    estimated_srf = np.full(mat.shape[0], np.nan, dtype=np.float32)
    fit_error = np.full(mat.shape[0], np.inf, dtype=np.float32)

    for gid, candidates in enumerate(candidate_groups):
        mask = group_id == gid
        if not np.any(mask):
            continue
        subset = mat[mask]

        best_idx = np.zeros(subset.shape[0], dtype=np.int16)
        best_srf = np.full(subset.shape[0], np.nan, dtype=np.float32)
        best_err = np.full(subset.shape[0], np.inf, dtype=np.float32)

        for spec_idx in candidates:
            lam, red_err = estimate_strength_reduction(subset, specs[spec_idx])
            total_err = red_err + elastic_rel_err[mask, spec_idx].astype(np.float32)
            better = total_err < best_err
            best_idx[better] = spec_idx
            best_srf[better] = lam[better]
            best_err[better] = total_err[better]

        material_id[mask] = best_idx
        estimated_srf[mask] = best_srf
        fit_error[mask] = best_err

    return {
        "material_id": material_id,
        "material_names": names,
        "estimated_strength_reduction": estimated_srf,
        "fit_error": fit_error,
    }
