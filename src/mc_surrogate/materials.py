"""Material parameter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DAVIS_TYPES = ("A", "B", "C")
DAVIS_ID_TO_TYPE = {0: "A", 1: "B", 2: "C"}
DAVIS_TYPE_TO_ID = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class ReducedMaterial:
    """Reduced constitutive parameters used by the pointwise operator."""

    c_bar: np.ndarray
    sin_phi: np.ndarray
    shear: np.ndarray
    bulk: np.ndarray
    lame: np.ndarray


def isotropic_moduli_from_young_poisson(
    young: np.ndarray | float,
    poisson: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shear modulus G, bulk modulus K, and Lamé coefficient λ."""
    young = np.asarray(young, dtype=float)
    poisson = np.asarray(poisson, dtype=float)
    young, poisson = np.broadcast_arrays(young, poisson)
    if np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
        raise ValueError("Poisson ratio must lie in (-1, 0.5).")
    shear = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    lame = bulk - (2.0 / 3.0) * shear
    return shear.reshape(-1), bulk.reshape(-1), lame.reshape(-1)


def _broadcast_1d(*arrays: np.ndarray | float) -> list[np.ndarray]:
    out = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])
    return [a.reshape(-1) for a in out]


def _normalize_davis_type(davis_type: str | int | Iterable[str | int], n: int) -> np.ndarray:
    if isinstance(davis_type, str):
        if davis_type not in DAVIS_TYPE_TO_ID:
            raise ValueError(f"Unsupported Davis type {davis_type!r}.")
        return np.full(n, DAVIS_TYPE_TO_ID[davis_type], dtype=int)
    if isinstance(davis_type, (int, np.integer)):
        if int(davis_type) not in DAVIS_ID_TO_TYPE:
            raise ValueError(f"Unsupported Davis id {davis_type}.")
        return np.full(n, int(davis_type), dtype=int)
    seq = np.asarray(list(davis_type))
    if seq.ndim != 1:
        raise ValueError("Davis type must be scalar or 1D sequence.")
    if seq.shape[0] != n:
        raise ValueError(f"Expected {n} Davis labels, got {seq.shape[0]}.")
    if seq.dtype.kind in {"U", "S", "O"}:
        ids = np.empty(n, dtype=int)
        for i, item in enumerate(seq):
            label = str(item)
            if label not in DAVIS_TYPE_TO_ID:
                raise ValueError(f"Unsupported Davis type {label!r}.")
            ids[i] = DAVIS_TYPE_TO_ID[label]
        return ids
    ids = seq.astype(int)
    if np.any((ids < 0) | (ids > 2)):
        raise ValueError("Davis ids must lie in {0,1,2}.")
    return ids


def davis_reduction(
    c0: np.ndarray | float,
    phi_rad: np.ndarray | float,
    psi_rad: np.ndarray | float,
    strength_reduction: np.ndarray | float,
    davis_type: str | int | Iterable[str | int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Davis A/B/C reduction as in the original MATLAB/Octave code.

    Returns
    -------
    c_bar
        Reduced cohesion parameter 2*c_lambda*cos(phi_lambda).
    sin_phi
        Sine of the reduced friction angle.
    """
    c0, phi_rad, psi_rad, strength_reduction = _broadcast_1d(c0, phi_rad, psi_rad, strength_reduction)
    ids = _normalize_davis_type(davis_type, c0.size)

    c_bar = np.empty_like(c0)
    sin_phi = np.empty_like(c0)

    for davis_id in (0, 1, 2):
        mask = ids == davis_id
        if not np.any(mask):
            continue
        c = c0[mask]
        phi = phi_rad[mask]
        psi = psi_rad[mask]
        lam = strength_reduction[mask]

        if davis_id == 0:
            beta = np.cos(phi) * np.cos(psi) / (1.0 - np.sin(phi) * np.sin(psi))
            c_lambda = beta * c / lam
            phi_lambda = np.arctan(beta * np.tan(phi) / lam)
        elif davis_id == 1:
            c1 = c / lam
            phi1 = np.arctan(np.tan(phi) / lam)
            psi1 = np.arctan(np.tan(psi) / lam)
            beta = np.cos(phi1) * np.cos(psi1) / (1.0 - np.sin(phi1) * np.sin(psi1))
            c_lambda = beta * c1
            phi_lambda = np.arctan(beta * np.tan(phi1))
        else:
            c1 = c / lam
            phi1 = np.arctan(np.tan(phi) / lam)
            beta = np.where(
                phi1 > psi,
                np.cos(phi1) * np.cos(psi) / (1.0 - np.sin(phi1) * np.sin(psi)),
                1.0,
            )
            c_lambda = beta * c1
            phi_lambda = np.arctan(beta * np.tan(phi1))

        c_bar[mask] = 2.0 * c_lambda * np.cos(phi_lambda)
        sin_phi[mask] = np.sin(phi_lambda)

    return c_bar, sin_phi


def build_reduced_material_from_raw(
    c0: np.ndarray | float,
    phi_rad: np.ndarray | float,
    psi_rad: np.ndarray | float,
    young: np.ndarray | float,
    poisson: np.ndarray | float,
    strength_reduction: np.ndarray | float,
    davis_type: str | int | Iterable[str | int],
) -> ReducedMaterial:
    """Compute reduced constitutive parameters from raw geotechnical inputs."""
    c_bar, sin_phi = davis_reduction(c0, phi_rad, psi_rad, strength_reduction, davis_type)
    shear, bulk, lame = isotropic_moduli_from_young_poisson(young, poisson)
    return ReducedMaterial(c_bar=c_bar, sin_phi=sin_phi, shear=shear, bulk=bulk, lame=lame)
