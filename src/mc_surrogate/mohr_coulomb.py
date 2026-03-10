"""3D elastic-perfectly-plastic Mohr-Coulomb constitutive operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .voigt import (
    principal_values_and_vectors_from_strain,
    reconstruct_from_principal,
    strain_voigt_to_tensor,
    tensor_to_stress_voigt,
)

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")
BRANCH_TO_ID = {name: i for i, name in enumerate(BRANCH_NAMES)}


@dataclass
class ConstitutiveResult:
    """Full result of a constitutive evaluation."""

    stress: np.ndarray
    stress_principal: np.ndarray
    strain_principal: np.ndarray
    eigvecs: np.ndarray
    branch_id: np.ndarray
    f_trial: np.ndarray
    plastic_multiplier: np.ndarray
    tangent: np.ndarray | None = None


def _broadcast_materials(n: int, *arrays: np.ndarray | float) -> list[np.ndarray]:
    out = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])
    return [a.reshape(-1) if a.size == n else np.broadcast_to(a, (n,)).reshape(-1) for a in out]


def constitutive_update_3d(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
    *,
    return_tangent: bool = False,
    tangent_eps: float = 1.0e-8,
    branch_tol: float = 1.0e-10,
) -> ConstitutiveResult:
    """
    Evaluate the 3D Mohr-Coulomb constitutive operator.

    Parameters
    ----------
    strain_eng
        Engineering strain in Voigt order [e11, e22, e33, g12, g13, g23].
    c_bar, sin_phi, shear, bulk, lame
        Reduced material parameters of the constitutive model. Arrays are broadcast
        over the batch dimension.
    return_tangent
        If True, compute a numerical centered finite-difference tangent dσ/dε_eng.

    Notes
    -----
    This implementation follows the same branch decomposition used in the original
    MATLAB/Octave constitutive routine: elastic, smooth, left edge, right edge, apex.
    """
    strain = np.asarray(strain_eng, dtype=float)
    if strain.ndim == 1:
        strain = strain[None, :]
    if strain.ndim != 2 or strain.shape[1] != 6:
        raise ValueError(f"Expected strain shape (n,6), got {strain.shape}.")

    n = strain.shape[0]
    c_bar, sin_phi, shear, bulk, lame = _broadcast_materials(n, c_bar, sin_phi, shear, bulk, lame)
    sin_psi = sin_phi.copy()

    eigvals, eigvecs = principal_values_and_vectors_from_strain(strain)
    e1 = eigvals[:, 0]
    e2 = eigvals[:, 1]
    e3 = eigvals[:, 2]
    i1 = e1 + e2 + e3

    f_trial = 2.0 * shear * ((1.0 + sin_phi) * e1 - (1.0 - sin_phi) * e3) + 2.0 * (lame * sin_phi) * i1 - c_bar

    gamma_sl = (e1 - e2) / np.maximum(1.0 + sin_psi, 1.0e-14)
    gamma_sr = (e2 - e3) / np.maximum(1.0 - sin_psi, 1.0e-14)
    gamma_la = (e1 + e2 - 2.0 * e3) / np.maximum(3.0 - sin_psi, 1.0e-14)
    gamma_ra = (2.0 * e1 - e2 - e3) / np.maximum(3.0 + sin_psi, 1.0e-14)

    denom_s = 4.0 * lame * sin_phi * sin_psi + 4.0 * shear * (1.0 + sin_phi * sin_psi)
    denom_l = (
        4.0 * lame * sin_phi * sin_psi
        + shear * (1.0 + sin_phi) * (1.0 + sin_psi)
        + 2.0 * shear * (1.0 - sin_phi) * (1.0 - sin_psi)
    )
    denom_r = (
        4.0 * lame * sin_phi * sin_psi
        + 2.0 * shear * (1.0 + sin_phi) * (1.0 + sin_psi)
        + shear * (1.0 - sin_phi) * (1.0 - sin_psi)
    )
    denom_a = 4.0 * bulk * sin_phi * sin_psi

    safe_denom_s = np.where(np.abs(denom_s) > 1.0e-14, denom_s, np.inf)
    safe_denom_l = np.where(np.abs(denom_l) > 1.0e-14, denom_l, np.inf)
    safe_denom_r = np.where(np.abs(denom_r) > 1.0e-14, denom_r, np.inf)
    safe_denom_a = np.where(np.abs(denom_a) > 1.0e-14, denom_a, np.inf)

    lambda_s = f_trial / safe_denom_s
    lambda_l = (
        shear * ((1.0 + sin_phi) * (e1 + e2) - 2.0 * (1.0 - sin_phi) * e3)
        + 2.0 * lame * sin_phi * i1
        - c_bar
    ) / safe_denom_l
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * e1 - (1.0 - sin_phi) * (e2 + e3))
        + 2.0 * lame * sin_phi * i1
        - c_bar
    ) / safe_denom_r
    lambda_a = (2.0 * bulk * sin_phi * i1 - c_bar) / safe_denom_a

    test_el = f_trial <= branch_tol
    test_s = (~test_el) & (lambda_s <= np.minimum(gamma_sl, gamma_sr) + branch_tol)
    test_l = (
        (~test_el)
        & (~test_s)
        & (gamma_sl < gamma_sr + branch_tol)
        & (lambda_l >= gamma_sl - branch_tol)
        & (lambda_l <= gamma_la + branch_tol)
    )
    test_r = (
        (~test_el)
        & (~test_s)
        & (gamma_sl > gamma_sr - branch_tol)
        & (lambda_r >= gamma_sr - branch_tol)
        & (lambda_r <= gamma_ra + branch_tol)
    )
    test_a = ~(test_el | test_s | test_l | test_r)

    branch_id = np.full(n, -1, dtype=np.int64)
    branch_id[test_el] = BRANCH_TO_ID["elastic"]
    branch_id[test_s] = BRANCH_TO_ID["smooth"]
    branch_id[test_l] = BRANCH_TO_ID["left_edge"]
    branch_id[test_r] = BRANCH_TO_ID["right_edge"]
    branch_id[test_a] = BRANCH_TO_ID["apex"]

    stress_principal = np.zeros((n, 3), dtype=float)
    stress_tensor = np.zeros((n, 3, 3), dtype=float)
    plastic_multiplier = np.zeros(n, dtype=float)
    eye = np.eye(3, dtype=float)[None, :, :]

    sigma_trial = lame[:, None] * i1[:, None] + 2.0 * shear[:, None] * eigvals

    if np.any(test_el):
        idx = np.where(test_el)[0]
        stress_principal[idx] = sigma_trial[idx]
        stress_tensor[idx] = reconstruct_from_principal(stress_principal[idx], eigvecs[idx])

    if np.any(test_s):
        idx = np.where(test_s)[0]
        lam = lambda_s[idx]
        plastic_multiplier[idx] = lam
        sp = sin_psi[idx]
        ll = lame[idx]
        mu = shear[idx]
        i1s = i1[idx]
        s1 = ll * i1s + 2.0 * mu * e1[idx] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        s2 = ll * i1s + 2.0 * mu * e2[idx] - lam * (2.0 * ll * sp)
        s3 = ll * i1s + 2.0 * mu * e3[idx] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
        stress_principal[idx] = np.column_stack([s1, s2, s3])
        stress_tensor[idx] = reconstruct_from_principal(stress_principal[idx], eigvecs[idx])

    if np.any(test_l):
        idx = np.where(test_l)[0]
        lam = lambda_l[idx]
        plastic_multiplier[idx] = lam
        sp = sin_psi[idx]
        ll = lame[idx]
        mu = shear[idx]
        i1l = i1[idx]
        sigma12 = ll * i1l + mu * (e1[idx] + e2[idx]) - lam * (2.0 * ll * sp + mu * (1.0 + sp))
        sigma3 = ll * i1l + 2.0 * mu * e3[idx] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
        stress_principal[idx] = np.column_stack([sigma12, sigma12, sigma3])
        v3 = eigvecs[idx, :, 2]
        p3 = np.einsum("ni,nj->nij", v3, v3)
        stress_tensor[idx] = sigma12[:, None, None] * (eye[:, :, :] - p3) + sigma3[:, None, None] * p3

    if np.any(test_r):
        idx = np.where(test_r)[0]
        lam = lambda_r[idx]
        plastic_multiplier[idx] = lam
        sp = sin_psi[idx]
        ll = lame[idx]
        mu = shear[idx]
        i1r = i1[idx]
        sigma1 = ll * i1r + 2.0 * mu * e1[idx] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        sigma23 = ll * i1r + mu * (e2[idx] + e3[idx]) - lam * (2.0 * ll * sp - mu * (1.0 - sp))
        stress_principal[idx] = np.column_stack([sigma1, sigma23, sigma23])
        v1 = eigvecs[idx, :, 0]
        p1 = np.einsum("ni,nj->nij", v1, v1)
        stress_tensor[idx] = sigma1[:, None, None] * p1 + sigma23[:, None, None] * (eye[:, :, :] - p1)

    if np.any(test_a):
        idx = np.where(test_a)[0]
        denom = np.where(np.abs(sin_phi[idx]) > 1.0e-12, 2.0 * sin_phi[idx], np.inf)
        sigma_a = c_bar[idx] / denom
        plastic_multiplier[idx] = lambda_a[idx]
        stress_principal[idx] = np.column_stack([sigma_a, sigma_a, sigma_a])
        stress_tensor[idx] = sigma_a[:, None, None] * np.eye(3)[None, :, :]

    stress = tensor_to_stress_voigt(stress_tensor)

    tangent = None
    if return_tangent:
        tangent = numerical_tangent_fd(
            strain,
            c_bar=c_bar,
            sin_phi=sin_phi,
            shear=shear,
            bulk=bulk,
            lame=lame,
            eps=tangent_eps,
        )

    return ConstitutiveResult(
        stress=stress,
        stress_principal=stress_principal,
        strain_principal=eigvals,
        eigvecs=eigvecs,
        branch_id=branch_id,
        f_trial=f_trial,
        plastic_multiplier=plastic_multiplier,
        tangent=tangent,
    )


def numerical_tangent_fd(
    strain_eng: np.ndarray,
    *,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Numerical centered finite-difference tangent dσ/dε_eng."""
    strain = np.asarray(strain_eng, dtype=float)
    if strain.ndim == 1:
        strain = strain[None, :]
    n = strain.shape[0]
    tangent = np.zeros((n, 6, 6), dtype=float)
    for j in range(6):
        perturb = np.zeros_like(strain)
        perturb[:, j] = eps
        plus = constitutive_update_3d(
            strain + perturb,
            c_bar=c_bar,
            sin_phi=sin_phi,
            shear=shear,
            bulk=bulk,
            lame=lame,
            return_tangent=False,
        ).stress
        minus = constitutive_update_3d(
            strain - perturb,
            c_bar=c_bar,
            sin_phi=sin_phi,
            shear=shear,
            bulk=bulk,
            lame=lame,
            return_tangent=False,
        ).stress
        tangent[:, :, j] = (plus - minus) / (2.0 * eps)
    return tangent


def branch_name(branch_id: int | np.ndarray) -> str | np.ndarray:
    """Map branch ids to branch names."""
    if np.isscalar(branch_id):
        return BRANCH_NAMES[int(branch_id)]
    ids = np.asarray(branch_id, dtype=int)
    return np.array([BRANCH_NAMES[i] for i in ids], dtype=object)


def trial_strain_tensor(strain_eng: np.ndarray) -> np.ndarray:
    """Return the symmetric trial strain tensor."""
    return strain_voigt_to_tensor(strain_eng)
