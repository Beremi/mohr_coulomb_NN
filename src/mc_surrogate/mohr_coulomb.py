"""3D elastic-perfectly-plastic Mohr-Coulomb constitutive operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .voigt import (
    principal_values_and_vectors_from_strain,
    strain_voigt_to_tensor,
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


_IOTA = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)


def _strain_to_stress_notation(strain_eng: np.ndarray) -> np.ndarray:
    """Convert engineering Voigt strain to the MATLAB routine's stress notation."""
    strain = np.asarray(strain_eng, dtype=float)
    out = strain.copy()
    out[:, 3:] *= 0.5
    return out


def _square_in_stress_notation(strain_stress_notation: np.ndarray) -> np.ndarray:
    """Return the Voigt representation of E_tr^2 used by the MATLAB routine."""
    e = np.asarray(strain_stress_notation, dtype=float)
    out = np.empty_like(e)
    out[:, 0] = e[:, 0] ** 2 + e[:, 3] ** 2 + e[:, 5] ** 2
    out[:, 1] = e[:, 1] ** 2 + e[:, 3] ** 2 + e[:, 4] ** 2
    out[:, 2] = e[:, 2] ** 2 + e[:, 4] ** 2 + e[:, 5] ** 2
    out[:, 3] = e[:, 0] * e[:, 3] + e[:, 1] * e[:, 3] + e[:, 4] * e[:, 5]
    out[:, 4] = e[:, 3] * e[:, 5] + e[:, 1] * e[:, 4] + e[:, 2] * e[:, 4]
    out[:, 5] = e[:, 0] * e[:, 5] + e[:, 3] * e[:, 4] + e[:, 2] * e[:, 5]
    return out


def _matlab_principal_strains(strain_eng: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the trial-strain invariants and ordered principal values using the
    same closed-form formulas as the upstream MATLAB constitutive routine.
    """
    e_tr = _strain_to_stress_notation(strain_eng)
    e_sq = _square_in_stress_notation(e_tr)

    i1 = e_tr[:, 0] + e_tr[:, 1] + e_tr[:, 2]
    i2 = e_tr[:, 0] * e_tr[:, 1] + e_tr[:, 0] * e_tr[:, 2] + e_tr[:, 1] * e_tr[:, 2]
    i2 -= e_tr[:, 3] ** 2 + e_tr[:, 4] ** 2 + e_tr[:, 5] ** 2
    i3 = e_tr[:, 0] * e_tr[:, 1] * e_tr[:, 2]
    i3 -= e_tr[:, 2] * e_tr[:, 3] ** 2
    i3 -= e_tr[:, 1] * e_tr[:, 5] ** 2
    i3 -= e_tr[:, 0] * e_tr[:, 4] ** 2
    i3 += 2.0 * e_tr[:, 3] * e_tr[:, 4] * e_tr[:, 5]

    q = np.maximum(0.0, (i1**2 - 3.0 * i2) / 9.0)
    r = (-2.0 * i1**3 + 9.0 * i1 * i2 - 27.0 * i3) / 54.0
    theta0 = np.zeros_like(i1)
    mask = q > 0.0
    theta0[mask] = r[mask] / np.sqrt(q[mask] ** 3)
    theta = np.arccos(np.clip(theta0, -1.0, 1.0)) / 3.0
    sqrt_q = np.sqrt(q)

    eig1 = -2.0 * sqrt_q * np.cos(theta + 2.0 * np.pi / 3.0) + i1 / 3.0
    eig2 = -2.0 * sqrt_q * np.cos(theta - 2.0 * np.pi / 3.0) + i1 / 3.0
    eig3 = -2.0 * sqrt_q * np.cos(theta) + i1 / 3.0
    eigvals = np.column_stack([eig1, eig2, eig3])
    return eigvals, e_tr, e_sq, i1


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

    eigvals, e_tr, e_sq, i1 = _matlab_principal_strains(strain)
    _, eigvecs = principal_values_and_vectors_from_strain(strain)
    e1 = eigvals[:, 0]
    e2 = eigvals[:, 1]
    e3 = eigvals[:, 2]
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
    stress = np.zeros((n, 6), dtype=float)
    plastic_multiplier = np.zeros(n, dtype=float)

    sigma_trial = lame[:, None] * i1[:, None] + 2.0 * shear[:, None] * eigvals

    if np.any(test_el):
        idx = np.where(test_el)[0]
        stress_principal[idx] = sigma_trial[idx]
        stress[idx] = lame[idx, None] * i1[idx, None] * _IOTA[None, :]
        stress[idx] += 2.0 * shear[idx, None] * e_tr[idx]

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
        denom1 = np.where(np.abs((e1[idx] - e2[idx]) * (e1[idx] - e3[idx])) > 1.0e-14, (e1[idx] - e2[idx]) * (e1[idx] - e3[idx]), np.inf)
        denom2 = np.where(np.abs((e2[idx] - e1[idx]) * (e2[idx] - e3[idx])) > 1.0e-14, (e2[idx] - e1[idx]) * (e2[idx] - e3[idx]), np.inf)
        denom3 = np.where(np.abs((e3[idx] - e1[idx]) * (e3[idx] - e2[idx])) > 1.0e-14, (e3[idx] - e1[idx]) * (e3[idx] - e2[idx]), np.inf)
        eig1_proj = (e_sq[idx] - (e2[idx] + e3[idx])[:, None] * e_tr[idx] + (e2[idx] * e3[idx])[:, None] * _IOTA[None, :]) / denom1[:, None]
        eig2_proj = (e_sq[idx] - (e1[idx] + e3[idx])[:, None] * e_tr[idx] + (e1[idx] * e3[idx])[:, None] * _IOTA[None, :]) / denom2[:, None]
        eig3_proj = (e_sq[idx] - (e1[idx] + e2[idx])[:, None] * e_tr[idx] + (e1[idx] * e2[idx])[:, None] * _IOTA[None, :]) / denom3[:, None]
        stress[idx] = s1[:, None] * eig1_proj + s2[:, None] * eig2_proj + s3[:, None] * eig3_proj

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
        denom3 = np.where(np.abs((e3[idx] - e1[idx]) * (e3[idx] - e2[idx])) > 1.0e-14, (e3[idx] - e1[idx]) * (e3[idx] - e2[idx]), np.inf)
        eig3_proj = (e_sq[idx] - (e1[idx] + e2[idx])[:, None] * e_tr[idx] + (e1[idx] * e2[idx])[:, None] * _IOTA[None, :]) / denom3[:, None]
        eig12_proj = _IOTA[None, :] - eig3_proj
        stress[idx] = sigma12[:, None] * eig12_proj + sigma3[:, None] * eig3_proj

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
        denom1 = np.where(np.abs((e1[idx] - e2[idx]) * (e1[idx] - e3[idx])) > 1.0e-14, (e1[idx] - e2[idx]) * (e1[idx] - e3[idx]), np.inf)
        eig1_proj = (e_sq[idx] - (e2[idx] + e3[idx])[:, None] * e_tr[idx] + (e2[idx] * e3[idx])[:, None] * _IOTA[None, :]) / denom1[:, None]
        eig23_proj = _IOTA[None, :] - eig1_proj
        stress[idx] = sigma1[:, None] * eig1_proj + sigma23[:, None] * eig23_proj

    if np.any(test_a):
        idx = np.where(test_a)[0]
        denom = np.where(np.abs(sin_phi[idx]) > 1.0e-12, 2.0 * sin_phi[idx], np.inf)
        sigma_a = c_bar[idx] / denom
        plastic_multiplier[idx] = lambda_a[idx]
        stress_principal[idx] = np.column_stack([sigma_a, sigma_a, sigma_a])
        stress[idx] = sigma_a[:, None] * _IOTA[None, :]

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
