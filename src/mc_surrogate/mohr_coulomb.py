"""3D elastic-perfectly-plastic Mohr-Coulomb constitutive operator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .branch_geometry import compute_branch_geometry_principal
from .voigt import (
    principal_values_and_vectors_from_strain,
    strain_voigt_to_tensor,
)

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")
BRANCH_TO_ID = {name: i for i, name in enumerate(BRANCH_NAMES)}
_EXACT_LATENT_KIND_ELASTIC = "elastic_analytic"
_EXACT_LATENT_KIND_PLM = "plastic_multiplier"
_EXACT_LATENT_KIND_ANALYTIC = "analytic"
_EXACT_LATENT_KIND_BY_BRANCH = {
    BRANCH_TO_ID["elastic"]: _EXACT_LATENT_KIND_ELASTIC,
    BRANCH_TO_ID["smooth"]: _EXACT_LATENT_KIND_PLM,
    BRANCH_TO_ID["left_edge"]: _EXACT_LATENT_KIND_PLM,
    BRANCH_TO_ID["right_edge"]: _EXACT_LATENT_KIND_PLM,
    BRANCH_TO_ID["apex"]: _EXACT_LATENT_KIND_ANALYTIC,
}
_EXACT_LATENT_DIM_BY_BRANCH = {
    BRANCH_TO_ID["elastic"]: 0,
    BRANCH_TO_ID["smooth"]: 1,
    BRANCH_TO_ID["left_edge"]: 1,
    BRANCH_TO_ID["right_edge"]: 1,
    BRANCH_TO_ID["apex"]: 0,
}


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


@dataclass
class BranchHarmMetrics:
    """Stress-based harm classification for predicted-vs-exact branch outcomes."""

    exact_branch_id: np.ndarray
    predicted_branch_id: np.ndarray
    wrong_branch: np.ndarray
    benign_fail: np.ndarray
    harmful_fail: np.ndarray
    adjacent_fail: np.ndarray
    non_adjacent_fail: np.ndarray
    harmful_adjacent_fail: np.ndarray
    harmful_non_adjacent_fail: np.ndarray
    rel_e_sigma: np.ndarray
    abs_f_trial: np.ndarray
    abs_gamma_sl_minus_lambda_s: np.ndarray
    abs_gamma_sr_minus_lambda_s: np.ndarray
    abs_gamma_la_minus_lambda_l: np.ndarray
    abs_gamma_ra_minus_lambda_r: np.ndarray


@dataclass
class _PrincipalState3D:
    strain: np.ndarray
    c_bar: np.ndarray
    sin_phi: np.ndarray
    shear: np.ndarray
    bulk: np.ndarray
    lame: np.ndarray
    sin_psi: np.ndarray
    eigvals: np.ndarray
    e_tr: np.ndarray
    e_sq: np.ndarray
    i1: np.ndarray
    eigvecs: np.ndarray


@dataclass
class _BranchState3D(_PrincipalState3D):
    f_trial: np.ndarray
    gamma_sl: np.ndarray
    gamma_sr: np.ndarray
    gamma_la: np.ndarray
    gamma_ra: np.ndarray
    lambda_s: np.ndarray
    lambda_l: np.ndarray
    lambda_r: np.ndarray
    lambda_a: np.ndarray


def _broadcast_materials(n: int, *arrays: np.ndarray | float) -> list[np.ndarray]:
    out = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])
    return [a.reshape(-1) if a.size == n else np.broadcast_to(a, (n,)).reshape(-1) for a in out]


_IOTA = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
_ADJACENT_BRANCH_PAIRS = {
    (BRANCH_TO_ID["elastic"], BRANCH_TO_ID["smooth"]),
    (BRANCH_TO_ID["smooth"], BRANCH_TO_ID["left_edge"]),
    (BRANCH_TO_ID["smooth"], BRANCH_TO_ID["right_edge"]),
    (BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["apex"]),
    (BRANCH_TO_ID["right_edge"], BRANCH_TO_ID["apex"]),
}


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


def _coerce_strain_batch(strain_eng: np.ndarray) -> np.ndarray:
    strain = np.asarray(strain_eng, dtype=float)
    if strain.ndim == 1:
        strain = strain[None, :]
    if strain.ndim != 2 or strain.shape[1] != 6:
        raise ValueError(f"Expected strain shape (n,6), got {strain.shape}.")
    return strain


def _build_principal_state_3d(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
) -> _PrincipalState3D:
    strain = _coerce_strain_batch(strain_eng)
    n = strain.shape[0]
    c_bar, sin_phi, shear, bulk, lame = _broadcast_materials(n, c_bar, sin_phi, shear, bulk, lame)
    sin_psi = sin_phi.copy()
    eigvals, e_tr, e_sq, i1 = _matlab_principal_strains(strain)
    _, eigvecs = principal_values_and_vectors_from_strain(strain)
    return _PrincipalState3D(
        strain=strain,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
        sin_psi=sin_psi,
        eigvals=eigvals,
        e_tr=e_tr,
        e_sq=e_sq,
        i1=i1,
        eigvecs=eigvecs,
    )


def _build_branch_state_from_principal(principal: _PrincipalState3D) -> _BranchState3D:
    e1 = principal.eigvals[:, 0]
    e2 = principal.eigvals[:, 1]
    e3 = principal.eigvals[:, 2]

    f_trial = (
        2.0 * principal.shear * ((1.0 + principal.sin_phi) * e1 - (1.0 - principal.sin_phi) * e3)
        + 2.0 * (principal.lame * principal.sin_phi) * principal.i1
        - principal.c_bar
    )

    gamma_sl = (e1 - e2) / np.maximum(1.0 + principal.sin_psi, 1.0e-14)
    gamma_sr = (e2 - e3) / np.maximum(1.0 - principal.sin_psi, 1.0e-14)
    gamma_la = (e1 + e2 - 2.0 * e3) / np.maximum(3.0 - principal.sin_psi, 1.0e-14)
    gamma_ra = (2.0 * e1 - e2 - e3) / np.maximum(3.0 + principal.sin_psi, 1.0e-14)

    denom_s = 4.0 * principal.lame * principal.sin_phi * principal.sin_psi + 4.0 * principal.shear * (1.0 + principal.sin_phi * principal.sin_psi)
    denom_l = (
        4.0 * principal.lame * principal.sin_phi * principal.sin_psi
        + principal.shear * (1.0 + principal.sin_phi) * (1.0 + principal.sin_psi)
        + 2.0 * principal.shear * (1.0 - principal.sin_phi) * (1.0 - principal.sin_psi)
    )
    denom_r = (
        4.0 * principal.lame * principal.sin_phi * principal.sin_psi
        + 2.0 * principal.shear * (1.0 + principal.sin_phi) * (1.0 + principal.sin_psi)
        + principal.shear * (1.0 - principal.sin_phi) * (1.0 - principal.sin_psi)
    )
    denom_a = 4.0 * principal.bulk * principal.sin_phi * principal.sin_psi

    safe_denom_s = np.where(np.abs(denom_s) > 1.0e-14, denom_s, np.inf)
    safe_denom_l = np.where(np.abs(denom_l) > 1.0e-14, denom_l, np.inf)
    safe_denom_r = np.where(np.abs(denom_r) > 1.0e-14, denom_r, np.inf)
    safe_denom_a = np.where(np.abs(denom_a) > 1.0e-14, denom_a, np.inf)

    lambda_s = f_trial / safe_denom_s
    lambda_l = (
        principal.shear * ((1.0 + principal.sin_phi) * (e1 + e2) - 2.0 * (1.0 - principal.sin_phi) * e3)
        + 2.0 * principal.lame * principal.sin_phi * principal.i1
        - principal.c_bar
    ) / safe_denom_l
    lambda_r = (
        principal.shear * (2.0 * (1.0 + principal.sin_phi) * e1 - (1.0 - principal.sin_phi) * (e2 + e3))
        + 2.0 * principal.lame * principal.sin_phi * principal.i1
        - principal.c_bar
    ) / safe_denom_r
    lambda_a = (2.0 * principal.bulk * principal.sin_phi * principal.i1 - principal.c_bar) / safe_denom_a

    return _BranchState3D(
        **vars(principal),
        f_trial=f_trial,
        gamma_sl=gamma_sl,
        gamma_sr=gamma_sr,
        gamma_la=gamma_la,
        gamma_ra=gamma_ra,
        lambda_s=lambda_s,
        lambda_l=lambda_l,
        lambda_r=lambda_r,
        lambda_a=lambda_a,
    )


def _resolve_branch_id_from_state(
    state: _BranchState3D,
    *,
    branch_tol: float = 1.0e-10,
) -> np.ndarray:
    test_el = state.f_trial <= branch_tol
    test_s = (~test_el) & (state.lambda_s <= np.minimum(state.gamma_sl, state.gamma_sr) + branch_tol)
    test_l = (
        (~test_el)
        & (~test_s)
        & (state.gamma_sl < state.gamma_sr + branch_tol)
        & (state.lambda_l >= state.gamma_sl - branch_tol)
        & (state.lambda_l <= state.gamma_la + branch_tol)
    )
    test_r = (
        (~test_el)
        & (~test_s)
        & (state.gamma_sl > state.gamma_sr - branch_tol)
        & (state.lambda_r >= state.gamma_sr - branch_tol)
        & (state.lambda_r <= state.gamma_ra + branch_tol)
    )
    test_a = ~(test_el | test_s | test_l | test_r)

    branch_id = np.full(state.strain.shape[0], -1, dtype=np.int64)
    branch_id[test_el] = BRANCH_TO_ID["elastic"]
    branch_id[test_s] = BRANCH_TO_ID["smooth"]
    branch_id[test_l] = BRANCH_TO_ID["left_edge"]
    branch_id[test_r] = BRANCH_TO_ID["right_edge"]
    branch_id[test_a] = BRANCH_TO_ID["apex"]
    return branch_id


def _dispatch_from_branch_state(
    state: _BranchState3D,
    branch_id: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    n = state.strain.shape[0]
    if branch.size == 1 and n > 1:
        branch = np.full(n, int(branch.item()), dtype=np.int64)
    if branch.shape[0] != n:
        raise ValueError(f"Branch ids shape {branch.shape} does not match strain batch {n}.")
    if np.any((branch < 0) | (branch >= len(BRANCH_NAMES))):
        raise ValueError("Branch ids must be in [0, 4].")

    stress_principal = np.zeros((n, 3), dtype=float)
    stress = np.zeros((n, 6), dtype=float)
    plastic_multiplier = np.zeros(n, dtype=float)

    e1 = state.eigvals[:, 0]
    e2 = state.eigvals[:, 1]
    e3 = state.eigvals[:, 2]
    sigma_trial = state.lame[:, None] * state.i1[:, None] + 2.0 * state.shear[:, None] * state.eigvals

    test_el = branch == BRANCH_TO_ID["elastic"]
    test_s = branch == BRANCH_TO_ID["smooth"]
    test_l = branch == BRANCH_TO_ID["left_edge"]
    test_r = branch == BRANCH_TO_ID["right_edge"]
    test_a = branch == BRANCH_TO_ID["apex"]

    if np.any(test_el):
        idx = np.where(test_el)[0]
        stress_principal[idx] = sigma_trial[idx]
        stress[idx] = state.lame[idx, None] * state.i1[idx, None] * _IOTA[None, :]
        stress[idx] += 2.0 * state.shear[idx, None] * state.e_tr[idx]

    if np.any(test_s):
        idx = np.where(test_s)[0]
        lam = state.lambda_s[idx]
        plastic_multiplier[idx] = lam
        sp = state.sin_psi[idx]
        ll = state.lame[idx]
        mu = state.shear[idx]
        i1s = state.i1[idx]
        s1 = ll * i1s + 2.0 * mu * e1[idx] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        s2 = ll * i1s + 2.0 * mu * e2[idx] - lam * (2.0 * ll * sp)
        s3 = ll * i1s + 2.0 * mu * e3[idx] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
        stress_principal[idx] = np.column_stack([s1, s2, s3])
        denom1 = np.where(np.abs((e1[idx] - e2[idx]) * (e1[idx] - e3[idx])) > 1.0e-14, (e1[idx] - e2[idx]) * (e1[idx] - e3[idx]), np.inf)
        denom2 = np.where(np.abs((e2[idx] - e1[idx]) * (e2[idx] - e3[idx])) > 1.0e-14, (e2[idx] - e1[idx]) * (e2[idx] - e3[idx]), np.inf)
        denom3 = np.where(np.abs((e3[idx] - e1[idx]) * (e3[idx] - e2[idx])) > 1.0e-14, (e3[idx] - e1[idx]) * (e3[idx] - e2[idx]), np.inf)
        eig1_proj = (state.e_sq[idx] - (e2[idx] + e3[idx])[:, None] * state.e_tr[idx] + (e2[idx] * e3[idx])[:, None] * _IOTA[None, :]) / denom1[:, None]
        eig2_proj = (state.e_sq[idx] - (e1[idx] + e3[idx])[:, None] * state.e_tr[idx] + (e1[idx] * e3[idx])[:, None] * _IOTA[None, :]) / denom2[:, None]
        eig3_proj = (state.e_sq[idx] - (e1[idx] + e2[idx])[:, None] * state.e_tr[idx] + (e1[idx] * e2[idx])[:, None] * _IOTA[None, :]) / denom3[:, None]
        stress[idx] = s1[:, None] * eig1_proj + s2[:, None] * eig2_proj + s3[:, None] * eig3_proj

    if np.any(test_l):
        idx = np.where(test_l)[0]
        lam = state.lambda_l[idx]
        plastic_multiplier[idx] = lam
        sp = state.sin_psi[idx]
        ll = state.lame[idx]
        mu = state.shear[idx]
        i1l = state.i1[idx]
        sigma12 = ll * i1l + mu * (e1[idx] + e2[idx]) - lam * (2.0 * ll * sp + mu * (1.0 + sp))
        sigma3 = ll * i1l + 2.0 * mu * e3[idx] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
        stress_principal[idx] = np.column_stack([sigma12, sigma12, sigma3])
        denom3 = np.where(np.abs((e3[idx] - e1[idx]) * (e3[idx] - e2[idx])) > 1.0e-14, (e3[idx] - e1[idx]) * (e3[idx] - e2[idx]), np.inf)
        eig3_proj = (state.e_sq[idx] - (e1[idx] + e2[idx])[:, None] * state.e_tr[idx] + (e1[idx] * e2[idx])[:, None] * _IOTA[None, :]) / denom3[:, None]
        eig12_proj = _IOTA[None, :] - eig3_proj
        stress[idx] = sigma12[:, None] * eig12_proj + sigma3[:, None] * eig3_proj

    if np.any(test_r):
        idx = np.where(test_r)[0]
        lam = state.lambda_r[idx]
        plastic_multiplier[idx] = lam
        sp = state.sin_psi[idx]
        ll = state.lame[idx]
        mu = state.shear[idx]
        i1r = state.i1[idx]
        sigma1 = ll * i1r + 2.0 * mu * e1[idx] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        sigma23 = ll * i1r + mu * (e2[idx] + e3[idx]) - lam * (2.0 * ll * sp - mu * (1.0 - sp))
        stress_principal[idx] = np.column_stack([sigma1, sigma23, sigma23])
        denom1 = np.where(np.abs((e1[idx] - e2[idx]) * (e1[idx] - e3[idx])) > 1.0e-14, (e1[idx] - e2[idx]) * (e1[idx] - e3[idx]), np.inf)
        eig1_proj = (state.e_sq[idx] - (e2[idx] + e3[idx])[:, None] * state.e_tr[idx] + (e2[idx] * e3[idx])[:, None] * _IOTA[None, :]) / denom1[:, None]
        eig23_proj = _IOTA[None, :] - eig1_proj
        stress[idx] = sigma1[:, None] * eig1_proj + sigma23[:, None] * eig23_proj

    if np.any(test_a):
        idx = np.where(test_a)[0]
        denom = np.where(np.abs(state.sin_phi[idx]) > 1.0e-12, 2.0 * state.sin_phi[idx], np.inf)
        sigma_a = state.c_bar[idx] / denom
        plastic_multiplier[idx] = state.lambda_a[idx]
        stress_principal[idx] = np.column_stack([sigma_a, sigma_a, sigma_a])
        stress[idx] = sigma_a[:, None] * _IOTA[None, :]

    return stress, stress_principal, plastic_multiplier


def candidate_principal_stresses_3d(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
) -> np.ndarray:
    """
    Return the principal-stress candidate for every Mohr-Coulomb branch.

    Output shape is `(n, 5, 3)` ordered as:
    `elastic, smooth, left_edge, right_edge, apex`.
    """
    strain = _coerce_strain_batch(strain_eng)
    n = strain.shape[0]
    c_bar, sin_phi, shear, bulk, lame = _broadcast_materials(n, c_bar, sin_phi, shear, bulk, lame)
    sin_psi = sin_phi.copy()

    eigvals, _e_tr, _e_sq, i1 = _matlab_principal_strains(strain)
    e1 = eigvals[:, 0]
    e2 = eigvals[:, 1]
    e3 = eigvals[:, 2]
    f_trial = 2.0 * shear * ((1.0 + sin_phi) * e1 - (1.0 - sin_phi) * e3) + 2.0 * (lame * sin_phi) * i1 - c_bar

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

    sigma_trial = lame[:, None] * i1[:, None] + 2.0 * shear[:, None] * eigvals
    out = np.empty((n, len(BRANCH_NAMES), 3), dtype=float)
    out[:, BRANCH_TO_ID["elastic"], :] = sigma_trial

    s1 = lame * i1 + 2.0 * shear * e1 - lambda_s * (2.0 * lame * sin_psi + 2.0 * shear * (1.0 + sin_psi))
    s2 = lame * i1 + 2.0 * shear * e2 - lambda_s * (2.0 * lame * sin_psi)
    s3 = lame * i1 + 2.0 * shear * e3 - lambda_s * (2.0 * lame * sin_psi - 2.0 * shear * (1.0 - sin_psi))
    out[:, BRANCH_TO_ID["smooth"], :] = np.column_stack([s1, s2, s3])

    sigma12 = lame * i1 + shear * (e1 + e2) - lambda_l * (2.0 * lame * sin_psi + shear * (1.0 + sin_psi))
    sigma3 = lame * i1 + 2.0 * shear * e3 - lambda_l * (2.0 * lame * sin_psi - 2.0 * shear * (1.0 - sin_psi))
    out[:, BRANCH_TO_ID["left_edge"], :] = np.column_stack([sigma12, sigma12, sigma3])

    sigma1 = lame * i1 + 2.0 * shear * e1 - lambda_r * (2.0 * lame * sin_psi + 2.0 * shear * (1.0 + sin_psi))
    sigma23 = lame * i1 + shear * (e2 + e3) - lambda_r * (2.0 * lame * sin_psi - shear * (1.0 - sin_psi))
    out[:, BRANCH_TO_ID["right_edge"], :] = np.column_stack([sigma1, sigma23, sigma23])

    sigma_a = c_bar / np.where(np.abs(sin_phi) > 1.0e-12, 2.0 * sin_phi, np.inf)
    out[:, BRANCH_TO_ID["apex"], :] = np.column_stack([sigma_a, sigma_a, sigma_a])
    return out


def candidate_stresses_3d(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
) -> np.ndarray:
    """Return full Voigt stress candidates for every branch with shape `(n, 5, 6)`."""
    strain = _coerce_strain_batch(strain_eng)
    n = strain.shape[0]
    c_bar, sin_phi, shear, bulk, lame = _broadcast_materials(n, c_bar, sin_phi, shear, bulk, lame)
    sin_psi = sin_phi.copy()

    eigvals, e_tr, e_sq, i1 = _matlab_principal_strains(strain)
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
    safe_denom_s = np.where(np.abs(denom_s) > 1.0e-14, denom_s, np.inf)
    safe_denom_l = np.where(np.abs(denom_l) > 1.0e-14, denom_l, np.inf)
    safe_denom_r = np.where(np.abs(denom_r) > 1.0e-14, denom_r, np.inf)

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

    stress = np.zeros((n, len(BRANCH_NAMES), 6), dtype=float)
    sigma_trial = lame[:, None] * i1[:, None] + 2.0 * shear[:, None] * eigvals
    stress[:, BRANCH_TO_ID["elastic"], :] = lame[:, None] * i1[:, None] * _IOTA[None, :]
    stress[:, BRANCH_TO_ID["elastic"], :] += 2.0 * shear[:, None] * e_tr

    s1 = lame * i1 + 2.0 * shear * e1 - lambda_s * (2.0 * lame * sin_psi + 2.0 * shear * (1.0 + sin_psi))
    s2 = lame * i1 + 2.0 * shear * e2 - lambda_s * (2.0 * lame * sin_psi)
    s3 = lame * i1 + 2.0 * shear * e3 - lambda_s * (2.0 * lame * sin_psi - 2.0 * shear * (1.0 - sin_psi))
    denom1 = np.where(np.abs((e1 - e2) * (e1 - e3)) > 1.0e-14, (e1 - e2) * (e1 - e3), np.inf)
    denom2 = np.where(np.abs((e2 - e1) * (e2 - e3)) > 1.0e-14, (e2 - e1) * (e2 - e3), np.inf)
    denom3 = np.where(np.abs((e3 - e1) * (e3 - e2)) > 1.0e-14, (e3 - e1) * (e3 - e2), np.inf)
    eig1_proj = (e_sq - (e2 + e3)[:, None] * e_tr + (e2 * e3)[:, None] * _IOTA[None, :]) / denom1[:, None]
    eig2_proj = (e_sq - (e1 + e3)[:, None] * e_tr + (e1 * e3)[:, None] * _IOTA[None, :]) / denom2[:, None]
    eig3_proj = (e_sq - (e1 + e2)[:, None] * e_tr + (e1 * e2)[:, None] * _IOTA[None, :]) / denom3[:, None]
    stress[:, BRANCH_TO_ID["smooth"], :] = s1[:, None] * eig1_proj + s2[:, None] * eig2_proj + s3[:, None] * eig3_proj

    sigma12 = lame * i1 + shear * (e1 + e2) - lambda_l * (2.0 * lame * sin_psi + shear * (1.0 + sin_psi))
    sigma3 = lame * i1 + 2.0 * shear * e3 - lambda_l * (2.0 * lame * sin_psi - 2.0 * shear * (1.0 - sin_psi))
    eig12_proj = _IOTA[None, :] - eig3_proj
    stress[:, BRANCH_TO_ID["left_edge"], :] = sigma12[:, None] * eig12_proj + sigma3[:, None] * eig3_proj

    sigma1 = lame * i1 + 2.0 * shear * e1 - lambda_r * (2.0 * lame * sin_psi + 2.0 * shear * (1.0 + sin_psi))
    sigma23 = lame * i1 + shear * (e2 + e3) - lambda_r * (2.0 * lame * sin_psi - shear * (1.0 - sin_psi))
    eig23_proj = _IOTA[None, :] - eig1_proj
    stress[:, BRANCH_TO_ID["right_edge"], :] = sigma1[:, None] * eig1_proj + sigma23[:, None] * eig23_proj

    sigma_a = c_bar / np.where(np.abs(sin_phi) > 1.0e-12, 2.0 * sin_phi, np.inf)
    stress[:, BRANCH_TO_ID["apex"], :] = sigma_a[:, None] * _IOTA[None, :]
    return stress


def dispatch_branch_stress_3d(
    strain_eng: np.ndarray,
    branch_id: np.ndarray | int,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch branch ids through the exact branch formulas and return `(stress, principal_stress)`."""
    strain = _coerce_strain_batch(strain_eng)
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    if branch.size == 1 and strain.shape[0] > 1:
        branch = np.full(strain.shape[0], int(branch.item()), dtype=np.int64)
    if branch.shape[0] != strain.shape[0]:
        raise ValueError(f"Branch ids shape {branch.shape} does not match strain batch {strain.shape[0]}.")
    if np.any((branch < 0) | (branch >= len(BRANCH_NAMES))):
        raise ValueError("Branch ids must be in [0, 4].")
    principal_candidates = candidate_principal_stresses_3d(
        strain,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    stress_candidates = candidate_stresses_3d(
        strain,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    rows = np.arange(strain.shape[0], dtype=np.int64)
    return stress_candidates[rows, branch], principal_candidates[rows, branch]


def _adjacent_branch_mask(exact_branch_id: np.ndarray, predicted_branch_id: np.ndarray) -> np.ndarray:
    exact = np.asarray(exact_branch_id, dtype=np.int64).reshape(-1)
    pred = np.asarray(predicted_branch_id, dtype=np.int64).reshape(-1)
    out = np.zeros(exact.shape[0], dtype=bool)
    for left, right in _ADJACENT_BRANCH_PAIRS:
        out |= ((exact == left) & (pred == right)) | ((exact == right) & (pred == left))
    return out


def branch_harm_metrics_3d(
    strain_eng: np.ndarray,
    predicted_branch_id: np.ndarray | int,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
    *,
    tau: float = 1.0e-2,
) -> BranchHarmMetrics:
    """
    Classify wrong branch predictions as benign or harmful using principal-stress error.
    """
    strain = _coerce_strain_batch(strain_eng)
    n = strain.shape[0]
    pred = np.asarray(predicted_branch_id, dtype=np.int64).reshape(-1)
    if pred.size == 1 and n > 1:
        pred = np.full(n, int(pred.item()), dtype=np.int64)
    if pred.shape[0] != n:
        raise ValueError(f"Predicted branch ids shape {pred.shape} does not match strain batch {n}.")
    if np.any((pred < 0) | (pred >= len(BRANCH_NAMES))):
        raise ValueError("Predicted branch ids must be in [0, 4].")

    exact = constitutive_update_3d(
        strain,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
        return_tangent=False,
    )
    _pred_stress, pred_principal = dispatch_branch_stress_3d(
        strain,
        pred,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    c_bar_arr, sin_phi_arr, shear_arr, bulk_arr, lame_arr = _broadcast_materials(n, c_bar, sin_phi, shear, bulk, lame)
    geom = compute_branch_geometry_principal(
        exact.strain_principal,
        c_bar=c_bar_arr,
        sin_phi=sin_phi_arr,
        shear=shear_arr,
        bulk=bulk_arr,
        lame=lame_arr,
    )

    exact_branch = np.asarray(exact.branch_id, dtype=np.int64).reshape(-1)
    wrong = pred != exact_branch
    denom = np.linalg.norm(exact.stress_principal, axis=1) + np.maximum(c_bar_arr, 0.0) + 1.0e-12
    rel_e_sigma = np.linalg.norm(pred_principal - exact.stress_principal, axis=1) / denom
    harmful = wrong & (rel_e_sigma > tau)
    benign = wrong & ~harmful
    adjacent = wrong & _adjacent_branch_mask(exact_branch, pred)
    non_adjacent = wrong & ~adjacent
    harmful_adjacent = harmful & adjacent
    harmful_non_adjacent = harmful & non_adjacent

    return BranchHarmMetrics(
        exact_branch_id=exact_branch,
        predicted_branch_id=pred,
        wrong_branch=wrong,
        benign_fail=benign,
        harmful_fail=harmful,
        adjacent_fail=adjacent,
        non_adjacent_fail=non_adjacent,
        harmful_adjacent_fail=harmful_adjacent,
        harmful_non_adjacent_fail=harmful_non_adjacent,
        rel_e_sigma=rel_e_sigma.astype(float),
        abs_f_trial=np.abs(exact.f_trial).astype(float),
        abs_gamma_sl_minus_lambda_s=np.abs(geom.gamma_sl - geom.lambda_s).astype(float),
        abs_gamma_sr_minus_lambda_s=np.abs(geom.gamma_sr - geom.lambda_s).astype(float),
        abs_gamma_la_minus_lambda_l=np.abs(geom.gamma_la - geom.lambda_l).astype(float),
        abs_gamma_ra_minus_lambda_r=np.abs(geom.gamma_ra - geom.lambda_r).astype(float),
    )


def yield_function_principal_3d(
    stress_principal: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the Mohr-Coulomb yield function on ordered principal stress."""
    sigma = np.asarray(stress_principal, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma[None, :]
    if sigma.ndim != 2 or sigma.shape[1] != 3:
        raise ValueError(f"Expected stress_principal shape (n, 3), got {sigma.shape}.")
    sigma_sorted = np.sort(sigma, axis=1)[:, ::-1]
    c_bar_arr, sin_phi_arr, *_ = _broadcast_materials(sigma.shape[0], c_bar, sin_phi, 1.0, 1.0, 1.0)
    return ((1.0 + sin_phi_arr) * sigma_sorted[:, 0] - (1.0 - sin_phi_arr) * sigma_sorted[:, 2] - c_bar_arr).astype(float)


def encode_principal_to_abr(
    stress_principal: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Encode ordered principal stress into admissible-gap coordinates."""
    sigma = np.asarray(stress_principal, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma[None, :]
    if sigma.ndim != 2 or sigma.shape[1] != 3:
        raise ValueError(f"Expected stress_principal shape (n, 3), got {sigma.shape}.")
    sigma_sorted = np.sort(sigma, axis=1)[:, ::-1]
    a = sigma_sorted[:, 0] - sigma_sorted[:, 1]
    b = sigma_sorted[:, 1] - sigma_sorted[:, 2]
    r_raw = -yield_function_principal_3d(sigma_sorted, c_bar=c_bar, sin_phi=sin_phi)
    r_nonneg = np.maximum(r_raw, 0.0)
    return {
        "principal_sorted": sigma_sorted.astype(float),
        "a": a.astype(float),
        "b": b.astype(float),
        "r_raw": r_raw.astype(float),
        "r_nonneg": r_nonneg.astype(float),
        "abr_raw": np.column_stack([a, b, r_raw]).astype(float),
        "abr_nonneg": np.column_stack([a, b, r_nonneg]).astype(float),
    }


def decode_abr_to_principal(
    abr: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """Decode `(a, b, r)` coordinates back to ordered principal stress."""
    abr_arr = np.asarray(abr, dtype=float)
    if abr_arr.ndim == 1:
        abr_arr = abr_arr[None, :]
    if abr_arr.ndim != 2 or abr_arr.shape[1] != 3:
        raise ValueError(f"Expected abr shape (n, 3), got {abr_arr.shape}.")
    n = abr_arr.shape[0]
    c_bar_arr, sin_phi_arr, *_ = _broadcast_materials(n, c_bar, sin_phi, 1.0, 1.0, 1.0)
    denom = np.where(np.abs(sin_phi_arr) > 1.0e-14, 2.0 * sin_phi_arr, np.inf)
    a = abr_arr[:, 0]
    b = abr_arr[:, 1]
    r = abr_arr[:, 2]
    sigma3 = (c_bar_arr - r - (1.0 + sin_phi_arr) * (a + b)) / denom
    sigma2 = sigma3 + b
    sigma1 = sigma2 + a
    return np.column_stack([sigma1, sigma2, sigma3]).astype(float)


def encode_principal_to_grho(
    stress_principal: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    *,
    eps: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    """
    Encode ordered principal stress into plastic-surface `(g, rho)` coordinates.

    This representation is intended for plastic states where `r = 0` and the
    final stress lies on the Mohr-Coulomb surface. Elastic rows may still be
    encoded for diagnostics, but exact reconstruction then requires the
    separate elastic dispatch rather than the plastic decoder alone.
    """
    encoded = encode_principal_to_abr(stress_principal, c_bar=c_bar, sin_phi=sin_phi)
    a = encoded["a"]
    b = encoded["b"]
    g = a + b

    rho = np.zeros_like(g)
    lambda_coord = np.zeros_like(g)
    positive = g > eps
    rho[positive] = (a[positive] - b[positive]) / g[positive]
    lambda_coord[positive] = a[positive] / g[positive]
    rho = np.clip(rho, -1.0, 1.0)
    lambda_coord = np.clip(lambda_coord, 0.0, 1.0)

    return {
        **encoded,
        "g": g.astype(float),
        "rho": rho.astype(float),
        "lambda_coord": lambda_coord.astype(float),
        "grho": np.column_stack([g, rho]).astype(float),
        "glambda": np.column_stack([g, lambda_coord]).astype(float),
    }


def decode_grho_to_principal(
    grho: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """Decode plastic-surface `(g, rho)` coordinates to ordered principal stress."""
    grho_arr = np.asarray(grho, dtype=float)
    if grho_arr.ndim == 1:
        grho_arr = grho_arr[None, :]
    if grho_arr.ndim != 2 or grho_arr.shape[1] != 2:
        raise ValueError(f"Expected grho shape (n, 2), got {grho_arr.shape}.")
    n = grho_arr.shape[0]
    c_bar_arr, sin_phi_arr, *_ = _broadcast_materials(n, c_bar, sin_phi, 1.0, 1.0, 1.0)
    denom = np.where(np.abs(sin_phi_arr) > 1.0e-14, 2.0 * sin_phi_arr, np.inf)
    g = np.maximum(grho_arr[:, 0], 0.0)
    rho = np.clip(grho_arr[:, 1], -1.0, 1.0)
    a = 0.5 * g * (1.0 + rho)
    b = 0.5 * g * (1.0 - rho)
    sigma3 = (c_bar_arr - (1.0 + sin_phi_arr) * g) / denom
    sigma2 = sigma3 + b
    sigma1 = sigma2 + a
    return np.column_stack([sigma1, sigma2, sigma3]).astype(float)


def project_grho_to_branch_specialized(
    grho: np.ndarray,
    branch_id: np.ndarray,
) -> np.ndarray:
    """
    Snap `(g, rho)` coordinates to the exact branch-specialized plastic geometry.

    Smooth rows retain their free `rho`. Left/right edges pin `rho` to `-1/+1`,
    and apex rows decode from `g = 0`, `rho = 0`. Elastic rows are left
    unchanged so exact elastic dispatch can continue to live upstream.
    """
    grho_arr = np.asarray(grho, dtype=float)
    branch_arr = np.asarray(branch_id, dtype=int).reshape(-1)
    if grho_arr.ndim == 1:
        grho_arr = grho_arr[None, :]
    if grho_arr.ndim != 2 or grho_arr.shape[1] != 2:
        raise ValueError(f"Expected grho shape (n, 2), got {grho_arr.shape}.")
    if branch_arr.shape[0] != grho_arr.shape[0]:
        raise ValueError(f"Expected branch_id length {grho_arr.shape[0]}, got {branch_arr.shape[0]}.")

    projected = grho_arr.copy()
    projected[:, 0] = np.maximum(projected[:, 0], 0.0)
    projected[:, 1] = np.clip(projected[:, 1], -1.0, 1.0)
    projected[branch_arr == BRANCH_TO_ID["left_edge"], 1] = -1.0
    projected[branch_arr == BRANCH_TO_ID["right_edge"], 1] = 1.0
    projected[branch_arr == BRANCH_TO_ID["apex"], :] = 0.0
    return projected.astype(float)


def decode_branch_specialized_grho_to_principal(
    grho: np.ndarray,
    branch_id: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """Decode branch-specialized plastic-surface coordinates to ordered principal stress."""
    projected = project_grho_to_branch_specialized(grho, branch_id)
    return decode_grho_to_principal(projected, c_bar=c_bar, sin_phi=sin_phi)


def decode_grho_to_principal_plastic(
    grho: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """Backward-compatible alias for the plastic-surface `(g, rho)` decode."""
    return decode_grho_to_principal(grho, c_bar=c_bar, sin_phi=sin_phi)


def infer_branch_from_abr(
    abr: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """
    Infer branch id from decoded ABR geometry using small state-dependent tolerances.

    This helper is for analysis/reporting only and mirrors the geometric
    interpretation in the Stage 0 audit plan.
    """
    abr_arr = np.asarray(abr, dtype=float)
    if abr_arr.ndim == 1:
        abr_arr = abr_arr[None, :]
    if abr_arr.ndim != 2 or abr_arr.shape[1] != 3:
        raise ValueError(f"Expected abr shape (n, 3), got {abr_arr.shape}.")
    sigma = decode_abr_to_principal(abr_arr, c_bar=c_bar, sin_phi=sin_phi)
    n = abr_arr.shape[0]
    c_bar_arr, *_ = _broadcast_materials(n, c_bar, sin_phi, 1.0, 1.0, 1.0)
    a = abr_arr[:, 0]
    b = abr_arr[:, 1]
    r = abr_arr[:, 2]
    gap_tol = 1.0e-6 * np.maximum(a + b, 1.0)
    r_tol = 1.0e-6 * np.maximum(np.linalg.norm(sigma, axis=1) + np.maximum(c_bar_arr, 0.0), 1.0)

    branch_id = np.full(n, BRANCH_TO_ID["smooth"], dtype=np.int64)
    elastic = r > r_tol
    apex = (~elastic) & (a <= gap_tol) & (b <= gap_tol)
    left_edge = (~elastic) & (a <= gap_tol) & ~apex
    right_edge = (~elastic) & (b <= gap_tol) & ~apex

    branch_id[elastic] = BRANCH_TO_ID["elastic"]
    branch_id[left_edge] = BRANCH_TO_ID["left_edge"]
    branch_id[right_edge] = BRANCH_TO_ID["right_edge"]
    branch_id[apex] = BRANCH_TO_ID["apex"]
    return branch_id


def yield_violation_rel_principal_3d(
    stress_principal: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
) -> np.ndarray:
    """
    Return the normalized positive yield violation on predicted principal stress.

    `0` means the state is yield-admissible or exactly on the surface.
    """
    sigma = np.asarray(stress_principal, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma[None, :]
    f_sigma = yield_function_principal_3d(sigma, c_bar=c_bar, sin_phi=sin_phi)
    c_bar_arr, *_ = _broadcast_materials(sigma.shape[0], c_bar, sin_phi, 1.0, 1.0, 1.0)
    denom = np.linalg.norm(sigma, axis=1) + np.maximum(c_bar_arr, 0.0) + 1.0e-12
    return (np.maximum(f_sigma, 0.0) / denom).astype(float)


def principal_relative_error_3d(
    predicted_principal: np.ndarray,
    true_principal: np.ndarray,
    c_bar: np.ndarray | float,
) -> np.ndarray:
    """Return relative principal-stress error normalized by the true stress scale."""
    pred = np.asarray(predicted_principal, dtype=float)
    true = np.asarray(true_principal, dtype=float)
    if pred.ndim == 1:
        pred = pred[None, :]
    if true.ndim == 1:
        true = true[None, :]
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(
            f"Expected predicted and true principal stress shape (n, 3), got {pred.shape} and {true.shape}."
        )
    c_bar_arr, *_ = _broadcast_materials(pred.shape[0], c_bar, 0.0, 1.0, 1.0, 1.0)
    denom = np.linalg.norm(true, axis=1) + np.maximum(c_bar_arr, 0.0) + 1.0e-12
    return (np.linalg.norm(pred - true, axis=1) / denom).astype(float)


def exact_trial_principal_stress_3d(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
) -> np.ndarray:
    """Return the exact ordered elastic-trial principal stress used by the constitutive routine."""
    principal_state = _build_principal_state_3d(
        strain_eng,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    sigma_trial = principal_state.lame[:, None] * principal_state.i1[:, None]
    sigma_trial = sigma_trial + 2.0 * principal_state.shear[:, None] * principal_state.eigvals
    return sigma_trial.astype(float)


def _coerce_branch_id_batch(branch_id: np.ndarray | int, n: int | None = None) -> np.ndarray:
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    if n is not None:
        if branch.size == 1 and n > 1:
            branch = np.full(n, int(branch.item()), dtype=np.int64)
        if branch.shape[0] != n:
            raise ValueError(f"Branch ids shape {branch.shape} does not match batch size {n}.")
    if np.any((branch < 0) | (branch >= len(BRANCH_NAMES))):
        raise ValueError("Branch ids must be in [0, 4].")
    return branch


def _exact_latent_kind_and_dim(branch_id: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    kind = np.array([_EXACT_LATENT_KIND_BY_BRANCH[int(b)] for b in branch], dtype=object)
    dim = np.array([_EXACT_LATENT_DIM_BY_BRANCH[int(b)] for b in branch], dtype=np.int64)
    return kind, dim


def _exact_branch_latent_values_from_state(state: _BranchState3D, branch_id: np.ndarray) -> np.ndarray:
    branch = _coerce_branch_id_batch(branch_id, n=state.strain.shape[0])
    values = np.empty(branch.shape[0], dtype=object)
    for idx, bid in enumerate(branch):
        if bid in (BRANCH_TO_ID["smooth"], BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]):
            latent = np.array([state.lambda_s[idx] if bid == BRANCH_TO_ID["smooth"] else state.lambda_l[idx] if bid == BRANCH_TO_ID["left_edge"] else state.lambda_r[idx]], dtype=float)
        else:
            latent = np.zeros((0,), dtype=float)
        values[idx] = latent
    return values


def _principal_strain_from_trial_principal(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trial = np.asarray(trial_principal, dtype=float)
    if trial.ndim == 1:
        trial = trial[None, :]
    if trial.ndim != 2 or trial.shape[1] != 3:
        raise ValueError(f"Expected trial_principal shape (n, 3), got {trial.shape}.")

    material = np.asarray(material_reduced, dtype=float)
    if material.ndim == 1:
        material = material[None, :]
    if material.ndim != 2 or material.shape[1] < 5:
        raise ValueError(f"Expected material_reduced shape (n, >=5), got {material.shape}.")
    if material.shape[0] != trial.shape[0]:
        raise ValueError(f"Trial/material batch mismatch: {trial.shape[0]} vs {material.shape[0]}.")

    lame = material[:, 4]
    shear = material[:, 2]
    denom_i1 = 3.0 * lame + 2.0 * shear
    safe_denom_i1 = np.where(np.abs(denom_i1) > 1.0e-14, denom_i1, np.inf)
    i1 = np.sum(trial, axis=1) / safe_denom_i1
    safe_shear = np.where(np.abs(shear) > 1.0e-14, shear, np.inf)
    eigvals = (trial - lame[:, None] * i1[:, None]) / (2.0 * safe_shear[:, None])
    return eigvals.astype(float), i1.astype(float)


def _decode_exact_branch_principal_batch(
    branch_id: np.ndarray | int,
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
    latent_values: np.ndarray,
) -> np.ndarray:
    branch = _coerce_branch_id_batch(branch_id)
    trial = np.asarray(trial_principal, dtype=float)
    if trial.ndim == 1:
        trial = trial[None, :]
    if trial.ndim != 2 or trial.shape[1] != 3:
        raise ValueError(f"Expected trial_principal shape (n, 3), got {trial.shape}.")

    material = np.asarray(material_reduced, dtype=float)
    if material.ndim == 1:
        material = material[None, :]
    if material.ndim != 2 or material.shape[1] < 2:
        raise ValueError(f"Expected material_reduced shape (n, >=2), got {material.shape}.")
    if material.shape[0] != trial.shape[0]:
        raise ValueError(f"Trial/material batch mismatch: {trial.shape[0]} vs {material.shape[0]}.")
    branch = _coerce_branch_id_batch(branch, n=trial.shape[0])

    latent = np.asarray(latent_values, dtype=float)
    if latent.ndim == 1:
        latent = latent[:, None]
    if latent.ndim != 2:
        raise ValueError(f"Expected latent_values shape (n, k), got {latent.shape}.")
    if latent.shape[0] != trial.shape[0]:
        raise ValueError(f"Latent batch mismatch: {latent.shape[0]} vs {trial.shape[0]}.")
    unique_branch = np.unique(branch)
    if unique_branch.size != 1:
        raise ValueError(
            "decode_exact_branch_latents_to_principal expects a homogeneous branch batch; "
            f"got branch ids {unique_branch.tolist()}."
        )

    eigvals, i1 = _principal_strain_from_trial_principal(trial, material)
    out = np.zeros((trial.shape[0], 3), dtype=float)
    e1 = eigvals[:, 0]
    e2 = eigvals[:, 1]
    e3 = eigvals[:, 2]
    c_bar = material[:, 0]
    sin_phi = material[:, 1]
    shear = material[:, 2]
    bulk = material[:, 3]
    lame = material[:, 4]
    sin_psi = sin_phi.copy()

    test_el = branch == BRANCH_TO_ID["elastic"]
    test_s = branch == BRANCH_TO_ID["smooth"]
    test_l = branch == BRANCH_TO_ID["left_edge"]
    test_r = branch == BRANCH_TO_ID["right_edge"]
    test_a = branch == BRANCH_TO_ID["apex"]
    if np.any(test_el):
        out[test_el] = trial[test_el]
    if np.any(test_s):
        idx = np.where(test_s)[0]
        lam = latent[idx, 0]
        sp = sin_psi[idx]
        ll = lame[idx]
        mu = shear[idx]
        i1s = i1[idx]
        out[idx, 0] = ll * i1s + 2.0 * mu * e1[idx] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        out[idx, 1] = ll * i1s + 2.0 * mu * e2[idx] - lam * (2.0 * ll * sp)
        out[idx, 2] = ll * i1s + 2.0 * mu * e3[idx] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
    if np.any(test_l):
        idx = np.where(test_l)[0]
        lam = latent[idx, 0]
        sp = sin_psi[idx]
        ll = lame[idx]
        mu = shear[idx]
        i1l = i1[idx]
        sigma12 = ll * i1l + mu * (e1[idx] + e2[idx]) - lam * (2.0 * ll * sp + mu * (1.0 + sp))
        sigma3 = ll * i1l + 2.0 * mu * e3[idx] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
        out[idx, 0] = sigma12
        out[idx, 1] = sigma12
        out[idx, 2] = sigma3
    if np.any(test_r):
        idx = np.where(test_r)[0]
        lam = latent[idx, 0]
        sp = sin_psi[idx]
        ll = lame[idx]
        mu = shear[idx]
        i1r = i1[idx]
        sigma1 = ll * i1r + 2.0 * mu * e1[idx] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        sigma23 = ll * i1r + mu * (e2[idx] + e3[idx]) - lam * (2.0 * ll * sp - mu * (1.0 - sp))
        out[idx, 0] = sigma1
        out[idx, 1] = sigma23
        out[idx, 2] = sigma23
    if np.any(test_a):
        idx = np.where(test_a)[0]
        denom = np.where(np.abs(sin_phi[idx]) > 1.0e-12, 2.0 * sin_phi[idx], np.inf)
        sigma_a = c_bar[idx] / denom
        out[idx] = np.column_stack([sigma_a, sigma_a, sigma_a])
    return out


def extract_exact_branch_latents(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
    *,
    branch_tol: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Extract the exact branchwise latent payload from the constitutive solver."""
    principal_state = _build_principal_state_3d(
        strain_eng,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    state = _build_branch_state_from_principal(principal_state)
    branch_id = _resolve_branch_id_from_state(state, branch_tol=branch_tol)
    stress, stress_principal, plastic_multiplier = _dispatch_from_branch_state(state, branch_id)
    trial_principal = exact_trial_principal_stress_3d(
        principal_state.strain,
        c_bar=principal_state.c_bar,
        sin_phi=principal_state.sin_phi,
        shear=principal_state.shear,
        bulk=principal_state.bulk,
        lame=principal_state.lame,
    )
    latent_kind, latent_dim = _exact_latent_kind_and_dim(branch_id)
    latent_values = _exact_branch_latent_values_from_state(state, branch_id)
    return {
        "branch_id": branch_id.astype(np.int64),
        "latent_kind": latent_kind,
        "latent_dim": latent_dim,
        "latent_values": latent_values,
        "sigma_principal": stress_principal.astype(float),
        "tau_principal": trial_principal.astype(float),
        "trial_principal": trial_principal.astype(float),
        "stress": stress.astype(float),
        "material_reduced": np.column_stack([state.c_bar, state.sin_phi, state.shear, state.bulk, state.lame]).astype(float),
        "f_trial": state.f_trial.astype(float),
        "plastic_multiplier": plastic_multiplier.astype(float),
        "eigvecs": state.eigvecs.astype(float),
    }


def decode_exact_branch_latents_to_principal(
    branch_id: np.ndarray | int,
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
    latent_values: np.ndarray,
) -> np.ndarray:
    """Decode exact branch latents back to ordered principal stress."""
    return _decode_exact_branch_principal_batch(branch_id, trial_principal, material_reduced, latent_values)


def audit_exact_branch_latent_roundtrip(
    strain_eng: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
    *,
    branch_tol: float = 1.0e-10,
) -> dict[str, float | int | dict[str, dict[str, float | int]]]:
    """Audit exact-latent round-trip reconstruction against the exact solver."""
    extracted = extract_exact_branch_latents(
        strain_eng,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
        branch_tol=branch_tol,
    )
    exact = constitutive_update_3d(
        strain_eng,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
        branch_tol=branch_tol,
    )

    branch_id = extracted["branch_id"].astype(np.int64)
    sigma_true = exact.stress_principal.astype(float)
    sigma_pred = np.zeros_like(sigma_true)
    branchwise: dict[str, dict[str, float | int]] = {}
    for bid, name in enumerate(BRANCH_NAMES):
        mask = branch_id == bid
        if not np.any(mask):
            branchwise[name] = {
                "count": 0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "yield_p95": 0.0,
                "yield_max": 0.0,
            }
            continue
        latent_list = extracted["latent_values"][mask]
        if len(latent_list) == 0:
            latent_matrix = np.zeros((0, _EXACT_LATENT_DIM_BY_BRANCH[bid]), dtype=float)
        elif _EXACT_LATENT_DIM_BY_BRANCH[bid] == 0:
            latent_matrix = np.zeros((mask.sum(), 0), dtype=float)
        else:
            latent_matrix = np.vstack([np.asarray(item, dtype=float).reshape(1, -1) for item in latent_list])
        sigma_pred[mask] = decode_exact_branch_latents_to_principal(
            branch_id[mask],
            extracted["trial_principal"][mask],
            extracted["material_reduced"][mask],
            latent_matrix,
        )
        yield_rel = yield_violation_rel_principal_3d(
            sigma_pred[mask],
            c_bar=extracted["material_reduced"][mask, 0],
            sin_phi=extracted["material_reduced"][mask, 1],
        )
        abs_err = np.abs(sigma_pred[mask] - sigma_true[mask])
        branchwise[name] = {
            "count": int(np.sum(mask)),
            "mean_abs": float(np.mean(abs_err)),
            "max_abs": float(np.max(abs_err)),
            "yield_p95": float(np.quantile(yield_rel, 0.95)) if yield_rel.size else 0.0,
            "yield_max": float(np.max(yield_rel)) if yield_rel.size else 0.0,
        }

    overall_abs = np.abs(sigma_pred - sigma_true)
    yield_rel = yield_violation_rel_principal_3d(
        sigma_pred,
        c_bar=extracted["material_reduced"][:, 0],
        sin_phi=extracted["material_reduced"][:, 1],
    )
    return {
        "n_rows": int(sigma_true.shape[0]),
        "mean_abs": float(np.mean(overall_abs)),
        "max_abs": float(np.max(overall_abs)),
        "yield_p95": float(np.quantile(yield_rel, 0.95)) if yield_rel.size else 0.0,
        "yield_max": float(np.max(yield_rel)) if yield_rel.size else 0.0,
        "branch_agreement": float(np.mean(exact.branch_id.astype(np.int64) == branch_id)),
        "branch_counts": {name: int(np.sum(branch_id == idx)) for idx, name in enumerate(BRANCH_NAMES)},
        "branchwise": branchwise,
    }


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
    principal_state = _build_principal_state_3d(
        strain_eng,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    state = _build_branch_state_from_principal(principal_state)
    branch_id = _resolve_branch_id_from_state(state, branch_tol=branch_tol)
    stress, stress_principal, plastic_multiplier = _dispatch_from_branch_state(state, branch_id)

    tangent = None
    if return_tangent:
        tangent = numerical_tangent_fd(
            principal_state.strain,
            c_bar=principal_state.c_bar,
            sin_phi=principal_state.sin_phi,
            shear=principal_state.shear,
            bulk=principal_state.bulk,
            lame=principal_state.lame,
            eps=tangent_eps,
        )

    return ConstitutiveResult(
        stress=stress,
        stress_principal=stress_principal,
        strain_principal=state.eigvals,
        eigvecs=state.eigvecs,
        branch_id=branch_id,
        f_trial=state.f_trial,
        plastic_multiplier=plastic_multiplier,
        tangent=tangent,
    )


def profile_constitutive_update_3d(
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
) -> tuple[ConstitutiveResult, dict[str, float | int | dict[str, int]]]:
    """Run the exact constitutive update and return phase timings and branch counters."""
    total_start = perf_counter()

    principal_start = perf_counter()
    principal_state = _build_principal_state_3d(
        strain_eng,
        c_bar=c_bar,
        sin_phi=sin_phi,
        shear=shear,
        bulk=bulk,
        lame=lame,
    )
    principal_time = perf_counter() - principal_start

    scalar_start = perf_counter()
    state = _build_branch_state_from_principal(principal_state)
    branch_id = _resolve_branch_id_from_state(state, branch_tol=branch_tol)
    branch_scalar_time = perf_counter() - scalar_start

    dispatch_start = perf_counter()
    stress, stress_principal, plastic_multiplier = _dispatch_from_branch_state(state, branch_id)
    branch_dispatch_time = perf_counter() - dispatch_start

    tangent = None
    tangent_time = 0.0
    if return_tangent:
        tangent_start = perf_counter()
        tangent = numerical_tangent_fd(
            principal_state.strain,
            c_bar=principal_state.c_bar,
            sin_phi=principal_state.sin_phi,
            shear=principal_state.shear,
            bulk=principal_state.bulk,
            lame=principal_state.lame,
            eps=tangent_eps,
        )
        tangent_time = perf_counter() - tangent_start

    total_time = perf_counter() - total_start
    counts = np.bincount(branch_id.astype(np.int64), minlength=len(BRANCH_NAMES))
    profile = {
        "n_rows": int(principal_state.strain.shape[0]),
        "principal_eig_s": float(principal_time),
        "branch_scalar_s": float(branch_scalar_time),
        "branch_dispatch_s": float(branch_dispatch_time),
        "tangent_fd_s": float(tangent_time),
        "total_s": float(total_time),
        "residual_overhead_s": float(total_time - principal_time - branch_scalar_time - branch_dispatch_time - tangent_time),
        "elastic_rows": int(counts[BRANCH_TO_ID["elastic"]]),
        "plastic_rows": int(np.sum(counts[1:])),
        "plastic_fraction": float(np.mean(branch_id > 0)),
        "branch_counts": {name: int(counts[idx]) for idx, name in enumerate(BRANCH_NAMES)},
    }
    return (
        ConstitutiveResult(
            stress=stress,
            stress_principal=stress_principal,
            strain_principal=state.eigvals,
            eigvecs=state.eigvecs,
            branch_id=branch_id,
            f_trial=state.f_trial,
            plastic_multiplier=plastic_multiplier,
            tangent=tangent,
        ),
        profile,
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
