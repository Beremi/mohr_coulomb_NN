"""Analytic branch-geometry helpers in ordered principal-strain coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SurfaceName = Literal["yield", "smooth_left", "smooth_right", "left_apex", "right_apex"]


@dataclass
class BranchGeometry:
    principal_strain: np.ndarray
    i1: np.ndarray
    delta12: np.ndarray
    delta23: np.ndarray
    f_trial: np.ndarray
    gamma_sl: np.ndarray
    gamma_sr: np.ndarray
    gamma_la: np.ndarray
    gamma_ra: np.ndarray
    lambda_s: np.ndarray
    lambda_l: np.ndarray
    lambda_r: np.ndarray
    lambda_a: np.ndarray
    m_yield: np.ndarray
    m_smooth_left: np.ndarray
    m_smooth_right: np.ndarray
    m_left_apex: np.ndarray
    m_right_apex: np.ndarray
    m_left_vs_right: np.ndarray


def principal_from_gap_coords(i1: np.ndarray, delta12: np.ndarray, delta23: np.ndarray) -> np.ndarray:
    i1 = np.asarray(i1, dtype=float)
    d12 = np.asarray(delta12, dtype=float)
    d23 = np.asarray(delta23, dtype=float)
    e1 = (i1 + 2.0 * d12 + d23) / 3.0
    e2 = (i1 - d12 + d23) / 3.0
    e3 = (i1 - d12 - 2.0 * d23) / 3.0
    return np.stack([e1, e2, e3], axis=-1)


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    den = np.asarray(den, dtype=float)
    den = np.where(np.abs(den) > eps, den, np.sign(den) * eps + (den == 0.0) * eps)
    return np.asarray(num, dtype=float) / den


def _normalized_signed_gap(lhs: np.ndarray, rhs: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    scale = np.abs(lhs) + np.abs(rhs) + eps
    return (lhs - rhs) / scale


def compute_branch_geometry_principal(
    principal_strain: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
) -> BranchGeometry:
    eps = np.asarray(principal_strain, dtype=float)
    if eps.ndim == 1:
        eps = eps[None, :]
    if eps.ndim != 2 or eps.shape[1] != 3:
        raise ValueError(f"Expected shape (n,3), got {eps.shape}.")

    e1, e2, e3 = eps[:, 0], eps[:, 1], eps[:, 2]
    i1 = e1 + e2 + e3
    d12 = e1 - e2
    d23 = e2 - e3

    c_bar, sin_phi, shear, bulk, lame = np.broadcast_arrays(
        np.asarray(c_bar, dtype=float),
        np.asarray(sin_phi, dtype=float),
        np.asarray(shear, dtype=float),
        np.asarray(bulk, dtype=float),
        np.asarray(lame, dtype=float),
    )
    c_bar = c_bar.reshape(-1)
    sin_phi = sin_phi.reshape(-1)
    shear = shear.reshape(-1)
    bulk = bulk.reshape(-1)
    lame = lame.reshape(-1)
    n = eps.shape[0]
    if c_bar.size == 1:
        c_bar = np.full(n, c_bar.item())
    if sin_phi.size == 1:
        sin_phi = np.full(n, sin_phi.item())
    if shear.size == 1:
        shear = np.full(n, shear.item())
    if bulk.size == 1:
        bulk = np.full(n, bulk.item())
    if lame.size == 1:
        lame = np.full(n, lame.item())

    sp = sin_phi
    f_trial = 2.0 * shear * ((1.0 + sp) * e1 - (1.0 - sp) * e3) + 2.0 * lame * sp * i1 - c_bar

    gamma_sl = _safe_div(d12, 1.0 + sp)
    gamma_sr = _safe_div(d23, 1.0 - sp)
    gamma_la = _safe_div(d12 + 2.0 * d23, 3.0 - sp)
    gamma_ra = _safe_div(2.0 * d12 + d23, 3.0 + sp)

    denom_s = 4.0 * lame * sp * sp + 4.0 * shear * (1.0 + sp * sp)
    denom_l = 4.0 * lame * sp * sp + shear * (1.0 + sp) * (1.0 + sp) + 2.0 * shear * (1.0 - sp) * (1.0 - sp)
    denom_r = 4.0 * lame * sp * sp + 2.0 * shear * (1.0 + sp) * (1.0 + sp) + shear * (1.0 - sp) * (1.0 - sp)
    denom_a = 4.0 * bulk * sp * sp

    lambda_s = _safe_div(f_trial, denom_s)
    lambda_l = _safe_div(
        shear * ((1.0 + sp) * (e1 + e2) - 2.0 * (1.0 - sp) * e3) + 2.0 * lame * sp * i1 - c_bar,
        denom_l,
    )
    lambda_r = _safe_div(
        shear * (2.0 * (1.0 + sp) * e1 - (1.0 - sp) * (e2 + e3)) + 2.0 * lame * sp * i1 - c_bar,
        denom_r,
    )
    lambda_a = _safe_div(2.0 * bulk * sp * i1 - c_bar, denom_a)

    m_yield = _safe_div(
        f_trial,
        np.abs(2.0 * shear * ((1.0 + sp) * e1 - (1.0 - sp) * e3)) + np.abs(2.0 * lame * sp * i1) + np.abs(c_bar) + 1.0e-12,
    )
    m_smooth_left = _normalized_signed_gap(gamma_sl, lambda_s)
    m_smooth_right = _normalized_signed_gap(gamma_sr, lambda_s)
    m_left_apex = _normalized_signed_gap(gamma_la, lambda_l)
    m_right_apex = _normalized_signed_gap(gamma_ra, lambda_r)
    m_left_vs_right = _normalized_signed_gap(gamma_sl, gamma_sr)

    return BranchGeometry(
        principal_strain=eps,
        i1=i1,
        delta12=d12,
        delta23=d23,
        f_trial=f_trial,
        gamma_sl=gamma_sl,
        gamma_sr=gamma_sr,
        gamma_la=gamma_la,
        gamma_ra=gamma_ra,
        lambda_s=lambda_s,
        lambda_l=lambda_l,
        lambda_r=lambda_r,
        lambda_a=lambda_a,
        m_yield=m_yield,
        m_smooth_left=m_smooth_left,
        m_smooth_right=m_smooth_right,
        m_left_apex=m_left_apex,
        m_right_apex=m_right_apex,
        m_left_vs_right=m_left_vs_right,
    )


def solve_alpha_for_surface(
    principal_direction: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    shear: np.ndarray | float,
    bulk: np.ndarray | float,
    lame: np.ndarray | float,
    surface: SurfaceName,
) -> np.ndarray:
    d = np.asarray(principal_direction, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"Expected shape (n,3), got {d.shape}.")

    d1, d2, d3 = d[:, 0], d[:, 1], d[:, 2]
    i1d = d1 + d2 + d3

    c_bar, sin_phi, shear, bulk, lame = np.broadcast_arrays(
        np.asarray(c_bar, dtype=float),
        np.asarray(sin_phi, dtype=float),
        np.asarray(shear, dtype=float),
        np.asarray(bulk, dtype=float),
        np.asarray(lame, dtype=float),
    )
    c_bar = c_bar.reshape(-1)
    sin_phi = sin_phi.reshape(-1)
    shear = shear.reshape(-1)
    bulk = bulk.reshape(-1)
    lame = lame.reshape(-1)
    n = d.shape[0]
    if c_bar.size == 1:
        c_bar = np.full(n, c_bar.item())
    if sin_phi.size == 1:
        sin_phi = np.full(n, sin_phi.item())
    if shear.size == 1:
        shear = np.full(n, shear.item())
    if bulk.size == 1:
        bulk = np.full(n, bulk.item())
    if lame.size == 1:
        lame = np.full(n, lame.item())

    sp = sin_phi
    a = 2.0 * shear * ((1.0 + sp) * d1 - (1.0 - sp) * d3) + 2.0 * lame * sp * i1d
    ds = 4.0 * lame * sp * sp + 4.0 * shear * (1.0 + sp * sp)
    dl = 4.0 * lame * sp * sp + shear * (1.0 + sp) * (1.0 + sp) + 2.0 * shear * (1.0 - sp) * (1.0 - sp)
    dr = 4.0 * lame * sp * sp + 2.0 * shear * (1.0 + sp) * (1.0 + sp) + shear * (1.0 - sp) * (1.0 - sp)

    if surface == "yield":
        denom = a
    elif surface == "smooth_left":
        b = _safe_div(d1 - d2, 1.0 + sp)
        denom = a - ds * b
    elif surface == "smooth_right":
        b = _safe_div(d2 - d3, 1.0 - sp)
        denom = a - ds * b
    elif surface == "left_apex":
        c = shear * ((1.0 + sp) * (d1 + d2) - 2.0 * (1.0 - sp) * d3) + 2.0 * lame * sp * i1d
        b = _safe_div(d1 + d2 - 2.0 * d3, 3.0 - sp)
        denom = c - dl * b
    elif surface == "right_apex":
        c = shear * (2.0 * (1.0 + sp) * d1 - (1.0 - sp) * (d2 + d3)) + 2.0 * lame * sp * i1d
        b = _safe_div(2.0 * d1 - d2 - d3, 3.0 + sp)
        denom = c - dr * b
    else:
        raise ValueError(f"Unsupported surface {surface!r}")

    denom = np.where(np.abs(denom) > 1.0e-12, denom, np.nan)
    return c_bar / denom
