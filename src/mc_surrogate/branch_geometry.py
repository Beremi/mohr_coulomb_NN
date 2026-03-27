"""Analytic branch-geometry helpers in ordered principal-strain coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SurfaceName = Literal["yield", "smooth_left", "smooth_right", "left_apex", "right_apex"]
GLOBAL_DISTANCE_TERM_NAMES = (
    "yield",
    "smooth_left",
    "smooth_right",
    "left_apex",
    "right_apex",
    "gap12_norm",
    "gap23_norm",
)
BRANCH_DISTANCE_NAMES = ("smooth", "left_edge", "right_edge", "apex")
BRANCH_DISTANCE_TERM_NAMES: dict[str, tuple[str, ...]] = {
    "smooth": ("yield", "smooth_left", "smooth_right", "gap12_norm", "gap23_norm"),
    "left_edge": ("smooth_left", "left_apex", "gap12_norm"),
    "right_edge": ("smooth_right", "right_apex", "gap23_norm"),
    "apex": ("left_apex", "right_apex", "gap12_norm", "gap23_norm"),
}


@dataclass
class BranchGeometry:
    principal_strain: np.ndarray
    i1: np.ndarray
    delta12: np.ndarray
    delta23: np.ndarray
    gap12_norm: np.ndarray
    gap23_norm: np.ndarray
    gap_min_norm: np.ndarray
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
    d_geom: np.ndarray
    d_smooth: np.ndarray
    d_left: np.ndarray
    d_right: np.ndarray
    d_apex: np.ndarray


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


def normalized_gap_metrics_principal(
    principal_strain: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized ordered-principal gap metrics."""
    eps_p = np.asarray(principal_strain, dtype=float)
    if eps_p.ndim == 1:
        eps_p = eps_p[None, :]
    if eps_p.ndim != 2 or eps_p.shape[1] != 3:
        raise ValueError(f"Expected shape (n,3), got {eps_p.shape}.")

    i1 = np.sum(eps_p, axis=1)
    delta12 = eps_p[:, 0] - eps_p[:, 1]
    delta23 = eps_p[:, 1] - eps_p[:, 2]
    denom = np.abs(i1) + np.abs(delta12) + np.abs(delta23) + eps
    gap12_norm = np.abs(delta12) / denom
    gap23_norm = np.abs(delta23) / denom
    gap_min_norm = np.minimum(gap12_norm, gap23_norm)
    return gap12_norm, gap23_norm, gap_min_norm


def geometry_safety_score(
    *,
    m_yield: np.ndarray,
    m_smooth_left: np.ndarray,
    m_smooth_right: np.ndarray,
    m_left_apex: np.ndarray,
    m_right_apex: np.ndarray,
    gap12_norm: np.ndarray,
    gap23_norm: np.ndarray,
    ) -> np.ndarray:
    """Return the analytic safety margin used for hybrid fallback decisions."""
    return np.minimum.reduce(
        [
            np.abs(np.asarray(m_yield, dtype=float)),
            np.abs(np.asarray(m_smooth_left, dtype=float)),
            np.abs(np.asarray(m_smooth_right, dtype=float)),
            np.abs(np.asarray(m_left_apex, dtype=float)),
            np.abs(np.asarray(m_right_apex, dtype=float)),
            np.asarray(gap12_norm, dtype=float),
            np.asarray(gap23_norm, dtype=float),
        ]
    )


def branch_conditioned_safety_scores(
    *,
    m_yield: np.ndarray,
    m_smooth_left: np.ndarray,
    m_smooth_right: np.ndarray,
    m_left_apex: np.ndarray,
    m_right_apex: np.ndarray,
    gap12_norm: np.ndarray,
    gap23_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return branch-conditioned geometry distances for plastic regions."""
    return {
        "smooth": np.minimum.reduce(
            [
                np.abs(np.asarray(m_yield, dtype=float)),
                np.abs(np.asarray(m_smooth_left, dtype=float)),
                np.abs(np.asarray(m_smooth_right, dtype=float)),
                np.asarray(gap12_norm, dtype=float),
                np.asarray(gap23_norm, dtype=float),
            ]
        ),
        "left_edge": np.minimum.reduce(
            [
                np.abs(np.asarray(m_smooth_left, dtype=float)),
                np.abs(np.asarray(m_left_apex, dtype=float)),
                np.asarray(gap12_norm, dtype=float),
            ]
        ),
        "right_edge": np.minimum.reduce(
            [
                np.abs(np.asarray(m_smooth_right, dtype=float)),
                np.abs(np.asarray(m_right_apex, dtype=float)),
                np.asarray(gap23_norm, dtype=float),
            ]
        ),
        "apex": np.minimum.reduce(
            [
                np.abs(np.asarray(m_left_apex, dtype=float)),
                np.abs(np.asarray(m_right_apex, dtype=float)),
                np.asarray(gap12_norm, dtype=float),
                np.asarray(gap23_norm, dtype=float),
            ]
        ),
    }


def _select_margin_terms(
    names: tuple[str, ...],
    values: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([np.asarray(v, dtype=float) for v in values], axis=1)
    term_idx = np.argmin(stacked, axis=1)
    distance = stacked[np.arange(stacked.shape[0]), term_idx]
    term_names = np.asarray(names, dtype=object)[term_idx]
    return distance.astype(np.float32), term_names.astype(object)


def global_min_term_names(geom: BranchGeometry) -> np.ndarray:
    """Return the winning global `d_geom` term name for every row."""
    _, term_names = _select_margin_terms(
        GLOBAL_DISTANCE_TERM_NAMES,
        (
            np.abs(geom.m_yield),
            np.abs(geom.m_smooth_left),
            np.abs(geom.m_smooth_right),
            np.abs(geom.m_left_apex),
            np.abs(geom.m_right_apex),
            geom.gap12_norm,
            geom.gap23_norm,
        ),
    )
    return term_names


def branch_min_term_names(
    geom: BranchGeometry,
    branch_name: str,
) -> np.ndarray:
    """Return the winning term name for one branch-conditioned distance."""
    if branch_name == "smooth":
        _, term_names = _select_margin_terms(
            BRANCH_DISTANCE_TERM_NAMES["smooth"],
            (
                np.abs(geom.m_yield),
                np.abs(geom.m_smooth_left),
                np.abs(geom.m_smooth_right),
                geom.gap12_norm,
                geom.gap23_norm,
            ),
        )
        return term_names
    if branch_name == "left_edge":
        _, term_names = _select_margin_terms(
            BRANCH_DISTANCE_TERM_NAMES["left_edge"],
            (
                np.abs(geom.m_smooth_left),
                np.abs(geom.m_left_apex),
                geom.gap12_norm,
            ),
        )
        return term_names
    if branch_name == "right_edge":
        _, term_names = _select_margin_terms(
            BRANCH_DISTANCE_TERM_NAMES["right_edge"],
            (
                np.abs(geom.m_smooth_right),
                np.abs(geom.m_right_apex),
                geom.gap23_norm,
            ),
        )
        return term_names
    if branch_name == "apex":
        _, term_names = _select_margin_terms(
            BRANCH_DISTANCE_TERM_NAMES["apex"],
            (
                np.abs(geom.m_left_apex),
                np.abs(geom.m_right_apex),
                geom.gap12_norm,
                geom.gap23_norm,
            ),
        )
        return term_names
    raise ValueError(f"Unsupported branch_name {branch_name!r}.")


def select_branch_conditioned_distance(
    geom: BranchGeometry,
    branch_id: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select branch-conditioned distances and min-term names by branch id.

    Accepted branch ids are the plastic Mohr-Coulomb ids:
    `1=smooth, 2=left_edge, 3=right_edge, 4=apex`.
    """
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    n = geom.d_geom.shape[0]
    if branch.size == 1 and n > 1:
        branch = np.full(n, int(branch.item()), dtype=np.int64)
    if branch.shape[0] != n:
        raise ValueError(f"branch_id shape {branch.shape} does not match geometry batch {n}.")

    distance = np.full(n, np.nan, dtype=np.float32)
    branch_names = np.full(n, "", dtype=object)
    term_names = np.full(n, "", dtype=object)

    smooth_terms = branch_min_term_names(geom, "smooth")
    left_terms = branch_min_term_names(geom, "left_edge")
    right_terms = branch_min_term_names(geom, "right_edge")
    apex_terms = branch_min_term_names(geom, "apex")

    masks = {
        1: ("smooth", geom.d_smooth.astype(np.float32), smooth_terms),
        2: ("left_edge", geom.d_left.astype(np.float32), left_terms),
        3: ("right_edge", geom.d_right.astype(np.float32), right_terms),
        4: ("apex", geom.d_apex.astype(np.float32), apex_terms),
    }
    for bid, (name, values, terms) in masks.items():
        mask = branch == bid
        if np.any(mask):
            distance[mask] = values[mask]
            branch_names[mask] = name
            term_names[mask] = terms[mask]
    return distance, branch_names, term_names


def soft_admissible_atlas_route_targets(
    branch_id: np.ndarray,
    geom: BranchGeometry,
    *,
    margin_scale: float = 0.10,
    base_weight: float = 1.0,
) -> np.ndarray:
    """
    Return 4-way soft targets for the packet4 plastic atlas route head.

    Target order is `(smooth, left_edge, right_edge, apex)`. Each true branch
    only mixes with its adjacent branches, and neighbor mass decays with the
    magnitude of the corresponding exact geometry margin.
    """
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    n = branch.shape[0]
    target = np.zeros((n, len(BRANCH_DISTANCE_NAMES)), dtype=np.float32)
    scale = max(float(margin_scale), 1.0e-12)

    def _adjacent_score(margin: np.ndarray) -> np.ndarray:
        return np.exp(-np.abs(np.asarray(margin, dtype=float)) / scale).astype(np.float32)

    smooth = branch == 1
    left = branch == 2
    right = branch == 3
    apex = branch == 4

    if np.any(smooth):
        target[smooth, 0] = float(base_weight)
        target[smooth, 1] = _adjacent_score(geom.m_smooth_left[smooth])
        target[smooth, 2] = _adjacent_score(geom.m_smooth_right[smooth])
    if np.any(left):
        target[left, 0] = _adjacent_score(geom.m_smooth_left[left])
        target[left, 1] = float(base_weight)
        target[left, 3] = _adjacent_score(geom.m_left_apex[left])
    if np.any(right):
        target[right, 0] = _adjacent_score(geom.m_smooth_right[right])
        target[right, 2] = float(base_weight)
        target[right, 3] = _adjacent_score(geom.m_right_apex[right])
    if np.any(apex):
        target[apex, 1] = _adjacent_score(geom.m_left_apex[apex])
        target[apex, 2] = _adjacent_score(geom.m_right_apex[apex])
        target[apex, 3] = float(base_weight)

    row_sum = np.sum(target, axis=1, keepdims=True)
    valid = row_sum[:, 0] > 0.0
    if np.any(valid):
        target[valid] = target[valid] / row_sum[valid]
    return target.astype(np.float32)


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
    gap12_norm, gap23_norm, gap_min_norm = normalized_gap_metrics_principal(eps)

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
    branch_scores = branch_conditioned_safety_scores(
        m_yield=m_yield,
        m_smooth_left=m_smooth_left,
        m_smooth_right=m_smooth_right,
        m_left_apex=m_left_apex,
        m_right_apex=m_right_apex,
        gap12_norm=gap12_norm,
        gap23_norm=gap23_norm,
    )
    d_geom = geometry_safety_score(
        m_yield=m_yield,
        m_smooth_left=m_smooth_left,
        m_smooth_right=m_smooth_right,
        m_left_apex=m_left_apex,
        m_right_apex=m_right_apex,
        gap12_norm=gap12_norm,
        gap23_norm=gap23_norm,
    )

    return BranchGeometry(
        principal_strain=eps,
        i1=i1,
        delta12=d12,
        delta23=d23,
        gap12_norm=gap12_norm,
        gap23_norm=gap23_norm,
        gap_min_norm=gap_min_norm,
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
        d_geom=d_geom,
        d_smooth=branch_scores["smooth"].astype(np.float32),
        d_left=branch_scores["left_edge"].astype(np.float32),
        d_right=branch_scores["right_edge"].astype(np.float32),
        d_apex=branch_scores["apex"].astype(np.float32),
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
