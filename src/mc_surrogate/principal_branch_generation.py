"""Principal-space synthetic generators for pointwise branch-classification experiments."""

from __future__ import annotations

import math

import numpy as np

from .branch_geometry import compute_branch_geometry_principal, principal_from_gap_coords, solve_alpha_for_surface
from .mohr_coulomb import constitutive_update_3d
from .voigt import (
    principal_values_and_vectors_from_strain,
    reconstruct_from_principal,
    tensor_to_strain_voigt,
)


def fit_principal_hybrid_bank(
    strain_fit: np.ndarray,
    branch_fit: np.ndarray,
    material_fit: np.ndarray,
) -> dict[str, np.ndarray | str]:
    """Build a pointwise principal-space bank from real cover-layer constitutive states."""
    if strain_fit.ndim == 3:
        strain_flat = strain_fit.reshape(-1, 6).astype(np.float32)
    else:
        strain_flat = np.asarray(strain_fit, dtype=np.float32).reshape(-1, 6)
    if branch_fit.ndim > 1:
        branch_flat = branch_fit.reshape(-1).astype(np.int64)
    else:
        branch_flat = np.asarray(branch_fit, dtype=np.int64).reshape(-1)
    if material_fit.ndim == 2 and material_fit.shape[0] * 11 == strain_flat.shape[0]:
        material_flat = np.repeat(material_fit.astype(np.float32), 11, axis=0)
    else:
        material_flat = np.asarray(material_fit, dtype=np.float32).reshape(-1, 5)

    principal, eigvecs = principal_values_and_vectors_from_strain(strain_flat)
    geom = compute_branch_geometry_principal(
        principal,
        c_bar=material_flat[:, 0],
        sin_phi=material_flat[:, 1],
        shear=material_flat[:, 2],
        bulk=material_flat[:, 3],
        lame=material_flat[:, 4],
    )
    c_bar = np.maximum(material_flat[:, 0], 1.0e-6)
    shear = np.maximum(material_flat[:, 2], 1.0e-6)
    bulk = np.maximum(material_flat[:, 3], 1.0e-6)

    z_i1 = bulk * geom.i1 / c_bar
    z_d12 = np.log(np.maximum(shear * geom.delta12 / c_bar, 1.0e-8))
    z_d23 = np.log(np.maximum(shear * geom.delta23 / c_bar, 1.0e-8))
    constitutive_coords = np.column_stack([z_i1, z_d12, z_d23]).astype(np.float32)
    coord_std = np.maximum(np.std(constitutive_coords, axis=0), np.array([1.0e-5, 1.0e-4, 1.0e-4], dtype=np.float32))

    branch_counts = np.bincount(branch_flat, minlength=5).astype(np.float64)
    inv_branch = 1.0 / np.maximum(branch_counts, 1.0)
    branch_balanced = inv_branch[branch_flat]
    branch_balanced /= np.sum(branch_balanced)

    sr_closeness = np.exp(-np.abs(geom.m_smooth_right) / 0.05).astype(np.float64)
    sl_closeness = np.exp(-np.abs(geom.m_smooth_left) / 0.05).astype(np.float64)
    la_closeness = np.exp(-np.abs(geom.m_left_apex) / 0.05).astype(np.float64)
    ra_closeness = np.exp(-np.abs(geom.m_right_apex) / 0.05).astype(np.float64)
    smooth_flag = (branch_flat == 1).astype(np.float64)
    right_flag = (branch_flat == 3).astype(np.float64)
    left_flag = (branch_flat == 2).astype(np.float64)
    apex_flag = (branch_flat == 4).astype(np.float64)
    strain_norm = np.linalg.norm(strain_flat, axis=1)
    strain_norm_scaled = strain_norm / max(float(np.mean(strain_norm)), 1.0e-8)

    smooth_focus = 1.0 + 2.0 * smooth_flag + 1.5 * sr_closeness + 0.5 * strain_norm_scaled
    smooth_focus /= np.sum(smooth_focus)
    smooth_edge = 1.0 + 4.0 * sr_closeness + 2.0 * smooth_flag + 1.5 * right_flag + 0.5 * strain_norm_scaled
    smooth_edge /= np.sum(smooth_edge)
    left_focus = 1.0 + 2.5 * left_flag + 0.5 * strain_norm_scaled
    left_focus /= np.sum(left_focus)
    yield_closeness = np.exp(-np.abs(geom.m_yield) / 0.05).astype(np.float64)
    gap_ratio = np.minimum(geom.delta12, geom.delta23) / (
        np.abs(principal[:, 0]) + np.abs(principal[:, 1]) + np.abs(principal[:, 2]) + 1.0e-12
    )
    small_gap_closeness = np.exp(-gap_ratio / 0.03).astype(np.float64)
    boundary_closeness = 0.25 * (sr_closeness + sl_closeness + la_closeness + ra_closeness)
    elastic_scale = np.where(branch_flat == 0, 0.30 + 1.70 * yield_closeness, 1.0)
    margin_bulk = (0.25 + 1.75 * yield_closeness + 1.00 * boundary_closeness + 0.40 * strain_norm_scaled) * elastic_scale
    margin_bulk /= np.sum(margin_bulk)
    yield_tube = (1.0 + 6.0 * yield_closeness + 0.50 * strain_norm_scaled) * np.where(branch_flat == 0, 1.20, 1.0)
    yield_tube /= np.sum(yield_tube)
    small_gap = 1.0 + 8.0 * small_gap_closeness + 0.50 * yield_closeness + 0.25 * strain_norm_scaled
    small_gap /= np.sum(small_gap)

    tail_threshold = np.quantile(strain_norm, 0.80)
    tail_weights = 1.0 + 4.0 * (strain_norm >= tail_threshold).astype(np.float64) + 0.75 * strain_norm_scaled
    tail_weights /= np.sum(tail_weights)

    boundary_sr = 1.0 + 6.0 * sr_closeness + 1.5 * smooth_flag + 1.5 * right_flag
    boundary_sr /= np.sum(boundary_sr)
    boundary_sl = 1.0 + 6.0 * sl_closeness + 1.5 * smooth_flag + 1.5 * left_flag
    boundary_sl /= np.sum(boundary_sl)
    edge_apex_right = 1.0 + 5.0 * ra_closeness + 1.5 * right_flag + 1.5 * apex_flag + 0.5 * strain_norm_scaled
    edge_apex_right /= np.sum(edge_apex_right)
    edge_apex_left = 1.0 + 5.0 * la_closeness + 1.5 * left_flag + 1.5 * apex_flag + 0.5 * strain_norm_scaled
    edge_apex_left /= np.sum(edge_apex_left)

    return {
        "generator_kind": "principal_hybrid",
        "strain_fit": strain_flat.astype(np.float32),
        "principal_fit": principal.astype(np.float32),
        "eigvecs_fit": eigvecs.astype(np.float32),
        "branch_fit": branch_flat.astype(np.int64),
        "material_fit": material_flat.astype(np.float32),
        "constitutive_coords": constitutive_coords.astype(np.float32),
        "coord_std": coord_std.astype(np.float32),
        "weights_uniform": np.full(strain_flat.shape[0], 1.0 / strain_flat.shape[0], dtype=np.float32),
        "weights_branch_balanced": branch_balanced.astype(np.float32),
        "weights_smooth_focus": smooth_focus.astype(np.float32),
        "weights_smooth_edge": smooth_edge.astype(np.float32),
        "weights_left_focus": left_focus.astype(np.float32),
        "weights_margin_bulk": margin_bulk.astype(np.float32),
        "weights_yield_tube": yield_tube.astype(np.float32),
        "weights_small_gap": small_gap.astype(np.float32),
        "weights_tail": tail_weights.astype(np.float32),
        "weights_boundary_sr": boundary_sr.astype(np.float32),
        "weights_boundary_sl": boundary_sl.astype(np.float32),
        "weights_edge_apex_right": edge_apex_right.astype(np.float32),
        "weights_edge_apex_left": edge_apex_left.astype(np.float32),
    }


def _pick_weights(bank: dict[str, np.ndarray | str], selection: str) -> np.ndarray:
    if selection == "uniform":
        return np.asarray(bank["weights_uniform"], dtype=np.float64)
    if selection == "branch_balanced":
        return np.asarray(bank["weights_branch_balanced"], dtype=np.float64)
    if selection == "smooth_focus":
        return np.asarray(bank["weights_smooth_focus"], dtype=np.float64)
    if selection == "smooth_edge":
        return np.asarray(bank["weights_smooth_edge"], dtype=np.float64)
    if selection == "left_focus":
        return np.asarray(bank["weights_left_focus"], dtype=np.float64)
    if selection == "margin_bulk":
        return np.asarray(bank["weights_margin_bulk"], dtype=np.float64)
    if selection == "yield_tube":
        return np.asarray(bank["weights_yield_tube"], dtype=np.float64)
    if selection == "small_gap":
        return np.asarray(bank["weights_small_gap"], dtype=np.float64)
    if selection == "tail":
        return np.asarray(bank["weights_tail"], dtype=np.float64)
    if selection == "boundary_smooth_right":
        return np.asarray(bank["weights_boundary_sr"], dtype=np.float64)
    if selection == "boundary_smooth_left":
        return np.asarray(bank["weights_boundary_sl"], dtype=np.float64)
    if selection == "edge_apex_right":
        return np.asarray(bank["weights_edge_apex_right"], dtype=np.float64)
    if selection == "edge_apex_left":
        return np.asarray(bank["weights_edge_apex_left"], dtype=np.float64)
    raise ValueError(f"Unknown principal-hybrid selection {selection!r}.")


def _reconstruct_strain(
    principal: np.ndarray,
    eigvecs: np.ndarray,
) -> np.ndarray:
    strain_tensor = reconstruct_from_principal(principal, eigvecs)
    return tensor_to_strain_voigt(strain_tensor).astype(np.float32)


def _evaluate_exact(
    strain: np.ndarray,
    material: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    exact = constitutive_update_3d(
        strain,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    return strain.astype(np.float32), exact.branch_id.astype(np.int64)


def _real_like_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, selection)
    idx = rng.choice(weights.shape[0], size=sample_count, replace=True, p=weights)
    z0 = np.asarray(bank["constitutive_coords"], dtype=np.float32)[idx]
    eigvecs = np.asarray(bank["eigvecs_fit"], dtype=np.float32)[idx]
    material = np.asarray(bank["material_fit"], dtype=np.float32)[idx]
    coord_std = np.asarray(bank["coord_std"], dtype=np.float32)
    local_scale = np.maximum(0.15 * coord_std[None, :], 0.20 * np.abs(z0))
    noise = rng.normal(size=z0.shape).astype(np.float32) * local_scale * float(noise_scale)
    z = z0 + noise

    c_bar = np.maximum(material[:, 0], 1.0e-6)
    shear = np.maximum(material[:, 2], 1.0e-6)
    bulk = np.maximum(material[:, 3], 1.0e-6)
    i1 = z[:, 0] * c_bar / bulk
    d12 = np.exp(z[:, 1]) * c_bar / shear
    d23 = np.exp(z[:, 2]) * c_bar / shear
    principal = principal_from_gap_coords(i1, d12, d23).astype(np.float32)
    strain = _reconstruct_strain(principal, eigvecs)
    strain, branch = _evaluate_exact(strain, material)
    return strain, branch, material.astype(np.float32)


def _boundary_smooth_right_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    tube_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, "boundary_smooth_right")
    principal_fit = np.asarray(bank["principal_fit"], dtype=np.float32)
    eigvecs_fit = np.asarray(bank["eigvecs_fit"], dtype=np.float32)
    material_fit = np.asarray(bank["material_fit"], dtype=np.float32)

    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    total = 0
    attempts = 0
    while total < sample_count:
        attempts += 1
        need = sample_count - total
        draw = max(need * 2, 256)
        idx = rng.choice(weights.shape[0], size=draw, replace=True, p=weights)
        principal_seed = principal_fit[idx]
        eigvecs = eigvecs_fit[idx]
        material = material_fit[idx]
        scale = np.maximum(np.linalg.norm(principal_seed, axis=1, keepdims=True), 1.0e-8)
        direction = principal_seed / scale
        alpha = solve_alpha_for_surface(
            direction,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
            shear=material[:, 2],
            bulk=material[:, 3],
            lame=material[:, 4],
            surface="smooth_right",
        )
        jitter = rng.uniform(-tube_width, tube_width, size=alpha.shape)
        alpha_jitter = alpha * (1.0 + jitter)
        valid = np.isfinite(alpha_jitter) & (alpha_jitter > 0.0)
        if not np.any(valid):
            if attempts >= 20:
                break
            continue
        principal = (alpha_jitter[valid, None] * direction[valid]).astype(np.float32)
        strain = _reconstruct_strain(principal, eigvecs[valid])
        strain, branch = _evaluate_exact(strain, material[valid])
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material[valid].astype(np.float32))
        total += strain.shape[0]
        if attempts >= 20:
            break
    if total == 0:
        return _real_like_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed + 1701,
            noise_scale=max(0.04, tube_width),
            selection="boundary_smooth_right",
        )
    strain_full = np.concatenate(strain_parts, axis=0)[:sample_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:sample_count]
    material_full = np.concatenate(material_parts, axis=0)[:sample_count]
    return strain_full, branch_full, material_full


def _yield_tube_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    tube_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, "yield_tube")
    principal_fit = np.asarray(bank["principal_fit"], dtype=np.float32)
    eigvecs_fit = np.asarray(bank["eigvecs_fit"], dtype=np.float32)
    material_fit = np.asarray(bank["material_fit"], dtype=np.float32)

    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    total = 0
    attempts = 0
    while total < sample_count:
        attempts += 1
        need = sample_count - total
        draw = max(need * 2, 256)
        idx = rng.choice(weights.shape[0], size=draw, replace=True, p=weights)
        principal_seed = principal_fit[idx]
        eigvecs = eigvecs_fit[idx]
        material = material_fit[idx]
        scale = np.maximum(np.linalg.norm(principal_seed, axis=1, keepdims=True), 1.0e-8)
        direction = principal_seed / scale
        alpha = solve_alpha_for_surface(
            direction,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
            shear=material[:, 2],
            bulk=material[:, 3],
            lame=material[:, 4],
            surface="yield",
        )
        jitter = rng.uniform(-tube_width, tube_width, size=alpha.shape)
        alpha_jitter = alpha * (1.0 + jitter)
        valid = np.isfinite(alpha_jitter) & (alpha_jitter > 0.0)
        if not np.any(valid):
            if attempts >= 20:
                break
            continue
        principal = (alpha_jitter[valid, None] * direction[valid]).astype(np.float32)
        strain = _reconstruct_strain(principal, eigvecs[valid])
        strain, branch = _evaluate_exact(strain, material[valid])
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material[valid].astype(np.float32))
        total += strain.shape[0]
        if attempts >= 20:
            break
    if total == 0:
        return _real_like_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed + 1702,
            noise_scale=max(0.03, tube_width),
            selection="yield_tube",
        )
    strain_full = np.concatenate(strain_parts, axis=0)[:sample_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:sample_count]
    material_full = np.concatenate(material_parts, axis=0)[:sample_count]
    return strain_full, branch_full, material_full


def _boundary_smooth_left_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    tube_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, "boundary_smooth_left")
    principal_fit = np.asarray(bank["principal_fit"], dtype=np.float32)
    eigvecs_fit = np.asarray(bank["eigvecs_fit"], dtype=np.float32)
    material_fit = np.asarray(bank["material_fit"], dtype=np.float32)

    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    total = 0
    attempts = 0
    while total < sample_count:
        attempts += 1
        need = sample_count - total
        draw = max(need * 2, 256)
        idx = rng.choice(weights.shape[0], size=draw, replace=True, p=weights)
        principal_seed = principal_fit[idx]
        eigvecs = eigvecs_fit[idx]
        material = material_fit[idx]
        scale = np.maximum(np.linalg.norm(principal_seed, axis=1, keepdims=True), 1.0e-8)
        direction = principal_seed / scale
        alpha = solve_alpha_for_surface(
            direction,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
            shear=material[:, 2],
            bulk=material[:, 3],
            lame=material[:, 4],
            surface="smooth_left",
        )
        jitter = rng.uniform(-tube_width, tube_width, size=alpha.shape)
        alpha_jitter = alpha * (1.0 + jitter)
        valid = np.isfinite(alpha_jitter) & (alpha_jitter > 0.0)
        if not np.any(valid):
            if attempts >= 20:
                break
            continue
        principal = (alpha_jitter[valid, None] * direction[valid]).astype(np.float32)
        strain = _reconstruct_strain(principal, eigvecs[valid])
        strain, branch = _evaluate_exact(strain, material[valid])
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material[valid].astype(np.float32))
        total += strain.shape[0]
        if attempts >= 20:
            break
    if total == 0:
        return _real_like_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed + 1703,
            noise_scale=max(0.04, tube_width),
            selection="boundary_smooth_left",
        )
    strain_full = np.concatenate(strain_parts, axis=0)[:sample_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:sample_count]
    material_full = np.concatenate(material_parts, axis=0)[:sample_count]
    return strain_full, branch_full, material_full


def _small_gap_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, "small_gap")
    principal_fit = np.asarray(bank["principal_fit"], dtype=np.float32)
    eigvecs_fit = np.asarray(bank["eigvecs_fit"], dtype=np.float32)
    material_fit = np.asarray(bank["material_fit"], dtype=np.float32)
    idx = rng.choice(weights.shape[0], size=sample_count, replace=True, p=weights)
    principal_seed = principal_fit[idx]
    eigvecs = eigvecs_fit[idx]
    material = material_fit[idx]

    i1_seed = np.sum(principal_seed, axis=1)
    amp = np.maximum(np.linalg.norm(principal_seed, axis=1), 1.0e-4)
    gap_total = rng.beta(1.5, 8.0, size=sample_count).astype(np.float32) * (0.08 + 0.35 * float(noise_scale)) * amp
    split = rng.uniform(0.05, 0.95, size=sample_count).astype(np.float32)
    d12 = gap_total * split
    d23 = gap_total * (1.0 - split)
    i1 = i1_seed * rng.uniform(0.85, 1.15, size=sample_count).astype(np.float32)
    i1 += rng.normal(scale=(0.05 + 0.10 * float(noise_scale)) * amp, size=sample_count).astype(np.float32)
    principal = principal_from_gap_coords(i1, d12, d23).astype(np.float32)
    strain = _reconstruct_strain(principal, eigvecs)
    strain, branch = _evaluate_exact(strain, material)
    return strain, branch, material.astype(np.float32)


def _edge_apex_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    tube_width: float,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported edge/apex side {side!r}.")
    rng = np.random.default_rng(seed)
    selection = "edge_apex_left" if side == "left" else "edge_apex_right"
    surface = "left_apex" if side == "left" else "right_apex"
    weights = _pick_weights(bank, selection)
    principal_fit = np.asarray(bank["principal_fit"], dtype=np.float32)
    eigvecs_fit = np.asarray(bank["eigvecs_fit"], dtype=np.float32)
    material_fit = np.asarray(bank["material_fit"], dtype=np.float32)

    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    total = 0
    attempts = 0
    while total < sample_count:
        attempts += 1
        need = sample_count - total
        draw = max(need * 2, 256)
        idx = rng.choice(weights.shape[0], size=draw, replace=True, p=weights)
        principal_seed = principal_fit[idx]
        eigvecs = eigvecs_fit[idx]
        material = material_fit[idx]
        scale = np.maximum(np.linalg.norm(principal_seed, axis=1, keepdims=True), 1.0e-8)
        direction = principal_seed / scale
        alpha = solve_alpha_for_surface(
            direction,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
            shear=material[:, 2],
            bulk=material[:, 3],
            lame=material[:, 4],
            surface=surface,
        )
        jitter = rng.uniform(-tube_width, tube_width, size=alpha.shape)
        alpha_jitter = alpha * (1.0 + jitter)
        valid = np.isfinite(alpha_jitter) & (alpha_jitter > 0.0)
        if not np.any(valid):
            if attempts >= 20:
                break
            continue
        principal = (alpha_jitter[valid, None] * direction[valid]).astype(np.float32)
        strain = _reconstruct_strain(principal, eigvecs[valid])
        strain, branch = _evaluate_exact(strain, material[valid])
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material[valid].astype(np.float32))
        total += strain.shape[0]
        if attempts >= 20:
            break
    if total == 0:
        return _real_like_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed + 1704,
            noise_scale=max(0.08, tube_width),
            selection=selection,
        )
    strain_full = np.concatenate(strain_parts, axis=0)[:sample_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:sample_count]
    material_full = np.concatenate(material_parts, axis=0)[:sample_count]
    return strain_full, branch_full, material_full


def _loading_paths_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, "margin_bulk")
    eigvecs_fit = np.asarray(bank["eigvecs_fit"], dtype=np.float32)
    material_fit = np.asarray(bank["material_fit"], dtype=np.float32)
    idx = rng.choice(weights.shape[0], size=sample_count, replace=True, p=weights)
    eigvecs = eigvecs_fit[idx]
    material = material_fit[idx]

    path_specs = [
        ("yield", np.array([1.0, 0.0, -1.0], dtype=np.float32)),
        ("yield", np.array([0.6, 0.4, -1.0], dtype=np.float32)),
        ("left_apex", principal_from_gap_coords(np.array([0.20]), np.array([0.20]), np.array([1.20]))[0].astype(np.float32)),
        ("right_apex", principal_from_gap_coords(np.array([0.20]), np.array([1.20]), np.array([0.20]))[0].astype(np.float32)),
        ("right_apex", np.array([1.0, 0.98, 0.95], dtype=np.float32)),
    ]

    principal_parts: list[np.ndarray] = []
    cursor = 0
    for spec_index, (surface, base_direction) in enumerate(path_specs):
        if spec_index == len(path_specs) - 1:
            count = sample_count - cursor
        else:
            count = sample_count // len(path_specs)
        if count <= 0:
            continue
        rows = slice(cursor, cursor + count)
        cursor += count
        direction = np.repeat(base_direction[None, :], count, axis=0)
        direction *= rng.normal(loc=1.0, scale=0.03 + 0.05 * float(noise_scale), size=direction.shape).astype(np.float32)
        direction.sort(axis=1)
        direction = direction[:, ::-1]
        alpha = solve_alpha_for_surface(
            direction,
            c_bar=material[rows, 0],
            sin_phi=material[rows, 1],
            shear=material[rows, 2],
            bulk=material[rows, 3],
            lame=material[rows, 4],
            surface=surface,  # type: ignore[arg-type]
        )
        if surface == "right_apex":
            hydro_idx = np.arange(count)
            hydro_mask = np.abs(direction[:, 0] - direction[:, 2]) < 0.08
            if np.any(hydro_mask):
                sp = np.maximum(np.abs(material[rows, 1][hydro_mask]), 1.0e-6)
                alpha_h = material[rows, 0][hydro_mask] / np.maximum(2.0 * material[rows, 3][hydro_mask] * sp * np.sum(direction[hydro_mask], axis=1), 1.0e-8)
                alpha[hydro_mask] = alpha_h
        alpha = np.where(np.isfinite(alpha), alpha, np.nan)
        jitter = rng.uniform(-0.04 - 0.08 * float(noise_scale), 0.04 + 0.08 * float(noise_scale), size=alpha.shape)
        alpha = alpha * (1.0 + jitter)
        alpha = np.where(alpha > 0.0, alpha, np.nan)
        valid = np.isfinite(alpha)
        if not np.any(valid):
            continue
        principal = (alpha[valid, None] * direction[valid]).astype(np.float32)
        principal_parts.append((rows, valid, principal))

    principal_full = np.zeros((sample_count, 3), dtype=np.float32)
    filled = np.zeros(sample_count, dtype=bool)
    for rows, valid, principal in principal_parts:
        target = np.arange(rows.start, rows.stop)[valid]
        principal_full[target] = principal
        filled[target] = True
    if not np.all(filled):
        fallback_principal, fallback_branch, fallback_material = _yield_tube_from_bank(
            bank,
            sample_count=int(np.sum(~filled)),
            seed=seed + 7001,
            tube_width=max(0.03, 0.5 * float(noise_scale)),
        )
        strain = np.zeros((sample_count, 6), dtype=np.float32)
        strain[filled] = _reconstruct_strain(principal_full[filled], eigvecs[filled])
        strain[~filled] = fallback_principal
        material_out = material.astype(np.float32)
        material_out[~filled] = fallback_material
        strain, branch = _evaluate_exact(strain, material_out)
        return strain, branch, material_out
    strain = _reconstruct_strain(principal_full, eigvecs)
    strain, branch = _evaluate_exact(strain, material)
    return strain, branch, material.astype(np.float32)


def _tail_extension_from_bank(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = _pick_weights(bank, "tail")
    idx = rng.choice(weights.shape[0], size=sample_count, replace=True, p=weights)
    principal = np.asarray(bank["principal_fit"], dtype=np.float32)[idx]
    eigvecs = np.asarray(bank["eigvecs_fit"], dtype=np.float32)[idx]
    material = np.asarray(bank["material_fit"], dtype=np.float32)[idx]
    i1 = np.sum(principal, axis=1)
    d12 = principal[:, 0] - principal[:, 1]
    d23 = principal[:, 1] - principal[:, 2]
    beta_v = rng.uniform(1.0, 1.0 + noise_scale, size=sample_count).astype(np.float32)
    beta_d = rng.uniform(1.0, 1.0 + 1.1 * noise_scale, size=sample_count).astype(np.float32)
    principal_tail = principal_from_gap_coords(i1 * beta_v, d12 * beta_d, d23 * beta_d).astype(np.float32)
    strain = _reconstruct_strain(principal_tail, eigvecs)
    strain, branch = _evaluate_exact(strain, material)
    return strain, branch, material.astype(np.float32)


def synthesize_from_principal_hybrid(
    bank: dict[str, np.ndarray | str],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pointwise constitutive states directly in principal coordinates."""
    if selection in {"uniform", "branch_balanced", "smooth_focus", "smooth_edge", "left_focus", "margin_bulk"}:
        strain, branch, material = _real_like_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
            selection=selection,
        )
    elif selection == "yield_tube":
        strain, branch, material = _yield_tube_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            tube_width=max(float(noise_scale), 1.0e-3),
        )
    elif selection == "boundary_smooth_right":
        strain, branch, material = _boundary_smooth_right_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            tube_width=max(float(noise_scale), 1.0e-3),
        )
    elif selection == "boundary_smooth_left":
        strain, branch, material = _boundary_smooth_left_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            tube_width=max(float(noise_scale), 1.0e-3),
        )
    elif selection == "edge_apex_right":
        strain, branch, material = _edge_apex_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            tube_width=max(float(noise_scale), 1.0e-3),
            side="right",
        )
    elif selection == "edge_apex_left":
        strain, branch, material = _edge_apex_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            tube_width=max(float(noise_scale), 1.0e-3),
            side="left",
        )
    elif selection == "tail":
        strain, branch, material = _tail_extension_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
        )
    elif selection == "small_gap":
        strain, branch, material = _small_gap_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
        )
    elif selection == "loading_paths":
        strain, branch, material = _loading_paths_from_bank(
            bank,
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
        )
    else:
        raise ValueError(f"Unknown principal-hybrid selection {selection!r}.")
    valid = np.ones(strain.shape[0], dtype=bool)
    return strain.astype(np.float32), branch.astype(np.int64), material.astype(np.float32), valid


def summarize_branch_geometry(
    strain: np.ndarray,
    branch: np.ndarray,
    material: np.ndarray,
) -> dict[str, float]:
    strain = np.asarray(strain, dtype=np.float32).reshape(-1, 6)
    branch = np.asarray(branch).reshape(-1)
    if material.ndim == 2 and material.shape[0] != strain.shape[0]:
        if material.shape[0] * 11 == strain.shape[0]:
            material = np.repeat(material.astype(np.float32), 11, axis=0)
        else:
            raise ValueError("Material shape does not align with strain shape.")
    material = np.asarray(material, dtype=np.float32).reshape(-1, 5)
    principal, _ = principal_values_and_vectors_from_strain(strain)
    geom = compute_branch_geometry_principal(
        principal,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    strain_norm = np.linalg.norm(strain, axis=1)
    sr_abs = np.abs(geom.m_smooth_right)
    sl_abs = np.abs(geom.m_smooth_left)
    la_abs = np.abs(geom.m_left_apex)
    ra_abs = np.abs(geom.m_right_apex)
    yield_abs = np.abs(geom.m_yield)
    gap_ratio = np.minimum(geom.delta12, geom.delta23) / (
        np.abs(principal[:, 0]) + np.abs(principal[:, 1]) + np.abs(principal[:, 2]) + 1.0e-12
    )
    return {
        "strain_norm_p95": float(np.quantile(strain_norm, 0.95)),
        "strain_norm_p99": float(np.quantile(strain_norm, 0.99)),
        "i1_p05": float(np.quantile(geom.i1, 0.05)),
        "i1_p50": float(np.quantile(geom.i1, 0.50)),
        "i1_p95": float(np.quantile(geom.i1, 0.95)),
        "delta12_p50": float(np.quantile(geom.delta12, 0.50)),
        "delta12_p95": float(np.quantile(geom.delta12, 0.95)),
        "delta23_p50": float(np.quantile(geom.delta23, 0.50)),
        "delta23_p95": float(np.quantile(geom.delta23, 0.95)),
        "yield_abs_p50": float(np.quantile(yield_abs, 0.50)),
        "yield_abs_p90": float(np.quantile(yield_abs, 0.90)),
        "yield_band_005": float(np.mean(yield_abs < 0.05)),
        "yield_band_010": float(np.mean(yield_abs < 0.10)),
        "sr_abs_p50": float(np.quantile(sr_abs, 0.50)),
        "sr_abs_p90": float(np.quantile(sr_abs, 0.90)),
        "sr_abs_p95": float(np.quantile(sr_abs, 0.95)),
        "sl_band_005": float(np.mean(sl_abs < 0.05)),
        "sr_band_002": float(np.mean(sr_abs < 0.02)),
        "sr_band_005": float(np.mean(sr_abs < 0.05)),
        "sr_band_010": float(np.mean(sr_abs < 0.10)),
        "la_band_005": float(np.mean(la_abs < 0.05)),
        "ra_band_005": float(np.mean(ra_abs < 0.05)),
        "small_gap_band_003": float(np.mean(gap_ratio < 0.03)),
        "small_gap_p50": float(np.quantile(gap_ratio, 0.50)),
        "small_gap_p90": float(np.quantile(gap_ratio, 0.90)),
        "smooth_fraction": float(np.mean(branch == 1)),
        "right_fraction": float(np.mean(branch == 3)),
        "elastic_fraction": float(np.mean(branch == 0)),
        "left_fraction": float(np.mean(branch == 2)),
        "apex_fraction": float(np.mean(branch == 4)),
    }
