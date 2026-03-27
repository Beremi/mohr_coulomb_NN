"""Neural-network models and feature builders for constitutive surrogates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .branch_geometry import compute_branch_geometry_principal
from .mohr_coulomb import exact_trial_principal_stress_3d
from .voigt import principal_values_and_vectors_from_strain, tensor_to_stress_voigt


def _safe_log(x: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), floor))


def lode_cos3theta_from_principal(principal_values: np.ndarray) -> np.ndarray:
    """Compute cos(3θ) from ordered principal values of a symmetric tensor."""
    p = np.asarray(principal_values, dtype=float)
    mean = np.mean(p, axis=1, keepdims=True)
    s = p - mean
    j2 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) / 6.0
    j3 = s[:, 0] * s[:, 1] * s[:, 2]
    denom = np.maximum(j2 ** 1.5, 1.0e-14)
    cos3theta = (3.0 * np.sqrt(3.0) / 2.0) * j3 / denom
    return np.clip(cos3theta, -1.0, 1.0)


def equivalent_deviatoric_measure(principal_values: np.ndarray) -> np.ndarray:
    """Simple equivalent magnitude of the deviatoric principal values."""
    p = np.asarray(principal_values, dtype=float)
    mean = np.mean(p, axis=1, keepdims=True)
    s = p - mean
    return np.sqrt(np.sum(s**2, axis=1))


def build_principal_features(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Build invariant-aware features for the recommended principal-stress model."""
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]
    trace = np.sum(eps_p, axis=1)
    eq_dev = equivalent_deviatoric_measure(eps_p)
    lode = lode_cos3theta_from_principal(eps_p)
    return np.column_stack(
        [
            eps_p,
            trace,
            eq_dev,
            lode,
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def build_raw_features(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Build raw-Voigt features for the baseline model."""
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]
    return np.column_stack(
        [
            strain_eng,
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def build_trial_features(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """
    Build raw-stress features around the elastic trial stress.

    This representation is better aligned with the actual return mapping and is
    more stable on real simulation states with very large signed strain ranges.
    """
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]

    trial = compute_trial_stress(strain_eng, material_reduced)

    scale = np.maximum(c_bar, 1.0)
    return np.column_stack(
        [
            np.arcsinh(strain_eng),
            np.arcsinh(trial / scale[:, None]),
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def build_trial_principal_features(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
) -> np.ndarray:
    """
    Build principal/invariant features around the elastic trial principal stress.

    This is intended for structured plastic-correction models: the network sees
    ordered principal strains plus the corresponding elastic trial stress state
    and learns only the non-elastic correction.
    """
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    trial_p = np.asarray(trial_principal, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]

    trace_eps = np.sum(eps_p, axis=1)
    eq_dev_eps = equivalent_deviatoric_measure(eps_p)
    lode_eps = lode_cos3theta_from_principal(eps_p)
    eps_gap_12 = eps_p[:, 0] - eps_p[:, 1]
    eps_gap_23 = eps_p[:, 1] - eps_p[:, 2]

    scale = np.maximum(c_bar, 1.0)
    mean_trial = np.mean(trial_p, axis=1)
    eq_dev_trial = equivalent_deviatoric_measure(trial_p)
    lode_trial = lode_cos3theta_from_principal(trial_p)

    return np.column_stack(
        [
            eps_p,
            trace_eps,
            eq_dev_eps,
            lode_eps,
            eps_gap_12,
            eps_gap_23,
            np.arcsinh(trial_p / scale[:, None]),
            mean_trial / scale,
            eq_dev_trial / scale,
            lode_trial,
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def build_trial_principal_geom_features(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
) -> np.ndarray:
    """
    Build principal/trial features augmented with exact constitutive geometry.

    The geometry features are computed analytically from the ordered principal
    strain state and reduced material parameters so the network can see how far
    the current point sits from the constitutive interfaces.
    """
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    trial_p = np.asarray(trial_principal, dtype=float)

    base = build_trial_principal_features(eps_p, mat, trial_p).astype(np.float32)
    geom = compute_branch_geometry_principal(
        eps_p,
        c_bar=mat[:, 0],
        sin_phi=mat[:, 1],
        shear=mat[:, 2],
        bulk=mat[:, 3],
        lame=mat[:, 4],
    )

    geom_features = np.column_stack(
        [
            geom.m_yield,
            geom.m_smooth_left,
            geom.m_smooth_right,
            geom.m_left_apex,
            geom.m_right_apex,
            geom.m_left_vs_right,
            geom.gap12_norm,
            geom.gap23_norm,
            geom.gap_min_norm,
            np.arcsinh(geom.lambda_s),
            np.arcsinh(geom.lambda_l),
            np.arcsinh(geom.lambda_r),
            np.arcsinh(geom.lambda_a),
            geom.d_geom,
        ]
    ).astype(np.float32)
    return np.concatenate([base, geom_features], axis=1).astype(np.float32)


def compute_trial_surface_feature_stats(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> dict[str, list[float] | float]:
    """Compute train-only robust statistics for the plastic-surface feature transforms."""
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    p_tr = np.mean(trial_p, axis=1)
    g_tr = trial_p[:, 0] - trial_p[:, 2]
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    sin_phi_safe = np.maximum(np.abs(mat[:, 1]), 1.0e-6)
    p_apex = c_bar_safe / (2.0 * sin_phi_safe)
    p_tilde = p_tr / np.maximum(np.abs(p_apex), 1.0)
    g_tilde = ((1.0 + mat[:, 1]) * g_tr) / c_bar_safe
    f_tilde = f_tr / c_bar_safe

    material_mean = mat.mean(axis=0)
    material_std = mat.std(axis=0)
    material_std = np.where(material_std < 1.0e-12, 1.0, material_std)

    return {
        "s_p": float(max(np.quantile(np.abs(p_tilde), 0.95), 1.0)),
        "s_g": float(max(np.quantile(np.maximum(g_tilde, 0.0), 0.95), 1.0)),
        "s_f": float(max(np.quantile(np.abs(f_tilde), 0.95), 1.0)),
        "material_mean": material_mean.astype(np.float32).tolist(),
        "material_std": material_std.astype(np.float32).tolist(),
    }


def build_trial_surface_features_f1(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
    feature_stats: dict[str, list[float] | float],
) -> np.ndarray:
    """Build the plastic-surface `F1` vector from exact trial principal stress and geometry margins."""
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    trial_p = np.asarray(trial_principal, dtype=float)
    material_mean = np.asarray(feature_stats["material_mean"], dtype=float)
    material_std = np.asarray(feature_stats["material_std"], dtype=float)
    mat_norm = (mat - material_mean) / material_std

    p_tr = np.mean(trial_p, axis=1)
    g_tr = trial_p[:, 0] - trial_p[:, 2]
    rho_tr = (trial_p[:, 0] - 2.0 * trial_p[:, 1] + trial_p[:, 2]) / (g_tr + 1.0e-12)
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    sin_phi_safe = np.maximum(np.abs(mat[:, 1]), 1.0e-6)
    p_apex = c_bar_safe / (2.0 * sin_phi_safe)
    p_tilde = p_tr / np.maximum(np.abs(p_apex), 1.0)
    g_tilde = ((1.0 + mat[:, 1]) * g_tr) / c_bar_safe
    f_tilde = f_tr / c_bar_safe

    geom = compute_branch_geometry_principal(
        eps_p,
        c_bar=mat[:, 0],
        sin_phi=mat[:, 1],
        shear=mat[:, 2],
        bulk=mat[:, 3],
        lame=mat[:, 4],
    )
    geom_features = np.column_stack(
        [
            geom.m_yield,
            geom.m_smooth_left,
            geom.m_smooth_right,
            geom.m_left_apex,
            geom.m_right_apex,
            geom.gap12_norm,
            geom.gap23_norm,
            geom.m_left_vs_right,
            geom.d_geom,
        ]
    ).astype(np.float32)

    return np.column_stack(
        [
            np.arcsinh(p_tilde / float(feature_stats["s_p"])),
            np.log1p(np.maximum(g_tilde, 0.0) / float(feature_stats["s_g"])),
            np.arctanh(np.clip(rho_tr, -0.999999, 0.999999)),
            np.arcsinh(f_tilde / float(feature_stats["s_f"])),
            geom_features,
            mat_norm,
        ]
    ).astype(np.float32)


def build_trial_principal_surface_features(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
    feature_stats: dict[str, list[float] | float],
) -> np.ndarray:
    """Compatibility wrapper for the primary plastic-surface feature family."""
    return build_trial_surface_features_f1(
        strain_principal,
        material_reduced,
        trial_principal,
        feature_stats,
    )


def _safe_logit(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def compute_trial_exact_latent_feature_stats(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> dict[str, list[float] | float]:
    """Compute train-only robust statistics for Program X exact-latent features."""
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    p_tr = np.mean(trial_p, axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    g_tr = a_tr + b_tr
    rho_tr = (a_tr - b_tr) / np.maximum(g_tr, 1.0e-12)
    lambda_tr = np.clip(a_tr / np.maximum(g_tr, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]

    material_mean = mat.mean(axis=0)
    material_std = mat.std(axis=0)
    material_std = np.where(material_std < 1.0e-12, 1.0, material_std)

    return {
        "s_p": float(max(np.quantile(np.abs(p_tr), 0.95), 1.0)),
        "s_g": float(max(np.quantile(np.maximum(g_tr, 0.0), 0.95), 1.0)),
        "s_f": float(max(np.quantile(np.abs(f_tr), 0.95), 1.0)),
        "material_mean": material_mean.astype(np.float32).tolist(),
        "material_std": material_std.astype(np.float32).tolist(),
        "rho_tr_q05": float(np.quantile(rho_tr, 0.05)),
        "rho_tr_q95": float(np.quantile(rho_tr, 0.95)),
        "lambda_min": float(np.quantile(lambda_tr, 0.05)),
        "lambda_max": float(np.quantile(lambda_tr, 0.95)),
    }


def build_trial_exact_latent_features_f1(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
    feature_stats: dict[str, list[float] | float],
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Build the Program X exact-latent feature vector."""
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    trial_p = np.asarray(trial_principal, dtype=float)
    material_mean = np.asarray(feature_stats["material_mean"], dtype=float)
    material_std = np.asarray(feature_stats["material_std"], dtype=float)
    mat_norm = (mat - material_mean) / material_std

    p_tr = np.mean(trial_p, axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    g_tr = a_tr + b_tr
    rho_tr = (a_tr - b_tr) / np.maximum(g_tr, eps)
    lambda_tr = np.clip(a_tr / np.maximum(g_tr, eps), 1.0e-6, 1.0 - 1.0e-6)
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]

    geom = compute_branch_geometry_principal(
        eps_p,
        c_bar=mat[:, 0],
        sin_phi=mat[:, 1],
        shear=mat[:, 2],
        bulk=mat[:, 3],
        lame=mat[:, 4],
    )
    geom_features = np.column_stack(
        [
            geom.m_yield,
            geom.m_smooth_left,
            geom.m_smooth_right,
            geom.m_left_apex,
            geom.m_right_apex,
            geom.gap12_norm,
            geom.gap23_norm,
            geom.gap_min_norm,
            geom.m_left_vs_right,
            geom.d_geom,
        ]
    ).astype(np.float32)

    return np.column_stack(
        [
            np.arcsinh(p_tr / float(feature_stats["s_p"])),
            np.log1p(np.maximum(g_tr, 0.0) / float(feature_stats["s_g"])),
            np.arctanh(np.clip(rho_tr, -0.999999, 0.999999)),
            _safe_logit(lambda_tr, eps=eps),
            np.arcsinh(f_tr / float(feature_stats["s_f"])),
            geom_features,
            mat_norm,
        ]
    ).astype(np.float32)


def compute_trial_soft_atlas_feature_stats(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> dict[str, list[float] | float]:
    """Compute train-only robust statistics for the packet4 soft-atlas feature transforms."""
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    p_tr = np.mean(trial_p, axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    g_tr = a_tr + b_tr
    lambda_tr = np.clip(np.divide(a_tr, g_tr + 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]

    material_mean = mat.mean(axis=0)
    material_std = mat.std(axis=0)
    material_std = np.where(material_std < 1.0e-12, 1.0, material_std)

    return {
        "s_p": float(max(np.quantile(np.abs(p_tr), 0.95), 1.0)),
        "s_g": float(max(np.quantile(np.maximum(g_tr, 0.0), 0.95), 1.0)),
        "s_f": float(max(np.quantile(np.abs(f_tr), 0.95), 1.0)),
        "material_mean": material_mean.astype(np.float32).tolist(),
        "material_std": material_std.astype(np.float32).tolist(),
        "lambda_min": float(np.quantile(lambda_tr, 0.05)),
        "lambda_max": float(np.quantile(lambda_tr, 0.95)),
    }


def build_trial_soft_atlas_features_f1(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
    feature_stats: dict[str, list[float] | float],
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Build the packet4 soft-atlas feature vector from trial principal stress and geometry margins."""
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    trial_p = np.asarray(trial_principal, dtype=float)
    material_mean = np.asarray(feature_stats["material_mean"], dtype=float)
    material_std = np.asarray(feature_stats["material_std"], dtype=float)
    mat_norm = (mat - material_mean) / material_std

    p_tr = np.mean(trial_p, axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    g_tr = a_tr + b_tr
    rho_tr = (a_tr - b_tr) / (g_tr + eps)
    lambda_tr = a_tr / (g_tr + eps)
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]

    geom = compute_branch_geometry_principal(
        eps_p,
        c_bar=mat[:, 0],
        sin_phi=mat[:, 1],
        shear=mat[:, 2],
        bulk=mat[:, 3],
        lame=mat[:, 4],
    )
    geom_features = np.column_stack(
        [
            geom.m_yield,
            geom.m_smooth_left,
            geom.m_smooth_right,
            geom.m_left_apex,
            geom.m_right_apex,
            geom.gap12_norm,
            geom.gap23_norm,
        ]
    ).astype(np.float32)

    return np.column_stack(
        [
            np.arcsinh(p_tr / float(feature_stats["s_p"])),
            np.log1p(np.maximum(g_tr, 0.0) / float(feature_stats["s_g"])),
            _safe_logit(lambda_tr, eps=eps),
            np.arcsinh(f_tr / float(feature_stats["s_f"])),
            rho_tr,
            geom_features,
            mat_norm,
        ]
    ).astype(np.float32)


def build_trial_soft_atlas_surface_features(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
    trial_principal: np.ndarray,
    feature_stats: dict[str, list[float] | float],
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compatibility wrapper for the packet4 soft-atlas feature family."""
    return build_trial_soft_atlas_features_f1(
        strain_principal,
        material_reduced,
        trial_principal,
        feature_stats,
        eps=eps,
    )


build_trial_soft_atlas_principal_features = build_trial_soft_atlas_features_f1
compute_trial_soft_atlas_surface_feature_stats = compute_trial_soft_atlas_feature_stats


def compute_trial_abr_feature_stats(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> dict[str, list[float] | float]:
    """Compute train-only robust statistics for the ACN trial feature transforms."""
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    p_tr = np.mean(trial_p, axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]

    material_mean = mat.mean(axis=0)
    material_std = mat.std(axis=0)
    material_std = np.where(material_std < 1.0e-12, 1.0, material_std)

    return {
        "s_p": float(max(np.quantile(np.abs(p_tr), 0.95), 1.0)),
        "s_a_tr": float(max(np.quantile(a_tr, 0.95), 1.0)),
        "s_b_tr": float(max(np.quantile(b_tr, 0.95), 1.0)),
        "s_f": float(max(np.quantile(np.abs(f_tr), 0.95), 1.0)),
        "material_mean": material_mean.astype(np.float32).tolist(),
        "material_std": material_std.astype(np.float32).tolist(),
    }


def build_trial_abr_features_f1(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
    feature_stats: dict[str, list[float] | float],
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Build the transformed Stage 1 `F1` vector from exact trial principal stress."""
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    material_mean = np.asarray(feature_stats["material_mean"], dtype=float)
    material_std = np.asarray(feature_stats["material_std"], dtype=float)

    p_tr = np.mean(trial_p, axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    f_tr = (1.0 + mat[:, 1]) * trial_p[:, 0] - (1.0 - mat[:, 1]) * trial_p[:, 2] - mat[:, 0]
    rho_tr = (a_tr - b_tr) / (a_tr + b_tr + eps)
    mat_norm = (mat - material_mean) / material_std

    return np.column_stack(
        [
            np.arcsinh(p_tr / float(feature_stats["s_p"])),
            np.log1p(np.maximum(a_tr, 0.0) / float(feature_stats["s_a_tr"])),
            np.log1p(np.maximum(b_tr, 0.0) / float(feature_stats["s_b_tr"])),
            np.arcsinh(f_tr / float(feature_stats["s_f"])),
            rho_tr,
            mat_norm,
        ]
    ).astype(np.float32)


def build_trial_abr_features_f2(
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
    feature_stats: dict[str, list[float] | float],
) -> np.ndarray:
    """Build the Stage 0 ablation helper `F2` vector."""
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    material_mean = np.asarray(feature_stats["material_mean"], dtype=float)
    material_std = np.asarray(feature_stats["material_std"], dtype=float)
    mat_norm = (mat - material_mean) / material_std
    return np.concatenate([trial_p, mat_norm], axis=1).astype(np.float32)


def compute_trial_stress(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Compute elastic trial stress in Voigt notation from engineering strain."""
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    shear = mat[:, 2]
    lame = mat[:, 4]
    trace = np.sum(strain_eng[:, :3], axis=1)
    trial = np.empty_like(strain_eng, dtype=float)
    trial[:, :3] = lame[:, None] * trace[:, None] + 2.0 * shear[:, None] * strain_eng[:, :3]
    trial[:, 3:] = shear[:, None] * strain_eng[:, 3:]
    return trial.astype(np.float32)


@dataclass
class Standardizer:
    """Feature or target standardization parameters."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def identity(cls, dim: int) -> "Standardizer":
        return cls(mean=np.zeros(dim, dtype=np.float32), std=np.ones(dim, dtype=np.float32))

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Standardizer":
        arr = np.asarray(array, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1.0e-12, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=float)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=float)
        return (arr * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, dct: dict[str, list[float]]) -> "Standardizer":
        return cls(mean=np.asarray(dct["mean"], dtype=np.float32), std=np.asarray(dct["std"], dtype=np.float32))


class ResidualBlock(nn.Module):
    """Small residual block for tabular MLPs."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.out_norm = nn.LayerNorm(width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.out_norm(x + self.net(x)))


class PrincipalStressNet(nn.Module):
    """Recommended spectral MLP with auxiliary branch classification head."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        n_branches: int = 5,
        sort_output: bool = True,
    ) -> None:
        super().__init__()
        self.sort_output = sort_output
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_stress = nn.Linear(width, 3)
        self.head_branch = nn.Linear(width, n_branches)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        stress = self.head_stress(h)
        if self.sort_output:
            stress, _ = torch.sort(stress, dim=-1, descending=True)
        branch_logits = self.head_branch(h)
        return {"stress": stress, "branch_logits": branch_logits}


class RawStressNet(nn.Module):
    """Baseline raw-input MLP that predicts all six stress components directly."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_stress = nn.Linear(width, 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return {"stress": self.head_stress(h)}


class RawStressBranchNet(nn.Module):
    """Raw-stress MLP with an auxiliary branch classification head."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        n_branches: int = 5,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_stress = nn.Linear(width, 6)
        self.head_branch = nn.Linear(width, n_branches)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return {
            "stress": self.head_stress(h),
            "branch_logits": self.head_branch(h),
        }


class AcnStressNet(nn.Module):
    """Small branchless MLP that predicts raw ABR coordinates."""

    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        self.net = nn.Sequential(*layers)
        self.head_stress = nn.Linear(in_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {"stress": self.head_stress(h)}


class PlasticSurfaceNet(nn.Module):
    """Small branchless MLP that predicts plastic-surface coordinates."""

    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        self.net = nn.Sequential(*layers)
        self.head_surface = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {"stress": self.head_surface(h)}


class ExactLatentRegressorNet(nn.Module):
    """Small branchwise MLP that predicts exact latent values."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        self.net = nn.Sequential(*layers)
        self.head_latent = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {"stress": self.head_latent(h)}


class PlasticBranchStructuredSurfaceNet(nn.Module):
    """Small plastic-surface MLP with a learned 4-way plastic branch head."""

    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        self.net = nn.Sequential(*layers)
        self.head_branch = nn.Linear(in_dim, 4)
        self.head_smooth = nn.Linear(in_dim, 2)
        self.head_left = nn.Linear(in_dim, 1)
        self.head_right = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(x)
        return {
            "stress": torch.cat(
                [
                    self.head_smooth(h),
                    self.head_left(h),
                    self.head_right(h),
                ],
                dim=1,
            ),
            "branch_logits": self.head_branch(h),
        }


class _SoftAtlasBaseNet(nn.Module):
    """Shared trunk and atlas heads for the packet4 soft admissible atlas."""

    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.trunk_in = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_route = nn.Linear(width, 4)
        self.head_smooth = nn.Linear(width, 2)
        self.head_left = nn.Linear(width, 1)
        self.head_right = nn.Linear(width, 1)

    def _run_trunk(self, x: torch.Tensor, film_params: torch.Tensor | None = None) -> torch.Tensor:
        h = self.trunk_in(x)
        for idx, block in enumerate(self.blocks):
            h = block(h)
            if film_params is not None:
                gamma = 0.5 * torch.tanh(film_params[:, idx, 0, :])
                beta = 0.5 * torch.tanh(film_params[:, idx, 1, :])
                h = h * (1.0 + gamma) + beta
        return h

    def _head_outputs(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        smooth = self.head_smooth(h)
        left = self.head_left(h)
        right = self.head_right(h)
        route_logits = self.head_route(h)
        apex = h.new_zeros((h.shape[0], 2))
        return {
            "stress": torch.cat([smooth, left, right], dim=1),
            "branch_logits": route_logits,
            "route_logits": route_logits,
            "smooth": smooth,
            "left": left,
            "right": right,
            "apex": apex,
        }


class SoftAtlasConcatNet(_SoftAtlasBaseNet):
    """Packet4 soft-atlas network with plain feature concatenation."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._run_trunk(x)
        return self._head_outputs(h)


class SoftAtlasFiLMNet(_SoftAtlasBaseNet):
    """Packet4 soft-atlas network with material-conditioned FiLM modulation."""

    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
        material_dim: int = 5,
    ) -> None:
        super().__init__(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
        self.material_dim = material_dim
        cond_hidden = max(16, width)
        self.conditioner = nn.Sequential(
            nn.Linear(material_dim, cond_hidden),
            nn.SiLU(),
            nn.Linear(cond_hidden, 2 * depth * width),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[1] < self.material_dim:
            raise ValueError(
                f"Expected at least {self.material_dim} material features in the packet4 input, got {x.shape[1]}."
            )
        material = x[:, -self.material_dim :]
        film = self.conditioner(material)
        film = film.view(x.shape[0], self.depth, 2, self.width)
        h = self._run_trunk(x, film_params=film)
        return self._head_outputs(h)


def build_model(
    model_kind: str,
    input_dim: int,
    width: int = 256,
    depth: int = 4,
    dropout: float = 0.0,
) -> nn.Module:
    """Factory for supported surrogate architectures."""
    if model_kind == "principal":
        return PrincipalStressNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout, sort_output=True)
    if model_kind == "trial_principal_branch_residual":
        return PrincipalStressNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout, sort_output=False)
    if model_kind == "trial_principal_geom_plastic_branch_residual":
        return PrincipalStressNet(
            input_dim=input_dim,
            width=width,
            depth=depth,
            dropout=dropout,
            n_branches=4,
            sort_output=False,
        )
    if model_kind == "trial_principal_geom_projected_student":
        return PrincipalStressNet(
            input_dim=input_dim,
            width=width,
            depth=depth,
            dropout=dropout,
            n_branches=4,
            sort_output=False,
        )
    if model_kind == "trial_abr_acn_f1":
        return AcnStressNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind == "trial_surface_acn_f1":
        return PlasticSurfaceNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind == "exact_latent_regressor":
        return ExactLatentRegressorNet(input_dim=input_dim, output_dim=1, width=width, depth=depth, dropout=dropout)
    if model_kind == "trial_surface_branch_structured_f1":
        return PlasticBranchStructuredSurfaceNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind == "trial_surface_soft_atlas_f1_concat":
        return SoftAtlasConcatNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind == "trial_surface_soft_atlas_f1_film":
        return SoftAtlasFiLMNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind in {"raw", "trial_raw", "trial_raw_residual"}:
        return RawStressNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind in {"raw_branch", "trial_raw_branch", "trial_raw_branch_residual"}:
        return RawStressBranchNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    raise ValueError(f"Unsupported model kind {model_kind!r}.")


def stress_tensor_from_principal_torch(principal_stress: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    """Reconstruct stress tensors V diag(s) V^T in torch."""
    return torch.einsum("nij,nj,nkj->nik", eigvecs, principal_stress, eigvecs)


def stress_voigt_from_principal_torch(principal_stress: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    """Reconstruct stress Voigt vectors from principal stresses and eigenvectors."""
    tensor = stress_tensor_from_principal_torch(principal_stress, eigvecs)
    out = torch.zeros((tensor.shape[0], 6), dtype=tensor.dtype, device=tensor.device)
    out[:, 0] = tensor[:, 0, 0]
    out[:, 1] = tensor[:, 1, 1]
    out[:, 2] = tensor[:, 2, 2]
    out[:, 3] = tensor[:, 0, 1]
    out[:, 4] = tensor[:, 0, 2]
    out[:, 5] = tensor[:, 1, 2]
    return out


def stress_voigt_from_principal_numpy(principal_stress: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Reconstruct stress Voigt vectors from principal stresses and eigenvectors."""
    tensor = np.einsum("nij,nj,nkj->nik", eigvecs, principal_stress, eigvecs)
    return tensor_to_stress_voigt(tensor)


def spectral_decomposition_from_strain(strain_eng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute principal strains and eigenvectors from engineering strains."""
    return principal_values_and_vectors_from_strain(strain_eng)


def exact_trial_principal_from_strain(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Return the constitutive operator's exact ordered elastic-trial principal stress."""
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    return exact_trial_principal_stress_3d(
        strain_eng,
        c_bar=mat[:, 0],
        sin_phi=mat[:, 1],
        shear=mat[:, 2],
        bulk=mat[:, 3],
        lame=mat[:, 4],
    ).astype(np.float32)


def decode_abr_to_principal_torch(
    abr: torch.Tensor,
    *,
    c_bar: torch.Tensor,
    sin_phi: torch.Tensor,
) -> torch.Tensor:
    """Decode `(a, b, r)` coordinates to ordered principal stress in torch."""
    denom = torch.where(torch.abs(sin_phi) > 1.0e-14, 2.0 * sin_phi, torch.full_like(sin_phi, float("inf")))
    sigma3 = (c_bar - abr[:, 2] - (1.0 + sin_phi) * (abr[:, 0] + abr[:, 1])) / denom
    sigma2 = sigma3 + abr[:, 1]
    sigma1 = sigma2 + abr[:, 0]
    return torch.stack([sigma1, sigma2, sigma3], dim=1)


def decode_exact_branch_latents_to_principal_torch(
    latent_values: torch.Tensor,
    branch_id: torch.Tensor,
    trial_principal: torch.Tensor,
    material_reduced: torch.Tensor,
) -> torch.Tensor:
    """Decode exact branch latents to ordered principal stress in torch."""
    latent = latent_values
    if latent.ndim == 1:
        latent = latent[:, None]
    branch = branch_id.reshape(-1).to(dtype=torch.long, device=trial_principal.device)
    trial = trial_principal.to(dtype=torch.float32)
    material = material_reduced.to(dtype=torch.float32)
    if trial.ndim != 2 or trial.shape[1] != 3:
        raise ValueError(f"Expected trial_principal shape (n, 3), got {tuple(trial.shape)}.")
    if material.ndim != 2 or material.shape[1] != 5:
        raise ValueError(f"Expected material_reduced shape (n, 5), got {tuple(material.shape)}.")
    if branch.shape[0] != trial.shape[0]:
        raise ValueError(f"Expected branch_id length {trial.shape[0]}, got {branch.shape[0]}.")
    if latent.shape[0] != trial.shape[0]:
        raise ValueError(f"Expected latent_values length {trial.shape[0]}, got {latent.shape[0]}.")

    out = trial.clone()
    c_bar = material[:, 0]
    sin_phi = material[:, 1]
    shear = material[:, 2]
    lame = material[:, 4]

    smooth = branch == 1
    left = branch == 2
    right = branch == 3
    apex = branch == 4

    if torch.any(smooth):
        lam = latent[smooth, 0]
        sp = sin_phi[smooth]
        ll = lame[smooth]
        mu = shear[smooth]
        out[smooth, 0] = trial[smooth, 0] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        out[smooth, 1] = trial[smooth, 1] - lam * (2.0 * ll * sp)
        out[smooth, 2] = trial[smooth, 2] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
    if torch.any(left):
        lam = latent[left, 0]
        sp = sin_phi[left]
        ll = lame[left]
        mu = shear[left]
        sigma12 = 0.5 * (trial[left, 0] + trial[left, 1]) - lam * (2.0 * ll * sp + mu * (1.0 + sp))
        sigma3 = trial[left, 2] - lam * (2.0 * ll * sp - 2.0 * mu * (1.0 - sp))
        out[left, 0] = sigma12
        out[left, 1] = sigma12
        out[left, 2] = sigma3
    if torch.any(right):
        lam = latent[right, 0]
        sp = sin_phi[right]
        ll = lame[right]
        mu = shear[right]
        sigma1 = trial[right, 0] - lam * (2.0 * ll * sp + 2.0 * mu * (1.0 + sp))
        sigma23 = 0.5 * (trial[right, 1] + trial[right, 2]) - lam * (2.0 * ll * sp - mu * (1.0 - sp))
        out[right, 0] = sigma1
        out[right, 1] = sigma23
        out[right, 2] = sigma23
    if torch.any(apex):
        sigma_a = c_bar[apex] / torch.clamp(2.0 * sin_phi[apex], min=1.0e-12)
        out[apex] = sigma_a[:, None]

    out, _ = torch.sort(out, dim=1, descending=True)
    return out


def decode_grho_to_principal_torch(
    grho: torch.Tensor,
    *,
    c_bar: torch.Tensor,
    sin_phi: torch.Tensor,
) -> torch.Tensor:
    """Decode `(g, rho)` coordinates to ordered principal stress in torch."""
    denom = torch.where(torch.abs(sin_phi) > 1.0e-14, 2.0 * sin_phi, torch.full_like(sin_phi, float("inf")))
    g = grho[:, 0]
    rho = torch.clamp(grho[:, 1], -1.0, 1.0)
    a = 0.5 * g * (1.0 + rho)
    b = 0.5 * g * (1.0 - rho)
    sigma3 = (c_bar - (1.0 + sin_phi) * g) / denom
    sigma2 = sigma3 + b
    sigma1 = sigma2 + a
    return torch.stack([sigma1, sigma2, sigma3], dim=1)


def project_grho_to_branch_specialized_torch(
    grho: torch.Tensor,
    branch_id: torch.Tensor,
) -> torch.Tensor:
    """Torch version of the branch-specialized plastic-surface projection."""
    if grho.ndim != 2 or grho.shape[1] != 2:
        raise ValueError(f"Expected grho shape (n, 2), got {tuple(grho.shape)}.")
    branch = branch_id.reshape(-1).to(dtype=torch.long, device=grho.device)
    if branch.shape[0] != grho.shape[0]:
        raise ValueError(f"Expected branch_id length {grho.shape[0]}, got {branch.shape[0]}.")
    g = torch.clamp(grho[:, 0], min=0.0)
    rho = torch.clamp(grho[:, 1], min=-1.0, max=1.0)
    left = branch == 2
    right = branch == 3
    apex = branch == 4
    rho = torch.where(left, -torch.ones_like(rho), rho)
    rho = torch.where(right, torch.ones_like(rho), rho)
    g = torch.where(apex, torch.zeros_like(g), g)
    rho = torch.where(apex, torch.zeros_like(rho), rho)
    return torch.stack([g, rho], dim=1)


def decode_branch_specialized_grho_to_principal_torch(
    grho: torch.Tensor,
    branch_id: torch.Tensor,
    *,
    c_bar: torch.Tensor,
    sin_phi: torch.Tensor,
) -> torch.Tensor:
    """Decode branch-specialized plastic-surface coordinates to ordered principal stress."""
    projected = project_grho_to_branch_specialized_torch(grho, branch_id)
    return decode_grho_to_principal_torch(projected, c_bar=c_bar, sin_phi=sin_phi)
