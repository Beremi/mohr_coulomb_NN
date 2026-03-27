"""Training and evaluation utilities for constitutive surrogates."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .branch_geometry import compute_branch_geometry_principal
from .data import load_arrays
from .models import (
    Standardizer,
    build_model,
    build_principal_features,
    build_raw_features,
    build_trial_abr_features_f1,
    build_trial_surface_features_f1,
    build_trial_soft_atlas_features_f1,
    build_trial_principal_geom_features,
    build_trial_principal_features,
    build_trial_features,
    decode_abr_to_principal_torch,
    decode_grho_to_principal_torch,
    compute_trial_soft_atlas_feature_stats,
    exact_trial_principal_from_strain,
    compute_trial_stress,
    compute_trial_surface_feature_stats,
    decode_branch_specialized_grho_to_principal_torch,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
    stress_voigt_from_principal_torch,
)
from .mohr_coulomb import BRANCH_NAMES, BRANCH_TO_ID, encode_principal_to_abr, encode_principal_to_grho, yield_function_principal_3d
from .principal_projection import project_mc_principal_torch
from .voigt import stress_voigt_to_tensor

PLASTIC_BRANCH_NAMES = BRANCH_NAMES[1:]


def choose_device(device: str = "auto") -> torch.device:
    """Choose a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _clone_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _update_ema_state_dict(
    ema_state_dict: dict[str, Any],
    model: nn.Module,
    decay: float,
) -> None:
    if decay <= 0.0:
        return
    model_state = model.state_dict()
    with torch.no_grad():
        for key, ema_value in ema_state_dict.items():
            model_value = model_state[key]
            if torch.is_tensor(ema_value) and torch.is_tensor(model_value):
                if torch.is_floating_point(ema_value) or torch.is_complex(ema_value):
                    ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
                else:
                    ema_state_dict[key] = model_value.detach().clone()
            else:
                ema_state_dict[key] = copy.deepcopy(model_value)


@dataclass
class TrainingConfig:
    """Hyperparameters and file paths for training."""
    dataset: str
    run_dir: str
    model_kind: str = "principal"
    epochs: int = 150
    batch_size: int = 2048
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    width: int = 256
    depth: int = 4
    dropout: float = 0.0
    seed: int = 0
    patience: int = 25
    grad_clip: float = 1.0
    branch_loss_weight: float = 0.1
    num_workers: int = 0
    device: str = "auto"
    scheduler_kind: str = "plateau"
    warmup_epochs: int = 0
    min_lr: float = 1.0e-6
    plateau_factor: float = 0.5
    plateau_patience: int | None = None
    lbfgs_epochs: int = 0
    lbfgs_lr: float = 0.25
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 100
    log_every_epochs: int = 0
    stress_weight_alpha: float = 0.0
    stress_weight_scale: float = 250.0
    checkpoint_metric: str = "loss"
    init_checkpoint: str | None = None
    regression_loss_kind: str = "mse"
    huber_delta: float = 1.0
    voigt_mae_weight: float = 0.0
    snapshot_every_epochs: int = 0
    tangent_loss_weight: float = 0.0
    tangent_fd_scale: float = 1.0e-6
    projection_mode: str = "exact"
    projection_tau: float = 0.05
    ema_decay: float = 0.0
    ema_eval: bool = False
    ema_start_epoch: int = 1
    projected_student_hard_loss_multiplier: float = 1.0
    projected_student_any_boundary_loss_multiplier: float = 1.0
    projected_student_high_disp_loss_multiplier: float = 1.0
    projected_student_teacher_gap_q90_loss_multiplier: float = 1.0
    projected_student_teacher_gap_q95_loss_multiplier: float = 1.0
    projected_student_delta_gap_q90_loss_multiplier: float = 1.0
    projected_student_edge_apex_loss_multiplier: float = 1.0
    projected_student_candidate_loss_weights: dict[int, float] | None = None
    projected_student_branch_loss_weights: dict[int, float] | None = None
    projected_student_teacher_alignment_focus_multiplier: float = 1.0
    projected_student_hard_quantile: float = 0.0
    projected_student_hard_quantile_weight: float = 0.0
    projected_student_high_disp_threshold: float | None = None


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_keys(dataset_path: str) -> set[str]:
    with h5py.File(dataset_path, "r") as f:
        return set(f.keys())


def _dataset_attrs(dataset_path: str) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as f:
        return {key: (value.decode() if isinstance(value, bytes) else value) for key, value in f.attrs.items()}


def _principal_stress_from_stress(stress_voigt: np.ndarray) -> np.ndarray:
    tensor = stress_voigt_to_tensor(stress_voigt)
    vals = np.linalg.eigvalsh(tensor)
    return vals[:, ::-1].astype(np.float32)


def _json_attr_dict(attrs: dict[str, Any], key: str) -> dict[str, Any] | None:
    raw = attrs.get(key)
    if raw is None:
        return None
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _is_acn_model(model_kind: str) -> bool:
    return model_kind == "trial_abr_acn_f1"


def _is_surface_model(model_kind: str) -> bool:
    return model_kind in {"trial_surface_acn_f1", "trial_surface_branch_structured_f1", "trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"}


def _is_branch_structured_surface_model(model_kind: str) -> bool:
    return model_kind == "trial_surface_branch_structured_f1"


def _is_soft_atlas_surface_model(model_kind: str) -> bool:
    return model_kind in {"trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"}


def _is_projected_student_model(model_kind: str) -> bool:
    return model_kind == "trial_principal_geom_projected_student"


def _branch_head_is_plastic_only(model_kind: str) -> bool:
    return model_kind in {"trial_principal_geom_plastic_branch_residual", "trial_principal_geom_projected_student"}


def _branch_names_for_model(model_kind: str) -> list[str]:
    if _branch_head_is_plastic_only(model_kind):
        return list(PLASTIC_BRANCH_NAMES)
    return list(BRANCH_NAMES)


def _is_principal_model(model_kind: str) -> bool:
    return model_kind in {"principal", "trial_principal_branch_residual", "trial_principal_geom_plastic_branch_residual"}


def _outputs_principal_stress(model_kind: str) -> bool:
    return model_kind in {
        "principal",
        "trial_principal_branch_residual",
        "trial_principal_geom_plastic_branch_residual",
        "trial_principal_geom_projected_student",
        "trial_abr_acn_f1",
        "trial_surface_acn_f1",
        "trial_surface_branch_structured_f1",
        "trial_surface_soft_atlas_f1_concat",
        "trial_surface_soft_atlas_f1_film",
    }


def _uses_raw_features(model_kind: str) -> bool:
    return model_kind in {"raw", "raw_branch"}


def _uses_trial_features(model_kind: str) -> bool:
    return model_kind in {
        "trial_raw",
        "trial_raw_branch",
        "trial_raw_residual",
        "trial_raw_branch_residual",
    }


def _uses_trial_principal_features(model_kind: str) -> bool:
    return model_kind in {"trial_principal_branch_residual", "trial_principal_geom_plastic_branch_residual"}


def _uses_surface_features(model_kind: str) -> bool:
    return _is_surface_model(model_kind)


def _uses_residual_target(model_kind: str) -> bool:
    return model_kind in {
        "trial_raw_residual",
        "trial_raw_branch_residual",
        "trial_principal_branch_residual",
        "trial_principal_geom_plastic_branch_residual",
    }


def _plastic_only_regression(model_kind: str) -> bool:
    return model_kind in {
        "trial_principal_branch_residual",
        "trial_principal_geom_plastic_branch_residual",
        "trial_principal_geom_projected_student",
    }


def _plastic_only_surface_regression(model_kind: str) -> bool:
    return _is_surface_model(model_kind)


def _coord_scales_from_abr(abr_nonneg: np.ndarray) -> dict[str, float]:
    abr = np.asarray(abr_nonneg, dtype=float)
    if abr.ndim != 2 or abr.shape[1] != 3:
        raise ValueError(f"Expected abr_nonneg shape (n, 3), got {abr.shape}.")
    return {
        "scale_a": float(max(np.quantile(abr[:, 0], 0.95), 1.0)),
        "scale_b": float(max(np.quantile(abr[:, 1], 0.95), 1.0)),
        "scale_r": float(max(np.quantile(abr[:, 2], 0.95), 1.0)),
    }


def _coord_scales_from_grho(
    grho: np.ndarray,
    material_reduced: np.ndarray,
) -> dict[str, float]:
    coords = np.asarray(grho, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected grho shape (n, 2), got {coords.shape}.")
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    g_tilde = ((1.0 + mat[:, 1]) * coords[:, 0]) / c_bar_safe
    return {
        "scale_g": float(max(np.quantile(np.maximum(g_tilde, 0.0), 0.95), 1.0)),
    }


def _coord_scales_from_branch_structured_grho(
    grho: np.ndarray,
    material_reduced: np.ndarray,
    branch_id: np.ndarray,
) -> dict[str, float]:
    coords = np.asarray(grho, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected grho shape (n, 2), got {coords.shape}.")
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    g_tilde = ((1.0 + mat[:, 1]) * coords[:, 0]) / c_bar_safe

    def _branch_scale(branch_name: str) -> float:
        mask = branch == BRANCH_TO_ID[branch_name]
        if not np.any(mask):
            return 1.0
        return float(max(np.quantile(np.maximum(g_tilde[mask], 0.0), 0.95), 1.0))

    return {
        "scale_g_smooth": _branch_scale("smooth"),
        "scale_g_left": _branch_scale("left_edge"),
        "scale_g_right": _branch_scale("right_edge"),
    }


def _transform_abr_target(
    abr_nonneg: np.ndarray,
    coordinate_scales: dict[str, float],
) -> np.ndarray:
    abr = np.asarray(abr_nonneg, dtype=float)
    scale_vec = np.asarray(
        [
            coordinate_scales["scale_a"],
            coordinate_scales["scale_b"],
            coordinate_scales["scale_r"],
        ],
        dtype=float,
    )
    return np.log1p(np.maximum(abr, 0.0) / scale_vec[None, :]).astype(np.float32)


def _softplus_inverse_np(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.maximum(arr, eps)
    return np.log(np.expm1(arr)).astype(np.float32)


def _transform_grho_target(
    grho: np.ndarray,
    coordinate_scales: dict[str, float],
    material_reduced: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(grho, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    scale_g = float(coordinate_scales["scale_g"])
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    g_tilde = ((1.0 + mat[:, 1]) * coords[:, 0]) / c_bar_safe
    g_target = np.log1p(np.maximum(g_tilde, 0.0) / scale_g).astype(np.float32)
    rho_target = np.arctanh(np.clip(coords[:, 1], -0.999999, 0.999999)).astype(np.float32)
    return np.column_stack([g_target, rho_target]).astype(np.float32)


def _transform_branch_structured_surface_target(
    grho: np.ndarray,
    coordinate_scales: dict[str, float],
    material_reduced: np.ndarray,
    branch_id: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(grho, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    g_tilde = ((1.0 + mat[:, 1]) * coords[:, 0]) / c_bar_safe
    target = np.zeros((coords.shape[0], 4), dtype=np.float32)

    smooth = branch == BRANCH_TO_ID["smooth"]
    left = branch == BRANCH_TO_ID["left_edge"]
    right = branch == BRANCH_TO_ID["right_edge"]

    if np.any(smooth):
        target[smooth, 0] = np.log1p(np.maximum(g_tilde[smooth], 0.0) / float(coordinate_scales["scale_g_smooth"])).astype(np.float32)
        target[smooth, 1] = np.arctanh(np.clip(coords[smooth, 1], -0.999999, 0.999999)).astype(np.float32)
    if np.any(left):
        target[left, 2] = np.log1p(np.maximum(g_tilde[left], 0.0) / float(coordinate_scales["scale_g_left"])).astype(np.float32)
    if np.any(right):
        target[right, 3] = np.log1p(np.maximum(g_tilde[right], 0.0) / float(coordinate_scales["scale_g_right"])).astype(np.float32)
    return target


def _soft_atlas_route_targets(
    branch_id: np.ndarray,
    *,
    m_yield: np.ndarray,
    m_smooth_left: np.ndarray,
    m_smooth_right: np.ndarray,
    m_left_apex: np.ndarray,
    m_right_apex: np.ndarray,
    temperature: float = 0.10,
) -> np.ndarray:
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    scores = np.zeros((branch.shape[0], 4), dtype=np.float64)
    smooth = branch == BRANCH_TO_ID["smooth"]
    left = branch == BRANCH_TO_ID["left_edge"]
    right = branch == BRANCH_TO_ID["right_edge"]
    apex = branch == BRANCH_TO_ID["apex"]

    def _normalize(local_scores: np.ndarray) -> np.ndarray:
        local_scores = np.asarray(local_scores, dtype=np.float64)
        local_scores_sum = np.sum(local_scores, axis=1, keepdims=True)
        return np.divide(local_scores, np.maximum(local_scores_sum, 1.0e-12))

    if np.any(smooth):
        idx = np.where(smooth)[0]
        local_scores = np.zeros((idx.size, 4), dtype=np.float64)
        local_scores[:, 0] = 1.0
        local_scores[:, 1] = np.exp(-np.abs(np.asarray(m_smooth_left, dtype=float)[idx]) / max(temperature, 1.0e-6))
        local_scores[:, 2] = np.exp(-np.abs(np.asarray(m_smooth_right, dtype=float)[idx]) / max(temperature, 1.0e-6))
        scores[idx] = _normalize(local_scores)
    if np.any(left):
        idx = np.where(left)[0]
        local_scores = np.zeros((idx.size, 4), dtype=np.float64)
        local_scores[:, 1] = 1.0
        smooth_scores = np.exp(-np.abs(np.asarray(m_smooth_left, dtype=float)[idx]) / max(temperature, 1.0e-6))
        apex_scores = np.exp(-np.abs(np.asarray(m_left_apex, dtype=float)[idx]) / max(temperature, 1.0e-6))
        prefer_smooth = smooth_scores >= apex_scores
        local_scores[:, 0] = np.where(prefer_smooth, smooth_scores, 0.0)
        local_scores[:, 3] = np.where(prefer_smooth, 0.0, apex_scores)
        scores[idx] = _normalize(local_scores)
    if np.any(right):
        idx = np.where(right)[0]
        local_scores = np.zeros((idx.size, 4), dtype=np.float64)
        local_scores[:, 2] = 1.0
        smooth_scores = np.exp(-np.abs(np.asarray(m_smooth_right, dtype=float)[idx]) / max(temperature, 1.0e-6))
        apex_scores = np.exp(-np.abs(np.asarray(m_right_apex, dtype=float)[idx]) / max(temperature, 1.0e-6))
        prefer_smooth = smooth_scores >= apex_scores
        local_scores[:, 0] = np.where(prefer_smooth, smooth_scores, 0.0)
        local_scores[:, 3] = np.where(prefer_smooth, 0.0, apex_scores)
        scores[idx] = _normalize(local_scores)
    if np.any(apex):
        idx = np.where(apex)[0]
        local_scores = np.zeros((idx.size, 4), dtype=np.float64)
        local_scores[:, 3] = 1.0
        local_scores[:, 1] = np.exp(-np.abs(np.asarray(m_left_apex, dtype=float)[idx]) / max(temperature, 1.0e-6))
        local_scores[:, 2] = np.exp(-np.abs(np.asarray(m_right_apex, dtype=float)[idx]) / max(temperature, 1.0e-6))
        scores[idx] = _normalize(local_scores)
    return scores.astype(np.float32)


def _soft_atlas_target_transform(
    stress_principal: np.ndarray,
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
    branch_id: np.ndarray,
) -> np.ndarray:
    true_p = np.asarray(stress_principal, dtype=float)
    trial_p = np.asarray(trial_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)

    a_true = true_p[:, 0] - true_p[:, 1]
    b_true = true_p[:, 1] - true_p[:, 2]
    g_true = a_true + b_true
    lambda_true = np.divide(a_true, g_true + 1.0e-12)
    lambda_true = np.clip(lambda_true, 1.0e-6, 1.0 - 1.0e-6)

    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    g_tr = a_tr + b_tr
    lambda_tr = np.clip(np.divide(a_tr, g_tr + 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)

    target = np.zeros((true_p.shape[0], 4), dtype=np.float32)
    smooth = branch == BRANCH_TO_ID["smooth"]
    left = branch == BRANCH_TO_ID["left_edge"]
    right = branch == BRANCH_TO_ID["right_edge"]

    if np.any(smooth):
        target[smooth, 0] = np.log(np.maximum(g_true[smooth], 1.0e-12) / np.maximum(g_tr[smooth], 1.0e-12)).astype(np.float32)
        target[smooth, 1] = (
            np.log(lambda_true[smooth]) - np.log1p(-lambda_true[smooth]) - (np.log(lambda_tr[smooth]) - np.log1p(-lambda_tr[smooth]))
        ).astype(np.float32)
    if np.any(left):
        target[left, 2] = np.log(np.maximum(b_true[left], 1.0e-12) / np.maximum(b_tr[left], 1.0e-12)).astype(np.float32)
    if np.any(right):
        target[right, 3] = np.log(np.maximum(a_true[right], 1.0e-12) / np.maximum(a_tr[right], 1.0e-12)).astype(np.float32)
    return target


def _mirror_soft_atlas_features(features: torch.Tensor) -> torch.Tensor:
    """Mirror packet4 input features left-to-right in the raw feature layout."""
    x = features.clone()
    if x.shape[1] < 17:
        return x
    # [0]=asinh(p), [1]=log1p(g), [2]=logit(lambda), [3]=asinh(f), [4]=rho
    # [5]=m_yield, [6]=m_smooth_left, [7]=m_smooth_right, [8]=m_left_apex,
    # [9]=m_right_apex, [10]=gap12, [11]=gap23, [12..16]=material_norm
    x[:, 2] = -x[:, 2]
    x[:, 4] = -x[:, 4]
    x[:, 6], x[:, 7] = x[:, 7].clone(), x[:, 6].clone()
    x[:, 8], x[:, 9] = x[:, 9].clone(), x[:, 8].clone()
    x[:, 10], x[:, 11] = x[:, 11].clone(), x[:, 10].clone()
    return x


def _soft_atlas_mirror_trial_principal(trial_principal: torch.Tensor) -> torch.Tensor:
    """Mirror the trial principal stress by swapping the two plastic gaps."""
    p = trial_principal.mean(dim=1, keepdim=True)
    a = trial_principal[:, 0:1] - trial_principal[:, 1:2]
    b = trial_principal[:, 1:2] - trial_principal[:, 2:3]
    a_m = b
    b_m = a
    tau1 = p + (2.0 * a_m + b_m) / 3.0
    tau2 = p + (-a_m + b_m) / 3.0
    tau3 = p - (a_m + 2.0 * b_m) / 3.0
    return torch.cat([tau1, tau2, tau3], dim=1)


def _soft_atlas_packet4_sample_weights(
    branch_id: np.ndarray,
    hard_mask: np.ndarray | None,
    plastic_mask: np.ndarray | None,
) -> np.ndarray:
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    if hard_mask is None:
        hard = np.zeros_like(branch, dtype=bool)
    else:
        hard = np.asarray(hard_mask, dtype=bool).reshape(-1)
    if plastic_mask is None:
        plastic = branch > 0
    else:
        plastic = np.asarray(plastic_mask, dtype=bool).reshape(-1)

    weights = np.zeros(branch.shape[0], dtype=np.float64)
    valid = plastic & np.isin(branch, list(BRANCH_TO_ID.values()))
    if not np.any(valid):
        return weights

    bucket_counts: dict[tuple[bool, int], int] = {}
    for hard_flag in (False, True):
        for branch_id_i in range(1, len(BRANCH_NAMES)):
            bucket = valid & (hard == hard_flag) & (branch == branch_id_i)
            if np.any(bucket):
                bucket_counts[(hard_flag, branch_id_i)] = int(np.sum(bucket))

    for (hard_flag, branch_id_i), count in bucket_counts.items():
        bucket = valid & (hard == hard_flag) & (branch == branch_id_i)
        if count > 0:
            weights[bucket] = 1.0 / float(count)
    return weights


def _prepare_model_inputs(
    model_kind: str,
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    feature_stats: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    strain_eng = np.asarray(strain_eng, dtype=np.float32)
    material_reduced = np.asarray(material_reduced, dtype=np.float32)
    trial_stress = compute_trial_stress(strain_eng, material_reduced)
    trial_principal = exact_trial_principal_from_strain(strain_eng, material_reduced)
    strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)

    if _is_acn_model(model_kind):
        if feature_stats is None:
            raise ValueError("ACN feature preparation requires feature_stats.")
        features = build_trial_abr_features_f1(trial_principal, material_reduced, feature_stats)
    elif _is_soft_atlas_surface_model(model_kind):
        if feature_stats is None:
            feature_stats = compute_trial_soft_atlas_feature_stats(trial_principal, material_reduced)
        features = build_trial_soft_atlas_features_f1(strain_principal, material_reduced, trial_principal, feature_stats)
    elif _is_surface_model(model_kind):
        if feature_stats is None:
            feature_stats = compute_trial_surface_feature_stats(trial_principal, material_reduced)
        features = build_trial_surface_features_f1(strain_principal, material_reduced, trial_principal, feature_stats)
    elif model_kind in {"trial_principal_geom_plastic_branch_residual", "trial_principal_geom_projected_student"}:
        features = build_trial_principal_geom_features(strain_principal, material_reduced, trial_principal)
    elif _uses_trial_principal_features(model_kind):
        features = build_trial_principal_features(strain_principal, material_reduced, trial_principal)
    elif _is_principal_model(model_kind):
        features = build_principal_features(strain_principal, material_reduced)
    elif _uses_raw_features(model_kind):
        features = build_raw_features(strain_eng, material_reduced)
    elif _uses_trial_features(model_kind):
        features = build_trial_features(strain_eng, material_reduced)
    else:
        raise ValueError(f"Unsupported model kind {model_kind!r}.")

    return {
        "features": features.astype(np.float32),
        "strain_principal": strain_principal.astype(np.float32),
        "eigvecs": eigvecs.astype(np.float32),
        "trial_stress": trial_stress.astype(np.float32),
        "trial_principal": trial_principal.astype(np.float32),
    }


def _branch_targets_for_model(model_kind: str, branch_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _branch_head_is_plastic_only(model_kind) or _is_branch_structured_surface_model(model_kind):
        valid = branch_true > 0
        return valid, branch_true[valid] - 1
    valid = branch_true >= 0
    return valid, branch_true[valid]


def _load_split_for_training(
    dataset_path: str,
    split: str,
    model_kind: str,
    *,
    include_tangent: bool = False,
    feature_stats: dict[str, Any] | None = None,
    coordinate_scales: dict[str, float] | None = None,
    projected_student_high_disp_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    available = _dataset_keys(dataset_path)
    keys = ["strain_eng", "stress", "material_reduced"]
    optional = [
        "stress_principal",
        "branch_id",
        "eigvecs",
        "hard_mask",
        "plastic_mask",
        "teacher_stress_principal",
        "teacher_provisional_stress_principal",
        "teacher_projected_stress_principal",
        "teacher_projection_delta_principal",
        "teacher_projection_candidate_id",
        "teacher_projection_disp_norm",
        "ds_valid_mask",
        "sampling_weight",
        "any_boundary_mask",
        "teacher_gap_q90_mask",
        "teacher_gap_q95_mask",
        "delta_gap_q90_mask",
        "edge_apex_adjacent_mask",
        "near_yield_mask",
        "near_smooth_left_mask",
        "near_smooth_right_mask",
        "near_left_apex_mask",
        "near_right_apex_mask",
    ]
    if include_tangent and "tangent" in available:
        optional.append("tangent")
    elif include_tangent and "DS" in available:
        optional.append("DS")
    if _is_acn_model(model_kind):
        for key in ("feature_f1", "abr_raw", "abr_nonneg", "trial_principal", "trial_stress"):
            if key in available:
                optional.append(key)
    if _is_surface_model(model_kind):
        for key in ("surface_feature_f1", "soft_atlas_feature_f1", "soft_atlas_route_target", "trial_principal", "trial_stress"):
            if key in available:
                optional.append(key)
    if model_kind in {"trial_principal_geom_plastic_branch_residual", "trial_principal_geom_projected_student"}:
        for key in ("trial_principal_geom_feature_f1", "trial_principal", "trial_stress"):
            if key in available:
                optional.append(key)
    keys.extend([key for key in optional if key in available])
    arrays = load_arrays(dataset_path, keys, split=split)

    stress_principal = arrays.get("stress_principal")
    if stress_principal is None:
        stress_principal = _principal_stress_from_stress(arrays["stress"])
    branch_id = arrays.get("branch_id")
    if branch_id is None:
        branch_id = np.full(arrays["strain_eng"].shape[0], -1, dtype=np.int64)
    else:
        branch_id = branch_id.astype(np.int64)
    hard_mask = arrays.get("hard_mask")
    if hard_mask is not None:
        hard_mask = hard_mask.astype(bool)
    plastic_mask = arrays.get("plastic_mask")
    if plastic_mask is not None:
        plastic_mask = plastic_mask.astype(bool)
    teacher_provisional_stress_principal = arrays.get("teacher_provisional_stress_principal")
    if teacher_provisional_stress_principal is None:
        teacher_provisional_stress_principal = arrays.get("teacher_stress_principal")
    if teacher_provisional_stress_principal is None:
        teacher_provisional_stress_principal = stress_principal.astype(np.float32)
    else:
        teacher_provisional_stress_principal = teacher_provisional_stress_principal.astype(np.float32)
    teacher_projected_stress_principal = arrays.get("teacher_projected_stress_principal")
    if teacher_projected_stress_principal is None:
        teacher_projected_stress_principal = np.full_like(stress_principal, np.nan, dtype=np.float32)
    else:
        teacher_projected_stress_principal = teacher_projected_stress_principal.astype(np.float32)
    teacher_projection_delta_principal = arrays.get("teacher_projection_delta_principal")
    if teacher_projection_delta_principal is None:
        teacher_projection_delta_principal = np.full_like(stress_principal, np.nan, dtype=np.float32)
    else:
        teacher_projection_delta_principal = teacher_projection_delta_principal.astype(np.float32)
    teacher_projection_candidate_id = arrays.get("teacher_projection_candidate_id")
    if teacher_projection_candidate_id is None:
        teacher_projection_candidate_id = np.full(arrays["strain_eng"].shape[0], -1, dtype=np.int64)
    else:
        teacher_projection_candidate_id = teacher_projection_candidate_id.astype(np.int64)
    teacher_projection_disp_norm = arrays.get("teacher_projection_disp_norm")
    if teacher_projection_disp_norm is None:
        teacher_projection_disp_norm = np.full(arrays["strain_eng"].shape[0], np.nan, dtype=np.float32)
    else:
        teacher_projection_disp_norm = teacher_projection_disp_norm.astype(np.float32)
    near_yield_mask = arrays.get("near_yield_mask")
    near_smooth_left_mask = arrays.get("near_smooth_left_mask")
    near_smooth_right_mask = arrays.get("near_smooth_right_mask")
    near_left_apex_mask = arrays.get("near_left_apex_mask")
    near_right_apex_mask = arrays.get("near_right_apex_mask")
    any_boundary_mask = arrays.get("any_boundary_mask")
    if near_yield_mask is None:
        near_yield_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        near_yield_mask = near_yield_mask.astype(bool)
    if near_smooth_left_mask is None:
        near_smooth_left_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        near_smooth_left_mask = near_smooth_left_mask.astype(bool)
    if near_smooth_right_mask is None:
        near_smooth_right_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        near_smooth_right_mask = near_smooth_right_mask.astype(bool)
    if near_left_apex_mask is None:
        near_left_apex_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        near_left_apex_mask = near_left_apex_mask.astype(bool)
    if near_right_apex_mask is None:
        near_right_apex_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        near_right_apex_mask = near_right_apex_mask.astype(bool)
    if any_boundary_mask is None:
        any_boundary_mask = (
            near_yield_mask
            | near_smooth_left_mask
            | near_smooth_right_mask
            | near_left_apex_mask
            | near_right_apex_mask
        )
    else:
        any_boundary_mask = any_boundary_mask.astype(bool)
    teacher_gap_q90_mask = arrays.get("teacher_gap_q90_mask")
    if teacher_gap_q90_mask is None:
        teacher_gap_q90_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        teacher_gap_q90_mask = teacher_gap_q90_mask.astype(bool)
    teacher_gap_q95_mask = arrays.get("teacher_gap_q95_mask")
    if teacher_gap_q95_mask is None:
        teacher_gap_q95_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        teacher_gap_q95_mask = teacher_gap_q95_mask.astype(bool)
    delta_gap_q90_mask = arrays.get("delta_gap_q90_mask")
    if delta_gap_q90_mask is None:
        delta_gap_q90_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        delta_gap_q90_mask = delta_gap_q90_mask.astype(bool)
    edge_apex_adjacent_mask = arrays.get("edge_apex_adjacent_mask")
    if edge_apex_adjacent_mask is None:
        edge_apex_adjacent_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        edge_apex_adjacent_mask = edge_apex_adjacent_mask.astype(bool)
    ds_valid_mask = arrays.get("ds_valid_mask")
    if ds_valid_mask is not None:
        ds_valid_mask = ds_valid_mask.astype(bool)
    sampling_weight = arrays.get("sampling_weight")
    if sampling_weight is not None:
        sampling_weight = sampling_weight.astype(np.float32)
    attrs = _dataset_attrs(dataset_path)
    high_disp_threshold = projected_student_high_disp_threshold
    if high_disp_threshold is None:
        raw_threshold = attrs.get("high_disp_focus_threshold")
        if raw_threshold is None:
            raw_threshold = attrs.get("teacher_projection_disp_p90_train")
        if raw_threshold is not None:
            high_disp_threshold = float(raw_threshold)
    if high_disp_threshold is None:
        high_disp_mask = np.zeros(arrays["strain_eng"].shape[0], dtype=bool)
    else:
        high_disp_mask = np.isfinite(teacher_projection_disp_norm) & (teacher_projection_disp_norm >= float(high_disp_threshold))

    prepared = None
    if _is_acn_model(model_kind):
        if feature_stats is None:
            attrs = _dataset_attrs(dataset_path)
            feature_stats = _json_attr_dict(attrs, "acn_feature_stats_json")
        if coordinate_scales is None:
            attrs = _dataset_attrs(dataset_path)
            coordinate_scales = _json_attr_dict(attrs, "acn_coordinate_scales_json")
        if coordinate_scales is None:
            raise ValueError("ACN split loading requires coordinate scales.")
        if "feature_f1" in arrays:
            features = arrays["feature_f1"].astype(np.float32)
        else:
            prepared = _prepare_model_inputs(
                model_kind,
                arrays["strain_eng"],
                arrays["material_reduced"],
                feature_stats=feature_stats,
            )
            features = prepared["features"].astype(np.float32)
        abr_raw = arrays.get("abr_raw")
        if abr_raw is None:
            encoded = encode_principal_to_abr(stress_principal, c_bar=arrays["material_reduced"][:, 0], sin_phi=arrays["material_reduced"][:, 1])
            abr_raw = encoded["abr_raw"].astype(np.float32)
            abr_nonneg = encoded["abr_nonneg"].astype(np.float32)
        else:
            abr_raw = abr_raw.astype(np.float32)
            abr_nonneg = arrays["abr_nonneg"].astype(np.float32) if "abr_nonneg" in arrays else abr_raw.copy()
            abr_nonneg[:, 2] = np.maximum(abr_nonneg[:, 2], 0.0)
        if coordinate_scales is None:
            coordinate_scales = _coord_scales_from_abr(abr_nonneg)
        target = _transform_abr_target(abr_nonneg, coordinate_scales)
    elif _is_soft_atlas_surface_model(model_kind):
        if feature_stats is None:
            attrs = _dataset_attrs(dataset_path)
            feature_stats = _json_attr_dict(attrs, "soft_atlas_feature_stats_json")
        if "soft_atlas_feature_f1" in arrays:
            features = arrays["soft_atlas_feature_f1"].astype(np.float32)
        else:
            prepared = _prepare_model_inputs(
                model_kind,
                arrays["strain_eng"],
                arrays["material_reduced"],
                feature_stats=feature_stats,
            )
            features = prepared["features"].astype(np.float32)
        encoded = encode_principal_to_grho(
            stress_principal,
            c_bar=arrays["material_reduced"][:, 0],
            sin_phi=arrays["material_reduced"][:, 1],
        )
        grho_true = encoded["grho"].astype(np.float32)
        if prepared is None:
            prepared = _prepare_model_inputs(
                model_kind,
                arrays["strain_eng"],
                arrays["material_reduced"],
                feature_stats=feature_stats,
            )
        route_target = arrays.get("soft_atlas_route_target")
        if route_target is None:
            geom = compute_branch_geometry_principal(
                prepared["strain_principal"],
                c_bar=arrays["material_reduced"][:, 0],
                sin_phi=arrays["material_reduced"][:, 1],
                shear=arrays["material_reduced"][:, 2],
                bulk=arrays["material_reduced"][:, 3],
                lame=arrays["material_reduced"][:, 4],
            )
            route_target = _soft_atlas_route_targets(
                branch_id,
                m_yield=geom.m_yield,
                m_smooth_left=geom.m_smooth_left,
                m_smooth_right=geom.m_smooth_right,
                m_left_apex=geom.m_left_apex,
                m_right_apex=geom.m_right_apex,
            )
        else:
            route_target = route_target.astype(np.float32)
        target = _soft_atlas_target_transform(
            stress_principal,
            prepared["trial_principal"],
            arrays["material_reduced"],
            branch_id,
        )
    elif _is_surface_model(model_kind):
        if feature_stats is None:
            attrs = _dataset_attrs(dataset_path)
            feature_stats = _json_attr_dict(attrs, "surface_feature_stats_json")
            if feature_stats is None:
                feature_stats = _json_attr_dict(attrs, "acn_feature_stats_json")
        if coordinate_scales is None:
            attrs = _dataset_attrs(dataset_path)
            coordinate_scales = _json_attr_dict(attrs, "surface_coordinate_scales_json")
        encoded = encode_principal_to_grho(
            stress_principal,
            c_bar=arrays["material_reduced"][:, 0],
            sin_phi=arrays["material_reduced"][:, 1],
        )
        grho_true = encoded["grho"].astype(np.float32)
        if _is_branch_structured_surface_model(model_kind):
            required = {"scale_g_smooth", "scale_g_left", "scale_g_right"}
            if coordinate_scales is None or not required.issubset(set(coordinate_scales)):
                coordinate_scales = _coord_scales_from_branch_structured_grho(
                    grho_true,
                    arrays["material_reduced"],
                    branch_id,
                )
        elif coordinate_scales is None:
            coordinate_scales = _coord_scales_from_grho(grho_true, arrays["material_reduced"])
        if "surface_feature_f1" in arrays:
            features = arrays["surface_feature_f1"].astype(np.float32)
        else:
            prepared = _prepare_model_inputs(
                model_kind,
                arrays["strain_eng"],
                arrays["material_reduced"],
                feature_stats=feature_stats,
            )
            features = prepared["features"].astype(np.float32)
        if _is_branch_structured_surface_model(model_kind):
            target = _transform_branch_structured_surface_target(
                grho_true,
                coordinate_scales,
                arrays["material_reduced"],
                branch_id,
            )
        else:
            target = _transform_grho_target(grho_true, coordinate_scales, arrays["material_reduced"])
    else:
        prepared = _prepare_model_inputs(model_kind, arrays["strain_eng"], arrays["material_reduced"])
        if model_kind in {"trial_principal_geom_plastic_branch_residual", "trial_principal_geom_projected_student"} and "trial_principal_geom_feature_f1" in arrays:
            features = arrays["trial_principal_geom_feature_f1"].astype(np.float32)
        else:
            features = prepared["features"].astype(np.float32)

    if _is_projected_student_model(model_kind):
        target = (stress_principal.astype(np.float32) - prepared["trial_principal"]).astype(np.float32)
    elif _uses_trial_principal_features(model_kind):
        target = (stress_principal.astype(np.float32) - prepared["trial_principal"]).astype(np.float32)
    elif _is_principal_model(model_kind):
        target = stress_principal.astype(np.float32)
    elif _uses_residual_target(model_kind):
        target = (arrays["stress"] - prepared["trial_stress"]).astype(np.float32)
    elif not (_is_acn_model(model_kind) or _is_surface_model(model_kind)):
        target = arrays["stress"].astype(np.float32)

    tangent = arrays.get("tangent")
    if tangent is None:
        tangent = arrays.get("DS")
    tangent_out = None
    if tangent is not None:
        tangent_out = tangent.astype(np.float32)
        if tangent_out.ndim == 2 and tangent_out.shape[1] == 36:
            tangent_out = tangent_out.reshape(-1, 6, 6)

    return {
        "features": features.astype(np.float32),
        "target": target.astype(np.float32),
        "stress_true": arrays["stress"].astype(np.float32),
        "stress_principal_true": stress_principal.astype(np.float32),
        "branch_id": branch_id,
        "eigvecs": (prepared["eigvecs"] if prepared is not None else arrays["eigvecs"]).astype(np.float32),
        "trial_stress": (prepared["trial_stress"] if prepared is not None else arrays["trial_stress"]).astype(np.float32),
        "trial_principal": (prepared["trial_principal"] if prepared is not None else arrays["trial_principal"]).astype(np.float32),
        "soft_atlas_route_target": (route_target.astype(np.float32) if _is_soft_atlas_surface_model(model_kind) else np.full((arrays["stress"].shape[0], 4), np.nan, dtype=np.float32)),
        "abr_true_raw": (abr_raw.astype(np.float32) if _is_acn_model(model_kind) else np.full((arrays["stress"].shape[0], 3), np.nan, dtype=np.float32)),
        "abr_true_nonneg": (abr_nonneg.astype(np.float32) if _is_acn_model(model_kind) else np.full((arrays["stress"].shape[0], 3), np.nan, dtype=np.float32)),
        "grho_true": (grho_true.astype(np.float32) if _is_surface_model(model_kind) else np.full((arrays["stress"].shape[0], 2), np.nan, dtype=np.float32)),
        "strain_eng": arrays["strain_eng"].astype(np.float32),
        "material_reduced": arrays["material_reduced"].astype(np.float32),
        "teacher_provisional_stress_principal": teacher_provisional_stress_principal.astype(np.float32),
        "teacher_projected_stress_principal": teacher_projected_stress_principal.astype(np.float32),
        "teacher_projection_delta_principal": teacher_projection_delta_principal.astype(np.float32),
        "teacher_projection_candidate_id": teacher_projection_candidate_id.astype(np.int64),
        "teacher_projection_disp_norm": teacher_projection_disp_norm.astype(np.float32),
        "near_yield_mask": near_yield_mask.astype(bool),
        "near_smooth_left_mask": near_smooth_left_mask.astype(bool),
        "near_smooth_right_mask": near_smooth_right_mask.astype(bool),
        "near_left_apex_mask": near_left_apex_mask.astype(bool),
        "near_right_apex_mask": near_right_apex_mask.astype(bool),
        "any_boundary_mask": any_boundary_mask.astype(bool),
        "teacher_gap_q90_mask": teacher_gap_q90_mask.astype(bool),
        "teacher_gap_q95_mask": teacher_gap_q95_mask.astype(bool),
        "delta_gap_q90_mask": delta_gap_q90_mask.astype(bool),
        "edge_apex_adjacent_mask": edge_apex_adjacent_mask.astype(bool),
        "high_disp_mask": high_disp_mask.astype(bool),
        "hard_mask": (hard_mask.astype(bool) if hard_mask is not None else np.zeros(arrays["strain_eng"].shape[0], dtype=bool)),
        "plastic_mask": (
            plastic_mask.astype(bool) if plastic_mask is not None else (branch_id > 0)
        ),
        "ds_valid_mask": (ds_valid_mask.astype(bool) if ds_valid_mask is not None else np.zeros(arrays["strain_eng"].shape[0], dtype=bool)),
        "sampling_weight": sampling_weight,
        "tangent_true": tangent_out,
    }


def _build_tensor_dataset(
    split_arrays: dict[str, np.ndarray],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
) -> TensorDataset:
    x = torch.from_numpy(x_scaler.transform(split_arrays["features"]))
    y = torch.from_numpy(y_scaler.transform(split_arrays["target"]))
    branch = torch.from_numpy(split_arrays["branch_id"])
    stress_true = torch.from_numpy(split_arrays["stress_true"])
    stress_principal_true = torch.from_numpy(split_arrays["stress_principal_true"])
    eigvecs = torch.from_numpy(split_arrays["eigvecs"])
    trial_stress = torch.from_numpy(split_arrays["trial_stress"])
    trial_principal = torch.from_numpy(split_arrays["trial_principal"])
    abr_true_raw = torch.from_numpy(split_arrays["abr_true_raw"])
    abr_true_nonneg = torch.from_numpy(split_arrays["abr_true_nonneg"])
    grho_true = torch.from_numpy(split_arrays["grho_true"])
    soft_atlas_route_target = torch.from_numpy(split_arrays["soft_atlas_route_target"])
    strain_eng = torch.from_numpy(split_arrays["strain_eng"])
    material_reduced = torch.from_numpy(split_arrays["material_reduced"])
    teacher_provisional_stress_principal = torch.from_numpy(split_arrays["teacher_provisional_stress_principal"])
    teacher_projected_stress_principal = torch.from_numpy(split_arrays["teacher_projected_stress_principal"])
    teacher_projection_delta_principal = torch.from_numpy(split_arrays["teacher_projection_delta_principal"])
    hard_mask = torch.from_numpy(split_arrays["hard_mask"].astype(np.int8))
    any_boundary_mask = torch.from_numpy(split_arrays["any_boundary_mask"].astype(np.int8))
    teacher_gap_q90_mask = torch.from_numpy(split_arrays["teacher_gap_q90_mask"].astype(np.int8))
    teacher_gap_q95_mask = torch.from_numpy(split_arrays["teacher_gap_q95_mask"].astype(np.int8))
    delta_gap_q90_mask = torch.from_numpy(split_arrays["delta_gap_q90_mask"].astype(np.int8))
    edge_apex_adjacent_mask = torch.from_numpy(split_arrays["edge_apex_adjacent_mask"].astype(np.int8))
    high_disp_mask = torch.from_numpy(split_arrays["high_disp_mask"].astype(np.int8))
    teacher_projection_candidate_id = torch.from_numpy(split_arrays["teacher_projection_candidate_id"])
    tangent_true = split_arrays.get("tangent_true")
    if tangent_true is None:
        tangent = torch.full((x.shape[0], 6, 6), float("nan"), dtype=torch.float32)
    else:
        tangent = torch.from_numpy(tangent_true)
    return TensorDataset(
        x,
        y,
        branch,
        stress_true,
        stress_principal_true,
        eigvecs,
        trial_stress,
        trial_principal,
        abr_true_raw,
        abr_true_nonneg,
        grho_true,
        soft_atlas_route_target,
        strain_eng,
        material_reduced,
        teacher_provisional_stress_principal,
        teacher_projected_stress_principal,
        teacher_projection_delta_principal,
        hard_mask,
        any_boundary_mask,
        teacher_gap_q90_mask,
        teacher_gap_q95_mask,
        delta_gap_q90_mask,
        edge_apex_adjacent_mask,
        high_disp_mask,
        teacher_projection_candidate_id,
        tangent,
    )


def _decode_projected_student_outputs(
    pred_norm: torch.Tensor,
    y_scaler: Standardizer,
    material_reduced: torch.Tensor,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    *,
    projection_mode: str,
    projection_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
        y_scaler.mean, device=pred_norm.device
    )
    provisional_principal = pred + trial_principal
    provisional_principal = torch.sort(provisional_principal, dim=-1, descending=True).values
    projected_principal = provisional_principal
    plastic_mask = (
        (1.0 + material_reduced[:, 1]) * trial_principal[:, 0]
        - (1.0 - material_reduced[:, 1]) * trial_principal[:, 2]
        - material_reduced[:, 0]
    ) > 0.0
    if torch.any(plastic_mask):
        projected_principal = projected_principal.clone()
        projected_principal[plastic_mask] = project_mc_principal_torch(
            provisional_principal[plastic_mask],
            c_bar=material_reduced[plastic_mask, 0],
            sin_phi=material_reduced[plastic_mask, 1],
            mode=projection_mode,
            tau=projection_tau,
        )
    elastic_mask = ~plastic_mask
    if torch.any(elastic_mask):
        projected_principal = projected_principal.clone()
        projected_principal[elastic_mask] = trial_principal[elastic_mask]
    stress = stress_voigt_from_principal_torch(projected_principal, eigvecs)
    if torch.any(elastic_mask):
        stress = stress.clone()
        stress[elastic_mask] = trial_stress[elastic_mask]
    return stress, projected_principal, provisional_principal


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1.0e-12)


def _normalize_weight_map(weight_map: Mapping[int, float] | None) -> dict[int, float]:
    if not weight_map:
        return {}
    normalized: dict[int, float] = {}
    for key, value in weight_map.items():
        normalized[int(key)] = float(value)
    return normalized


def _projected_student_row_weights(
    *,
    hard_mask: torch.Tensor,
    any_boundary_mask: torch.Tensor,
    teacher_gap_q90_mask: torch.Tensor,
    teacher_gap_q95_mask: torch.Tensor,
    delta_gap_q90_mask: torch.Tensor,
    edge_apex_adjacent_mask: torch.Tensor,
    high_disp_mask: torch.Tensor,
    teacher_projection_candidate_id: torch.Tensor,
    branch_true: torch.Tensor,
    hard_loss_multiplier: float,
    any_boundary_loss_multiplier: float,
    high_disp_loss_multiplier: float,
    teacher_gap_q90_loss_multiplier: float,
    teacher_gap_q95_loss_multiplier: float,
    delta_gap_q90_loss_multiplier: float,
    edge_apex_loss_multiplier: float,
    candidate_loss_weights: Mapping[int, float] | None,
    branch_loss_weights: Mapping[int, float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.ones(hard_mask.shape[0], dtype=torch.float32, device=hard_mask.device)
    focus_mask = torch.zeros(hard_mask.shape[0], dtype=torch.bool, device=hard_mask.device)

    if hard_loss_multiplier != 1.0:
        hard_local = hard_mask > 0
        weights = weights * torch.where(hard_local, weights.new_tensor(float(hard_loss_multiplier)), weights.new_tensor(1.0))
        focus_mask |= hard_local
    if any_boundary_loss_multiplier != 1.0:
        boundary_local = any_boundary_mask > 0
        weights = weights * torch.where(boundary_local, weights.new_tensor(float(any_boundary_loss_multiplier)), weights.new_tensor(1.0))
        focus_mask |= boundary_local
    if teacher_gap_q90_loss_multiplier != 1.0:
        teacher_gap_local = teacher_gap_q90_mask > 0
        weights = weights * torch.where(teacher_gap_local, weights.new_tensor(float(teacher_gap_q90_loss_multiplier)), weights.new_tensor(1.0))
        focus_mask |= teacher_gap_local
    if teacher_gap_q95_loss_multiplier != 1.0:
        teacher_gap_extreme_local = teacher_gap_q95_mask > 0
        weights = weights * torch.where(
            teacher_gap_extreme_local,
            weights.new_tensor(float(teacher_gap_q95_loss_multiplier)),
            weights.new_tensor(1.0),
        )
        focus_mask |= teacher_gap_extreme_local
    if delta_gap_q90_loss_multiplier != 1.0:
        delta_gap_local = delta_gap_q90_mask > 0
        weights = weights * torch.where(delta_gap_local, weights.new_tensor(float(delta_gap_q90_loss_multiplier)), weights.new_tensor(1.0))
        focus_mask |= delta_gap_local
    if edge_apex_loss_multiplier != 1.0:
        edge_apex_local = edge_apex_adjacent_mask > 0
        weights = weights * torch.where(edge_apex_local, weights.new_tensor(float(edge_apex_loss_multiplier)), weights.new_tensor(1.0))
        focus_mask |= edge_apex_local
    if high_disp_loss_multiplier != 1.0:
        high_disp_local = high_disp_mask > 0
        weights = weights * torch.where(high_disp_local, weights.new_tensor(float(high_disp_loss_multiplier)), weights.new_tensor(1.0))
        focus_mask |= high_disp_local

    for bucket_id, multiplier in _normalize_weight_map(candidate_loss_weights).items():
        if multiplier == 1.0:
            continue
        local = teacher_projection_candidate_id == int(bucket_id)
        weights = weights * torch.where(local, weights.new_tensor(float(multiplier)), weights.new_tensor(1.0))
        focus_mask |= local

    for bucket_id, multiplier in _normalize_weight_map(branch_loss_weights).items():
        if multiplier == 1.0:
            continue
        local = branch_true == int(bucket_id)
        weights = weights * torch.where(local, weights.new_tensor(float(multiplier)), weights.new_tensor(1.0))
        focus_mask |= local

    return weights, focus_mask


def _decode_acn_outputs(
    pred_norm: torch.Tensor,
    material_reduced: torch.Tensor,
    eigvecs: torch.Tensor,
    coordinate_scales: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_vec = pred_norm.new_tensor(
        [
            coordinate_scales["scale_a"],
            coordinate_scales["scale_b"],
            coordinate_scales["scale_r"],
        ]
    )
    abr_pos = torch.nn.functional.softplus(pred_norm) * scale_vec.unsqueeze(0)
    stress_principal = decode_abr_to_principal_torch(
        abr_pos,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
    )
    stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
    t_abr = torch.log1p(abr_pos / scale_vec.unsqueeze(0))
    return stress, stress_principal, abr_pos, t_abr


def _branch_structured_surface_head_outputs(
    pred_norm: torch.Tensor,
    material_reduced: torch.Tensor,
    coordinate_scales: dict[str, float],
) -> dict[str, torch.Tensor]:
    c_bar_safe = torch.clamp(material_reduced[:, 0:1], min=1.0e-6)
    denom = torch.clamp(1.0 + material_reduced[:, 1:2], min=1.0e-6)

    scale_smooth = pred_norm.new_tensor(float(coordinate_scales["scale_g_smooth"]))
    scale_left = pred_norm.new_tensor(float(coordinate_scales["scale_g_left"]))
    scale_right = pred_norm.new_tensor(float(coordinate_scales["scale_g_right"]))

    smooth_g_tilde = torch.nn.functional.softplus(pred_norm[:, 0:1]) * scale_smooth
    smooth_rho = torch.tanh(pred_norm[:, 1:2])
    left_g_tilde = torch.nn.functional.softplus(pred_norm[:, 2:3]) * scale_left
    right_g_tilde = torch.nn.functional.softplus(pred_norm[:, 3:4]) * scale_right

    smooth_g = smooth_g_tilde * c_bar_safe / denom
    left_g = left_g_tilde * c_bar_safe / denom
    right_g = right_g_tilde * c_bar_safe / denom

    return {
        "smooth_grho": torch.cat([smooth_g, smooth_rho], dim=1),
        "left_grho": torch.cat([left_g, -torch.ones_like(left_g)], dim=1),
        "right_grho": torch.cat([right_g, torch.ones_like(right_g)], dim=1),
        "apex_grho": torch.zeros((pred_norm.shape[0], 2), dtype=pred_norm.dtype, device=pred_norm.device),
        "t_pred": torch.cat(
            [
                torch.log1p(smooth_g_tilde / scale_smooth),
                torch.atanh(torch.clamp(smooth_rho, -0.999999, 0.999999)),
                torch.log1p(left_g_tilde / scale_left),
                torch.log1p(right_g_tilde / scale_right),
            ],
            dim=1,
        ),
    }


def _decode_surface_outputs(
    pred_norm: torch.Tensor,
    material_reduced: torch.Tensor,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    coordinate_scales: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_g = pred_norm.new_tensor(float(coordinate_scales["scale_g"]))
    g_tilde = torch.nn.functional.softplus(pred_norm[:, :1]) * scale_g
    rho = torch.tanh(pred_norm[:, 1:2])
    c_bar_safe = torch.clamp(material_reduced[:, 0:1], min=1.0e-6)
    g = g_tilde * c_bar_safe / torch.clamp(1.0 + material_reduced[:, 1:2], min=1.0e-6)
    grho = torch.cat([g, rho], dim=1)
    stress_principal = decode_grho_to_principal_torch(
        grho,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
    )
    trial_yield = (1.0 + material_reduced[:, 1]) * trial_principal[:, 0] - (1.0 - material_reduced[:, 1]) * trial_principal[:, 2] - material_reduced[:, 0]
    elastic_mask = trial_yield <= 0.0
    if torch.any(elastic_mask):
        stress_principal = stress_principal.clone()
        stress_principal[elastic_mask] = trial_principal[elastic_mask]
    stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
    if torch.any(elastic_mask):
        stress = stress.clone()
        stress[elastic_mask] = trial_stress[elastic_mask]
    t_grho = torch.cat(
        [
            torch.log1p(g_tilde / scale_g),
            torch.atanh(torch.clamp(rho, -0.999999, 0.999999)),
        ],
        dim=1,
    )
    return stress, stress_principal, grho, t_grho


def _decode_branch_structured_surface_outputs(
    pred_norm: torch.Tensor,
    branch_route: torch.Tensor,
    material_reduced: torch.Tensor,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    coordinate_scales: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    head_outputs = _branch_structured_surface_head_outputs(pred_norm, material_reduced, coordinate_scales)
    route = branch_route.reshape(-1).to(device=pred_norm.device, dtype=torch.long)
    route_decode = route.clone()
    route_decode[route_decode <= 0] = BRANCH_TO_ID["smooth"]

    grho = head_outputs["smooth_grho"]
    left = route_decode == BRANCH_TO_ID["left_edge"]
    right = route_decode == BRANCH_TO_ID["right_edge"]
    apex = route_decode == BRANCH_TO_ID["apex"]
    grho = torch.where(left.unsqueeze(1), head_outputs["left_grho"], grho)
    grho = torch.where(right.unsqueeze(1), head_outputs["right_grho"], grho)
    grho = torch.where(apex.unsqueeze(1), head_outputs["apex_grho"], grho)

    stress_principal = decode_branch_specialized_grho_to_principal_torch(
        grho,
        route_decode,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
    )
    trial_yield = (1.0 + material_reduced[:, 1]) * trial_principal[:, 0] - (1.0 - material_reduced[:, 1]) * trial_principal[:, 2] - material_reduced[:, 0]
    elastic_mask = trial_yield <= 0.0
    if torch.any(elastic_mask):
        stress_principal = stress_principal.clone()
        stress_principal[elastic_mask] = trial_principal[elastic_mask]
    stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
    if torch.any(elastic_mask):
        stress = stress.clone()
        stress[elastic_mask] = trial_stress[elastic_mask]
    return stress, stress_principal, grho, head_outputs["t_pred"], route_decode


def _decode_soft_atlas_surface_outputs(
    pred_raw: torch.Tensor,
    route_logits: torch.Tensor,
    material_reduced: torch.Tensor,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    *,
    route_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    route_temp = max(float(route_temperature), 1.0e-6)
    route_probs = torch.softmax(route_logits / route_temp, dim=1)
    g_tr = trial_principal[:, 0] - trial_principal[:, 2]
    b_tr = trial_principal[:, 1] - trial_principal[:, 2]
    a_tr = trial_principal[:, 0] - trial_principal[:, 1]
    lambda_tr = torch.clamp(a_tr / torch.clamp(g_tr, min=1.0e-12), 1.0e-6, 1.0 - 1.0e-6)

    dlog_kappa_g = pred_raw[:, 0:1]
    d_lambda = pred_raw[:, 1:2]
    dlog_kappa_l = pred_raw[:, 2:3]
    dlog_kappa_r = pred_raw[:, 3:4]

    g_s = torch.clamp(g_tr[:, None], min=1.0e-12) * torch.exp(dlog_kappa_g)
    lambda_s = torch.sigmoid(torch.logit(lambda_tr)[:, None] + d_lambda)
    b_l = torch.clamp(b_tr[:, None], min=1.0e-12) * torch.exp(dlog_kappa_l)
    a_r = torch.clamp(a_tr[:, None], min=1.0e-12) * torch.exp(dlog_kappa_r)

    a = route_probs[:, 0:1] * (lambda_s * g_s) + route_probs[:, 2:3] * a_r
    b = route_probs[:, 0:1] * ((1.0 - lambda_s) * g_s) + route_probs[:, 1:2] * b_l
    r = torch.zeros_like(a)
    abr = torch.cat([a, b, r], dim=1)

    stress_principal = decode_abr_to_principal_torch(
        abr,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
    )
    trial_yield = (1.0 + material_reduced[:, 1]) * trial_principal[:, 0] - (1.0 - material_reduced[:, 1]) * trial_principal[:, 2] - material_reduced[:, 0]
    elastic_mask = trial_yield <= 0.0
    if torch.any(elastic_mask):
        stress_principal = stress_principal.clone()
        stress_principal[elastic_mask] = trial_principal[elastic_mask]
    stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
    if torch.any(elastic_mask):
        stress = stress.clone()
        stress[elastic_mask] = trial_stress[elastic_mask]
    g = a + b
    rho = (a - b) / torch.clamp(g, min=1.0e-12)
    grho = torch.cat([g, rho], dim=1)
    route_branch = torch.argmax(route_probs, dim=1) + 1
    return stress, stress_principal, grho, pred_raw, route_probs, route_branch


def _soft_atlas_symmetry_loss(
    *,
    model: nn.Module,
    xb: torch.Tensor | None,
    x_scaler: Standardizer | None,
    y_scaler: Standardizer,
    pred_raw: torch.Tensor,
    route_logits: torch.Tensor,
    trial_principal: torch.Tensor,
    material_reduced: torch.Tensor,
    branch_true: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if xb is None or x_scaler is None:
        return pred_raw.new_tensor(0.0), {"symmetry_loss": 0.0}

    if branch_true is not None:
        valid = branch_true > 0
        if not torch.any(valid):
            return pred_raw.new_tensor(0.0), {"symmetry_loss": 0.0}
        xb = xb[valid]
        pred_raw = pred_raw[valid]
        route_logits = route_logits[valid]
        trial_principal = trial_principal[valid]
        material_reduced = material_reduced[valid]

    x_raw = xb * torch.as_tensor(x_scaler.std, device=xb.device) + torch.as_tensor(x_scaler.mean, device=xb.device)
    x_mirror_raw = _mirror_soft_atlas_features(x_raw)
    x_mirror = torch.from_numpy(x_scaler.transform(x_mirror_raw.detach().cpu().numpy())).to(device=xb.device)
    out_mirror = model(x_mirror)
    pred_mirror_raw = out_mirror["stress"] * torch.as_tensor(y_scaler.std, device=xb.device) + torch.as_tensor(y_scaler.mean, device=xb.device)
    route_mirror = out_mirror.get("route_logits")
    if route_mirror is None:
        return pred_raw.new_tensor(0.0), {"symmetry_loss": 0.0}

    def _chart_components(raw: torch.Tensor, trial_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g_tr = trial_p[:, 0] - trial_p[:, 2]
        b_tr = trial_p[:, 1] - trial_p[:, 2]
        a_tr = trial_p[:, 0] - trial_p[:, 1]
        lambda_tr = torch.clamp(a_tr / torch.clamp(g_tr, min=1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
        g_s = torch.clamp(g_tr[:, None], min=1.0e-12) * torch.exp(raw[:, 0:1])
        lambda_s = torch.sigmoid(torch.logit(lambda_tr)[:, None] + raw[:, 1:2])
        b_l = torch.clamp(b_tr[:, None], min=1.0e-12) * torch.exp(raw[:, 2:3])
        a_r = torch.clamp(a_tr[:, None], min=1.0e-12) * torch.exp(raw[:, 3:4])
        return g_s, lambda_s, b_l, a_r

    trial_principal_mirror = _soft_atlas_mirror_trial_principal(trial_principal)
    g_s, lambda_s, b_l, a_r = _chart_components(pred_raw, trial_principal)
    g_s_m, lambda_s_m, b_l_m, a_r_m = _chart_components(pred_mirror_raw, trial_principal_mirror)
    route_swapped = route_logits[:, [0, 2, 1, 3]]
    route_loss = nn.functional.mse_loss(torch.softmax(route_mirror, dim=1), torch.softmax(route_swapped, dim=1))
    chart_loss = (
        nn.functional.mse_loss(g_s_m, g_s)
        + nn.functional.mse_loss(lambda_s_m, 1.0 - lambda_s)
        + nn.functional.mse_loss(b_l_m, a_r)
        + nn.functional.mse_loss(a_r_m, b_l)
    )
    loss = 0.5 * route_loss + 0.5 * chart_loss
    return loss, {"symmetry_loss": float(loss.detach().cpu())}


def _decode_stress_prediction(
    *,
    model_kind: str,
    pred_norm: torch.Tensor,
    y_scaler: Standardizer,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    material_reduced: torch.Tensor,
    branch_logits: torch.Tensor | None = None,
    coordinate_scales: dict[str, float] | None = None,
    projection_mode: str = "exact",
    projection_tau: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _is_acn_model(model_kind):
        if coordinate_scales is None:
            raise ValueError("ACN decoding requires coordinate scales.")
        stress, stress_principal, _abr_pos, _t_abr = _decode_acn_outputs(
            pred_norm,
            material_reduced,
            eigvecs,
            coordinate_scales,
        )
        return stress, stress_principal

    if _is_surface_model(model_kind):
        if _is_soft_atlas_surface_model(model_kind):
            pred_raw = pred_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
                y_scaler.mean, device=pred_norm.device
            )
            if branch_logits is None:
                raise ValueError("Soft-atlas decoding requires route logits.")
            stress, stress_principal, _grho, _atlas_raw, _route_probs, _route_branch = _decode_soft_atlas_surface_outputs(
                pred_raw,
                branch_logits,
                material_reduced,
                eigvecs,
                trial_stress,
                trial_principal,
            )
            return stress, stress_principal
        if coordinate_scales is None:
            raise ValueError("Plastic-surface decoding requires coordinate scales.")
        if _is_branch_structured_surface_model(model_kind):
            if branch_logits is None:
                raise ValueError("Branch-structured plastic-surface decoding requires branch logits.")
            branch_route = branch_logits.argmax(dim=1) + 1
            stress, stress_principal, _grho_pos, _t_branch, _route = _decode_branch_structured_surface_outputs(
                pred_norm,
                branch_route,
                material_reduced,
                eigvecs,
                trial_stress,
                trial_principal,
                coordinate_scales,
            )
            return stress, stress_principal
        stress, stress_principal, _grho_pos, _t_grho = _decode_surface_outputs(
            pred_norm,
            material_reduced,
            eigvecs,
            trial_stress,
            trial_principal,
            coordinate_scales,
        )
        return stress, stress_principal

    pred = pred_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
        y_scaler.mean, device=pred_norm.device
    )

    if model_kind == "principal":
        stress_principal = pred
        stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
        return stress, stress_principal

    if model_kind == "trial_principal_branch_residual":
        stress_principal = pred + trial_principal
        stress_principal, _ = torch.sort(stress_principal, dim=-1, descending=True)
        if branch_logits is not None:
            pred_branch = branch_logits.argmax(dim=1)
            elastic_mask = pred_branch == 0
            if torch.any(elastic_mask):
                stress_principal = stress_principal.clone()
                stress_principal[elastic_mask] = trial_principal[elastic_mask]
        stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
        return stress, stress_principal

    if model_kind == "trial_principal_geom_plastic_branch_residual":
        stress_principal = pred + trial_principal
        stress_principal, _ = torch.sort(stress_principal, dim=-1, descending=True)
        stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
        return stress, stress_principal

    if _is_projected_student_model(model_kind):
        stress, stress_principal, _provisional = _decode_projected_student_outputs(
            pred_norm,
            y_scaler,
            material_reduced,
            eigvecs,
            trial_stress,
            trial_principal,
            projection_mode=projection_mode,
            projection_tau=projection_tau,
        )
        return stress, stress_principal

    if _uses_residual_target(model_kind):
        return pred + trial_stress, None

    return pred, None


def _regression_loss(
    model_kind: str,
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    y_scaler: Standardizer,
    xb: torch.Tensor | None,
    x_scaler: Standardizer | None,
    model: nn.Module | None,
    branch_true: torch.Tensor,
    eigvecs: torch.Tensor,
    stress_true: torch.Tensor,
    stress_principal_true: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    abr_true_nonneg: torch.Tensor,
    grho_true: torch.Tensor,
    soft_route_target: torch.Tensor | None,
    material_reduced: torch.Tensor,
    teacher_provisional_stress_principal: torch.Tensor,
    teacher_projected_stress_principal: torch.Tensor,
    teacher_projection_delta_principal: torch.Tensor,
    hard_mask: torch.Tensor,
    any_boundary_mask: torch.Tensor,
    teacher_gap_q90_mask: torch.Tensor,
    teacher_gap_q95_mask: torch.Tensor,
    delta_gap_q90_mask: torch.Tensor,
    edge_apex_adjacent_mask: torch.Tensor,
    high_disp_mask: torch.Tensor,
    teacher_projection_candidate_id: torch.Tensor,
    branch_logits: torch.Tensor | None,
    stress_weight_alpha: float,
    stress_weight_scale: float,
    regression_loss_kind: str,
    huber_delta: float,
    voigt_mae_weight: float,
    projected_student_hard_loss_multiplier: float = 1.0,
    projected_student_any_boundary_loss_multiplier: float = 1.0,
    projected_student_high_disp_loss_multiplier: float = 1.0,
    projected_student_teacher_gap_q90_loss_multiplier: float = 1.0,
    projected_student_teacher_gap_q95_loss_multiplier: float = 1.0,
    projected_student_delta_gap_q90_loss_multiplier: float = 1.0,
    projected_student_edge_apex_loss_multiplier: float = 1.0,
    projected_student_candidate_loss_weights: Mapping[int, float] | None = None,
    projected_student_branch_loss_weights: Mapping[int, float] | None = None,
    projected_student_teacher_alignment_focus_multiplier: float = 1.0,
    projected_student_hard_quantile: float = 0.0,
    projected_student_hard_quantile_weight: float = 0.0,
    coordinate_scales: dict[str, float] | None = None,
    projection_mode: str = "exact",
    projection_tau: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    if _is_projected_student_model(model_kind):
        stress_pred, principal_pred, provisional_principal = _decode_projected_student_outputs(
            pred_norm,
            y_scaler,
            material_reduced,
            eigvecs,
            trial_stress,
            trial_principal,
            projection_mode=projection_mode,
            projection_tau=projection_tau,
        )
        valid = branch_true > 0
        if torch.any(valid):
            stress_pred_sel = stress_pred[valid]
            stress_true_sel = stress_true[valid]
            principal_pred_sel = principal_pred[valid]
            principal_true_sel = stress_principal_true[valid]
            provisional_sel = provisional_principal[valid]
            teacher_provisional_sel = teacher_provisional_stress_principal[valid]
            teacher_projected_sel = teacher_projected_stress_principal[valid]
            teacher_delta_sel = teacher_projection_delta_principal[valid]
            c_bar_sel = material_reduced[valid, 0]
            hard_sel = hard_mask[valid]
            any_boundary_sel = any_boundary_mask[valid]
            teacher_gap_q90_sel = teacher_gap_q90_mask[valid]
            teacher_gap_q95_sel = teacher_gap_q95_mask[valid]
            delta_gap_q90_sel = delta_gap_q90_mask[valid]
            edge_apex_sel = edge_apex_adjacent_mask[valid]
            high_disp_sel = high_disp_mask[valid]
            candidate_sel = teacher_projection_candidate_id[valid]
            branch_logits_sel = branch_logits[valid] if branch_logits is not None else None
            branch_true_sel = branch_true[valid]
            branch_target = branch_true_sel - 1
        else:
            stress_pred_sel = stress_pred[:1]
            stress_true_sel = stress_true[:1]
            principal_pred_sel = principal_pred[:1]
            principal_true_sel = stress_principal_true[:1]
            provisional_sel = provisional_principal[:1]
            teacher_provisional_sel = teacher_provisional_stress_principal[:1]
            teacher_projected_sel = teacher_projected_stress_principal[:1]
            teacher_delta_sel = teacher_projection_delta_principal[:1]
            c_bar_sel = material_reduced[:1, 0]
            hard_sel = hard_mask[:1]
            any_boundary_sel = any_boundary_mask[:1]
            teacher_gap_q90_sel = teacher_gap_q90_mask[:1]
            teacher_gap_q95_sel = teacher_gap_q95_mask[:1]
            delta_gap_q90_sel = delta_gap_q90_mask[:1]
            edge_apex_sel = edge_apex_adjacent_mask[:1]
            high_disp_sel = high_disp_mask[:1]
            candidate_sel = teacher_projection_candidate_id[:1]
            branch_logits_sel = branch_logits[:1] if branch_logits is not None else None
            branch_true_sel = branch_true[:1]
            branch_target = torch.zeros((1,), dtype=torch.long, device=pred_norm.device)

        use_preservation_targets = bool(
            torch.isfinite(teacher_projected_sel).all().detach().cpu()
            and torch.isfinite(teacher_delta_sel).all().detach().cpu()
        )
        principal_row_huber = nn.functional.huber_loss(
            principal_pred_sel,
            principal_true_sel,
            delta=huber_delta,
            reduction="none",
        ).mean(dim=1)
        principal_huber = torch.mean(principal_row_huber)
        stress_mae = torch.mean(torch.abs(stress_pred_sel - stress_true_sel))
        if branch_logits_sel is not None:
            branch_ce_row = nn.functional.cross_entropy(branch_logits_sel, branch_target, reduction="none")
            branch_ce = torch.mean(branch_ce_row)
        else:
            branch_ce_row = pred_norm.new_zeros(principal_pred_sel.shape[0])
            branch_ce = pred_norm.new_tensor(0.0)

        row_weights, focus_mask = _projected_student_row_weights(
            hard_mask=hard_sel,
            any_boundary_mask=any_boundary_sel,
            teacher_gap_q90_mask=teacher_gap_q90_sel,
            teacher_gap_q95_mask=teacher_gap_q95_sel,
            delta_gap_q90_mask=delta_gap_q90_sel,
            edge_apex_adjacent_mask=edge_apex_sel,
            high_disp_mask=high_disp_sel,
            teacher_projection_candidate_id=candidate_sel,
            branch_true=branch_true_sel,
            hard_loss_multiplier=projected_student_hard_loss_multiplier,
            any_boundary_loss_multiplier=projected_student_any_boundary_loss_multiplier,
            high_disp_loss_multiplier=projected_student_high_disp_loss_multiplier,
            teacher_gap_q90_loss_multiplier=projected_student_teacher_gap_q90_loss_multiplier,
            teacher_gap_q95_loss_multiplier=projected_student_teacher_gap_q95_loss_multiplier,
            delta_gap_q90_loss_multiplier=projected_student_delta_gap_q90_loss_multiplier,
            edge_apex_loss_multiplier=projected_student_edge_apex_loss_multiplier,
            candidate_loss_weights=projected_student_candidate_loss_weights,
            branch_loss_weights=projected_student_branch_loss_weights,
        )
        teacher_row_weights = row_weights.clone()
        if projected_student_teacher_alignment_focus_multiplier != 1.0:
            teacher_row_weights = teacher_row_weights * torch.where(
                focus_mask,
                teacher_row_weights.new_tensor(float(projected_student_teacher_alignment_focus_multiplier)),
                teacher_row_weights.new_tensor(1.0),
            )

        if use_preservation_targets:
            projected_teacher_row_huber = nn.functional.huber_loss(
                principal_pred_sel,
                teacher_projected_sel,
                delta=huber_delta,
                reduction="none",
            ).mean(dim=1)
            provisional_teacher_row_huber = nn.functional.huber_loss(
                provisional_sel,
                teacher_provisional_sel,
                delta=huber_delta,
                reduction="none",
            ).mean(dim=1)
            projection_delta_student = principal_pred_sel - provisional_sel
            projection_delta_row_huber = nn.functional.huber_loss(
                projection_delta_student,
                teacher_delta_sel,
                delta=huber_delta,
                reduction="none",
            ).mean(dim=1)
            projected_teacher_huber = _weighted_mean(projected_teacher_row_huber, teacher_row_weights)
            provisional_teacher_huber = _weighted_mean(provisional_teacher_row_huber, row_weights)
            projection_delta_huber = _weighted_mean(
                projection_delta_row_huber,
                teacher_row_weights,
            )
            principal_huber = _weighted_mean(principal_row_huber, row_weights)
            branch_ce = _weighted_mean(branch_ce_row, row_weights) if branch_logits_sel is not None else pred_norm.new_tensor(0.0)
            total = (
                1.00 * principal_huber
                + 0.75 * projected_teacher_huber
                + 0.50 * provisional_teacher_huber
                + 0.25 * projection_delta_huber
                + 0.10 * branch_ce
            )
            hard_quantile_huber = pred_norm.new_tensor(0.0)
            hard_quantile_threshold = float("nan")
            hard_quantile_rows = 0
            if projected_student_hard_quantile_weight > 0.0 and projected_student_hard_quantile > 0.0:
                hard_local = hard_sel > 0
                if torch.any(hard_local):
                    hard_max_abs = torch.max(torch.abs(principal_pred_sel - principal_true_sel), dim=1).values[hard_local]
                    hard_quantile_threshold = float(torch.quantile(hard_max_abs, float(projected_student_hard_quantile)).detach().cpu())
                    tail_mask = hard_local & (torch.max(torch.abs(principal_pred_sel - principal_true_sel), dim=1).values >= hard_max_abs.new_tensor(hard_quantile_threshold))
                    hard_quantile_rows = int(torch.sum(tail_mask).detach().cpu())
                    if hard_quantile_rows > 0:
                        hard_quantile_huber = nn.functional.huber_loss(
                            principal_pred_sel[tail_mask],
                            principal_true_sel[tail_mask],
                            delta=huber_delta,
                        )
                        total = total + float(projected_student_hard_quantile_weight) * hard_quantile_huber
            metrics = {
                "regression_mse": float(torch.mean((principal_pred_sel - principal_true_sel) ** 2).detach().cpu()),
                "stress_mse": float(torch.mean((stress_pred - stress_true) ** 2).detach().cpu()),
                "stress_mae": float(torch.mean(torch.abs(stress_pred - stress_true)).detach().cpu()),
                "principal_mse": float(torch.mean((principal_pred - stress_principal_true) ** 2).detach().cpu()),
                "projected_teacher_huber_loss": float(projected_teacher_huber.detach().cpu()),
                "provisional_teacher_huber_loss": float(provisional_teacher_huber.detach().cpu()),
                "projection_delta_huber_loss": float(projection_delta_huber.detach().cpu()),
                "projection_disp_mean": float(torch.mean(torch.linalg.norm(principal_pred_sel - provisional_sel, dim=1)).detach().cpu()),
                "branch_ce_loss": float(branch_ce.detach().cpu()),
                "projected_student_focus_fraction": float(torch.mean(focus_mask.float()).detach().cpu()),
                "projected_student_row_weight_mean": float(torch.mean(row_weights).detach().cpu()),
                "projected_student_teacher_row_weight_mean": float(torch.mean(teacher_row_weights).detach().cpu()),
                "hard_quantile_huber_loss": float(hard_quantile_huber.detach().cpu()),
                "hard_quantile_threshold": hard_quantile_threshold,
                "hard_quantile_rows": float(hard_quantile_rows),
            }
        else:
            teacher_row_huber = nn.functional.huber_loss(
                provisional_sel,
                teacher_provisional_sel,
                delta=huber_delta,
                reduction="none",
            ).mean(dim=1)
            disp_norm = torch.linalg.norm(principal_pred_sel - provisional_sel, dim=1)
            disp_denom = torch.linalg.norm(teacher_provisional_sel, dim=1) + c_bar_sel + 1.0
            disp_penalty = _weighted_mean(disp_norm / torch.clamp(disp_denom, min=1.0e-12), row_weights)
            principal_huber = _weighted_mean(principal_row_huber, row_weights)
            teacher_huber = _weighted_mean(teacher_row_huber, teacher_row_weights)
            branch_ce = _weighted_mean(branch_ce_row, row_weights) if branch_logits_sel is not None else pred_norm.new_tensor(0.0)
            total = principal_huber + 0.5 * stress_mae + 0.25 * teacher_huber + 0.05 * disp_penalty + 0.05 * branch_ce
            metrics = {
                "regression_mse": float(torch.mean((principal_pred_sel - principal_true_sel) ** 2).detach().cpu()),
                "stress_mse": float(torch.mean((stress_pred - stress_true) ** 2).detach().cpu()),
                "stress_mae": float(torch.mean(torch.abs(stress_pred - stress_true)).detach().cpu()),
                "principal_mse": float(torch.mean((principal_pred - stress_principal_true) ** 2).detach().cpu()),
                "projection_disp_loss": float(disp_penalty.detach().cpu()),
                "projection_disp_mean": float(torch.mean(torch.linalg.norm(principal_pred_sel - provisional_sel, dim=1)).detach().cpu()),
                "teacher_huber_loss": float(teacher_huber.detach().cpu()),
                "branch_ce_loss": float(branch_ce.detach().cpu()),
                "projected_student_focus_fraction": float(torch.mean(focus_mask.float()).detach().cpu()),
                "projected_student_row_weight_mean": float(torch.mean(row_weights).detach().cpu()),
                "projected_student_teacher_row_weight_mean": float(torch.mean(teacher_row_weights).detach().cpu()),
                "hard_quantile_huber_loss": 0.0,
                "hard_quantile_threshold": float("nan"),
                "hard_quantile_rows": 0.0,
            }
        return total, metrics

    if _is_soft_atlas_surface_model(model_kind):
        pred_raw = pred_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
            y_scaler.mean, device=pred_norm.device
        )
        target_raw = target_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
            y_scaler.mean, device=pred_norm.device
        )
        if branch_logits is None or soft_route_target is None:
            raise ValueError("Soft-atlas regression requires route logits and soft route targets.")
        stress_pred, principal_pred, grho_pred, atlas_pred_raw, route_probs, _route_branch = _decode_soft_atlas_surface_outputs(
            pred_raw,
            branch_logits,
            material_reduced,
            eigvecs,
            trial_stress,
            trial_principal,
        )
        valid = branch_true > 0
        smooth = branch_true == BRANCH_TO_ID["smooth"]
        left = branch_true == BRANCH_TO_ID["left_edge"]
        right = branch_true == BRANCH_TO_ID["right_edge"]
        apex = branch_true == BRANCH_TO_ID["apex"]

        if torch.any(valid):
            stress_true_sel = stress_true[valid]
            principal_true_sel = stress_principal_true[valid]
            stress_pred_sel = stress_pred[valid]
            principal_pred_sel = principal_pred[valid]
            pred_raw_sel = pred_raw[valid]
            target_raw_sel = target_raw[valid]
            route_logits_sel = branch_logits[valid]
            route_target_sel = soft_route_target[valid]
        else:
            stress_true_sel = stress_true[:1]
            principal_true_sel = stress_principal_true[:1]
            stress_pred_sel = stress_pred[:1]
            principal_pred_sel = principal_pred[:1]
            pred_raw_sel = pred_raw[:1]
            target_raw_sel = target_raw[:1]
            route_logits_sel = branch_logits[:1]
            route_target_sel = soft_route_target[:1]

        denom = torch.linalg.norm(principal_true_sel, dim=1) + torch.clamp(material_reduced[valid, 0] if torch.any(valid) else material_reduced[:1, 0], min=1.0e-6) + 1.0e-6
        principal_rel = torch.linalg.norm(principal_pred_sel - principal_true_sel, dim=1) / denom
        principal_zero = torch.zeros_like(principal_rel)
        principal_loss = nn.functional.huber_loss(principal_rel, principal_zero, delta=huber_delta)
        voigt_loss = torch.mean(torch.abs(stress_pred_sel - stress_true_sel))

        def _masked_loss(pred_cols: torch.Tensor, target_cols: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if not torch.any(mask):
                return pred_norm.new_tensor(0.0)
            if regression_loss_kind == "mse":
                return torch.mean((pred_cols[mask] - target_cols[mask]) ** 2)
            if regression_loss_kind == "huber":
                return nn.functional.huber_loss(pred_cols[mask], target_cols[mask], delta=huber_delta)
            raise ValueError(f"Unsupported regression_loss_kind {regression_loss_kind!r}.")

        chart_terms: list[torch.Tensor] = []
        if torch.any(smooth):
            chart_terms.append(_masked_loss(pred_raw[:, 0:2], target_raw[:, 0:2], smooth))
        if torch.any(left):
            chart_terms.append(_masked_loss(pred_raw[:, 2:3], target_raw[:, 2:3], left))
        if torch.any(right):
            chart_terms.append(_masked_loss(pred_raw[:, 3:4], target_raw[:, 3:4], right))
        if torch.any(apex):
            chart_terms.append(_masked_loss(pred_raw, torch.zeros_like(pred_raw), apex))
        chart_loss = torch.stack(chart_terms).mean() if chart_terms else pred_norm.new_tensor(0.0)

        route_valid = branch_true > 0
        if torch.any(route_valid):
            route_loss = -(route_target_sel * torch.log_softmax(route_logits_sel, dim=1)).sum(dim=1).mean()
        else:
            route_loss = pred_norm.new_tensor(0.0)

        g_tr = trial_principal[:, 0] - trial_principal[:, 2]
        b_tr = trial_principal[:, 1] - trial_principal[:, 2]
        a_tr = trial_principal[:, 0] - trial_principal[:, 1]
        g_s = g_tr[:, None] * torch.exp(pred_raw[:, 0:1])
        lambda_s = torch.sigmoid(torch.logit(torch.clamp(a_tr / torch.clamp(g_tr, min=1.0e-12), 1.0e-6, 1.0 - 1.0e-6))[:, None] + pred_raw[:, 1:2])
        a_s = lambda_s * g_s
        b_s = (1.0 - lambda_s) * g_s
        a_r = torch.clamp(a_tr[:, None], min=1.0e-12) * torch.exp(pred_raw[:, 3:4])
        b_l = torch.clamp(b_tr[:, None], min=1.0e-12) * torch.exp(pred_raw[:, 2:3])
        trial_scale = torch.clamp(g_tr[:, None].abs() + torch.clamp(material_reduced[:, 0:1], min=1.0e-6), min=1.0e-6)
        edge_terms: list[torch.Tensor] = []
        if torch.any(left):
            edge_terms.append(torch.mean((a_s[left] / trial_scale[left]) ** 2))
        if torch.any(right):
            edge_terms.append(torch.mean((b_s[right] / trial_scale[right]) ** 2))
        if torch.any(apex):
            edge_terms.append(torch.mean(((g_s[apex] + a_r[apex] + b_l[apex]) / trial_scale[apex]) ** 2))
        edge_loss = torch.stack(edge_terms).mean() if edge_terms else pred_norm.new_tensor(0.0)

        symmetry_loss = pred_norm.new_tensor(0.0)
        if model is not None and xb is not None and x_scaler is not None:
            symmetry_loss, _ = _soft_atlas_symmetry_loss(
                model=model,
                xb=xb,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                pred_raw=pred_raw,
                route_logits=branch_logits,
                trial_principal=trial_principal,
                material_reduced=material_reduced,
                branch_true=branch_true,
            )

        total = (
            1.0 * principal_loss
            + 0.5 * voigt_loss
            + 0.5 * chart_loss
            + 0.25 * route_loss
            + 0.20 * edge_loss
            + 0.05 * symmetry_loss
        )
        metrics = {
            "regression_mse": float(chart_loss.detach().cpu()),
            "stress_mse": float(torch.mean((stress_pred - stress_true) ** 2).detach().cpu()),
            "stress_mae": float(torch.mean(torch.abs(stress_pred - stress_true)).detach().cpu()),
            "principal_mse": float(torch.mean((principal_pred - stress_principal_true) ** 2).detach().cpu()),
            "coord_mse": float(torch.mean((pred_raw - target_raw) ** 2).detach().cpu()),
            "route_soft_loss": float(route_loss.detach().cpu()),
            "edge_degeneracy_loss": float(edge_loss.detach().cpu()),
            "symmetry_loss": float(symmetry_loss.detach().cpu()),
        }
        return total, metrics

    if _is_surface_model(model_kind):
        if coordinate_scales is None:
            raise ValueError("Plastic-surface regression requires coordinate scales.")
        if _is_branch_structured_surface_model(model_kind):
            stress_pred, principal_pred, grho_pred, t_pred, _route = _decode_branch_structured_surface_outputs(
                pred_norm,
                branch_true,
                material_reduced,
                eigvecs,
                trial_stress,
                trial_principal,
                coordinate_scales,
            )
            valid = branch_true > 0
            smooth = branch_true == BRANCH_TO_ID["smooth"]
            left = branch_true == BRANCH_TO_ID["left_edge"]
            right = branch_true == BRANCH_TO_ID["right_edge"]
            if torch.any(valid):
                stress_true_sel = stress_true[valid]
                principal_true_sel = stress_principal_true[valid]
                stress_pred_sel = stress_pred[valid]
                principal_pred_sel = principal_pred[valid]
                grho_pred_sel = grho_pred[valid]
                grho_true_sel = grho_true[valid]
            else:
                stress_true_sel = stress_true[:1]
                principal_true_sel = stress_principal_true[:1]
                stress_pred_sel = stress_pred[:1]
                principal_pred_sel = principal_pred[:1]
                grho_pred_sel = grho_pred[:1]
                grho_true_sel = grho_true[:1]

            def _masked_component_loss(pred_col: torch.Tensor, target_col: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                if not torch.any(mask):
                    return pred_norm.new_tensor(0.0)
                if regression_loss_kind == "mse":
                    return torch.mean((pred_col[mask] - target_col[mask]) ** 2)
                if regression_loss_kind == "huber":
                    return nn.functional.huber_loss(pred_col[mask], target_col[mask], delta=huber_delta)
                raise ValueError(f"Unsupported regression_loss_kind {regression_loss_kind!r}.")

            smooth_g_loss = _masked_component_loss(t_pred[:, 0], target_norm[:, 0], smooth)
            smooth_rho_loss = _masked_component_loss(t_pred[:, 1], target_norm[:, 1], smooth)
            left_g_loss = _masked_component_loss(t_pred[:, 2], target_norm[:, 2], left)
            right_g_loss = _masked_component_loss(t_pred[:, 3], target_norm[:, 3], right)
            g_loss_terms = [loss for loss, mask in ((smooth_g_loss, smooth), (left_g_loss, left), (right_g_loss, right)) if torch.any(mask)]
            if g_loss_terms:
                g_loss = torch.stack(g_loss_terms).mean()
            else:
                g_loss = pred_norm.new_tensor(0.0)
            rho_loss = smooth_rho_loss

            if regression_loss_kind == "mse":
                stress_loss = torch.mean((stress_pred_sel - stress_true_sel) ** 2)
                principal_loss = torch.mean((principal_pred_sel - principal_true_sel) ** 2)
            elif regression_loss_kind == "huber":
                stress_loss = nn.functional.huber_loss(stress_pred_sel, stress_true_sel, delta=huber_delta)
                principal_loss = nn.functional.huber_loss(principal_pred_sel, principal_true_sel, delta=huber_delta)
            else:
                raise ValueError(f"Unsupported regression_loss_kind {regression_loss_kind!r}.")
            stress_mae = torch.mean(torch.abs(stress_pred - stress_true))
            stress_mse = torch.mean((stress_pred - stress_true) ** 2)
            total = 0.20 * stress_loss + 0.30 * principal_loss + 1.00 * g_loss + 2.00 * rho_loss + voigt_mae_weight * stress_mae
            metrics = {
                "regression_mse": float((g_loss + rho_loss).detach().cpu()),
                "stress_mse": float(stress_mse.detach().cpu()),
                "stress_mae": float(stress_mae.detach().cpu()),
                "principal_mse": float(torch.mean((principal_pred - stress_principal_true) ** 2).detach().cpu()),
                "coord_mse": float((g_loss + rho_loss).detach().cpu()),
                "g_mae": float(torch.mean(torch.abs(grho_pred_sel[:, 0] - grho_true_sel[:, 0])).detach().cpu()),
                "rho_mae": float(torch.mean(torch.abs(grho_pred_sel[:, 1] - grho_true_sel[:, 1])).detach().cpu()),
            }
            return total, metrics
        stress_pred, principal_pred, grho_pred, t_pred = _decode_surface_outputs(
            pred_norm,
            material_reduced,
            eigvecs,
            trial_stress,
            trial_principal,
            coordinate_scales,
        )
        valid = branch_true > 0
        if torch.any(valid):
            stress_true_sel = stress_true[valid]
            principal_true_sel = stress_principal_true[valid]
            target_sel = target_norm[valid]
            t_pred_sel = t_pred[valid]
            stress_pred_sel = stress_pred[valid]
            principal_pred_sel = principal_pred[valid]
            grho_pred_sel = grho_pred[valid]
            grho_true_sel = grho_true[valid]
        else:
            stress_true_sel = stress_true[:1]
            principal_true_sel = stress_principal_true[:1]
            target_sel = target_norm[:1]
            t_pred_sel = t_pred[:1]
            stress_pred_sel = stress_pred[:1]
            principal_pred_sel = principal_pred[:1]
            grho_pred_sel = grho_pred[:1]
            grho_true_sel = grho_true[:1]
        if regression_loss_kind == "mse":
            g_loss = torch.mean((t_pred_sel[:, :1] - target_sel[:, :1]) ** 2)
            rho_loss = torch.mean((t_pred_sel[:, 1:] - target_sel[:, 1:]) ** 2)
            stress_loss = torch.mean((stress_pred_sel - stress_true_sel) ** 2)
            principal_loss = torch.mean((principal_pred_sel - principal_true_sel) ** 2)
        elif regression_loss_kind == "huber":
            g_loss = nn.functional.huber_loss(t_pred_sel[:, :1], target_sel[:, :1], delta=huber_delta)
            rho_loss = nn.functional.huber_loss(t_pred_sel[:, 1:], target_sel[:, 1:], delta=huber_delta)
            stress_loss = nn.functional.huber_loss(stress_pred_sel, stress_true_sel, delta=huber_delta)
            principal_loss = nn.functional.huber_loss(principal_pred_sel, principal_true_sel, delta=huber_delta)
        else:
            raise ValueError(f"Unsupported regression_loss_kind {regression_loss_kind!r}.")
        stress_mae = torch.mean(torch.abs(stress_pred - stress_true))
        stress_mse = torch.mean((stress_pred - stress_true) ** 2)
        total = 0.20 * stress_loss + 0.30 * principal_loss + 1.00 * g_loss + 2.00 * rho_loss + voigt_mae_weight * stress_mae
        metrics = {
            "regression_mse": float((g_loss + rho_loss).detach().cpu()),
            "stress_mse": float(stress_mse.detach().cpu()),
            "stress_mae": float(stress_mae.detach().cpu()),
            "principal_mse": float(torch.mean((principal_pred - stress_principal_true) ** 2).detach().cpu()),
            "coord_mse": float(torch.mean((t_pred_sel - target_sel) ** 2).detach().cpu()),
            "g_mae": float(torch.mean(torch.abs(grho_pred_sel[:, 0] - grho_true_sel[:, 0])).detach().cpu()),
            "rho_mae": float(torch.mean(torch.abs(grho_pred_sel[:, 1] - grho_true_sel[:, 1])).detach().cpu()),
        }
        return total, metrics

    if _is_acn_model(model_kind):
        if coordinate_scales is None:
            raise ValueError("ACN regression requires coordinate scales.")
        stress_pred, principal_pred, _abr_pos, t_pred = _decode_acn_outputs(
            pred_norm,
            material_reduced,
            eigvecs,
            coordinate_scales,
        )
        if regression_loss_kind == "mse":
            coord_loss = torch.mean((t_pred - target_norm) ** 2)
            stress_loss = torch.mean((stress_pred - stress_true) ** 2)
            principal_loss = torch.mean((principal_pred - stress_principal_true) ** 2)
        elif regression_loss_kind == "huber":
            coord_loss = nn.functional.huber_loss(t_pred, target_norm, delta=huber_delta)
            stress_loss = nn.functional.huber_loss(stress_pred, stress_true, delta=huber_delta)
            principal_loss = nn.functional.huber_loss(principal_pred, stress_principal_true, delta=huber_delta)
        else:
            raise ValueError(f"Unsupported regression_loss_kind {regression_loss_kind!r}.")
        stress_mae = torch.mean(torch.abs(stress_pred - stress_true))
        total = stress_loss + 0.25 * principal_loss + 0.25 * coord_loss + voigt_mae_weight * stress_mae
        metrics = {
            "regression_mse": float(coord_loss.detach().cpu()),
            "stress_mse": float(torch.mean((stress_pred - stress_true) ** 2).detach().cpu()),
            "stress_mae": float(stress_mae.detach().cpu()),
            "principal_mse": float(torch.mean((principal_pred - stress_principal_true) ** 2).detach().cpu()),
            "coord_mse": float(torch.mean((t_pred - target_norm) ** 2).detach().cpu()),
        }
        return total, metrics

    if regression_loss_kind == "mse":
        per_sample_loss = torch.mean((pred_norm - target_norm) ** 2, dim=1)
    elif regression_loss_kind == "huber":
        per_sample_loss = nn.functional.huber_loss(pred_norm, target_norm, delta=huber_delta, reduction="none").mean(dim=1)
    else:
        raise ValueError(f"Unsupported regression_loss_kind {regression_loss_kind!r}.")
    if _plastic_only_regression(model_kind):
        valid = branch_true > 0
        if torch.any(valid):
            per_sample_loss = per_sample_loss[valid]
        else:
            per_sample_loss = pred_norm.new_zeros((1,))
    if stress_weight_alpha > 0.0:
        sample_mag = torch.amax(torch.abs(stress_true), dim=1)
        if _plastic_only_regression(model_kind):
            valid = branch_true > 0
            sample_mag = sample_mag[valid] if torch.any(valid) else sample_mag[:1]
        weights = 1.0 + stress_weight_alpha * torch.log1p(sample_mag / max(stress_weight_scale, 1.0e-12))
        reg = torch.mean(weights * per_sample_loss)
    else:
        reg = torch.mean(per_sample_loss)
    metrics = {"regression_mse": float(reg.detach().cpu())}
    stress_pred, _ = _decode_stress_prediction(
        model_kind=model_kind,
        pred_norm=pred_norm,
        y_scaler=y_scaler,
        eigvecs=eigvecs,
        trial_stress=trial_stress,
        trial_principal=trial_principal,
        material_reduced=material_reduced,
        coordinate_scales=coordinate_scales,
        branch_logits=branch_logits,
        projection_mode=projection_mode,
        projection_tau=projection_tau,
    )

    stress_mse = nn.functional.mse_loss(stress_pred, stress_true)
    stress_mae = torch.mean(torch.abs(stress_pred - stress_true))
    metrics["stress_mse"] = float(stress_mse.detach().cpu())
    metrics["stress_mae"] = float(stress_mae.detach().cpu())
    total = reg + voigt_mae_weight * stress_mae
    return total, metrics


def _predict_stress_from_raw_batch(
    *,
    model: nn.Module,
    model_kind: str,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    strain_eng: torch.Tensor,
    material_reduced: torch.Tensor,
    device: torch.device,
    feature_stats: dict[str, Any] | None = None,
    coordinate_scales: dict[str, float] | None = None,
    projection_mode: str = "exact",
    projection_tau: float = 0.05,
) -> torch.Tensor:
    prepared = _prepare_model_inputs(
        model_kind,
        strain_eng.detach().cpu().numpy(),
        material_reduced.detach().cpu().numpy(),
        feature_stats=feature_stats,
    )
    x = torch.from_numpy(x_scaler.transform(prepared["features"])).to(device)
    eigvecs = torch.from_numpy(prepared["eigvecs"]).to(device)
    trial_stress = torch.from_numpy(prepared["trial_stress"]).to(device)
    trial_principal = torch.from_numpy(prepared["trial_principal"]).to(device)
    material_reduced_t = material_reduced.to(device)
    out = model(x)
    stress_pred, _ = _decode_stress_prediction(
        model_kind=model_kind,
        pred_norm=out["stress"],
        y_scaler=y_scaler,
        eigvecs=eigvecs,
        trial_stress=trial_stress,
        trial_principal=trial_principal,
        material_reduced=material_reduced_t,
        coordinate_scales=coordinate_scales,
        branch_logits=out.get("branch_logits"),
        projection_mode=projection_mode,
        projection_tau=projection_tau,
    )
    return stress_pred


def _tangent_direction_loss(
    *,
    model: nn.Module,
    model_kind: str,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    strain_eng: torch.Tensor,
    material_reduced: torch.Tensor,
    tangent_true: torch.Tensor,
    branch_true: torch.Tensor,
    device: torch.device,
    tangent_fd_scale: float,
    feature_stats: dict[str, Any] | None = None,
    coordinate_scales: dict[str, float] | None = None,
    projection_mode: str = "exact",
    projection_tau: float = 0.05,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    tangent = tangent_true
    tangent_valid = torch.isfinite(tangent).all(dim=(1, 2))
    tangent_valid &= torch.linalg.matrix_norm(torch.nan_to_num(tangent), dim=(1, 2)) > 0.0
    tangent_valid &= branch_true > 0
    if not torch.any(tangent_valid):
        return None, {"tangent_dir_mse": 0.0}

    strain_sel = strain_eng[tangent_valid]
    material_sel = material_reduced[tangent_valid]
    tangent_sel = tangent[tangent_valid]
    direction = torch.randn((strain_sel.shape[0], 6), device=device, dtype=strain_sel.dtype)
    direction = direction / torch.clamp(torch.linalg.norm(direction, dim=1, keepdim=True), min=1.0e-12)
    h = tangent_fd_scale * torch.clamp(torch.amax(torch.abs(strain_sel), dim=1, keepdim=True), min=1.0)
    stress_plus = _predict_stress_from_raw_batch(
        model=model,
        model_kind=model_kind,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_stats=feature_stats,
        coordinate_scales=coordinate_scales,
        strain_eng=strain_sel + h * direction,
        material_reduced=material_sel,
        device=device,
        projection_mode=projection_mode,
        projection_tau=projection_tau,
    )
    stress_minus = _predict_stress_from_raw_batch(
        model=model,
        model_kind=model_kind,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_stats=feature_stats,
        coordinate_scales=coordinate_scales,
        strain_eng=strain_sel - h * direction,
        material_reduced=material_sel,
        device=device,
        projection_mode=projection_mode,
        projection_tau=projection_tau,
    )
    jv_pred = (stress_plus - stress_minus) / (2.0 * h)
    jv_target = torch.bmm(tangent_sel, direction.unsqueeze(-1)).squeeze(-1)
    loss = nn.functional.mse_loss(jv_pred, jv_target)
    return loss, {"tangent_dir_mse": float(loss.detach().cpu())}


def _epoch_loop(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    model_kind: str,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    branch_loss_weight: float,
    device: torch.device,
    grad_clip: float,
    stress_weight_alpha: float,
    stress_weight_scale: float,
    regression_loss_kind: str,
    huber_delta: float,
    voigt_mae_weight: float,
    tangent_loss_weight: float,
    tangent_fd_scale: float,
    projected_student_hard_loss_multiplier: float = 1.0,
    projected_student_any_boundary_loss_multiplier: float = 1.0,
    projected_student_high_disp_loss_multiplier: float = 1.0,
    projected_student_teacher_gap_q90_loss_multiplier: float = 1.0,
    projected_student_teacher_gap_q95_loss_multiplier: float = 1.0,
    projected_student_delta_gap_q90_loss_multiplier: float = 1.0,
    projected_student_edge_apex_loss_multiplier: float = 1.0,
    projected_student_candidate_loss_weights: Mapping[int, float] | None = None,
    projected_student_branch_loss_weights: Mapping[int, float] | None = None,
    projected_student_teacher_alignment_focus_multiplier: float = 1.0,
    projected_student_hard_quantile: float = 0.0,
    projected_student_hard_quantile_weight: float = 0.0,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
    feature_stats: dict[str, Any] | None = None,
    coordinate_scales: dict[str, float] | None = None,
    symmetry_loss_weight: float = 0.05,
    projection_mode: str = "exact",
    projection_tau: float = 0.05,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_reg = 0.0
    total_stress = 0.0
    total_stress_mae = 0.0
    total_branch = 0.0
    total_branch_correct = 0.0
    total_tangent = 0.0
    n_branch_samples = 0
    n_samples = 0

    for xb, yb, branch, stress_true, stress_principal_true, eigvecs, trial_stress, trial_principal, abr_true_raw, abr_true_nonneg, grho_true, soft_atlas_route_target, strain_eng, material_reduced, teacher_provisional_stress_principal, teacher_projected_stress_principal, teacher_projection_delta_principal, hard_mask, any_boundary_mask, teacher_gap_q90_mask, teacher_gap_q95_mask, delta_gap_q90_mask, edge_apex_adjacent_mask, high_disp_mask, teacher_projection_candidate_id, tangent_true in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        branch = branch.to(device)
        stress_true = stress_true.to(device)
        stress_principal_true = stress_principal_true.to(device)
        eigvecs = eigvecs.to(device)
        trial_stress = trial_stress.to(device)
        trial_principal = trial_principal.to(device)
        abr_true_raw = abr_true_raw.to(device)
        abr_true_nonneg = abr_true_nonneg.to(device)
        grho_true = grho_true.to(device)
        soft_atlas_route_target = soft_atlas_route_target.to(device)
        strain_eng = strain_eng.to(device)
        material_reduced = material_reduced.to(device)
        teacher_provisional_stress_principal = teacher_provisional_stress_principal.to(device)
        teacher_projected_stress_principal = teacher_projected_stress_principal.to(device)
        teacher_projection_delta_principal = teacher_projection_delta_principal.to(device)
        hard_mask = hard_mask.to(device)
        any_boundary_mask = any_boundary_mask.to(device)
        teacher_gap_q90_mask = teacher_gap_q90_mask.to(device)
        teacher_gap_q95_mask = teacher_gap_q95_mask.to(device)
        delta_gap_q90_mask = delta_gap_q90_mask.to(device)
        edge_apex_adjacent_mask = edge_apex_adjacent_mask.to(device)
        high_disp_mask = high_disp_mask.to(device)
        teacher_projection_candidate_id = teacher_projection_candidate_id.to(device)
        tangent_true = tangent_true.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        out = model(xb)
        reg_loss, reg_metrics = _regression_loss(
            model_kind=model_kind,
            pred_norm=out["stress"],
            target_norm=yb,
            y_scaler=y_scaler,
            branch_true=branch,
            eigvecs=eigvecs,
            stress_true=stress_true,
            stress_principal_true=stress_principal_true,
            trial_stress=trial_stress,
            trial_principal=trial_principal,
            abr_true_nonneg=abr_true_nonneg,
            grho_true=grho_true,
            soft_route_target=soft_atlas_route_target if _is_soft_atlas_surface_model(model_kind) else None,
            model=model if _is_soft_atlas_surface_model(model_kind) else None,
            xb=xb if _is_soft_atlas_surface_model(model_kind) else None,
            x_scaler=x_scaler if _is_soft_atlas_surface_model(model_kind) else None,
            material_reduced=material_reduced,
            teacher_provisional_stress_principal=teacher_provisional_stress_principal,
            teacher_projected_stress_principal=teacher_projected_stress_principal,
            teacher_projection_delta_principal=teacher_projection_delta_principal,
            hard_mask=hard_mask,
            any_boundary_mask=any_boundary_mask,
            teacher_gap_q90_mask=teacher_gap_q90_mask,
            teacher_gap_q95_mask=teacher_gap_q95_mask,
            delta_gap_q90_mask=delta_gap_q90_mask,
            edge_apex_adjacent_mask=edge_apex_adjacent_mask,
            high_disp_mask=high_disp_mask,
            teacher_projection_candidate_id=teacher_projection_candidate_id,
            coordinate_scales=coordinate_scales,
            branch_logits=out.get("branch_logits"),
            stress_weight_alpha=stress_weight_alpha,
            stress_weight_scale=stress_weight_scale,
            regression_loss_kind=regression_loss_kind,
            huber_delta=huber_delta,
            voigt_mae_weight=voigt_mae_weight,
            projected_student_hard_loss_multiplier=projected_student_hard_loss_multiplier,
            projected_student_any_boundary_loss_multiplier=projected_student_any_boundary_loss_multiplier,
            projected_student_high_disp_loss_multiplier=projected_student_high_disp_loss_multiplier,
            projected_student_teacher_gap_q90_loss_multiplier=projected_student_teacher_gap_q90_loss_multiplier,
            projected_student_teacher_gap_q95_loss_multiplier=projected_student_teacher_gap_q95_loss_multiplier,
            projected_student_delta_gap_q90_loss_multiplier=projected_student_delta_gap_q90_loss_multiplier,
            projected_student_edge_apex_loss_multiplier=projected_student_edge_apex_loss_multiplier,
            projected_student_candidate_loss_weights=projected_student_candidate_loss_weights,
            projected_student_branch_loss_weights=projected_student_branch_loss_weights,
            projected_student_teacher_alignment_focus_multiplier=projected_student_teacher_alignment_focus_multiplier,
            projected_student_hard_quantile=projected_student_hard_quantile,
            projected_student_hard_quantile_weight=projected_student_hard_quantile_weight,
            projection_mode=projection_mode,
            projection_tau=projection_tau,
        )

        loss = reg_loss
        branch_loss_value = 0.0
        branch_acc_value = 0.0
        branch_valid_count = 0
        tangent_loss_value = 0.0

        if _is_soft_atlas_surface_model(model_kind) and "route_logits" in out:
            valid_branch = branch > 0
            if torch.any(valid_branch):
                route_logits = out["route_logits"][valid_branch]
                route_target = soft_atlas_route_target[valid_branch]
                branch_loss_value = float(reg_metrics.get("route_soft_loss", 0.0))
                pred_branch = route_logits.argmax(dim=1)
                branch_acc_value = float((pred_branch == route_target.argmax(dim=1)).float().mean().detach().cpu())
                branch_valid_count = int(valid_branch.sum().detach().cpu())
        elif _is_projected_student_model(model_kind) and "branch_logits" in out:
            valid_branch, branch_target = _branch_targets_for_model(model_kind, branch)
            if torch.any(valid_branch):
                branch_loss_value = float(reg_metrics.get("branch_ce_loss", 0.0))
                pred_branch = out["branch_logits"][valid_branch].argmax(dim=1)
                branch_acc_value = float((pred_branch == branch_target).float().mean().detach().cpu())
                branch_valid_count = int(valid_branch.sum().detach().cpu())
        elif "branch_logits" in out:
            valid_branch, branch_target = _branch_targets_for_model(model_kind, branch)
            if torch.any(valid_branch):
                branch_loss = nn.functional.cross_entropy(out["branch_logits"][valid_branch], branch_target)
                loss = loss + branch_loss_weight * branch_loss
                branch_loss_value = float(branch_loss.detach().cpu())
                pred_branch = out["branch_logits"][valid_branch].argmax(dim=1)
                branch_acc_value = float((pred_branch == branch_target).float().mean().detach().cpu())
                branch_valid_count = int(valid_branch.sum().detach().cpu())

        if tangent_loss_weight > 0.0:
            tangent_loss, tangent_metrics = _tangent_direction_loss(
                model=model,
                model_kind=model_kind,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                feature_stats=feature_stats,
                coordinate_scales=coordinate_scales,
                strain_eng=strain_eng,
                material_reduced=material_reduced,
                tangent_true=tangent_true,
                branch_true=branch,
                device=device,
                tangent_fd_scale=tangent_fd_scale,
                projection_mode=projection_mode,
                projection_tau=projection_tau,
            )
            if tangent_loss is not None:
                loss = loss + tangent_loss_weight * tangent_loss
                tangent_loss_value = tangent_metrics["tangent_dir_mse"]

        if training:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema_state_dict is not None and ema_decay > 0.0:
                _update_ema_state_dict(ema_state_dict, model, ema_decay)

        batch_size = xb.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_reg += reg_metrics["regression_mse"] * batch_size
        total_stress += reg_metrics["stress_mse"] * batch_size
        total_stress_mae += reg_metrics["stress_mae"] * batch_size
        total_branch += branch_loss_value * max(branch_valid_count, 0)
        total_branch_correct += branch_acc_value * max(branch_valid_count, 0)
        n_branch_samples += branch_valid_count
        total_tangent += tangent_loss_value * batch_size
        n_samples += batch_size

    return {
        "loss": total_loss / max(n_samples, 1),
        "regression_mse": total_reg / max(n_samples, 1),
        "stress_mse": total_stress / max(n_samples, 1),
        "stress_mae": total_stress_mae / max(n_samples, 1),
        "branch_loss": total_branch / max(n_branch_samples, 1),
        "branch_accuracy": total_branch_correct / max(n_branch_samples, 1),
        "tangent_dir_mse": total_tangent / max(n_samples, 1),
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, str]:
    """Build an epoch scheduler and describe how it should be stepped."""
    if config.scheduler_kind == "none":
        return None, "none"

    if config.scheduler_kind == "plateau":
        plateau_patience = config.plateau_patience
        if plateau_patience is None:
            plateau_patience = max(3, config.patience // 4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=plateau_patience,
            min_lr=config.min_lr,
        )
        return scheduler, "val"

    if config.scheduler_kind == "cosine":
        if config.epochs <= 0:
            return None, "none"
        if config.warmup_epochs > 0:
            start_factor = max(config.min_lr / max(config.lr, 1.0e-12), 1.0e-3)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=min(start_factor, 1.0),
                end_factor=1.0,
                total_iters=config.warmup_epochs,
            )
            remain = max(1, config.epochs - config.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remain,
                eta_min=config.min_lr,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[config.warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, config.epochs),
                eta_min=config.min_lr,
            )
        return scheduler, "epoch"

    raise ValueError(f"Unsupported scheduler kind {config.scheduler_kind!r}.")


def _write_history_row(
    history_path: Path,
    *,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lbfgs_phase: int,
) -> None:
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                lr,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["regression_mse"],
                val_metrics["regression_mse"],
                train_metrics["stress_mse"],
                val_metrics["stress_mse"],
                train_metrics["stress_mae"],
                val_metrics["stress_mae"],
                train_metrics["branch_loss"],
                val_metrics["branch_loss"],
                train_metrics["branch_accuracy"],
                val_metrics["branch_accuracy"],
                train_metrics["tangent_dir_mse"],
                val_metrics["tangent_dir_mse"],
                lbfgs_phase,
            ]
        )


def _maybe_print_epoch_status(
    *,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val: float,
    lbfgs_phase: int,
    config: TrainingConfig,
) -> None:
    if config.log_every_epochs <= 0:
        return
    should_print = epoch == 1 or epoch % config.log_every_epochs == 0
    if not should_print:
        return
    phase_name = "LBFGS" if lbfgs_phase else "Adam"
    print(
        f"[{phase_name}] epoch={epoch} "
        f"lr={lr:.3e} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"val_loss={val_metrics['loss']:.6f} "
        f"train_stress_mse={train_metrics['stress_mse']:.6f} "
        f"val_stress_mse={val_metrics['stress_mse']:.6f} "
        f"val_stress_mae={val_metrics['stress_mae']:.6f} "
        f"val_tangent_dir_mse={val_metrics['tangent_dir_mse']:.6f} "
        f"best_val={best_val:.6f}"
    )


def _save_snapshot(
    *,
    run_dir: Path,
    epoch: int,
    checkpoint: dict[str, Any],
    snapshot_every_epochs: int,
) -> None:
    if snapshot_every_epochs <= 0 or epoch % snapshot_every_epochs != 0:
        return
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, snapshot_dir / f"epoch_{epoch:04d}.pt")


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Train a constitutive surrogate and save history/checkpoints."""
    set_seed(config.seed)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)

    dataset_attrs = _dataset_attrs(config.dataset)
    feature_stats = None
    coordinate_scales = None
    if _is_acn_model(config.model_kind):
        feature_stats = _json_attr_dict(dataset_attrs, "acn_feature_stats_json")
        coordinate_scales = _json_attr_dict(dataset_attrs, "acn_coordinate_scales_json")
    elif _is_soft_atlas_surface_model(config.model_kind):
        feature_stats = _json_attr_dict(dataset_attrs, "soft_atlas_feature_stats_json")
    elif _is_surface_model(config.model_kind):
        feature_stats = _json_attr_dict(dataset_attrs, "surface_feature_stats_json")
        coordinate_scales = _json_attr_dict(dataset_attrs, "surface_coordinate_scales_json")
    train_arrays = _load_split_for_training(
        config.dataset,
        "train",
        config.model_kind,
        include_tangent=config.tangent_loss_weight > 0.0,
        feature_stats=feature_stats,
        coordinate_scales=coordinate_scales,
        projected_student_high_disp_threshold=config.projected_student_high_disp_threshold,
    )
    if _is_acn_model(config.model_kind):
        if feature_stats is None:
            raise ValueError("ACN training requires acn_feature_stats_json in the dataset attrs.")
        if coordinate_scales is None:
            coordinate_scales = _coord_scales_from_abr(train_arrays["abr_true_nonneg"])
    elif _is_soft_atlas_surface_model(config.model_kind):
        if feature_stats is None:
            feature_stats = compute_trial_soft_atlas_feature_stats(
                train_arrays["trial_principal"],
                train_arrays["material_reduced"],
            )
    elif _is_surface_model(config.model_kind):
        if feature_stats is None:
            feature_stats = compute_trial_surface_feature_stats(train_arrays["trial_principal"], train_arrays["material_reduced"])
        if _is_branch_structured_surface_model(config.model_kind):
            required = {"scale_g_smooth", "scale_g_left", "scale_g_right"}
            if coordinate_scales is None or not required.issubset(set(coordinate_scales)):
                coordinate_scales = _coord_scales_from_branch_structured_grho(
                    train_arrays["grho_true"],
                    train_arrays["material_reduced"],
                    train_arrays["branch_id"],
                )
        elif coordinate_scales is None:
            coordinate_scales = _coord_scales_from_grho(train_arrays["grho_true"], train_arrays["material_reduced"])
    val_arrays = _load_split_for_training(
        config.dataset,
        "val",
        config.model_kind,
        include_tangent=config.tangent_loss_weight > 0.0,
        feature_stats=feature_stats,
        coordinate_scales=coordinate_scales,
        projected_student_high_disp_threshold=config.projected_student_high_disp_threshold,
    )

    if _is_surface_model(config.model_kind) and np.any(train_arrays["branch_id"] > 0):
        x_scaler = Standardizer.from_array(train_arrays["features"][train_arrays["branch_id"] > 0])
    else:
        x_scaler = Standardizer.from_array(train_arrays["features"])
    y_scaler = (
        Standardizer.identity(train_arrays["target"].shape[1])
        if (_is_acn_model(config.model_kind) or _is_surface_model(config.model_kind))
        else Standardizer.from_array(train_arrays["target"])
    )

    train_ds = _build_tensor_dataset(train_arrays, x_scaler, y_scaler)
    val_ds = _build_tensor_dataset(val_arrays, x_scaler, y_scaler)

    sample_weights = train_arrays.get("sampling_weight")
    if sample_weights is not None and np.any(sample_weights > 0.0):
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(len(train_ds)),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers)
    elif _is_soft_atlas_surface_model(config.model_kind):
        sample_weights = _soft_atlas_packet4_sample_weights(
            train_arrays["branch_id"],
            train_arrays.get("hard_mask"),
            train_arrays.get("plastic_mask"),
        )
        if np.any(sample_weights > 0.0):
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=int(len(train_ds)),
                replacement=True,
            )
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers)
        else:
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(
        model_kind=config.model_kind,
        input_dim=train_arrays["features"].shape[1],
        width=config.width,
        depth=config.depth,
        dropout=config.dropout,
    ).to(device)
    if config.init_checkpoint:
        init_ckpt = torch.load(config.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
    if config.ema_decay < 0.0 or config.ema_decay >= 1.0:
        raise ValueError(f"ema_decay must satisfy 0 <= decay < 1, got {config.ema_decay!r}.")
    ema_enabled = config.ema_decay > 0.0
    if config.ema_eval and not ema_enabled:
        raise ValueError("ema_eval requires ema_decay > 0.")
    ema_state_dict = _clone_state_dict(model.state_dict()) if ema_enabled else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler, scheduler_step_mode = _build_scheduler(optimizer, config)

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "train_regression_mse",
                "val_regression_mse",
                "train_stress_mse",
                "val_stress_mse",
                "train_stress_mae",
                "val_stress_mae",
                "train_branch_loss",
                "val_branch_loss",
                "train_branch_accuracy",
                "val_branch_accuracy",
                "train_tangent_dir_mse",
                "val_tangent_dir_mse",
                "lbfgs_phase",
            ]
        )

    if config.checkpoint_metric not in {"loss", "stress_mse", "stress_mae"}:
        raise ValueError(f"Unsupported checkpoint_metric {config.checkpoint_metric!r}.")
    best_val = float("inf")
    best_epoch = 0
    completed_epochs = 0
    epochs_without_improvement = 0

    metadata = {
        "config": asdict(config),
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "feature_stats": feature_stats,
        "coordinate_scales": coordinate_scales,
        "branch_names": _branch_names_for_model(config.model_kind),
        "plastic_branch_names": (
            list(PLASTIC_BRANCH_NAMES)
            if (_is_branch_structured_surface_model(config.model_kind) or _is_soft_atlas_surface_model(config.model_kind))
            else None
        ),
        "branch_coordinate_scheme": (
            "branch_specialized"
            if _is_branch_structured_surface_model(config.model_kind)
            else "soft_atlas"
            if _is_soft_atlas_surface_model(config.model_kind)
            else None
        ),
        "branch_head_kind": (
            "learned_plastic_4way"
            if _is_branch_structured_surface_model(config.model_kind)
            else "soft_atlas"
            if _is_soft_atlas_surface_model(config.model_kind)
            else "plastic_only"
            if _branch_head_is_plastic_only(config.model_kind)
            else "full"
        ),
        "ema_decay": config.ema_decay,
        "ema_enabled": ema_enabled,
        "elastic_handling": "exact_upstream" if (_branch_head_is_plastic_only(config.model_kind) or _is_surface_model(config.model_kind)) else "model_decoded",
        "soft_atlas_route_temperature": (1.0 if _is_soft_atlas_surface_model(config.model_kind) else None),
        "soft_atlas_loss_weights": (
            {
                "principal_relative": 1.0,
                "voigt_absolute": 0.5,
                "chart_masked": 0.5,
                "route_soft": 0.25,
                "edge_degeneracy": 0.20,
                "symmetry": 0.05,
            }
            if _is_soft_atlas_surface_model(config.model_kind)
            else None
        ),
        "soft_atlas_sampling": ("hard_broad_branch_balanced" if _is_soft_atlas_surface_model(config.model_kind) else None),
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _epoch_loop(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_kind=config.model_kind,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_stats=feature_stats,
            coordinate_scales=coordinate_scales,
            projection_mode=config.projection_mode,
            projection_tau=config.projection_tau,
            branch_loss_weight=config.branch_loss_weight,
            device=device,
            grad_clip=config.grad_clip,
            stress_weight_alpha=config.stress_weight_alpha,
            stress_weight_scale=config.stress_weight_scale,
            regression_loss_kind=config.regression_loss_kind,
            huber_delta=config.huber_delta,
            voigt_mae_weight=config.voigt_mae_weight,
            tangent_loss_weight=config.tangent_loss_weight,
            tangent_fd_scale=config.tangent_fd_scale,
            projected_student_hard_loss_multiplier=config.projected_student_hard_loss_multiplier,
            projected_student_any_boundary_loss_multiplier=config.projected_student_any_boundary_loss_multiplier,
            projected_student_high_disp_loss_multiplier=config.projected_student_high_disp_loss_multiplier,
            projected_student_teacher_gap_q90_loss_multiplier=config.projected_student_teacher_gap_q90_loss_multiplier,
            projected_student_teacher_gap_q95_loss_multiplier=config.projected_student_teacher_gap_q95_loss_multiplier,
            projected_student_delta_gap_q90_loss_multiplier=config.projected_student_delta_gap_q90_loss_multiplier,
            projected_student_edge_apex_loss_multiplier=config.projected_student_edge_apex_loss_multiplier,
            projected_student_candidate_loss_weights=config.projected_student_candidate_loss_weights,
            projected_student_branch_loss_weights=config.projected_student_branch_loss_weights,
            projected_student_teacher_alignment_focus_multiplier=config.projected_student_teacher_alignment_focus_multiplier,
            projected_student_hard_quantile=config.projected_student_hard_quantile,
            projected_student_hard_quantile_weight=config.projected_student_hard_quantile_weight,
            ema_state_dict=(ema_state_dict if ema_enabled and epoch >= max(1, config.ema_start_epoch) else None),
            ema_decay=config.ema_decay,
        )
        raw_state_dict = _clone_state_dict(model.state_dict())
        if config.ema_eval and ema_enabled and ema_state_dict is not None and epoch >= max(1, config.ema_start_epoch):
            model.load_state_dict(ema_state_dict)
        val_metrics = _epoch_loop(
            model=model,
            loader=val_loader,
            optimizer=None,
            model_kind=config.model_kind,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_stats=feature_stats,
            coordinate_scales=coordinate_scales,
            projection_mode=config.projection_mode,
            projection_tau=config.projection_tau,
            branch_loss_weight=config.branch_loss_weight,
            device=device,
            grad_clip=config.grad_clip,
            stress_weight_alpha=config.stress_weight_alpha,
            stress_weight_scale=config.stress_weight_scale,
            regression_loss_kind=config.regression_loss_kind,
            huber_delta=config.huber_delta,
            voigt_mae_weight=config.voigt_mae_weight,
            tangent_loss_weight=config.tangent_loss_weight,
            tangent_fd_scale=config.tangent_fd_scale,
            projected_student_hard_loss_multiplier=config.projected_student_hard_loss_multiplier,
            projected_student_any_boundary_loss_multiplier=config.projected_student_any_boundary_loss_multiplier,
            projected_student_high_disp_loss_multiplier=config.projected_student_high_disp_loss_multiplier,
            projected_student_teacher_gap_q90_loss_multiplier=config.projected_student_teacher_gap_q90_loss_multiplier,
            projected_student_teacher_gap_q95_loss_multiplier=config.projected_student_teacher_gap_q95_loss_multiplier,
            projected_student_delta_gap_q90_loss_multiplier=config.projected_student_delta_gap_q90_loss_multiplier,
            projected_student_edge_apex_loss_multiplier=config.projected_student_edge_apex_loss_multiplier,
            projected_student_candidate_loss_weights=config.projected_student_candidate_loss_weights,
            projected_student_branch_loss_weights=config.projected_student_branch_loss_weights,
            projected_student_teacher_alignment_focus_multiplier=config.projected_student_teacher_alignment_focus_multiplier,
            projected_student_hard_quantile=config.projected_student_hard_quantile,
            projected_student_hard_quantile_weight=config.projected_student_hard_quantile_weight,
        )
        if config.ema_eval and ema_enabled:
            model.load_state_dict(raw_state_dict)

        if scheduler is not None:
            if scheduler_step_mode == "val":
                scheduler.step(val_metrics[config.checkpoint_metric])
            elif scheduler_step_mode == "epoch":
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        _write_history_row(
            history_path,
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lbfgs_phase=0,
        )
        completed_epochs = epoch

        checkpoint_state = _clone_state_dict(
            ema_state_dict
            if config.ema_eval and ema_enabled and ema_state_dict is not None and epoch >= max(1, config.ema_start_epoch)
            else model.state_dict()
        )
        checkpoint = {
            "model_state_dict": checkpoint_state,
            "metadata": metadata,
        }
        torch.save(checkpoint, last_checkpoint_path)
        _save_snapshot(
            run_dir=run_dir,
            epoch=epoch,
            checkpoint=checkpoint,
            snapshot_every_epochs=config.snapshot_every_epochs,
        )

        current_metric = val_metrics[config.checkpoint_metric]
        if current_metric < best_val:
            best_val = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        _maybe_print_epoch_status(
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val=best_val,
            lbfgs_phase=0,
            config=config,
        )

        if epochs_without_improvement >= config.patience:
            break

    if config.lbfgs_epochs > 0:
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.to(device)
        train_full = tuple(t.to(device) for t in train_ds.tensors)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        for lbfgs_epoch in range(1, config.lbfgs_epochs + 1):
            xb, yb, branch, stress_true, stress_principal_true, eigvecs, trial_stress, trial_principal, abr_true_raw, abr_true_nonneg, grho_true, soft_atlas_route_target, strain_eng, material_reduced, teacher_provisional_stress_principal, teacher_projected_stress_principal, teacher_projection_delta_principal, hard_mask, any_boundary_mask, teacher_gap_q90_mask, teacher_gap_q95_mask, delta_gap_q90_mask, edge_apex_adjacent_mask, high_disp_mask, teacher_projection_candidate_id, tangent_true = train_full

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                out = model(xb)
                reg_loss, _ = _regression_loss(
                    model_kind=config.model_kind,
                    pred_norm=out["stress"],
                    target_norm=yb,
                    y_scaler=y_scaler,
                    branch_true=branch,
                    eigvecs=eigvecs,
                    stress_true=stress_true,
                    stress_principal_true=stress_principal_true,
                    trial_stress=trial_stress,
                    trial_principal=trial_principal,
                    abr_true_nonneg=abr_true_nonneg,
                    grho_true=grho_true,
                    model=model if _is_soft_atlas_surface_model(config.model_kind) else None,
                    xb=xb if _is_soft_atlas_surface_model(config.model_kind) else None,
                    x_scaler=x_scaler if _is_soft_atlas_surface_model(config.model_kind) else None,
                    soft_route_target=soft_atlas_route_target if _is_soft_atlas_surface_model(config.model_kind) else None,
                    material_reduced=material_reduced,
                    teacher_provisional_stress_principal=teacher_provisional_stress_principal,
                    teacher_projected_stress_principal=teacher_projected_stress_principal,
                    teacher_projection_delta_principal=teacher_projection_delta_principal,
                    hard_mask=hard_mask,
                    any_boundary_mask=any_boundary_mask,
                    teacher_gap_q90_mask=teacher_gap_q90_mask,
                    teacher_gap_q95_mask=teacher_gap_q95_mask,
                    delta_gap_q90_mask=delta_gap_q90_mask,
                    edge_apex_adjacent_mask=edge_apex_adjacent_mask,
                    high_disp_mask=high_disp_mask,
                    teacher_projection_candidate_id=teacher_projection_candidate_id,
                    coordinate_scales=coordinate_scales,
                    branch_logits=out.get("branch_logits"),
                    stress_weight_alpha=config.stress_weight_alpha,
                    stress_weight_scale=config.stress_weight_scale,
                    regression_loss_kind=config.regression_loss_kind,
                    huber_delta=config.huber_delta,
                    voigt_mae_weight=config.voigt_mae_weight,
                    projected_student_hard_loss_multiplier=config.projected_student_hard_loss_multiplier,
                    projected_student_any_boundary_loss_multiplier=config.projected_student_any_boundary_loss_multiplier,
                    projected_student_high_disp_loss_multiplier=config.projected_student_high_disp_loss_multiplier,
                    projected_student_teacher_gap_q90_loss_multiplier=config.projected_student_teacher_gap_q90_loss_multiplier,
                    projected_student_teacher_gap_q95_loss_multiplier=config.projected_student_teacher_gap_q95_loss_multiplier,
                    projected_student_delta_gap_q90_loss_multiplier=config.projected_student_delta_gap_q90_loss_multiplier,
                    projected_student_edge_apex_loss_multiplier=config.projected_student_edge_apex_loss_multiplier,
                    projected_student_candidate_loss_weights=config.projected_student_candidate_loss_weights,
                    projected_student_branch_loss_weights=config.projected_student_branch_loss_weights,
                    projected_student_teacher_alignment_focus_multiplier=config.projected_student_teacher_alignment_focus_multiplier,
                    projected_student_hard_quantile=config.projected_student_hard_quantile,
                    projected_student_hard_quantile_weight=config.projected_student_hard_quantile_weight,
                    projection_mode=config.projection_mode,
                    projection_tau=config.projection_tau,
                )
                loss = reg_loss
                if (not _is_soft_atlas_surface_model(config.model_kind)) and (not _is_projected_student_model(config.model_kind)) and "branch_logits" in out:
                    valid_branch, branch_target = _branch_targets_for_model(config.model_kind, branch)
                    if torch.any(valid_branch):
                        branch_loss = nn.functional.cross_entropy(out["branch_logits"][valid_branch], branch_target)
                        loss = loss + config.branch_loss_weight * branch_loss
                if config.tangent_loss_weight > 0.0:
                    tangent_loss, _ = _tangent_direction_loss(
                        model=model,
                        model_kind=config.model_kind,
                        x_scaler=x_scaler,
                        y_scaler=y_scaler,
                        feature_stats=feature_stats,
                        coordinate_scales=coordinate_scales,
                        strain_eng=strain_eng,
                        material_reduced=material_reduced,
                        tangent_true=tangent_true,
                        branch_true=branch,
                        device=device,
                        tangent_fd_scale=config.tangent_fd_scale,
                        projection_mode=config.projection_mode,
                        projection_tau=config.projection_tau,
                    )
                    if tangent_loss is not None:
                        loss = loss + config.tangent_loss_weight * tangent_loss
                loss.backward()
                return loss

            model.train(True)
            lbfgs.step(closure)
            if ema_enabled and ema_state_dict is not None:
                _update_ema_state_dict(ema_state_dict, model, config.ema_decay)

            train_metrics = _epoch_loop(
                model=model,
                loader=train_loader,
                optimizer=None,
                model_kind=config.model_kind,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                feature_stats=feature_stats,
                coordinate_scales=coordinate_scales,
                projection_mode=config.projection_mode,
                projection_tau=config.projection_tau,
                branch_loss_weight=config.branch_loss_weight,
                device=device,
                grad_clip=config.grad_clip,
                stress_weight_alpha=config.stress_weight_alpha,
                stress_weight_scale=config.stress_weight_scale,
                regression_loss_kind=config.regression_loss_kind,
                huber_delta=config.huber_delta,
                voigt_mae_weight=config.voigt_mae_weight,
                tangent_loss_weight=config.tangent_loss_weight,
                tangent_fd_scale=config.tangent_fd_scale,
                projected_student_hard_loss_multiplier=config.projected_student_hard_loss_multiplier,
                projected_student_any_boundary_loss_multiplier=config.projected_student_any_boundary_loss_multiplier,
                projected_student_high_disp_loss_multiplier=config.projected_student_high_disp_loss_multiplier,
                projected_student_teacher_gap_q90_loss_multiplier=config.projected_student_teacher_gap_q90_loss_multiplier,
                projected_student_teacher_gap_q95_loss_multiplier=config.projected_student_teacher_gap_q95_loss_multiplier,
                projected_student_delta_gap_q90_loss_multiplier=config.projected_student_delta_gap_q90_loss_multiplier,
                projected_student_edge_apex_loss_multiplier=config.projected_student_edge_apex_loss_multiplier,
                projected_student_candidate_loss_weights=config.projected_student_candidate_loss_weights,
                projected_student_branch_loss_weights=config.projected_student_branch_loss_weights,
                projected_student_teacher_alignment_focus_multiplier=config.projected_student_teacher_alignment_focus_multiplier,
                projected_student_hard_quantile=config.projected_student_hard_quantile,
                projected_student_hard_quantile_weight=config.projected_student_hard_quantile_weight,
                ema_state_dict=ema_state_dict,
                ema_decay=config.ema_decay,
            )
            raw_state_dict = _clone_state_dict(model.state_dict())
            if config.ema_eval and ema_enabled and ema_state_dict is not None:
                model.load_state_dict(ema_state_dict)
            val_metrics = _epoch_loop(
                model=model,
                loader=val_loader,
                optimizer=None,
                model_kind=config.model_kind,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                feature_stats=feature_stats,
                coordinate_scales=coordinate_scales,
                projection_mode=config.projection_mode,
                projection_tau=config.projection_tau,
                branch_loss_weight=config.branch_loss_weight,
                device=device,
                grad_clip=config.grad_clip,
                stress_weight_alpha=config.stress_weight_alpha,
                stress_weight_scale=config.stress_weight_scale,
                regression_loss_kind=config.regression_loss_kind,
                huber_delta=config.huber_delta,
                voigt_mae_weight=config.voigt_mae_weight,
                tangent_loss_weight=config.tangent_loss_weight,
                tangent_fd_scale=config.tangent_fd_scale,
                projected_student_hard_loss_multiplier=config.projected_student_hard_loss_multiplier,
                projected_student_any_boundary_loss_multiplier=config.projected_student_any_boundary_loss_multiplier,
                projected_student_high_disp_loss_multiplier=config.projected_student_high_disp_loss_multiplier,
                projected_student_teacher_gap_q90_loss_multiplier=config.projected_student_teacher_gap_q90_loss_multiplier,
                projected_student_teacher_gap_q95_loss_multiplier=config.projected_student_teacher_gap_q95_loss_multiplier,
                projected_student_delta_gap_q90_loss_multiplier=config.projected_student_delta_gap_q90_loss_multiplier,
                projected_student_edge_apex_loss_multiplier=config.projected_student_edge_apex_loss_multiplier,
                projected_student_candidate_loss_weights=config.projected_student_candidate_loss_weights,
                projected_student_branch_loss_weights=config.projected_student_branch_loss_weights,
                projected_student_teacher_alignment_focus_multiplier=config.projected_student_teacher_alignment_focus_multiplier,
                projected_student_hard_quantile=config.projected_student_hard_quantile,
                projected_student_hard_quantile_weight=config.projected_student_hard_quantile_weight,
                ema_state_dict=ema_state_dict,
                ema_decay=config.ema_decay,
            )
            if config.ema_eval and ema_enabled:
                model.load_state_dict(raw_state_dict)
            epoch = completed_epochs + lbfgs_epoch
            current_lr = lbfgs.param_groups[0]["lr"]
            _write_history_row(
                history_path,
                epoch=epoch,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lbfgs_phase=1,
            )

            checkpoint_state = _clone_state_dict(
                ema_state_dict if config.ema_eval and ema_enabled and ema_state_dict is not None else model.state_dict()
            )
            checkpoint = {
                "model_state_dict": checkpoint_state,
                "metadata": metadata,
            }
            torch.save(checkpoint, last_checkpoint_path)
            _save_snapshot(
                run_dir=run_dir,
                epoch=epoch,
                checkpoint=checkpoint,
                snapshot_every_epochs=config.snapshot_every_epochs,
            )

            current_metric = val_metrics[config.checkpoint_metric]
            if current_metric < best_val:
                best_val = current_metric
                best_epoch = epoch
                torch.save(checkpoint, best_checkpoint_path)
            completed_epochs = epoch
            _maybe_print_epoch_status(
                epoch=epoch,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val=best_val,
                lbfgs_phase=1,
                config=config,
            )

    summary = {
        "best_val_loss": best_val,
        "checkpoint_metric": config.checkpoint_metric,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "history_csv": str(history_path),
        "device": str(device),
        "ema_decay": config.ema_decay,
        "ema_enabled": ema_enabled,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained model checkpoint."""
    device_obj = choose_device(device)
    ckpt = torch.load(checkpoint_path, map_location=device_obj)
    metadata = ckpt["metadata"]
    cfg = metadata["config"]
    model = build_model(
        model_kind=cfg["model_kind"],
        input_dim=len(metadata["x_scaler"]["mean"]),
        width=cfg["width"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, metadata


def predict_with_loaded_checkpoint(
    model: nn.Module,
    metadata: dict[str, Any],
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    route_temperature: float | None = None,
    branch_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Predict constitutive stresses using an already-loaded checkpoint."""
    device_obj = choose_device(device)
    model = model.to(device_obj)
    cfg = metadata["config"]
    x_scaler = Standardizer.from_dict(metadata["x_scaler"])
    y_scaler = Standardizer.from_dict(metadata["y_scaler"])
    feature_stats = metadata.get("feature_stats")
    coordinate_scales = metadata.get("coordinate_scales")
    branch_head_kind = metadata.get("branch_head_kind", "full")
    projection_mode = cfg.get("projection_mode", "exact")
    projection_tau = float(cfg.get("projection_tau", 0.05))

    strain_eng = np.asarray(strain_eng, dtype=float)
    material_reduced = np.asarray(material_reduced, dtype=float)
    if route_temperature is None:
        route_temperature = metadata.get("soft_atlas_route_temperature", 1.0)
    if route_temperature is None:
        route_temperature = 1.0
    route_temperature = float(route_temperature)
    prepared = _prepare_model_inputs(
        cfg["model_kind"],
        strain_eng,
        material_reduced,
        feature_stats=feature_stats,
    )
    features = prepared["features"]
    strain_principal = prepared["strain_principal"]
    eigvecs = prepared["eigvecs"]
    trial_stress = prepared["trial_stress"]
    trial_principal = prepared["trial_principal"]

    if batch_size is None or batch_size <= 0:
        batch_size = int(features.shape[0])

    pred_chunks: list[np.ndarray] = []
    branch_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(x_scaler.transform(features[start:stop])).to(device_obj)
            out = model(x)
            pred_chunks.append(out["stress"].cpu().numpy())
            if "branch_logits" in out:
                branch_chunks.append(out["branch_logits"].cpu().numpy())
    pred_norm = np.concatenate(pred_chunks, axis=0)
    branch_probs = None
    predicted_branch_id = None
    if branch_chunks:
        logits = np.concatenate(branch_chunks, axis=0)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        if branch_head_kind in {"plastic_only", "learned_plastic_4way", "soft_atlas"}:
            expanded = np.zeros((probs.shape[0], len(BRANCH_NAMES)), dtype=np.float32)
            expanded[:, 1:] = probs.astype(np.float32)
            branch_probs = expanded
        else:
            branch_probs = probs.astype(np.float32)
        predicted_branch_id = np.argmax(branch_probs, axis=1).astype(np.int64)
    pred = y_scaler.inverse_transform(pred_norm)

    if cfg["model_kind"] == "trial_abr_acn_f1":
        scale_vec = np.asarray(
            [
                coordinate_scales["scale_a"],
                coordinate_scales["scale_b"],
                coordinate_scales["scale_r"],
            ],
            dtype=np.float32,
        )
        abr_pred = torch.nn.functional.softplus(torch.from_numpy(pred_norm)).numpy().astype(np.float32) * scale_vec[None, :]
        stress_principal = decode_abr_to_principal_torch(
            torch.from_numpy(abr_pred),
            c_bar=torch.from_numpy(material_reduced[:, 0].astype(np.float32)),
            sin_phi=torch.from_numpy(material_reduced[:, 1].astype(np.float32)),
        ).numpy()
        stress = stress_voigt_from_principal_numpy(stress_principal, eigvecs)
    elif cfg["model_kind"] == "trial_surface_acn_f1":
        stress_t, stress_principal_t, grho_t, _t_grho = _decode_surface_outputs(
            torch.from_numpy(pred_norm.astype(np.float32)),
            torch.from_numpy(material_reduced.astype(np.float32)),
            torch.from_numpy(eigvecs.astype(np.float32)),
            torch.from_numpy(trial_stress.astype(np.float32)),
            torch.from_numpy(trial_principal.astype(np.float32)),
            coordinate_scales,
        )
        stress = stress_t.numpy()
        stress_principal = stress_principal_t.numpy()
        grho_pred = grho_t.numpy().astype(np.float32)
    elif cfg["model_kind"] == "trial_surface_branch_structured_f1":
        if branch_override is not None:
            route_branch_id = np.asarray(branch_override, dtype=np.int64).reshape(-1)
            if route_branch_id.shape[0] != pred_norm.shape[0]:
                raise ValueError(f"Expected branch_override length {pred_norm.shape[0]}, got {route_branch_id.shape[0]}.")
        elif predicted_branch_id is not None:
            route_branch_id = predicted_branch_id
        else:
            route_branch_id = np.full(pred_norm.shape[0], BRANCH_TO_ID["smooth"], dtype=np.int64)
        stress_t, stress_principal_t, grho_t, _t_branch, _route = _decode_branch_structured_surface_outputs(
            torch.from_numpy(pred_norm.astype(np.float32)),
            torch.from_numpy(route_branch_id.astype(np.int64)),
            torch.from_numpy(material_reduced.astype(np.float32)),
            torch.from_numpy(eigvecs.astype(np.float32)),
            torch.from_numpy(trial_stress.astype(np.float32)),
            torch.from_numpy(trial_principal.astype(np.float32)),
            coordinate_scales,
        )
        stress = stress_t.numpy()
        stress_principal = stress_principal_t.numpy()
        grho_pred = grho_t.numpy().astype(np.float32)
    elif cfg["model_kind"] in {"trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"}:
        stress_t, stress_principal_t, grho_t, atlas_t, route_probs_t, _route_branch = _decode_soft_atlas_surface_outputs(
            torch.from_numpy(pred_norm.astype(np.float32)),
            torch.from_numpy(np.concatenate(branch_chunks, axis=0).astype(np.float32)) if branch_chunks else torch.zeros((pred_norm.shape[0], 4), dtype=torch.float32),
            torch.from_numpy(material_reduced.astype(np.float32)),
            torch.from_numpy(eigvecs.astype(np.float32)),
            torch.from_numpy(trial_stress.astype(np.float32)),
            torch.from_numpy(trial_principal.astype(np.float32)),
            route_temperature=route_temperature,
        )
        stress = stress_t.numpy()
        stress_principal = stress_principal_t.numpy()
        grho_pred = grho_t.numpy().astype(np.float32)
        soft_atlas_chart = atlas_t.numpy().astype(np.float32)
        soft_atlas_route_probs = route_probs_t.numpy().astype(np.float32)
        branch_probs = np.zeros((soft_atlas_route_probs.shape[0], len(BRANCH_NAMES)), dtype=np.float32)
        branch_probs[:, 1:] = soft_atlas_route_probs
        predicted_branch_id = np.argmax(branch_probs, axis=1).astype(np.int64)
    elif cfg["model_kind"] == "principal":
        stress = stress_voigt_from_principal_numpy(pred, eigvecs)
        stress_principal = pred
    elif cfg["model_kind"] in {"trial_principal_branch_residual", "trial_principal_geom_plastic_branch_residual"}:
        stress_principal = pred + trial_principal
        stress_principal = np.sort(stress_principal, axis=1)[:, ::-1]
        if cfg["model_kind"] == "trial_principal_branch_residual" and branch_probs is not None:
            pred_branch = np.argmax(branch_probs, axis=1)
            elastic_mask = pred_branch == 0
            if np.any(elastic_mask):
                stress_principal = stress_principal.copy()
                stress_principal[elastic_mask] = trial_principal[elastic_mask]
        stress = stress_voigt_from_principal_numpy(stress_principal, eigvecs)
    elif _is_projected_student_model(cfg["model_kind"]):
        stress_t, stress_principal_t, provisional_t = _decode_projected_student_outputs(
            torch.from_numpy(pred_norm.astype(np.float32)),
            y_scaler,
            torch.from_numpy(material_reduced.astype(np.float32)),
            torch.from_numpy(eigvecs.astype(np.float32)),
            torch.from_numpy(trial_stress.astype(np.float32)),
            torch.from_numpy(trial_principal.astype(np.float32)),
            projection_mode=projection_mode,
            projection_tau=projection_tau,
        )
        stress = stress_t.numpy()
        stress_principal = stress_principal_t.numpy()
        provisional_principal = provisional_t.numpy()
    elif _uses_residual_target(cfg["model_kind"]):
        stress = pred + trial_stress
        stress_principal = None
    else:
        stress = pred
        stress_principal = None

    result = {
        "stress": stress.astype(np.float32),
        "branch_probabilities": branch_probs,
        "branch_head_kind": branch_head_kind,
        "strain_principal": strain_principal.astype(np.float32),
        "eigvecs": eigvecs.astype(np.float32),
    }
    if predicted_branch_id is not None:
        result["predicted_branch_id"] = predicted_branch_id.astype(np.int64)
    if cfg["model_kind"] in {"trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"} and predicted_branch_id is not None:
        result["route_branch_id"] = predicted_branch_id.astype(np.int64)
    if cfg["model_kind"] == "trial_surface_branch_structured_f1":
        result["route_branch_id"] = route_branch_id.astype(np.int64)
    if stress_principal is not None:
        result["stress_principal"] = stress_principal.astype(np.float32)
    if _is_projected_student_model(cfg["model_kind"]):
        result["provisional_stress_principal"] = provisional_principal.astype(np.float32)
    if cfg["model_kind"] == "trial_abr_acn_f1":
        result["abr_nonneg"] = abr_pred.astype(np.float32)
    if cfg["model_kind"] in {"trial_surface_acn_f1", "trial_surface_branch_structured_f1", "trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"}:
        result["grho"] = grho_pred.astype(np.float32)
    if cfg["model_kind"] in {"trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"}:
        result["soft_atlas_chart"] = soft_atlas_chart.astype(np.float32)
        result["soft_atlas_route_probs"] = soft_atlas_route_probs.astype(np.float32)
    return result


def predict_with_checkpoint(
    checkpoint_path: str | Path,
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    route_temperature: float | None = None,
    branch_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Predict constitutive stresses using a saved checkpoint."""
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    return predict_with_loaded_checkpoint(
        model,
        metadata,
        strain_eng,
        material_reduced,
        device=device,
        batch_size=batch_size,
        route_temperature=route_temperature,
        branch_override=branch_override,
    )


def evaluate_checkpoint_on_dataset(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    split: str = "test",
    device: str = "cpu",
    batch_size: int | None = None,
    route_temperature: float | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a dataset split and return metrics and predictions."""
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    cfg = metadata["config"]
    available = _dataset_keys(str(dataset_path))
    keys = ["strain_eng", "stress", "material_reduced"]
    if "stress_principal" in available:
        keys.append("stress_principal")
    if "branch_id" in available:
        keys.append("branch_id")
    arrays = load_arrays(dataset_path, keys, split=split)
    if "stress_principal" not in arrays:
        arrays["stress_principal"] = _principal_stress_from_stress(arrays["stress"])
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
        route_temperature=route_temperature,
    )
    stress_pred = pred["stress"]
    stress_true = arrays["stress"]
    abs_err = np.abs(stress_pred - stress_true)

    metrics: dict[str, Any] = {
        "split": split,
        "n_samples": int(stress_true.shape[0]),
        "stress_mae": float(np.mean(abs_err)),
        "stress_rmse": float(np.sqrt(np.mean((stress_pred - stress_true) ** 2))),
        "stress_max_abs": float(np.max(abs_err)),
        "per_component_mae": np.mean(abs_err, axis=0).tolist(),
    }

    if _outputs_principal_stress(cfg["model_kind"]):
        stress_principal_true = arrays["stress_principal"]
        stress_principal_pred = pred["stress_principal"]
        metrics["principal_mae"] = float(np.mean(np.abs(stress_principal_pred - stress_principal_true)))
        metrics["principal_rmse"] = float(np.sqrt(np.mean((stress_principal_pred - stress_principal_true) ** 2)))
    if "soft_atlas_route_probs" in pred:
        route_probs = pred["soft_atlas_route_probs"]
        route_entropy = -np.sum(route_probs * np.log(np.clip(route_probs, 1.0e-12, 1.0)), axis=1)
        metrics["soft_atlas_route_entropy"] = float(np.mean(route_entropy))
        metrics["soft_atlas_route_maxprob"] = float(np.mean(np.max(route_probs, axis=1)))

    if pred["branch_probabilities"] is not None and "branch_id" in arrays and np.any(arrays["branch_id"] >= 0):
        branch_pred = pred.get("predicted_branch_id")
        if branch_pred is None:
            branch_pred = np.argmax(pred["branch_probabilities"], axis=1)
        branch_true = arrays["branch_id"].astype(int)
        if metadata.get("branch_head_kind", "full") in {"plastic_only", "learned_plastic_4way", "soft_atlas"}:
            plastic = branch_true > 0
            if np.any(plastic):
                metrics["plastic_branch_accuracy"] = float(np.mean(branch_pred[plastic] == branch_true[plastic]))
                metrics["plastic_branch_confusion"] = [
                    [
                        int(np.sum((branch_true[plastic] == i) & (branch_pred[plastic] == j)))
                        for j in range(1, len(BRANCH_NAMES))
                    ]
                    for i in range(1, len(BRANCH_NAMES))
                ]
                metrics["predicted_plastic_branch_counts"] = {
                    BRANCH_NAMES[i]: int(np.sum(branch_pred[plastic] == i))
                    for i in range(1, len(BRANCH_NAMES))
                }
        else:
            metrics["branch_accuracy"] = float(np.mean(branch_pred == branch_true))
            metrics["branch_confusion"] = [
                [
                    int(np.sum((branch_true == i) & (branch_pred == j)))
                    for j in range(len(BRANCH_NAMES))
                ]
                for i in range(len(BRANCH_NAMES))
            ]

    if "branch_id" in arrays and np.any(arrays["branch_id"] >= 0):
        per_branch_mae = {}
        for i, name in enumerate(BRANCH_NAMES):
            mask = arrays["branch_id"] == i
            if np.any(mask):
                per_branch_mae[name] = float(np.mean(np.abs(stress_pred[mask] - stress_true[mask])))
        metrics["per_branch_stress_mae"] = per_branch_mae

    return {
        "metrics": metrics,
        "arrays": arrays,
        "predictions": pred,
    }
