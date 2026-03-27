"""Helpers for the March 27 projection-student preservation/compression packet."""

from __future__ import annotations

import numpy as np

from .models import (
    compute_trial_stress,
    exact_trial_principal_from_strain,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
)
from .mohr_coulomb import BRANCH_NAMES
from .principal_projection import PROJECTION_CANDIDATE_NAMES, PROJECTION_CANDIDATE_TO_ID, project_mc_principal_numpy
from .voigt import stress_voigt_to_tensor


def quantile_or_zero(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def principal_and_eigvecs_from_stress(stress_voigt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stress_tensor = stress_voigt_to_tensor(np.asarray(stress_voigt, dtype=float))
    vals, vecs = np.linalg.eigh(stress_tensor)
    return vals[:, ::-1].astype(np.float32), vecs[:, :, ::-1].astype(np.float32)


def summarize_array(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "p50": quantile_or_zero(arr, 0.50),
        "p95": quantile_or_zero(arr, 0.95),
        "max": float(np.max(arr)),
    }


def directional_error_arrays(
    jv_pred: np.ndarray,
    jv_true: np.ndarray,
) -> dict[str, np.ndarray]:
    pred = np.asarray(jv_pred, dtype=float)
    true = np.asarray(jv_true, dtype=float)
    diff = pred - true
    abs_error = np.linalg.norm(diff, axis=1)
    true_norm = np.linalg.norm(true, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    denom = np.maximum(true_norm, 1.0e-12)
    cosine = np.sum(pred * true, axis=1) / np.maximum(pred_norm * true_norm, 1.0e-12)
    return {
        "abs_error_norm": abs_error.astype(np.float32),
        "relative_error": (abs_error / denom).astype(np.float32),
        "cosine_similarity": np.clip(cosine, -1.0, 1.0).astype(np.float32),
        "pred_norm": pred_norm.astype(np.float32),
        "true_norm": true_norm.astype(np.float32),
    }


def projection_switch_mask(
    candidate_base: np.ndarray,
    candidate_plus: np.ndarray,
    candidate_minus: np.ndarray,
) -> np.ndarray:
    base = np.asarray(candidate_base, dtype=np.int64).reshape(-1)
    plus = np.asarray(candidate_plus, dtype=np.int64).reshape(-1)
    minus = np.asarray(candidate_minus, dtype=np.int64).reshape(-1)
    return ((plus != base) | (minus != base) | (plus != minus)).astype(bool)


def build_sampling_weights(
    *,
    train_mask: np.ndarray,
    hard_mask: np.ndarray,
    teacher_projection_candidate_id: np.ndarray,
    teacher_projection_disp_norm: np.ndarray,
) -> tuple[np.ndarray, float]:
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    hard = np.asarray(hard_mask, dtype=bool).reshape(-1)
    candidate_id = np.asarray(teacher_projection_candidate_id, dtype=np.int64).reshape(-1)
    disp = np.asarray(teacher_projection_disp_norm, dtype=float).reshape(-1)

    train_disp = disp[train & np.isfinite(disp)]
    disp_threshold = quantile_or_zero(train_disp, 0.90)

    weights = np.ones(candidate_id.shape[0], dtype=np.float32)
    edge_mask = np.isin(
        candidate_id,
        [
            PROJECTION_CANDIDATE_TO_ID["left_edge"],
            PROJECTION_CANDIDATE_TO_ID["right_edge"],
        ],
    )
    high_disp_mask = np.isfinite(disp) & (disp >= disp_threshold)
    weights[hard] *= 1.5
    weights[edge_mask] *= 1.5
    weights[high_disp_mask] *= 1.5
    return weights.astype(np.float32), float(disp_threshold)


def build_teacher_projection_cache_arrays(
    arrays_all: dict[str, np.ndarray],
    panel_all: dict[str, np.ndarray],
    teacher_pred: dict[str, np.ndarray],
    *,
    projection_mode: str = "exact",
    projection_tau: float = 0.05,
) -> dict[str, np.ndarray]:
    strain_eng = np.asarray(arrays_all["strain_eng"], dtype=np.float32)
    material_reduced = np.asarray(arrays_all["material_reduced"], dtype=np.float32)
    exact_stress = np.asarray(arrays_all["stress"], dtype=np.float32)
    exact_stress_principal = np.asarray(arrays_all["stress_principal"], dtype=np.float32)
    eigvecs = np.asarray(arrays_all["eigvecs"], dtype=np.float32)
    split_id = np.asarray(arrays_all["split_id"], dtype=np.int8)
    branch_id = np.asarray(arrays_all["branch_id"], dtype=np.int8)
    source_call_id = np.asarray(
        arrays_all.get("source_call_id", np.full(split_id.shape[0], -1, dtype=np.int32)),
        dtype=np.int32,
    )
    source_row_in_call = np.asarray(
        arrays_all.get("source_row_in_call", np.full(split_id.shape[0], -1, dtype=np.int32)),
        dtype=np.int32,
    )

    trial_stress = compute_trial_stress(strain_eng, material_reduced).astype(np.float32)
    trial_principal = exact_trial_principal_from_strain(strain_eng, material_reduced).astype(np.float32)

    checkpoint_stress = np.asarray(teacher_pred["stress"], dtype=np.float32)
    checkpoint_principal, checkpoint_eigvecs = principal_and_eigvecs_from_stress(checkpoint_stress)

    plastic_mask = np.asarray(panel_all["plastic_mask"], dtype=bool)
    elastic_mask = ~plastic_mask

    dispatch_stress = checkpoint_stress.copy()
    dispatch_principal = checkpoint_principal.copy()
    dispatch_stress[elastic_mask] = trial_stress[elastic_mask]
    dispatch_principal[elastic_mask] = trial_principal[elastic_mask]

    projected_principal, projection_details = project_mc_principal_numpy(
        dispatch_principal,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        mode=projection_mode,
        tau=projection_tau,
        return_details=True,
    )
    projected_stress = stress_voigt_from_principal_numpy(projected_principal, checkpoint_eigvecs).astype(np.float32)
    projected_stress[elastic_mask] = trial_stress[elastic_mask]
    projected_principal[elastic_mask] = trial_principal[elastic_mask]

    branch_probabilities = teacher_pred.get("branch_probabilities")
    if branch_probabilities is None:
        branch_probabilities = np.full((strain_eng.shape[0], len(BRANCH_NAMES)), np.nan, dtype=np.float32)
    else:
        branch_probabilities = np.asarray(branch_probabilities, dtype=np.float32)
    predicted_branch_id = teacher_pred.get("predicted_branch_id")
    if predicted_branch_id is None:
        predicted_branch_id = np.full(strain_eng.shape[0], -1, dtype=np.int64)
    else:
        predicted_branch_id = np.asarray(predicted_branch_id, dtype=np.int64)

    disp_principal = projected_principal.astype(np.float32) - dispatch_principal.astype(np.float32)
    cache_arrays = {
        "split_id": split_id,
        "source_call_id": source_call_id,
        "source_row_in_call": source_row_in_call,
        "branch_id": branch_id,
        "hard_mask": np.asarray(panel_all["hard_mask"], dtype=np.int8),
        "plastic_mask": np.asarray(panel_all["plastic_mask"], dtype=np.int8),
        "ds_valid_mask": np.asarray(panel_all["ds_valid_mask"], dtype=np.int8),
        "near_yield_mask": np.asarray(panel_all["near_yield_mask"], dtype=np.int8),
        "near_smooth_left_mask": np.asarray(panel_all["near_smooth_left_mask"], dtype=np.int8),
        "near_smooth_right_mask": np.asarray(panel_all["near_smooth_right_mask"], dtype=np.int8),
        "near_left_apex_mask": np.asarray(panel_all["near_left_apex_mask"], dtype=np.int8),
        "near_right_apex_mask": np.asarray(panel_all["near_right_apex_mask"], dtype=np.int8),
        "exact_stress": exact_stress,
        "exact_stress_principal": exact_stress_principal,
        "trial_principal": trial_principal,
        "teacher_checkpoint_stress": checkpoint_stress.astype(np.float32),
        "teacher_checkpoint_stress_principal": checkpoint_principal.astype(np.float32),
        "teacher_dispatch_stress": dispatch_stress.astype(np.float32),
        "teacher_dispatch_stress_principal": dispatch_principal.astype(np.float32),
        "teacher_projected_stress": projected_stress.astype(np.float32),
        "teacher_projected_stress_principal": projected_principal.astype(np.float32),
        "teacher_projection_delta_principal": disp_principal.astype(np.float32),
        "teacher_projection_disp_principal": disp_principal.astype(np.float32),
        "teacher_projection_disp_norm": projection_details.displacement_norm.astype(np.float32),
        "teacher_projection_candidate_id": projection_details.selected_index.astype(np.int8),
        "teacher_projection_candidate_principal": projection_details.candidate_principal.astype(np.float32),
        "teacher_projection_candidate_sq_distance": projection_details.candidate_sq_distance.astype(np.float32),
        "teacher_projection_feasible_mask": projection_details.feasible_mask.astype(np.int8),
        "teacher_branch_probabilities": branch_probabilities.astype(np.float32),
        "teacher_predicted_branch_id": predicted_branch_id.astype(np.int64),
    }
    return cache_arrays


def project_teacher_checkpoint_stress(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    checkpoint_stress: np.ndarray,
    *,
    mode: str = "exact",
    tau: float = 0.05,
) -> dict[str, np.ndarray]:
    strain_eng = np.asarray(strain_eng, dtype=np.float32)
    material_reduced = np.asarray(material_reduced, dtype=np.float32)
    checkpoint_stress = np.asarray(checkpoint_stress, dtype=np.float32)

    trial_stress = compute_trial_stress(strain_eng, material_reduced).astype(np.float32)
    trial_principal = exact_trial_principal_from_strain(strain_eng, material_reduced).astype(np.float32)
    _strain_principal, _eigvecs = spectral_decomposition_from_strain(strain_eng)
    checkpoint_principal, checkpoint_eigvecs = principal_and_eigvecs_from_stress(checkpoint_stress)

    plastic_mask = (
        (1.0 + material_reduced[:, 1]) * trial_principal[:, 0]
        - (1.0 - material_reduced[:, 1]) * trial_principal[:, 2]
        - material_reduced[:, 0]
    ) > 0.0
    elastic_mask = ~plastic_mask

    dispatch_stress = checkpoint_stress.copy()
    dispatch_principal = checkpoint_principal.copy()
    dispatch_stress[elastic_mask] = trial_stress[elastic_mask]
    dispatch_principal[elastic_mask] = trial_principal[elastic_mask]

    projected_principal, projection_details = project_mc_principal_numpy(
        dispatch_principal,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        mode=mode,
        tau=tau,
        return_details=True,
    )
    projected_stress = stress_voigt_from_principal_numpy(projected_principal, checkpoint_eigvecs).astype(np.float32)
    projected_stress[elastic_mask] = trial_stress[elastic_mask]
    projected_principal[elastic_mask] = trial_principal[elastic_mask]

    return {
        "trial_stress": trial_stress.astype(np.float32),
        "trial_principal": trial_principal.astype(np.float32),
        "teacher_dispatch_stress": dispatch_stress.astype(np.float32),
        "teacher_dispatch_stress_principal": dispatch_principal.astype(np.float32),
        "teacher_projected_stress": projected_stress.astype(np.float32),
        "teacher_projected_stress_principal": projected_principal.astype(np.float32),
        "teacher_projection_delta_principal": (projected_principal - dispatch_principal).astype(np.float32),
        "teacher_projection_disp_norm": projection_details.displacement_norm.astype(np.float32),
        "teacher_projection_candidate_id": projection_details.selected_index.astype(np.int8),
    }


__all__ = [
    "PROJECTION_CANDIDATE_NAMES",
    "build_sampling_weights",
    "build_teacher_projection_cache_arrays",
    "directional_error_arrays",
    "principal_and_eigvecs_from_stress",
    "project_teacher_checkpoint_stress",
    "projection_switch_mask",
    "quantile_or_zero",
    "summarize_array",
]
