"""Helpers for the March 27 projection-student preservation/compression packet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

BOUNDARY_MASK_KEYS = (
    "near_yield_mask",
    "near_smooth_left_mask",
    "near_smooth_right_mask",
    "near_left_apex_mask",
    "near_right_apex_mask",
)


def build_any_boundary_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    n_rows = None
    any_boundary = None
    for key in BOUNDARY_MASK_KEYS:
        if key not in arrays:
            continue
        mask = np.asarray(arrays[key], dtype=bool).reshape(-1)
        if n_rows is None:
            n_rows = mask.shape[0]
            any_boundary = np.zeros(n_rows, dtype=bool)
        any_boundary |= mask
    if any_boundary is None:
        return np.zeros(0, dtype=bool)
    return any_boundary


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
    any_boundary_mask: np.ndarray | None = None,
    branch_id: np.ndarray | None = None,
    hard_multiplier: float = 1.5,
    edge_candidate_multiplier: float = 1.5,
    high_disp_multiplier: float = 1.5,
    boundary_multiplier: float = 1.0,
    candidate_weight_map: Mapping[int, float] | None = None,
    branch_weight_map: Mapping[int, float] | None = None,
    high_disp_quantile: float = 0.90,
    high_disp_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    hard = np.asarray(hard_mask, dtype=bool).reshape(-1)
    candidate_id = np.asarray(teacher_projection_candidate_id, dtype=np.int64).reshape(-1)
    disp = np.asarray(teacher_projection_disp_norm, dtype=float).reshape(-1)
    any_boundary = (
        np.asarray(any_boundary_mask, dtype=bool).reshape(-1)
        if any_boundary_mask is not None
        else np.zeros(candidate_id.shape[0], dtype=bool)
    )
    branch = (
        np.asarray(branch_id, dtype=np.int64).reshape(-1)
        if branch_id is not None
        else np.full(candidate_id.shape[0], -1, dtype=np.int64)
    )

    train_disp = disp[train & np.isfinite(disp)]
    disp_threshold = float(high_disp_threshold) if high_disp_threshold is not None else quantile_or_zero(train_disp, high_disp_quantile)

    weights = np.ones(candidate_id.shape[0], dtype=np.float32)
    high_disp_mask = np.isfinite(disp) & (disp >= disp_threshold)
    if hard_multiplier != 1.0:
        weights[hard] *= float(hard_multiplier)
    if boundary_multiplier != 1.0:
        weights[any_boundary] *= float(boundary_multiplier)
    if candidate_weight_map is not None:
        for bucket_id, multiplier in candidate_weight_map.items():
            if float(multiplier) != 1.0:
                weights[candidate_id == int(bucket_id)] *= float(multiplier)
    elif edge_candidate_multiplier != 1.0:
        edge_mask = np.isin(
            candidate_id,
            [
                PROJECTION_CANDIDATE_TO_ID["left_edge"],
                PROJECTION_CANDIDATE_TO_ID["right_edge"],
            ],
        )
        weights[edge_mask] *= float(edge_candidate_multiplier)
    if branch_weight_map is not None:
        for bucket_id, multiplier in branch_weight_map.items():
            if float(multiplier) != 1.0:
                weights[branch == int(bucket_id)] *= float(multiplier)
    if high_disp_multiplier != 1.0:
        weights[high_disp_mask] *= float(high_disp_multiplier)
    return weights.astype(np.float32), float(disp_threshold)


def principal_abs_error_arrays(
    pred_principal: np.ndarray,
    true_principal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_principal, dtype=np.float32)
    true = np.asarray(true_principal, dtype=np.float32)
    abs_error = np.abs(pred - true).astype(np.float32)
    max_abs_error = np.max(abs_error, axis=1).astype(np.float32)
    return abs_error, max_abs_error


def displacement_decile_edges(
    values: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    n_bins: int = 10,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if reference_mask is not None:
        ref = arr[np.asarray(reference_mask, dtype=bool).reshape(-1)]
    else:
        ref = arr
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    edges = np.quantile(ref, quantiles).astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for idx in range(1, edges.shape[0] - 1):
        prev = edges[idx - 1]
        if edges[idx] <= prev:
            edges[idx] = np.nextafter(prev, np.inf)
    return edges


def assign_quantile_bins(
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite_fill = np.nanmin(edges[np.isfinite(edges)]) if np.any(np.isfinite(edges)) else 0.0
    safe = np.where(np.isfinite(arr), arr, finite_fill)
    bins = np.digitize(safe, edges[1:-1], right=True)
    return np.clip(bins.astype(np.int8), 0, max(edges.shape[0] - 2, 0))


def build_top_fraction_mask(
    values: np.ndarray,
    *,
    pool_mask: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    pool = np.asarray(pool_mask, dtype=bool).reshape(-1)
    out = np.zeros(arr.shape[0], dtype=bool)
    pool_values = arr[pool & np.isfinite(arr)]
    if pool_values.size == 0:
        return out, float("nan")
    threshold = float(np.quantile(pool_values, max(0.0, 1.0 - float(fraction))))
    out = pool & np.isfinite(arr) & (arr >= threshold)
    return out, threshold


def build_control_zero_rowwise_arrays(
    split_arrays: Mapping[str, np.ndarray],
    control_predictions: Mapping[str, np.ndarray],
    *,
    disp_decile_edges: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | list[float]]]:
    exact = np.asarray(split_arrays["exact_stress_principal"], dtype=np.float32)
    control = np.asarray(control_predictions["stress_principal"], dtype=np.float32)
    control_abs_error, control_max_abs_error = principal_abs_error_arrays(control, exact)
    any_boundary_mask = build_any_boundary_mask(split_arrays)
    disp_norm = np.asarray(split_arrays["teacher_projection_disp_norm"], dtype=np.float32)
    if disp_decile_edges is None:
        disp_decile_edges = displacement_decile_edges(
            disp_norm,
            reference_mask=np.asarray(split_arrays["plastic_mask"], dtype=bool),
            n_bins=10,
        )
    disp_decile = assign_quantile_bins(disp_norm, disp_decile_edges)
    provisional = control_predictions.get("provisional_stress_principal")
    predicted_branch = control_predictions.get("predicted_branch_id")
    if provisional is None:
        provisional = np.full_like(control, np.nan, dtype=np.float32)
    else:
        provisional = np.asarray(provisional, dtype=np.float32)
    if predicted_branch is None:
        predicted_branch = np.full(control.shape[0], -1, dtype=np.int64)
    else:
        predicted_branch = np.asarray(predicted_branch, dtype=np.int64).reshape(-1)

    rowwise = {
        "split_id": np.asarray(split_arrays["split_id"], dtype=np.int8),
        "source_call_id": np.asarray(split_arrays["source_call_id"], dtype=np.int32),
        "source_row_in_call": np.asarray(split_arrays["source_row_in_call"], dtype=np.int32),
        "branch_id": np.asarray(split_arrays["branch_id"], dtype=np.int8),
        "hard_mask": np.asarray(split_arrays["hard_mask"], dtype=np.int8),
        "plastic_mask": np.asarray(split_arrays["plastic_mask"], dtype=np.int8),
        "ds_valid_mask": np.asarray(split_arrays["ds_valid_mask"], dtype=np.int8),
        "teacher_projection_candidate_id": np.asarray(split_arrays["teacher_projection_candidate_id"], dtype=np.int8),
        "teacher_projection_disp_norm": disp_norm.astype(np.float32),
        "teacher_projection_disp_decile": disp_decile.astype(np.int8),
        "any_boundary_mask": any_boundary_mask.astype(np.int8),
        "exact_stress_principal": exact.astype(np.float32),
        "teacher_projected_stress_principal": np.asarray(split_arrays["teacher_projected_stress_principal"], dtype=np.float32),
        "control_zero_projected_stress_principal": control.astype(np.float32),
        "control_zero_provisional_stress_principal": provisional.astype(np.float32),
        "control_zero_predicted_branch_id": predicted_branch.astype(np.int64),
        "control_zero_principal_abs_error": control_abs_error.astype(np.float32),
        "control_zero_principal_max_abs_error": control_max_abs_error.astype(np.float32),
    }
    for key in BOUNDARY_MASK_KEYS:
        rowwise[key] = np.asarray(split_arrays.get(key, np.zeros(control.shape[0], dtype=np.int8)), dtype=np.int8)
    meta = {
        "teacher_projection_disp_decile_edges": [float(x) for x in np.asarray(disp_decile_edges, dtype=float)],
    }
    return rowwise, meta


def build_slice_summary_rows(
    *,
    scope_name: str,
    rowwise_arrays: Mapping[str, np.ndarray],
    group_name: str,
    group_values: np.ndarray,
    labels: Mapping[int, str] | Sequence[str],
    base_mask: np.ndarray,
    hard_top_fraction_masks: Mapping[str, np.ndarray],
    error_key: str = "control_zero_principal_max_abs_error",
) -> list[dict[str, float | int | str]]:
    values = np.asarray(group_values)
    base = np.asarray(base_mask, dtype=bool).reshape(-1)
    hard = np.asarray(rowwise_arrays["hard_mask"], dtype=bool).reshape(-1)
    max_abs = np.asarray(rowwise_arrays[error_key], dtype=float).reshape(-1)
    hard_base = base & hard
    hard_base_count = int(np.sum(hard_base))

    if isinstance(labels, Mapping):
        label_map = {int(key): str(value) for key, value in labels.items()}
    else:
        label_map = {idx: str(value) for idx, value in enumerate(labels)}

    rows: list[dict[str, float | int | str]] = []
    for bucket_id, bucket_label in label_map.items():
        mask = base & (values == bucket_id)
        if not np.any(mask):
            continue
        hard_count = int(np.sum(mask & hard))
        row = {
            "scope": scope_name,
            "group_name": group_name,
            "group_value": bucket_label,
            "group_id": int(bucket_id),
            "n_rows": int(np.sum(mask)),
            "n_hard_rows": hard_count,
            "hard_base_share": float(hard_count / max(hard_base_count, 1)),
            "mean_principal_max_abs_error": float(np.mean(max_abs[mask])),
            "p95_principal_max_abs_error": quantile_or_zero(max_abs[mask], 0.95),
            "p99_principal_max_abs_error": quantile_or_zero(max_abs[mask], 0.99),
        }
        for suffix, top_mask in hard_top_fraction_masks.items():
            denom = int(np.sum(top_mask))
            numer = int(np.sum(mask & top_mask))
            row[f"hard_top{suffix}_count"] = numer
            row[f"hard_top{suffix}_share"] = float(numer / max(denom, 1))
            base_share = float(hard_count / max(hard_base_count, 1))
            row[f"hard_top{suffix}_over_index"] = float((numer / max(denom, 1)) / max(base_share, 1.0e-12))
        rows.append(row)
    return rows


def build_call_concentration_rows(
    *,
    scope_name: str,
    source_call_id: np.ndarray,
    base_mask: np.ndarray,
    hard_mask: np.ndarray,
    principal_max_abs_error: np.ndarray,
    hard_top_mask: np.ndarray,
    top_n: int = 20,
) -> list[dict[str, float | int | str]]:
    call_id = np.asarray(source_call_id, dtype=np.int64).reshape(-1)
    base = np.asarray(base_mask, dtype=bool).reshape(-1)
    hard = np.asarray(hard_mask, dtype=bool).reshape(-1)
    error = np.asarray(principal_max_abs_error, dtype=float).reshape(-1)
    top_mask = np.asarray(hard_top_mask, dtype=bool).reshape(-1)
    valid = base & (call_id >= 0)
    if not np.any(valid):
        return []
    total_top = int(np.sum(top_mask & valid))
    rows: list[dict[str, float | int | str]] = []
    for bucket in np.unique(call_id[valid]):
        call_mask = valid & (call_id == bucket)
        hard_rows = call_mask & hard
        if not np.any(hard_rows):
            continue
        top_rows = call_mask & top_mask
        rows.append(
            {
                "scope": scope_name,
                "source_call_id": int(bucket),
                "n_rows": int(np.sum(call_mask)),
                "n_hard_rows": int(np.sum(hard_rows)),
                "hard_top5_count": int(np.sum(top_rows)),
                "hard_top5_share": float(np.sum(top_rows) / max(total_top, 1)),
                "mean_principal_max_abs_error": float(np.mean(error[call_mask])),
                "p95_principal_max_abs_error": quantile_or_zero(error[call_mask], 0.95),
                "p99_principal_max_abs_error": quantile_or_zero(error[call_mask], 0.99),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["hard_top5_share"]),
            -int(row["hard_top5_count"]),
            -float(row["p95_principal_max_abs_error"]),
        )
    )
    return rows[:top_n]


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
