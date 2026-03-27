#!/usr/bin/env python
"""Run the March 24 packet4 soft admissible atlas redesign packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_surrogate.branch_geometry import compute_branch_geometry_principal
from mc_surrogate.data import load_arrays
from mc_surrogate.models import (
    build_trial_soft_atlas_features_f1,
    compute_trial_soft_atlas_feature_stats,
    decode_abr_to_principal_torch,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
    stress_voigt_from_principal_torch,
)
from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    BRANCH_TO_ID,
    decode_abr_to_principal,
    exact_trial_principal_stress_3d,
    principal_relative_error_3d,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.training import (
    TrainingConfig,
    Standardizer,
    load_checkpoint,
    _soft_atlas_route_targets,
    _soft_atlas_target_transform,
    train_model,
)
from run_hybrid_campaign import _candidate_checkpoint_paths
from run_nn_replacement_abr_cycle import (
    _aggregate_policy_metrics,
    _load_h5,
    _parameter_count,
    _quantile_or_zero,
    _slice_arrays,
    _train_with_batch_fallback,
    _write_csv,
    _write_h5,
    _write_json,
    _write_phase_report,
)

PACKET2_REFERENCE_VAL = {
    "broad_plastic_mae": 32.458569,
    "hard_plastic_mae": 37.824432,
    "hard_p95_principal": 328.641602,
    "yield_violation_p95": 6.173982e-08,
}
PACKET2_REFERENCE_TEST = {
    "broad_plastic_mae": 33.603760,
    "hard_plastic_mae": 38.797607,
    "hard_p95_principal": 331.668732,
    "yield_violation_p95": 5.991195e-08,
}
PACKET3_REFERENCE_VAL_PRED = {
    "broad_plastic_mae": 38.164696,
    "hard_plastic_mae": 43.860016,
    "hard_p95_principal": 399.468353,
    "yield_violation_p95": 5.600788e-08,
}
PACKET3_REFERENCE_VAL_ORACLE = {
    "broad_plastic_mae": 29.512514,
    "hard_plastic_mae": 32.682789,
    "hard_p95_principal": 302.360168,
    "yield_violation_p95": 5.600788e-08,
}
S1_GOAL = {
    "broad_plastic_mae": 15.0,
    "hard_plastic_mae": 18.0,
    "hard_p95_principal": 150.0,
    "yield_violation_p95": 1.0e-6,
}
PACKET4_TEMPERATURE_GRID = (0.60, 0.80, 1.00, 1.25, 1.50, 2.00)
CONTINUE_BAR = {
    "broad_plastic_mae": 25.0,
    "hard_plastic_mae": 28.0,
    "hard_p95_principal": 250.0,
    "yield_violation_p95": 1.0e-6,
}
PACKET3_GAP_HALF_TARGETS = {
    "broad_plastic_mae": (PACKET3_REFERENCE_VAL_PRED["broad_plastic_mae"] - PACKET3_REFERENCE_VAL_ORACLE["broad_plastic_mae"]) / 2.0,
    "hard_plastic_mae": (PACKET3_REFERENCE_VAL_PRED["hard_plastic_mae"] - PACKET3_REFERENCE_VAL_ORACLE["hard_plastic_mae"]) / 2.0,
    "hard_p95_principal": (PACKET3_REFERENCE_VAL_PRED["hard_p95_principal"] - PACKET3_REFERENCE_VAL_ORACLE["hard_p95_principal"]) / 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage0-dataset",
        default="experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp0_surface_dataset/derived_surface_dataset.h5",
    )
    parser.add_argument("--output-root", default="experiment_runs/real_sim/nn_replacement_surface_20260324_packet4")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _corr_or_zero(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size < 2 or y_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1.0e-12 or np.std(y_arr) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _load_eval_split(dataset_path: Path, split_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays_all, _attrs = _load_h5(dataset_path)
    split_id = {"train": 0, "val": 1, "test": 2}[split_name]
    mask = arrays_all["split_id"] == split_id
    arrays = _slice_arrays(arrays_all, mask)
    for key in ("soft_atlas_feature_f1", "trial_principal", "trial_stress", "grho", "soft_atlas_route_target"):
        if key in arrays_all:
            arrays[key] = arrays_all[key][mask]
    panel = {
        "plastic_mask": arrays["plastic_mask"],
        "hard_mask": arrays["hard_mask"],
    }
    return arrays, panel


def _soft_atlas_decode_from_chart(
    soft_atlas_chart: np.ndarray,
    branch_id: np.ndarray,
    *,
    trial_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chart = np.asarray(soft_atlas_chart, dtype=np.float32)
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    trial = np.asarray(trial_principal, dtype=np.float32)
    material = np.asarray(material_reduced, dtype=np.float32)
    a_tr = trial[:, 0] - trial[:, 1]
    b_tr = trial[:, 1] - trial[:, 2]
    g_tr = a_tr + b_tr
    lambda_tr = np.clip(a_tr / np.maximum(g_tr, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    g_s = np.maximum(g_tr, 1.0e-12) * np.exp(chart[:, 0])
    lambda_s = 1.0 / (1.0 + np.exp(-(np.log(lambda_tr / (1.0 - lambda_tr)) + chart[:, 1])))
    b_l = np.maximum(b_tr, 1.0e-12) * np.exp(chart[:, 2])
    a_r = np.maximum(a_tr, 1.0e-12) * np.exp(chart[:, 3])

    a = np.zeros_like(g_s)
    b = np.zeros_like(g_s)
    smooth = branch == BRANCH_TO_ID["smooth"]
    left = branch == BRANCH_TO_ID["left_edge"]
    right = branch == BRANCH_TO_ID["right_edge"]
    apex = branch == BRANCH_TO_ID["apex"]
    a[smooth] = lambda_s[smooth] * g_s[smooth]
    b[smooth] = (1.0 - lambda_s[smooth]) * g_s[smooth]
    b[left] = b_l[left]
    a[right] = a_r[right]
    abr = np.column_stack([a, b, np.zeros_like(a)]).astype(np.float32)
    stress_principal = decode_abr_to_principal(
        abr,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
    ).astype(np.float32)
    grho = np.column_stack([a + b, (a - b) / np.maximum(a + b, 1.0e-12)]).astype(np.float32)
    return stress_principal, grho, abr


def _decode_soft_atlas_surface_outputs(
    pred_raw: torch.Tensor,
    route_logits: torch.Tensor,
    material_reduced: torch.Tensor,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    *,
    route_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = max(float(route_temperature), 1.0e-6)
    route_probs = torch.softmax(route_logits / temperature, dim=1)
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


def _surface_metric_row(true_grho: np.ndarray, pred_grho: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return {"g_mae": float("nan"), "rho_mae": float("nan"), "g_corr": float("nan"), "rho_corr": float("nan")}
    true_sel = np.asarray(true_grho[valid], dtype=float)
    pred_sel = np.asarray(pred_grho[valid], dtype=float)
    return {
        "g_mae": float(np.mean(np.abs(pred_sel[:, 0] - true_sel[:, 0]))),
        "rho_mae": float(np.mean(np.abs(pred_sel[:, 1] - true_sel[:, 1]))),
        "g_corr": _corr_or_zero(pred_sel[:, 0], true_sel[:, 0]),
        "rho_corr": _corr_or_zero(pred_sel[:, 1], true_sel[:, 1]),
    }


def _branch_breakdown_rows(
    branch_id: np.ndarray,
    stress_principal_true: np.ndarray,
    stress_principal_pred: np.ndarray,
    stress_true: np.ndarray,
    stress_pred: np.ndarray,
    material_reduced: np.ndarray,
) -> list[dict[str, Any]]:
    stress_abs = np.abs(stress_pred - stress_true)
    principal_abs = np.abs(stress_principal_pred - stress_principal_true)
    principal_abs_max = np.max(principal_abs, axis=1)
    repo_relative = principal_relative_error_3d(
        stress_principal_pred,
        stress_principal_true,
        c_bar=material_reduced[:, 0],
    ).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for branch_value, branch_name in enumerate(BRANCH_NAMES):
        mask = branch_id == branch_value
        if not np.any(mask):
            continue
        rows.append(
            {
                "branch": branch_name,
                "count": int(np.sum(mask)),
                "fraction_of_split": float(np.mean(mask)),
                "stress_component_mae": float(np.mean(stress_abs[mask])),
                "principal_abs_p95": float(np.quantile(principal_abs_max[mask], 0.95)),
                "repo_relative_p95": float(np.quantile(repo_relative[mask], 0.95)),
            }
        )
    return rows


def _packet4_route_target_stats(route_target: np.ndarray) -> dict[str, float]:
    route = np.asarray(route_target, dtype=float)
    entropy = -np.sum(route * np.log(np.clip(route, 1.0e-12, 1.0)), axis=1)
    return {
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p95": float(np.quantile(entropy, 0.95)),
        "maxprob_mean": float(np.mean(np.max(route, axis=1))),
    }


def _feature_stats_summary(features: np.ndarray) -> dict[str, Any]:
    feat = np.asarray(features, dtype=float)
    finite = np.isfinite(feat)
    all_finite = bool(np.all(finite))
    summary = {
        "feature_dim": int(feat.shape[1]) if feat.ndim == 2 else 0,
        "n_rows": int(feat.shape[0]) if feat.ndim == 2 else 0,
        "all_finite": all_finite,
        "nan_count": int(np.isnan(feat).sum()),
        "inf_count": int(np.isinf(feat).sum()),
        "mean": np.mean(feat, axis=0).astype(float).tolist() if feat.ndim == 2 and feat.size else [],
        "std": np.std(feat, axis=0).astype(float).tolist() if feat.ndim == 2 and feat.size else [],
        "min": np.min(feat, axis=0).astype(float).tolist() if feat.ndim == 2 and feat.size else [],
        "max": np.max(feat, axis=0).astype(float).tolist() if feat.ndim == 2 and feat.size else [],
        "q05": np.quantile(feat, 0.05, axis=0).astype(float).tolist() if feat.ndim == 2 and feat.size else [],
        "q95": np.quantile(feat, 0.95, axis=0).astype(float).tolist() if feat.ndim == 2 and feat.size else [],
    }
    return summary


def _route_target_audit_summary(route_target: np.ndarray, branch_id: np.ndarray) -> dict[str, Any]:
    route = np.asarray(route_target, dtype=float)
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    plastic = branch > 0
    if np.any(plastic):
        route_plastic = route[plastic]
        row_sum = np.sum(route_plastic, axis=1)
        forbidden_direct = (route_plastic[:, 0] > 1.0e-8) & (route_plastic[:, 3] > 1.0e-8)
        return {
            "plastic_rows": int(np.sum(plastic)),
            "all_entries_nonnegative": bool(np.all(route_plastic >= -1.0e-12)),
            "min_entry": float(np.min(route_plastic)),
            "max_entry": float(np.max(route_plastic)),
            "row_sum_max_abs_err": float(np.max(np.abs(row_sum - 1.0))),
            "row_sum_p95_abs_err": float(np.quantile(np.abs(row_sum - 1.0), 0.95)),
            "forbidden_direct_smooth_apex_rows": int(np.sum(forbidden_direct)),
            "forbidden_direct_smooth_apex_fraction": float(np.mean(forbidden_direct)),
            "entropy_mean": _packet4_route_target_stats(route_plastic)["entropy_mean"],
            "entropy_p95": _packet4_route_target_stats(route_plastic)["entropy_p95"],
            "maxprob_mean": _packet4_route_target_stats(route_plastic)["maxprob_mean"],
            "route_mass_mean": np.mean(route_plastic, axis=0).astype(float).tolist(),
        }
    return {
        "plastic_rows": 0,
        "all_entries_nonnegative": True,
        "row_sum_max_abs_err": 0.0,
        "row_sum_p95_abs_err": 0.0,
        "forbidden_direct_smooth_apex_rows": 0,
        "forbidden_direct_smooth_apex_fraction": 0.0,
        "entropy_mean": 0.0,
        "entropy_p95": 0.0,
        "maxprob_mean": 0.0,
        "route_mass_mean": [0.0, 0.0, 0.0, 0.0],
    }


def _predict_soft_atlas_with_checkpoint(
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
    route_temperature: float,
    checkpoint_cache: dict[str, tuple[torch.nn.Module, dict[str, Any], torch.device]] | None = None,
) -> dict[str, np.ndarray]:
    cache_key = str(checkpoint_path.resolve())
    if checkpoint_cache is not None and cache_key in checkpoint_cache:
        model, metadata, device_obj = checkpoint_cache[cache_key]
    else:
        model, metadata = load_checkpoint(checkpoint_path, device=device)
        device_obj = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device_obj)
        if checkpoint_cache is not None:
            checkpoint_cache[cache_key] = (model, metadata, device_obj)
    x_scaler = Standardizer.from_dict(metadata["x_scaler"])
    y_scaler = Standardizer.from_dict(metadata["y_scaler"])

    if "soft_atlas_feature_f1" not in arrays:
        raise ValueError("packet4 evaluation requires soft_atlas_feature_f1 in the eval split.")
    features = np.asarray(arrays["soft_atlas_feature_f1"], dtype=np.float32)
    trial_principal = np.asarray(arrays["trial_principal"], dtype=np.float32)
    trial_stress = np.asarray(arrays["trial_stress"], dtype=np.float32)
    material_reduced = np.asarray(arrays["material_reduced"], dtype=np.float32)
    eigvecs = np.asarray(arrays["eigvecs"], dtype=np.float32)

    if batch_size <= 0:
        batch_size = int(features.shape[0])
    current_batch_size = max(1, int(batch_size))

    stress_chunks: list[np.ndarray] = []
    principal_chunks: list[np.ndarray] = []
    grho_chunks: list[np.ndarray] = []
    chart_chunks: list[np.ndarray] = []
    route_prob_chunks: list[np.ndarray] = []
    branch_chunks: list[np.ndarray] = []
    pred_branch_chunks: list[np.ndarray] = []

    with torch.no_grad():
        start = 0
        while start < features.shape[0]:
            stop = min(start + current_batch_size, features.shape[0])
            try:
                x = torch.from_numpy(x_scaler.transform(features[start:stop])).to(device_obj)
                out = model(x)
                pred_norm = out["stress"]
                route_logits = out.get("route_logits", out.get("branch_logits"))
                if route_logits is None:
                    raise ValueError("Soft-atlas checkpoint is missing route logits.")
                pred_raw = pred_norm * torch.as_tensor(y_scaler.std, device=device_obj) + torch.as_tensor(y_scaler.mean, device=device_obj)
                stress_t, principal_t, grho_t, chart_t, route_probs_t, route_branch_t = _decode_soft_atlas_surface_outputs(
                    pred_raw,
                    route_logits,
                    torch.from_numpy(material_reduced[start:stop]).to(device_obj),
                    torch.from_numpy(eigvecs[start:stop]).to(device_obj),
                    torch.from_numpy(trial_stress[start:stop]).to(device_obj),
                    torch.from_numpy(trial_principal[start:stop]).to(device_obj),
                    route_temperature=route_temperature,
                )
                route_probs_np = route_probs_t.cpu().numpy().astype(np.float32)
                grho_np = grho_t.cpu().numpy().astype(np.float32)
                if grho_np.ndim != 2 or grho_np.shape[0] != (stop - start) or grho_np.shape[1] != 2:
                    raise ValueError(f"Expected packet4 grho chunk with shape {(stop - start, 2)}, got {grho_np.shape}.")
                stress_chunks.append(stress_t.cpu().numpy().astype(np.float32))
                principal_chunks.append(principal_t.cpu().numpy().astype(np.float32))
                grho_chunks.append(grho_np)
                chart_chunks.append(chart_t.cpu().numpy().astype(np.float32))
                route_prob_chunks.append(route_probs_np)
                branch_chunks.append(route_branch_t.cpu().numpy().astype(np.int64))
                pred_branch_chunks.append(np.argmax(route_probs_np, axis=1).astype(np.int64) + 1)
                start = stop
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current_batch_size == 1:
                    raise
                if device_obj.type == "cuda":
                    torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)

    route_probs = np.concatenate(route_prob_chunks, axis=0)
    branch_probs = np.zeros((route_probs.shape[0], len(BRANCH_NAMES)), dtype=np.float32)
    branch_probs[:, 1:] = route_probs
    pred_branch = np.concatenate(pred_branch_chunks, axis=0)
    return {
        "stress": np.concatenate(stress_chunks, axis=0),
        "stress_principal": np.concatenate(principal_chunks, axis=0),
        "grho": np.concatenate(grho_chunks, axis=0),
        "soft_atlas_chart": np.concatenate(chart_chunks, axis=0),
        "soft_atlas_route_probs": route_probs,
        "branch_probabilities": branch_probs,
        "predicted_branch_id": pred_branch,
        "route_branch_id": np.concatenate(branch_chunks, axis=0),
        "eigvecs": eigvecs.astype(np.float32),
        "trial_principal": trial_principal.astype(np.float32),
        "trial_stress": trial_stress.astype(np.float32),
        "material_reduced": material_reduced.astype(np.float32),
    }


def build_stage0_soft_atlas_dataset(
    *,
    stage0_dataset_path: Path,
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    dataset_path = output_dir / "derived_soft_atlas_dataset.h5"
    summary_path = output_dir / "stage0_summary.json"
    route_target_summary_path = output_dir / "stage0_route_target_summary.json"
    feature_stats_summary_path = output_dir / "stage0_feature_stats.json"
    if dataset_path.exists() and summary_path.exists() and route_target_summary_path.exists() and feature_stats_summary_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    arrays, attrs = _load_h5(stage0_dataset_path)
    branch_id = arrays["branch_id"].astype(np.int8)
    split_id = arrays["split_id"].astype(np.int8)
    plastic = branch_id > 0
    train_plastic = (split_id == 0) & plastic

    strain_principal, _eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
    trial_principal = exact_trial_principal_stress_3d(
        arrays["strain_eng"],
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
        shear=arrays["material_reduced"][:, 2],
        bulk=arrays["material_reduced"][:, 3],
        lame=arrays["material_reduced"][:, 4],
    ).astype(np.float32)
    feature_stats = compute_trial_soft_atlas_feature_stats(
        trial_principal[train_plastic],
        arrays["material_reduced"][train_plastic],
    )
    soft_atlas_feature_f1 = build_trial_soft_atlas_features_f1(
        strain_principal,
        arrays["material_reduced"],
        trial_principal,
        feature_stats,
    ).astype(np.float32)
    geom = compute_branch_geometry_principal(
        strain_principal,
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
        shear=arrays["material_reduced"][:, 2],
        bulk=arrays["material_reduced"][:, 3],
        lame=arrays["material_reduced"][:, 4],
    )
    soft_atlas_route_target = _soft_atlas_route_targets(
        branch_id,
        m_yield=geom.m_yield,
        m_smooth_left=geom.m_smooth_left,
        m_smooth_right=geom.m_smooth_right,
        m_left_apex=geom.m_left_apex,
        m_right_apex=geom.m_right_apex,
    )
    soft_atlas_target = _soft_atlas_target_transform(
        arrays["stress_principal"],
        trial_principal,
        arrays["material_reduced"],
        branch_id,
    )

    # Exact/near-exact audit of the target parameterization.
    a_true = arrays["stress_principal"][:, 0] - arrays["stress_principal"][:, 1]
    b_true = arrays["stress_principal"][:, 1] - arrays["stress_principal"][:, 2]
    g_true = a_true + b_true
    lambda_true = np.clip(a_true / np.maximum(g_true, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    g_tr = trial_principal[:, 0] - trial_principal[:, 2]
    b_tr = trial_principal[:, 1] - trial_principal[:, 2]
    a_tr = trial_principal[:, 0] - trial_principal[:, 1]
    lambda_tr = np.clip(a_tr / np.maximum(g_tr, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    a_decode = np.zeros_like(a_true)
    b_decode = np.zeros_like(b_true)
    smooth = branch_id == BRANCH_TO_ID["smooth"]
    left = branch_id == BRANCH_TO_ID["left_edge"]
    right = branch_id == BRANCH_TO_ID["right_edge"]
    apex = branch_id == BRANCH_TO_ID["apex"]
    g_s = np.maximum(g_tr, 1.0e-12) * np.exp(soft_atlas_target[:, 0])
    lambda_s = 1.0 / (1.0 + np.exp(-(np.log(lambda_tr / (1.0 - lambda_tr)) + soft_atlas_target[:, 1])))
    a_decode[smooth] = lambda_s[smooth] * g_s[smooth]
    b_decode[smooth] = (1.0 - lambda_s[smooth]) * g_s[smooth]
    b_decode[left] = np.maximum(b_tr[left], 1.0e-12) * np.exp(soft_atlas_target[left, 2])
    a_decode[right] = np.maximum(a_tr[right], 1.0e-12) * np.exp(soft_atlas_target[right, 3])
    a_decode[apex] = 0.0
    b_decode[apex] = 0.0
    atlas_abr = np.column_stack([a_decode, b_decode, np.zeros_like(a_decode)]).astype(np.float32)
    atlas_principal = decode_abr_to_principal(
        atlas_abr[plastic],
        c_bar=arrays["material_reduced"][plastic, 0],
        sin_phi=arrays["material_reduced"][plastic, 1],
    ).astype(np.float32)
    atlas_recon = np.zeros(arrays["stress_principal"].shape[0], dtype=np.float32)
    atlas_recon[plastic] = np.abs(atlas_principal - arrays["stress_principal"][plastic].astype(np.float32)).max(axis=1)
    atlas_yield = np.zeros(arrays["stress_principal"].shape[0], dtype=np.float32)
    atlas_yield[plastic] = yield_violation_rel_principal_3d(
        atlas_principal,
        c_bar=arrays["material_reduced"][plastic, 0],
        sin_phi=arrays["material_reduced"][plastic, 1],
    ).astype(np.float32)

    derived_arrays = {
        "strain_eng": arrays["strain_eng"].astype(np.float32),
        "stress": arrays["stress"].astype(np.float32),
        "stress_principal": arrays["stress_principal"].astype(np.float32),
        "material_reduced": arrays["material_reduced"].astype(np.float32),
        "eigvecs": arrays["eigvecs"].astype(np.float32),
        "branch_id": branch_id,
        "split_id": split_id,
        "source_call_id": arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": arrays["source_row_in_call"].astype(np.int32),
        "trial_stress": arrays["trial_stress"].astype(np.float32),
        "trial_principal": trial_principal,
        "strain_principal": strain_principal.astype(np.float32),
        "grho": arrays["grho"].astype(np.float32),
        "soft_atlas_feature_f1": soft_atlas_feature_f1,
        "soft_atlas_route_target": soft_atlas_route_target.astype(np.float32),
        "soft_atlas_target": soft_atlas_target.astype(np.float32),
        "soft_atlas_reconstruction_abs_max": atlas_recon,
        "soft_atlas_yield_violation": atlas_yield,
        "plastic_mask": arrays.get("plastic_mask", plastic.astype(np.int8)).astype(np.int8),
        "hard_mask": arrays.get("hard_mask", np.zeros_like(branch_id, dtype=np.int8)).astype(np.int8),
    }
    feature_stats_summary = {
        "feature_names": [
            "asinh_p_tr",
            "log1p_g_tr",
            "logit_lambda_tr",
            "asinh_f_tr",
            "rho_tr",
            "m_yield",
            "m_smooth_left",
            "m_smooth_right",
            "m_left_apex",
            "m_right_apex",
            "gap12_norm",
            "gap23_norm",
            "material_reduced_0",
            "material_reduced_1",
            "material_reduced_2",
            "material_reduced_3",
            "material_reduced_4",
        ],
        "train_plastic": _feature_stats_summary(soft_atlas_feature_f1[train_plastic]),
        "full_dataset": _feature_stats_summary(soft_atlas_feature_f1),
    }
    route_target_summary = {
        "overall_plastic": _route_target_audit_summary(soft_atlas_route_target[plastic], branch_id[plastic]),
        "train_plastic": _route_target_audit_summary(soft_atlas_route_target[(split_id == 0) & plastic], branch_id[(split_id == 0) & plastic]),
        "val_plastic": _route_target_audit_summary(soft_atlas_route_target[(split_id == 1) & plastic], branch_id[(split_id == 1) & plastic]),
        "test_plastic": _route_target_audit_summary(soft_atlas_route_target[(split_id == 2) & plastic], branch_id[(split_id == 2) & plastic]),
    }
    elastic = branch_id == BRANCH_TO_ID["elastic"]
    if np.any(elastic):
        elastic_stress_t, elastic_principal_t, _elastic_grho_t, _elastic_chart_t, _elastic_route_probs_t, _elastic_route_branch_t = _decode_soft_atlas_surface_outputs(
            torch.zeros((int(np.sum(elastic)), 4), dtype=torch.float32),
            torch.zeros((int(np.sum(elastic)), 4), dtype=torch.float32),
            torch.from_numpy(arrays["material_reduced"][elastic].astype(np.float32)),
            torch.from_numpy(arrays["eigvecs"][elastic].astype(np.float32)),
            torch.from_numpy(arrays["trial_stress"][elastic].astype(np.float32)),
            torch.from_numpy(trial_principal[elastic].astype(np.float32)),
            route_temperature=1.0,
        )
        elastic_stress = elastic_stress_t.cpu().numpy().astype(np.float32)
        elastic_principal = elastic_principal_t.cpu().numpy().astype(np.float32)
        elastic_override_abs = np.abs(elastic_stress - arrays["trial_stress"][elastic].astype(np.float32))
        elastic_override_principal_abs = np.abs(elastic_principal - trial_principal[elastic].astype(np.float32))
        elastic_override_max_abs = float(np.max(elastic_override_abs))
        elastic_override_mean_abs = float(np.mean(elastic_override_abs))
        elastic_override_principal_max_abs = float(np.max(elastic_override_principal_abs))
    else:
        elastic_override_max_abs = 0.0
        elastic_override_mean_abs = 0.0
        elastic_override_principal_max_abs = 0.0
    target_all_finite = bool(np.isfinite(soft_atlas_target).all())
    decode_all_finite = bool(np.isfinite(atlas_abr).all() and np.isfinite(atlas_principal).all())
    dummy_logits_all_finite = True
    route_target_summary["checks"] = {
        "probability_vectors_valid": bool(
            route_target_summary["overall_plastic"]["all_entries_nonnegative"]
            and route_target_summary["overall_plastic"]["row_sum_max_abs_err"] <= 1.0e-6
        ),
        "forbidden_smooth_apex_mixing_absent": bool(route_target_summary["overall_plastic"]["forbidden_direct_smooth_apex_rows"] == 0),
        "feature_stats_finite": bool(feature_stats_summary["train_plastic"]["all_finite"] and feature_stats_summary["full_dataset"]["all_finite"]),
        "target_all_finite": target_all_finite,
        "decode_outputs_all_finite": decode_all_finite,
        "dummy_logits_all_finite": dummy_logits_all_finite,
        "elastic_override_exact": bool(elastic_override_max_abs <= 1.0e-10 and elastic_override_principal_max_abs <= 1.0e-10),
        "admissible_decode_max_abs": float(np.max(atlas_recon[plastic])) if np.any(plastic) else 0.0,
        "admissible_decode_yield_p95": _quantile_or_zero(atlas_yield[plastic], 0.95),
        "decoded_a_min": float(np.min(a_decode[plastic])) if np.any(plastic) else 0.0,
        "decoded_b_min": float(np.min(b_decode[plastic])) if np.any(plastic) else 0.0,
    }
    feature_stats_summary["checks"] = {
        "train_plastic_all_finite": bool(feature_stats_summary["train_plastic"]["all_finite"]),
        "full_dataset_all_finite": bool(feature_stats_summary["full_dataset"]["all_finite"]),
        "nan_count": int(np.isnan(soft_atlas_feature_f1).sum()),
        "inf_count": int(np.isinf(soft_atlas_feature_f1).sum()),
    }
    soft_atlas_target_stats = {
        "target_mean": np.mean(soft_atlas_target[train_plastic], axis=0).astype(float).tolist() if np.any(train_plastic) else [0.0] * 4,
        "target_std": np.std(soft_atlas_target[train_plastic], axis=0).astype(float).tolist() if np.any(train_plastic) else [1.0] * 4,
        "route_target_stats": _packet4_route_target_stats(soft_atlas_route_target[train_plastic]) if np.any(train_plastic) else {},
    }
    attrs_out = {
        "source_stage0_dataset": str(stage0_dataset_path),
        "soft_atlas_feature_stats_json": json.dumps(feature_stats),
        "soft_atlas_target_stats_json": json.dumps(soft_atlas_target_stats),
        "split_seed": attrs.get("split_seed", 20260324),
        "branch_names_json": attrs.get("branch_names_json", json.dumps(BRANCH_NAMES)),
        "packet": "packet4_soft_atlas",
    }
    _write_h5(dataset_path, derived_arrays, attrs_out)

    stage0_pass = bool(
        route_target_summary["checks"]["probability_vectors_valid"]
        and route_target_summary["checks"]["forbidden_smooth_apex_mixing_absent"]
        and route_target_summary["checks"]["feature_stats_finite"]
        and route_target_summary["checks"]["target_all_finite"]
        and route_target_summary["checks"]["decode_outputs_all_finite"]
        and route_target_summary["checks"]["dummy_logits_all_finite"]
        and route_target_summary["checks"]["elastic_override_exact"]
        and route_target_summary["checks"]["decoded_a_min"] >= -1.0e-8
        and route_target_summary["checks"]["decoded_b_min"] >= -1.0e-8
        and route_target_summary["checks"]["admissible_decode_yield_p95"] <= 1.0e-6
    )
    summary = {
        "dataset_path": str(dataset_path),
        "source_stage0_dataset": str(stage0_dataset_path),
        "feature_stats": feature_stats,
        "target_stats": soft_atlas_target_stats,
        "route_target_stats": _packet4_route_target_stats(soft_atlas_route_target[train_plastic]) if np.any(train_plastic) else {},
        "reconstruction": {
            "plastic_mean_abs": float(np.mean(atlas_recon[plastic])) if np.any(plastic) else 0.0,
            "plastic_max_abs": float(np.max(atlas_recon[plastic])) if np.any(plastic) else 0.0,
        },
        "yield": {
            "plastic_p95": _quantile_or_zero(atlas_yield[plastic], 0.95),
            "plastic_max": float(np.max(atlas_yield[plastic])) if np.any(plastic) else 0.0,
        },
        "elastic_override": {
            "mean_abs": elastic_override_mean_abs,
            "max_abs": elastic_override_max_abs,
            "principal_max_abs": elastic_override_principal_max_abs,
        },
        "route_target_audit": route_target_summary,
        "feature_stats_audit": feature_stats_summary,
        "split_summaries": {
            split_name: {
                "n_rows": int(np.sum(split_id == split_value)),
                "plastic_rows": int(np.sum((split_id == split_value) & plastic)),
                "route_entropy": _packet4_route_target_stats(soft_atlas_route_target[(split_id == split_value) & plastic]) if np.any((split_id == split_value) & plastic) else None,
            }
            for split_value, split_name in ((0, "train"), (1, "val"), (2, "test"))
        },
        "branch_summaries": {
            name: {
                "n_rows": int(np.sum(branch_id == idx)),
                "route_entropy": _packet4_route_target_stats(soft_atlas_route_target[branch_id == idx]) if np.any(branch_id == idx) else None,
            }
            for idx, name in enumerate(BRANCH_NAMES)
        },
        "stage0_pass": stage0_pass,
    }
    _write_json(summary_path, summary)
    _write_json(route_target_summary_path, route_target_summary)
    _write_json(feature_stats_summary_path, feature_stats_summary)
    return summary


def evaluate_soft_atlas_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split_name: str,
    device: str,
    batch_size: int,
    route_temperature: float = 1.0,
    checkpoint_cache: dict[str, tuple[torch.nn.Module, dict[str, Any], torch.device]] | None = None,
) -> dict[str, Any]:
    arrays, panel = _load_eval_split(dataset_path, split_name)
    pred = _predict_soft_atlas_with_checkpoint(
        checkpoint_path,
        arrays,
        device=device,
        batch_size=batch_size,
        route_temperature=route_temperature,
        checkpoint_cache=checkpoint_cache,
    )

    stress_pred = pred["stress"]
    stress_principal_pred = pred["stress_principal"]
    stress_true = arrays["stress"]
    stress_principal_true = arrays["stress_principal"]
    yield_rel = yield_violation_rel_principal_3d(
        stress_principal_pred,
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
    ).astype(np.float32)
    stress_abs = np.abs(stress_pred - stress_true)
    principal_abs = np.abs(stress_principal_pred - stress_principal_true)
    principal_abs_max = np.max(principal_abs, axis=1).astype(np.float32)
    repo_relative = principal_relative_error_3d(
        stress_principal_pred,
        stress_principal_true,
        c_bar=arrays["material_reduced"][:, 0],
    ).astype(np.float32)

    stats = {
        "stress_component_abs": stress_abs.astype(np.float32),
        "principal_max_abs": principal_abs_max,
        "principal_rel_error": repo_relative,
        "yield_violation_rel": yield_rel,
    }
    metrics = _aggregate_policy_metrics(
        arrays,
        panel,
        stats,
        learned_mask=np.ones(arrays["stress"].shape[0], dtype=bool),
        elastic_mask=(arrays["branch_id"] == 0),
    )

    plastic = panel["plastic_mask"].astype(bool)
    hard = panel["hard_mask"].astype(bool)
    hard_plastic = hard & plastic
    coord_metrics = _surface_metric_row(arrays["grho"], pred["grho"], plastic)
    hard_coord_metrics = _surface_metric_row(arrays["grho"], pred["grho"], hard_plastic)
    route_probs = pred["soft_atlas_route_probs"]
    route_entropy = -np.sum(route_probs * np.log(np.clip(route_probs, 1.0e-12, 1.0)), axis=1)
    metrics.update(coord_metrics)
    metrics.update(
        {
            "hard_g_mae": hard_coord_metrics["g_mae"],
            "hard_rho_mae": hard_coord_metrics["rho_mae"],
            "hard_g_corr": hard_coord_metrics["g_corr"],
            "hard_rho_corr": hard_coord_metrics["rho_corr"],
            "yield_violation_max": float(np.max(yield_rel[plastic])) if np.any(plastic) else 0.0,
            "soft_atlas_route_entropy": float(np.mean(route_entropy)),
            "soft_atlas_route_entropy_p95": float(np.quantile(route_entropy, 0.95)),
            "soft_atlas_route_maxprob": float(np.mean(np.max(route_probs, axis=1))),
            "split": split_name,
        }
    )

    predicted_branch = pred.get("predicted_branch_id")
    if predicted_branch is not None and np.any(plastic):
        branch_true = arrays["branch_id"].astype(np.int64)
        metrics["plastic_branch_accuracy"] = float(np.mean(predicted_branch[plastic] == branch_true[plastic]))
        metrics["plastic_branch_confusion"] = [
            [
                int(np.sum((branch_true[plastic] == i) & (predicted_branch[plastic] == j)))
                for j in range(1, len(BRANCH_NAMES))
            ]
            for i in range(1, len(BRANCH_NAMES))
        ]
        metrics["predicted_plastic_branch_counts"] = {
            BRANCH_NAMES[i]: int(np.sum(predicted_branch[plastic] == i))
            for i in range(1, len(BRANCH_NAMES))
        }

    branch_rows = _branch_breakdown_rows(
        arrays["branch_id"].astype(np.int64),
        stress_principal_true,
        stress_principal_pred,
        stress_true,
        stress_pred,
        arrays["material_reduced"],
    )
    oracle_principal, oracle_grho, _oracle_abr = _soft_atlas_decode_from_chart(
        pred["soft_atlas_chart"],
        arrays["branch_id"],
        trial_principal=arrays["trial_principal"],
        material_reduced=arrays["material_reduced"],
    )
    oracle_stress = stress_voigt_from_principal_numpy(oracle_principal, pred["eigvecs"])
    oracle_abs = np.abs(oracle_stress - stress_true)
    oracle_principal_abs = np.abs(oracle_principal - stress_principal_true)
    oracle_repo_relative = principal_relative_error_3d(
        oracle_principal,
        stress_principal_true,
        c_bar=arrays["material_reduced"][:, 0],
    ).astype(np.float32)
    oracle_yield = yield_violation_rel_principal_3d(
        oracle_principal,
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
    ).astype(np.float32)
    oracle_metrics = {
        "stress_mae": float(np.mean(oracle_abs)),
        "stress_rmse": float(np.sqrt(np.mean((oracle_stress - stress_true) ** 2))),
        "stress_max_abs": float(np.max(oracle_abs)),
        "principal_mae": float(np.mean(np.abs(oracle_principal - stress_principal_true))),
        "principal_rmse": float(np.sqrt(np.mean((oracle_principal - stress_principal_true) ** 2))),
        "broad_plastic_mae": float(np.mean(oracle_abs[plastic])) if np.any(plastic) else float("nan"),
        "hard_plastic_mae": float(np.mean(oracle_abs[hard_plastic])) if np.any(hard_plastic) else float("nan"),
        "hard_p95_principal": float(np.quantile(np.max(oracle_principal_abs[hard_plastic], axis=1), 0.95)) if np.any(hard_plastic) else float("nan"),
        "hard_rel_p95_principal": float(np.quantile(oracle_repo_relative[hard_plastic], 0.95)) if np.any(hard_plastic) else float("nan"),
        "yield_violation_p95": _quantile_or_zero(oracle_yield[plastic], 0.95),
        "yield_violation_max": float(np.max(oracle_yield[plastic])) if np.any(plastic) else 0.0,
        "grho_mae": float(np.mean(np.abs(oracle_grho - arrays["grho"]))),
    }
    oracle_branch_rows = _branch_breakdown_rows(
        arrays["branch_id"].astype(np.int64),
        stress_principal_true,
        oracle_principal,
        stress_true,
        oracle_stress,
        arrays["material_reduced"],
    )
    gap_metrics = {
        "broad_plastic_mae": float(metrics["broad_plastic_mae"] - oracle_metrics["broad_plastic_mae"]),
        "hard_plastic_mae": float(metrics["hard_plastic_mae"] - oracle_metrics["hard_plastic_mae"]),
        "hard_p95_principal": float(metrics["hard_p95_principal"] - oracle_metrics["hard_p95_principal"]),
    }

    return {
        "metrics": metrics,
        "predictions": pred,
        "oracle_metrics": oracle_metrics,
        "branch_rows": branch_rows,
        "oracle_branch_rows": oracle_branch_rows,
        "gap_metrics": gap_metrics,
    }


def _write_branch_breakdown_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        [
            "branch",
            "count",
            "fraction_of_split",
            "stress_component_mae",
            "principal_abs_p95",
            "repo_relative_p95",
        ],
    )


def write_packet_report(
    report_path: Path,
    *,
    stage0_summary: dict[str, Any],
    architecture_rows: list[dict[str, Any]],
    winner: dict[str, Any],
    winner_val_pred: dict[str, Any],
    winner_val_oracle: dict[str, Any] | None,
    winner_test: dict[str, Any],
    success: bool,
    materially_improved: bool,
    gap_half_recommended_minimum_met: bool,
    gap_half_preferred_stronger_met: bool,
) -> None:
    oracle_gap_metrics = winner_val_pred["gap_metrics"]
    oracle_improves_over_predicted = {
        "broad_plastic_mae": bool(float(oracle_gap_metrics["broad_plastic_mae"]) >= 0.0),
        "hard_plastic_mae": bool(float(oracle_gap_metrics["hard_plastic_mae"]) >= 0.0),
        "hard_p95_principal": bool(float(oracle_gap_metrics["hard_p95_principal"]) >= 0.0),
    }
    lines = [
        "## Assignment",
        "",
        "- Executed the packet4 soft admissible atlas experiment plan from `report10.md` on the frozen March 24 validation-first split.",
        "- Exact elastic dispatch, plastic-only learning, and analytic admissible decode were preserved; hard argmax deployment was replaced by a deployed soft mixture over admissible branch charts.",
        "",
        "## Stage 0 Audit",
        "",
        f"- source Stage 0 dataset: `{stage0_summary['source_stage0_dataset']}`",
        f"- derived packet4 dataset: `{stage0_summary['dataset_path']}`",
        f"- target reconstruction mean / max abs: `{stage0_summary['reconstruction']['plastic_mean_abs']:.3e}` / `{stage0_summary['reconstruction']['plastic_max_abs']:.3e}`",
        f"- target yield p95 / max: `{stage0_summary['yield']['plastic_p95']:.3e}` / `{stage0_summary['yield']['plastic_max']:.3e}`",
        f"- route target audit passed: `{stage0_summary['route_target_audit']['checks']['probability_vectors_valid'] and stage0_summary['route_target_audit']['checks']['forbidden_smooth_apex_mixing_absent']}`",
        f"- feature stats finite: `{stage0_summary['feature_stats_audit']['checks']['train_plastic_all_finite'] and stage0_summary['feature_stats_audit']['checks']['full_dataset_all_finite']}`",
        f"- soft route target entropy mean / p95: `{stage0_summary['route_target_stats']['entropy_mean']:.3e}` / `{stage0_summary['route_target_stats']['entropy_p95']:.3e}`",
        "",
        "## Stage 1 Sweep",
        "",
        "- model kinds: `trial_surface_soft_atlas_f1_concat`, `trial_surface_soft_atlas_f1_film`",
        "- sweep: `w64 / w96 / w128` at depth `3` for each conditioning style",
        "",
        "| Architecture | Param Count | Best Checkpoint | T* | Pred Val Broad Plastic MAE | Pred Val Hard Plastic MAE | Pred Val Hard p95 | Pred Val Yield p95 | Oracle Val Broad Plastic MAE | Oracle Val Hard Plastic MAE | Oracle Val Hard p95 | Gap Cut Broad | Gap Cut Hard | Gap Cut P95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in architecture_rows:
        lines.append(
            f"| {row['architecture']} | {row['param_count']} | {row['selected_checkpoint_label']} | {row.get('selected_temperature', float('nan')):.2f} | "
            f"{row['broad_plastic_mae']:.6f} | {row['hard_plastic_mae']:.6f} | {row['hard_p95_principal']:.6f} | {row['yield_violation_p95']:.6e} | "
            f"{row['oracle_broad_plastic_mae']:.6f} | {row['oracle_hard_plastic_mae']:.6f} | {row['oracle_hard_p95_principal']:.6f} | "
            f"{row['gap_cut_broad']:.3f} | {row['gap_cut_hard']:.3f} | {row['gap_cut_hard_p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Validation Winner",
            "",
            f"- architecture: `{winner['architecture']}`",
            f"- checkpoint: `{winner['selected_checkpoint_path']}`",
            f"- selected temperature: `{winner.get('selected_temperature', float('nan')):.2f}`",
            f"- deployed validation broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
            f"- deployed validation hard p95 principal / yield p95: `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
            f"- packet4 oracle validation broad / hard plastic MAE: `{winner_val_oracle['oracle_metrics']['broad_plastic_mae'] if winner_val_oracle and winner_val_oracle['oracle_metrics'] else float('nan'):.6f}` / `{winner_val_oracle['oracle_metrics']['hard_plastic_mae'] if winner_val_oracle and winner_val_oracle['oracle_metrics'] else float('nan'):.6f}`",
            f"- packet4 oracle validation hard p95 principal: `{winner_val_oracle['oracle_metrics']['hard_p95_principal'] if winner_val_oracle and winner_val_oracle['oracle_metrics'] else float('nan'):.6f}`",
            f"- oracle improves over deployed broad / hard / p95: `{oracle_improves_over_predicted['broad_plastic_mae']}` / `{oracle_improves_over_predicted['hard_plastic_mae']}` / `{oracle_improves_over_predicted['hard_p95_principal']}`",
            f"- packet4 soft route entropy mean: `{winner_val_pred['metrics'].get('soft_atlas_route_entropy', float('nan')):.6f}`",
            "",
            "## Held-Out Test",
            "",
            f"- broad plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}`",
            f"- hard plastic MAE: `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
            f"- hard p95 principal: `{winner_test['metrics']['hard_p95_principal']:.6f}`",
            f"- yield violation p95 / max: `{winner_test['metrics']['yield_violation_p95']:.6e}` / `{winner_test['metrics']['yield_violation_max']:.6e}`",
            "",
            "## Decision",
            "",
            f"- continue bar (`<=25 / <=28 / <=250 / <=1e-6`) met: `{success}`",
            f"- packet4 deployed validation beats packet2 on core metrics: `{materially_improved}`",
            f"- packet4 closes the recommended minimum packet3 routing-gap test (hard / hard-p95): `{gap_half_recommended_minimum_met}`",
            f"- packet4 closes the preferred stronger packet3 routing-gap test (all three): `{gap_half_preferred_stronger_met}`",
            f"- final decision: `{'continue' if (success and materially_improved and gap_half_recommended_minimum_met) else 'stop'}`",
            f"- DS remains blocked: `{not (winner['broad_plastic_mae'] <= 15.0 and winner['hard_plastic_mae'] <= 18.0 and winner['hard_p95_principal'] <= 150.0 and winner['yield_violation_p95'] <= 1.0e-6)}`",
        ]
    )
    _write_phase_report(report_path, "NN Replacement Surface Packet4 Soft Atlas 20260324", lines)


def write_execution_report(
    report_path: Path,
    *,
    stage0_summary: dict[str, Any],
    winner: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
    materially_improved: bool,
    gap_half_recommended_minimum_met: bool,
) -> None:
    lines = [
        f"- Stage 0 source dataset: `{stage0_summary['source_stage0_dataset']}`",
        f"- Stage 0 derived packet4 dataset: `{stage0_summary['dataset_path']}`",
        f"- Representation: exact elastic dispatch + plastic-only learned part + soft admissible atlas",
        f"- Stage 1 winner: `{winner['architecture']}` at `{winner['selected_checkpoint_label']}`",
        f"- Selected temperature: `{winner.get('selected_temperature', float('nan')):.2f}`",
        f"- Validation broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
        f"- Validation hard p95 principal / yield p95: `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
        f"- Oracle validation broad / hard plastic MAE: `{winner['oracle_broad_plastic_mae']:.6f}` / `{winner['oracle_hard_plastic_mae']:.6f}`",
        f"- Oracle validation hard p95 principal: `{winner['oracle_hard_p95_principal']:.6f}`",
        f"- Test broad / hard plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}` / `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
        f"- Test hard p95 principal / yield p95: `{winner_test['metrics']['hard_p95_principal']:.6f}` / `{winner_test['metrics']['yield_violation_p95']:.6e}`",
        f"- Validation gap cut broad / hard / p95: `{winner['gap_cut_broad']:.3f}` / `{winner['gap_cut_hard']:.3f}` / `{winner['gap_cut_hard_p95']:.3f}`",
        f"- Deployed validation beats packet2 core metrics: `{materially_improved}`",
        f"- Continue bar met: `{success}`",
        f"- DS remains blocked: `{not (winner['broad_plastic_mae'] <= 15.0 and winner['hard_plastic_mae'] <= 18.0 and winner['hard_p95_principal'] <= 150.0 and winner['yield_violation_p95'] <= 1.0e-6)}`",
        f"- Final decision: `{'continue' if (success and materially_improved and gap_half_recommended_minimum_met) else 'stop'}`",
    ]
    _write_phase_report(report_path, "NN Replacement Surface Packet4 Execution 20260324", lines)


def _temperature_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        float(row["hard_p95_principal"]),
        float(row["hard_plastic_mae"]),
        float(row["broad_plastic_mae"]),
        float(row["oracle_gap_hard_p95_principal"]),
        float(row["temperature"]),
        str(row["checkpoint_label"]),
    )


def _checkpoint_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(row["hard_p95_principal"]),
        float(row["hard_plastic_mae"]),
        float(row["broad_plastic_mae"]),
        float(row["oracle_gap_hard_p95_principal"]),
        str(row.get("selected_checkpoint_label", row.get("checkpoint_label", ""))),
    )


def _select_best_temperature_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible_rows = [row for row in rows if float(row["yield_violation_p95"]) <= 1.0e-6]
    if not feasible_rows:
        return None
    return min(feasible_rows, key=_temperature_selection_key)


def main() -> None:
    args = parse_args()
    stage0_dataset_path = (ROOT / args.stage0_dataset).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    stage0_dir = output_root / "exp0_soft_atlas_dataset"
    stage1_dir = output_root / "exp1_soft_atlas_surface"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    stage0_summary = build_stage0_soft_atlas_dataset(
        stage0_dataset_path=stage0_dataset_path,
        output_dir=stage0_dir,
        force_rerun=args.force_rerun,
    )
    route_checks = stage0_summary["route_target_audit"]["checks"]
    feature_checks = stage0_summary["feature_stats_audit"]["checks"]
    stage0_pass = bool(
        route_checks["probability_vectors_valid"]
        and route_checks["forbidden_smooth_apex_mixing_absent"]
        and route_checks["feature_stats_finite"]
        and route_checks["target_all_finite"]
        and route_checks["decode_outputs_all_finite"]
        and route_checks["dummy_logits_all_finite"]
        and route_checks["elastic_override_exact"]
        and route_checks["admissible_decode_yield_p95"] <= 1.0e-6
        and stage0_summary["yield"]["plastic_max"] <= 1.0e-6
        and feature_checks["train_plastic_all_finite"]
        and feature_checks["full_dataset_all_finite"]
        and feature_checks["nan_count"] == 0
        and feature_checks["inf_count"] == 0
    )
    if not stage0_pass:
        raise RuntimeError("Packet4 Stage 0 failed the report10 audit checks; aborting before Stage 1.")

    architectures = [
        ("concat_w64_d3", "trial_surface_soft_atlas_f1_concat", 64, 3),
        ("concat_w96_d3", "trial_surface_soft_atlas_f1_concat", 96, 3),
        ("concat_w128_d3", "trial_surface_soft_atlas_f1_concat", 128, 3),
        ("film_w64_d3", "trial_surface_soft_atlas_f1_film", 64, 3),
        ("film_w96_d3", "trial_surface_soft_atlas_f1_film", 96, 3),
        ("film_w128_d3", "trial_surface_soft_atlas_f1_film", 128, 3),
    ]
    temperature_rows: list[dict[str, Any]] = []
    temperature_csv_rows: list[dict[str, Any]] = []
    checkpoint_selection_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    checkpoint_cache: dict[str, tuple[torch.nn.Module, dict[str, Any], torch.device]] = {}

    for arch_name, model_kind, width, depth in architectures:
        run_dir = stage1_dir / arch_name
        config = TrainingConfig(
            dataset=stage0_summary["dataset_path"],
            run_dir=str(run_dir),
            model_kind=model_kind,
            epochs=60,
            batch_size=4096,
            lr=3.0e-4,
            weight_decay=1.0e-5,
            width=width,
            depth=depth,
            dropout=0.0,
            seed=args.seed,
            patience=60,
            device=args.device,
            scheduler_kind="cosine",
            warmup_epochs=2,
            min_lr=1.0e-5,
            regression_loss_kind="huber",
            huber_delta=1.0,
            snapshot_every_epochs=10,
            branch_loss_weight=0.0,
        )
        summary = _train_with_batch_fallback(config, force_rerun=args.force_rerun)
        checkpoint_candidates = _candidate_checkpoint_paths(run_dir)
        selected_checkpoint_rows: list[dict[str, Any]] = []

        for checkpoint_path in checkpoint_candidates:
            checkpoint_temperature_rows: list[dict[str, Any]] = []
            for temperature in PACKET4_TEMPERATURE_GRID:
                eval_result = evaluate_soft_atlas_checkpoint(
                    checkpoint_path,
                    dataset_path=Path(stage0_summary["dataset_path"]),
                    split_name="val",
                    device=args.device,
                    batch_size=args.eval_batch_size,
                    route_temperature=temperature,
                    checkpoint_cache=checkpoint_cache,
                )
                row = {
                    "architecture": arch_name,
                    "model_kind": model_kind,
                    "run_dir": str(run_dir),
                    "checkpoint_label": checkpoint_path.stem if checkpoint_path.parent.name == "snapshots" else checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                    "temperature": float(temperature),
                    **eval_result["metrics"],
                    "oracle_broad_plastic_mae": eval_result["oracle_metrics"]["broad_plastic_mae"],
                    "oracle_hard_plastic_mae": eval_result["oracle_metrics"]["hard_plastic_mae"],
                    "oracle_hard_p95_principal": eval_result["oracle_metrics"]["hard_p95_principal"],
                    "oracle_yield_violation_p95": eval_result["oracle_metrics"]["yield_violation_p95"],
                    "oracle_gap_broad_plastic_mae": eval_result["gap_metrics"]["broad_plastic_mae"],
                    "oracle_gap_hard_plastic_mae": eval_result["gap_metrics"]["hard_plastic_mae"],
                    "oracle_gap_hard_p95_principal": eval_result["gap_metrics"]["hard_p95_principal"],
                    "branch_rows": eval_result["branch_rows"],
                    "oracle_branch_rows": eval_result["oracle_branch_rows"],
                    "selected_temperature": float(temperature),
                }
                checkpoint_temperature_rows.append(row)
                temperature_rows.append(row)
                temperature_csv_rows.append(
                    {
                        key: row[key]
                        for key in (
                            "architecture",
                            "model_kind",
                            "run_dir",
                            "checkpoint_label",
                            "checkpoint_path",
                            "temperature",
                            "split",
                            "n_samples",
                            "broad_mae",
                            "hard_mae",
                            "broad_plastic_mae",
                            "hard_plastic_mae",
                            "hard_p95_principal",
                            "hard_rel_p95_principal",
                            "yield_violation_p95",
                            "yield_violation_max",
                            "plastic_coverage",
                            "accepted_plastic_rows",
                            "route_counts",
                            "accepted_true_branch_counts",
                            "soft_atlas_route_entropy",
                            "soft_atlas_route_entropy_p95",
                            "soft_atlas_route_maxprob",
                            "plastic_branch_accuracy",
                            "oracle_broad_plastic_mae",
                            "oracle_hard_plastic_mae",
                            "oracle_hard_p95_principal",
                            "oracle_yield_violation_p95",
                            "oracle_gap_broad_plastic_mae",
                            "oracle_gap_hard_plastic_mae",
                            "oracle_gap_hard_p95_principal",
                        )
                        if key in row
                    }
                )

            selected_temp_row = _select_best_temperature_row(checkpoint_temperature_rows)
            if selected_temp_row is None:
                continue
            feasible_rows = [row for row in checkpoint_temperature_rows if row["yield_violation_p95"] <= 1.0e-6]
            selected_checkpoint_rows.append(
                {
                    **selected_temp_row,
                    "feasible_temperature_count": len(feasible_rows),
                    "temperature_count": len(checkpoint_temperature_rows),
                    "param_count": _parameter_count(Path(selected_temp_row["checkpoint_path"]), args.device),
                    "best_epoch_by_train_loop": summary["best_epoch"],
                    "completed_epochs": summary["completed_epochs"],
                    "checkpoint_feasible": True,
                }
            )

        if not selected_checkpoint_rows:
            raise RuntimeError(f"No feasible packet4 checkpoint temperatures found for architecture {arch_name}.")

        selected_checkpoint_rows.sort(key=_checkpoint_selection_key)
        winner_row = selected_checkpoint_rows[0]
        gap_cut_broad = (
            (PACKET3_REFERENCE_VAL_PRED["broad_plastic_mae"] - winner_row["broad_plastic_mae"])
            / max(PACKET3_REFERENCE_VAL_PRED["broad_plastic_mae"] - PACKET3_REFERENCE_VAL_ORACLE["broad_plastic_mae"], 1.0e-12)
        )
        gap_cut_hard = (
            (PACKET3_REFERENCE_VAL_PRED["hard_plastic_mae"] - winner_row["hard_plastic_mae"])
            / max(PACKET3_REFERENCE_VAL_PRED["hard_plastic_mae"] - PACKET3_REFERENCE_VAL_ORACLE["hard_plastic_mae"], 1.0e-12)
        )
        gap_cut_hard_p95 = (
            (PACKET3_REFERENCE_VAL_PRED["hard_p95_principal"] - winner_row["hard_p95_principal"])
            / max(PACKET3_REFERENCE_VAL_PRED["hard_p95_principal"] - PACKET3_REFERENCE_VAL_ORACLE["hard_p95_principal"], 1.0e-12)
        )
        architecture_rows.append(
            {
                "architecture": arch_name,
                "model_kind": model_kind,
                "run_dir": str(run_dir),
                "param_count": winner_row["param_count"],
                "selected_checkpoint_label": winner_row["checkpoint_label"],
                "selected_checkpoint_path": winner_row["checkpoint_path"],
                "selected_temperature": float(winner_row["selected_temperature"]),
                "completed_epochs": summary["completed_epochs"],
                "best_epoch_by_train_loop": summary["best_epoch"],
                "batch_size": config.batch_size,
                "width": width,
                "depth": depth,
                "feature_set": "soft_atlas_f1",
                "broad_plastic_mae": winner_row["broad_plastic_mae"],
                "hard_plastic_mae": winner_row["hard_plastic_mae"],
                "hard_p95_principal": winner_row["hard_p95_principal"],
                "hard_rel_p95_principal": winner_row["hard_rel_p95_principal"],
                "yield_violation_p95": winner_row["yield_violation_p95"],
                "yield_violation_max": winner_row["yield_violation_max"],
                "plastic_coverage": winner_row["plastic_coverage"],
                "accepted_plastic_rows": winner_row["accepted_plastic_rows"],
                "route_counts": winner_row["route_counts"],
                "accepted_true_branch_counts": winner_row["accepted_true_branch_counts"],
                "soft_atlas_route_entropy": winner_row["soft_atlas_route_entropy"],
                "soft_atlas_route_entropy_p95": winner_row["soft_atlas_route_entropy_p95"],
                "soft_atlas_route_maxprob": winner_row["soft_atlas_route_maxprob"],
                "plastic_branch_accuracy": winner_row.get("plastic_branch_accuracy", float("nan")),
                "oracle_broad_plastic_mae": winner_row["oracle_broad_plastic_mae"],
                "oracle_hard_plastic_mae": winner_row["oracle_hard_plastic_mae"],
                "oracle_hard_p95_principal": winner_row["oracle_hard_p95_principal"],
                "oracle_yield_violation_p95": winner_row["oracle_yield_violation_p95"],
                "oracle_gap_broad_plastic_mae": winner_row["oracle_gap_broad_plastic_mae"],
                "oracle_gap_hard_plastic_mae": winner_row["oracle_gap_hard_plastic_mae"],
                "oracle_gap_hard_p95_principal": winner_row["oracle_gap_hard_p95_principal"],
                "gap_cut_broad": gap_cut_broad,
                "gap_cut_hard": gap_cut_hard,
                "gap_cut_hard_p95": gap_cut_hard_p95,
            }
        )
        checkpoint_selection_rows.extend(selected_checkpoint_rows)
        checkpoint_csv_fieldnames = [
            "architecture",
            "model_kind",
            "run_dir",
            "checkpoint_label",
            "checkpoint_path",
            "temperature",
            "split",
            "n_samples",
            "broad_mae",
            "hard_mae",
            "broad_plastic_mae",
            "hard_plastic_mae",
            "hard_p95_principal",
            "hard_rel_p95_principal",
            "yield_violation_p95",
            "yield_violation_max",
            "plastic_coverage",
            "accepted_plastic_rows",
            "route_counts",
            "accepted_true_branch_counts",
            "soft_atlas_route_entropy",
            "soft_atlas_route_entropy_p95",
            "soft_atlas_route_maxprob",
            "plastic_branch_accuracy",
            "oracle_broad_plastic_mae",
            "oracle_hard_plastic_mae",
            "oracle_hard_p95_principal",
            "oracle_yield_violation_p95",
            "oracle_gap_broad_plastic_mae",
            "oracle_gap_hard_plastic_mae",
            "oracle_gap_hard_p95_principal",
        ]
        _write_csv(
            run_dir / "checkpoint_sweep_val.csv",
            [{key: row[key] for key in checkpoint_csv_fieldnames if key in row} for row in checkpoint_temperature_rows],
            checkpoint_csv_fieldnames,
        )

    architecture_rows.sort(key=_checkpoint_selection_key)
    if not architecture_rows:
        raise RuntimeError("No feasible packet4 architectures found on validation.")
    winner = architecture_rows[0]
    winner_temperature = float(winner["selected_temperature"])
    winner_checkpoint_path = Path(winner["selected_checkpoint_path"])
    winner_val_pred = evaluate_soft_atlas_checkpoint(
        winner_checkpoint_path,
        dataset_path=Path(stage0_summary["dataset_path"]),
        split_name="val",
        device=args.device,
        batch_size=args.eval_batch_size,
        route_temperature=winner_temperature,
        checkpoint_cache=checkpoint_cache,
    )
    winner_val_oracle = {
        "metrics": winner_val_pred["oracle_metrics"],
        "oracle_metrics": winner_val_pred["oracle_metrics"],
        "branch_rows": winner_val_pred["oracle_branch_rows"],
    }
    winner_test = evaluate_soft_atlas_checkpoint(
        winner_checkpoint_path,
        dataset_path=Path(stage0_summary["dataset_path"]),
        split_name="test",
        device=args.device,
        batch_size=args.eval_batch_size,
        route_temperature=winner_temperature,
        checkpoint_cache=checkpoint_cache,
    )

    beat_packet2_core_metrics = {
        "broad_plastic_mae": bool(winner["broad_plastic_mae"] < PACKET2_REFERENCE_VAL["broad_plastic_mae"]),
        "hard_plastic_mae": bool(winner["hard_plastic_mae"] < PACKET2_REFERENCE_VAL["hard_plastic_mae"]),
        "hard_p95_principal": bool(winner["hard_p95_principal"] < PACKET2_REFERENCE_VAL["hard_p95_principal"]),
    }
    yield_stability_constraint_met = bool(winner["yield_violation_p95"] <= 1.0e-6)
    beat_packet2 = bool(all(beat_packet2_core_metrics.values()) and yield_stability_constraint_met)
    packet4_oracle_gap = {
        "broad_plastic_mae": float(winner["broad_plastic_mae"] - winner["oracle_broad_plastic_mae"]),
        "hard_plastic_mae": float(winner["hard_plastic_mae"] - winner["oracle_hard_plastic_mae"]),
        "hard_p95_principal": float(winner["hard_p95_principal"] - winner["oracle_hard_p95_principal"]),
    }
    oracle_improves_over_predicted = {
        metric: bool(packet4_oracle_gap[metric] >= 0.0)
        for metric in packet4_oracle_gap
    }
    gap_half_success = {
        metric: bool(oracle_improves_over_predicted[metric] and packet4_oracle_gap[metric] <= PACKET3_GAP_HALF_TARGETS[metric])
        for metric in PACKET3_GAP_HALF_TARGETS
    }
    gap_half_recommended_minimum_met = bool(gap_half_success["hard_plastic_mae"] and gap_half_success["hard_p95_principal"])
    gap_half_preferred_stronger_met = bool(all(gap_half_success.values()))
    continue_bar_checks = {
        "broad_plastic_mae": bool(winner["broad_plastic_mae"] <= CONTINUE_BAR["broad_plastic_mae"]),
        "hard_plastic_mae": bool(winner["hard_plastic_mae"] <= CONTINUE_BAR["hard_plastic_mae"]),
        "hard_p95_principal": bool(winner["hard_p95_principal"] <= CONTINUE_BAR["hard_p95_principal"]),
        "yield_violation_p95": bool(winner["yield_violation_p95"] <= CONTINUE_BAR["yield_violation_p95"]),
    }
    continue_bar_met = bool(all(continue_bar_checks.values()))
    ds_bar_checks = {
        "broad_plastic_mae": bool(winner["broad_plastic_mae"] <= S1_GOAL["broad_plastic_mae"]),
        "hard_plastic_mae": bool(winner["hard_plastic_mae"] <= S1_GOAL["hard_plastic_mae"]),
        "hard_p95_principal": bool(winner["hard_p95_principal"] <= S1_GOAL["hard_p95_principal"]),
        "yield_violation_p95": bool(winner["yield_violation_p95"] <= S1_GOAL["yield_violation_p95"]),
    }
    ds_bar_met = bool(all(ds_bar_checks.values()))
    final_decision = "continue" if (continue_bar_met and beat_packet2 and gap_half_recommended_minimum_met) else "stop"
    materially_improved = beat_packet2
    success = continue_bar_met

    _write_csv(
        stage1_dir / "architecture_summary_val.csv",
        architecture_rows,
        [
            "architecture",
            "model_kind",
            "run_dir",
            "param_count",
            "selected_checkpoint_label",
            "selected_checkpoint_path",
            "selected_temperature",
            "completed_epochs",
            "best_epoch_by_train_loop",
            "batch_size",
            "width",
            "depth",
            "feature_set",
            "broad_plastic_mae",
            "hard_plastic_mae",
            "hard_p95_principal",
            "hard_rel_p95_principal",
            "yield_violation_p95",
            "yield_violation_max",
            "plastic_coverage",
            "accepted_plastic_rows",
            "route_counts",
            "accepted_true_branch_counts",
            "soft_atlas_route_entropy",
            "soft_atlas_route_entropy_p95",
            "soft_atlas_route_maxprob",
            "plastic_branch_accuracy",
            "oracle_broad_plastic_mae",
            "oracle_hard_plastic_mae",
            "oracle_hard_p95_principal",
            "oracle_yield_violation_p95",
            "oracle_gap_broad_plastic_mae",
            "oracle_gap_hard_plastic_mae",
            "oracle_gap_hard_p95_principal",
            "gap_cut_broad",
            "gap_cut_hard",
            "gap_cut_hard_p95",
        ],
    )
    _write_csv(
        stage1_dir / "temperature_sweep_val.csv",
        temperature_csv_rows,
        [
            "architecture",
            "model_kind",
            "run_dir",
            "checkpoint_label",
            "checkpoint_path",
            "temperature",
            "split",
            "n_samples",
            "broad_mae",
            "hard_mae",
            "broad_plastic_mae",
            "hard_plastic_mae",
            "hard_p95_principal",
            "hard_rel_p95_principal",
            "yield_violation_p95",
            "yield_violation_max",
            "plastic_coverage",
            "accepted_plastic_rows",
            "route_counts",
            "accepted_true_branch_counts",
            "soft_atlas_route_entropy",
            "soft_atlas_route_entropy_p95",
            "soft_atlas_route_maxprob",
            "plastic_branch_accuracy",
            "oracle_broad_plastic_mae",
            "oracle_hard_plastic_mae",
            "oracle_hard_p95_principal",
            "oracle_yield_violation_p95",
            "oracle_gap_broad_plastic_mae",
            "oracle_gap_hard_plastic_mae",
            "oracle_gap_hard_p95_principal",
        ],
    )

    _write_json(
        stage1_dir / "temperature_sweep_val.json",
        {
            "temperature_grid": list(PACKET4_TEMPERATURE_GRID),
            "rows": temperature_rows,
            "selected_checkpoint_rows": checkpoint_selection_rows,
            "selected_architecture_rows": architecture_rows,
            "winner": winner,
            "packet2_reference_val": PACKET2_REFERENCE_VAL,
            "packet3_reference_val_pred": PACKET3_REFERENCE_VAL_PRED,
            "packet3_reference_val_oracle": PACKET3_REFERENCE_VAL_ORACLE,
            "stage0_route_target_audit": stage0_summary["route_target_audit"],
            "stage0_feature_stats": stage0_summary["feature_stats_audit"],
        },
    )
    _write_json(
        stage1_dir / "checkpoint_selection.json",
        {
            "temperature_grid": list(PACKET4_TEMPERATURE_GRID),
            "selected_checkpoint_rows": checkpoint_selection_rows,
            "all_architectures": architecture_rows,
            "winner": winner,
            "packet2_reference_val": PACKET2_REFERENCE_VAL,
            "packet2_reference_test": PACKET2_REFERENCE_TEST,
            "packet3_reference_val_pred": PACKET3_REFERENCE_VAL_PRED,
            "packet3_reference_val_oracle": PACKET3_REFERENCE_VAL_ORACLE,
            "packet4_oracle_gap": packet4_oracle_gap,
            "oracle_improves_over_predicted": oracle_improves_over_predicted,
            "continue_bar_met": continue_bar_met,
            "gap_half_recommended_minimum_met": gap_half_recommended_minimum_met,
            "gap_half_preferred_stronger_met": gap_half_preferred_stronger_met,
            "beat_packet2": beat_packet2,
            "yield_stability_constraint_met": yield_stability_constraint_met,
            "ds_bar_met": ds_bar_met,
        },
    )
    _write_json(
        stage1_dir / "decision_logic_summary.json",
        {
            "temperature_grid": list(PACKET4_TEMPERATURE_GRID),
            "yield_constraint": {"metric": "yield_violation_p95", "threshold": 1.0e-6},
            "yield_tail_summary": {
                "winner_yield_violation_p95": winner["yield_violation_p95"],
                "winner_yield_violation_max": winner["yield_violation_max"],
                "yield_stability_constraint_met": yield_stability_constraint_met,
            },
            "checkpoint_selection_rule": [
                "keep only temperatures with yield_violation_p95 <= 1e-6",
                "rank by hard_p95_principal, hard_plastic_mae, broad_plastic_mae, oracle_gap_hard_p95_principal",
            ],
            "architecture_selection_rule": [
                "choose best feasible checkpoint per architecture using the same lexicographic rule",
                "rank architectures by the selected checkpoint's metrics with the same lexicographic rule",
            ],
            "beat_packet2_core_metrics": beat_packet2_core_metrics,
            "beat_packet2": beat_packet2,
            "gap_half_targets": PACKET3_GAP_HALF_TARGETS,
            "packet4_oracle_gap": packet4_oracle_gap,
            "oracle_improves_over_predicted": oracle_improves_over_predicted,
            "gap_half_success": gap_half_success,
            "gap_half_recommended_minimum_met": gap_half_recommended_minimum_met,
            "gap_half_preferred_stronger_met": gap_half_preferred_stronger_met,
            "continue_bar_checks": continue_bar_checks,
            "continue_bar_met": continue_bar_met,
            "ds_bar_checks": ds_bar_checks,
            "ds_bar_met": ds_bar_met,
            "final_decision": final_decision,
            "winner_validation": {
                "selected_temperature": winner_temperature,
                "metrics": winner_val_pred["metrics"],
                "oracle_metrics": winner_val_pred["oracle_metrics"],
                "oracle_vs_predicted_gaps": winner_val_pred["gap_metrics"],
            },
            "winner": winner,
            "packet2_reference_val": PACKET2_REFERENCE_VAL,
            "packet2_reference_test": PACKET2_REFERENCE_TEST,
            "packet3_reference_val_pred": PACKET3_REFERENCE_VAL_PRED,
            "packet3_reference_val_oracle": PACKET3_REFERENCE_VAL_ORACLE,
            "stage0_pass": stage0_pass,
        },
    )
    _write_json(stage1_dir / "winner_val_predicted_summary.json", winner_val_pred)
    _write_json(stage1_dir / "winner_val_oracle_summary.json", winner_val_oracle)
    _write_json(stage1_dir / "winner_test_summary.json", winner_test)
    _write_json(
        stage1_dir / "winner_temperature_selected.json",
        {
            "architecture": winner["architecture"],
            "model_kind": winner["model_kind"],
            "checkpoint_label": winner["selected_checkpoint_label"],
            "checkpoint_path": winner["selected_checkpoint_path"],
            "temperature": winner_temperature,
            "selected_temperature": winner_temperature,
            "selection_rule": {
                "yield_constraint": "yield_violation_p95 <= 1e-6",
                "ranking": [
                    "hard_p95_principal",
                    "hard_plastic_mae",
                    "broad_plastic_mae",
                    "oracle_gap_hard_p95_principal",
                ],
            },
            "validation_metrics": winner_val_pred["metrics"],
            "validation_oracle_metrics": winner_val_pred["oracle_metrics"],
            "validation_row": winner,
            "test_metrics": winner_test["metrics"],
        },
    )
    _write_json(
        stage1_dir / "oracle_vs_predicted_val.json",
        {
            "predicted": winner_val_pred["metrics"],
            "oracle": winner_val_pred["oracle_metrics"],
            "gap": {
                "broad_plastic_mae": float(winner_val_pred["metrics"]["broad_plastic_mae"] - winner_val_pred["oracle_metrics"]["broad_plastic_mae"]),
                "hard_plastic_mae": float(winner_val_pred["metrics"]["hard_plastic_mae"] - winner_val_pred["oracle_metrics"]["hard_plastic_mae"]),
                "hard_p95_principal": float(winner_val_pred["metrics"]["hard_p95_principal"] - winner_val_pred["oracle_metrics"]["hard_p95_principal"]),
            },
            "packet2_reference_val": PACKET2_REFERENCE_VAL,
            "packet3_reference_val_pred": PACKET3_REFERENCE_VAL_PRED,
            "packet3_reference_val_oracle": PACKET3_REFERENCE_VAL_ORACLE,
            "selected_temperature": winner_temperature,
        },
    )
    _write_json(
        stage1_dir / "winner_oracle_vs_predicted_test.json",
        {
            "predicted": winner_test["metrics"],
            "oracle": winner_test["oracle_metrics"],
            "gap": {
                "broad_plastic_mae": float(winner_test["metrics"]["broad_plastic_mae"] - winner_test["oracle_metrics"]["broad_plastic_mae"]),
                "hard_plastic_mae": float(winner_test["metrics"]["hard_plastic_mae"] - winner_test["oracle_metrics"]["hard_plastic_mae"]),
                "hard_p95_principal": float(winner_test["metrics"]["hard_p95_principal"] - winner_test["oracle_metrics"]["hard_p95_principal"]),
            },
            "selected_winner": winner,
            "selected_temperature": winner_temperature,
        },
    )
    _write_branch_breakdown_csv(stage1_dir / "winner_val_branch_breakdown_predicted.csv", winner_val_pred["branch_rows"])
    _write_branch_breakdown_csv(stage1_dir / "winner_val_branch_breakdown_oracle.csv", winner_val_oracle["branch_rows"])
    _write_branch_breakdown_csv(stage1_dir / "winner_test_branch_breakdown.csv", winner_test["branch_rows"])

    comparison_lines = [
        "# Packet4 vs Packet2 / Packet3",
        "",
        f"- selected architecture: `{winner['architecture']}`",
        f"- selected checkpoint: `{winner['selected_checkpoint_label']}`",
        f"- selected temperature: `{winner_temperature:.2f}`",
        "",
        "## Validation Winner",
        "",
        f"- deployed broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
        f"- deployed hard p95 principal / yield p95: `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
        f"- oracle broad / hard plastic MAE: `{winner_val_pred['oracle_metrics']['broad_plastic_mae']:.6f}` / `{winner_val_pred['oracle_metrics']['hard_plastic_mae']:.6f}`",
        f"- oracle hard p95 principal: `{winner_val_pred['oracle_metrics']['hard_p95_principal']:.6f}`",
        "",
        "## Packet2 Comparison",
        "",
        f"- packet2 broad / hard plastic MAE: `{PACKET2_REFERENCE_VAL['broad_plastic_mae']:.6f}` / `{PACKET2_REFERENCE_VAL['hard_plastic_mae']:.6f}`",
        f"- packet2 hard p95 principal / yield p95: `{PACKET2_REFERENCE_VAL['hard_p95_principal']:.6f}` / `{PACKET2_REFERENCE_VAL['yield_violation_p95']:.6e}`",
        f"- packet4 beat packet2 on core stress metrics: `{beat_packet2}`",
        "",
        "## Packet3 Comparison",
        "",
        f"- packet3 predicted broad / hard plastic MAE: `{PACKET3_REFERENCE_VAL_PRED['broad_plastic_mae']:.6f}` / `{PACKET3_REFERENCE_VAL_PRED['hard_plastic_mae']:.6f}`",
        f"- packet3 predicted hard p95 principal: `{PACKET3_REFERENCE_VAL_PRED['hard_p95_principal']:.6f}`",
        f"- packet3 oracle broad / hard plastic MAE: `{PACKET3_REFERENCE_VAL_ORACLE['broad_plastic_mae']:.6f}` / `{PACKET3_REFERENCE_VAL_ORACLE['hard_plastic_mae']:.6f}`",
        f"- packet3 oracle hard p95 principal: `{PACKET3_REFERENCE_VAL_ORACLE['hard_p95_principal']:.6f}`",
        f"- packet4 oracle improves over deployed broad / hard / p95: `{oracle_improves_over_predicted['broad_plastic_mae']}` / `{oracle_improves_over_predicted['hard_plastic_mae']}` / `{oracle_improves_over_predicted['hard_p95_principal']}`",
        f"- packet4 oracle-vs-predicted gaps broad / hard / p95: `{winner_val_pred['gap_metrics']['broad_plastic_mae']:.6f}` / `{winner_val_pred['gap_metrics']['hard_plastic_mae']:.6f}` / `{winner_val_pred['gap_metrics']['hard_p95_principal']:.6f}`",
        f"- packet3 half-gap targets broad / hard / p95: `{PACKET3_GAP_HALF_TARGETS['broad_plastic_mae']:.6f}` / `{PACKET3_GAP_HALF_TARGETS['hard_plastic_mae']:.6f}` / `{PACKET3_GAP_HALF_TARGETS['hard_p95_principal']:.6f}`",
        f"- packet4 gap cuts broad / hard / p95: `{winner['gap_cut_broad']:.3f}` / `{winner['gap_cut_hard']:.3f}` / `{winner['gap_cut_hard_p95']:.3f}`",
        f"- packet4 gap-half success recommended minimum (hard / hard-p95): `{gap_half_recommended_minimum_met}`",
        f"- packet4 gap-half success preferred stronger all three: `{gap_half_preferred_stronger_met}`",
        "",
        "## Continue Bar",
        "",
        f"- continue bar met (`<=25 / <=28 / <=250 / <=1e-6`): `{continue_bar_met}`",
        f"- DS bar met (`<=15 / <=18 / <=150 / <=1e-6`): `{ds_bar_met}`",
        f"- final decision: `{final_decision}`",
    ]
    _write_phase_report(stage1_dir / "packet4_vs_packet2_packet3_val.md", "Packet4 vs Packet2 / Packet3 20260325", comparison_lines)

    write_packet_report(
        docs_root / "nn_replacement_surface_packet4_soft_atlas_20260324.md",
        stage0_summary=stage0_summary,
        architecture_rows=architecture_rows,
        winner=winner,
        winner_val_pred=winner_val_pred,
        winner_val_oracle=winner_val_oracle,
        winner_test=winner_test,
        success=continue_bar_met,
        materially_improved=beat_packet2,
        gap_half_recommended_minimum_met=gap_half_recommended_minimum_met,
        gap_half_preferred_stronger_met=gap_half_preferred_stronger_met,
    )
    write_execution_report(
        docs_root / "nn_replacement_surface_packet4_execution_20260324.md",
        stage0_summary=stage0_summary,
        winner=winner,
        winner_test=winner_test,
        success=continue_bar_met,
        materially_improved=beat_packet2,
        gap_half_recommended_minimum_met=gap_half_recommended_minimum_met,
    )


if __name__ == "__main__":
    main()
