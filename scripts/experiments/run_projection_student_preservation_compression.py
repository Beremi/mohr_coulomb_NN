#!/usr/bin/env python
"""Run the March 27 projection-student preservation/compression packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_surrogate.models import build_trial_principal_geom_features, spectral_decomposition_from_strain
from mc_surrogate.mohr_coulomb import BRANCH_NAMES
from mc_surrogate.projection_student_preservation import (
    PROJECTION_CANDIDATE_NAMES,
    build_sampling_weights,
    build_teacher_projection_cache_arrays,
    directional_error_arrays,
    project_teacher_checkpoint_stress,
    projection_switch_mask,
    quantile_or_zero,
    summarize_array,
)
from mc_surrogate.training import TrainingConfig, load_checkpoint, predict_with_checkpoint, predict_with_loaded_checkpoint
from run_nn_replacement_abr_cycle import (
    _aggregate_policy_metrics,
    _load_h5,
    _parameter_count,
    _pointwise_prediction_stats,
    _slice_arrays,
    _train_with_batch_fallback,
    _write_csv,
    _write_h5,
    _write_json,
)

SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
TEACHER_CHECKPOINT = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt"
TEACHER_SUMMARY = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/summary.json"
GROUPED_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5"
PANEL_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5"
MARCH26_PHASE0_SUMMARY = ROOT / "experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_summary.json"
MARCH26_PHASE2_SUMMARY = ROOT / "experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/phase2_summary.json"
CANDIDATE_B_WARMSTART = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/candidate_b/finetune/best.pt"
WORK_PACKET_BASENAME = "projection_student_preservation_compression_work_packet_20260327.md"

PROBE_DIRECTIONS = 3
PROBE_SEED = 20260327
PROJECTION_TAU = 0.05
FD_SCALE = 1.0e-6
TIER1_GATE = {
    "broad_plastic_mae": 18.0,
    "hard_plastic_mae": 20.0,
    "hard_p95_principal": 180.0,
    "yield_violation_p95": 1.0e-6,
}
TIER2_GATE = {
    "broad_plastic_mae": 15.0,
    "hard_plastic_mae": 18.0,
    "hard_p95_principal": 160.0,
    "yield_violation_p95": 1.0e-6,
}
TIER1_ARCHS = (
    {"name": "w160_d2", "width": 160, "depth": 2},
    {"name": "w144_d3", "width": 144, "depth": 3},
    {"name": "w128_d4", "width": 128, "depth": 4},
)
TIER2_ARCHS = (
    {"name": "w144_d2", "width": 144, "depth": 2},
    {"name": "w96_d5", "width": 96, "depth": 5},
    {"name": "w128_d2", "width": 128, "depth": 2},
)
BOUNDARY_MASK_KEYS = (
    "near_yield_mask",
    "near_smooth_left_mask",
    "near_smooth_right_mask",
    "near_left_apex_mask",
    "near_right_apex_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="all",
        choices=[
            "all",
            "exp0_teacher_cache",
            "exp1_teacher_ds_probe",
            "exp2_preservation_dataset",
            "exp3_preservation_control",
            "exp4_tier1_compression",
            "exp5_tier2_compression",
        ],
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/projection_student_preservation_compression_20260327",
    )
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_mask(split_id: np.ndarray, split_name: str) -> np.ndarray:
    return np.asarray(split_id == SPLIT_NAME_TO_ID[split_name], dtype=bool)


def _meets_bar(metrics: dict[str, float], bar: dict[str, float]) -> bool:
    return all(float(metrics[key]) <= float(bar[key]) for key in bar)


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    gate_fail = 1.0 if float(row["val_yield_violation_p95"]) > 1.0e-6 else 0.0
    return (
        gate_fail,
        float(row["val_hard_p95_principal"]),
        float(row["val_hard_plastic_mae"]),
        float(row["val_broad_plastic_mae"]),
    )


def _medium_exact_reference() -> dict[str, float]:
    winner = _read_json(MARCH26_PHASE2_SUMMARY)["winner"]
    return {
        "broad_plastic_mae": float(winner["val_broad_plastic_mae"]),
        "hard_plastic_mae": float(winner["val_hard_plastic_mae"]),
        "hard_p95_principal": float(winner["val_hard_p95_principal"]),
        "yield_violation_p95": float(winner["val_yield_violation_p95"]),
    }


def _evaluate_policy(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    *,
    stress_pred: np.ndarray,
    stress_principal_pred: np.ndarray,
) -> dict[str, Any]:
    plastic_mask = panel["plastic_mask"].astype(bool)
    elastic_mask = ~plastic_mask
    stats = _pointwise_prediction_stats(
        arrays,
        stress_pred=stress_pred,
        stress_principal_pred=stress_principal_pred,
    )
    metrics = _aggregate_policy_metrics(
        arrays,
        panel,
        stats,
        learned_mask=plastic_mask,
        elastic_mask=elastic_mask,
    )
    yield_plastic = stats["yield_violation_rel"][plastic_mask]
    metrics["yield_violation_max"] = float(np.max(yield_plastic)) if yield_plastic.size else 0.0
    metrics["stress_mae"] = float(np.mean(stats["stress_component_abs"]))
    metrics["principal_mae"] = float(np.mean(np.abs(stress_principal_pred - arrays["stress_principal"])))
    return {"metrics": metrics, "stats": stats}


def _evaluate_projected_checkpoint_on_split(
    checkpoint_path: Path,
    arrays_all: dict[str, np.ndarray],
    panel_all: dict[str, np.ndarray],
    *,
    split_name: str,
    device: str,
    eval_batch_size: int,
) -> dict[str, Any]:
    mask = _split_mask(arrays_all["split_id"], split_name)
    arrays = _slice_arrays(arrays_all, mask)
    panel = _slice_arrays(panel_all, mask)
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=eval_batch_size,
    )
    return _evaluate_policy(
        arrays,
        panel,
        stress_pred=pred["stress"].astype(np.float32),
        stress_principal_pred=pred["stress_principal"].astype(np.float32),
    ) | {"predictions": pred}


def _probe_directions(n_rows: int, *, seed: int, n_dirs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_rows, n_dirs, 6)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return directions / np.maximum(norms, 1.0e-12)


def _probe_scope_summary(
    metric_arrays: dict[str, np.ndarray],
    switch_events: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return {
            "n_rows": 0,
            "n_direction_samples": 0,
            "abs_error_norm": summarize_array(np.asarray([], dtype=np.float32)),
            "relative_error": summarize_array(np.asarray([], dtype=np.float32)),
            "cosine_similarity": summarize_array(np.asarray([], dtype=np.float32)),
            "candidate_switch_rate": 0.0,
        }
    return {
        "n_rows": int(np.sum(mask)),
        "n_direction_samples": int(metric_arrays["abs_error_norm"][mask].size),
        "abs_error_norm": summarize_array(metric_arrays["abs_error_norm"][mask]),
        "relative_error": summarize_array(metric_arrays["relative_error"][mask]),
        "cosine_similarity": summarize_array(metric_arrays["cosine_similarity"][mask]),
        "candidate_switch_rate": float(np.mean(switch_events[mask])),
    }


def _boundary_summary(
    metric_arrays: dict[str, np.ndarray],
    switch_events: np.ndarray,
    split_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    any_boundary = np.zeros(split_arrays["branch_id"].shape[0], dtype=bool)
    for key in BOUNDARY_MASK_KEYS:
        mask = split_arrays[key].astype(bool)
        any_boundary |= mask
        out[key] = _probe_scope_summary(metric_arrays, switch_events, mask)
    out["any_boundary"] = _probe_scope_summary(metric_arrays, switch_events, any_boundary)
    return out


def _branchwise_probe_summary(
    metric_arrays: dict[str, np.ndarray],
    switch_events: np.ndarray,
    branch_id: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    branch = np.asarray(branch_id, dtype=np.int64).reshape(-1)
    for idx, name in enumerate(BRANCH_NAMES[1:], start=1):
        out[name] = _probe_scope_summary(metric_arrays, switch_events, branch == idx)
    return out


def _probe_mode_on_split(
    split_arrays: dict[str, np.ndarray],
    *,
    teacher_model: Any,
    teacher_metadata: dict[str, Any],
    mode: str,
    directions: np.ndarray,
    device: str,
    eval_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_rows = split_arrays["strain_eng"].shape[0]
    h = FD_SCALE * np.maximum(np.amax(np.abs(split_arrays["strain_eng"]), axis=1, keepdims=True), 1.0).astype(np.float32)
    base_projection = project_teacher_checkpoint_stress(
        split_arrays["strain_eng"],
        split_arrays["material_reduced"],
        split_arrays["teacher_checkpoint_stress"],
        mode=mode,
        tau=PROJECTION_TAU,
    )

    abs_error_norm = np.zeros((n_rows, directions.shape[1]), dtype=np.float32)
    relative_error = np.zeros((n_rows, directions.shape[1]), dtype=np.float32)
    cosine_similarity = np.zeros((n_rows, directions.shape[1]), dtype=np.float32)
    switch_events = np.zeros((n_rows, directions.shape[1]), dtype=np.int8)
    candidate_plus = np.full((n_rows, directions.shape[1]), -1, dtype=np.int8)
    candidate_minus = np.full((n_rows, directions.shape[1]), -1, dtype=np.int8)

    tangent_true = split_arrays["DS"].astype(np.float32)
    for dir_idx in range(directions.shape[1]):
        direction = directions[:, dir_idx, :]
        strain_plus = split_arrays["strain_eng"] + h * direction
        strain_minus = split_arrays["strain_eng"] - h * direction

        strain_pm = np.concatenate([strain_plus, strain_minus], axis=0)
        material_pm = np.concatenate([split_arrays["material_reduced"], split_arrays["material_reduced"]], axis=0)
        pred_pm = predict_with_loaded_checkpoint(
            teacher_model,
            teacher_metadata,
            strain_pm,
            material_pm,
            device=device,
            batch_size=eval_batch_size,
        )
        plus_pred = {key: value[:n_rows] for key, value in pred_pm.items() if isinstance(value, np.ndarray)}
        minus_pred = {key: value[n_rows:] for key, value in pred_pm.items() if isinstance(value, np.ndarray)}
        plus_projection = project_teacher_checkpoint_stress(
            strain_plus,
            split_arrays["material_reduced"],
            plus_pred["stress"],
            mode=mode,
            tau=PROJECTION_TAU,
        )
        minus_projection = project_teacher_checkpoint_stress(
            strain_minus,
            split_arrays["material_reduced"],
            minus_pred["stress"],
            mode=mode,
            tau=PROJECTION_TAU,
        )
        jv_pred = (plus_projection["teacher_projected_stress"] - minus_projection["teacher_projected_stress"]) / (2.0 * h)
        jv_true = np.einsum("nij,nj->ni", tangent_true, direction, optimize=True)
        metrics = directional_error_arrays(jv_pred, jv_true)
        abs_error_norm[:, dir_idx] = metrics["abs_error_norm"]
        relative_error[:, dir_idx] = metrics["relative_error"]
        cosine_similarity[:, dir_idx] = metrics["cosine_similarity"]
        switch_events[:, dir_idx] = projection_switch_mask(
            base_projection["teacher_projection_candidate_id"],
            plus_projection["teacher_projection_candidate_id"],
            minus_projection["teacher_projection_candidate_id"],
        ).astype(np.int8)
        candidate_plus[:, dir_idx] = plus_projection["teacher_projection_candidate_id"]
        candidate_minus[:, dir_idx] = minus_projection["teacher_projection_candidate_id"]

    rowwise = {
        "source_call_id": split_arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": split_arrays["source_row_in_call"].astype(np.int32),
        "branch_id": split_arrays["branch_id"].astype(np.int8),
        "hard_mask": split_arrays["hard_mask"].astype(np.int8),
        "plastic_mask": split_arrays["plastic_mask"].astype(np.int8),
        "ds_valid_mask": split_arrays["ds_valid_mask"].astype(np.int8),
        "near_yield_mask": split_arrays["near_yield_mask"].astype(np.int8),
        "near_smooth_left_mask": split_arrays["near_smooth_left_mask"].astype(np.int8),
        "near_smooth_right_mask": split_arrays["near_smooth_right_mask"].astype(np.int8),
        "near_left_apex_mask": split_arrays["near_left_apex_mask"].astype(np.int8),
        "near_right_apex_mask": split_arrays["near_right_apex_mask"].astype(np.int8),
        "direction": directions.astype(np.float32),
        "fd_step": h.astype(np.float32),
        f"{mode}_candidate_base": base_projection["teacher_projection_candidate_id"].astype(np.int8),
        f"{mode}_candidate_plus": candidate_plus.astype(np.int8),
        f"{mode}_candidate_minus": candidate_minus.astype(np.int8),
        f"{mode}_abs_error_norm": abs_error_norm.astype(np.float32),
        f"{mode}_relative_error": relative_error.astype(np.float32),
        f"{mode}_cosine_similarity": cosine_similarity.astype(np.float32),
        f"{mode}_candidate_switch": switch_events.astype(np.int8),
    }
    metric_arrays = {
        "abs_error_norm": abs_error_norm,
        "relative_error": relative_error,
        "cosine_similarity": cosine_similarity,
    }
    summary = {
        "overall": _probe_scope_summary(metric_arrays, switch_events, np.ones(n_rows, dtype=bool)),
        "plastic_only": _probe_scope_summary(metric_arrays, switch_events, split_arrays["branch_id"] > 0),
        "branchwise": _branchwise_probe_summary(metric_arrays, switch_events, split_arrays["branch_id"]),
        "boundary_instability": _boundary_summary(metric_arrays, switch_events, split_arrays),
    }
    return rowwise, summary


def _probe_summary_rows(split_name: str, mode: str, scope_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "split": split_name,
            "mode": mode,
            "scope": scope_name,
            "metric": "abs_error_norm",
            "mean": payload["abs_error_norm"]["mean"],
            "p50": payload["abs_error_norm"]["p50"],
            "p95": payload["abs_error_norm"]["p95"],
            "max": payload["abs_error_norm"]["max"],
            "candidate_switch_rate": payload["candidate_switch_rate"],
        },
        {
            "split": split_name,
            "mode": mode,
            "scope": scope_name,
            "metric": "relative_error",
            "mean": payload["relative_error"]["mean"],
            "p50": payload["relative_error"]["p50"],
            "p95": payload["relative_error"]["p95"],
            "max": payload["relative_error"]["max"],
            "candidate_switch_rate": payload["candidate_switch_rate"],
        },
        {
            "split": split_name,
            "mode": mode,
            "scope": scope_name,
            "metric": "cosine_similarity",
            "mean": payload["cosine_similarity"]["mean"],
            "p50": payload["cosine_similarity"]["p50"],
            "p95": payload["cosine_similarity"]["p95"],
            "max": payload["cosine_similarity"]["max"],
            "candidate_switch_rate": payload["candidate_switch_rate"],
        },
    ]
    return rows


def run_phase0_teacher_cache(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp0_teacher_cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    cache_path = phase_dir / "teacher_projection_preservation_cache.h5"
    summary_path = phase_dir / "teacher_projection_preservation_summary.json"
    if cache_path.exists() and summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    arrays_all, dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, panel_attrs = _load_h5(PANEL_DATASET)
    teacher_pred = predict_with_checkpoint(
        TEACHER_CHECKPOINT,
        arrays_all["strain_eng"],
        arrays_all["material_reduced"],
        device=device,
        batch_size=eval_batch_size,
    )
    cache_arrays = build_teacher_projection_cache_arrays(
        arrays_all,
        panel_all,
        teacher_pred,
        projection_mode="exact",
        projection_tau=PROJECTION_TAU,
    )
    _write_h5(
        cache_path,
        cache_arrays,
        {
            "teacher_checkpoint": str(TEACHER_CHECKPOINT),
            "teacher_summary": str(TEACHER_SUMMARY),
            "source_grouped_dataset": str(GROUPED_DATASET),
            "source_panel_dataset": str(PANEL_DATASET),
            "projection_mode": "exact",
            "projection_tau": PROJECTION_TAU,
            "projection_candidate_names_json": list(PROJECTION_CANDIDATE_NAMES),
            "dataset_attrs_json": dataset_attrs,
            "panel_attrs_json": panel_attrs,
        },
    )

    split_summaries: dict[str, Any] = {}
    for split_name in ("val", "test"):
        mask = _split_mask(cache_arrays["split_id"], split_name)
        split_arrays = _slice_arrays(arrays_all, mask)
        split_panel = _slice_arrays(panel_all, mask)
        dispatch_eval = _evaluate_policy(
            split_arrays,
            split_panel,
            stress_pred=cache_arrays["teacher_dispatch_stress"][mask],
            stress_principal_pred=cache_arrays["teacher_dispatch_stress_principal"][mask],
        )
        projected_eval = _evaluate_policy(
            split_arrays,
            split_panel,
            stress_pred=cache_arrays["teacher_projected_stress"][mask],
            stress_principal_pred=cache_arrays["teacher_projected_stress_principal"][mask],
        )
        split_plastic = split_panel["plastic_mask"].astype(bool)
        split_summaries[split_name] = {
            "dispatch_teacher": dispatch_eval["metrics"],
            "projected_teacher": projected_eval["metrics"],
            "projection_disp_norm": summarize_array(cache_arrays["teacher_projection_disp_norm"][mask][split_plastic]),
            "projection_candidate_counts": {
                name: int(np.sum(cache_arrays["teacher_projection_candidate_id"][mask][split_plastic] == idx))
                for idx, name in enumerate(PROJECTION_CANDIDATE_NAMES)
            },
            "ds_valid_count": int(np.sum(cache_arrays["ds_valid_mask"][mask] > 0)),
            "hard_count": int(np.sum(cache_arrays["hard_mask"][mask] > 0)),
        }

    summary = {
        "teacher_checkpoint": str(TEACHER_CHECKPOINT),
        "teacher_summary": str(TEACHER_SUMMARY),
        "cache_path": str(cache_path),
        "n_rows": int(cache_arrays["split_id"].shape[0]),
        "split_summaries": split_summaries,
    }
    _write_json(summary_path, summary)
    return summary


def run_phase1_teacher_ds_probe(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp1_teacher_ds_probe"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "teacher_ds_probe_summary.json"
    csv_path = phase_dir / "teacher_ds_probe_summary.csv"
    fe_status_path = output_root / "fe_shadow_status.json"
    if summary_path.exists() and csv_path.exists() and fe_status_path.exists() and not force_rerun:
        return _read_json(summary_path)

    cache_path = output_root / "exp0_teacher_cache" / "teacher_projection_preservation_cache.h5"
    if not cache_path.exists():
        run_phase0_teacher_cache(
            output_root=output_root,
            device=device,
            eval_batch_size=eval_batch_size,
            force_rerun=force_rerun,
        )
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, _cache_attrs = _load_h5(cache_path)

    tangent_key = "DS" if "DS" in arrays_all else "tangent"
    if tangent_key not in arrays_all:
        summary = {
            "status": "blocked_no_tangent",
            "reason": "Grouped dataset does not contain DS/tangent tensors.",
        }
        _write_json(summary_path, summary)
        _write_json(
            fe_status_path,
            {
                "status": "skipped_no_reusable_harness",
                "reason": "No in-repo FE shadow harness is available for the canonical benchmark.",
            },
        )
        return summary

    summary_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    teacher_model, teacher_metadata = load_checkpoint(TEACHER_CHECKPOINT, device=device)
    for split_idx, split_name in enumerate(("val", "test"), start=1):
        mask = _split_mask(arrays_all["split_id"], split_name) & panel_all["ds_valid_mask"].astype(bool)
        split_arrays = {
            "strain_eng": arrays_all["strain_eng"][mask].astype(np.float32),
            "material_reduced": arrays_all["material_reduced"][mask].astype(np.float32),
            "branch_id": arrays_all["branch_id"][mask].astype(np.int8),
            "source_call_id": arrays_all["source_call_id"][mask].astype(np.int32),
            "source_row_in_call": arrays_all["source_row_in_call"][mask].astype(np.int32),
            "hard_mask": panel_all["hard_mask"][mask].astype(np.int8),
            "plastic_mask": panel_all["plastic_mask"][mask].astype(np.int8),
            "ds_valid_mask": panel_all["ds_valid_mask"][mask].astype(np.int8),
            "near_yield_mask": panel_all["near_yield_mask"][mask].astype(np.int8),
            "near_smooth_left_mask": panel_all["near_smooth_left_mask"][mask].astype(np.int8),
            "near_smooth_right_mask": panel_all["near_smooth_right_mask"][mask].astype(np.int8),
            "near_left_apex_mask": panel_all["near_left_apex_mask"][mask].astype(np.int8),
            "near_right_apex_mask": panel_all["near_right_apex_mask"][mask].astype(np.int8),
            "teacher_checkpoint_stress": cache_arrays["teacher_checkpoint_stress"][mask].astype(np.float32),
            "teacher_dispatch_stress": cache_arrays["teacher_dispatch_stress"][mask].astype(np.float32),
            "teacher_dispatch_stress_principal": cache_arrays["teacher_dispatch_stress_principal"][mask].astype(np.float32),
            "teacher_projected_stress": cache_arrays["teacher_projected_stress"][mask].astype(np.float32),
            "teacher_projected_stress_principal": cache_arrays["teacher_projected_stress_principal"][mask].astype(np.float32),
            "DS": arrays_all[tangent_key][mask].astype(np.float32),
        }
        directions = _probe_directions(split_arrays["strain_eng"].shape[0], seed=PROBE_SEED + split_idx, n_dirs=PROBE_DIRECTIONS)
        exact_rowwise, exact_summary = _probe_mode_on_split(
            split_arrays,
            teacher_model=teacher_model,
            teacher_metadata=teacher_metadata,
            mode="exact",
            directions=directions,
            device=device,
            eval_batch_size=eval_batch_size,
        )
        softmin_rowwise, softmin_summary = _probe_mode_on_split(
            split_arrays,
            teacher_model=teacher_model,
            teacher_metadata=teacher_metadata,
            mode="softmin",
            directions=directions,
            device=device,
            eval_batch_size=eval_batch_size,
        )

        softmin_forward = project_teacher_checkpoint_stress(
            split_arrays["strain_eng"],
            split_arrays["material_reduced"],
            split_arrays["teacher_checkpoint_stress"],
            mode="softmin",
            tau=PROJECTION_TAU,
        )
        forward_stress_mae = np.mean(
            np.abs(softmin_forward["teacher_projected_stress"] - split_arrays["teacher_projected_stress"]),
            axis=1,
        ).astype(np.float32)
        forward_principal_max_abs = np.max(
            np.abs(softmin_forward["teacher_projected_stress_principal"] - split_arrays["teacher_projected_stress_principal"]),
            axis=1,
        ).astype(np.float32)
        softmin_summary["forward_delta_vs_exact"] = {
            "stress_component_mae": summarize_array(forward_stress_mae),
            "principal_max_abs": summarize_array(forward_principal_max_abs),
        }

        any_boundary_exact = exact_summary["boundary_instability"]["any_boundary"]["candidate_switch_rate"]
        any_boundary_softmin = softmin_summary["boundary_instability"]["any_boundary"]["candidate_switch_rate"]
        ds_status = "blocked"
        if forward_stress_mae.size and forward_stress_mae.mean() <= 0.5 and any_boundary_softmin < any_boundary_exact:
            ds_status = "viable"
        elif forward_stress_mae.size and forward_stress_mae.mean() <= 1.0 and any_boundary_softmin <= any_boundary_exact:
            ds_status = "mixed"

        split_summary = {
            "n_rows": int(split_arrays["strain_eng"].shape[0]),
            "n_plastic_rows": int(np.sum(split_arrays["branch_id"] > 0)),
            "exact": exact_summary,
            "softmin": softmin_summary,
            "ds_status": ds_status,
        }
        split_summaries[split_name] = split_summary

        summary_rows.extend(_probe_summary_rows(split_name, "exact", "overall", exact_summary["overall"]))
        summary_rows.extend(_probe_summary_rows(split_name, "exact", "plastic_only", exact_summary["plastic_only"]))
        summary_rows.extend(_probe_summary_rows(split_name, "softmin", "overall", softmin_summary["overall"]))
        summary_rows.extend(_probe_summary_rows(split_name, "softmin", "plastic_only", softmin_summary["plastic_only"]))

        _write_h5(
            phase_dir / f"teacher_ds_probe_{split_name}.h5",
            {
                **exact_rowwise,
                **softmin_rowwise,
                "softmin_forward_stress_component_mae": forward_stress_mae.astype(np.float32),
                "softmin_forward_principal_max_abs": forward_principal_max_abs.astype(np.float32),
            },
            {
                "split": split_name,
                "n_directions": PROBE_DIRECTIONS,
                "fd_scale": FD_SCALE,
                "projection_tau": PROJECTION_TAU,
                "teacher_checkpoint": str(TEACHER_CHECKPOINT),
            },
        )

    overall_status = "blocked"
    if any(split_summaries[name]["ds_status"] == "viable" for name in split_summaries):
        overall_status = "viable"
    elif any(split_summaries[name]["ds_status"] == "mixed" for name in split_summaries):
        overall_status = "mixed"
    summary = {
        "status": overall_status,
        "projection_tau": PROJECTION_TAU,
        "n_directions": PROBE_DIRECTIONS,
        "fd_scale": FD_SCALE,
        "split_summaries": split_summaries,
    }
    _write_json(summary_path, summary)
    _write_csv(
        csv_path,
        summary_rows,
        ["split", "mode", "scope", "metric", "mean", "p50", "p95", "max", "candidate_switch_rate"],
    )
    _write_json(
        fe_status_path,
        {
            "status": "skipped_no_reusable_harness",
            "reason": "No in-repo FE shadow harness is available for the canonical benchmark.",
        },
    )
    return summary


def run_phase2_preservation_dataset(
    *,
    output_root: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp2_preservation_dataset"
    phase_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = phase_dir / "projection_student_preservation_plastic.h5"
    summary_path = phase_dir / "preservation_dataset_summary.json"
    if dataset_path.exists() and summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    cache_path = output_root / "exp0_teacher_cache" / "teacher_projection_preservation_cache.h5"
    if not cache_path.exists():
        raise FileNotFoundError(f"Teacher cache is required before preservation dataset build: {cache_path}")
    cache_arrays, cache_attrs = _load_h5(cache_path)
    arrays_all, dataset_attrs = _load_h5(GROUPED_DATASET)
    _panel_arrays, panel_attrs = _load_h5(PANEL_DATASET)

    plastic_mask = cache_arrays["plastic_mask"].astype(bool)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(arrays_all["strain_eng"])
    geom_features = build_trial_principal_geom_features(
        strain_principal,
        arrays_all["material_reduced"],
        cache_arrays["trial_principal"],
    ).astype(np.float32)
    sampling_weight, disp_threshold = build_sampling_weights(
        train_mask=cache_arrays["split_id"][plastic_mask] == SPLIT_NAME_TO_ID["train"],
        hard_mask=cache_arrays["hard_mask"][plastic_mask] > 0,
        teacher_projection_candidate_id=cache_arrays["teacher_projection_candidate_id"][plastic_mask],
        teacher_projection_disp_norm=cache_arrays["teacher_projection_disp_norm"][plastic_mask],
    )

    derived_arrays = {
        "split_id": cache_arrays["split_id"][plastic_mask].astype(np.int8),
        "source_call_id": cache_arrays["source_call_id"][plastic_mask].astype(np.int32),
        "source_row_in_call": cache_arrays["source_row_in_call"][plastic_mask].astype(np.int32),
        "strain_eng": arrays_all["strain_eng"][plastic_mask].astype(np.float32),
        "stress": cache_arrays["exact_stress"][plastic_mask].astype(np.float32),
        "stress_principal": cache_arrays["exact_stress_principal"][plastic_mask].astype(np.float32),
        "material_reduced": arrays_all["material_reduced"][plastic_mask].astype(np.float32),
        "eigvecs": arrays_all["eigvecs"][plastic_mask].astype(np.float32),
        "branch_id": cache_arrays["branch_id"][plastic_mask].astype(np.int8),
        "trial_principal": cache_arrays["trial_principal"][plastic_mask].astype(np.float32),
        "hard_mask": cache_arrays["hard_mask"][plastic_mask].astype(np.int8),
        "ds_valid_mask": cache_arrays["ds_valid_mask"][plastic_mask].astype(np.int8),
        "plastic_mask": np.ones(int(np.sum(plastic_mask)), dtype=np.int8),
        "teacher_stress_principal": cache_arrays["teacher_dispatch_stress_principal"][plastic_mask].astype(np.float32),
        "teacher_provisional_stress_principal": cache_arrays["teacher_dispatch_stress_principal"][plastic_mask].astype(np.float32),
        "teacher_projected_stress_principal": cache_arrays["teacher_projected_stress_principal"][plastic_mask].astype(np.float32),
        "teacher_projection_delta_principal": cache_arrays["teacher_projection_delta_principal"][plastic_mask].astype(np.float32),
        "teacher_projection_candidate_id": cache_arrays["teacher_projection_candidate_id"][plastic_mask].astype(np.int8),
        "teacher_projection_disp_norm": cache_arrays["teacher_projection_disp_norm"][plastic_mask].astype(np.float32),
        "sampling_weight": sampling_weight.astype(np.float32),
        "trial_principal_geom_feature_f1": geom_features[plastic_mask].astype(np.float32),
    }
    _write_h5(
        dataset_path,
        derived_arrays,
        {
            "source_teacher_cache": str(cache_path),
            "source_grouped_dataset": str(GROUPED_DATASET),
            "source_panel_dataset": str(PANEL_DATASET),
            "teacher_checkpoint": str(TEACHER_CHECKPOINT),
            "teacher_summary": str(TEACHER_SUMMARY),
            "teacher_projection_disp_p90_train": disp_threshold,
            "sampling_weight_rule": {
                "base": 1.0,
                "hard_mask": 1.5,
                "edge_candidate": 1.5,
                "high_disp_train_p90": 1.5,
                "max_by_construction": 3.375,
            },
            "cache_attrs_json": cache_attrs,
            "dataset_attrs_json": dataset_attrs,
            "panel_attrs_json": panel_attrs,
        },
    )

    split_id = derived_arrays["split_id"]
    summary = {
        "dataset_path": str(dataset_path),
        "n_rows": int(derived_arrays["stress"].shape[0]),
        "split_counts": {name: int(np.sum(split_id == idx)) for name, idx in SPLIT_NAME_TO_ID.items()},
        "hard_counts": {
            name: int(np.sum((split_id == idx) & (derived_arrays["hard_mask"] > 0)))
            for name, idx in SPLIT_NAME_TO_ID.items()
        },
        "ds_valid_counts": {
            name: int(np.sum((split_id == idx) & (derived_arrays["ds_valid_mask"] > 0)))
            for name, idx in SPLIT_NAME_TO_ID.items()
        },
        "sampling_weight_summary": summarize_array(derived_arrays["sampling_weight"]),
        "teacher_projection_disp_norm_summary": summarize_array(derived_arrays["teacher_projection_disp_norm"]),
    }
    _write_json(summary_path, summary)
    return summary


def _projected_student_config(
    *,
    dataset_path: Path,
    run_dir: Path,
    width: int,
    depth: int,
    seed: int,
    init_checkpoint: str | None,
) -> TrainingConfig:
    return TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_geom_projected_student",
        epochs=60,
        batch_size=4096,
        lr=3.0e-4,
        weight_decay=1.0e-5,
        width=width,
        depth=depth,
        dropout=0.0,
        seed=seed,
        patience=12,
        grad_clip=1.0,
        branch_loss_weight=0.0,
        num_workers=0,
        device="auto",
        scheduler_kind="plateau",
        plateau_factor=0.5,
        regression_loss_kind="huber",
        huber_delta=1.0,
        checkpoint_metric="loss",
        log_every_epochs=5,
        projection_mode="exact",
        projection_tau=PROJECTION_TAU,
        init_checkpoint=init_checkpoint,
    )


def run_phase3_preservation_control(
    *,
    output_root: Path,
    docs_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp3_preservation_control"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase3_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    dataset_path = output_root / "exp2_preservation_dataset" / "projection_student_preservation_plastic.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preservation dataset is required before the control run: {dataset_path}")
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    medium_exact = _medium_exact_reference()

    run_dir = phase_dir / "control_exact_w256_d4"
    config = _projected_student_config(
        dataset_path=dataset_path,
        run_dir=run_dir,
        width=256,
        depth=4,
        seed=20260327,
        init_checkpoint=str(CANDIDATE_B_WARMSTART),
    )
    train_summary = _train_with_batch_fallback(config, force_rerun=force_rerun)
    best_checkpoint = Path(train_summary["best_checkpoint"])
    val_eval = _evaluate_projected_checkpoint_on_split(
        best_checkpoint,
        arrays_all,
        panel_all,
        split_name="val",
        device=device,
        eval_batch_size=eval_batch_size,
    )
    test_eval = _evaluate_projected_checkpoint_on_split(
        best_checkpoint,
        arrays_all,
        panel_all,
        split_name="test",
        device=device,
        eval_batch_size=eval_batch_size,
    )
    row = {
        "width": 256,
        "depth": 4,
        "param_count": _parameter_count(best_checkpoint, device=device),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "completed_epochs": int(train_summary["completed_epochs"]),
        "best_epoch": int(train_summary["best_epoch"]),
        "val_broad_plastic_mae": float(val_eval["metrics"]["broad_plastic_mae"]),
        "val_hard_plastic_mae": float(val_eval["metrics"]["hard_plastic_mae"]),
        "val_hard_p95_principal": float(val_eval["metrics"]["hard_p95_principal"]),
        "val_yield_violation_p95": float(val_eval["metrics"]["yield_violation_p95"]),
        "test_broad_plastic_mae": float(test_eval["metrics"]["broad_plastic_mae"]),
        "test_hard_plastic_mae": float(test_eval["metrics"]["hard_plastic_mae"]),
        "test_hard_p95_principal": float(test_eval["metrics"]["hard_p95_principal"]),
        "test_yield_violation_p95": float(test_eval["metrics"]["yield_violation_p95"]),
    }
    _write_json(run_dir / "projected_student_eval_val.json", val_eval["metrics"])
    _write_json(run_dir / "projected_student_eval_test.json", test_eval["metrics"])

    beats_medium_exact_all = (
        row["val_broad_plastic_mae"] < medium_exact["broad_plastic_mae"]
        and row["val_hard_plastic_mae"] < medium_exact["hard_plastic_mae"]
        and row["val_hard_p95_principal"] < medium_exact["hard_p95_principal"]
        and row["val_yield_violation_p95"] <= 1.0e-6
    )
    opens_tier1 = beats_medium_exact_all and _meets_bar(
        {
            "broad_plastic_mae": row["val_broad_plastic_mae"],
            "hard_plastic_mae": row["val_hard_plastic_mae"],
            "hard_p95_principal": row["val_hard_p95_principal"],
            "yield_violation_p95": row["val_yield_violation_p95"],
        },
        TIER1_GATE,
    )
    summary = {
        "control": row,
        "medium_exact_reference": medium_exact,
        "beats_medium_exact_all": beats_medium_exact_all,
        "opens_tier1": opens_tier1,
    }
    _write_json(summary_path, summary)
    return summary


def _run_architecture_sweep(
    *,
    phase_dir: Path,
    dataset_path: Path,
    arch_specs: tuple[dict[str, Any], ...],
    arrays_all: dict[str, np.ndarray],
    panel_all: dict[str, np.ndarray],
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arch in arch_specs:
        run_dir = phase_dir / arch["name"]
        config = _projected_student_config(
            dataset_path=dataset_path,
            run_dir=run_dir,
            width=int(arch["width"]),
            depth=int(arch["depth"]),
            seed=seed,
            init_checkpoint=None,
        )
        train_summary = _train_with_batch_fallback(config, force_rerun=force_rerun)
        best_checkpoint = Path(train_summary["best_checkpoint"])
        val_eval = _evaluate_projected_checkpoint_on_split(
            best_checkpoint,
            arrays_all,
            panel_all,
            split_name="val",
            device=device,
            eval_batch_size=eval_batch_size,
        )
        test_eval = _evaluate_projected_checkpoint_on_split(
            best_checkpoint,
            arrays_all,
            panel_all,
            split_name="test",
            device=device,
            eval_batch_size=eval_batch_size,
        )
        row = {
            "architecture": arch["name"],
            "width": int(arch["width"]),
            "depth": int(arch["depth"]),
            "param_count": _parameter_count(best_checkpoint, device=device),
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint),
            "completed_epochs": int(train_summary["completed_epochs"]),
            "best_epoch": int(train_summary["best_epoch"]),
            "val_broad_plastic_mae": float(val_eval["metrics"]["broad_plastic_mae"]),
            "val_hard_plastic_mae": float(val_eval["metrics"]["hard_plastic_mae"]),
            "val_hard_p95_principal": float(val_eval["metrics"]["hard_p95_principal"]),
            "val_yield_violation_p95": float(val_eval["metrics"]["yield_violation_p95"]),
            "test_broad_plastic_mae": float(test_eval["metrics"]["broad_plastic_mae"]),
            "test_hard_plastic_mae": float(test_eval["metrics"]["hard_plastic_mae"]),
            "test_hard_p95_principal": float(test_eval["metrics"]["hard_p95_principal"]),
            "test_yield_violation_p95": float(test_eval["metrics"]["yield_violation_p95"]),
        }
        _write_json(run_dir / "projected_student_eval_val.json", val_eval["metrics"])
        _write_json(run_dir / "projected_student_eval_test.json", test_eval["metrics"])
        rows.append(row)
    return rows


def run_phase4_tier1_compression(
    *,
    output_root: Path,
    docs_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp4_tier1_compression"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase4_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase3 = _read_json(output_root / "exp3_preservation_control" / "phase3_summary.json")
    if not phase3["opens_tier1"]:
        summary = {
            "status": "skipped",
            "reason": "Preservation control did not clear the Tier 1 opening gate.",
            "control": phase3["control"],
        }
        _write_json(summary_path, summary)
        return summary

    dataset_path = output_root / "exp2_preservation_dataset" / "projection_student_preservation_plastic.h5"
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    rows = _run_architecture_sweep(
        phase_dir=phase_dir,
        dataset_path=dataset_path,
        arch_specs=TIER1_ARCHS,
        arrays_all=arrays_all,
        panel_all=panel_all,
        device=device,
        eval_batch_size=eval_batch_size,
        force_rerun=force_rerun,
        seed=20260327,
    )
    _write_csv(
        phase_dir / "tier1_architecture_summary.csv",
        rows,
        [
            "architecture",
            "width",
            "depth",
            "param_count",
            "run_dir",
            "best_checkpoint",
            "completed_epochs",
            "best_epoch",
            "val_broad_plastic_mae",
            "val_hard_plastic_mae",
            "val_hard_p95_principal",
            "val_yield_violation_p95",
            "test_broad_plastic_mae",
            "test_hard_plastic_mae",
            "test_hard_p95_principal",
            "test_yield_violation_p95",
        ],
    )
    winner = min(rows, key=_ranking_key)
    tier2_open = _meets_bar(
        {
            "broad_plastic_mae": winner["val_broad_plastic_mae"],
            "hard_plastic_mae": winner["val_hard_plastic_mae"],
            "hard_p95_principal": winner["val_hard_p95_principal"],
            "yield_violation_p95": winner["val_yield_violation_p95"],
        },
        TIER2_GATE,
    )
    summary = {
        "status": "completed",
        "all_runs": rows,
        "winner": winner,
        "tier2_open": tier2_open,
    }
    _write_json(summary_path, summary)
    return summary


def run_phase5_tier2_compression(
    *,
    output_root: Path,
    docs_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp5_tier2_compression"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase5_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase4 = _read_json(output_root / "exp4_tier1_compression" / "phase4_summary.json")
    if phase4.get("status") != "completed" or not phase4.get("tier2_open", False):
        summary = {
            "status": "skipped",
            "reason": "Tier 1 did not clear the Tier 2 opening gate.",
            "tier1": phase4,
        }
        _write_json(summary_path, summary)
        return summary

    dataset_path = output_root / "exp2_preservation_dataset" / "projection_student_preservation_plastic.h5"
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    rows = _run_architecture_sweep(
        phase_dir=phase_dir,
        dataset_path=dataset_path,
        arch_specs=TIER2_ARCHS,
        arrays_all=arrays_all,
        panel_all=panel_all,
        device=device,
        eval_batch_size=eval_batch_size,
        force_rerun=force_rerun,
        seed=20260328,
    )
    _write_csv(
        phase_dir / "tier2_architecture_summary.csv",
        rows,
        [
            "architecture",
            "width",
            "depth",
            "param_count",
            "run_dir",
            "best_checkpoint",
            "completed_epochs",
            "best_epoch",
            "val_broad_plastic_mae",
            "val_hard_plastic_mae",
            "val_hard_p95_principal",
            "val_yield_violation_p95",
            "test_broad_plastic_mae",
            "test_hard_plastic_mae",
            "test_hard_p95_principal",
            "test_yield_violation_p95",
        ],
    )
    summary = {
        "status": "completed",
        "all_runs": rows,
        "winner": min(rows, key=_ranking_key),
    }
    _write_json(summary_path, summary)
    return summary


def write_execution_report(
    *,
    output_root: Path,
    docs_root: Path,
) -> None:
    docs_path = docs_root / "executions" / WORK_PACKET_BASENAME
    phase0_path = output_root / "exp0_teacher_cache" / "teacher_projection_preservation_summary.json"
    phase1_path = output_root / "exp1_teacher_ds_probe" / "teacher_ds_probe_summary.json"
    phase2_path = output_root / "exp2_preservation_dataset" / "preservation_dataset_summary.json"
    phase3_path = output_root / "exp3_preservation_control" / "phase3_summary.json"
    phase4_path = output_root / "exp4_tier1_compression" / "phase4_summary.json"
    phase5_path = output_root / "exp5_tier2_compression" / "phase5_summary.json"
    fe_path = output_root / "fe_shadow_status.json"

    phase0 = _read_json(phase0_path) if phase0_path.exists() else None
    phase1 = _read_json(phase1_path) if phase1_path.exists() else None
    phase2 = _read_json(phase2_path) if phase2_path.exists() else None
    phase3 = _read_json(phase3_path) if phase3_path.exists() else None
    phase4 = _read_json(phase4_path) if phase4_path.exists() else None
    phase5 = _read_json(phase5_path) if phase5_path.exists() else None
    fe_status = _read_json(fe_path) if fe_path.exists() else None
    medium_exact = _medium_exact_reference()

    task_path = ROOT / "docs/tasks/projection_student_preservation_compression_work_packet_20260327.md"

    def _fmt_metric_tuple(metrics: dict[str, Any]) -> str:
        return (
            f"`{float(metrics['broad_plastic_mae']):.6f}` / "
            f"`{float(metrics['hard_plastic_mae']):.6f}` / "
            f"`{float(metrics['hard_p95_principal']):.6f}` / "
            f"`{float(metrics['yield_violation_p95']):.6e}`"
        )

    def _fmt_summary_stats(stats: dict[str, Any]) -> str:
        return (
            f"`mean {float(stats['mean']):.6f}`, "
            f"`p50 {float(stats['p50']):.6f}`, "
            f"`p95 {float(stats['p95']):.6f}`, "
            f"`max {float(stats['max']):.6f}`"
        )

    def _fmt_bool(value: bool) -> str:
        return "`True`" if bool(value) else "`False`"

    teacher_gate = {
        "val": phase0["split_summaries"]["val"]["projected_teacher"],
        "test": phase0["split_summaries"]["test"]["projected_teacher"],
    } if phase0 else None
    ds_reopen_bar = {
        "broad_plastic_mae": 10.0,
        "hard_plastic_mae": 12.0,
        "hard_p95_principal": 120.0,
        "yield_violation_p95": 1.0e-6,
    }
    ds_still_blocked = True
    if phase3:
        ds_still_blocked = not _meets_bar(
            {
                "broad_plastic_mae": float(phase3["control"]["val_broad_plastic_mae"]),
                "hard_plastic_mae": float(phase3["control"]["val_hard_plastic_mae"]),
                "hard_p95_principal": float(phase3["control"]["val_hard_p95_principal"]),
                "yield_violation_p95": float(phase3["control"]["val_yield_violation_p95"]),
            },
            ds_reopen_bar,
        )
    compression_verdict = "not_run"
    if phase4:
        if phase4.get("status") == "completed":
            compression_verdict = "alive" if phase4.get("tier2_open", False) else "mixed"
        else:
            compression_verdict = "dead"
    route_decision = "stop"
    if phase3 and phase3.get("beats_medium_exact_all", False):
        route_decision = "continue" if phase3.get("opens_tier1", False) else "narrow"
    elif phase1 and phase1.get("status") in {"viable", "mixed"}:
        route_decision = "narrow"

    lines: list[str] = [
        "# Projection-Student Preservation Compression Work Packet",
        "",
        "Execution report for the March 27, 2026 `projection-student` preservation/compression packet.",
        "",
        "Paired task memo:",
        f"[`docs/tasks/projection_student_preservation_compression_work_packet_20260327.md`]({task_path})",
        "",
        "Repository:",
        f"`{ROOT}`",
        "",
        "Output root:",
        f"`{output_root}`",
        "",
        "## Strategic Setup",
        "",
        "This packet continued the only still-live direct-replacement route after the March 26 projection audit:",
        "",
        "- preserve the projected teacher first",
        "- compress only after preservation works",
        "- keep `projection-student` as the only active direct-replacement family",
        "",
        "What remained closed during this packet:",
        "",
        "- packet2 / packet3 / packet4 as active research families",
        "- Program X follow-ons",
        "- packet5",
        "- new atlas variants",
        "- routing redesign",
        "- exact-latent follow-ons",
        "- separate `DS` heads",
        "",
        "Controlling scientific questions:",
        "",
        "- Is the frozen projected teacher numerically stable enough to justify later tangent work?",
        "- Can a same-capacity projected student preserve the projected teacher materially better than the March 26 `medium_exact` baseline?",
        "- Does the route clear the explicit gate for bounded compression, or should compression stay closed?",
        "",
        "## Benchmark and Anchors",
        "",
        "Canonical benchmark:",
        "",
        "- grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`",
        "- panel sidecar: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5`",
        "- split seed: `20260324`",
        "- split unit: constitutive call",
        "- split fractions: `70 / 15 / 15`",
        "- samples per call: `512`",
        "",
        "Canonical sizes:",
        "",
        "- `train`: `198656`",
        "- `val`: `42496`",
        "- `test`: `42496`",
        "- `hard_val`: `21198`",
        "- `hard_test`: `22705`",
        "- `ds_valid`: `73261`",
        "- `ds_valid_val`: `11224`",
        "- `ds_valid_test`: `9839`",
        "",
        "Teacher used:",
        "",
        "- checkpoint: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt`",
        "- summary: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/summary.json`",
        "",
        "March 26 continuation anchors:",
        "",
        f"- projected teacher validation broad / hard / p95 / yield: {_fmt_metric_tuple(teacher_gate['val']) if teacher_gate else '`not_run`'}",
        f"- projected teacher test broad / hard / p95 / yield: {_fmt_metric_tuple(teacher_gate['test']) if teacher_gate else '`not_run`'}",
        f"- March 26 `medium_exact` validation broad / hard / p95 / yield: {_fmt_metric_tuple(medium_exact)}",
        "",
        "Compression and `DS` gates carried into this packet:",
        "",
        "- Tier 1 open bar: `broad <= 18`, `hard <= 20`, `hard_p95 <= 180`, `yield <= 1e-6`",
        "- broader `DS` reopen bar: `broad <= 10`, `hard <= 12`, `hard_p95 <= 120`, `yield <= 1e-6`",
        "",
        "## Implementation",
        "",
        "Primary code paths used or extended in this packet:",
        "",
        f"- [`scripts/experiments/run_projection_student_preservation_compression.py`]({ROOT / 'scripts/experiments/run_projection_student_preservation_compression.py'})",
        f"- [`src/mc_surrogate/projection_student_preservation.py`]({ROOT / 'src/mc_surrogate/projection_student_preservation.py'})",
        f"- [`src/mc_surrogate/training.py`]({ROOT / 'src/mc_surrogate/training.py'})",
        f"- [`src/mc_surrogate/principal_projection.py`]({ROOT / 'src/mc_surrogate/principal_projection.py'})",
        f"- [`tests/test_projection_student_preservation.py`]({ROOT / 'tests/test_projection_student_preservation.py'})",
        f"- [`tests/test_training.py`]({ROOT / 'tests/test_training.py'})",
        "",
        "### Phase A0: candidate-zero teacher cache",
        "",
        "Implemented a full-row cache that joins the frozen teacher checkpoint outputs with the canonical grouped dataset and panel sidecar. The cache stores:",
        "",
        "- exact stress and exact principal stress",
        "- exact trial principal stress",
        "- teacher checkpoint stress before elastic dispatch",
        "- elastic-dispatched teacher stress",
        "- exact-projected teacher stress",
        "- projection displacement vector and norm",
        "- selected projection candidate, full candidate tensor, squared distances, and feasible-mask audit",
        "- split ids, call-local row ids, branch ids, hard masks, `ds_valid` masks, and boundary-nearness masks",
        "",
        "### Phase A1: projected-teacher `DS` probe",
        "",
        "Implemented a finite-difference directional tangent probe on canonical `ds_valid_val` and `ds_valid_test` rows:",
        "",
        "- centered finite differences with `h = 1e-6 * max(max(|strain|), 1)`",
        "- `3` fixed random unit directions per row from seed `20260327`",
        "- both `exact` and `softmin(tau=0.05)` projection modes",
        "- per-row outputs for `||Jv_pred - DS_true v||`, relative error, cosine similarity, base / plus / minus candidate ids, and candidate-switch events",
        "- explicit boundary-instability summaries over the saved `near_*` masks",
        "",
        "Implementation note:",
        "",
        "- the probe path was tightened to reuse a loaded teacher checkpoint and batch `+h/-h` perturbations together, so the packet can run the full probe without repeatedly reloading the same checkpoint",
        "",
        "### Phase B0 and B1: preservation dataset and same-capacity control",
        "",
        "Extended the projected-student training path to consume preservation targets and sample weights:",
        "",
        "- optional dataset keys: `teacher_provisional_stress_principal`, `teacher_projected_stress_principal`, `teacher_projection_delta_principal`, `teacher_projection_candidate_id`, `teacher_projection_disp_norm`, `ds_valid_mask`, `sampling_weight`, and `trial_principal_geom_feature_f1`",
        "- backward compatibility: legacy `teacher_stress_principal` still works as the provisional-teacher alias",
        "- training now auto-uses `WeightedRandomSampler` when `sampling_weight` is present",
        "- projected-student loss now follows the preservation objective: exact-principal fit, projected-teacher preservation, provisional-teacher preservation, projection-delta preservation, and branch CE over the existing 4-way plastic head",
        "",
        "Preservation dataset weighting rule used in this packet:",
        "",
        "- base `1.0`",
        "- `x1.5` for `hard_mask`",
        "- `x1.5` for teacher candidate in `{left_edge, right_edge}`",
        "- `x1.5` for `teacher_projection_disp_norm >= train p90`",
        "",
        "Same-capacity control configuration:",
        "",
        "- model: `trial_principal_geom_projected_student`",
        "- width / depth: `256 / 4`",
        "- params: `540935`",
        "- projection mode: `exact`",
        "- batch size: `4096`",
        "- lr: `3e-4`",
        "- weight decay: `1e-5`",
        "- patience: `12`",
        "- epochs: `60`",
        "- seed: `20260327`",
        "- warm start: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/candidate_b/finetune/best.pt`",
        "",
        "### Phase C: bounded compression gating",
        "",
        "Compression logic stayed strict in this packet:",
        "",
        "- open Tier 1 only if the same-capacity preservation control beats `medium_exact` on all three primary validation stress metrics, preserves numerical-zero yield, and clears the explicit `18 / 20 / 180 / 1e-6` gate",
        "- open Tier 2 only if a Tier 1 winner clears the stricter `15 / 18 / 160 / 1e-6` gate",
        "- if the same-capacity control misses the Tier 1 bar, compression stops rather than expanding the search",
        "",
        "## Results",
        "",
    ]

    if phase0:
        val_dispatch = phase0["split_summaries"]["val"]["dispatch_teacher"]
        val_projected = phase0["split_summaries"]["val"]["projected_teacher"]
        test_dispatch = phase0["split_summaries"]["test"]["dispatch_teacher"]
        test_projected = phase0["split_summaries"]["test"]["projected_teacher"]
        lines.extend(
            [
                "### Phase A0: candidate-zero teacher cache",
                "",
                "Validation:",
                "",
                "| Evaluation | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Dispatch teacher | {float(val_dispatch['broad_plastic_mae']):.6f} | {float(val_dispatch['hard_plastic_mae']):.6f} | {float(val_dispatch['hard_p95_principal']):.6f} | {float(val_dispatch['yield_violation_p95']):.6e} |",
                f"| Projected teacher | {float(val_projected['broad_plastic_mae']):.6f} | {float(val_projected['hard_plastic_mae']):.6f} | {float(val_projected['hard_p95_principal']):.6f} | {float(val_projected['yield_violation_p95']):.6e} |",
                "",
                "Test:",
                "",
                "| Evaluation | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Dispatch teacher | {float(test_dispatch['broad_plastic_mae']):.6f} | {float(test_dispatch['hard_plastic_mae']):.6f} | {float(test_dispatch['hard_p95_principal']):.6f} | {float(test_dispatch['yield_violation_p95']):.6e} |",
                f"| Projected teacher | {float(test_projected['broad_plastic_mae']):.6f} | {float(test_projected['hard_plastic_mae']):.6f} | {float(test_projected['hard_p95_principal']):.6f} | {float(test_projected['yield_violation_p95']):.6e} |",
                "",
                "Projection displacement summary on plastic rows:",
                "",
                f"- validation: {_fmt_summary_stats(phase0['split_summaries']['val']['projection_disp_norm'])}",
                f"- test: {_fmt_summary_stats(phase0['split_summaries']['test']['projection_disp_norm'])}",
                f"- validation candidate counts: `{phase0['split_summaries']['val']['projection_candidate_counts']}`",
                f"- test candidate counts: `{phase0['split_summaries']['test']['projection_candidate_counts']}`",
                "",
                "Interpretation:",
                "",
                "- candidate zero remained scientifically strong: projection drove yield back to numerical zero while preserving the frozen teacher's forward-stress quality close to the March 26 audit",
                "",
            ]
        )

    if phase1:
        lines.extend(
            [
                "### Phase A1: projected-teacher `DS` probe",
                "",
                "| Split | Rows | Plastic rows | Exact plastic rel-error p95 | Exact plastic cosine mean | Exact boundary switch | Softmin plastic rel-error p95 | Softmin plastic cosine mean | Softmin boundary switch | Softmin forward stress MAE vs exact | Status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for split_name in ("val", "test"):
            payload = phase1["split_summaries"][split_name]
            exact = payload["exact"]["plastic_only"]
            softmin = payload["softmin"]["plastic_only"]
            lines.append(
                f"| {split_name} | {int(payload['n_rows'])} | {int(payload['n_plastic_rows'])} | "
                f"{float(exact['relative_error']['p95']):.6f} | {float(exact['cosine_similarity']['mean']):.6f} | "
                f"{float(payload['exact']['boundary_instability']['any_boundary']['candidate_switch_rate']):.6f} | "
                f"{float(softmin['relative_error']['p95']):.6f} | {float(softmin['cosine_similarity']['mean']):.6f} | "
                f"{float(payload['softmin']['boundary_instability']['any_boundary']['candidate_switch_rate']):.6f} | "
                f"{float(payload['softmin']['forward_delta_vs_exact']['stress_component_mae']['mean']):.6f} | `{payload['ds_status']}` |"
            )
        lines.extend(
            [
                "",
                "Boundary observations:",
                "",
                f"- validation exact boundary switch rate: `{phase1['split_summaries']['val']['exact']['boundary_instability']['any_boundary']['candidate_switch_rate']:.6f}`",
                f"- validation softmin boundary switch rate: `{phase1['split_summaries']['val']['softmin']['boundary_instability']['any_boundary']['candidate_switch_rate']:.6f}`",
                f"- test exact boundary switch rate: `{phase1['split_summaries']['test']['exact']['boundary_instability']['any_boundary']['candidate_switch_rate']:.6f}`",
                f"- test softmin boundary switch rate: `{phase1['split_summaries']['test']['softmin']['boundary_instability']['any_boundary']['candidate_switch_rate']:.6f}`",
                "",
                "Interpretation:",
                "",
                f"- overall probe status: `{phase1['status']}`",
                f"- validation / test status: `{phase1['split_summaries']['val']['ds_status']}` / `{phase1['split_summaries']['test']['ds_status']}`",
                "- candidate-switch rates were very small, so the projection map was not thrashing across candidate boundaries",
                "- the remaining problem was tangent quality: directional relative-error tails stayed too large, especially on plastic and boundary-near rows",
                "- softmin did not create a clean forward-equivalent tangent path in validation because its forward stress delta versus exact stayed around one stress unit on average",
                "",
            ]
        )

    lines.extend(
        [
            "### Phase A2: FE shadow",
            "",
            f"- status: `{fe_status['status'] if fe_status else 'not_run'}`",
            f"- reason: `{fe_status['reason'] if fe_status else 'not_run'}`",
            "- no new FE harness was built in this packet, per the task stop rule",
            "",
        ]
    )

    if phase2:
        lines.extend(
            [
                "### Phase B0: preservation dataset",
                "",
                f"- dataset path: `{phase2['dataset_path']}`",
                f"- rows: `{phase2['n_rows']}`",
                f"- split counts: `{phase2['split_counts']}`",
                f"- hard counts: `{phase2['hard_counts']}`",
                f"- `ds_valid` counts: `{phase2['ds_valid_counts']}`",
                f"- sampling-weight summary: {_fmt_summary_stats(phase2['sampling_weight_summary'])}",
                f"- teacher projection-displacement summary: {_fmt_summary_stats(phase2['teacher_projection_disp_norm_summary'])}",
                "",
                "Interpretation:",
                "",
                "- the preservation dataset captured the projected teacher as a reusable supervision target rather than forcing every later run to rebuild projections from scratch",
                "- weighting biased sampling toward hard rows, edge candidates, and large teacher-displacement rows exactly as scoped",
                "",
            ]
        )

    if phase3:
        control = phase3["control"]
        val_improvements = {
            "broad": medium_exact["broad_plastic_mae"] - control["val_broad_plastic_mae"],
            "hard": medium_exact["hard_plastic_mae"] - control["val_hard_plastic_mae"],
            "p95": medium_exact["hard_p95_principal"] - control["val_hard_p95_principal"],
        }
        lines.extend(
            [
                "### Phase B1: same-capacity preservation control",
                "",
                "Validation comparison:",
                "",
                "| Model | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Projected teacher | {float(teacher_gate['val']['broad_plastic_mae']):.6f} | {float(teacher_gate['val']['hard_plastic_mae']):.6f} | {float(teacher_gate['val']['hard_p95_principal']):.6f} | {float(teacher_gate['val']['yield_violation_p95']):.6e} |" if teacher_gate else "| Projected teacher | not_run | not_run | not_run | not_run |",
                f"| March 26 `medium_exact` | {float(medium_exact['broad_plastic_mae']):.6f} | {float(medium_exact['hard_plastic_mae']):.6f} | {float(medium_exact['hard_p95_principal']):.6f} | {float(medium_exact['yield_violation_p95']):.6e} |",
                f"| March 27 preservation control | {float(control['val_broad_plastic_mae']):.6f} | {float(control['val_hard_plastic_mae']):.6f} | {float(control['val_hard_p95_principal']):.6f} | {float(control['val_yield_violation_p95']):.6e} |",
                "",
                "Held-out test comparison:",
                "",
                "| Model | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Projected teacher | {float(teacher_gate['test']['broad_plastic_mae']):.6f} | {float(teacher_gate['test']['hard_plastic_mae']):.6f} | {float(teacher_gate['test']['hard_p95_principal']):.6f} | {float(teacher_gate['test']['yield_violation_p95']):.6e} |" if teacher_gate else "| Projected teacher | not_run | not_run | not_run | not_run |",
                f"| March 27 preservation control | {float(control['test_broad_plastic_mae']):.6f} | {float(control['test_hard_plastic_mae']):.6f} | {float(control['test_hard_p95_principal']):.6f} | {float(control['test_yield_violation_p95']):.6e} |",
                "",
                "Validation improvement versus March 26 `medium_exact`:",
                "",
                f"- broad plastic MAE improvement: `{val_improvements['broad']:.6f}`",
                f"- hard plastic MAE improvement: `{val_improvements['hard']:.6f}`",
                f"- hard p95 principal improvement: `{val_improvements['p95']:.6f}`",
                f"- materially beats `medium_exact` on validation: {_fmt_bool(phase3['beats_medium_exact_all'])}",
                "",
                "Tier 1 opening gate check:",
                "",
                "| Gate | Threshold | Control value | Pass |",
                "| --- | ---: | ---: | --- |",
                f"| broad plastic MAE | 18.000000 | {float(control['val_broad_plastic_mae']):.6f} | {_fmt_bool(float(control['val_broad_plastic_mae']) <= TIER1_GATE['broad_plastic_mae'])} |",
                f"| hard plastic MAE | 20.000000 | {float(control['val_hard_plastic_mae']):.6f} | {_fmt_bool(float(control['val_hard_plastic_mae']) <= TIER1_GATE['hard_plastic_mae'])} |",
                f"| hard p95 principal | 180.000000 | {float(control['val_hard_p95_principal']):.6f} | {_fmt_bool(float(control['val_hard_p95_principal']) <= TIER1_GATE['hard_p95_principal'])} |",
                f"| yield p95 | 1.000000e-06 | {float(control['val_yield_violation_p95']):.6e} | {_fmt_bool(float(control['val_yield_violation_p95']) <= TIER1_GATE['yield_violation_p95'])} |",
                "",
                "Interpretation:",
                "",
                "- this was the main success of the packet: a same-capacity preservation control materially closed the gap to the projected teacher relative to the March 26 student",
                "- the remaining blocker was hard-tail quality, not admissibility or average stress accuracy",
                f"- Tier 1 open decision: {_fmt_bool(phase3['opens_tier1'])}",
                "",
            ]
        )

    lines.extend(
        [
            "### Phase C: bounded compression decision",
            "",
            f"- Tier 1 status: `{phase4['status'] if phase4 else 'not_run'}`",
            f"- Tier 2 status: `{phase5['status'] if phase5 else 'not_run'}`",
        ]
    )
    if phase4:
        lines.append(f"- Tier 1 reason: `{phase4.get('reason', 'completed')}`")
    if phase5:
        lines.append(f"- Tier 2 reason: `{phase5.get('reason', 'completed')}`")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- compression did not open because the packet stop rule was triggered before the sweep, not because an opened compression sweep failed",
            "- in practical terms, Tier 1 looked dead for this packet because the preservation control was still too tail-heavy to justify spending budget on smaller models",
            "",
            "## Decision",
            "",
            f"- projected-teacher `DS` viability: `{phase1['status'] if phase1 else 'not_run'}`",
            f"- same-capacity preservation materially closed the gap: {_fmt_bool(phase3['beats_medium_exact_all'] if phase3 else False)}",
            f"- Tier 1 compression verdict: `{compression_verdict}`",
            f"- `DS` still blocked: {_fmt_bool(ds_still_blocked)}",
            f"- route decision: `{route_decision}`",
            "",
            "Decision reasoning:",
            "",
            "- candidate zero remains strong enough that the route is worth preserving",
            "- the teacher `DS` probe did not justify opening broader tangent work yet",
            "- the same-capacity preservation control clearly improved on the March 26 student, so the route is not dead",
            "- the remaining gap is concentrated in hard-tail principal-stress error, so the next packet should narrow onto that preservation problem before reopening compression",
            "",
            "Bottom line:",
            "",
            "The projection-student line should continue only in narrowed preservation-first mode. `DS` remains blocked, and bounded compression should remain closed until the preserved student closes the remaining `hard_p95` gap to the projected teacher.",
            "",
            "## Artifact Map",
            "",
            "Primary packet artifacts:",
            "",
            f"- teacher cache summary: [`exp0_teacher_cache/teacher_projection_preservation_summary.json`]({output_root / 'exp0_teacher_cache/teacher_projection_preservation_summary.json'})",
            f"- teacher cache H5: [`exp0_teacher_cache/teacher_projection_preservation_cache.h5`]({output_root / 'exp0_teacher_cache/teacher_projection_preservation_cache.h5'})",
            f"- teacher `DS` probe summary: [`exp1_teacher_ds_probe/teacher_ds_probe_summary.json`]({output_root / 'exp1_teacher_ds_probe/teacher_ds_probe_summary.json'})",
            f"- teacher `DS` probe CSV: [`exp1_teacher_ds_probe/teacher_ds_probe_summary.csv`]({output_root / 'exp1_teacher_ds_probe/teacher_ds_probe_summary.csv'})",
            f"- teacher `DS` rowwise val H5: [`exp1_teacher_ds_probe/teacher_ds_probe_val.h5`]({output_root / 'exp1_teacher_ds_probe/teacher_ds_probe_val.h5'})",
            f"- teacher `DS` rowwise test H5: [`exp1_teacher_ds_probe/teacher_ds_probe_test.h5`]({output_root / 'exp1_teacher_ds_probe/teacher_ds_probe_test.h5'})",
            f"- preservation dataset summary: [`exp2_preservation_dataset/preservation_dataset_summary.json`]({output_root / 'exp2_preservation_dataset/preservation_dataset_summary.json'})",
            f"- preservation dataset H5: [`exp2_preservation_dataset/projection_student_preservation_plastic.h5`]({output_root / 'exp2_preservation_dataset/projection_student_preservation_plastic.h5'})",
            f"- preservation control summary: [`exp3_preservation_control/phase3_summary.json`]({output_root / 'exp3_preservation_control/phase3_summary.json'})",
            f"- preservation control checkpoint: [`exp3_preservation_control/control_exact_w256_d4/best.pt`]({output_root / 'exp3_preservation_control/control_exact_w256_d4/best.pt'})",
            f"- preservation control validation metrics: [`exp3_preservation_control/control_exact_w256_d4/projected_student_eval_val.json`]({output_root / 'exp3_preservation_control/control_exact_w256_d4/projected_student_eval_val.json'})",
            f"- preservation control test metrics: [`exp3_preservation_control/control_exact_w256_d4/projected_student_eval_test.json`]({output_root / 'exp3_preservation_control/control_exact_w256_d4/projected_student_eval_test.json'})",
            f"- Tier 1 compression status: [`exp4_tier1_compression/phase4_summary.json`]({output_root / 'exp4_tier1_compression/phase4_summary.json'})",
            f"- Tier 2 compression status: [`exp5_tier2_compression/phase5_summary.json`]({output_root / 'exp5_tier2_compression/phase5_summary.json'})",
            f"- FE shadow status: [`fe_shadow_status.json`]({output_root / 'fe_shadow_status.json'})",
            "",
        ]
    )

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    docs_root = ROOT / args.docs_root
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    if args.phase in {"all", "exp0_teacher_cache"}:
        run_phase0_teacher_cache(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp1_teacher_ds_probe"}:
        run_phase1_teacher_ds_probe(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp2_preservation_dataset"}:
        run_phase2_preservation_dataset(
            output_root=output_root,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp3_preservation_control"}:
        run_phase3_preservation_control(
            output_root=output_root,
            docs_root=docs_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp4_tier1_compression"}:
        run_phase4_tier1_compression(
            output_root=output_root,
            docs_root=docs_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp5_tier2_compression"}:
        run_phase5_tier2_compression(
            output_root=output_root,
            docs_root=docs_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    write_execution_report(output_root=output_root, docs_root=docs_root)


if __name__ == "__main__":
    main()
