#!/usr/bin/env python
"""Run the March 27 projection-student hard-tail closure packet."""

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
from mc_surrogate.principal_projection import PROJECTION_CANDIDATE_NAMES, PROJECTION_CANDIDATE_TO_ID
from mc_surrogate.projection_student_preservation import (
    BOUNDARY_MASK_KEYS,
    build_any_boundary_mask,
    build_call_concentration_rows,
    build_control_zero_rowwise_arrays,
    build_sampling_weights,
    build_slice_summary_rows,
    build_top_fraction_mask,
    displacement_decile_edges,
    principal_abs_error_arrays,
    quantile_or_zero,
    summarize_array,
)
from mc_surrogate.training import TrainingConfig, predict_with_checkpoint
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
GROUPED_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5"
PANEL_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5"
PREV_PACKET_ROOT = ROOT / "experiment_runs/real_sim/projection_student_preservation_compression_20260327"
TEACHER_CACHE_PATH = PREV_PACKET_ROOT / "exp0_teacher_cache/teacher_projection_preservation_cache.h5"
OLD_PRESERVATION_DATASET = PREV_PACKET_ROOT / "exp2_preservation_dataset/projection_student_preservation_plastic.h5"
CONTROL_ZERO_CHECKPOINT = PREV_PACKET_ROOT / "exp3_preservation_control/control_exact_w256_d4/best.pt"
CONTROL_ZERO_VAL_METRICS = PREV_PACKET_ROOT / "exp3_preservation_control/control_exact_w256_d4/projected_student_eval_val.json"
CONTROL_ZERO_TEST_METRICS = PREV_PACKET_ROOT / "exp3_preservation_control/control_exact_w256_d4/projected_student_eval_test.json"
CANDIDATE_B_WARMSTART = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/candidate_b/finetune/best.pt"
WORK_PACKET_BASENAME = "projection_student_hard_tail_closure_work_packet_20260327.md"
TIER1_OPEN_GATE = {
    "broad_plastic_mae": 18.0,
    "hard_plastic_mae": 20.0,
    "hard_p95_principal": 180.0,
    "yield_violation_p95": 1.0e-6,
}
COMPRESSION_REOPEN_BAR = {
    "broad_plastic_mae": 15.0,
    "hard_plastic_mae": 18.0,
    "hard_p95_principal": 160.0,
    "yield_violation_p95": 1.0e-6,
}
DS_REOPEN_BAR = {
    "broad_plastic_mae": 10.0,
    "hard_plastic_mae": 12.0,
    "hard_p95_principal": 120.0,
    "yield_violation_p95": 1.0e-6,
}
SAME_CAPACITY_VARIANTS = ("weighted_focus", "tail_loss_focus", "combined_focus", "dominant_slice_focus")
NEAR_SAME_CAPACITY_ARCHS = (
    {"name": "w288_d4", "width": 288, "depth": 4},
    {"name": "w256_d5", "width": 256, "depth": 5},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="all",
        choices=("all", "exp0_control_zero_slices", "exp1_tail_closure_controls", "exp2_near_same_capacity_followon"),
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/projection_student_hard_tail_closure_20260327",
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


def _build_generic_rowwise_arrays(
    *,
    prefix: str,
    split_arrays: dict[str, np.ndarray],
    predicted_principal: np.ndarray,
    provisional_principal: np.ndarray | None,
    predicted_branch_id: np.ndarray | None,
    disp_decile_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    abs_error, max_abs_error = principal_abs_error_arrays(predicted_principal, split_arrays["exact_stress_principal"])
    disp_decile = np.digitize(
        np.where(np.isfinite(split_arrays["teacher_projection_disp_norm"]), split_arrays["teacher_projection_disp_norm"], 0.0),
        disp_decile_edges[1:-1],
        right=True,
    ).astype(np.int8)
    any_boundary = build_any_boundary_mask(split_arrays)
    rowwise = {
        "split_id": np.asarray(split_arrays["split_id"], dtype=np.int8),
        "source_call_id": np.asarray(split_arrays["source_call_id"], dtype=np.int32),
        "source_row_in_call": np.asarray(split_arrays["source_row_in_call"], dtype=np.int32),
        "branch_id": np.asarray(split_arrays["branch_id"], dtype=np.int8),
        "hard_mask": np.asarray(split_arrays["hard_mask"], dtype=np.int8),
        "plastic_mask": np.asarray(split_arrays["plastic_mask"], dtype=np.int8),
        "ds_valid_mask": np.asarray(split_arrays["ds_valid_mask"], dtype=np.int8),
        "teacher_projection_candidate_id": np.asarray(split_arrays["teacher_projection_candidate_id"], dtype=np.int8),
        "teacher_projection_disp_norm": np.asarray(split_arrays["teacher_projection_disp_norm"], dtype=np.float32),
        "teacher_projection_disp_decile": disp_decile.astype(np.int8),
        "any_boundary_mask": any_boundary.astype(np.int8),
        "exact_stress_principal": np.asarray(split_arrays["exact_stress_principal"], dtype=np.float32),
        "teacher_projected_stress_principal": np.asarray(split_arrays["teacher_projected_stress_principal"], dtype=np.float32),
        f"{prefix}_projected_stress_principal": np.asarray(predicted_principal, dtype=np.float32),
        f"{prefix}_principal_abs_error": abs_error.astype(np.float32),
        f"{prefix}_principal_max_abs_error": max_abs_error.astype(np.float32),
    }
    rowwise[f"{prefix}_provisional_stress_principal"] = (
        np.asarray(provisional_principal, dtype=np.float32)
        if provisional_principal is not None
        else np.full_like(predicted_principal, np.nan, dtype=np.float32)
    )
    rowwise[f"{prefix}_predicted_branch_id"] = (
        np.asarray(predicted_branch_id, dtype=np.int64)
        if predicted_branch_id is not None
        else np.full(predicted_principal.shape[0], -1, dtype=np.int64)
    )
    for key in BOUNDARY_MASK_KEYS:
        rowwise[key] = np.asarray(split_arrays.get(key, np.zeros(predicted_principal.shape[0], dtype=np.int8)), dtype=np.int8)
    return rowwise


def _find_group_row(rows: list[dict[str, Any]], group_name: str, group_id: int) -> dict[str, Any] | None:
    for row in rows:
        if row["group_name"] == group_name and int(row["group_id"]) == int(group_id):
            return row
    return None


def _summarize_rowwise(
    *,
    scope_name: str,
    rowwise_arrays: dict[str, np.ndarray],
    error_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plastic_mask = np.asarray(rowwise_arrays["plastic_mask"], dtype=bool)
    hard_mask = np.asarray(rowwise_arrays["hard_mask"], dtype=bool)
    base_mask = plastic_mask
    hard_pool = base_mask & hard_mask
    error = np.asarray(rowwise_arrays[error_key], dtype=np.float32)
    hard_top10, hard_top10_threshold = build_top_fraction_mask(error, pool_mask=hard_pool, fraction=0.10)
    hard_top5, hard_top5_threshold = build_top_fraction_mask(error, pool_mask=hard_pool, fraction=0.05)
    hard_top1, hard_top1_threshold = build_top_fraction_mask(error, pool_mask=hard_pool, fraction=0.01)
    top_masks = {
        "10": hard_top10,
        "5": hard_top5,
        "1": hard_top1,
    }

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(
        build_slice_summary_rows(
            scope_name=scope_name,
            rowwise_arrays=rowwise_arrays,
            group_name="true_branch",
            group_values=np.asarray(rowwise_arrays["branch_id"], dtype=np.int64),
            labels={idx: name for idx, name in enumerate(BRANCH_NAMES)},
            base_mask=base_mask,
            hard_top_fraction_masks=top_masks,
            error_key=error_key,
        )
    )
    summary_rows.extend(
        build_slice_summary_rows(
            scope_name=scope_name,
            rowwise_arrays=rowwise_arrays,
            group_name="teacher_projection_candidate",
            group_values=np.asarray(rowwise_arrays["teacher_projection_candidate_id"], dtype=np.int64),
            labels={idx: name for idx, name in enumerate(PROJECTION_CANDIDATE_NAMES)},
            base_mask=base_mask,
            hard_top_fraction_masks=top_masks,
            error_key=error_key,
        )
    )
    summary_rows.extend(
        build_slice_summary_rows(
            scope_name=scope_name,
            rowwise_arrays=rowwise_arrays,
            group_name="hard_panel",
            group_values=np.asarray(rowwise_arrays["hard_mask"], dtype=np.int64),
            labels={0: "non_hard", 1: "hard"},
            base_mask=base_mask,
            hard_top_fraction_masks=top_masks,
            error_key=error_key,
        )
    )
    for key in (*BOUNDARY_MASK_KEYS, "any_boundary_mask"):
        summary_rows.extend(
            build_slice_summary_rows(
                scope_name=scope_name,
                rowwise_arrays=rowwise_arrays,
                group_name=key,
                group_values=np.asarray(rowwise_arrays[key], dtype=np.int64),
                labels={0: "false", 1: "true"},
                base_mask=base_mask,
                hard_top_fraction_masks=top_masks,
                error_key=error_key,
            )
        )
    summary_rows.extend(
        build_slice_summary_rows(
            scope_name=scope_name,
            rowwise_arrays=rowwise_arrays,
            group_name="teacher_projection_disp_decile",
            group_values=np.asarray(rowwise_arrays["teacher_projection_disp_decile"], dtype=np.int64),
            labels={idx: f"decile_{idx + 1}" for idx in range(10)},
            base_mask=base_mask,
            hard_top_fraction_masks=top_masks,
            error_key=error_key,
        )
    )

    call_rows = build_call_concentration_rows(
        scope_name=scope_name,
        source_call_id=np.asarray(rowwise_arrays["source_call_id"], dtype=np.int32),
        base_mask=base_mask,
        hard_mask=hard_mask,
        principal_max_abs_error=error,
        hard_top_mask=hard_top5,
        top_n=20,
    )

    hard_panel_mean = float(np.mean(error[hard_pool])) if np.any(hard_pool) else 0.0
    boundary_true = _find_group_row(summary_rows, "any_boundary_mask", 1) or {}
    disp_top_decile = _find_group_row(summary_rows, "teacher_projection_disp_decile", 9) or {}
    best_candidate = None
    best_candidate_over = -np.inf
    for idx in range(len(PROJECTION_CANDIDATE_NAMES)):
        row = _find_group_row(summary_rows, "teacher_projection_candidate", idx)
        if row is None:
            continue
        if float(row["hard_top5_share"]) >= 0.45 and float(row["hard_top5_over_index"]) >= 1.75 and float(row["hard_top5_over_index"]) > best_candidate_over:
            best_candidate = row
            best_candidate_over = float(row["hard_top5_over_index"])
    best_branch = None
    best_branch_over = -np.inf
    for idx in range(1, len(BRANCH_NAMES)):
        row = _find_group_row(summary_rows, "true_branch", idx)
        if row is None:
            continue
        if float(row["hard_top5_share"]) >= 0.45 and float(row["hard_top5_over_index"]) >= 1.75 and float(row["hard_top5_over_index"]) > best_branch_over:
            best_branch = row
            best_branch_over = float(row["hard_top5_over_index"])

    dominant_focus: dict[str, Any] | None = None
    if best_candidate is not None or best_branch is not None:
        if best_candidate is None:
            dominant_focus = {
                "kind": "true_branch",
                "id": int(best_branch["group_id"]),
                "label": str(best_branch["group_value"]),
                "hard_top5_share": float(best_branch["hard_top5_share"]),
                "hard_top5_over_index": float(best_branch["hard_top5_over_index"]),
            }
        elif best_branch is None:
            dominant_focus = {
                "kind": "teacher_projection_candidate",
                "id": int(best_candidate["group_id"]),
                "label": str(best_candidate["group_value"]),
                "hard_top5_share": float(best_candidate["hard_top5_share"]),
                "hard_top5_over_index": float(best_candidate["hard_top5_over_index"]),
            }
        elif float(best_candidate["hard_top5_over_index"]) >= float(best_branch["hard_top5_over_index"]):
            dominant_focus = {
                "kind": "teacher_projection_candidate",
                "id": int(best_candidate["group_id"]),
                "label": str(best_candidate["group_value"]),
                "hard_top5_share": float(best_candidate["hard_top5_share"]),
                "hard_top5_over_index": float(best_candidate["hard_top5_over_index"]),
            }
        else:
            dominant_focus = {
                "kind": "true_branch",
                "id": int(best_branch["group_id"]),
                "label": str(best_branch["group_value"]),
                "hard_top5_share": float(best_branch["hard_top5_share"]),
                "hard_top5_over_index": float(best_branch["hard_top5_over_index"]),
            }

    focus_decision = {
        "any_boundary": bool(
            float(boundary_true.get("hard_top5_share", 0.0)) >= 0.50
            or float(boundary_true.get("hard_top5_over_index", 0.0)) >= 2.0
        ),
        "high_disp": bool(
            float(disp_top_decile.get("hard_top5_share", 0.0)) >= 0.40
            and float(disp_top_decile.get("mean_principal_max_abs_error", 0.0)) >= 1.5 * max(hard_panel_mean, 1.0e-12)
        ),
        "dominant_focus": dominant_focus,
    }

    summary = {
        "scope": scope_name,
        "n_rows": int(np.sum(base_mask)),
        "n_hard_rows": int(np.sum(hard_pool)),
        "hard_panel_mean_principal_max_abs_error": hard_panel_mean,
        "hard_panel_p95_principal_max_abs_error": quantile_or_zero(error[hard_pool], 0.95),
        "hard_panel_p99_principal_max_abs_error": quantile_or_zero(error[hard_pool], 0.99),
        "hard_top10_threshold": float(hard_top10_threshold) if np.isfinite(hard_top10_threshold) else None,
        "hard_top5_threshold": float(hard_top5_threshold) if np.isfinite(hard_top5_threshold) else None,
        "hard_top1_threshold": float(hard_top1_threshold) if np.isfinite(hard_top1_threshold) else None,
        "hard_top10_count": int(np.sum(hard_top10)),
        "hard_top5_count": int(np.sum(hard_top5)),
        "hard_top1_count": int(np.sum(hard_top1)),
        "focus_decision": focus_decision,
        "call_concentration_top5_total_rows": int(np.sum(hard_top5)),
    }
    return summary, summary_rows, call_rows


def _write_slice_bundle(
    *,
    phase_dir: Path,
    stem: str,
    rowwise_arrays: dict[str, np.ndarray] | None,
    rowwise_attrs: dict[str, Any] | None,
    summary: dict[str, Any],
    slice_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> None:
    if rowwise_arrays is not None:
        _write_h5(phase_dir / f"{stem}_rowwise.h5", rowwise_arrays, rowwise_attrs or {})
    _write_json(phase_dir / f"{stem}_summary.json", summary)
    _write_csv(
        phase_dir / f"{stem}_slice_summary.csv",
        slice_rows,
        [
            "scope",
            "group_name",
            "group_value",
            "group_id",
            "n_rows",
            "n_hard_rows",
            "hard_base_share",
            "mean_principal_max_abs_error",
            "p95_principal_max_abs_error",
            "p99_principal_max_abs_error",
            "hard_top10_count",
            "hard_top10_share",
            "hard_top10_over_index",
            "hard_top5_count",
            "hard_top5_share",
            "hard_top5_over_index",
            "hard_top1_count",
            "hard_top1_share",
            "hard_top1_over_index",
        ],
    )
    _write_csv(
        phase_dir / f"{stem}_call_concentration.csv",
        call_rows,
        [
            "scope",
            "source_call_id",
            "n_rows",
            "n_hard_rows",
            "hard_top5_count",
            "hard_top5_share",
            "mean_principal_max_abs_error",
            "p95_principal_max_abs_error",
            "p99_principal_max_abs_error",
        ],
    )


def _control_reference_metrics() -> dict[str, Any]:
    return {
        "val": _read_json(CONTROL_ZERO_VAL_METRICS),
        "test": _read_json(CONTROL_ZERO_TEST_METRICS),
    }


def _materially_better(row: dict[str, Any], control_val: dict[str, float]) -> bool:
    return (
        float(control_val["hard_p95_principal"]) - float(row["val_hard_p95_principal"]) >= 5.0
        and float(row["val_broad_plastic_mae"]) <= float(control_val["broad_plastic_mae"]) + 0.75
        and float(row["val_hard_plastic_mae"]) <= float(control_val["hard_plastic_mae"]) + 1.0
        and float(row["val_yield_violation_p95"]) <= 1.0e-6
    )


def _same_capacity_config(
    *,
    dataset_path: Path,
    run_dir: Path,
    width: int,
    depth: int,
    seed: int,
    init_checkpoint: str | None,
    tail_config: dict[str, Any],
    lr: float,
    epochs: int,
    patience: int,
) -> TrainingConfig:
    return TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_geom_projected_student",
        epochs=epochs,
        batch_size=4096,
        lr=lr,
        weight_decay=1.0e-5,
        width=width,
        depth=depth,
        dropout=0.0,
        seed=seed,
        patience=patience,
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
        projection_tau=0.05,
        init_checkpoint=init_checkpoint,
        projected_student_hard_loss_multiplier=float(tail_config.get("hard_loss_multiplier", 1.0)),
        projected_student_any_boundary_loss_multiplier=float(tail_config.get("any_boundary_loss_multiplier", 1.0)),
        projected_student_high_disp_loss_multiplier=float(tail_config.get("high_disp_loss_multiplier", 1.0)),
        projected_student_candidate_loss_weights=tail_config.get("candidate_loss_weights"),
        projected_student_branch_loss_weights=tail_config.get("branch_loss_weights"),
        projected_student_teacher_alignment_focus_multiplier=float(tail_config.get("teacher_alignment_focus_multiplier", 1.0)),
        projected_student_hard_quantile=float(tail_config.get("hard_quantile", 0.0)),
        projected_student_hard_quantile_weight=float(tail_config.get("hard_quantile_weight", 0.0)),
        projected_student_high_disp_threshold=(
            float(tail_config["high_disp_threshold"]) if tail_config.get("high_disp_threshold") is not None else None
        ),
    )


def _evaluate_checkpoint_with_rowwise(
    *,
    checkpoint_path: Path,
    split_name: str,
    arrays_all: dict[str, np.ndarray],
    panel_all: dict[str, np.ndarray],
    cache_arrays: dict[str, np.ndarray],
    disp_edges: np.ndarray,
    device: str,
    eval_batch_size: int,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mask = _split_mask(cache_arrays["split_id"], split_name)
    arrays = _slice_arrays(arrays_all, mask)
    panel = _slice_arrays(panel_all, mask)
    split_cache = _slice_arrays(cache_arrays, mask)
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=eval_batch_size,
    )
    eval_result = _evaluate_policy(
        arrays,
        panel,
        stress_pred=pred["stress"].astype(np.float32),
        stress_principal_pred=pred["stress_principal"].astype(np.float32),
    )
    rowwise = _build_generic_rowwise_arrays(
        prefix=prefix,
        split_arrays=split_cache,
        predicted_principal=pred["stress_principal"].astype(np.float32),
        provisional_principal=pred.get("provisional_stress_principal"),
        predicted_branch_id=pred.get("predicted_branch_id"),
        disp_decile_edges=disp_edges,
    )
    summary, slice_rows, call_rows = _summarize_rowwise(
        scope_name=split_name,
        rowwise_arrays=rowwise,
        error_key=f"{prefix}_principal_max_abs_error",
    )
    return eval_result["metrics"], rowwise, summary, slice_rows, call_rows


def run_phase0_control_zero_slices(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp0_control_zero_slices"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase0_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    if not TEACHER_CACHE_PATH.exists():
        raise FileNotFoundError(f"Expected March 27 teacher cache at {TEACHER_CACHE_PATH}")
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, cache_attrs = _load_h5(TEACHER_CACHE_PATH)

    split_payloads: dict[str, Any] = {}
    focus_source = None
    for split_name in ("val", "test"):
        mask = _split_mask(cache_arrays["split_id"], split_name)
        split_cache = _slice_arrays(cache_arrays, mask)
        disp_edges = displacement_decile_edges(
            split_cache["teacher_projection_disp_norm"],
            reference_mask=np.asarray(split_cache["plastic_mask"], dtype=bool),
            n_bins=10,
        )
        split_arrays = _slice_arrays(arrays_all, mask)
        panel = _slice_arrays(panel_all, mask)
        pred = predict_with_checkpoint(
            CONTROL_ZERO_CHECKPOINT,
            split_arrays["strain_eng"],
            split_arrays["material_reduced"],
            device=device,
            batch_size=eval_batch_size,
        )
        metrics = _evaluate_policy(
            split_arrays,
            panel,
            stress_pred=pred["stress"].astype(np.float32),
            stress_principal_pred=pred["stress_principal"].astype(np.float32),
        )["metrics"]
        rowwise, rowwise_meta = build_control_zero_rowwise_arrays(
            split_cache,
            pred,
            disp_decile_edges=disp_edges,
        )
        slice_summary, slice_rows, call_rows = _summarize_rowwise(
            scope_name=split_name,
            rowwise_arrays=rowwise,
            error_key="control_zero_principal_max_abs_error",
        )
        rowwise_meta.update(
            {
                "teacher_cache_path": str(TEACHER_CACHE_PATH),
                "control_zero_checkpoint": str(CONTROL_ZERO_CHECKPOINT),
                "projection_candidate_names_json": list(PROJECTION_CANDIDATE_NAMES),
            }
        )
        _write_slice_bundle(
            phase_dir=phase_dir,
            stem=f"control_zero_{split_name}",
            rowwise_arrays=rowwise,
            rowwise_attrs=rowwise_meta,
            summary={**slice_summary, "metrics": metrics},
            slice_rows=slice_rows,
            call_rows=call_rows,
        )
        split_payloads[split_name] = {
            "metrics": metrics,
            "slice_summary": slice_summary,
            "disp_decile_edges": [float(x) for x in disp_edges],
        }
        if split_name == "val":
            focus_source = slice_summary["focus_decision"]

    summary = {
        "teacher_cache_path": str(TEACHER_CACHE_PATH),
        "control_zero_checkpoint": str(CONTROL_ZERO_CHECKPOINT),
        "split_payloads": split_payloads,
        "focus_source_split": "val",
        "focus_decision": focus_source,
        "teacher_cache_attrs": cache_attrs,
    }
    _write_json(summary_path, summary)
    return summary


def _build_base_preservation_dataset(
    *,
    phase_dir: Path,
    force_rerun: bool,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = phase_dir / "base_preservation_dataset.h5"
    summary_path = phase_dir / "base_preservation_dataset_summary.json"
    if dataset_path.exists() and summary_path.exists() and not force_rerun:
        return dataset_path, _read_json(summary_path)

    if not TEACHER_CACHE_PATH.exists():
        raise FileNotFoundError(f"Expected March 27 teacher cache at {TEACHER_CACHE_PATH}")
    cache_arrays, cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    arrays_all, dataset_attrs = _load_h5(GROUPED_DATASET)
    _panel_arrays, panel_attrs = _load_h5(PANEL_DATASET)

    plastic_mask = cache_arrays["plastic_mask"].astype(bool)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(arrays_all["strain_eng"])
    geom_features = build_trial_principal_geom_features(
        strain_principal,
        arrays_all["material_reduced"],
        cache_arrays["trial_principal"],
    ).astype(np.float32)
    any_boundary_mask = build_any_boundary_mask(cache_arrays)
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
        "any_boundary_mask": any_boundary_mask[plastic_mask].astype(np.int8),
    }
    for key in BOUNDARY_MASK_KEYS:
        derived_arrays[key] = cache_arrays[key][plastic_mask].astype(np.int8)

    _write_h5(
        dataset_path,
        derived_arrays,
        {
            "source_teacher_cache": str(TEACHER_CACHE_PATH),
            "source_grouped_dataset": str(GROUPED_DATASET),
            "source_panel_dataset": str(PANEL_DATASET),
            "high_disp_focus_threshold": disp_threshold,
            "teacher_projection_disp_p90_train": disp_threshold,
            "sampling_weight_rule": {
                "base": 1.0,
                "hard_mask": 1.5,
                "edge_candidate": 1.5,
                "high_disp_train_p90": 1.5,
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
        "any_boundary_counts": {
            name: int(np.sum((split_id == idx) & (derived_arrays["any_boundary_mask"] > 0)))
            for name, idx in SPLIT_NAME_TO_ID.items()
        },
        "sampling_weight_summary": summarize_array(derived_arrays["sampling_weight"]),
        "teacher_projection_disp_norm_summary": summarize_array(derived_arrays["teacher_projection_disp_norm"]),
        "high_disp_focus_threshold": float(disp_threshold),
    }
    _write_json(summary_path, summary)
    return dataset_path, summary


def _load_dataset_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return _load_h5(path)


def _variant_dataset(
    *,
    base_dataset_path: Path,
    variant_name: str,
    phase_dir: Path,
    focus_decision: dict[str, Any],
    force_rerun: bool,
) -> Path:
    if variant_name not in {"weighted_focus", "combined_focus", "dominant_slice_focus"}:
        return base_dataset_path
    variant_dir = phase_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = variant_dir / "dataset.h5"
    if dataset_path.exists() and not force_rerun:
        return dataset_path

    arrays, attrs = _load_dataset_arrays(base_dataset_path)
    dominant = focus_decision.get("dominant_focus")
    candidate_map = None
    branch_map = None
    dominant_multiplier = 2.0 if variant_name == "dominant_slice_focus" else 1.75
    if dominant is not None:
        if dominant["kind"] == "teacher_projection_candidate":
            candidate_map = {int(dominant["id"]): dominant_multiplier}
        else:
            branch_map = {int(dominant["id"]): dominant_multiplier}
    weights, _threshold = build_sampling_weights(
        train_mask=arrays["split_id"] == SPLIT_NAME_TO_ID["train"],
        hard_mask=arrays["hard_mask"] > 0,
        teacher_projection_candidate_id=arrays["teacher_projection_candidate_id"],
        teacher_projection_disp_norm=arrays["teacher_projection_disp_norm"],
        any_boundary_mask=arrays["any_boundary_mask"] > 0,
        branch_id=arrays["branch_id"],
        hard_multiplier=2.0,
        boundary_multiplier=1.5 if focus_decision.get("any_boundary", False) else 1.0,
        high_disp_multiplier=1.5 if focus_decision.get("high_disp", False) else 1.0,
        candidate_weight_map=candidate_map,
        branch_weight_map=branch_map,
        high_disp_threshold=float(attrs.get("high_disp_focus_threshold", attrs.get("teacher_projection_disp_p90_train", 0.0))),
    )
    arrays["sampling_weight"] = weights.astype(np.float32)
    attrs["variant_sampling_rule"] = {
        "hard_multiplier": 2.0,
        "any_boundary_multiplier": 1.5 if focus_decision.get("any_boundary", False) else 1.0,
        "high_disp_multiplier": 1.5 if focus_decision.get("high_disp", False) else 1.0,
        "candidate_weight_map": candidate_map,
        "branch_weight_map": branch_map,
    }
    _write_h5(dataset_path, arrays, attrs)
    return dataset_path


def _tail_config_for_variant(
    *,
    variant_name: str,
    focus_decision: dict[str, Any],
    high_disp_threshold: float | None,
) -> dict[str, Any]:
    dominant = focus_decision.get("dominant_focus")
    candidate_loss_weights = None
    branch_loss_weights = None
    dominant_loss_multiplier = 1.75 if variant_name == "dominant_slice_focus" else 1.5
    if dominant is not None:
        if dominant["kind"] == "teacher_projection_candidate":
            candidate_loss_weights = {int(dominant["id"]): dominant_loss_multiplier}
        else:
            branch_loss_weights = {int(dominant["id"]): dominant_loss_multiplier}
    if variant_name == "weighted_focus":
        return {
            "high_disp_threshold": high_disp_threshold,
        }
    if variant_name == "tail_loss_focus":
        return {
            "any_boundary_loss_multiplier": 1.5 if focus_decision.get("any_boundary", False) else 1.0,
            "high_disp_loss_multiplier": 1.5 if focus_decision.get("high_disp", False) else 1.0,
            "candidate_loss_weights": candidate_loss_weights,
            "branch_loss_weights": branch_loss_weights,
            "teacher_alignment_focus_multiplier": 1.5,
            "hard_quantile": 0.90,
            "hard_quantile_weight": 0.30,
            "high_disp_threshold": high_disp_threshold,
        }
    if variant_name == "combined_focus":
        return {
            "any_boundary_loss_multiplier": 1.5 if focus_decision.get("any_boundary", False) else 1.0,
            "high_disp_loss_multiplier": 1.5 if focus_decision.get("high_disp", False) else 1.0,
            "candidate_loss_weights": candidate_loss_weights,
            "branch_loss_weights": branch_loss_weights,
            "teacher_alignment_focus_multiplier": 1.5,
            "hard_quantile": 0.90,
            "hard_quantile_weight": 0.30,
            "high_disp_threshold": high_disp_threshold,
        }
    if variant_name == "dominant_slice_focus":
        return {
            "any_boundary_loss_multiplier": 1.5 if focus_decision.get("any_boundary", False) else 1.0,
            "high_disp_loss_multiplier": 1.5 if focus_decision.get("high_disp", False) else 1.0,
            "candidate_loss_weights": candidate_loss_weights,
            "branch_loss_weights": branch_loss_weights,
            "teacher_alignment_focus_multiplier": 1.75,
            "hard_quantile": 0.90,
            "hard_quantile_weight": 0.30,
            "high_disp_threshold": high_disp_threshold,
        }
    raise ValueError(f"Unsupported variant {variant_name!r}")


def _train_variant(
    *,
    config: TrainingConfig,
    force_rerun: bool,
) -> dict[str, Any]:
    try:
        return _train_with_batch_fallback(config, force_rerun=force_rerun)
    except RuntimeError:
        if config.init_checkpoint == str(CANDIDATE_B_WARMSTART):
            raise
        fallback = TrainingConfig(**{**config.__dict__, "init_checkpoint": str(CANDIDATE_B_WARMSTART)})
        return _train_with_batch_fallback(fallback, force_rerun=force_rerun)


def run_phase1_tail_closure_controls(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp1_tail_closure_controls"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase1_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase0 = _read_json(output_root / "exp0_control_zero_slices" / "phase0_summary.json")
    focus_decision = phase0["focus_decision"]
    base_dataset_path, base_dataset_summary = _build_base_preservation_dataset(phase_dir=phase_dir, force_rerun=force_rerun)
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, _cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    control_val = _control_reference_metrics()["val"]
    val_disp_edges = np.asarray(phase0["split_payloads"]["val"]["disp_decile_edges"], dtype=float)
    test_disp_edges = np.asarray(phase0["split_payloads"]["test"]["disp_decile_edges"], dtype=float)
    high_disp_threshold = float(base_dataset_summary["high_disp_focus_threshold"])

    rows: list[dict[str, Any]] = [
        {
            "variant": "control_zero_baseline",
            "width": 256,
            "depth": 4,
            "param_count": _parameter_count(CONTROL_ZERO_CHECKPOINT, device=device),
            "run_dir": str(CONTROL_ZERO_CHECKPOINT.parent),
            "best_checkpoint": str(CONTROL_ZERO_CHECKPOINT),
            "completed_epochs": None,
            "best_epoch": None,
            "val_broad_plastic_mae": float(control_val["broad_plastic_mae"]),
            "val_hard_plastic_mae": float(control_val["hard_plastic_mae"]),
            "val_hard_p95_principal": float(control_val["hard_p95_principal"]),
            "val_yield_violation_p95": float(control_val["yield_violation_p95"]),
            "test_broad_plastic_mae": float(_control_reference_metrics()["test"]["broad_plastic_mae"]),
            "test_hard_plastic_mae": float(_control_reference_metrics()["test"]["hard_plastic_mae"]),
            "test_hard_p95_principal": float(_control_reference_metrics()["test"]["hard_p95_principal"]),
            "test_yield_violation_p95": float(_control_reference_metrics()["test"]["yield_violation_p95"]),
            "materially_better_than_control": False,
            "tail_config": None,
            "dataset_path": str(base_dataset_path),
        }
    ]

    for idx, variant_name in enumerate(SAME_CAPACITY_VARIANTS):
        if variant_name == "dominant_slice_focus" and focus_decision.get("dominant_focus") is None:
            continue
        run_dir = phase_dir / variant_name
        dataset_path = _variant_dataset(
            base_dataset_path=base_dataset_path,
            variant_name=variant_name,
            phase_dir=phase_dir,
            focus_decision=focus_decision,
            force_rerun=force_rerun,
        )
        tail_config = _tail_config_for_variant(
            variant_name=variant_name,
            focus_decision=focus_decision,
            high_disp_threshold=high_disp_threshold,
        )
        config = _same_capacity_config(
            dataset_path=dataset_path,
            run_dir=run_dir,
            width=256,
            depth=4,
            seed=20260327 + idx,
            init_checkpoint=str(CONTROL_ZERO_CHECKPOINT),
            tail_config=tail_config,
            lr=1.0e-4,
            epochs=30,
            patience=8,
        )
        train_summary = _train_variant(config=config, force_rerun=force_rerun)
        best_checkpoint = Path(train_summary["best_checkpoint"])
        val_metrics, _val_rowwise, val_summary, val_slice_rows, val_call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=best_checkpoint,
            split_name="val",
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=val_disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix=variant_name,
        )
        test_metrics, _test_rowwise, test_summary, test_slice_rows, test_call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=best_checkpoint,
            split_name="test",
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=test_disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix=variant_name,
        )
        _write_json(run_dir / "eval_val.json", val_metrics)
        _write_json(run_dir / "eval_test.json", test_metrics)
        _write_slice_bundle(
            phase_dir=run_dir,
            stem="val",
            rowwise_arrays=None,
            rowwise_attrs=None,
            summary=val_summary,
            slice_rows=val_slice_rows,
            call_rows=val_call_rows,
        )
        _write_slice_bundle(
            phase_dir=run_dir,
            stem="test",
            rowwise_arrays=None,
            rowwise_attrs=None,
            summary=test_summary,
            slice_rows=test_slice_rows,
            call_rows=test_call_rows,
        )
        row = {
            "variant": variant_name,
            "width": 256,
            "depth": 4,
            "param_count": _parameter_count(best_checkpoint, device=device),
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint),
            "completed_epochs": int(train_summary["completed_epochs"]),
            "best_epoch": int(train_summary["best_epoch"]),
            "val_broad_plastic_mae": float(val_metrics["broad_plastic_mae"]),
            "val_hard_plastic_mae": float(val_metrics["hard_plastic_mae"]),
            "val_hard_p95_principal": float(val_metrics["hard_p95_principal"]),
            "val_yield_violation_p95": float(val_metrics["yield_violation_p95"]),
            "test_broad_plastic_mae": float(test_metrics["broad_plastic_mae"]),
            "test_hard_plastic_mae": float(test_metrics["hard_plastic_mae"]),
            "test_hard_p95_principal": float(test_metrics["hard_p95_principal"]),
            "test_yield_violation_p95": float(test_metrics["yield_violation_p95"]),
            "materially_better_than_control": _materially_better(
                {
                    "val_broad_plastic_mae": val_metrics["broad_plastic_mae"],
                    "val_hard_plastic_mae": val_metrics["hard_plastic_mae"],
                    "val_hard_p95_principal": val_metrics["hard_p95_principal"],
                    "val_yield_violation_p95": val_metrics["yield_violation_p95"],
                },
                control_val,
            ),
            "tail_config": tail_config,
            "dataset_path": str(dataset_path),
        }
        rows.append(row)

    _write_csv(
        phase_dir / "same_capacity_summary.csv",
        rows,
        [
            "variant",
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
            "materially_better_than_control",
            "tail_config",
            "dataset_path",
        ],
    )

    candidate_rows = [row for row in rows if row["variant"] != "control_zero_baseline"]
    winner = min(candidate_rows, key=_ranking_key) if candidate_rows else None
    if winner is not None:
        best_val_metrics, best_val_rowwise, best_val_summary, best_val_slice_rows, best_val_call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=Path(winner["best_checkpoint"]),
            split_name="val",
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=val_disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix="best_same_capacity",
        )
        best_test_metrics, best_test_rowwise, best_test_summary, best_test_slice_rows, best_test_call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=Path(winner["best_checkpoint"]),
            split_name="test",
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=test_disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix="best_same_capacity",
        )
        _write_slice_bundle(
            phase_dir=phase_dir,
            stem="best_same_capacity_val",
            rowwise_arrays=best_val_rowwise,
            rowwise_attrs={
                "source_checkpoint": winner["best_checkpoint"],
                "disp_decile_edges": [float(x) for x in val_disp_edges],
            },
            summary={**best_val_summary, "metrics": best_val_metrics},
            slice_rows=best_val_slice_rows,
            call_rows=best_val_call_rows,
        )
        _write_slice_bundle(
            phase_dir=phase_dir,
            stem="best_same_capacity_test",
            rowwise_arrays=best_test_rowwise,
            rowwise_attrs={
                "source_checkpoint": winner["best_checkpoint"],
                "disp_decile_edges": [float(x) for x in test_disp_edges],
            },
            summary={**best_test_summary, "metrics": best_test_metrics},
            slice_rows=best_test_slice_rows,
            call_rows=best_test_call_rows,
        )
        winner["best_same_capacity_val_summary"] = best_val_summary
        winner["best_same_capacity_test_summary"] = best_test_summary

    winner_gate = bool(
        winner is not None
        and (
            _meets_bar(
                {
                    "broad_plastic_mae": float(winner["val_broad_plastic_mae"]),
                    "hard_plastic_mae": float(winner["val_hard_plastic_mae"]),
                    "hard_p95_principal": float(winner["val_hard_p95_principal"]),
                    "yield_violation_p95": float(winner["val_yield_violation_p95"]),
                },
                TIER1_OPEN_GATE,
            )
            or (
                float(winner["val_hard_p95_principal"]) <= 185.0
                and float(control_val["hard_p95_principal"]) - float(winner["test_hard_p95_principal"]) >= 5.0
                and float(winner["val_broad_plastic_mae"]) <= float(control_val["broad_plastic_mae"]) + 0.75
                and float(winner["val_hard_plastic_mae"]) <= float(control_val["hard_plastic_mae"]) + 1.0
                and float(winner["val_yield_violation_p95"]) <= 1.0e-6
            )
        )
    )
    summary = {
        "base_dataset_summary": base_dataset_summary,
        "focus_decision": focus_decision,
        "rows": rows,
        "winner": winner,
        "near_same_capacity_open": winner_gate,
        "control_zero_reference": _control_reference_metrics(),
    }
    _write_json(summary_path, summary)
    return summary


def run_phase2_near_same_capacity_followon(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp2_near_same_capacity_followon"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase2_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase1 = _read_json(output_root / "exp1_tail_closure_controls" / "phase1_summary.json")
    if not phase1.get("near_same_capacity_open", False):
        summary = {
            "status": "skipped",
            "reason": "Same-capacity sweep did not justify the near-same-capacity follow-on.",
            "phase1": phase1,
        }
        _write_json(summary_path, summary)
        return summary

    winner = phase1["winner"]
    focus_decision = phase1["focus_decision"]
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, _cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    val_disp_edges = np.asarray(_read_json(output_root / "exp0_control_zero_slices" / "phase0_summary.json")["split_payloads"]["val"]["disp_decile_edges"], dtype=float)
    test_disp_edges = np.asarray(_read_json(output_root / "exp0_control_zero_slices" / "phase0_summary.json")["split_payloads"]["test"]["disp_decile_edges"], dtype=float)
    high_disp_threshold = float(phase1["base_dataset_summary"]["high_disp_focus_threshold"])

    base_dataset_path = Path(phase1["base_dataset_summary"]["dataset_path"])
    rows: list[dict[str, Any]] = []
    for idx, arch in enumerate(NEAR_SAME_CAPACITY_ARCHS):
        run_dir = phase_dir / arch["name"]
        tail_config = _tail_config_for_variant(
            variant_name=str(winner["variant"]),
            focus_decision=focus_decision,
            high_disp_threshold=high_disp_threshold,
        )
        config = _same_capacity_config(
            dataset_path=base_dataset_path,
            run_dir=run_dir,
            width=int(arch["width"]),
            depth=int(arch["depth"]),
            seed=20260340 + idx,
            init_checkpoint=None,
            tail_config=tail_config,
            lr=3.0e-4,
            epochs=60,
            patience=12,
        )
        train_summary = _train_variant(config=config, force_rerun=force_rerun)
        best_checkpoint = Path(train_summary["best_checkpoint"])
        val_metrics, _val_rowwise, val_summary, val_slice_rows, val_call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=best_checkpoint,
            split_name="val",
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=val_disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix=arch["name"],
        )
        test_metrics, _test_rowwise, test_summary, test_slice_rows, test_call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=best_checkpoint,
            split_name="test",
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=test_disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix=arch["name"],
        )
        _write_json(run_dir / "eval_val.json", val_metrics)
        _write_json(run_dir / "eval_test.json", test_metrics)
        _write_slice_bundle(
            phase_dir=run_dir,
            stem="val",
            rowwise_arrays=None,
            rowwise_attrs=None,
            summary=val_summary,
            slice_rows=val_slice_rows,
            call_rows=val_call_rows,
        )
        _write_slice_bundle(
            phase_dir=run_dir,
            stem="test",
            rowwise_arrays=None,
            rowwise_attrs=None,
            summary=test_summary,
            slice_rows=test_slice_rows,
            call_rows=test_call_rows,
        )
        rows.append(
            {
                "variant": arch["name"],
                "width": int(arch["width"]),
                "depth": int(arch["depth"]),
                "param_count": _parameter_count(best_checkpoint, device=device),
                "run_dir": str(run_dir),
                "best_checkpoint": str(best_checkpoint),
                "completed_epochs": int(train_summary["completed_epochs"]),
                "best_epoch": int(train_summary["best_epoch"]),
                "val_broad_plastic_mae": float(val_metrics["broad_plastic_mae"]),
                "val_hard_plastic_mae": float(val_metrics["hard_plastic_mae"]),
                "val_hard_p95_principal": float(val_metrics["hard_p95_principal"]),
                "val_yield_violation_p95": float(val_metrics["yield_violation_p95"]),
                "test_broad_plastic_mae": float(test_metrics["broad_plastic_mae"]),
                "test_hard_plastic_mae": float(test_metrics["hard_plastic_mae"]),
                "test_hard_p95_principal": float(test_metrics["hard_p95_principal"]),
                "test_yield_violation_p95": float(test_metrics["yield_violation_p95"]),
            }
        )
    _write_csv(
        phase_dir / "near_same_capacity_summary.csv",
        rows,
        [
            "variant",
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
        "rows": rows,
        "winner": min(rows, key=_ranking_key),
        "source_same_capacity_winner": winner,
    }
    _write_json(summary_path, summary)
    return summary


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"`{float(metrics['broad_plastic_mae']):.6f}` / "
        f"`{float(metrics['hard_plastic_mae']):.6f}` / "
        f"`{float(metrics['hard_p95_principal']):.6f}` / "
        f"`{float(metrics['yield_violation_p95']):.6e}`"
    )


def _write_execution_report(*, output_root: Path, docs_root: Path) -> None:
    docs_path = docs_root / "executions" / WORK_PACKET_BASENAME
    phase0 = _read_json(output_root / "exp0_control_zero_slices" / "phase0_summary.json")
    phase1_path = output_root / "exp1_tail_closure_controls" / "phase1_summary.json"
    phase1 = _read_json(phase1_path) if phase1_path.exists() else None
    phase2_path = output_root / "exp2_near_same_capacity_followon" / "phase2_summary.json"
    phase2 = _read_json(phase2_path) if phase2_path.exists() else None
    control = _control_reference_metrics()
    winner = phase1.get("winner") if phase1 is not None else None
    near = phase2.get("winner") if phase2 and phase2.get("status") == "completed" else None
    overall_candidates = [row for row in [winner, near] if row is not None]
    final_anchor = min(overall_candidates, key=_ranking_key) if overall_candidates else None

    lines = [
        "# Projection-Student Hard-Tail Closure Work Packet",
        "",
        "Execution report for the March 27, 2026 hard-tail closure packet.",
        "",
        f"Paired task memo: [`docs/tasks/projection_student_hard_tail_closure_work_packet_20260327.md`]({ROOT / 'docs/tasks/projection_student_hard_tail_closure_work_packet_20260327.md'})",
        "",
        f"Repository: `{ROOT}`",
        f"Output root: `{output_root}`",
        "",
        "## Baseline",
        "",
        f"- frozen control-zero checkpoint: `{CONTROL_ZERO_CHECKPOINT}`",
        f"- frozen validation broad / hard / p95 / yield: {_fmt_metrics(control['val'])}",
        f"- frozen test broad / hard / p95 / yield: {_fmt_metrics(control['test'])}",
        f"- control-zero focus decision from validation: `{phase0['focus_decision']}`",
        "",
        "## Phase A",
        "",
        f"- validation control-zero slice summary: [`exp0_control_zero_slices/control_zero_val_summary.json`]({output_root / 'exp0_control_zero_slices/control_zero_val_summary.json'})",
        f"- test control-zero slice summary: [`exp0_control_zero_slices/control_zero_test_summary.json`]({output_root / 'exp0_control_zero_slices/control_zero_test_summary.json'})",
        f"- validation control-zero rowwise H5: [`exp0_control_zero_slices/control_zero_val_rowwise.h5`]({output_root / 'exp0_control_zero_slices/control_zero_val_rowwise.h5'})",
        f"- test control-zero rowwise H5: [`exp0_control_zero_slices/control_zero_test_rowwise.h5`]({output_root / 'exp0_control_zero_slices/control_zero_test_rowwise.h5'})",
        "",
        "Interpretation:",
        "",
        f"- boundary focus triggered: `{bool(phase0['focus_decision'].get('any_boundary', False))}`",
        f"- high-displacement focus triggered: `{bool(phase0['focus_decision'].get('high_disp', False))}`",
        f"- dominant slice focus: `{phase0['focus_decision'].get('dominant_focus')}`",
        "",
        "## Phase B/C",
        "",
        f"- rebuilt preservation dataset summary: [`exp1_tail_closure_controls/base_preservation_dataset_summary.json`]({output_root / 'exp1_tail_closure_controls/base_preservation_dataset_summary.json'})" if (output_root / "exp1_tail_closure_controls/base_preservation_dataset_summary.json").exists() else "- rebuilt preservation dataset summary: `not_run`",
        f"- same-capacity summary CSV: [`exp1_tail_closure_controls/same_capacity_summary.csv`]({output_root / 'exp1_tail_closure_controls/same_capacity_summary.csv'})" if (output_root / "exp1_tail_closure_controls/same_capacity_summary.csv").exists() else "- same-capacity summary CSV: `not_run`",
        f"- same-capacity summary JSON: [`exp1_tail_closure_controls/phase1_summary.json`]({output_root / 'exp1_tail_closure_controls/phase1_summary.json'})" if phase1 is not None else "- same-capacity summary JSON: `not_run`",
    ]
    if winner is not None:
        lines.extend(
            [
                "",
                f"- best same-capacity run: `{winner['variant']}`",
                f"- best same-capacity validation broad / hard / p95 / yield: `{winner['val_broad_plastic_mae']:.6f}` / `{winner['val_hard_plastic_mae']:.6f}` / `{winner['val_hard_p95_principal']:.6f}` / `{winner['val_yield_violation_p95']:.6e}`",
                f"- best same-capacity test broad / hard / p95 / yield: `{winner['test_broad_plastic_mae']:.6f}` / `{winner['test_hard_plastic_mae']:.6f}` / `{winner['test_hard_p95_principal']:.6f}` / `{winner['test_yield_violation_p95']:.6e}`",
                f"- materially better than control by packet rule: `{bool(winner.get('materially_better_than_control', False))}`",
                f"- same-capacity best rowwise val H5: [`exp1_tail_closure_controls/best_same_capacity_val_rowwise.h5`]({output_root / 'exp1_tail_closure_controls/best_same_capacity_val_rowwise.h5'})",
                f"- same-capacity best rowwise test H5: [`exp1_tail_closure_controls/best_same_capacity_test_rowwise.h5`]({output_root / 'exp1_tail_closure_controls/best_same_capacity_test_rowwise.h5'})",
            ]
        )
    lines.extend(
        [
            "",
            "## Phase D",
            "",
            f"- near-same-capacity status: `{phase2['status'] if phase2 else 'not_run'}`",
        ]
    )
    if phase2 is not None and phase2.get("status") == "completed" and near is not None:
        lines.extend(
            [
                f"- best near-same-capacity run: `{near['variant']}`",
                f"- best near-same-capacity validation broad / hard / p95 / yield: `{near['val_broad_plastic_mae']:.6f}` / `{near['val_hard_plastic_mae']:.6f}` / `{near['val_hard_p95_principal']:.6f}` / `{near['val_yield_violation_p95']:.6e}`",
            ]
        )
    elif phase2 is not None:
        lines.append(f"- near-same-capacity reason: `{phase2.get('reason', 'n/a')}`")

    if final_anchor is not None:
        lines.extend(
            [
                "",
                "Overall best packet anchor:",
                "",
                f"- overall best run: `{final_anchor['variant']}`",
                f"- overall best validation broad / hard / p95 / yield: `{final_anchor['val_broad_plastic_mae']:.6f}` / `{final_anchor['val_hard_plastic_mae']:.6f}` / `{final_anchor['val_hard_p95_principal']:.6f}` / `{final_anchor['val_yield_violation_p95']:.6e}`",
                f"- overall best test broad / hard / p95 / yield: `{final_anchor['test_broad_plastic_mae']:.6f}` / `{final_anchor['test_hard_plastic_mae']:.6f}` / `{final_anchor['test_hard_p95_principal']:.6f}` / `{final_anchor['test_yield_violation_p95']:.6e}`",
            ]
        )

    compression_reopen = final_anchor is not None and _meets_bar(
        {
            "broad_plastic_mae": float(final_anchor["val_broad_plastic_mae"]),
            "hard_plastic_mae": float(final_anchor["val_hard_plastic_mae"]),
            "hard_p95_principal": float(final_anchor["val_hard_p95_principal"]),
            "yield_violation_p95": float(final_anchor["val_yield_violation_p95"]),
        },
        COMPRESSION_REOPEN_BAR,
    )
    ds_reopen = final_anchor is not None and _meets_bar(
        {
            "broad_plastic_mae": float(final_anchor["val_broad_plastic_mae"]),
            "hard_plastic_mae": float(final_anchor["val_hard_plastic_mae"]),
            "hard_p95_principal": float(final_anchor["val_hard_p95_principal"]),
            "yield_violation_p95": float(final_anchor["val_yield_violation_p95"]),
        },
        DS_REOPEN_BAR,
    )
    cleared_tier1 = final_anchor is not None and _meets_bar(
        {
            "broad_plastic_mae": float(final_anchor["val_broad_plastic_mae"]),
            "hard_plastic_mae": float(final_anchor["val_hard_plastic_mae"]),
            "hard_p95_principal": float(final_anchor["val_hard_p95_principal"]),
            "yield_violation_p95": float(final_anchor["val_yield_violation_p95"]),
        },
        TIER1_OPEN_GATE,
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- route cleared `hard_p95 <= 180` gate: `{bool(cleared_tier1)}`",
            f"- compression remains closed: `{not compression_reopen}`",
            f"- `DS` remains blocked: `{not ds_reopen}`",
            f"- continuation state: `{'continue_projection_student' if (winner is not None and (winner.get('materially_better_than_control', False) or phase0['focus_decision'].get('dominant_focus') is not None)) or winner is None else 'stop_projection_student'}`",
            "",
            "Bottom line:",
            "",
            "This packet localized the hard tail first, then tested bounded in-family tail-closure mechanics before any capacity increase. The route should only continue if the best same-capacity or near-same-capacity result materially improved the hard tail or if the localization stayed concentrated enough to justify another narrow packet.",
            "",
        ]
    )
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    docs_root = ROOT / args.docs_root
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    if args.phase in {"all", "exp0_control_zero_slices"}:
        run_phase0_control_zero_slices(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp1_tail_closure_controls"}:
        run_phase1_tail_closure_controls(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp2_near_same_capacity_followon"}:
        run_phase2_near_same_capacity_followon(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    _write_execution_report(output_root=output_root, docs_root=docs_root)


if __name__ == "__main__":
    main()
