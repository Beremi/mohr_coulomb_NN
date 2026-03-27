#!/usr/bin/env python
"""Run the March 27 projection-student tail reopen packet."""

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

from mc_surrogate.mohr_coulomb import BRANCH_NAMES, BRANCH_TO_ID
from mc_surrogate.principal_projection import PROJECTION_CANDIDATE_NAMES
from mc_surrogate.projection_student_preservation import (
    BOUNDARY_MASK_KEYS,
    build_any_boundary_mask,
    build_sampling_weights,
    build_top_fraction_mask,
    directional_error_arrays,
    displacement_decile_edges,
    principal_abs_error_arrays,
    project_teacher_checkpoint_stress,
    projection_switch_mask,
    quantile_or_zero,
    summarize_array,
)
from mc_surrogate.training import TrainingConfig, load_checkpoint, predict_with_checkpoint, predict_with_loaded_checkpoint
from run_nn_replacement_abr_cycle import (
    _load_h5,
    _parameter_count,
    _slice_arrays,
    _train_with_batch_fallback,
    _write_csv,
    _write_h5,
    _write_json,
)
from run_projection_student_hard_tail_closure import (
    _build_generic_rowwise_arrays,
    _evaluate_checkpoint_with_rowwise,
    _evaluate_policy,
    _summarize_rowwise,
)

SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
GROUPED_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5"
PANEL_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5"
PANEL_SUMMARY_PATH = PANEL_DATASET
PREV_PRESERVATION_ROOT = ROOT / "experiment_runs/real_sim/projection_student_preservation_compression_20260327"
PREV_HARD_TAIL_ROOT = ROOT / "experiment_runs/real_sim/projection_student_hard_tail_closure_20260327"
TEACHER_CACHE_PATH = PREV_PRESERVATION_ROOT / "exp0_teacher_cache/teacher_projection_preservation_cache.h5"
PREV_PHASE1_SUMMARY = PREV_HARD_TAIL_ROOT / "exp1_tail_closure_controls/phase1_summary.json"
PREV_SAME_CAPACITY_CSV = PREV_HARD_TAIL_ROOT / "exp1_tail_closure_controls/same_capacity_summary.csv"
WORK_PACKET_BASENAME = "projection_student_tail_reopen_work_packet_20260327.md"
REOPEN_BAR = {
    "broad_plastic_mae": 15.0,
    "hard_plastic_mae": 18.0,
    "hard_p95_principal": 160.0,
    "yield_violation_p95": 1.0e-6,
}
PHASE4_TRIGGER_BAR = {
    "broad_plastic_mae": 15.8,
    "hard_plastic_mae": 18.4,
    "hard_p95_principal": 168.0,
    "yield_violation_p95": 1.0e-6,
}
ANCHOR_EXPECTED_VAL = {
    "broad_plastic_mae": 16.68758201599121,
    "hard_plastic_mae": 18.9135799407959,
    "hard_p95_principal": 176.1450653076172,
    "yield_violation_p95": 2.310791025195158e-08,
}
ANCHOR_EXPECTED_TEST = {
    "broad_plastic_mae": 17.59446144104004,
    "hard_plastic_mae": 19.734207153320312,
    "hard_p95_principal": 183.9632568359375,
    "yield_violation_p95": 2.290212286482074e-08,
}
PROBE_TAU_GRID = (0.25, 0.50, 1.00)
PROBE_DIRECTIONS = 4
PROBE_SEED = 20260327
FD_SCALE = 1.0e-6
PHASE3_VARIANTS = (
    "anchor_ema_restart",
    "teacher_gap_cvar",
    "edge_apex_delta_focus",
    "full_combo_ema",
)
SLICE_SUMMARY_COLUMNS = [
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
]
CALL_SUMMARY_COLUMNS = [
    "scope",
    "source_call_id",
    "n_rows",
    "n_hard_rows",
    "hard_top5_count",
    "hard_top5_share",
    "mean_principal_max_abs_error",
    "p95_principal_max_abs_error",
    "p99_principal_max_abs_error",
]
PHASE1_SUMMARY_COLUMNS = [
    "table",
    "group_name",
    "group_id",
    "group_value",
    "n_rows",
    "n_hard_rows",
    "hard_share",
    "any_boundary_share",
    "apex_adjacent_share",
    "edge_branch_share",
    "student_post_mean",
    "student_post_p95",
    "teacher_post_mean",
    "teacher_post_p95",
    "gap_post_mean",
    "gap_post_p95",
    "gap_pre_mean",
    "gap_pre_p95",
    "delta_gap_mean",
    "delta_gap_p95",
    "teacher_disp_mean",
    "teacher_disp_p95",
    "student_minus_teacher_mean",
]
PROBE_SUMMARY_COLUMNS = [
    "split",
    "mode",
    "tau",
    "scope",
    "metric",
    "mean",
    "p50",
    "p95",
    "max",
    "finite_rate",
    "candidate_switch_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="all",
        choices=(
            "all",
            "exp0_anchor_freeze",
            "exp1_teacher_gap_audit",
            "exp2_projected_teacher_ds_probe",
            "exp3_same_capacity_refinement",
            "exp4_full_real_distill_followon",
            "exp5_reopen_decision",
        ),
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/projection_student_tail_reopen_20260327",
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


def _summary_attr_json(path: Path) -> dict[str, Any]:
    try:
        raw = _read_json(path)
    except json.JSONDecodeError:
        return {}
    return raw


def _resolve_anchor_checkpoint() -> Path:
    summary = _read_json(PREV_PHASE1_SUMMARY)
    winner = summary["winner"]
    checkpoint = Path(winner["best_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anchor checkpoint missing: {checkpoint}")
    return checkpoint


def _freeze_anchor_link(anchor_checkpoint: Path, phase_dir: Path) -> Path:
    link_dir = phase_dir / "anchor_checkpoint"
    link_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / "best.pt"
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == anchor_checkpoint.resolve():
            return link_path
        link_path.unlink()
    link_path.symlink_to(anchor_checkpoint.resolve())
    return link_path


def _repro_ok(metrics: dict[str, float], reference: dict[str, float]) -> bool:
    return (
        abs(float(metrics["broad_plastic_mae"]) - float(reference["broad_plastic_mae"])) <= 0.05
        and abs(float(metrics["hard_plastic_mae"]) - float(reference["hard_plastic_mae"])) <= 0.05
        and abs(float(metrics["hard_p95_principal"]) - float(reference["hard_p95_principal"])) <= 0.5
        and abs(float(metrics["yield_violation_p95"]) - float(reference["yield_violation_p95"])) <= 1.0e-8
    )


def _write_rowwise_bundle(
    *,
    phase_dir: Path,
    stem: str,
    metrics: dict[str, Any],
    rowwise_arrays: dict[str, np.ndarray],
    rowwise_attrs: dict[str, Any],
    summary: dict[str, Any],
    slice_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> None:
    _write_json(phase_dir / f"{stem}_eval.json", metrics)
    _write_json(phase_dir / f"{stem}_summary.json", summary)
    _write_h5(phase_dir / f"{stem}_rowwise.h5", rowwise_arrays, rowwise_attrs)
    _write_csv(phase_dir / f"{stem}_slice_summary.csv", slice_rows, SLICE_SUMMARY_COLUMNS)
    _write_csv(phase_dir / f"{stem}_call_concentration.csv", call_rows, CALL_SUMMARY_COLUMNS)


def _metric_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    local = arr[np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(arr)]
    if local.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(local)),
        "p50": quantile_or_zero(local, 0.50),
        "p95": quantile_or_zero(local, 0.95),
        "max": float(np.max(local)),
    }


def _group_summary_row(
    *,
    table: str,
    group_name: str,
    group_id: int,
    group_value: str,
    rowwise: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    local = np.asarray(mask, dtype=bool) & (np.asarray(rowwise["plastic_mask"], dtype=bool))
    hard = np.asarray(rowwise["hard_mask"], dtype=bool)
    apex_adjacent = np.asarray(rowwise["apex_adjacent_mask"], dtype=bool)
    any_boundary = np.asarray(rowwise["any_boundary_mask"], dtype=bool)
    branch = np.asarray(rowwise["branch_id"], dtype=np.int64)
    if not np.any(local):
        return {
            "table": table,
            "group_name": group_name,
            "group_id": int(group_id),
            "group_value": group_value,
            "n_rows": 0,
            "n_hard_rows": 0,
            "hard_share": 0.0,
            "any_boundary_share": 0.0,
            "apex_adjacent_share": 0.0,
            "edge_branch_share": 0.0,
            "student_post_mean": 0.0,
            "student_post_p95": 0.0,
            "teacher_post_mean": 0.0,
            "teacher_post_p95": 0.0,
            "gap_post_mean": 0.0,
            "gap_post_p95": 0.0,
            "gap_pre_mean": 0.0,
            "gap_pre_p95": 0.0,
            "delta_gap_mean": 0.0,
            "delta_gap_p95": 0.0,
            "teacher_disp_mean": 0.0,
            "teacher_disp_p95": 0.0,
            "student_minus_teacher_mean": 0.0,
        }
    student_stats = _metric_stats(rowwise["err_student_post_max_abs"], local)
    teacher_stats = _metric_stats(rowwise["err_teacher_post_max_abs"], local)
    gap_post_stats = _metric_stats(rowwise["gap_to_teacher_post_max_abs"], local)
    gap_pre_stats = _metric_stats(rowwise["gap_to_teacher_pre_max_abs"], local)
    delta_gap_stats = _metric_stats(rowwise["delta_gap_max_abs"], local)
    disp_stats = _metric_stats(rowwise["teacher_projection_disp_norm"], local)
    return {
        "table": table,
        "group_name": group_name,
        "group_id": int(group_id),
        "group_value": group_value,
        "n_rows": int(np.sum(local)),
        "n_hard_rows": int(np.sum(local & hard)),
        "hard_share": float(np.mean(hard[local])),
        "any_boundary_share": float(np.mean(any_boundary[local])),
        "apex_adjacent_share": float(np.mean(apex_adjacent[local])),
        "edge_branch_share": float(np.mean(np.isin(branch[local], [BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]]))),
        "student_post_mean": student_stats["mean"],
        "student_post_p95": student_stats["p95"],
        "teacher_post_mean": teacher_stats["mean"],
        "teacher_post_p95": teacher_stats["p95"],
        "gap_post_mean": gap_post_stats["mean"],
        "gap_post_p95": gap_post_stats["p95"],
        "gap_pre_mean": gap_pre_stats["mean"],
        "gap_pre_p95": gap_pre_stats["p95"],
        "delta_gap_mean": delta_gap_stats["mean"],
        "delta_gap_p95": delta_gap_stats["p95"],
        "teacher_disp_mean": disp_stats["mean"],
        "teacher_disp_p95": disp_stats["p95"],
        "student_minus_teacher_mean": float(
            np.mean(np.asarray(rowwise["err_student_post_max_abs"], dtype=float)[local] - np.asarray(rowwise["err_teacher_post_max_abs"], dtype=float)[local])
        ),
    }


def _phase1_group_rows(rowwise: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch = np.asarray(rowwise["branch_id"], dtype=np.int64)
    for idx, name in enumerate(BRANCH_NAMES):
        rows.append(
            _group_summary_row(
                table="true_branch",
                group_name="branch_id",
                group_id=idx,
                group_value=name,
                rowwise=rowwise,
                mask=branch == idx,
            )
        )
    for key in (*BOUNDARY_MASK_KEYS, "any_boundary_mask", "apex_adjacent_mask"):
        values = np.asarray(rowwise[key], dtype=bool)
        rows.append(_group_summary_row(table="boundary_mask", group_name=key, group_id=0, group_value="false", rowwise=rowwise, mask=~values))
        rows.append(_group_summary_row(table="boundary_mask", group_name=key, group_id=1, group_value="true", rowwise=rowwise, mask=values))
    decile = np.asarray(rowwise["teacher_projection_disp_decile"], dtype=np.int64)
    for idx in range(10):
        rows.append(
            _group_summary_row(
                table="teacher_disp_decile",
                group_name="teacher_projection_disp_decile",
                group_id=idx,
                group_value=f"decile_{idx + 1}",
                rowwise=rowwise,
                mask=decile == idx,
            )
        )
    apex_adjacent = np.asarray(rowwise["apex_adjacent_mask"], dtype=bool)
    any_boundary = np.asarray(rowwise["any_boundary_mask"], dtype=bool)
    edge_mask = np.isin(branch, [BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]])
    intersection_masks = {
        "left_edge_apex_adjacent": (branch == BRANCH_TO_ID["left_edge"]) & apex_adjacent,
        "right_edge_apex_adjacent": (branch == BRANCH_TO_ID["right_edge"]) & apex_adjacent,
        "smooth_any_boundary": (branch == BRANCH_TO_ID["smooth"]) & any_boundary,
        "edge_top_disp_decile": edge_mask & (decile == 9),
    }
    for idx, (name, mask) in enumerate(intersection_masks.items()):
        rows.append(
            _group_summary_row(
                table="intersection",
                group_name="intersection",
                group_id=idx,
                group_value=name,
                rowwise=rowwise,
                mask=mask,
            )
        )
    plastic = np.asarray(rowwise["plastic_mask"], dtype=bool)
    hard = np.asarray(rowwise["hard_mask"], dtype=bool)
    criteria = {
        "student_hard_error": np.asarray(rowwise["err_student_post_max_abs"], dtype=float),
        "student_teacher_post_gap": np.asarray(rowwise["gap_to_teacher_post_max_abs"], dtype=float),
        "projection_delta_gap": np.asarray(rowwise["delta_gap_max_abs"], dtype=float),
    }
    for criterion, values in criteria.items():
        for fraction in (0.05, 0.01):
            pool = plastic & (hard if criterion == "student_hard_error" else np.ones(plastic.shape[0], dtype=bool))
            top_mask, _threshold = build_top_fraction_mask(values, pool_mask=pool, fraction=fraction)
            rows.append(
                _group_summary_row(
                    table="top_fraction",
                    group_name=criterion,
                    group_id=int(fraction * 100),
                    group_value=f"top_{int(fraction * 100)}pct",
                    rowwise=rowwise,
                    mask=top_mask,
                )
            )
            rows.append(
                _group_summary_row(
                    table="top_fraction",
                    group_name=f"{criterion}_any_boundary",
                    group_id=int(fraction * 100),
                    group_value=f"top_{int(fraction * 100)}pct_any_boundary",
                    rowwise=rowwise,
                    mask=top_mask & any_boundary,
                )
            )
            rows.append(
                _group_summary_row(
                    table="top_fraction",
                    group_name=f"{criterion}_edge_apex_adjacent",
                    group_id=int(fraction * 100),
                    group_value=f"top_{int(fraction * 100)}pct_edge_apex_adjacent",
                    rowwise=rowwise,
                    mask=top_mask & np.asarray(rowwise["edge_apex_adjacent_mask"], dtype=bool),
                )
            )
    return rows


def _phase1_weighting_bands(rowwise: dict[str, np.ndarray]) -> dict[str, Any]:
    plastic = np.asarray(rowwise["plastic_mask"], dtype=bool)
    gap_post = np.asarray(rowwise["gap_to_teacher_post_max_abs"], dtype=float)
    delta_gap = np.asarray(rowwise["delta_gap_max_abs"], dtype=float)
    disp = np.asarray(rowwise["teacher_projection_disp_norm"], dtype=float)
    gap_q90 = quantile_or_zero(gap_post[plastic], 0.90)
    gap_q95 = quantile_or_zero(gap_post[plastic], 0.95)
    delta_q90 = quantile_or_zero(delta_gap[plastic], 0.90)
    teacher_disp_q90 = quantile_or_zero(disp[plastic], 0.90)
    gap_q95_mask = plastic & (gap_post >= gap_q95)
    delta_q90_mask = plastic & (delta_gap >= delta_q90)
    any_boundary = np.asarray(rowwise["any_boundary_mask"], dtype=bool)
    apex_adjacent = np.asarray(rowwise["apex_adjacent_mask"], dtype=bool)
    edge_apex_adjacent = np.asarray(rowwise["edge_apex_adjacent_mask"], dtype=bool)
    return {
        "gap_q90": float(gap_q90),
        "gap_q95": float(gap_q95),
        "delta_q90": float(delta_q90),
        "teacher_disp_q90": float(teacher_disp_q90),
        "concentration": {
            "gap_q95_any_boundary_share": float(np.mean(any_boundary[gap_q95_mask])) if np.any(gap_q95_mask) else 0.0,
            "gap_q95_apex_adjacent_share": float(np.mean(apex_adjacent[gap_q95_mask])) if np.any(gap_q95_mask) else 0.0,
            "gap_q95_edge_apex_adjacent_share": float(np.mean(edge_apex_adjacent[gap_q95_mask])) if np.any(gap_q95_mask) else 0.0,
            "delta_q90_any_boundary_share": float(np.mean(any_boundary[delta_q90_mask])) if np.any(delta_q90_mask) else 0.0,
            "delta_q90_edge_apex_adjacent_share": float(np.mean(edge_apex_adjacent[delta_q90_mask])) if np.any(delta_q90_mask) else 0.0,
        },
        "recommended_sampling_multipliers": {
            "teacher_gap_q90": 1.50,
            "teacher_gap_q95": 2.00,
            "delta_gap_q90": 1.40,
            "edge_apex_adjacent": 1.75,
            "any_boundary": 1.25,
            "high_disp": 1.25,
            "edge_branch": 1.15,
        },
    }


def _probe_directions(n_rows: int, *, seed: int, n_dirs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_rows, n_dirs, 6)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return directions / np.maximum(norms, 1.0e-12)


def _probe_scope_summary(
    metric_arrays: dict[str, np.ndarray],
    finite_mask: np.ndarray,
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
            "finite_rate": 0.0,
            "candidate_switch_rate": 0.0,
        }
    return {
        "n_rows": int(np.sum(mask)),
        "n_direction_samples": int(metric_arrays["abs_error_norm"][mask].size),
        "abs_error_norm": summarize_array(metric_arrays["abs_error_norm"][mask]),
        "relative_error": summarize_array(metric_arrays["relative_error"][mask]),
        "cosine_similarity": summarize_array(metric_arrays["cosine_similarity"][mask]),
        "finite_rate": float(np.mean(finite_mask[mask])),
        "candidate_switch_rate": float(np.mean(switch_events[mask])),
    }


def _probe_summary_rows(split_name: str, mode: str, tau: float | None, scope_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in ("abs_error_norm", "relative_error", "cosine_similarity"):
        rows.append(
            {
                "split": split_name,
                "mode": mode,
                "tau": "" if tau is None else f"{tau:.2f}",
                "scope": scope_name,
                "metric": metric,
                "mean": payload[metric]["mean"],
                "p50": payload[metric]["p50"],
                "p95": payload[metric]["p95"],
                "max": payload[metric]["max"],
                "finite_rate": payload["finite_rate"],
                "candidate_switch_rate": payload["candidate_switch_rate"],
            }
        )
    return rows


def _probe_mode_on_split(
    split_arrays: dict[str, np.ndarray],
    *,
    teacher_model: Any,
    teacher_metadata: dict[str, Any],
    mode: str,
    tau: float | None,
    directions: np.ndarray,
    device: str,
    eval_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_rows = split_arrays["strain_eng"].shape[0]
    h = FD_SCALE * np.maximum(np.amax(np.abs(split_arrays["strain_eng"]), axis=1, keepdims=True), 1.0).astype(np.float32)
    proj_tau = 0.05 if tau is None else float(tau)
    base_projection = project_teacher_checkpoint_stress(
        split_arrays["strain_eng"],
        split_arrays["material_reduced"],
        split_arrays["teacher_checkpoint_stress"],
        mode=mode,
        tau=proj_tau,
    )

    abs_error_norm = np.zeros((n_rows, directions.shape[1]), dtype=np.float32)
    relative_error = np.zeros((n_rows, directions.shape[1]), dtype=np.float32)
    cosine_similarity = np.zeros((n_rows, directions.shape[1]), dtype=np.float32)
    finite_mask = np.zeros((n_rows, directions.shape[1]), dtype=np.int8)
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
            tau=proj_tau,
        )
        minus_projection = project_teacher_checkpoint_stress(
            strain_minus,
            split_arrays["material_reduced"],
            minus_pred["stress"],
            mode=mode,
            tau=proj_tau,
        )
        jv_pred = (plus_projection["teacher_projected_stress"] - minus_projection["teacher_projected_stress"]) / (2.0 * h)
        jv_true = np.einsum("nij,nj->ni", tangent_true, direction, optimize=True)
        metrics = directional_error_arrays(np.nan_to_num(jv_pred, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(jv_true, nan=0.0))
        finite = np.isfinite(jv_pred).all(axis=1) & np.isfinite(jv_true).all(axis=1)
        abs_error_norm[:, dir_idx] = metrics["abs_error_norm"]
        relative_error[:, dir_idx] = metrics["relative_error"]
        cosine_similarity[:, dir_idx] = metrics["cosine_similarity"]
        finite_mask[:, dir_idx] = finite.astype(np.int8)
        switch_events[:, dir_idx] = projection_switch_mask(
            base_projection["teacher_projection_candidate_id"],
            plus_projection["teacher_projection_candidate_id"],
            minus_projection["teacher_projection_candidate_id"],
        ).astype(np.int8)
        candidate_plus[:, dir_idx] = plus_projection["teacher_projection_candidate_id"]
        candidate_minus[:, dir_idx] = minus_projection["teacher_projection_candidate_id"]

    any_boundary = build_any_boundary_mask(split_arrays)
    apex_adjacent = split_arrays["near_left_apex_mask"].astype(bool) | split_arrays["near_right_apex_mask"].astype(bool)
    metric_arrays = {
        "abs_error_norm": abs_error_norm,
        "relative_error": relative_error,
        "cosine_similarity": cosine_similarity,
    }
    summary = {
        "overall": _probe_scope_summary(metric_arrays, finite_mask, switch_events, np.ones(n_rows, dtype=bool)),
        "plastic_only": _probe_scope_summary(metric_arrays, finite_mask, switch_events, split_arrays["branch_id"] > 0),
        "true_branch": {
            name: _probe_scope_summary(metric_arrays, finite_mask, switch_events, split_arrays["branch_id"] == idx)
            for idx, name in enumerate(BRANCH_NAMES[1:], start=1)
        },
        "any_boundary": {
            "true": _probe_scope_summary(metric_arrays, finite_mask, switch_events, any_boundary),
            "false": _probe_scope_summary(metric_arrays, finite_mask, switch_events, ~any_boundary),
        },
        "apex_adjacent": {
            "true": _probe_scope_summary(metric_arrays, finite_mask, switch_events, apex_adjacent),
            "false": _probe_scope_summary(metric_arrays, finite_mask, switch_events, ~apex_adjacent),
        },
        "candidate_base": {
            name: _probe_scope_summary(metric_arrays, finite_mask, switch_events, base_projection["teacher_projection_candidate_id"] == idx)
            for idx, name in enumerate(PROJECTION_CANDIDATE_NAMES)
        },
    }
    rowwise = {
        "source_call_id": split_arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": split_arrays["source_row_in_call"].astype(np.int32),
        "branch_id": split_arrays["branch_id"].astype(np.int8),
        "hard_mask": split_arrays["hard_mask"].astype(np.int8),
        "plastic_mask": split_arrays["plastic_mask"].astype(np.int8),
        "ds_valid_mask": split_arrays["ds_valid_mask"].astype(np.int8),
        "any_boundary_mask": any_boundary.astype(np.int8),
        "apex_adjacent_mask": apex_adjacent.astype(np.int8),
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
        f"{mode}_finite_mask": finite_mask.astype(np.int8),
        f"{mode}_candidate_switch": switch_events.astype(np.int8),
    }
    return rowwise, summary


def run_phase0_anchor_freeze(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp0_anchor_freeze"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase0_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    anchor_checkpoint = _resolve_anchor_checkpoint()
    frozen_link = _freeze_anchor_link(anchor_checkpoint, phase_dir)
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, cache_attrs = _load_h5(TEACHER_CACHE_PATH)

    split_summaries: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        split_cache = _slice_arrays(cache_arrays, _split_mask(cache_arrays["split_id"], split_name))
        disp_edges = displacement_decile_edges(
            split_cache["teacher_projection_disp_norm"],
            reference_mask=np.asarray(split_cache["plastic_mask"], dtype=bool),
            n_bins=10,
        )
        metrics, rowwise, summary, slice_rows, call_rows = _evaluate_checkpoint_with_rowwise(
            checkpoint_path=anchor_checkpoint,
            split_name=split_name,
            arrays_all=arrays_all,
            panel_all=panel_all,
            cache_arrays=cache_arrays,
            disp_edges=disp_edges,
            device=device,
            eval_batch_size=eval_batch_size,
            prefix="anchor",
        )
        _write_h5(
            phase_dir / f"anchor_rowwise_{split_name}.h5",
            rowwise,
            {
                "source_checkpoint": str(anchor_checkpoint),
                "disp_decile_edges": [float(x) for x in disp_edges],
                "teacher_cache_path": str(TEACHER_CACHE_PATH),
            },
        )
        _write_json(phase_dir / f"anchor_eval_{split_name}.json", metrics)
        _write_json(phase_dir / f"anchor_summary_{split_name}.json", summary)
        _write_csv(phase_dir / f"anchor_slice_summary_{split_name}.csv", slice_rows, SLICE_SUMMARY_COLUMNS)
        _write_csv(phase_dir / f"anchor_call_concentration_{split_name}.csv", call_rows, CALL_SUMMARY_COLUMNS)
        split_summaries[split_name] = {
            "metrics": metrics,
            "summary": summary,
            "disp_decile_edges": [float(x) for x in disp_edges],
        }

    if not _repro_ok(split_summaries["val"]["metrics"], ANCHOR_EXPECTED_VAL) or not _repro_ok(split_summaries["test"]["metrics"], ANCHOR_EXPECTED_TEST):
        raise RuntimeError(
            "Frozen combined_focus anchor did not reproduce the expected val/test metrics closely enough."
        )

    summary = {
        "anchor_checkpoint": str(anchor_checkpoint),
        "frozen_anchor_path": str(frozen_link),
        "teacher_cache_path": str(TEACHER_CACHE_PATH),
        "cache_attrs": cache_attrs,
        "split_summaries": split_summaries,
        "repro_ok": True,
    }
    _write_json(summary_path, summary)
    return summary


def run_phase1_teacher_gap_audit(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp1_teacher_gap_audit"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase1_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase0 = _read_json(output_root / "exp0_anchor_freeze" / "phase0_summary.json")
    anchor_checkpoint = Path(phase0["anchor_checkpoint"])
    cache_arrays, _cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    train_mask = _split_mask(cache_arrays["split_id"], "train")
    train_cache = _slice_arrays(cache_arrays, train_mask)
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    train_arrays = _slice_arrays(arrays_all, train_mask)
    pred = predict_with_checkpoint(
        anchor_checkpoint,
        train_arrays["strain_eng"],
        train_arrays["material_reduced"],
        device=device,
        batch_size=eval_batch_size,
    )

    disp_edges = np.asarray(phase0["split_summaries"]["train"]["disp_decile_edges"], dtype=float)
    rowwise = _build_generic_rowwise_arrays(
        prefix="anchor",
        split_arrays=train_cache,
        predicted_principal=pred["stress_principal"].astype(np.float32),
        provisional_principal=pred.get("provisional_stress_principal"),
        predicted_branch_id=pred.get("predicted_branch_id"),
        disp_decile_edges=disp_edges,
    )
    exact = np.asarray(train_cache["exact_stress_principal"], dtype=np.float32)
    teacher_post = np.asarray(train_cache["teacher_projected_stress_principal"], dtype=np.float32)
    teacher_pre = np.asarray(train_cache["teacher_dispatch_stress_principal"], dtype=np.float32)
    student_post = np.asarray(pred["stress_principal"], dtype=np.float32)
    student_pre = np.asarray(pred["provisional_stress_principal"], dtype=np.float32)
    err_student_post, err_student_post_max = principal_abs_error_arrays(student_post, exact)
    err_teacher_post, err_teacher_post_max = principal_abs_error_arrays(teacher_post, exact)
    gap_post, gap_post_max = principal_abs_error_arrays(student_post, teacher_post)
    gap_pre, gap_pre_max = principal_abs_error_arrays(student_pre, teacher_pre)
    delta_student = (student_post - student_pre).astype(np.float32)
    delta_teacher = (teacher_post - teacher_pre).astype(np.float32)
    delta_gap, delta_gap_max = principal_abs_error_arrays(delta_student, delta_teacher)
    apex_adjacent = train_cache["near_left_apex_mask"].astype(bool) | train_cache["near_right_apex_mask"].astype(bool)
    edge_apex_adjacent = apex_adjacent & np.isin(train_cache["branch_id"], [BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]])
    rowwise.update(
        {
            "teacher_dispatch_stress_principal": teacher_pre,
            "err_student_post_abs": err_student_post.astype(np.float32),
            "err_student_post_max_abs": err_student_post_max.astype(np.float32),
            "err_teacher_post_abs": err_teacher_post.astype(np.float32),
            "err_teacher_post_max_abs": err_teacher_post_max.astype(np.float32),
            "gap_to_teacher_post_abs": gap_post.astype(np.float32),
            "gap_to_teacher_post_max_abs": gap_post_max.astype(np.float32),
            "gap_to_teacher_pre_abs": gap_pre.astype(np.float32),
            "gap_to_teacher_pre_max_abs": gap_pre_max.astype(np.float32),
            "delta_proj_student": delta_student.astype(np.float32),
            "delta_proj_teacher": delta_teacher.astype(np.float32),
            "delta_gap_abs": delta_gap.astype(np.float32),
            "delta_gap_max_abs": delta_gap_max.astype(np.float32),
            "apex_adjacent_mask": apex_adjacent.astype(np.int8),
            "edge_apex_adjacent_mask": edge_apex_adjacent.astype(np.int8),
        }
    )

    summary_rows = _phase1_group_rows(rowwise)
    weighting_bands = _phase1_weighting_bands(rowwise)
    _write_h5(
        phase_dir / "teacher_gap_train_rowwise.h5",
        rowwise,
        {
            "source_checkpoint": str(anchor_checkpoint),
            "disp_decile_edges": [float(x) for x in disp_edges],
            "weighting_bands_json": weighting_bands,
        },
    )
    _write_csv(phase_dir / "teacher_gap_train_slice_summary.csv", summary_rows, PHASE1_SUMMARY_COLUMNS)
    _write_json(phase_dir / "teacher_gap_weighting_bands.json", weighting_bands)
    summary = {
        "anchor_checkpoint": str(anchor_checkpoint),
        "rowwise_path": str(phase_dir / "teacher_gap_train_rowwise.h5"),
        "slice_summary_path": str(phase_dir / "teacher_gap_train_slice_summary.csv"),
        "weighting_bands": weighting_bands,
    }
    _write_json(summary_path, summary)
    return summary


def run_phase2_projected_teacher_ds_probe(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp2_projected_teacher_ds_probe"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase2_summary.json"
    csv_path = phase_dir / "teacher_ds_probe_summary.csv"
    if summary_path.exists() and csv_path.exists() and not force_rerun:
        return _read_json(summary_path)

    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, _cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    tangent_key = "DS" if "DS" in arrays_all else "tangent"
    if tangent_key not in arrays_all:
        summary = {
            "status": "blocked_no_tangent",
            "reason": "Grouped dataset does not contain DS/tangent tensors.",
        }
        _write_json(summary_path, summary)
        return summary

    teacher_model, teacher_metadata = load_checkpoint(
        PREV_PRESERVATION_ROOT / "exp0_teacher_cache/../../hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt",
        device=device,
    )
    summary_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
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
            "near_left_apex_mask": panel_all["near_left_apex_mask"][mask].astype(np.int8),
            "near_right_apex_mask": panel_all["near_right_apex_mask"][mask].astype(np.int8),
            "teacher_checkpoint_stress": cache_arrays["teacher_checkpoint_stress"][mask].astype(np.float32),
            "DS": arrays_all[tangent_key][mask].astype(np.float32),
        }
        for key in ("near_yield_mask", "near_smooth_left_mask", "near_smooth_right_mask"):
            split_arrays[key] = panel_all[key][mask].astype(np.int8)
        directions = _probe_directions(split_arrays["strain_eng"].shape[0], seed=PROBE_SEED + split_idx, n_dirs=PROBE_DIRECTIONS)
        mode_payloads: dict[str, Any] = {}
        rowwise_payload: dict[str, np.ndarray] = {}
        exact_rowwise, exact_summary = _probe_mode_on_split(
            split_arrays,
            teacher_model=teacher_model,
            teacher_metadata=teacher_metadata,
            mode="exact",
            tau=None,
            directions=directions,
            device=device,
            eval_batch_size=eval_batch_size,
        )
        rowwise_payload.update(exact_rowwise)
        mode_payloads["exact"] = {"summary": exact_summary, "tau": None}
        summary_rows.extend(_probe_summary_rows(split_name, "exact", None, "overall", exact_summary["overall"]))
        summary_rows.extend(_probe_summary_rows(split_name, "exact", None, "plastic_only", exact_summary["plastic_only"]))
        for tau in PROBE_TAU_GRID:
            mode_name = f"softmin_tau_{tau:.2f}"
            soft_rowwise, soft_summary = _probe_mode_on_split(
                split_arrays,
                teacher_model=teacher_model,
                teacher_metadata=teacher_metadata,
                mode="softmin",
                tau=tau,
                directions=directions,
                device=device,
                eval_batch_size=eval_batch_size,
            )
            softmin_forward = project_teacher_checkpoint_stress(
                split_arrays["strain_eng"],
                split_arrays["material_reduced"],
                split_arrays["teacher_checkpoint_stress"],
                mode="softmin",
                tau=tau,
            )
            exact_forward = project_teacher_checkpoint_stress(
                split_arrays["strain_eng"],
                split_arrays["material_reduced"],
                split_arrays["teacher_checkpoint_stress"],
                mode="exact",
                tau=0.05,
            )
            soft_summary["forward_delta_vs_exact"] = {
                "stress_component_mae": summarize_array(
                    np.mean(np.abs(softmin_forward["teacher_projected_stress"] - exact_forward["teacher_projected_stress"]), axis=1).astype(np.float32)
                ),
                "principal_max_abs": summarize_array(
                    np.max(np.abs(softmin_forward["teacher_projected_stress_principal"] - exact_forward["teacher_projected_stress_principal"]), axis=1).astype(np.float32)
                ),
            }
            rowwise_payload.update({f"{mode_name}_{key}": value for key, value in soft_rowwise.items() if key.startswith("softmin_")})
            mode_payloads[mode_name] = {"summary": soft_summary, "tau": tau}
            summary_rows.extend(_probe_summary_rows(split_name, "softmin", tau, "overall", soft_summary["overall"]))
            summary_rows.extend(_probe_summary_rows(split_name, "softmin", tau, "plastic_only", soft_summary["plastic_only"]))

        _write_h5(
            phase_dir / f"teacher_ds_probe_{split_name}.h5",
            rowwise_payload,
            {
                "split": split_name,
                "n_directions": PROBE_DIRECTIONS,
                "fd_scale": FD_SCALE,
                "tau_grid": list(PROBE_TAU_GRID),
            },
        )
        split_summaries[split_name] = mode_payloads

    summary = {
        "status": "completed",
        "n_directions": PROBE_DIRECTIONS,
        "fd_scale": FD_SCALE,
        "tau_grid": list(PROBE_TAU_GRID),
        "split_summaries": split_summaries,
    }
    _write_json(summary_path, summary)
    _write_csv(csv_path, summary_rows, PROBE_SUMMARY_COLUMNS)
    return summary


def _phase3_base_dataset(
    *,
    phase_dir: Path,
    phase1: dict[str, Any],
    force_rerun: bool,
) -> tuple[Path, dict[str, Any]]:
    dataset_path = phase_dir / "base_tail_reopen_dataset.h5"
    summary_path = phase_dir / "base_tail_reopen_dataset_summary.json"
    if dataset_path.exists() and summary_path.exists() and not force_rerun:
        return dataset_path, _read_json(summary_path)

    cache_arrays, cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    arrays_all, dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, panel_attrs = _load_h5(PANEL_DATASET)
    phase1_rowwise, _rowwise_attrs = _load_h5(Path(phase1["rowwise_path"]))
    weighting_bands = phase1["weighting_bands"]

    plastic_mask = cache_arrays["plastic_mask"].astype(bool)
    global_split = cache_arrays["split_id"][plastic_mask]
    any_boundary = build_any_boundary_mask(cache_arrays)[plastic_mask]
    apex_adjacent = (cache_arrays["near_left_apex_mask"].astype(bool) | cache_arrays["near_right_apex_mask"].astype(bool))[plastic_mask]
    edge_apex_adjacent = apex_adjacent & np.isin(cache_arrays["branch_id"][plastic_mask], [BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]])

    train_phase1_plastic = phase1_rowwise["plastic_mask"].astype(bool)
    train_gap_q90_mask = phase1_rowwise["gap_to_teacher_post_max_abs"][train_phase1_plastic] >= float(weighting_bands["gap_q90"])
    train_gap_q95_mask = phase1_rowwise["gap_to_teacher_post_max_abs"][train_phase1_plastic] >= float(weighting_bands["gap_q95"])
    train_delta_q90_mask = phase1_rowwise["delta_gap_max_abs"][train_phase1_plastic] >= float(weighting_bands["delta_q90"])
    n_plastic = int(np.sum(plastic_mask))
    teacher_gap_q90_mask = np.zeros(n_plastic, dtype=np.int8)
    teacher_gap_q95_mask = np.zeros(n_plastic, dtype=np.int8)
    delta_gap_q90_mask = np.zeros(n_plastic, dtype=np.int8)
    train_plastic_mask = global_split == SPLIT_NAME_TO_ID["train"]
    teacher_gap_q90_mask[train_plastic_mask] = train_gap_q90_mask.astype(np.int8)
    teacher_gap_q95_mask[train_plastic_mask] = train_gap_q95_mask.astype(np.int8)
    delta_gap_q90_mask[train_plastic_mask] = train_delta_q90_mask.astype(np.int8)

    base_sampling_weight, disp_threshold = build_sampling_weights(
        train_mask=global_split == SPLIT_NAME_TO_ID["train"],
        hard_mask=cache_arrays["hard_mask"][plastic_mask] > 0,
        teacher_projection_candidate_id=cache_arrays["teacher_projection_candidate_id"][plastic_mask],
        teacher_projection_disp_norm=cache_arrays["teacher_projection_disp_norm"][plastic_mask],
        any_boundary_mask=any_boundary,
        branch_id=cache_arrays["branch_id"][plastic_mask],
        hard_multiplier=1.25,
        boundary_multiplier=1.10,
        edge_candidate_multiplier=1.15,
        high_disp_multiplier=1.15,
        high_disp_threshold=float(weighting_bands["teacher_disp_q90"]),
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
        "plastic_mask": np.ones(n_plastic, dtype=np.int8),
        "teacher_stress_principal": cache_arrays["teacher_dispatch_stress_principal"][plastic_mask].astype(np.float32),
        "teacher_provisional_stress_principal": cache_arrays["teacher_dispatch_stress_principal"][plastic_mask].astype(np.float32),
        "teacher_projected_stress_principal": cache_arrays["teacher_projected_stress_principal"][plastic_mask].astype(np.float32),
        "teacher_projection_delta_principal": cache_arrays["teacher_projection_delta_principal"][plastic_mask].astype(np.float32),
        "teacher_projection_candidate_id": cache_arrays["teacher_projection_candidate_id"][plastic_mask].astype(np.int8),
        "teacher_projection_disp_norm": cache_arrays["teacher_projection_disp_norm"][plastic_mask].astype(np.float32),
        "sampling_weight": base_sampling_weight.astype(np.float32),
        "any_boundary_mask": any_boundary.astype(np.int8),
        "apex_adjacent_mask": apex_adjacent.astype(np.int8),
        "edge_apex_adjacent_mask": edge_apex_adjacent.astype(np.int8),
        "teacher_gap_q90_mask": teacher_gap_q90_mask.astype(np.int8),
        "teacher_gap_q95_mask": teacher_gap_q95_mask.astype(np.int8),
        "delta_gap_q90_mask": delta_gap_q90_mask.astype(np.int8),
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
            "teacher_gap_q90": float(weighting_bands["gap_q90"]),
            "teacher_gap_q95": float(weighting_bands["gap_q95"]),
            "delta_gap_q90": float(weighting_bands["delta_q90"]),
            "high_disp_focus_threshold": float(weighting_bands["teacher_disp_q90"]),
            "weighting_bands_json": weighting_bands,
            "cache_attrs_json": cache_attrs,
            "dataset_attrs_json": dataset_attrs,
            "panel_attrs_json": panel_attrs,
        },
    )
    summary = {
        "dataset_path": str(dataset_path),
        "n_rows": n_plastic,
        "split_counts": {name: int(np.sum(global_split == idx)) for name, idx in SPLIT_NAME_TO_ID.items()},
        "teacher_gap_q90_train_count": int(np.sum(teacher_gap_q90_mask)),
        "teacher_gap_q95_train_count": int(np.sum(teacher_gap_q95_mask)),
        "delta_gap_q90_train_count": int(np.sum(delta_gap_q90_mask)),
        "edge_apex_adjacent_count": int(np.sum(edge_apex_adjacent)),
        "sampling_weight_summary": summarize_array(base_sampling_weight),
    }
    _write_json(summary_path, summary)
    return dataset_path, summary


def _phase3_variant_dataset(
    *,
    base_dataset_path: Path,
    variant_name: str,
    phase_dir: Path,
    weighting_bands: dict[str, Any],
    force_rerun: bool,
) -> Path:
    if variant_name == "anchor_ema_restart":
        return base_dataset_path
    variant_dir = phase_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = variant_dir / "dataset.h5"
    if dataset_path.exists() and not force_rerun:
        return dataset_path

    arrays, attrs = _load_h5(base_dataset_path)
    weights = np.ones_like(arrays["sampling_weight"], dtype=np.float32)
    train_mask = arrays["split_id"] == SPLIT_NAME_TO_ID["train"]
    edge_branch = np.isin(arrays["branch_id"], [BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]])
    high_disp = arrays["teacher_projection_disp_norm"] >= float(weighting_bands["teacher_disp_q90"])

    if variant_name == "teacher_gap_cvar":
        weights[train_mask & (arrays["teacher_gap_q90_mask"] > 0)] *= 1.50
        weights[train_mask & (arrays["teacher_gap_q95_mask"] > 0)] *= 2.00
        weights[train_mask & (arrays["delta_gap_q90_mask"] > 0)] *= 1.40
        weights[train_mask & high_disp] *= 1.15
        weights[train_mask & (arrays["hard_mask"] > 0)] *= 1.15
    elif variant_name == "edge_apex_delta_focus":
        weights[train_mask & (arrays["edge_apex_adjacent_mask"] > 0)] *= 1.75
        weights[train_mask & (arrays["delta_gap_q90_mask"] > 0)] *= 1.40
        weights[train_mask & (arrays["any_boundary_mask"] > 0)] *= 1.25
        weights[train_mask & high_disp] *= 1.15
        weights[train_mask & edge_branch] *= 1.15
    elif variant_name == "full_combo_ema":
        weights[train_mask & (arrays["teacher_gap_q90_mask"] > 0)] *= 1.50
        weights[train_mask & (arrays["teacher_gap_q95_mask"] > 0)] *= 2.00
        weights[train_mask & (arrays["delta_gap_q90_mask"] > 0)] *= 1.40
        weights[train_mask & (arrays["edge_apex_adjacent_mask"] > 0)] *= 1.75
        weights[train_mask & (arrays["any_boundary_mask"] > 0)] *= 1.25
        weights[train_mask & high_disp] *= 1.25
        weights[train_mask & edge_branch] *= 1.15
        weights[train_mask & (arrays["hard_mask"] > 0)] *= 1.15
    else:
        raise ValueError(f"Unsupported phase3 variant {variant_name!r}.")

    arrays["sampling_weight"] = weights.astype(np.float32)
    attrs["variant_sampling_rule"] = {
        "variant": variant_name,
        "weighting_bands": weighting_bands,
    }
    _write_h5(dataset_path, arrays, attrs)
    return dataset_path


def _phase3_config(
    *,
    dataset_path: Path,
    run_dir: Path,
    variant_name: str,
    anchor_checkpoint: Path,
    weighting_bands: dict[str, Any],
    seed: int,
) -> TrainingConfig:
    common = dict(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_geom_projected_student",
        batch_size=4096,
        weight_decay=1.0e-5,
        width=256,
        depth=4,
        dropout=0.0,
        seed=seed,
        grad_clip=1.0,
        branch_loss_weight=0.0,
        num_workers=0,
        device="auto",
        regression_loss_kind="huber",
        huber_delta=1.0,
        checkpoint_metric="loss",
        log_every_epochs=5,
        projection_mode="exact",
        projection_tau=0.05,
        init_checkpoint=str(anchor_checkpoint),
        projected_student_high_disp_threshold=float(weighting_bands["teacher_disp_q90"]),
    )
    if variant_name == "anchor_ema_restart":
        return TrainingConfig(
            **common,
            epochs=18,
            lr=5.0e-5,
            patience=6,
            scheduler_kind="cosine",
            warmup_epochs=2,
            min_lr=5.0e-6,
            ema_decay=0.999,
            ema_eval=True,
            ema_start_epoch=1,
            projected_student_hard_quantile=0.90,
            projected_student_hard_quantile_weight=0.20,
        )
    if variant_name == "teacher_gap_cvar":
        return TrainingConfig(
            **common,
            epochs=22,
            lr=7.5e-5,
            patience=6,
            scheduler_kind="plateau",
            ema_decay=0.0,
            projected_student_teacher_gap_q90_loss_multiplier=1.35,
            projected_student_teacher_gap_q95_loss_multiplier=1.75,
            projected_student_delta_gap_q90_loss_multiplier=1.20,
            projected_student_teacher_alignment_focus_multiplier=1.50,
            projected_student_hard_quantile=0.92,
            projected_student_hard_quantile_weight=0.25,
        )
    if variant_name == "edge_apex_delta_focus":
        return TrainingConfig(
            **common,
            epochs=22,
            lr=6.0e-5,
            patience=6,
            scheduler_kind="plateau",
            projected_student_any_boundary_loss_multiplier=1.15,
            projected_student_high_disp_loss_multiplier=1.20,
            projected_student_delta_gap_q90_loss_multiplier=1.35,
            projected_student_edge_apex_loss_multiplier=1.75,
            projected_student_branch_loss_weights={
                int(BRANCH_TO_ID["left_edge"]): 1.15,
                int(BRANCH_TO_ID["right_edge"]): 1.15,
            },
            projected_student_teacher_alignment_focus_multiplier=1.60,
            projected_student_hard_quantile=0.92,
            projected_student_hard_quantile_weight=0.30,
        )
    if variant_name == "full_combo_ema":
        return TrainingConfig(
            **common,
            epochs=26,
            lr=5.0e-5,
            patience=8,
            scheduler_kind="cosine",
            warmup_epochs=2,
            min_lr=5.0e-6,
            ema_decay=0.999,
            ema_eval=True,
            ema_start_epoch=1,
            projected_student_any_boundary_loss_multiplier=1.20,
            projected_student_high_disp_loss_multiplier=1.25,
            projected_student_teacher_gap_q90_loss_multiplier=1.35,
            projected_student_teacher_gap_q95_loss_multiplier=1.75,
            projected_student_delta_gap_q90_loss_multiplier=1.40,
            projected_student_edge_apex_loss_multiplier=1.75,
            projected_student_branch_loss_weights={
                int(BRANCH_TO_ID["left_edge"]): 1.15,
                int(BRANCH_TO_ID["right_edge"]): 1.15,
            },
            projected_student_teacher_alignment_focus_multiplier=1.75,
            projected_student_hard_quantile=0.92,
            projected_student_hard_quantile_weight=0.35,
        )
    raise ValueError(f"Unsupported phase3 variant {variant_name!r}.")


def _phase3_material_improvement(row: dict[str, Any], anchor_metrics: dict[str, float]) -> bool:
    return (
        float(anchor_metrics["hard_p95_principal"]) - float(row["val_hard_p95_principal"]) >= 5.0
        and float(anchor_metrics["broad_plastic_mae"]) - float(row["val_broad_plastic_mae"]) >= 0.5
        and float(anchor_metrics["hard_plastic_mae"]) - float(row["val_hard_plastic_mae"]) >= 0.3
        and float(row["val_yield_violation_p95"]) <= 1.0e-6
    )


def run_phase3_same_capacity_refinement(
    *,
    output_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp3_same_capacity_refinement"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase3_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase0 = _read_json(output_root / "exp0_anchor_freeze" / "phase0_summary.json")
    phase1 = _read_json(output_root / "exp1_teacher_gap_audit" / "phase1_summary.json")
    anchor_checkpoint = Path(phase0["anchor_checkpoint"])
    weighting_bands = phase1["weighting_bands"]
    base_dataset_path, base_dataset_summary = _phase3_base_dataset(
        phase_dir=phase_dir,
        phase1=phase1,
        force_rerun=force_rerun,
    )
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    cache_arrays, _cache_attrs = _load_h5(TEACHER_CACHE_PATH)
    val_disp_edges = np.asarray(phase0["split_summaries"]["val"]["disp_decile_edges"], dtype=float)
    test_disp_edges = np.asarray(phase0["split_summaries"]["test"]["disp_decile_edges"], dtype=float)
    anchor_val_metrics = phase0["split_summaries"]["val"]["metrics"]

    rows: list[dict[str, Any]] = []
    for idx, variant_name in enumerate(PHASE3_VARIANTS):
        run_dir = phase_dir / variant_name
        dataset_path = _phase3_variant_dataset(
            base_dataset_path=base_dataset_path,
            variant_name=variant_name,
            phase_dir=phase_dir,
            weighting_bands=weighting_bands,
            force_rerun=force_rerun,
        )
        config = _phase3_config(
            dataset_path=dataset_path,
            run_dir=run_dir,
            variant_name=variant_name,
            anchor_checkpoint=anchor_checkpoint,
            weighting_bands=weighting_bands,
            seed=20260340 + idx,
        )
        train_summary = _train_with_batch_fallback(config, force_rerun=force_rerun)
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
        _write_json(run_dir / "val_summary.json", val_summary)
        _write_json(run_dir / "test_summary.json", test_summary)
        _write_csv(run_dir / "val_slice_summary.csv", val_slice_rows, SLICE_SUMMARY_COLUMNS)
        _write_csv(run_dir / "test_slice_summary.csv", test_slice_rows, SLICE_SUMMARY_COLUMNS)
        _write_csv(run_dir / "val_call_concentration.csv", val_call_rows, CALL_SUMMARY_COLUMNS)
        _write_csv(run_dir / "test_call_concentration.csv", test_call_rows, CALL_SUMMARY_COLUMNS)
        rows.append(
            {
                "variant": variant_name,
                "run_dir": str(run_dir),
                "dataset_path": str(dataset_path),
                "best_checkpoint": str(best_checkpoint),
                "param_count": _parameter_count(best_checkpoint, device=device),
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
                "material_improvement": _phase3_material_improvement(
                    {
                        "val_broad_plastic_mae": val_metrics["broad_plastic_mae"],
                        "val_hard_plastic_mae": val_metrics["hard_plastic_mae"],
                        "val_hard_p95_principal": val_metrics["hard_p95_principal"],
                        "val_yield_violation_p95": val_metrics["yield_violation_p95"],
                    },
                    anchor_val_metrics,
                ),
                "meets_reopen_bar": _meets_bar(
                    {
                        "broad_plastic_mae": float(val_metrics["broad_plastic_mae"]),
                        "hard_plastic_mae": float(val_metrics["hard_plastic_mae"]),
                        "hard_p95_principal": float(val_metrics["hard_p95_principal"]),
                        "yield_violation_p95": float(val_metrics["yield_violation_p95"]),
                    },
                    REOPEN_BAR,
                ),
            }
        )

    _write_csv(
        phase_dir / "same_capacity_refinement_summary.csv",
        rows,
        [
            "variant",
            "run_dir",
            "dataset_path",
            "best_checkpoint",
            "param_count",
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
            "material_improvement",
            "meets_reopen_bar",
        ],
    )

    winner = min(rows, key=_ranking_key)
    best_val_metrics, best_val_rowwise, best_val_summary, best_val_slice_rows, best_val_call_rows = _evaluate_checkpoint_with_rowwise(
        checkpoint_path=Path(winner["best_checkpoint"]),
        split_name="val",
        arrays_all=arrays_all,
        panel_all=panel_all,
        cache_arrays=cache_arrays,
        disp_edges=val_disp_edges,
        device=device,
        eval_batch_size=eval_batch_size,
        prefix="best_phase3",
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
        prefix="best_phase3",
    )
    _write_h5(
        phase_dir / "best_phase3_val_rowwise.h5",
        best_val_rowwise,
        {"source_checkpoint": winner["best_checkpoint"], "disp_decile_edges": [float(x) for x in val_disp_edges]},
    )
    _write_h5(
        phase_dir / "best_phase3_test_rowwise.h5",
        best_test_rowwise,
        {"source_checkpoint": winner["best_checkpoint"], "disp_decile_edges": [float(x) for x in test_disp_edges]},
    )
    _write_json(phase_dir / "best_phase3_val_summary.json", {**best_val_summary, "metrics": best_val_metrics})
    _write_json(phase_dir / "best_phase3_test_summary.json", {**best_test_summary, "metrics": best_test_metrics})
    _write_csv(phase_dir / "best_phase3_val_slice_summary.csv", best_val_slice_rows, SLICE_SUMMARY_COLUMNS)
    _write_csv(phase_dir / "best_phase3_test_slice_summary.csv", best_test_slice_rows, SLICE_SUMMARY_COLUMNS)
    _write_csv(phase_dir / "best_phase3_val_call_concentration.csv", best_val_call_rows, CALL_SUMMARY_COLUMNS)
    _write_csv(phase_dir / "best_phase3_test_call_concentration.csv", best_test_call_rows, CALL_SUMMARY_COLUMNS)

    phase4_eligible = _phase3_material_improvement(
        {
            "val_broad_plastic_mae": winner["val_broad_plastic_mae"],
            "val_hard_plastic_mae": winner["val_hard_plastic_mae"],
            "val_hard_p95_principal": winner["val_hard_p95_principal"],
            "val_yield_violation_p95": winner["val_yield_violation_p95"],
        },
        anchor_val_metrics,
    ) and _meets_bar(
        {
            "broad_plastic_mae": float(winner["val_broad_plastic_mae"]),
            "hard_plastic_mae": float(winner["val_hard_plastic_mae"]),
            "hard_p95_principal": float(winner["val_hard_p95_principal"]),
            "yield_violation_p95": float(winner["val_yield_violation_p95"]),
        },
        PHASE4_TRIGGER_BAR,
    )
    summary = {
        "anchor_checkpoint": str(anchor_checkpoint),
        "base_dataset_summary": base_dataset_summary,
        "weighting_bands": weighting_bands,
        "rows": rows,
        "winner": winner,
        "phase4_eligible": phase4_eligible,
        "stop_rule_triggered": not any(bool(row["material_improvement"]) for row in rows),
    }
    _write_json(summary_path, summary)
    return summary


def run_phase4_full_real_distill_followon(
    *,
    output_root: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp4_full_real_distill_followon"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase4_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase3 = _read_json(output_root / "exp3_same_capacity_refinement" / "phase3_summary.json")
    if not phase3.get("phase4_eligible", False):
        summary = {
            "status": "skipped_not_justified",
            "reason": "Best Phase 3 run did not meet the narrow Phase 4 opening condition.",
            "source_phase3_winner": phase3["winner"],
        }
        _write_json(summary_path, summary)
        return summary

    summary = {
        "status": "skipped_no_bounded_followon_implemented",
        "reason": "Phase 4 trigger fired, but this packet did not open an additional full-real export path.",
        "source_phase3_winner": phase3["winner"],
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


def run_phase5_reopen_decision(
    *,
    output_root: Path,
    docs_root: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp5_reopen_decision"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase5_summary.json"
    docs_path = docs_root / "executions" / WORK_PACKET_BASENAME
    if summary_path.exists() and docs_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase0 = _read_json(output_root / "exp0_anchor_freeze" / "phase0_summary.json")
    phase1 = _read_json(output_root / "exp1_teacher_gap_audit" / "phase1_summary.json")
    phase2 = _read_json(output_root / "exp2_projected_teacher_ds_probe" / "phase2_summary.json")
    phase3 = _read_json(output_root / "exp3_same_capacity_refinement" / "phase3_summary.json")
    phase4 = _read_json(output_root / "exp4_full_real_distill_followon" / "phase4_summary.json")

    anchor = phase0["split_summaries"]["val"]["metrics"]
    winner = phase3["winner"]
    reopen_cleared = _meets_bar(
        {
            "broad_plastic_mae": float(winner["val_broad_plastic_mae"]),
            "hard_plastic_mae": float(winner["val_hard_plastic_mae"]),
            "hard_p95_principal": float(winner["val_hard_p95_principal"]),
            "yield_violation_p95": float(winner["val_yield_violation_p95"]),
        },
        REOPEN_BAR,
    )
    route_should_stop = bool(phase3["stop_rule_triggered"]) and not reopen_cleared

    lines = [
        "# Projection-Student Tail Reopen Work Packet",
        "",
        "Execution report for the March 27, 2026 projection-student tail reopen packet.",
        "",
        f"Paired task memo: [`docs/tasks/projection_student_tail_reopen_work_packet_20260327.md`]({ROOT / 'docs/tasks/projection_student_tail_reopen_work_packet_20260327.md'})",
        "",
        f"Repository: `{ROOT}`",
        f"Output root: `{output_root}`",
        "",
        "## Outcome",
        "",
        f"- reopen bar `<= 15 / <= 18 / <= 160 / <= 1e-6`: `{reopen_cleared}`",
        f"- frozen anchor validation broad / hard / p95 / yield: {_fmt_metrics(anchor)}",
        f"- best Phase 3 validation broad / hard / p95 / yield: `{winner['val_broad_plastic_mae']:.6f}` / `{winner['val_hard_plastic_mae']:.6f}` / `{winner['val_hard_p95_principal']:.6f}` / `{winner['val_yield_violation_p95']:.6e}`",
        f"- best Phase 3 test broad / hard / p95 / yield: `{winner['test_broad_plastic_mae']:.6f}` / `{winner['test_hard_plastic_mae']:.6f}` / `{winner['test_hard_p95_principal']:.6f}` / `{winner['test_yield_violation_p95']:.6e}`",
        "",
        "## Evidence",
        "",
        f"- anchor freeze summary: [`exp0_anchor_freeze/phase0_summary.json`]({output_root / 'exp0_anchor_freeze/phase0_summary.json'})",
        f"- teacher-gap weighting bands: [`exp1_teacher_gap_audit/teacher_gap_weighting_bands.json`]({output_root / 'exp1_teacher_gap_audit/teacher_gap_weighting_bands.json'})",
        f"- projected-teacher DS probe summary: [`exp2_projected_teacher_ds_probe/phase2_summary.json`]({output_root / 'exp2_projected_teacher_ds_probe/phase2_summary.json'})",
        f"- same-capacity refinement summary: [`exp3_same_capacity_refinement/phase3_summary.json`]({output_root / 'exp3_same_capacity_refinement/phase3_summary.json'})",
        f"- same-capacity refinement CSV: [`exp3_same_capacity_refinement/same_capacity_refinement_summary.csv`]({output_root / 'exp3_same_capacity_refinement/same_capacity_refinement_summary.csv'})",
        f"- best Phase 3 val rowwise: [`exp3_same_capacity_refinement/best_phase3_val_rowwise.h5`]({output_root / 'exp3_same_capacity_refinement/best_phase3_val_rowwise.h5'})",
        f"- best Phase 3 test rowwise: [`exp3_same_capacity_refinement/best_phase3_test_rowwise.h5`]({output_root / 'exp3_same_capacity_refinement/best_phase3_test_rowwise.h5'})",
        "",
        "Interpretation:",
        "",
        f"- same-capacity changes that helped most: `{winner['variant']}`",
        f"- stop rule triggered: `{phase3['stop_rule_triggered']}`",
        f"- Phase 4 status: `{phase4['status']}`",
        f"- compression remains closed: `{not reopen_cleared}`",
        f"- `DS` remains blocked: `{True}`",
        f"- route should continue narrowly: `{(not route_should_stop) and (not reopen_cleared)}`",
        f"- route should stop: `{route_should_stop}`",
    ]
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "reopen_bar_cleared": reopen_cleared,
        "winner": winner,
        "phase4": phase4,
        "stop_state": route_should_stop,
        "compression_closed": not reopen_cleared,
        "ds_blocked": True,
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    docs_root = ROOT / args.docs_root
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    if args.phase in {"all", "exp0_anchor_freeze"}:
        run_phase0_anchor_freeze(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp1_teacher_gap_audit"}:
        run_phase1_teacher_gap_audit(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp2_projected_teacher_ds_probe"}:
        run_phase2_projected_teacher_ds_probe(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp3_same_capacity_refinement"}:
        run_phase3_same_capacity_refinement(
            output_root=output_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp4_full_real_distill_followon"}:
        run_phase4_full_real_distill_followon(
            output_root=output_root,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp5_reopen_decision"}:
        run_phase5_reopen_decision(
            output_root=output_root,
            docs_root=docs_root,
            force_rerun=args.force_rerun,
        )


if __name__ == "__main__":
    main()
