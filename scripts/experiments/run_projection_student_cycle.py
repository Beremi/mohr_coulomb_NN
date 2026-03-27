#!/usr/bin/env python
"""Run the March 26 projection-student program."""

from __future__ import annotations

import argparse
import csv
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

from mc_surrogate.models import (
    build_trial_principal_geom_features,
    compute_trial_stress,
    exact_trial_principal_from_strain,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
)
from mc_surrogate.mohr_coulomb import BRANCH_NAMES
from mc_surrogate.principal_projection import PROJECTION_CANDIDATE_NAMES, project_mc_principal_numpy
from mc_surrogate.training import TrainingConfig, predict_with_checkpoint
from mc_surrogate.voigt import stress_voigt_to_tensor
from run_nn_replacement_abr_cycle import (
    PANEL_MASK_KEYS,
    _aggregate_policy_metrics,
    _load_h5,
    _parameter_count,
    _pointwise_prediction_stats,
    _quantile_or_zero,
    _slice_arrays,
    _train_with_batch_fallback,
    _write_csv,
    _write_h5,
    _write_json,
    _write_phase_report,
)

SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
TEACHER_CHECKPOINT = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/best.pt"
TEACHER_SUMMARY = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/baseline/rb_staged_w512_d6_valfirst/summary.json"
GROUPED_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5"
PANEL_DATASET = ROOT / "experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5"

CANONICAL_RAW_METRICS = {
    "val": {
        "broad_plastic_mae": 5.771003,
        "hard_plastic_mae": 6.949571,
        "hard_p95_principal": 76.398964,
        "yield_violation_p95": 1.001326e-01,
    },
    "test": {
        "broad_plastic_mae": 5.981674,
        "hard_plastic_mae": 7.241915,
        "hard_p95_principal": 79.455505,
        "yield_violation_p95": 1.061293e-01,
    },
}

CONTINUE_BAR = {
    "broad_plastic_mae": 10.0,
    "hard_plastic_mae": 12.0,
    "hard_p95_principal": 120.0,
    "yield_violation_p95": 1.0e-6,
}
DS_BAR = {
    "broad_plastic_mae": 8.0,
    "hard_plastic_mae": 10.0,
    "hard_p95_principal": 100.0,
    "yield_violation_p95": 1.0e-6,
}

SIZE_BANDS = {
    "small": {"width": 48, "depth": 2},
    "medium": {"width": 80, "depth": 2},
    "large": {"width": 96, "depth": 2},
}

REFERENCE_FILES = {
    "packet2_val": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp1_branchless_surface/checkpoint_selection.json",
    "packet2_test": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp1_branchless_surface/winner_test_summary.json",
    "packet3_pred_val": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/winner_val_predicted_summary.json",
    "packet3_oracle_val": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/winner_val_oracle_summary.json",
    "packet3_test": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/winner_test_summary.json",
    "packet4_val": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_val_predicted_summary.json",
    "packet4_test": ROOT / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_test_summary.json",
    "programx_val": ROOT / "experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/oracle_val_summary.json",
    "programx_test": ROOT / "experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/oracle_test_summary.json",
}
WORK_PACKET_BASENAME = "projection_student_work_packet_20260326.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "exp0_phase0_audit", "exp1_student_dataset", "exp2_projected_student", "exp3_ds_probe"],
    )
    parser.add_argument("--output-root", default="experiment_runs/real_sim/projection_student_20260326")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_mask(split_id: np.ndarray, split_name: str) -> np.ndarray:
    return np.asarray(split_id == SPLIT_NAME_TO_ID[split_name], dtype=bool)


def _reference_metrics() -> dict[str, dict[str, float]]:
    packet2_val = _read_json(REFERENCE_FILES["packet2_val"])["winner"]
    return {
        "packet2_val": packet2_val,
        "packet2_test": _read_json(REFERENCE_FILES["packet2_test"])["metrics"],
        "packet3_pred_val": _read_json(REFERENCE_FILES["packet3_pred_val"])["metrics"],
        "packet3_oracle_val": _read_json(REFERENCE_FILES["packet3_oracle_val"])["metrics"],
        "packet3_test": _read_json(REFERENCE_FILES["packet3_test"])["metrics"],
        "packet4_val": _read_json(REFERENCE_FILES["packet4_val"])["metrics"],
        "packet4_test": _read_json(REFERENCE_FILES["packet4_test"])["metrics"],
        "programx_val": _read_json(REFERENCE_FILES["programx_val"])["metrics"],
        "programx_test": _read_json(REFERENCE_FILES["programx_test"])["metrics"],
    }


def _principal_and_eigvecs_from_stress(stress_voigt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stress_tensor = stress_voigt_to_tensor(stress_voigt)
    vals, vecs = np.linalg.eigh(stress_tensor)
    return vals[:, ::-1].astype(np.float32), vecs[:, :, ::-1].astype(np.float32)


def _meets_bar(metrics: dict[str, float], bar: dict[str, float]) -> bool:
    return all(float(metrics[key]) <= float(bar[key]) for key in bar)


def _history_is_nan_or_flat(history_csv: Path) -> tuple[bool, str]:
    if not history_csv.exists():
        return False, ""
    with history_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, ""
    val_loss = np.asarray([float(row["val_loss"]) for row in rows], dtype=float)
    train_loss = np.asarray([float(row["train_loss"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(val_loss)) or not np.all(np.isfinite(train_loss)):
        return True, "nan_loss"
    window = min(10, val_loss.shape[0])
    if window >= 10 and (np.max(val_loss[:window]) - np.min(val_loss[:window])) < 1.0e-4:
        return True, "flat_val"
    return False, ""


def _metric_row(label: str, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "label": label,
        "broad_plastic_mae": float(metrics["broad_plastic_mae"]),
        "hard_plastic_mae": float(metrics["hard_plastic_mae"]),
        "hard_p95_principal": float(metrics["hard_p95_principal"]),
        "hard_rel_p95_principal": float(metrics.get("hard_rel_p95_principal", float("nan"))),
        "yield_violation_p95": float(metrics["yield_violation_p95"]),
        "yield_violation_max": float(metrics.get("yield_violation_max", float("nan"))),
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


def _branchwise_projection_summary(
    arrays: dict[str, np.ndarray],
    *,
    projected_stress: np.ndarray,
    displacement_norm: np.ndarray,
    candidate_id: np.ndarray,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for branch_idx, branch_name in enumerate(BRANCH_NAMES[1:], start=1):
        mask = arrays["branch_id"] == branch_idx
        if not np.any(mask):
            out[branch_name] = {
                "n_rows": 0,
                "disp_mean": 0.0,
                "disp_p95": 0.0,
                "disp_max": 0.0,
                "post_mae": float("nan"),
                "candidate_counts": {name: 0 for name in PROJECTION_CANDIDATE_NAMES},
            }
            continue
        branch_disp = displacement_norm[mask]
        branch_candidate = candidate_id[mask]
        out[branch_name] = {
            "n_rows": int(np.sum(mask)),
            "disp_mean": float(np.mean(branch_disp)),
            "disp_p95": _quantile_or_zero(branch_disp, 0.95),
            "disp_max": float(np.max(branch_disp)),
            "post_mae": float(np.mean(np.abs(projected_stress[mask] - arrays["stress"][mask]))),
            "candidate_counts": {
                name: int(np.sum(branch_candidate == idx))
                for idx, name in enumerate(PROJECTION_CANDIDATE_NAMES)
            },
        }
    return out


def run_phase0_projection_audit(
    *,
    output_root: Path,
    docs_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp0_phase0_audit"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase0_summary.json"
    cache_path = phase_dir / "teacher_projection_cache.h5"
    if summary_path.exists() and cache_path.exists() and not force_rerun:
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

    trial_stress = compute_trial_stress(arrays_all["strain_eng"], arrays_all["material_reduced"]).astype(np.float32)
    trial_principal = exact_trial_principal_from_strain(arrays_all["strain_eng"], arrays_all["material_reduced"]).astype(np.float32)
    teacher_raw_stress = teacher_pred["stress"].astype(np.float32)
    teacher_raw_principal, teacher_raw_eigvecs = _principal_and_eigvecs_from_stress(teacher_raw_stress)

    plastic_mask = panel_all["plastic_mask"].astype(bool)
    elastic_mask = ~plastic_mask

    teacher_dispatch_stress = teacher_raw_stress.copy()
    teacher_dispatch_principal = teacher_raw_principal.copy()
    teacher_dispatch_stress[elastic_mask] = trial_stress[elastic_mask]
    teacher_dispatch_principal[elastic_mask] = trial_principal[elastic_mask]

    teacher_projected_stress = teacher_dispatch_stress.copy()
    teacher_projected_principal = teacher_dispatch_principal.copy()
    projection_disp_norm = np.zeros(teacher_raw_stress.shape[0], dtype=np.float32)
    projection_candidate_id = np.full(teacher_raw_stress.shape[0], -1, dtype=np.int8)

    if np.any(plastic_mask):
        projected_plastic, projection_details = project_mc_principal_numpy(
            teacher_raw_principal[plastic_mask],
            c_bar=arrays_all["material_reduced"][plastic_mask, 0],
            sin_phi=arrays_all["material_reduced"][plastic_mask, 1],
            mode="exact",
            return_details=True,
        )
        teacher_projected_principal[plastic_mask] = projected_plastic.astype(np.float32)
        teacher_projected_stress[plastic_mask] = stress_voigt_from_principal_numpy(
            projected_plastic,
            teacher_raw_eigvecs[plastic_mask],
        ).astype(np.float32)
        projection_disp_norm[plastic_mask] = projection_details.displacement_norm.astype(np.float32)
        projection_candidate_id[plastic_mask] = projection_details.selected_index.astype(np.int8)

    cache_arrays = {
        "split_id": arrays_all["split_id"].astype(np.int8),
        "branch_id": arrays_all["branch_id"].astype(np.int8),
        "teacher_raw_stress": teacher_dispatch_stress.astype(np.float32),
        "teacher_raw_stress_principal": teacher_dispatch_principal.astype(np.float32),
        "teacher_projected_stress": teacher_projected_stress.astype(np.float32),
        "teacher_projected_stress_principal": teacher_projected_principal.astype(np.float32),
        "teacher_projection_disp_norm": projection_disp_norm.astype(np.float32),
        "teacher_projection_candidate_id": projection_candidate_id.astype(np.int8),
    }
    cache_attrs = {
        "teacher_checkpoint": str(TEACHER_CHECKPOINT),
        "teacher_summary": str(TEACHER_SUMMARY),
        "source_grouped_dataset": str(GROUPED_DATASET),
        "source_panel_dataset": str(PANEL_DATASET),
        "projection_operator": (
            "Exact Euclidean projection of ordered principal stress onto the convex Mohr-Coulomb admissible set "
            "using pass-through, smooth-face, left-edge, right-edge, and apex candidates with deterministic tie-breaking."
        ),
        "dataset_attrs_json": dataset_attrs,
        "panel_attrs_json": panel_attrs,
    }
    _write_h5(cache_path, cache_arrays, cache_attrs)

    references = _reference_metrics()
    split_summaries: dict[str, Any] = {}
    comparison_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    phase0_success = True

    for split_name in ("val", "test"):
        mask = _split_mask(arrays_all["split_id"], split_name)
        split_arrays = _slice_arrays(arrays_all, mask)
        split_panel = _slice_arrays(panel_all, mask)

        raw_eval = _evaluate_policy(
            split_arrays,
            split_panel,
            stress_pred=teacher_dispatch_stress[mask],
            stress_principal_pred=teacher_dispatch_principal[mask],
        )
        projected_eval = _evaluate_policy(
            split_arrays,
            split_panel,
            stress_pred=teacher_projected_stress[mask],
            stress_principal_pred=teacher_projected_principal[mask],
        )
        split_plastic = split_panel["plastic_mask"].astype(bool)
        split_arrays_plastic = _slice_arrays(split_arrays, split_plastic)
        projected_metrics = projected_eval["metrics"]
        projected_metrics["projection_disp_mean"] = float(np.mean(projection_disp_norm[mask][split_plastic])) if np.any(split_plastic) else 0.0
        projected_metrics["projection_disp_p50"] = _quantile_or_zero(projection_disp_norm[mask][split_plastic], 0.50)
        projected_metrics["projection_disp_p95"] = _quantile_or_zero(projection_disp_norm[mask][split_plastic], 0.95)
        projected_metrics["projection_disp_max"] = float(np.max(projection_disp_norm[mask][split_plastic])) if np.any(split_plastic) else 0.0
        projected_metrics["projection_candidate_counts"] = {
            name: int(np.sum(projection_candidate_id[mask][split_plastic] == idx))
            for idx, name in enumerate(PROJECTION_CANDIDATE_NAMES)
        }
        branch_summary = _branchwise_projection_summary(
            split_arrays_plastic,
            projected_stress=teacher_projected_stress[mask][split_plastic],
            displacement_norm=projection_disp_norm[mask][split_plastic],
            candidate_id=projection_candidate_id[mask][split_plastic],
        )
        split_summaries[split_name] = {
            "raw_teacher": raw_eval["metrics"],
            "projected_teacher": projected_metrics,
            "branchwise_projection": branch_summary,
            "raw_canonical_delta": {
                key: float(raw_eval["metrics"][key] - CANONICAL_RAW_METRICS[split_name][key])
                for key in CANONICAL_RAW_METRICS[split_name]
            },
            "meets_continue_bar": _meets_bar(projected_metrics, CONTINUE_BAR),
        }
        phase0_success &= bool(split_summaries[split_name]["meets_continue_bar"])

        split_ref_rows = [
            _metric_row("teacher_raw", raw_eval["metrics"]),
            _metric_row("teacher_projected", projected_metrics),
        ]
        if split_name == "val":
            split_ref_rows.extend(
                [
                    _metric_row("packet2_deployed", references["packet2_val"]),
                    _metric_row("packet3_predicted", references["packet3_pred_val"]),
                    _metric_row("packet3_oracle", references["packet3_oracle_val"]),
                    _metric_row("packet4_deployed", references["packet4_val"]),
                    _metric_row("programx_oracle", references["programx_val"]),
                ]
            )
        else:
            split_ref_rows.extend(
                [
                    _metric_row("packet2_deployed", references["packet2_test"]),
                    _metric_row("packet3_structured", references["packet3_test"]),
                    _metric_row("packet4_deployed", references["packet4_test"]),
                    _metric_row("programx_oracle", references["programx_test"]),
                ]
            )
        comparison_rows_by_split[split_name] = split_ref_rows
        _write_csv(
            phase_dir / f"phase0_comparison_{split_name}.csv",
            split_ref_rows,
            ["label", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "yield_violation_max"],
        )

    summary = {
        "teacher_checkpoint": str(TEACHER_CHECKPOINT),
        "teacher_summary": str(TEACHER_SUMMARY),
        "cache_path": str(cache_path),
        "phase0_success": phase0_success,
        "projection_operator": cache_attrs["projection_operator"],
        "split_summaries": split_summaries,
        "comparison_csv": {
            split_name: str(phase_dir / f"phase0_comparison_{split_name}.csv")
            for split_name in ("val", "test")
        },
    }
    _write_json(summary_path, summary)

    val_proj = split_summaries["val"]["projected_teacher"]
    test_proj = split_summaries["test"]["projected_teacher"]
    val_raw = split_summaries["val"]["raw_teacher"]
    test_raw = split_summaries["test"]["raw_teacher"]
    report_lines = [
        "Teacher checkpoint",
        f"- checkpoint: `{TEACHER_CHECKPOINT}`",
        f"- summary: `{TEACHER_SUMMARY}`",
        "",
        "Projection operator",
        f"- {cache_attrs['projection_operator']}",
        "",
        "Phase 0 decision",
        f"- continue bar on validation: `{split_summaries['val']['meets_continue_bar']}`",
        f"- continue bar on test: `{split_summaries['test']['meets_continue_bar']}`",
        f"- overall Phase 0 status: `{'pass' if phase0_success else 'fail'}`",
        "",
        "Validation",
        f"- raw teacher broad / hard plastic MAE: `{val_raw['broad_plastic_mae']:.6f}` / `{val_raw['hard_plastic_mae']:.6f}`",
        f"- raw teacher hard p95 / yield p95: `{val_raw['hard_p95_principal']:.6f}` / `{val_raw['yield_violation_p95']:.6e}`",
        f"- projected teacher broad / hard plastic MAE: `{val_proj['broad_plastic_mae']:.6f}` / `{val_proj['hard_plastic_mae']:.6f}`",
        f"- projected teacher hard p95 / yield p95: `{val_proj['hard_p95_principal']:.6f}` / `{val_proj['yield_violation_p95']:.6e}`",
        f"- projection displacement p50 / p95: `{val_proj['projection_disp_p50']:.6f}` / `{val_proj['projection_disp_p95']:.6f}`",
        "",
        "Test",
        f"- raw teacher broad / hard plastic MAE: `{test_raw['broad_plastic_mae']:.6f}` / `{test_raw['hard_plastic_mae']:.6f}`",
        f"- raw teacher hard p95 / yield p95: `{test_raw['hard_p95_principal']:.6f}` / `{test_raw['yield_violation_p95']:.6e}`",
        f"- projected teacher broad / hard plastic MAE: `{test_proj['broad_plastic_mae']:.6f}` / `{test_proj['hard_plastic_mae']:.6f}`",
        f"- projected teacher hard p95 / yield p95: `{test_proj['hard_p95_principal']:.6f}` / `{test_proj['yield_violation_p95']:.6e}`",
        f"- projection displacement p50 / p95: `{test_proj['projection_disp_p50']:.6f}` / `{test_proj['projection_disp_p95']:.6f}`",
        "",
        "Comparison anchors",
        f"- packet2 validation broad / hard / p95: `{references['packet2_val']['broad_plastic_mae']:.6f}` / `{references['packet2_val']['hard_plastic_mae']:.6f}` / `{references['packet2_val']['hard_p95_principal']:.6f}`",
        f"- packet3 predicted validation broad / hard / p95: `{references['packet3_pred_val']['broad_plastic_mae']:.6f}` / `{references['packet3_pred_val']['hard_plastic_mae']:.6f}` / `{references['packet3_pred_val']['hard_p95_principal']:.6f}`",
        f"- packet3 oracle validation broad / hard / p95: `{references['packet3_oracle_val']['broad_plastic_mae']:.6f}` / `{references['packet3_oracle_val']['hard_plastic_mae']:.6f}` / `{references['packet3_oracle_val']['hard_p95_principal']:.6f}`",
        f"- packet4 validation broad / hard / p95: `{references['packet4_val']['broad_plastic_mae']:.6f}` / `{references['packet4_val']['hard_plastic_mae']:.6f}` / `{references['packet4_val']['hard_p95_principal']:.6f}`",
        "",
        "Artifacts",
        f"- cache H5: `{cache_path}`",
        f"- validation comparison CSV: `{phase_dir / 'phase0_comparison_val.csv'}`",
        f"- test comparison CSV: `{phase_dir / 'phase0_comparison_test.csv'}`",
    ]
    _write_phase_report(
        phase_dir / "phase0_report.md",
        "Projection-Student Phase 0 Audit",
        report_lines,
    )
    return summary


def run_phase1_student_dataset(
    *,
    output_root: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp1_student_dataset"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "dataset_summary.json"
    dataset_path = phase_dir / "projection_student_plastic_only.h5"
    if summary_path.exists() and dataset_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase0_cache_path = output_root / "exp0_phase0_audit" / "teacher_projection_cache.h5"
    if not phase0_cache_path.exists():
        raise FileNotFoundError(f"Phase 0 cache is required before Phase 1: {phase0_cache_path}")

    real_arrays, real_attrs = _load_h5(GROUPED_DATASET)
    panel_arrays, panel_attrs = _load_h5(PANEL_DATASET)
    phase0_cache, _cache_attrs = _load_h5(phase0_cache_path)

    plastic_mask = panel_arrays["plastic_mask"].astype(bool)
    trial_principal = exact_trial_principal_from_strain(real_arrays["strain_eng"], real_arrays["material_reduced"]).astype(np.float32)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(real_arrays["strain_eng"])
    geom_features = build_trial_principal_geom_features(
        strain_principal,
        real_arrays["material_reduced"],
        trial_principal,
    ).astype(np.float32)

    derived_arrays = {
        "split_id": real_arrays["split_id"][plastic_mask].astype(np.int8),
        "strain_eng": real_arrays["strain_eng"][plastic_mask].astype(np.float32),
        "stress": real_arrays["stress"][plastic_mask].astype(np.float32),
        "stress_principal": real_arrays["stress_principal"][plastic_mask].astype(np.float32),
        "material_reduced": real_arrays["material_reduced"][plastic_mask].astype(np.float32),
        "eigvecs": real_arrays["eigvecs"][plastic_mask].astype(np.float32),
        "branch_id": real_arrays["branch_id"][plastic_mask].astype(np.int8),
        "trial_principal": trial_principal[plastic_mask].astype(np.float32),
        "hard_mask": panel_arrays["hard_mask"][plastic_mask].astype(np.int8),
        "ds_valid_mask": panel_arrays["ds_valid_mask"][plastic_mask].astype(np.int8),
        "plastic_mask": np.ones(int(np.sum(plastic_mask)), dtype=np.int8),
        "teacher_stress_principal": phase0_cache["teacher_raw_stress_principal"][plastic_mask].astype(np.float32),
        "trial_principal_geom_feature_f1": geom_features[plastic_mask].astype(np.float32),
        "source_call_id": real_arrays["source_call_id"][plastic_mask].astype(np.int32),
        "source_row_in_call": real_arrays["source_row_in_call"][plastic_mask].astype(np.int32),
    }
    derived_attrs = {
        "source_grouped_dataset": str(GROUPED_DATASET),
        "source_panel_dataset": str(PANEL_DATASET),
        "source_phase0_cache": str(phase0_cache_path),
        "teacher_checkpoint": str(TEACHER_CHECKPOINT),
        "teacher_summary": str(TEACHER_SUMMARY),
        "real_dataset_attrs_json": real_attrs,
        "panel_attrs_json": panel_attrs,
    }
    _write_h5(dataset_path, derived_arrays, derived_attrs)

    split_id = derived_arrays["split_id"]
    branch_id = derived_arrays["branch_id"]
    summary = {
        "dataset_path": str(dataset_path),
        "n_rows": int(derived_arrays["stress"].shape[0]),
        "split_counts": {
            split_name: int(np.sum(split_id == split_value))
            for split_name, split_value in SPLIT_NAME_TO_ID.items()
        },
        "hard_counts": {
            split_name: int(np.sum((split_id == split_value) & (derived_arrays["hard_mask"] > 0)))
            for split_name, split_value in SPLIT_NAME_TO_ID.items()
        },
        "ds_valid_counts": {
            split_name: int(np.sum((split_id == split_value) & (derived_arrays["ds_valid_mask"] > 0)))
            for split_name, split_value in SPLIT_NAME_TO_ID.items()
        },
        "branch_counts": {
            BRANCH_NAMES[idx]: int(np.sum(branch_id == idx))
            for idx in range(len(BRANCH_NAMES))
        },
    }
    _write_json(summary_path, summary)
    return summary


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


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    gate_fail = 1.0 if float(row["val_yield_violation_p95"]) > 1.0e-6 else 0.0
    return (
        gate_fail,
        float(row["val_hard_p95_principal"]),
        float(row["val_hard_plastic_mae"]),
        float(row["val_broad_plastic_mae"]),
    )


def run_phase2_projected_student(
    *,
    output_root: Path,
    docs_root: Path,
    device: str,
    eval_batch_size: int,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp2_projected_student"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "phase2_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    dataset_path = output_root / "exp1_student_dataset" / "projection_student_plastic_only.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Phase 1 dataset is required before Phase 2: {dataset_path}")
    arrays_all, _dataset_attrs = _load_h5(GROUPED_DATASET)
    panel_all, _panel_attrs = _load_h5(PANEL_DATASET)
    phase0 = _read_json(output_root / "exp0_phase0_audit" / "phase0_summary.json")
    references = _reference_metrics()

    results: list[dict[str, Any]] = []
    preferred_mode = "exact"
    for size_name, arch in SIZE_BANDS.items():
        candidate_modes = [preferred_mode]
        if size_name == "small":
            candidate_modes = ["exact"]

        band_best: dict[str, Any] | None = None
        for projection_mode in candidate_modes:
            run_dir = phase_dir / f"{size_name}_{projection_mode}"
            config = TrainingConfig(
                dataset=str(dataset_path),
                run_dir=str(run_dir),
                model_kind="trial_principal_geom_projected_student",
                epochs=60,
                batch_size=4096,
                lr=3.0e-4,
                weight_decay=1.0e-5,
                width=arch["width"],
                depth=arch["depth"],
                dropout=0.0,
                seed=20260326,
                patience=12,
                grad_clip=1.0,
                branch_loss_weight=0.0,
                num_workers=0,
                device=device,
                scheduler_kind="plateau",
                plateau_factor=0.5,
                regression_loss_kind="huber",
                huber_delta=1.0,
                checkpoint_metric="loss",
                log_every_epochs=5,
                projection_mode=projection_mode,
                projection_tau=0.05,
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
                "size_band": size_name,
                "projection_mode": projection_mode,
                "width": arch["width"],
                "depth": arch["depth"],
                "param_count": _parameter_count(best_checkpoint, device=device),
                "run_dir": str(run_dir),
                "best_checkpoint": str(best_checkpoint),
                "completed_epochs": int(train_summary["completed_epochs"]),
                "best_epoch": int(train_summary["best_epoch"]),
                "val_broad_plastic_mae": float(val_eval["metrics"]["broad_plastic_mae"]),
                "val_hard_plastic_mae": float(val_eval["metrics"]["hard_plastic_mae"]),
                "val_hard_p95_principal": float(val_eval["metrics"]["hard_p95_principal"]),
                "val_hard_rel_p95_principal": float(val_eval["metrics"]["hard_rel_p95_principal"]),
                "val_yield_violation_p95": float(val_eval["metrics"]["yield_violation_p95"]),
                "test_broad_plastic_mae": float(test_eval["metrics"]["broad_plastic_mae"]),
                "test_hard_plastic_mae": float(test_eval["metrics"]["hard_plastic_mae"]),
                "test_hard_p95_principal": float(test_eval["metrics"]["hard_p95_principal"]),
                "test_hard_rel_p95_principal": float(test_eval["metrics"]["hard_rel_p95_principal"]),
                "test_yield_violation_p95": float(test_eval["metrics"]["yield_violation_p95"]),
            }
            _write_json(run_dir / "projected_student_eval_val.json", val_eval["metrics"])
            _write_json(run_dir / "projected_student_eval_test.json", test_eval["metrics"])
            results.append(row)
            if band_best is None or _ranking_key(row) < _ranking_key(band_best):
                band_best = row

        if size_name == "small":
            history_flag, history_reason = _history_is_nan_or_flat(Path(band_best["run_dir"]) / "history.csv")
            if history_flag:
                softmin_dir = phase_dir / "small_softmin"
                softmin_config = TrainingConfig(
                    dataset=str(dataset_path),
                    run_dir=str(softmin_dir),
                    model_kind="trial_principal_geom_projected_student",
                    epochs=60,
                    batch_size=4096,
                    lr=3.0e-4,
                    weight_decay=1.0e-5,
                    width=SIZE_BANDS["small"]["width"],
                    depth=SIZE_BANDS["small"]["depth"],
                    dropout=0.0,
                    seed=20260326,
                    patience=12,
                    grad_clip=1.0,
                    branch_loss_weight=0.0,
                    num_workers=0,
                    device=device,
                    scheduler_kind="plateau",
                    plateau_factor=0.5,
                    regression_loss_kind="huber",
                    huber_delta=1.0,
                    checkpoint_metric="loss",
                    log_every_epochs=5,
                    projection_mode="softmin",
                    projection_tau=0.05,
                )
                softmin_summary = _train_with_batch_fallback(softmin_config, force_rerun=force_rerun)
                best_checkpoint = Path(softmin_summary["best_checkpoint"])
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
                softmin_row = {
                    "size_band": "small",
                    "projection_mode": "softmin",
                    "width": SIZE_BANDS["small"]["width"],
                    "depth": SIZE_BANDS["small"]["depth"],
                    "param_count": _parameter_count(best_checkpoint, device=device),
                    "run_dir": str(softmin_dir),
                    "best_checkpoint": str(best_checkpoint),
                    "completed_epochs": int(softmin_summary["completed_epochs"]),
                    "best_epoch": int(softmin_summary["best_epoch"]),
                    "val_broad_plastic_mae": float(val_eval["metrics"]["broad_plastic_mae"]),
                    "val_hard_plastic_mae": float(val_eval["metrics"]["hard_plastic_mae"]),
                    "val_hard_p95_principal": float(val_eval["metrics"]["hard_p95_principal"]),
                    "val_hard_rel_p95_principal": float(val_eval["metrics"]["hard_rel_p95_principal"]),
                    "val_yield_violation_p95": float(val_eval["metrics"]["yield_violation_p95"]),
                    "test_broad_plastic_mae": float(test_eval["metrics"]["broad_plastic_mae"]),
                    "test_hard_plastic_mae": float(test_eval["metrics"]["hard_plastic_mae"]),
                    "test_hard_p95_principal": float(test_eval["metrics"]["hard_p95_principal"]),
                    "test_hard_rel_p95_principal": float(test_eval["metrics"]["hard_rel_p95_principal"]),
                    "test_yield_violation_p95": float(test_eval["metrics"]["yield_violation_p95"]),
                    "fallback_reason": history_reason,
                }
                results.append(softmin_row)
                preferred_mode = "softmin" if _ranking_key(softmin_row) < _ranking_key(band_best) else "exact"
            else:
                preferred_mode = band_best["projection_mode"]

    winner = min(results, key=_ranking_key)
    _write_csv(
        phase_dir / "projected_student_architecture_summary.csv",
        results,
        [
            "size_band",
            "projection_mode",
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
            "val_hard_rel_p95_principal",
            "val_yield_violation_p95",
            "test_broad_plastic_mae",
            "test_hard_plastic_mae",
            "test_hard_p95_principal",
            "test_hard_rel_p95_principal",
            "test_yield_violation_p95",
            "fallback_reason",
        ],
    )

    continuation = (
        winner["val_broad_plastic_mae"] < references["packet2_val"]["broad_plastic_mae"]
        and winner["val_hard_plastic_mae"] < references["packet2_val"]["hard_plastic_mae"]
        and winner["val_hard_p95_principal"] < references["packet2_val"]["hard_p95_principal"]
        and winner["val_yield_violation_p95"] <= 1.0e-6
    )
    ds_ready = _meets_bar(
        {
            "broad_plastic_mae": winner["val_broad_plastic_mae"],
            "hard_plastic_mae": winner["val_hard_plastic_mae"],
            "hard_p95_principal": winner["val_hard_p95_principal"],
            "yield_violation_p95": winner["val_yield_violation_p95"],
        },
        DS_BAR,
    )

    summary = {
        "phase0_success": bool(phase0["phase0_success"]),
        "preferred_projection_mode": preferred_mode,
        "winner": winner,
        "all_runs": results,
        "continue_projection_student": bool(continuation),
        "ds_probe_ready": bool(ds_ready),
    }
    _write_json(summary_path, summary)

    report_lines = [
        "Teacher and projection setup",
        f"- teacher checkpoint: `{TEACHER_CHECKPOINT}`",
        f"- projection operator: `{phase0['projection_operator']}`",
        f"- Phase 0 status: `{'pass' if phase0['phase0_success'] else 'fail'}`",
        "",
        "Winner",
        f"- size band / mode: `{winner['size_band']}` / `{winner['projection_mode']}`",
        f"- param count: `{winner['param_count']}`",
        f"- validation broad / hard plastic MAE: `{winner['val_broad_plastic_mae']:.6f}` / `{winner['val_hard_plastic_mae']:.6f}`",
        f"- validation hard p95 / yield p95: `{winner['val_hard_p95_principal']:.6f}` / `{winner['val_yield_violation_p95']:.6e}`",
        f"- test broad / hard plastic MAE: `{winner['test_broad_plastic_mae']:.6f}` / `{winner['test_hard_plastic_mae']:.6f}`",
        f"- test hard p95 / yield p95: `{winner['test_hard_p95_principal']:.6f}` / `{winner['test_yield_violation_p95']:.6e}`",
        "",
        "Decision",
        f"- materially beats packet2 on validation while preserving admissibility: `{continuation}`",
        f"- ready for DS probe bar on validation: `{ds_ready}`",
        f"- direct-replacement continuation state: `{'continue_projection_student' if continuation else 'stop_direct_replacement_again'}`",
        "",
        "Packet anchors",
        f"- packet2 validation broad / hard / p95: `{references['packet2_val']['broad_plastic_mae']:.6f}` / `{references['packet2_val']['hard_plastic_mae']:.6f}` / `{references['packet2_val']['hard_p95_principal']:.6f}`",
        f"- packet3 oracle validation broad / hard / p95: `{references['packet3_oracle_val']['broad_plastic_mae']:.6f}` / `{references['packet3_oracle_val']['hard_plastic_mae']:.6f}` / `{references['packet3_oracle_val']['hard_p95_principal']:.6f}`",
        f"- packet4 validation broad / hard / p95: `{references['packet4_val']['broad_plastic_mae']:.6f}` / `{references['packet4_val']['hard_plastic_mae']:.6f}` / `{references['packet4_val']['hard_p95_principal']:.6f}`",
        "",
        "Artifacts",
        f"- architecture summary CSV: `{phase_dir / 'projected_student_architecture_summary.csv'}`",
        f"- phase2 summary JSON: `{summary_path}`",
    ]
    execution_docs_dir = docs_root / "executions"
    execution_docs_dir.mkdir(parents=True, exist_ok=True)
    _write_phase_report(
        execution_docs_dir / WORK_PACKET_BASENAME,
        "Projection-Student Execution",
        report_lines,
    )
    return summary


def run_phase3_ds_probe(
    *,
    output_root: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    phase_dir = output_root / "exp3_ds_probe"
    phase_dir.mkdir(parents=True, exist_ok=True)
    summary_path = phase_dir / "ds_probe_summary.json"
    if summary_path.exists() and not force_rerun:
        return _read_json(summary_path)

    phase2_path = output_root / "exp2_projected_student" / "phase2_summary.json"
    if not phase2_path.exists():
        raise FileNotFoundError(f"Phase 2 summary is required before Phase 3: {phase2_path}")
    phase2 = _read_json(phase2_path)
    winner = phase2["winner"]
    ds_ready = phase2["ds_probe_ready"]
    if not ds_ready:
        summary = {
            "status": "skipped",
            "reason": "Projected student did not reach the DS-opening validation bar.",
            "winner": winner,
        }
        _write_json(summary_path, summary)
        return summary

    summary = {
        "status": "blocked",
        "reason": (
            "Phase 3 remains blocked in this pass: the projected-student training path is in place, "
            "but the repo still lacks a pure-torch feature-preparation path for full JVP-through-input execution."
        ),
        "winner": winner,
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    docs_root = ROOT / args.docs_root
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    if args.phase in {"all", "exp0_phase0_audit"}:
        run_phase0_projection_audit(
            output_root=output_root,
            docs_root=docs_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp1_student_dataset"}:
        run_phase1_student_dataset(
            output_root=output_root,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp2_projected_student"}:
        run_phase2_projected_student(
            output_root=output_root,
            docs_root=docs_root,
            device=args.device,
            eval_batch_size=args.eval_batch_size,
            force_rerun=args.force_rerun,
        )
    if args.phase in {"all", "exp3_ds_probe"}:
        run_phase3_ds_probe(
            output_root=output_root,
            force_rerun=args.force_rerun,
        )


if __name__ == "__main__":
    main()
