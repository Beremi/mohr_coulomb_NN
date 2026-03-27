#!/usr/bin/env python
"""Run the March 24 branch-structured plastic-surface redesign packet."""

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

from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    BRANCH_TO_ID,
    decode_branch_specialized_grho_to_principal,
    principal_relative_error_3d,
    project_grho_to_branch_specialized,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.training import TrainingConfig, predict_with_checkpoint
from run_hybrid_campaign import _candidate_checkpoint_paths
from run_nn_replacement_abr_cycle import (
    _aggregate_policy_metrics,
    _load_h5,
    _parameter_count,
    _quantile_or_zero,
    _slice_arrays,
    _train_with_batch_fallback,
    _write_csv,
    _write_json,
    _write_phase_report,
)
from run_nn_replacement_surface_cycle import S1_GOAL, _surface_metric_row

SURFACE_REFERENCE_VAL = {
    "broad_plastic_mae": 32.458569,
    "hard_plastic_mae": 37.824432,
    "hard_p95_principal": 328.641602,
    "yield_violation_p95": 6.173982e-08,
}
SURFACE_REFERENCE_TEST = {
    "broad_plastic_mae": 33.603760,
    "hard_plastic_mae": 38.797607,
    "hard_p95_principal": 331.668732,
    "yield_violation_p95": 5.991195e-08,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage0-dataset",
        default="experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp0_surface_dataset/derived_surface_dataset.h5",
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/nn_replacement_surface_20260324_packet3",
    )
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
    panel = {
        "plastic_mask": arrays["plastic_mask"],
        "hard_mask": arrays["hard_mask"],
    }
    return arrays, panel


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


def build_stage0_branch_structured_audit(
    *,
    stage0_dataset_path: Path,
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "stage0_summary.json"
    if summary_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    arrays, attrs = _load_h5(stage0_dataset_path)
    branch_id = arrays["branch_id"].astype(np.int8)
    plastic = branch_id > 0
    projected_grho = project_grho_to_branch_specialized(arrays["grho"], branch_id)
    decoded = decode_branch_specialized_grho_to_principal(
        projected_grho[plastic],
        branch_id[plastic],
        c_bar=arrays["material_reduced"][plastic, 0],
        sin_phi=arrays["material_reduced"][plastic, 1],
    ).astype(np.float32)
    reconstruction_abs = np.abs(decoded - arrays["stress_principal"][plastic].astype(np.float32))
    reconstruction_abs_max = np.max(reconstruction_abs, axis=1).astype(np.float32) if np.any(plastic) else np.zeros(0, dtype=np.float32)
    yield_rel = yield_violation_rel_principal_3d(
        decoded,
        c_bar=arrays["material_reduced"][plastic, 0],
        sin_phi=arrays["material_reduced"][plastic, 1],
    ).astype(np.float32) if np.any(plastic) else np.zeros(0, dtype=np.float32)

    branch_summaries: dict[str, Any] = {}
    for branch_value, branch_name in enumerate(BRANCH_NAMES):
        if branch_value == BRANCH_TO_ID["elastic"]:
            continue
        mask = branch_id == branch_value
        if not np.any(mask):
            branch_summaries[branch_name] = None
            continue
        decoded_branch = decode_branch_specialized_grho_to_principal(
            projected_grho[mask],
            branch_id[mask],
            c_bar=arrays["material_reduced"][mask, 0],
            sin_phi=arrays["material_reduced"][mask, 1],
        ).astype(np.float32)
        recon_branch = np.abs(decoded_branch - arrays["stress_principal"][mask].astype(np.float32))
        recon_branch_max = np.max(recon_branch, axis=1).astype(np.float32)
        yield_branch = yield_violation_rel_principal_3d(
            decoded_branch,
            c_bar=arrays["material_reduced"][mask, 0],
            sin_phi=arrays["material_reduced"][mask, 1],
        ).astype(np.float32)
        branch_summaries[branch_name] = {
            "count": int(np.sum(mask)),
            "reconstruction_mean_abs": float(np.mean(recon_branch_max)),
            "reconstruction_max_abs": float(np.max(recon_branch_max)),
            "yield_p95": _quantile_or_zero(yield_branch, 0.95),
            "yield_max": float(np.max(yield_branch)),
            "rho_fixed_value": (
                -1.0 if branch_name == "left_edge" else 1.0 if branch_name == "right_edge" else 0.0 if branch_name == "apex" else None
            ),
        }

    summary = {
        "source_stage0_dataset": str(stage0_dataset_path),
        "source_grouped_dataset": attrs.get("source_real_dataset"),
        "reconstruction": {
            "plastic_mean_abs": float(np.mean(reconstruction_abs_max)) if reconstruction_abs_max.size else 0.0,
            "plastic_max_abs": float(np.max(reconstruction_abs_max)) if reconstruction_abs_max.size else 0.0,
        },
        "yield": {
            "plastic_p95": _quantile_or_zero(yield_rel, 0.95),
            "plastic_max": float(np.max(yield_rel)) if yield_rel.size else 0.0,
        },
        "branch_summaries": branch_summaries,
        "stop_rules": {
            "plastic_reconstruction_max_le_5e4": bool((float(np.max(reconstruction_abs_max)) if reconstruction_abs_max.size else 0.0) <= 5.0e-4),
            "plastic_yield_max_le_1e6": bool((float(np.max(yield_rel)) if yield_rel.size else 0.0) <= 1.0e-6),
            "apex_reconstruction_max_le_1e6": bool(
                branch_summaries["apex"] is not None and branch_summaries["apex"]["reconstruction_max_abs"] <= 1.0e-6
            ),
            "apex_yield_max_le_1e6": bool(
                branch_summaries["apex"] is not None and branch_summaries["apex"]["yield_max"] <= 1.0e-6
            ),
        },
    }
    _write_json(summary_path, summary)
    return summary


def evaluate_branch_structured_surface_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split_name: str,
    device: str,
    batch_size: int,
    route_mode: str = "predicted",
) -> dict[str, Any]:
    arrays, panel = _load_eval_split(dataset_path, split_name)
    branch_override = arrays["branch_id"] if route_mode == "oracle" else None
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
        branch_override=branch_override,
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
    metrics.update(coord_metrics)
    metrics.update(
        {
            "hard_g_mae": hard_coord_metrics["g_mae"],
            "hard_rho_mae": hard_coord_metrics["rho_mae"],
            "hard_g_corr": hard_coord_metrics["g_corr"],
            "hard_rho_corr": hard_coord_metrics["rho_corr"],
            "yield_violation_max": float(np.max(yield_rel[plastic])) if np.any(plastic) else 0.0,
            "route_mode": route_mode,
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
    route_branch = pred.get("route_branch_id")
    if route_branch is not None and np.any(plastic):
        metrics["route_plastic_branch_counts"] = {
            BRANCH_NAMES[i]: int(np.sum(route_branch[plastic] == i))
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
    return {
        "metrics": metrics,
        "predictions": pred,
        "branch_rows": branch_rows,
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
    winner_val_oracle: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
    materially_improved: bool,
) -> None:
    lines = [
        "## Assignment",
        "",
        "- replacement brief used because `report8.md` was empty: one bounded branch-structured plastic redesign packet only",
        "- benchmark kept frozen at the March 24 validation-first split from the grouped `512`-samples-per-call dataset",
        "- exact elastic dispatch and analytic admissible decode were preserved",
        "",
        "## Stage 0 Audit",
        "",
        f"- source Stage 0 dataset: `{stage0_summary['source_stage0_dataset']}`",
        f"- branch-specialized plastic reconstruction mean / max abs: `{stage0_summary['reconstruction']['plastic_mean_abs']:.3e}` / `{stage0_summary['reconstruction']['plastic_max_abs']:.3e}`",
        f"- branch-specialized plastic yield p95 / max: `{stage0_summary['yield']['plastic_p95']:.3e}` / `{stage0_summary['yield']['plastic_max']:.3e}`",
        f"- apex reconstruction max abs: `{stage0_summary['branch_summaries']['apex']['reconstruction_max_abs']:.3e}`",
        f"- apex yield max: `{stage0_summary['branch_summaries']['apex']['yield_max']:.3e}`",
        "",
        "## Stage 1 Sweep",
        "",
        "- model kind: `trial_surface_branch_structured_f1`",
        "- shared trunk + learned plastic 4-way branch head over `smooth / left_edge / right_edge / apex`",
        "- smooth head predicts `(g, rho)`; left and right heads predict scalar `g`; apex decodes exactly from material with `g = 0`",
        "",
        "| Architecture | Param Count | Best Checkpoint | Pred Val Broad Plastic MAE | Pred Val Hard Plastic MAE | Pred Val Hard p95 | Oracle Val Broad Plastic MAE | Oracle Val Hard Plastic MAE | Oracle Val Hard p95 | Branch Acc | Yield p95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in architecture_rows:
        lines.append(
            f"| {row['architecture']} | {row['param_count']} | {row['selected_checkpoint_label']} | "
            f"{row['broad_plastic_mae']:.6f} | {row['hard_plastic_mae']:.6f} | {row['hard_p95_principal']:.6f} | "
            f"{row['oracle_broad_plastic_mae']:.6f} | {row['oracle_hard_plastic_mae']:.6f} | {row['oracle_hard_p95_principal']:.6f} | "
            f"{row['plastic_branch_accuracy']:.6f} | {row['yield_violation_p95']:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Validation Winner",
            "",
            f"- architecture: `{winner['architecture']}`",
            f"- checkpoint: `{winner['selected_checkpoint_path']}`",
            f"- predicted-route validation broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
            f"- predicted-route validation hard p95 principal / yield p95: `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
            f"- oracle-route validation broad / hard plastic MAE: `{winner['oracle_broad_plastic_mae']:.6f}` / `{winner['oracle_hard_plastic_mae']:.6f}`",
            f"- oracle-route validation hard p95 principal: `{winner['oracle_hard_p95_principal']:.6f}`",
            f"- plastic branch accuracy on validation: `{winner['plastic_branch_accuracy']:.6f}`",
            f"- predicted-route material improvement vs packet2 winner: `{materially_improved}`",
            "",
            "## Held-Out Test",
            "",
            f"- broad plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}`",
            f"- hard plastic MAE: `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
            f"- hard p95 principal: `{winner_test['metrics']['hard_p95_principal']:.6f}`",
            f"- g / rho MAE: `{winner_test['metrics']['g_mae']:.6f}` / `{winner_test['metrics']['rho_mae']:.6f}`",
            f"- yield violation p95 / max: `{winner_test['metrics']['yield_violation_p95']:.6e}` / `{winner_test['metrics']['yield_violation_max']:.6e}`",
            "",
            "## Decision",
            "",
            f"- internal Stage 1 gate (`<=15 / <=18 / <=150 / <=1e-6`) met: `{success}`",
            f"- predicted-route winner beats packet2 validation core metrics: `{materially_improved}`",
            f"- oracle beats packet2 while predicted does not: `{(winner['oracle_broad_plastic_mae'] < SURFACE_REFERENCE_VAL['broad_plastic_mae']) and (winner['oracle_hard_plastic_mae'] < SURFACE_REFERENCE_VAL['hard_plastic_mae']) and (winner['oracle_hard_p95_principal'] < SURFACE_REFERENCE_VAL['hard_p95_principal']) and (not materially_improved)}`",
            f"- continue to `DS` from this packet: `{success and materially_improved}`",
        ]
    )
    _write_phase_report(report_path, "NN Replacement Surface Packet3 Branch Structured 20260324", lines)


def write_execution_report(
    report_path: Path,
    *,
    stage0_summary: dict[str, Any],
    winner: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
    materially_improved: bool,
) -> None:
    lines = [
        f"- Stage 0 source dataset reused from packet2: `{stage0_summary['source_stage0_dataset']}`",
        "- Representation: exact elastic dispatch + learned plastic 4-way branch head + branch-specialized admissible decode",
        f"- Stage 1 winner: `{winner['architecture']}` at `{winner['selected_checkpoint_label']}`",
        f"- Validation predicted broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
        f"- Validation predicted hard p95 principal / yield p95: `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
        f"- Validation oracle broad / hard plastic MAE: `{winner['oracle_broad_plastic_mae']:.6f}` / `{winner['oracle_hard_plastic_mae']:.6f}`",
        f"- Validation plastic branch accuracy: `{winner['plastic_branch_accuracy']:.6f}`",
        f"- Test broad / hard plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}` / `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
        f"- Test hard p95 principal / yield p95: `{winner_test['metrics']['hard_p95_principal']:.6f}` / `{winner_test['metrics']['yield_violation_p95']:.6e}`",
        f"- Improvement vs packet2 validation broad plastic MAE: `{SURFACE_REFERENCE_VAL['broad_plastic_mae'] - winner['broad_plastic_mae']:.6f}`",
        f"- Materially improved vs packet2 validation core metrics: `{materially_improved}`",
        f"- Stage 1 go decision: `{success and materially_improved}`",
    ]
    _write_phase_report(report_path, "NN Replacement Surface Packet3 Execution 20260324", lines)


def main() -> None:
    args = parse_args()
    stage0_dataset_path = (ROOT / args.stage0_dataset).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    stage0_dir = output_root / "exp0_branch_specialized_audit"
    stage1_dir = output_root / "exp1_branch_structured_surface"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    stage0_summary = build_stage0_branch_structured_audit(
        stage0_dataset_path=stage0_dataset_path,
        output_dir=stage0_dir,
        force_rerun=args.force_rerun,
    )
    stop_rules = stage0_summary["stop_rules"]
    if not all(stop_rules.values()):
        raise RuntimeError(f"Stage 0 audit failed stop rules: {stop_rules}")

    architectures = [
        ("p3_w64_d2", 64, 2),
        ("p3_w64_d3", 64, 3),
        ("p3_w96_d3", 96, 3),
        ("p3_w128_d3", 128, 3),
    ]
    architecture_rows: list[dict[str, Any]] = []
    for arch_name, width, depth in architectures:
        run_dir = stage1_dir / arch_name
        config = TrainingConfig(
            dataset=str(stage0_dataset_path),
            run_dir=str(run_dir),
            model_kind="trial_surface_branch_structured_f1",
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
            branch_loss_weight=0.1,
        )
        summary = _train_with_batch_fallback(config, force_rerun=args.force_rerun)
        checkpoint_rows: list[dict[str, Any]] = []
        for checkpoint_path in _candidate_checkpoint_paths(run_dir):
            pred_eval = evaluate_branch_structured_surface_checkpoint(
                checkpoint_path,
                dataset_path=stage0_dataset_path,
                split_name="val",
                device=args.device,
                batch_size=args.eval_batch_size,
                route_mode="predicted",
            )
            oracle_eval = evaluate_branch_structured_surface_checkpoint(
                checkpoint_path,
                dataset_path=stage0_dataset_path,
                split_name="val",
                device=args.device,
                batch_size=args.eval_batch_size,
                route_mode="oracle",
            )
            pred_metrics = pred_eval["metrics"]
            oracle_metrics = oracle_eval["metrics"]
            checkpoint_rows.append(
                {
                    "checkpoint_label": checkpoint_path.stem if checkpoint_path.parent.name == "snapshots" else checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                    "split": pred_metrics["split"],
                    "route_mode": pred_metrics["route_mode"],
                    "n_rows": pred_metrics["n_rows"],
                    "broad_plastic_mae": pred_metrics["broad_plastic_mae"],
                    "hard_plastic_mae": pred_metrics["hard_plastic_mae"],
                    "hard_p95_principal": pred_metrics["hard_p95_principal"],
                    "yield_violation_p95": pred_metrics["yield_violation_p95"],
                    "yield_violation_max": pred_metrics["yield_violation_max"],
                    "g_mae": pred_metrics["g_mae"],
                    "rho_mae": pred_metrics["rho_mae"],
                    "g_corr": pred_metrics["g_corr"],
                    "rho_corr": pred_metrics["rho_corr"],
                    "hard_g_mae": pred_metrics["hard_g_mae"],
                    "hard_rho_mae": pred_metrics["hard_rho_mae"],
                    "hard_g_corr": pred_metrics["hard_g_corr"],
                    "hard_rho_corr": pred_metrics["hard_rho_corr"],
                    "plastic_branch_accuracy": pred_metrics.get("plastic_branch_accuracy", float("nan")),
                    "plastic_branch_confusion": pred_metrics.get("plastic_branch_confusion"),
                    "predicted_plastic_branch_counts": pred_metrics.get("predicted_plastic_branch_counts"),
                    "oracle_broad_plastic_mae": oracle_metrics["broad_plastic_mae"],
                    "oracle_hard_plastic_mae": oracle_metrics["hard_plastic_mae"],
                    "oracle_hard_p95_principal": oracle_metrics["hard_p95_principal"],
                    "oracle_yield_violation_p95": oracle_metrics["yield_violation_p95"],
                }
            )
        checkpoint_rows.sort(
            key=lambda row: (
                row["broad_plastic_mae"],
                row["hard_plastic_mae"],
                row["hard_p95_principal"],
                row["yield_violation_p95"],
                row["checkpoint_label"],
            )
        )
        selected = checkpoint_rows[0]
        _write_csv(
            run_dir / "checkpoint_sweep_val.csv",
            checkpoint_rows,
            [
                "checkpoint_label",
                "checkpoint_path",
                "split",
                "route_mode",
                "n_rows",
                "broad_plastic_mae",
                "hard_plastic_mae",
                "hard_p95_principal",
                "yield_violation_p95",
                "yield_violation_max",
                "g_mae",
                "rho_mae",
                "g_corr",
                "rho_corr",
                "hard_g_mae",
                "hard_rho_mae",
                "hard_g_corr",
                "hard_rho_corr",
                "plastic_branch_accuracy",
                "plastic_branch_confusion",
                "predicted_plastic_branch_counts",
                "oracle_broad_plastic_mae",
                "oracle_hard_plastic_mae",
                "oracle_hard_p95_principal",
                "oracle_yield_violation_p95",
            ],
        )
        architecture_rows.append(
            {
                "architecture": arch_name,
                "run_dir": str(run_dir),
                "param_count": _parameter_count(Path(selected["checkpoint_path"]), args.device),
                "selected_checkpoint_label": selected["checkpoint_label"],
                "selected_checkpoint_path": selected["checkpoint_path"],
                "completed_epochs": summary["completed_epochs"],
                "best_epoch_by_train_loop": summary["best_epoch"],
                "batch_size": config.batch_size,
                "width": width,
                "depth": depth,
                "feature_set": "surface_f1_real_only",
                "branch_coordinate_scheme": "branch_specialized",
                **{
                    key: selected[key]
                    for key in (
                        "broad_plastic_mae",
                        "hard_plastic_mae",
                        "hard_p95_principal",
                        "yield_violation_p95",
                        "yield_violation_max",
                        "g_mae",
                        "rho_mae",
                        "g_corr",
                        "rho_corr",
                        "hard_g_mae",
                        "hard_rho_mae",
                        "hard_g_corr",
                        "hard_rho_corr",
                        "plastic_branch_accuracy",
                        "plastic_branch_confusion",
                        "predicted_plastic_branch_counts",
                        "oracle_broad_plastic_mae",
                        "oracle_hard_plastic_mae",
                        "oracle_hard_p95_principal",
                        "oracle_yield_violation_p95",
                    )
                },
            }
        )

    architecture_rows.sort(
        key=lambda row: (
            row["broad_plastic_mae"],
            row["hard_plastic_mae"],
            row["hard_p95_principal"],
            row["yield_violation_p95"],
            row["selected_checkpoint_label"],
        )
    )
    winner = architecture_rows[0]
    winner_val_pred = evaluate_branch_structured_surface_checkpoint(
        Path(winner["selected_checkpoint_path"]),
        dataset_path=stage0_dataset_path,
        split_name="val",
        device=args.device,
        batch_size=args.eval_batch_size,
        route_mode="predicted",
    )
    winner_val_oracle = evaluate_branch_structured_surface_checkpoint(
        Path(winner["selected_checkpoint_path"]),
        dataset_path=stage0_dataset_path,
        split_name="val",
        device=args.device,
        batch_size=args.eval_batch_size,
        route_mode="oracle",
    )
    winner_test = evaluate_branch_structured_surface_checkpoint(
        Path(winner["selected_checkpoint_path"]),
        dataset_path=stage0_dataset_path,
        split_name="test",
        device=args.device,
        batch_size=args.eval_batch_size,
        route_mode="predicted",
    )

    success = (
        winner["yield_violation_p95"] <= S1_GOAL["yield_violation_p95"]
        and winner["broad_plastic_mae"] <= S1_GOAL["broad_plastic_mae"]
        and winner["hard_plastic_mae"] <= S1_GOAL["hard_plastic_mae"]
        and winner["hard_p95_principal"] <= S1_GOAL["hard_p95_principal"]
    )
    materially_improved = (
        winner["yield_violation_p95"] <= 1.0e-6
        and winner["broad_plastic_mae"] < SURFACE_REFERENCE_VAL["broad_plastic_mae"]
        and winner["hard_plastic_mae"] < SURFACE_REFERENCE_VAL["hard_plastic_mae"]
        and winner["hard_p95_principal"] < SURFACE_REFERENCE_VAL["hard_p95_principal"]
    )

    _write_csv(
        stage1_dir / "architecture_summary_val.csv",
        architecture_rows,
        [
            "architecture",
            "run_dir",
            "param_count",
            "selected_checkpoint_label",
            "selected_checkpoint_path",
            "completed_epochs",
            "best_epoch_by_train_loop",
            "batch_size",
            "width",
            "depth",
            "feature_set",
            "branch_coordinate_scheme",
            "broad_plastic_mae",
            "hard_plastic_mae",
            "hard_p95_principal",
            "yield_violation_p95",
            "yield_violation_max",
            "g_mae",
            "rho_mae",
            "g_corr",
            "rho_corr",
            "hard_g_mae",
            "hard_rho_mae",
            "hard_g_corr",
            "hard_rho_corr",
            "plastic_branch_accuracy",
            "plastic_branch_confusion",
            "predicted_plastic_branch_counts",
            "oracle_broad_plastic_mae",
            "oracle_hard_plastic_mae",
            "oracle_hard_p95_principal",
            "oracle_yield_violation_p95",
        ],
    )
    _write_json(
        stage1_dir / "checkpoint_selection.json",
        {
            "winner": winner,
            "all_architectures": architecture_rows,
            "surface_reference_val": SURFACE_REFERENCE_VAL,
            "surface_reference_test": SURFACE_REFERENCE_TEST,
            "s1_goal": S1_GOAL,
            "materially_improved": materially_improved,
        },
    )
    _write_json(stage1_dir / "winner_val_predicted_summary.json", winner_val_pred)
    _write_json(stage1_dir / "winner_val_oracle_summary.json", winner_val_oracle)
    _write_json(stage1_dir / "winner_test_summary.json", winner_test)
    _write_json(
        stage1_dir / "oracle_vs_predicted_val.json",
        {
            "predicted": winner_val_pred["metrics"],
            "oracle": winner_val_oracle["metrics"],
            "packet2_reference_val": SURFACE_REFERENCE_VAL,
        },
    )
    _write_branch_breakdown_csv(stage1_dir / "winner_val_branch_breakdown_predicted.csv", winner_val_pred["branch_rows"])
    _write_branch_breakdown_csv(stage1_dir / "winner_val_branch_breakdown_oracle.csv", winner_val_oracle["branch_rows"])
    _write_branch_breakdown_csv(stage1_dir / "winner_test_branch_breakdown.csv", winner_test["branch_rows"])

    write_packet_report(
        docs_root / "nn_replacement_surface_packet3_branch_structured_20260324.md",
        stage0_summary=stage0_summary,
        architecture_rows=architecture_rows,
        winner=winner,
        winner_val_pred=winner_val_pred,
        winner_val_oracle=winner_val_oracle,
        winner_test=winner_test,
        success=success,
        materially_improved=materially_improved,
    )
    write_execution_report(
        docs_root / "nn_replacement_surface_packet3_execution_20260324.md",
        stage0_summary=stage0_summary,
        winner=winner,
        winner_test=winner_test,
        success=success,
        materially_improved=materially_improved,
    )


if __name__ == "__main__":
    main()
