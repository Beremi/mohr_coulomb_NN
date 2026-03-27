#!/usr/bin/env python
"""Run the March 24 plastic-surface replacement packet."""

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

from mc_surrogate.models import (
    build_trial_surface_features_f1,
    compute_trial_stress,
    compute_trial_surface_feature_stats,
    spectral_decomposition_from_strain,
)
from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    decode_grho_to_principal,
    encode_principal_to_grho,
    exact_trial_principal_stress_3d,
    principal_relative_error_3d,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.training import TrainingConfig, predict_with_checkpoint
from run_hybrid_campaign import _candidate_checkpoint_paths
from run_nn_replacement_abr_cycle import (
    PANEL_MASK_KEYS,
    _aggregate_policy_metrics,
    _branch_counts,
    _load_h5,
    _parameter_count,
    _percentile_summary,
    _plot_histograms,
    _quantile_or_zero,
    _slice_arrays,
    _train_with_batch_fallback,
    _write_csv,
    _write_h5,
    _write_json,
    _write_phase_report,
)

ABR_REFERENCE_VAL = {
    "broad_plastic_mae": 42.877327,
    "hard_plastic_mae": 47.608658,
    "hard_p95_principal": 383.443207,
    "yield_violation_p95": 0.0,
}
ABR_REFERENCE_TEST = {
    "broad_plastic_mae": 44.258335,
    "hard_plastic_mae": 48.720104,
    "hard_p95_principal": 391.887390,
    "yield_violation_p95": 0.0,
}
S1_GOAL = {
    "broad_plastic_mae": 15.0,
    "hard_plastic_mae": 18.0,
    "hard_p95_principal": 150.0,
    "yield_violation_p95": 1.0e-6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redesign-root", default="experiment_runs/real_sim/hybrid_gate_redesign_20260324")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/nn_replacement_surface_20260324")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _coord_scales_from_grho(grho: np.ndarray, material_reduced: np.ndarray) -> dict[str, float]:
    coords = np.asarray(grho, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar_safe = np.maximum(mat[:, 0], 1.0e-6)
    g_tilde = ((1.0 + mat[:, 1]) * coords[:, 0]) / c_bar_safe
    return {"scale_g": float(max(np.quantile(np.maximum(g_tilde, 0.0), 0.95), 1.0))}


def _corr_or_zero(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size < 2 or y_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1.0e-12 or np.std(y_arr) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _surface_metric_row(true_grho: np.ndarray, pred_grho: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return {
            "g_mae": float("nan"),
            "rho_mae": float("nan"),
            "g_corr": float("nan"),
            "rho_corr": float("nan"),
        }
    true_sel = np.asarray(true_grho[valid], dtype=float)
    pred_sel = np.asarray(pred_grho[valid], dtype=float)
    return {
        "g_mae": float(np.mean(np.abs(pred_sel[:, 0] - true_sel[:, 0]))),
        "rho_mae": float(np.mean(np.abs(pred_sel[:, 1] - true_sel[:, 1]))),
        "g_corr": _corr_or_zero(pred_sel[:, 0], true_sel[:, 0]),
        "rho_corr": _corr_or_zero(pred_sel[:, 1], true_sel[:, 1]),
    }


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


def build_stage0_surface_dataset(
    *,
    real_dataset_path: Path,
    panel_path: Path,
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    dataset_path = output_dir / "derived_surface_dataset.h5"
    summary_path = output_dir / "stage0_summary.json"
    if dataset_path.exists() and summary_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    real_arrays, real_attrs = _load_h5(real_dataset_path)
    panel_arrays, panel_attrs = _load_h5(panel_path)

    split_id = real_arrays["split_id"].astype(np.int8)
    branch_id = real_arrays["branch_id"].astype(np.int8)
    plastic = branch_id > 0
    train_plastic = (split_id == 0) & plastic

    trial_stress = compute_trial_stress(real_arrays["strain_eng"], real_arrays["material_reduced"]).astype(np.float32)
    trial_principal = exact_trial_principal_stress_3d(
        real_arrays["strain_eng"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
        shear=real_arrays["material_reduced"][:, 2],
        bulk=real_arrays["material_reduced"][:, 3],
        lame=real_arrays["material_reduced"][:, 4],
    ).astype(np.float32)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(real_arrays["strain_eng"])

    encoded = encode_principal_to_grho(
        real_arrays["stress_principal"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
    )
    feature_stats = compute_trial_surface_feature_stats(
        trial_principal[train_plastic],
        real_arrays["material_reduced"][train_plastic],
    )
    coordinate_scales = _coord_scales_from_grho(
        encoded["grho"][train_plastic],
        real_arrays["material_reduced"][train_plastic],
    )
    surface_feature_f1 = build_trial_surface_features_f1(
        strain_principal,
        real_arrays["material_reduced"],
        trial_principal,
        feature_stats,
    ).astype(np.float32)

    reconstruction_abs_max = np.zeros(real_arrays["stress"].shape[0], dtype=np.float32)
    surface_decode_yield = np.zeros(real_arrays["stress"].shape[0], dtype=np.float32)
    if np.any(plastic):
        decoded_plastic = decode_grho_to_principal(
            encoded["grho"][plastic],
            c_bar=real_arrays["material_reduced"][plastic, 0],
            sin_phi=real_arrays["material_reduced"][plastic, 1],
        ).astype(np.float32)
        reconstruction_abs = np.abs(decoded_plastic - real_arrays["stress_principal"][plastic].astype(np.float32))
        reconstruction_abs_max[plastic] = np.max(reconstruction_abs, axis=1)
        surface_decode_yield[plastic] = yield_violation_rel_principal_3d(
            decoded_plastic,
            c_bar=real_arrays["material_reduced"][plastic, 0],
            sin_phi=real_arrays["material_reduced"][plastic, 1],
        ).astype(np.float32)

    elastic = branch_id == 0
    elastic_identity = np.abs(trial_stress - real_arrays["stress"].astype(np.float32))
    elastic_identity_abs_max = np.max(elastic_identity, axis=1).astype(np.float32)

    derived_arrays = {
        "strain_eng": real_arrays["strain_eng"].astype(np.float32),
        "stress": real_arrays["stress"].astype(np.float32),
        "stress_principal": real_arrays["stress_principal"].astype(np.float32),
        "material_reduced": real_arrays["material_reduced"].astype(np.float32),
        "eigvecs": real_arrays["eigvecs"].astype(np.float32),
        "branch_id": branch_id,
        "split_id": split_id,
        "source_call_id": real_arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": real_arrays["source_row_in_call"].astype(np.int32),
        "trial_stress": trial_stress,
        "trial_principal": trial_principal,
        "strain_principal": strain_principal.astype(np.float32),
        "surface_feature_f1": surface_feature_f1,
        "grho": encoded["grho"].astype(np.float32),
        "g": encoded["g"].astype(np.float32),
        "rho": encoded["rho"].astype(np.float32),
        "lambda_coord": encoded["lambda_coord"].astype(np.float32),
        "surface_reconstruction_abs_max": reconstruction_abs_max,
        "surface_decode_yield_violation": surface_decode_yield,
        "elastic_identity_abs_max": elastic_identity_abs_max,
        "plastic_mask": panel_arrays["plastic_mask"].astype(np.int8),
        **{key: panel_arrays[key].astype(np.int8) for key in PANEL_MASK_KEYS},
    }
    attrs = {
        "source_real_dataset": str(real_dataset_path),
        "source_panel_path": str(panel_path),
        "surface_feature_stats_json": feature_stats,
        "surface_coordinate_scales_json": coordinate_scales,
        "split_seed": real_attrs.get("split_seed", 20260324),
        "branch_names_json": real_attrs.get("branch_names_json", json.dumps(BRANCH_NAMES)),
        "panel_summary_json": panel_attrs.get("panel_summary_json", ""),
    }
    _write_h5(dataset_path, derived_arrays, attrs)

    split_summaries: dict[str, Any] = {}
    for split_value, split_name in ((0, "train"), (1, "val"), (2, "test")):
        mask = split_id == split_value
        plastic_mask = mask & plastic
        split_summaries[split_name] = {
            "n_rows": int(np.sum(mask)),
            "plastic_rows": int(np.sum(plastic_mask)),
            "g": _percentile_summary(encoded["g"][plastic_mask]) if np.any(plastic_mask) else None,
            "rho": _percentile_summary(encoded["rho"][plastic_mask]) if np.any(plastic_mask) else None,
        }

    branch_summaries: dict[str, Any] = {}
    for branch_value, branch_name in enumerate(BRANCH_NAMES):
        mask = branch_id == branch_value
        branch_summaries[branch_name] = {
            "n_rows": int(np.sum(mask)),
            "g": _percentile_summary(encoded["g"][mask]) if np.any(mask) else None,
            "rho": _percentile_summary(encoded["rho"][mask]) if np.any(mask) else None,
        }

    _plot_histograms(
        output_dir / "surface_by_split.png",
        {
            "train_g": encoded["g"][(split_id == 0) & plastic],
            "val_g": encoded["g"][(split_id == 1) & plastic],
            "test_g": encoded["g"][(split_id == 2) & plastic],
            "train_rho": encoded["rho"][(split_id == 0) & plastic],
            "val_rho": encoded["rho"][(split_id == 1) & plastic],
            "test_rho": encoded["rho"][(split_id == 2) & plastic],
        },
        title="Plastic-surface coordinates by split",
    )
    _plot_histograms(
        output_dir / "surface_by_branch.png",
        {
            f"{name}_rho": encoded["rho"][branch_id == idx]
            for idx, name in enumerate(BRANCH_NAMES)
        },
        title="Plastic-surface rho by branch",
    )

    summary = {
        "dataset_path": str(dataset_path),
        "feature_stats": feature_stats,
        "coordinate_scales": coordinate_scales,
        "split_summaries": split_summaries,
        "branch_summaries": branch_summaries,
        "reconstruction": {
            "plastic_mean_abs": float(np.mean(reconstruction_abs_max[plastic])) if np.any(plastic) else 0.0,
            "plastic_max_abs": float(np.max(reconstruction_abs_max[plastic])) if np.any(plastic) else 0.0,
        },
        "surface_decode_yield": {
            "plastic_p95": _quantile_or_zero(surface_decode_yield[plastic], 0.95),
            "plastic_max": float(np.max(surface_decode_yield[plastic])) if np.any(plastic) else 0.0,
        },
        "elastic_identity": {
            "mean_abs": float(np.mean(elastic_identity[elastic])) if np.any(elastic) else 0.0,
            "max_abs": float(np.max(elastic_identity[elastic])) if np.any(elastic) else 0.0,
        },
    }
    _write_json(summary_path, summary)
    return summary


def evaluate_surface_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split_name: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    arrays, panel = _load_eval_split(dataset_path, split_name)
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
    )
    stress_pred = pred["stress"]
    stress_principal_pred = pred["stress_principal"]
    stress_true = arrays["stress"]
    stress_principal_true = arrays["stress_principal"]
    stress_abs = np.abs(stress_pred - stress_true)
    principal_abs = np.abs(stress_principal_pred - stress_principal_true)
    yield_rel = yield_violation_rel_principal_3d(
        stress_principal_pred,
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
    ).astype(np.float32)

    stats = {
        "stress_component_abs": stress_abs.astype(np.float32),
        "principal_max_abs": np.max(principal_abs, axis=1).astype(np.float32),
        "principal_rel_error": principal_relative_error_3d(
            stress_principal_pred,
            stress_principal_true,
            c_bar=arrays["material_reduced"][:, 0],
        ).astype(np.float32),
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
    coord_metrics = _surface_metric_row(arrays["grho"], pred["grho"], plastic)
    hard_coord_metrics = _surface_metric_row(arrays["grho"], pred["grho"], hard & plastic)
    metrics.update(coord_metrics)
    metrics.update(
        {
            "hard_g_mae": hard_coord_metrics["g_mae"],
            "hard_rho_mae": hard_coord_metrics["rho_mae"],
            "hard_g_corr": hard_coord_metrics["g_corr"],
            "hard_rho_corr": hard_coord_metrics["rho_corr"],
            "yield_violation_max": float(np.max(yield_rel[plastic])) if np.any(plastic) else 0.0,
        }
    )
    metrics["split"] = split_name
    return {
        "metrics": metrics,
        "predictions": pred,
    }


def write_stage0_report(report_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "## Frozen Split",
        "",
        "- source grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`",
        "- split seed: `20260324`",
        "- split fractions: `70 / 15 / 15`",
        "- Stage 1 training uses the full real train split with exact elastic dispatch upstream.",
        "",
        "## Representation",
        "",
        "- learned plastic coordinates: `g = a + b`, `rho = (a - b) / (a + b + eps)`",
        "- exact elastic rows are not learned; they are returned analytically from the trial state.",
        "",
        "## Exactness",
        "",
        f"- plastic decode mean / max abs: `{summary['reconstruction']['plastic_mean_abs']:.3e}` / `{summary['reconstruction']['plastic_max_abs']:.3e}`",
        f"- plastic decode yield p95 / max: `{summary['surface_decode_yield']['plastic_p95']:.3e}` / `{summary['surface_decode_yield']['plastic_max']:.3e}`",
        f"- elastic identity mean / max abs: `{summary['elastic_identity']['mean_abs']:.3e}` / `{summary['elastic_identity']['max_abs']:.3e}`",
        "- the small plastic decode mismatch is the expected projection of exact-solver surface tolerance onto `r = 0`, not an admissibility failure of the `g, rho` chart.",
        "",
        "## Derived Dataset",
        "",
        f"- derived dataset: `{summary['dataset_path']}`",
        f"- feature stats: `{json.dumps(summary['feature_stats'])}`",
        f"- coordinate scales: `{json.dumps(summary['coordinate_scales'])}`",
    ]
    _write_phase_report(report_path, "NN Replacement Surface Stage 0 20260324", lines)


def write_stage1_report(
    report_path: Path,
    *,
    stage0_summary: dict[str, Any],
    architecture_rows: list[dict[str, Any]],
    winner: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
) -> None:
    lines = [
        "## Training Setup",
        "",
        f"- dataset: `{stage0_summary['dataset_path']}`",
        f"- feature stats: `{json.dumps(stage0_summary['feature_stats'])}`",
        f"- coordinate scales: `{json.dumps(stage0_summary['coordinate_scales'])}`",
        f"- model kind: `trial_surface_acn_f1`",
        f"- exact elastic handling: `f_trial <= 0 -> exact elastic stress`",
        "",
        "## Architecture Sweep",
        "",
        "| Architecture | Param Count | Best Checkpoint | Broad Plastic MAE | Hard Plastic MAE | Hard p95 Principal | g MAE | rho MAE | Yield Violation p95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in architecture_rows:
        lines.append(
            f"| {row['architecture']} | {row['param_count']} | {row['selected_checkpoint_label']} | "
            f"{row['broad_plastic_mae']:.6f} | {row['hard_plastic_mae']:.6f} | {row['hard_p95_principal']:.6f} | "
            f"{row['g_mae']:.6f} | {row['rho_mae']:.6f} | {row['yield_violation_p95']:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Validation Winner",
            "",
            f"- architecture: `{winner['architecture']}`",
            f"- checkpoint: `{winner['selected_checkpoint_path']}`",
            f"- validation broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
            f"- validation hard p95 principal: `{winner['hard_p95_principal']:.6f}`",
            f"- validation g / rho MAE: `{winner['g_mae']:.6f}` / `{winner['rho_mae']:.6f}`",
            "",
            "## Comparison To First ABR Packet",
            "",
            f"- validation broad plastic MAE delta vs ABR: `{winner['broad_plastic_mae'] - ABR_REFERENCE_VAL['broad_plastic_mae']:.6f}`",
            f"- validation hard plastic MAE delta vs ABR: `{winner['hard_plastic_mae'] - ABR_REFERENCE_VAL['hard_plastic_mae']:.6f}`",
            f"- validation hard p95 principal delta vs ABR: `{winner['hard_p95_principal'] - ABR_REFERENCE_VAL['hard_p95_principal']:.6f}`",
            "",
            "## One-Shot Test",
            "",
            f"- broad plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}`",
            f"- hard plastic MAE: `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
            f"- hard p95 principal: `{winner_test['metrics']['hard_p95_principal']:.6f}`",
            f"- g / rho MAE: `{winner_test['metrics']['g_mae']:.6f}` / `{winner_test['metrics']['rho_mae']:.6f}`",
            f"- yield violation p95 / max: `{winner_test['metrics']['yield_violation_p95']:.6e}` / `{winner_test['metrics']['yield_violation_max']:.6e}`",
            "",
            "## Decision",
            "",
            f"- S1 gate (`<=15 / <=18 / <=150 / <=1e-6`) met: `{success}`",
            f"- continue to DS from this packet: `{success}`",
        ]
    )
    _write_phase_report(report_path, "NN Replacement Surface Stage 1 20260324", lines)


def write_execution_report(
    report_path: Path,
    *,
    stage0_summary: dict[str, Any],
    winner: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
) -> None:
    lines = [
        f"- Stage 0 dataset: `{stage0_summary['dataset_path']}`",
        f"- Representation: exact elastic dispatch + plastic-only `g, rho` surface decode",
        f"- Stage 1 winner: `{winner['architecture']}` at `{winner['selected_checkpoint_label']}`",
        f"- Validation broad / hard plastic MAE: `{winner['broad_plastic_mae']:.6f}` / `{winner['hard_plastic_mae']:.6f}`",
        f"- Validation hard p95 principal / yield p95: `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
        f"- Test broad / hard plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}` / `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
        f"- Test hard p95 principal / yield p95: `{winner_test['metrics']['hard_p95_principal']:.6f}` / `{winner_test['metrics']['yield_violation_p95']:.6e}`",
        f"- Improvement vs first ABR validation broad plastic MAE: `{ABR_REFERENCE_VAL['broad_plastic_mae'] - winner['broad_plastic_mae']:.6f}`",
        f"- Stage 1 go decision: `{success}`",
    ]
    _write_phase_report(report_path, "NN Replacement Surface Execution 20260324", lines)


def main() -> None:
    args = parse_args()
    redesign_root = (ROOT / args.redesign_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    real_dataset_path = redesign_root / "real_grouped_sampled_512.h5"
    panel_path = redesign_root / "panels" / "panel_sidecar.h5"

    stage0_dir = output_root / "exp0_surface_dataset"
    stage1_dir = output_root / "exp1_branchless_surface"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    stage0_summary = build_stage0_surface_dataset(
        real_dataset_path=real_dataset_path,
        panel_path=panel_path,
        output_dir=stage0_dir,
        force_rerun=args.force_rerun,
    )
    write_stage0_report(docs_root / "nn_replacement_surface_exp0_20260324.md", stage0_summary)

    if (
        stage0_summary["reconstruction"]["plastic_max_abs"] > 5.0e-4
        or stage0_summary["surface_decode_yield"]["plastic_max"] > 1.0e-6
        or stage0_summary["elastic_identity"]["max_abs"] > 1.0e-10
    ):
        raise RuntimeError("Stage 0 failed exactness stop rules; aborting before Stage 1.")

    architectures = [
        ("p1_w32_d2", 32, 2),
        ("p1_w64_d2", 64, 2),
        ("p1_w64_d3", 64, 3),
        ("p1_w96_d3", 96, 3),
        ("p1_w128_d3", 128, 3),
    ]
    architecture_rows: list[dict[str, Any]] = []
    for arch_name, width, depth in architectures:
        run_dir = stage1_dir / arch_name
        config = TrainingConfig(
            dataset=stage0_summary["dataset_path"],
            run_dir=str(run_dir),
            model_kind="trial_surface_acn_f1",
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
        )
        summary = _train_with_batch_fallback(config, force_rerun=args.force_rerun)
        checkpoint_rows: list[dict[str, Any]] = []
        for checkpoint_path in _candidate_checkpoint_paths(run_dir):
            eval_result = evaluate_surface_checkpoint(
                checkpoint_path,
                dataset_path=Path(stage0_summary["dataset_path"]),
                split_name="val",
                device=args.device,
                batch_size=args.eval_batch_size,
            )
            checkpoint_rows.append(
                {
                    "checkpoint_label": checkpoint_path.stem if checkpoint_path.parent.name == "snapshots" else checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                    **eval_result["metrics"],
                }
            )
        checkpoint_rows.sort(
            key=lambda row: (
                row["broad_plastic_mae"],
                row["hard_plastic_mae"],
                row["hard_p95_principal"],
                row["yield_violation_p95"],
                row["rho_mae"],
                row["g_mae"],
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
                "n_rows",
                "broad_mae",
                "hard_mae",
                "broad_plastic_mae",
                "hard_plastic_mae",
                "hard_p95_principal",
                "hard_rel_p95_principal",
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
                "plastic_coverage",
                "accepted_plastic_rows",
                "route_counts",
                "accepted_true_branch_counts",
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
                "coordinate_scales": stage0_summary["coordinate_scales"],
                **{
                    key: selected[key]
                    for key in (
                        "broad_plastic_mae",
                        "hard_plastic_mae",
                        "hard_p95_principal",
                        "hard_rel_p95_principal",
                        "yield_violation_p95",
                        "yield_violation_max",
                        "g_mae",
                        "rho_mae",
                        "g_corr",
                        "rho_corr",
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
            row["rho_mae"],
            row["g_mae"],
            row["selected_checkpoint_label"],
        )
    )
    winner = architecture_rows[0]
    winner_test = evaluate_surface_checkpoint(
        Path(winner["selected_checkpoint_path"]),
        dataset_path=Path(stage0_summary["dataset_path"]),
        split_name="test",
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    success = (
        winner["yield_violation_p95"] <= S1_GOAL["yield_violation_p95"]
        and winner["broad_plastic_mae"] <= S1_GOAL["broad_plastic_mae"]
        and winner["hard_plastic_mae"] <= S1_GOAL["hard_plastic_mae"]
        and winner["hard_p95_principal"] <= S1_GOAL["hard_p95_principal"]
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
            "coordinate_scales",
            "broad_plastic_mae",
            "hard_plastic_mae",
            "hard_p95_principal",
            "hard_rel_p95_principal",
            "yield_violation_p95",
            "yield_violation_max",
            "g_mae",
            "rho_mae",
            "g_corr",
            "rho_corr",
        ],
    )
    _write_json(stage1_dir / "checkpoint_selection.json", {"winner": winner, "all_architectures": architecture_rows})
    _write_json(stage1_dir / "winner_test_summary.json", winner_test)

    write_stage1_report(
        docs_root / "nn_replacement_surface_exp1_branchless_20260324.md",
        stage0_summary=stage0_summary,
        architecture_rows=architecture_rows,
        winner=winner,
        winner_test=winner_test,
        success=success,
    )
    write_execution_report(
        docs_root / "nn_replacement_surface_execution_20260324.md",
        stage0_summary=stage0_summary,
        winner=winner,
        winner_test=winner_test,
        success=success,
    )


if __name__ == "__main__":
    main()
