#!/usr/bin/env python
"""Run the validation-first hybrid gate redesign cycle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_hybrid_real_panels import build_hybrid_real_panels
from mc_surrogate.branch_geometry import compute_branch_geometry_principal
from mc_surrogate.data import SPLIT_TO_ID
from mc_surrogate.inference import apply_hybrid_gate, prepare_hybrid_gate_inputs
from mc_surrogate.models import Standardizer, build_model, compute_trial_stress, spectral_decomposition_from_strain
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, principal_relative_error_3d, yield_violation_rel_principal_3d
from mc_surrogate.training import (
    TrainingConfig,
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    predict_with_checkpoint,
    set_seed,
    train_model,
)
from mc_surrogate.voigt import stress_voigt_to_tensor
from run_cover_layer_single_material_plan import _plot_history_log
from run_hybrid_campaign import (
    _candidate_checkpoint_paths,
    _json_safe as _campaign_json_safe,
    _maybe_train,
    build_finetune_dataset,
    build_synthetic_pretrain_dataset,
)


DELTA_GRID = (0.0, 1.0e-4, 3.0e-4, 1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 7.0e-3, 1.0e-2, 1.5e-2, 2.0e-2, 3.0e-2, 5.0e-2, 8.0e-2)
ENTROPY_GRID = (0.15, 0.25, 0.35, 0.45, 0.55, 0.70)
TAU_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
D_BANDS = (
    (0.0, 0.002),
    (0.002, 0.005),
    (0.005, 0.01),
    (0.01, 0.02),
    (0.02, 0.05),
    (0.05, math.inf),
)


@dataclass(frozen=True)
class AdamStage:
    name: str
    batch_size: int
    lr_start: float
    lr_end: float
    epochs: int
    lbfgs_steps_after: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", default="constitutive_problem_3D_full.h5")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/hybrid_gate_redesign_20260324")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-call", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--baseline-width", type=int, default=512)
    parser.add_argument("--baseline-depth", type=int, default=6)
    parser.add_argument("--baseline-seed", type=int, default=20260324)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--pretrain-train-rows", type=int, default=131072)
    parser.add_argument("--pretrain-val-rows", type=int, default=16384)
    parser.add_argument("--pretrain-test-rows", type=int, default=16384)
    parser.add_argument("--finetune-train-rows", type=int, default=131072)
    parser.add_argument("--finetune-real-fraction", type=float, default=0.80)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--candidate-width", type=int, default=256)
    parser.add_argument("--candidate-depth", type=int, default=4)
    parser.add_argument("--candidate-train-batch-size", type=int, default=4096)
    parser.add_argument("--rejector-epochs", type=int, default=40)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    return _campaign_json_safe(value)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


def _load_h5(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _split_mask(arrays: dict[str, np.ndarray], split_name: str) -> np.ndarray:
    return arrays["split_id"] == SPLIT_TO_ID[split_name]


def _slice_dict(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: value[mask] if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0] else value
        for key, value in data.items()
    }


def _quantile_or_zero(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _pointwise_prediction_stats(
    arrays: dict[str, np.ndarray],
    *,
    stress_pred: np.ndarray,
    stress_principal_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    stress_true = arrays["stress"]
    principal_true = arrays["stress_principal"]
    material = arrays["material_reduced"]
    stress_abs = np.abs(stress_pred - stress_true)
    principal_abs = np.abs(stress_principal_pred - principal_true)
    return {
        "stress_component_abs": stress_abs.astype(np.float32),
        "stress_sample_mae": np.mean(stress_abs, axis=1).astype(np.float32),
        "stress_sample_rmse": np.sqrt(np.mean((stress_pred - stress_true) ** 2, axis=1)).astype(np.float32),
        "principal_max_abs": np.max(principal_abs, axis=1).astype(np.float32),
        "principal_rel_error": principal_relative_error_3d(
            stress_principal_pred,
            principal_true,
            c_bar=material[:, 0],
        ).astype(np.float32),
        "yield_violation_rel": yield_violation_rel_principal_3d(
            stress_principal_pred,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
        ).astype(np.float32),
    }


def _branch_count_dict(branch_id: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    counts = np.bincount(branch_id[mask].astype(np.int64), minlength=len(BRANCH_NAMES))
    return {name: int(counts[idx]) for idx, name in enumerate(BRANCH_NAMES)}


def _aggregate_policy_metrics(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    *,
    learned_mask: np.ndarray | None = None,
    fallback_mask: np.ndarray | None = None,
    elastic_mask: np.ndarray | None = None,
    accepted_yield_violation_rel: np.ndarray | None = None,
) -> dict[str, Any]:
    plastic = panel["plastic_mask"].astype(bool)
    hard = panel["hard_mask"].astype(bool)
    hard_plastic = hard & plastic
    if learned_mask is None:
        learned_mask = plastic.copy()
    if fallback_mask is None:
        fallback_mask = np.zeros_like(plastic, dtype=bool)
    if elastic_mask is None:
        elastic_mask = np.zeros_like(plastic, dtype=bool)
    if accepted_yield_violation_rel is None:
        accepted_yield_violation_rel = stats["yield_violation_rel"]

    learned_plastic = learned_mask & plastic
    coverage = float(np.mean(learned_mask[plastic])) if np.any(plastic) else 0.0
    accepted_yield = accepted_yield_violation_rel[learned_plastic]
    if accepted_yield.size == 0:
        accepted_yield = np.zeros((0,), dtype=np.float32)

    return {
        "n_rows": int(arrays["stress"].shape[0]),
        "broad_mae": float(np.mean(stats["stress_component_abs"])),
        "hard_mae": float(np.mean(stats["stress_component_abs"][hard])) if np.any(hard) else float("nan"),
        "broad_plastic_mae": float(np.mean(stats["stress_component_abs"][plastic])) if np.any(plastic) else float("nan"),
        "hard_plastic_mae": float(np.mean(stats["stress_component_abs"][hard_plastic])) if np.any(hard_plastic) else float("nan"),
        "hard_p95_principal": _quantile_or_zero(stats["principal_max_abs"][hard], 0.95),
        "hard_rel_p95_principal": _quantile_or_zero(stats["principal_rel_error"][hard], 0.95),
        "yield_violation_p95": _quantile_or_zero(accepted_yield, 0.95),
        "plastic_coverage": coverage,
        "accepted_plastic_rows": int(np.sum(learned_plastic)),
        "route_counts": {
            "elastic": int(np.sum(elastic_mask)),
            "fallback": int(np.sum(fallback_mask)),
            "learned": int(np.sum(learned_mask)),
        },
        "accepted_true_branch_counts": _branch_count_dict(arrays["branch_id"], learned_plastic) if np.any(learned_plastic) else {name: 0 for name in BRANCH_NAMES},
    }


def _coverage_under_risk_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    metrics = row["metrics"]
    return (
        -metrics["plastic_coverage"],
        metrics["hard_p95_principal"],
        metrics["hard_plastic_mae"],
        metrics["broad_plastic_mae"],
        float(row.get("delta_geom", 0.0)),
        float(row.get("entropy_threshold", row.get("tau", 0.0))),
    )


def _risk_excess_key(row: dict[str, Any], ceilings: dict[str, float]) -> tuple[float, float, float, float, float]:
    metrics = row["metrics"]
    excess = 0.0
    for key, ceiling_key in (
        ("broad_plastic_mae", "broad_plastic_mae_ceiling"),
        ("hard_plastic_mae", "hard_plastic_mae_ceiling"),
        ("hard_p95_principal", "hard_p95_principal_ceiling"),
        ("yield_violation_p95", "yield_violation_p95_ceiling"),
    ):
        ceiling = max(abs(ceilings[ceiling_key]), 1.0e-12)
        excess += max(metrics[key] - ceilings[ceiling_key], 0.0) / ceiling
    return (
        excess,
        -metrics["plastic_coverage"],
        metrics["hard_p95_principal"],
        metrics["hard_plastic_mae"],
        metrics["broad_plastic_mae"],
    )


def _is_feasible(metrics: dict[str, Any], ceilings: dict[str, float], *, tol: float = 1.0e-12) -> bool:
    return (
        metrics["broad_plastic_mae"] <= ceilings["broad_plastic_mae_ceiling"] + tol
        and metrics["hard_plastic_mae"] <= ceilings["hard_plastic_mae_ceiling"] + tol
        and metrics["hard_p95_principal"] <= ceilings["hard_p95_principal_ceiling"] + tol
        and metrics["yield_violation_p95"] <= ceilings["yield_violation_p95_ceiling"] + tol
    )


def _select_frontier_row(rows: list[dict[str, Any]], ceilings: dict[str, float]) -> dict[str, Any]:
    feasible = [row for row in rows if _is_feasible(row["metrics"], ceilings)]
    if feasible:
        best = min(feasible, key=_coverage_under_risk_sort_key)
        return {
            "status": "feasible",
            "best_row": best,
            "feasible_rows": len(feasible),
        }
    best = min(rows, key=lambda row: _risk_excess_key(row, ceilings))
    return {
        "status": "no_feasible_row",
        "best_row": best,
        "feasible_rows": 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _plot_frontier(path: Path, rows: list[dict[str, Any]], *, title: str, color_key: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage = np.array([row["metrics"]["plastic_coverage"] for row in rows], dtype=float)
    hard_p95 = np.array([row["metrics"]["hard_p95_principal"] for row in rows], dtype=float)
    hard_mae = np.array([row["metrics"]["hard_plastic_mae"] for row in rows], dtype=float)
    broad_mae = np.array([row["metrics"]["broad_plastic_mae"] for row in rows], dtype=float)
    color = np.array([float(row.get(color_key, 0.0)) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter = axes[0].scatter(coverage, hard_p95, c=color, cmap="viridis", s=30)
    axes[0].set_xlabel("learned plastic coverage")
    axes[0].set_ylabel("hard p95 principal")
    axes[0].set_title("Coverage vs hard tail")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axes[0], label=color_key)

    axes[1].scatter(coverage, hard_mae, c=color, cmap="viridis", s=30)
    axes[1].set_xlabel("learned plastic coverage")
    axes[1].set_ylabel("hard plastic MAE")
    axes[1].set_title("Coverage vs hard MAE")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(coverage, broad_mae, c=color, cmap="viridis", s=30)
    axes[2].set_xlabel("learned plastic coverage")
    axes[2].set_ylabel("broad plastic MAE")
    axes[2].set_title("Coverage vs broad MAE")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _stage_schedule() -> list[AdamStage]:
    return [
        AdamStage("adam_bs1024", batch_size=1024, lr_start=1.0e-3, lr_end=3.0e-4, epochs=140, lbfgs_steps_after=2),
        AdamStage("adam_bs2048", batch_size=2048, lr_start=3.0e-4, lr_end=1.0e-4, epochs=120, lbfgs_steps_after=2),
        AdamStage("adam_bs4096", batch_size=4096, lr_start=1.0e-4, lr_end=3.0e-5, epochs=120, lbfgs_steps_after=3),
        AdamStage("adam_bs6144", batch_size=6144, lr_start=3.0e-5, lr_end=1.0e-6, epochs=160, lbfgs_steps_after=4),
    ]


def _plot_staged_history(history_csv: Path, output_path: Path) -> Path:
    rows = []
    with history_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No history rows found in {history_csv}.")

    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    train_loss = np.array([float(r["train_loss"]) for r in rows], dtype=float)
    val_loss = np.array([float(r["val_loss"]) for r in rows], dtype=float)
    train_stress = np.array([float(r["train_stress_mse"]) for r in rows], dtype=float)
    val_stress = np.array([float(r["val_stress_mse"]) for r in rows], dtype=float)
    stage_names = [r["stage_name"] for r in rows]

    boundaries: list[tuple[int, str]] = []
    prev = None
    for e, s in zip(epoch, stage_names):
        if s != prev:
            boundaries.append((int(e), s))
            prev = s

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(epoch, train_loss, label="train loss")
    axes[0].plot(epoch, val_loss, label="val loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch, train_stress, label="train stress mse")
    axes[1].plot(epoch, val_stress, label="val stress mse")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("stress mse")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    for ax in axes:
        for x, _label in boundaries:
            ax.axvline(x, color="k", linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_rejector_history(history_csv: Path, output_path: Path) -> Path:
    rows = []
    with history_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    train_loss = np.array([float(r["train_loss"]) for r in rows], dtype=float)
    val_loss = np.array([float(r["val_loss"]) for r in rows], dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch, train_loss, label="train")
    ax.plot(epoch, val_loss, label="val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Rejector training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def train_validation_first_staged_baseline(
    *,
    dataset_path: Path,
    run_dir: Path,
    device: str,
    width: int,
    depth: int,
    seed: int,
    schedule: list[AdamStage] | None = None,
    min_delta: float = 1.0,
    stop_patience: int = 140,
    force_rerun: bool = False,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_ckpt = run_dir / "best.pt"
    if summary_path.exists() and best_ckpt.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    set_seed(seed)
    device_obj = choose_device(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_schedule = schedule or _stage_schedule()

    train_arrays = _load_split_for_training(str(dataset_path), "train", "raw_branch")
    val_arrays = _load_split_for_training(str(dataset_path), "val", "raw_branch")

    x_scaler = Standardizer.from_array(train_arrays["features"])
    y_scaler = Standardizer.from_array(train_arrays["target"])
    train_ds = _build_tensor_dataset(train_arrays, x_scaler, y_scaler)
    val_ds = _build_tensor_dataset(val_arrays, x_scaler, y_scaler)

    model = build_model("raw_branch", input_dim=train_arrays["features"].shape[1], width=width, depth=depth, dropout=0.0).to(device_obj)
    metadata = {
        "config": {
            "dataset": str(dataset_path),
            "run_dir": str(run_dir),
            "model_kind": "raw_branch",
            "width": width,
            "depth": depth,
            "dropout": 0.0,
            "seed": seed,
            "selection_metric": "val_stress_mse",
            "schedule": [asdict(stage) for stage in stage_schedule],
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
        "branch_head_kind": "full",
        "elastic_handling": "model_decoded",
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "stage_name",
                "stage_kind",
                "batch_size",
                "lr",
                "train_loss",
                "val_loss",
                "train_stress_mse",
                "val_stress_mse",
                "train_branch_accuracy",
                "val_branch_accuracy",
                "best_val_stress_mse",
                "is_best",
            ]
        )

    last_ckpt = run_dir / "last.pt"
    best_val_stress_mse = float("inf")
    best_epoch = 0
    best_stage = ""
    global_epoch = 0
    stagnant_steps = 0

    for stage in stage_schedule:
        batch_size = stage.batch_size
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage.lr_start, weight_decay=2.0e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage.epochs, 1), eta_min=stage.lr_end)

        for _ in range(stage.epochs):
            global_epoch += 1
            train_metrics = _epoch_loop(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                model_kind="raw_branch",
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                branch_loss_weight=0.1,
                device=device_obj,
                grad_clip=1.0,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
                regression_loss_kind="mse",
                huber_delta=1.0,
                voigt_mae_weight=0.0,
                tangent_loss_weight=0.0,
                tangent_fd_scale=1.0e-6,
            )
            val_metrics = _epoch_loop(
                model=model,
                loader=val_loader,
                optimizer=None,
                model_kind="raw_branch",
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                branch_loss_weight=0.1,
                device=device_obj,
                grad_clip=1.0,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
                regression_loss_kind="mse",
                huber_delta=1.0,
                voigt_mae_weight=0.0,
                tangent_loss_weight=0.0,
                tangent_fd_scale=1.0e-6,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            improved = val_metrics["stress_mse"] < (best_val_stress_mse - min_delta)
            if improved:
                best_val_stress_mse = val_metrics["stress_mse"]
                best_epoch = global_epoch
                best_stage = stage.name
                stagnant_steps = 0
                torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, best_ckpt)
            else:
                stagnant_steps += 1
            torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, last_ckpt)
            with history_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        global_epoch,
                        stage.name,
                        "adam",
                        batch_size,
                        current_lr,
                        train_metrics["loss"],
                        val_metrics["loss"],
                        train_metrics["stress_mse"],
                        val_metrics["stress_mse"],
                        train_metrics["branch_accuracy"],
                        val_metrics["branch_accuracy"],
                        best_val_stress_mse,
                        1 if improved else 0,
                    ]
                )
            if stagnant_steps >= stop_patience:
                break

        if stagnant_steps >= stop_patience:
            break

        if stage.lbfgs_steps_after > 0:
            full_train = tuple(t.to(device_obj) for t in train_ds.tensors)
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.15,
                max_iter=16,
                history_size=100,
                line_search_fn="strong_wolfe",
            )
            for _step in range(stage.lbfgs_steps_after):
                global_epoch += 1
                xb, yb, branch, _stress_true, _eigvecs, _trial_stress, _trial_principal, _strain_eng, _material_reduced, _tangent = full_train

                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = torch.mean((out["stress"] - yb) ** 2)
                    valid_branch = branch >= 0
                    if torch.any(valid_branch):
                        branch_loss = nn.functional.cross_entropy(out["branch_logits"][valid_branch], branch[valid_branch])
                        loss = loss + 0.1 * branch_loss
                    loss.backward()
                    return loss

                model.train(True)
                lbfgs.step(closure)
                train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                val_loader_eval = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                train_metrics = _epoch_loop(
                    model=model,
                    loader=train_loader_eval,
                    optimizer=None,
                    model_kind="raw_branch",
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                    branch_loss_weight=0.1,
                    device=device_obj,
                    grad_clip=1.0,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                    regression_loss_kind="mse",
                    huber_delta=1.0,
                    voigt_mae_weight=0.0,
                    tangent_loss_weight=0.0,
                    tangent_fd_scale=1.0e-6,
                )
                val_metrics = _epoch_loop(
                    model=model,
                    loader=val_loader_eval,
                    optimizer=None,
                    model_kind="raw_branch",
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                    branch_loss_weight=0.1,
                    device=device_obj,
                    grad_clip=1.0,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                    regression_loss_kind="mse",
                    huber_delta=1.0,
                    voigt_mae_weight=0.0,
                    tangent_loss_weight=0.0,
                    tangent_fd_scale=1.0e-6,
                )
                improved = val_metrics["stress_mse"] < (best_val_stress_mse - min_delta)
                if improved:
                    best_val_stress_mse = val_metrics["stress_mse"]
                    best_epoch = global_epoch
                    best_stage = f"{stage.name}_lbfgs"
                    stagnant_steps = 0
                    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, best_ckpt)
                else:
                    stagnant_steps += 1
                torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, last_ckpt)
                with history_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            global_epoch,
                            f"{stage.name}_lbfgs",
                            "lbfgs",
                            batch_size,
                            lbfgs.param_groups[0]["lr"],
                            train_metrics["loss"],
                            val_metrics["loss"],
                            train_metrics["stress_mse"],
                            val_metrics["stress_mse"],
                            train_metrics["branch_accuracy"],
                            val_metrics["branch_accuracy"],
                            best_val_stress_mse,
                            1 if improved else 0,
                        ]
                    )
                if stagnant_steps >= stop_patience:
                    break
        if stagnant_steps >= stop_patience:
            break

    plot_path = _plot_staged_history(history_path, run_dir / "staged_history_log.png")
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_val_stress_mse": best_val_stress_mse,
        "history_csv": str(history_path),
        "history_plot": str(plot_path),
        "selection_metric": "val_stress_mse",
        "schedule": [asdict(stage) for stage in stage_schedule],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _direct_prediction_for_split(
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
    )
    stress_principal_pred = pred.get("stress_principal")
    if stress_principal_pred is None:
        stress_principal_pred = np.linalg.eigvalsh(stress_voigt_to_tensor(pred["stress"]))[:, ::-1].astype(np.float32)
    stats = _pointwise_prediction_stats(
        arrays,
        stress_pred=pred["stress"],
        stress_principal_pred=stress_principal_pred,
    )
    pred["stress_principal"] = stress_principal_pred.astype(np.float32)
    return pred, stats


def _evaluate_global_frontier(
    *,
    prepared: dict[str, Any],
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    delta_grid: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta in delta_grid:
        routed = apply_hybrid_gate(prepared, delta_geom=delta, gate_mode="global")
        stats = _pointwise_prediction_stats(
            arrays,
            stress_pred=routed["stress"],
            stress_principal_pred=routed["stress_principal"],
        )
        metrics = _aggregate_policy_metrics(
            arrays,
            panel,
            stats,
            learned_mask=routed["learned_mask"],
            fallback_mask=routed["fallback_mask"],
            elastic_mask=routed["elastic_mask"],
            accepted_yield_violation_rel=routed["accepted_yield_violation_rel"],
        )
        rows.append(
            {
                "gate_mode": "global",
                "delta_geom": float(delta),
                "metrics": metrics,
                "route_counts": routed["route_counts"],
                "predicted_branch_counts": routed["predicted_branch_counts"],
            }
        )
    return rows


def _evaluate_oracle_frontier(
    *,
    prepared: dict[str, Any],
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    delta_grid: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta in delta_grid:
        routed = apply_hybrid_gate(
            prepared,
            delta_geom=delta,
            gate_mode="oracle_branch",
            branch_id_for_gate=arrays["branch_id"],
        )
        stats = _pointwise_prediction_stats(
            arrays,
            stress_pred=routed["stress"],
            stress_principal_pred=routed["stress_principal"],
        )
        metrics = _aggregate_policy_metrics(
            arrays,
            panel,
            stats,
            learned_mask=routed["learned_mask"],
            fallback_mask=routed["fallback_mask"],
            elastic_mask=routed["elastic_mask"],
            accepted_yield_violation_rel=routed["accepted_yield_violation_rel"],
        )
        rows.append(
            {
                "gate_mode": "oracle_branch",
                "delta_geom": float(delta),
                "metrics": metrics,
                "route_counts": routed["route_counts"],
            }
        )
    return rows


def _evaluate_predicted_branch_frontier(
    *,
    prepared: dict[str, Any],
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    delta_grid: tuple[float, ...],
    entropy_grid: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entropy in entropy_grid:
        for delta in delta_grid:
            routed = apply_hybrid_gate(
                prepared,
                delta_geom=delta,
                gate_mode="predicted_branch",
                entropy_threshold=entropy,
            )
            stats = _pointwise_prediction_stats(
                arrays,
                stress_pred=routed["stress"],
                stress_principal_pred=routed["stress_principal"],
            )
            metrics = _aggregate_policy_metrics(
                arrays,
                panel,
                stats,
                learned_mask=routed["learned_mask"],
                fallback_mask=routed["fallback_mask"],
                elastic_mask=routed["elastic_mask"],
                accepted_yield_violation_rel=routed["accepted_yield_violation_rel"],
            )
            rows.append(
                {
                    "gate_mode": "predicted_branch",
                    "delta_geom": float(delta),
                    "entropy_threshold": float(entropy),
                    "metrics": metrics,
                    "route_counts": routed["route_counts"],
                    "predicted_branch_counts": routed["predicted_branch_counts"],
                }
            )
    return rows


def _format_delta(delta: float) -> str:
    return f"{delta:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _runtime_feature_bank(
    arrays: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    material = arrays["material_reduced"]
    strain_principal, _eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
    geom = compute_branch_geometry_principal(
        strain_principal,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    trial_stress = compute_trial_stress(arrays["strain_eng"], material)
    trial_principal = np.linalg.eigvalsh(stress_voigt_to_tensor(trial_stress))[:, ::-1].astype(np.float32)
    branch_probs = prediction["branch_probabilities"][:, 1:].astype(np.float32)
    branch_probs_sum = np.sum(branch_probs, axis=1, keepdims=True)
    branch_probs = np.divide(branch_probs, np.maximum(branch_probs_sum, 1.0e-12), out=np.zeros_like(branch_probs), where=branch_probs_sum > 0.0)
    predicted_branch_id = np.argmax(branch_probs, axis=1).astype(np.int8) + 1
    predicted_branch_distance = np.zeros(arrays["strain_eng"].shape[0], dtype=np.float32)
    predicted_branch_distance[predicted_branch_id == 1] = geom.d_smooth[predicted_branch_id == 1]
    predicted_branch_distance[predicted_branch_id == 2] = geom.d_left[predicted_branch_id == 2]
    predicted_branch_distance[predicted_branch_id == 3] = geom.d_right[predicted_branch_id == 3]
    predicted_branch_distance[predicted_branch_id == 4] = geom.d_apex[predicted_branch_id == 4]
    trial_mag = np.linalg.norm(trial_principal, axis=1) / np.maximum(material[:, 0], 1.0)
    predicted_correction_norm_rel = np.linalg.norm(prediction["stress"] - trial_stress, axis=1) / (
        np.linalg.norm(prediction["stress"], axis=1) + np.maximum(material[:, 0], 0.0) + 1.0e-12
    )
    entropy = -np.sum(np.where(branch_probs > 0.0, branch_probs * np.log(np.maximum(branch_probs, 1.0e-12)), 0.0), axis=1) / np.log(4.0)
    features = np.column_stack(
        [
            np.abs(geom.m_yield),
            np.abs(geom.m_smooth_left),
            np.abs(geom.m_smooth_right),
            np.abs(geom.m_left_apex),
            np.abs(geom.m_right_apex),
            geom.gap12_norm,
            geom.gap23_norm,
            geom.gap_min_norm,
            geom.d_geom,
            predicted_branch_distance,
            material,
            trial_mag[:, None],
            branch_probs,
            entropy[:, None],
            predicted_correction_norm_rel[:, None],
        ]
    ).astype(np.float32)
    aux = {
        "predicted_branch_id": predicted_branch_id.astype(np.int8),
        "predicted_branch_distance": predicted_branch_distance.astype(np.float32),
        "branch_entropy": entropy.astype(np.float32),
        "predicted_correction_norm_rel": predicted_correction_norm_rel.astype(np.float32),
        "trial_mag": trial_mag.astype(np.float32),
        "d_geom": geom.d_geom.astype(np.float32),
    }
    return features, aux


class RejectorNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_rejector(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    run_dir: Path,
    device: str,
    epochs: int,
    force_rerun: bool = False,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    if summary_path.exists() and best_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    device_obj = choose_device(device)
    x_mean = train_x.mean(axis=0).astype(np.float32)
    x_std = np.where(train_x.std(axis=0) < 1.0e-12, 1.0, train_x.std(axis=0)).astype(np.float32)
    train_xn = ((train_x - x_mean) / x_std).astype(np.float32)
    val_xn = ((val_x - x_mean) / x_std).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(train_xn), torch.from_numpy(train_y.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(val_xn), torch.from_numpy(val_y.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

    model = RejectorNet(train_x.shape[1]).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1.0e-5)
    pos_count = max(float(np.sum(train_y > 0.5)), 1.0)
    neg_count = max(float(np.sum(train_y <= 0.5)), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_positive_rate"])

    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train(True)
        train_loss = 0.0
        train_rows = 0
        for xb, yb in train_loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * xb.shape[0]
            train_rows += xb.shape[0]

        model.train(False)
        val_loss = 0.0
        val_rows = 0
        val_probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_obj)
                yb = yb.to(device_obj)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.detach().cpu()) * xb.shape[0]
                val_rows += xb.shape[0]
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
        scheduler.step()
        mean_train = train_loss / max(train_rows, 1)
        mean_val = val_loss / max(val_rows, 1)
        val_positive_rate = float(np.mean(np.concatenate(val_probs, axis=0) >= 0.5))
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, optimizer.param_groups[0]["lr"], mean_train, mean_val, val_positive_rate])
        if mean_val < best_val:
            best_val = mean_val
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        "x_mean": x_mean.tolist(),
                        "x_std": x_std.tolist(),
                        "input_dim": int(train_x.shape[1]),
                    },
                },
                best_path,
            )

    _plot_rejector_history(history_path, run_dir / "history_log.png")
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history_csv": str(history_path),
        "history_plot": str(run_dir / "history_log.png"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_rejector(checkpoint_path: Path, x: np.ndarray, *, device: str) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location=choose_device(device))
    device_obj = choose_device(device)
    meta = ckpt["metadata"]
    model = RejectorNet(int(meta["input_dim"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device_obj)
    model.eval()
    mean = np.asarray(meta["x_mean"], dtype=np.float32)
    std = np.asarray(meta["x_std"], dtype=np.float32)
    xn = ((x.astype(np.float32) - mean) / std).astype(np.float32)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, xn.shape[0], 4096):
            xb = torch.from_numpy(xn[start : start + 4096]).to(device_obj)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _rows_to_flat(rows: list[dict[str, Any]], extra_keys: list[str]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        payload = {
            key: row.get(key)
            for key in extra_keys
        }
        metrics = row["metrics"]
        payload.update(metrics)
        flat.append(payload)
    return flat


def _write_phase_report(path: Path, title: str, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f"# {title}", ""]
    content.extend(lines)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    full_export = (ROOT / args.full_export).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    exp0_report = docs_root / "hybrid_gate_redesign_exp0_panels_20260324.md"
    execution_doc = docs_root / "hybrid_gate_redesign_execution_20260324.md"
    taskgiver_report = docs_root / "hybrid_gate_redesign_taskgiver_report_20260324.md"
    exp_a_report = docs_root / "hybrid_gate_redesign_expA_global_20260324.md"
    exp_b_report = docs_root / "hybrid_gate_redesign_expB_audit_20260324.md"
    exp_c_report = docs_root / "hybrid_gate_redesign_expC_oracle_20260324.md"
    exp_d_report = docs_root / "hybrid_gate_redesign_expD_predicted_branch_20260324.md"
    exp_e_report = docs_root / "hybrid_gate_redesign_expE_rejector_20260324.md"
    pivot_report = docs_root / "hybrid_gate_redesign_pivot_memo_20260324.md"

    panel_result = build_hybrid_real_panels(
        full_export=full_export,
        output_root=output_root,
        report_path=exp0_report,
        samples_per_call=args.samples_per_call,
        seed=args.seed,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
    )
    dataset_path = Path(panel_result["dataset_path"])
    panel_path = Path(panel_result["panel_path"])
    arrays_all, _attrs = _load_h5(dataset_path)
    panel_all, _panel_attrs = _load_h5(panel_path)

    baseline_run_dir = output_root / "baseline" / "rb_staged_w512_d6_valfirst"
    baseline_summary = train_validation_first_staged_baseline(
        dataset_path=dataset_path,
        run_dir=baseline_run_dir,
        device=args.device,
        width=args.baseline_width,
        depth=args.baseline_depth,
        seed=args.baseline_seed,
        force_rerun=args.force_rerun,
    )
    baseline_ckpt = Path(baseline_summary["best_checkpoint"])

    candidate_data_root = output_root / "candidate_b" / "data"
    pretrain_dataset = build_synthetic_pretrain_dataset(
        real_dataset_path=dataset_path,
        output_root=candidate_data_root,
        train_rows=args.pretrain_train_rows,
        val_rows=args.pretrain_val_rows,
        test_rows=args.pretrain_test_rows,
        seed=args.seed,
    )
    finetune_dataset = build_finetune_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        synthetic_dataset_path=Path(pretrain_dataset["dataset_path"]),
        output_root=candidate_data_root,
        train_rows=args.finetune_train_rows,
        real_fraction=args.finetune_real_fraction,
        seed=args.seed,
    )

    pretrain_config = TrainingConfig(
        dataset=str(pretrain_dataset["dataset_path"]),
        run_dir=str(output_root / "candidate_b" / "pretrain"),
        model_kind="trial_principal_geom_plastic_branch_residual",
        epochs=args.pretrain_epochs,
        batch_size=args.candidate_train_batch_size,
        lr=1.0e-3,
        weight_decay=1.0e-4,
        width=args.candidate_width,
        depth=args.candidate_depth,
        dropout=0.0,
        seed=args.seed,
        patience=max(10, args.pretrain_epochs),
        device=args.device,
        scheduler_kind="cosine",
        warmup_epochs=2,
        min_lr=1.0e-5,
        branch_loss_weight=0.05,
        regression_loss_kind="huber",
        huber_delta=1.0,
        voigt_mae_weight=0.25,
        snapshot_every_epochs=10,
    )
    finetune_config = TrainingConfig(
        dataset=str(finetune_dataset["dataset_path"]),
        run_dir=str(output_root / "candidate_b" / "finetune"),
        model_kind="trial_principal_geom_plastic_branch_residual",
        epochs=args.finetune_epochs,
        batch_size=args.candidate_train_batch_size,
        lr=3.0e-4,
        weight_decay=1.0e-5,
        width=args.candidate_width,
        depth=args.candidate_depth,
        dropout=0.0,
        seed=args.seed,
        patience=max(8, args.finetune_epochs),
        device=args.device,
        scheduler_kind="plateau",
        plateau_factor=0.5,
        plateau_patience=4,
        min_lr=1.0e-5,
        branch_loss_weight=0.05,
        regression_loss_kind="huber",
        huber_delta=1.0,
        voigt_mae_weight=0.25,
        snapshot_every_epochs=10,
        init_checkpoint=str(output_root / "candidate_b" / "pretrain" / "best.pt"),
    )
    pretrain_summary = _maybe_train(pretrain_config, "Candidate B pretrain")
    finetune_summary = _maybe_train(finetune_config, "Candidate B finetune")

    val_arrays = _slice_dict(arrays_all, _split_mask(arrays_all, "val"))
    val_panel = _slice_dict(panel_all, _split_mask(arrays_all, "val"))
    test_arrays = _slice_dict(arrays_all, _split_mask(arrays_all, "test"))
    test_panel = _slice_dict(panel_all, _split_mask(arrays_all, "test"))
    train_arrays = _slice_dict(arrays_all, _split_mask(arrays_all, "train"))
    train_panel = _slice_dict(panel_all, _split_mask(arrays_all, "train"))

    baseline_val_pred, baseline_val_stats = _direct_prediction_for_split(
        baseline_ckpt,
        val_arrays,
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    baseline_val_metrics = _aggregate_policy_metrics(val_arrays, val_panel, baseline_val_stats)
    ceilings = {
        "broad_plastic_mae_ceiling": baseline_val_metrics["broad_plastic_mae"],
        "hard_plastic_mae_ceiling": baseline_val_metrics["hard_plastic_mae"],
        "hard_p95_principal_ceiling": baseline_val_metrics["hard_p95_principal"],
        "yield_violation_p95_ceiling": baseline_val_metrics["yield_violation_p95"],
        "baseline_hard_rel_p95": baseline_val_metrics["hard_rel_p95_principal"],
    }

    baseline_val_prepared = prepare_hybrid_gate_inputs(
        baseline_ckpt,
        val_arrays["strain_eng"],
        val_arrays["material_reduced"],
        device=args.device,
        batch_size=args.eval_batch_size,
        include_exact=True,
    )
    baseline_global_rows = _evaluate_global_frontier(
        prepared=baseline_val_prepared,
        arrays=val_arrays,
        panel=val_panel,
        delta_grid=DELTA_GRID,
    )
    baseline_global_selection = _select_frontier_row(baseline_global_rows, ceilings)

    snapshot_rows: list[dict[str, Any]] = []
    snapshot_details: dict[str, Any] = {}
    for checkpoint_path in _candidate_checkpoint_paths(Path(finetune_summary["run_dir"])):
        label = checkpoint_path.stem if checkpoint_path.parent.name == "snapshots" else checkpoint_path.name
        prepared = prepare_hybrid_gate_inputs(
            checkpoint_path,
            val_arrays["strain_eng"],
            val_arrays["material_reduced"],
            device=args.device,
            batch_size=args.eval_batch_size,
            include_exact=True,
        )
        rows = _evaluate_global_frontier(prepared=prepared, arrays=val_arrays, panel=val_panel, delta_grid=DELTA_GRID)
        selection = _select_frontier_row(rows, ceilings)
        snapshot_details[label] = {"rows": rows, "selection": selection, "checkpoint_path": str(checkpoint_path)}
        best_row = selection["best_row"]
        snapshot_rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_path": str(checkpoint_path),
                "selection_status": selection["status"],
                "feasible_rows": int(selection["feasible_rows"]),
                "delta_geom": best_row["delta_geom"],
                "plastic_coverage": best_row["metrics"]["plastic_coverage"],
                "hard_p95_principal": best_row["metrics"]["hard_p95_principal"],
                "hard_plastic_mae": best_row["metrics"]["hard_plastic_mae"],
                "broad_plastic_mae": best_row["metrics"]["broad_plastic_mae"],
                "yield_violation_p95": best_row["metrics"]["yield_violation_p95"],
            }
        )

    selected_snapshot = min(
        snapshot_rows,
        key=lambda row: _coverage_under_risk_sort_key(snapshot_details[row["checkpoint_label"]]["selection"]["best_row"])
        if snapshot_details[row["checkpoint_label"]]["selection"]["status"] == "feasible"
        else _risk_excess_key(snapshot_details[row["checkpoint_label"]]["selection"]["best_row"], ceilings),
    )
    candidate_ckpt = Path(selected_snapshot["checkpoint_path"])
    candidate_val_prepared = prepare_hybrid_gate_inputs(
        candidate_ckpt,
        val_arrays["strain_eng"],
        val_arrays["material_reduced"],
        device=args.device,
        batch_size=args.eval_batch_size,
        include_exact=True,
    )
    candidate_global_rows = snapshot_details[selected_snapshot["checkpoint_label"]]["rows"]
    candidate_global_selection = snapshot_details[selected_snapshot["checkpoint_label"]]["selection"]
    best_global_row = candidate_global_selection["best_row"]

    baseline_band_pred, baseline_band_stats = baseline_val_pred, baseline_val_stats
    candidate_val_pred, candidate_val_stats = _direct_prediction_for_split(
        candidate_ckpt,
        val_arrays,
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    d_geom_val = candidate_val_prepared["d_geom"]
    plastic_val = val_panel["plastic_mask"].astype(bool)
    band_rows: list[dict[str, Any]] = []
    for low, high in D_BANDS:
        band_mask = plastic_val & (d_geom_val >= low) & (d_geom_val < high)
        if not np.any(band_mask):
            continue
        band_label = f"[{low:.3f}, {high if math.isfinite(high) else '+inf'})"
        band_rows.append(
            {
                "band": band_label,
                "n_rows": int(np.sum(band_mask)),
                "baseline_mae": float(np.mean(baseline_band_stats["stress_component_abs"][band_mask])),
                "candidate_mae": float(np.mean(candidate_val_stats["stress_component_abs"][band_mask])),
                "baseline_hard_p95": _quantile_or_zero(baseline_band_stats["principal_max_abs"][band_mask], 0.95),
                "candidate_hard_p95": _quantile_or_zero(candidate_val_stats["principal_max_abs"][band_mask], 0.95),
                "baseline_yield_violation_p95": _quantile_or_zero(baseline_band_stats["yield_violation_rel"][band_mask], 0.95),
                "candidate_yield_violation_p95": _quantile_or_zero(candidate_val_stats["yield_violation_rel"][band_mask], 0.95),
            }
        )

    exp_a_dir = output_root / "exp_a_global"
    baseline_frontier_csv = _write_csv(
        exp_a_dir / "baseline_global_frontier_val.csv",
        _rows_to_flat(baseline_global_rows, ["gate_mode", "delta_geom"]),
        ["gate_mode", "delta_geom", "n_rows", "broad_mae", "hard_mae", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "plastic_coverage", "accepted_plastic_rows", "route_counts", "accepted_true_branch_counts"],
    )
    candidate_frontier_csv = _write_csv(
        exp_a_dir / "candidate_b_global_frontier_val.csv",
        _rows_to_flat(candidate_global_rows, ["gate_mode", "delta_geom"]),
        ["gate_mode", "delta_geom", "n_rows", "broad_mae", "hard_mae", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "plastic_coverage", "accepted_plastic_rows", "route_counts", "accepted_true_branch_counts"],
    )
    band_csv = _write_csv(
        exp_a_dir / "d_geom_bandwise_val.csv",
        band_rows,
        ["band", "n_rows", "baseline_mae", "candidate_mae", "baseline_hard_p95", "candidate_hard_p95", "baseline_yield_violation_p95", "candidate_yield_violation_p95"],
    )
    snapshot_csv = _write_csv(
        exp_a_dir / "candidate_b_snapshot_sweep_val.csv",
        snapshot_rows,
        ["checkpoint_label", "checkpoint_path", "selection_status", "feasible_rows", "delta_geom", "plastic_coverage", "hard_p95_principal", "hard_plastic_mae", "broad_plastic_mae", "yield_violation_p95"],
    )
    _plot_frontier(exp_a_dir / "baseline_global_frontier_val.png", baseline_global_rows, title="Baseline global frontier", color_key="delta_geom")
    _plot_frontier(exp_a_dir / "candidate_b_global_frontier_val.png", candidate_global_rows, title="Candidate B global frontier", color_key="delta_geom")
    _write_json(exp_a_dir / "baseline_selection.json", baseline_global_selection)
    _write_json(exp_a_dir / "candidate_b_selection.json", candidate_global_selection)
    snapshot_selection_json = _write_json(
        exp_a_dir / "candidate_b_snapshot_selection.json",
        {
            "selected_checkpoint_label": selected_snapshot["checkpoint_label"],
            "selected_checkpoint_path": selected_snapshot["checkpoint_path"],
            "selection_status": selected_snapshot["selection_status"],
            "selected_delta_geom": selected_snapshot["delta_geom"],
            "selected_summary": selected_snapshot,
            "selection": candidate_global_selection,
        },
    )

    best_global_delta = float(best_global_row["delta_geom"])
    candidate_global_pred = apply_hybrid_gate(candidate_val_prepared, delta_geom=best_global_delta, gate_mode="global")
    candidate_global_stats = _pointwise_prediction_stats(
        val_arrays,
        stress_pred=candidate_global_pred["stress"],
        stress_principal_pred=candidate_global_pred["stress_principal"],
    )
    low_risk_mask = (
        (candidate_val_stats["principal_rel_error"] <= ceilings["baseline_hard_rel_p95"])
        & (candidate_val_stats["yield_violation_rel"] <= ceilings["yield_violation_p95_ceiling"])
    )
    accept_columns = {
        f"accept_delta_{_format_delta(delta)}": (apply_hybrid_gate(candidate_val_prepared, delta_geom=delta, gate_mode="global")["learned_mask"] & plastic_val).astype(np.int8)
        for delta in DELTA_GRID
    }
    min_term_rows: list[dict[str, Any]] = []
    for model_family, stats in (("baseline_raw_branch", baseline_band_stats), ("candidate_b", candidate_val_stats)):
        for idx in np.where(plastic_val)[0]:
            row = {
                "row_index": int(idx),
                "model_family": model_family,
                "true_branch": BRANCH_NAMES[int(val_arrays["branch_id"][idx])],
                "winning_term": str(candidate_val_prepared["d_geom_min_term"][idx]),
                "d_geom": float(candidate_val_prepared["d_geom"][idx]),
                "principal_rel_error": float(stats["principal_rel_error"][idx]),
                "yield_violation_rel": float(stats["yield_violation_rel"][idx]),
                "selected_delta_accept": int(candidate_global_pred["learned_mask"][idx]),
                "candidate_low_risk": int(low_risk_mask[idx]),
            }
            for name, values in accept_columns.items():
                row[name] = int(values[idx])
            min_term_rows.append(row)

    attribution_summary: list[dict[str, Any]] = []
    for winning_term in sorted({row["winning_term"] for row in min_term_rows}):
        for branch_name in BRANCH_NAMES[1:]:
            candidate_rows = [
                row for row in min_term_rows
                if row["model_family"] == "candidate_b" and row["winning_term"] == winning_term and row["true_branch"] == branch_name
            ]
            if not candidate_rows:
                continue
            rejected = [row for row in candidate_rows if row["selected_delta_accept"] == 0]
            attribution_summary.append(
                {
                    "winning_term": winning_term,
                    "true_branch": branch_name,
                    "n_rows": len(candidate_rows),
                    "rejected_rows": len(rejected),
                    "rejected_low_risk_fraction": float(np.mean([row["candidate_low_risk"] for row in rejected])) if rejected else 0.0,
                }
            )
    exp_b_dir = output_root / "exp_b_audit"
    _write_csv(
        exp_b_dir / "candidate_b_min_term_rows_val.csv",
        min_term_rows,
        ["row_index", "model_family", "true_branch", "winning_term", "d_geom", "principal_rel_error", "yield_violation_rel", "selected_delta_accept", "candidate_low_risk"] + list(accept_columns.keys()),
    )
    _write_csv(
        exp_b_dir / "candidate_b_min_term_summary_val.csv",
        attribution_summary,
        ["winning_term", "true_branch", "n_rows", "rejected_rows", "rejected_low_risk_fraction"],
    )

    exp_c_dir = output_root / "exp_c_oracle"
    oracle_rows = _evaluate_oracle_frontier(
        prepared=candidate_val_prepared,
        arrays=val_arrays,
        panel=val_panel,
        delta_grid=DELTA_GRID,
    )
    oracle_selection = _select_frontier_row(oracle_rows, ceilings)
    _write_csv(
        exp_c_dir / "oracle_frontier_val.csv",
        _rows_to_flat(oracle_rows, ["gate_mode", "delta_geom"]),
        ["gate_mode", "delta_geom", "n_rows", "broad_mae", "hard_mae", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "plastic_coverage", "accepted_plastic_rows", "route_counts", "accepted_true_branch_counts"],
    )
    _write_json(exp_c_dir / "oracle_selection.json", oracle_selection)
    _plot_frontier(exp_c_dir / "oracle_frontier_val.png", oracle_rows, title="Oracle branch frontier", color_key="delta_geom")
    oracle_improvement = oracle_selection["best_row"]["metrics"]["plastic_coverage"] - best_global_row["metrics"]["plastic_coverage"]

    best_redesign_kind = None
    best_redesign_policy: dict[str, Any] | None = None
    best_redesign_summary: dict[str, Any] | None = None
    rejector_summary: dict[str, Any] | None = None

    if oracle_improvement < 0.05:
        pivot_lines = [
            "Oracle branch-conditioning did not recover enough validation coverage to justify a deployable redesign follow-on.",
            "",
            "## Decision",
            "",
            f"- best feasible global-gate coverage: `{best_global_row['metrics']['plastic_coverage']:.6f}`",
            f"- best feasible oracle coverage: `{oracle_selection['best_row']['metrics']['plastic_coverage']:.6f}`",
            f"- absolute improvement: `{oracle_improvement:.6f}`",
            "- predicted-branch gating and learned rejector were skipped under the plan's stop rule.",
            "- the next credible path is a return-mapping-assisted acceleration route.",
        ]
        _write_phase_report(pivot_report, "Hybrid Gate Redesign Pivot Memo", pivot_lines)
    else:
        exp_d_dir = output_root / "exp_d_predicted_branch"
        predicted_rows = _evaluate_predicted_branch_frontier(
            prepared=candidate_val_prepared,
            arrays=val_arrays,
            panel=val_panel,
            delta_grid=DELTA_GRID,
            entropy_grid=ENTROPY_GRID,
        )
        predicted_selection = _select_frontier_row(predicted_rows, ceilings)
        _write_csv(
            exp_d_dir / "predicted_branch_frontier_val.csv",
            _rows_to_flat(predicted_rows, ["gate_mode", "delta_geom", "entropy_threshold"]),
            ["gate_mode", "delta_geom", "entropy_threshold", "n_rows", "broad_mae", "hard_mae", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "plastic_coverage", "accepted_plastic_rows", "route_counts", "accepted_true_branch_counts"],
        )
        _write_json(exp_d_dir / "predicted_selection.json", predicted_selection)
        _plot_frontier(exp_d_dir / "predicted_branch_frontier_val.png", predicted_rows, title="Predicted branch frontier", color_key="entropy_threshold")
        pred_improvement = predicted_selection["best_row"]["metrics"]["plastic_coverage"] - best_global_row["metrics"]["plastic_coverage"]

        if pred_improvement < 0.03:
            best_redesign_kind = "predicted_branch"
            best_redesign_policy = predicted_selection["best_row"]
            best_redesign_summary = predicted_selection
            pivot_lines = [
                "Predicted branch-conditioning improved on the global gate, but not enough to justify a learned rejector stage.",
                "",
                "## Decision",
                "",
                f"- best feasible global-gate coverage: `{best_global_row['metrics']['plastic_coverage']:.6f}`",
                f"- best feasible predicted-branch coverage: `{predicted_selection['best_row']['metrics']['plastic_coverage']:.6f}`",
                f"- absolute improvement: `{pred_improvement:.6f}`",
                "- learned rejector was skipped under the plan's stop rule.",
            ]
            _write_phase_report(pivot_report, "Hybrid Gate Redesign Pivot Memo", pivot_lines)
        else:
            train_prediction, train_stats = _direct_prediction_for_split(
                candidate_ckpt,
                train_arrays,
                device=args.device,
                batch_size=args.eval_batch_size,
            )
            val_prediction = candidate_val_pred
            train_features, _train_aux = _runtime_feature_bank(train_arrays, train_prediction)
            val_features, _val_aux = _runtime_feature_bank(val_arrays, val_prediction)
            train_plastic = train_panel["plastic_mask"].astype(bool)
            val_plastic = val_panel["plastic_mask"].astype(bool)
            train_unsafe = (
                (train_stats["principal_rel_error"] > ceilings["baseline_hard_rel_p95"])
                | (train_stats["yield_violation_rel"] > ceilings["yield_violation_p95_ceiling"])
            ).astype(np.float32)
            val_unsafe = (
                (candidate_val_stats["principal_rel_error"] > ceilings["baseline_hard_rel_p95"])
                | (candidate_val_stats["yield_violation_rel"] > ceilings["yield_violation_p95_ceiling"])
            ).astype(np.float32)
            exp_e_dir = output_root / "exp_e_rejector"
            rejector_summary = train_rejector(
                train_x=train_features[train_plastic],
                train_y=train_unsafe[train_plastic],
                val_x=val_features[val_plastic],
                val_y=val_unsafe[val_plastic],
                run_dir=exp_e_dir / "training",
                device=args.device,
                epochs=args.rejector_epochs,
                force_rerun=args.force_rerun,
            )
            rejector_val_score = np.zeros(val_arrays["strain_eng"].shape[0], dtype=np.float32)
            rejector_val_score[val_plastic] = predict_rejector(Path(rejector_summary["best_checkpoint"]), val_features[val_plastic], device=args.device)
            rejector_rows: list[dict[str, Any]] = []
            for tau in TAU_GRID:
                routed = apply_hybrid_gate(
                    candidate_val_prepared,
                    delta_geom=0.0,
                    gate_mode="rejector",
                    rejector_score=rejector_val_score,
                    rejector_threshold=tau,
                )
                stats = _pointwise_prediction_stats(
                    val_arrays,
                    stress_pred=routed["stress"],
                    stress_principal_pred=routed["stress_principal"],
                )
                metrics = _aggregate_policy_metrics(
                    val_arrays,
                    val_panel,
                    stats,
                    learned_mask=routed["learned_mask"],
                    fallback_mask=routed["fallback_mask"],
                    elastic_mask=routed["elastic_mask"],
                    accepted_yield_violation_rel=routed["accepted_yield_violation_rel"],
                )
                rejector_rows.append(
                    {
                        "gate_mode": "rejector",
                        "tau": float(tau),
                        "metrics": metrics,
                        "route_counts": routed["route_counts"],
                    }
                )
            rejector_selection = _select_frontier_row(rejector_rows, ceilings)
            _write_csv(
                exp_e_dir / "rejector_frontier_val.csv",
                _rows_to_flat(rejector_rows, ["gate_mode", "tau"]),
                ["gate_mode", "tau", "n_rows", "broad_mae", "hard_mae", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "plastic_coverage", "accepted_plastic_rows", "route_counts", "accepted_true_branch_counts"],
            )
            _write_json(exp_e_dir / "rejector_selection.json", rejector_selection)
            _plot_frontier(exp_e_dir / "rejector_frontier_val.png", rejector_rows, title="Rejector frontier", color_key="tau")
            best_redesign_kind = "rejector"
            best_redesign_policy = rejector_selection["best_row"]
            best_redesign_summary = rejector_selection

    baseline_test_pred, baseline_test_stats = _direct_prediction_for_split(
        baseline_ckpt,
        test_arrays,
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    baseline_test_metrics = _aggregate_policy_metrics(test_arrays, test_panel, baseline_test_stats)
    baseline_feasible_delta = float(baseline_global_selection["best_row"]["delta_geom"])
    baseline_test_prepared = prepare_hybrid_gate_inputs(
        baseline_ckpt,
        test_arrays["strain_eng"],
        test_arrays["material_reduced"],
        device=args.device,
        batch_size=args.eval_batch_size,
        include_exact=True,
    )
    baseline_feasible_pred = apply_hybrid_gate(
        baseline_test_prepared,
        delta_geom=baseline_feasible_delta,
        gate_mode="global",
    )
    baseline_feasible_stats = _pointwise_prediction_stats(
        test_arrays,
        stress_pred=baseline_feasible_pred["stress"],
        stress_principal_pred=baseline_feasible_pred["stress_principal"],
    )
    baseline_feasible_test_metrics = _aggregate_policy_metrics(
        test_arrays,
        test_panel,
        baseline_feasible_stats,
        learned_mask=baseline_feasible_pred["learned_mask"],
        fallback_mask=baseline_feasible_pred["fallback_mask"],
        elastic_mask=baseline_feasible_pred["elastic_mask"],
        accepted_yield_violation_rel=baseline_feasible_pred["accepted_yield_violation_rel"],
    )

    candidate_test_prepared = prepare_hybrid_gate_inputs(
        candidate_ckpt,
        test_arrays["strain_eng"],
        test_arrays["material_reduced"],
        device=args.device,
        batch_size=args.eval_batch_size,
        include_exact=True,
    )
    global_test_pred = apply_hybrid_gate(candidate_test_prepared, delta_geom=best_global_delta, gate_mode="global")
    global_test_stats = _pointwise_prediction_stats(
        test_arrays,
        stress_pred=global_test_pred["stress"],
        stress_principal_pred=global_test_pred["stress_principal"],
    )
    global_test_metrics = _aggregate_policy_metrics(
        test_arrays,
        test_panel,
        global_test_stats,
        learned_mask=global_test_pred["learned_mask"],
        fallback_mask=global_test_pred["fallback_mask"],
        elastic_mask=global_test_pred["elastic_mask"],
        accepted_yield_violation_rel=global_test_pred["accepted_yield_violation_rel"],
    )

    redesigned_test_metrics = None
    redesigned_test_label = "not_available"
    if best_redesign_kind == "predicted_branch" and best_redesign_policy is not None:
        redesign_pred = apply_hybrid_gate(
            candidate_test_prepared,
            delta_geom=float(best_redesign_policy["delta_geom"]),
            gate_mode="predicted_branch",
            entropy_threshold=float(best_redesign_policy["entropy_threshold"]),
        )
        redesign_stats = _pointwise_prediction_stats(
            test_arrays,
            stress_pred=redesign_pred["stress"],
            stress_principal_pred=redesign_pred["stress_principal"],
        )
        redesigned_test_metrics = _aggregate_policy_metrics(
            test_arrays,
            test_panel,
            redesign_stats,
            learned_mask=redesign_pred["learned_mask"],
            fallback_mask=redesign_pred["fallback_mask"],
            elastic_mask=redesign_pred["elastic_mask"],
            accepted_yield_violation_rel=redesign_pred["accepted_yield_violation_rel"],
        )
        redesigned_test_label = "predicted_branch"
    elif best_redesign_kind == "rejector" and best_redesign_policy is not None and rejector_summary is not None:
        test_prediction, _test_stats_raw = _direct_prediction_for_split(
            candidate_ckpt,
            test_arrays,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        test_features, _test_aux = _runtime_feature_bank(test_arrays, test_prediction)
        test_plastic = test_panel["plastic_mask"].astype(bool)
        rejector_test_score = np.zeros(test_arrays["strain_eng"].shape[0], dtype=np.float32)
        rejector_test_score[test_plastic] = predict_rejector(Path(rejector_summary["best_checkpoint"]), test_features[test_plastic], device=args.device)
        redesign_pred = apply_hybrid_gate(
            candidate_test_prepared,
            delta_geom=0.0,
            gate_mode="rejector",
            rejector_score=rejector_test_score,
            rejector_threshold=float(best_redesign_policy["tau"]),
        )
        redesign_stats = _pointwise_prediction_stats(
            test_arrays,
            stress_pred=redesign_pred["stress"],
            stress_principal_pred=redesign_pred["stress_principal"],
        )
        redesigned_test_metrics = _aggregate_policy_metrics(
            test_arrays,
            test_panel,
            redesign_stats,
            learned_mask=redesign_pred["learned_mask"],
            fallback_mask=redesign_pred["fallback_mask"],
            elastic_mask=redesign_pred["elastic_mask"],
            accepted_yield_violation_rel=redesign_pred["accepted_yield_violation_rel"],
        )
        redesigned_test_label = "rejector"

    final_summary = {
        "baseline_raw_test": baseline_test_metrics,
        "baseline_feasible_gate_test": {
            "delta_geom": baseline_feasible_delta,
            "selection_status": baseline_global_selection["status"],
            **baseline_feasible_test_metrics,
        },
        "best_global_gate_test": global_test_metrics,
        "best_redesign_kind": best_redesign_kind,
        "best_redesign_test": redesigned_test_metrics,
        "ceilings_from_val": ceilings,
    }
    _write_json(output_root / "final" / "final_test_summary.json", final_summary)

    redesign_improvement = (
        float(redesigned_test_metrics["plastic_coverage"]) - float(global_test_metrics["plastic_coverage"])
        if redesigned_test_metrics is not None
        else float("-inf")
    )
    redesign_passes_test = (
        redesigned_test_metrics is not None
        and redesigned_test_metrics["plastic_coverage"] >= global_test_metrics["plastic_coverage"] + 0.05
        and _is_feasible(redesigned_test_metrics, ceilings)
    )

    _write_phase_report(
        exp_a_report,
        "Hybrid Gate Redesign Experiment A",
        [
            "Dense global-gate frontier on dev validation for the retrained March-12-style baseline and the selected Candidate B checkpoint.",
            "",
            "## Artifacts",
            "",
            f"- baseline frontier csv: `{baseline_frontier_csv}`",
            f"- candidate frontier csv: `{candidate_frontier_csv}`",
            f"- Candidate B snapshot sweep csv: `{snapshot_csv}`",
            f"- Candidate B snapshot selection json: `{snapshot_selection_json}`",
            f"- d_geom bandwise csv: `{band_csv}`",
            "",
            "## Baseline Ceilings",
            "",
            f"- broad plastic MAE ceiling: `{ceilings['broad_plastic_mae_ceiling']:.6f}`",
            f"- hard plastic MAE ceiling: `{ceilings['hard_plastic_mae_ceiling']:.6f}`",
            f"- hard p95 principal ceiling: `{ceilings['hard_p95_principal_ceiling']:.6f}`",
            f"- yield violation p95 ceiling: `{ceilings['yield_violation_p95_ceiling']:.6f}`",
            "",
            f"- validation-feasible baseline global delta: `{baseline_feasible_delta:.6f}`",
            "- Candidate B checkpoint was selected from a validation frontier sweep over `best.pt`, `last.pt`, and all saved snapshot checkpoints.",
            f"- selected Candidate B checkpoint label: `{selected_snapshot['checkpoint_label']}`",
            f"- selected Candidate B checkpoint status: `{selected_snapshot['selection_status']}`",
            f"- selected Candidate B checkpoint: `{candidate_ckpt}`",
            f"- best Candidate B global delta: `{best_global_delta:.6f}`",
            f"- best Candidate B global coverage: `{best_global_row['metrics']['plastic_coverage']:.6f}`",
        ],
    )
    _write_phase_report(
        exp_b_report,
        "Hybrid Gate Redesign Experiment B",
        [
            "Global `d_geom` min-term attribution audit on dev validation.",
            "",
            "## Artifacts",
            "",
            f"- row table: `{exp_b_dir / 'candidate_b_min_term_rows_val.csv'}`",
            f"- grouped summary: `{exp_b_dir / 'candidate_b_min_term_summary_val.csv'}`",
            "",
            f"- selected global delta for audit summaries: `{best_global_delta:.6f}`",
        ],
    )
    _write_phase_report(
        exp_c_report,
        "Hybrid Gate Redesign Experiment C",
        [
            "Oracle branch-conditioned gate on dev validation.",
            "",
            "## Result",
            "",
            f"- best oracle coverage: `{oracle_selection['best_row']['metrics']['plastic_coverage']:.6f}`",
            f"- best global coverage: `{best_global_row['metrics']['plastic_coverage']:.6f}`",
            f"- absolute improvement: `{oracle_improvement:.6f}`",
        ],
    )
    if oracle_improvement >= 0.05:
        _write_phase_report(
            exp_d_report,
            "Hybrid Gate Redesign Experiment D",
            [
                "Predicted branch-conditioned gate on dev validation.",
                "",
                f"- selected policy summary: `{output_root / 'exp_d_predicted_branch' / 'predicted_selection.json'}`",
            ],
        )
    if rejector_summary is not None:
        _write_phase_report(
            exp_e_report,
            "Hybrid Gate Redesign Experiment E",
            [
                "Learned rejector on Candidate B runtime features.",
                "",
                f"- rejector summary: `{Path(rejector_summary['run_dir']) / 'summary.json'}`",
                f"- selected policy summary: `{output_root / 'exp_e_rejector' / 'rejector_selection.json'}`",
            ],
        )

    execution_lines = [
        "This document tracks the redesign cycle execution status against `report4.md`.",
        "",
        "## Completed",
        "",
        f"- [x] Fresh grouped dataset and frozen panels: `{dataset_path}` / `{panel_path}`",
        f"- [x] Validation-first baseline retraining: `{baseline_ckpt}`",
        f"- [x] Candidate B retraining and snapshot selection: `{candidate_ckpt}` via `{snapshot_selection_json}`",
        f"- [x] Experiment A dense global frontier: `{exp_a_report}`",
        f"- [x] Experiment B min-term attribution audit: `{exp_b_report}`",
        f"- [x] Experiment C oracle branch gate: `{exp_c_report}`",
        f"- [{'x' if oracle_improvement >= 0.05 else ' '}] Experiment D predicted-branch gate",
        f"- [{'x' if rejector_summary is not None else ' '}] Experiment E learned rejector",
        f"- [x] Final test freeze and summary: `{output_root / 'final' / 'final_test_summary.json'}`",
        "",
        "## Outcome",
        "",
        f"- validation-feasible baseline global delta: `{baseline_feasible_delta:.6f}`",
        f"- best global-gate validation coverage: `{best_global_row['metrics']['plastic_coverage']:.6f}`",
        f"- oracle improvement over global: `{oracle_improvement:.6f}`",
        f"- redesigned final-test coverage improvement over global: `{redesign_improvement if np.isfinite(redesign_improvement) else float('nan'):.6f}`",
        f"- final redesign passes promotion bar: `{bool(redesign_passes_test)}`",
    ]
    _write_phase_report(execution_doc, "Hybrid Gate Redesign Execution 20260324", execution_lines)

    taskgiver_lines = [
        "Prepared for taskgiver review. This report is self-contained and summarizes the redesign cycle, the exact training setup, the gating experiments, and the final recommendation.",
        "",
        "## Executed Setup",
        "",
        f"- source full export: `{full_export}`",
        f"- output root: `{output_root}`",
        f"- split seed: `{args.seed}`",
        f"- call-level split fractions: `{(args.train_frac, args.val_frac, args.test_frac)}`",
        f"- samples per call: `{args.samples_per_call}`",
        "",
        "## Network And Training Setup",
        "",
        f"- baseline model: `raw_branch`, width `{args.baseline_width}`, depth `{args.baseline_depth}`, dropout `0.0`, staged Adam/LBFGS schedule, validation-first checkpoint selection",
        f"- Candidate B model: `trial_principal_geom_plastic_branch_residual`, width `{args.candidate_width}`, depth `{args.candidate_depth}`, dropout `0.0`",
        f"- Candidate B pretrain rows: `{args.pretrain_train_rows}` train / `{args.pretrain_val_rows}` val / `{args.pretrain_test_rows}` test",
        f"- Candidate B fine-tune rows: `{args.finetune_train_rows}` train with `{args.finetune_real_fraction:.2f}` real fraction",
        "",
        "## Validation Ceilings",
        "",
        f"- broad plastic MAE ceiling: `{ceilings['broad_plastic_mae_ceiling']:.6f}`",
        f"- hard plastic MAE ceiling: `{ceilings['hard_plastic_mae_ceiling']:.6f}`",
        f"- hard p95 principal ceiling: `{ceilings['hard_p95_principal_ceiling']:.6f}`",
        f"- yield violation p95 ceiling: `{ceilings['yield_violation_p95_ceiling']:.6f}`",
        "",
        "## Main Results",
        "",
        "- Candidate B checkpoint was selected from a validation frontier sweep over `best.pt`, `last.pt`, and saved snapshots.",
        f"- selected Candidate B checkpoint label: `{selected_snapshot['checkpoint_label']}`",
        f"- selected Candidate B checkpoint: `{candidate_ckpt}`",
        f"- validation-feasible baseline global test delta: `{baseline_feasible_delta:.6f}`",
        f"- validation-feasible baseline global final-test coverage: `{baseline_feasible_test_metrics['plastic_coverage']:.6f}`",
        f"- validation-feasible baseline global final-test yield violation p95: `{baseline_feasible_test_metrics['yield_violation_p95']:.6f}`",
        f"- best feasible Candidate B global gate delta: `{best_global_delta:.6f}`",
        f"- best feasible Candidate B global validation coverage: `{best_global_row['metrics']['plastic_coverage']:.6f}`",
        f"- best feasible Candidate B global validation hard p95: `{best_global_row['metrics']['hard_p95_principal']:.6f}`",
        f"- oracle improvement over global coverage: `{oracle_improvement:.6f}`",
        f"- final redesigned policy kind: `{redesigned_test_label}`",
        f"- final-test baseline-feasible gate coverage: `{baseline_feasible_test_metrics['plastic_coverage']:.6f}`",
        f"- final-test global coverage: `{global_test_metrics['plastic_coverage']:.6f}`",
        f"- final-test redesigned coverage: `{redesigned_test_metrics['plastic_coverage']:.6f}`" if redesigned_test_metrics is not None else "- final-test redesigned coverage: `not available`",
        f"- final-test coverage improvement over global: `{redesign_improvement if np.isfinite(redesign_improvement) else float('nan'):.6f}`",
        "",
        "## Recommendation",
        "",
        "- Continue only if the redesigned deployable gate clears the final-test promotion rule.",
        f"- Current redesign promotion result: `{bool(redesign_passes_test)}`",
        "- If promotion failed, pivot to return-mapping-assisted acceleration rather than more direct-surrogate polishing.",
    ]
    _write_phase_report(taskgiver_report, "Hybrid Gate Redesign Taskgiver Report", taskgiver_lines)

    print(
        json.dumps(
            _json_safe(
                {
                    "dataset_path": str(dataset_path),
                    "panel_path": str(panel_path),
                    "baseline_checkpoint": str(baseline_ckpt),
                    "candidate_checkpoint": str(candidate_ckpt),
                    "best_global_delta": best_global_delta,
                    "oracle_improvement": oracle_improvement,
                    "best_redesign_kind": best_redesign_kind,
                    "redesign_final_improvement": redesign_improvement if np.isfinite(redesign_improvement) else None,
                    "redesign_passes_test": redesign_passes_test,
                    "taskgiver_report": str(taskgiver_report),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
