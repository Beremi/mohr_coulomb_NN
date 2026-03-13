#!/usr/bin/env python
"""Adaptive cover-layer-only training run for a 1024x6 raw-branch network."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.models import Standardizer, build_model
from mc_surrogate.mohr_coulomb import BRANCH_NAMES
from mc_surrogate.training import (
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    evaluate_checkpoint_on_dataset,
    set_seed,
)
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


def _load_cover_layer_cyclic_module() -> Any:
    module_path = ROOT / "scripts" / "experiments" / "cover_layer_cyclic_sweep.py"
    spec = importlib.util.spec_from_file_location("cover_layer_cyclic_sweep_mod", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CYCLIC_HELPERS = _load_cover_layer_cyclic_module()


@dataclass(frozen=True)
class RunSpec:
    name: str
    width: int
    depth: int
    weight_decay: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-primary",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5",
    )
    parser.add_argument(
        "--real-cross",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_cross.h5",
    )
    parser.add_argument(
        "--hybrid-dataset",
        default="experiment_runs/real_sim/per_material_hybrid_hardcases_20260312/hybrid_datasets/cover_layer_hybrid.h5",
        help="Used only for coverage diagnostics against the old augmented training set.",
    )
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_adaptive_20260312")
    parser.add_argument("--report-md", default="docs/cover_layer_adaptive_w1024_d6.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--base-lr", type=float, default=1.0e-3)
    parser.add_argument("--cycle-lr-decay", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--stage-patience", type=int, default=20)
    parser.add_argument("--improvement-rel-tol", type=float, default=1.0e-4)
    parser.add_argument("--improvement-abs-tol", type=float, default=1.0e-7)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=8201)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _stage_initial_lr(*, cycle_idx: int, batch_size: int, base_lr: float, cycle_lr_decay: float) -> float:
    cycle_lr = base_lr * (cycle_lr_decay ** max(cycle_idx - 1, 0))
    batch_scale = math.sqrt(64.0 / float(batch_size))
    return cycle_lr * batch_scale


def _format_runtime(seconds: float) -> str:
    total_seconds = int(max(seconds, 0.0))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _maybe_improved(current: float, best: float, rel_tol: float, abs_tol: float) -> bool:
    if not math.isfinite(best):
        return True
    threshold = max(abs_tol, rel_tol * max(abs(best), 1.0))
    return current < best - threshold


def _plot_branch_accuracy(history_csv: Path, output_path: Path) -> Path:
    rows = list(CYCLIC_HELPERS.csv.DictReader(history_csv.open("r", encoding="utf-8")))
    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    stage_names = [r["stage_name"] for r in rows]
    boundaries: list[tuple[int, str]] = []
    prev = None
    for e, s in zip(epoch, stage_names):
        if s != prev:
            boundaries.append((int(e), s))
            prev = s

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(epoch, arr("train_branch_accuracy"), label="train")
    ax.plot(epoch, arr("real_val_branch_accuracy"), label="real val")
    ax.plot(epoch, arr("real_test_branch_accuracy"), label="real test")
    ax.plot(epoch, arr("cross_test_branch_accuracy"), label="cross test")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("branch accuracy")
    ax.set_xlabel("global epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for x, _ in boundaries:
        ax.axvline(x, color="k", linestyle="--", alpha=0.15)
    ymax = 0.99
    for x, label in boundaries:
        ax.text(x + 1, ymax, label, rotation=90, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _load_prior_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _train_one(
    *,
    exact_dataset: Path,
    real_primary: Path,
    real_cross: Path,
    run_dir: Path,
    spec: RunSpec,
    device: torch.device,
    eval_batch_size: int,
    branch_loss_weight: float,
    grad_clip: float,
    cycles: int,
    batch_sizes: list[int],
    base_lr: float,
    cycle_lr_decay: float,
    min_lr: float,
    plateau_patience: int,
    stage_patience: int,
    improvement_rel_tol: float,
    improvement_abs_tol: float,
) -> dict[str, Any]:
    set_seed(spec.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    train_arrays = _load_split_for_training(str(exact_dataset), "train", "raw_branch")
    synth_val_arrays = _load_split_for_training(str(exact_dataset), "val", "raw_branch")
    real_val_arrays = _load_split_for_training(str(real_primary), "val", "raw_branch")
    real_test_arrays = _load_split_for_training(str(real_primary), "test", "raw_branch")
    cross_test_arrays = _load_split_for_training(str(real_cross), "test", "raw_branch")

    x_scaler = Standardizer.from_array(train_arrays["features"])
    y_scaler = Standardizer.from_array(train_arrays["target"])
    train_ds = _build_tensor_dataset(train_arrays, x_scaler, y_scaler)
    synth_val_loader = CYCLIC_HELPERS._make_eval_loader(synth_val_arrays, x_scaler, y_scaler, eval_batch_size)
    real_val_loader = CYCLIC_HELPERS._make_eval_loader(real_val_arrays, x_scaler, y_scaler, eval_batch_size)
    real_test_loader = CYCLIC_HELPERS._make_eval_loader(real_test_arrays, x_scaler, y_scaler, eval_batch_size)
    cross_test_loader = CYCLIC_HELPERS._make_eval_loader(cross_test_arrays, x_scaler, y_scaler, eval_batch_size)

    model = build_model(
        "raw_branch",
        input_dim=train_arrays["features"].shape[1],
        width=spec.width,
        depth=spec.depth,
        dropout=0.0,
    ).to(device)
    metadata = {
        "config": {
            "dataset": str(exact_dataset),
            "real_primary": str(real_primary),
            "real_cross": str(real_cross),
            "run_dir": str(run_dir),
            "model_kind": "raw_branch",
            "width": spec.width,
            "depth": spec.depth,
            "dropout": 0.0,
            "seed": spec.seed,
            "weight_decay": spec.weight_decay,
            "cycles": cycles,
            "batch_sizes": batch_sizes,
            "base_lr": base_lr,
            "cycle_lr_decay": cycle_lr_decay,
            "min_lr": min_lr,
            "plateau_patience": plateau_patience,
            "stage_patience": stage_patience,
            "improvement_rel_tol": improvement_rel_tol,
            "improvement_abs_tol": improvement_abs_tol,
            "selection_metric": "real_val_stress_mse",
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_csv = run_dir / "history.csv"
    CYCLIC_HELPERS._write_history_header(history_csv)
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    best_real_val_stress_mse = float("inf")
    best_epoch = 0
    global_epoch = 0
    total_start = time.perf_counter()

    for cycle_idx in range(1, cycles + 1):
        for batch_size in batch_sizes:
            stage_name = f"cycle{cycle_idx}_bs{batch_size}"
            stage_start = time.perf_counter()
            stage_lr = _stage_initial_lr(
                cycle_idx=cycle_idx,
                batch_size=batch_size,
                base_lr=base_lr,
                cycle_lr_decay=cycle_lr_decay,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=spec.weight_decay)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

            stage_epoch = 0
            stage_bad_epochs = 0
            lr_bad_epochs = 0
            best_stage_train_loss = float("inf")
            best_stage_epoch = 0

            _log_message(
                log_path,
                (
                    f"[stage-start] cycle={cycle_idx}/{cycles} batch={batch_size} lr={stage_lr:.3e} "
                    f"global_epoch={global_epoch} runtime={_format_runtime(time.perf_counter() - total_start)}"
                ),
            )

            while True:
                stage_epoch += 1
                global_epoch += 1

                train_metrics = _epoch_loop(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                synth_val_metrics = _epoch_loop(
                    model=model,
                    loader=synth_val_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                real_val_metrics = _epoch_loop(
                    model=model,
                    loader=real_val_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                real_test_metrics = _epoch_loop(
                    model=model,
                    loader=real_test_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                cross_test_metrics = _epoch_loop(
                    model=model,
                    loader=cross_test_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )

                checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
                torch.save(checkpoint, last_ckpt)

                current_lr = optimizer.param_groups[0]["lr"]
                is_best = False
                if real_val_metrics["stress_mse"] < best_real_val_stress_mse:
                    best_real_val_stress_mse = real_val_metrics["stress_mse"]
                    best_epoch = global_epoch
                    CYCLIC_HELPERS._save_checkpoint(best_ckpt, model, metadata)
                    is_best = True

                CYCLIC_HELPERS._append_history_row(
                    history_csv,
                    epoch=global_epoch,
                    stage_name=stage_name,
                    stage_kind="adam",
                    batch_size=batch_size,
                    lr=current_lr,
                    train_metrics=train_metrics,
                    synth_val_metrics=synth_val_metrics,
                    real_val_metrics=real_val_metrics,
                    real_test_metrics=real_test_metrics,
                    cross_test_metrics=cross_test_metrics,
                    best_real_val_stress_mse=best_real_val_stress_mse,
                    is_best=is_best,
                )

                runtime = time.perf_counter() - total_start
                _log_message(
                    log_path,
                    (
                        f"[adam] epoch={global_epoch} stage_epoch={stage_epoch} cycle={cycle_idx}/{cycles} "
                        f"batch={batch_size} lr={current_lr:.3e} runtime={_format_runtime(runtime)} "
                        f"train_loss={train_metrics['loss']:.6f} real_val_stress_mse={real_val_metrics['stress_mse']:.6f} "
                        f"real_test_stress_mse={real_test_metrics['stress_mse']:.6f} best_real_val={best_real_val_stress_mse:.6f}"
                    ),
                )

                if _maybe_improved(
                    train_metrics["loss"],
                    best_stage_train_loss,
                    rel_tol=improvement_rel_tol,
                    abs_tol=improvement_abs_tol,
                ):
                    best_stage_train_loss = train_metrics["loss"]
                    best_stage_epoch = stage_epoch
                    stage_bad_epochs = 0
                    lr_bad_epochs = 0
                else:
                    stage_bad_epochs += 1
                    lr_bad_epochs += 1
                    if lr_bad_epochs >= plateau_patience and current_lr > min_lr * (1.0 + 1.0e-12):
                        new_lr = max(current_lr * 0.5, min_lr)
                        for group in optimizer.param_groups:
                            group["lr"] = new_lr
                        lr_bad_epochs = 0
                        _log_message(
                            log_path,
                            (
                                f"[lr-drop] epoch={global_epoch} cycle={cycle_idx}/{cycles} batch={batch_size} "
                                f"old_lr={current_lr:.3e} new_lr={new_lr:.3e} "
                                f"runtime={_format_runtime(time.perf_counter() - total_start)}"
                            ),
                        )

                if stage_bad_epochs >= stage_patience:
                    _log_message(
                        log_path,
                        (
                            f"[stage-stop] cycle={cycle_idx}/{cycles} batch={batch_size} stage_epochs={stage_epoch} "
                            f"best_stage_epoch={best_stage_epoch} best_stage_train_loss={best_stage_train_loss:.6f} "
                            f"stage_runtime={_format_runtime(time.perf_counter() - stage_start)} "
                            f"total_runtime={_format_runtime(time.perf_counter() - total_start)}"
                        ),
                    )
                    break

    total_runtime = time.perf_counter() - total_start
    history_plot = CYCLIC_HELPERS._plot_history(history_csv, run_dir / "history_log.png")
    branch_plot = _plot_branch_accuracy(history_csv, run_dir / "branch_accuracy.png")
    primary_eval = evaluate_checkpoint_on_dataset(best_ckpt, real_primary, split="test", device=str(device), batch_size=eval_batch_size)
    cross_eval = evaluate_checkpoint_on_dataset(best_ckpt, real_cross, split="test", device=str(device), batch_size=eval_batch_size)

    primary_dir = run_dir / "eval_primary"
    cross_dir = run_dir / "eval_cross"
    primary_dir.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(primary_eval["arrays"]["stress"], primary_eval["predictions"]["stress"], primary_dir / "parity_stress.png", label="stress")
    error_histogram(primary_eval["predictions"]["stress"] - primary_eval["arrays"]["stress"], primary_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in primary_eval["metrics"]:
        branch_confusion_plot(primary_eval["metrics"]["branch_confusion"], primary_dir / "branch_confusion.png")
    parity_plot(cross_eval["arrays"]["stress"], cross_eval["predictions"]["stress"], cross_dir / "parity_stress.png", label="stress")
    error_histogram(cross_eval["predictions"]["stress"] - cross_eval["arrays"]["stress"], cross_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in cross_eval["metrics"]:
        branch_confusion_plot(cross_eval["metrics"]["branch_confusion"], cross_dir / "branch_confusion.png")

    summary = {
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "best_real_val_stress_mse": best_real_val_stress_mse,
        "history_csv": str(history_csv),
        "history_plot": str(history_plot),
        "branch_plot": str(branch_plot),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "log_path": str(log_path),
        "elapsed_seconds": total_runtime,
        "elapsed_hms": _format_runtime(total_runtime),
        "primary_metrics": primary_eval["metrics"],
        "cross_metrics": cross_eval["metrics"],
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _write_report(
    report_md: Path,
    *,
    exact_dataset_path: Path,
    coverage: dict[str, Any],
    mex_check: dict[str, Any],
    summary: dict[str, Any],
    config: argparse.Namespace,
) -> None:
    def rel_repo(path: str | Path) -> str:
        return Path(path).resolve().relative_to(ROOT).as_posix()

    prior_mae = _load_prior_summary(
        ROOT
        / "experiment_runs"
        / "real_sim"
        / "cover_layer_cyclic_20260312"
        / "cover_raw_branch_w384_d6"
        / "summary.json"
    )
    prior_rmse = _load_prior_summary(
        ROOT
        / "experiment_runs"
        / "real_sim"
        / "cover_layer_cyclic_20260312"
        / "cover_raw_branch_w512_d6"
        / "summary.json"
    )

    p = summary["primary_metrics"]
    c = summary["cross_metrics"]
    report_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cover Layer Adaptive 1024x6 Run",
        "",
        "This run keeps the cover-layer-only exact-domain setup from the previous sweep,",
        "but replaces the short fixed stage schedule with a longer adaptive cycle schedule.",
        "",
        "## Configuration",
        "",
        f"- network: `{summary['spec']['name']}` (`width={summary['spec']['width']}`, `depth={summary['spec']['depth']}`)",
        f"- cycles: `{config.cycles}`",
        f"- batch sizes: `{config.batch_sizes}`",
        f"- stage initial LR formula: `base_lr * cycle_lr_decay^(cycle-1) * sqrt(64 / batch_size)`",
        f"- base LR: `{config.base_lr:.3e}`",
        f"- cycle LR decay: `{config.cycle_lr_decay:.3f}`",
        f"- min LR: `{config.min_lr:.3e}`",
        f"- LR reduction rule: halve LR after `{config.plateau_patience}` bad epochs on train loss",
        f"- stage advance rule: move to the next batch size after `{config.stage_patience}` bad epochs on train loss",
        f"- runtime: `{summary['elapsed_hms']}`",
        f"- train log: `{summary['log_path']}`",
        "",
        f"Exact-domain dataset: `{exact_dataset_path}`",
        "",
    ]
    if summary.get("interrupted"):
        lines.extend(
            [
                "Training was interrupted after the later stages stopped improving the best saved checkpoint.",
                "",
            ]
        )
    lines.extend(
        [
        "## Coverage Validation",
        "",
        "The same cover-layer domain check still holds: the training inputs do cover the real cover-layer test inputs.",
        "The main mismatch in the older synthetic set was target-tail inflation, not input-domain miss.",
        "",
        f"- exact-domain relabel vs stored real MAE: `{coverage['exact_domain_attrs']['exact_vs_real_mae']:.6e}`",
        f"- exact-domain relabel vs stored real RMSE: `{coverage['exact_domain_attrs']['exact_vs_real_rmse']:.6e}`",
        f"- exact-domain relabel vs stored real max abs: `{coverage['exact_domain_attrs']['exact_vs_real_max_abs']:.6e}`",
        "",
        "| Dataset | Train N | Out-of-box sample rate vs real test | NN z mean | Stress q95 | Stress q995 |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in coverage["datasets"].items():
        lines.append(
            f"| {name} | {stats['n_train']} | {stats['sample_out_of_box_rate']:.4f} | "
            f"{stats['nn_z_mean']:.4f} | {stats['stress_mag_q95']:.4f} | {stats['stress_mag_q995']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## MEX Cross-Check",
            "",
            f"- samples: `{mex_check['samples']}`",
            f"- C-vs-Python MAE: `{mex_check['mae']:.6e}`",
            f"- C-vs-Python RMSE: `{mex_check['rmse']:.6e}`",
            f"- C-vs-Python max abs: `{mex_check['max_abs']:.6e}`",
            "",
            "## Results",
            "",
            "| Split | Stress MAE | Stress RMSE | Stress Max Abs | Branch Acc |",
            "|---|---:|---:|---:|---:|",
            f"| primary | {p['stress_mae']:.4f} | {p['stress_rmse']:.4f} | {p['stress_max_abs']:.4f} | {p.get('branch_accuracy', float('nan')):.4f} |",
            f"| cross | {c['stress_mae']:.4f} | {c['stress_rmse']:.4f} | {c['stress_max_abs']:.4f} | {c.get('branch_accuracy', float('nan')):.4f} |",
            "",
            f"- best epoch by real-val stress MSE: `{summary['best_epoch']}`",
            f"- best real-val stress MSE: `{summary['best_real_val_stress_mse']:.6f}`",
        ]
    )

    if prior_mae is not None:
        lines.extend(
            [
                "",
                "Primary-split comparison to the previous cover-layer sweep best-by-MAE (`w384 d6`):",
                "",
                f"- previous primary MAE / RMSE / max abs: `{prior_mae['primary_metrics']['stress_mae']:.4f}` / "
                f"`{prior_mae['primary_metrics']['stress_rmse']:.4f}` / `{prior_mae['primary_metrics']['stress_max_abs']:.4f}`",
                f"- current primary MAE / RMSE / max abs: `{p['stress_mae']:.4f}` / `{p['stress_rmse']:.4f}` / `{p['stress_max_abs']:.4f}`",
            ]
        )
    if prior_rmse is not None:
        lines.extend(
            [
                "",
                "Cross-split comparison to the previous cover-layer sweep best-by-RMSE (`w512 d6`):",
                "",
                f"- previous cross MAE / RMSE / max abs: `{prior_rmse['cross_metrics']['stress_mae']:.4f}` / "
                f"`{prior_rmse['cross_metrics']['stress_rmse']:.4f}` / `{prior_rmse['cross_metrics']['stress_max_abs']:.4f}`",
                f"- current cross MAE / RMSE / max abs: `{c['stress_mae']:.4f}` / `{c['stress_rmse']:.4f}` / `{c['stress_max_abs']:.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## History",
            "",
            f"![adaptive training history](../{rel_repo(summary['history_plot'])})",
            "",
            f"![adaptive branch accuracy](../{rel_repo(summary['branch_plot'])})",
            "",
            "## Validation Plots",
            "",
            "Primary real test:",
            "",
            f"![primary parity](../{rel_repo(Path(summary['history_plot']).parent / 'eval_primary' / 'parity_stress.png')})",
            "",
            f"![primary error histogram](../{rel_repo(Path(summary['history_plot']).parent / 'eval_primary' / 'stress_error_hist.png')})",
            "",
            f"![primary branch confusion](../{rel_repo(Path(summary['history_plot']).parent / 'eval_primary' / 'branch_confusion.png')})",
            "",
            "Cross real test:",
            "",
            f"![cross parity](../{rel_repo(Path(summary['history_plot']).parent / 'eval_cross' / 'parity_stress.png')})",
            "",
            f"![cross error histogram](../{rel_repo(Path(summary['history_plot']).parent / 'eval_cross' / 'stress_error_hist.png')})",
            "",
            f"![cross branch confusion](../{rel_repo(Path(summary['history_plot']).parent / 'eval_cross' / 'branch_confusion.png')})",
            "",
        ]
    )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    spec = RunSpec(
        name=f"cover_raw_branch_w{args.width}_d{args.depth}_adaptive",
        width=args.width,
        depth=args.depth,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    real_primary = Path(args.real_primary)
    real_cross = Path(args.real_cross)
    hybrid_dataset = Path(args.hybrid_dataset)
    exact_dataset_path, exact_meta = CYCLIC_HELPERS._build_exact_domain_dataset(
        real_primary, output_root / "cover_layer_exact_domain.h5"
    )
    coverage = CYCLIC_HELPERS._coverage_summary(real_primary, hybrid_dataset, exact_dataset_path)
    coverage["exact_domain_attrs"] = exact_meta
    mex_check = CYCLIC_HELPERS._mex_kernel_check(real_primary)

    summary = _train_one(
        exact_dataset=exact_dataset_path,
        real_primary=real_primary,
        real_cross=real_cross,
        run_dir=output_root / spec.name,
        spec=spec,
        device=device,
        eval_batch_size=args.eval_batch_size,
        branch_loss_weight=args.branch_loss_weight,
        grad_clip=args.grad_clip,
        cycles=args.cycles,
        batch_sizes=args.batch_sizes,
        base_lr=args.base_lr,
        cycle_lr_decay=args.cycle_lr_decay,
        min_lr=args.min_lr,
        plateau_patience=args.plateau_patience,
        stage_patience=args.stage_patience,
        improvement_rel_tol=args.improvement_rel_tol,
        improvement_abs_tol=args.improvement_abs_tol,
    )

    _write_report(
        Path(args.report_md),
        exact_dataset_path=exact_dataset_path,
        coverage=coverage,
        mex_check=mex_check,
        summary=summary,
        config=args,
    )
    (output_root / "coverage_summary.json").write_text(json.dumps(_json_safe(coverage), indent=2), encoding="utf-8")
    (output_root / "mex_check.json").write_text(json.dumps(_json_safe(mex_check), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "report_md": str(Path(args.report_md)),
                "device": str(device),
                "best_checkpoint": summary["best_checkpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
