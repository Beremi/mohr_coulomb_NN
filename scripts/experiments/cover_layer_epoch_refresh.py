#!/usr/bin/env python
"""Train cover-layer surrogates with freshly regenerated synthetic data every epoch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import REDUCED_MATERIAL_COLUMNS, SPLIT_NAMES, SPLIT_TO_ID
from mc_surrogate.models import (
    Standardizer,
    build_model,
    build_raw_features,
    compute_trial_stress,
    spectral_decomposition_from_strain,
)
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, BRANCH_TO_ID, constitutive_update_3d
from mc_surrogate.sampling import (
    _principal_direction_template,
    _principal_to_global_engineering_strain,
    _scale_factor_for_branch,
    _yield_scale_from_direction,
    random_rotation_matrices,
)
from mc_surrogate.training import (
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    evaluate_checkpoint_on_dataset,
    set_seed,
)
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


@dataclass(frozen=True)
class SyntheticEpochConfig:
    n_samples: int
    candidate_batch: int = 4096
    max_abs_principal_strain: float | None = None
    branch_fractions: tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2)


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
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_epoch_refresh_20260312")
    parser.add_argument("--report-md", default="docs/cover_layer_epoch_refresh.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=8301)
    parser.add_argument("--train-size", type=int, default=65536)
    parser.add_argument("--synthetic-test-size", type=int, default=16384)
    parser.add_argument("--candidate-batch", type=int, default=4096)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--base-lr", type=float, default=1.0e-4)
    parser.add_argument("--cycle-lr-decay", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--stage-patience", type=int, default=15)
    parser.add_argument("--improvement-rel-tol", type=float, default=1.0e-4)
    parser.add_argument("--improvement-abs-tol", type=float, default=1.0e-7)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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


def _format_runtime(seconds: float) -> str:
    total_seconds = int(max(seconds, 0.0))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _stage_initial_lr(*, cycle_idx: int, batch_size: int, base_lr: float, cycle_lr_decay: float) -> float:
    cycle_lr = base_lr * (cycle_lr_decay ** max(cycle_idx - 1, 0))
    batch_scale = math.sqrt(64.0 / float(batch_size))
    return cycle_lr * batch_scale


def _maybe_improved(current: float, best: float, rel_tol: float, abs_tol: float) -> bool:
    if not math.isfinite(best):
        return True
    threshold = max(abs_tol, rel_tol * max(abs(best), 1.0))
    return current < best - threshold


def _load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
    return arrays


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.attrs["branch_names_json"] = json.dumps(BRANCH_NAMES)
        f.attrs["reduced_material_columns_json"] = json.dumps(REDUCED_MATERIAL_COLUMNS)
        f.attrs["split_names_json"] = json.dumps(SPLIT_NAMES)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value
    return path


def _distribution_summary(strain_eng: np.ndarray, stress: np.ndarray, branch_id: np.ndarray) -> dict[str, Any]:
    strain_principal, _ = spectral_decomposition_from_strain(strain_eng)
    max_abs_principal = np.max(np.abs(strain_principal), axis=1)
    stress_mag = np.linalg.norm(stress, axis=1)
    return {
        "n_samples": int(strain_eng.shape[0]),
        "max_abs_principal_q50": float(np.quantile(max_abs_principal, 0.5)),
        "max_abs_principal_q95": float(np.quantile(max_abs_principal, 0.95)),
        "max_abs_principal_q995": float(np.quantile(max_abs_principal, 0.995)),
        "stress_mag_q50": float(np.quantile(stress_mag, 0.5)),
        "stress_mag_q95": float(np.quantile(stress_mag, 0.95)),
        "stress_mag_q995": float(np.quantile(stress_mag, 0.995)),
        "branch_counts": {name: int(np.sum(branch_id == i)) for i, name in enumerate(BRANCH_NAMES)},
    }


def _estimate_principal_cap(real_primary_path: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    arrays = _load_arrays(real_primary_path)
    train_mask = arrays["split_id"] == SPLIT_TO_ID["train"]
    material_pool = arrays["material_reduced"][train_mask].astype(np.float32)
    train_strain = arrays["strain_eng"][train_mask].astype(np.float32)
    train_branch = arrays["branch_id"][train_mask].astype(np.int8)
    train_stress = arrays["stress"][train_mask].astype(np.float32)

    strain_principal, _ = spectral_decomposition_from_strain(train_strain)
    principal_cap = float(np.quantile(np.max(np.abs(strain_principal), axis=1), 0.995) * 1.10)
    info = {
        "n_material_rows": int(material_pool.shape[0]),
        "n_unique_reduced_rows": int(np.unique(np.round(material_pool, decimals=8), axis=0).shape[0]),
        "principal_cap": principal_cap,
        "real_train_distribution": _distribution_summary(train_strain, train_stress, train_branch),
    }
    return material_pool, principal_cap, info


def _desired_branch_counts(cfg: SyntheticEpochConfig) -> dict[str, int]:
    fracs = np.asarray(cfg.branch_fractions, dtype=float)
    if fracs.shape != (len(BRANCH_NAMES),):
        raise ValueError("branch_fractions must have one entry per branch.")
    if not np.isclose(float(fracs.sum()), 1.0):
        raise ValueError("branch_fractions must sum to 1.")
    counts: dict[str, int] = {}
    assigned = 0
    for name, frac in zip(BRANCH_NAMES[:-1], fracs[:-1]):
        count = int(round(float(frac) * cfg.n_samples))
        counts[name] = count
        assigned += count
    counts[BRANCH_NAMES[-1]] = cfg.n_samples - assigned
    return counts


def generate_branch_balanced_from_material_pool(
    *,
    material_pool: np.ndarray,
    cfg: SyntheticEpochConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    desired_counts = _desired_branch_counts(cfg)
    chunks: list[dict[str, np.ndarray]] = []

    for branch in BRANCH_NAMES:
        remaining = desired_counts[branch]
        attempts = 0
        while remaining > 0:
            attempts += 1
            if attempts > 4000:
                raise RuntimeError(f"Failed to realize enough samples for branch {branch!r}.")
            batch = max(cfg.candidate_batch, remaining * 2)
            idx_mat = rng.integers(0, material_pool.shape[0], size=batch)
            mat = material_pool[idx_mat]
            direction = _principal_direction_template(branch, batch, rng)
            alpha_yield = _yield_scale_from_direction(
                direction,
                mat[:, 0],
                mat[:, 1],
                mat[:, 2],
                mat[:, 4],
            )
            factor = _scale_factor_for_branch(branch, batch, rng)
            principal_strain = direction * alpha_yield[:, None] * factor[:, None]
            rotations = random_rotation_matrices(batch, rng)
            strain_eng = _principal_to_global_engineering_strain(principal_strain, rotations)

            response = constitutive_update_3d(
                strain_eng,
                c_bar=mat[:, 0],
                sin_phi=mat[:, 1],
                shear=mat[:, 2],
                bulk=mat[:, 3],
                lame=mat[:, 4],
            )
            matched = response.branch_id == BRANCH_TO_ID[branch]
            if cfg.max_abs_principal_strain is not None:
                matched &= np.max(np.abs(principal_strain), axis=1) <= cfg.max_abs_principal_strain
            take = np.flatnonzero(matched)
            if take.size == 0:
                continue
            take = take[:remaining]
            chunks.append(
                {
                    "strain_eng": strain_eng[take].astype(np.float32),
                    "stress": response.stress[take].astype(np.float32),
                    "material_reduced": mat[take].astype(np.float32),
                    "branch_id": response.branch_id[take].astype(np.int8),
                    "eigvecs": response.eigvecs[take].astype(np.float32),
                    "stress_principal": response.stress_principal[take].astype(np.float32),
                }
            )
            remaining -= take.size

    arrays: dict[str, np.ndarray] = {}
    keys = chunks[0].keys()
    for key in keys:
        arrays[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    perm = rng.permutation(arrays["strain_eng"].shape[0])
    arrays = {key: value[perm] for key, value in arrays.items()}
    return arrays


def _split_arrays_for_raw_branch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "features": build_raw_features(arrays["strain_eng"], arrays["material_reduced"]).astype(np.float32),
        "target": arrays["stress"].astype(np.float32),
        "stress_true": arrays["stress"].astype(np.float32),
        "branch_id": arrays["branch_id"].astype(np.int64),
        "eigvecs": arrays["eigvecs"].astype(np.float32),
        "trial_stress": compute_trial_stress(arrays["strain_eng"], arrays["material_reduced"]).astype(np.float32),
    }


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "stage_name",
                "batch_size",
                "train_seed",
                "lr",
                "train_loss",
                "train_stress_mse",
                "real_val_loss",
                "real_val_stress_mse",
                "real_test_loss",
                "real_test_stress_mse",
                "synthetic_test_loss",
                "synthetic_test_stress_mse",
                "train_branch_accuracy",
                "real_val_branch_accuracy",
                "real_test_branch_accuracy",
                "synthetic_test_branch_accuracy",
                "best_real_val_stress_mse",
                "is_best",
            ]
        )


def _append_history_row(
    path: Path,
    *,
    epoch: int,
    stage_name: str,
    batch_size: int,
    train_seed: int,
    lr: float,
    train_metrics: dict[str, float],
    real_val_metrics: dict[str, float],
    real_test_metrics: dict[str, float],
    synthetic_test_metrics: dict[str, float],
    best_real_val_stress_mse: float,
    is_best: bool,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                stage_name,
                batch_size,
                train_seed,
                lr,
                train_metrics["loss"],
                train_metrics["stress_mse"],
                real_val_metrics["loss"],
                real_val_metrics["stress_mse"],
                real_test_metrics["loss"],
                real_test_metrics["stress_mse"],
                synthetic_test_metrics["loss"],
                synthetic_test_metrics["stress_mse"],
                train_metrics["branch_accuracy"],
                real_val_metrics["branch_accuracy"],
                real_test_metrics["branch_accuracy"],
                synthetic_test_metrics["branch_accuracy"],
                best_real_val_stress_mse,
                1 if is_best else 0,
            ]
        )


def _plot_history(history_csv: Path, output_path: Path) -> Path:
    rows = list(csv.DictReader(history_csv.open("r", encoding="utf-8")))
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

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(epoch, arr("train_loss"), label="train")
    axes[0].plot(epoch, arr("real_val_loss"), label="real val")
    axes[0].plot(epoch, arr("real_test_loss"), label="real test")
    axes[0].plot(epoch, arr("synthetic_test_loss"), label="synthetic test")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch, arr("train_stress_mse"), label="train")
    axes[1].plot(epoch, arr("real_val_stress_mse"), label="real val")
    axes[1].plot(epoch, arr("real_test_stress_mse"), label="real test")
    axes[1].plot(epoch, arr("synthetic_test_stress_mse"), label="synthetic test")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("stress mse")
    axes[1].set_xlabel("global epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ymax0 = float(max(arr("train_loss").max(), arr("real_val_loss").max(), arr("real_test_loss").max(), arr("synthetic_test_loss").max()))
    ymax1 = float(max(arr("train_stress_mse").max(), arr("real_val_stress_mse").max(), arr("real_test_stress_mse").max(), arr("synthetic_test_stress_mse").max()))
    for ax in axes:
        for x, _ in boundaries:
            ax.axvline(x, color="k", linestyle="--", alpha=0.2)
    for x, label in boundaries:
        axes[0].text(x + 1, ymax0, label, rotation=90, va="top", ha="left", fontsize=8)
        axes[1].text(x + 1, ymax1, label, rotation=90, va="top", ha="left", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_branch_accuracy(history_csv: Path, output_path: Path) -> Path:
    rows = list(csv.DictReader(history_csv.open("r", encoding="utf-8")))
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
    ax.plot(epoch, arr("synthetic_test_branch_accuracy"), label="synthetic test")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("branch accuracy")
    ax.set_xlabel("global epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for x, _ in boundaries:
        ax.axvline(x, color="k", linestyle="--", alpha=0.15)
    for x, label in boundaries:
        ax.text(x + 1, 0.99, label, rotation=90, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def _train_refresh_run(
    *,
    real_primary: Path,
    synthetic_test_path: Path,
    run_dir: Path,
    spec: RunSpec,
    device: torch.device,
    eval_batch_size: int,
    branch_loss_weight: float,
    grad_clip: float,
    train_epoch_cfg: SyntheticEpochConfig,
    calibration_cfg: SyntheticEpochConfig,
    material_pool: np.ndarray,
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

    calibration_arrays = generate_branch_balanced_from_material_pool(
        material_pool=material_pool,
        cfg=calibration_cfg,
        seed=spec.seed + 101,
    )
    calibration_split = _split_arrays_for_raw_branch(calibration_arrays)
    x_scaler = Standardizer.from_array(calibration_split["features"])
    y_scaler = Standardizer.from_array(calibration_split["target"])

    real_val_arrays = _load_split_for_training(str(real_primary), "val", "raw_branch")
    real_test_arrays = _load_split_for_training(str(real_primary), "test", "raw_branch")
    synthetic_test_arrays = _load_split_for_training(str(synthetic_test_path), "test", "raw_branch")

    real_val_loader = DataLoader(_build_tensor_dataset(real_val_arrays, x_scaler, y_scaler), batch_size=eval_batch_size, shuffle=False, num_workers=0)
    real_test_loader = DataLoader(_build_tensor_dataset(real_test_arrays, x_scaler, y_scaler), batch_size=eval_batch_size, shuffle=False, num_workers=0)
    synthetic_test_loader = DataLoader(_build_tensor_dataset(synthetic_test_arrays, x_scaler, y_scaler), batch_size=eval_batch_size, shuffle=False, num_workers=0)

    model = build_model(
        "raw_branch",
        input_dim=calibration_split["features"].shape[1],
        width=spec.width,
        depth=spec.depth,
        dropout=0.0,
    ).to(device)
    metadata = {
        "config": {
            "dataset": "epoch_refreshed_synthetic_train",
            "real_primary": str(real_primary),
            "synthetic_test": str(synthetic_test_path),
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
            "selection_metric": "real_val_stress_mse",
            "train_epoch_cfg": asdict(train_epoch_cfg),
            "calibration_cfg": asdict(calibration_cfg),
            "material_pool_rows": int(material_pool.shape[0]),
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_csv = run_dir / "history.csv"
    _write_history_header(history_csv)
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    best_real_val_stress_mse = float("inf")
    best_epoch = 0
    global_epoch = 0
    total_start = time.perf_counter()

    for cycle_idx in range(1, cycles + 1):
        for batch_size in batch_sizes:
            stage_name = f"cycle{cycle_idx}_bs{batch_size}"
            stage_lr = _stage_initial_lr(
                cycle_idx=cycle_idx,
                batch_size=batch_size,
                base_lr=base_lr,
                cycle_lr_decay=cycle_lr_decay,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=spec.weight_decay)
            stage_best_val = float("inf")
            stage_bad_epochs = 0
            lr_bad_epochs = 0
            stage_start = time.perf_counter()

            _log_message(
                log_path,
                (
                    f"[stage-start] cycle={cycle_idx}/{cycles} batch={batch_size} lr={stage_lr:.3e} "
                    f"global_epoch={global_epoch} runtime={_format_runtime(time.perf_counter() - total_start)}"
                ),
            )

            while True:
                global_epoch += 1
                train_seed = spec.seed + 100000 + global_epoch * 7919
                train_arrays = generate_branch_balanced_from_material_pool(
                    material_pool=material_pool,
                    cfg=train_epoch_cfg,
                    seed=train_seed,
                )
                train_split = _split_arrays_for_raw_branch(train_arrays)
                train_ds = _build_tensor_dataset(train_split, x_scaler, y_scaler)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

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
                synthetic_test_metrics = _epoch_loop(
                    model=model,
                    loader=synthetic_test_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )

                current_lr = optimizer.param_groups[0]["lr"]
                checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
                torch.save(checkpoint, last_ckpt)

                is_best = False
                if real_val_metrics["stress_mse"] < best_real_val_stress_mse:
                    best_real_val_stress_mse = real_val_metrics["stress_mse"]
                    best_epoch = global_epoch
                    _save_checkpoint(best_ckpt, model, metadata)
                    is_best = True

                _append_history_row(
                    history_csv,
                    epoch=global_epoch,
                    stage_name=stage_name,
                    batch_size=batch_size,
                    train_seed=train_seed,
                    lr=current_lr,
                    train_metrics=train_metrics,
                    real_val_metrics=real_val_metrics,
                    real_test_metrics=real_test_metrics,
                    synthetic_test_metrics=synthetic_test_metrics,
                    best_real_val_stress_mse=best_real_val_stress_mse,
                    is_best=is_best,
                )
                _log_message(
                    log_path,
                    (
                        f"[adam] epoch={global_epoch} cycle={cycle_idx}/{cycles} batch={batch_size} lr={current_lr:.3e} "
                        f"train_seed={train_seed} runtime={_format_runtime(time.perf_counter() - total_start)} "
                        f"train_loss={train_metrics['loss']:.6f} real_val_stress_mse={real_val_metrics['stress_mse']:.6f} "
                        f"real_test_stress_mse={real_test_metrics['stress_mse']:.6f} synth_test_stress_mse={synthetic_test_metrics['stress_mse']:.6f} "
                        f"best_real_val={best_real_val_stress_mse:.6f}"
                    ),
                )

                if _maybe_improved(
                    real_val_metrics["stress_mse"],
                    stage_best_val,
                    rel_tol=improvement_rel_tol,
                    abs_tol=improvement_abs_tol,
                ):
                    stage_best_val = real_val_metrics["stress_mse"]
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
                            f"[stage-stop] cycle={cycle_idx}/{cycles} batch={batch_size} stage_runtime={_format_runtime(time.perf_counter() - stage_start)} "
                            f"total_runtime={_format_runtime(time.perf_counter() - total_start)} "
                            f"best_stage_real_val={stage_best_val:.6f}"
                        ),
                    )
                    break

    elapsed = time.perf_counter() - total_start
    history_plot = _plot_history(history_csv, run_dir / "history_log.png")
    branch_plot = _plot_branch_accuracy(history_csv, run_dir / "branch_accuracy.png")
    real_eval = evaluate_checkpoint_on_dataset(best_ckpt, real_primary, split="test", device=str(device), batch_size=eval_batch_size)
    synthetic_eval = evaluate_checkpoint_on_dataset(best_ckpt, synthetic_test_path, split="test", device=str(device), batch_size=eval_batch_size)

    real_dir = run_dir / "eval_real"
    synth_dir = run_dir / "eval_synth"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(real_eval["arrays"]["stress"], real_eval["predictions"]["stress"], real_dir / "parity_stress.png", label="stress")
    error_histogram(real_eval["predictions"]["stress"] - real_eval["arrays"]["stress"], real_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in real_eval["metrics"]:
        branch_confusion_plot(real_eval["metrics"]["branch_confusion"], real_dir / "branch_confusion.png")
    parity_plot(synthetic_eval["arrays"]["stress"], synthetic_eval["predictions"]["stress"], synth_dir / "parity_stress.png", label="stress")
    error_histogram(synthetic_eval["predictions"]["stress"] - synthetic_eval["arrays"]["stress"], synth_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in synthetic_eval["metrics"]:
        branch_confusion_plot(synthetic_eval["metrics"]["branch_confusion"], synth_dir / "branch_confusion.png")

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
        "elapsed_seconds": elapsed,
        "elapsed_hms": _format_runtime(elapsed),
        "real_test_metrics": real_eval["metrics"],
        "synthetic_test_metrics": synthetic_eval["metrics"],
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _write_report(
    report_md: Path,
    *,
    summary: dict[str, Any],
    train_info: dict[str, Any],
    real_test_distribution: dict[str, Any],
    synthetic_test_distribution: dict[str, Any],
    synthetic_test_path: Path,
) -> None:
    def rel_repo(path: str | Path) -> str:
        return Path(path).resolve().relative_to(ROOT).as_posix()

    prior = json.loads(
        (
            ROOT
            / "experiment_runs"
            / "real_sim"
            / "cover_layer_cyclic_20260312"
            / "cover_raw_branch_w384_d6"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    p = summary["real_test_metrics"]
    s = summary["synthetic_test_metrics"]

    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cover Layer Epoch-Refresh Training",
        "",
        "This run keeps the cover-layer material family fixed, but regenerates a new synthetic training set every epoch.",
        "The held-out evaluations are:",
        f"- real cover-layer test split from `{train_info['real_primary']}`",
        f"- independent synthetic holdout test from `{synthetic_test_path}`",
        "",
        "## Setup",
        "",
        f"- model: `{summary['spec']['name']}` (`width={summary['spec']['width']}`, `depth={summary['spec']['depth']}`)",
        f"- runtime: `{summary['elapsed_hms']}`",
        f"- best epoch by real-val stress MSE: `{summary['best_epoch']}`",
        f"- best real-val stress MSE: `{summary['best_real_val_stress_mse']:.6f}`",
        f"- real material pool rows used for synthetic generation: `{train_info['n_material_rows']}`",
        f"- unique reduced material rows in that pool: `{train_info['n_unique_reduced_rows']}`",
        f"- synthetic train samples regenerated each epoch: `{train_info['train_samples_per_epoch']}`",
        f"- synthetic test samples: `{train_info['synthetic_test_samples']}`",
        f"- principal-strain cap: `{train_info['principal_cap']:.6e}`",
        f"- branch fractions per regenerated epoch: `{train_info['branch_fractions']}`",
        "",
        "## Distribution Check",
        "",
        "| Dataset | N | max |eps_principal| q95 | q995 | stress | q95 | q995 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| real test | {real_test_distribution['n_samples']} | {real_test_distribution['max_abs_principal_q95']:.6e} | {real_test_distribution['max_abs_principal_q995']:.6e} | {real_test_distribution['stress_mag_q95']:.4f} | {real_test_distribution['stress_mag_q995']:.4f} |",
        f"| synthetic holdout | {synthetic_test_distribution['n_samples']} | {synthetic_test_distribution['max_abs_principal_q95']:.6e} | {synthetic_test_distribution['max_abs_principal_q995']:.6e} | {synthetic_test_distribution['stress_mag_q95']:.4f} | {synthetic_test_distribution['stress_mag_q995']:.4f} |",
        "",
        "## Results",
        "",
        "| Test set | Stress MAE | Stress RMSE | Stress Max Abs | Branch Acc |",
        "|---|---:|---:|---:|---:|",
        f"| real | {p['stress_mae']:.4f} | {p['stress_rmse']:.4f} | {p['stress_max_abs']:.4f} | {p.get('branch_accuracy', float('nan')):.4f} |",
        f"| synthetic | {s['stress_mae']:.4f} | {s['stress_rmse']:.4f} | {s['stress_max_abs']:.4f} | {s.get('branch_accuracy', float('nan')):.4f} |",
        "",
        "Comparison to the previous fixed-dataset cover-layer best (`w384 d6` on exact-domain training file):",
        "",
        f"- previous real test MAE / RMSE / max abs: `{prior['primary_metrics']['stress_mae']:.4f}` / "
        f"`{prior['primary_metrics']['stress_rmse']:.4f}` / `{prior['primary_metrics']['stress_max_abs']:.4f}`",
        f"- refreshed-data real test MAE / RMSE / max abs: `{p['stress_mae']:.4f}` / `{p['stress_rmse']:.4f}` / `{p['stress_max_abs']:.4f}`",
        "",
        "## History",
        "",
        f"![epoch refresh history](../{rel_repo(summary['history_plot'])})",
        "",
        f"![epoch refresh branch accuracy](../{rel_repo(summary['branch_plot'])})",
        "",
        "## Real Test Plots",
        "",
        f"![real parity](../{rel_repo(Path(summary['history_plot']).parent / 'eval_real' / 'parity_stress.png')})",
        "",
        f"![real error histogram](../{rel_repo(Path(summary['history_plot']).parent / 'eval_real' / 'stress_error_hist.png')})",
        "",
        f"![real branch confusion](../{rel_repo(Path(summary['history_plot']).parent / 'eval_real' / 'branch_confusion.png')})",
        "",
        "## Synthetic Holdout Plots",
        "",
        f"![synthetic parity](../{rel_repo(Path(summary['history_plot']).parent / 'eval_synth' / 'parity_stress.png')})",
        "",
        f"![synthetic error histogram](../{rel_repo(Path(summary['history_plot']).parent / 'eval_synth' / 'stress_error_hist.png')})",
        "",
        f"![synthetic branch confusion](../{rel_repo(Path(summary['history_plot']).parent / 'eval_synth' / 'branch_confusion.png')})",
        "",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    real_primary = Path(args.real_primary)

    material_pool, principal_cap, train_info = _estimate_principal_cap(real_primary)
    train_epoch_cfg = SyntheticEpochConfig(
        n_samples=args.train_size,
        candidate_batch=args.candidate_batch,
        max_abs_principal_strain=principal_cap,
    )
    calibration_cfg = SyntheticEpochConfig(
        n_samples=args.train_size,
        candidate_batch=args.candidate_batch,
        max_abs_principal_strain=principal_cap,
    )
    synthetic_test_cfg = SyntheticEpochConfig(
        n_samples=args.synthetic_test_size,
        candidate_batch=args.candidate_batch,
        max_abs_principal_strain=principal_cap,
    )

    synthetic_test_arrays = generate_branch_balanced_from_material_pool(
        material_pool=material_pool,
        cfg=synthetic_test_cfg,
        seed=args.seed + 500000,
    )
    synthetic_test_path = _write_dataset(
        output_root / "synthetic_test.h5",
        {
            **synthetic_test_arrays,
            "split_id": np.full(synthetic_test_arrays["strain_eng"].shape[0], SPLIT_TO_ID["test"], dtype=np.int8),
        },
        {
            "generator": "epoch_refresh_synthetic_holdout",
            "seed": int(args.seed + 500000),
            "principal_cap": principal_cap,
        },
    )

    spec = RunSpec(
        name=f"cover_raw_branch_w{args.width}_d{args.depth}_epoch_refresh",
        width=args.width,
        depth=args.depth,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    summary = _train_refresh_run(
        real_primary=real_primary,
        synthetic_test_path=synthetic_test_path,
        run_dir=output_root / spec.name,
        spec=spec,
        device=device,
        eval_batch_size=args.eval_batch_size,
        branch_loss_weight=args.branch_loss_weight,
        grad_clip=args.grad_clip,
        train_epoch_cfg=train_epoch_cfg,
        calibration_cfg=calibration_cfg,
        material_pool=material_pool,
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

    real_arrays = _load_arrays(real_primary)
    real_test_mask = real_arrays["split_id"] == SPLIT_TO_ID["test"]
    real_test_distribution = _distribution_summary(
        real_arrays["strain_eng"][real_test_mask],
        real_arrays["stress"][real_test_mask],
        real_arrays["branch_id"][real_test_mask],
    )
    synthetic_test_distribution = _distribution_summary(
        synthetic_test_arrays["strain_eng"],
        synthetic_test_arrays["stress"],
        synthetic_test_arrays["branch_id"],
    )
    train_info.update(
        {
            "real_primary": str(real_primary),
            "train_samples_per_epoch": args.train_size,
            "synthetic_test_samples": args.synthetic_test_size,
            "branch_fractions": list(train_epoch_cfg.branch_fractions),
        }
    )
    (output_root / "distribution_summary.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "train_info": train_info,
                    "real_test_distribution": real_test_distribution,
                    "synthetic_test_distribution": synthetic_test_distribution,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        Path(args.report_md),
        summary=summary,
        train_info=train_info,
        real_test_distribution=real_test_distribution,
        synthetic_test_distribution=synthetic_test_distribution,
        synthetic_test_path=synthetic_test_path,
    )
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
