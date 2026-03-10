"""Training and evaluation utilities for constitutive surrogates."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_arrays
from .models import (
    Standardizer,
    build_model,
    build_principal_features,
    build_raw_features,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
    stress_voigt_from_principal_torch,
)
from .mohr_coulomb import BRANCH_NAMES


def choose_device(device: str = "auto") -> torch.device:
    """Choose a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TrainingConfig:
    """Hyperparameters and file paths for training."""
    dataset: str
    run_dir: str
    model_kind: str = "principal"
    epochs: int = 150
    batch_size: int = 2048
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    width: int = 256
    depth: int = 4
    dropout: float = 0.0
    seed: int = 0
    patience: int = 25
    grad_clip: float = 1.0
    branch_loss_weight: float = 0.1
    num_workers: int = 0
    device: str = "auto"
    scheduler_kind: str = "plateau"
    warmup_epochs: int = 0
    min_lr: float = 1.0e-6
    lbfgs_epochs: int = 0
    lbfgs_lr: float = 0.25
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 100


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split_for_training(dataset_path: str, split: str, model_kind: str) -> dict[str, np.ndarray]:
    keys = ["strain_eng", "stress", "stress_principal", "material_reduced", "branch_id", "eigvecs"]
    arrays = load_arrays(dataset_path, keys, split=split)

    if model_kind == "principal":
        # Recompute principal strains from the stored engineering strain so the
        # training features remain self-contained and consistent with inference.
        strain_principal, eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
        features = build_principal_features(strain_principal, arrays["material_reduced"])
        target = arrays["stress_principal"].astype(np.float32)
        out = {
            "features": features,
            "target": target,
            "stress_true": arrays["stress"].astype(np.float32),
            "branch_id": arrays["branch_id"].astype(np.int64),
            "eigvecs": eigvecs.astype(np.float32),
        }
        return out

    if model_kind == "raw":
        features = build_raw_features(arrays["strain_eng"], arrays["material_reduced"])
        target = arrays["stress"].astype(np.float32)
        return {
            "features": features,
            "target": target,
            "stress_true": arrays["stress"].astype(np.float32),
            "branch_id": arrays["branch_id"].astype(np.int64),
            "eigvecs": arrays["eigvecs"].astype(np.float32),
        }

    raise ValueError(f"Unsupported model kind {model_kind!r}.")


def _build_tensor_dataset(
    split_arrays: dict[str, np.ndarray],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
) -> TensorDataset:
    x = torch.from_numpy(x_scaler.transform(split_arrays["features"]))
    y = torch.from_numpy(y_scaler.transform(split_arrays["target"]))
    branch = torch.from_numpy(split_arrays["branch_id"])
    stress_true = torch.from_numpy(split_arrays["stress_true"])
    eigvecs = torch.from_numpy(split_arrays["eigvecs"])
    return TensorDataset(x, y, branch, stress_true, eigvecs)


def _regression_loss(
    model_kind: str,
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    y_scaler: Standardizer,
    eigvecs: torch.Tensor,
    stress_true: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.functional.mse_loss(pred_norm, target_norm)
    metrics = {"regression_mse": float(mse.detach().cpu())}

    pred = pred_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
        y_scaler.mean, device=pred_norm.device
    )

    if model_kind == "principal":
        stress_pred = stress_voigt_from_principal_torch(pred, eigvecs)
    else:
        stress_pred = pred

    stress_mse = nn.functional.mse_loss(stress_pred, stress_true)
    metrics["stress_mse"] = float(stress_mse.detach().cpu())
    return mse, metrics


def _epoch_loop(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    model_kind: str,
    y_scaler: Standardizer,
    branch_loss_weight: float,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_reg = 0.0
    total_stress = 0.0
    total_branch = 0.0
    total_branch_correct = 0.0
    n_samples = 0

    for xb, yb, branch, stress_true, eigvecs in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        branch = branch.to(device)
        stress_true = stress_true.to(device)
        eigvecs = eigvecs.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        out = model(xb)
        reg_loss, reg_metrics = _regression_loss(
            model_kind=model_kind,
            pred_norm=out["stress"],
            target_norm=yb,
            y_scaler=y_scaler,
            eigvecs=eigvecs,
            stress_true=stress_true,
        )

        loss = reg_loss
        branch_loss_value = 0.0
        branch_acc_value = 0.0

        if "branch_logits" in out:
            branch_loss = nn.functional.cross_entropy(out["branch_logits"], branch)
            loss = loss + branch_loss_weight * branch_loss
            branch_loss_value = float(branch_loss.detach().cpu())
            pred_branch = out["branch_logits"].argmax(dim=1)
            branch_acc_value = float((pred_branch == branch).float().mean().detach().cpu())

        if training:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = xb.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_reg += reg_metrics["regression_mse"] * batch_size
        total_stress += reg_metrics["stress_mse"] * batch_size
        total_branch += branch_loss_value * batch_size
        total_branch_correct += branch_acc_value * batch_size
        n_samples += batch_size

    return {
        "loss": total_loss / max(n_samples, 1),
        "regression_mse": total_reg / max(n_samples, 1),
        "stress_mse": total_stress / max(n_samples, 1),
        "branch_loss": total_branch / max(n_samples, 1),
        "branch_accuracy": total_branch_correct / max(n_samples, 1),
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, str]:
    """Build an epoch scheduler and describe how it should be stepped."""
    if config.scheduler_kind == "none":
        return None, "none"

    if config.scheduler_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(3, config.patience // 4),
            min_lr=config.min_lr,
        )
        return scheduler, "val"

    if config.scheduler_kind == "cosine":
        if config.epochs <= 0:
            return None, "none"
        if config.warmup_epochs > 0:
            start_factor = max(config.min_lr / max(config.lr, 1.0e-12), 1.0e-3)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=min(start_factor, 1.0),
                end_factor=1.0,
                total_iters=config.warmup_epochs,
            )
            remain = max(1, config.epochs - config.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remain,
                eta_min=config.min_lr,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[config.warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, config.epochs),
                eta_min=config.min_lr,
            )
        return scheduler, "epoch"

    raise ValueError(f"Unsupported scheduler kind {config.scheduler_kind!r}.")


def _write_history_row(
    history_path: Path,
    *,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lbfgs_phase: int,
) -> None:
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                lr,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["regression_mse"],
                val_metrics["regression_mse"],
                train_metrics["stress_mse"],
                val_metrics["stress_mse"],
                train_metrics["branch_loss"],
                val_metrics["branch_loss"],
                train_metrics["branch_accuracy"],
                val_metrics["branch_accuracy"],
                lbfgs_phase,
            ]
        )


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Train a constitutive surrogate and save history/checkpoints."""
    set_seed(config.seed)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)

    train_arrays = _load_split_for_training(config.dataset, "train", config.model_kind)
    val_arrays = _load_split_for_training(config.dataset, "val", config.model_kind)

    x_scaler = Standardizer.from_array(train_arrays["features"])
    y_scaler = Standardizer.from_array(train_arrays["target"])

    train_ds = _build_tensor_dataset(train_arrays, x_scaler, y_scaler)
    val_ds = _build_tensor_dataset(val_arrays, x_scaler, y_scaler)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(
        model_kind=config.model_kind,
        input_dim=train_arrays["features"].shape[1],
        width=config.width,
        depth=config.depth,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler, scheduler_step_mode = _build_scheduler(optimizer, config)

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "train_regression_mse",
                "val_regression_mse",
                "train_stress_mse",
                "val_stress_mse",
                "train_branch_loss",
                "val_branch_loss",
                "train_branch_accuracy",
                "val_branch_accuracy",
                "lbfgs_phase",
            ]
        )

    best_val = float("inf")
    best_epoch = 0
    completed_epochs = 0
    epochs_without_improvement = 0

    metadata = {
        "config": asdict(config),
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _epoch_loop(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_kind=config.model_kind,
            y_scaler=y_scaler,
            branch_loss_weight=config.branch_loss_weight,
            device=device,
            grad_clip=config.grad_clip,
        )
        val_metrics = _epoch_loop(
            model=model,
            loader=val_loader,
            optimizer=None,
            model_kind=config.model_kind,
            y_scaler=y_scaler,
            branch_loss_weight=config.branch_loss_weight,
            device=device,
            grad_clip=config.grad_clip,
        )

        if scheduler is not None:
            if scheduler_step_mode == "val":
                scheduler.step(val_metrics["loss"])
            elif scheduler_step_mode == "epoch":
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        _write_history_row(
            history_path,
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lbfgs_phase=0,
        )
        completed_epochs = epoch

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        }
        torch.save(checkpoint, last_checkpoint_path)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if config.lbfgs_epochs > 0:
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.to(device)
        train_full = tuple(t.to(device) for t in train_ds.tensors)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        for lbfgs_epoch in range(1, config.lbfgs_epochs + 1):
            xb, yb, branch, stress_true, eigvecs = train_full

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                out = model(xb)
                reg_loss, _ = _regression_loss(
                    model_kind=config.model_kind,
                    pred_norm=out["stress"],
                    target_norm=yb,
                    y_scaler=y_scaler,
                    eigvecs=eigvecs,
                    stress_true=stress_true,
                )
                loss = reg_loss
                if "branch_logits" in out:
                    branch_loss = nn.functional.cross_entropy(out["branch_logits"], branch)
                    loss = loss + config.branch_loss_weight * branch_loss
                loss.backward()
                return loss

            model.train(True)
            lbfgs.step(closure)

            train_metrics = _epoch_loop(
                model=model,
                loader=train_loader,
                optimizer=None,
                model_kind=config.model_kind,
                y_scaler=y_scaler,
                branch_loss_weight=config.branch_loss_weight,
                device=device,
                grad_clip=config.grad_clip,
            )
            val_metrics = _epoch_loop(
                model=model,
                loader=val_loader,
                optimizer=None,
                model_kind=config.model_kind,
                y_scaler=y_scaler,
                branch_loss_weight=config.branch_loss_weight,
                device=device,
                grad_clip=config.grad_clip,
            )
            epoch = completed_epochs + lbfgs_epoch
            current_lr = lbfgs.param_groups[0]["lr"]
            _write_history_row(
                history_path,
                epoch=epoch,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lbfgs_phase=1,
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
            }
            torch.save(checkpoint, last_checkpoint_path)

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                torch.save(checkpoint, best_checkpoint_path)
            completed_epochs = epoch

    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "history_csv": str(history_path),
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
    metadata = ckpt["metadata"]
    cfg = metadata["config"]
    model = build_model(
        model_kind=cfg["model_kind"],
        input_dim=len(metadata["x_scaler"]["mean"]),
        width=cfg["width"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, metadata


def predict_with_checkpoint(
    checkpoint_path: str | Path,
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Predict constitutive stresses using a saved checkpoint."""
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    model = model.to(torch.device(device))
    cfg = metadata["config"]
    x_scaler = Standardizer.from_dict(metadata["x_scaler"])
    y_scaler = Standardizer.from_dict(metadata["y_scaler"])

    strain_eng = np.asarray(strain_eng, dtype=float)
    material_reduced = np.asarray(material_reduced, dtype=float)

    if cfg["model_kind"] == "principal":
        strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)
        features = build_principal_features(strain_principal, material_reduced)
    else:
        strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)
        features = build_raw_features(strain_eng, material_reduced)

    x = torch.from_numpy(x_scaler.transform(features)).to(torch.device(device))
    with torch.no_grad():
        out = model(x)
        pred_norm = out["stress"].cpu().numpy()
    pred = y_scaler.inverse_transform(pred_norm)

    if cfg["model_kind"] == "principal":
        stress = stress_voigt_from_principal_numpy(pred, eigvecs)
        stress_principal = pred
    else:
        stress = pred
        stress_principal = None

    result = {
        "stress": stress.astype(np.float32),
        "branch_probabilities": None,
        "strain_principal": strain_principal.astype(np.float32),
        "eigvecs": eigvecs.astype(np.float32),
    }
    if stress_principal is not None:
        result["stress_principal"] = stress_principal.astype(np.float32)
    if "branch_logits" in out:
        logits = out["branch_logits"].cpu().numpy()
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        result["branch_probabilities"] = probs.astype(np.float32)
    return result


def evaluate_checkpoint_on_dataset(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    split: str = "test",
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a checkpoint on a dataset split and return metrics and predictions."""
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    cfg = metadata["config"]
    arrays = load_arrays(
        dataset_path,
        ["strain_eng", "stress", "stress_principal", "material_reduced", "branch_id"],
        split=split,
    )
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
    )
    stress_pred = pred["stress"]
    stress_true = arrays["stress"]
    abs_err = np.abs(stress_pred - stress_true)

    metrics: dict[str, Any] = {
        "split": split,
        "n_samples": int(stress_true.shape[0]),
        "stress_mae": float(np.mean(abs_err)),
        "stress_rmse": float(np.sqrt(np.mean((stress_pred - stress_true) ** 2))),
        "stress_max_abs": float(np.max(abs_err)),
        "per_component_mae": np.mean(abs_err, axis=0).tolist(),
    }

    if cfg["model_kind"] == "principal":
        stress_principal_true = arrays["stress_principal"]
        stress_principal_pred = pred["stress_principal"]
        metrics["principal_mae"] = float(np.mean(np.abs(stress_principal_pred - stress_principal_true)))
        metrics["principal_rmse"] = float(np.sqrt(np.mean((stress_principal_pred - stress_principal_true) ** 2)))

    if pred["branch_probabilities"] is not None:
        branch_pred = np.argmax(pred["branch_probabilities"], axis=1)
        branch_true = arrays["branch_id"].astype(int)
        metrics["branch_accuracy"] = float(np.mean(branch_pred == branch_true))
        metrics["branch_confusion"] = [
            [
                int(np.sum((branch_true == i) & (branch_pred == j)))
                for j in range(len(BRANCH_NAMES))
            ]
            for i in range(len(BRANCH_NAMES))
        ]

    per_branch_mae = {}
    for i, name in enumerate(BRANCH_NAMES):
        mask = arrays["branch_id"] == i
        if np.any(mask):
            per_branch_mae[name] = float(np.mean(np.abs(stress_pred[mask] - stress_true[mask])))
    metrics["per_branch_stress_mae"] = per_branch_mae

    return {
        "metrics": metrics,
        "arrays": arrays,
        "predictions": pred,
    }
