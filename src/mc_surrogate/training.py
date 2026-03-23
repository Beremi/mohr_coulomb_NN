"""Training and evaluation utilities for constitutive surrogates."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import h5py
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
    build_trial_principal_features,
    build_trial_features,
    compute_trial_stress,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
    stress_voigt_from_principal_torch,
)
from .mohr_coulomb import BRANCH_NAMES
from .voigt import stress_voigt_to_tensor


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
    plateau_factor: float = 0.5
    plateau_patience: int | None = None
    lbfgs_epochs: int = 0
    lbfgs_lr: float = 0.25
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 100
    log_every_epochs: int = 0
    stress_weight_alpha: float = 0.0
    stress_weight_scale: float = 250.0
    checkpoint_metric: str = "loss"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_keys(dataset_path: str) -> set[str]:
    with h5py.File(dataset_path, "r") as f:
        return set(f.keys())


def _principal_stress_from_stress(stress_voigt: np.ndarray) -> np.ndarray:
    tensor = stress_voigt_to_tensor(stress_voigt)
    vals = np.linalg.eigvalsh(tensor)
    return vals[:, ::-1].astype(np.float32)


def _is_principal_model(model_kind: str) -> bool:
    return model_kind in {"principal", "trial_principal_branch_residual"}


def _uses_raw_features(model_kind: str) -> bool:
    return model_kind in {"raw", "raw_branch"}


def _uses_trial_features(model_kind: str) -> bool:
    return model_kind in {
        "trial_raw",
        "trial_raw_branch",
        "trial_raw_residual",
        "trial_raw_branch_residual",
    }


def _uses_trial_principal_features(model_kind: str) -> bool:
    return model_kind in {"trial_principal_branch_residual"}


def _uses_residual_target(model_kind: str) -> bool:
    return model_kind in {"trial_raw_residual", "trial_raw_branch_residual", "trial_principal_branch_residual"}


def _plastic_only_regression(model_kind: str) -> bool:
    return model_kind in {"trial_principal_branch_residual"}


def _load_split_for_training(dataset_path: str, split: str, model_kind: str) -> dict[str, np.ndarray]:
    available = _dataset_keys(dataset_path)
    keys = ["strain_eng", "stress", "material_reduced"]
    optional = ["stress_principal", "branch_id", "eigvecs"]
    keys.extend([key for key in optional if key in available])
    arrays = load_arrays(dataset_path, keys, split=split)

    stress_principal = arrays.get("stress_principal")
    if stress_principal is None:
        stress_principal = _principal_stress_from_stress(arrays["stress"])
    branch_id = arrays.get("branch_id")
    if branch_id is None:
        branch_id = np.full(arrays["strain_eng"].shape[0], -1, dtype=np.int64)
    else:
        branch_id = branch_id.astype(np.int64)

    trial_stress = compute_trial_stress(arrays["strain_eng"], arrays["material_reduced"])
    trial_principal = _principal_stress_from_stress(trial_stress)

    if _uses_trial_principal_features(model_kind):
        strain_principal, eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
        features = build_trial_principal_features(strain_principal, arrays["material_reduced"], trial_principal)
        target = (stress_principal.astype(np.float32) - trial_principal.astype(np.float32)).astype(np.float32)
        return {
            "features": features,
            "target": target,
            "stress_true": arrays["stress"].astype(np.float32),
            "branch_id": branch_id,
            "eigvecs": eigvecs.astype(np.float32),
            "trial_stress": trial_stress.astype(np.float32),
            "trial_principal": trial_principal.astype(np.float32),
        }

    if _is_principal_model(model_kind):
        # Recompute principal strains from the stored engineering strain so the
        # training features remain self-contained and consistent with inference.
        strain_principal, eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
        features = build_principal_features(strain_principal, arrays["material_reduced"])
        target = stress_principal.astype(np.float32)
        out = {
            "features": features,
            "target": target,
            "stress_true": arrays["stress"].astype(np.float32),
            "branch_id": branch_id,
            "eigvecs": eigvecs.astype(np.float32),
            "trial_stress": np.zeros_like(arrays["stress"], dtype=np.float32),
            "trial_principal": np.zeros_like(stress_principal, dtype=np.float32),
        }
        return out

    if _uses_raw_features(model_kind):
        features = build_raw_features(arrays["strain_eng"], arrays["material_reduced"])
        target = arrays["stress"].astype(np.float32)
        eigvecs = arrays.get("eigvecs")
        if eigvecs is None:
            _, eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
        return {
            "features": features,
            "target": target,
            "stress_true": arrays["stress"].astype(np.float32),
            "branch_id": branch_id,
            "eigvecs": eigvecs.astype(np.float32),
            "trial_stress": trial_stress.astype(np.float32),
            "trial_principal": trial_principal.astype(np.float32),
        }

    if _uses_trial_features(model_kind):
        features = build_trial_features(arrays["strain_eng"], arrays["material_reduced"])
        if _uses_residual_target(model_kind):
            target = (arrays["stress"] - trial_stress).astype(np.float32)
        else:
            target = arrays["stress"].astype(np.float32)
        eigvecs = arrays.get("eigvecs")
        if eigvecs is None:
            _, eigvecs = spectral_decomposition_from_strain(arrays["strain_eng"])
        return {
            "features": features,
            "target": target,
            "stress_true": arrays["stress"].astype(np.float32),
            "branch_id": branch_id,
            "eigvecs": eigvecs.astype(np.float32),
            "trial_stress": trial_stress.astype(np.float32),
            "trial_principal": trial_principal.astype(np.float32),
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
    trial_stress = torch.from_numpy(split_arrays["trial_stress"])
    trial_principal = torch.from_numpy(split_arrays["trial_principal"])
    return TensorDataset(x, y, branch, stress_true, eigvecs, trial_stress, trial_principal)


def _decode_stress_prediction(
    *,
    model_kind: str,
    pred_norm: torch.Tensor,
    y_scaler: Standardizer,
    eigvecs: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    branch_logits: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pred = pred_norm * torch.as_tensor(y_scaler.std, device=pred_norm.device) + torch.as_tensor(
        y_scaler.mean, device=pred_norm.device
    )

    if model_kind == "principal":
        stress_principal = pred
        stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
        return stress, stress_principal

    if model_kind == "trial_principal_branch_residual":
        stress_principal = pred + trial_principal
        stress_principal, _ = torch.sort(stress_principal, dim=-1, descending=True)
        if branch_logits is not None:
            pred_branch = branch_logits.argmax(dim=1)
            elastic_mask = pred_branch == 0
            if torch.any(elastic_mask):
                stress_principal = stress_principal.clone()
                stress_principal[elastic_mask] = trial_principal[elastic_mask]
        stress = stress_voigt_from_principal_torch(stress_principal, eigvecs)
        return stress, stress_principal

    if _uses_residual_target(model_kind):
        return pred + trial_stress, None

    return pred, None


def _regression_loss(
    model_kind: str,
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    y_scaler: Standardizer,
    branch_true: torch.Tensor,
    eigvecs: torch.Tensor,
    stress_true: torch.Tensor,
    trial_stress: torch.Tensor,
    trial_principal: torch.Tensor,
    branch_logits: torch.Tensor | None,
    stress_weight_alpha: float,
    stress_weight_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_sample_mse = torch.mean((pred_norm - target_norm) ** 2, dim=1)
    if _plastic_only_regression(model_kind):
        valid = branch_true > 0
        if torch.any(valid):
            per_sample_mse = per_sample_mse[valid]
        else:
            per_sample_mse = pred_norm.new_zeros((1,))
    if stress_weight_alpha > 0.0:
        sample_mag = torch.amax(torch.abs(stress_true), dim=1)
        if _plastic_only_regression(model_kind):
            valid = branch_true > 0
            sample_mag = sample_mag[valid] if torch.any(valid) else sample_mag[:1]
        weights = 1.0 + stress_weight_alpha * torch.log1p(sample_mag / max(stress_weight_scale, 1.0e-12))
        mse = torch.mean(weights * per_sample_mse)
    else:
        mse = torch.mean(per_sample_mse)
    metrics = {"regression_mse": float(mse.detach().cpu())}
    stress_pred, _ = _decode_stress_prediction(
        model_kind=model_kind,
        pred_norm=pred_norm,
        y_scaler=y_scaler,
        eigvecs=eigvecs,
        trial_stress=trial_stress,
        trial_principal=trial_principal,
        branch_logits=branch_logits,
    )

    stress_mse = nn.functional.mse_loss(stress_pred, stress_true)
    stress_mae = torch.mean(torch.abs(stress_pred - stress_true))
    metrics["stress_mse"] = float(stress_mse.detach().cpu())
    metrics["stress_mae"] = float(stress_mae.detach().cpu())
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
    stress_weight_alpha: float,
    stress_weight_scale: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_reg = 0.0
    total_stress = 0.0
    total_stress_mae = 0.0
    total_branch = 0.0
    total_branch_correct = 0.0
    n_branch_samples = 0
    n_samples = 0

    for xb, yb, branch, stress_true, eigvecs, trial_stress, trial_principal in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        branch = branch.to(device)
        stress_true = stress_true.to(device)
        eigvecs = eigvecs.to(device)
        trial_stress = trial_stress.to(device)
        trial_principal = trial_principal.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        out = model(xb)
        reg_loss, reg_metrics = _regression_loss(
            model_kind=model_kind,
            pred_norm=out["stress"],
            target_norm=yb,
            y_scaler=y_scaler,
            branch_true=branch,
            eigvecs=eigvecs,
            stress_true=stress_true,
            trial_stress=trial_stress,
            trial_principal=trial_principal,
            branch_logits=out.get("branch_logits"),
            stress_weight_alpha=stress_weight_alpha,
            stress_weight_scale=stress_weight_scale,
        )

        loss = reg_loss
        branch_loss_value = 0.0
        branch_acc_value = 0.0

        if "branch_logits" in out:
            valid_branch = branch >= 0
            if torch.any(valid_branch):
                branch_loss = nn.functional.cross_entropy(out["branch_logits"][valid_branch], branch[valid_branch])
                loss = loss + branch_loss_weight * branch_loss
                branch_loss_value = float(branch_loss.detach().cpu())
                pred_branch = out["branch_logits"][valid_branch].argmax(dim=1)
                branch_acc_value = float((pred_branch == branch[valid_branch]).float().mean().detach().cpu())
                n_branch_samples += int(valid_branch.sum().detach().cpu())

        if training:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = xb.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_reg += reg_metrics["regression_mse"] * batch_size
        total_stress += reg_metrics["stress_mse"] * batch_size
        total_stress_mae += reg_metrics["stress_mae"] * batch_size
        total_branch += branch_loss_value * max(int((branch >= 0).sum().detach().cpu()), 0)
        total_branch_correct += branch_acc_value * max(int((branch >= 0).sum().detach().cpu()), 0)
        n_samples += batch_size

    return {
        "loss": total_loss / max(n_samples, 1),
        "regression_mse": total_reg / max(n_samples, 1),
        "stress_mse": total_stress / max(n_samples, 1),
        "stress_mae": total_stress_mae / max(n_samples, 1),
        "branch_loss": total_branch / max(n_branch_samples, 1),
        "branch_accuracy": total_branch_correct / max(n_branch_samples, 1),
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, str]:
    """Build an epoch scheduler and describe how it should be stepped."""
    if config.scheduler_kind == "none":
        return None, "none"

    if config.scheduler_kind == "plateau":
        plateau_patience = config.plateau_patience
        if plateau_patience is None:
            plateau_patience = max(3, config.patience // 4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=plateau_patience,
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
                train_metrics["stress_mae"],
                val_metrics["stress_mae"],
                train_metrics["branch_loss"],
                val_metrics["branch_loss"],
                train_metrics["branch_accuracy"],
                val_metrics["branch_accuracy"],
                lbfgs_phase,
            ]
        )


def _maybe_print_epoch_status(
    *,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val: float,
    lbfgs_phase: int,
    config: TrainingConfig,
) -> None:
    if config.log_every_epochs <= 0:
        return
    should_print = epoch == 1 or epoch % config.log_every_epochs == 0
    if not should_print:
        return
    phase_name = "LBFGS" if lbfgs_phase else "Adam"
    print(
        f"[{phase_name}] epoch={epoch} "
        f"lr={lr:.3e} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"val_loss={val_metrics['loss']:.6f} "
        f"train_stress_mse={train_metrics['stress_mse']:.6f} "
        f"val_stress_mse={val_metrics['stress_mse']:.6f} "
        f"val_stress_mae={val_metrics['stress_mae']:.6f} "
        f"best_val={best_val:.6f}"
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
                "train_stress_mae",
                "val_stress_mae",
                "train_branch_loss",
                "val_branch_loss",
                "train_branch_accuracy",
                "val_branch_accuracy",
                "lbfgs_phase",
            ]
        )

    if config.checkpoint_metric not in {"loss", "stress_mse", "stress_mae"}:
        raise ValueError(f"Unsupported checkpoint_metric {config.checkpoint_metric!r}.")
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
            stress_weight_alpha=config.stress_weight_alpha,
            stress_weight_scale=config.stress_weight_scale,
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
            stress_weight_alpha=config.stress_weight_alpha,
            stress_weight_scale=config.stress_weight_scale,
        )

        if scheduler is not None:
            if scheduler_step_mode == "val":
                scheduler.step(val_metrics[config.checkpoint_metric])
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

        current_metric = val_metrics[config.checkpoint_metric]
        if current_metric < best_val:
            best_val = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        _maybe_print_epoch_status(
            epoch=epoch,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val=best_val,
            lbfgs_phase=0,
            config=config,
        )

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
            xb, yb, branch, stress_true, eigvecs, trial_stress, trial_principal = train_full

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                out = model(xb)
                reg_loss, _ = _regression_loss(
                    model_kind=config.model_kind,
                    pred_norm=out["stress"],
                    target_norm=yb,
                    y_scaler=y_scaler,
                    branch_true=branch,
                    eigvecs=eigvecs,
                    stress_true=stress_true,
                    trial_stress=trial_stress,
                    trial_principal=trial_principal,
                    branch_logits=out.get("branch_logits"),
                    stress_weight_alpha=config.stress_weight_alpha,
                    stress_weight_scale=config.stress_weight_scale,
                )
                loss = reg_loss
                if "branch_logits" in out:
                    valid_branch = branch >= 0
                    if torch.any(valid_branch):
                        branch_loss = nn.functional.cross_entropy(out["branch_logits"][valid_branch], branch[valid_branch])
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
                stress_weight_alpha=config.stress_weight_alpha,
                stress_weight_scale=config.stress_weight_scale,
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
                stress_weight_alpha=config.stress_weight_alpha,
                stress_weight_scale=config.stress_weight_scale,
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

            current_metric = val_metrics[config.checkpoint_metric]
            if current_metric < best_val:
                best_val = current_metric
                best_epoch = epoch
                torch.save(checkpoint, best_checkpoint_path)
            completed_epochs = epoch
            _maybe_print_epoch_status(
                epoch=epoch,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val=best_val,
                lbfgs_phase=1,
                config=config,
            )

    summary = {
        "best_val_loss": best_val,
        "checkpoint_metric": config.checkpoint_metric,
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
    device_obj = choose_device(device)
    ckpt = torch.load(checkpoint_path, map_location=device_obj)
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
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Predict constitutive stresses using a saved checkpoint."""
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    device_obj = choose_device(device)
    model = model.to(device_obj)
    cfg = metadata["config"]
    x_scaler = Standardizer.from_dict(metadata["x_scaler"])
    y_scaler = Standardizer.from_dict(metadata["y_scaler"])

    strain_eng = np.asarray(strain_eng, dtype=float)
    material_reduced = np.asarray(material_reduced, dtype=float)

    if _uses_trial_principal_features(cfg["model_kind"]):
        strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)
        trial_stress = compute_trial_stress(strain_eng, material_reduced)
        trial_principal = _principal_stress_from_stress(trial_stress)
        features = build_trial_principal_features(strain_principal, material_reduced, trial_principal)
    elif _is_principal_model(cfg["model_kind"]):
        strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)
        features = build_principal_features(strain_principal, material_reduced)
    elif _uses_raw_features(cfg["model_kind"]):
        strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)
        features = build_raw_features(strain_eng, material_reduced)
    elif _uses_trial_features(cfg["model_kind"]):
        strain_principal, eigvecs = spectral_decomposition_from_strain(strain_eng)
        features = build_trial_features(strain_eng, material_reduced)
    else:
        raise ValueError(f"Unsupported model kind {cfg['model_kind']!r}.")

    if batch_size is None or batch_size <= 0:
        batch_size = int(features.shape[0])

    pred_chunks: list[np.ndarray] = []
    branch_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(x_scaler.transform(features[start:stop])).to(device_obj)
            out = model(x)
            pred_chunks.append(out["stress"].cpu().numpy())
            if "branch_logits" in out:
                branch_chunks.append(out["branch_logits"].cpu().numpy())
    pred_norm = np.concatenate(pred_chunks, axis=0)
    trial_stress = compute_trial_stress(strain_eng, material_reduced)
    trial_principal = _principal_stress_from_stress(trial_stress)
    branch_probs = None
    if branch_chunks:
        logits = np.concatenate(branch_chunks, axis=0)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        branch_probs = probs.astype(np.float32)
    pred = y_scaler.inverse_transform(pred_norm)

    if cfg["model_kind"] == "principal":
        stress = stress_voigt_from_principal_numpy(pred, eigvecs)
        stress_principal = pred
    elif cfg["model_kind"] == "trial_principal_branch_residual":
        stress_principal = pred + trial_principal
        stress_principal = np.sort(stress_principal, axis=1)[:, ::-1]
        if branch_probs is not None:
            pred_branch = np.argmax(branch_probs, axis=1)
            elastic_mask = pred_branch == 0
            if np.any(elastic_mask):
                stress_principal = stress_principal.copy()
                stress_principal[elastic_mask] = trial_principal[elastic_mask]
        stress = stress_voigt_from_principal_numpy(stress_principal, eigvecs)
    elif _uses_residual_target(cfg["model_kind"]):
        stress = pred + trial_stress
        stress_principal = None
    else:
        stress = pred
        stress_principal = None

    result = {
        "stress": stress.astype(np.float32),
        "branch_probabilities": branch_probs,
        "strain_principal": strain_principal.astype(np.float32),
        "eigvecs": eigvecs.astype(np.float32),
    }
    if stress_principal is not None:
        result["stress_principal"] = stress_principal.astype(np.float32)
    return result


def evaluate_checkpoint_on_dataset(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    split: str = "test",
    device: str = "cpu",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a dataset split and return metrics and predictions."""
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    cfg = metadata["config"]
    available = _dataset_keys(str(dataset_path))
    keys = ["strain_eng", "stress", "material_reduced"]
    if "stress_principal" in available:
        keys.append("stress_principal")
    if "branch_id" in available:
        keys.append("branch_id")
    arrays = load_arrays(dataset_path, keys, split=split)
    if "stress_principal" not in arrays:
        arrays["stress_principal"] = _principal_stress_from_stress(arrays["stress"])
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
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

    if cfg["model_kind"] in {"principal", "trial_principal_branch_residual"}:
        stress_principal_true = arrays["stress_principal"]
        stress_principal_pred = pred["stress_principal"]
        metrics["principal_mae"] = float(np.mean(np.abs(stress_principal_pred - stress_principal_true)))
        metrics["principal_rmse"] = float(np.sqrt(np.mean((stress_principal_pred - stress_principal_true) ** 2)))

    if pred["branch_probabilities"] is not None and "branch_id" in arrays and np.any(arrays["branch_id"] >= 0):
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

    if "branch_id" in arrays and np.any(arrays["branch_id"] >= 0):
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
