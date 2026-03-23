from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mc_surrogate.cover_branch_generation import (
    collect_blocks,
    fit_seed_noise_bank,
    load_call_regimes,
    load_split_calls,
    pick_calls,
    synthesize_element_states_from_seeded_noise,
)
from mc_surrogate.full_export import canonicalize_p2_element_states

matplotlib.use("Agg")

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")


class ElementBranchMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_points: int = 11, out_classes: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        layers.append(nn.Linear(width, out_points * out_classes))
        self.net = nn.Sequential(*layers)
        self.out_points = out_points
        self.out_classes = out_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(x.shape[0], self.out_points, self.out_classes)


def _element_features(coords: np.ndarray, disp: np.ndarray) -> np.ndarray:
    canonical = canonicalize_p2_element_states(coords, disp)
    feat = np.concatenate(
        [
            canonical.local_coords.reshape(canonical.local_coords.shape[0], -1),
            canonical.local_displacements.reshape(canonical.local_displacements.shape[0], -1),
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = torch.argmax(logits, dim=-1)
    acc = float((pred == labels).float().mean().item())
    pattern_acc = float(torch.all(pred == labels, dim=1).float().mean().item())
    recalls = []
    for branch_id in range(len(BRANCH_NAMES)):
        mask = labels == branch_id
        if int(mask.sum().item()) == 0:
            recalls.append(float("nan"))
        else:
            recalls.append(float((pred[mask] == labels[mask]).float().mean().item()))
    return {
        "accuracy": acc,
        "macro_recall": float(np.nanmean(recalls)),
        "pattern_accuracy": pattern_acc,
        "recall_elastic": recalls[0],
        "recall_smooth": recalls[1],
        "recall_left_edge": recalls[2],
        "recall_right_edge": recalls[3],
        "recall_apex": recalls[4],
    }


def _confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
    true = labels.detach().cpu().numpy().reshape(-1)
    cm = np.zeros((len(BRANCH_NAMES), len(BRANCH_NAMES)), dtype=np.int64)
    np.add.at(cm, (true, pred), 1)
    return cm


def _branch_frequencies(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)
    counts /= np.sum(counts)
    return {name: float(counts[i]) for i, name in enumerate(BRANCH_NAMES)}


def _score_tuple(iid_metrics: dict[str, float], hard_metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        0.5 * (iid_metrics["pattern_accuracy"] + hard_metrics["pattern_accuracy"]),
        0.5 * (iid_metrics["macro_recall"] + hard_metrics["macro_recall"]),
        0.5 * (iid_metrics["accuracy"] + hard_metrics["accuracy"]),
    )


def _score_to_dict(score: tuple[float, float, float]) -> dict[str, float]:
    return {
        "pattern_mean": float(score[0]),
        "macro_recall_mean": float(score[1]),
        "accuracy_mean": float(score[2]),
    }


def _sample_exact_count(
    seed_bank: dict[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_parts: list[np.ndarray] = []
    disp_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    total = 0
    attempt = 0
    while total < sample_count:
        attempt += 1
        need = sample_count - total
        draw_count = max(int(math.ceil(need * 1.4)), 256)
        coords, disp, _strain, branch, _material, _valid = synthesize_element_states_from_seeded_noise(
            seed_bank,
            sample_count=draw_count,
            seed=seed + 97 * attempt,
            noise_scale=noise_scale,
            selection=selection,
        )
        if coords.shape[0] == 0:
            if attempt >= 40:
                raise RuntimeError("Failed to draw any valid synthetic element states.")
            continue
        coords_parts.append(coords)
        disp_parts.append(disp)
        branch_parts.append(branch)
        total += coords.shape[0]
        if attempt >= 40 and total < sample_count:
            raise RuntimeError(
                f"Failed to reach requested synthetic count {sample_count}; only collected {total} valid elements."
            )
    coords_full = np.concatenate(coords_parts, axis=0)[:sample_count]
    disp_full = np.concatenate(disp_parts, axis=0)[:sample_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:sample_count]
    return coords_full, disp_full, branch_full


def _draw_streaming_recipe(
    seed_bank: dict[str, np.ndarray],
    *,
    recipe: list[dict[str, float | str]],
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_parts: list[np.ndarray] = []
    disp_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    assigned = 0
    for idx, item in enumerate(recipe):
        if idx == len(recipe) - 1:
            part_count = sample_count - assigned
        else:
            part_count = int(round(sample_count * float(item["fraction"])))
            part_count = min(part_count, sample_count - assigned)
        if part_count <= 0:
            continue
        coords, disp, branch = _sample_exact_count(
            seed_bank,
            sample_count=part_count,
            seed=seed + 1000 * (idx + 1),
            noise_scale=float(item["noise_scale"]),
            selection=str(item["selection"]),
        )
        coords_parts.append(coords)
        disp_parts.append(disp)
        branch_parts.append(branch)
        assigned += part_count
    coords_full = np.concatenate(coords_parts, axis=0)
    disp_full = np.concatenate(disp_parts, axis=0)
    branch_full = np.concatenate(branch_parts, axis=0)
    return coords_full, disp_full, branch_full


def _spread_pick_exact(
    call_names: list[str],
    *,
    count: int,
    regimes: dict[str, dict[str, float]],
) -> list[str]:
    ranked = sorted(call_names, key=lambda name: regimes[name]["strain_norm_p95"])
    positions = np.linspace(0, len(ranked) - 1, num=count)
    picked: list[str] = []
    used: set[str] = set()
    for pos in positions:
        center = int(round(pos))
        offsets = sorted(range(len(ranked)), key=lambda idx: abs(idx - center))
        for idx in offsets:
            name = ranked[idx]
            if name not in used:
                used.add(name)
                picked.append(name)
                break
    if len(picked) != count:
        raise RuntimeError(f"Expected {count} picked calls, got {len(picked)}.")
    return picked


def _split_train_eval_seed_calls(
    generator_fit_calls: list[str],
    *,
    regimes: dict[str, dict[str, float]],
) -> tuple[list[str], list[str]]:
    selected = _spread_pick_exact(generator_fit_calls, count=32, regimes=regimes)
    eval_positions = np.linspace(0, len(selected) - 1, num=8)
    eval_idx = {int(round(pos)) for pos in eval_positions}
    eval_calls = [name for idx, name in enumerate(selected) if idx in eval_idx]
    train_calls = [name for idx, name in enumerate(selected) if idx not in eval_idx]
    if len(train_calls) != 24 or len(eval_calls) != 8:
        raise RuntimeError("Failed to create 24/8 disjoint synthetic seed-call split.")
    return train_calls, eval_calls


def _scale_fn(x_mean: np.ndarray, x_std: np.ndarray):
    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    return scale


def _tensorize(x_np: np.ndarray, y_np: np.ndarray, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np.astype(np.int64)).to(device)


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _evaluate_named_sets(
    model: nn.Module,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for name, (x, y) in eval_sets.items():
            logits = model(x)
            out[name] = _metrics(logits, y)
    return out


def _best_from_eval(metrics_by_name: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    return _score_tuple(metrics_by_name["synthetic_iid_val"], metrics_by_name["synthetic_hard_val"])


def _lbfgs_tail(
    model: nn.Module,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    class_weights: torch.Tensor,
    epochs: int,
    lr: float,
    max_iter: int,
    history_size: int,
    best_score: tuple[float, float, float],
) -> tuple[tuple[float, float, float], dict[str, torch.Tensor] | None, list[dict[str, float]]]:
    logs: list[dict[str, float]] = []
    accepted_state: dict[str, torch.Tensor] | None = None
    best_local = best_score
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(x_train.device))
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    for epoch in range(1, epochs + 1):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train)
            loss = loss_fn(logits.reshape(-1, len(BRANCH_NAMES)), y_train.reshape(-1))
            loss.backward()
            return loss

        model.train(True)
        optimizer.step(closure)
        train_loss = float(
            loss_fn(model(x_train).reshape(-1, len(BRANCH_NAMES)), y_train.reshape(-1)).detach().cpu().item()
        )
        metrics_by_name = _evaluate_named_sets(model, eval_sets)
        score = _best_from_eval(metrics_by_name)
        logs.append(
            {
                "lbfgs_epoch": epoch,
                "train_loss": train_loss,
                **{f"{name}_{k}": float(v) for name, metrics in metrics_by_name.items() for k, v in metrics.items()},
                **{f"score_{k}": v for k, v in _score_to_dict(score).items()},
            }
        )
        if score > best_local:
            best_local = score
            accepted_state = copy.deepcopy(model.state_dict())
    return best_local, accepted_state, logs


def _plot_gate_curves(rows: list[dict[str, float]], output_path: Path, *, title: str) -> None:
    if not rows:
        return
    epoch = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epoch, [r["train_loss"] for r in rows], label="train_loss")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    for key, label in [
        ("val_accuracy", "point acc"),
        ("val_macro_recall", "macro recall"),
        ("val_pattern_accuracy", "pattern acc"),
    ]:
        axes[1].plot(epoch, [r[key] for r in rows], label=label)
    axes[1].set_title("Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for key, label in [
        ("lr", "lr"),
    ]:
        axes[2].plot(epoch, [r[key] for r in rows], label=label)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_streaming_history(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    epoch = [r["global_epoch"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epoch, [r["train_loss"] for r in rows], label="train_loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].grid(True, alpha=0.3)

    for key, label in [
        ("synthetic_iid_val_accuracy", "iid acc"),
        ("synthetic_iid_val_macro_recall", "iid macro"),
        ("synthetic_iid_val_pattern_accuracy", "iid pattern"),
        ("synthetic_hard_val_accuracy", "hard acc"),
        ("synthetic_hard_val_macro_recall", "hard macro"),
        ("synthetic_hard_val_pattern_accuracy", "hard pattern"),
    ]:
        axes[0, 1].plot(epoch, [r[key] for r in rows], label=label)
    axes[0, 1].set_title("Synthetic Validation")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    for key, label in [
        ("real_val_accuracy", "real val acc"),
        ("real_val_macro_recall", "real val macro"),
        ("real_test_accuracy", "real test acc"),
        ("real_test_macro_recall", "real test macro"),
    ]:
        axes[1, 0].plot(epoch, [r[key] for r in rows], label=label)
    axes[1, 0].set_title("Real Diagnostics")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(epoch, [r["lr"] for r in rows], label="lr")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_frequencies(branch_freqs: dict[str, dict[str, float]], output_path: Path) -> None:
    names = list(branch_freqs.keys())
    x = np.arange(len(BRANCH_NAMES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, name in enumerate(names):
        vals = [branch_freqs[name][branch] for branch in BRANCH_NAMES]
        ax.bar(x + (idx - (len(names) - 1) / 2) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(BRANCH_NAMES, rotation=20)
    ax.set_ylabel("Fraction")
    ax.set_title("Branch Frequency by Benchmark Split")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(confusions: dict[str, np.ndarray], output_path: Path) -> None:
    names = list(confusions.keys())
    cols = 2
    rows = int(math.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax, name in zip(axes_arr.flat, names):
        cm = confusions[name].astype(np.float64)
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        norm = cm / row_sum
        im = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(name)
        ax.set_xticks(range(len(BRANCH_NAMES)))
        ax.set_xticklabels(BRANCH_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(len(BRANCH_NAMES)))
        ax.set_yticklabels(BRANCH_NAMES)
        for i in range(len(BRANCH_NAMES)):
            for j in range(len(BRANCH_NAMES)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    for ax in axes_arr.flat[len(names):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes_arr.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _train_fixed_gate(
    *,
    gate_name: str,
    model_specs: list[tuple[int, int]],
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    scale,
    device: torch.device,
    output_dir: Path,
    lr: float,
    weight_decay: float,
    batch_sizes: list[int],
    stage_max_epochs: int,
    stage_patience: int,
    plateau_patience: int,
    plateau_factor: float,
    min_lr: float,
    lbfgs_epochs: int,
    lbfgs_lr: float,
    lbfgs_max_iter: int,
    lbfgs_history_size: int,
    use_stage_schedule: bool,
    success_point_acc: float,
    success_pattern_acc: float,
) -> dict[str, object]:
    x_train = torch.from_numpy(scale(x_train_np)).to(device)
    y_train = torch.from_numpy(y_train_np.astype(np.int64)).to(device)
    x_val = torch.from_numpy(scale(x_val_np)).to(device)
    y_val = torch.from_numpy(y_val_np.astype(np.int64)).to(device)
    class_weights = _compute_class_weights(y_train_np)
    eval_sets = {"synthetic_iid_val": (x_val, y_val), "synthetic_hard_val": (x_val, y_val)}

    gate_result: dict[str, object] = {"models": [], "passed": False}

    for width, depth in model_specs:
        spec_name = f"w{width}_d{depth}"
        model = ElementBranchMLP(in_dim=x_train.shape[1], width=width, depth=depth).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        history: list[dict[str, float]] = []
        best_score = (-float("inf"), -float("inf"), -float("inf"))
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        stage_lr = lr
        global_epoch = 0

        if use_stage_schedule:
            stage_plan = [(bs, stage_max_epochs, stage_patience) for bs in batch_sizes]
        else:
            stage_plan = [(batch_sizes[0], stage_max_epochs, stage_max_epochs + 1)]

        for stage_index, (batch_size, stage_epochs, stage_pat) in enumerate(stage_plan, start=1):
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=plateau_factor,
                patience=plateau_patience,
                min_lr=min_lr,
            )
            no_improve = 0
            best_stage = (-float("inf"), -float("inf"), -float("inf"))

            loader = DataLoader(TensorDataset(x_train.cpu(), y_train.cpu()), batch_size=batch_size, shuffle=True)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

            for _ in range(stage_epochs):
                global_epoch += 1
                model.train(True)
                train_loss = 0.0
                train_count = 0
                for xb_cpu, yb_cpu in loader:
                    xb = xb_cpu.to(device)
                    yb = yb_cpu.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits.reshape(-1, len(BRANCH_NAMES)), yb.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item()) * xb.shape[0]
                    train_count += xb.shape[0]
                train_loss /= max(train_count, 1)

                metrics = _metrics(model(x_val), y_val)
                score = (metrics["pattern_accuracy"], metrics["macro_recall"], metrics["accuracy"])
                scheduler.step(score[0])
                row = {
                    "epoch": global_epoch,
                    "stage_index": stage_index,
                    "batch_size": batch_size,
                    "train_loss": train_loss,
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_recall": metrics["macro_recall"],
                    "val_pattern_accuracy": metrics["pattern_accuracy"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                history.append(row)
                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = global_epoch
                if score > best_stage:
                    best_stage = score
                    no_improve = 0
                else:
                    no_improve += 1
                if global_epoch == 1 or global_epoch % 50 == 0:
                    print(
                        f"[{gate_name} {spec_name}] epoch={global_epoch} stage={stage_index}/{len(stage_plan)} "
                        f"batch={batch_size} lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"train_loss={train_loss:.4f} val_acc={metrics['accuracy']:.4f} "
                        f"val_macro={metrics['macro_recall']:.4f} val_pattern={metrics['pattern_accuracy']:.4f}"
                    )
                if use_stage_schedule and no_improve >= stage_pat:
                    break
            stage_lr = float(optimizer.param_groups[0]["lr"])

        if best_state is None:
            raise RuntimeError(f"{gate_name} {spec_name} produced no checkpoint.")

        model.load_state_dict(best_state)
        lbfgs_logs: list[dict[str, float]] = []
        accepted_lbfgs = False
        if lbfgs_epochs > 0:
            lbfgs_best, lbfgs_state, lbfgs_logs = _lbfgs_tail(
                model,
                x_train=x_train,
                y_train=y_train,
                eval_sets=eval_sets,
                class_weights=class_weights,
                epochs=lbfgs_epochs,
                lr=lbfgs_lr,
                max_iter=lbfgs_max_iter,
                history_size=lbfgs_history_size,
                best_score=best_score,
            )
            if lbfgs_state is not None and lbfgs_best > best_score:
                accepted_lbfgs = True
                best_score = lbfgs_best
                best_state = lbfgs_state
                model.load_state_dict(best_state)

        final_metrics = _metrics(model(x_val), y_val)
        model_out_dir = output_dir / f"{gate_name}_{spec_name}"
        model_out_dir.mkdir(parents=True, exist_ok=True)
        (model_out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (model_out_dir / "lbfgs.json").write_text(json.dumps(lbfgs_logs, indent=2), encoding="utf-8")
        torch.save(
            {
                "state_dict": best_state,
                "width": width,
                "depth": depth,
            },
            model_out_dir / "best.pt",
        )
        _plot_gate_curves(history, model_out_dir / "history.png", title=f"{gate_name} {spec_name}")
        result = {
            "width": width,
            "depth": depth,
            "best_epoch": best_epoch,
            "accepted_lbfgs": accepted_lbfgs,
            "metrics": final_metrics,
            "score": _score_to_dict(best_score),
            "checkpoint": str(model_out_dir / "best.pt"),
            "history_plot": str(model_out_dir / "history.png"),
        }
        gate_result["models"].append(result)

        if (
            final_metrics["accuracy"] >= success_point_acc
            and final_metrics["pattern_accuracy"] >= success_pattern_acc
            and not gate_result["passed"]
        ):
            gate_result["passed"] = True
            gate_result["winner"] = result

    if not gate_result["passed"]:
        gate_result["winner"] = max(gate_result["models"], key=lambda row: tuple(row["score"].values()))
    return gate_result


def _train_gate_c(
    *,
    checkpoint_path: Path,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    train_seed_bank: dict[str, np.ndarray],
    eval_sets_np: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    output_dir: Path,
    cycles: list[dict[str, object]],
    batch_sizes: list[int],
    stage_max_epochs: int,
    stage_patience: int,
    plateau_patience: int,
    plateau_factor: float,
    min_lr: float,
    weight_decay: float,
    lbfgs_epochs: int,
    lbfgs_lr: float,
    lbfgs_max_iter: int,
    lbfgs_history_size: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    width = int(checkpoint["width"])
    depth = int(checkpoint["depth"])
    model = ElementBranchMLP(in_dim=int(x_mean.shape[0]), width=width, depth=depth).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    scale = _scale_fn(x_mean, x_std)
    eval_sets = {
        name: _tensorize(scale(x_np), y_np, device=device)
        for name, (x_np, y_np) in eval_sets_np.items()
    }
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_cycle = 0
    history: list[dict[str, float]] = []
    global_epoch = 0
    start = time.time()

    for cycle_idx, cycle_cfg in enumerate(cycles, start=1):
        stage_lr = float(cycle_cfg["base_lr"])
        cycle_best_score = (-float("inf"), -float("inf"), -float("inf"))
        cycle_best_state = copy.deepcopy(model.state_dict())
        cycle_best_epoch = global_epoch
        recipe = cycle_cfg["recipe"]
        print(
            f"[gate_c cycle-start] cycle={cycle_idx}/{len(cycles)} base_lr={stage_lr:.2e} "
            f"train_elements={cycle_cfg['train_elements_per_epoch']}"
        )

        for stage_index, batch_size in enumerate(batch_sizes, start=1):
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=plateau_factor,
                patience=plateau_patience,
                min_lr=min_lr,
            )
            stage_best = (-float("inf"), -float("inf"), -float("inf"))
            stage_no_improve = 0
            for _ in range(stage_max_epochs):
                global_epoch += 1
                coords_syn, disp_syn, branch_syn = _draw_streaming_recipe(
                    train_seed_bank,
                    recipe=recipe,
                    sample_count=int(cycle_cfg["train_elements_per_epoch"]),
                    seed=880000 + 1000 * cycle_idx + global_epoch,
                )
                x_train_np = scale(_element_features(coords_syn, disp_syn))
                y_train_np = branch_syn.astype(np.int64)
                x_train = torch.from_numpy(x_train_np)
                y_train = torch.from_numpy(y_train_np)
                loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
                loss_fn = nn.CrossEntropyLoss(weight=_compute_class_weights(y_train_np).to(device))

                model.train(True)
                train_loss = 0.0
                train_count = 0
                for xb_cpu, yb_cpu in loader:
                    xb = xb_cpu.to(device)
                    yb = yb_cpu.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits.reshape(-1, len(BRANCH_NAMES)), yb.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item()) * xb.shape[0]
                    train_count += xb.shape[0]
                train_loss /= max(train_count, 1)

                metrics_by_name = _evaluate_named_sets(model, eval_sets)
                score = _best_from_eval(metrics_by_name)
                scheduler.step(score[0])
                row = {
                    "global_epoch": global_epoch,
                    "cycle": cycle_idx,
                    "stage_index": stage_index,
                    "batch_size": batch_size,
                    "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "runtime_s": time.time() - start,
                }
                for name, metrics in metrics_by_name.items():
                    for key, value in metrics.items():
                        row[f"{name}_{key}"] = float(value)
                for key, value in _score_to_dict(score).items():
                    row[f"score_{key}"] = value
                history.append(row)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = global_epoch
                    best_cycle = cycle_idx
                if score > cycle_best_score:
                    cycle_best_score = score
                    cycle_best_state = copy.deepcopy(model.state_dict())
                    cycle_best_epoch = global_epoch
                if score > stage_best:
                    stage_best = score
                    stage_no_improve = 0
                else:
                    stage_no_improve += 1

                if global_epoch == 1 or global_epoch % 10 == 0:
                    print(
                        f"[gate_c epoch {global_epoch:04d}] cycle={cycle_idx}/{len(cycles)} "
                        f"stage={stage_index}/{len(batch_sizes)} batch={batch_size} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} runtime={time.time() - start:.1f}s "
                        f"train_loss={train_loss:.4f} iid_pattern={metrics_by_name['synthetic_iid_val']['pattern_accuracy']:.4f} "
                        f"hard_pattern={metrics_by_name['synthetic_hard_val']['pattern_accuracy']:.4f} "
                        f"iid_acc={metrics_by_name['synthetic_iid_val']['accuracy']:.4f} "
                        f"hard_acc={metrics_by_name['synthetic_hard_val']['accuracy']:.4f} "
                        f"real_test_acc={metrics_by_name['real_test']['accuracy']:.4f}"
                    )
                if stage_no_improve >= stage_patience:
                    break
            stage_lr = float(optimizer.param_groups[0]["lr"])

        model.load_state_dict(cycle_best_state)
        coords_cache, disp_cache, branch_cache = _draw_streaming_recipe(
            train_seed_bank,
            recipe=recipe,
            sample_count=4096,
            seed=990000 + cycle_idx,
        )
        x_cache, y_cache = _tensorize(scale(_element_features(coords_cache, disp_cache)), branch_cache.astype(np.int64), device=device)
        lbfgs_best, lbfgs_state, lbfgs_logs = _lbfgs_tail(
            model,
            x_train=x_cache,
            y_train=y_cache,
            eval_sets=eval_sets,
            class_weights=_compute_class_weights(branch_cache.astype(np.int64)),
            epochs=lbfgs_epochs,
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            best_score=cycle_best_score,
        )
        (output_dir / f"cycle_{cycle_idx}_lbfgs.json").write_text(json.dumps(lbfgs_logs, indent=2), encoding="utf-8")
        if lbfgs_state is not None and lbfgs_best > cycle_best_score:
            cycle_best_score = lbfgs_best
            cycle_best_state = lbfgs_state
        model.load_state_dict(cycle_best_state)
        if cycle_best_score > best_score:
            best_score = cycle_best_score
            best_state = copy.deepcopy(cycle_best_state)
            best_epoch = cycle_best_epoch
            best_cycle = cycle_idx

    model.load_state_dict(best_state)
    final_metrics = _evaluate_named_sets(model, eval_sets)
    history_path = output_dir / "gate_c_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_streaming_history(history, output_dir / "gate_c_history.png")
    torch.save({"state_dict": best_state, "width": width, "depth": depth, "x_mean": x_mean, "x_std": x_std}, output_dir / "best.pt")
    confusions = {}
    for name, (x, y) in eval_sets.items():
        confusions[name] = _confusion_matrix(model(x), y)
    _plot_confusions(confusions, output_dir / "gate_c_confusions.png")
    return {
        "passed_iid": final_metrics["synthetic_iid_test"]["accuracy"] >= 0.98
        and final_metrics["synthetic_iid_test"]["pattern_accuracy"] >= 0.90,
        "passed_hard": final_metrics["synthetic_hard_test"]["accuracy"] >= 0.95
        and final_metrics["synthetic_hard_test"]["pattern_accuracy"] >= 0.80,
        "best_epoch": best_epoch,
        "best_cycle": best_cycle,
        "score": _score_to_dict(best_score),
        "metrics": final_metrics,
        "checkpoint": str(output_dir / "best.pt"),
        "history_plot": str(output_dir / "gate_c_history.png"),
        "confusion_plot": str(output_dir / "gate_c_confusions.png"),
    }


def _write_report(
    report_path: Path,
    *,
    artifact_dir: Path,
    benchmark_summary: dict[str, object],
    gate_a: dict[str, object],
    gate_b: dict[str, object] | None,
    gate_c: dict[str, object] | None,
) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    lines: list[str] = [
        "# Cover Layer Direct Element Branch Predictor: Synthetic-Domain Mastery",
        "",
        "## Summary",
        "",
        "This report focuses on the **direct one-material raw-element model**:",
        "",
        "- input: canonicalized `coords(10x3) + disp(10x3)`",
        "- output: `11 x 5` branch logits",
        "- training objective for this phase: **master the synthetic domain first**",
        "- real validation/test are tracked only as diagnostics",
        "",
        "## Frozen Synthetic Benchmark",
        "",
        "- synthetic train seed calls: `24`",
        "- synthetic eval seed calls: `8`",
        "- synthetic IID val/test: `4096 / 8192`",
        "- synthetic hard val/test: `4096 / 8192`",
        "- fixed hard benchmark noise scale: `0.35`",
        "",
        "![Branch benchmark frequencies](../" + str(rel / "benchmark_branch_frequencies.png") + ")",
        "",
        "Benchmark branch frequencies:",
        "",
        "```json",
        json.dumps(benchmark_summary["branch_frequencies"], indent=2),
        "```",
        "",
        "## Gate A: Tiny Memorization",
        "",
        "Gate A asks whether the direct model can fit a fixed synthetic slice at all.",
        "",
    ]
    for model in gate_a["models"]:
        metrics = model["metrics"]
        lines.extend(
            [
                f"### `w{model['width']} d{model['depth']}`",
                "",
                f"- best epoch: `{model['best_epoch']}`",
                f"- val point accuracy: `{metrics['accuracy']:.4f}`",
                f"- val macro recall: `{metrics['macro_recall']:.4f}`",
                f"- val pattern accuracy: `{metrics['pattern_accuracy']:.4f}`",
                f"- LBFGS accepted: `{model['accepted_lbfgs']}`",
                "",
                f"![Gate A history](../{Path(model['history_plot']).relative_to(report_path.parent.parent)})",
                "",
            ]
        )
    if gate_a["passed"]:
        winner = gate_a["winner"]
        lines.extend(
            [
                "Gate A **passed**.",
                "",
                f"- winner: `w{winner['width']} d{winner['depth']}`",
                "",
            ]
        )
    else:
        winner = gate_a["winner"]
        lines.extend(
            [
                "Gate A **failed** the synthetic memorization target.",
                "",
                f"- best available model: `w{winner['width']} d{winner['depth']}`",
                "- per the ladder, later stages were not executed because this is no longer a simple undertraining signal.",
                "",
            ]
        )

    if gate_b is not None:
        lines.extend(
            [
                "## Gate B: Fixed Medium Synthetic Mastery",
                "",
            ]
        )
        for model in gate_b["models"]:
            metrics = model["metrics"]
            lines.extend(
                [
                    f"### `w{model['width']} d{model['depth']}`",
                    "",
                    f"- val point accuracy: `{metrics['accuracy']:.4f}`",
                    f"- val macro recall: `{metrics['macro_recall']:.4f}`",
                    f"- val pattern accuracy: `{metrics['pattern_accuracy']:.4f}`",
                    f"![Gate B history](../{Path(model['history_plot']).relative_to(report_path.parent.parent)})",
                    "",
                ]
            )
        lines.extend([f"Gate B passed: `{gate_b['passed']}`", ""])

    if gate_c is not None:
        metrics = gate_c["metrics"]
        lines.extend(
            [
                "## Gate C: Streaming Synthetic-Domain Training",
                "",
                f"- best epoch: `{gate_c['best_epoch']}`",
                f"- best cycle: `{gate_c['best_cycle']}`",
                f"- synthetic IID test accuracy / pattern: `{metrics['synthetic_iid_test']['accuracy']:.4f}` / `{metrics['synthetic_iid_test']['pattern_accuracy']:.4f}`",
                f"- synthetic hard test accuracy / pattern: `{metrics['synthetic_hard_test']['accuracy']:.4f}` / `{metrics['synthetic_hard_test']['pattern_accuracy']:.4f}`",
                f"- real test accuracy / pattern: `{metrics['real_test']['accuracy']:.4f}` / `{metrics['real_test']['pattern_accuracy']:.4f}`",
                "",
                "![Gate C history](../" + str(Path(gate_c["history_plot"]).relative_to(report_path.parent.parent)) + ")",
                "",
                "![Gate C confusions](../" + str(Path(gate_c["confusion_plot"]).relative_to(report_path.parent.parent)) + ")",
                "",
                f"Synthetic IID target passed: `{gate_c['passed_iid']}`",
                f"Synthetic hard target passed: `{gate_c['passed_hard']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Conclusion",
            "",
            "This phase is successful only if the direct raw-element MLP masters its **synthetic** domain first.",
            "",
            "- If Gate A fails, the issue is no longer just schedule length; there is a deeper model/data-alignment problem.",
            "- If Gate A passes but Gate B or C fail, the issue is likely full-pattern label complexity or model inductive bias.",
            "- Only after synthetic mastery should the real diagnostic gap be used to argue about generator-domain mismatch.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-domain mastery study for the direct cover-layer element branch predictor.")
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json"),
    )
    parser.add_argument(
        "--regime-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_regimes.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_element_branch_synth_mastery_20260314"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_element_branch_predictor_synth_mastery.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_calls(args.split_json)
    regimes = load_call_regimes(args.regime_json)
    synth_train_seed_calls, synth_eval_seed_calls = _split_train_eval_seed_calls(splits["generator_fit"], regimes=regimes)
    real_val_calls = pick_calls(splits["real_val"], count=4, selection="spread_p95", regimes=regimes)
    real_test_calls = pick_calls(splits["real_test"], count=4, selection="spread_p95", regimes=regimes)

    coords_train_seed, disp_train_seed, _strain_train_seed, branch_train_seed, material_train_seed = collect_blocks(
        args.export,
        call_names=synth_train_seed_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 1,
    )
    coords_eval_seed, disp_eval_seed, _strain_eval_seed, branch_eval_seed, material_eval_seed = collect_blocks(
        args.export,
        call_names=synth_eval_seed_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 2,
    )
    train_seed_bank = fit_seed_noise_bank(coords_train_seed, disp_train_seed, branch_train_seed, material_train_seed)
    eval_seed_bank = fit_seed_noise_bank(coords_eval_seed, disp_eval_seed, branch_eval_seed, material_eval_seed)

    coords_calib, disp_calib, branch_calib = _sample_exact_count(
        train_seed_bank,
        sample_count=8192,
        seed=args.seed + 3,
        noise_scale=0.20,
        selection="branch_balanced",
    )
    x_calib = _element_features(coords_calib, disp_calib)
    x_mean = x_calib.mean(axis=0)
    x_std = np.where(x_calib.std(axis=0) < 1.0e-6, 1.0, x_calib.std(axis=0))
    scale = _scale_fn(x_mean, x_std)

    def fixed_eval(seed_bank, count, noise_scale, selection, seed):
        coords, disp, branch = _sample_exact_count(
            seed_bank,
            sample_count=count,
            seed=seed,
            noise_scale=noise_scale,
            selection=selection,
        )
        return _element_features(coords, disp), branch.astype(np.int64)

    synthetic_iid_val = fixed_eval(eval_seed_bank, 4096, 0.20, "branch_balanced", args.seed + 10)
    synthetic_iid_test = fixed_eval(eval_seed_bank, 8192, 0.20, "branch_balanced", args.seed + 11)
    synthetic_hard_val = fixed_eval(eval_seed_bank, 4096, 0.35, "branch_balanced", args.seed + 12)
    synthetic_hard_test = fixed_eval(eval_seed_bank, 8192, 0.35, "branch_balanced", args.seed + 13)

    coords_real_val, disp_real_val, _strain_real_val, branch_real_val, _ = collect_blocks(
        args.export,
        call_names=real_val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 20,
    )
    coords_real_test, disp_real_test, _strain_real_test, branch_real_test, _ = collect_blocks(
        args.export,
        call_names=real_test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 21,
    )
    real_val = (_element_features(coords_real_val, disp_real_val), branch_real_val.astype(np.int64))
    real_test = (_element_features(coords_real_test, disp_real_test), branch_real_test.astype(np.int64))

    benchmark_summary = {
        "synth_train_seed_calls": synth_train_seed_calls,
        "synth_eval_seed_calls": synth_eval_seed_calls,
        "real_val_calls": real_val_calls,
        "real_test_calls": real_test_calls,
        "branch_frequencies": {
            "train_seed_real": _branch_frequencies(branch_train_seed.astype(np.int64)),
            "synthetic_iid_val": _branch_frequencies(synthetic_iid_val[1]),
            "synthetic_iid_test": _branch_frequencies(synthetic_iid_test[1]),
            "synthetic_hard_val": _branch_frequencies(synthetic_hard_val[1]),
            "synthetic_hard_test": _branch_frequencies(synthetic_hard_test[1]),
            "real_val": _branch_frequencies(real_val[1]),
            "real_test": _branch_frequencies(real_test[1]),
        },
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    _plot_benchmark_frequencies(benchmark_summary["branch_frequencies"], args.output_dir / "benchmark_branch_frequencies.png")

    gate_a_train = fixed_eval(train_seed_bank, 256, 0.20, "branch_balanced", args.seed + 30)
    gate_a = _train_fixed_gate(
        gate_name="gate_a",
        model_specs=[(128, 4), (256, 6)],
        x_train_np=gate_a_train[0],
        y_train_np=gate_a_train[1],
        x_val_np=gate_a_train[0],
        y_val_np=gate_a_train[1],
        scale=scale,
        device=device,
        output_dir=args.output_dir,
        lr=1.0e-3,
        weight_decay=1.0e-4,
        batch_sizes=[64],
        stage_max_epochs=600,
        stage_patience=601,
        plateau_patience=20,
        plateau_factor=0.5,
        min_lr=1.0e-6,
        lbfgs_epochs=3,
        lbfgs_lr=0.25,
        lbfgs_max_iter=20,
        lbfgs_history_size=50,
        use_stage_schedule=False,
        success_point_acc=0.99,
        success_pattern_acc=0.95,
    )
    (args.output_dir / "gate_a_summary.json").write_text(json.dumps(gate_a, indent=2), encoding="utf-8")

    gate_b = None
    gate_c = None
    if gate_a["passed"]:
        winner = gate_a["winner"]
        gate_b_train = fixed_eval(train_seed_bank, 2048, 0.20, "branch_balanced", args.seed + 40)
        gate_b_val = fixed_eval(eval_seed_bank, 2048, 0.20, "branch_balanced", args.seed + 41)
        gate_b = _train_fixed_gate(
            gate_name="gate_b",
            model_specs=[(int(winner["width"]), int(winner["depth"]))],
            x_train_np=gate_b_train[0],
            y_train_np=gate_b_train[1],
            x_val_np=gate_b_val[0],
            y_val_np=gate_b_val[1],
            scale=scale,
            device=device,
            output_dir=args.output_dir,
            lr=1.0e-3,
            weight_decay=1.0e-4,
            batch_sizes=[32, 64, 128, 256],
            stage_max_epochs=80,
            stage_patience=24,
            plateau_patience=8,
            plateau_factor=0.5,
            min_lr=1.0e-6,
            lbfgs_epochs=5,
            lbfgs_lr=0.25,
            lbfgs_max_iter=20,
            lbfgs_history_size=50,
            use_stage_schedule=True,
            success_point_acc=0.98,
            success_pattern_acc=0.90,
        )
        if not gate_b["passed"]:
            alternate = None
            for model_row in gate_a["models"]:
                spec = (int(model_row["width"]), int(model_row["depth"]))
                if spec != (int(winner["width"]), int(winner["depth"])):
                    alternate = spec
                    break
        else:
            alternate = None
        if not gate_b["passed"] and alternate is not None:
            gate_b_alt = _train_fixed_gate(
                gate_name="gate_b_alt",
                model_specs=[alternate],
                x_train_np=gate_b_train[0],
                y_train_np=gate_b_train[1],
                x_val_np=gate_b_val[0],
                y_val_np=gate_b_val[1],
                scale=scale,
                device=device,
                output_dir=args.output_dir,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                batch_sizes=[32, 64, 128, 256],
                stage_max_epochs=80,
                stage_patience=24,
                plateau_patience=8,
                plateau_factor=0.5,
                min_lr=1.0e-6,
                lbfgs_epochs=5,
                lbfgs_lr=0.25,
                lbfgs_max_iter=20,
                lbfgs_history_size=50,
                use_stage_schedule=True,
                success_point_acc=0.98,
                success_pattern_acc=0.90,
            )
            gate_b["models"].extend(gate_b_alt["models"])
            if gate_b_alt["passed"]:
                gate_b["passed"] = True
                gate_b["winner"] = gate_b_alt["winner"]
            else:
                gate_b["winner"] = max(gate_b["models"], key=lambda row: tuple(row["score"].values()))
        (args.output_dir / "gate_b_summary.json").write_text(json.dumps(gate_b, indent=2), encoding="utf-8")

        if gate_b["passed"]:
            gate_c = _train_gate_c(
                checkpoint_path=Path(gate_b["winner"]["checkpoint"]),
                x_mean=x_mean,
                x_std=x_std,
                train_seed_bank=train_seed_bank,
                eval_sets_np={
                    "synthetic_iid_val": synthetic_iid_val,
                    "synthetic_iid_test": synthetic_iid_test,
                    "synthetic_hard_val": synthetic_hard_val,
                    "synthetic_hard_test": synthetic_hard_test,
                    "real_val": real_val,
                    "real_test": real_test,
                },
                device=device,
                output_dir=args.output_dir,
                cycles=[
                    {
                        "base_lr": 1.0e-3,
                        "train_elements_per_epoch": 4096,
                        "recipe": [{"fraction": 1.0, "selection": "uniform", "noise_scale": 0.10}],
                    },
                    {
                        "base_lr": 5.0e-4,
                        "train_elements_per_epoch": 6144,
                        "recipe": [{"fraction": 1.0, "selection": "branch_balanced", "noise_scale": 0.20}],
                    },
                    {
                        "base_lr": 2.5e-4,
                        "train_elements_per_epoch": 8192,
                        "recipe": [
                            {"fraction": 0.70, "selection": "branch_balanced", "noise_scale": 0.20},
                            {"fraction": 0.30, "selection": "branch_balanced", "noise_scale": 0.35},
                        ],
                    },
                    {
                        "base_lr": 1.0e-4,
                        "train_elements_per_epoch": 8192,
                        "recipe": [
                            {"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.20},
                            {"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.35},
                        ],
                    },
                ],
                batch_sizes=[32, 64, 128, 256, 512],
                stage_max_epochs=60,
                stage_patience=20,
                plateau_patience=6,
                plateau_factor=0.5,
                min_lr=1.0e-6,
                weight_decay=1.0e-4,
                lbfgs_epochs=3,
                lbfgs_lr=0.25,
                lbfgs_max_iter=20,
                lbfgs_history_size=50,
            )
            (args.output_dir / "gate_c_summary.json").write_text(json.dumps(gate_c, indent=2), encoding="utf-8")

    _write_report(
        args.report_path,
        artifact_dir=args.output_dir,
        benchmark_summary=benchmark_summary,
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
    )
    print(json.dumps({"benchmark_summary": benchmark_summary, "gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c}, indent=2))


if __name__ == "__main__":
    main()
