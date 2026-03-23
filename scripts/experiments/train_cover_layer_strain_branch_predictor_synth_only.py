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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mc_surrogate.cover_branch_generation import (
    collect_blocks,
    fit_seed_noise_bank,
    load_call_regimes,
    load_split_calls,
    synthesize_from_seeded_noise,
)
from mc_surrogate.principal_branch_generation import (
    fit_principal_hybrid_bank,
    summarize_branch_geometry,
    synthesize_from_principal_hybrid,
)
from mc_surrogate.models import (
    build_trial_features,
    build_trial_principal_features,
    compute_trial_stress,
)
from mc_surrogate.voigt import principal_values_and_vectors_from_strain, stress_voigt_to_tensor

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")
PLASTIC_BRANCH_NAMES = BRANCH_NAMES[1:]


class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalBranchNet(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        self.trunk = nn.Sequential(*layers)
        self.elastic_head = nn.Linear(width, 2)
        self.plastic_head = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.elastic_head(h), self.plastic_head(h)


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


def _split_seed_calls(
    generator_fit_calls: list[str],
    *,
    regimes: dict[str, dict[str, float]],
    train_count: int,
    eval_count: int,
) -> tuple[list[str], list[str]]:
    selected = _spread_pick_exact(generator_fit_calls, count=train_count + eval_count, regimes=regimes)
    eval_positions = np.linspace(0, len(selected) - 1, num=eval_count)
    eval_idx = {int(round(pos)) for pos in eval_positions}
    eval_calls = [name for idx, name in enumerate(selected) if idx in eval_idx]
    train_calls = [name for idx, name in enumerate(selected) if idx not in eval_idx]
    if len(train_calls) != train_count or len(eval_calls) != eval_count:
        raise RuntimeError("Failed to create requested train/eval call split.")
    return train_calls, eval_calls


def _sample_exact_count(
    seed_bank: dict[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    generator_kind = str(seed_bank.get("generator_kind", "element_seed"))
    target_count = sample_count * 11 if generator_kind == "principal_hybrid" else sample_count
    total = 0
    attempt = 0
    while total < target_count:
        attempt += 1
        need = target_count - total
        draw_count = max(int(math.ceil(need * 1.4)), 256)
        if generator_kind == "principal_hybrid":
            point_count = max(draw_count, 256)
            strain, branch, material, _valid = synthesize_from_principal_hybrid(
                seed_bank,
                sample_count=point_count,
                seed=seed + 97 * attempt,
                noise_scale=noise_scale,
                selection=selection,
            )
        else:
            strain, branch, material, _valid = synthesize_from_seeded_noise(
                seed_bank,
                sample_count=draw_count,
                seed=seed + 97 * attempt,
                noise_scale=noise_scale,
                selection=selection,
            )
        if strain.shape[0] == 0:
            if attempt >= 40:
                raise RuntimeError("Failed to draw any valid synthetic states.")
            continue
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material)
        total += strain.shape[0]
        if attempt >= 40 and total < sample_count:
            raise RuntimeError(f"Failed to reach requested synthetic count {target_count}; collected {total}.")
    strain_full = np.concatenate(strain_parts, axis=0)[:target_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:target_count]
    material_full = np.concatenate(material_parts, axis=0)[:target_count]
    return strain_full, branch_full, material_full


def _principal_stresses_from_voigt(stress_voigt: np.ndarray) -> np.ndarray:
    stress_tensor = stress_voigt_to_tensor(stress_voigt)
    eigvals = np.linalg.eigvalsh(stress_tensor)
    return eigvals[:, ::-1].astype(np.float32)


def _trial_yield_scalar(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    strain_principal: np.ndarray,
) -> np.ndarray:
    c_bar = material_reduced[:, 0]
    sin_phi = np.clip(material_reduced[:, 1], -0.999999, 0.999999)
    shear = material_reduced[:, 2]
    lame = material_reduced[:, 4]
    i1 = np.sum(strain_principal, axis=1)
    e1 = strain_principal[:, 0]
    e3 = strain_principal[:, 2]
    f_trial = 2.0 * shear * ((1.0 + sin_phi) * e1 - (1.0 - sin_phi) * e3) + 2.0 * (lame * sin_phi) * i1 - c_bar
    scale = np.maximum(c_bar, 1.0)
    return np.arcsinh(f_trial / scale).astype(np.float32)[:, None]


def _build_point_features(
    strain: np.ndarray,
    material: np.ndarray,
    *,
    feature_set: str,
) -> np.ndarray:
    strain_arr = np.asarray(strain, dtype=np.float32)
    material_arr = np.asarray(material, dtype=np.float32)
    strain_flat = strain_arr.reshape(-1, 6).astype(np.float32)
    if strain_arr.ndim == 3:
        material_flat = np.repeat(material_arr.astype(np.float32), strain_arr.shape[1], axis=0)
    elif strain_arr.ndim == 2:
        if material_arr.shape[0] != strain_arr.shape[0]:
            raise ValueError(
                f"Pointwise strain/material shapes do not align: {strain_arr.shape} vs {material_arr.shape}."
            )
        material_flat = material_arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported strain shape {strain_arr.shape}.")
    if feature_set == "strain_only":
        return np.arcsinh(strain_flat).astype(np.float32)
    if feature_set == "trial_raw_material":
        return build_trial_features(strain_flat, material_flat)
    if feature_set == "trial_principal_yield_material":
        strain_principal, _ = principal_values_and_vectors_from_strain(strain_flat)
        trial_stress = compute_trial_stress(strain_flat, material_flat)
        trial_principal = _principal_stresses_from_voigt(trial_stress)
        base = build_trial_principal_features(strain_principal, material_flat, trial_principal)
        yield_scalar = _trial_yield_scalar(strain_flat, material_flat, strain_principal)
        return np.column_stack([base, yield_scalar]).astype(np.float32)
    raise ValueError(f"Unknown feature set {feature_set!r}.")


def _flatten_pointwise_labels(branch: np.ndarray) -> np.ndarray:
    return np.asarray(branch).reshape(-1).astype(np.int64)


def _feature_description(feature_set: str) -> tuple[str, str]:
    if feature_set == "strain_only":
        return (
            "transformed engineering strain only (`asinh(E)` followed by z-score)",
            "no material or trial-stress features",
        )
    if feature_set == "trial_raw_material":
        return (
            "engineering strain, elastic trial stress, and reduced material features",
            "includes `asinh(E)`, `asinh(sigma_trial / c_bar)`, and reduced material logs",
        )
    if feature_set == "trial_principal_yield_material":
        return (
            "principal/invariant trial-state features plus reduced material and trial-yield scalar",
            "includes ordered principal strains, trial principal stresses, invariant-like scalars, reduced material logs, and `asinh(f_trial / c_bar)`",
        )
    raise ValueError(f"Unknown feature set {feature_set!r}.")


def _branch_frequencies(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)
    counts /= np.sum(counts)
    return {name: float(counts[i]) for i, name in enumerate(BRANCH_NAMES)}


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(BRANCH_NAMES)).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _binary_class_weights(labels: np.ndarray) -> torch.Tensor:
    binary = (labels != 0).astype(np.int64)
    counts = np.bincount(binary, minlength=2).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _plastic_class_weights(labels: np.ndarray) -> torch.Tensor:
    plastic = labels[labels != 0] - 1
    if plastic.size == 0:
        return torch.ones(4, dtype=torch.float32)
    counts = np.bincount(plastic, minlength=4).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _predict_labels(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        elastic_logits, plastic_logits = out
        elastic_pred = torch.argmax(elastic_logits, dim=1)
        plastic_pred = torch.argmax(plastic_logits, dim=1) + 1
        return torch.where(elastic_pred == 0, torch.zeros_like(elastic_pred), plastic_pred)
    return torch.argmax(out, dim=1)


def _metrics(pred: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    acc = float((pred == labels).float().mean().item())
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
        "recall_elastic": recalls[0],
        "recall_smooth": recalls[1],
        "recall_left_edge": recalls[2],
        "recall_right_edge": recalls[3],
        "recall_apex": recalls[4],
    }


def _confusion_matrix(pred: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    pred = pred.detach().cpu().numpy()
    true = labels.detach().cpu().numpy()
    cm = np.zeros((len(BRANCH_NAMES), len(BRANCH_NAMES)), dtype=np.int64)
    np.add.at(cm, (true, pred), 1)
    return cm


def _score(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    edge_mean = 0.5 * (metrics["recall_left_edge"] + metrics["recall_right_edge"])
    return (metrics["macro_recall"], metrics["recall_smooth"], edge_mean, metrics["accuracy"])


def _score_dict(score: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "macro_recall": float(score[0]),
        "smooth_recall": float(score[1]),
        "edge_recall_mean": float(score[2]),
        "accuracy": float(score[3]),
    }


def _compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    model_type: str,
    class_weights: torch.Tensor,
    binary_weights: torch.Tensor | None = None,
    plastic_weights: torch.Tensor | None = None,
    plastic_loss_weight: float = 1.0,
) -> torch.Tensor:
    out = model(x)
    if model_type == "flat":
        if isinstance(out, tuple):
            raise RuntimeError("Flat model unexpectedly returned hierarchical outputs.")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(x.device))
        return loss_fn(out, y)
    if model_type == "hierarchical":
        if not isinstance(out, tuple):
            raise RuntimeError("Hierarchical model unexpectedly returned flat outputs.")
        elastic_logits, plastic_logits = out
        binary_targets = (y != 0).long()
        bin_loss = nn.CrossEntropyLoss(weight=binary_weights.to(x.device))(elastic_logits, binary_targets)
        plastic_mask = y != 0
        if int(plastic_mask.sum().item()) == 0:
            return bin_loss
        plastic_targets = (y[plastic_mask] - 1).long()
        plast_loss = nn.CrossEntropyLoss(weight=plastic_weights.to(x.device))(plastic_logits[plastic_mask], plastic_targets)
        return bin_loss + plastic_loss_weight * plast_loss
    raise ValueError(f"Unknown model_type {model_type!r}.")


def _draw_training_recipe(
    seed_bank: dict[str, np.ndarray],
    *,
    recipe: list[dict[str, float | str]],
    element_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    assigned = 0
    for idx, item in enumerate(recipe):
        if idx == len(recipe) - 1:
            part_count = element_count - assigned
        else:
            part_count = int(round(element_count * float(item["fraction"])))
            part_count = min(part_count, element_count - assigned)
        if part_count <= 0:
            continue
        strain, branch, material = _sample_exact_count(
            seed_bank,
            sample_count=part_count,
            seed=seed + 1000 * (idx + 1),
            noise_scale=float(item["noise_scale"]),
            selection=str(item["selection"]),
        )
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material)
        assigned += part_count
    return (
        np.concatenate(strain_parts, axis=0),
        np.concatenate(branch_parts, axis=0),
        np.concatenate(material_parts, axis=0),
    )


def _evaluate_sets(
    model: nn.Module,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for name, (x, y) in eval_sets.items():
            pred = _predict_labels(model, x)
            out[name] = _metrics(pred, y)
    return out


def _lbfgs_tail(
    model: nn.Module,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    model_type: str,
    class_weights: torch.Tensor,
    binary_weights: torch.Tensor | None,
    plastic_weights: torch.Tensor | None,
    plastic_loss_weight: float,
    epochs: int,
    lr: float,
    max_iter: int,
    history_size: int,
    best_score: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], dict[str, torch.Tensor] | None]:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    best_local = best_score
    accepted_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = _compute_loss(
                model,
                x_train,
                y_train,
                model_type=model_type,
                class_weights=class_weights,
                binary_weights=binary_weights,
                plastic_weights=plastic_weights,
                plastic_loss_weight=plastic_loss_weight,
            )
            loss.backward()
            return loss

        model.train(True)
        optimizer.step(closure)
        metrics_by_name = _evaluate_sets(model, eval_sets)
        score = _score(metrics_by_name["synthetic_val"])
        if score > best_local:
            best_local = score
            accepted_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_local, accepted_state


def _plot_history(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    epoch = [row["global_epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(epoch, [row["train_loss"] for row in rows], label="train_loss")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)

    for key, label in [
        ("synthetic_val_accuracy", "syn val acc"),
        ("synthetic_val_macro_recall", "syn val macro"),
        ("synthetic_test_accuracy", "syn test acc"),
        ("synthetic_test_macro_recall", "syn test macro"),
    ]:
        axes[0, 1].plot(epoch, [row[key] for row in rows], label=label)
    axes[0, 1].set_title("Synthetic Benchmarks")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    for key, label in [
        ("real_val_accuracy", "real val acc"),
        ("real_val_macro_recall", "real val macro"),
        ("real_test_accuracy", "real test acc"),
        ("real_test_macro_recall", "real test macro"),
    ]:
        axes[1, 0].plot(epoch, [row[key] for row in rows], label=label)
    axes[1, 0].set_title("Real Diagnostics")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(epoch, [row["lr"] for row in rows], label="lr")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(confusions: dict[str, np.ndarray], output_path: Path) -> None:
    names = list(confusions.keys())
    rows = int(math.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 4.5 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, 2)
    im = None
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
    if im is not None:
        fig.colorbar(im, ax=axes_arr.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_branch_frequencies(freqs: dict[str, dict[str, float]], output_path: Path) -> None:
    names = list(freqs.keys())
    x = np.arange(len(BRANCH_NAMES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, name in enumerate(names):
        vals = [freqs[name][branch] for branch in BRANCH_NAMES]
        ax.bar(x + (idx - (len(names) - 1) / 2) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(BRANCH_NAMES, rotation=20)
    ax.set_ylabel("Fraction")
    ax.set_title("Branch Frequency by Synthetic/Real Split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_model_card(
    report_path: Path,
    *,
    artifact_dir: Path,
    summary: dict[str, object],
    benchmark_summary: dict[str, object],
) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    metrics = summary["metrics"]
    feature_input_line, feature_detail_line = _feature_description(str(summary["feature_set"]))
    lines = [
        "# Cover Layer Branch Predictor Model Card",
        "",
        "## Summary",
        "",
        "This card describes the current **synthetic-only branch** classifier for `cover_layer`.",
        "",
        "- task: `E -> branch`",
        f"- feature set: `{summary['feature_set']}`",
        f"- model type: `{summary['model_type']}`",
        f"- generator recipe: `{summary['recipe_mode']}`",
        f"- input: {feature_input_line}",
        "- output: one of `elastic / smooth / left_edge / right_edge / apex`",
        f"- detail: {feature_detail_line}",
        "",
        "## Architecture",
        "",
        f"- model type: `{summary['model_type']}`",
        f"- input dimension: `{summary['input_dim']}`",
        f"- hidden width: `{summary['width']}`",
        f"- depth: `{summary['depth']}`",
        f"- activation: `GELU`",
        f"- output classes: `5`",
        f"- plastic loss weight: `{summary['plastic_loss_weight']}`",
        "",
        "## Synthetic Benchmark",
        "",
        f"- synthetic train seed calls: `{len(benchmark_summary['train_seed_calls'])}`",
        f"- synthetic eval seed calls: `{len(benchmark_summary['eval_seed_calls'])}`",
        f"- synthetic val elements: `{benchmark_summary['synthetic_val_elements']}`",
        f"- synthetic test elements: `{benchmark_summary['synthetic_test_elements']}`",
        f"- generator: `{summary['generator_name']}`",
        "",
        "![Branch frequencies](../" + str(rel / "benchmark_branch_frequencies.png") + ")",
        "",
        "## Training",
        "",
        f"- cycles: `{summary['cycles']}`",
        f"- batch-size schedule: `{summary['batch_sizes']}`",
        f"- base learning rates by cycle: `{summary['cycle_base_lrs']}`",
        f"- stage max epochs: `{summary['stage_max_epochs']}`",
        f"- stage patience: `{summary['stage_patience']}`",
        f"- plateau patience/factor: `{summary['plateau_patience']} / {summary['plateau_factor']}`",
        f"- LBFGS tail per cycle: `{summary['lbfgs_epochs']}` epochs",
        "",
        "Training regenerates synthetic data every epoch and checkpoints only on the fixed synthetic validation split.",
        "",
        "## Convergence",
        "",
        "![Training history](../" + str(rel / "training_history.png") + ")",
        "",
        "## Metrics",
        "",
        f"- synthetic val accuracy / macro recall: `{metrics['synthetic_val']['accuracy']:.4f}` / `{metrics['synthetic_val']['macro_recall']:.4f}`",
        f"- synthetic test accuracy / macro recall: `{metrics['synthetic_test']['accuracy']:.4f}` / `{metrics['synthetic_test']['macro_recall']:.4f}`",
        f"- real val accuracy / macro recall: `{metrics['real_val']['accuracy']:.4f}` / `{metrics['real_val']['macro_recall']:.4f}`",
        f"- real test accuracy / macro recall: `{metrics['real_test']['accuracy']:.4f}` / `{metrics['real_test']['macro_recall']:.4f}`",
        "",
        "## Confusions",
        "",
        "![Confusions](../" + str(rel / "confusions.png") + ")",
        "",
        "## Assessment",
        "",
        "This run should first be judged by synthetic validation/test quality, because the whole purpose was to prove the synthetic-domain training loop itself. Real validation/test are diagnostic only.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-only cover-layer strain-to-branch staged trainer.")
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
        default=Path("experiment_runs/real_sim/cover_layer_strain_branch_predictor_synth_only_20260314"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_strain_branch_predictor_model_card.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--train-seed-calls", type=int, default=24)
    parser.add_argument("--eval-seed-calls", type=int, default=8)
    parser.add_argument("--synthetic-calibration-elements", type=int, default=4096)
    parser.add_argument("--synthetic-val-elements", type=int, default=4096)
    parser.add_argument("--synthetic-test-elements", type=int, default=16384)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--model-type", choices=("flat", "hierarchical"), default="flat")
    parser.add_argument("--generator-kind", choices=("element_seed", "principal_hybrid"), default="element_seed")
    parser.add_argument(
        "--feature-set",
        choices=("strain_only", "trial_raw_material", "trial_principal_yield_material"),
        default="strain_only",
    )
    parser.add_argument("--recipe-mode", choices=("default", "smooth_focus", "expert_principal"), default="default")
    parser.add_argument("--plastic-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=6)
    parser.add_argument("--stage-patience", type=int, default=20)
    parser.add_argument("--stage-max-epochs", type=int, default=60)
    parser.add_argument("--batch-sizes", type=str, default="64,128,256,512,1024")
    parser.add_argument("--cycle-elements", type=str, default="")
    parser.add_argument("--cycle-base-lrs", type=str, default="")
    parser.add_argument("--lbfgs-epochs", type=int, default=3)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_calls(args.split_json)
    regimes = load_call_regimes(args.regime_json)
    train_seed_calls, eval_seed_calls = _split_seed_calls(
        splits["generator_fit"],
        regimes=regimes,
        train_count=args.train_seed_calls,
        eval_count=args.eval_seed_calls,
    )
    real_val_calls = _spread_pick_exact(splits["real_val"], count=4, regimes=regimes)
    real_test_calls = _spread_pick_exact(splits["real_test"], count=4, regimes=regimes)

    def build_seed_bank(call_names: list[str], seed: int) -> dict[str, np.ndarray]:
        coords, disp, strain, branch, material = collect_blocks(
            args.export,
            call_names=call_names,
            max_elements_per_call=args.max_elements_per_call,
            seed=seed,
        )
        if args.generator_kind == "principal_hybrid":
            return fit_principal_hybrid_bank(strain, branch, material)
        bank = fit_seed_noise_bank(coords, disp, branch, material)
        bank["generator_kind"] = "element_seed"
        return bank

    train_seed_bank = build_seed_bank(train_seed_calls, args.seed + 1)
    eval_seed_bank = build_seed_bank(eval_seed_calls, args.seed + 2)

    strain_calib, branch_calib, material_calib = _sample_exact_count(
        train_seed_bank,
        sample_count=args.synthetic_calibration_elements,
        seed=args.seed + 3,
        noise_scale=0.20,
        selection="branch_balanced",
    )
    x_calib = _build_point_features(strain_calib, material_calib, feature_set=args.feature_set)
    y_calib = _flatten_pointwise_labels(branch_calib)
    x_mean = x_calib.mean(axis=0)
    x_std = np.where(x_calib.std(axis=0) < 1.0e-6, 1.0, x_calib.std(axis=0))

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    benchmark_recipe = [{"fraction": 1.0, "selection": "branch_balanced", "noise_scale": 0.20}]
    if args.generator_kind == "principal_hybrid" and args.recipe_mode == "expert_principal":
        benchmark_recipe = [
            {"fraction": 0.60, "selection": "branch_balanced", "noise_scale": 0.18},
            {"fraction": 0.25, "selection": "boundary_smooth_right", "noise_scale": 0.05},
            {"fraction": 0.15, "selection": "tail", "noise_scale": 0.25},
        ]
    strain_syn_val, branch_syn_val, material_syn_val = _draw_training_recipe(
        eval_seed_bank,
        recipe=benchmark_recipe,
        element_count=args.synthetic_val_elements,
        seed=args.seed + 10,
    )
    strain_syn_test, branch_syn_test, material_syn_test = _draw_training_recipe(
        eval_seed_bank,
        recipe=benchmark_recipe,
        element_count=args.synthetic_test_elements,
        seed=args.seed + 11,
    )
    x_syn_val_np = _build_point_features(strain_syn_val, material_syn_val, feature_set=args.feature_set)
    y_syn_val_np = _flatten_pointwise_labels(branch_syn_val)
    x_syn_test_np = _build_point_features(strain_syn_test, material_syn_test, feature_set=args.feature_set)
    y_syn_test_np = _flatten_pointwise_labels(branch_syn_test)
    x_syn_val = torch.from_numpy(scale(x_syn_val_np)).to(device)
    y_syn_val = torch.from_numpy(y_syn_val_np).to(device)
    x_syn_test = torch.from_numpy(scale(x_syn_test_np)).to(device)
    y_syn_test = torch.from_numpy(y_syn_test_np).to(device)

    _, _, strain_real_val, branch_real_val, material_real_val = collect_blocks(
        args.export,
        call_names=real_val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 20,
    )
    _, _, strain_real_test, branch_real_test, material_real_test = collect_blocks(
        args.export,
        call_names=real_test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 21,
    )
    x_real_val_np = _build_point_features(strain_real_val, material_real_val, feature_set=args.feature_set)
    y_real_val_np = _flatten_pointwise_labels(branch_real_val)
    x_real_test_np = _build_point_features(strain_real_test, material_real_test, feature_set=args.feature_set)
    y_real_test_np = _flatten_pointwise_labels(branch_real_test)
    x_real_val = torch.from_numpy(scale(x_real_val_np)).to(device)
    y_real_val = torch.from_numpy(y_real_val_np).to(device)
    x_real_test = torch.from_numpy(scale(x_real_test_np)).to(device)
    y_real_test = torch.from_numpy(y_real_test_np).to(device)

    eval_sets = {
        "synthetic_val": (x_syn_val, y_syn_val),
        "synthetic_test": (x_syn_test, y_syn_test),
        "real_val": (x_real_val, y_real_val),
        "real_test": (x_real_test, y_real_test),
    }

    benchmark_summary = {
        "generator_kind": args.generator_kind,
        "train_seed_calls": train_seed_calls,
        "eval_seed_calls": eval_seed_calls,
        "real_val_calls": real_val_calls,
        "real_test_calls": real_test_calls,
        "synthetic_val_elements": args.synthetic_val_elements,
        "synthetic_test_elements": args.synthetic_test_elements,
        "branch_frequencies": {
            "synthetic_val": _branch_frequencies(y_syn_val_np),
            "synthetic_test": _branch_frequencies(y_syn_test_np),
            "real_val": _branch_frequencies(y_real_val_np),
            "real_test": _branch_frequencies(y_real_test_np),
        },
    }
    if args.generator_kind == "principal_hybrid":
        benchmark_summary["coverage"] = {
            "synthetic_val": summarize_branch_geometry(strain_syn_val, branch_syn_val, material_syn_val),
            "synthetic_test": summarize_branch_geometry(strain_syn_test, branch_syn_test, material_syn_test),
            "real_val": summarize_branch_geometry(strain_real_val, branch_real_val, material_real_val),
            "real_test": summarize_branch_geometry(strain_real_test, branch_real_test, material_real_test),
        }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    _plot_branch_frequencies(benchmark_summary["branch_frequencies"], args.output_dir / "benchmark_branch_frequencies.png")

    if args.model_type == "flat":
        model = BranchMLP(in_dim=int(x_calib.shape[1]), width=args.width, depth=args.depth).to(device)
    elif args.model_type == "hierarchical":
        model = HierarchicalBranchNet(in_dim=int(x_calib.shape[1]), width=args.width, depth=args.depth).to(device)
    else:
        raise ValueError(f"Unknown model_type {args.model_type!r}.")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    cycle_specs = [
        {"name": "easy", "base_lr": 1.0e-3, "elements_per_epoch": 4096, "recipe": [{"fraction": 1.0, "selection": "uniform", "noise_scale": 0.10}]},
        {"name": "match", "base_lr": 5.0e-4, "elements_per_epoch": 6144, "recipe": [{"fraction": 1.0, "selection": "branch_balanced", "noise_scale": 0.20}]},
        {"name": "coverage", "base_lr": 2.5e-4, "elements_per_epoch": 8192, "recipe": [{"fraction": 0.70, "selection": "branch_balanced", "noise_scale": 0.20}, {"fraction": 0.30, "selection": "branch_balanced", "noise_scale": 0.25}]},
        {"name": "hard", "base_lr": 1.0e-4, "elements_per_epoch": 8192, "recipe": [{"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.20}, {"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.30}]},
    ]
    if args.recipe_mode == "smooth_focus":
        cycle_specs = [
            {
                "name": "easy",
                "base_lr": 1.0e-3,
                "elements_per_epoch": 4096,
                "recipe": [
                    {"fraction": 0.60, "selection": "uniform", "noise_scale": 0.10},
                    {"fraction": 0.40, "selection": "smooth_focus", "noise_scale": 0.10},
                ],
            },
            {
                "name": "match",
                "base_lr": 5.0e-4,
                "elements_per_epoch": 6144,
                "recipe": [
                    {"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.20},
                    {"fraction": 0.25, "selection": "smooth_focus", "noise_scale": 0.20},
                    {"fraction": 0.25, "selection": "smooth_edge", "noise_scale": 0.20},
                ],
            },
            {
                "name": "coverage",
                "base_lr": 2.5e-4,
                "elements_per_epoch": 8192,
                "recipe": [
                    {"fraction": 0.40, "selection": "branch_balanced", "noise_scale": 0.20},
                    {"fraction": 0.30, "selection": "smooth_focus", "noise_scale": 0.20},
                    {"fraction": 0.30, "selection": "smooth_edge", "noise_scale": 0.25},
                ],
            },
            {
                "name": "hard",
                "base_lr": 1.0e-4,
                "elements_per_epoch": 8192,
                "recipe": [
                    {"fraction": 0.30, "selection": "branch_balanced", "noise_scale": 0.20},
                    {"fraction": 0.35, "selection": "smooth_focus", "noise_scale": 0.25},
                    {"fraction": 0.35, "selection": "smooth_edge", "noise_scale": 0.30},
                ],
            },
        ]
    if args.recipe_mode == "expert_principal":
        cycle_specs = [
            {
                "name": "easy",
                "base_lr": 1.0e-3,
                "elements_per_epoch": 4096,
                "recipe": [
                    {"fraction": 0.70, "selection": "branch_balanced", "noise_scale": 0.12},
                    {"fraction": 0.30, "selection": "smooth_focus", "noise_scale": 0.10},
                ],
            },
            {
                "name": "match",
                "base_lr": 5.0e-4,
                "elements_per_epoch": 6144,
                "recipe": [
                    {"fraction": 0.60, "selection": "branch_balanced", "noise_scale": 0.18},
                    {"fraction": 0.25, "selection": "boundary_smooth_right", "noise_scale": 0.04},
                    {"fraction": 0.15, "selection": "tail", "noise_scale": 0.18},
                ],
            },
            {
                "name": "coverage",
                "base_lr": 2.5e-4,
                "elements_per_epoch": 8192,
                "recipe": [
                    {"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.20},
                    {"fraction": 0.30, "selection": "boundary_smooth_right", "noise_scale": 0.05},
                    {"fraction": 0.20, "selection": "tail", "noise_scale": 0.25},
                ],
            },
            {
                "name": "hard",
                "base_lr": 1.0e-4,
                "elements_per_epoch": 8192,
                "recipe": [
                    {"fraction": 0.35, "selection": "smooth_edge", "noise_scale": 0.22},
                    {"fraction": 0.40, "selection": "boundary_smooth_right", "noise_scale": 0.06},
                    {"fraction": 0.25, "selection": "tail", "noise_scale": 0.30},
                ],
            },
        ]
    if args.cycle_elements:
        cycle_elements = [int(x) for x in args.cycle_elements.split(",") if x.strip()]
        if len(cycle_elements) != len(cycle_specs):
            raise ValueError(f"--cycle-elements must provide {len(cycle_specs)} integers.")
        for cycle, elem_count in zip(cycle_specs, cycle_elements):
            cycle["elements_per_epoch"] = elem_count
    if args.cycle_base_lrs:
        cycle_base_lrs = [float(x) for x in args.cycle_base_lrs.split(",") if x.strip()]
        if len(cycle_base_lrs) != len(cycle_specs):
            raise ValueError(f"--cycle-base-lrs must provide {len(cycle_specs)} floats.")
        for cycle, base_lr in zip(cycle_specs, cycle_base_lrs):
            cycle["base_lr"] = base_lr

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_epoch = 0
    best_cycle = ""
    global_epoch = 0
    start = time.time()

    for cycle_index, cycle in enumerate(cycle_specs, start=1):
        stage_lr = float(cycle["base_lr"])
        cycle_best_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
        cycle_best_state = copy.deepcopy(model.state_dict())
        cycle_best_epoch = global_epoch
        print(
            f"[cycle-start] cycle={cycle_index}/{len(cycle_specs)} name={cycle['name']} "
            f"base_lr={stage_lr:.2e} elements_per_epoch={cycle['elements_per_epoch']}"
        )

        for stage_index, batch_size in enumerate(batch_sizes, start=1):
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                min_lr=args.min_lr,
            )
            stage_best = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
            stage_no_improve = 0
            for _ in range(args.stage_max_epochs):
                global_epoch += 1
                strain_syn, branch_syn, material_syn = _draw_training_recipe(
                    train_seed_bank,
                    recipe=cycle["recipe"],
                    element_count=int(cycle["elements_per_epoch"]),
                    seed=args.seed + 10000 * cycle_index + global_epoch,
                )
                x_train_np = _build_point_features(strain_syn, material_syn, feature_set=args.feature_set)
                y_train_np = _flatten_pointwise_labels(branch_syn)
                x_train = torch.from_numpy(scale(x_train_np))
                y_train = torch.from_numpy(y_train_np)
                class_weights = _class_weights(y_train_np)
                binary_weights = _binary_class_weights(y_train_np)
                plastic_weights = _plastic_class_weights(y_train_np)
                loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

                model.train(True)
                train_loss = 0.0
                train_count = 0
                for xb_cpu, yb_cpu in loader:
                    xb = xb_cpu.to(device)
                    yb = yb_cpu.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _compute_loss(
                        model,
                        xb,
                        yb,
                        model_type=args.model_type,
                        class_weights=class_weights,
                        binary_weights=binary_weights,
                        plastic_weights=plastic_weights,
                        plastic_loss_weight=args.plastic_loss_weight,
                    )
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item()) * xb.shape[0]
                    train_count += xb.shape[0]
                train_loss /= max(train_count, 1)

                metrics_by_name = _evaluate_sets(model, eval_sets)
                score = _score(metrics_by_name["synthetic_val"])
                scheduler.step(score[0])
                row = {
                    "global_epoch": global_epoch,
                    "cycle_index": cycle_index,
                    "cycle_name": cycle["name"],
                    "stage_index": stage_index,
                    "batch_size": batch_size,
                    "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "runtime_s": time.time() - start,
                }
                for split_name, split_metrics in metrics_by_name.items():
                    for key, value in split_metrics.items():
                        row[f"{split_name}_{key}"] = float(value)
                history.append(row)

                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = global_epoch
                    best_cycle = str(cycle["name"])
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
                        f"[epoch {global_epoch:04d}] cycle={cycle['name']} stage={stage_index}/{len(batch_sizes)} "
                        f"batch={batch_size} lr={optimizer.param_groups[0]['lr']:.2e} runtime={time.time()-start:.1f}s "
                        f"train_loss={train_loss:.4f} syn_val_acc={metrics_by_name['synthetic_val']['accuracy']:.4f} "
                        f"syn_val_macro={metrics_by_name['synthetic_val']['macro_recall']:.4f} "
                        f"syn_test_acc={metrics_by_name['synthetic_test']['accuracy']:.4f} "
                        f"syn_test_macro={metrics_by_name['synthetic_test']['macro_recall']:.4f} "
                        f"real_test_acc={metrics_by_name['real_test']['accuracy']:.4f}"
                    )
                if stage_no_improve >= args.stage_patience:
                    break
            stage_lr = float(optimizer.param_groups[0]["lr"])

        model.load_state_dict(cycle_best_state)
        strain_cache, branch_cache, material_cache = _draw_training_recipe(
            train_seed_bank,
            recipe=cycle["recipe"],
            element_count=2048,
            seed=args.seed + 990000 + cycle_index,
        )
        x_cache_np = _build_point_features(strain_cache, material_cache, feature_set=args.feature_set)
        y_cache_np = _flatten_pointwise_labels(branch_cache)
        x_cache = torch.from_numpy(scale(x_cache_np)).to(device)
        y_cache = torch.from_numpy(y_cache_np).to(device)
        lbfgs_best, lbfgs_state = _lbfgs_tail(
            model,
            x_train=x_cache,
            y_train=y_cache,
            eval_sets=eval_sets,
            model_type=args.model_type,
            class_weights=_class_weights(y_cache_np),
            binary_weights=_binary_class_weights(y_cache_np),
            plastic_weights=_plastic_class_weights(y_cache_np),
            plastic_loss_weight=args.plastic_loss_weight,
            epochs=args.lbfgs_epochs,
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size,
            best_score=cycle_best_score,
        )
        if lbfgs_state is not None and lbfgs_best > cycle_best_score:
            cycle_best_score = lbfgs_best
            cycle_best_state = lbfgs_state
        model.load_state_dict(cycle_best_state)
        if cycle_best_score > best_score:
            best_score = cycle_best_score
            best_state = {k: v.detach().cpu().clone() for k, v in cycle_best_state.items()}
            best_epoch = cycle_best_epoch
            best_cycle = str(cycle["name"])

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")
    model.load_state_dict(best_state)
    final_metrics = _evaluate_sets(model, eval_sets)
    confusions = {}
    for name, (x, y) in eval_sets.items():
        pred = _predict_labels(model, x)
        confusions[name] = _confusion_matrix(pred, y)
    _plot_history(history, args.output_dir / "training_history.png")
    _plot_confusions(confusions, args.output_dir / "confusions.png")
    checkpoint = {
        "state_dict": best_state,
        "x_mean": x_mean,
        "x_std": x_std,
        "width": args.width,
        "depth": args.depth,
        "input_dim": int(x_calib.shape[1]),
        "feature_set": args.feature_set,
        "model_type": args.model_type,
        "generator_kind": args.generator_kind,
        "recipe_mode": args.recipe_mode,
        "plastic_loss_weight": args.plastic_loss_weight,
        "branch_names": BRANCH_NAMES,
        "train_seed_calls": train_seed_calls,
        "eval_seed_calls": eval_seed_calls,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    summary = {
        "generator_name": args.generator_kind,
        "feature_set": args.feature_set,
        "model_type": args.model_type,
        "generator_kind": args.generator_kind,
        "recipe_mode": args.recipe_mode,
        "plastic_loss_weight": args.plastic_loss_weight,
        "input_dim": int(x_calib.shape[1]),
        "width": args.width,
        "depth": args.depth,
        "cycles": len(cycle_specs),
        "batch_sizes": batch_sizes,
        "cycle_base_lrs": [cycle["base_lr"] for cycle in cycle_specs],
        "stage_max_epochs": args.stage_max_epochs,
        "stage_patience": args.stage_patience,
        "plateau_patience": args.plateau_patience,
        "plateau_factor": args.plateau_factor,
        "lbfgs_epochs": args.lbfgs_epochs,
        "best_epoch": best_epoch,
        "best_cycle": best_cycle,
        "score": _score_dict(best_score),
        "metrics": final_metrics,
        "checkpoint": str(args.output_dir / "best.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _write_model_card(args.report_path, artifact_dir=args.output_dir, summary=summary, benchmark_summary=benchmark_summary)
    print(json.dumps({"summary": summary, "benchmark_summary": benchmark_summary}, indent=2))


if __name__ == "__main__":
    main()
