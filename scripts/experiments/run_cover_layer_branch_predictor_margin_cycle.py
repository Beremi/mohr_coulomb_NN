from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mc_surrogate.branch_geometry import compute_branch_geometry_principal
from mc_surrogate.mohr_coulomb import (
    branch_harm_metrics_3d,
    constitutive_update_3d,
    dispatch_branch_stress_3d,
)
from mc_surrogate.principal_branch_generation import (
    fit_principal_hybrid_bank,
    summarize_branch_geometry,
    synthesize_from_principal_hybrid,
)
from mc_surrogate.voigt import principal_values_and_vectors_from_strain

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")


def _load_trainer_module():
    script_path = Path(__file__).with_name("train_cover_layer_strain_branch_predictor_synth_only.py")
    spec = importlib.util.spec_from_file_location("cover_branch_trainer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load trainer module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict_numpy(model: nn.Module, x_np: np.ndarray, *, device: torch.device, batch_size: int = 16384) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.from_numpy(x_np[start:start + batch_size]).to(device)
            out = model(xb)
            if isinstance(out, tuple):
                elastic_logits, plastic_logits = out
                elastic_pred = torch.argmax(elastic_logits, dim=1)
                plastic_pred = torch.argmax(plastic_logits, dim=1) + 1
                pred = torch.where(elastic_pred == 0, torch.zeros_like(elastic_pred), plastic_pred)
            else:
                pred = torch.argmax(out, dim=1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _metrics_from_np(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = pred.reshape(-1)
    labels = labels.reshape(-1)
    acc = float(np.mean(pred == labels))
    recalls = []
    out = {"accuracy": acc}
    for idx, name in enumerate(BRANCH_NAMES):
        mask = labels == idx
        if np.any(mask):
            val = float(np.mean(pred[mask] == labels[mask]))
        else:
            val = float("nan")
        recalls.append(val)
        out[f"recall_{name}"] = val
    out["macro_recall"] = float(np.nanmean(recalls))
    return out


def _confusion_np(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(BRANCH_NAMES), len(BRANCH_NAMES)), dtype=np.int64)
    np.add.at(cm, (labels.reshape(-1), pred.reshape(-1)), 1)
    return cm


def _harm_summary(strain: np.ndarray, material: np.ndarray, labels: np.ndarray, pred: np.ndarray, *, tau: float) -> dict[str, float]:
    if material.ndim == 2 and material.shape[0] * 11 == strain.shape[0]:
        material_point = np.repeat(material.astype(np.float32), 11, axis=0)
    else:
        material_point = material.astype(np.float32).reshape(-1, 5)
    harm = branch_harm_metrics_3d(
        strain,
        pred,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
        tau=tau,
    )
    wrong_mask = harm.wrong_branch
    rel_wrong = harm.rel_e_sigma[wrong_mask]
    return {
        "wrong_rate": float(np.mean(wrong_mask)),
        "benign_fail_rate": float(np.mean(harm.benign_fail)),
        "harmful_fail_rate": float(np.mean(harm.harmful_fail)),
        "harmful_adjacent_fail_rate": float(np.mean(harm.harmful_adjacent_fail)),
        "harmful_non_adjacent_fail_rate": float(np.mean(harm.harmful_non_adjacent_fail)),
        "median_rel_e_sigma_wrong": float(np.median(rel_wrong)) if rel_wrong.size else 0.0,
        "p95_rel_e_sigma_wrong": float(np.quantile(rel_wrong, 0.95)) if rel_wrong.size else 0.0,
    }


def _harm_confusion_table(strain: np.ndarray, material: np.ndarray, labels: np.ndarray, pred: np.ndarray, *, tau: float) -> list[dict[str, object]]:
    if material.ndim == 2 and material.shape[0] * 11 == strain.shape[0]:
        material_point = np.repeat(material.astype(np.float32), 11, axis=0)
    else:
        material_point = material.astype(np.float32).reshape(-1, 5)
    harm = branch_harm_metrics_3d(
        strain,
        pred,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
        tau=tau,
    )
    rows: list[dict[str, object]] = []
    for true_id, true_name in enumerate(BRANCH_NAMES):
        for pred_id, pred_name in enumerate(BRANCH_NAMES):
            if true_id == pred_id:
                continue
            mask = (labels.reshape(-1) == true_id) & (pred.reshape(-1) == pred_id)
            if not np.any(mask):
                continue
            rel = harm.rel_e_sigma[mask]
            rows.append(
                {
                    "true_branch": true_name,
                    "pred_branch": pred_name,
                    "count": int(np.sum(mask)),
                    "harmful_rate": float(np.mean(harm.harmful_fail[mask])),
                    "median_rel_e_sigma": float(np.median(rel)),
                    "p95_rel_e_sigma": float(np.quantile(rel, 0.95)),
                }
            )
    rows.sort(key=lambda row: (row["harmful_rate"], row["count"]), reverse=True)
    return rows


def _stress_dispatch_metrics(strain: np.ndarray, material: np.ndarray, labels: np.ndarray, pred: np.ndarray, *, tau: float) -> dict[str, float]:
    if material.ndim == 2 and material.shape[0] * 11 == strain.shape[0]:
        material_point = np.repeat(material.astype(np.float32), 11, axis=0)
    else:
        material_point = material.astype(np.float32).reshape(-1, 5)
    exact = constitutive_update_3d(
        strain,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
    )
    pred_stress, pred_principal = dispatch_branch_stress_3d(
        strain,
        pred,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
    )
    harm = branch_harm_metrics_3d(
        strain,
        pred,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
        tau=tau,
    )
    denom = np.linalg.norm(exact.stress, axis=1) + material_point[:, 0] + 1.0e-12
    rel_full = np.linalg.norm(pred_stress - exact.stress, axis=1) / denom
    denom_pr = np.linalg.norm(exact.stress_principal, axis=1) + material_point[:, 0] + 1.0e-12
    rel_pr = np.linalg.norm(pred_principal - exact.stress_principal, axis=1) / denom_pr
    wrong_mask = labels.reshape(-1) != pred.reshape(-1)
    harmful_mask = harm.harmful_fail

    def subset_stats(name: str, mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {
                f"{name}_mean_rel_full_stress": 0.0,
                f"{name}_p95_rel_full_stress": 0.0,
                f"{name}_mean_rel_principal_stress": 0.0,
                f"{name}_p95_rel_principal_stress": 0.0,
            }
        return {
            f"{name}_mean_rel_full_stress": float(np.mean(rel_full[mask])),
            f"{name}_p95_rel_full_stress": float(np.quantile(rel_full[mask], 0.95)),
            f"{name}_mean_rel_principal_stress": float(np.mean(rel_pr[mask])),
            f"{name}_p95_rel_principal_stress": float(np.quantile(rel_pr[mask], 0.95)),
        }

    out = {}
    out.update(subset_stats("overall", np.ones(rel_full.shape[0], dtype=bool)))
    out.update(subset_stats("wrong_branch", wrong_mask))
    out.update(subset_stats("harmful_fail", harmful_mask))
    return out


def _selector_score(metrics: dict[str, dict[str, float]]) -> float:
    broad_macro = float(metrics["real_val_large"]["macro_recall"])
    hard_macro = float(metrics["real_val_hard"]["macro_recall"])
    smooth_hard = float(metrics["real_val_hard"]["recall_smooth"])
    return 0.5 * broad_macro + 0.3 * hard_macro + 0.2 * smooth_hard


def _synthetic_constraints_ok(metrics: dict[str, dict[str, float]], baseline_metrics: dict[str, dict[str, float]]) -> bool:
    core_drop = float(baseline_metrics["synthetic_core_val"]["accuracy"]) - float(metrics["synthetic_core_val"]["accuracy"])
    hard_drop = float(baseline_metrics["synthetic_hard_val"]["macro_recall"]) - float(metrics["synthetic_hard_val"]["macro_recall"])
    return core_drop <= 0.003 and hard_drop <= 0.005


def _evaluate_splits(
    model: nn.Module,
    eval_splits: dict[str, dict[str, np.ndarray]],
    *,
    device: torch.device,
    tau: float,
    batch_size: int = 16384,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, float]] = {}
    confusions: dict[str, np.ndarray] = {}
    harms: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {}
    for name, split in eval_splits.items():
        pred = _predict_numpy(model, split["x"], device=device, batch_size=batch_size)
        preds[name] = pred
        metrics[name] = _metrics_from_np(pred, split["y"])
        confusions[name] = _confusion_np(pred, split["y"])
        harms[name] = _harm_summary(split["strain"], split["material"], split["y"], pred, tau=tau)
    return metrics, confusions, harms, preds


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


def _hierarchical_loss(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    binary_weights: torch.Tensor,
    plastic_weights: torch.Tensor,
    plastic_loss_weight: float,
) -> torch.Tensor:
    elastic_logits, plastic_logits = model(xb)
    binary_targets = (yb != 0).long()
    bin_loss = F.cross_entropy(elastic_logits, binary_targets, weight=binary_weights.to(xb.device))
    plastic_mask = yb != 0
    if int(plastic_mask.sum().item()) == 0:
        return bin_loss
    plastic_targets = (yb[plastic_mask] - 1).long()
    plast_loss = F.cross_entropy(plastic_logits[plastic_mask], plastic_targets, weight=plastic_weights.to(xb.device))
    return bin_loss + plastic_loss_weight * plast_loss


def _draw_recipe_points(
    seed_bank: dict[str, np.ndarray],
    *,
    total_points: int,
    recipe: list[dict[str, float | str]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    bucket_parts: list[np.ndarray] = []
    assigned = 0
    for idx, item in enumerate(recipe):
        if idx == len(recipe) - 1:
            part_count = total_points - assigned
        else:
            part_count = int(round(total_points * float(item["fraction"])))
            part_count = min(part_count, total_points - assigned)
        if part_count <= 0:
            continue
        strain, branch, material, _valid = synthesize_from_principal_hybrid(
            seed_bank,
            sample_count=part_count,
            seed=seed + 1000 * (idx + 1),
            noise_scale=float(item["noise_scale"]),
            selection=str(item["selection"]),
        )
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material)
        bucket_parts.append(np.full(branch.shape[0], str(item["selection"]), dtype=object))
        assigned += part_count
    return (
        np.concatenate(strain_parts, axis=0),
        np.concatenate(branch_parts, axis=0),
        np.concatenate(material_parts, axis=0),
        np.concatenate(bucket_parts, axis=0),
    )


def _save_checkpoint(path: Path, *, base_ckpt: dict[str, object], model: nn.Module, extra: dict[str, object] | None = None) -> None:
    ckpt = copy.deepcopy(base_ckpt)
    ckpt["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def _inflate_input_checkpoint(
    ckpt: dict[str, object],
    trainer,
    *,
    new_input_dim: int,
    noise_scale: float,
    seed: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> dict[str, object]:
    old_input_dim = int(ckpt["input_dim"])
    if new_input_dim < old_input_dim:
        raise ValueError(f"Requested input dim {new_input_dim} is smaller than checkpoint input dim {old_input_dim}.")
    if new_input_dim == old_input_dim:
        inflated = copy.deepcopy(ckpt)
        inflated["x_mean"] = x_mean.astype(np.float32)
        inflated["x_std"] = x_std.astype(np.float32)
        return inflated

    model = trainer.HierarchicalBranchNet(
        in_dim=new_input_dim,
        width=int(ckpt["width"]),
        depth=int(ckpt["depth"]),
    )
    new_state = model.state_dict()
    old_state = ckpt["state_dict"]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    for name, tensor in new_state.items():
        old_tensor = old_state.get(name)
        if old_tensor is None:
            continue
        if tuple(old_tensor.shape) == tuple(tensor.shape):
            new_state[name] = old_tensor.detach().cpu().clone()
            continue
        if name == "trunk.0.weight" and old_tensor.shape[0] == tensor.shape[0]:
            filled = torch.empty_like(tensor)
            filled.normal_(mean=0.0, std=noise_scale, generator=rng)
            filled[:, : old_tensor.shape[1]] = old_tensor.detach().cpu()
            new_state[name] = filled
            continue
        raise RuntimeError(f"Unsupported shape inflation for {name}: {tuple(old_tensor.shape)} -> {tuple(tensor.shape)}")

    inflated = copy.deepcopy(ckpt)
    inflated["input_dim"] = int(new_input_dim)
    inflated["state_dict"] = {k: v.detach().cpu().clone() for k, v in new_state.items()}
    inflated["x_mean"] = x_mean.astype(np.float32)
    inflated["x_std"] = x_std.astype(np.float32)
    inflated["context_summary"] = True
    return inflated


def _plot_history(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    steps = [float(row["global_step"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].plot(steps, [float(row["train_loss"]) for row in rows], label="train loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].grid(True, alpha=0.3)

    for key, label in [
        ("selector_score", "selector"),
        ("real_val_large_macro_recall", "broad macro"),
        ("real_val_hard_macro_recall", "hard macro"),
        ("real_val_hard_recall_smooth", "hard smooth"),
    ]:
        axes[0, 1].plot(steps, [float(row[key]) for row in rows], label=label)
    axes[0, 1].set_title("Selector Metrics")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    for key, label in [
        ("real_val_large_harmful_fail_rate", "broad harmful"),
        ("real_val_large_harmful_adjacent_fail_rate", "broad harmful adj"),
        ("real_val_hard_harmful_fail_rate", "hard harmful"),
        ("synthetic_hard_val_macro_recall", "synthetic hard"),
    ]:
        axes[1, 0].plot(steps, [float(row[key]) for row in rows], label=label)
    axes[1, 0].set_title("Harm / Hard Metrics")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(steps, [float(row["lr"]) for row in rows], label="lr")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(confusions: dict[str, np.ndarray], output_path: Path) -> None:
    names = list(confusions.keys())
    rows = int(math.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.5 * rows))
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
    width = 0.10
    fig, ax = plt.subplots(figsize=(15, 5))
    for idx, name in enumerate(names):
        vals = [freqs[name][branch] for branch in BRANCH_NAMES]
        ax.bar(x + (idx - (len(names) - 1) / 2) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(BRANCH_NAMES, rotation=20)
    ax.set_ylabel("Fraction")
    ax.set_title("Branch Frequency by Split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _margin_features(strain: np.ndarray, material: np.ndarray) -> np.ndarray:
    strain_flat = np.asarray(strain, dtype=np.float32).reshape(-1, 6)
    if material.ndim == 2 and material.shape[0] * 11 == strain_flat.shape[0]:
        material_point = np.repeat(material.astype(np.float32), 11, axis=0)
    else:
        material_point = np.asarray(material, dtype=np.float32).reshape(-1, 5)
    principal, _ = principal_values_and_vectors_from_strain(strain_flat)
    geom = compute_branch_geometry_principal(
        principal,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
    )
    return np.column_stack(
        [
            geom.f_trial,
            geom.i1,
            geom.delta12,
            geom.delta23,
            geom.m_smooth_left,
            geom.m_smooth_right,
            geom.m_left_apex,
            geom.m_right_apex,
        ]
    ).astype(np.float32)


def _append_summary_context(base_features: np.ndarray, strain: np.ndarray, material: np.ndarray) -> np.ndarray:
    margin = _margin_features(strain, material)
    n = margin.shape[0]
    group_size = 11
    out = np.zeros((n, margin.shape[1] * 3), dtype=np.float32)
    start = 0
    while start < n:
        end = min(start + group_size, n)
        chunk = margin[start:end]
        summary = np.concatenate([chunk.min(axis=0), chunk.mean(axis=0), chunk.max(axis=0)], axis=0).astype(np.float32)
        out[start:end] = summary[None, :]
        start = end
    return np.concatenate([base_features.astype(np.float32), out], axis=1)


def _evaluate_full_real_val(
    trainer,
    model: nn.Module,
    *,
    export_path: Path,
    call_names: list[str],
    max_elements_per_call: int,
    seed: int,
    scale_fn,
    feature_set: str,
    append_context: bool,
    device: torch.device,
    tau: float,
) -> dict[str, object]:
    _, _, strain, branch, material = trainer.collect_blocks(
        export_path,
        call_names=call_names,
        max_elements_per_call=max_elements_per_call,
        seed=seed,
    )
    base = trainer._build_point_features(strain, material, feature_set=feature_set)
    x_np = _append_summary_context(base, strain, material) if append_context else base.astype(np.float32)
    x_scaled = scale_fn(x_np)
    y_np = branch.reshape(-1).astype(np.int64)
    pred = _predict_numpy(model, x_scaled, device=device)
    return {
        "metrics": _metrics_from_np(pred, y_np),
        "harm": _harm_summary(strain.reshape(-1, 6), material, y_np, pred, tau=tau),
    }


def main() -> None:
    trainer = _load_trainer_module()

    parser = argparse.ArgumentParser(description="Margin-aware branch-only cycle for the cover-layer predictor.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_100cycles_20260315/loop_17_accepted.pt"),
    )
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
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_margin_cycle_20260316"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_margin_cycle.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--train-seed-calls", type=int, default=24)
    parser.add_argument("--eval-seed-calls", type=int, default=8)
    parser.add_argument("--real-val-large-calls", type=int, default=32)
    parser.add_argument("--phase1-datasets", type=int, default=30)
    parser.add_argument("--phase1-eval-every", type=int, default=2)
    parser.add_argument("--phase1-lr", type=float, default=5.0e-7)
    parser.add_argument("--phase1-lbfgs-lr", type=float, default=1.0e-2)
    parser.add_argument("--phase1-top-datasets", type=int, default=10)
    parser.add_argument("--phase1-lbfgs-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--train-points", type=int, default=81920)
    parser.add_argument("--tau-harm", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--run-context-phase", action="store_true")
    parser.add_argument("--context-datasets", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_set = str(base_ckpt["feature_set"])
    model_type = str(base_ckpt.get("model_type", "hierarchical"))
    if model_type != "hierarchical":
        raise RuntimeError(f"This cycle expects hierarchical checkpoint, got {model_type!r}.")
    x_mean = np.asarray(base_ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(base_ckpt["x_std"], dtype=np.float32)
    x_std = np.where(np.abs(x_std) < 1.0e-6, 1.0, x_std)
    plastic_loss_weight = float(base_ckpt.get("plastic_loss_weight", 1.0))

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    model = trainer.HierarchicalBranchNet(
        in_dim=int(base_ckpt["input_dim"]),
        width=int(base_ckpt["width"]),
        depth=int(base_ckpt["depth"]),
    ).to(device)
    model.load_state_dict(base_ckpt["state_dict"])

    splits = trainer.load_split_calls(args.split_json)
    regimes = trainer.load_call_regimes(args.regime_json)
    train_seed_calls, eval_seed_calls = trainer._split_seed_calls(
        splits["generator_fit"],
        regimes=regimes,
        train_count=args.train_seed_calls,
        eval_count=args.eval_seed_calls,
    )
    real_val_slice_calls = trainer._spread_pick_exact(splits["real_val"], count=4, regimes=regimes)
    real_val_large_calls = trainer._spread_pick_exact(splits["real_val"], count=args.real_val_large_calls, regimes=regimes)
    real_test_calls = trainer._spread_pick_exact(splits["real_test"], count=4, regimes=regimes)

    def build_seed_bank(call_names: list[str], seed: int) -> dict[str, np.ndarray]:
        _, _, strain, branch, material = trainer.collect_blocks(
            args.export,
            call_names=call_names,
            max_elements_per_call=args.max_elements_per_call,
            seed=seed,
        )
        return fit_principal_hybrid_bank(strain, branch, material)

    train_seed_bank = build_seed_bank(train_seed_calls, args.seed + 1)
    eval_seed_bank = build_seed_bank(eval_seed_calls, args.seed + 2)

    generator_v2_recipe = [
        {"fraction": 0.35, "selection": "margin_bulk", "noise_scale": 0.18},
        {"fraction": 0.08, "selection": "boundary_smooth_right", "noise_scale": 0.05},
        {"fraction": 0.08, "selection": "boundary_smooth_left", "noise_scale": 0.05},
        {"fraction": 0.07, "selection": "edge_apex_right", "noise_scale": 0.16},
        {"fraction": 0.07, "selection": "edge_apex_left", "noise_scale": 0.16},
        {"fraction": 0.05, "selection": "yield_tube", "noise_scale": 0.03},
        {"fraction": 0.15, "selection": "tail", "noise_scale": 0.25},
        {"fraction": 0.10, "selection": "small_gap", "noise_scale": 0.08},
        {"fraction": 0.05, "selection": "loading_paths", "noise_scale": 0.10},
    ]
    hard_recipe = [
        {"fraction": 0.20, "selection": "margin_bulk", "noise_scale": 0.20},
        {"fraction": 0.12, "selection": "boundary_smooth_right", "noise_scale": 0.05},
        {"fraction": 0.12, "selection": "boundary_smooth_left", "noise_scale": 0.05},
        {"fraction": 0.10, "selection": "edge_apex_right", "noise_scale": 0.16},
        {"fraction": 0.10, "selection": "edge_apex_left", "noise_scale": 0.16},
        {"fraction": 0.08, "selection": "yield_tube", "noise_scale": 0.03},
        {"fraction": 0.18, "selection": "tail", "noise_scale": 0.30},
        {"fraction": 0.15, "selection": "small_gap", "noise_scale": 0.10},
        {"fraction": 0.05, "selection": "loading_paths", "noise_scale": 0.12},
    ]

    eval_splits: dict[str, dict[str, np.ndarray]] = {}
    synthetic_geometry: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for name, points, recipe, seed_offset in [
        ("synthetic_core_val", 180224, generator_v2_recipe, 10),
        ("synthetic_hard_val", 180224, hard_recipe, 11),
    ]:
        strain, branch, material, _bucket = _draw_recipe_points(
            eval_seed_bank,
            total_points=points,
            recipe=recipe,
            seed=args.seed + seed_offset,
        )
        synthetic_geometry[name] = (strain, branch, material)
        base_features = trainer._build_point_features(strain, material, feature_set=feature_set)
        eval_splits[name] = {
            "x": scale(base_features),
            "y": branch.astype(np.int64),
            "strain": strain.reshape(-1, 6).astype(np.float32),
            "material": material.astype(np.float32),
        }

    _, _, strain_val_slice, branch_val_slice, material_val_slice = trainer.collect_blocks(
        args.export,
        call_names=real_val_slice_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 20,
    )
    _, _, strain_val_large, branch_val_large, material_val_large = trainer.collect_blocks(
        args.export,
        call_names=real_val_large_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 21,
    )
    _, _, strain_real_test, branch_real_test, material_real_test = trainer.collect_blocks(
        args.export,
        call_names=real_test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 22,
    )

    def add_real_split(name: str, strain: np.ndarray, branch: np.ndarray, material: np.ndarray) -> None:
        base_features = trainer._build_point_features(strain, material, feature_set=feature_set)
        eval_splits[name] = {
            "x": scale(base_features),
            "y": branch.reshape(-1).astype(np.int64),
            "strain": strain.reshape(-1, 6).astype(np.float32),
            "material": material.astype(np.float32),
        }

    add_real_split("real_val_slice", strain_val_slice, branch_val_slice, material_val_slice)
    add_real_split("real_val_large", strain_val_large, branch_val_large, material_val_large)
    add_real_split("real_test", strain_real_test, branch_real_test, material_real_test)

    principal_large, _ = principal_values_and_vectors_from_strain(strain_val_large.reshape(-1, 6))
    material_large_point = np.repeat(material_val_large.astype(np.float32), 11, axis=0)
    geom_large = compute_branch_geometry_principal(
        principal_large,
        c_bar=material_large_point[:, 0],
        sin_phi=material_large_point[:, 1],
        shear=material_large_point[:, 2],
        bulk=material_large_point[:, 3],
        lame=material_large_point[:, 4],
    )
    strain_norm_large = np.linalg.norm(strain_val_large.reshape(-1, 6), axis=1)
    gap_ratio_large = np.minimum(geom_large.delta12, geom_large.delta23) / (
        np.abs(principal_large[:, 0]) + np.abs(principal_large[:, 1]) + np.abs(principal_large[:, 2]) + 1.0e-12
    )
    hard_mask = (
        (np.abs(geom_large.m_yield) <= 0.05)
        | (np.abs(geom_large.m_smooth_left) <= 0.05)
        | (np.abs(geom_large.m_smooth_right) <= 0.05)
        | (np.abs(geom_large.m_left_apex) <= 0.05)
        | (np.abs(geom_large.m_right_apex) <= 0.05)
        | (gap_ratio_large <= 0.03)
        | (strain_norm_large >= float(np.quantile(strain_norm_large, 0.90)))
    )
    eval_splits["real_val_hard"] = {
        "x": eval_splits["real_val_large"]["x"][hard_mask],
        "y": eval_splits["real_val_large"]["y"][hard_mask],
        "strain": eval_splits["real_val_large"]["strain"][hard_mask],
        "material": material_large_point[hard_mask],
    }

    metrics0, confusions0, harms0, preds0 = _evaluate_splits(
        model,
        eval_splits,
        device=device,
        tau=args.tau_harm,
        batch_size=args.batch_size,
    )
    baseline_selector = _selector_score(metrics0)
    baseline_harmful = float(harms0["real_val_large"]["harmful_fail_rate"])
    selected_principal_check = dispatch_branch_stress_3d(
        eval_splits["real_val_slice"]["strain"][:2048],
        constitutive_update_3d(
            eval_splits["real_val_slice"]["strain"][:2048],
            c_bar=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 0],
            sin_phi=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 1],
            shear=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 2],
            bulk=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 3],
            lame=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 4],
        ).branch_id,
        c_bar=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 0],
        sin_phi=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 1],
        shear=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 2],
        bulk=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 3],
        lame=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 4],
    )
    exact_selected = constitutive_update_3d(
        eval_splits["real_val_slice"]["strain"][:2048],
        c_bar=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 0],
        sin_phi=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 1],
        shear=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 2],
        bulk=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 3],
        lame=np.repeat(material_val_slice.astype(np.float32), 11, axis=0)[:2048, 4],
    )
    phase0_checks = {
        "max_selected_dispatch_stress_error": float(np.max(np.abs(selected_principal_check[0] - exact_selected.stress))),
        "max_selected_dispatch_principal_error": float(np.max(np.abs(selected_principal_check[1] - exact_selected.stress_principal))),
    }
    harm_repeat = _harm_summary(
        eval_splits["real_val_large"]["strain"],
        eval_splits["real_val_large"]["material"],
        eval_splits["real_val_large"]["y"],
        preds0["real_val_large"],
        tau=args.tau_harm,
    )
    phase0_checks["harm_repeat_matches"] = bool(
        abs(harm_repeat["harmful_fail_rate"] - harms0["real_val_large"]["harmful_fail_rate"]) < 1.0e-12
    )

    branch_freqs = {
        name: {branch: float(np.mean(split["y"] == idx)) for idx, branch in enumerate(BRANCH_NAMES)}
        for name, split in eval_splits.items()
    }
    benchmark_summary = {
        "train_seed_calls": train_seed_calls,
        "eval_seed_calls": eval_seed_calls,
        "real_val_slice_calls": real_val_slice_calls,
        "real_val_large_calls": real_val_large_calls,
        "real_test_calls": real_test_calls,
        "hard_panel_points": int(eval_splits["real_val_hard"]["y"].shape[0]),
        "baseline_selector_score": baseline_selector,
        "baseline_large_val_harmful_fail_rate": baseline_harmful,
        "phase0_checks": phase0_checks,
        "branch_frequencies": branch_freqs,
        "coverage": {
            "synthetic_core_val": summarize_branch_geometry(*synthetic_geometry["synthetic_core_val"]),
            "synthetic_hard_val": summarize_branch_geometry(*synthetic_geometry["synthetic_hard_val"]),
            "real_val_large": summarize_branch_geometry(strain_val_large, branch_val_large, material_val_large),
            "real_val_hard": summarize_branch_geometry(
                eval_splits["real_val_hard"]["strain"],
                eval_splits["real_val_hard"]["y"],
                eval_splits["real_val_hard"]["material"],
            ),
            "real_test": summarize_branch_geometry(strain_real_test, branch_real_test, material_real_test),
        },
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    _plot_branch_frequencies(branch_freqs, args.output_dir / "benchmark_branch_frequencies.png")
    _plot_confusions(
        {
            "real_val_slice": confusions0["real_val_slice"],
            "real_val_large": confusions0["real_val_large"],
            "real_val_hard": confusions0["real_val_hard"],
            "real_test": confusions0["real_test"],
        },
        args.output_dir / "baseline_confusions.png",
    )
    baseline_harm_table = _harm_confusion_table(
        eval_splits["real_val_large"]["strain"],
        eval_splits["real_val_large"]["material"],
        eval_splits["real_val_large"]["y"],
        preds0["real_val_large"],
        tau=args.tau_harm,
    )
    (args.output_dir / "baseline_harm_confusions.json").write_text(json.dumps(baseline_harm_table, indent=2), encoding="utf-8")

    history: list[dict[str, object]] = []
    top_selector_candidates: list[dict[str, object]] = []
    top_phase1_datasets: list[dict[str, object]] = []
    tracks = {
        "best_selector": {"score": baseline_selector, "path": str(args.checkpoint), "metrics": copy.deepcopy(metrics0), "harms": copy.deepcopy(harms0)},
        "best_harm": {"score": -baseline_harmful, "path": str(args.checkpoint), "metrics": copy.deepcopy(metrics0), "harms": copy.deepcopy(harms0)},
        "best_synthetic": {
            "score": (
                float(metrics0["synthetic_core_val"]["accuracy"]),
                float(metrics0["synthetic_hard_val"]["macro_recall"]),
            ),
            "path": str(args.checkpoint),
            "metrics": copy.deepcopy(metrics0),
            "harms": copy.deepcopy(harms0),
        },
    }

    dataset_cache_dir = args.output_dir / "phase1_datasets"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    global_step = 0
    phase1_best_model_state = copy.deepcopy(model.state_dict())

    for dataset_index in range(1, args.phase1_datasets + 1):
        strain_train, branch_train, material_train, bucket_train = _draw_recipe_points(
            train_seed_bank,
            total_points=args.train_points,
            recipe=generator_v2_recipe,
            seed=args.seed + 1000 + dataset_index,
        )
        x_train_base = trainer._build_point_features(strain_train, material_train, feature_set=feature_set)
        x_train = scale(x_train_base)
        y_train = branch_train.astype(np.int64)
        dataset_path = dataset_cache_dir / f"dataset_{dataset_index:03d}.npz"
        np.savez_compressed(
            dataset_path,
            x=x_train.astype(np.float32),
            y=y_train.astype(np.int64),
            bucket=bucket_train.astype(object),
        )

        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        binary_weights = _binary_class_weights(y_train)
        plastic_weights = _plastic_class_weights(y_train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.phase1_lr, weight_decay=args.weight_decay)

        model.train(True)
        train_loss = 0.0
        count = 0
        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _hierarchical_loss(
                model,
                xb,
                yb,
                binary_weights=binary_weights,
                plastic_weights=plastic_weights,
                plastic_loss_weight=plastic_loss_weight,
            )
            loss.backward()
            optimizer.step()
            batch_n = int(yb.shape[0])
            train_loss += float(loss.item()) * batch_n
            count += batch_n
        train_loss /= max(count, 1)
        global_step += 1

        if dataset_index % args.phase1_eval_every != 0:
            continue

        metrics, confusions, harms, preds = _evaluate_splits(
            model,
            eval_splits,
            device=device,
            tau=args.tau_harm,
            batch_size=args.batch_size,
        )
        selector = _selector_score(metrics)
        synthetic_ok = _synthetic_constraints_ok(metrics, metrics0)
        harmful_large = float(harms["real_val_large"]["harmful_fail_rate"])

        if synthetic_ok and selector > float(tracks["best_selector"]["score"]):
            path = checkpoints_dir / "phase1_best_selector.pt"
            _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"track_name": "best_selector", "selector_score": selector})
            tracks["best_selector"] = {"score": selector, "path": str(path), "metrics": copy.deepcopy(metrics), "harms": copy.deepcopy(harms)}
            phase1_best_model_state = copy.deepcopy(model.state_dict())
        if synthetic_ok and (-harmful_large) > float(tracks["best_harm"]["score"]):
            path = checkpoints_dir / "phase1_best_harm.pt"
            _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"track_name": "best_harm", "harmful_fail_rate": harmful_large})
            tracks["best_harm"] = {"score": -harmful_large, "path": str(path), "metrics": copy.deepcopy(metrics), "harms": copy.deepcopy(harms)}
        synth_score = (
            float(metrics["synthetic_core_val"]["accuracy"]),
            float(metrics["synthetic_hard_val"]["macro_recall"]),
        )
        if synth_score > tuple(tracks["best_synthetic"]["score"]):
            path = checkpoints_dir / "phase1_best_synthetic.pt"
            _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"track_name": "best_synthetic"})
            tracks["best_synthetic"] = {"score": synth_score, "path": str(path), "metrics": copy.deepcopy(metrics), "harms": copy.deepcopy(harms)}

        if synthetic_ok:
            cand_path = checkpoints_dir / f"phase1_selector_candidate_{dataset_index:03d}.pt"
            _save_checkpoint(
                cand_path,
                base_ckpt=base_ckpt,
                model=model,
                extra={"track_name": "selector_candidate", "dataset_index": dataset_index, "selector_score": selector},
            )
            top_selector_candidates.append(
                {
                    "path": str(cand_path),
                    "selector_score": selector,
                    "dataset_index": dataset_index,
                    "metrics": copy.deepcopy(metrics),
                    "harms": copy.deepcopy(harms),
                }
            )
            top_selector_candidates = sorted(top_selector_candidates, key=lambda row: float(row["selector_score"]), reverse=True)[:5]
            top_phase1_datasets.append(
                {
                    "dataset_path": str(dataset_path),
                    "selector_score": selector,
                    "dataset_index": dataset_index,
                }
            )
            top_phase1_datasets = sorted(top_phase1_datasets, key=lambda row: float(row["selector_score"]), reverse=True)[
                : args.phase1_top_datasets
            ]

        history_row: dict[str, object] = {
            "phase": "phase1",
            "global_step": global_step,
            "dataset_index": dataset_index,
            "optimizer": "adamw",
            "lr": args.phase1_lr,
            "train_loss": train_loss,
            "selector_score": selector,
            "synthetic_constraints_ok": synthetic_ok,
            "runtime_s": time.time() - start_time,
        }
        for split_name, split_metrics in metrics.items():
            for key, value in split_metrics.items():
                history_row[f"{split_name}_{key}"] = float(value)
        for split_name, split_harm in harms.items():
            for key, value in split_harm.items():
                history_row[f"{split_name}_{key}"] = float(value)
        history.append(history_row)
        print(
            f"[phase1] dataset={dataset_index}/{args.phase1_datasets} "
            f"selector={selector:.4f} broad_macro={metrics['real_val_large']['macro_recall']:.4f} "
            f"hard_smooth={metrics['real_val_hard']['recall_smooth']:.4f} "
            f"harmful={harmful_large:.4f} synth_ok={synthetic_ok}"
        )

    top10_datasets = [Path(row["dataset_path"]) for row in top_phase1_datasets]
    if top10_datasets:
        best_selector_ckpt = torch.load(Path(tracks["best_selector"]["path"]), map_location="cpu", weights_only=False)
        model.load_state_dict(best_selector_ckpt["state_dict"])
    for dataset_path in top10_datasets:
        data = np.load(dataset_path, allow_pickle=True)
        x_train = data["x"].astype(np.float32)
        y_train = data["y"].astype(np.int64)
        x_t = torch.from_numpy(x_train)
        y_t = torch.from_numpy(y_train)
        binary_weights = _binary_class_weights(y_train)
        plastic_weights = _plastic_class_weights(y_train)
        for step_idx in range(1, args.phase1_lbfgs_steps + 1):
            batch_rng = np.random.default_rng(args.seed + 90000 + int(dataset_path.stem.split("_")[-1]) * 100 + step_idx)
            batch_idx = batch_rng.choice(x_train.shape[0], size=min(args.batch_size, x_train.shape[0]), replace=False)
            xb = x_t[batch_idx].to(device)
            yb = y_t[batch_idx].to(device)
            optimizer = torch.optim.LBFGS(model.parameters(), lr=args.phase1_lbfgs_lr, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = _hierarchical_loss(
                    model,
                    xb,
                    yb,
                    binary_weights=binary_weights,
                    plastic_weights=plastic_weights,
                    plastic_loss_weight=plastic_loss_weight,
                )
                loss.backward()
                return loss

            loss = float(optimizer.step(closure).item())
            global_step += 1
            if step_idx % 2 != 0:
                continue
            metrics, confusions, harms, preds = _evaluate_splits(
                model,
                eval_splits,
                device=device,
                tau=args.tau_harm,
                batch_size=args.batch_size,
            )
            selector = _selector_score(metrics)
            synthetic_ok = _synthetic_constraints_ok(metrics, metrics0)
            harmful_large = float(harms["real_val_large"]["harmful_fail_rate"])
            if synthetic_ok and selector > float(tracks["best_selector"]["score"]):
                path = checkpoints_dir / "phase1_best_selector.pt"
                _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"track_name": "best_selector_lbfgs", "selector_score": selector})
                tracks["best_selector"] = {"score": selector, "path": str(path), "metrics": copy.deepcopy(metrics), "harms": copy.deepcopy(harms)}
            if synthetic_ok and (-harmful_large) > float(tracks["best_harm"]["score"]):
                path = checkpoints_dir / "phase1_best_harm.pt"
                _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"track_name": "best_harm_lbfgs", "harmful_fail_rate": harmful_large})
                tracks["best_harm"] = {"score": -harmful_large, "path": str(path), "metrics": copy.deepcopy(metrics), "harms": copy.deepcopy(harms)}
            history_row = {
                "phase": "phase1_lbfgs",
                "global_step": global_step,
                "dataset_index": int(dataset_path.stem.split("_")[-1]),
                "optimizer": "lbfgs",
                "lr": args.phase1_lbfgs_lr,
                "train_loss": loss,
                "selector_score": selector,
                "synthetic_constraints_ok": synthetic_ok,
                "runtime_s": time.time() - start_time,
            }
            for split_name, split_metrics in metrics.items():
                for key, value in split_metrics.items():
                    history_row[f"{split_name}_{key}"] = float(value)
            for split_name, split_harm in harms.items():
                for key, value in split_harm.items():
                    history_row[f"{split_name}_{key}"] = float(value)
            history.append(history_row)
            print(
                f"[phase1-lbfgs] dataset={dataset_path.stem} step={step_idx}/{args.phase1_lbfgs_steps} "
                f"selector={selector:.4f} broad_macro={metrics['real_val_large']['macro_recall']:.4f} "
                f"hard_smooth={metrics['real_val_hard']['recall_smooth']:.4f} harmful={harmful_large:.4f}"
            )

    selector_candidates = sorted(top_selector_candidates, key=lambda row: float(row["selector_score"]), reverse=True)[:3]
    best_full = None
    for rank, candidate in enumerate(selector_candidates, start=1):
        ckpt = torch.load(candidate["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        full_eval = _evaluate_full_real_val(
            trainer,
            model,
            export_path=args.export,
            call_names=splits["real_val"],
            max_elements_per_call=args.max_elements_per_call,
            seed=args.seed + 700 + rank,
            scale_fn=scale,
            feature_set=feature_set,
            append_context=False,
            device=device,
            tau=args.tau_harm,
        )
        candidate["full_real_val"] = full_eval
        if best_full is None or float(full_eval["metrics"]["macro_recall"]) > float(best_full["full_real_val"]["metrics"]["macro_recall"]):
            best_full = candidate
    if best_full is not None:
        best_full_path = checkpoints_dir / "phase1_best_real_val_full.pt"
        ckpt = torch.load(best_full["path"], map_location="cpu", weights_only=False)
        torch.save(ckpt, best_full_path)
        tracks["best_real_val_full"] = {
            "score": float(best_full["full_real_val"]["metrics"]["macro_recall"]),
            "path": str(best_full_path),
            "metrics": best_full["full_real_val"]["metrics"],
            "harms": best_full["full_real_val"]["harm"],
        }
    else:
        tracks["best_real_val_full"] = {"score": None, "path": None, "metrics": None, "harms": None}

    best_selector_metrics = tracks["best_selector"]["metrics"]
    best_selector_harms = tracks["best_selector"]["harms"]
    phase1_success = (
        float(tracks["best_selector"]["score"]) > baseline_selector
        and (
            float(best_selector_metrics["real_val_large"]["macro_recall"]) >= float(metrics0["real_val_large"]["macro_recall"]) + 0.010
            or float(best_selector_metrics["real_val_hard"]["recall_smooth"]) >= float(metrics0["real_val_hard"]["recall_smooth"]) + 0.030
        )
        and float(best_selector_harms["real_val_large"]["harmful_fail_rate"]) <= 0.8 * baseline_harmful
    )

    final_track_name = "best_selector"
    context_result = None
    if phase1_success and args.run_context_phase:
        hard_improved_enough = float(best_selector_metrics["real_val_hard"]["recall_smooth"]) >= float(metrics0["real_val_hard"]["recall_smooth"]) + 0.030
        harm_improved_enough = float(best_selector_harms["real_val_large"]["harmful_fail_rate"]) <= 0.8 * baseline_harmful
        if not (hard_improved_enough and harm_improved_enough):
            strain_calib_ctx, branch_calib_ctx, material_calib_ctx, _bucket_ctx = _draw_recipe_points(
                train_seed_bank,
                total_points=min(args.train_points, 32768),
                recipe=generator_v2_recipe,
                seed=args.seed + 500000,
            )
            x_calib_base = trainer._build_point_features(strain_calib_ctx, material_calib_ctx, feature_set=feature_set)
            x_calib_ctx = _append_summary_context(x_calib_base, strain_calib_ctx, material_calib_ctx)
            x_mean_ctx = x_calib_ctx.mean(axis=0)
            x_std_ctx = np.where(x_calib_ctx.std(axis=0) < 1.0e-6, 1.0, x_calib_ctx.std(axis=0))

            def scale_ctx(x: np.ndarray) -> np.ndarray:
                return ((x - x_mean_ctx) / x_std_ctx).astype(np.float32)

            context_ckpt = _inflate_input_checkpoint(
                torch.load(tracks["best_selector"]["path"], map_location="cpu", weights_only=False),
                trainer,
                new_input_dim=int(x_calib_ctx.shape[1]),
                noise_scale=1.0e-9,
                seed=args.seed + 600000,
                x_mean=x_mean_ctx,
                x_std=x_std_ctx,
            )
            context_model = trainer.HierarchicalBranchNet(
                in_dim=int(context_ckpt["input_dim"]),
                width=int(context_ckpt["width"]),
                depth=int(context_ckpt["depth"]),
            ).to(device)
            context_model.load_state_dict(context_ckpt["state_dict"])

            context_eval_splits: dict[str, dict[str, np.ndarray]] = {}
            for split_name, split in eval_splits.items():
                base_feat = trainer._build_point_features(split["strain"], split["material"], feature_set=feature_set)
                x_ctx = _append_summary_context(base_feat, split["strain"], split["material"])
                context_eval_splits[split_name] = {
                    "x": scale_ctx(x_ctx),
                    "y": split["y"],
                    "strain": split["strain"],
                    "material": split["material"],
                }

            context_baseline_metrics, _context_conf0, context_baseline_harms, _ = _evaluate_splits(
                context_model,
                context_eval_splits,
                device=device,
                tau=args.tau_harm,
                batch_size=args.batch_size,
            )
            context_best = {
                "score": _selector_score(context_baseline_metrics),
                "metrics": copy.deepcopy(context_baseline_metrics),
                "harms": copy.deepcopy(context_baseline_harms),
                "path": None,
            }
            for dataset_index in range(1, args.context_datasets + 1):
                strain_train, branch_train, material_train, _bucket_train = _draw_recipe_points(
                    train_seed_bank,
                    total_points=args.train_points,
                    recipe=generator_v2_recipe,
                    seed=args.seed + 700000 + dataset_index,
                )
                x_train_base = trainer._build_point_features(strain_train, material_train, feature_set=feature_set)
                x_train_ctx = scale_ctx(_append_summary_context(x_train_base, strain_train, material_train))
                y_train = branch_train.astype(np.int64)
                loader = DataLoader(
                    TensorDataset(torch.from_numpy(x_train_ctx), torch.from_numpy(y_train)),
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                binary_weights = _binary_class_weights(y_train)
                plastic_weights = _plastic_class_weights(y_train)
                optimizer = torch.optim.AdamW(context_model.parameters(), lr=args.phase1_lr, weight_decay=args.weight_decay)
                for xb_cpu, yb_cpu in loader:
                    xb = xb_cpu.to(device)
                    yb = yb_cpu.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _hierarchical_loss(
                        context_model,
                        xb,
                        yb,
                        binary_weights=binary_weights,
                        plastic_weights=plastic_weights,
                        plastic_loss_weight=plastic_loss_weight,
                    )
                    loss.backward()
                    optimizer.step()
                if dataset_index % 2 != 0:
                    continue
                ctx_metrics, _ctx_conf, ctx_harms, _ctx_preds = _evaluate_splits(
                    context_model,
                    context_eval_splits,
                    device=device,
                    tau=args.tau_harm,
                    batch_size=args.batch_size,
                )
                selector_ctx = _selector_score(ctx_metrics)
                synth_ok_ctx = _synthetic_constraints_ok(ctx_metrics, best_selector_metrics)
                better_selector = selector_ctx > float(context_best["score"])
                better_harm = float(ctx_harms["real_val_large"]["harmful_fail_rate"]) < float(context_best["harms"]["real_val_large"]["harmful_fail_rate"])
                if synth_ok_ctx and (better_selector or better_harm):
                    ctx_path = checkpoints_dir / "phase2_context_best.pt"
                    _save_checkpoint(
                        ctx_path,
                        base_ckpt=context_ckpt,
                        model=context_model,
                        extra={"track_name": "phase2_context_best", "selector_score": selector_ctx},
                    )
                    context_best = {
                        "score": selector_ctx,
                        "metrics": copy.deepcopy(ctx_metrics),
                        "harms": copy.deepcopy(ctx_harms),
                        "path": str(ctx_path),
                    }
                    print(
                        f"[phase2-context] dataset={dataset_index}/{args.context_datasets} "
                        f"selector={selector_ctx:.4f} harmful={ctx_harms['real_val_large']['harmful_fail_rate']:.4f}"
                    )
            context_helped = (
                float(context_best["score"]) > float(tracks["best_selector"]["score"])
                or float(context_best["harms"]["real_val_large"]["harmful_fail_rate"]) < float(best_selector_harms["real_val_large"]["harmful_fail_rate"])
            )
            context_result = {
                "status": "ran",
                "helped": context_helped,
                "best_selector_score": float(context_best["score"]),
                "best_large_val_harmful_fail_rate": float(context_best["harms"]["real_val_large"]["harmful_fail_rate"]),
                "path": context_best["path"],
            }
            if context_helped and context_best["path"] is not None:
                final_track_name = "context_best"
                tracks["context_best"] = {
                    "score": float(context_best["score"]),
                    "path": str(context_best["path"]),
                    "metrics": copy.deepcopy(context_best["metrics"]),
                    "harms": copy.deepcopy(context_best["harms"]),
                }

    best_path = Path(tracks[final_track_name]["path"])
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    final_metrics, final_confusions, final_harms, final_preds = _evaluate_splits(
        model,
        eval_splits,
        device=device,
        tau=args.tau_harm,
        batch_size=args.batch_size,
    )
    stress_probe = {
        "real_val_large": _stress_dispatch_metrics(
            eval_splits["real_val_large"]["strain"],
            eval_splits["real_val_large"]["material"],
            eval_splits["real_val_large"]["y"],
            final_preds["real_val_large"],
            tau=args.tau_harm,
        ),
        "real_val_hard": _stress_dispatch_metrics(
            eval_splits["real_val_hard"]["strain"],
            eval_splits["real_val_hard"]["material"],
            eval_splits["real_val_hard"]["y"],
            final_preds["real_val_hard"],
            tau=args.tau_harm,
        ),
        "real_test": _stress_dispatch_metrics(
            eval_splits["real_test"]["strain"],
            eval_splits["real_test"]["material"],
            eval_splits["real_test"]["y"],
            final_preds["real_test"],
            tau=args.tau_harm,
        ),
    }
    final_harm_table = _harm_confusion_table(
        eval_splits["real_val_large"]["strain"],
        eval_splits["real_val_large"]["material"],
        eval_splits["real_val_large"]["y"],
        final_preds["real_val_large"],
        tau=args.tau_harm,
    )
    (args.output_dir / "final_harm_confusions.json").write_text(json.dumps(final_harm_table, indent=2), encoding="utf-8")

    if history:
        fieldnames = sorted({key for row in history for key in row.keys()})
        with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
    _plot_history(history, args.output_dir / "training_history.png")
    _plot_confusions(
        {
            "real_val_large": final_confusions["real_val_large"],
            "real_val_hard": final_confusions["real_val_hard"],
            "real_test": final_confusions["real_test"],
            "synthetic_hard_val": final_confusions["synthetic_hard_val"],
        },
        args.output_dir / "final_confusions.png",
    )

    report_lines = [
        "# Cover Layer Branch Predictor Margin Cycle",
        "",
        "## Summary",
        "",
        "- model family: `hierarchical trial_raw_material w1024 d6`",
        f"- base checkpoint: `{args.checkpoint}`",
        "- selector: `0.5 * broad_macro + 0.3 * hard_macro + 0.2 * smooth_hard`",
        f"- baseline selector score: `{baseline_selector:.4f}`",
        f"- final selector score: `{_selector_score(final_metrics):.4f}`",
        f"- phase1 success gate passed: `{phase1_success}`",
        "",
        "## Baseline",
        "",
        f"- current 4-call slice accuracy / macro: `{metrics0['real_val_slice']['accuracy']:.4f}` / `{metrics0['real_val_slice']['macro_recall']:.4f}`",
        f"- large real-val accuracy / macro: `{metrics0['real_val_large']['accuracy']:.4f}` / `{metrics0['real_val_large']['macro_recall']:.4f}`",
        f"- hard-panel accuracy / macro / smooth: `{metrics0['real_val_hard']['accuracy']:.4f}` / `{metrics0['real_val_hard']['macro_recall']:.4f}` / `{metrics0['real_val_hard']['recall_smooth']:.4f}`",
        f"- real test accuracy / macro: `{metrics0['real_test']['accuracy']:.4f}` / `{metrics0['real_test']['macro_recall']:.4f}`",
        f"- large real-val harmful fail rate: `{harms0['real_val_large']['harmful_fail_rate']:.4f}`",
        f"- phase0 dispatch check max stress error: `{phase0_checks['max_selected_dispatch_stress_error']:.4e}`",
        "",
        "## Phase 1",
        "",
        f"- best selector checkpoint: `{tracks['best_selector']['path']}`",
        f"- best selector large real-val accuracy / macro: `{best_selector_metrics['real_val_large']['accuracy']:.4f}` / `{best_selector_metrics['real_val_large']['macro_recall']:.4f}`",
        f"- best selector hard-panel accuracy / macro / smooth: `{best_selector_metrics['real_val_hard']['accuracy']:.4f}` / `{best_selector_metrics['real_val_hard']['macro_recall']:.4f}` / `{best_selector_metrics['real_val_hard']['recall_smooth']:.4f}`",
        f"- best selector real test accuracy / macro: `{best_selector_metrics['real_test']['accuracy']:.4f}` / `{best_selector_metrics['real_test']['macro_recall']:.4f}`",
        f"- best selector large real-val harmful fail rate: `{best_selector_harms['real_val_large']['harmful_fail_rate']:.4f}`",
        f"- best selector large real-val harmful adjacent fail rate: `{best_selector_harms['real_val_large']['harmful_adjacent_fail_rate']:.4f}`",
        "",
    ]
    if tracks["best_real_val_full"]["path"] is not None:
        report_lines.extend(
            [
                "## Full Real-Val Finalists",
                "",
                f"- best full real-val checkpoint: `{tracks['best_real_val_full']['path']}`",
                f"- full real-val macro recall: `{tracks['best_real_val_full']['metrics']['macro_recall']:.4f}`",
                f"- full real-val harmful fail rate: `{tracks['best_real_val_full']['harms']['harmful_fail_rate']:.4f}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Stress Usefulness",
            "",
            f"- broad real-val wrong-branch mean / p95 full-stress error: `{stress_probe['real_val_large']['wrong_branch_mean_rel_full_stress']:.4f}` / `{stress_probe['real_val_large']['wrong_branch_p95_rel_full_stress']:.4f}`",
            f"- broad real-val harmful-fail mean / p95 full-stress error: `{stress_probe['real_val_large']['harmful_fail_mean_rel_full_stress']:.4f}` / `{stress_probe['real_val_large']['harmful_fail_p95_rel_full_stress']:.4f}`",
            f"- hard-panel harmful-fail mean / p95 full-stress error: `{stress_probe['real_val_hard']['harmful_fail_mean_rel_full_stress']:.4f}` / `{stress_probe['real_val_hard']['harmful_fail_p95_rel_full_stress']:.4f}`",
            f"- real test overall mean / p95 full-stress error: `{stress_probe['real_test']['overall_mean_rel_full_stress']:.4f}` / `{stress_probe['real_test']['overall_p95_rel_full_stress']:.4f}`",
            "",
            "## Final Decision",
            "",
        ]
    )
    if phase1_success:
        report_lines.append("- branch-only line remains viable under the new margin-aware selector")
    else:
        report_lines.append("- branch-only line did not clear the planned success gate; next move should pivot")
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- baseline harm-confusion table: `baseline_harm_confusions.json`",
            "- final harm-confusion table: `final_harm_confusions.json`",
            "- benchmark summary: `benchmark_summary.json`",
            "- training history: `history.csv` and `training_history.png`",
            "- final confusions: `final_confusions.png`",
            "",
        ]
    )
    args.report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "base_checkpoint": str(args.checkpoint),
        "phase0_checks": phase0_checks,
        "baseline_metrics": metrics0,
        "baseline_harms": harms0,
        "tracks": tracks,
        "phase1_success": phase1_success,
        "final_metrics": final_metrics,
        "final_harms": final_harms,
        "stress_probe": stress_probe,
        "context_result": context_result,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
