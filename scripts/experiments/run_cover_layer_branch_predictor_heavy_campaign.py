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

from mc_surrogate.principal_branch_generation import (
    fit_principal_hybrid_bank,
    summarize_branch_geometry,
    synthesize_from_principal_hybrid,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")
BRANCH_IDS = {name: idx for idx, name in enumerate(BRANCH_NAMES)}
REPLAY_QUEUE_NAMES = ("smooth_right_fail", "smooth_left_fail", "edge_apex_fail", "tail_fail")


def _load_trainer_module():
    script_path = Path(__file__).with_name("train_cover_layer_strain_branch_predictor_synth_only.py")
    spec = importlib.util.spec_from_file_location("cover_branch_trainer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load trainer module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inflate_tensor(
    old_tensor: torch.Tensor,
    new_shape: torch.Size,
    *,
    generator: torch.Generator,
    noise_scale: float,
) -> torch.Tensor:
    new_tensor = torch.empty(new_shape, dtype=old_tensor.dtype)
    new_tensor.normal_(mean=0.0, std=noise_scale, generator=generator)
    slices = tuple(slice(0, dim) for dim in old_tensor.shape)
    new_tensor[slices] = old_tensor
    return new_tensor


def _inflate_checkpoint(
    ckpt: dict[str, object],
    trainer,
    *,
    new_width: int,
    noise_scale: float,
    seed: int,
) -> dict[str, object]:
    old_width = int(ckpt["width"])
    if new_width < old_width:
        raise ValueError(f"Requested width {new_width} is smaller than checkpoint width {old_width}.")
    if new_width == old_width:
        return copy.deepcopy(ckpt)

    model_type = str(ckpt.get("model_type", "hierarchical"))
    input_dim = int(ckpt["input_dim"])
    depth = int(ckpt["depth"])
    if model_type == "flat":
        new_model = trainer.BranchMLP(in_dim=input_dim, width=new_width, depth=depth)
    elif model_type == "hierarchical":
        new_model = trainer.HierarchicalBranchNet(in_dim=input_dim, width=new_width, depth=depth)
    else:
        raise ValueError(f"Unsupported model_type {model_type!r}.")

    old_state = ckpt["state_dict"]
    new_state: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for name, param in new_model.state_dict().items():
        old_param = old_state[name]
        if tuple(old_param.shape) == tuple(param.shape):
            new_state[name] = old_param.detach().cpu().clone()
        else:
            new_state[name] = _inflate_tensor(
                old_param.detach().cpu(),
                param.shape,
                generator=generator,
                noise_scale=noise_scale,
            )

    inflated = copy.deepcopy(ckpt)
    inflated["width"] = new_width
    inflated["state_dict"] = new_state
    return inflated


def _prediction_change_rate(
    trainer,
    model_old: nn.Module,
    model_new: nn.Module,
    x_np: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 16384,
) -> float:
    pred_old = _predict_numpy(model_old, x_np, device=device, batch_size=batch_size)
    pred_new = _predict_numpy(model_new, x_np, device=device, batch_size=batch_size)
    return float(np.mean(pred_old != pred_new))


def _draw_principal_recipe(
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
        bucket_parts.append(np.full(branch.shape[0], str(item["bucket"]), dtype=object))
        assigned += part_count
    return (
        np.concatenate(strain_parts, axis=0),
        np.concatenate(branch_parts, axis=0),
        np.concatenate(material_parts, axis=0),
        np.concatenate(bucket_parts, axis=0),
    )


def _branch_freq(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)
    total = max(float(counts.sum()), 1.0)
    counts /= total
    return {name: float(counts[idx]) for idx, name in enumerate(BRANCH_NAMES)}


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


def _evaluate_sets(
    model: nn.Module,
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    batch_size: int = 16384,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, float]] = {}
    confusions: dict[str, np.ndarray] = {}
    for name, (x_np, y_np) in eval_sets.items():
        pred = _predict_numpy(model, x_np, device=device, batch_size=batch_size)
        metrics[name] = _metrics_from_np(pred, y_np)
        confusions[name] = _confusion_np(pred, y_np)
    return metrics, confusions


def _evaluate_real_val_only(
    model: nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 16384,
) -> dict[str, float]:
    pred = _predict_numpy(model, x_np, device=device, batch_size=batch_size)
    return _metrics_from_np(pred, y_np)


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


def _weighted_hierarchical_loss(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    binary_weights: torch.Tensor,
    plastic_weights: torch.Tensor,
    plastic_loss_weight: float,
) -> torch.Tensor:
    elastic_logits, plastic_logits = model(xb)
    binary_targets = (yb != 0).long()
    bin_loss = F.cross_entropy(
        elastic_logits,
        binary_targets,
        weight=binary_weights.to(xb.device),
        reduction="none",
    )
    total = bin_loss * sample_weight
    plastic_mask = yb != 0
    if int(plastic_mask.sum().item()) > 0:
        plastic_targets = (yb[plastic_mask] - 1).long()
        plast_loss = F.cross_entropy(
            plastic_logits[plastic_mask],
            plastic_targets,
            weight=plastic_weights.to(xb.device),
            reduction="none",
        )
        total[plastic_mask] = total[plastic_mask] + plastic_loss_weight * plast_loss * sample_weight[plastic_mask]
    denom = torch.clamp(sample_weight.sum(), min=1.0)
    return total.sum() / denom


def _plot_history(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    epoch = [float(row["global_step"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0, 0].plot(epoch, [float(row["train_loss"]) for row in rows], label="train loss")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)

    for key, label in [
        ("synthetic_core_val_accuracy", "core acc"),
        ("synthetic_hard_val_macro_recall", "hard macro"),
        ("real_val_macro_recall", "real val macro"),
        ("real_test_macro_recall", "real test macro"),
    ]:
        ys = [float(row.get(key, float("nan"))) for row in rows]
        axes[0, 1].plot(epoch, ys, label=label)
    axes[0, 1].set_title("Primary Metrics")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    for key, label in [
        ("synthetic_boundary_right_val_macro_recall", "right boundary"),
        ("synthetic_boundary_left_val_macro_recall", "left boundary"),
        ("synthetic_apex_val_macro_recall", "apex"),
    ]:
        ys = [float(row.get(key, float("nan"))) for row in rows]
        axes[1, 0].plot(epoch, ys, label=label)
    axes[1, 0].set_title("Boundary Buckets")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(epoch, [float(row["lr"]) for row in rows], label="lr")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_yscale("log")
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
    width = 0.12
    fig, ax = plt.subplots(figsize=(14, 5))
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


def _save_checkpoint(
    path: Path,
    *,
    base_ckpt: dict[str, object],
    model: nn.Module,
    extra: dict[str, object] | None = None,
) -> None:
    ckpt = copy.deepcopy(base_ckpt)
    ckpt["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def _real_val_score(metrics: dict[str, dict[str, float]]) -> tuple[float, float, float, float, float]:
    rv = metrics["real_val"]
    return (
        float(rv["macro_recall"]),
        float(rv["accuracy"]),
        float(rv["recall_smooth"]),
        0.5 * (float(rv["recall_left_edge"]) + float(rv["recall_right_edge"])),
        float(rv["recall_apex"]),
    )


def _synthetic_score(metrics: dict[str, dict[str, float]]) -> tuple[float, float, float, float]:
    core = metrics["synthetic_core_val"]
    hard = metrics["synthetic_hard_val"]
    return (
        float(core["macro_recall"]),
        float(hard["macro_recall"]),
        float(core["accuracy"]),
        float(hard["accuracy"]),
    )


def _balanced_ok(metrics: dict[str, dict[str, float]], baseline_metrics: dict[str, dict[str, float]]) -> bool:
    core = metrics["synthetic_core_val"]
    hard = metrics["synthetic_hard_val"]
    rv = metrics["real_val"]
    base_core = baseline_metrics["synthetic_core_val"]
    base_hard = baseline_metrics["synthetic_hard_val"]
    if float(base_core["accuracy"]) - float(core["accuracy"]) > 0.003:
        return False
    if float(base_hard["macro_recall"]) - float(hard["macro_recall"]) > 0.005:
        return False
    if float(rv["recall_smooth"]) < 0.70:
        return False
    if float(rv["recall_left_edge"]) < 0.88:
        return False
    if float(rv["recall_right_edge"]) < 0.84:
        return False
    if float(rv["recall_apex"]) < 0.86:
        return False
    return True


def _balanced_score(metrics: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    rv = metrics["real_val"]
    return (
        float(rv["macro_recall"]),
        float(rv["accuracy"]),
        float(rv["recall_smooth"]),
    )


def _phase_gate_success(
    phase_name: str,
    *,
    baseline_metrics: dict[str, dict[str, float]],
    phase_tracks: dict[str, dict[str, object]],
) -> tuple[bool, str]:
    if phase_name == "phase0":
        return True, "inflation preserved predictions on all frozen eval sets"
    if phase_name == "phase1":
        best_real = phase_tracks["best_real_val"]["metrics"]
        best_bal = phase_tracks["best_balanced"]["metrics"]
        baseline = baseline_metrics["real_val"]["macro_recall"]
        ok = False
        reason = "no improvement"
        if best_real is not None and best_real["real_val"]["macro_recall"] > baseline:
            ok = True
            reason = "best_real_val improved over baseline"
        elif best_bal is not None and best_bal["real_val"]["macro_recall"] > baseline:
            ok = True
            reason = "best_balanced improved over baseline"
        return ok, reason
    if phase_name == "phase2":
        best_real = phase_tracks["best_real_val"]["metrics"]
        if best_real is None:
            return False, "no real-val checkpoint recorded"
        base = baseline_metrics["real_val"]
        delta = float(best_real["real_val"]["macro_recall"]) - float(base["macro_recall"])
        smooth_ok = float(best_real["real_val"]["recall_smooth"]) >= float(base["recall_smooth"]) - 1.0e-6
        left_ok = float(best_real["real_val"]["recall_left_edge"]) >= float(base["recall_left_edge"]) - 0.01
        right_ok = float(best_real["real_val"]["recall_right_edge"]) >= float(base["recall_right_edge"]) - 0.01
        hard_ok = _balanced_ok(best_real, baseline_metrics)
        ok = delta >= 0.003 and smooth_ok and left_ok and right_ok and hard_ok
        return ok, f"delta_macro={delta:.4f}, smooth_ok={smooth_ok}, left_ok={left_ok}, right_ok={right_ok}, hard_ok={hard_ok}"
    if phase_name == "phase3":
        best_real = phase_tracks["best_real_val"]["metrics"]
        if best_real is None:
            return False, "no LBFGS checkpoint recorded"
        delta = float(best_real["real_val"]["macro_recall"]) - float(baseline_metrics["real_val"]["macro_recall"])
        ok = delta > 0.0 and _balanced_ok(best_real, baseline_metrics)
        return ok, f"delta_macro={delta:.4f}"
    raise ValueError(f"Unknown phase {phase_name!r}.")


def _phase_report(
    report_path: Path,
    *,
    phase_name: str,
    artifact_dir: Path,
    gate_ok: bool,
    gate_reason: str,
    baseline_metrics: dict[str, dict[str, float]],
    phase_tracks: dict[str, dict[str, object]],
    history_rows: list[dict[str, object]],
    confusion_path: Path | None,
) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    lines = [
        f"# {phase_name.title()} Report",
        "",
        "## Gate",
        "",
        f"- passed: `{gate_ok}`",
        f"- reason: `{gate_reason}`",
        "",
        "## Track Summary",
        "",
    ]
    for track_name in ("best_real_val", "best_balanced", "best_synthetic", "best_internal_step"):
        track = phase_tracks[track_name]
        lines.append(f"### {track_name}")
        lines.append("")
        if track["metrics"] is None:
            lines.append("- no checkpoint saved")
        else:
            metrics = track["metrics"]
            lines.append(f"- checkpoint: `{track['path']}`")
            lines.append(f"- real val accuracy / macro: `{metrics['real_val']['accuracy']:.4f}` / `{metrics['real_val']['macro_recall']:.4f}`")
            lines.append(f"- real test accuracy / macro: `{metrics['real_test']['accuracy']:.4f}` / `{metrics['real_test']['macro_recall']:.4f}`")
            lines.append(f"- core val accuracy / macro: `{metrics['synthetic_core_val']['accuracy']:.4f}` / `{metrics['synthetic_core_val']['macro_recall']:.4f}`")
            lines.append(f"- hard val accuracy / macro: `{metrics['synthetic_hard_val']['accuracy']:.4f}` / `{metrics['synthetic_hard_val']['macro_recall']:.4f}`")
        lines.append("")
    lines.extend(
        [
            "## Baseline",
            "",
            f"- baseline real val accuracy / macro: `{baseline_metrics['real_val']['accuracy']:.4f}` / `{baseline_metrics['real_val']['macro_recall']:.4f}`",
            f"- baseline real test accuracy / macro: `{baseline_metrics['real_test']['accuracy']:.4f}` / `{baseline_metrics['real_test']['macro_recall']:.4f}`",
            f"- history rows: `{len(history_rows)}`",
            "",
        ]
    )
    if confusion_path is not None:
        lines.extend(
            [
                "## Confusions",
                "",
                "![Confusions](../" + str(rel / confusion_path.name) + ")",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _final_report(
    report_path: Path,
    *,
    artifact_dir: Path,
    baseline_metrics: dict[str, dict[str, float]],
    phase_results: dict[str, dict[str, dict[str, object]]],
    final_paths: dict[str, Path | None],
) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    rows = [
        ("baseline", baseline_metrics),
    ]
    for phase_name in ("phase1", "phase2", "phase3"):
        tracks = phase_results.get(phase_name, {})
        for track_name in ("best_real_val", "best_balanced"):
            track = tracks.get(track_name)
            if track and track.get("metrics") is not None:
                rows.append((f"{phase_name}_{track_name}", track["metrics"]))
    lines = [
        "# Cover Layer Branch Predictor Heavy Post-Train Campaign",
        "",
        "## Summary",
        "",
        "- base architecture: `hierarchical w2048 d6`",
        "- features: `trial_raw_material`",
        "- generator: `expert principal hybrid`",
        "- selection policy: multi-track (`best_real_val`, `best_balanced`, `best_synthetic`, `best_internal_step`)",
        "",
        "## Comparison",
        "",
        "| checkpoint | real val macro | real test macro | core val acc | hard val macro |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['real_val']['macro_recall']:.4f} | {metrics['real_test']['macro_recall']:.4f} | "
            f"{metrics['synthetic_core_val']['accuracy']:.4f} | {metrics['synthetic_hard_val']['macro_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Deliverables",
            "",
            f"- final_best_real_val: `{final_paths.get('final_best_real_val')}`",
            f"- final_best_balanced: `{final_paths.get('final_best_balanced')}`",
            f"- final_best_synthetic: `{final_paths.get('final_best_synthetic')}`",
            "",
            "## Recommendation",
            "",
        ]
    )
    best_real = (
        phase_results.get("phase3", {}).get("best_real_val")
        or phase_results.get("phase2", {}).get("best_real_val")
        or phase_results.get("phase1", {}).get("best_real_val")
    )
    best_balanced = (
        phase_results.get("phase3", {}).get("best_balanced")
        or phase_results.get("phase2", {}).get("best_balanced")
        or phase_results.get("phase1", {}).get("best_balanced")
    )
    if best_real and best_real.get("path") is not None:
        lines.append(f"- deployment: `{best_real['path']}`")
    if best_balanced and best_balanced.get("path") is not None:
        lines.append(f"- balanced continuation anchor: `{best_balanced['path']}`")
    synth_track = (
        phase_results.get("phase3", {}).get("best_synthetic")
        or phase_results.get("phase2", {}).get("best_synthetic")
        or phase_results.get("phase1", {}).get("best_synthetic")
    )
    if synth_track and synth_track.get("path") is not None:
        lines.append(f"- synthetic-faithful analysis: `{synth_track['path']}`")
    history_plot = artifact_dir / "campaign_history.png"
    if history_plot.exists():
        lines.extend(
            [
                "",
                "## Curves",
                "",
                "![Campaign history](../" + str(rel / history_plot.name) + ")",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _campaign_summary(
    *,
    args,
    feature_set: str,
    model_type: str,
    inflated_ckpt: dict[str, object],
    inflation_check: dict[str, float],
    phase_results: dict[str, dict[str, dict[str, object]]],
    final_paths: dict[str, Path | None],
) -> dict[str, object]:
    return {
        "base_checkpoint": str(args.checkpoint),
        "feature_set": feature_set,
        "model_type": model_type,
        "width": int(inflated_ckpt["width"]),
        "depth": int(inflated_ckpt["depth"]),
        "train_points": args.train_points,
        "batch_size": args.batch_size,
        "phase1_datasets": args.phase1_datasets,
        "phase2_datasets": args.phase2_datasets,
        "phase3_top_datasets": args.phase3_top_datasets,
        "phase3_lbfgs_steps": args.phase3_lbfgs_steps,
        "inflation_check": inflation_check,
        "phase_results": {
            phase_name: {
                track_name: {"path": track["path"], "score": track["score"], "metrics": track["metrics"]}
                for track_name, track in tracks.items()
            }
            for phase_name, tracks in phase_results.items()
        },
        "final_paths": {key: str(val) if val is not None else None for key, val in final_paths.items()},
    }


def _bucket_loss_weight(phase_name: str, bucket: str) -> float:
    if phase_name == "phase1":
        return 1.0
    if bucket == "replay":
        return 2.0
    if bucket == "tail":
        return 1.5
    if bucket in {"boundary_smooth_right", "boundary_smooth_left", "smooth_edge", "edge_apex_right", "edge_apex_left"}:
        return 1.5
    return 1.0


def _family_from_real_val(metrics: dict[str, float]) -> str:
    families = {
        "smooth_right_fail": 0.5 * (float(metrics["recall_smooth"]) + float(metrics["recall_right_edge"])),
        "smooth_left_fail": 0.5 * (float(metrics["recall_smooth"]) + float(metrics["recall_left_edge"])),
        "edge_apex_fail": (float(metrics["recall_left_edge"]) + float(metrics["recall_right_edge"]) + float(metrics["recall_apex"])) / 3.0,
    }
    return min(families, key=families.get)


def _weakest_boundary_bucket(metrics_by_name: dict[str, dict[str, float]]) -> str:
    options = {
        "smooth_right_fail": float(metrics_by_name["synthetic_boundary_right_val"]["macro_recall"]),
        "smooth_left_fail": float(metrics_by_name["synthetic_boundary_left_val"]["macro_recall"]),
        "edge_apex_fail": float(metrics_by_name["synthetic_apex_val"]["macro_recall"]),
    }
    return min(options, key=options.get)


def _queue_append(
    queue_bank: dict[str, list[np.ndarray]],
    queue_name: str,
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    *,
    max_points: int,
) -> None:
    queue_bank[f"{queue_name}_x"].append(x_chunk.astype(np.float32, copy=False))
    queue_bank[f"{queue_name}_y"].append(y_chunk.astype(np.int64, copy=False))
    total = int(sum(chunk.shape[0] for chunk in queue_bank[f"{queue_name}_y"]))
    while total > max_points and queue_bank[f"{queue_name}_y"]:
        drop_x = queue_bank[f"{queue_name}_x"].pop(0)
        drop_y = queue_bank[f"{queue_name}_y"].pop(0)
        total -= int(drop_y.shape[0])


def _queue_sample(
    x_chunks: list[np.ndarray],
    y_chunks: list[np.ndarray],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if count <= 0 or not x_chunks:
        return None, None
    x_full = np.concatenate(x_chunks, axis=0)
    y_full = np.concatenate(y_chunks, axis=0)
    if x_full.shape[0] == 0:
        return None, None
    replace = x_full.shape[0] < count
    idx = rng.choice(x_full.shape[0], size=count, replace=replace)
    return x_full[idx], y_full[idx]


def _sample_replay_mix(
    queue_bank: dict[str, list[np.ndarray]],
    *,
    total_count: int,
    seed: int,
    weakest_family: str,
    weakest_boundary: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if total_count <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    counts = {
        weakest_family: int(round(0.40 * total_count)),
        weakest_boundary: int(round(0.30 * total_count)),
        "tail_fail": int(round(0.20 * total_count)),
    }
    counts["generic"] = total_count - sum(counts.values())
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for queue_name, count in counts.items():
        if count <= 0:
            continue
        if queue_name == "generic":
            x_all: list[np.ndarray] = []
            y_all: list[np.ndarray] = []
            for name in REPLAY_QUEUE_NAMES:
                x_all.extend(queue_bank[f"{name}_x"])
                y_all.extend(queue_bank[f"{name}_y"])
            x_sel, y_sel = _queue_sample(x_all, y_all, count=count, rng=rng)
        else:
            x_sel, y_sel = _queue_sample(queue_bank[f"{queue_name}_x"], queue_bank[f"{queue_name}_y"], count=count, rng=rng)
        if x_sel is None:
            continue
        x_parts.append(x_sel)
        y_parts.append(y_sel)
    if not x_parts:
        return None, None
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def _categorize_wrong_points(
    *,
    x_scaled: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bucket_names: np.ndarray,
    strain_raw: np.ndarray,
    queue_bank: dict[str, list[np.ndarray]],
    max_points_per_queue: int,
) -> None:
    wrong_mask = y_true != y_pred
    if not np.any(wrong_mask):
        return
    wrong_x = x_scaled[wrong_mask]
    wrong_y = y_true[wrong_mask]
    wrong_bucket = bucket_names[wrong_mask]
    wrong_strain = strain_raw[wrong_mask]
    strain_norm = np.linalg.norm(wrong_strain, axis=1)
    if strain_norm.size == 0:
        return
    tail_threshold = float(np.quantile(np.linalg.norm(strain_raw, axis=1), 0.80))
    sr_mask = np.isin(wrong_bucket, ["boundary_smooth_right"]) | np.isin(wrong_y, [BRANCH_IDS["smooth"], BRANCH_IDS["right_edge"]])
    sl_mask = np.isin(wrong_bucket, ["boundary_smooth_left"]) | np.isin(wrong_y, [BRANCH_IDS["smooth"], BRANCH_IDS["left_edge"]])
    ea_mask = np.isin(wrong_bucket, ["edge_apex_right", "edge_apex_left"]) | np.isin(wrong_y, [BRANCH_IDS["left_edge"], BRANCH_IDS["right_edge"], BRANCH_IDS["apex"]])
    tail_mask = np.isin(wrong_bucket, ["tail"]) | (strain_norm >= tail_threshold)

    for name, mask in [
        ("smooth_right_fail", sr_mask),
        ("smooth_left_fail", sl_mask),
        ("edge_apex_fail", ea_mask),
        ("tail_fail", tail_mask),
    ]:
        if not np.any(mask):
            continue
        _queue_append(queue_bank, name, wrong_x[mask], wrong_y[mask], max_points=max_points_per_queue)


def _queue_sizes(queue_bank: dict[str, list[np.ndarray]]) -> dict[str, int]:
    return {name: int(sum(chunk.shape[0] for chunk in queue_bank[f"{name}_y"])) for name in REPLAY_QUEUE_NAMES}


def _build_phase_tracks() -> dict[str, dict[str, object]]:
    return {
        "best_real_val": {"score": None, "path": None, "metrics": None},
        "best_balanced": {"score": None, "path": None, "metrics": None},
        "best_synthetic": {"score": None, "path": None, "metrics": None},
        "best_internal_step": {"score": None, "path": None, "metrics": None},
    }


def _maybe_update_tracks(
    *,
    phase_tracks: dict[str, dict[str, object]],
    metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
    checkpoints_dir: Path,
    phase_name: str,
    step_label: str,
    model: nn.Module,
    base_ckpt: dict[str, object],
) -> list[str]:
    saved: list[str] = []

    real_score = _real_val_score(metrics)
    if phase_tracks["best_real_val"]["score"] is None or real_score > phase_tracks["best_real_val"]["score"]:
        path = checkpoints_dir / f"{phase_name}_best_real_val.pt"
        _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"phase_name": phase_name, "step_label": step_label, "track_name": "best_real_val"})
        phase_tracks["best_real_val"] = {"score": real_score, "path": str(path), "metrics": copy.deepcopy(metrics)}
        saved.append("best_real_val")

    synth_score = _synthetic_score(metrics)
    if phase_tracks["best_synthetic"]["score"] is None or synth_score > phase_tracks["best_synthetic"]["score"]:
        path = checkpoints_dir / f"{phase_name}_best_synthetic.pt"
        _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"phase_name": phase_name, "step_label": step_label, "track_name": "best_synthetic"})
        phase_tracks["best_synthetic"] = {"score": synth_score, "path": str(path), "metrics": copy.deepcopy(metrics)}
        saved.append("best_synthetic")

    if _balanced_ok(metrics, baseline_metrics):
        bal_score = _balanced_score(metrics)
        if phase_tracks["best_balanced"]["score"] is None or bal_score > phase_tracks["best_balanced"]["score"]:
            path = checkpoints_dir / f"{phase_name}_best_balanced.pt"
            _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"phase_name": phase_name, "step_label": step_label, "track_name": "best_balanced"})
            phase_tracks["best_balanced"] = {"score": bal_score, "path": str(path), "metrics": copy.deepcopy(metrics)}
            saved.append("best_balanced")

    if phase_tracks["best_internal_step"]["score"] is None or real_score > phase_tracks["best_internal_step"]["score"]:
        path = checkpoints_dir / f"{phase_name}_best_internal_step.pt"
        _save_checkpoint(path, base_ckpt=base_ckpt, model=model, extra={"phase_name": phase_name, "step_label": step_label, "track_name": "best_internal_step"})
        phase_tracks["best_internal_step"] = {"score": real_score, "path": str(path), "metrics": copy.deepcopy(metrics)}
        saved.append("best_internal_step")
    return saved


def _history_row(
    *,
    phase_name: str,
    global_step: int,
    dataset_index: int,
    step_in_dataset: int,
    optimizer_name: str,
    learning_rate: float,
    train_loss: float,
    recipe_fractions: dict[str, float],
    metrics: dict[str, dict[str, float]],
    replay_sizes: dict[str, int],
    saved_tracks: list[str],
    runtime_s: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "phase": phase_name,
        "global_step": global_step,
        "dataset_index": dataset_index,
        "step_in_dataset": step_in_dataset,
        "optimizer": optimizer_name,
        "lr": learning_rate,
        "train_loss": train_loss,
        "recipe_fractions": json.dumps(recipe_fractions, sort_keys=True),
        "replay_sizes": json.dumps(replay_sizes, sort_keys=True),
        "accepted_tracks": ",".join(saved_tracks),
        "runtime_s": runtime_s,
    }
    for split_name, split_metrics in metrics.items():
        for key, value in split_metrics.items():
            row[f"{split_name}_{key}"] = float(value)
    return row


def _save_phase_dataset(path: Path, *, x: np.ndarray, y: np.ndarray, weight: np.ndarray, bucket_code: np.ndarray) -> None:
    np.savez_compressed(path, x=x.astype(np.float32), y=y.astype(np.int64), weight=weight.astype(np.float32), bucket=bucket_code.astype(np.int16))


def _load_phase_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64), data["weight"].astype(np.float32)


def _bucket_code(bucket_names: np.ndarray) -> np.ndarray:
    mapping = {
        "branch_balanced": 0,
        "boundary_smooth_right": 1,
        "boundary_smooth_left": 2,
        "smooth_edge": 3,
        "edge_apex_right": 4,
        "edge_apex_left": 5,
        "tail": 6,
        "replay": 7,
    }
    return np.asarray([mapping.get(str(name), 99) for name in bucket_names], dtype=np.int16)


def main() -> None:
    trainer = _load_trainer_module()

    parser = argparse.ArgumentParser(description="Inspectable heavy post-train campaign for the cover-layer branch predictor.")
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
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_heavy_campaign_20260316"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_heavy_campaign.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--train-seed-calls", type=int, default=24)
    parser.add_argument("--eval-seed-calls", type=int, default=8)
    parser.add_argument("--inflate-width", type=int, default=2048)
    parser.add_argument("--inflate-noise-scale", type=float, default=1.0e-9)
    parser.add_argument("--phase1-datasets", type=int, default=20)
    parser.add_argument("--phase2-datasets", type=int, default=100)
    parser.add_argument("--phase3-top-datasets", type=int, default=12)
    parser.add_argument("--phase3-lbfgs-steps", type=int, default=10)
    parser.add_argument("--phase1-lr", type=float, default=1.0e-6)
    parser.add_argument("--phase2-lr", type=float, default=5.0e-7)
    parser.add_argument("--phase3-lr", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--train-points", type=int, default=81920)
    parser.add_argument("--replay-cap-active", type=int, default=16384)
    parser.add_argument("--replay-cap-queue", type=int, default=131072)
    parser.add_argument("--eval-every-phase1", type=int, default=2)
    parser.add_argument("--eval-every-phase2", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    phase2_dataset_dir = args.output_dir / "phase2_datasets"
    phase_reports_dir = args.output_dir / "phase_reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    phase2_dataset_dir.mkdir(parents=True, exist_ok=True)
    phase_reports_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    inflated_ckpt = _inflate_checkpoint(
        base_ckpt,
        trainer,
        new_width=args.inflate_width,
        noise_scale=args.inflate_noise_scale,
        seed=args.seed + 991,
    )

    feature_set = str(inflated_ckpt["feature_set"])
    model_type = str(inflated_ckpt.get("model_type", "hierarchical"))
    plastic_loss_weight = float(inflated_ckpt.get("plastic_loss_weight", 1.0))
    x_mean = np.asarray(inflated_ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(inflated_ckpt["x_std"], dtype=np.float32)
    x_std = np.where(np.abs(x_std) < 1.0e-6, 1.0, x_std)

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    if model_type != "hierarchical":
        raise RuntimeError(f"This campaign expects the hierarchical model family, got {model_type!r}.")

    model = trainer.HierarchicalBranchNet(in_dim=int(inflated_ckpt["input_dim"]), width=int(inflated_ckpt["width"]), depth=int(inflated_ckpt["depth"])).to(device)
    model.load_state_dict(inflated_ckpt["state_dict"])
    inflated_init_path = checkpoints_dir / "inflated_init.pt"
    _save_checkpoint(inflated_init_path, base_ckpt=inflated_ckpt, model=model, extra={"phase_name": "phase0", "track_name": "inflated_init"})

    base_model = trainer.HierarchicalBranchNet(in_dim=int(base_ckpt["input_dim"]), width=int(base_ckpt["width"]), depth=int(base_ckpt["depth"])).to(device)
    base_model.load_state_dict(base_ckpt["state_dict"])

    splits = trainer.load_split_calls(args.split_json)
    regimes = trainer.load_call_regimes(args.regime_json)
    train_seed_calls, eval_seed_calls = trainer._split_seed_calls(
        splits["generator_fit"],
        regimes=regimes,
        train_count=args.train_seed_calls,
        eval_count=args.eval_seed_calls,
    )
    real_val_calls = trainer._spread_pick_exact(splits["real_val"], count=4, regimes=regimes)
    real_test_calls = trainer._spread_pick_exact(splits["real_test"], count=4, regimes=regimes)

    def build_seed_bank(call_names: list[str], seed: int) -> dict[str, np.ndarray]:
        _coords, _disp, strain, branch, material = trainer.collect_blocks(
            args.export,
            call_names=call_names,
            max_elements_per_call=args.max_elements_per_call,
            seed=seed,
        )
        return fit_principal_hybrid_bank(strain, branch, material)

    train_seed_bank = build_seed_bank(train_seed_calls, args.seed + 1)
    eval_seed_bank = build_seed_bank(eval_seed_calls, args.seed + 2)

    core_recipe = [
        {"fraction": 0.60, "selection": "branch_balanced", "noise_scale": 0.18, "bucket": "branch_balanced"},
        {"fraction": 0.25, "selection": "boundary_smooth_right", "noise_scale": 0.05, "bucket": "boundary_smooth_right"},
        {"fraction": 0.15, "selection": "tail", "noise_scale": 0.25, "bucket": "tail"},
    ]
    hard_recipe = [
        {"fraction": 0.30, "selection": "branch_balanced", "noise_scale": 0.22, "bucket": "branch_balanced"},
        {"fraction": 0.20, "selection": "boundary_smooth_right", "noise_scale": 0.05, "bucket": "boundary_smooth_right"},
        {"fraction": 0.20, "selection": "boundary_smooth_left", "noise_scale": 0.05, "bucket": "boundary_smooth_left"},
        {"fraction": 0.15, "selection": "edge_apex_right", "noise_scale": 0.16, "bucket": "edge_apex_right"},
        {"fraction": 0.10, "selection": "edge_apex_left", "noise_scale": 0.16, "bucket": "edge_apex_left"},
        {"fraction": 0.05, "selection": "tail", "noise_scale": 0.28, "bucket": "tail"},
    ]
    boundary_right_recipe = [{"fraction": 1.0, "selection": "boundary_smooth_right", "noise_scale": 0.05, "bucket": "boundary_smooth_right"}]
    boundary_left_recipe = [{"fraction": 1.0, "selection": "boundary_smooth_left", "noise_scale": 0.05, "bucket": "boundary_smooth_left"}]
    apex_recipe = [
        {"fraction": 0.50, "selection": "edge_apex_right", "noise_scale": 0.16, "bucket": "edge_apex_right"},
        {"fraction": 0.50, "selection": "edge_apex_left", "noise_scale": 0.16, "bucket": "edge_apex_left"},
    ]

    synthetic_eval_geometry: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    eval_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, point_count, recipe, seed_offset in [
        ("synthetic_core_val", 180224, core_recipe, 10),
        ("synthetic_hard_val", 180224, hard_recipe, 11),
        ("synthetic_boundary_right_val", 90112, boundary_right_recipe, 12),
        ("synthetic_boundary_left_val", 90112, boundary_left_recipe, 13),
        ("synthetic_apex_val", 90112, apex_recipe, 14),
    ]:
        strain, branch, material, _bucket = _draw_principal_recipe(eval_seed_bank, total_points=point_count, recipe=recipe, seed=args.seed + seed_offset)
        synthetic_eval_geometry[name] = (strain, branch, material)
        x_np = trainer._build_point_features(strain, material, feature_set=feature_set)
        eval_raw[name] = (x_np.astype(np.float32), branch.astype(np.int64), material.astype(np.float32))

    _, _, strain_real_val, branch_real_val, material_real_val = trainer.collect_blocks(
        args.export,
        call_names=real_val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 20,
    )
    _, _, strain_real_test, branch_real_test, material_real_test = trainer.collect_blocks(
        args.export,
        call_names=real_test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 21,
    )
    x_real_val_np = trainer._build_point_features(strain_real_val, material_real_val, feature_set=feature_set)
    x_real_test_np = trainer._build_point_features(strain_real_test, material_real_test, feature_set=feature_set)
    eval_raw["real_val"] = (x_real_val_np.astype(np.float32), branch_real_val.reshape(-1).astype(np.int64), material_real_val.astype(np.float32))
    eval_raw["real_test"] = (x_real_test_np.astype(np.float32), branch_real_test.reshape(-1).astype(np.int64), material_real_test.astype(np.float32))

    eval_sets = {name: (scale(x_np), y_np) for name, (x_np, y_np, _mat) in eval_raw.items()}

    inflation_check = {
        name: _prediction_change_rate(trainer, base_model, model, eval_sets[name][0], device=device)
        for name in eval_sets
    }
    baseline_metrics, baseline_confusions = _evaluate_sets(model, eval_sets, device=device)

    benchmark_summary = {
        "train_seed_calls": train_seed_calls,
        "eval_seed_calls": eval_seed_calls,
        "real_val_calls": real_val_calls,
        "real_test_calls": real_test_calls,
        "inflation_check": inflation_check,
        "branch_frequencies": {
            name: _branch_freq(labels)
            for name, (_x, labels) in eval_sets.items()
        },
        "coverage": {},
    }
    for name, (strain, branch, material) in synthetic_eval_geometry.items():
        benchmark_summary["coverage"][name] = summarize_branch_geometry(strain, branch, material)
    benchmark_summary["coverage"]["real_val"] = summarize_branch_geometry(strain_real_val, branch_real_val, material_real_val)
    benchmark_summary["coverage"]["real_test"] = summarize_branch_geometry(strain_real_test, branch_real_test, material_real_test)
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    _plot_branch_frequencies(benchmark_summary["branch_frequencies"], args.output_dir / "benchmark_branch_frequencies.png")
    _plot_confusions(
        {
            "real_val": baseline_confusions["real_val"],
            "synthetic_boundary_right_val": baseline_confusions["synthetic_boundary_right_val"],
            "synthetic_boundary_left_val": baseline_confusions["synthetic_boundary_left_val"],
            "synthetic_apex_val": baseline_confusions["synthetic_apex_val"],
        },
        args.output_dir / "baseline_confusions.png",
    )

    phase_results: dict[str, dict[str, dict[str, object]]] = {}
    all_history: list[dict[str, object]] = []
    global_step = 0
    start_time = time.time()

    phase0_tracks = _build_phase_tracks()
    phase0_metrics = copy.deepcopy(baseline_metrics)
    phase0_ok = all(value == 0.0 for value in inflation_check.values())
    phase0_reason = "inflation preserved predictions on all frozen eval sets" if phase0_ok else f"prediction drift detected: {inflation_check}"
    for track_name in phase0_tracks:
        path = checkpoints_dir / f"phase0_{track_name}.pt"
        _save_checkpoint(
            path,
            base_ckpt=inflated_ckpt,
            model=model,
            extra={"phase_name": "phase0", "track_name": track_name, "step_label": "baseline"},
        )
        phase0_tracks[track_name] = {"score": "baseline", "path": str(path), "metrics": copy.deepcopy(phase0_metrics)}
    phase_results["phase0"] = phase0_tracks
    _phase_report(
        phase_reports_dir / "phase0_report.md",
        phase_name="phase0",
        artifact_dir=args.output_dir,
        gate_ok=phase0_ok,
        gate_reason=phase0_reason,
        baseline_metrics=baseline_metrics,
        phase_tracks=phase0_tracks,
        history_rows=[],
        confusion_path=args.output_dir / "baseline_confusions.png",
    )
    if not phase0_ok:
        raise RuntimeError(f"Phase 0 gate failed: {phase0_reason}")

    queue_bank = {f"{name}_x": [] for name in REPLAY_QUEUE_NAMES} | {f"{name}_y": [] for name in REPLAY_QUEUE_NAMES}

    def draw_training_dataset(
        phase_name: str,
        dataset_seed: int,
        weakest_family: str,
        weakest_boundary: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replay_target = int(round(args.train_points * 0.05))
        replay_x, replay_y = _sample_replay_mix(
            queue_bank,
            total_count=min(args.replay_cap_active, replay_target),
            seed=dataset_seed + 900000,
            weakest_family=weakest_family,
            weakest_boundary=weakest_boundary,
        )
        replay_count = 0 if replay_x is None else int(replay_x.shape[0])
        fresh_points = args.train_points - replay_count
        fresh_recipe = [
            {"fraction": 30.0 / 95.0, "selection": "branch_balanced", "noise_scale": 0.20, "bucket": "branch_balanced"},
            {"fraction": 20.0 / 95.0, "selection": "boundary_smooth_right", "noise_scale": 0.05, "bucket": "boundary_smooth_right"},
            {"fraction": 15.0 / 95.0, "selection": "boundary_smooth_left", "noise_scale": 0.05, "bucket": "boundary_smooth_left"},
            {"fraction": 10.0 / 95.0, "selection": "smooth_edge", "noise_scale": 0.22, "bucket": "smooth_edge"},
            {"fraction": 10.0 / 95.0, "selection": "edge_apex_right", "noise_scale": 0.16, "bucket": "edge_apex_right"},
            {"fraction": 5.0 / 95.0, "selection": "edge_apex_left", "noise_scale": 0.16, "bucket": "edge_apex_left"},
            {"fraction": 5.0 / 95.0, "selection": "tail", "noise_scale": 0.25, "bucket": "tail"},
        ]
        strain, branch, material, buckets = _draw_principal_recipe(
            train_seed_bank,
            total_points=fresh_points,
            recipe=fresh_recipe,
            seed=dataset_seed,
        )
        x_fresh = scale(trainer._build_point_features(strain, material, feature_set=feature_set))
        y_fresh = branch.astype(np.int64)
        weights = np.asarray([_bucket_loss_weight(phase_name, str(bucket)) for bucket in buckets], dtype=np.float32)
        if replay_x is not None and replay_y is not None:
            x_all = np.concatenate([x_fresh, replay_x], axis=0)
            y_all = np.concatenate([y_fresh, replay_y], axis=0)
            w_all = np.concatenate([weights, np.full(replay_x.shape[0], _bucket_loss_weight(phase_name, "replay"), dtype=np.float32)], axis=0)
            bucket_all = np.concatenate([buckets, np.full(replay_x.shape[0], "replay", dtype=object)], axis=0)
            strain_all = np.concatenate([strain, np.zeros((replay_x.shape[0], strain.shape[1]), dtype=np.float32)], axis=0)
        else:
            x_all = x_fresh
            y_all = y_fresh
            w_all = weights
            bucket_all = buckets
            strain_all = strain
        return x_all, y_all, w_all, bucket_all, strain_all

    # Phase 1
    current_real_val_metrics = baseline_metrics["real_val"]
    current_boundary_metrics = baseline_metrics
    phase1_tracks = _build_phase_tracks()
    phase1_history: list[dict[str, object]] = []
    binary_weights = None
    plastic_weights = None
    for dataset_index in range(1, args.phase1_datasets + 1):
        weakest_family = _family_from_real_val(current_real_val_metrics)
        weakest_boundary = _weakest_boundary_bucket(current_boundary_metrics)
        x_train, y_train, sample_weights, bucket_names, strain_raw = draw_training_dataset("phase1", args.seed + 1000 + dataset_index, weakest_family, weakest_boundary)
        tensor_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(sample_weights))
        loader = DataLoader(tensor_ds, batch_size=args.batch_size, shuffle=True)
        binary_weights = _binary_class_weights(y_train)
        plastic_weights = _plastic_class_weights(y_train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.phase1_lr, weight_decay=args.weight_decay)

        model.train(True)
        train_loss = 0.0
        count = 0
        for xb_cpu, yb_cpu, wb_cpu in loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)
            wb = wb_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_hierarchical_loss(
                model,
                xb,
                yb,
                wb,
                binary_weights=binary_weights,
                plastic_weights=plastic_weights,
                plastic_loss_weight=plastic_loss_weight,
            )
            loss.backward()
            optimizer.step()
            n = int(yb.shape[0])
            train_loss += float(loss.item()) * n
            count += n
        train_loss /= max(count, 1)
        global_step += 1

        pred_train = _predict_numpy(model, x_train, device=device, batch_size=args.batch_size)
        _categorize_wrong_points(
            x_scaled=x_train,
            y_true=y_train,
            y_pred=pred_train,
            bucket_names=bucket_names,
            strain_raw=strain_raw,
            queue_bank=queue_bank,
            max_points_per_queue=args.replay_cap_queue,
        )

        if dataset_index % args.eval_every_phase1 == 0:
            metrics, confusions = _evaluate_sets(model, eval_sets, device=device)
            saved = _maybe_update_tracks(
                phase_tracks=phase1_tracks,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                checkpoints_dir=checkpoints_dir,
                phase_name="phase1",
                step_label=f"dataset_{dataset_index}",
                model=model,
                base_ckpt=inflated_ckpt,
            )
            current_real_val_metrics = metrics["real_val"]
            current_boundary_metrics = metrics
            row = _history_row(
                phase_name="phase1",
                global_step=global_step,
                dataset_index=dataset_index,
                step_in_dataset=1,
                optimizer_name="adamw",
                learning_rate=args.phase1_lr,
                train_loss=train_loss,
                recipe_fractions={"fresh": 0.95, "replay": 0.05},
                metrics=metrics,
                replay_sizes=_queue_sizes(queue_bank),
                saved_tracks=saved,
                runtime_s=time.time() - start_time,
            )
            phase1_history.append(row)
            all_history.append(row)
            print(
                f"[phase1] dataset={dataset_index}/{args.phase1_datasets} loss={train_loss:.6f} "
                f"real_val_macro={metrics['real_val']['macro_recall']:.4f} "
                f"real_test_macro={metrics['real_test']['macro_recall']:.4f} "
                f"saved={saved}"
            )
    phase1_last = checkpoints_dir / "phase1_last.pt"
    _save_checkpoint(phase1_last, base_ckpt=inflated_ckpt, model=model, extra={"phase_name": "phase1", "track_name": "last"})
    phase1_ok, phase1_reason = _phase_gate_success("phase1", baseline_metrics=baseline_metrics, phase_tracks=phase1_tracks)
    phase_results["phase1"] = phase1_tracks
    phase1_confusions = _evaluate_sets(model, eval_sets, device=device)[1]
    _plot_confusions(
        {
            "real_val": phase1_confusions["real_val"],
            "synthetic_boundary_right_val": phase1_confusions["synthetic_boundary_right_val"],
            "synthetic_boundary_left_val": phase1_confusions["synthetic_boundary_left_val"],
            "synthetic_apex_val": phase1_confusions["synthetic_apex_val"],
        },
        args.output_dir / "phase1_confusions.png",
    )
    _phase_report(
        phase_reports_dir / "phase1_report.md",
        phase_name="phase1",
        artifact_dir=args.output_dir,
        gate_ok=phase1_ok,
        gate_reason=phase1_reason,
        baseline_metrics=baseline_metrics,
        phase_tracks=phase1_tracks,
        history_rows=phase1_history,
        confusion_path=args.output_dir / "phase1_confusions.png",
    )
    if not phase1_ok:
        _plot_history(all_history, args.output_dir / "campaign_history.png")
        final_paths = {
            "final_best_real_val": Path(phase1_tracks["best_real_val"]["path"]) if phase1_tracks["best_real_val"]["path"] else None,
            "final_best_balanced": Path(phase1_tracks["best_balanced"]["path"]) if phase1_tracks["best_balanced"]["path"] else None,
            "final_best_synthetic": Path(phase1_tracks["best_synthetic"]["path"]) if phase1_tracks["best_synthetic"]["path"] else None,
        }
        _final_report(args.report_path, artifact_dir=args.output_dir, baseline_metrics=baseline_metrics, phase_results=phase_results, final_paths=final_paths)
        summary = _campaign_summary(
            args=args,
            feature_set=feature_set,
            model_type=model_type,
            inflated_ckpt=inflated_ckpt,
            inflation_check=inflation_check,
            phase_results=phase_results,
            final_paths=final_paths,
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    if phase1_tracks["best_balanced"]["path"]:
        model.load_state_dict(torch.load(phase1_tracks["best_balanced"]["path"], map_location="cpu", weights_only=False)["state_dict"])
    elif phase1_tracks["best_real_val"]["path"]:
        model.load_state_dict(torch.load(phase1_tracks["best_real_val"]["path"], map_location="cpu", weights_only=False)["state_dict"])

    # Phase 2
    phase2_tracks = _build_phase_tracks()
    phase2_history: list[dict[str, object]] = []
    top_phase2_datasets: list[dict[str, object]] = []
    current_real_val_metrics = phase1_tracks["best_real_val"]["metrics"]["real_val"] if phase1_tracks["best_real_val"]["metrics"] else baseline_metrics["real_val"]
    current_boundary_metrics = phase1_tracks["best_real_val"]["metrics"] if phase1_tracks["best_real_val"]["metrics"] else baseline_metrics
    for dataset_index in range(1, args.phase2_datasets + 1):
        if dataset_index % 10 == 1 and phase2_tracks["best_real_val"]["metrics"] is not None:
            current_real_val_metrics = phase2_tracks["best_real_val"]["metrics"]["real_val"]
            current_boundary_metrics = phase2_tracks["best_real_val"]["metrics"]
        weakest_family = _family_from_real_val(current_real_val_metrics)
        weakest_boundary = _weakest_boundary_bucket(current_boundary_metrics)
        x_train, y_train, sample_weights, bucket_names, strain_raw = draw_training_dataset("phase2", args.seed + 10000 + dataset_index, weakest_family, weakest_boundary)
        tensor_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(sample_weights))
        loader = DataLoader(tensor_ds, batch_size=args.batch_size, shuffle=True)
        binary_weights = _binary_class_weights(y_train)
        plastic_weights = _plastic_class_weights(y_train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=args.weight_decay)

        model.train(True)
        train_loss = 0.0
        count = 0
        for xb_cpu, yb_cpu, wb_cpu in loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)
            wb = wb_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_hierarchical_loss(
                model,
                xb,
                yb,
                wb,
                binary_weights=binary_weights,
                plastic_weights=plastic_weights,
                plastic_loss_weight=plastic_loss_weight,
            )
            loss.backward()
            optimizer.step()
            n = int(yb.shape[0])
            train_loss += float(loss.item()) * n
            count += n
        train_loss /= max(count, 1)
        global_step += 1

        real_val_only = _evaluate_real_val_only(model, eval_sets["real_val"][0], eval_sets["real_val"][1], device=device)
        dataset_score = (
            float(real_val_only["macro_recall"]),
            float(real_val_only["accuracy"]),
            float(real_val_only["recall_smooth"]),
        )
        dataset_path = phase2_dataset_dir / f"phase2_dataset_{dataset_index:03d}.npz"
        _save_phase_dataset(dataset_path, x=x_train, y=y_train, weight=sample_weights, bucket_code=_bucket_code(bucket_names))
        top_phase2_datasets.append({"path": str(dataset_path), "score": dataset_score, "dataset_index": dataset_index, "metrics": real_val_only})
        top_phase2_datasets = sorted(top_phase2_datasets, key=lambda item: item["score"], reverse=True)[: args.phase3_top_datasets]
        top_paths = {item["path"] for item in top_phase2_datasets}
        for candidate in list(phase2_dataset_dir.glob("phase2_dataset_*.npz")):
            if str(candidate) not in top_paths and int(candidate.stem.split("_")[-1]) < dataset_index - 2:
                candidate.unlink(missing_ok=True)

        pred_train = _predict_numpy(model, x_train, device=device, batch_size=args.batch_size)
        _categorize_wrong_points(
            x_scaled=x_train,
            y_true=y_train,
            y_pred=pred_train,
            bucket_names=bucket_names,
            strain_raw=strain_raw,
            queue_bank=queue_bank,
            max_points_per_queue=args.replay_cap_queue,
        )

        if dataset_index % args.eval_every_phase2 == 0:
            metrics, confusions = _evaluate_sets(model, eval_sets, device=device)
            saved = _maybe_update_tracks(
                phase_tracks=phase2_tracks,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                checkpoints_dir=checkpoints_dir,
                phase_name="phase2",
                step_label=f"dataset_{dataset_index}",
                model=model,
                base_ckpt=inflated_ckpt,
            )
            current_real_val_metrics = metrics["real_val"]
            current_boundary_metrics = metrics
            row = _history_row(
                phase_name="phase2",
                global_step=global_step,
                dataset_index=dataset_index,
                step_in_dataset=1,
                optimizer_name="adamw",
                learning_rate=args.phase2_lr,
                train_loss=train_loss,
                recipe_fractions={"fresh": 0.95, "replay": 0.05},
                metrics=metrics,
                replay_sizes=_queue_sizes(queue_bank),
                saved_tracks=saved,
                runtime_s=time.time() - start_time,
            )
            phase2_history.append(row)
            all_history.append(row)
            print(
                f"[phase2] dataset={dataset_index}/{args.phase2_datasets} loss={train_loss:.6f} "
                f"real_val_macro={metrics['real_val']['macro_recall']:.4f} "
                f"real_test_macro={metrics['real_test']['macro_recall']:.4f} "
                f"saved={saved}"
            )
    phase2_ok, phase2_reason = _phase_gate_success("phase2", baseline_metrics=baseline_metrics, phase_tracks=phase2_tracks)
    phase_results["phase2"] = phase2_tracks
    phase2_confusions = _evaluate_sets(model, eval_sets, device=device)[1]
    _plot_confusions(
        {
            "real_val": phase2_confusions["real_val"],
            "synthetic_boundary_right_val": phase2_confusions["synthetic_boundary_right_val"],
            "synthetic_boundary_left_val": phase2_confusions["synthetic_boundary_left_val"],
            "synthetic_apex_val": phase2_confusions["synthetic_apex_val"],
        },
        args.output_dir / "phase2_confusions.png",
    )
    _phase_report(
        phase_reports_dir / "phase2_report.md",
        phase_name="phase2",
        artifact_dir=args.output_dir,
        gate_ok=phase2_ok,
        gate_reason=phase2_reason,
        baseline_metrics=baseline_metrics,
        phase_tracks=phase2_tracks,
        history_rows=phase2_history,
        confusion_path=args.output_dir / "phase2_confusions.png",
    )
    if not phase2_ok:
        _plot_history(all_history, args.output_dir / "campaign_history.png")
        final_paths = {
            "final_best_real_val": Path(phase2_tracks["best_real_val"]["path"]) if phase2_tracks["best_real_val"]["path"] else None,
            "final_best_balanced": Path(phase2_tracks["best_balanced"]["path"]) if phase2_tracks["best_balanced"]["path"] else None,
            "final_best_synthetic": Path(phase2_tracks["best_synthetic"]["path"]) if phase2_tracks["best_synthetic"]["path"] else None,
        }
        _final_report(args.report_path, artifact_dir=args.output_dir, baseline_metrics=baseline_metrics, phase_results=phase_results, final_paths=final_paths)
        summary = _campaign_summary(
            args=args,
            feature_set=feature_set,
            model_type=model_type,
            inflated_ckpt=inflated_ckpt,
            inflation_check=inflation_check,
            phase_results=phase_results,
            final_paths=final_paths,
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    phase2_best_balanced_path = phase2_tracks["best_balanced"]["path"] or phase2_tracks["best_real_val"]["path"]
    if phase2_best_balanced_path is None:
        raise RuntimeError("Phase 2 finished without a usable checkpoint.")
    model.load_state_dict(torch.load(phase2_best_balanced_path, map_location="cpu", weights_only=False)["state_dict"])

    # Phase 3
    phase3_tracks = _build_phase_tracks()
    phase3_history: list[dict[str, object]] = []
    selected_phase2 = sorted(top_phase2_datasets, key=lambda item: item["score"], reverse=True)[: args.phase3_top_datasets]
    for dataset_rank, dataset_info in enumerate(selected_phase2, start=1):
        x_train, y_train, sample_weights = _load_phase_dataset(Path(dataset_info["path"]))
        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train)
        w_train_t = torch.from_numpy(sample_weights)
        binary_weights = _binary_class_weights(y_train)
        plastic_weights = _plastic_class_weights(y_train)

        model.train(True)
        for step_idx in range(1, args.phase3_lbfgs_steps + 1):
            batch_rng = np.random.default_rng(args.seed + 500000 + dataset_rank * 100 + step_idx)
            batch_idx = batch_rng.choice(x_train.shape[0], size=min(args.batch_size, x_train.shape[0]), replace=False)
            xb = x_train_t[batch_idx].to(device)
            yb = y_train_t[batch_idx].to(device)
            wb = w_train_t[batch_idx].to(device)
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=args.phase3_lr,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = _weighted_hierarchical_loss(
                    model,
                    xb,
                    yb,
                    wb,
                    binary_weights=binary_weights,
                    plastic_weights=plastic_weights,
                    plastic_loss_weight=plastic_loss_weight,
                )
                loss.backward()
                return loss

            loss = float(optimizer.step(closure).item())
            global_step += 1
            if step_idx % 2 == 0:
                metrics, confusions = _evaluate_sets(model, eval_sets, device=device)
                saved = _maybe_update_tracks(
                    phase_tracks=phase3_tracks,
                    metrics=metrics,
                    baseline_metrics=baseline_metrics,
                    checkpoints_dir=checkpoints_dir,
                    phase_name="phase3",
                    step_label=f"dataset_{dataset_info['dataset_index']}_step_{step_idx}",
                    model=model,
                    base_ckpt=inflated_ckpt,
                )
                if saved:
                    specific_path = checkpoints_dir / f"phase3_dataset_{dataset_info['dataset_index']}_step_{step_idx}.pt"
                    _save_checkpoint(
                        specific_path,
                        base_ckpt=inflated_ckpt,
                        model=model,
                        extra={
                            "phase_name": "phase3",
                            "step_label": f"dataset_{dataset_info['dataset_index']}_step_{step_idx}",
                            "track_name": "event_snapshot",
                        },
                    )
                row = _history_row(
                    phase_name="phase3",
                    global_step=global_step,
                    dataset_index=int(dataset_info["dataset_index"]),
                    step_in_dataset=step_idx,
                    optimizer_name="lbfgs",
                    learning_rate=args.phase3_lr,
                    train_loss=loss,
                    recipe_fractions={"selected_phase2_dataset_rank": float(dataset_rank)},
                    metrics=metrics,
                    replay_sizes=_queue_sizes(queue_bank),
                    saved_tracks=saved,
                    runtime_s=time.time() - start_time,
                )
                phase3_history.append(row)
                all_history.append(row)
                print(
                    f"[phase3] dataset_rank={dataset_rank}/{len(selected_phase2)} dataset={dataset_info['dataset_index']} "
                    f"step={step_idx}/{args.phase3_lbfgs_steps} loss={loss:.6f} "
                    f"real_val_macro={metrics['real_val']['macro_recall']:.4f} "
                    f"real_test_macro={metrics['real_test']['macro_recall']:.4f} "
                    f"saved={saved}"
                )

    phase3_ok, phase3_reason = _phase_gate_success("phase3", baseline_metrics=baseline_metrics, phase_tracks=phase3_tracks)
    phase_results["phase3"] = phase3_tracks
    phase3_confusions = _evaluate_sets(model, eval_sets, device=device)[1]
    _plot_confusions(
        {
            "real_val": phase3_confusions["real_val"],
            "synthetic_boundary_right_val": phase3_confusions["synthetic_boundary_right_val"],
            "synthetic_boundary_left_val": phase3_confusions["synthetic_boundary_left_val"],
            "synthetic_apex_val": phase3_confusions["synthetic_apex_val"],
        },
        args.output_dir / "phase3_confusions.png",
    )
    _phase_report(
        phase_reports_dir / "phase3_report.md",
        phase_name="phase3",
        artifact_dir=args.output_dir,
        gate_ok=phase3_ok,
        gate_reason=phase3_reason,
        baseline_metrics=baseline_metrics,
        phase_tracks=phase3_tracks,
        history_rows=phase3_history,
        confusion_path=args.output_dir / "phase3_confusions.png",
    )

    _plot_history(all_history, args.output_dir / "campaign_history.png")
    history_path = args.output_dir / "campaign_history.csv"
    if all_history:
        fieldnames = sorted({key for row in all_history for key in row.keys()})
        with history_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_history:
                writer.writerow(row)

    final_paths = {
        "final_best_real_val": Path(phase3_tracks["best_real_val"]["path"]) if phase3_tracks["best_real_val"]["path"] else (Path(phase2_tracks["best_real_val"]["path"]) if phase2_tracks["best_real_val"]["path"] else None),
        "final_best_balanced": Path(phase3_tracks["best_balanced"]["path"]) if phase3_tracks["best_balanced"]["path"] else (Path(phase2_tracks["best_balanced"]["path"]) if phase2_tracks["best_balanced"]["path"] else None),
        "final_best_synthetic": Path(phase3_tracks["best_synthetic"]["path"]) if phase3_tracks["best_synthetic"]["path"] else (Path(phase2_tracks["best_synthetic"]["path"]) if phase2_tracks["best_synthetic"]["path"] else None),
    }
    for key, src_path in final_paths.items():
        if src_path is None:
            continue
        dst = checkpoints_dir / f"{key}.pt"
        if src_path.resolve() != dst.resolve():
            ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
            torch.save(ckpt, dst)
            final_paths[key] = dst

    summary = _campaign_summary(
        args=args,
        feature_set=feature_set,
        model_type=model_type,
        inflated_ckpt=inflated_ckpt,
        inflation_check=inflation_check,
        phase_results=phase_results,
        final_paths=final_paths,
    )
    summary["inflated_checkpoint"] = str(inflated_init_path)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _final_report(args.report_path, artifact_dir=args.output_dir, baseline_metrics=baseline_metrics, phase_results=phase_results, final_paths=final_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
