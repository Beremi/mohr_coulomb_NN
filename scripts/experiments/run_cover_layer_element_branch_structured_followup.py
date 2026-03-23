from __future__ import annotations

import argparse
import copy
import json
import math
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        return {"branch_logits": logits.view(x.shape[0], self.out_points, self.out_classes)}


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        inner = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForwardBlock(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qh = self.norm_q(q)
        kvh = self.norm_kv(kv)
        out, _ = self.attn(qh, kvh, kvh, need_weights=False)
        q = q + out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class StructuredElementBranchNet(nn.Module):
    def __init__(
        self,
        *,
        node_input_dim: int = 6,
        node_dim: int = 128,
        encoder_depth: int = 3,
        cross_depth: int = 2,
        heads: int = 4,
        out_points: int = 11,
        out_classes: int = 5,
    ) -> None:
        super().__init__()
        self.node_input_dim = node_input_dim
        self.out_points = out_points
        self.out_classes = out_classes
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.node_role = nn.Embedding(10, node_dim)
        self.query_embed = nn.Embedding(out_points, node_dim)
        self.global_proj = nn.Linear(node_dim, node_dim)
        self.encoder = nn.ModuleList([SelfAttentionBlock(node_dim, heads=heads) for _ in range(encoder_depth)])
        self.cross = nn.ModuleList([CrossAttentionBlock(node_dim, heads=heads) for _ in range(cross_depth)])
        self.branch_head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, out_classes),
        )
        self.strain_head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 6),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        node_feat = x.view(batch, 10, self.node_input_dim)
        role_idx = torch.arange(10, device=x.device)
        h = self.node_proj(node_feat) + self.node_role(role_idx)[None, :, :]
        for block in self.encoder:
            h = block(h)
        global_token = h.mean(dim=1, keepdim=True)
        query_idx = torch.arange(self.out_points, device=x.device)
        q = self.query_embed(query_idx)[None, :, :].expand(batch, -1, -1)
        q = q + self.global_proj(global_token)
        for block in self.cross:
            q = block(q, h)
        return {
            "branch_logits": self.branch_head(q),
            "strain_pred": self.strain_head(q),
        }


def _canonical_features(coords: np.ndarray, disp: np.ndarray) -> np.ndarray:
    canonical = canonicalize_p2_element_states(coords, disp)
    per_node = np.concatenate([canonical.local_coords, canonical.local_displacements], axis=2)
    return per_node.astype(np.float32)


def _flatten_features(per_node_feat: np.ndarray) -> np.ndarray:
    return per_node_feat.reshape(per_node_feat.shape[0], -1).astype(np.float32)


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


def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["pattern_accuracy"], metrics["macro_recall"], metrics["accuracy"])


def _compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


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
        raise RuntimeError("Failed to build requested train/eval call split.")
    return train_calls, eval_calls


def _sample_exact_count(
    seed_bank: dict[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_parts: list[np.ndarray] = []
    disp_parts: list[np.ndarray] = []
    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    total = 0
    attempt = 0
    while total < sample_count:
        attempt += 1
        need = sample_count - total
        draw_count = max(int(math.ceil(need * 1.4)), 256)
        coords, disp, strain, branch, _material, _valid = synthesize_element_states_from_seeded_noise(
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
        strain_parts.append(strain)
        branch_parts.append(branch)
        total += coords.shape[0]
        if attempt >= 40 and total < sample_count:
            raise RuntimeError(f"Failed to reach requested synthetic count {sample_count}; only collected {total}.")
    coords_full = np.concatenate(coords_parts, axis=0)[:sample_count]
    disp_full = np.concatenate(disp_parts, axis=0)[:sample_count]
    strain_full = np.concatenate(strain_parts, axis=0)[:sample_count]
    branch_full = np.concatenate(branch_parts, axis=0)[:sample_count]
    return coords_full, disp_full, strain_full, branch_full


def _scale_fit(train_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = _flatten_features(train_feat)
    mean = flat.mean(axis=0)
    std = np.where(flat.std(axis=0) < 1.0e-6, 1.0, flat.std(axis=0))
    return mean.astype(np.float32), std.astype(np.float32)


def _scale_apply(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    flat = _flatten_features(feat)
    return ((flat - mean) / std).astype(np.float32)


def _tensorize(
    feat_np: np.ndarray,
    branch_np: np.ndarray,
    strain_np: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(feat_np).to(device),
        torch.from_numpy(branch_np.astype(np.int64)).to(device),
        torch.from_numpy(strain_np.astype(np.float32)).to(device),
    )


def _evaluate_model(
    model: nn.Module,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    model.eval()
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, (x, branch, strain) in eval_sets.items():
            pred = model(x)
            branch_metrics = _metrics(pred["branch_logits"], branch)
            if "strain_pred" in pred:
                strain_mae = float(torch.mean(torch.abs(pred["strain_pred"] - strain)).cpu().item())
            else:
                strain_mae = float("nan")
            out[name] = {**branch_metrics, "strain_mae": strain_mae}
    return out


def _loss_fn(
    pred: dict[str, torch.Tensor],
    branch_true: torch.Tensor,
    strain_true: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    strain_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = nn.functional.cross_entropy(
        pred["branch_logits"].reshape(-1, len(BRANCH_NAMES)),
        branch_true.reshape(-1),
        weight=class_weights,
    )
    total = ce
    strain_loss_value = 0.0
    if strain_weight > 0.0 and "strain_pred" in pred:
        strain_loss = nn.functional.smooth_l1_loss(pred["strain_pred"], strain_true)
        total = total + strain_weight * strain_loss
        strain_loss_value = float(strain_loss.detach().cpu().item())
    return total, {"branch_ce": float(ce.detach().cpu().item()), "strain_loss": strain_loss_value}


def _lbfgs_tail(
    model: nn.Module,
    *,
    x_train: torch.Tensor,
    branch_train: torch.Tensor,
    strain_train: torch.Tensor,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    class_weights: torch.Tensor,
    strain_weight: float,
    epochs: int,
    lr: float,
    max_iter: int,
    history_size: int,
    best_score: tuple[float, float, float],
) -> tuple[tuple[float, float, float], dict[str, torch.Tensor] | None]:
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
            pred = model(x_train)
            loss, _ = _loss_fn(
                pred,
                branch_train,
                strain_train,
                class_weights=class_weights,
                strain_weight=strain_weight,
            )
            loss.backward()
            return loss

        model.train(True)
        optimizer.step(closure)
        metrics = _evaluate_model(model, eval_sets)
        score = _score(metrics["synthetic_val"])
        if score > best_local:
            best_local = score
            accepted_state = copy.deepcopy(model.state_dict())
    return best_local, accepted_state


def _train_fixed_experiment(
    *,
    name: str,
    model_kind: str,
    train_feat: np.ndarray,
    train_branch: np.ndarray,
    train_strain: np.ndarray,
    val_feat: np.ndarray,
    val_branch: np.ndarray,
    val_strain: np.ndarray,
    diagnostic_sets_np: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    device: torch.device,
    width: int = 256,
    depth: int = 6,
    node_dim: int = 128,
    encoder_depth: int = 3,
    cross_depth: int = 2,
    strain_weight: float = 0.0,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    batch_sizes: list[int] | None = None,
    stage_max_epochs: int = 80,
    stage_patience: int = 24,
    plateau_patience: int = 8,
    plateau_factor: float = 0.5,
    min_lr: float = 1.0e-6,
    lbfgs_epochs: int = 5,
    lbfgs_lr: float = 0.25,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 50,
) -> dict[str, object]:
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 256]

    x_mean, x_std = _scale_fit(train_feat)
    x_train_np = _scale_apply(train_feat, x_mean, x_std)
    x_val_np = _scale_apply(val_feat, x_mean, x_std)
    x_train, y_train, s_train = _tensorize(x_train_np, train_branch, train_strain, device=device)
    x_val, y_val, s_val = _tensorize(x_val_np, val_branch, val_strain, device=device)
    eval_sets = {"synthetic_val": (x_val, y_val, s_val)}
    for key, (feat, branch, strain) in diagnostic_sets_np.items():
        eval_sets[key] = _tensorize(_scale_apply(feat, x_mean, x_std), branch, strain, device=device)

    if model_kind == "baseline_mlp":
        model = ElementBranchMLP(in_dim=x_train.shape[1], width=width, depth=depth).to(device)
    elif model_kind == "structured":
        model = StructuredElementBranchNet(node_dim=node_dim, encoder_depth=encoder_depth, cross_depth=cross_depth).to(device)
    else:
        raise ValueError(f"Unknown model_kind {model_kind!r}.")

    class_weights = _compute_class_weights(train_branch).to(device)
    history: list[dict[str, float]] = []
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stage_lr = lr
    global_epoch = 0
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    for stage_index, batch_size in enumerate(batch_sizes, start=1):
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
        loader = DataLoader(TensorDataset(x_train.cpu(), y_train.cpu(), s_train.cpu()), batch_size=batch_size, shuffle=True)
        for _ in range(stage_max_epochs):
            global_epoch += 1
            model.train(True)
            train_loss = 0.0
            train_count = 0
            branch_ce_mean = 0.0
            strain_loss_mean = 0.0
            for xb_cpu, yb_cpu, sb_cpu in loader:
                xb = xb_cpu.to(device)
                yb = yb_cpu.to(device)
                sb = sb_cpu.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss, pieces = _loss_fn(
                    pred,
                    yb,
                    sb,
                    class_weights=class_weights,
                    strain_weight=strain_weight,
                )
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * xb.shape[0]
                branch_ce_mean += pieces["branch_ce"] * xb.shape[0]
                strain_loss_mean += pieces["strain_loss"] * xb.shape[0]
                train_count += xb.shape[0]
            train_loss /= max(train_count, 1)
            branch_ce_mean /= max(train_count, 1)
            strain_loss_mean /= max(train_count, 1)
            metrics = _evaluate_model(model, eval_sets)
            score = _score(metrics["synthetic_val"])
            scheduler.step(score[0])
            row = {
                "epoch": global_epoch,
                "stage_index": stage_index,
                "batch_size": batch_size,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_branch_ce": branch_ce_mean,
                "train_strain_loss": strain_loss_mean,
            }
            for split_name, split_metrics in metrics.items():
                for key, value in split_metrics.items():
                    row[f"{split_name}_{key}"] = float(value)
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

            if global_epoch == 1 or global_epoch % 25 == 0:
                print(
                    f"[{name}] epoch={global_epoch} stage={stage_index}/{len(batch_sizes)} batch={batch_size} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} train_loss={train_loss:.4f} "
                    f"val_acc={metrics['synthetic_val']['accuracy']:.4f} "
                    f"val_macro={metrics['synthetic_val']['macro_recall']:.4f} "
                    f"val_pattern={metrics['synthetic_val']['pattern_accuracy']:.4f} "
                    f"val_strain_mae={metrics['synthetic_val']['strain_mae']:.4f}"
                )
            if no_improve >= stage_patience:
                break
        stage_lr = float(optimizer.param_groups[0]["lr"])

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    if lbfgs_epochs > 0:
        lbfgs_best, lbfgs_state = _lbfgs_tail(
            model,
            x_train=x_train,
            branch_train=y_train,
            strain_train=s_train,
            eval_sets=eval_sets,
            class_weights=class_weights,
            strain_weight=strain_weight,
            epochs=lbfgs_epochs,
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            best_score=best_score,
        )
        if lbfgs_state is not None and lbfgs_best > best_score:
            best_score = lbfgs_best
            best_state = lbfgs_state
            model.load_state_dict(best_state)

    metrics = _evaluate_model(model, eval_sets)
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "x_mean": x_mean,
            "x_std": x_std,
            "model_kind": model_kind,
            "width": width,
            "depth": depth,
            "node_dim": node_dim,
            "encoder_depth": encoder_depth,
            "cross_depth": cross_depth,
            "strain_weight": strain_weight,
        },
        exp_dir / "best.pt",
    )
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (exp_dir / "summary.json").write_text(
        json.dumps(
            {
                "model_kind": model_kind,
                "best_epoch": best_epoch,
                "best_score": {
                    "pattern_accuracy": best_score[0],
                    "macro_recall": best_score[1],
                    "accuracy": best_score[2],
                },
                "metrics": metrics,
                "checkpoint": str(exp_dir / "best.pt"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_history(history, exp_dir / "history.png")
    return {
        "model_kind": model_kind,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "checkpoint": str(exp_dir / "best.pt"),
        "history_plot": str(exp_dir / "history.png"),
    }


def _plot_history(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    epoch = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epoch, [row["train_loss"] for row in rows], label="train")
    axes[0].set_title("Train Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    for key, label in [
        ("synthetic_val_accuracy", "acc"),
        ("synthetic_val_macro_recall", "macro"),
        ("synthetic_val_pattern_accuracy", "pattern"),
    ]:
        axes[1].plot(epoch, [row[key] for row in rows], label=label)
    axes[1].set_title("Synthetic Val")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(epoch, [row["lr"] for row in rows], label="lr")
    axes[2].set_title("Learning Rate")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_compare(rows: list[tuple[str, float]], title: str, ylabel: str, output_path: Path) -> None:
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    report_path: Path,
    *,
    artifact_dir: Path,
    diagnostics: dict[str, object],
    structured_gate_a: dict[str, object],
    structured_gate_b: dict[str, object],
) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    diag_rows = []
    for name, payload in diagnostics.items():
        m = payload["metrics"]["synthetic_val"]
        diag_rows.append(
            f"- `{name}`: val acc `{m['accuracy']:.4f}`, val macro `{m['macro_recall']:.4f}`, "
            f"val pattern `{m['pattern_accuracy']:.4f}`"
        )
    gate_a_metrics = structured_gate_a["metrics"]["synthetic_val"]
    gate_b_syn = structured_gate_b["metrics"]["synthetic_val"]
    gate_b_real = structured_gate_b["metrics"]["real_test"]
    lines = [
        "# Cover Layer Direct Element Branch Predictor: Structured Follow-Up",
        "",
        "## Summary",
        "",
        "This follow-up does two things:",
        "",
        "1. separates the old Gate B failure into `B1/B2/B3` on the flat direct MLP",
        "2. replaces the flat MLP with a structured raw-element model that also predicts auxiliary strain",
        "",
        "The input/output contract is unchanged:",
        "",
        "- input: canonicalized `coords(10x3) + disp(10x3)`",
        "- output: `11 x 5` branch logits",
        "",
        "## B1 / B2 / B3 Diagnosis with the Flat Direct MLP",
        "",
        "`B1` = same train seed pool, disjoint synthetic samples.",
        "`B2` = disjoint synthetic seed-call validation.",
        "`B3` = expanded train seed-call pool (`48`) against disjoint eval seed calls (`8`).",
        "",
        *diag_rows,
        "",
        "![B-diagnostic pattern accuracy](../" + str(rel / "b_diagnostic_pattern_accuracy.png") + ")",
        "![B-diagnostic point accuracy](../" + str(rel / "b_diagnostic_point_accuracy.png") + ")",
        "",
        "Interpretation:",
        "",
        "- if `B1` is already poor, the flat direct MLP does not generalize within the synthetic domain itself",
        "- if `B1` is good but `B2/B3` collapse, the main problem is cross-seed-call generalization",
        "",
        "## Structured Raw-Element Model",
        "",
        "Architecture:",
        "",
        "- node input: `10 x 6` canonicalized `(x,y,z,u,v,w)`",
        "- node encoder with role embeddings for the `10` P2 nodes",
        "- self-attention across nodes",
        "- `11` learned integration-point queries with cross-attention into node tokens",
        "- branch head: `11 x 5`",
        "- auxiliary strain head: `11 x 6`",
        "",
        "The strain head is supervised with exact synthetic strain and used only as an auxiliary training target.",
        "",
        "## Structured Gate A",
        "",
        f"- synthetic val acc: `{gate_a_metrics['accuracy']:.4f}`",
        f"- synthetic val macro recall: `{gate_a_metrics['macro_recall']:.4f}`",
        f"- synthetic val pattern accuracy: `{gate_a_metrics['pattern_accuracy']:.4f}`",
        f"- synthetic val strain MAE: `{gate_a_metrics['strain_mae']:.4f}`",
        "",
        "![Structured Gate A history](../" + str(Path(structured_gate_a["history_plot"]).relative_to(report_path.parent.parent)) + ")",
        "",
        "## Structured Gate B",
        "",
        f"- synthetic val acc: `{gate_b_syn['accuracy']:.4f}`",
        f"- synthetic val macro recall: `{gate_b_syn['macro_recall']:.4f}`",
        f"- synthetic val pattern accuracy: `{gate_b_syn['pattern_accuracy']:.4f}`",
        f"- synthetic val strain MAE: `{gate_b_syn['strain_mae']:.4f}`",
        "",
        f"- real test acc: `{gate_b_real['accuracy']:.4f}`",
        f"- real test macro recall: `{gate_b_real['macro_recall']:.4f}`",
        f"- real test pattern accuracy: `{gate_b_real['pattern_accuracy']:.4f}`",
        "",
        "![Structured Gate B history](../" + str(Path(structured_gate_b["history_plot"]).relative_to(report_path.parent.parent)) + ")",
        "",
        "## Conclusion",
        "",
        "This follow-up is trying to answer whether the next move is still training recipe or now architecture.",
        "",
        "- If the structured model improves Gate B materially, then the flat MLP was the main bottleneck.",
        "- If the structured model is still poor on synthetic Gate B, then the synthetic domain itself is too diverse for the current direct formulation, and the next move should be to narrow or factor the task.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured follow-up for the direct cover-layer element branch predictor.")
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
        default=Path("experiment_runs/real_sim/cover_layer_element_branch_structured_followup_20260314"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_element_branch_structured_followup.md"),
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
    train_24, eval_8 = _split_seed_calls(splits["generator_fit"], regimes=regimes, train_count=24, eval_count=8)
    train_48, eval_8_b3 = _split_seed_calls(splits["generator_fit"], regimes=regimes, train_count=48, eval_count=8)

    def build_seed_bank(call_names: list[str], seed: int) -> dict[str, np.ndarray]:
        coords, disp, _strain, branch, material = collect_blocks(
            args.export,
            call_names=call_names,
            max_elements_per_call=args.max_elements_per_call,
            seed=seed,
        )
        return fit_seed_noise_bank(coords, disp, branch, material)

    train_bank_24 = build_seed_bank(train_24, args.seed + 1)
    eval_bank_8 = build_seed_bank(eval_8, args.seed + 2)
    train_bank_48 = build_seed_bank(train_48, args.seed + 3)
    real_val_calls = _spread_pick_exact(splits["real_val"], count=4, regimes=regimes)
    real_test_calls = _spread_pick_exact(splits["real_test"], count=4, regimes=regimes)
    coords_real_val, disp_real_val, strain_real_val, branch_real_val, _ = collect_blocks(
        args.export,
        call_names=real_val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 20,
    )
    coords_real_test, disp_real_test, strain_real_test, branch_real_test, _ = collect_blocks(
        args.export,
        call_names=real_test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 21,
    )
    real_val = (_canonical_features(coords_real_val, disp_real_val), branch_real_val.astype(np.int64), strain_real_val.astype(np.float32))
    real_test = (_canonical_features(coords_real_test, disp_real_test), branch_real_test.astype(np.int64), strain_real_test.astype(np.float32))

    def fixed_set(bank: dict[str, np.ndarray], *, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords, disp, strain, branch = _sample_exact_count(
            bank,
            sample_count=count,
            seed=seed,
            noise_scale=0.20,
            selection="branch_balanced",
        )
        return _canonical_features(coords, disp), branch.astype(np.int64), strain.astype(np.float32)

    b1_train = fixed_set(train_bank_24, count=2048, seed=args.seed + 40)
    b1_val = fixed_set(train_bank_24, count=2048, seed=args.seed + 41)
    b2_train = fixed_set(train_bank_24, count=2048, seed=args.seed + 42)
    b2_val = fixed_set(eval_bank_8, count=2048, seed=args.seed + 43)
    b3_train = fixed_set(train_bank_48, count=2048, seed=args.seed + 44)
    b3_val = fixed_set(build_seed_bank(eval_8_b3, args.seed + 4), count=2048, seed=args.seed + 45)

    diagnostics: dict[str, object] = {}
    for name, train_set, val_set in [
        ("b1_same_seed_pool", b1_train, b1_val),
        ("b2_disjoint_seed_calls", b2_train, b2_val),
        ("b3_expanded_train_calls", b3_train, b3_val),
    ]:
        diagnostics[name] = _train_fixed_experiment(
            name=name,
            model_kind="baseline_mlp",
            train_feat=train_set[0],
            train_branch=train_set[1],
            train_strain=train_set[2],
            val_feat=val_set[0],
            val_branch=val_set[1],
            val_strain=val_set[2],
            diagnostic_sets_np={"real_val": real_val, "real_test": real_test},
            output_dir=args.output_dir,
            device=device,
            width=256,
            depth=6,
            strain_weight=0.0,
            batch_sizes=[32, 64, 128, 256],
            stage_max_epochs=80,
            stage_patience=24,
            plateau_patience=8,
        )

    _plot_bar_compare(
        [(name, payload["metrics"]["synthetic_val"]["pattern_accuracy"]) for name, payload in diagnostics.items()],
        "Flat MLP Gate-B Diagnostics: Pattern Accuracy",
        "Pattern accuracy",
        args.output_dir / "b_diagnostic_pattern_accuracy.png",
    )
    _plot_bar_compare(
        [(name, payload["metrics"]["synthetic_val"]["accuracy"]) for name, payload in diagnostics.items()],
        "Flat MLP Gate-B Diagnostics: Point Accuracy",
        "Point accuracy",
        args.output_dir / "b_diagnostic_point_accuracy.png",
    )
    (args.output_dir / "b_diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    gate_a_train = fixed_set(train_bank_24, count=256, seed=args.seed + 60)
    structured_gate_a = _train_fixed_experiment(
        name="structured_gate_a",
        model_kind="structured",
        train_feat=gate_a_train[0],
        train_branch=gate_a_train[1],
        train_strain=gate_a_train[2],
        val_feat=gate_a_train[0],
        val_branch=gate_a_train[1],
        val_strain=gate_a_train[2],
        diagnostic_sets_np={"real_val": real_val, "real_test": real_test},
        output_dir=args.output_dir,
        device=device,
        node_dim=128,
        encoder_depth=3,
        cross_depth=2,
        strain_weight=0.25,
        batch_sizes=[64],
        stage_max_epochs=600,
        stage_patience=601,
        plateau_patience=20,
        lbfgs_epochs=3,
    )

    structured_gate_b = _train_fixed_experiment(
        name="structured_gate_b",
        model_kind="structured",
        train_feat=b2_train[0],
        train_branch=b2_train[1],
        train_strain=b2_train[2],
        val_feat=b2_val[0],
        val_branch=b2_val[1],
        val_strain=b2_val[2],
        diagnostic_sets_np={"real_val": real_val, "real_test": real_test},
        output_dir=args.output_dir,
        device=device,
        node_dim=128,
        encoder_depth=3,
        cross_depth=2,
        strain_weight=0.25,
        batch_sizes=[32, 64, 128, 256],
        stage_max_epochs=80,
        stage_patience=24,
        plateau_patience=8,
        lbfgs_epochs=5,
    )

    _write_report(
        args.report_path,
        artifact_dir=args.output_dir,
        diagnostics=diagnostics,
        structured_gate_a=structured_gate_a,
        structured_gate_b=structured_gate_b,
    )
    print(
        json.dumps(
            {
                "diagnostics": diagnostics,
                "structured_gate_a": structured_gate_a,
                "structured_gate_b": structured_gate_b,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
