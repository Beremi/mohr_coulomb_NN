#!/usr/bin/env python
"""Train dedicated branch gates for the cover-layer branch experts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.models import ResidualBlock, Standardizer, build_raw_features, build_trial_features, compute_trial_stress
from mc_surrogate.mohr_coulomb import BRANCH_NAMES
from mc_surrogate.training import choose_device, predict_with_checkpoint, set_seed

sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from run_cover_layer_branch_experts import _eval_metrics  # noqa: E402
from run_cover_layer_single_material_plan import _compute_real_dissection, _plot_error_vs_magnitude, _plot_relative_error_cdf  # noqa: E402


class GateNet(nn.Module):
    def __init__(self, input_dim: int, width: int = 512, depth: int = 4, n_branches: int = 5) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=0.0) for _ in range(depth)])
        self.head = nn.Linear(width, n_branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="experiment_runs/real_sim/cover_layer_single_material_20260313")
    parser.add_argument("--experts-root", default="experiment_runs/real_sim/cover_layer_branch_experts_20260313")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_gate_experiments_20260313")
    parser.add_argument("--report-md", default="docs/cover_layer_gate_experiments.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--plateau-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--log-every-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1661)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_split_arrays(dataset_path: Path, split: str) -> dict[str, np.ndarray]:
    with h5py.File(dataset_path, "r") as f:
        split_id = f["split_id"][:]
        split_map = {"train": 0, "val": 1, "test": 2}
        mask = split_id == split_map[split]
        return {
            "strain_eng": f["strain_eng"][mask],
            "stress": f["stress"][mask],
            "material_reduced": f["material_reduced"][mask],
            "branch_id": f["branch_id"][mask],
        }


def _build_features(feature_kind: str, strain_eng: np.ndarray, material_reduced: np.ndarray) -> np.ndarray:
    if feature_kind == "raw":
        return build_raw_features(strain_eng, material_reduced)
    if feature_kind == "trial":
        return build_trial_features(strain_eng, material_reduced)
    raise ValueError(feature_kind)


def _macro_recall(branch_true: np.ndarray, branch_pred: np.ndarray) -> float:
    recalls = []
    for i in range(len(BRANCH_NAMES)):
        mask = branch_true == i
        if np.any(mask):
            recalls.append(float(np.mean(branch_pred[mask] == i)))
    return float(np.mean(recalls))


def _train_gate(
    *,
    name: str,
    feature_kind: str,
    dataset_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    train = _load_split_arrays(dataset_path, "train")
    val = _load_split_arrays(dataset_path, "val")
    x_train = _build_features(feature_kind, train["strain_eng"], train["material_reduced"])
    x_val = _build_features(feature_kind, val["strain_eng"], val["material_reduced"])
    scaler = Standardizer.from_array(x_train)

    train_ds = TensorDataset(
        torch.from_numpy(scaler.transform(x_train)),
        torch.from_numpy(train["branch_id"].astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(scaler.transform(x_val)),
        torch.from_numpy(val["branch_id"].astype(np.int64)),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GateNet(input_dim=x_train.shape[1], width=args.width, depth=args.depth).to(device)
    train_counts = np.bincount(train["branch_id"].astype(int), minlength=len(BRANCH_NAMES)).astype(np.float32)
    class_weights = train_counts.mean() / np.maximum(train_counts, 1.0)
    class_weights = torch.from_numpy(class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.plateau_patience,
        min_lr=args.min_lr,
    )

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_macro_recall", "val_macro_recall"])

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train(True)
        train_loss_sum = 0.0
        train_true_parts = []
        train_pred_parts = []
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * xb.shape[0]
            train_true_parts.append(yb.detach().cpu().numpy())
            train_pred_parts.append(logits.argmax(dim=1).detach().cpu().numpy())
            n_train += xb.shape[0]

        model.train(False)
        val_loss_sum = 0.0
        val_true_parts = []
        val_pred_parts = []
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb, weight=class_weights)
                val_loss_sum += float(loss.detach().cpu()) * xb.shape[0]
                val_true_parts.append(yb.detach().cpu().numpy())
                val_pred_parts.append(logits.argmax(dim=1).detach().cpu().numpy())
                n_val += xb.shape[0]

        train_true = np.concatenate(train_true_parts)
        train_pred = np.concatenate(train_pred_parts)
        val_true = np.concatenate(val_true_parts)
        val_pred = np.concatenate(val_pred_parts)
        train_loss = train_loss_sum / max(n_train, 1)
        val_loss = val_loss_sum / max(n_val, 1)
        train_acc = float(np.mean(train_true == train_pred))
        val_acc = float(np.mean(val_true == val_pred))
        train_macro = _macro_recall(train_true, train_pred)
        val_macro = _macro_recall(val_true, val_pred)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, lr, train_loss, val_loss, train_acc, val_acc, train_macro, val_macro])

        if epoch == 1 or epoch % args.log_every_epochs == 0:
            print(
                f"[gate {name}] epoch={epoch} lr={lr:.3e} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_acc={val_acc:.6f} val_macro={val_macro:.6f} best_macro={best_metric:.6f}"
            )

        if val_macro > best_metric:
            best_metric = val_macro
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        "feature_kind": feature_kind,
                        "input_dim": x_train.shape[1],
                        "width": args.width,
                        "depth": args.depth,
                        "x_scaler": scaler.to_dict(),
                    },
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    _plot_gate_history(history_path, run_dir / "history.png", title=name)
    summary = {
        "best_macro_recall": best_metric,
        "best_epoch": best_epoch,
        "history_csv": str(history_path),
        "checkpoint": str(best_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _plot_gate_history(history_csv: Path, output_path: Path, title: str) -> Path:
    data = np.genfromtxt(history_csv, delimiter=",", names=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(data["epoch"], data["train_loss"], label="train")
    axes[0].plot(data["epoch"], data["val_loss"], label="val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(data["epoch"], data["train_macro_recall"], label="train macro recall")
    axes[1].plot(data["epoch"], data["val_macro_recall"], label="val macro recall")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro recall")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _load_gate_checkpoint(path: Path, device: str) -> tuple[GateNet, dict[str, Any]]:
    device_obj = choose_device(device)
    ckpt = torch.load(path, map_location=device_obj)
    meta = ckpt["metadata"]
    model = GateNet(input_dim=meta["input_dim"], width=meta["width"], depth=meta["depth"]).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta


def _predict_gate(checkpoint_path: Path, feature_kind: str, strain_eng: np.ndarray, material_reduced: np.ndarray, device: str) -> np.ndarray:
    model, meta = _load_gate_checkpoint(checkpoint_path, device)
    scaler = Standardizer.from_dict(meta["x_scaler"])
    features = _build_features(feature_kind, strain_eng, material_reduced)
    x = torch.from_numpy(scaler.transform(features)).to(choose_device(device))
    probs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 16384):
            logits = model(x[start : start + 16384])
            logits = logits - logits.max(dim=1, keepdim=True).values
            prob = torch.softmax(logits, dim=1)
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def _load_expert_paths(experts_root: Path) -> dict[int, Path]:
    return {
        1: experts_root / "expert_smooth" / "best.pt",
        2: experts_root / "expert_left_edge" / "best.pt",
        3: experts_root / "expert_right_edge" / "best.pt",
        4: experts_root / "expert_apex" / "best.pt",
    }


def _predict_all_experts(expert_paths: dict[int, Path], strain_eng: np.ndarray, material_reduced: np.ndarray, device: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for branch_id, path in expert_paths.items():
        out[branch_id] = predict_with_checkpoint(path, strain_eng, material_reduced, device=device, batch_size=16384)["stress"]
    return out


def _baseline_predictions(reference_root: Path, strain_eng: np.ndarray, material_reduced: np.ndarray, device: str) -> dict[str, np.ndarray]:
    ckpt = reference_root / "baseline_raw_branch" / "best.pt"
    pred = predict_with_checkpoint(ckpt, strain_eng, material_reduced, device=device, batch_size=16384)
    return {"stress": pred["stress"], "branch_probs": pred["branch_probabilities"]}


def _ensemble_from_gate(
    *,
    gate_probs: np.ndarray,
    trial_stress: np.ndarray,
    baseline_stress: np.ndarray,
    expert_stress: dict[int, np.ndarray],
    mode: str,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    pred_branch = np.argmax(gate_probs, axis=1)
    if mode == "hard":
        stress = np.empty_like(baseline_stress)
        for branch_id in range(len(BRANCH_NAMES)):
            mask = pred_branch == branch_id
            if not np.any(mask):
                continue
            if branch_id == 0:
                stress[mask] = trial_stress[mask]
            else:
                stress[mask] = expert_stress[branch_id][mask]
        return stress.astype(np.float32), pred_branch.astype(np.int64)

    if mode == "soft":
        stress = gate_probs[:, [0]] * trial_stress
        for branch_id in range(1, len(BRANCH_NAMES)):
            stress = stress + gate_probs[:, [branch_id]] * expert_stress[branch_id]
        return stress.astype(np.float32), pred_branch.astype(np.int64)

    if mode == "threshold":
        if threshold is None:
            raise ValueError("threshold mode requires threshold")
        stress, pred_branch = _ensemble_from_gate(
            gate_probs=gate_probs,
            trial_stress=trial_stress,
            baseline_stress=baseline_stress,
            expert_stress=expert_stress,
            mode="hard",
            threshold=None,
        )
        conf = np.max(gate_probs, axis=1)
        mask = conf < threshold
        if np.any(mask):
            stress[mask] = baseline_stress[mask]
        return stress.astype(np.float32), pred_branch.astype(np.int64)

    raise ValueError(mode)


def _plot_confusion(confusion: list[list[int]], output_path: Path) -> Path:
    mat = np.asarray(confusion, dtype=float)
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(BRANCH_NAMES)), BRANCH_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(BRANCH_NAMES)), BRANCH_NAMES)
    plt.xlabel("predicted branch")
    plt.ylabel("true branch")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_parity(stress_true: np.ndarray, stress_pred: np.ndarray, output_path: Path) -> Path:
    if stress_true.shape[0] > 4000:
        rng = np.random.default_rng(0)
        idx = rng.choice(stress_true.shape[0], size=4000, replace=False)
        stress_true = stress_true[idx]
        stress_pred = stress_pred[idx]
    lo = float(min(stress_true.min(), stress_pred.min()))
    hi = float(max(stress_true.max(), stress_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(stress_true.reshape(-1), stress_pred.reshape(-1), s=6, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("true stress")
    plt.ylabel("predicted stress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _evaluate_gate_mode(name: str, stress_true: np.ndarray, branch_true: np.ndarray, stress_pred: np.ndarray, branch_pred: np.ndarray, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _eval_metrics(stress_true, stress_pred, branch_true, branch_pred)
    (output_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _plot_parity(stress_true, stress_pred, output_dir / "parity.png")
    _plot_relative_error_cdf(stress_true, stress_pred, output_dir / "relative_error_cdf.png", title=f"{name} relative error")
    _plot_error_vs_magnitude(stress_true, stress_pred, output_dir / "error_vs_magnitude.png", title=f"{name} error vs magnitude")
    _plot_confusion(metrics["branch_confusion"], output_dir / "branch_confusion.png")
    return metrics


def _plot_mode_compare(rows: list[dict[str, Any]], output_path: Path, key: str, title: str) -> Path:
    names = [row["name"] for row in rows]
    real = [row["real"][key] for row in rows]
    synth = [row["synthetic"][key] for row in rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, real, width=width, label="real")
    ax.bar(x + width / 2, synth, width=width, label="synthetic")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_report(report_path: Path, output_root: Path, gate_summaries: dict[str, Any], rows: list[dict[str, Any]], best_name: str, best_dissection: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Cover-Layer Gate Experiments")
    lines.append("")
    lines.append("This report trains dedicated branch gates for the already-learned plastic experts and tests whether routing quality is now enough to beat the direct baseline.")
    lines.append("")
    lines.append("## Gate Training")
    lines.append("")
    lines.append("| Gate | Best Val Macro Recall | Best Epoch |")
    lines.append("|---|---:|---:|")
    for name, summary in gate_summaries.items():
        lines.append(f"| {name} | {summary['best_macro_recall']:.4f} | {summary['best_epoch']} |")
    lines.append("")
    lines.append("## Ensemble Results")
    lines.append("")
    lines.append("| Mode | Real MAE | Real RMSE | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['real']['stress_mae']:.4f} | {row['real']['stress_rmse']:.4f} | {row['real']['branch_accuracy']:.4f} | "
            f"{row['synthetic']['stress_mae']:.4f} | {row['synthetic']['stress_rmse']:.4f} | {row['synthetic']['branch_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append(f"Best real-holdout mode: `{best_name}`")
    lines.append("")
    lines.append(f"![MAE comparison]({(output_root / 'compare_mae.png').as_posix()})")
    lines.append("")
    lines.append(f"![RMSE comparison]({(output_root / 'compare_rmse.png').as_posix()})")
    lines.append("")
    lines.append("## Gate Histories")
    lines.append("")
    for name in gate_summaries:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"![history]({(output_root / name / 'history.png').as_posix()})")
        lines.append("")
    lines.append("## Mode Figures")
    lines.append("")
    for row in rows:
        mode_dir = output_root / row["name"]
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- real parity: ![real parity]({(mode_dir / 'real' / 'parity.png').as_posix()})")
        lines.append(f"- real relative error: ![real rel]({(mode_dir / 'real' / 'relative_error_cdf.png').as_posix()})")
        lines.append(f"- real error vs magnitude: ![real mag]({(mode_dir / 'real' / 'error_vs_magnitude.png').as_posix()})")
        lines.append(f"- real branch confusion: ![real branch]({(mode_dir / 'real' / 'branch_confusion.png').as_posix()})")
        lines.append(f"- synthetic parity: ![synth parity]({(mode_dir / 'synthetic' / 'parity.png').as_posix()})")
        lines.append(f"- synthetic relative error: ![synth rel]({(mode_dir / 'synthetic' / 'relative_error_cdf.png').as_posix()})")
        lines.append(f"- synthetic error vs magnitude: ![synth mag]({(mode_dir / 'synthetic' / 'error_vs_magnitude.png').as_posix()})")
        lines.append(f"- synthetic branch confusion: ![synth branch]({(mode_dir / 'synthetic' / 'branch_confusion.png').as_posix()})")
        lines.append("")
    lines.append("## Best Real-Holdout Dissection")
    lines.append("")
    lines.append(f"- mean relative sample error: `{best_dissection['relative_error_mean']:.4f}`")
    lines.append(f"- median relative sample error: `{best_dissection['relative_error_median']:.4f}`")
    lines.append(f"- p90 relative sample error: `{best_dissection['relative_error_p90']:.4f}`")
    lines.append(f"- p99 relative sample error: `{best_dissection['relative_error_p99']:.4f}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- If a dedicated gate beats the baseline in deployable hard or soft routing, the branch-expert path is now validated end to end.")
    lines.append("- If soft routing helps but hard routing still fails, the gate probabilities are useful but not sharp enough for top-1 dispatch.")
    lines.append("- If nothing beats the baseline, then even with dedicated gates the branch-expert route is still not enough.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reference_root = (ROOT / args.reference_root).resolve()
    experts_root = (ROOT / args.experts_root).resolve()
    output_root = ROOT / args.output_root
    report_path = ROOT / args.report_md
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    synth_dataset = reference_root / "cover_layer_full_synthetic_holdout.h5"
    gate_specs = [("gate_raw", "raw"), ("gate_trial", "trial")]
    gate_summaries: dict[str, Any] = {}
    for idx, (name, feature_kind) in enumerate(gate_specs):
        run_dir = output_root / name
        if not (run_dir / "best.pt").exists():
            summary = _train_gate(
                name=name,
                feature_kind=feature_kind,
                dataset_path=real_dataset,
                output_root=output_root,
                args=args,
                seed=args.seed + idx,
            )
        else:
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        gate_summaries[name] = summary

    real_arrays = _load_split_arrays(real_dataset, "test")
    real_val = _load_split_arrays(real_dataset, "val")
    synth_arrays = _load_split_arrays(synth_dataset, "test")
    baseline_real = _baseline_predictions(reference_root, real_arrays["strain_eng"], real_arrays["material_reduced"], args.device)
    baseline_val = _baseline_predictions(reference_root, real_val["strain_eng"], real_val["material_reduced"], args.device)
    baseline_synth = _baseline_predictions(reference_root, synth_arrays["strain_eng"], synth_arrays["material_reduced"], args.device)
    expert_paths = _load_expert_paths(experts_root)
    real_experts = _predict_all_experts(expert_paths, real_arrays["strain_eng"], real_arrays["material_reduced"], args.device)
    val_experts = _predict_all_experts(expert_paths, real_val["strain_eng"], real_val["material_reduced"], args.device)
    synth_experts = _predict_all_experts(expert_paths, synth_arrays["strain_eng"], synth_arrays["material_reduced"], args.device)
    trial_real = compute_trial_stress(real_arrays["strain_eng"], real_arrays["material_reduced"])
    trial_val = compute_trial_stress(real_val["strain_eng"], real_val["material_reduced"])
    trial_synth = compute_trial_stress(synth_arrays["strain_eng"], synth_arrays["material_reduced"])

    rows: list[dict[str, Any]] = []

    # References.
    baseline_ref_real = _eval_metrics(real_arrays["stress"], baseline_real["stress"], real_arrays["branch_id"], np.argmax(baseline_real["branch_probs"], axis=1))
    baseline_ref_synth = _eval_metrics(synth_arrays["stress"], baseline_synth["stress"], synth_arrays["branch_id"], np.argmax(baseline_synth["branch_probs"], axis=1))
    rows.append({"name": "baseline_reference", "real": baseline_ref_real, "synthetic": baseline_ref_synth})

    # Oracle expert ceiling.
    oracle_real_stress = trial_real.copy()
    oracle_synth_stress = trial_synth.copy()
    for branch_id in range(1, len(BRANCH_NAMES)):
        mask_real = real_arrays["branch_id"] == branch_id
        mask_synth = synth_arrays["branch_id"] == branch_id
        oracle_real_stress[mask_real] = real_experts[branch_id][mask_real]
        oracle_synth_stress[mask_synth] = synth_experts[branch_id][mask_synth]
    oracle_real_metrics = _eval_metrics(real_arrays["stress"], oracle_real_stress, real_arrays["branch_id"], real_arrays["branch_id"])
    oracle_synth_metrics = _eval_metrics(synth_arrays["stress"], oracle_synth_stress, synth_arrays["branch_id"], synth_arrays["branch_id"])
    rows.append({"name": "oracle_reference", "real": oracle_real_metrics, "synthetic": oracle_synth_metrics})

    for gate_name, feature_kind in gate_specs:
        gate_ckpt = output_root / gate_name / "best.pt"
        gate_probs_real = _predict_gate(gate_ckpt, feature_kind, real_arrays["strain_eng"], real_arrays["material_reduced"], args.device)
        gate_probs_val = _predict_gate(gate_ckpt, feature_kind, real_val["strain_eng"], real_val["material_reduced"], args.device)
        gate_probs_synth = _predict_gate(gate_ckpt, feature_kind, synth_arrays["strain_eng"], synth_arrays["material_reduced"], args.device)

        hard_real, hard_branch_real = _ensemble_from_gate(
            gate_probs=gate_probs_real,
            trial_stress=trial_real,
            baseline_stress=baseline_real["stress"],
            expert_stress=real_experts,
            mode="hard",
            threshold=None,
        )
        hard_synth, hard_branch_synth = _ensemble_from_gate(
            gate_probs=gate_probs_synth,
            trial_stress=trial_synth,
            baseline_stress=baseline_synth["stress"],
            expert_stress=synth_experts,
            mode="hard",
            threshold=None,
        )
        hard_real_metrics = _evaluate_gate_mode(
            name=f"{gate_name}_hard",
            stress_true=real_arrays["stress"],
            branch_true=real_arrays["branch_id"],
            stress_pred=hard_real,
            branch_pred=hard_branch_real,
            output_dir=output_root / f"{gate_name}_hard" / "real",
        )
        hard_synth_metrics = _evaluate_gate_mode(
            name=f"{gate_name}_hard",
            stress_true=synth_arrays["stress"],
            branch_true=synth_arrays["branch_id"],
            stress_pred=hard_synth,
            branch_pred=hard_branch_synth,
            output_dir=output_root / f"{gate_name}_hard" / "synthetic",
        )
        rows.append({"name": f"{gate_name}_hard", "real": hard_real_metrics, "synthetic": hard_synth_metrics})

        soft_real, soft_branch_real = _ensemble_from_gate(
            gate_probs=gate_probs_real,
            trial_stress=trial_real,
            baseline_stress=baseline_real["stress"],
            expert_stress=real_experts,
            mode="soft",
            threshold=None,
        )
        soft_synth, soft_branch_synth = _ensemble_from_gate(
            gate_probs=gate_probs_synth,
            trial_stress=trial_synth,
            baseline_stress=baseline_synth["stress"],
            expert_stress=synth_experts,
            mode="soft",
            threshold=None,
        )
        soft_real_metrics = _evaluate_gate_mode(
            name=f"{gate_name}_soft",
            stress_true=real_arrays["stress"],
            branch_true=real_arrays["branch_id"],
            stress_pred=soft_real,
            branch_pred=np.argmax(gate_probs_real, axis=1),
            output_dir=output_root / f"{gate_name}_soft" / "real",
        )
        soft_synth_metrics = _evaluate_gate_mode(
            name=f"{gate_name}_soft",
            stress_true=synth_arrays["stress"],
            branch_true=synth_arrays["branch_id"],
            stress_pred=soft_synth,
            branch_pred=np.argmax(gate_probs_synth, axis=1),
            output_dir=output_root / f"{gate_name}_soft" / "synthetic",
        )
        rows.append({"name": f"{gate_name}_soft", "real": soft_real_metrics, "synthetic": soft_synth_metrics})

        best_threshold = None
        best_val_mae = float("inf")
        for threshold in [0.45, 0.55, 0.65, 0.75, 0.85]:
            thresh_val, thresh_branch_val = _ensemble_from_gate(
                gate_probs=gate_probs_val,
                trial_stress=trial_val,
                baseline_stress=baseline_val["stress"],
                expert_stress=val_experts,
                mode="threshold",
                threshold=threshold,
            )
            val_metrics = _eval_metrics(real_val["stress"], thresh_val, real_val["branch_id"], thresh_branch_val)
            if val_metrics["stress_mae"] < best_val_mae:
                best_val_mae = val_metrics["stress_mae"]
                best_threshold = threshold
        thresh_real, thresh_branch_real = _ensemble_from_gate(
            gate_probs=gate_probs_real,
            trial_stress=trial_real,
            baseline_stress=baseline_real["stress"],
            expert_stress=real_experts,
            mode="threshold",
            threshold=best_threshold,
        )
        thresh_synth, thresh_branch_synth = _ensemble_from_gate(
            gate_probs=gate_probs_synth,
            trial_stress=trial_synth,
            baseline_stress=baseline_synth["stress"],
            expert_stress=synth_experts,
            mode="threshold",
            threshold=best_threshold,
        )
        thresh_real_metrics = _evaluate_gate_mode(
            name=f"{gate_name}_threshold_t{best_threshold:.2f}",
            stress_true=real_arrays["stress"],
            branch_true=real_arrays["branch_id"],
            stress_pred=thresh_real,
            branch_pred=thresh_branch_real,
            output_dir=output_root / f"{gate_name}_threshold_t{best_threshold:.2f}" / "real",
        )
        thresh_synth_metrics = _evaluate_gate_mode(
            name=f"{gate_name}_threshold_t{best_threshold:.2f}",
            stress_true=synth_arrays["stress"],
            branch_true=synth_arrays["branch_id"],
            stress_pred=thresh_synth,
            branch_pred=thresh_branch_synth,
            output_dir=output_root / f"{gate_name}_threshold_t{best_threshold:.2f}" / "synthetic",
        )
        rows.append({"name": f"{gate_name}_threshold_t{best_threshold:.2f}", "real": thresh_real_metrics, "synthetic": thresh_synth_metrics})
        (output_root / f"{gate_name}_threshold_selection.json").write_text(json.dumps({"best_threshold": best_threshold, "val_stress_mae": best_val_mae}, indent=2), encoding="utf-8")

    _plot_mode_compare(rows, output_root / "compare_mae.png", key="stress_mae", title="Gate ensemble MAE")
    _plot_mode_compare(rows, output_root / "compare_rmse.png", key="stress_rmse", title="Gate ensemble RMSE")

    best_row = min(rows, key=lambda row: row["real"]["stress_mae"])
    best_name = best_row["name"]
    if best_name == "baseline_reference":
        best_predictions = baseline_real["stress"]
    elif best_name == "oracle_reference":
        best_predictions = oracle_real_stress
    else:
        if best_name.startswith("gate_raw_"):
            gate_name = "gate_raw"
            mode_name = best_name[len("gate_raw_") :]
            feature_kind = "raw"
        elif best_name.startswith("gate_trial_"):
            gate_name = "gate_trial"
            mode_name = best_name[len("gate_trial_") :]
            feature_kind = "trial"
        else:
            raise ValueError(f"Unexpected best mode name: {best_name}")
        gate_ckpt = output_root / gate_name / "best.pt"
        probs = _predict_gate(gate_ckpt, feature_kind, real_arrays["strain_eng"], real_arrays["material_reduced"], args.device)
        if mode_name == "hard":
            best_predictions, _ = _ensemble_from_gate(
                gate_probs=probs,
                trial_stress=trial_real,
                baseline_stress=baseline_real["stress"],
                expert_stress=real_experts,
                mode="hard",
                threshold=None,
            )
        elif mode_name == "soft":
            best_predictions, _ = _ensemble_from_gate(
                gate_probs=probs,
                trial_stress=trial_real,
                baseline_stress=baseline_real["stress"],
                expert_stress=real_experts,
                mode="soft",
                threshold=None,
            )
        else:
            threshold = float(best_name.rsplit("t", 1)[1])
            best_predictions, _ = _ensemble_from_gate(
                gate_probs=probs,
                trial_stress=trial_real,
                baseline_stress=baseline_real["stress"],
                expert_stress=real_experts,
                mode="threshold",
                threshold=threshold,
            )

    best_dissection = _compute_real_dissection(real_dataset_path=real_dataset, predictions=best_predictions)
    (output_root / "best_real_dissection.json").write_text(json.dumps(_json_safe(best_dissection), indent=2), encoding="utf-8")

    _write_report(report_path, output_root, gate_summaries, rows, best_name, best_dissection)


if __name__ == "__main__":
    main()
