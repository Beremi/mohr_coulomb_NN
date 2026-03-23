from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mc_surrogate.cover_branch_generation import (
    collect_blocks,
    draw_latent,
    fit_seed_noise_bank,
    fit_latent_mixture,
    fit_pca,
    synthesize_from_latent,
    synthesize_from_seeded_noise,
)
from mc_surrogate.full_export import canonicalize_p2_element_states
from mc_surrogate.models import build_trial_features

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")


class BranchMLP(torch.nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int = 5) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, width), torch.nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(width, width), torch.nn.GELU()])
        layers.append(torch.nn.Linear(width, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flatten_pointwise(strain: np.ndarray, material: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_elem, n_q, _ = strain.shape
    material_rep = np.repeat(material[:, None, :], n_q, axis=1).reshape(n_elem * n_q, material.shape[1])
    features = build_trial_features(strain.reshape(n_elem * n_q, 6), material_rep)
    labels = branch.reshape(n_elem * n_q).astype(np.int64)
    return features.astype(np.float32), labels


def _evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, *, device: torch.device, x_mean: np.ndarray, x_std: np.ndarray) -> dict[str, object]:
    x_scaled = ((x - x_mean) / x_std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    acc = float(np.mean(pred == y))
    recalls = []
    confusion = np.zeros((len(BRANCH_NAMES), len(BRANCH_NAMES)), dtype=np.int64)
    for truth, guess in zip(y, pred, strict=False):
        confusion[int(truth), int(guess)] += 1
    for branch_id in range(len(BRANCH_NAMES)):
        mask = y == branch_id
        if np.sum(mask) == 0:
            recalls.append(float("nan"))
        else:
            recalls.append(float(np.mean(pred[mask] == y[mask])))
    return {
        "accuracy": acc,
        "macro_recall": float(np.nanmean(recalls)),
        "recalls": recalls,
        "confusion": confusion,
        "pred": pred,
        "truth": y,
    }


def _plot_history(history_rows: list[dict[str, float]], output: Path) -> None:
    epoch = np.array([row["global_epoch"] if "global_epoch" in row else row["epoch"] for row in history_rows], dtype=float)
    train_loss = np.array([row["train_loss"] for row in history_rows], dtype=float)
    val_acc = np.array([row["val_accuracy"] for row in history_rows], dtype=float)
    test_acc = np.array([row["test_accuracy"] for row in history_rows], dtype=float)
    syn_acc = np.array([row.get("synthetic_accuracy", np.nan) for row in history_rows], dtype=float)
    val_macro = np.array([row["val_macro_recall"] for row in history_rows], dtype=float)
    test_macro = np.array([row["test_macro_recall"] for row in history_rows], dtype=float)
    syn_macro = np.array([row.get("synthetic_macro_recall", np.nan) for row in history_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(epoch, train_loss, color="#1b5e20", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epoch, val_acc, label="Real val acc", linewidth=2)
    axes[1].plot(epoch, test_acc, label="Real test acc", linewidth=2)
    if not np.all(np.isnan(syn_acc)):
        axes[1].plot(epoch, syn_acc, label="Synthetic holdout acc", linewidth=2)
    axes[1].plot(epoch, val_macro, label="Real val macro", linewidth=2, linestyle="--")
    axes[1].plot(epoch, test_macro, label="Real test macro", linewidth=2, linestyle="--")
    if not np.all(np.isnan(syn_macro)):
        axes[1].plot(epoch, syn_macro, label="Synthetic holdout macro", linewidth=2, linestyle=":")
    axes[1].set_title("Validation Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    if "stage_index" in history_rows[0]:
        last = (history_rows[0]["cycle"], history_rows[0]["stage_index"])
        for row in history_rows[1:]:
            current = (row["cycle"], row["stage_index"])
            if current != last:
                x = row["global_epoch"]
                for ax in axes:
                    ax.axvline(x=x, color="#999999", alpha=0.25, linewidth=1)
                last = current

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_confusions(evals: dict[str, dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, len(evals), figsize=(4.5 * len(evals), 4.2))
    if len(evals) == 1:
        axes = [axes]
    for ax, (name, payload) in zip(axes, evals.items(), strict=False):
        conf = payload["confusion"].astype(np.float64)
        row_sum = np.maximum(conf.sum(axis=1, keepdims=True), 1.0)
        conf_n = conf / row_sum
        im = ax.imshow(conf_n, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        ax.set_title(name)
        ax.set_xticks(range(len(BRANCH_NAMES)), BRANCH_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(BRANCH_NAMES)), BRANCH_NAMES)
        for i in range(conf_n.shape[0]):
            for j in range(conf_n.shape[1]):
                ax.text(j, i, f"{conf_n[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=np.atleast_1d(axes).ravel().tolist(), shrink=0.82)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_recall_bars(evals: dict[str, dict[str, object]], output: Path) -> None:
    x = np.arange(len(BRANCH_NAMES))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for idx, (name, payload) in enumerate(evals.items()):
        ax.bar(x + (idx - (len(evals) - 1) / 2.0) * width, payload["recalls"], width=width, label=name)
    ax.set_xticks(x, BRANCH_NAMES, rotation=20)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Branch Recall by Split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _sample_synthetic_holdout(
    mode: str,
    *,
    coords_fit: np.ndarray,
    disp_fit: np.ndarray,
    branch_fit: np.ndarray,
    material_fit: np.ndarray,
    pca: dict[str, np.ndarray],
    mixture: dict[str, np.ndarray],
    seed_bank: dict[str, np.ndarray | float],
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode in {"seeded_local_noise_uniform", "seeded_local_noise_branch_balanced"}:
        selection = "branch_balanced" if mode.endswith("branch_balanced") else "uniform"
        strain_syn, branch_syn, material_syn, _ = synthesize_from_seeded_noise(
            seed_bank,
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
            selection=selection,
        )
        return strain_syn, branch_syn, material_syn

    latent_holdout = draw_latent(
        mode,
        pca=pca,
        mixture=mixture,
        sample_count=sample_count,
        seed=seed,
        noise_scale=noise_scale,
    )
    strain_syn, branch_syn, material_syn, _ = synthesize_from_latent(
        latent_holdout,
        coords_fit=coords_fit,
        material_fit=material_fit,
        pca=pca,
        seed=seed + 1000,
    )
    return strain_syn, branch_syn, material_syn


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown model card for a cover-layer branch-predictor run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_baseline_20260314"),
    )
    parser.add_argument(
        "--card-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_model_card_20260314"),
    )
    parser.add_argument(
        "--card-md",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_model_card.md"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Cover Layer Branch Predictor Model Card",
    )
    args = parser.parse_args()

    args.card_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((args.run_dir / "train_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)

    with (args.run_dir / "history.csv").open("r", encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                parsed[k] = float(v)
            rows.append(parsed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BranchMLP(in_dim=int(checkpoint["input_dim"]), width=int(checkpoint["width"]), depth=int(checkpoint["depth"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    export = Path(summary["export_path"])
    fit_calls = summary["fit_calls"]
    val_calls = summary["val_calls"]
    test_calls = summary["test_calls"]

    coords_fit, disp_fit, _, branch_fit, material_fit = collect_blocks(
        export,
        call_names=fit_calls,
        max_elements_per_call=int(summary["max_elements_per_call"]),
        seed=int(config["seed"]),
    )
    _, _, strain_val, branch_val, material_val = collect_blocks(
        export,
        call_names=val_calls,
        max_elements_per_call=int(summary["max_elements_per_call"]),
        seed=int(config["seed"]) + 1,
    )
    _, _, strain_test, branch_test, material_test = collect_blocks(
        export,
        call_names=test_calls,
        max_elements_per_call=int(summary["max_elements_per_call"]),
        seed=int(config["seed"]) + 2,
    )

    canonical_fit = canonicalize_p2_element_states(coords_fit, disp_fit)
    disp_fit_flat = canonical_fit.local_displacements.reshape(canonical_fit.local_displacements.shape[0], -1)
    pca = fit_pca(disp_fit_flat, explained_variance=float(config["explained_variance"]), max_rank=int(config["max_rank"]))
    mixture = fit_latent_mixture(pca["latent"], n_clusters=min(int(config["clusters"]), pca["latent"].shape[0]), seed=int(config["seed"]))
    seed_bank = fit_seed_noise_bank(
        coords_fit,
        disp_fit,
        branch_fit,
        material_fit,
    )
    strain_syn, branch_syn, material_syn = _sample_synthetic_holdout(
        summary["generator_mode"],
        coords_fit=coords_fit,
        disp_fit=disp_fit,
        branch_fit=branch_fit,
        material_fit=material_fit,
        pca=pca,
        mixture=mixture,
        seed_bank=seed_bank,
        sample_count=int(summary["synthetic_holdout_elements"]),
        seed=int(config["seed"]) + 999,
        noise_scale=float(config["noise_scale"]),
    )

    x_val, y_val = _flatten_pointwise(strain_val, material_val, branch_val)
    x_test, y_test = _flatten_pointwise(strain_test, material_test, branch_test)
    x_syn, y_syn = _flatten_pointwise(strain_syn, material_syn, branch_syn)
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]

    evals = {
        "Real val": _evaluate(model, x_val, y_val, device=device, x_mean=x_mean, x_std=x_std),
        "Real test": _evaluate(model, x_test, y_test, device=device, x_mean=x_mean, x_std=x_std),
        "Synthetic holdout": _evaluate(model, x_syn, y_syn, device=device, x_mean=x_mean, x_std=x_std),
    }

    history_png = args.card_dir / "training_history.png"
    confusion_png = args.card_dir / "confusions.png"
    recall_png = args.card_dir / "recalls.png"
    _plot_history(rows, history_png)
    _plot_confusions(evals, confusion_png)
    _plot_recall_bars(evals, recall_png)

    n_params = sum(p.numel() for p in model.parameters())
    stage_text = ""
    if "cycles" in config:
        stage_text = (
            f"- cycles: `{config['cycles']}`\n"
            f"- batch-size schedule per cycle: `{config['batch_sizes']}`\n"
            f"- stage max epochs: `{config['stage_max_epochs']}`\n"
            f"- plateau patience/factor: `{config['plateau_patience']}` / `{config['plateau_factor']}`\n"
        )

    md = f"""# {args.title}

## Summary

This card describes the current **synthetic-only** cover-layer branch predictor run.

- checkpoint: `{args.run_dir / 'best.pt'}`
- material scope: `cover_layer` only
- task: predict one of `elastic / smooth / left_edge / right_edge / apex`
- prediction granularity: pointwise, one prediction per integration point
- deployment intent: use exact FE kinematics to compute `E`, then run the classifier

## Architecture

- model type: shared pointwise MLP
- input dimension: `{int(checkpoint["input_dim"])}`
- hidden width: `{int(checkpoint["width"])}`
- depth: `{int(checkpoint["depth"])}`
- output classes: `5`
- activation: `GELU`
- parameter count: `{n_params:,}`

Input features are built with `build_trial_features(...)`:

- `asinh(strain_eng)` -> `6`
- `asinh(trial_stress / c_bar)` -> `6`
- reduced material features -> `5`

Total: `17` features per integration point.

## Training Setup

- training data: synthetic only
- synthetic generator mode: `{summary["generator_mode"]}`
- synthetic elements per epoch: `{summary["synthetic_elements_per_epoch"]}`
- synthetic holdout elements: `{summary["synthetic_holdout_elements"]}`
- real fit calls used to fit the synthetic generator: `{len(fit_calls)}`
- real validation calls: `{len(val_calls)}`
- real test calls: `{len(test_calls)}`
- max elements per call in this run: `{summary["max_elements_per_call"]}`
- optimizer: `AdamW`
- base learning rate: `{config["lr"]}`
- weight decay: `{config["weight_decay"]}`
{stage_text}- best epoch: `{summary["best_epoch"]}`

This run regenerates synthetic training data throughout training and never uses real labels for optimization.

## Convergence

![Training History](../{history_png.as_posix()})

What to look at:

- training loss should keep decreasing without the early collapse we saw in the short baseline
- real validation, real test, and synthetic holdout curves are plotted together
- if stage markers are present, they show batch-size transitions across cycles

## Accuracy

| Split | Accuracy | Macro Recall |
|---|---:|---:|
| Real val | {evals["Real val"]["accuracy"]:.4f} | {evals["Real val"]["macro_recall"]:.4f} |
| Real test | {evals["Real test"]["accuracy"]:.4f} | {evals["Real test"]["macro_recall"]:.4f} |
| Synthetic holdout | {evals["Synthetic holdout"]["accuracy"]:.4f} | {evals["Synthetic holdout"]["macro_recall"]:.4f} |

So the direct answer to the synthetic-test question is:

- synthetic holdout accuracy: `{evals["Synthetic holdout"]["accuracy"]:.4f}`
- synthetic holdout macro recall: `{evals["Synthetic holdout"]["macro_recall"]:.4f}`

## Per-Branch Recall

![Per-Branch Recall](../{recall_png.as_posix()})

| Branch | Real val | Real test | Synthetic holdout |
|---|---:|---:|---:|
| elastic | {evals["Real val"]["recalls"][0]:.4f} | {evals["Real test"]["recalls"][0]:.4f} | {evals["Synthetic holdout"]["recalls"][0]:.4f} |
| smooth | {evals["Real val"]["recalls"][1]:.4f} | {evals["Real test"]["recalls"][1]:.4f} | {evals["Synthetic holdout"]["recalls"][1]:.4f} |
| left_edge | {evals["Real val"]["recalls"][2]:.4f} | {evals["Real test"]["recalls"][2]:.4f} | {evals["Synthetic holdout"]["recalls"][2]:.4f} |
| right_edge | {evals["Real val"]["recalls"][3]:.4f} | {evals["Real test"]["recalls"][3]:.4f} | {evals["Synthetic holdout"]["recalls"][3]:.4f} |
| apex | {evals["Real val"]["recalls"][4]:.4f} | {evals["Real test"]["recalls"][4]:.4f} | {evals["Synthetic holdout"]["recalls"][4]:.4f} |

## Confusion Structure

![Confusion Matrices](../{confusion_png.as_posix()})

Key interpretation:

- if real and synthetic confusion look similar, the bottleneck is likely still the synthetic generator
- if synthetic is good but real is poor, the bottleneck is transfer / coverage mismatch
- for this task, `smooth`, `left_edge`, and `right_edge` are the branches to watch

## Current Assessment

This model should be judged by hard-branch recall and macro recall, not overall accuracy alone.
"""
    args.card_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.card_md}")


if __name__ == "__main__":
    main()
