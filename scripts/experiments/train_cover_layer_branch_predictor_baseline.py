from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mc_surrogate.cover_branch_generation import (
    collect_blocks,
    draw_latent,
    fit_latent_mixture,
    fit_pca,
    load_call_regimes,
    load_split_calls,
    pick_calls,
    synthesize_from_latent,
)
from mc_surrogate.models import build_trial_features

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")


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


def _flatten_pointwise(strain: np.ndarray, material: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_elem, n_q, _ = strain.shape
    material_rep = np.repeat(material[:, None, :], n_q, axis=1).reshape(n_elem * n_q, material.shape[1])
    features = build_trial_features(strain.reshape(n_elem * n_q, 6), material_rep)
    labels = branch.reshape(n_elem * n_q).astype(np.int64)
    return features.astype(np.float32), labels


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == labels).float().mean().item())
    recalls = []
    for branch_id in range(len(BRANCH_NAMES)):
        mask = labels == branch_id
        if int(mask.sum().item()) == 0:
            recalls.append(float("nan"))
            continue
        recalls.append(float((pred[mask] == labels[mask]).float().mean().item()))
    macro = float(np.nanmean(recalls))
    out = {"accuracy": acc, "macro_recall": macro}
    for i, name in enumerate(BRANCH_NAMES):
        out[f"recall_{name}"] = recalls[i]
    return out


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(BRANCH_NAMES)).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a first synthetic-only cover-layer branch predictor baseline.")
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
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_baseline_20260314"),
    )
    parser.add_argument("--fit-calls", type=int, default=8)
    parser.add_argument("--val-calls", type=int, default=4)
    parser.add_argument("--test-calls", type=int, default=4)
    parser.add_argument("--max-elements-per-call", type=int, default=96)
    parser.add_argument("--synthetic-elements-per-epoch", type=int, default=1500)
    parser.add_argument("--synthetic-holdout-elements", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--explained-variance", type=float, default=0.995)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_calls(args.split_json)
    regimes = load_call_regimes(args.regime_json)
    fit_calls = pick_calls(splits["generator_fit"], count=args.fit_calls, selection="spread_p95", regimes=regimes)
    val_calls = pick_calls(splits["real_val"], count=args.val_calls, selection="spread_p95", regimes=regimes)
    test_calls = pick_calls(splits["real_test"], count=args.test_calls, selection="spread_p95", regimes=regimes)

    coords_fit, disp_fit, _, _, material_fit = collect_blocks(
        args.export,
        call_names=fit_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed,
    )
    _, _, strain_val, branch_val, material_val = collect_blocks(
        args.export,
        call_names=val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 1,
    )
    _, _, strain_test, branch_test, material_test = collect_blocks(
        args.export,
        call_names=test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 2,
    )

    from mc_surrogate.full_export import canonicalize_p2_element_states

    canonical_fit = canonicalize_p2_element_states(coords_fit, disp_fit)
    disp_fit_flat = canonical_fit.local_displacements.reshape(canonical_fit.local_displacements.shape[0], -1)
    pca = fit_pca(disp_fit_flat, explained_variance=args.explained_variance, max_rank=args.max_rank)
    mixture = fit_latent_mixture(pca["latent"], n_clusters=min(args.clusters, pca["latent"].shape[0]), seed=args.seed)

    x_val_np, y_val_np = _flatten_pointwise(strain_val, material_val, branch_val)
    x_test_np, y_test_np = _flatten_pointwise(strain_test, material_test, branch_test)
    latent_holdout = draw_latent(
        "empirical_local_noise",
        pca=pca,
        mixture=mixture,
        sample_count=args.synthetic_holdout_elements,
        seed=args.seed + 999,
        noise_scale=args.noise_scale,
    )
    strain_syn_holdout, branch_syn_holdout, material_syn_holdout, _ = synthesize_from_latent(
        latent_holdout,
        coords_fit=coords_fit,
        material_fit=material_fit,
        pca=pca,
        seed=args.seed + 1999,
    )
    x_syn_np, y_syn_np = _flatten_pointwise(strain_syn_holdout, material_syn_holdout, branch_syn_holdout)
    x_mean = x_val_np.mean(axis=0)
    x_std = np.where(x_val_np.std(axis=0) < 1.0e-6, 1.0, x_val_np.std(axis=0))

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    x_val = torch.from_numpy(scale(x_val_np)).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)
    x_test = torch.from_numpy(scale(x_test_np)).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    x_syn = torch.from_numpy(scale(x_syn_np)).to(device)
    y_syn = torch.from_numpy(y_syn_np).to(device)

    model = BranchMLP(in_dim=x_val.shape[1], width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["epoch", "train_loss", "val_accuracy", "val_macro_recall", "test_accuracy", "test_macro_recall", "lr"],
        )
        writer.writeheader()

        best_score = -float("inf")
        best_epoch = -1
        best_metrics: dict[str, float] = {}
        best_state = None
        no_improve = 0
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            latent = draw_latent(
                "empirical_local_noise",
                pca=pca,
                mixture=mixture,
                sample_count=args.synthetic_elements_per_epoch,
                seed=args.seed + epoch,
                noise_scale=args.noise_scale,
            )
            strain_syn, branch_syn, material_syn, _ = synthesize_from_latent(
                latent,
                coords_fit=coords_fit,
                material_fit=material_fit,
                pca=pca,
                seed=args.seed + 1000 + epoch,
            )
            x_train_np, y_train_np = _flatten_pointwise(strain_syn, material_syn, branch_syn)
            x_train = torch.from_numpy(scale(x_train_np))
            y_train = torch.from_numpy(y_train_np)

            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            class_weights = _class_weights(y_train_np).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            model.train()
            train_loss = 0.0
            train_count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * xb.shape[0]
                train_count += xb.shape[0]
            train_loss /= max(train_count, 1)

            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                test_logits = model(x_test)
                syn_logits = model(x_syn)
            val_metrics = _metrics(val_logits, y_val)
            test_metrics = _metrics(test_logits, y_test)
            syn_metrics = _metrics(syn_logits, y_syn)
            score = val_metrics["macro_recall"] + 0.5 * val_metrics["accuracy"]
            scheduler.step(score)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_recall": test_metrics["macro_recall"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_metrics = {
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_recall": test_metrics["macro_recall"],
                    "synthetic_accuracy": syn_metrics["accuracy"],
                    "synthetic_macro_recall": syn_metrics["macro_recall"],
                    **{f"val_{k}": v for k, v in val_metrics.items() if k.startswith("recall_")},
                    **{f"test_{k}": v for k, v in test_metrics.items() if k.startswith("recall_")},
                    **{f"synthetic_{k}": v for k, v in syn_metrics.items() if k.startswith("recall_")},
                }
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 20 == 0:
                elapsed = time.time() - start
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} val_macro={val_metrics['macro_recall']:.4f} "
                    f"test_acc={test_metrics['accuracy']:.4f} synth_acc={syn_metrics['accuracy']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"elapsed={elapsed:.1f}s"
                )

            if no_improve >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce any checkpoint.")

    torch.save(
        {
            "state_dict": best_state,
            "x_mean": x_mean,
            "x_std": x_std,
            "branch_names": BRANCH_NAMES,
            "fit_calls": fit_calls,
            "val_calls": val_calls,
            "test_calls": test_calls,
            "input_dim": int(x_val.shape[1]),
            "width": args.width,
            "depth": args.depth,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "synthetic_holdout_elements": args.synthetic_holdout_elements,
        },
        args.output_dir / "best.pt",
    )
    summary = {
        "export_path": str(args.export),
        "fit_calls": fit_calls,
        "val_calls": val_calls,
        "test_calls": test_calls,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "device": str(device),
        "synthetic_elements_per_epoch": args.synthetic_elements_per_epoch,
        "synthetic_holdout_elements": args.synthetic_holdout_elements,
        "max_elements_per_call": args.max_elements_per_call,
        "generator_mode": "empirical_local_noise",
        **best_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "fit_calls": fit_calls,
                "val_calls": val_calls,
                "test_calls": test_calls,
                "fit_calls_count": args.fit_calls,
                "val_calls_count": args.val_calls,
                "test_calls_count": args.test_calls,
                "max_elements_per_call": args.max_elements_per_call,
                "synthetic_elements_per_epoch": args.synthetic_elements_per_epoch,
                "synthetic_holdout_elements": args.synthetic_holdout_elements,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "width": args.width,
                "depth": args.depth,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "explained_variance": args.explained_variance,
                "max_rank": args.max_rank,
                "clusters": args.clusters,
                "noise_scale": args.noise_scale,
                "seed": args.seed,
                "generator_mode": "empirical_local_noise",
                "call_selection": "spread_p95",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
