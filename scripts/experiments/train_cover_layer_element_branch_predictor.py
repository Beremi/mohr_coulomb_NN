from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mc_surrogate.cover_branch_generation import (
    collect_blocks,
    draw_latent,
    fit_latent_mixture,
    fit_pca,
    fit_seed_noise_bank,
    load_call_regimes,
    load_split_calls,
    pick_calls,
    synthesize_element_states_from_latent,
    synthesize_element_states_from_seeded_noise,
)
from mc_surrogate.full_export import canonicalize_p2_element_states

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


def _class_weights(labels: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    counts = np.bincount(labels.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)
    loss_weights = 1.0 / np.maximum(counts, 1.0)
    loss_weights = loss_weights / np.mean(loss_weights)
    elem_weights = np.mean(loss_weights[labels], axis=1)
    return torch.tensor(loss_weights, dtype=torch.float32), elem_weights.astype(np.float64)


def _score(metrics: dict[str, float]) -> float:
    edge_mean = 0.5 * (metrics["recall_left_edge"] + metrics["recall_right_edge"])
    return float(
        metrics["macro_recall"]
        + 0.20 * metrics["accuracy"]
        + 0.20 * metrics["pattern_accuracy"]
        + 0.20 * edge_mean
    )


def _build_generator(
    export: Path,
    *,
    fit_calls: list[str],
    max_elements_per_call: int,
    fit_call_selection: str,
    regimes: dict[str, dict[str, float]],
    fit_calls_count: int,
    explained_variance: float,
    max_rank: int,
    clusters: int,
    seed: int,
) -> dict[str, object]:
    selected_fit_calls = pick_calls(fit_calls, count=fit_calls_count, selection=fit_call_selection, regimes=regimes)
    coords_fit, disp_fit, _, branch_fit, material_fit = collect_blocks(
        export,
        call_names=selected_fit_calls,
        max_elements_per_call=max_elements_per_call,
        seed=seed,
    )
    canonical_fit = canonicalize_p2_element_states(coords_fit, disp_fit)
    disp_fit_flat = canonical_fit.local_displacements.reshape(canonical_fit.local_displacements.shape[0], -1)
    pca = fit_pca(disp_fit_flat, explained_variance=explained_variance, max_rank=max_rank)
    mixture = fit_latent_mixture(pca["latent"], n_clusters=min(clusters, pca["latent"].shape[0]), seed=seed)
    return {
        "fit_calls": selected_fit_calls,
        "coords_fit": coords_fit,
        "disp_fit": disp_fit,
        "material_fit": material_fit,
        "branch_fit": branch_fit,
        "pca": pca,
        "mixture": mixture,
        "seed_bank": fit_seed_noise_bank(coords_fit, disp_fit, branch_fit, material_fit),
    }


def _sample_synthetic_elements(
    generator: dict[str, object],
    *,
    mode: str,
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode in {"seeded_local_noise_uniform", "seeded_local_noise_branch_balanced"}:
        selection = "uniform" if mode.endswith("uniform") else "branch_balanced"
        coords, disp, _, branch, _material, _valid = synthesize_element_states_from_seeded_noise(
            generator["seed_bank"],
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
            selection=selection,
        )
        return coords, disp, branch

    latent = draw_latent(
        mode,
        pca=generator["pca"],
        mixture=generator["mixture"],
        sample_count=sample_count,
        seed=seed,
        noise_scale=noise_scale,
    )
    coords, disp, _, branch, _material, _valid = synthesize_element_states_from_latent(
        latent,
        coords_fit=generator["coords_fit"],
        material_fit=generator["material_fit"],
        pca=generator["pca"],
        seed=seed + 1000,
    )
    return coords, disp, branch


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-only element-level cover-layer branch predictor.")
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
        default=Path("experiment_runs/real_sim/cover_layer_element_branch_predictor_20260314"),
    )
    parser.add_argument("--fit-calls", type=int, default=12)
    parser.add_argument("--val-calls", type=int, default=4)
    parser.add_argument("--test-calls", type=int, default=4)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--fit-call-selection", choices=("first", "spread_p95"), default="spread_p95")
    parser.add_argument("--generator-fit-elements-per-cycle", type=int, default=128)
    parser.add_argument("--synthetic-elements-per-epoch", type=int, default=2500)
    parser.add_argument("--synthetic-holdout-elements", type=int, default=1500)
    parser.add_argument(
        "--generator-mode",
        choices=("empirical_local_noise", "seeded_local_noise_uniform", "seeded_local_noise_branch_balanced"),
        default="seeded_local_noise_branch_balanced",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--batch-sizes", type=str, default="32,64,128,256,512")
    parser.add_argument("--stage-max-epochs", type=int, default=45)
    parser.add_argument("--stage-patience", type=int, default=14)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--explained-variance", type=float, default=0.995)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_calls(args.split_json)
    regimes = load_call_regimes(args.regime_json)
    all_fit_calls = pick_calls(splits["generator_fit"], count=args.fit_calls, selection=args.fit_call_selection, regimes=regimes)
    val_calls = pick_calls(splits["real_val"], count=args.val_calls, selection="spread_p95", regimes=regimes)
    test_calls = pick_calls(splits["real_test"], count=args.test_calls, selection="spread_p95", regimes=regimes)

    coords_val, disp_val, _, branch_val, _ = collect_blocks(
        args.export,
        call_names=val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 1,
    )
    coords_test, disp_test, _, branch_test, _ = collect_blocks(
        args.export,
        call_names=test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 2,
    )

    gen0 = _build_generator(
        args.export,
        fit_calls=all_fit_calls,
        max_elements_per_call=args.generator_fit_elements_per_cycle,
        fit_call_selection="first",
        regimes=regimes,
        fit_calls_count=len(all_fit_calls),
        explained_variance=args.explained_variance,
        max_rank=args.max_rank,
        clusters=args.clusters,
        seed=args.seed,
    )
    coords_calib, disp_calib, branch_calib = _sample_synthetic_elements(
        gen0,
        mode=args.generator_mode,
        sample_count=max(1000, args.synthetic_holdout_elements),
        seed=args.seed + 77,
        noise_scale=args.noise_scale,
    )
    x_calib = _element_features(coords_calib, disp_calib)
    x_mean = x_calib.mean(axis=0)
    x_std = np.where(x_calib.std(axis=0) < 1.0e-6, 1.0, x_calib.std(axis=0))

    coords_syn_holdout, disp_syn_holdout, branch_syn_holdout = _sample_synthetic_elements(
        gen0,
        mode=args.generator_mode,
        sample_count=args.synthetic_holdout_elements,
        seed=args.seed + 999,
        noise_scale=args.noise_scale,
    )

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    x_val_np = _element_features(coords_val, disp_val)
    x_test_np = _element_features(coords_test, disp_test)
    x_syn_np = _element_features(coords_syn_holdout, disp_syn_holdout)
    y_val_np = branch_val.astype(np.int64)
    y_test_np = branch_test.astype(np.int64)
    y_syn_np = branch_syn_holdout.astype(np.int64)

    x_val = torch.from_numpy(scale(x_val_np)).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)
    x_test = torch.from_numpy(scale(x_test_np)).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    x_syn = torch.from_numpy(scale(x_syn_np)).to(device)
    y_syn = torch.from_numpy(y_syn_np).to(device)

    model = ElementBranchMLP(in_dim=x_val.shape[1], width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "global_epoch",
                "cycle",
                "stage_index",
                "batch_size",
                "train_loss",
                "val_accuracy",
                "val_macro_recall",
                "val_pattern_accuracy",
                "test_accuracy",
                "test_macro_recall",
                "test_pattern_accuracy",
                "synthetic_accuracy",
                "synthetic_macro_recall",
                "synthetic_pattern_accuracy",
                "lr",
            ],
        )
        writer.writeheader()

        best_score = -float("inf")
        best_epoch = -1
        best_metrics: dict[str, float] = {}
        best_state: dict[str, torch.Tensor] | None = None
        best_cycle_details: dict[str, int | float | str] = {}
        start = time.time()
        global_epoch = 0

        for cycle in range(1, args.cycles + 1):
            cycle_seed = args.seed + 10000 * cycle
            generator = _build_generator(
                args.export,
                fit_calls=all_fit_calls,
                max_elements_per_call=args.generator_fit_elements_per_cycle,
                fit_call_selection="spread_p95",
                regimes=regimes,
                fit_calls_count=len(all_fit_calls),
                explained_variance=args.explained_variance,
                max_rank=args.max_rank,
                clusters=args.clusters,
                seed=cycle_seed,
            )
            base_lr_cycle = max(args.min_lr, args.lr * (args.plateau_factor ** (cycle - 1)))
            for g in optimizer.param_groups:
                g["lr"] = base_lr_cycle

            print(f"[cycle-start] cycle={cycle}/{args.cycles} base_lr={base_lr_cycle:.2e} fit_calls={len(generator['fit_calls'])}")

            for stage_index, batch_size in enumerate(batch_sizes, start=1):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=args.plateau_factor,
                    patience=args.plateau_patience,
                    min_lr=args.min_lr,
                )
                stage_best = -float("inf")
                stage_no_improve = 0
                stage_epoch = 0

                print(
                    f"[stage-start] cycle={cycle}/{args.cycles} stage={stage_index}/{len(batch_sizes)} "
                    f"batch={batch_size} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                while stage_epoch < args.stage_max_epochs and stage_no_improve < args.stage_patience:
                    stage_epoch += 1
                    global_epoch += 1

                    coords_syn, disp_syn, branch_syn = _sample_synthetic_elements(
                        generator,
                        mode=args.generator_mode,
                        sample_count=args.synthetic_elements_per_epoch,
                        seed=cycle_seed + global_epoch,
                        noise_scale=args.noise_scale,
                    )
                    x_train_np = _element_features(coords_syn, disp_syn)
                    y_train_np = branch_syn.astype(np.int64)
                    x_train = torch.from_numpy(scale(x_train_np))
                    y_train = torch.from_numpy(y_train_np)
                    dataset = TensorDataset(x_train, y_train)
                    loss_weights, elem_weights = _class_weights(y_train_np)
                    sampler = WeightedRandomSampler(elem_weights, num_samples=len(elem_weights), replacement=True)
                    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
                    loss_fn = nn.CrossEntropyLoss(weight=loss_weights.to(device))

                    model.train()
                    train_loss = 0.0
                    train_count = 0
                    for xb, yb in loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(xb)
                        loss = loss_fn(logits.reshape(-1, len(BRANCH_NAMES)), yb.reshape(-1))
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
                    score = _score(val_metrics)
                    scheduler.step(score)

                    writer.writerow(
                        {
                            "global_epoch": global_epoch,
                            "cycle": cycle,
                            "stage_index": stage_index,
                            "batch_size": batch_size,
                            "train_loss": train_loss,
                            "val_accuracy": val_metrics["accuracy"],
                            "val_macro_recall": val_metrics["macro_recall"],
                            "val_pattern_accuracy": val_metrics["pattern_accuracy"],
                            "test_accuracy": test_metrics["accuracy"],
                            "test_macro_recall": test_metrics["macro_recall"],
                            "test_pattern_accuracy": test_metrics["pattern_accuracy"],
                            "synthetic_accuracy": syn_metrics["accuracy"],
                            "synthetic_macro_recall": syn_metrics["macro_recall"],
                            "synthetic_pattern_accuracy": syn_metrics["pattern_accuracy"],
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

                    if score > best_score:
                        best_score = score
                        best_epoch = global_epoch
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        best_metrics = {
                            "val_accuracy": val_metrics["accuracy"],
                            "val_macro_recall": val_metrics["macro_recall"],
                            "val_pattern_accuracy": val_metrics["pattern_accuracy"],
                            "test_accuracy": test_metrics["accuracy"],
                            "test_macro_recall": test_metrics["macro_recall"],
                            "test_pattern_accuracy": test_metrics["pattern_accuracy"],
                            "synthetic_accuracy": syn_metrics["accuracy"],
                            "synthetic_macro_recall": syn_metrics["macro_recall"],
                            "synthetic_pattern_accuracy": syn_metrics["pattern_accuracy"],
                            **{f"val_{k}": v for k, v in val_metrics.items() if k.startswith("recall_")},
                            **{f"test_{k}": v for k, v in test_metrics.items() if k.startswith("recall_")},
                            **{f"synthetic_{k}": v for k, v in syn_metrics.items() if k.startswith("recall_")},
                        }
                        best_cycle_details = {
                            "cycle": cycle,
                            "stage_index": stage_index,
                            "batch_size": batch_size,
                            "lr": optimizer.param_groups[0]["lr"],
                        }

                    if score > stage_best:
                        stage_best = score
                        stage_no_improve = 0
                    else:
                        stage_no_improve += 1

                    if global_epoch == 1 or global_epoch % 25 == 0:
                        elapsed = time.time() - start
                        print(
                            f"[epoch {global_epoch:04d}] cycle={cycle}/{args.cycles} stage={stage_index}/{len(batch_sizes)} "
                            f"batch={batch_size} lr={optimizer.param_groups[0]['lr']:.2e} runtime={elapsed:.1f}s "
                            f"train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                            f"val_macro={val_metrics['macro_recall']:.4f} val_pattern={val_metrics['pattern_accuracy']:.4f} "
                            f"test_acc={test_metrics['accuracy']:.4f} test_macro={test_metrics['macro_recall']:.4f} "
                            f"test_pattern={test_metrics['pattern_accuracy']:.4f} synth_acc={syn_metrics['accuracy']:.4f} "
                            f"synth_macro={syn_metrics['macro_recall']:.4f} synth_pattern={syn_metrics['pattern_accuracy']:.4f}"
                        )

        if best_state is None:
            raise RuntimeError("No checkpoint was produced.")

    checkpoint = {
        "state_dict": best_state,
        "x_mean": x_mean,
        "x_std": x_std,
        "branch_names": BRANCH_NAMES,
        "fit_calls": all_fit_calls,
        "val_calls": val_calls,
        "test_calls": test_calls,
        "input_dim": int(x_val.shape[1]),
        "width": args.width,
        "depth": args.depth,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "cycles": args.cycles,
        "batch_sizes": batch_sizes,
        "stage_max_epochs": args.stage_max_epochs,
        "stage_patience": args.stage_patience,
        "plateau_patience": args.plateau_patience,
        "plateau_factor": args.plateau_factor,
        "generator_mode": args.generator_mode,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")

    summary = {
        "export_path": str(args.export),
        "fit_calls": all_fit_calls,
        "val_calls": val_calls,
        "test_calls": test_calls,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "generator_mode": args.generator_mode,
        "synthetic_elements_per_epoch": args.synthetic_elements_per_epoch,
        "synthetic_holdout_elements": args.synthetic_holdout_elements,
        "max_elements_per_call": args.max_elements_per_call,
        "generator_fit_elements_per_cycle": args.generator_fit_elements_per_cycle,
        **best_cycle_details,
        **best_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "train_config.json").write_text(
        json.dumps(
            {
                "split_json": str(args.split_json),
                "regime_json": str(args.regime_json),
                "fit_calls_count": args.fit_calls,
                "val_calls_count": args.val_calls,
                "test_calls_count": args.test_calls,
                "fit_call_selection": args.fit_call_selection,
                "max_elements_per_call": args.max_elements_per_call,
                "generator_fit_elements_per_cycle": args.generator_fit_elements_per_cycle,
                "synthetic_elements_per_epoch": args.synthetic_elements_per_epoch,
                "synthetic_holdout_elements": args.synthetic_holdout_elements,
                "cycles": args.cycles,
                "batch_sizes": batch_sizes,
                "stage_max_epochs": args.stage_max_epochs,
                "stage_patience": args.stage_patience,
                "plateau_patience": args.plateau_patience,
                "plateau_factor": args.plateau_factor,
                "min_lr": args.min_lr,
                "width": args.width,
                "depth": args.depth,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "explained_variance": args.explained_variance,
                "max_rank": args.max_rank,
                "clusters": args.clusters,
                "noise_scale": args.noise_scale,
                "seed": args.seed,
                "generator_mode": args.generator_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
