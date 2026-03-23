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
    synthesize_from_seeded_noise,
    synthesize_from_latent,
)
from mc_surrogate.full_export import canonicalize_p2_element_states
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


def _build_pointwise_features(strain: np.ndarray, material: np.ndarray) -> np.ndarray:
    n_elem, n_q, _ = strain.shape
    material_rep = np.repeat(material[:, None, :], n_q, axis=1).reshape(n_elem * n_q, material.shape[1])
    features = build_trial_features(strain.reshape(n_elem * n_q, 6), material_rep)
    return features.astype(np.float32)


def _flatten_pointwise(strain: np.ndarray, material: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_elem, n_q, _ = strain.shape
    features = _build_pointwise_features(strain, material)
    labels = branch.reshape(n_elem * n_q).astype(np.int64)
    return features, labels


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == labels).float().mean().item())
    recalls = []
    for branch_id in range(len(BRANCH_NAMES)):
        mask = labels == branch_id
        if int(mask.sum().item()) == 0:
            recalls.append(float("nan"))
        else:
            recalls.append(float((pred[mask] == labels[mask]).float().mean().item()))
    out = {
        "accuracy": acc,
        "macro_recall": float(np.nanmean(recalls)),
        "recall_elastic": recalls[0],
        "recall_smooth": recalls[1],
        "recall_left_edge": recalls[2],
        "recall_right_edge": recalls[3],
        "recall_apex": recalls[4],
    }
    return out


def _class_weights(labels: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    counts = np.bincount(labels, minlength=len(BRANCH_NAMES)).astype(np.float64)
    loss_weights = 1.0 / np.maximum(counts, 1.0)
    loss_weights = loss_weights / np.mean(loss_weights)
    sample_weights = loss_weights[labels]
    return torch.tensor(loss_weights, dtype=torch.float32), sample_weights.astype(np.float64)


def _score(metrics: dict[str, float]) -> float:
    edge_mean = 0.5 * (metrics["recall_left_edge"] + metrics["recall_right_edge"])
    return float(metrics["macro_recall"] + 0.20 * metrics["accuracy"] + 0.25 * edge_mean)


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
        "material_fit": material_fit,
        "branch_fit": branch_fit,
        "pca": pca,
        "mixture": mixture,
        "seed_bank": fit_seed_noise_bank(coords_fit, disp_fit, branch_fit, material_fit),
    }


def _sample_synthetic_batch(
    generator: dict[str, object],
    *,
    mode: str,
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "seeded_smooth_fail_mixture":
        focus_fraction = float(generator["focus_mix_fraction"])
        focus_noise_multiplier = float(generator["focus_noise_multiplier"])
        focus_count = int(round(sample_count * focus_fraction))
        base_count = max(0, sample_count - focus_count)

        batches = []
        if base_count > 0:
            batches.append(
                synthesize_from_seeded_noise(
                    generator["seed_bank"],
                    sample_count=base_count,
                    seed=seed,
                    noise_scale=noise_scale,
                    selection="branch_balanced",
                )[:3]
            )
        if focus_count > 0:
            batches.append(
                synthesize_from_seeded_noise(
                    generator["focus_seed_bank"],
                    sample_count=focus_count,
                    seed=seed + 700000,
                    noise_scale=noise_scale * focus_noise_multiplier,
                    selection="branch_balanced",
                )[:3]
            )
        if not batches:
            raise RuntimeError("No synthetic batches were generated.")
        if len(batches) == 1:
            return batches[0]
        strain = np.concatenate([row[0] for row in batches], axis=0)
        branch = np.concatenate([row[1] for row in batches], axis=0)
        material = np.concatenate([row[2] for row in batches], axis=0)
        return strain, branch, material

    if mode in {"seeded_local_noise_uniform", "seeded_local_noise_branch_balanced"}:
        selection = "uniform" if mode.endswith("uniform") else "branch_balanced"
        strain, branch, material, _ = synthesize_from_seeded_noise(
            generator["seed_bank"],
            sample_count=sample_count,
            seed=seed,
            noise_scale=noise_scale,
            selection=selection,
        )
        return strain, branch, material

    latent = draw_latent(
        mode,
        pca=generator["pca"],
        mixture=generator["mixture"],
        sample_count=sample_count,
        seed=seed,
        noise_scale=noise_scale,
    )
    strain, branch, material, _ = synthesize_from_latent(
        latent,
        coords_fit=generator["coords_fit"],
        material_fit=generator["material_fit"],
        pca=generator["pca"],
        seed=seed + 1000,
    )
    return strain, branch, material


def _load_bootstrap_model(run_dir: Path, *, device: torch.device) -> tuple[BranchMLP, dict[str, object]]:
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = BranchMLP(
        in_dim=int(checkpoint["input_dim"]),
        width=int(checkpoint["width"]),
        depth=int(checkpoint["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _predict_element_branches(
    model: BranchMLP,
    checkpoint: dict[str, object],
    *,
    strain: np.ndarray,
    material: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = _build_pointwise_features(strain, material)
    x_scaled = ((x - checkpoint["x_mean"]) / checkpoint["x_std"]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    return pred.reshape(strain.shape[0], strain.shape[1]).astype(np.int64)


def _build_focus_seed_bank(
    export: Path,
    *,
    call_names: list[str],
    max_elements_per_call: int,
    seed: int,
    bootstrap_run_dir: Path,
    focus_branch: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, int | str]]:
    coords, disp, strain, branch, material = collect_blocks(
        export,
        call_names=call_names,
        max_elements_per_call=max_elements_per_call,
        seed=seed,
    )
    model, checkpoint = _load_bootstrap_model(bootstrap_run_dir, device=device)
    pred = _predict_element_branches(model, checkpoint, strain=strain, material=material, device=device)
    focus_branch_id = BRANCH_NAMES.index(focus_branch)
    contains_focus = np.any(branch == focus_branch_id, axis=1)
    focus_fail = contains_focus & np.any((pred != branch) & (branch == focus_branch_id), axis=1)
    if not np.any(focus_fail):
        focus_fail = contains_focus
    bank = fit_seed_noise_bank(coords[focus_fail], disp[focus_fail], branch[focus_fail], material[focus_fail])
    summary = {
        "focus_source_calls": len(call_names),
        "focus_total_elements": int(coords.shape[0]),
        "focus_branch": focus_branch,
        "focus_contains_branch_elements": int(np.sum(contains_focus)),
        "focus_failed_elements": int(np.sum(focus_fail)),
    }
    return bank, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Long staged synthetic-only cover-layer branch-predictor training.")
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
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_staged_20260314"),
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
        choices=(
            "empirical_local_noise",
            "seeded_local_noise_uniform",
            "seeded_local_noise_branch_balanced",
            "seeded_smooth_fail_mixture",
        ),
        default="empirical_local_noise",
    )
    parser.add_argument(
        "--bootstrap-run-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_staged_seeded_balanced_20260314"),
    )
    parser.add_argument("--focus-source", choices=("real_val",), default="real_val")
    parser.add_argument("--focus-branch", type=str, default="smooth")
    parser.add_argument("--focus-mix-fraction", type=float, default=0.45)
    parser.add_argument("--focus-noise-multiplier", type=float, default=1.15)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--batch-sizes", type=str, default="64,128,256,512,1024")
    parser.add_argument("--stage-max-epochs", type=int, default=45)
    parser.add_argument("--stage-patience", type=int, default=14)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--width", type=int, default=512)
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

    focus_seed_bank = None
    focus_summary: dict[str, int | str | float] = {}
    if args.generator_mode == "seeded_smooth_fail_mixture":
        focus_calls = val_calls
        focus_seed_bank, focus_summary = _build_focus_seed_bank(
            args.export,
            call_names=focus_calls,
            max_elements_per_call=args.max_elements_per_call,
            seed=args.seed + 17,
            bootstrap_run_dir=args.bootstrap_run_dir,
            focus_branch=args.focus_branch,
            device=device,
        )
        focus_summary.update(
            {
                "focus_source": args.focus_source,
                "focus_mix_fraction": args.focus_mix_fraction,
                "focus_noise_multiplier": args.focus_noise_multiplier,
                "bootstrap_run_dir": str(args.bootstrap_run_dir),
            }
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
    if focus_seed_bank is not None:
        gen0["focus_seed_bank"] = focus_seed_bank
        gen0["focus_mix_fraction"] = args.focus_mix_fraction
        gen0["focus_noise_multiplier"] = args.focus_noise_multiplier
    strain_calib, branch_calib, material_calib = _sample_synthetic_batch(
        gen0,
        mode=args.generator_mode,
        sample_count=max(1000, args.synthetic_holdout_elements),
        seed=args.seed + 77,
        noise_scale=args.noise_scale,
    )
    x_calib, _ = _flatten_pointwise(strain_calib, material_calib, branch_calib)
    x_mean = x_calib.mean(axis=0)
    x_std = np.where(x_calib.std(axis=0) < 1.0e-6, 1.0, x_calib.std(axis=0))

    strain_syn_holdout, branch_syn_holdout, material_syn_holdout = _sample_synthetic_batch(
        gen0,
        mode=args.generator_mode,
        sample_count=args.synthetic_holdout_elements,
        seed=args.seed + 999,
        noise_scale=args.noise_scale,
    )

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    x_val_np, y_val_np = _flatten_pointwise(strain_val, material_val, branch_val)
    x_test_np, y_test_np = _flatten_pointwise(strain_test, material_test, branch_test)
    x_syn_np, y_syn_np = _flatten_pointwise(strain_syn_holdout, material_syn_holdout, branch_syn_holdout)

    x_val = torch.from_numpy(scale(x_val_np)).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)
    x_test = torch.from_numpy(scale(x_test_np)).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    x_syn = torch.from_numpy(scale(x_syn_np)).to(device)
    y_syn = torch.from_numpy(y_syn_np).to(device)

    model = BranchMLP(in_dim=x_val.shape[1], width=args.width, depth=args.depth).to(device)
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
                "test_accuracy",
                "test_macro_recall",
                "synthetic_accuracy",
                "synthetic_macro_recall",
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
            if focus_seed_bank is not None:
                generator["focus_seed_bank"] = focus_seed_bank
                generator["focus_mix_fraction"] = args.focus_mix_fraction
                generator["focus_noise_multiplier"] = args.focus_noise_multiplier
            base_lr_cycle = max(args.min_lr, args.lr * (args.plateau_factor ** (cycle - 1)))
            for g in optimizer.param_groups:
                g["lr"] = base_lr_cycle

            if focus_summary:
                print(
                    f"[cycle-start] cycle={cycle}/{args.cycles} base_lr={base_lr_cycle:.2e} "
                    f"fit_calls={len(generator['fit_calls'])} focus_failed={focus_summary['focus_failed_elements']}"
                )
            else:
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

                    strain_syn, branch_syn, material_syn = _sample_synthetic_batch(
                        generator,
                        mode=args.generator_mode,
                        sample_count=args.synthetic_elements_per_epoch,
                        seed=cycle_seed + global_epoch,
                        noise_scale=args.noise_scale,
                    )
                    x_train_np, y_train_np = _flatten_pointwise(strain_syn, material_syn, branch_syn)
                    x_train = torch.from_numpy(scale(x_train_np))
                    y_train = torch.from_numpy(y_train_np)
                    dataset = TensorDataset(x_train, y_train)
                    loss_weights, sample_weights = _class_weights(y_train_np)
                    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
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
                            "test_accuracy": test_metrics["accuracy"],
                            "test_macro_recall": test_metrics["macro_recall"],
                            "synthetic_accuracy": syn_metrics["accuracy"],
                            "synthetic_macro_recall": syn_metrics["macro_recall"],
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
                            "test_accuracy": test_metrics["accuracy"],
                            "test_macro_recall": test_metrics["macro_recall"],
                            "synthetic_accuracy": syn_metrics["accuracy"],
                            "synthetic_macro_recall": syn_metrics["macro_recall"],
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
                            f"val_macro={val_metrics['macro_recall']:.4f} test_acc={test_metrics['accuracy']:.4f} "
                            f"test_macro={test_metrics['macro_recall']:.4f} synth_acc={syn_metrics['accuracy']:.4f} "
                            f"synth_macro={syn_metrics['macro_recall']:.4f}"
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
        "bootstrap_run_dir": str(args.bootstrap_run_dir),
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
        **focus_summary,
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
                "bootstrap_run_dir": str(args.bootstrap_run_dir),
                "focus_source": args.focus_source,
                "focus_branch": args.focus_branch,
                "focus_mix_fraction": args.focus_mix_fraction,
                "focus_noise_multiplier": args.focus_noise_multiplier,
                **focus_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
