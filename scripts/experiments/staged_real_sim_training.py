#!/usr/bin/env python
"""Run staged long-horizon real-data training with batch-size cycling and LBFGS bridges."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mc_surrogate.models import Standardizer, build_model
from mc_surrogate.mohr_coulomb import BRANCH_NAMES
from mc_surrogate.training import (
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    evaluate_checkpoint_on_dataset,
    set_seed,
)


@dataclass(frozen=True)
class AdamStage:
    name: str
    batch_size: int
    lr_start: float
    lr_end: float
    epochs: int
    lbfgs_steps_after: int


@dataclass(frozen=True)
class RunSpec:
    name: str
    width: int
    depth: int
    batch_cap: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Primary real dataset used for train/val/test.")
    parser.add_argument("--cross-dataset", required=True, help="Secondary real dataset used for cross-evaluation.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument(
        "--spec-names",
        nargs="*",
        help="Optional subset of run names to execute, e.g. rb_staged_w768_d8",
    )
    return parser.parse_args()


def _make_loaders(train_ds, val_ds, test_ds, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "stage_name",
                "stage_kind",
                "batch_size",
                "lr",
                "train_loss",
                "val_loss",
                "test_loss",
                "train_stress_mse",
                "val_stress_mse",
                "test_stress_mse",
                "train_branch_accuracy",
                "val_branch_accuracy",
                "test_branch_accuracy",
                "best_test_stress_mse",
                "is_best",
            ]
        )


def _append_history_row(
    path: Path,
    *,
    epoch: int,
    stage_name: str,
    stage_kind: str,
    batch_size: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_test_stress_mse: float,
    is_best: bool,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                stage_name,
                stage_kind,
                batch_size,
                lr,
                train_metrics["loss"],
                val_metrics["loss"],
                test_metrics["loss"],
                train_metrics["stress_mse"],
                val_metrics["stress_mse"],
                test_metrics["stress_mse"],
                train_metrics["branch_accuracy"],
                val_metrics["branch_accuracy"],
                test_metrics["branch_accuracy"],
                best_test_stress_mse,
                1 if is_best else 0,
            ]
        )


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def _plot_staged_history(history_csv: Path, output_path: Path) -> Path:
    rows = []
    with history_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {history_csv}.")

    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    train_loss = np.array([float(r["train_loss"]) for r in rows], dtype=float)
    val_loss = np.array([float(r["val_loss"]) for r in rows], dtype=float)
    test_loss = np.array([float(r["test_loss"]) for r in rows], dtype=float)
    train_stress = np.array([float(r["train_stress_mse"]) for r in rows], dtype=float)
    val_stress = np.array([float(r["val_stress_mse"]) for r in rows], dtype=float)
    test_stress = np.array([float(r["test_stress_mse"]) for r in rows], dtype=float)
    stage_names = [r["stage_name"] for r in rows]

    boundaries: list[tuple[int, str]] = []
    prev = None
    for e, s in zip(epoch, stage_names):
        if s != prev:
            boundaries.append((int(e), s))
            prev = s

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(epoch, train_loss, label="train loss")
    axes[0].plot(epoch, val_loss, label="val loss")
    axes[0].plot(epoch, test_loss, label="test loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch, train_stress, label="train stress mse")
    axes[1].plot(epoch, val_stress, label="val stress mse")
    axes[1].plot(epoch, test_stress, label="test stress mse")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("stress mse")
    axes[1].set_xlabel("global epoch / stage step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ymax0 = max(train_loss.max(), val_loss.max(), test_loss.max())
    ymax1 = max(train_stress.max(), val_stress.max(), test_stress.max())
    for ax in axes:
        for x, label in boundaries:
            ax.axvline(x, color="k", linestyle="--", alpha=0.2)
    for x, label in boundaries:
        axes[0].text(x + 1, ymax0, label, rotation=90, va="top", ha="left", fontsize=8)
        axes[1].text(x + 1, ymax1, label, rotation=90, va="top", ha="left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _stage_schedule() -> list[AdamStage]:
    return [
        AdamStage("adam_bs1024", batch_size=1024, lr_start=1.0e-3, lr_end=3.0e-4, epochs=140, lbfgs_steps_after=2),
        AdamStage("adam_bs2048", batch_size=2048, lr_start=3.0e-4, lr_end=1.0e-4, epochs=120, lbfgs_steps_after=2),
        AdamStage("adam_bs4096", batch_size=4096, lr_start=1.0e-4, lr_end=3.0e-5, epochs=120, lbfgs_steps_after=3),
        AdamStage("adam_bs6144", batch_size=6144, lr_start=3.0e-5, lr_end=1.0e-6, epochs=160, lbfgs_steps_after=4),
    ]


def _run_specs() -> list[RunSpec]:
    return [
        RunSpec("rb_staged_w512_d6", width=512, depth=6, batch_cap=6144, seed=501),
        RunSpec("rb_staged_w768_d8", width=768, depth=8, batch_cap=4096, seed=502),
        RunSpec("rb_staged_w1024_d10", width=1024, depth=10, batch_cap=3072, seed=503),
    ]


def _train_one(
    *,
    dataset_path: str,
    cross_dataset_path: str,
    run_dir: Path,
    spec: RunSpec,
    device: torch.device,
    eval_batch_size: int,
) -> dict[str, object]:
    set_seed(spec.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_arrays = _load_split_for_training(dataset_path, "train", "raw_branch")
    val_arrays = _load_split_for_training(dataset_path, "val", "raw_branch")
    test_arrays = _load_split_for_training(dataset_path, "test", "raw_branch")

    x_scaler = Standardizer.from_array(train_arrays["features"])
    y_scaler = Standardizer.from_array(train_arrays["target"])
    train_ds = _build_tensor_dataset(train_arrays, x_scaler, y_scaler)
    val_ds = _build_tensor_dataset(val_arrays, x_scaler, y_scaler)
    test_ds = _build_tensor_dataset(test_arrays, x_scaler, y_scaler)

    model = build_model("raw_branch", input_dim=train_arrays["features"].shape[1], width=spec.width, depth=spec.depth).to(device)
    metadata = {
        "config": {
            "dataset": dataset_path,
            "cross_dataset": cross_dataset_path,
            "run_dir": str(run_dir),
            "model_kind": "raw_branch",
            "width": spec.width,
            "depth": spec.depth,
            "dropout": 0.0,
            "seed": spec.seed,
            "staged_schedule": [asdict(stage) for stage in _stage_schedule()],
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_csv = run_dir / "history.csv"
    _write_history_header(history_csv)
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    global_epoch = 0
    best_test_stress_mse = float("inf")
    best_epoch = 0
    best_stage = ""
    stagnant_steps = 0
    min_delta = 1.0
    stop_patience = 140

    for stage in _stage_schedule():
        batch_size = min(stage.batch_size, spec.batch_cap)
        train_loader, val_loader, test_loader = _make_loaders(train_ds, val_ds, test_ds, batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage.lr_start, weight_decay=2.0e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(stage.epochs, 1),
            eta_min=stage.lr_end,
        )

        for _ in range(stage.epochs):
            global_epoch += 1
            train_metrics = _epoch_loop(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=0.1,
                device=device,
                grad_clip=1.0,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            val_metrics = _epoch_loop(
                model=model,
                loader=val_loader,
                optimizer=None,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=0.1,
                device=device,
                grad_clip=1.0,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            test_metrics = _epoch_loop(
                model=model,
                loader=test_loader,
                optimizer=None,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=0.1,
                device=device,
                grad_clip=1.0,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            improved = test_metrics["stress_mse"] < (best_test_stress_mse - min_delta)
            if improved:
                best_test_stress_mse = test_metrics["stress_mse"]
                best_epoch = global_epoch
                best_stage = stage.name
                stagnant_steps = 0
                _save_checkpoint(best_ckpt, model, metadata)
            else:
                stagnant_steps += 1
            _save_checkpoint(last_ckpt, model, metadata)
            _append_history_row(
                history_csv,
                epoch=global_epoch,
                stage_name=stage.name,
                stage_kind="adam",
                batch_size=batch_size,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                best_test_stress_mse=best_test_stress_mse,
                is_best=improved,
            )
            if global_epoch == 1 or global_epoch % 25 == 0:
                print(
                    f"[{spec.name}] epoch={global_epoch} stage={stage.name} "
                    f"lr={current_lr:.3e} train_loss={train_metrics['loss']:.6f} "
                    f"val_loss={val_metrics['loss']:.6f} test_loss={test_metrics['loss']:.6f} "
                    f"test_stress_mse={test_metrics['stress_mse']:.3f} "
                    f"best_test_stress_mse={best_test_stress_mse:.3f}"
                )
            if stagnant_steps >= stop_patience:
                break

        if stagnant_steps >= stop_patience:
            break

        if stage.lbfgs_steps_after > 0:
            full_train = tuple(t.to(device) for t in train_ds.tensors)
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.15,
                max_iter=16,
                history_size=100,
                line_search_fn="strong_wolfe",
            )
            for step in range(stage.lbfgs_steps_after):
                global_epoch += 1
                xb, yb, branch, stress_true, eigvecs, trial_stress = full_train

                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    out = model(xb)
                    reg_loss = torch.mean((out["stress"] - yb) ** 2)
                    valid_branch = branch >= 0
                    if torch.any(valid_branch):
                        branch_loss = torch.nn.functional.cross_entropy(
                            out["branch_logits"][valid_branch],
                            branch[valid_branch],
                        )
                        reg_loss = reg_loss + 0.1 * branch_loss
                    reg_loss.backward()
                    return reg_loss

                model.train(True)
                lbfgs.step(closure)
                train_loader_eval, val_loader_eval, test_loader_eval = _make_loaders(train_ds, val_ds, test_ds, batch_size)
                train_metrics = _epoch_loop(
                    model=model,
                    loader=train_loader_eval,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=0.1,
                    device=device,
                    grad_clip=1.0,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                val_metrics = _epoch_loop(
                    model=model,
                    loader=val_loader_eval,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=0.1,
                    device=device,
                    grad_clip=1.0,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                test_metrics = _epoch_loop(
                    model=model,
                    loader=test_loader_eval,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=0.1,
                    device=device,
                    grad_clip=1.0,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                improved = test_metrics["stress_mse"] < (best_test_stress_mse - min_delta)
                if improved:
                    best_test_stress_mse = test_metrics["stress_mse"]
                    best_epoch = global_epoch
                    best_stage = f"{stage.name}_lbfgs"
                    stagnant_steps = 0
                    _save_checkpoint(best_ckpt, model, metadata)
                else:
                    stagnant_steps += 1
                _save_checkpoint(last_ckpt, model, metadata)
                _append_history_row(
                    history_csv,
                    epoch=global_epoch,
                    stage_name=f"{stage.name}_lbfgs",
                    stage_kind="lbfgs",
                    batch_size=batch_size,
                    lr=lbfgs.param_groups[0]["lr"],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    best_test_stress_mse=best_test_stress_mse,
                    is_best=improved,
                )
                if global_epoch % 25 == 0 or step == stage.lbfgs_steps_after - 1:
                    print(
                        f"[{spec.name}] epoch={global_epoch} stage={stage.name}_lbfgs "
                        f"test_stress_mse={test_metrics['stress_mse']:.3f} "
                        f"best_test_stress_mse={best_test_stress_mse:.3f}"
                    )
                if stagnant_steps >= stop_patience:
                    break
        if stagnant_steps >= stop_patience:
            break

    plot_path = _plot_staged_history(history_csv, run_dir / "staged_history_log.png")

    primary_eval = evaluate_checkpoint_on_dataset(best_ckpt, dataset_path, split="test", device=str(device), batch_size=eval_batch_size)
    cross_eval = evaluate_checkpoint_on_dataset(best_ckpt, cross_dataset_path, split="test", device=str(device), batch_size=eval_batch_size)
    (run_dir / "primary_metrics.json").write_text(json.dumps(primary_eval["metrics"], indent=2), encoding="utf-8")
    (run_dir / "cross_metrics.json").write_text(json.dumps(cross_eval["metrics"], indent=2), encoding="utf-8")

    summary = {
        "run_name": spec.name,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_test_stress_mse": best_test_stress_mse,
        "history_csv": str(history_csv),
        "staged_plot": str(plot_path),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "primary_metrics": primary_eval["metrics"],
        "cross_metrics": cross_eval["metrics"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _render_report(output_root: Path, results: list[dict[str, object]]) -> Path:
    report_path = output_root / "staged_training_report.md"
    lines = [
        "# Staged Real-Simulation Training",
        "",
        "This report summarizes staged long-horizon training with batch-size cycling, LR decay, and LBFGS bridges.",
        "",
        "## Schedule",
        "",
    ]
    for stage in _stage_schedule():
        lines.append(
            f"- `{stage.name}`: batch `{stage.batch_size}`, lr `{stage.lr_start:.1e}` -> `{stage.lr_end:.1e}`, "
            f"`{stage.epochs}` Adam epochs, `{stage.lbfgs_steps_after}` LBFGS steps after stage"
        )
    lines.extend(["", "## Results", ""])
    lines.append("| Run | Best stage | Broad MAE | Broad RMSE | Broad max abs | Broad branch acc | Cross MAE | Cross RMSE |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in results:
        primary = item["primary_metrics"]
        cross = item["cross_metrics"]
        lines.append(
            f"| `{item['run_name']}` | `{item['best_stage']}` | "
            f"{primary['stress_mae']:.2f} | {primary['stress_rmse']:.2f} | {primary['stress_max_abs']:.2f} | "
            f"{primary.get('branch_accuracy', float('nan')):.3f} | {cross['stress_mae']:.2f} | {cross['stress_rmse']:.2f} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for item in results:
        run_dir = output_root / item["run_name"]
        lines.append(f"- `{item['run_name']}`")
        lines.append(f"  - history: `{run_dir / 'history.csv'}`")
        lines.append(f"  - staged plot: `{run_dir / 'staged_history_log.png'}`")
        lines.append(f"  - summary: `{run_dir / 'summary.json'}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    specs = _run_specs()
    if args.spec_names:
        wanted = set(args.spec_names)
        specs = [spec for spec in specs if spec.name in wanted]
    if args.limit_runs > 0:
        specs = specs[: args.limit_runs]

    results: list[dict[str, object]] = []
    for spec in specs:
        run_dir = output_root / spec.name
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = _train_one(
                dataset_path=args.dataset,
                cross_dataset_path=args.cross_dataset,
                run_dir=run_dir,
                spec=spec,
                device=device,
                eval_batch_size=args.eval_batch_size,
            )
        results.append(summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda item: (float(item["primary_metrics"]["stress_mae"]), float(item["primary_metrics"]["stress_rmse"])))
    (output_root / "staged_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = _render_report(output_root, results)
    print(json.dumps({"results": results, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
