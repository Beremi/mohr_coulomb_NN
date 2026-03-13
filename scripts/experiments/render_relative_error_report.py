#!/usr/bin/env python
"""Render test-loss and relative-error visual reports for a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mc_surrogate.training import evaluate_checkpoint_on_dataset


COMPONENT_LABELS = ("s11", "s22", "s33", "s12", "s13", "s23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history-csv", required=True)
    parser.add_argument("--primary-dataset", required=True)
    parser.add_argument("--cross-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    return parser.parse_args()


def _load_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stage_boundaries(rows: list[dict[str, str]]) -> list[tuple[int, str]]:
    boundaries: list[tuple[int, str]] = []
    prev = None
    for row in rows:
        stage = row["stage_name"]
        epoch = int(row["epoch"])
        if stage != prev:
            boundaries.append((epoch, stage))
            prev = stage
    return boundaries


def _plot_test_loss(rows: list[dict[str, str]], output_path: Path) -> Path:
    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    train_loss = np.array([float(r["train_loss"]) for r in rows], dtype=float)
    val_loss = np.array([float(r["val_loss"]) for r in rows], dtype=float)
    test_loss = np.array([float(r["test_loss"]) for r in rows], dtype=float)
    best_test = np.minimum.accumulate(test_loss)
    boundaries = _stage_boundaries(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.plot(epoch, train_loss, label="train loss")
    plt.plot(epoch, val_loss, label="val loss")
    plt.plot(epoch, test_loss, label="test loss")
    plt.plot(epoch, best_test, label="best-so-far test loss", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("global epoch / stage step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    ymax = max(train_loss.max(), val_loss.max(), test_loss.max())
    for x, label in boundaries:
        plt.axvline(x, color="k", linestyle="--", alpha=0.2)
        plt.text(x + 1, ymax, label, rotation=90, va="top", ha="left", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_test_stress_mse(rows: list[dict[str, str]], output_path: Path) -> Path:
    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    train_stress = np.array([float(r["train_stress_mse"]) for r in rows], dtype=float)
    val_stress = np.array([float(r["val_stress_mse"]) for r in rows], dtype=float)
    test_stress = np.array([float(r["test_stress_mse"]) for r in rows], dtype=float)
    best_test = np.minimum.accumulate(test_stress)
    boundaries = _stage_boundaries(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.plot(epoch, train_stress, label="train stress mse")
    plt.plot(epoch, val_stress, label="val stress mse")
    plt.plot(epoch, test_stress, label="test stress mse")
    plt.plot(epoch, best_test, label="best-so-far test stress mse", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("global epoch / stage step")
    plt.ylabel("stress mse")
    plt.grid(True, alpha=0.3)
    ymax = max(train_stress.max(), val_stress.max(), test_stress.max())
    for x, label in boundaries:
        plt.axvline(x, color="k", linestyle="--", alpha=0.2)
        plt.text(x + 1, ymax, label, rotation=90, va="top", ha="left", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _samplewise_relative_metrics(stress_true: np.ndarray, stress_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = stress_pred - stress_true
    sample_rel_l2 = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(stress_true, axis=1), 1.0)
    comp_rel_signed = diff / np.maximum(np.abs(stress_true), 1.0)
    return sample_rel_l2.astype(np.float32), comp_rel_signed.astype(np.float32)


def _plot_relative_histogram(sample_rel_l2: np.ndarray, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(sample_rel_l2, bins=80)
    plt.xlabel(r"relative sample error $\|\hat{\sigma}-\sigma\|_2 / \max(\|\sigma\|_2, 1)$")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_relative_cdf(sample_rel_l2: np.ndarray, output_path: Path, title: str) -> Path:
    x = np.sort(sample_rel_l2)
    y = np.linspace(0.0, 1.0, x.size, endpoint=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel(r"relative sample error $\|\hat{\sigma}-\sigma\|_2 / \max(\|\sigma\|_2, 1)$")
    plt.ylabel("fraction of samples")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_random_relative_scatter(
    sample_rel_l2: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(indices.size), sample_rel_l2[indices], s=18, alpha=0.7)
    plt.xlabel("random sample order")
    plt.ylabel("relative sample error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_random_relative_heatmap(
    comp_rel_signed: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    arr = comp_rel_signed[indices]
    vmax = float(np.quantile(np.abs(arr), 0.95))
    vmax = max(vmax, 0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 9))
    plt.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(label=r"$(\hat{\sigma} - \sigma) / \max(|\sigma|, 1)$")
    plt.xticks(np.arange(len(COMPONENT_LABELS)), COMPONENT_LABELS)
    plt.yticks(np.arange(indices.size), [str(int(i)) for i in indices], fontsize=7)
    plt.xlabel("stress component")
    plt.ylabel("random sample index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _render_dataset_section(
    *,
    checkpoint: Path,
    dataset_path: Path,
    name: str,
    output_dir: Path,
    device: str,
    eval_batch_size: int,
) -> dict[str, object]:
    result = evaluate_checkpoint_on_dataset(
        checkpoint_path=checkpoint,
        dataset_path=dataset_path,
        split="test",
        device=device,
        batch_size=eval_batch_size,
    )
    metrics = result["metrics"]
    stress_true = result["arrays"]["stress"]
    stress_pred = result["predictions"]["stress"]
    sample_rel_l2, comp_rel_signed = _samplewise_relative_metrics(stress_true, stress_pred)

    rng = np.random.default_rng(0)
    n_random = min(96, stress_true.shape[0])
    indices = np.sort(rng.choice(stress_true.shape[0], size=n_random, replace=False))

    section_dir = output_dir / name
    section_dir.mkdir(parents=True, exist_ok=True)
    hist_png = _plot_relative_histogram(sample_rel_l2, section_dir / "relative_hist.png", f"{name}: relative error histogram")
    cdf_png = _plot_relative_cdf(sample_rel_l2, section_dir / "relative_cdf.png", f"{name}: relative error CDF")
    scatter_png = _plot_random_relative_scatter(
        sample_rel_l2,
        indices,
        section_dir / "random_relative_scatter.png",
        f"{name}: random-sample relative errors",
    )
    heatmap_png = _plot_random_relative_heatmap(
        comp_rel_signed,
        indices,
        section_dir / "random_component_relative_heatmap.png",
        f"{name}: signed componentwise relative difference on random samples",
    )

    summary = {
        "metrics": metrics,
        "relative_summary": {
            "sample_rel_l2_mean": float(np.mean(sample_rel_l2)),
            "sample_rel_l2_median": float(np.median(sample_rel_l2)),
            "sample_rel_l2_p90": float(np.quantile(sample_rel_l2, 0.9)),
            "sample_rel_l2_p99": float(np.quantile(sample_rel_l2, 0.99)),
        },
        "images": {
            "relative_hist": str(hist_png),
            "relative_cdf": str(cdf_png),
            "random_relative_scatter": str(scatter_png),
            "random_component_relative_heatmap": str(heatmap_png),
        },
    }
    (section_dir / "relative_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _relpath(target: Path, start: Path) -> str:
    return str(target.relative_to(start.parent.resolve() if start.is_absolute() else start.parent))


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    history_csv = Path(args.history_csv).resolve()
    primary_dataset = Path(args.primary_dataset).resolve()
    cross_dataset = Path(args.cross_dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_md = Path(args.report_md).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_history(history_csv)
    test_loss_png = _plot_test_loss(rows, output_dir / "test_loss_log.png")
    test_stress_png = _plot_test_stress_mse(rows, output_dir / "test_stress_mse_log.png")

    primary = _render_dataset_section(
        checkpoint=checkpoint,
        dataset_path=primary_dataset,
        name="primary_d512",
        output_dir=output_dir,
        device=args.device,
        eval_batch_size=args.eval_batch_size,
    )
    cross = _render_dataset_section(
        checkpoint=checkpoint,
        dataset_path=cross_dataset,
        name="cross_d256",
        output_dir=output_dir,
        device=args.device,
        eval_batch_size=args.eval_batch_size,
    )

    report = [
        "# Relative Error Visual Report",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "This report focuses on:",
        "",
        "- staged test-loss evolution",
        "- robust relative error on held-out samples",
        "- random-sample relative differences between predicted and true stresses",
        "",
        "Relative metrics in this report use robust denominators:",
        "",
        r"- sample relative error: `||pred - true||_2 / max(||true||_2, 1)`",
        r"- signed componentwise relative difference: `(pred - true) / max(|true|, 1)`",
        "",
        "## Test Loss Curves",
        "",
        "![Test loss log](../experiment_runs/real_sim/staged_20260312/relative_error_report/test_loss_log.png)",
        "",
        "![Test stress mse log](../experiment_runs/real_sim/staged_20260312/relative_error_report/test_stress_mse_log.png)",
        "",
        "## Primary Test Split `d512`",
        "",
        f"- `stress_mae = {primary['metrics']['stress_mae']:.4f}`",
        f"- `stress_rmse = {primary['metrics']['stress_rmse']:.4f}`",
        f"- `stress_max_abs = {primary['metrics']['stress_max_abs']:.4f}`",
        f"- `branch_accuracy = {primary['metrics'].get('branch_accuracy', float('nan')):.4f}`",
        f"- `mean relative sample error = {primary['relative_summary']['sample_rel_l2_mean']:.4f}`",
        f"- `median relative sample error = {primary['relative_summary']['sample_rel_l2_median']:.4f}`",
        f"- `p90 relative sample error = {primary['relative_summary']['sample_rel_l2_p90']:.4f}`",
        f"- `p99 relative sample error = {primary['relative_summary']['sample_rel_l2_p99']:.4f}`",
        "",
        "![Primary relative histogram](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/relative_hist.png)",
        "",
        "![Primary relative CDF](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/relative_cdf.png)",
        "",
        "![Primary random relative scatter](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/random_relative_scatter.png)",
        "",
        "![Primary random component heatmap](../experiment_runs/real_sim/staged_20260312/relative_error_report/primary_d512/random_component_relative_heatmap.png)",
        "",
        "## Cross-Check Test Split `d256`",
        "",
        f"- `stress_mae = {cross['metrics']['stress_mae']:.4f}`",
        f"- `stress_rmse = {cross['metrics']['stress_rmse']:.4f}`",
        f"- `stress_max_abs = {cross['metrics']['stress_max_abs']:.4f}`",
        f"- `branch_accuracy = {cross['metrics'].get('branch_accuracy', float('nan')):.4f}`",
        f"- `mean relative sample error = {cross['relative_summary']['sample_rel_l2_mean']:.4f}`",
        f"- `median relative sample error = {cross['relative_summary']['sample_rel_l2_median']:.4f}`",
        f"- `p90 relative sample error = {cross['relative_summary']['sample_rel_l2_p90']:.4f}`",
        f"- `p99 relative sample error = {cross['relative_summary']['sample_rel_l2_p99']:.4f}`",
        "",
        "![Cross relative histogram](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/relative_hist.png)",
        "",
        "![Cross relative CDF](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/relative_cdf.png)",
        "",
        "![Cross random relative scatter](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/random_relative_scatter.png)",
        "",
        "![Cross random component heatmap](../experiment_runs/real_sim/staged_20260312/relative_error_report/cross_d256/random_component_relative_heatmap.png)",
        "",
    ]
    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "checkpoint": str(checkpoint),
        "history_csv": str(history_csv),
        "output_dir": str(output_dir),
        "report_md": str(report_md),
        "primary": primary,
        "cross": cross,
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
