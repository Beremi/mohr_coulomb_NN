#!/usr/bin/env python
"""Render a material-by-material prediction-vs-reality report on the real test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.training import evaluate_checkpoint_on_dataset
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-material-summary",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/summary.json",
    )
    parser.add_argument(
        "--hybrid-summary",
        default="experiment_runs/real_sim/per_material_hybrid_hardcases_20260312/summary.json",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_runs/real_sim/per_material_prediction_report_20260312",
    )
    parser.add_argument(
        "--report-md",
        default="docs/per_material_prediction_vs_reality.md",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    return parser.parse_args()


def _plot_history_log(history_csv: Path, output_path: Path) -> Path:
    import csv

    with history_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epoch = np.array([float(row["epoch"]) for row in rows], dtype=float)
    train_loss = np.array([float(row["train_loss"]) for row in rows], dtype=float)
    val_loss = np.array([float(row["val_loss"]) for row in rows], dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch, train_loss, label="train loss")
    plt.plot(epoch, val_loss, label="val loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _sample_loss_arrays(stress_true: np.ndarray, stress_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = stress_pred - stress_true
    sample_l2 = np.linalg.norm(diff, axis=1)
    true_l2 = np.linalg.norm(stress_true, axis=1)
    rel_l2 = sample_l2 / np.maximum(true_l2, 1.0)
    return sample_l2.astype(np.float32), true_l2.astype(np.float32), rel_l2.astype(np.float32)


def _plot_sample_loss_hist(sample_l2: np.ndarray, output_path: Path, *, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(sample_l2, bins=80)
    plt.xlabel(r"sample test loss $\|\hat{\sigma}-\sigma\|_2$")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_loss_vs_true_norm(
    sample_l2: np.ndarray,
    true_l2: np.ndarray,
    output_path: Path,
    *,
    title: str,
    max_points: int = 4000,
) -> Path:
    if sample_l2.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(sample_l2.size, size=max_points, replace=False)
        sample_l2 = sample_l2[idx]
        true_l2 = true_l2[idx]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(true_l2, sample_l2, s=8, alpha=0.35)
    plt.xlabel(r"true stress magnitude $\|\sigma\|_2$")
    plt.ylabel(r"sample test loss $\|\hat{\sigma}-\sigma\|_2$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_relative_cdf(rel_l2: np.ndarray, output_path: Path, *, title: str) -> Path:
    x = np.sort(rel_l2)
    y = np.linspace(0.0, 1.0, x.size, endpoint=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(r"relative sample test loss $\|\hat{\sigma}-\sigma\|_2 / \max(\|\sigma\|_2, 1)$")
    plt.ylabel("fraction of test samples")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_metric_bars(material_rows: list[dict[str, object]], output_path: Path) -> Path:
    names = [str(row["material"]) for row in material_rows]
    selected_mae = [float(row["selected_metrics"]["stress_mae"]) for row in material_rows]
    baseline_mae = [float(row["baseline_metrics"]["stress_mae"]) for row in material_rows]
    x = np.arange(len(names))
    width = 0.36

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2.0, selected_mae, width=width, label="selected per-material model")
    plt.bar(x + width / 2.0, baseline_mae, width=width, label="global real-trained baseline")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("stress MAE on primary real test split")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _relpath(target: Path, report_md: Path) -> str:
    return str(target.resolve().relative_to(report_md.resolve().parent.parent))


def _render_one(
    *,
    material: str,
    checkpoint: Path,
    history_csv: Path,
    dataset_path: Path,
    baseline_checkpoint: Path,
    output_dir: Path,
    device: str,
    eval_batch_size: int,
) -> dict[str, object]:
    section_dir = output_dir / material
    section_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(history_csv, section_dir / "history_linear.png")
    history_log_png = _plot_history_log(history_csv, section_dir / "history_log.png")

    selected = evaluate_checkpoint_on_dataset(
        checkpoint_path=checkpoint,
        dataset_path=dataset_path,
        split="test",
        device=device,
        batch_size=eval_batch_size,
    )
    baseline = evaluate_checkpoint_on_dataset(
        checkpoint_path=baseline_checkpoint,
        dataset_path=dataset_path,
        split="test",
        device=device,
        batch_size=eval_batch_size,
    )

    stress_true = selected["arrays"]["stress"]
    stress_pred = selected["predictions"]["stress"]
    sample_l2, true_l2, rel_l2 = _sample_loss_arrays(stress_true, stress_pred)

    parity_png = parity_plot(stress_true, stress_pred, section_dir / "parity_stress.png", label="stress")
    error_png = error_histogram(stress_pred - stress_true, section_dir / "stress_error_hist.png", label="stress error")
    sample_loss_png = _plot_sample_loss_hist(sample_l2, section_dir / "sample_l2_hist.png", title=f"{material}: test loss histogram")
    loss_vs_norm_png = _plot_loss_vs_true_norm(
        sample_l2,
        true_l2,
        section_dir / "sample_l2_vs_true_norm.png",
        title=f"{material}: test loss vs true stress magnitude",
    )
    rel_cdf_png = _plot_relative_cdf(rel_l2, section_dir / "relative_loss_cdf.png", title=f"{material}: relative test loss CDF")

    branch_png = None
    if "branch_confusion" in selected["metrics"]:
        branch_png = branch_confusion_plot(selected["metrics"]["branch_confusion"], section_dir / "branch_confusion.png")

    out = {
        "material": material,
        "checkpoint": str(checkpoint),
        "history_csv": str(history_csv),
        "dataset_path": str(dataset_path),
        "selected_metrics": selected["metrics"],
        "baseline_metrics": baseline["metrics"],
        "relative_summary": {
            "sample_l2_mean": float(np.mean(sample_l2)),
            "sample_l2_median": float(np.median(sample_l2)),
            "sample_l2_p90": float(np.quantile(sample_l2, 0.9)),
            "sample_l2_p99": float(np.quantile(sample_l2, 0.99)),
            "relative_l2_mean": float(np.mean(rel_l2)),
            "relative_l2_median": float(np.median(rel_l2)),
            "relative_l2_p90": float(np.quantile(rel_l2, 0.9)),
            "relative_l2_p99": float(np.quantile(rel_l2, 0.99)),
        },
        "images": {
            "history_linear": str(section_dir / "history_linear.png"),
            "history_log": str(history_log_png),
            "parity_stress": str(parity_png),
            "stress_error_hist": str(error_png),
            "sample_l2_hist": str(sample_loss_png),
            "sample_l2_vs_true_norm": str(loss_vs_norm_png),
            "relative_loss_cdf": str(rel_cdf_png),
            "branch_confusion": str(branch_png) if branch_png is not None else None,
        },
    }
    (section_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    per_material_summary = json.loads(Path(args.per_material_summary).read_text(encoding="utf-8"))
    hybrid_summary = json.loads(Path(args.hybrid_summary).read_text(encoding="utf-8"))
    baseline_checkpoint = Path(args.baseline_checkpoint)
    output_dir = Path(args.output_dir)
    report_md = Path(args.report_md)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, object]] = []

    # Keep the empirical per-material models where they were clearly good.
    empirical_materials = ("general_foundation", "weak_foundation")
    for material in empirical_materials:
        row = per_material_summary["materials"][material]
        selected_rows.append(
            _render_one(
                material=material,
                checkpoint=Path(row["train_summary"]["best_checkpoint"]),
                history_csv=Path(row["train_summary"]["history_csv"]),
                dataset_path=Path(row["primary_dataset"]),
                baseline_checkpoint=baseline_checkpoint,
                output_dir=output_dir,
                device=args.device,
                eval_batch_size=args.eval_batch_size,
            )
        )

    # For the hard materials, use the improved hybrid models.
    for material in ("general_slope", "cover_layer"):
        row = hybrid_summary["hardcase_results"][material]
        dataset_path = Path(per_material_summary["materials"][material]["primary_dataset"])
        selected_rows.append(
            _render_one(
                material=material,
                checkpoint=Path(row["train_summary"]["best_checkpoint"]),
                history_csv=Path(row["train_summary"]["history_csv"]),
                dataset_path=dataset_path,
                baseline_checkpoint=baseline_checkpoint,
                output_dir=output_dir,
                device=args.device,
                eval_batch_size=args.eval_batch_size,
            )
        )

    mae_bar_png = _plot_metric_bars(selected_rows, output_dir / "primary_mae_by_material.png")

    summary = {"materials": selected_rows, "summary_plot": str(mae_bar_png)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Per-Material Prediction vs Reality",
        "",
        "This report shows the prediction-vs-reality behavior on the primary real test split for the best",
        "per-material checkpoint available for each material family:",
        "- `general_foundation`: empirical exact relabel model",
        "- `weak_foundation`: empirical exact relabel model",
        "- `general_slope`: hybrid hardcase synthetic model",
        "- `cover_layer`: hybrid hardcase synthetic model",
        "",
        "The comparison target in the tables is the current global real-trained baseline.",
        "",
        "## Summary",
        "",
        "![Primary MAE by material](../experiment_runs/real_sim/per_material_prediction_report_20260312/primary_mae_by_material.png)",
        "",
        "| Material | Selected MAE | Baseline MAE | Selected RMSE | Baseline RMSE | Selected Branch Acc | Baseline Branch Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in selected_rows:
        s = row["selected_metrics"]
        b = row["baseline_metrics"]
        lines.append(
            f"| {row['material']} | {s['stress_mae']:.4f} | {b['stress_mae']:.4f} | "
            f"{s['stress_rmse']:.4f} | {b['stress_rmse']:.4f} | "
            f"{s.get('branch_accuracy', float('nan')):.4f} | {b.get('branch_accuracy', float('nan')):.4f} |"
        )

    for row in selected_rows:
        material = row["material"]
        metrics = row["selected_metrics"]
        rel = row["relative_summary"]
        images = row["images"]
        lines.extend(
            [
                "",
                f"## {material}",
                "",
                f"- checkpoint: `{row['checkpoint']}`",
                f"- primary test dataset: `{row['dataset_path']}`",
                f"- stress MAE: `{metrics['stress_mae']:.4f}`",
                f"- stress RMSE: `{metrics['stress_rmse']:.4f}`",
                f"- stress max abs: `{metrics['stress_max_abs']:.4f}`",
                f"- branch accuracy: `{metrics.get('branch_accuracy', float('nan')):.4f}`",
                f"- sample L2 test loss mean / p90 / p99: `{rel['sample_l2_mean']:.4f}` / `{rel['sample_l2_p90']:.4f}` / `{rel['sample_l2_p99']:.4f}`",
                f"- relative sample test loss mean / p90 / p99: `{rel['relative_l2_mean']:.4f}` / `{rel['relative_l2_p90']:.4f}` / `{rel['relative_l2_p99']:.4f}`",
                "",
                "Training history:",
                "",
                f"![{material} history log](../{Path(images['history_log']).as_posix()})",
                "",
                "Prediction vs reality parity on the primary real test split:",
                "",
                f"![{material} parity](../{Path(images['parity_stress']).as_posix()})",
                "",
                "Test-loss histogram over samples:",
                "",
                f"![{material} sample loss histogram](../{Path(images['sample_l2_hist']).as_posix()})",
                "",
                "Test loss versus true stress magnitude:",
                "",
                f"![{material} loss vs true norm](../{Path(images['sample_l2_vs_true_norm']).as_posix()})",
                "",
                "Relative test-loss CDF:",
                "",
                f"![{material} relative loss cdf](../{Path(images['relative_loss_cdf']).as_posix()})",
                "",
                "Componentwise stress-error histogram:",
                "",
                f"![{material} stress error histogram](../{Path(images['stress_error_hist']).as_posix()})",
            ]
        )
        if images["branch_confusion"] is not None:
            lines.extend(
                [
                    "",
                    "Branch confusion on the same test split:",
                    "",
                    f"![{material} branch confusion](../{Path(images['branch_confusion']).as_posix()})",
                ]
            )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report_md": str(report_md)}, indent=2))


if __name__ == "__main__":
    main()
