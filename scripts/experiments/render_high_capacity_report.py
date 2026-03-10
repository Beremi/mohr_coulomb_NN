#!/usr/bin/env python
"""Render plots and a markdown report for the high-capacity study."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, help="Path to the study output root.")
    parser.add_argument("--dataset", required=True, help="Path to the shared dataset.")
    parser.add_argument("--report-md", default="docs/high_capacity_study.md", help="Markdown report output path.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, object]] = []
    for row in rows:
        parsed: dict[str, object] = {}
        for key, value in row.items():
            if value is None or value == "":
                parsed[key] = value
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def load_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[float]] = {}
    for key in rows[0]:
        out[key] = [float(row[key]) for row in rows]
    return {key: np.asarray(value, dtype=float) for key, value in out.items()}


def plot_pilot_comparison(rows: list[dict[str, object]], output_path: Path) -> None:
    names = [str(row["name"]) for row in rows]
    stress_mae = [float(row["test_stress_mae"]) for row in rows]
    bench_mae = [float(row["benchmark_mean_principal_mae"]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, stress_mae)
    axes[0].set_ylabel("test stress MAE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(names, bench_mae)
    axes[1].set_ylabel("benchmark mean principal MAE")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_repeat_distributions(rows: list[dict[str, object]], output_path: Path) -> None:
    metrics = [
        ("test_stress_mae", "Test stress MAE"),
        ("test_stress_rmse", "Test stress RMSE"),
        ("benchmark_mean_principal_mae", "Benchmark mean principal MAE"),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (key, label) in zip(axes, metrics, strict=True):
        vals = [float(row[key]) for row in rows]
        ax.boxplot(vals, vert=True)
        ax.scatter(np.ones(len(vals)), vals, s=12, alpha=0.6)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_repeat_history_overlay(history_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for history_path in history_paths:
        hist = load_history(history_path)
        axes[0].plot(hist["epoch"], hist["val_loss"], alpha=0.35)
        axes[1].plot(hist["epoch"], hist["val_stress_mse"], alpha=0.35)
    axes[0].set_title("Validation loss across repeats")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation stress MSE across repeats")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val stress MSE")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_history(history_path: Path, output_path: Path) -> None:
    hist = load_history(history_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("Best run loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["train_stress_mse"], label="train")
    axes[1].plot(hist["epoch"], hist["val_stress_mse"], label="val")
    axes[1].set_title("Best run stress MSE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("stress MSE")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    lbfgs_phase = hist.get("lbfgs_phase")
    if lbfgs_phase is not None and np.any(lbfgs_phase > 0.5):
        lbfgs_start = int(hist["epoch"][np.argmax(lbfgs_phase > 0.5)])
        for ax in axes:
            ax.axvline(lbfgs_start, color="tab:red", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def benchmark_rows_from_summary(summary_path: Path) -> list[dict[str, object]]:
    summary = json.loads(summary_path.read_text())
    return summary["benchmark_paths"]["rows"]


def write_markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], float_fmt: str = ".4f") -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                cells.append(format(value, float_fmt))
            else:
                cells.append(str(value))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines])


def relpath_for_markdown(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target, start=base_dir)


def main() -> None:
    args = parse_args()
    study_root = Path(args.study_root)
    dataset_path = Path(args.dataset)
    report_md = ROOT / args.report_md
    report_dir = study_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    pilot_rows = read_csv_rows(study_root / "pilot" / "pilot_results.csv")
    repeat_rows = read_csv_rows(study_root / "repeats" / "repeat_results.csv")
    aggregate = json.loads((study_root / "aggregate_summary.json").read_text())

    best_repeat = min(repeat_rows, key=lambda row: float(row["test_stress_mae"]))
    worst_repeat = max(repeat_rows, key=lambda row: float(row["test_stress_mae"]))
    repeat_rows_sorted = sorted(repeat_rows, key=lambda row: float(row["test_stress_mae"]))
    median_repeat = repeat_rows_sorted[len(repeat_rows_sorted) // 2]

    best_run_root = study_root / "repeats" / str(best_repeat["name"])
    best_history = best_run_root / "history.csv"
    best_checkpoint = best_run_root / "best.pt"
    best_summary = best_run_root / "summary.json"

    pilot_plot = report_dir / "pilot_comparison.png"
    repeat_dist_plot = report_dir / "repeat_distributions.png"
    repeat_overlay_plot = report_dir / "repeat_history_overlay.png"
    best_history_plot = report_dir / "best_history.png"
    plot_pilot_comparison(pilot_rows, pilot_plot)
    plot_repeat_distributions(repeat_rows, repeat_dist_plot)
    plot_repeat_history_overlay(
        [study_root / "repeats" / str(row["name"]) / "history.csv" for row in repeat_rows],
        repeat_overlay_plot,
    )
    plot_best_history(best_history, best_history_plot)

    best_eval = evaluate_checkpoint_on_dataset(best_checkpoint, dataset_path, split="test", device="cuda")
    stress_true = best_eval["arrays"]["stress"]
    stress_pred = best_eval["predictions"]["stress"]
    abs_err = np.abs(stress_pred - stress_true)
    mean_abs_stress = float(np.mean(np.abs(stress_true)))
    relative_mae = float(np.mean(abs_err) / max(mean_abs_stress, 1.0e-12))

    parity_png = parity_plot(stress_true, stress_pred, report_dir / "best_parity_stress.png", label="stress")
    error_png = error_histogram(stress_pred - stress_true, report_dir / "best_stress_error_hist.png", label="stress error")
    branch_png = None
    if "branch_confusion" in best_eval["metrics"]:
        branch_png = branch_confusion_plot(best_eval["metrics"]["branch_confusion"], report_dir / "best_branch_confusion.png")

    benchmark_rows = benchmark_rows_from_summary(best_summary)
    worst_paths = sorted(benchmark_rows, key=lambda row: float(row["principal_mae"]), reverse=True)[:8]

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        "\n".join(
            [
                "# High-Capacity Constitutive Surrogate Study",
                "",
                "## Setup",
                "",
                f"- Dataset: `{dataset_path}`",
                f"- Study root: `{study_root}`",
                f"- Shared dataset size: `{best_eval['metrics']['n_samples']}` test points out of the generated split",
                f"- Selected pilot by score: `{aggregate['selected_pilot']['name']}`",
                f"- Repeat runs: `{aggregate['repeat_summary']['n_runs']}`",
                "",
                "The study used a larger principal-stress network, cosine learning-rate scheduling, and an LBFGS fine-tune stage after AdamW. A composite dataset was generated to widen coverage: broad branch-balanced samples, focused branch-balanced samples, and benchmark path augmentation.",
                "",
                "## Pilot Comparison",
                "",
                f"![Pilot comparison]({relpath_for_markdown(pilot_plot, report_md.parent)})",
                "",
                write_markdown_table(
                    pilot_rows,
                    [
                        ("name", "Pilot"),
                        ("width", "Width"),
                        ("depth", "Depth"),
                        ("epochs", "Adam epochs"),
                        ("lbfgs_epochs", "LBFGS"),
                        ("test_stress_mae", "Test stress MAE"),
                        ("test_stress_rmse", "Test stress RMSE"),
                        ("test_stress_max_abs", "Test max abs"),
                        ("benchmark_mean_principal_mae", "Benchmark mean principal MAE"),
                        ("pilot_score", "Score"),
                    ],
                ),
                "",
                "## Repeat Stability",
                "",
                f"![Repeat distributions]({relpath_for_markdown(repeat_dist_plot, report_md.parent)})",
                "",
                f"![Repeat histories]({relpath_for_markdown(repeat_overlay_plot, report_md.parent)})",
                "",
                write_markdown_table(
                    [
                        {
                            "metric": "test stress MAE",
                            "mean": aggregate["repeat_summary"]["mean_test_stress_mae"],
                            "std": aggregate["repeat_summary"]["std_test_stress_mae"],
                            "best": min(float(row["test_stress_mae"]) for row in repeat_rows),
                            "worst": max(float(row["test_stress_mae"]) for row in repeat_rows),
                        },
                        {
                            "metric": "test stress RMSE",
                            "mean": aggregate["repeat_summary"]["mean_test_stress_rmse"],
                            "std": aggregate["repeat_summary"]["std_test_stress_rmse"],
                            "best": min(float(row["test_stress_rmse"]) for row in repeat_rows),
                            "worst": max(float(row["test_stress_rmse"]) for row in repeat_rows),
                        },
                        {
                            "metric": "benchmark mean principal MAE",
                            "mean": aggregate["repeat_summary"]["mean_benchmark_principal_mae"],
                            "std": aggregate["repeat_summary"]["std_benchmark_principal_mae"],
                            "best": min(float(row["benchmark_mean_principal_mae"]) for row in repeat_rows),
                            "worst": max(float(row["benchmark_mean_principal_mae"]) for row in repeat_rows),
                        },
                    ],
                    [("metric", "Metric"), ("mean", "Mean"), ("std", "Std"), ("best", "Best"), ("worst", "Worst")],
                ),
                "",
                "Best / median / worst repeats by test stress MAE:",
                "",
                write_markdown_table(
                    [best_repeat, median_repeat, worst_repeat],
                    [
                        ("name", "Run"),
                        ("seed", "Seed"),
                        ("test_stress_mae", "Test stress MAE"),
                        ("test_stress_rmse", "Test stress RMSE"),
                        ("test_stress_max_abs", "Test max abs"),
                        ("benchmark_mean_principal_mae", "Benchmark mean principal MAE"),
                        ("benchmark_worst_principal_mae", "Benchmark worst principal MAE"),
                    ],
                ),
                "",
                "## Best Run Diagnostics",
                "",
                f"![Best history]({relpath_for_markdown(best_history_plot, report_md.parent)})",
                "",
                f"![Best parity]({relpath_for_markdown(parity_png, report_md.parent)})",
                "",
                f"![Best error histogram]({relpath_for_markdown(error_png, report_md.parent)})",
                "",
                (f"![Best branch confusion]({relpath_for_markdown(branch_png, report_md.parent)})" if branch_png is not None else ""),
                "",
                f"- Best repeat: `{best_repeat['name']}`",
                f"- Test stress MAE: `{best_repeat['test_stress_mae']:.6f}`",
                f"- Test stress RMSE: `{best_repeat['test_stress_rmse']:.6f}`",
                f"- Test stress max abs: `{best_repeat['test_stress_max_abs']:.6f}`",
                f"- Relative MAE against mean absolute test stress: `{relative_mae:.4%}`",
                f"- Branch accuracy: `{best_eval['metrics']['branch_accuracy']:.4%}`",
                "",
                "Worst benchmark path cases for the best repeat:",
                "",
                write_markdown_table(
                    worst_paths,
                    [
                        ("material", "Material"),
                        ("path_kind", "Path"),
                        ("strength_reduction", "SRF"),
                        ("stress_mae", "Stress MAE"),
                        ("stress_max_abs", "Stress max abs"),
                        ("principal_mae", "Principal MAE"),
                        ("principal_max_abs", "Principal max abs"),
                    ],
                ),
                "",
                "## Insights",
                "",
                "- The larger network plus cosine scheduling and LBFGS made the training stable, but the improvement over the earlier tuned demo saturated quickly.",
                "- Across the repeated runs, randomness had very little effect on the final test MAE. The configuration appears highly repeatable.",
                "- The broad high-coverage dataset is materially harder than the earlier focused demo dataset, which explains why the absolute error is higher than in the narrower notebook run.",
                "- Benchmark path errors remain largest on left-edge cases and on weaker materials, which is consistent with the branch geometry being hardest there.",
                "",
                "## Replacement Assessment",
                "",
                "This study gives a stronger local approximation than the earlier notebook, and it is stable across repeated runs. However, based on these results alone, it is **not yet justified to claim that the surrogate can replace the constitutive relationship in production code while preserving the same limit-analysis result**.",
                "",
                "Reasons:",
                "",
                f"- Even the best repeated run still has local test stress MAE around `{best_repeat['test_stress_mae']:.3f}` and worst benchmark principal MAE around `{best_repeat['benchmark_worst_principal_mae']:.3f}`.",
                "- The study validates the local constitutive map only. It does not insert the surrogate into the FE limit-analysis loop and measure the resulting factor of safety, collapse multiplier, plastic-zone shape, or Newton/continuation behavior.",
                "- Limit analysis can be sensitive to local errors near branch transitions and in the active plastic zone, so local surrogate accuracy alone is not sufficient to guarantee the same global result.",
                "",
                "Current judgment:",
                "",
                "- Good enough for continued surrogate research and for controlled FE integration tests.",
                "- Not enough evidence yet for a drop-in replacement when the requirement is to keep the final limit-analysis result the same.",
                "",
                "Recommended next step:",
                "",
                "Run the surrogate inside the actual limit-analysis code on a small benchmark set and compare factor of safety, load multiplier, iteration counts, and plastic-zone localization against the exact constitutive call.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report_md)


if __name__ == "__main__":
    main()
