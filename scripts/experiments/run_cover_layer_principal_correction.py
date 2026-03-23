#!/usr/bin/env python
"""Train corrected principal-space plastic-correction models for the cover layer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.training import TrainingConfig, train_model

sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from run_cover_layer_single_material_plan import (  # noqa: E402
    _compute_real_dissection,
    _evaluate_and_plot,
    _json_safe,
    _plot_metric_bars,
    _plot_per_branch_mae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        default="experiment_runs/real_sim/cover_layer_single_material_20260313",
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/cover_layer_principal_correction_20260313",
    )
    parser.add_argument(
        "--report-md",
        default="docs/cover_layer_principal_correction.md",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=700)
    parser.add_argument("--plateau-patience", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--lbfgs-epochs", type=int, default=8)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--log-every-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1441)
    return parser.parse_args()


def _train_one(
    *,
    experiment_name: str,
    dataset_path: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_root / experiment_name
    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_branch_residual",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        dropout=0.0,
        seed=args.seed,
        patience=args.patience,
        grad_clip=1.0,
        branch_loss_weight=args.branch_loss_weight,
        num_workers=0,
        device=args.device,
        scheduler_kind="plateau",
        min_lr=args.min_lr,
        plateau_factor=0.5,
        plateau_patience=args.plateau_patience,
        lbfgs_epochs=args.lbfgs_epochs,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_max_iter=20,
        lbfgs_history_size=100,
        log_every_epochs=args.log_every_epochs,
        stress_weight_alpha=0.0,
        stress_weight_scale=250.0,
        checkpoint_metric="stress_mae",
    )
    summary = train_model(config)
    (run_dir / "config_used.json").write_text(json.dumps(_json_safe(asdict(config)), indent=2), encoding="utf-8")
    return summary


def _write_report(
    *,
    report_path: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
    real_dissection: dict[str, Any],
    reference_root: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Cover-Layer Principal Correction")
    lines.append("")
    lines.append("This report tests the corrected structured next step after the single-material execution study:")
    lines.append("")
    lines.append("- principal-space plastic correction")
    lines.append("- exact elastic gating through the branch head")
    lines.append("- plastic-only regression loss")
    lines.append("- checkpoint selection by real validation stress MAE")
    lines.append("")
    lines.append("Reference control from the previous study:")
    lines.append(f"- baseline checkpoint: `{reference_root / 'baseline_raw_branch' / 'best.pt'}`")
    lines.append("")
    lines.append("## Result Table")
    lines.append("")
    lines.append("| Experiment | Real MAE | Real RMSE | Real Max Abs | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        real = row["real"]
        synth = row["synthetic"]
        lines.append(
            f"| {row['name']} | {real['stress_mae']:.4f} | {real['stress_rmse']:.4f} | {real['stress_max_abs']:.4f} | "
            f"{real.get('branch_accuracy', float('nan')):.4f} | {synth['stress_mae']:.4f} | {synth['stress_rmse']:.4f} | "
            f"{synth.get('branch_accuracy', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append(f"![Metric comparison]({(output_root / 'comparison_metrics.png').as_posix()})")
    lines.append("")
    lines.append(f"![Per-branch MAE]({(output_root / 'comparison_per_branch.png').as_posix()})")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for row in rows:
        exp_dir = Path(row["run_dir"])
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- history: ![history]({(exp_dir / 'history_log.png').as_posix()})")
        lines.append(f"- real parity: ![real parity]({(exp_dir / 'eval' / 'real_parity.png').as_posix()})")
        lines.append(f"- real relative error: ![real rel]({(exp_dir / 'eval' / 'real_relative_error_cdf.png').as_posix()})")
        lines.append(f"- real error vs magnitude: ![real mag]({(exp_dir / 'eval' / 'real_error_vs_magnitude.png').as_posix()})")
        lines.append(f"- real branch confusion: ![real branch]({(exp_dir / 'eval' / 'real_branch_confusion.png').as_posix()})")
        lines.append(f"- synthetic parity: ![synth parity]({(exp_dir / 'eval' / 'synthetic_parity.png').as_posix()})")
        lines.append(f"- synthetic relative error: ![synth rel]({(exp_dir / 'eval' / 'synthetic_relative_error_cdf.png').as_posix()})")
        lines.append(f"- synthetic error vs magnitude: ![synth mag]({(exp_dir / 'eval' / 'synthetic_error_vs_magnitude.png').as_posix()})")
        lines.append(f"- synthetic branch confusion: ![synth branch]({(exp_dir / 'eval' / 'synthetic_branch_confusion.png').as_posix()})")
        lines.append("")
    lines.append("## Best-Model Real Holdout Dissection")
    lines.append("")
    lines.append(f"- mean relative sample error: `{real_dissection['relative_error_mean']:.4f}`")
    lines.append(f"- median relative sample error: `{real_dissection['relative_error_median']:.4f}`")
    lines.append(f"- p90 relative sample error: `{real_dissection['relative_error_p90']:.4f}`")
    lines.append(f"- p99 relative sample error: `{real_dissection['relative_error_p99']:.4f}`")
    lines.append("")
    lines.append("### Per-Branch Mean Relative Error")
    lines.append("")
    for name, value in real_dissection["per_branch_mean_relative"].items():
        lines.append(f"- `{name}`: `{value:.4f}`")
    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append("This experiment is successful only if the corrected principal-space model closes the gap to the old direct baseline on real stress metrics, not just on normalized training loss.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- If the corrected real-only model beats the old baseline, the core issue was target/metric design, not the structured idea itself.")
    lines.append("- If the hybrid corrected model then improves further, `U/B` augmentation becomes worth keeping.")
    lines.append("- If both remain behind the old baseline, the next move should be a different structured target altogether rather than more tuning of this residual family.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reference_root = (ROOT / args.reference_root).resolve()
    output_root = ROOT / args.output_root
    report_path = ROOT / args.report_md
    output_root.mkdir(parents=True, exist_ok=True)

    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    synthetic_holdout = reference_root / "cover_layer_full_synthetic_holdout.h5"
    hybrid_dataset = reference_root / "cover_layer_full_hybrid_train.h5"

    baseline_row = _evaluate_and_plot(
        name="baseline_raw_branch_reference",
        checkpoint_path=reference_root / "baseline_raw_branch" / "best.pt",
        real_dataset_path=real_dataset,
        synthetic_dataset_path=synthetic_holdout,
        output_dir=output_root / "baseline_raw_branch_reference" / "eval",
        device=args.device,
    )
    baseline_row["run_dir"] = str(reference_root / "baseline_raw_branch")

    experiments = (
        ("principal_trial_branch_residual_real", real_dataset),
        ("principal_trial_branch_residual_hybrid", hybrid_dataset),
    )
    rows = [baseline_row]

    from run_cover_layer_single_material_plan import _plot_history_log  # noqa: E402

    for idx, (name, dataset_path) in enumerate(experiments):
        run_dir = output_root / name
        if not (run_dir / "best.pt").exists():
            local_args = argparse.Namespace(**vars(args))
            local_args.seed = args.seed + idx
            _train_one(experiment_name=name, dataset_path=dataset_path, output_root=output_root, args=local_args)
            _plot_history_log(run_dir / "history.csv", run_dir / "history_log.png", title=name)
        row = _evaluate_and_plot(
            name=name,
            checkpoint_path=run_dir / "best.pt",
            real_dataset_path=real_dataset,
            synthetic_dataset_path=synthetic_holdout,
            output_dir=run_dir / "eval",
            device=args.device,
        )
        row["run_dir"] = str(run_dir)
        rows.append(row)

    _plot_metric_bars(rows, output_root / "comparison_metrics.png")
    _plot_per_branch_mae(rows, output_root / "comparison_per_branch.png")

    best_row = min(rows, key=lambda row: row["real"]["stress_mae"])
    best_predictions = best_row.pop("real_predictions")
    real_dissection = _compute_real_dissection(real_dataset_path=real_dataset, predictions=best_predictions)
    (output_root / "real_dissection.json").write_text(json.dumps(_json_safe(real_dissection), indent=2), encoding="utf-8")

    _write_report(
        report_path=report_path,
        output_root=output_root,
        rows=rows,
        real_dissection=real_dissection,
        reference_root=reference_root,
    )


if __name__ == "__main__":
    main()
