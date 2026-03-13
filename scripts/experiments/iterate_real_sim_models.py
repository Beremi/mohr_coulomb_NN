#!/usr/bin/env python
"""Run the next targeted real-simulation experiments and collect metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Primary real sampled dataset used for train/val/test.")
    parser.add_argument(
        "--cross-dataset",
        help="Optional second real sampled dataset used only for cross-evaluation of the trained checkpoints.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    return parser.parse_args()


def _evaluate(summary: dict[str, object], dataset: str, device: str, batch_size: int) -> dict[str, object]:
    result = evaluate_checkpoint_on_dataset(
        summary["best_checkpoint"],
        dataset,
        split="test",
        device=device,
        batch_size=batch_size,
    )
    return {
        "metrics": result["metrics"],
        "prediction_keys": sorted(result["predictions"].keys()),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "raw_branch_w512_d6",
            "model_kind": "raw_branch",
            "width": 512,
            "depth": 6,
            "epochs": 220,
            "branch_loss_weight": 0.10,
            "lbfgs_epochs": 6,
            "seed": 301,
        },
        {
            "name": "trial_raw_residual_w512_d6",
            "model_kind": "trial_raw_residual",
            "width": 512,
            "depth": 6,
            "epochs": 220,
            "branch_loss_weight": 0.0,
            "lbfgs_epochs": 6,
            "seed": 302,
        },
        {
            "name": "trial_raw_branch_residual_w512_d6",
            "model_kind": "trial_raw_branch_residual",
            "width": 512,
            "depth": 6,
            "epochs": 220,
            "branch_loss_weight": 0.10,
            "lbfgs_epochs": 6,
            "seed": 303,
        },
    ]

    records: list[dict[str, object]] = []
    for cfg in configs:
        run_dir = output_root / cfg["name"]
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = train_model(
                TrainingConfig(
                    dataset=args.dataset,
                    run_dir=str(run_dir),
                    model_kind=str(cfg["model_kind"]),
                    epochs=int(cfg["epochs"]),
                    batch_size=args.train_batch_size,
                    lr=1.0e-3,
                    weight_decay=5.0e-5,
                    width=int(cfg["width"]),
                    depth=int(cfg["depth"]),
                    dropout=0.0,
                    seed=int(cfg["seed"]),
                    patience=60,
                    grad_clip=1.0,
                    branch_loss_weight=float(cfg["branch_loss_weight"]),
                    num_workers=0,
                    device=args.device,
                    scheduler_kind="plateau",
                    min_lr=1.0e-6,
                    plateau_factor=0.5,
                    plateau_patience=15,
                    lbfgs_epochs=int(cfg["lbfgs_epochs"]),
                    lbfgs_lr=0.2,
                    lbfgs_max_iter=12,
                    lbfgs_history_size=50,
                    log_every_epochs=20,
                )
            )

        primary_eval = _evaluate(summary, args.dataset, args.device, args.eval_batch_size)
        (run_dir / "primary_test_metrics.json").write_text(
            json.dumps(primary_eval["metrics"], indent=2),
            encoding="utf-8",
        )

        record: dict[str, object] = {
            "name": cfg["name"],
            "model_kind": cfg["model_kind"],
            "summary": summary,
            "primary_test": primary_eval["metrics"],
        }

        if args.cross_dataset:
            cross_eval = _evaluate(summary, args.cross_dataset, args.device, args.eval_batch_size)
            cross_dir = run_dir / "cross_test_metrics.json"
            cross_dir.write_text(json.dumps(cross_eval["metrics"], indent=2), encoding="utf-8")
            record["cross_test"] = cross_eval["metrics"]

        records.append(record)

    def score(item: dict[str, object]) -> tuple[float, float]:
        primary = item["primary_test"]
        return float(primary["stress_mae"]), float(primary["stress_rmse"])

    records.sort(key=score)
    (output_root / "iteration_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
