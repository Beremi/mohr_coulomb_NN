#!/usr/bin/env python
"""Run a small architecture sweep on the sampled real-simulation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "principal_w512_d6",
            "model_kind": "principal",
            "width": 512,
            "depth": 6,
            "epochs": 160,
            "branch_loss_weight": 0.1,
            "lbfgs_epochs": 4,
        },
        {
            "name": "raw_w512_d6",
            "model_kind": "raw",
            "width": 512,
            "depth": 6,
            "epochs": 160,
            "branch_loss_weight": 0.0,
            "lbfgs_epochs": 4,
        },
        {
            "name": "trial_raw_w512_d6",
            "model_kind": "trial_raw",
            "width": 512,
            "depth": 6,
            "epochs": 160,
            "branch_loss_weight": 0.0,
            "lbfgs_epochs": 4,
        },
    ]

    results: list[dict[str, object]] = []
    for seed, cfg in enumerate(configs, start=1):
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
                    seed=100 + seed,
                    patience=40,
                    grad_clip=1.0,
                    branch_loss_weight=float(cfg["branch_loss_weight"]),
                    num_workers=0,
                    device=args.device,
                    scheduler_kind="plateau",
                    min_lr=1.0e-6,
                    plateau_factor=0.5,
                    plateau_patience=12,
                    lbfgs_epochs=int(cfg["lbfgs_epochs"]),
                    lbfgs_lr=0.2,
                    lbfgs_max_iter=12,
                    lbfgs_history_size=50,
                    log_every_epochs=20,
                )
            )
        eval_result = evaluate_checkpoint_on_dataset(
            summary["best_checkpoint"],
            args.dataset,
            split="test",
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        record = {
            "name": cfg["name"],
            "model_kind": cfg["model_kind"],
            "summary": summary,
            "metrics": eval_result["metrics"],
        }
        (run_dir / "real_test_metrics.json").write_text(json.dumps(eval_result["metrics"], indent=2), encoding="utf-8")
        results.append(record)

    results.sort(key=lambda item: float(item["metrics"]["stress_mae"]))
    (output_root / "sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
