#!/usr/bin/env python
"""Run a longer, larger-capacity sweep for the real-simulation raw-branch model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-256", required=True)
    parser.add_argument("--dataset-512", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--limit-runs", type=int, default=0, help="If > 0, run only the first N configs.")
    return parser.parse_args()


def _evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    result = evaluate_checkpoint_on_dataset(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        split="test",
        device=device,
        batch_size=batch_size,
    )
    return result["metrics"]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "d256": args.dataset_256,
        "d512": args.dataset_512,
    }

    configs = [
        {
            "name": "rb_d256_w512_d6_long",
            "dataset_key": "d256",
            "width": 512,
            "depth": 6,
            "batch_size": 4096,
            "lr": 1.0e-3,
            "epochs": 900,
            "patience": 220,
            "plateau_patience": 30,
            "lbfgs_epochs": 12,
            "seed": 401,
        },
        {
            "name": "rb_d512_w512_d6_long",
            "dataset_key": "d512",
            "width": 512,
            "depth": 6,
            "batch_size": 4096,
            "lr": 1.0e-3,
            "epochs": 900,
            "patience": 220,
            "plateau_patience": 30,
            "lbfgs_epochs": 12,
            "seed": 402,
        },
        {
            "name": "rb_d256_w768_d8_long",
            "dataset_key": "d256",
            "width": 768,
            "depth": 8,
            "batch_size": 3072,
            "lr": 8.0e-4,
            "epochs": 1000,
            "patience": 260,
            "plateau_patience": 35,
            "lbfgs_epochs": 12,
            "seed": 403,
        },
        {
            "name": "rb_d512_w768_d8_long",
            "dataset_key": "d512",
            "width": 768,
            "depth": 8,
            "batch_size": 3072,
            "lr": 8.0e-4,
            "epochs": 1000,
            "patience": 260,
            "plateau_patience": 35,
            "lbfgs_epochs": 12,
            "seed": 404,
        },
        {
            "name": "rb_d256_w1024_d8_long",
            "dataset_key": "d256",
            "width": 1024,
            "depth": 8,
            "batch_size": 2048,
            "lr": 6.0e-4,
            "epochs": 1200,
            "patience": 320,
            "plateau_patience": 40,
            "lbfgs_epochs": 12,
            "seed": 405,
        },
        {
            "name": "rb_d512_w1024_d8_long",
            "dataset_key": "d512",
            "width": 1024,
            "depth": 8,
            "batch_size": 2048,
            "lr": 6.0e-4,
            "epochs": 1200,
            "patience": 320,
            "plateau_patience": 40,
            "lbfgs_epochs": 12,
            "seed": 406,
        },
    ]

    if args.limit_runs > 0:
        configs = configs[: args.limit_runs]

    records: list[dict[str, object]] = []
    for cfg in configs:
        run_dir = output_root / cfg["name"]
        summary_path = run_dir / "summary.json"
        dataset_path = datasets[str(cfg["dataset_key"])]

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = train_model(
                TrainingConfig(
                    dataset=dataset_path,
                    run_dir=str(run_dir),
                    model_kind="raw_branch",
                    epochs=int(cfg["epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    lr=float(cfg["lr"]),
                    weight_decay=2.0e-5,
                    width=int(cfg["width"]),
                    depth=int(cfg["depth"]),
                    dropout=0.0,
                    seed=int(cfg["seed"]),
                    patience=int(cfg["patience"]),
                    grad_clip=1.0,
                    branch_loss_weight=0.1,
                    num_workers=0,
                    device=args.device,
                    scheduler_kind="plateau",
                    min_lr=1.0e-7,
                    plateau_factor=0.5,
                    plateau_patience=int(cfg["plateau_patience"]),
                    lbfgs_epochs=int(cfg["lbfgs_epochs"]),
                    lbfgs_lr=0.15,
                    lbfgs_max_iter=16,
                    lbfgs_history_size=100,
                    log_every_epochs=50,
                )
            )

        evals = {}
        for dataset_name, eval_dataset in datasets.items():
            metrics = _evaluate_checkpoint(summary["best_checkpoint"], eval_dataset, args.device, args.eval_batch_size)
            evals[dataset_name] = metrics
            (run_dir / f"{dataset_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        record = {
            "name": cfg["name"],
            "train_dataset": cfg["dataset_key"],
            "width": cfg["width"],
            "depth": cfg["depth"],
            "summary": summary,
            "evals": evals,
        }
        records.append(record)

    def score(item: dict[str, object]) -> tuple[float, float, float]:
        evals = item["evals"]
        d512 = evals["d512"]
        return (
            float(d512["stress_mae"]),
            float(d512["stress_rmse"]),
            float(d512["stress_max_abs"]),
        )

    records.sort(key=score)
    (output_root / "long_sweep_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
