#!/usr/bin/env python
"""Train a neural surrogate from an HDF5 constitutive dataset."""

from __future__ import annotations

import argparse
import json

from mc_surrogate.training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-kind", choices=("principal", "raw"), default="principal")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        dataset=args.dataset,
        run_dir=args.run_dir,
        model_kind=args.model_kind,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        seed=args.seed,
        patience=args.patience,
        grad_clip=args.grad_clip,
        branch_loss_weight=args.branch_loss_weight,
        num_workers=args.num_workers,
        device=args.device,
    )
    summary = train_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
