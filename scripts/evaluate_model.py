#!/usr/bin/env python
"""Evaluate a trained surrogate on a dataset split and save plots/metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from mc_surrogate.training import evaluate_checkpoint_on_dataset
from mc_surrogate.viz import (
    branch_confusion_plot,
    error_histogram,
    parity_plot,
    save_metrics_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_checkpoint_on_dataset(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        split=args.split,
        device=args.device,
    )
    metrics = result["metrics"]
    arrays = result["arrays"]
    pred = result["predictions"]

    save_metrics_json(metrics, out_dir / "metrics.json")
    parity_plot(arrays["stress"], pred["stress"], out_dir / "parity_stress.png", label="stress")
    error_histogram(pred["stress"] - arrays["stress"], out_dir / "stress_error_hist.png", label="stress error")

    if "stress_principal" in pred:
        parity_plot(arrays["stress_principal"], pred["stress_principal"], out_dir / "parity_principal.png", label="principal stress")

    if "branch_confusion" in metrics:
        branch_confusion_plot(metrics["branch_confusion"], out_dir / "branch_confusion.png")

    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
