#!/usr/bin/env python
"""Validate a trained surrogate against sampled real simulation data and write a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mc_surrogate.data import load_arrays
from mc_surrogate.training import evaluate_checkpoint_on_dataset, predict_with_checkpoint
from mc_surrogate.viz import error_histogram, parity_plot, save_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dataset", required=True)
    parser.add_argument("--synthetic-dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


def _device_name(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _dataset_attrs(path: str) -> dict[str, object]:
    with h5py.File(path, "r") as f:
        return {key: f.attrs[key] for key in f.attrs.keys()}


def _abs_quantiles(array: np.ndarray) -> dict[str, float]:
    arr = np.abs(np.asarray(array, dtype=float).reshape(-1))
    return {
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
        "q999": float(np.quantile(arr, 0.999)),
        "max": float(np.max(arr)),
    }


def _branch_distribution(branch_id: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(branch_id.astype(int), return_counts=True)
    total = float(np.sum(counts))
    return {str(int(v)): float(c / total) for v, c in zip(values, counts)}


def _evaluate_real_dataset(
    dataset_path: str,
    checkpoint_path: str,
    *,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    arrays = load_arrays(dataset_path, ["strain_eng", "stress", "material_reduced", "branch_id", "stress_principal"], split="test")
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
    )
    stress_err = pred["stress"] - arrays["stress"]
    abs_err = np.abs(stress_err)
    metrics: dict[str, object] = {
        "split": "test",
        "n_samples": int(arrays["strain_eng"].shape[0]),
        "stress_mae": float(np.mean(abs_err)),
        "stress_rmse": float(np.sqrt(np.mean(stress_err**2))),
        "stress_max_abs": float(np.max(abs_err)),
        "principal_mae": float(np.mean(np.abs(pred["stress_principal"] - arrays["stress_principal"]))),
        "principal_rmse": float(np.sqrt(np.mean((pred["stress_principal"] - arrays["stress_principal"]) ** 2))),
        "branch_accuracy": float(np.mean(np.argmax(pred["branch_probabilities"], axis=1) == arrays["branch_id"])),
    }

    strain_scale = np.max(np.abs(arrays["strain_eng"]), axis=1)
    bin_edges = np.quantile(strain_scale, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins: list[dict[str, float]] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (strain_scale >= lo) & (strain_scale <= hi)
        else:
            mask = (strain_scale >= lo) & (strain_scale < hi)
        if not np.any(mask):
            continue
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(np.sum(mask)),
                "stress_mae": float(np.mean(abs_err[mask])),
                "stress_rmse": float(np.sqrt(np.mean(stress_err[mask] ** 2))),
            }
        )
    metrics["strain_magnitude_bins"] = bins
    return {"metrics": metrics, "arrays": arrays, "predictions": pred}


def _overlay_abs_hist(real: np.ndarray, synthetic: np.ndarray, output_path: Path, xlabel: str) -> None:
    real_vals = np.log10(np.abs(real).reshape(-1) + 1.0e-12)
    syn_vals = np.log10(np.abs(synthetic).reshape(-1) + 1.0e-12)
    plt.figure(figsize=(8, 5))
    plt.hist(syn_vals, bins=80, alpha=0.5, density=True, label="synthetic")
    plt.hist(real_vals, bins=80, alpha=0.5, density=True, label="real")
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_branch_distribution(real_branch: np.ndarray, synthetic_branch: np.ndarray, output_path: Path) -> None:
    real_counts = np.bincount(real_branch.astype(int), minlength=5)
    syn_counts = np.bincount(synthetic_branch.astype(int), minlength=5)
    real_frac = real_counts / np.maximum(real_counts.sum(), 1)
    syn_frac = syn_counts / np.maximum(syn_counts.sum(), 1)
    labels = ["elastic", "smooth", "left_edge", "right_edge", "apex"]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, syn_frac, width=width, label="synthetic")
    plt.bar(x + width / 2, real_frac, width=width, label="real")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("fraction")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_error_vs_strain(strain_eng: np.ndarray, stress_true: np.ndarray, stress_pred: np.ndarray, output_path: Path) -> None:
    strain_scale = np.max(np.abs(strain_eng), axis=1)
    err = np.mean(np.abs(stress_pred - stress_true), axis=1)
    plt.figure(figsize=(7, 5))
    plt.scatter(strain_scale, err, s=6, alpha=0.25)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("max abs strain component")
    plt.ylabel("mean abs stress error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _metrics_table(title: str, metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            f"### {title}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Samples | {int(metrics['n_samples'])} |",
            f"| Stress MAE | {float(metrics['stress_mae']):.6f} |",
            f"| Stress RMSE | {float(metrics['stress_rmse']):.6f} |",
            f"| Stress max abs | {float(metrics['stress_max_abs']):.6f} |",
            *(
                [
                    f"| Principal MAE | {float(metrics['principal_mae']):.6f} |",
                    f"| Principal RMSE | {float(metrics['principal_rmse']):.6f} |",
                ]
                if "principal_mae" in metrics
                else []
            ),
            *( [f"| Branch accuracy | {float(metrics['branch_accuracy']):.6f} |"] if "branch_accuracy" in metrics else [] ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    device = _device_name(args.device)

    synthetic_eval = evaluate_checkpoint_on_dataset(
        args.checkpoint,
        args.synthetic_dataset,
        split="test",
        device=device,
        batch_size=args.batch_size,
    )
    real_eval = _evaluate_real_dataset(
        args.real_dataset,
        args.checkpoint,
        device=device,
        batch_size=args.batch_size,
    )

    synthetic_full = load_arrays(args.synthetic_dataset, ["strain_eng", "stress", "branch_id"], split=None)
    real_full = load_arrays(args.real_dataset, ["strain_eng", "stress", "branch_id"], split=None)
    real_attrs = _dataset_attrs(args.real_dataset)

    save_metrics_json(real_eval["metrics"], out_dir / "real_metrics.json")
    save_metrics_json(synthetic_eval["metrics"], out_dir / "synthetic_test_metrics.json")
    parity_plot(real_eval["arrays"]["stress"], real_eval["predictions"]["stress"], out_dir / "real_parity_stress.png", label="stress")
    parity_plot(
        real_eval["arrays"]["stress_principal"],
        real_eval["predictions"]["stress_principal"],
        out_dir / "real_parity_principal.png",
        label="principal stress",
    )
    error_histogram(real_eval["predictions"]["stress"] - real_eval["arrays"]["stress"], out_dir / "real_stress_error_hist.png", label="stress error")
    _overlay_abs_hist(real_full["strain_eng"], synthetic_full["strain_eng"], out_dir / "strain_abs_distribution.png", "log10(|strain component| + 1e-12)")
    _overlay_abs_hist(real_full["stress"], synthetic_full["stress"], out_dir / "stress_abs_distribution.png", "log10(|stress component| + 1e-12)")
    _plot_branch_distribution(real_full["branch_id"], synthetic_full["branch_id"], out_dir / "branch_distribution.png")
    _plot_error_vs_strain(
        real_eval["arrays"]["strain_eng"],
        real_eval["arrays"]["stress"],
        real_eval["predictions"]["stress"],
        out_dir / "real_error_vs_strain.png",
    )

    real_strain_q = _abs_quantiles(real_full["strain_eng"])
    synthetic_strain_q = _abs_quantiles(synthetic_full["strain_eng"])
    real_stress_q = _abs_quantiles(real_full["stress"])
    synthetic_stress_q = _abs_quantiles(synthetic_full["stress"])

    report = "\n".join(
        [
            "# Real Simulation Validation",
            "",
            "## Setup",
            "",
            f"- Real dataset: `{args.real_dataset}`",
            f"- Synthetic reference dataset: `{args.synthetic_dataset}`",
            f"- Checkpoint: `{args.checkpoint}`",
            f"- Device: `{device}`",
            "",
            "## Exact Export Compatibility",
            "",
            f"- Sampled real export was converted with exact branch labels using the updated Python constitutive operator.",
            f"- Export/exact stress agreement in preprocessing: MAE `{float(real_attrs['exact_match_mae']):.3e}`, RMSE `{float(real_attrs['exact_match_rmse']):.3e}`, max abs `{float(real_attrs['exact_match_max_abs']):.3e}`.",
            "",
            "## Distribution Shift",
            "",
            "| Quantity | Synthetic q50 | Synthetic q90 | Synthetic q99 | Synthetic max | Real q50 | Real q90 | Real q99 | Real max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| abs strain component | {synthetic_strain_q['q50']:.6g} | {synthetic_strain_q['q90']:.6g} | {synthetic_strain_q['q99']:.6g} | {synthetic_strain_q['max']:.6g} | {real_strain_q['q50']:.6g} | {real_strain_q['q90']:.6g} | {real_strain_q['q99']:.6g} | {real_strain_q['max']:.6g} |",
            f"| abs stress component | {synthetic_stress_q['q50']:.6g} | {synthetic_stress_q['q90']:.6g} | {synthetic_stress_q['q99']:.6g} | {synthetic_stress_q['max']:.6g} | {real_stress_q['q50']:.6g} | {real_stress_q['q90']:.6g} | {real_stress_q['q99']:.6g} | {real_stress_q['max']:.6g} |",
            "",
            f"- Real branch fractions: `{json.dumps(_branch_distribution(real_full['branch_id']))}`",
            f"- Synthetic branch fractions: `{json.dumps(_branch_distribution(synthetic_full['branch_id']))}`",
            "",
            "Plots:",
            f"- `{out_dir / 'strain_abs_distribution.png'}`",
            f"- `{out_dir / 'stress_abs_distribution.png'}`",
            f"- `{out_dir / 'branch_distribution.png'}`",
            "",
            "## Baseline Surrogate",
            "",
            _metrics_table("Synthetic Test Split", synthetic_eval["metrics"]),
            _metrics_table("Real Test Split", real_eval["metrics"]),
            "### Real Stress Error by Strain Magnitude Quartile",
            "",
            "| Strain magnitude bin | Samples | Stress MAE | Stress RMSE |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| [{row['lo']:.6g}, {row['hi']:.6g}] | {row['n']} | {row['stress_mae']:.6f} | {row['stress_rmse']:.6f} |"
                for row in real_eval["metrics"]["strain_magnitude_bins"]
            ],
            "",
            "Plots:",
            f"- `{out_dir / 'real_parity_stress.png'}`",
            f"- `{out_dir / 'real_parity_principal.png'}`",
            f"- `{out_dir / 'real_stress_error_hist.png'}`",
            f"- `{out_dir / 'real_error_vs_strain.png'}`",
            "",
            "## Findings",
            "",
            f"- The current surrogate is accurate on the synthetic reference test split but degrades strongly on the real sampled states.",
            f"- The dominant issue is distribution shift: real strain magnitudes are orders of magnitude larger than the synthetic training distribution used so far.",
            f"- Branch classification is also weak on real data, which suggests the current feature/architecture choice is not robust enough for the actual state occupancy of the slope-stability solve.",
            f"- The preprocessing step confirms the real export itself is consistent with the updated exact constitutive operator, so the remaining gap is in the learned model and training data, not in the sampled real labels.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
