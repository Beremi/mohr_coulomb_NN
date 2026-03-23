#!/usr/bin/env python
"""Build a self-contained model card for the current best cover-layer route."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            out.append(parsed)
    return out


def _count_params(checkpoint_path: Path) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return int(sum(t.numel() for t in ckpt["model_state_dict"].values()))


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt(value: float, ndigits: int = 4) -> str:
    return f"{value:.{ndigits}f}"


def _rel_path(doc_path: Path, target: Path) -> str:
    return os.path.relpath(target, start=doc_path.parent)


def _stress_bin_labels(dissection: dict[str, Any]) -> list[str]:
    return [
        f"{row['stress_mag_lo']:.0f}-{row['stress_mag_hi']:.0f}"
        for row in dissection["stress_magnitude_bins"]
    ]


def _plot_architecture(output_path: Path, param_counts: dict[str, int]) -> Path:
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.5,
            facecolor=color,
            edgecolor="#243746",
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.035, title, fontsize=13, fontweight="bold", va="top")
        ax.text(x + 0.02, y + h - 0.085, body, fontsize=10.5, va="top", family="monospace")

    def arrow(x0: float, y0: float, x1: float, y1: float, text: str = "") -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#243746"))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, text, ha="center", va="bottom", fontsize=10)

    box(
        0.03,
        0.58,
        0.21,
        0.29,
        "Input Features",
        "Engineering strain E (6)\n[e11,e22,e33,g12,g13,g23]\n+\nReduced material (5)\n[log c_bar, atanh sin(phi),\n log G, log K, log lambda]\n=> raw feature vector (11)",
        "#E9F5DB",
    )
    box(
        0.29,
        0.62,
        0.24,
        0.24,
        "Baseline Network",
        "RawStressBranchNet\nLinear 11->1024 + GELU\n6 x ResidualBlock(1024)\nStress head: 1024->6\nBranch head: 1024->5\nparams: " + _fmt_int(param_counts["baseline"]),
        "#DCEBFA",
    )
    box(
        0.29,
        0.28,
        0.24,
        0.24,
        "Gate Network",
        "GateNet\nLinear 11->512 + GELU\n4 x ResidualBlock(512)\nGate head: 512->5 logits\nthreshold = 0.65\nparams: " + _fmt_int(param_counts["gate_raw"]),
        "#FDECC8",
    )
    box(
        0.58,
        0.63,
        0.19,
        0.23,
        "Frozen Experts",
        "smooth expert\nRawStressNet\n11->512\n6 residual blocks\nhead 512->6\nparams each:\n" + _fmt_int(param_counts["smooth"]) + "\n\napex expert\n" + _fmt_int(param_counts["apex"]),
        "#F8E1E7",
    )
    box(
        0.58,
        0.30,
        0.20,
        0.28,
        "Hard-Mined Edge Experts",
        "left_edge expert\nRawStressNet 11->512\n6 residual blocks\nhead 512->6\nparams: "
        + _fmt_int(param_counts["left_hard"])
        + "\n\nright_edge expert\nparams: "
        + _fmt_int(param_counts["right_hard"]),
        "#F9DCC4",
    )
    box(
        0.82,
        0.42,
        0.16,
        0.34,
        "Routing Logic",
        "1. run baseline + gate\n2. if gate conf < 0.65:\n   use baseline stress\n3. else if branch == elastic:\n   use exact trial stress\n4. else dispatch to\n   branch expert\nOutput: stress (6)\n+ gate branch (5 classes)",
        "#E4F0D0",
    )
    box(
        0.04,
        0.06,
        0.92,
        0.13,
        "Residual Block Detail",
        "LayerNorm(width) -> Linear(width,width) -> GELU -> Dropout(0.0) -> Linear(width,width) -> residual add -> LayerNorm(width) -> GELU",
        "#F5F5F5",
    )

    arrow(0.24, 0.74, 0.29, 0.74, "raw(11)")
    arrow(0.24, 0.40, 0.29, 0.40, "raw(11)")
    arrow(0.53, 0.74, 0.58, 0.74, "fallback stress")
    arrow(0.53, 0.40, 0.58, 0.46, "gate branch")
    arrow(0.77, 0.74, 0.82, 0.67, "smooth/apex")
    arrow(0.78, 0.44, 0.82, 0.56, "left/right")
    arrow(0.53, 0.72, 0.82, 0.60, "baseline stress")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_convergence(
    output_path: Path,
    baseline_hist: list[dict[str, float]],
    gate_hist: list[dict[str, float]],
    left_histories: dict[str, list[dict[str, float]]],
    right_histories: dict[str, list[dict[str, float]]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = [row["epoch"] for row in baseline_hist]
    ax = axes[0, 0]
    ax.plot(epochs, [row["val_stress_mse"] for row in baseline_hist], label="val stress MSE", color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val stress MSE")
    ax.set_title("Baseline raw_branch convergence")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(epochs, [row["val_branch_accuracy"] for row in baseline_hist], label="val branch acc", color="#d62728")
    ax2.set_ylabel("val branch accuracy")

    epochs = [row["epoch"] for row in gate_hist]
    ax = axes[0, 1]
    ax.plot(epochs, [row["val_loss"] for row in gate_hist], label="val CE loss", color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title("Gate raw convergence")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(epochs, [row["val_macro_recall"] for row in gate_hist], label="val macro recall", color="#2ca02c")
    ax2.set_ylabel("val macro recall")

    colors = {"control": "#1f77b4", "tail_weighted": "#ff7f0e", "hard_mined": "#2ca02c"}
    ax = axes[1, 0]
    for key, hist in left_histories.items():
        ax.plot(
            [row["epoch"] for row in hist],
            [row["val_weighted_rmse"] for row in hist],
            label=key,
            color=colors[key],
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val weighted RMSE")
    ax.set_title("left_edge expert variants")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for key, hist in right_histories.items():
        ax.plot(
            [row["epoch"] for row in hist],
            [row["val_weighted_rmse"] for row in hist],
            label=key,
            color=colors[key],
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val weighted RMSE")
    ax.set_title("right_edge expert variants")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_breakdown(
    output_path: Path,
    title_prefix: str,
    baseline_metrics: dict[str, Any],
    baseline_dissection: dict[str, Any] | None,
    winner_metrics: dict[str, Any],
    winner_dissection: dict[str, Any] | None,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metric_names = ["stress_mae", "stress_rmse", "relative_error_p90", "relative_error_p99"]
    labels = ["MAE", "RMSE", "p90 rel", "p99 rel"]
    base_vals = [baseline_metrics[name] for name in metric_names]
    win_vals = [winner_metrics[name] for name in metric_names]
    x = np.arange(len(labels))
    width = 0.36
    ax = axes[0, 0]
    ax.bar(x - width / 2, base_vals, width, label="baseline")
    ax.bar(x + width / 2, win_vals, width, label="winner")
    ax.set_xticks(x, labels)
    ax.set_title(f"{title_prefix}: overall metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    branch_labels = list(winner_metrics["per_branch_stress_mae"].keys())
    base_branch = [baseline_metrics["per_branch_stress_mae"][k] for k in branch_labels]
    win_branch = [winner_metrics["per_branch_stress_mae"][k] for k in branch_labels]
    x = np.arange(len(branch_labels))
    ax = axes[0, 1]
    ax.bar(x - width / 2, base_branch, width, label="baseline")
    ax.bar(x + width / 2, win_branch, width, label="winner")
    ax.set_xticks(x, branch_labels, rotation=25, ha="right")
    ax.set_title(f"{title_prefix}: per-branch stress MAE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    if baseline_dissection is not None and winner_dissection is not None:
        bin_labels = _stress_bin_labels(winner_dissection)
        base_bins = [row["sample_mae"] for row in baseline_dissection["stress_magnitude_bins"]]
        win_bins = [row["sample_mae"] for row in winner_dissection["stress_magnitude_bins"]]
        x = np.arange(len(bin_labels))
        ax = axes[1, 0]
        ax.bar(x - width / 2, base_bins, width, label="baseline")
        ax.bar(x + width / 2, win_bins, width, label="winner")
        ax.set_xticks(x, bin_labels, rotation=25, ha="right")
        ax.set_title(f"{title_prefix}: sample MAE by stress bin")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        rel_labels = list(winner_dissection["per_branch_mean_relative"].keys())
        base_rel = [baseline_dissection["per_branch_mean_relative"][k] for k in rel_labels]
        win_rel = [winner_dissection["per_branch_mean_relative"][k] for k in rel_labels]
        x = np.arange(len(rel_labels))
        ax = axes[1, 1]
        ax.bar(x - width / 2, base_rel, width, label="baseline")
        ax.bar(x + width / 2, win_rel, width, label="winner")
        ax.set_xticks(x, rel_labels, rotation=25, ha="right")
        ax.set_title(f"{title_prefix}: mean relative error by branch")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_dataset_counts(path: Path) -> dict[str, Any]:
    split_names = {0: "train", 1: "val", 2: "test"}
    with h5py.File(path, "r") as f:
        split_id = f["split_id"][:]
        branch_id = f["branch_id"][:]
    out = {
        "total": int(split_id.shape[0]),
        "splits": {split_names[i]: int(np.sum(split_id == i)) for i in split_names},
        "branches": {name: int(np.sum(branch_id == i)) for i, name in enumerate(["elastic", "smooth", "left_edge", "right_edge", "apex"])},
    }
    return out


def main() -> None:
    output_root = ROOT / "experiment_runs" / "real_sim" / "cover_layer_model_card_20260314"
    output_root.mkdir(parents=True, exist_ok=True)
    doc_path = ROOT / "docs" / "cover_layer_best_candidate_model_card.md"

    routes = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "routes_summary.json")
    tail_summary = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "training_hard_mined_summary.json")
    gate_raw_summary = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_gate_experiments_20260313" / "gate_raw" / "summary.json")
    gate_threshold = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_gate_experiments_20260313" / "gate_raw_threshold_selection.json")
    controls = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "controls_summary.json")
    mining_summary = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "mining_summary.json")

    baseline_real_dataset = ROOT / "experiment_runs" / "real_sim" / "cover_layer_single_material_20260313" / "cover_layer_full_real_exact_256.h5"
    synth_dataset = ROOT / "experiment_runs" / "real_sim" / "cover_layer_single_material_20260313" / "cover_layer_full_synthetic_holdout.h5"
    left_dataset = ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "datasets" / "left_edge_dataset.h5"
    right_dataset = ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "datasets" / "right_edge_dataset.h5"

    baseline_route = controls["modes"][0]
    winner_route = routes["routes"][0]
    alt_route = routes["routes"][1]

    baseline_real_metrics = baseline_route["real"]
    baseline_synth_metrics = baseline_route["synthetic"]
    baseline_real_dissection = _load_json(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "controls" / "baseline_reference" / "real" / "dissection.json")

    winner_real_metrics = winner_route["real"]
    winner_synth_metrics = winner_route["synthetic"]
    winner_real_dissection = winner_route["real_dissection"]

    param_counts = {
        "baseline": _count_params(ROOT / "experiment_runs" / "real_sim" / "cover_layer_single_material_20260313" / "baseline_raw_branch" / "best.pt"),
        "gate_raw": _count_params(ROOT / "experiment_runs" / "real_sim" / "cover_layer_gate_experiments_20260313" / "gate_raw" / "best.pt"),
        "smooth": _count_params(ROOT / "experiment_runs" / "real_sim" / "cover_layer_branch_experts_20260313" / "expert_smooth" / "best.pt"),
        "apex": _count_params(ROOT / "experiment_runs" / "real_sim" / "cover_layer_branch_experts_20260313" / "expert_apex" / "best.pt"),
        "left_hard": _count_params(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "left_edge_edge_hard_mined" / "best.pt"),
        "right_hard": _count_params(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "right_edge_edge_hard_mined" / "best.pt"),
    }
    deployed_param_total = param_counts["baseline"] + param_counts["gate_raw"] + param_counts["smooth"] + param_counts["apex"] + param_counts["left_hard"] + param_counts["right_hard"]

    arch_path = _plot_architecture(output_root / "architecture_overview.png", param_counts)
    convergence_path = _plot_convergence(
        output_root / "component_convergence.png",
        baseline_hist=_load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_single_material_20260313" / "baseline_raw_branch" / "history.csv"),
        gate_hist=_load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_gate_experiments_20260313" / "gate_raw" / "history.csv"),
        left_histories={
            "control": _load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "left_edge_edge_control" / "history.csv"),
            "tail_weighted": _load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "left_edge_edge_tail_weighted" / "history.csv"),
            "hard_mined": _load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "left_edge_edge_hard_mined" / "history.csv"),
        },
        right_histories={
            "control": _load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "right_edge_edge_control" / "history.csv"),
            "tail_weighted": _load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "right_edge_edge_tail_weighted" / "history.csv"),
            "hard_mined": _load_history(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "right_edge_edge_hard_mined" / "history.csv"),
        },
    )
    real_breakdown_path = _plot_breakdown(
        output_root / "real_breakdown.png",
        "Real holdout",
        baseline_real_metrics,
        baseline_real_dissection,
        winner_real_metrics,
        winner_real_dissection,
    )
    synth_breakdown_path = _plot_breakdown(
        output_root / "synthetic_breakdown.png",
        "Synthetic holdout",
        baseline_synth_metrics,
        None,
        winner_synth_metrics,
        None,
    )

    real_counts = _load_dataset_counts(baseline_real_dataset)
    synth_counts = _load_dataset_counts(synth_dataset)
    left_counts = _load_dataset_counts(left_dataset)
    right_counts = _load_dataset_counts(right_dataset)

    from mc_surrogate.real_materials import default_slope_material_specs

    cover = next(spec for spec in default_slope_material_specs() if spec.name == "cover_layer")

    rel_real = {
        "mae": 1.0 - (winner_real_metrics["stress_mae"] / baseline_real_metrics["stress_mae"]),
        "rmse": 1.0 - (winner_real_metrics["stress_rmse"] / baseline_real_metrics["stress_rmse"]),
        "p99": 1.0 - (winner_real_metrics["relative_error_p99"] / baseline_real_metrics["relative_error_p99"]),
        "edge": 1.0 - (winner_real_metrics["edge_combined_mae"] / baseline_real_metrics["edge_combined_mae"]),
    }
    rel_synth = {
        "mae": 1.0 - (winner_synth_metrics["stress_mae"] / baseline_synth_metrics["stress_mae"]),
        "rmse": 1.0 - (winner_synth_metrics["stress_rmse"] / baseline_synth_metrics["stress_rmse"]),
    }

    baseline_ckpt = torch.load(ROOT / "experiment_runs" / "real_sim" / "cover_layer_single_material_20260313" / "baseline_raw_branch" / "best.pt", map_location="cpu")["metadata"]["config"]
    left_hard_ckpt = torch.load(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "left_edge_edge_hard_mined" / "best.pt", map_location="cpu")["metadata"]["config"]
    right_hard_ckpt = torch.load(ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "right_edge_edge_hard_mined" / "best.pt", map_location="cpu")["metadata"]["config"]
    smooth_ckpt = torch.load(ROOT / "experiment_runs" / "real_sim" / "cover_layer_branch_experts_20260313" / "expert_smooth" / "best.pt", map_location="cpu")["metadata"]["config"]
    apex_ckpt = torch.load(ROOT / "experiment_runs" / "real_sim" / "cover_layer_branch_experts_20260313" / "expert_apex" / "best.pt", map_location="cpu")["metadata"]["config"]
    gate_meta = torch.load(ROOT / "experiment_runs" / "real_sim" / "cover_layer_gate_experiments_20260313" / "gate_raw" / "best.pt", map_location="cpu")["metadata"]

    doc = f"""# Cover-Layer Best-Candidate Model Card

This model card documents the current best deployable cover-layer surrogate route:
`edge_hard_mined_gate_raw_threshold_t0.65`.

It is intended to be readable on its own and sharable with collaborators who were not involved in the training loop.

## 1. At A Glance

- **Target operator:** fixed-material cover-layer constitutive surrogate from engineering strain `E in R^6` to stress `S in R^6`
- **Winning route:** raw gate with confidence threshold `0.65`, baseline fallback, exact elastic trial for branch `elastic`, frozen `smooth/apex` experts, hard-mined `left_edge/right_edge` experts
- **Primary validation target:** real exact holdout from the exported slope-stability constitutive calls
- **Status:** best local candidate so far; strong enough for a solver shadow test, not yet proven safe for constitutive replacement

## 2. Material Scope

This card is for the **cover-layer** material family only, using the raw material family recovered from the slope model:

| Material | c0 | phi [deg] | psi [deg] | Young | Poisson |
|---|---:|---:|---:|---:|---:|
| cover_layer | {cover.c0:.1f} | {cover.phi_deg:.1f} | {cover.psi_deg:.1f} | {cover.young:.1f} | {cover.poisson:.2f} |

Important nuance: the model family is fixed to this raw material family, but the reduced material inputs still vary with the strength-reduction factor captured in the real export. So the last five features are **not** constant over the dataset.

## 3. System Architecture

![Architecture overview]({_rel_path(doc_path, arch_path)})

### 3.1 Inputs

Every neural component in the winning route uses the same raw feature vector of length `11`:

| Index | Feature | Notes |
|---|---|---|
| 1-6 | `[e11, e22, e33, g12, g13, g23]` | engineering strain in Voigt form |
| 7 | `log(c_bar)` | reduced cohesion |
| 8 | `atanh(sin(phi_bar))` | reduced friction term |
| 9 | `log(G)` | shear modulus |
| 10 | `log(K)` | bulk modulus |
| 11 | `log(lambda)` | Lamé parameter |

### 3.2 Neural Components

| Component | Role | Input Dim | Hidden Width | Residual Blocks | Output | Activation | Parameters |
|---|---|---:|---:|---:|---|---|---:|
| `baseline_raw_branch` | fallback stress model + auxiliary branch head | 11 | 1024 | 6 | stress(6) + branch logits(5) | GELU | {_fmt_int(param_counts["baseline"])} |
| `gate_raw` | route selector | 11 | 512 | 4 | branch logits(5) | GELU | {_fmt_int(param_counts["gate_raw"])} |
| `smooth` expert | plastic expert for `smooth` | 11 | 512 | 6 | stress(6) | GELU | {_fmt_int(param_counts["smooth"])} |
| `apex` expert | plastic expert for `apex` | 11 | 512 | 6 | stress(6) | GELU | {_fmt_int(param_counts["apex"])} |
| `left_edge` expert | plastic expert for `left_edge` | 11 | 512 | 6 | stress(6) | GELU | {_fmt_int(param_counts["left_hard"])} |
| `right_edge` expert | plastic expert for `right_edge` | 11 | 512 | 6 | stress(6) | GELU | {_fmt_int(param_counts["right_hard"])} |

### 3.3 Residual Block

All MLP components use the same tabular residual block:

`LayerNorm(width) -> Linear(width,width) -> GELU -> Dropout(0) -> Linear(width,width) -> residual add -> LayerNorm(width) -> GELU`

### 3.4 Final Routing Logic

For each sample:

1. Compute raw features.
2. Run `baseline_raw_branch` and `gate_raw`.
3. If `max softmax(gate_raw) < 0.65`, return **baseline stress**.
4. Else, route by the predicted branch:
   - `elastic`: return **exact elastic trial stress**
   - `smooth`: return **smooth expert**
   - `left_edge`: return **left_edge hard-mined expert**
   - `right_edge`: return **right_edge hard-mined expert**
   - `apex`: return **apex expert**

Total parameters stored in the current deployment bundle: **{_fmt_int(deployed_param_total)}**.

## 4. How It Was Trained

This final route is not one monolithic end-to-end training run. It is a staged system assembled from separately trained components.

### 4.1 Datasets

| Dataset | Path | Total | Train | Val | Test | Purpose |
|---|---|---:|---:|---:|---:|---|
| real exact cover-layer | `{baseline_real_dataset}` | {_fmt_int(real_counts["total"])} | {_fmt_int(real_counts["splits"]["train"])} | {_fmt_int(real_counts["splits"]["val"])} | {_fmt_int(real_counts["splits"]["test"])} | main real supervision |
| synthetic U/B holdout | `{synth_dataset}` | {_fmt_int(synth_counts["total"])} | {_fmt_int(synth_counts["splits"]["train"])} | {_fmt_int(synth_counts["splits"]["val"])} | {_fmt_int(synth_counts["splits"]["test"])} | auxiliary generalization check |
| left_edge branch dataset | `{left_dataset}` | {_fmt_int(left_counts["total"])} | {_fmt_int(left_counts["splits"]["train"])} | {_fmt_int(left_counts["splits"]["val"])} | {_fmt_int(left_counts["splits"]["test"])} | edge-expert retraining |
| right_edge branch dataset | `{right_dataset}` | {_fmt_int(right_counts["total"])} | {_fmt_int(right_counts["splits"]["train"])} | {_fmt_int(right_counts["splits"]["val"])} | {_fmt_int(right_counts["splits"]["test"])} | edge-expert retraining |

Training supervision was **predominantly real exact data**, not synthetic data:

- `baseline_raw_branch`: trained on the real exact cover-layer dataset
- `gate_raw`: trained on the same real exact cover-layer dataset
- `smooth` and `apex` experts: trained on branch-filtered real exact datasets
- `left_edge` and `right_edge` hard-mined experts: trained on branch-filtered real exact datasets plus mined replay / local augmentation

The synthetic `U/B` dataset was **not** used as the main supervision source for the winning route. It is kept as an auxiliary test of domain coverage.

### 4.2 Training Stages

| Stage | Component(s) | Data | Optimizer / Schedule | Notes |
|---|---|---|---|---|
| A | `baseline_raw_branch` | real exact cover-layer | AdamW, plateau LR `3e-4 -> 1e-6`, LBFGS tail | direct stress + branch auxiliary head |
| B | `smooth`, `left_edge`, `right_edge`, `apex` experts | branch-filtered real exact | AdamW, plateau LR `3e-4 -> 1e-6`, LBFGS tail | first branch-specialized experts |
| C | `gate_raw` | real exact cover-layer | CE training on raw features | threshold later selected by validation MAE |
| D | `left_edge` + `right_edge` hard-mined experts | branch-filtered real exact + mined replay | AdamW, 3 cycles over batch `64->128->256->512->1024`, plateau drops by `0.5`, floor `1e-6` | this is the tail-safety phase |

### 4.3 Baseline Training Configuration

`baseline_raw_branch` config:

- model kind: `{baseline_ckpt["model_kind"]}`
- width/depth: `{baseline_ckpt["width"]} x {baseline_ckpt["depth"]}`
- epochs cap: `{baseline_ckpt["epochs"]}`
- batch size: `{baseline_ckpt["batch_size"]}`
- initial LR: `{baseline_ckpt["lr"]}`
- scheduler: `{baseline_ckpt["scheduler_kind"]}`
- plateau factor / patience: `{baseline_ckpt["plateau_factor"]}` / `{baseline_ckpt["plateau_patience"]}`
- patience: `{baseline_ckpt["patience"]}`
- min LR: `{baseline_ckpt["min_lr"]}`
- weight decay: `{baseline_ckpt["weight_decay"]}`
- LBFGS tail: `{baseline_ckpt["lbfgs_epochs"]}` epochs at LR `{baseline_ckpt["lbfgs_lr"]}`
- branch loss weight: `{baseline_ckpt["branch_loss_weight"]}`

### 4.4 Gate Training Configuration

`gate_raw` config:

- feature kind: `{gate_meta["feature_kind"]}`
- width/depth: `{gate_meta["width"]} x {gate_meta["depth"]}`
- input dim: `{gate_meta["input_dim"]}`
- objective: branch classification with macro-recall-driven checkpointing
- selected threshold: `{gate_threshold["best_threshold"]}` from validation stress MAE

### 4.5 Hard-Mined Edge Retraining

`left_edge` hard-mined config:

- width/depth: `{left_hard_ckpt["width"]} x {left_hard_ckpt["depth"]}`
- cycles: `{left_hard_ckpt["cycles"]}`
- batch schedule: `{left_hard_ckpt["batch_sizes"]}`
- base LR / floor: `{left_hard_ckpt["base_lr"]}` / `{left_hard_ckpt["min_lr"]}`
- plateau patience: `{left_hard_ckpt["plateau_patience"]}`
- stage patience: `{left_hard_ckpt["stage_patience"]}`
- replay ratio: `{left_hard_ckpt["hard_replay_ratio"]}`
- local noise scale: `{left_hard_ckpt["augment_noise_scale"]}`

`right_edge` hard-mined config is the same structure with a different seed (`{right_hard_ckpt["seed"]}`).

Mining table sizes used in the tail-safety phase:

| Mining Table | Rows | Val Rows | Test Rows | Relative-Error Threshold |
|---|---:|---:|---:|---:|
| left_edge | {_fmt_int(mining_summary["branches"]["left_edge"]["n_rows"])} | {_fmt_int(mining_summary["branches"]["left_edge"]["val_rows"])} | {_fmt_int(mining_summary["branches"]["left_edge"]["test_rows"])} | {_fmt(mining_summary["branches"]["left_edge"]["rel_threshold"])} |
| right_edge | {_fmt_int(mining_summary["branches"]["right_edge"]["n_rows"])} | {_fmt_int(mining_summary["branches"]["right_edge"]["val_rows"])} | {_fmt_int(mining_summary["branches"]["right_edge"]["test_rows"])} | {_fmt(mining_summary["branches"]["right_edge"]["rel_threshold"])} |

Important caveat: these mining tables were intentionally built from the fixed validation/test holdout of the current route. That made the tail-safety experiment effective, but it also means the resulting gains are **holdout-informed** and should be treated as optimistic until they are confirmed in a solver shadow test or on a fresh untouched split.

## 5. Training Convergence

![Component convergence]({_rel_path(doc_path, convergence_path)})

How to read this figure:

- **Top-left:** the baseline’s validation stress MSE falls steadily while branch accuracy rises. This is the original direct model that all routed variants fall back to.
- **Top-right:** the raw gate keeps improving macro recall while validation CE loss drops. This is why the final route uses `gate_raw` rather than `gate_trial`.
- **Bottom-left / bottom-right:** the tail-safety result is visible directly. The `edge_hard_mined` curves collapse the validation weighted RMSE for both edge branches, while `edge_control` and `edge_tail_weighted` plateau much higher.

What matters most in this convergence plot is not just that the losses go down, but **which curves separate**:

- the baseline converges to a strong general fallback
- the gate converges to a reliable branch selector
- the hard-mined edge experts are the only variant that materially changes the edge-tail regime

## 6. Validation Summary

### 6.1 Real Holdout: Winner vs Baseline

![Real breakdown]({_rel_path(doc_path, real_breakdown_path)})

| Metric | baseline_reference | winner route | Relative improvement |
|---|---:|---:|---:|
| stress MAE | {_fmt(baseline_real_metrics["stress_mae"])} | {_fmt(winner_real_metrics["stress_mae"])} | {100.0 * rel_real["mae"]:.1f}% |
| stress RMSE | {_fmt(baseline_real_metrics["stress_rmse"])} | {_fmt(winner_real_metrics["stress_rmse"])} | {100.0 * rel_real["rmse"]:.1f}% |
| p90 relative error | {_fmt(baseline_real_metrics["relative_error_p90"])} | {_fmt(winner_real_metrics["relative_error_p90"])} | {100.0 * (1.0 - winner_real_metrics["relative_error_p90"] / baseline_real_metrics["relative_error_p90"]):.1f}% |
| p99 relative error | {_fmt(baseline_real_metrics["relative_error_p99"])} | {_fmt(winner_real_metrics["relative_error_p99"])} | {100.0 * rel_real["p99"]:.1f}% |
| edge combined MAE | {_fmt(baseline_real_metrics["edge_combined_mae"])} | {_fmt(winner_real_metrics["edge_combined_mae"])} | {100.0 * rel_real["edge"]:.1f}% |
| branch accuracy | {_fmt(baseline_real_metrics["branch_accuracy"])} | {_fmt(winner_real_metrics["branch_accuracy"])} | {100.0 * (winner_real_metrics["branch_accuracy"] - baseline_real_metrics["branch_accuracy"]):.1f} pts |

### 6.2 Synthetic Holdout: Winner vs Baseline

![Synthetic breakdown]({_rel_path(doc_path, synth_breakdown_path)})

| Metric | baseline_reference | winner route | Relative improvement |
|---|---:|---:|---:|
| stress MAE | {_fmt(baseline_synth_metrics["stress_mae"])} | {_fmt(winner_synth_metrics["stress_mae"])} | {100.0 * rel_synth["mae"]:.1f}% |
| stress RMSE | {_fmt(baseline_synth_metrics["stress_rmse"])} | {_fmt(winner_synth_metrics["stress_rmse"])} | {100.0 * rel_synth["rmse"]:.1f}% |
| branch accuracy | {_fmt(baseline_synth_metrics["branch_accuracy"])} | {_fmt(winner_synth_metrics["branch_accuracy"])} | {100.0 * (winner_synth_metrics["branch_accuracy"] - baseline_synth_metrics["branch_accuracy"]):.1f} pts |

Interpretation:

- The real holdout is the primary target, and the winner is dramatically better there.
- The synthetic holdout also improves, but much less cleanly, especially for `left_edge`. This says the route is strongly tuned to the real exported cover-layer distribution, not universally solved across all synthetic regimes.

## 7. Detailed Real Validation

### 7.1 Winner Route Figures

- parity: ![winner real parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/parity.png)
- relative-error CDF: ![winner real cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/relative_error_cdf.png)
- error vs stress magnitude: ![winner real mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/error_vs_magnitude.png)
- branch confusion: ![winner real branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/real/branch_confusion.png)

### 7.2 Baseline Reference Figures

- parity: ![baseline real parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/parity.png)
- relative-error CDF: ![baseline real cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/relative_error_cdf.png)
- error vs stress magnitude: ![baseline real mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/error_vs_magnitude.png)
- branch confusion: ![baseline real branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/real/branch_confusion.png)

### 7.3 Real Holdout Error Tables

#### Per-Branch Stress MAE

| Branch | baseline_reference | winner route |
|---|---:|---:|
| elastic | {_fmt(baseline_real_metrics["per_branch_stress_mae"]["elastic"])} | {_fmt(winner_real_metrics["per_branch_stress_mae"]["elastic"])} |
| smooth | {_fmt(baseline_real_metrics["per_branch_stress_mae"]["smooth"])} | {_fmt(winner_real_metrics["per_branch_stress_mae"]["smooth"])} |
| left_edge | {_fmt(baseline_real_metrics["per_branch_stress_mae"]["left_edge"])} | {_fmt(winner_real_metrics["per_branch_stress_mae"]["left_edge"])} |
| right_edge | {_fmt(baseline_real_metrics["per_branch_stress_mae"]["right_edge"])} | {_fmt(winner_real_metrics["per_branch_stress_mae"]["right_edge"])} |
| apex | {_fmt(baseline_real_metrics["per_branch_stress_mae"]["apex"])} | {_fmt(winner_real_metrics["per_branch_stress_mae"]["apex"])} |

#### Stress-Magnitude-Bin Sample MAE

| Stress bin | baseline_reference | winner route |
|---|---:|---:|
"""

    for base_row, win_row in zip(baseline_real_dissection["stress_magnitude_bins"], winner_real_dissection["stress_magnitude_bins"]):
        label = f"{base_row['stress_mag_lo']:.0f} to {base_row['stress_mag_hi']:.0f}"
        doc += f"| {label} | {_fmt(base_row['sample_mae'])} | {_fmt(win_row['sample_mae'])} |\n"

    doc += f"""

#### Per-Branch Mean Relative Error

| Branch | baseline_reference | winner route |
|---|---:|---:|
"""
    for branch in winner_real_dissection["per_branch_mean_relative"]:
        doc += f"| {branch} | {_fmt(baseline_real_dissection['per_branch_mean_relative'][branch])} | {_fmt(winner_real_dissection['per_branch_mean_relative'][branch])} |\n"

    doc += """

#### Worst Real Holdout Calls For The Winner

| Call | N | Component MAE | Sample MAE | Mean Relative |
|---|---:|---:|---:|---:|
"""
    for row in winner_real_dissection["worst_calls_top10"]:
        doc += f"| {row['call_name']} | {row['n']} | {_fmt(row['component_mae'])} | {_fmt(row['sample_mae'])} | {_fmt(row['mean_relative'])} |\n"

    doc += f"""

## 8. Detailed Synthetic Validation

### 8.1 Winner Route Figures

- parity: ![winner synth parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/parity.png)
- relative-error CDF: ![winner synth cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/relative_error_cdf.png)
- error vs stress magnitude: ![winner synth mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/error_vs_magnitude.png)
- branch confusion: ![winner synth branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes/edge_hard_mined_gate_raw_threshold_t0.65/synthetic/branch_confusion.png)

### 8.2 Baseline Reference Figures

- parity: ![baseline synth parity](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/parity.png)
- relative-error CDF: ![baseline synth cdf](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/relative_error_cdf.png)
- error vs stress magnitude: ![baseline synth mag](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/error_vs_magnitude.png)
- branch confusion: ![baseline synth branch](../experiment_runs/real_sim/cover_layer_tail_safety_20260313/controls/baseline_reference/synthetic/branch_confusion.png)

### 8.3 Synthetic Holdout Per-Branch Stress MAE

| Branch | baseline_reference | winner route |
|---|---:|---:|
| elastic | {_fmt(baseline_synth_metrics["per_branch_stress_mae"]["elastic"])} | {_fmt(winner_synth_metrics["per_branch_stress_mae"]["elastic"])} |
| smooth | {_fmt(baseline_synth_metrics["per_branch_stress_mae"]["smooth"])} | {_fmt(winner_synth_metrics["per_branch_stress_mae"]["smooth"])} |
| left_edge | {_fmt(baseline_synth_metrics["per_branch_stress_mae"]["left_edge"])} | {_fmt(winner_synth_metrics["per_branch_stress_mae"]["left_edge"])} |
| right_edge | {_fmt(baseline_synth_metrics["per_branch_stress_mae"]["right_edge"])} | {_fmt(winner_synth_metrics["per_branch_stress_mae"]["right_edge"])} |
| apex | {_fmt(baseline_synth_metrics["per_branch_stress_mae"]["apex"])} | {_fmt(winner_synth_metrics["per_branch_stress_mae"]["apex"])} |

## 9. What To Tell Others

If you need a short summary for collaborators:

- The current best cover-layer surrogate is **not a single network**. It is a routed system composed of a strong baseline model, a dedicated raw gate, and branch-specific experts.
- It was trained mostly on **real exact relabeled constitutive-call data**, not on purely synthetic samples.
- The biggest gain came from **hard-mined retraining of the `left_edge` and `right_edge` experts**.
- On the real holdout, it improves over the direct baseline from MAE/RMSE `{_fmt(baseline_real_metrics["stress_mae"])}/{_fmt(baseline_real_metrics["stress_rmse"])}` to `{_fmt(winner_real_metrics["stress_mae"])}/{_fmt(winner_real_metrics["stress_rmse"])}`.
- The route is strong enough for a **solver shadow test**, but not yet proven safe for constitutive replacement because the hard-mining phase was holdout-informed.

## 10. Caveats

1. The strongest edge-expert gains came from holdout-informed mining. This makes the result useful, but optimistic.
2. The real holdout is where the route clearly wins. The synthetic holdout still shows large `left_edge` errors, so the route is not a universal fix over all synthetic regimes.
3. The current evaluation script computes all experts in batch and then routes for convenience. A deployment implementation should route more efficiently.
4. This card documents the **best local candidate**, not a solver-validated constitutive replacement.

## 11. Key Artifact Paths

- winning route metrics: `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "routes_summary.json"}`
- tail-safety execution report: `{ROOT / "docs" / "cover_layer_tail_safety_execution.md"}`
- baseline checkpoint: `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_single_material_20260313" / "baseline_raw_branch" / "best.pt"}`
- gate checkpoint: `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_gate_experiments_20260313" / "gate_raw" / "best.pt"}`
- left/right hard-mined checkpoints:
  - `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "left_edge_edge_hard_mined" / "best.pt"}`
  - `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_tail_safety_20260313" / "experts" / "right_edge_edge_hard_mined" / "best.pt"}`
- frozen smooth/apex checkpoints:
  - `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_branch_experts_20260313" / "expert_smooth" / "best.pt"}`
  - `{ROOT / "experiment_runs" / "real_sim" / "cover_layer_branch_experts_20260313" / "expert_apex" / "best.pt"}`
"""

    doc_path.write_text(doc, encoding="utf-8")
    print(doc_path)


if __name__ == "__main__":
    main()
