"""Visualization helpers for training/evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .mohr_coulomb import BRANCH_NAMES


def load_history_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a CSV training history into numpy arrays."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    out = {}
    for key in rows[0].keys():
        out[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return out


def plot_training_history(history_csv: str | Path, output_path: str | Path) -> Path:
    """Plot train/val loss curves."""
    hist = load_history_csv(history_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["train_loss"], label="train loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    *,
    label: str = "stress",
    max_points: int = 4000,
) -> Path:
    """Scatter parity plot for predicted versus true values."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(y_true.size, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.4)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel(f"true {label}")
    plt.ylabel(f"predicted {label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def error_histogram(errors: np.ndarray, output_path: str | Path, *, label: str = "stress error") -> Path:
    """Histogram of prediction errors."""
    errors = np.asarray(errors).reshape(-1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=60)
    plt.xlabel(label)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def branch_confusion_plot(confusion: Sequence[Sequence[int]], output_path: str | Path) -> Path:
    """Plot branch confusion matrix."""
    mat = np.asarray(confusion, dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(BRANCH_NAMES)), BRANCH_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(BRANCH_NAMES)), BRANCH_NAMES)
    plt.xlabel("predicted branch")
    plt.ylabel("true branch")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_path_comparison(
    path_parameter: np.ndarray,
    exact: np.ndarray,
    predicted: np.ndarray,
    output_path: str | Path,
    *,
    labels: tuple[str, str, str] = ("sigma1", "sigma2", "sigma3"),
    title: str = "Path comparison",
) -> Path:
    """Plot principal stress paths exact vs predicted."""
    path_parameter = np.asarray(path_parameter)
    exact = np.asarray(exact)
    predicted = np.asarray(predicted)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    for i in range(exact.shape[1]):
        plt.plot(path_parameter, exact[:, i], label=f"exact {labels[i]}")
        plt.plot(path_parameter, predicted[:, i], "--", label=f"pred {labels[i]}")
    plt.xlabel("path parameter")
    plt.ylabel("principal stress")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def save_metrics_json(metrics: dict, output_path: str | Path) -> Path:
    """Save metrics as pretty JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
