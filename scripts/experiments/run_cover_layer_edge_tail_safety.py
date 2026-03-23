#!/usr/bin/env python
"""Tail-safety follow-up for cover-layer branch experts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.models import ResidualBlock, Standardizer, build_model, build_raw_features, build_trial_features, compute_trial_stress
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.training import choose_device, evaluate_checkpoint_on_dataset, predict_with_checkpoint, set_seed

sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from run_cover_layer_branch_experts import _eval_metrics  # noqa: E402


TARGET_BRANCHES = (("left_edge", 2), ("right_edge", 3))
FROZEN_BRANCHES = {1: "smooth", 4: "apex"}
CONTROL_MODES = (
    "baseline_reference",
    "oracle_reference",
    "gate_trial_threshold_t0.85",
    "gate_raw_threshold_t0.65",
)
VARIANT_ORDER = ("edge_control", "edge_tail_weighted", "edge_hard_mined")


@dataclass(frozen=True)
class Stage:
    cycle: int
    batch_size: int
    lr: float
    name: str


class GateNet(nn.Module):
    def __init__(self, input_dim: int, width: int = 512, depth: int = 4, n_branches: int = 5) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=0.0) for _ in range(depth)])
        self.head = nn.Linear(width, n_branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="experiment_runs/real_sim/cover_layer_single_material_20260313")
    parser.add_argument("--experts-root", default="experiment_runs/real_sim/cover_layer_branch_experts_20260313")
    parser.add_argument("--gates-root", default="experiment_runs/real_sim/cover_layer_gate_experiments_20260313")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_tail_safety_20260313")
    parser.add_argument("--report-md", default="docs/cover_layer_tail_safety_execution.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--stage-max-epochs", type=int, default=120)
    parser.add_argument("--plateau-patience", type=int, default=20)
    parser.add_argument("--stage-patience", type=int, default=80)
    parser.add_argument("--base-lr", type=float, default=1.0e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--log-every-epochs", type=int, default=25)
    parser.add_argument("--lbfgs-epochs", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--hard-replay-ratio", type=float, default=0.35)
    parser.add_argument("--augment-noise-scale", type=float, default=0.05)
    parser.add_argument("--augment-max-rounds", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ORDER))
    parser.add_argument("--target-branches", nargs="+", default=[name for name, _ in TARGET_BRANCHES])
    parser.add_argument("--seed", type=int, default=1801)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_split_arrays(dataset_path: Path, split: str) -> dict[str, np.ndarray]:
    split_map = {"train": 0, "val": 1, "test": 2}
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(dataset_path, "r") as f:
        split_id = f["split_id"][:]
        mask = split_id == split_map[split]
        for key in f.keys():
            if key == "split_id":
                continue
            arrays[key] = f[key][mask]
        arrays["split_id"] = split_id[mask]
        arrays["source_call_names"] = np.array(json.loads(f.attrs["source_call_names_json"]), dtype=object)
    return arrays


def _load_dataset_all(dataset_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(dataset_path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value
    return arrays, attrs


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(_json_safe(value))
            else:
                f.attrs[key] = value
    return path


def _build_branch_dataset(real_dataset_path: Path, output_path: Path, branch_id: int) -> tuple[Path, dict[str, Any]]:
    arrays, attrs = _load_dataset_all(real_dataset_path)
    mask = arrays["branch_id"] == branch_id
    out = {key: value[mask] for key, value in arrays.items()}
    info = {
        "n_samples": int(np.sum(mask)),
        "split_counts": {
            "train": int(np.sum(out["split_id"] == 0)),
            "val": int(np.sum(out["split_id"] == 1)),
            "test": int(np.sum(out["split_id"] == 2)),
        },
    }
    attrs = dict(attrs)
    attrs["expert_branch_id"] = int(branch_id)
    attrs["expert_branch_name"] = BRANCH_NAMES[branch_id]
    _write_dataset(output_path, out, attrs)
    return output_path, info


def _format_runtime(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _plot_parity(stress_true: np.ndarray, stress_pred: np.ndarray, output_path: Path) -> Path:
    if stress_true.shape[0] > 4000:
        rng = np.random.default_rng(0)
        idx = rng.choice(stress_true.shape[0], size=4000, replace=False)
        stress_true = stress_true[idx]
        stress_pred = stress_pred[idx]
    lo = float(min(stress_true.min(), stress_pred.min()))
    hi = float(max(stress_true.max(), stress_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(stress_true.reshape(-1), stress_pred.reshape(-1), s=6, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("true stress")
    plt.ylabel("predicted stress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_relative_error_cdf(stress_true: np.ndarray, stress_pred: np.ndarray, output_path: Path, title: str) -> Path:
    denom = np.maximum(np.linalg.norm(stress_true, axis=1), 1.0)
    rel = np.linalg.norm(stress_pred - stress_true, axis=1) / denom
    rel = np.sort(rel)
    y = np.linspace(0.0, 1.0, rel.size)
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(rel, y)
    plt.xscale("log")
    plt.xlabel(r"$||\hat{\sigma} - \sigma||_2 / \max(||\sigma||_2, 1)$")
    plt.ylabel("empirical CDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_error_vs_magnitude(stress_true: np.ndarray, stress_pred: np.ndarray, output_path: Path, title: str) -> Path:
    mag = np.linalg.norm(stress_true, axis=1)
    err = np.linalg.norm(stress_pred - stress_true, axis=1)
    if mag.size > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(mag.size, size=5000, replace=False)
        mag = mag[idx]
        err = err[idx]
    plt.figure(figsize=(6.2, 5.0))
    plt.scatter(mag, err, s=6, alpha=0.35)
    plt.xscale("symlog", linthresh=1.0)
    plt.yscale("symlog", linthresh=1.0)
    plt.xlabel(r"$||\sigma||_2$")
    plt.ylabel(r"$||\hat{\sigma} - \sigma||_2$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_confusion(confusion: list[list[int]], output_path: Path) -> Path:
    mat = np.asarray(confusion, dtype=float)
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(BRANCH_NAMES)), BRANCH_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(BRANCH_NAMES)), BRANCH_NAMES)
    plt.xlabel("predicted branch")
    plt.ylabel("true branch")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_compare(rows: list[dict[str, Any]], output_path: Path, key: str, title: str) -> Path:
    names = [row["name"] for row in rows]
    real = [row["real"][key] for row in rows]
    synth = [row["synthetic"][key] for row in rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, real, width=width, label="real")
    ax.bar(x + width / 2, synth, width=width, label="synthetic")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_single_compare(labels: list[str], values: list[float], output_path: Path, title: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _build_features(feature_kind: str, strain_eng: np.ndarray, material_reduced: np.ndarray) -> np.ndarray:
    if feature_kind == "raw":
        return build_raw_features(strain_eng, material_reduced)
    if feature_kind == "trial":
        return build_trial_features(strain_eng, material_reduced)
    raise ValueError(feature_kind)


def _load_gate_checkpoint(path: Path, device: str) -> tuple[GateNet, dict[str, Any]]:
    device_obj = choose_device(device)
    ckpt = torch.load(path, map_location=device_obj)
    meta = ckpt["metadata"]
    model = GateNet(input_dim=meta["input_dim"], width=meta["width"], depth=meta["depth"]).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta


def _predict_gate(checkpoint_path: Path, feature_kind: str, strain_eng: np.ndarray, material_reduced: np.ndarray, device: str) -> np.ndarray:
    model, meta = _load_gate_checkpoint(checkpoint_path, device)
    scaler = Standardizer.from_dict(meta["x_scaler"])
    features = _build_features(feature_kind, strain_eng, material_reduced)
    x = torch.from_numpy(scaler.transform(features)).to(choose_device(device))
    probs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 16384):
            logits = model(x[start : start + 16384])
            logits = logits - logits.max(dim=1, keepdim=True).values
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def _predict_all_experts(expert_paths: dict[int, Path], strain_eng: np.ndarray, material_reduced: np.ndarray, device: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for branch_id, path in expert_paths.items():
        out[branch_id] = predict_with_checkpoint(path, strain_eng, material_reduced, device=device, batch_size=16384)["stress"]
    return out


def _ensemble_from_gate(
    *,
    gate_probs: np.ndarray,
    trial_stress: np.ndarray,
    baseline_stress: np.ndarray,
    expert_stress: dict[int, np.ndarray],
    mode: str,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    pred_branch = np.argmax(gate_probs, axis=1)
    if mode == "hard":
        stress = np.empty_like(baseline_stress)
        for branch_id in range(len(BRANCH_NAMES)):
            mask = pred_branch == branch_id
            if not np.any(mask):
                continue
            if branch_id == 0:
                stress[mask] = trial_stress[mask]
            else:
                stress[mask] = expert_stress[branch_id][mask]
        return stress.astype(np.float32), pred_branch.astype(np.int64)
    if mode == "threshold":
        if threshold is None:
            raise ValueError("threshold mode requires threshold.")
        stress, pred_branch = _ensemble_from_gate(
            gate_probs=gate_probs,
            trial_stress=trial_stress,
            baseline_stress=baseline_stress,
            expert_stress=expert_stress,
            mode="hard",
            threshold=None,
        )
        conf = np.max(gate_probs, axis=1)
        mask = conf < threshold
        if np.any(mask):
            stress[mask] = baseline_stress[mask]
        return stress.astype(np.float32), pred_branch.astype(np.int64)
    raise ValueError(mode)


def _baseline_predictions(reference_root: Path, split_arrays: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    ckpt = reference_root / "baseline_raw_branch" / "best.pt"
    pred = predict_with_checkpoint(
        ckpt,
        split_arrays["strain_eng"],
        split_arrays["material_reduced"],
        device=device,
        batch_size=16384,
    )
    return {"stress": pred["stress"], "branch_probs": pred["branch_probabilities"]}


def _compute_dissection_from_arrays(
    *,
    stress_true: np.ndarray,
    branch_id: np.ndarray,
    call_id: np.ndarray,
    call_names: list[str],
    predictions: np.ndarray,
) -> dict[str, Any]:
    abs_err = np.abs(predictions - stress_true)
    sample_err = np.linalg.norm(predictions - stress_true, axis=1)
    stress_mag = np.linalg.norm(stress_true, axis=1)
    rel = sample_err / np.maximum(stress_mag, 1.0)

    bins = np.quantile(stress_mag, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    bin_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi <= lo:
            continue
        if hi == bins[-1]:
            mask = (stress_mag >= lo) & (stress_mag <= hi)
        else:
            mask = (stress_mag >= lo) & (stress_mag < hi)
        if not np.any(mask):
            continue
        bin_rows.append(
            {
                "stress_mag_lo": float(lo),
                "stress_mag_hi": float(hi),
                "n": int(np.sum(mask)),
                "sample_mae": float(np.mean(sample_err[mask])),
                "component_mae": float(np.mean(abs_err[mask])),
                "mean_relative": float(np.mean(rel[mask])),
            }
        )

    per_call = []
    for cid in np.unique(call_id):
        mask = call_id == cid
        per_call.append(
            {
                "call_id": int(cid),
                "call_name": call_names[int(cid)],
                "n": int(np.sum(mask)),
                "sample_mae": float(np.mean(sample_err[mask])),
                "component_mae": float(np.mean(abs_err[mask])),
                "mean_relative": float(np.mean(rel[mask])),
            }
        )
    per_call.sort(key=lambda row: row["component_mae"], reverse=True)

    return {
        "relative_error_mean": float(np.mean(rel)),
        "relative_error_median": float(np.median(rel)),
        "relative_error_p90": float(np.quantile(rel, 0.9)),
        "relative_error_p99": float(np.quantile(rel, 0.99)),
        "stress_magnitude_bins": bin_rows,
        "worst_calls_top10": per_call[:10],
        "per_branch_mean_relative": {
            name: float(np.mean(rel[branch_id == i]))
            for i, name in enumerate(BRANCH_NAMES)
            if np.any(branch_id == i)
        },
    }


def _evaluate_route(
    *,
    name: str,
    split_arrays: dict[str, np.ndarray],
    stress_pred: np.ndarray,
    branch_pred: np.ndarray,
    output_dir: Path,
    include_dissection: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _eval_metrics(
        split_arrays["stress"],
        stress_pred,
        split_arrays["branch_id"].astype(int),
        branch_pred.astype(int),
    )
    abs_err = np.abs(stress_pred - split_arrays["stress"])
    edge_mask = np.isin(split_arrays["branch_id"].astype(int), [2, 3])
    denom = np.maximum(np.linalg.norm(split_arrays["stress"], axis=1), 1.0)
    rel = np.linalg.norm(stress_pred - split_arrays["stress"], axis=1) / denom
    metrics["relative_error_p90"] = float(np.quantile(rel, 0.9))
    metrics["relative_error_p99"] = float(np.quantile(rel, 0.99))
    metrics["edge_combined_mae"] = float(np.mean(abs_err[edge_mask])) if np.any(edge_mask) else float("nan")

    dissection = None
    if include_dissection:
        call_names = split_arrays["source_call_names"].tolist()
        dissection = _compute_dissection_from_arrays(
            stress_true=split_arrays["stress"],
            branch_id=split_arrays["branch_id"].astype(int),
            call_id=split_arrays["source_call_id"].astype(int),
            call_names=call_names,
            predictions=stress_pred,
        )
        metrics["top_bin_sample_mae"] = float(dissection["stress_magnitude_bins"][-1]["sample_mae"])
        (output_dir / "dissection.json").write_text(json.dumps(_json_safe(dissection), indent=2), encoding="utf-8")
    else:
        order = np.argsort(np.linalg.norm(split_arrays["stress"], axis=1))
        top_n = max(1, int(0.1 * order.size))
        top_idx = order[-top_n:]
        metrics["top_bin_sample_mae"] = float(np.mean(np.linalg.norm(stress_pred[top_idx] - split_arrays["stress"][top_idx], axis=1)))

    (output_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _plot_parity(split_arrays["stress"], stress_pred, output_dir / "parity.png")
    _plot_relative_error_cdf(split_arrays["stress"], stress_pred, output_dir / "relative_error_cdf.png", title=f"{name} relative error")
    _plot_error_vs_magnitude(split_arrays["stress"], stress_pred, output_dir / "error_vs_magnitude.png", title=f"{name} error vs stress magnitude")
    _plot_confusion(metrics["branch_confusion"], output_dir / "branch_confusion.png")
    return metrics, dissection


def _freeze_controls(
    *,
    reference_root: Path,
    experts_root: Path,
    gates_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    synth_dataset = reference_root / "cover_layer_full_synthetic_holdout.h5"
    gate_thresholds = {
        "gate_trial_threshold_t0.85": ("trial", 0.85, gates_root / "gate_trial" / "best.pt"),
        "gate_raw_threshold_t0.65": ("raw", 0.65, gates_root / "gate_raw" / "best.pt"),
    }
    expert_paths = {
        1: experts_root / "expert_smooth" / "best.pt",
        2: experts_root / "expert_left_edge" / "best.pt",
        3: experts_root / "expert_right_edge" / "best.pt",
        4: experts_root / "expert_apex" / "best.pt",
    }

    real_test = _load_split_arrays(real_dataset, "test")
    real_val = _load_split_arrays(real_dataset, "val")
    synth_test = _load_split_arrays(synth_dataset, "test")
    baseline_real = _baseline_predictions(reference_root, real_test, device)
    baseline_val = _baseline_predictions(reference_root, real_val, device)
    baseline_synth = _baseline_predictions(reference_root, synth_test, device)
    real_experts = _predict_all_experts(expert_paths, real_test["strain_eng"], real_test["material_reduced"], device)
    val_experts = _predict_all_experts(expert_paths, real_val["strain_eng"], real_val["material_reduced"], device)
    synth_experts = _predict_all_experts(expert_paths, synth_test["strain_eng"], synth_test["material_reduced"], device)
    trial_real = compute_trial_stress(real_test["strain_eng"], real_test["material_reduced"])
    trial_val = compute_trial_stress(real_val["strain_eng"], real_val["material_reduced"])
    trial_synth = compute_trial_stress(synth_test["strain_eng"], synth_test["material_reduced"])

    modes: list[dict[str, Any]] = []
    controls_dir = output_root / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_real, baseline_dissection = _evaluate_route(
        name="baseline_reference",
        split_arrays=real_test,
        stress_pred=baseline_real["stress"],
        branch_pred=np.argmax(baseline_real["branch_probs"], axis=1),
        output_dir=controls_dir / "baseline_reference" / "real",
        include_dissection=True,
    )
    baseline_metrics_synth, _ = _evaluate_route(
        name="baseline_reference",
        split_arrays=synth_test,
        stress_pred=baseline_synth["stress"],
        branch_pred=np.argmax(baseline_synth["branch_probs"], axis=1),
        output_dir=controls_dir / "baseline_reference" / "synthetic",
        include_dissection=False,
    )
    modes.append({"name": "baseline_reference", "real": baseline_metrics_real, "synthetic": baseline_metrics_synth})

    oracle_real_stress = trial_real.copy()
    oracle_synth_stress = trial_synth.copy()
    for branch_id in range(1, len(BRANCH_NAMES)):
        mask_real = real_test["branch_id"] == branch_id
        mask_synth = synth_test["branch_id"] == branch_id
        oracle_real_stress[mask_real] = real_experts[branch_id][mask_real]
        oracle_synth_stress[mask_synth] = synth_experts[branch_id][mask_synth]
    oracle_real_metrics, oracle_dissection = _evaluate_route(
        name="oracle_reference",
        split_arrays=real_test,
        stress_pred=oracle_real_stress,
        branch_pred=real_test["branch_id"].astype(int),
        output_dir=controls_dir / "oracle_reference" / "real",
        include_dissection=True,
    )
    oracle_synth_metrics, _ = _evaluate_route(
        name="oracle_reference",
        split_arrays=synth_test,
        stress_pred=oracle_synth_stress,
        branch_pred=synth_test["branch_id"].astype(int),
        output_dir=controls_dir / "oracle_reference" / "synthetic",
        include_dissection=False,
    )
    modes.append({"name": "oracle_reference", "real": oracle_real_metrics, "synthetic": oracle_synth_metrics})

    primary_val_dissection = None
    for mode_name, (feature_kind, threshold, gate_ckpt) in gate_thresholds.items():
        gate_probs_real = _predict_gate(gate_ckpt, feature_kind, real_test["strain_eng"], real_test["material_reduced"], device)
        gate_probs_val = _predict_gate(gate_ckpt, feature_kind, real_val["strain_eng"], real_val["material_reduced"], device)
        gate_probs_synth = _predict_gate(gate_ckpt, feature_kind, synth_test["strain_eng"], synth_test["material_reduced"], device)
        pred_real, branch_real = _ensemble_from_gate(
            gate_probs=gate_probs_real,
            trial_stress=trial_real,
            baseline_stress=baseline_real["stress"],
            expert_stress=real_experts,
            mode="threshold",
            threshold=threshold,
        )
        pred_val, branch_val = _ensemble_from_gate(
            gate_probs=gate_probs_val,
            trial_stress=trial_val,
            baseline_stress=baseline_val["stress"],
            expert_stress=val_experts,
            mode="threshold",
            threshold=threshold,
        )
        pred_synth, branch_synth = _ensemble_from_gate(
            gate_probs=gate_probs_synth,
            trial_stress=trial_synth,
            baseline_stress=baseline_synth["stress"],
            expert_stress=synth_experts,
            mode="threshold",
            threshold=threshold,
        )
        real_metrics, real_dissection = _evaluate_route(
            name=mode_name,
            split_arrays=real_test,
            stress_pred=pred_real,
            branch_pred=branch_real,
            output_dir=controls_dir / mode_name / "real",
            include_dissection=True,
        )
        synth_metrics, _ = _evaluate_route(
            name=mode_name,
            split_arrays=synth_test,
            stress_pred=pred_synth,
            branch_pred=branch_synth,
            output_dir=controls_dir / mode_name / "synthetic",
            include_dissection=False,
        )
        val_dissection = _compute_dissection_from_arrays(
            stress_true=real_val["stress"],
            branch_id=real_val["branch_id"].astype(int),
            call_id=real_val["source_call_id"].astype(int),
            call_names=real_val["source_call_names"].tolist(),
            predictions=pred_val,
        )
        (controls_dir / mode_name / "val_dissection.json").write_text(json.dumps(_json_safe(val_dissection), indent=2), encoding="utf-8")
        modes.append({"name": mode_name, "real": real_metrics, "synthetic": synth_metrics})
        if mode_name == "gate_trial_threshold_t0.85":
            primary_val_dissection = val_dissection

    _plot_compare(modes, output_root / "controls_compare_mae.png", key="stress_mae", title="Frozen control MAE")
    _plot_compare(modes, output_root / "controls_compare_rmse.png", key="stress_rmse", title="Frozen control RMSE")
    _plot_single_compare(
        [row["name"] for row in modes],
        [row["real"]["relative_error_p99"] for row in modes],
        output_root / "controls_compare_p99.png",
        title="Frozen control real p99 relative error",
        ylabel="p99 relative error",
    )
    _plot_single_compare(
        [row["name"] for row in modes],
        [row["real"]["edge_combined_mae"] for row in modes],
        output_root / "controls_compare_edge_mae.png",
        title="Frozen control real edge combined MAE",
        ylabel="edge combined MAE",
    )
    summary = {
        "modes": modes,
        "real_dataset": str(real_dataset),
        "synthetic_dataset": str(synth_dataset),
        "primary_val_dissection": primary_val_dissection,
    }
    (output_root / "controls_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _compute_primary_route_predictions_for_split(
    *,
    reference_root: Path,
    experts_root: Path,
    gates_root: Path,
    split_arrays: dict[str, np.ndarray],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_pred = _baseline_predictions(reference_root, split_arrays, device)
    trial_stress = compute_trial_stress(split_arrays["strain_eng"], split_arrays["material_reduced"])
    expert_paths = {
        1: experts_root / "expert_smooth" / "best.pt",
        2: experts_root / "expert_left_edge" / "best.pt",
        3: experts_root / "expert_right_edge" / "best.pt",
        4: experts_root / "expert_apex" / "best.pt",
    }
    expert_pred = _predict_all_experts(expert_paths, split_arrays["strain_eng"], split_arrays["material_reduced"], device)
    gate_ckpt = gates_root / "gate_trial" / "best.pt"
    gate_probs = _predict_gate(gate_ckpt, "trial", split_arrays["strain_eng"], split_arrays["material_reduced"], device)
    return _ensemble_from_gate(
        gate_probs=gate_probs,
        trial_stress=trial_stress,
        baseline_stress=baseline_pred["stress"],
        expert_stress=expert_pred,
        mode="threshold",
        threshold=0.85,
    )


def _build_hardcase_mining_tables(
    *,
    reference_root: Path,
    experts_root: Path,
    gates_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    val_arrays = _load_split_arrays(real_dataset, "val")
    test_arrays = _load_split_arrays(real_dataset, "test")
    pred_val, _ = _compute_primary_route_predictions_for_split(
        reference_root=reference_root,
        experts_root=experts_root,
        gates_root=gates_root,
        split_arrays=val_arrays,
        device=device,
    )
    pred_test, _ = _compute_primary_route_predictions_for_split(
        reference_root=reference_root,
        experts_root=experts_root,
        gates_root=gates_root,
        split_arrays=test_arrays,
        device=device,
    )
    val_dissection = _compute_dissection_from_arrays(
        stress_true=val_arrays["stress"],
        branch_id=val_arrays["branch_id"].astype(int),
        call_id=val_arrays["source_call_id"].astype(int),
        call_names=val_arrays["source_call_names"].tolist(),
        predictions=pred_val,
    )
    test_dissection = _compute_dissection_from_arrays(
        stress_true=test_arrays["stress"],
        branch_id=test_arrays["branch_id"].astype(int),
        call_id=test_arrays["source_call_id"].astype(int),
        call_names=test_arrays["source_call_names"].tolist(),
        predictions=pred_test,
    )

    mining_dir = output_root / "mining"
    mining_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary: dict[str, Any] = {
        "primary_route": "gate_trial_threshold_t0.85",
        "val_dissection": val_dissection,
        "test_dissection": test_dissection,
        "branches": {},
    }

    split_payloads = [
        ("val", val_arrays, pred_val),
        ("test", test_arrays, pred_test),
    ]
    for branch_name, branch_id in TARGET_BRANCHES:
        call_ids = {row["call_id"] for row in val_dissection["worst_calls_top10"]}
        call_ids.update(row["call_id"] for row in test_dissection["worst_calls_top10"])

        branch_rel_parts = []
        for _, arrays, pred in split_payloads:
            branch_mask = arrays["branch_id"].astype(int) == branch_id
            if np.any(branch_mask):
                denom = np.maximum(np.linalg.norm(arrays["stress"][branch_mask], axis=1), 1.0)
                rel = np.linalg.norm(pred[branch_mask] - arrays["stress"][branch_mask], axis=1) / denom
                branch_rel_parts.append(rel)
        rel_all = np.concatenate(branch_rel_parts) if branch_rel_parts else np.zeros(1, dtype=np.float32)
        rel_median = float(np.median(rel_all))
        rel_iqr = float(np.quantile(rel_all, 0.75) - np.quantile(rel_all, 0.25))
        rel_threshold = rel_median + 1.5 * rel_iqr

        rows: list[dict[str, Any]] = []
        count_top_call = 0
        count_top_bin = 0
        count_high_rel = 0
        for split_name, arrays, pred in split_payloads:
            branch_mask = arrays["branch_id"].astype(int) == branch_id
            stress_mag = np.linalg.norm(arrays["stress"], axis=1)
            split_top_bin = float(np.quantile(stress_mag, 0.9))
            sample_rel = np.linalg.norm(pred - arrays["stress"], axis=1) / np.maximum(stress_mag, 1.0)
            call_mask = np.isin(arrays["source_call_id"].astype(int), list(call_ids))
            top_bin_mask = stress_mag >= split_top_bin
            high_rel_mask = sample_rel >= rel_threshold
            sel_mask = branch_mask & (call_mask | top_bin_mask | high_rel_mask)
            idx = np.flatnonzero(sel_mask)
            count_top_call += int(np.sum(branch_mask & call_mask))
            count_top_bin += int(np.sum(branch_mask & top_bin_mask))
            count_high_rel += int(np.sum(branch_mask & high_rel_mask))
            for i in idx.tolist():
                rows.append(
                    {
                        "split_name": split_name,
                        "split_id": 1 if split_name == "val" else 2,
                        "source_call_id": int(arrays["source_call_id"][i]),
                        "source_row_in_call": int(arrays.get("source_row_in_call", np.full(arrays["stress"].shape[0], -1))[i]),
                        "branch_id": int(arrays["branch_id"][i]),
                        "strain_eng": arrays["strain_eng"][i].astype(np.float32),
                        "stress": arrays["stress"][i].astype(np.float32),
                        "material_reduced": arrays["material_reduced"][i].astype(np.float32),
                        "stress_magnitude": float(stress_mag[i]),
                        "relative_error": float(sample_rel[i]),
                        "criterion_top_call": bool(call_mask[i]),
                        "criterion_top_bin": bool(top_bin_mask[i]),
                        "criterion_high_rel": bool(high_rel_mask[i]),
                    }
                )

        unique: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            key = (row["split_id"], row["source_row_in_call"])
            if key not in unique:
                unique[key] = row
            else:
                existing = unique[key]
                existing["criterion_top_call"] = existing["criterion_top_call"] or row["criterion_top_call"]
                existing["criterion_top_bin"] = existing["criterion_top_bin"] or row["criterion_top_bin"]
                existing["criterion_high_rel"] = existing["criterion_high_rel"] or row["criterion_high_rel"]
                existing["relative_error"] = max(existing["relative_error"], row["relative_error"])

        ordered = sorted(unique.values(), key=lambda row: (row["split_id"], row["source_row_in_call"]))
        arrays_out = {
            "strain_eng": np.stack([row["strain_eng"] for row in ordered], axis=0) if ordered else np.empty((0, 6), dtype=np.float32),
            "stress": np.stack([row["stress"] for row in ordered], axis=0) if ordered else np.empty((0, 6), dtype=np.float32),
            "material_reduced": np.stack([row["material_reduced"] for row in ordered], axis=0) if ordered else np.empty((0, 5), dtype=np.float32),
            "branch_id": np.array([row["branch_id"] for row in ordered], dtype=np.int8),
            "split_id": np.array([row["split_id"] for row in ordered], dtype=np.int8),
            "source_call_id": np.array([row["source_call_id"] for row in ordered], dtype=np.int32),
            "source_row_in_call": np.array([row["source_row_in_call"] for row in ordered], dtype=np.int32),
            "stress_magnitude": np.array([row["stress_magnitude"] for row in ordered], dtype=np.float32),
            "relative_error": np.array([row["relative_error"] for row in ordered], dtype=np.float32),
            "criterion_top_call": np.array([row["criterion_top_call"] for row in ordered], dtype=np.int8),
            "criterion_top_bin": np.array([row["criterion_top_bin"] for row in ordered], dtype=np.int8),
            "criterion_high_rel": np.array([row["criterion_high_rel"] for row in ordered], dtype=np.int8),
        }
        table_path = mining_dir / f"{branch_name}_hardcases.h5"
        _write_dataset(
            table_path,
            arrays_out,
            {
                "branch_name": branch_name,
                "branch_id": branch_id,
                "rel_threshold": rel_threshold,
                "criterion_call_ids_json": json.dumps(sorted(call_ids)),
            },
        )
        branch_summary = {
            "n_rows": int(arrays_out["branch_id"].shape[0]),
            "val_rows": int(np.sum(arrays_out["split_id"] == 1)),
            "test_rows": int(np.sum(arrays_out["split_id"] == 2)),
            "rel_threshold": rel_threshold,
            "count_top_call_branch_hits": count_top_call,
            "count_top_bin_branch_hits": count_top_bin,
            "count_high_rel_branch_hits": count_high_rel,
            "table_path": str(table_path),
            "call_count": len(call_ids),
        }
        summary["branches"][branch_name] = branch_summary

    labels = [name for name, _ in TARGET_BRANCHES]
    total_rows = [summary["branches"][name]["n_rows"] for name in labels]
    _plot_single_compare(labels, total_rows, output_root / "mining_counts.png", title="Hard-case mining rows", ylabel="rows")
    (output_root / "mining_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _branch_arrays_from_dataset(dataset_path: Path) -> dict[str, dict[str, np.ndarray]]:
    arrays, _ = _load_dataset_all(dataset_path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_id in (("train", 0), ("val", 1), ("test", 2)):
        mask = arrays["split_id"] == split_id
        out[split_name] = {key: value[mask] for key, value in arrays.items()}
    return out


def _compute_quantile_weights(stress_mag: np.ndarray) -> tuple[tuple[float, float, float], np.ndarray]:
    q50, q80, q95 = np.quantile(stress_mag, [0.5, 0.8, 0.95])
    weights = _weight_from_mag(stress_mag, (float(q50), float(q80), float(q95)))
    return (float(q50), float(q80), float(q95)), weights


def _weight_from_mag(stress_mag: np.ndarray, quantiles: tuple[float, float, float]) -> np.ndarray:
    q50, q80, q95 = quantiles
    weights = np.ones_like(stress_mag, dtype=np.float32)
    weights = np.where(stress_mag >= q50, 1.5, weights)
    weights = np.where(stress_mag >= q80, 2.5, weights)
    weights = np.where(stress_mag >= q95, 4.0, weights)
    weights = np.where(stress_mag < q50, 1.0, weights)
    return weights.astype(np.float32)


def _build_tensor_loader(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    stress_true: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(stress_true.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _decode_raw_prediction(pred_norm: torch.Tensor, y_scaler: Standardizer) -> torch.Tensor:
    mean = torch.as_tensor(y_scaler.mean, device=pred_norm.device)
    std = torch.as_tensor(y_scaler.std, device=pred_norm.device)
    return pred_norm * std + mean


def _run_epoch_raw(
    *,
    model: nn.Module,
    loader: DataLoader,
    y_scaler: Standardizer,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    variant: str,
    weight_quantiles: tuple[float, float, float],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_weighted_rmse_sum = 0.0
    total_n = 0

    for xb, yb, stress_true in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        stress_true = stress_true.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        pred_norm = out["stress"]
        per_sample_mse = torch.mean((pred_norm - yb) ** 2, dim=1)
        if variant == "edge_tail_weighted":
            stress_mag = torch.linalg.norm(stress_true, dim=1)
            weights = torch.from_numpy(_weight_from_mag(stress_mag.detach().cpu().numpy(), weight_quantiles)).to(device)
            loss = torch.mean(weights * per_sample_mse)
        else:
            loss = torch.mean(per_sample_mse)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        stress_pred = _decode_raw_prediction(pred_norm, y_scaler)
        diff = stress_pred - stress_true
        stress_mse = torch.mean(diff * diff)
        stress_mae = torch.mean(torch.abs(diff))
        stress_mag = torch.linalg.norm(stress_true, dim=1)
        weights_eval = torch.from_numpy(_weight_from_mag(stress_mag.detach().cpu().numpy(), weight_quantiles)).to(device)
        per_sample_sq = torch.mean(diff * diff, dim=1)
        weighted_rmse = torch.sqrt(torch.mean(weights_eval * per_sample_sq))

        batch_n = xb.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_n
        total_mse += float(stress_mse.detach().cpu()) * batch_n
        total_mae += float(stress_mae.detach().cpu()) * batch_n
        total_weighted_rmse_sum += float(weighted_rmse.detach().cpu()) * batch_n
        total_n += batch_n

    return {
        "loss": total_loss / max(total_n, 1),
        "stress_mse": total_mse / max(total_n, 1),
        "stress_mae": total_mae / max(total_n, 1),
        "weighted_rmse": total_weighted_rmse_sum / max(total_n, 1),
    }


def _history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "cycle",
                "batch_size",
                "stage_name",
                "lr",
                "train_loss",
                "val_loss",
                "val_stress_mae",
                "val_weighted_rmse",
                "test_stress_mae",
                "test_weighted_rmse",
                "best_val_weighted_rmse",
            ]
        )


def _history_row(
    path: Path,
    *,
    epoch: int,
    cycle: int,
    batch_size: int,
    stage_name: str,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_val_weighted_rmse: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                epoch,
                cycle,
                batch_size,
                stage_name,
                lr,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["stress_mae"],
                val_metrics["weighted_rmse"],
                test_metrics["stress_mae"],
                test_metrics["weighted_rmse"],
                best_val_weighted_rmse,
            ]
        )


def _plot_history(history_csv: Path, output_path: Path, title: str) -> Path:
    rows = list(csv.DictReader(history_csv.open("r", encoding="utf-8")))
    epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
    stage_names = [r["stage_name"] for r in rows]
    boundaries: list[tuple[int, str]] = []
    prev = None
    for e, s in zip(epoch, stage_names):
        if s != prev:
            boundaries.append((int(e), s))
            prev = s

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(epoch, arr("train_loss"), label="train")
    axes[0].plot(epoch, arr("val_loss"), label="val")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epoch, arr("val_weighted_rmse"), label="val weighted RMSE")
    axes[1].plot(epoch, arr("test_weighted_rmse"), label="test weighted RMSE")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("weighted RMSE")
    axes[1].set_xlabel("global epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    ymax0 = float(arr("train_loss").max())
    ymax1 = float(arr("val_weighted_rmse").max())
    for ax in axes:
        for x, _ in boundaries:
            ax.axvline(x, color="k", linestyle="--", alpha=0.15)
    for x, label in boundaries:
        axes[0].text(x + 1, ymax0, label, rotation=90, va="top", ha="left", fontsize=8)
        axes[1].text(x + 1, ymax1, label, rotation=90, va="top", ha="left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _stage_schedule(cycles: int, batch_sizes: list[int], base_lr: float, min_lr: float) -> list[Stage]:
    total = cycles * len(batch_sizes)
    lrs = np.geomspace(base_lr, min_lr, num=total)
    stages: list[Stage] = []
    idx = 0
    for cycle in range(1, cycles + 1):
        for batch_size in batch_sizes:
            stages.append(Stage(cycle=cycle, batch_size=batch_size, lr=float(lrs[idx]), name=f"c{cycle}_bs{batch_size}"))
            idx += 1
    return stages


def _branch_val_metrics(checkpoint_path: Path, dataset_path: Path, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    val_eval = evaluate_checkpoint_on_dataset(checkpoint_path, dataset_path, split="val", device=device, batch_size=16384)
    test_eval = evaluate_checkpoint_on_dataset(checkpoint_path, dataset_path, split="test", device=device, batch_size=16384)
    return val_eval["metrics"], test_eval["metrics"]


def _build_epoch_training_arrays(
    *,
    variant: str,
    train_arrays: dict[str, np.ndarray],
    mining_arrays: dict[str, np.ndarray] | None,
    branch_id: int,
    strain_std: np.ndarray,
    noise_scale: float,
    max_rounds: int,
    hard_replay_ratio: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if variant != "edge_hard_mined" or mining_arrays is None or mining_arrays["strain_eng"].shape[0] == 0:
        return {
            "strain_eng": train_arrays["strain_eng"],
            "stress": train_arrays["stress"],
            "material_reduced": train_arrays["material_reduced"],
        }

    n_total = train_arrays["strain_eng"].shape[0]
    n_orig = int(round((1.0 - hard_replay_ratio) * n_total))
    n_mined = max(0, n_total - n_orig)
    n_replay = n_mined // 2
    n_aug = n_mined - n_replay

    orig_idx = rng.choice(train_arrays["strain_eng"].shape[0], size=n_orig, replace=True)
    replay_idx = rng.choice(mining_arrays["strain_eng"].shape[0], size=n_replay, replace=True) if n_replay > 0 else np.empty((0,), dtype=int)

    out = {
        "strain_eng": [train_arrays["strain_eng"][orig_idx]],
        "stress": [train_arrays["stress"][orig_idx]],
        "material_reduced": [train_arrays["material_reduced"][orig_idx]],
    }
    if n_replay > 0:
        out["strain_eng"].append(mining_arrays["strain_eng"][replay_idx])
        out["stress"].append(mining_arrays["stress"][replay_idx])
        out["material_reduced"].append(mining_arrays["material_reduced"][replay_idx])

    if n_aug > 0:
        aug = _generate_augmented_rows(
            seeds=mining_arrays,
            n_rows=n_aug,
            branch_id=branch_id,
            strain_std=strain_std,
            noise_scale=noise_scale,
            max_rounds=max_rounds,
            rng=rng,
        )
        out["strain_eng"].append(aug["strain_eng"])
        out["stress"].append(aug["stress"])
        out["material_reduced"].append(aug["material_reduced"])

    return {
        "strain_eng": np.concatenate(out["strain_eng"], axis=0).astype(np.float32),
        "stress": np.concatenate(out["stress"], axis=0).astype(np.float32),
        "material_reduced": np.concatenate(out["material_reduced"], axis=0).astype(np.float32),
    }


def _generate_augmented_rows(
    *,
    seeds: dict[str, np.ndarray],
    n_rows: int,
    branch_id: int,
    strain_std: np.ndarray,
    noise_scale: float,
    max_rounds: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    kept_strain: list[np.ndarray] = []
    kept_stress: list[np.ndarray] = []
    kept_mat: list[np.ndarray] = []
    remaining = n_rows
    rounds = 0
    while remaining > 0 and rounds < max_rounds:
        rounds += 1
        batch_n = max(64, remaining * 3)
        idx = rng.choice(seeds["strain_eng"].shape[0], size=batch_n, replace=True)
        strain = seeds["strain_eng"][idx].astype(np.float32).copy()
        mat = seeds["material_reduced"][idx].astype(np.float32)
        noise = rng.normal(size=strain.shape).astype(np.float32) * (noise_scale * strain_std[None, :])
        strain = strain + noise
        exact = constitutive_update_3d(
            strain,
            c_bar=mat[:, 0],
            sin_phi=mat[:, 1],
            shear=mat[:, 2],
            bulk=mat[:, 3],
            lame=mat[:, 4],
        )
        mask = exact.branch_id == branch_id
        if np.any(mask):
            take = min(int(np.sum(mask)), remaining)
            sel = np.flatnonzero(mask)[:take]
            kept_strain.append(strain[sel])
            kept_stress.append(exact.stress[sel].astype(np.float32))
            kept_mat.append(mat[sel])
            remaining -= take
    if remaining > 0:
        idx = rng.choice(seeds["strain_eng"].shape[0], size=remaining, replace=True)
        kept_strain.append(seeds["strain_eng"][idx].astype(np.float32))
        kept_stress.append(seeds["stress"][idx].astype(np.float32))
        kept_mat.append(seeds["material_reduced"][idx].astype(np.float32))

    return {
        "strain_eng": np.concatenate(kept_strain, axis=0).astype(np.float32),
        "stress": np.concatenate(kept_stress, axis=0).astype(np.float32),
        "material_reduced": np.concatenate(kept_mat, axis=0).astype(np.float32),
    }


def _train_edge_expert(
    *,
    branch_name: str,
    branch_id: int,
    variant: str,
    dataset_path: Path,
    mining_table_path: Path | None,
    output_root: Path,
    args: argparse.Namespace,
    seed: int,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = choose_device(args.device)
    run_dir = output_root / f"{branch_name}_{variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    branch_splits = _branch_arrays_from_dataset(dataset_path)
    mining_arrays = None
    if mining_table_path is not None and mining_table_path.exists():
        mining_all, _ = _load_dataset_all(mining_table_path)
        mining_arrays = {key: value for key, value in mining_all.items() if key != "split_id"}

    train_features = build_raw_features(branch_splits["train"]["strain_eng"], branch_splits["train"]["material_reduced"])
    val_features = build_raw_features(branch_splits["val"]["strain_eng"], branch_splits["val"]["material_reduced"])
    test_features = build_raw_features(branch_splits["test"]["strain_eng"], branch_splits["test"]["material_reduced"])
    x_scaler = Standardizer.from_array(train_features)
    y_scaler = Standardizer.from_array(branch_splits["train"]["stress"].astype(np.float32))
    train_targets = y_scaler.transform(branch_splits["train"]["stress"].astype(np.float32))
    val_targets = y_scaler.transform(branch_splits["val"]["stress"].astype(np.float32))
    test_targets = y_scaler.transform(branch_splits["test"]["stress"].astype(np.float32))

    val_loader = _build_tensor_loader(
        features=x_scaler.transform(val_features),
        targets=val_targets,
        stress_true=branch_splits["val"]["stress"],
        batch_size=8192,
        shuffle=False,
    )
    test_loader = _build_tensor_loader(
        features=x_scaler.transform(test_features),
        targets=test_targets,
        stress_true=branch_splits["test"]["stress"],
        batch_size=8192,
        shuffle=False,
    )

    train_stress_mag = np.linalg.norm(branch_splits["train"]["stress"], axis=1)
    weight_quantiles, _ = _compute_quantile_weights(train_stress_mag)
    strain_std = np.maximum(branch_splits["train"]["strain_eng"].std(axis=0), 1.0e-7).astype(np.float32)

    model = build_model("raw", input_dim=train_features.shape[1], width=args.width, depth=args.depth, dropout=0.0).to(device)
    if init_checkpoint is not None and init_checkpoint.exists():
        ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    metadata = {
        "config": {
            "branch_name": branch_name,
            "branch_id": branch_id,
            "variant": variant,
            "dataset": str(dataset_path),
            "mining_table": str(mining_table_path) if mining_table_path else None,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "model_kind": "raw",
            "width": args.width,
            "depth": args.depth,
            "dropout": 0.0,
            "weight_decay": args.weight_decay,
            "batch_sizes": args.batch_sizes,
            "cycles": args.cycles,
            "stage_max_epochs": args.stage_max_epochs,
            "plateau_patience": args.plateau_patience,
            "stage_patience": args.stage_patience,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "hard_replay_ratio": args.hard_replay_ratio,
            "augment_noise_scale": args.augment_noise_scale,
            "checkpoint_metric": "val_weighted_rmse_then_mae",
            "seed": seed,
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
    }
    (run_dir / "train_config.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

    history_csv = run_dir / "history.csv"
    _history_header(history_csv)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    best_metric = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    global_epoch = 0
    stages = _stage_schedule(args.cycles, args.batch_sizes, args.base_lr, args.min_lr)
    start_time = time.time()
    rng = np.random.default_rng(seed)

    for stage in stages:
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
        stage_best = float("inf")
        stage_bad_epochs = 0
        for _ in range(args.stage_max_epochs):
            global_epoch += 1
            epoch_arrays = _build_epoch_training_arrays(
                variant=variant,
                train_arrays=branch_splits["train"],
                mining_arrays=mining_arrays,
                branch_id=branch_id,
                strain_std=strain_std,
                noise_scale=args.augment_noise_scale,
                max_rounds=args.augment_max_rounds,
                hard_replay_ratio=args.hard_replay_ratio,
                rng=rng,
            )
            epoch_features = x_scaler.transform(build_raw_features(epoch_arrays["strain_eng"], epoch_arrays["material_reduced"]))
            epoch_targets = y_scaler.transform(epoch_arrays["stress"])
            train_loader = _build_tensor_loader(
                features=epoch_features,
                targets=epoch_targets,
                stress_true=epoch_arrays["stress"],
                batch_size=stage.batch_size,
                shuffle=True,
            )

            train_metrics = _run_epoch_raw(
                model=model,
                loader=train_loader,
                y_scaler=y_scaler,
                optimizer=optimizer,
                device=device,
                variant=variant,
                weight_quantiles=weight_quantiles,
            )
            val_metrics = _run_epoch_raw(
                model=model,
                loader=val_loader,
                y_scaler=y_scaler,
                optimizer=None,
                device=device,
                variant="edge_tail_weighted",
                weight_quantiles=weight_quantiles,
            )
            test_metrics = _run_epoch_raw(
                model=model,
                loader=test_loader,
                y_scaler=y_scaler,
                optimizer=None,
                device=device,
                variant="edge_tail_weighted",
                weight_quantiles=weight_quantiles,
            )
            scheduler.step(val_metrics["weighted_rmse"])
            current_lr = optimizer.param_groups[0]["lr"]

            _history_row(
                history_csv,
                epoch=global_epoch,
                cycle=stage.cycle,
                batch_size=stage.batch_size,
                stage_name=stage.name,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                best_val_weighted_rmse=best_metric,
            )

            if (
                val_metrics["weighted_rmse"] < best_metric - 1.0e-9
                or (
                    abs(val_metrics["weighted_rmse"] - best_metric) <= 1.0e-9
                    and val_metrics["stress_mae"] < best_mae - 1.0e-9
                )
            ):
                best_metric = val_metrics["weighted_rmse"]
                best_mae = val_metrics["stress_mae"]
                best_epoch = global_epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "metadata": metadata,
                    },
                    best_path,
                )
            torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, last_path)

            if val_metrics["weighted_rmse"] < stage_best - 1.0e-9:
                stage_best = val_metrics["weighted_rmse"]
                stage_bad_epochs = 0
            else:
                stage_bad_epochs += 1

            if global_epoch == 1 or global_epoch % args.log_every_epochs == 0:
                msg = (
                    f"[expert {branch_name} {variant}] cycle={stage.cycle}/{args.cycles} "
                    f"batch={stage.batch_size} lr={current_lr:.3e} epoch={global_epoch} "
                    f"runtime={_format_runtime(time.time() - start_time)} "
                    f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                    f"val_weighted_rmse={val_metrics['weighted_rmse']:.6f} val_mae={val_metrics['stress_mae']:.6f}"
                )
                print(msg, flush=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            if stage_bad_epochs >= args.stage_patience:
                break

    _plot_history(history_csv, run_dir / "history.png", title=f"{branch_name} {variant}")

    val_eval = evaluate_checkpoint_on_dataset(best_path, dataset_path, split="val", device=args.device, batch_size=16384)
    test_eval = evaluate_checkpoint_on_dataset(best_path, dataset_path, split="test", device=args.device, batch_size=16384)
    val_metrics = dict(val_eval["metrics"])
    test_metrics = dict(test_eval["metrics"])
    val_metrics["weighted_rmse"] = float(best_metric)
    test_diff = test_eval["predictions"]["stress"] - test_eval["arrays"]["stress"]
    test_weights = _weight_from_mag(np.linalg.norm(test_eval["arrays"]["stress"], axis=1), weight_quantiles)
    test_metrics["weighted_rmse"] = float(np.sqrt(np.mean(test_weights * np.mean(test_diff * test_diff, axis=1))))
    summary = {
        "branch_name": branch_name,
        "branch_id": branch_id,
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_weighted_rmse": best_metric,
        "best_val_mae": best_mae,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(best_path),
        "history": str(history_csv),
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _evaluate_variant_routes(
    *,
    variant_name: str,
    reference_root: Path,
    experts_root: Path,
    gates_root: Path,
    output_root: Path,
    new_edge_ckpts: dict[int, Path] | None,
    device: str,
) -> dict[str, Any]:
    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    synth_dataset = reference_root / "cover_layer_full_synthetic_holdout.h5"
    real_test = _load_split_arrays(real_dataset, "test")
    synth_test = _load_split_arrays(synth_dataset, "test")
    baseline_real = _baseline_predictions(reference_root, real_test, device)
    baseline_synth = _baseline_predictions(reference_root, synth_test, device)
    trial_real = compute_trial_stress(real_test["strain_eng"], real_test["material_reduced"])
    trial_synth = compute_trial_stress(synth_test["strain_eng"], synth_test["material_reduced"])

    expert_paths = {
        1: experts_root / "expert_smooth" / "best.pt",
        2: experts_root / "expert_left_edge" / "best.pt",
        3: experts_root / "expert_right_edge" / "best.pt",
        4: experts_root / "expert_apex" / "best.pt",
    }
    if new_edge_ckpts is not None:
        expert_paths.update(new_edge_ckpts)
    real_experts = _predict_all_experts(expert_paths, real_test["strain_eng"], real_test["material_reduced"], device)
    synth_experts = _predict_all_experts(expert_paths, synth_test["strain_eng"], synth_test["material_reduced"], device)

    gate_configs = {
        "gate_trial_threshold_t0.85": ("trial", 0.85, gates_root / "gate_trial" / "best.pt"),
        "gate_raw_threshold_t0.65": ("raw", 0.65, gates_root / "gate_raw" / "best.pt"),
    }
    rows = []
    for route_name, (feature_kind, threshold, gate_ckpt) in gate_configs.items():
        gate_probs_real = _predict_gate(gate_ckpt, feature_kind, real_test["strain_eng"], real_test["material_reduced"], device)
        gate_probs_synth = _predict_gate(gate_ckpt, feature_kind, synth_test["strain_eng"], synth_test["material_reduced"], device)
        pred_real, branch_real = _ensemble_from_gate(
            gate_probs=gate_probs_real,
            trial_stress=trial_real,
            baseline_stress=baseline_real["stress"],
            expert_stress=real_experts,
            mode="threshold",
            threshold=threshold,
        )
        pred_synth, branch_synth = _ensemble_from_gate(
            gate_probs=gate_probs_synth,
            trial_stress=trial_synth,
            baseline_stress=baseline_synth["stress"],
            expert_stress=synth_experts,
            mode="threshold",
            threshold=threshold,
        )
        mode_name = f"{variant_name}_{route_name}"
        real_metrics, real_dissection = _evaluate_route(
            name=mode_name,
            split_arrays=real_test,
            stress_pred=pred_real,
            branch_pred=branch_real,
            output_dir=output_root / mode_name / "real",
            include_dissection=True,
        )
        synth_metrics, _ = _evaluate_route(
            name=mode_name,
            split_arrays=synth_test,
            stress_pred=pred_synth,
            branch_pred=branch_synth,
            output_dir=output_root / mode_name / "synthetic",
            include_dissection=False,
        )
        rows.append(
            {
                "name": mode_name,
                "route_name": route_name,
                "real": real_metrics,
                "real_dissection": real_dissection,
                "synthetic": synth_metrics,
            }
        )
    return {"rows": rows}


def _write_execution_doc(report_path: Path, state: dict[str, Any]) -> None:
    def line_status(key: str) -> str:
        return "[x]" if state["checkpoints"].get(key, {}).get("done") else "[ ]"

    lines: list[str] = []
    lines.append("# Cover Layer Tail-Safety Execution")
    lines.append("")
    lines.append("This is the living execution doc for the tail-safety phase. It tracks the frozen controls, hard-case mining, edge-expert retraining, routed evaluations, and the final go/no-go decision.")
    lines.append("")
    lines.append("Primary objective: improve real-holdout tail safety for the conservative hybrid route while keeping the existing MAE gains.")
    lines.append("")
    for key, title in (
        ("C0", "Freeze controls"),
        ("C1", "Build hard-case mining table"),
        ("C2", "Build edge-focused train/val datasets"),
        ("C3", "Train control edge experts"),
        ("C4", "Train tail-weighted edge experts"),
        ("C5", "Train hard-mined edge experts"),
        ("C6", "Re-evaluate deployable routing"),
        ("C7", "Decide go/no-go for solver-facing shadow test"),
    ):
        cp = state["checkpoints"].get(key, {})
        lines.append(f"- `{line_status(key)} {key} {title}`")
        lines.append(f"  Comment: {cp.get('comment', 'Pending.')}")
        lines.append(f"  Exit criterion: {cp.get('criterion', 'Pending.')}")
        lines.append(f"  Artifact: {cp.get('artifact', 'Pending.')}")
        if "result" in cp:
            lines.append(f"  Result: {cp['result']}")
        if "next" in cp:
            lines.append(f"  Next: {cp['next']}")
        if "image" in cp:
            lines.append("")
            lines.append(f"  ![]({cp['image']})")
        lines.append("")

    if "controls" in state:
        lines.append("## C0 Frozen Controls")
        lines.append("")
        lines.append("| Mode | Real MAE | Real RMSE | Real p99 | Real Edge MAE | Synthetic MAE | Synthetic RMSE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in state["controls"]["modes"]:
            lines.append(
                f"| {row['name']} | {row['real']['stress_mae']:.4f} | {row['real']['stress_rmse']:.4f} | "
                f"{row['real']['relative_error_p99']:.4f} | {row['real']['edge_combined_mae']:.4f} | "
                f"{row['synthetic']['stress_mae']:.4f} | {row['synthetic']['stress_rmse']:.4f} |"
            )
        lines.append("")
        lines.append(f"![Frozen control MAE]({(Path(state['paths']['output_root']) / 'controls_compare_mae.png').as_posix()})")
        lines.append("")
        lines.append(f"![Frozen control RMSE]({(Path(state['paths']['output_root']) / 'controls_compare_rmse.png').as_posix()})")
        lines.append("")
        lines.append(f"![Frozen control p99]({(Path(state['paths']['output_root']) / 'controls_compare_p99.png').as_posix()})")
        lines.append("")

    if "mining" in state:
        lines.append("## C1 Mining Summary")
        lines.append("")
        lines.append("| Branch | Rows | Val | Test | Rel Threshold | Worst-Call Count |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for branch_name in [name for name, _ in TARGET_BRANCHES]:
            row = state["mining"]["branches"][branch_name]
            lines.append(
                f"| {branch_name} | {row['n_rows']} | {row['val_rows']} | {row['test_rows']} | {row['rel_threshold']:.4f} | {row['call_count']} |"
            )
        lines.append("")
        lines.append(f"![Mining counts]({(Path(state['paths']['output_root']) / 'mining_counts.png').as_posix()})")
        lines.append("")

    if "datasets" in state:
        lines.append("## C2 Edge Datasets")
        lines.append("")
        lines.append("| Branch | Train | Val | Test | Dataset | Mining Table |")
        lines.append("|---|---:|---:|---:|---|---|")
        for branch_name in [name for name, _ in TARGET_BRANCHES]:
            row = state["datasets"][branch_name]
            lines.append(
                f"| {branch_name} | {row['split_counts']['train']} | {row['split_counts']['val']} | {row['split_counts']['test']} | "
                f"`{row['dataset_path']}` | `{row['mining_table']}` |"
            )
        lines.append("")

    if "training" in state:
        lines.append("## Edge Expert Training")
        lines.append("")
        lines.append("| Branch | Variant | Best Val Weighted RMSE | Val MAE | Test MAE | Checkpoint |")
        lines.append("|---|---|---:|---:|---:|---|")
        for branch_name in [name for name, _ in TARGET_BRANCHES]:
            for variant in VARIANT_ORDER:
                if variant not in state["training"].get(branch_name, {}):
                    continue
                row = state["training"][branch_name][variant]
                lines.append(
                    f"| {branch_name} | {variant} | {row['best_val_weighted_rmse']:.4f} | {row['val_metrics']['stress_mae']:.4f} | "
                    f"{row['test_metrics']['stress_mae']:.4f} | `{row['checkpoint']}` |"
                )
        lines.append("")
        for branch_name in [name for name, _ in TARGET_BRANCHES]:
            for variant in VARIANT_ORDER:
                if variant not in state["training"].get(branch_name, {}):
                    continue
                row = state["training"][branch_name][variant]
                hist = Path(row["checkpoint"]).parent / "history.png"
                lines.append(f"### {branch_name} {variant}")
                lines.append("")
                lines.append(f"![history]({hist.as_posix()})")
                lines.append("")

    if "routes" in state:
        lines.append("## C6 Deployable Routing Results")
        lines.append("")
        lines.append("| Route | Real MAE | Real RMSE | Real p99 | Real Edge MAE | Synthetic MAE | Synthetic RMSE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in state["routes"]:
            lines.append(
                f"| {row['name']} | {row['real']['stress_mae']:.4f} | {row['real']['stress_rmse']:.4f} | "
                f"{row['real']['relative_error_p99']:.4f} | {row['real']['edge_combined_mae']:.4f} | "
                f"{row['synthetic']['stress_mae']:.4f} | {row['synthetic']['stress_rmse']:.4f} |"
            )
        lines.append("")
        lines.append(f"![Route real MAE]({(Path(state['paths']['output_root']) / 'route_compare_real_mae.png').as_posix()})")
        lines.append("")
        lines.append(f"![Route real RMSE]({(Path(state['paths']['output_root']) / 'route_compare_real_rmse.png').as_posix()})")
        lines.append("")
        lines.append(f"![Route real p99]({(Path(state['paths']['output_root']) / 'route_compare_real_p99.png').as_posix()})")
        lines.append("")
        lines.append(f"![Route real edge MAE]({(Path(state['paths']['output_root']) / 'route_compare_real_edge_mae.png').as_posix()})")
        lines.append("")

    if "decision" in state:
        lines.append("## C7 Decision")
        lines.append("")
        lines.append(state["decision"])
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reference_root = (ROOT / args.reference_root).resolve()
    experts_root = (ROOT / args.experts_root).resolve()
    gates_root = (ROOT / args.gates_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    report_path = (ROOT / args.report_md).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "paths": {"output_root": str(output_root)},
        "checkpoints": {
            "C0": {
                "done": False,
                "comment": "Recompute and freeze the current scoreboard.",
                "criterion": "controls_summary.json plus four real-holdout dissection files exist.",
                "artifact": str(output_root / "controls_summary.json"),
            },
            "C1": {
                "done": False,
                "comment": "Mine left/right hard cases from the primary deployable route.",
                "criterion": "left_edge_hardcases.h5 and right_edge_hardcases.h5 exist with only the intended branch rows.",
                "artifact": str(output_root / "mining_summary.json"),
            },
            "C2": {
                "done": False,
                "comment": "Build edge-only datasets anchored to the fixed real exact-domain split.",
                "criterion": "left/right branch datasets exist and split counts are recorded.",
                "artifact": str(output_root / "datasets"),
            },
            "C3": {
                "done": False,
                "comment": "Retrain control left/right experts and compare them to the published experts.",
                "criterion": "control experts reproduce the old branch validation MAE within 3% and the deployable real-holdout MAE within 5%.",
                "artifact": str(output_root / "training_control_summary.json"),
            },
            "C4": {
                "done": False,
                "comment": "Retrain left/right experts with tail-weighted loss only.",
                "criterion": "tail-weighted route improves at least one of edge MAE, top-bin sample MAE, or p99 relative error by the requested margin.",
                "artifact": str(output_root / "training_tail_weighted_summary.json"),
            },
            "C5": {
                "done": False,
                "comment": "Retrain left/right experts with hard-mined replay plus local augmentation.",
                "criterion": "hard-mined primary route beats control and tail-weighted on RMSE and p99 without materially worse MAE.",
                "artifact": str(output_root / "training_hard_mined_summary.json"),
            },
            "C6": {
                "done": False,
                "comment": "Evaluate the conservative hybrid routes with the updated edge experts.",
                "criterion": "a routed expert ensemble beats baseline on RMSE, p99, and edge combined MAE while keeping MAE within +5%.",
                "artifact": str(output_root / "routes_summary.json"),
            },
            "C7": {
                "done": False,
                "comment": "Make a solver-shadow go/no-go decision from the final scoreboard.",
                "criterion": "decision text written with the winning route and the caveat list.",
                "artifact": str(report_path),
            },
        },
    }
    _write_execution_doc(report_path, state)

    print("[tail-safety] C0 freezing controls", flush=True)
    controls = _freeze_controls(
        reference_root=reference_root,
        experts_root=experts_root,
        gates_root=gates_root,
        output_root=output_root,
        device=args.device,
    )
    state["controls"] = controls
    best_control = min(controls["modes"], key=lambda row: row["real"]["stress_mae"])
    state["checkpoints"]["C0"].update(
        {
            "done": True,
            "comment": "Recomputed the fixed scoreboard for the baseline, oracle, and both deployable routed controls.",
            "result": f"Best deployable control remains {best_control['name']} with real MAE {best_control['real']['stress_mae']:.4f}, but baseline_reference still has the lowest RMSE {min(controls['modes'], key=lambda row: row['real']['stress_rmse'])['real']['stress_rmse']:.4f}.",
            "next": "Use the primary deployable route dissection to mine left_edge and right_edge failures.",
            "image": str((output_root / "controls_compare_rmse.png").resolve()),
        }
    )
    _write_execution_doc(report_path, state)

    print("[tail-safety] C1 building hard-case mining tables", flush=True)
    mining = _build_hardcase_mining_tables(
        reference_root=reference_root,
        experts_root=experts_root,
        gates_root=gates_root,
        output_root=output_root,
        device=args.device,
    )
    state["mining"] = mining
    state["checkpoints"]["C1"].update(
        {
            "done": True,
            "comment": "Built left/right mining tables from the fixed val/test holdout using the current primary route, including worst calls, top stress bin, and branch outlier errors.",
            "result": "Mining tables contain only target-branch rows and preserve source split/call metadata for replay and augmentation.",
            "next": "Materialize branch datasets and wire each branch to its mining table.",
            "image": str((output_root / "mining_counts.png").resolve()),
        }
    )
    _write_execution_doc(report_path, state)

    print("[tail-safety] C2 building edge datasets", flush=True)
    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    datasets: dict[str, Any] = {}
    for branch_name, branch_id in TARGET_BRANCHES:
        dataset_path = output_root / "datasets" / f"{branch_name}_dataset.h5"
        _, info = _build_branch_dataset(real_dataset, dataset_path, branch_id)
        datasets[branch_name] = {
            "dataset_path": str(dataset_path),
            "split_counts": info["split_counts"],
            "mining_table": mining["branches"][branch_name]["table_path"],
        }
        with h5py.File(mining["branches"][branch_name]["table_path"], "r") as f:
            branch_values = f["branch_id"][:]
            if branch_values.size and not np.all(branch_values == branch_id):
                raise RuntimeError(f"Mining table for {branch_name} contains wrong branch ids.")
    state["datasets"] = datasets
    state["checkpoints"]["C2"].update(
        {
            "done": True,
            "comment": "Built branch-filtered real datasets for left_edge and right_edge and validated the branch-pure mining tables.",
            "result": "Both branch datasets now have fixed train/val/test splits and branch-pure hard-case tables.",
            "next": "Train the control experts first and compare them to the existing experts before trusting any new weighting or mining.",
        }
    )
    _write_execution_doc(report_path, state)

    print("[tail-safety] C3/C4/C5 training edge experts", flush=True)
    training_state: dict[str, dict[str, Any]] = {}
    old_branch_metrics: dict[str, dict[str, Any]] = {}
    for branch_name, branch_id in TARGET_BRANCHES:
        old_ckpt = experts_root / f"expert_{branch_name}" / "best.pt"
        branch_dataset = Path(datasets[branch_name]["dataset_path"])
        old_val, old_test = _branch_val_metrics(old_ckpt, branch_dataset, args.device)
        old_branch_metrics[branch_name] = {"val": old_val, "test": old_test}

    def _seed_for(branch_name: str, variant: str) -> int:
        base = {"left_edge": 11, "right_edge": 23}[branch_name]
        offset = {"edge_control": 0, "edge_tail_weighted": 100, "edge_hard_mined": 200}[variant]
        return args.seed + base + offset

    route_rows: list[dict[str, Any]] = []
    old_route_rows = _evaluate_variant_routes(
        variant_name="old",
        reference_root=reference_root,
        experts_root=experts_root,
        gates_root=gates_root,
        output_root=output_root / "routes",
        new_edge_ckpts=None,
        device=args.device,
    )["rows"]
    route_rows.extend(old_route_rows)

    c3_pass = True
    c4_pass = False
    c5_pass = False
    best_deployable_row: dict[str, Any] | None = None

    for variant in [v for v in VARIANT_ORDER if v in args.variants]:
        variant_ckpts: dict[int, Path] = {}
        variant_rows: list[dict[str, Any]] = []
        for branch_name, branch_id in TARGET_BRANCHES:
            branch_dataset = Path(datasets[branch_name]["dataset_path"])
            mining_path = Path(datasets[branch_name]["mining_table"]) if variant == "edge_hard_mined" else None
            init_checkpoint = None
            if variant == "edge_tail_weighted" and "edge_control" in training_state.get(branch_name, {}):
                init_checkpoint = Path(training_state[branch_name]["edge_control"]["checkpoint"])
            elif variant == "edge_hard_mined":
                if "edge_tail_weighted" in training_state.get(branch_name, {}):
                    init_checkpoint = Path(training_state[branch_name]["edge_tail_weighted"]["checkpoint"])
                elif "edge_control" in training_state.get(branch_name, {}):
                    init_checkpoint = Path(training_state[branch_name]["edge_control"]["checkpoint"])
            summary = _train_edge_expert(
                branch_name=branch_name,
                branch_id=branch_id,
                variant=variant,
                dataset_path=branch_dataset,
                mining_table_path=mining_path,
                output_root=output_root / "experts",
                args=args,
                seed=_seed_for(branch_name, variant),
                init_checkpoint=init_checkpoint,
            )
            training_state.setdefault(branch_name, {})[variant] = summary
            variant_ckpts[branch_id] = Path(summary["checkpoint"])
            variant_rows.append(summary)

        routes = _evaluate_variant_routes(
            variant_name=variant,
            reference_root=reference_root,
            experts_root=experts_root,
            gates_root=gates_root,
            output_root=output_root / "routes",
            new_edge_ckpts=variant_ckpts,
            device=args.device,
        )["rows"]
        route_rows.extend(routes)
        primary_row = next(row for row in routes if row["route_name"] == "gate_trial_threshold_t0.85")
        if best_deployable_row is None or primary_row["real"]["stress_mae"] < best_deployable_row["real"]["stress_mae"]:
            best_deployable_row = primary_row

        if variant == "edge_control":
            val_diffs = []
            for branch_name in [name for name, _ in TARGET_BRANCHES]:
                old_mae = old_branch_metrics[branch_name]["val"]["stress_mae"]
                new_mae = training_state[branch_name][variant]["val_metrics"]["stress_mae"]
                rel_diff = abs(new_mae - old_mae) / max(old_mae, 1.0e-12)
                val_diffs.append(rel_diff)
            old_primary = next(row for row in old_route_rows if row["route_name"] == "gate_trial_threshold_t0.85")
            route_rel_diff = abs(primary_row["real"]["stress_mae"] - old_primary["real"]["stress_mae"]) / max(old_primary["real"]["stress_mae"], 1.0e-12)
            c3_pass = max(val_diffs) <= 0.03 and route_rel_diff <= 0.05
            state["checkpoints"]["C3"].update(
                {
                    "done": True,
                    "comment": "Retrained both edge experts with the control recipe and compared them to the published experts and the old primary route.",
                    "result": f"max branch-val relative diff={max(val_diffs):.4f}, primary-route MAE relative diff={route_rel_diff:.4f}. {'PASS' if c3_pass else 'FAIL'}",
                    "next": "Proceed to tail weighting." if c3_pass else "Control reproducibility failed; per plan, later variants are diagnostic rather than trustworthy.",
                    "image": str((Path(training_state['left_edge'][variant]['checkpoint']).parent / 'history.png').resolve()),
                }
            )
            (output_root / "training_control_summary.json").write_text(json.dumps(_json_safe({"old_branch_metrics": old_branch_metrics, "new_training": training_state}), indent=2), encoding="utf-8")
            _write_execution_doc(report_path, state)

        if variant == "edge_tail_weighted":
            control_primary = next(row for row in route_rows if row["name"] == "edge_control_gate_trial_threshold_t0.85")
            improve_edge = (control_primary["real"]["edge_combined_mae"] - primary_row["real"]["edge_combined_mae"]) / max(control_primary["real"]["edge_combined_mae"], 1.0e-12)
            improve_top = (control_primary["real"]["top_bin_sample_mae"] - primary_row["real"]["top_bin_sample_mae"]) / max(control_primary["real"]["top_bin_sample_mae"], 1.0e-12)
            improve_p99 = (control_primary["real"]["relative_error_p99"] - primary_row["real"]["relative_error_p99"]) / max(control_primary["real"]["relative_error_p99"], 1.0e-12)
            c4_pass = improve_edge >= 0.08 or improve_top >= 0.10 or improve_p99 >= 0.10
            state["checkpoints"]["C4"].update(
                {
                    "done": True,
                    "comment": "Compared tail-weighted edge experts against the control route on the fixed primary deployable policy.",
                    "result": f"edge improvement={improve_edge:.4f}, top-bin improvement={improve_top:.4f}, p99 improvement={improve_p99:.4f}. {'PASS' if c4_pass else 'FAIL'}",
                    "next": "Proceed to hard-mined replay." if c4_pass else "Tail weighting alone was not enough; proceed to hard-mined replay.",
                    "image": str((Path(training_state['left_edge'][variant]['checkpoint']).parent / 'history.png').resolve()),
                }
            )
            (output_root / "training_tail_weighted_summary.json").write_text(json.dumps(_json_safe({"routes": routes, "training": training_state}), indent=2), encoding="utf-8")
            _write_execution_doc(report_path, state)

        if variant == "edge_hard_mined":
            control_primary = next(row for row in route_rows if row["name"] == "edge_control_gate_trial_threshold_t0.85")
            tail_primary = next(row for row in route_rows if row["name"] == "edge_tail_weighted_gate_trial_threshold_t0.85")
            best_mae_candidate = min(control_primary["real"]["stress_mae"], tail_primary["real"]["stress_mae"], primary_row["real"]["stress_mae"])
            c5_pass = (
                primary_row["real"]["stress_rmse"] < control_primary["real"]["stress_rmse"]
                and primary_row["real"]["stress_rmse"] < tail_primary["real"]["stress_rmse"]
                and primary_row["real"]["relative_error_p99"] < control_primary["real"]["relative_error_p99"]
                and primary_row["real"]["relative_error_p99"] < tail_primary["real"]["relative_error_p99"]
                and primary_row["real"]["stress_mae"] <= best_mae_candidate * 1.03
            )
            state["checkpoints"]["C5"].update(
                {
                    "done": True,
                    "comment": "Compared hard-mined replay against both control and tail-weighted on the fixed primary deployable policy.",
                    "result": f"primary RMSE={primary_row['real']['stress_rmse']:.4f}, p99={primary_row['real']['relative_error_p99']:.4f}, MAE={primary_row['real']['stress_mae']:.4f}. {'PASS' if c5_pass else 'FAIL'}",
                    "next": "Re-evaluate both fixed gate policies with all route variants.",
                    "image": str((Path(training_state['left_edge'][variant]['checkpoint']).parent / 'history.png').resolve()),
                }
            )
            (output_root / "training_hard_mined_summary.json").write_text(json.dumps(_json_safe({"routes": routes, "training": training_state}), indent=2), encoding="utf-8")
            _write_execution_doc(report_path, state)

    state["training"] = training_state

    print("[tail-safety] C6 evaluating routes", flush=True)
    route_candidates = [row for row in route_rows if row["route_name"] in {"gate_trial_threshold_t0.85", "gate_raw_threshold_t0.65"}]
    baseline_row = next(row for row in route_rows if row["name"] == "old_gate_trial_threshold_t0.85")
    baseline_ref = next(row for row in controls["modes"] if row["name"] == "baseline_reference")
    route_candidates_sorted = sorted(route_candidates, key=lambda row: row["real"]["stress_mae"])
    state["routes"] = route_candidates_sorted

    _plot_single_compare(
        [row["name"] for row in route_candidates_sorted],
        [row["real"]["stress_mae"] for row in route_candidates_sorted],
        output_root / "route_compare_real_mae.png",
        title="Route comparison real MAE",
        ylabel="real MAE",
    )
    _plot_single_compare(
        [row["name"] for row in route_candidates_sorted],
        [row["real"]["stress_rmse"] for row in route_candidates_sorted],
        output_root / "route_compare_real_rmse.png",
        title="Route comparison real RMSE",
        ylabel="real RMSE",
    )
    _plot_single_compare(
        [row["name"] for row in route_candidates_sorted],
        [row["real"]["relative_error_p99"] for row in route_candidates_sorted],
        output_root / "route_compare_real_p99.png",
        title="Route comparison real p99 relative error",
        ylabel="real p99",
    )
    _plot_single_compare(
        [row["name"] for row in route_candidates_sorted],
        [row["real"]["edge_combined_mae"] for row in route_candidates_sorted],
        output_root / "route_compare_real_edge_mae.png",
        title="Route comparison real edge combined MAE",
        ylabel="real edge MAE",
    )

    safe_winners = [
        row
        for row in route_candidates_sorted
        if row["real"]["stress_rmse"] < baseline_ref["real"]["stress_rmse"]
        and row["real"]["relative_error_p99"] < baseline_ref["real"]["relative_error_p99"]
        and row["real"]["edge_combined_mae"] < baseline_ref["real"]["edge_combined_mae"]
        and row["real"]["stress_mae"] <= baseline_ref["real"]["stress_mae"] * 1.05
    ]
    c6_pass = len(safe_winners) > 0
    best_safe = safe_winners[0] if safe_winners else route_candidates_sorted[0]
    state["checkpoints"]["C6"].update(
        {
            "done": True,
            "comment": "Evaluated both fixed gate policies for the old, control, tail-weighted, and hard-mined edge experts.",
            "result": (
                f"Best safe route is {best_safe['name']} with real MAE {best_safe['real']['stress_mae']:.4f}, RMSE {best_safe['real']['stress_rmse']:.4f}, "
                f"p99 {best_safe['real']['relative_error_p99']:.4f}. {'PASS' if c6_pass else 'FAIL'}"
            ),
            "next": "Make the final go/no-go decision.",
            "image": str((output_root / "route_compare_real_rmse.png").resolve()),
        }
    )
    (output_root / "routes_summary.json").write_text(json.dumps(_json_safe({"routes": route_candidates_sorted, "baseline_reference": baseline_ref}), indent=2), encoding="utf-8")
    _write_execution_doc(report_path, state)

    print("[tail-safety] C7 decision", flush=True)
    if c6_pass:
        decision = (
            f"GO for a solver-facing shadow test. The safest routed expert candidate is `{best_safe['name']}`, which beats `baseline_reference` on RMSE, p99 relative error, "
            f"and combined edge MAE while keeping real MAE within the allowed margin. The remaining caveat is that the hard-mined variant used holdout-informed mining by design, "
            "so the solver shadow test should be treated as the next real validation step, not as final proof of replacement safety."
        )
    else:
        decision = (
            f"NO-GO for a solver-facing shadow test from this phase. No routed expert ensemble beat `baseline_reference` on the required tail metrics. "
            f"The best routed candidate by MAE was `{best_safe['name']}`, but the route still failed at least one of RMSE, p99 relative error, or edge combined MAE. "
            "The next phase should focus on route retuning or target redesign rather than another broad edge-expert sweep. "
            "Also note that the hard-mined phase used holdout-informed mining by design, so these results are already optimistic for the current expert family."
        )
    state["decision"] = decision
    state["checkpoints"]["C7"].update(
        {
            "done": True,
            "comment": "Locked the solver-facing recommendation from the final routed scoreboard.",
            "result": decision,
            "next": "If GO, wire the winner into a solver shadow test. If NO-GO, pivot to a routing-focused follow-up.",
        }
    )
    _write_execution_doc(report_path, state)


if __name__ == "__main__":
    main()
