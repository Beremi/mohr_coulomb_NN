#!/usr/bin/env python
"""Execute the main cover-layer single-material workplan on the full export."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import SPLIT_NAMES, SPLIT_TO_ID, dataset_summary, load_arrays
from mc_surrogate.full_export import (
    build_test_only_dataset,
    load_cover_call_archive,
    load_sparse_B,
    sample_cover_family_dataset,
)
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", default="constitutive_problem_3D_full.h5")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_single_material_20260313")
    parser.add_argument("--report-md", default="docs/cover_layer_single_material_execution.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-call", type=int, default=256)
    parser.add_argument("--synthetic-samples-per-call", type=int, default=256)
    parser.add_argument("--synthetic-holdout-virtual-calls", type=int, default=96)
    parser.add_argument("--augment-virtual-calls", type=int, default=388)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=700)
    parser.add_argument("--plateau-patience", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--lbfgs-epochs", type=int, default=8)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--log-every-epochs", type=int, default=100)
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


def _read_attrs(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        return {key: f.attrs[key] for key in f.attrs.keys()}


def _call_names_for_split(path: Path, split: str) -> list[str]:
    attrs = _read_attrs(path)
    key = f"{split}_call_names_json"
    return json.loads(attrs[key])


def _load_problem_geometry(full_export_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(full_export_path, "r") as f:
        coord = f["problem/coord"][:].astype(np.float64)
        elem = f["problem/elem"][:].astype(np.int64) - 1
        material_identifier = f["problem/material_identifier"][0].astype(np.int64)
    return {
        "coord": coord,
        "elem": elem,
        "material_identifier": material_identifier,
    }


def _signed_tet_volumes(coords: np.ndarray) -> np.ndarray:
    a = coords[:, 1] - coords[:, 0]
    b = coords[:, 2] - coords[:, 0]
    c = coords[:, 3] - coords[:, 0]
    return np.einsum("ni,ni->n", np.cross(a, b), c) / 6.0


def _volume_ratio_ok(
    u_vec: np.ndarray,
    *,
    coord: np.ndarray,
    cover_corner_elem: np.ndarray,
    min_ratio: float = 0.1,
) -> bool:
    disp = np.asarray(u_vec, dtype=np.float64).reshape(-1, 3)
    ref = coord[cover_corner_elem]
    deformed = ref + disp[cover_corner_elem]
    ref_vol = _signed_tet_volumes(ref)
    def_vol = _signed_tet_volumes(deformed)
    ratio = def_vol / np.maximum(np.abs(ref_vol), 1.0e-12)
    return bool(np.all(ratio > min_ratio))


def _build_neighbor_table(material_reduced: np.ndarray, window: int = 6) -> list[np.ndarray]:
    order = np.argsort(material_reduced[:, 0])
    position = np.empty_like(order)
    position[order] = np.arange(order.size)
    neighbors: list[np.ndarray] = []
    for idx in range(order.size):
        pos = position[idx]
        lo = max(0, pos - window)
        hi = min(order.size, pos + window + 1)
        cand = order[lo:hi]
        cand = cand[cand != idx]
        if cand.size == 0:
            cand = np.array([idx], dtype=np.int64)
        neighbors.append(cand.astype(np.int64))
    return neighbors


def _generate_virtual_cover_dataset(
    *,
    full_export_path: Path,
    archive: dict[str, Any],
    call_names_pool: list[str],
    output_path: Path,
    rng: np.random.Generator,
    n_virtual_calls: int,
    samples_per_call: int,
    volume_ratio_min: float = 0.1,
) -> tuple[Path, dict[str, Any]]:
    b_all = load_sparse_B(full_export_path)
    b_cover = b_all[archive["b_row_mask"]]
    geom = _load_problem_geometry(full_export_path)
    cover_elem = geom["elem"][geom["material_identifier"] == 0, :4]

    u_archive = archive["U"].astype(np.float32)
    material_archive = archive["material_reduced"].astype(np.float32)
    neighbors = _build_neighbor_table(material_archive)
    n_cover_ip = int(archive["ip_mask"].sum())

    strain_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    principal_parts: list[np.ndarray] = []
    eigvec_parts: list[np.ndarray] = []
    call_id_parts: list[np.ndarray] = []
    source_type_parts: list[np.ndarray] = []

    attempts = 0
    accepted = 0
    while accepted < n_virtual_calls:
        attempts += 1
        base_idx = int(rng.integers(u_archive.shape[0]))
        neigh_choices = neighbors[base_idx]
        neigh_idx = int(rng.choice(neigh_choices))
        alpha = float(np.clip(rng.normal(loc=0.0, scale=0.18), -0.35, 0.35))

        u_base = u_archive[base_idx]
        u_neigh = u_archive[neigh_idx]
        u_syn = u_base + alpha * (u_neigh - u_base)
        if not _volume_ratio_ok(u_syn, coord=geom["coord"], cover_corner_elem=cover_elem, min_ratio=volume_ratio_min):
            continue

        mat_base = material_archive[base_idx]
        mat_neigh = material_archive[neigh_idx]
        mat_syn = (mat_base + alpha * (mat_neigh - mat_base)).astype(np.float32)

        strain_all = np.asarray(b_cover @ u_syn, dtype=np.float32).reshape(n_cover_ip, 6)
        sample_rows = np.sort(rng.choice(n_cover_ip, size=min(samples_per_call, n_cover_ip), replace=False))
        strain = strain_all[sample_rows]
        material = np.repeat(mat_syn[None, :], strain.shape[0], axis=0)

        exact = constitutive_update_3d(
            strain,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
            shear=material[:, 2],
            bulk=material[:, 3],
            lame=material[:, 4],
        )

        strain_parts.append(strain.astype(np.float32))
        stress_parts.append(exact.stress.astype(np.float32))
        material_parts.append(material.astype(np.float32))
        branch_parts.append(exact.branch_id.astype(np.int8))
        principal_parts.append(exact.stress_principal.astype(np.float32))
        eigvec_parts.append(exact.eigvecs.astype(np.float32))
        call_id_parts.append(np.full(strain.shape[0], base_idx, dtype=np.int32))
        source_type_parts.append(np.full(strain.shape[0], 1, dtype=np.int8))
        accepted += 1

    arrays = {
        "strain_eng": np.vstack(strain_parts),
        "stress": np.vstack(stress_parts),
        "material_reduced": np.vstack(material_parts),
        "stress_principal": np.vstack(principal_parts),
        "branch_id": np.concatenate(branch_parts),
        "eigvecs": np.vstack(eigvec_parts),
        "source_call_id": np.concatenate(call_id_parts),
        "source_type": np.concatenate(source_type_parts),
    }
    attrs = {
        "generator": "u_local_blend",
        "n_virtual_calls": int(n_virtual_calls),
        "samples_per_call": int(samples_per_call),
        "source_call_names_json": json.dumps(call_names_pool),
        "acceptance_rate": float(accepted / max(attempts, 1)),
        "volume_ratio_min": float(volume_ratio_min),
    }
    build_test_only_dataset(output_path, arrays, attrs=attrs)
    summary = dataset_summary(output_path)
    summary["acceptance_rate"] = attrs["acceptance_rate"]
    return output_path, summary


def _build_hybrid_dataset(
    *,
    real_dataset_path: Path,
    synthetic_train_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    real_all = load_arrays(
        real_dataset_path,
        ["strain_eng", "stress", "material_reduced", "stress_principal", "branch_id", "eigvecs", "source_call_id", "source_row_in_call"],
        split=None,
    )
    with h5py.File(real_dataset_path, "r") as f:
        split_id = f["split_id"][:]
        attrs = {key: f.attrs[key] for key in f.attrs.keys()}

    synth = load_arrays(
        synthetic_train_path,
        ["strain_eng", "stress", "material_reduced", "stress_principal", "branch_id", "eigvecs", "source_call_id"],
        split="test",
    )
    synth["source_row_in_call"] = np.full(synth["strain_eng"].shape[0], -1, dtype=np.int32)
    synth["source_type"] = np.full(synth["strain_eng"].shape[0], 1, dtype=np.int8)
    real_all["source_type"] = np.zeros(real_all["strain_eng"].shape[0], dtype=np.int8)

    arrays: dict[str, np.ndarray] = {}
    real_train_mask = split_id == SPLIT_TO_ID["train"]
    for key, value in real_all.items():
        if key in synth:
            arrays[key] = np.concatenate([value[real_train_mask], synth[key], value[~real_train_mask]], axis=0)
        else:
            arrays[key] = np.concatenate([value[real_train_mask], value[~real_train_mask]], axis=0)

    split_out = np.concatenate(
        [
            np.full(int(np.sum(real_train_mask)) + synth["strain_eng"].shape[0], SPLIT_TO_ID["train"], dtype=np.int8),
            split_id[~real_train_mask],
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.create_dataset("split_id", data=split_out, compression="gzip", shuffle=True)
        for key, value in attrs.items():
            f.attrs[key] = value
        f.attrs["hybrid_dataset"] = True
        f.attrs["synthetic_train_source"] = str(synthetic_train_path)
        f.attrs["synthetic_train_rows"] = int(synth["strain_eng"].shape[0])

    summary = dataset_summary(output_path)
    summary["synthetic_train_rows"] = int(synth["strain_eng"].shape[0])
    return output_path, summary


def _plot_history_log(history_csv: Path, output_path: Path, title: str) -> Path:
    data = np.genfromtxt(history_csv, delimiter=",", names=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(data["epoch"], data["train_loss"], label="train")
    axes[0].plot(data["epoch"], data["val_loss"], label="val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(data["epoch"], data["train_stress_mse"], label="train")
    axes[1].plot(data["epoch"], data["val_stress_mse"], label="val")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("stress MSE")
    axes[1].set_title("stress MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
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


def _plot_metric_bars(rows: list[dict[str, Any]], output_path: Path) -> Path:
    names = [row["name"] for row in rows]
    real_mae = [row["real"]["stress_mae"] for row in rows]
    synth_mae = [row["synthetic"]["stress_mae"] for row in rows]
    real_rmse = [row["real"]["stress_rmse"] for row in rows]
    synth_rmse = [row["synthetic"]["stress_rmse"] for row in rows]

    x = np.arange(len(names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - width / 2, real_mae, width=width, label="real")
    axes[0].bar(x + width / 2, synth_mae, width=width, label="synthetic")
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylabel("stress MAE")
    axes[0].set_title("MAE")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x - width / 2, real_rmse, width=width, label="real")
    axes[1].bar(x + width / 2, synth_rmse, width=width, label="synthetic")
    axes[1].set_xticks(x, names, rotation=20, ha="right")
    axes[1].set_ylabel("stress RMSE")
    axes[1].set_title("RMSE")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_per_branch_mae(rows: list[dict[str, Any]], output_path: Path) -> Path:
    x = np.arange(len(BRANCH_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, row in enumerate(rows):
        vals = [row["real"].get("per_branch_stress_mae", {}).get(name, math.nan) for name in BRANCH_NAMES]
        ax.bar(x + (idx - (len(rows) - 1) / 2.0) * width, vals, width=width, label=row["name"])
    ax.set_xticks(x, BRANCH_NAMES, rotation=20, ha="right")
    ax.set_ylabel("real-test branch stress MAE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _compute_real_dissection(
    *,
    real_dataset_path: Path,
    predictions: np.ndarray,
) -> dict[str, Any]:
    arrays = load_arrays(
        real_dataset_path,
        ["stress", "branch_id", "source_call_id"],
        split="test",
    )
    with h5py.File(real_dataset_path, "r") as f:
        call_names = json.loads(f.attrs["source_call_names_json"])

    stress_true = arrays["stress"]
    branch_id = arrays["branch_id"].astype(int)
    call_id = arrays["source_call_id"].astype(int)
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


def _evaluate_and_plot(
    *,
    name: str,
    checkpoint_path: Path,
    real_dataset_path: Path,
    synthetic_dataset_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    real_eval = evaluate_checkpoint_on_dataset(checkpoint_path, real_dataset_path, split="test", device=device, batch_size=16384)
    synth_eval = evaluate_checkpoint_on_dataset(checkpoint_path, synthetic_dataset_path, split="test", device=device, batch_size=16384)

    (output_dir / "real_metrics.json").write_text(json.dumps(_json_safe(real_eval["metrics"]), indent=2), encoding="utf-8")
    (output_dir / "synthetic_metrics.json").write_text(json.dumps(_json_safe(synth_eval["metrics"]), indent=2), encoding="utf-8")

    parity_plot(
        real_eval["arrays"]["stress"],
        real_eval["predictions"]["stress"],
        output_dir / "real_parity.png",
        label="stress",
    )
    error_histogram(
        real_eval["predictions"]["stress"] - real_eval["arrays"]["stress"],
        output_dir / "real_error_hist.png",
        label="real stress error",
    )
    _plot_relative_error_cdf(
        real_eval["arrays"]["stress"],
        real_eval["predictions"]["stress"],
        output_dir / "real_relative_error_cdf.png",
        title=f"{name} real-test relative error",
    )
    _plot_error_vs_magnitude(
        real_eval["arrays"]["stress"],
        real_eval["predictions"]["stress"],
        output_dir / "real_error_vs_magnitude.png",
        title=f"{name} real-test error vs stress magnitude",
    )
    if "branch_confusion" in real_eval["metrics"]:
        branch_confusion_plot(real_eval["metrics"]["branch_confusion"], output_dir / "real_branch_confusion.png")

    parity_plot(
        synth_eval["arrays"]["stress"],
        synth_eval["predictions"]["stress"],
        output_dir / "synthetic_parity.png",
        label="stress",
    )
    error_histogram(
        synth_eval["predictions"]["stress"] - synth_eval["arrays"]["stress"],
        output_dir / "synthetic_error_hist.png",
        label="synthetic stress error",
    )
    _plot_relative_error_cdf(
        synth_eval["arrays"]["stress"],
        synth_eval["predictions"]["stress"],
        output_dir / "synthetic_relative_error_cdf.png",
        title=f"{name} synthetic-holdout relative error",
    )
    _plot_error_vs_magnitude(
        synth_eval["arrays"]["stress"],
        synth_eval["predictions"]["stress"],
        output_dir / "synthetic_error_vs_magnitude.png",
        title=f"{name} synthetic-holdout error vs stress magnitude",
    )
    if "branch_confusion" in synth_eval["metrics"]:
        branch_confusion_plot(synth_eval["metrics"]["branch_confusion"], output_dir / "synthetic_branch_confusion.png")

    return {
        "name": name,
        "real": real_eval["metrics"],
        "synthetic": synth_eval["metrics"],
        "real_predictions": real_eval["predictions"]["stress"],
    }


def _train_one(
    *,
    experiment_name: str,
    dataset_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    model_kind: str,
) -> dict[str, Any]:
    run_dir = output_root / experiment_name
    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind=model_kind,
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
    )
    summary = train_model(config)
    _plot_history_log(run_dir / "history.csv", run_dir / "history_log.png", title=experiment_name)
    (run_dir / "train_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    (run_dir / "config_used.json").write_text(json.dumps(_json_safe(asdict(config)), indent=2), encoding="utf-8")
    return summary


def _write_report(
    *,
    report_path: Path,
    output_root: Path,
    real_dataset_path: Path,
    synthetic_dataset_path: Path,
    real_dataset_info: dict[str, Any],
    synthetic_dataset_info: dict[str, Any],
    hybrid_dataset_info: dict[str, Any],
    experiment_rows: list[dict[str, Any]],
    real_dissection: dict[str, Any],
) -> None:
    best_row = min(experiment_rows, key=lambda row: row["real"]["stress_mae"])
    lines: list[str] = []
    lines.append("# Cover-Layer Single-Material Execution")
    lines.append("")
    lines.append("This report executes the main cover-layer single-material workplan against the new full export and compares long-horizon models on a fixed real holdout split plus a fixed synthetic `U/B` holdout.")
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append(f"- Real exact-domain dataset: `{real_dataset_path}`")
    lines.append(f"  - total samples: `{real_dataset_info['n_samples']}`")
    lines.append(f"  - split counts: `{real_dataset_info['split_counts']}`")
    lines.append(f"  - branch counts: `{real_dataset_info['branch_counts']}`")
    lines.append("- Synthetic holdout dataset:")
    lines.append(f"  - path: `{synthetic_dataset_path}`")
    lines.append(f"  - total samples: `{synthetic_dataset_info['n_samples']}`")
    lines.append(f"  - branch counts: `{synthetic_dataset_info['branch_counts']}`")
    lines.append("- Hybrid training dataset:")
    lines.append(f"  - synthetic train rows added: `{hybrid_dataset_info['synthetic_train_rows']}`")
    lines.append(f"  - split counts: `{hybrid_dataset_info['split_counts']}`")
    lines.append("")
    lines.append("## Experiments")
    lines.append("")
    lines.append("Three long-horizon experiments were run:")
    lines.append("")
    lines.append("1. `baseline_raw_branch`: direct raw-space stress regression with auxiliary branch head on the real exact-domain split.")
    lines.append("2. `structured_trial_raw_branch_residual`: exact elastic trial features with a learned stress residual and branch head on the same real split.")
    lines.append("3. `hybrid_trial_raw_branch_residual`: the same structured model trained on a hybrid dataset that mixes the real train split with `U/B`-pushforward synthetic augmentation.")
    lines.append("")
    lines.append("All runs used long plateau training with the same width/depth and an LBFGS tail refinement.")
    lines.append("")
    lines.append("## Result Table")
    lines.append("")
    lines.append("| Experiment | Real MAE | Real RMSE | Real Max Abs | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in experiment_rows:
        real = row["real"]
        synth = row["synthetic"]
        lines.append(
            f"| {row['name']} | {real['stress_mae']:.4f} | {real['stress_rmse']:.4f} | {real['stress_max_abs']:.4f} | "
            f"{real.get('branch_accuracy', float('nan')):.4f} | {synth['stress_mae']:.4f} | {synth['stress_rmse']:.4f} | "
            f"{synth.get('branch_accuracy', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append(f"Best real-test model in this study: `{best_row['name']}`.")
    lines.append("")
    lines.append(f"![Metric comparison]({(output_root / 'comparison_metrics.png').as_posix()})")
    lines.append("")
    lines.append(f"![Per-branch MAE]({(output_root / 'comparison_per_branch.png').as_posix()})")
    lines.append("")
    lines.append("## Per-Experiment Figures")
    lines.append("")
    for row in experiment_rows:
        exp_dir = output_root / row["name"]
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- training history: ![history]({(exp_dir / 'history_log.png').as_posix()})")
        lines.append(f"- real parity: ![real parity]({(exp_dir / 'eval' / 'real_parity.png').as_posix()})")
        lines.append(f"- real relative error CDF: ![real rel]({(exp_dir / 'eval' / 'real_relative_error_cdf.png').as_posix()})")
        lines.append(f"- real error vs magnitude: ![real mag]({(exp_dir / 'eval' / 'real_error_vs_magnitude.png').as_posix()})")
        if (exp_dir / "eval" / "real_branch_confusion.png").exists():
            lines.append(f"- real branch confusion: ![real branch]({(exp_dir / 'eval' / 'real_branch_confusion.png').as_posix()})")
        lines.append(f"- synthetic parity: ![synth parity]({(exp_dir / 'eval' / 'synthetic_parity.png').as_posix()})")
        lines.append(f"- synthetic relative error CDF: ![synth rel]({(exp_dir / 'eval' / 'synthetic_relative_error_cdf.png').as_posix()})")
        lines.append(f"- synthetic error vs magnitude: ![synth mag]({(exp_dir / 'eval' / 'synthetic_error_vs_magnitude.png').as_posix()})")
        if (exp_dir / "eval" / "synthetic_branch_confusion.png").exists():
            lines.append(f"- synthetic branch confusion: ![synth branch]({(exp_dir / 'eval' / 'synthetic_branch_confusion.png').as_posix()})")
        lines.append("")
    lines.append("## Real-Test Dissection Of The Best Model")
    lines.append("")
    lines.append(f"Best model: `{best_row['name']}`")
    lines.append("")
    lines.append(f"- mean relative sample error: `{real_dissection['relative_error_mean']:.4f}`")
    lines.append(f"- median relative sample error: `{real_dissection['relative_error_median']:.4f}`")
    lines.append(f"- p90 relative sample error: `{real_dissection['relative_error_p90']:.4f}`")
    lines.append(f"- p99 relative sample error: `{real_dissection['relative_error_p99']:.4f}`")
    lines.append("")
    lines.append("### Error By Stress-Magnitude Bin")
    lines.append("")
    lines.append("| Stress-Magnitude Bin | N | Sample MAE | Component MAE | Mean Relative |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in real_dissection["stress_magnitude_bins"]:
        lines.append(
            f"| {row['stress_mag_lo']:.2f} to {row['stress_mag_hi']:.2f} | {row['n']} | {row['sample_mae']:.4f} | {row['component_mae']:.4f} | {row['mean_relative']:.4f} |"
        )
    lines.append("")
    lines.append("### Per-Branch Mean Relative Error")
    lines.append("")
    for name, value in real_dissection["per_branch_mean_relative"].items():
        lines.append(f"- `{name}`: `{value:.4f}`")
    lines.append("")
    lines.append("### Worst Real Holdout Calls")
    lines.append("")
    lines.append("| Call | N | Component MAE | Sample MAE | Mean Relative |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in real_dissection["worst_calls_top10"]:
        lines.append(
            f"| {row['call_name']} | {row['n']} | {row['component_mae']:.4f} | {row['sample_mae']:.4f} | {row['mean_relative']:.4f} |"
        )
    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append("The important comparison in this study is not only real-test error, but the gap between the real holdout and the synthetic `U/B` holdout.")
    lines.append("")
    lines.append("- If a model is good on synthetic but poor on real, the training distribution is still off.")
    lines.append("- If both improve together, the structure and the augmentation are helping in the right direction.")
    lines.append("- The hybrid result is the key test of whether the new full export is already useful for FE-compatible augmentation.")
    lines.append("")
    lines.append("The best next move after this study should follow the winner:")
    lines.append("")
    lines.append("- If the structured residual model wins without augmentation, keep the exact-elastic-plus-plastic-correction structure and improve the representation next.")
    lines.append("- If the hybrid model wins, make `U/B` augmentation part of the default cover-layer training recipe.")
    lines.append("- If neither beats the older heavy fitted-refresh baseline convincingly, the next bottleneck is still the synthetic state generator rather than the network itself.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    full_export_path = (ROOT / args.full_export).resolve()
    output_root = ROOT / args.output_root
    report_path = ROOT / args.report_md
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    real_dataset_path = output_root / "cover_layer_full_real_exact_256.h5"
    synthetic_holdout_path = output_root / "cover_layer_full_synthetic_holdout.h5"
    synthetic_train_path = output_root / "cover_layer_full_synthetic_train.h5"
    hybrid_dataset_path = output_root / "cover_layer_full_hybrid_train.h5"

    if not real_dataset_path.exists():
        sample_cover_family_dataset(
            full_export_path,
            real_dataset_path,
            family_name="cover_layer",
            samples_per_call=args.samples_per_call,
            seed=args.seed,
            use_exact_stress=True,
        )
    real_dataset_info = dataset_summary(real_dataset_path)

    test_call_names = _call_names_for_split(real_dataset_path, "test")
    train_call_names = _call_names_for_split(real_dataset_path, "train")

    if not synthetic_holdout_path.exists():
        holdout_archive = load_cover_call_archive(full_export_path, split_call_names=test_call_names)
        _, synthetic_dataset_info = _generate_virtual_cover_dataset(
            full_export_path=full_export_path,
            archive=holdout_archive,
            call_names_pool=test_call_names,
            output_path=synthetic_holdout_path,
            rng=np.random.default_rng(args.seed + 11),
            n_virtual_calls=min(args.synthetic_holdout_virtual_calls, len(test_call_names)),
            samples_per_call=args.synthetic_samples_per_call,
        )
    else:
        synthetic_dataset_info = dataset_summary(synthetic_holdout_path)

    if not synthetic_train_path.exists():
        train_archive = load_cover_call_archive(full_export_path, split_call_names=train_call_names)
        _, _ = _generate_virtual_cover_dataset(
            full_export_path=full_export_path,
            archive=train_archive,
            call_names_pool=train_call_names,
            output_path=synthetic_train_path,
            rng=np.random.default_rng(args.seed + 29),
            n_virtual_calls=min(args.augment_virtual_calls, len(train_call_names)),
            samples_per_call=args.synthetic_samples_per_call,
        )

    if not hybrid_dataset_path.exists():
        _, hybrid_dataset_info = _build_hybrid_dataset(
            real_dataset_path=real_dataset_path,
            synthetic_train_path=synthetic_train_path,
            output_path=hybrid_dataset_path,
        )
    else:
        hybrid_dataset_info = dataset_summary(hybrid_dataset_path)
        with h5py.File(hybrid_dataset_path, "r") as f:
            synth_rows = int(f.attrs.get("synthetic_train_rows", 0))
        hybrid_dataset_info["synthetic_train_rows"] = synth_rows

    experiments = (
        ("baseline_raw_branch", "raw_branch", real_dataset_path),
        ("structured_trial_raw_branch_residual", "trial_raw_branch_residual", real_dataset_path),
        ("hybrid_trial_raw_branch_residual", "trial_raw_branch_residual", hybrid_dataset_path),
    )

    experiment_rows: list[dict[str, Any]] = []
    for exp_name, model_kind, dataset_path in experiments:
        exp_dir = output_root / exp_name
        if not (exp_dir / "best.pt").exists():
            _train_one(
                experiment_name=exp_name,
                dataset_path=dataset_path,
                output_root=output_root,
                args=args,
                model_kind=model_kind,
            )
        row = _evaluate_and_plot(
            name=exp_name,
            checkpoint_path=exp_dir / "best.pt",
            real_dataset_path=real_dataset_path,
            synthetic_dataset_path=synthetic_holdout_path,
            output_dir=exp_dir / "eval",
            device=args.device,
        )
        experiment_rows.append(row)

    _plot_metric_bars(experiment_rows, output_root / "comparison_metrics.png")
    _plot_per_branch_mae(experiment_rows, output_root / "comparison_per_branch.png")

    best_row = min(experiment_rows, key=lambda row: row["real"]["stress_mae"])
    real_dissection = _compute_real_dissection(
        real_dataset_path=real_dataset_path,
        predictions=best_row["real_predictions"],
    )
    (output_root / "real_dissection.json").write_text(json.dumps(_json_safe(real_dissection), indent=2), encoding="utf-8")

    _write_report(
        report_path=report_path,
        output_root=output_root,
        real_dataset_path=real_dataset_path,
        synthetic_dataset_path=synthetic_holdout_path,
        real_dataset_info=real_dataset_info,
        synthetic_dataset_info=synthetic_dataset_info,
        hybrid_dataset_info=hybrid_dataset_info,
        experiment_rows=experiment_rows,
        real_dissection=real_dissection,
    )


if __name__ == "__main__":
    main()
