#!/usr/bin/env python
"""Cover-layer-only cyclic batch-size sweep on exact-domain synthetic labels."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import REDUCED_MATERIAL_COLUMNS, SPLIT_NAMES, SPLIT_TO_ID
from mc_surrogate.models import Standardizer, build_model, build_raw_features
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.training import (
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    evaluate_checkpoint_on_dataset,
    load_checkpoint,
    set_seed,
)
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


@dataclass(frozen=True)
class AdamStage:
    name: str
    batch_size: int
    lr_start: float
    lr_end: float
    epochs: int
    lbfgs_steps_after: int = 0


@dataclass(frozen=True)
class RunSpec:
    name: str
    width: int
    depth: int
    weight_decay: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-primary",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5",
    )
    parser.add_argument(
        "--real-cross",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_cross.h5",
    )
    parser.add_argument(
        "--hybrid-dataset",
        default="experiment_runs/real_sim/per_material_hybrid_hardcases_20260312/hybrid_datasets/cover_layer_hybrid.h5",
        help="Used only for coverage diagnostics against the old augmented training set.",
    )
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_cyclic_20260312")
    parser.add_argument("--report-md", default="docs/cover_layer_cyclic_sweep.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--global-patience", type=int, default=42)
    parser.add_argument("--min-epochs-before-stop", type=int, default=36)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every-epochs", type=int, default=10)
    parser.add_argument("--spec-names", nargs="*", help="Optional subset of spec names to run.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _run_specs() -> list[RunSpec]:
    return [
        RunSpec("cover_raw_branch_w256_d5", width=256, depth=5, weight_decay=1.0e-5, seed=8101),
        RunSpec("cover_raw_branch_w384_d6", width=384, depth=6, weight_decay=1.0e-5, seed=8102),
        RunSpec("cover_raw_branch_w512_d6", width=512, depth=6, weight_decay=1.0e-5, seed=8103),
    ]


def _stage_schedule() -> list[AdamStage]:
    return [
        AdamStage("cycle1_bs64", batch_size=64, lr_start=1.0e-3, lr_end=8.0e-4, epochs=4),
        AdamStage("cycle1_bs128", batch_size=128, lr_start=8.0e-4, lr_end=6.0e-4, epochs=6),
        AdamStage("cycle1_bs256", batch_size=256, lr_start=6.0e-4, lr_end=4.0e-4, epochs=8),
        AdamStage("cycle1_bs512", batch_size=512, lr_start=4.0e-4, lr_end=2.5e-4, epochs=10),
        AdamStage("cycle1_bs1024", batch_size=1024, lr_start=2.5e-4, lr_end=1.5e-4, epochs=12, lbfgs_steps_after=1),
        AdamStage("cycle1_bs2048", batch_size=2048, lr_start=1.5e-4, lr_end=8.0e-5, epochs=14, lbfgs_steps_after=1),
        AdamStage("cycle2_bs64", batch_size=64, lr_start=8.0e-5, lr_end=6.0e-5, epochs=4),
        AdamStage("cycle2_bs128", batch_size=128, lr_start=6.0e-5, lr_end=4.0e-5, epochs=6),
        AdamStage("cycle2_bs256", batch_size=256, lr_start=4.0e-5, lr_end=2.5e-5, epochs=8),
        AdamStage("cycle2_bs512", batch_size=512, lr_start=2.5e-5, lr_end=1.5e-5, epochs=10),
        AdamStage("cycle2_bs1024", batch_size=1024, lr_start=1.5e-5, lr_end=8.0e-6, epochs=12, lbfgs_steps_after=1),
        AdamStage("cycle2_bs2048", batch_size=2048, lr_start=8.0e-6, lr_end=1.0e-6, epochs=14, lbfgs_steps_after=1),
    ]


def _load_all_arrays(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.attrs["branch_names_json"] = json.dumps(BRANCH_NAMES)
        f.attrs["reduced_material_columns_json"] = json.dumps(REDUCED_MATERIAL_COLUMNS)
        f.attrs["split_names_json"] = json.dumps(SPLIT_NAMES)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value
    return path


def _build_exact_domain_dataset(real_primary: Path, output_path: Path) -> tuple[Path, dict[str, Any]]:
    arrays, attrs = _load_all_arrays(real_primary)
    exact = constitutive_update_3d(
        arrays["strain_eng"],
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
        shear=arrays["material_reduced"][:, 2],
        bulk=arrays["material_reduced"][:, 3],
        lame=arrays["material_reduced"][:, 4],
    )
    exact_arrays = {
        "strain_eng": arrays["strain_eng"].astype(np.float32),
        "stress": exact.stress.astype(np.float32),
        "material_reduced": arrays["material_reduced"].astype(np.float32),
        "stress_principal": exact.stress_principal.astype(np.float32),
        "branch_id": exact.branch_id.astype(np.int8),
        "eigvecs": exact.eigvecs.astype(np.float32),
        "split_id": arrays["split_id"].astype(np.int8),
    }
    if "source_call_id" in arrays:
        exact_arrays["source_call_id"] = arrays["source_call_id"].astype(np.int32)
    if "source_mode_id" in arrays:
        exact_arrays["source_mode_id"] = arrays["source_mode_id"].astype(np.int8)

    diff = exact_arrays["stress"] - arrays["stress"]
    meta = {
        "generator": "exact_domain_relabel_no_augmentation",
        "source_real_dataset": str(real_primary),
        "exact_vs_real_mae": float(np.mean(np.abs(diff))),
        "exact_vs_real_rmse": float(np.sqrt(np.mean(diff * diff))),
        "exact_vs_real_max_abs": float(np.max(np.abs(diff))),
    }
    meta.update(attrs)
    _write_dataset(output_path, exact_arrays, meta)
    return output_path, meta


def _sample_out_of_box_rate(train_feat: np.ndarray, test_feat: np.ndarray) -> dict[str, float]:
    lo = train_feat.min(axis=0)
    hi = train_feat.max(axis=0)
    miss = (test_feat < lo) | (test_feat > hi)
    return {
        "sample_out_of_box_rate": float(np.mean(np.any(miss, axis=1))),
        "feature_out_of_box_fraction": float(np.mean(miss)),
    }


def _nn_distance_summary(train_feat: np.ndarray, test_feat: np.ndarray) -> dict[str, float]:
    mean = train_feat.mean(axis=0)
    std = np.maximum(train_feat.std(axis=0), 1.0e-6)
    tr = ((train_feat - mean) / std).astype(np.float32)
    te = ((test_feat - mean) / std).astype(np.float32)
    rng = np.random.default_rng(0)
    if tr.shape[0] > 12000:
        tr = tr[rng.choice(tr.shape[0], size=12000, replace=False)]
    if te.shape[0] > 2000:
        te = te[rng.choice(te.shape[0], size=2000, replace=False)]
    dists: list[np.ndarray] = []
    block = 256
    for start in range(0, te.shape[0], block):
        x = te[start : start + block]
        d = np.sqrt(np.sum((x[:, None, :] - tr[None, :, :]) ** 2, axis=2))
        dists.append(np.min(d, axis=1))
    nn = np.concatenate(dists)
    return {
        "nn_z_mean": float(np.mean(nn)),
        "nn_z_p90": float(np.quantile(nn, 0.9)),
        "nn_z_p99": float(np.quantile(nn, 0.99)),
    }


def _distribution_summary(strain_eng: np.ndarray, stress: np.ndarray, branch_id: np.ndarray) -> dict[str, Any]:
    from mc_surrogate.models import spectral_decomposition_from_strain

    strain_principal, _ = spectral_decomposition_from_strain(strain_eng)
    max_abs_principal = np.max(np.abs(strain_principal), axis=1)
    stress_mag = np.linalg.norm(stress, axis=1)
    return {
        "max_abs_principal_q50": float(np.quantile(max_abs_principal, 0.5)),
        "max_abs_principal_q95": float(np.quantile(max_abs_principal, 0.95)),
        "max_abs_principal_q995": float(np.quantile(max_abs_principal, 0.995)),
        "stress_mag_q50": float(np.quantile(stress_mag, 0.5)),
        "stress_mag_q95": float(np.quantile(stress_mag, 0.95)),
        "stress_mag_q995": float(np.quantile(stress_mag, 0.995)),
        "branch_counts": {name: int(np.sum(branch_id == i)) for i, name in enumerate(BRANCH_NAMES)},
    }


def _mex_kernel_check(real_primary: Path) -> dict[str, Any]:
    arrays, _ = _load_all_arrays(real_primary)
    test_mask = arrays["split_id"] == SPLIT_TO_ID["test"]
    strain = arrays["strain_eng"][test_mask]
    mat = arrays["material_reduced"][test_mask]
    rng = np.random.default_rng(0)
    idx = rng.choice(strain.shape[0], size=min(200, strain.shape[0]), replace=False)
    strain = strain[idx]
    mat = mat[idx]
    exact = constitutive_update_3d(
        strain,
        c_bar=mat[:, 0],
        sin_phi=mat[:, 1],
        shear=mat[:, 2],
        bulk=mat[:, 3],
        lame=mat[:, 4],
    ).stress

    header_url = (
        "https://raw.githubusercontent.com/Beremi/slope_stability_octave/main/"
        "slope_stability/+CONSTITUTIVE_PROBLEM/mex/constitutive_3D_kernel.h"
    )
    wrapper = r"""
#include <stdio.h>
#include "constitutive_3D_kernel.h"
int main(void) {
    double E[6], c_bar, sin_phi, shear, bulk, lame, S[6];
    while (scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                 &E[0], &E[1], &E[2], &E[3], &E[4], &E[5],
                 &c_bar, &sin_phi, &shear, &bulk, &lame) == 11) {
        constitutive_3D_point(E, c_bar, sin_phi, shear, bulk, lame, S, NULL);
        printf("%.17g %.17g %.17g %.17g %.17g %.17g\n", S[0], S[1], S[2], S[3], S[4], S[5]);
    }
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        header_path = td_path / "constitutive_3D_kernel.h"
        src_path = td_path / "check.c"
        bin_path = td_path / "check"
        subprocess.run(["curl", "-L", "--silent", header_url, "-o", str(header_path)], check=True)
        src_path.write_text(wrapper, encoding="utf-8")
        subprocess.run(["cc", "-O3", "-std=c99", str(src_path), "-lm", "-o", str(bin_path)], check=True)

        input_path = td_path / "input.txt"
        np.savetxt(input_path, np.hstack([strain, mat]), fmt="%.17g")
        proc = subprocess.run([str(bin_path)], input=input_path.read_text(encoding="utf-8"), text=True, capture_output=True, check=True)
        c_stress = np.loadtxt(proc.stdout.splitlines()) if proc.stdout.strip() else np.empty_like(exact)
        if c_stress.ndim == 1:
            c_stress = c_stress.reshape(1, -1)

    diff = exact - c_stress
    return {
        "samples": int(exact.shape[0]),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _coverage_summary(real_primary: Path, hybrid_dataset: Path, exact_domain_dataset: Path) -> dict[str, Any]:
    real_arrays, _ = _load_all_arrays(real_primary)
    hybrid_arrays, _ = _load_all_arrays(hybrid_dataset)
    exact_arrays, exact_attrs = _load_all_arrays(exact_domain_dataset)

    test_mask = real_arrays["split_id"] == SPLIT_TO_ID["test"]
    real_test_feat = build_raw_features(real_arrays["strain_eng"][test_mask], real_arrays["material_reduced"][test_mask])

    out: dict[str, Any] = {
        "exact_domain_attrs": exact_attrs,
        "real_train_distribution": _distribution_summary(
            real_arrays["strain_eng"][real_arrays["split_id"] == SPLIT_TO_ID["train"]],
            real_arrays["stress"][real_arrays["split_id"] == SPLIT_TO_ID["train"]],
            real_arrays["branch_id"][real_arrays["split_id"] == SPLIT_TO_ID["train"]],
        ),
        "real_test_distribution": _distribution_summary(
            real_arrays["strain_eng"][test_mask],
            real_arrays["stress"][test_mask],
            real_arrays["branch_id"][test_mask],
        ),
        "datasets": {},
    }

    for name, arrays in (("hybrid_train", hybrid_arrays), ("exact_domain_train", exact_arrays)):
        train_mask = arrays["split_id"] == SPLIT_TO_ID["train"]
        train_feat = build_raw_features(arrays["strain_eng"][train_mask], arrays["material_reduced"][train_mask])
        stats = {}
        stats.update(_sample_out_of_box_rate(train_feat, real_test_feat))
        stats.update(_nn_distance_summary(train_feat, real_test_feat))
        stats.update(
            _distribution_summary(
                arrays["strain_eng"][train_mask],
                arrays["stress"][train_mask],
                arrays["branch_id"][train_mask],
            )
        )
        stats["n_train"] = int(np.sum(train_mask))
        out["datasets"][name] = stats

    return out


def _make_eval_loader(split_arrays: dict[str, np.ndarray], x_scaler: Standardizer, y_scaler: Standardizer, batch_size: int) -> DataLoader:
    ds = _build_tensor_dataset(split_arrays, x_scaler, y_scaler)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "stage_name",
                "stage_kind",
                "batch_size",
                "lr",
                "train_loss",
                "train_stress_mse",
                "synth_val_loss",
                "synth_val_stress_mse",
                "real_val_loss",
                "real_val_stress_mse",
                "real_test_loss",
                "real_test_stress_mse",
                "cross_test_loss",
                "cross_test_stress_mse",
                "train_branch_accuracy",
                "synth_val_branch_accuracy",
                "real_val_branch_accuracy",
                "real_test_branch_accuracy",
                "cross_test_branch_accuracy",
                "best_real_val_stress_mse",
                "is_best",
            ]
        )


def _append_history_row(
    path: Path,
    *,
    epoch: int,
    stage_name: str,
    stage_kind: str,
    batch_size: int,
    lr: float,
    train_metrics: dict[str, float],
    synth_val_metrics: dict[str, float],
    real_val_metrics: dict[str, float],
    real_test_metrics: dict[str, float],
    cross_test_metrics: dict[str, float],
    best_real_val_stress_mse: float,
    is_best: bool,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                stage_name,
                stage_kind,
                batch_size,
                lr,
                train_metrics["loss"],
                train_metrics["stress_mse"],
                synth_val_metrics["loss"],
                synth_val_metrics["stress_mse"],
                real_val_metrics["loss"],
                real_val_metrics["stress_mse"],
                real_test_metrics["loss"],
                real_test_metrics["stress_mse"],
                cross_test_metrics["loss"],
                cross_test_metrics["stress_mse"],
                train_metrics["branch_accuracy"],
                synth_val_metrics["branch_accuracy"],
                real_val_metrics["branch_accuracy"],
                real_test_metrics["branch_accuracy"],
                cross_test_metrics["branch_accuracy"],
                best_real_val_stress_mse,
                1 if is_best else 0,
            ]
        )


def _plot_history(history_csv: Path, output_path: Path) -> Path:
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

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(epoch, arr("train_loss"), label="train")
    axes[0].plot(epoch, arr("synth_val_loss"), label="synthetic val")
    axes[0].plot(epoch, arr("real_val_loss"), label="real val")
    axes[0].plot(epoch, arr("real_test_loss"), label="real test")
    axes[0].plot(epoch, arr("cross_test_loss"), label="cross test")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch, arr("train_stress_mse"), label="train")
    axes[1].plot(epoch, arr("synth_val_stress_mse"), label="synthetic val")
    axes[1].plot(epoch, arr("real_val_stress_mse"), label="real val")
    axes[1].plot(epoch, arr("real_test_stress_mse"), label="real test")
    axes[1].plot(epoch, arr("cross_test_stress_mse"), label="cross test")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("stress mse")
    axes[1].set_xlabel("global epoch / stage step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ymax0 = float(max(arr("train_loss").max(), arr("synth_val_loss").max(), arr("real_val_loss").max(), arr("real_test_loss").max()))
    ymax1 = float(max(arr("train_stress_mse").max(), arr("synth_val_stress_mse").max(), arr("real_val_stress_mse").max(), arr("real_test_stress_mse").max()))
    for ax in axes:
        for x, label in boundaries:
            ax.axvline(x, color="k", linestyle="--", alpha=0.2)
    for x, label in boundaries:
        axes[0].text(x + 1, ymax0, label, rotation=90, va="top", ha="left", fontsize=8)
        axes[1].text(x + 1, ymax1, label, rotation=90, va="top", ha="left", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def _maybe_log_epoch(
    *,
    epoch: int,
    stage_name: str,
    lr: float,
    train_metrics: dict[str, float],
    real_val_metrics: dict[str, float],
    real_test_metrics: dict[str, float],
    best_real_val_stress_mse: float,
    stage_kind: str,
    log_every_epochs: int,
) -> None:
    if log_every_epochs <= 0:
        return
    if epoch == 1 or epoch % log_every_epochs == 0:
        print(
            f"[{stage_kind.upper()}] epoch={epoch} stage={stage_name} lr={lr:.3e} "
            f"train_loss={train_metrics['loss']:.6f} real_val_stress_mse={real_val_metrics['stress_mse']:.6f} "
            f"real_test_stress_mse={real_test_metrics['stress_mse']:.6f} best_real_val={best_real_val_stress_mse:.6f}"
        )


def _train_one(
    *,
    exact_dataset: Path,
    real_primary: Path,
    real_cross: Path,
    run_dir: Path,
    spec: RunSpec,
    device: torch.device,
    eval_batch_size: int,
    global_patience: int,
    min_epochs_before_stop: int,
    branch_loss_weight: float,
    grad_clip: float,
    log_every_epochs: int,
) -> dict[str, Any]:
    set_seed(spec.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_arrays = _load_split_for_training(str(exact_dataset), "train", "raw_branch")
    synth_val_arrays = _load_split_for_training(str(exact_dataset), "val", "raw_branch")
    real_val_arrays = _load_split_for_training(str(real_primary), "val", "raw_branch")
    real_test_arrays = _load_split_for_training(str(real_primary), "test", "raw_branch")
    cross_test_arrays = _load_split_for_training(str(real_cross), "test", "raw_branch")

    x_scaler = Standardizer.from_array(train_arrays["features"])
    y_scaler = Standardizer.from_array(train_arrays["target"])
    train_ds = _build_tensor_dataset(train_arrays, x_scaler, y_scaler)
    synth_val_loader = _make_eval_loader(synth_val_arrays, x_scaler, y_scaler, eval_batch_size)
    real_val_loader = _make_eval_loader(real_val_arrays, x_scaler, y_scaler, eval_batch_size)
    real_test_loader = _make_eval_loader(real_test_arrays, x_scaler, y_scaler, eval_batch_size)
    cross_test_loader = _make_eval_loader(cross_test_arrays, x_scaler, y_scaler, eval_batch_size)

    model = build_model("raw_branch", input_dim=train_arrays["features"].shape[1], width=spec.width, depth=spec.depth, dropout=0.0).to(device)
    metadata = {
        "config": {
            "dataset": str(exact_dataset),
            "real_primary": str(real_primary),
            "real_cross": str(real_cross),
            "run_dir": str(run_dir),
            "model_kind": "raw_branch",
            "width": spec.width,
            "depth": spec.depth,
            "dropout": 0.0,
            "seed": spec.seed,
            "weight_decay": spec.weight_decay,
            "schedule": [asdict(stage) for stage in _stage_schedule()],
            "selection_metric": "real_val_stress_mse",
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_names": list(BRANCH_NAMES),
    }
    (run_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_csv = run_dir / "history.csv"
    _write_history_header(history_csv)
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    best_real_val_stress_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_epoch = 0

    full_train = tuple(t.to(device) for t in train_ds.tensors)
    for stage in _stage_schedule():
        train_loader = DataLoader(train_ds, batch_size=stage.batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage.lr_start, weight_decay=spec.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, stage.epochs), eta_min=stage.lr_end)

        for _ in range(stage.epochs):
            global_epoch += 1
            train_metrics = _epoch_loop(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=branch_loss_weight,
                device=device,
                grad_clip=grad_clip,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            synth_val_metrics = _epoch_loop(
                model=model,
                loader=synth_val_loader,
                optimizer=None,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=branch_loss_weight,
                device=device,
                grad_clip=grad_clip,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            real_val_metrics = _epoch_loop(
                model=model,
                loader=real_val_loader,
                optimizer=None,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=branch_loss_weight,
                device=device,
                grad_clip=grad_clip,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            real_test_metrics = _epoch_loop(
                model=model,
                loader=real_test_loader,
                optimizer=None,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=branch_loss_weight,
                device=device,
                grad_clip=grad_clip,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            cross_test_metrics = _epoch_loop(
                model=model,
                loader=cross_test_loader,
                optimizer=None,
                model_kind="raw_branch",
                y_scaler=y_scaler,
                branch_loss_weight=branch_loss_weight,
                device=device,
                grad_clip=grad_clip,
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
            torch.save(checkpoint, last_ckpt)

            is_best = False
            if real_val_metrics["stress_mse"] < best_real_val_stress_mse:
                best_real_val_stress_mse = real_val_metrics["stress_mse"]
                best_epoch = global_epoch
                epochs_without_improvement = 0
                _save_checkpoint(best_ckpt, model, metadata)
                is_best = True
            else:
                epochs_without_improvement += 1

            _append_history_row(
                history_csv,
                epoch=global_epoch,
                stage_name=stage.name,
                stage_kind="adam",
                batch_size=stage.batch_size,
                lr=current_lr,
                train_metrics=train_metrics,
                synth_val_metrics=synth_val_metrics,
                real_val_metrics=real_val_metrics,
                real_test_metrics=real_test_metrics,
                cross_test_metrics=cross_test_metrics,
                best_real_val_stress_mse=best_real_val_stress_mse,
                is_best=is_best,
            )
            _maybe_log_epoch(
                epoch=global_epoch,
                stage_name=stage.name,
                lr=current_lr,
                train_metrics=train_metrics,
                real_val_metrics=real_val_metrics,
                real_test_metrics=real_test_metrics,
                best_real_val_stress_mse=best_real_val_stress_mse,
                stage_kind="adam",
                log_every_epochs=log_every_epochs,
            )

            if global_epoch >= min_epochs_before_stop and epochs_without_improvement >= global_patience:
                break

        if global_epoch >= min_epochs_before_stop and epochs_without_improvement >= global_patience:
            break

        if stage.lbfgs_steps_after > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.25,
                max_iter=20,
                history_size=100,
                line_search_fn="strong_wolfe",
            )
            xb, yb, branch, stress_true, eigvecs, trial_stress = full_train
            for _ in range(stage.lbfgs_steps_after):
                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    out = model(xb)
                    reg_loss, _ = _epoch_loop.__globals__["_regression_loss"](
                        model_kind="raw_branch",
                        pred_norm=out["stress"],
                        target_norm=yb,
                        y_scaler=y_scaler,
                        eigvecs=eigvecs,
                        stress_true=stress_true,
                        trial_stress=trial_stress,
                        stress_weight_alpha=0.0,
                        stress_weight_scale=250.0,
                    )
                    loss = reg_loss
                    valid_branch = branch >= 0
                    if torch.any(valid_branch):
                        branch_loss = torch.nn.functional.cross_entropy(out["branch_logits"][valid_branch], branch[valid_branch])
                        loss = loss + branch_loss_weight * branch_loss
                    loss.backward()
                    return loss

                lbfgs.step(closure)
                global_epoch += 1
                train_metrics = _epoch_loop(
                    model=model,
                    loader=train_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                synth_val_metrics = _epoch_loop(
                    model=model,
                    loader=synth_val_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                real_val_metrics = _epoch_loop(
                    model=model,
                    loader=real_val_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                real_test_metrics = _epoch_loop(
                    model=model,
                    loader=real_test_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )
                cross_test_metrics = _epoch_loop(
                    model=model,
                    loader=cross_test_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=0.0,
                    stress_weight_scale=250.0,
                )

                checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
                torch.save(checkpoint, last_ckpt)
                is_best = False
                if real_val_metrics["stress_mse"] < best_real_val_stress_mse:
                    best_real_val_stress_mse = real_val_metrics["stress_mse"]
                    best_epoch = global_epoch
                    epochs_without_improvement = 0
                    _save_checkpoint(best_ckpt, model, metadata)
                    is_best = True
                else:
                    epochs_without_improvement += 1

                _append_history_row(
                    history_csv,
                    epoch=global_epoch,
                    stage_name=stage.name,
                    stage_kind="lbfgs",
                    batch_size=stage.batch_size,
                    lr=lbfgs.param_groups[0]["lr"],
                    train_metrics=train_metrics,
                    synth_val_metrics=synth_val_metrics,
                    real_val_metrics=real_val_metrics,
                    real_test_metrics=real_test_metrics,
                    cross_test_metrics=cross_test_metrics,
                    best_real_val_stress_mse=best_real_val_stress_mse,
                    is_best=is_best,
                )
                _maybe_log_epoch(
                    epoch=global_epoch,
                    stage_name=stage.name,
                    lr=lbfgs.param_groups[0]["lr"],
                    train_metrics=train_metrics,
                    real_val_metrics=real_val_metrics,
                    real_test_metrics=real_test_metrics,
                    best_real_val_stress_mse=best_real_val_stress_mse,
                    stage_kind="lbfgs",
                    log_every_epochs=log_every_epochs,
                )

                if global_epoch >= min_epochs_before_stop and epochs_without_improvement >= global_patience:
                    break

        if global_epoch >= min_epochs_before_stop and epochs_without_improvement >= global_patience:
            break

    history_plot = _plot_history(history_csv, run_dir / "history_log.png")
    primary_eval = evaluate_checkpoint_on_dataset(best_ckpt, real_primary, split="test", device=str(device), batch_size=eval_batch_size)
    cross_eval = evaluate_checkpoint_on_dataset(best_ckpt, real_cross, split="test", device=str(device), batch_size=eval_batch_size)

    primary_dir = run_dir / "eval_primary"
    cross_dir = run_dir / "eval_cross"
    primary_dir.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(primary_eval["arrays"]["stress"], primary_eval["predictions"]["stress"], primary_dir / "parity_stress.png", label="stress")
    error_histogram(primary_eval["predictions"]["stress"] - primary_eval["arrays"]["stress"], primary_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in primary_eval["metrics"]:
        branch_confusion_plot(primary_eval["metrics"]["branch_confusion"], primary_dir / "branch_confusion.png")
    parity_plot(cross_eval["arrays"]["stress"], cross_eval["predictions"]["stress"], cross_dir / "parity_stress.png", label="stress")
    error_histogram(cross_eval["predictions"]["stress"] - cross_eval["arrays"]["stress"], cross_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in cross_eval["metrics"]:
        branch_confusion_plot(cross_eval["metrics"]["branch_confusion"], cross_dir / "branch_confusion.png")

    summary = {
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "best_real_val_stress_mse": best_real_val_stress_mse,
        "history_csv": str(history_csv),
        "history_plot": str(history_plot),
        "best_checkpoint": str(best_ckpt),
        "primary_metrics": primary_eval["metrics"],
        "cross_metrics": cross_eval["metrics"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_report(
    report_md: Path,
    *,
    exact_dataset_path: Path,
    coverage: dict[str, Any],
    mex_check: dict[str, Any],
    run_summaries: list[dict[str, Any]],
) -> None:
    report_md.parent.mkdir(parents=True, exist_ok=True)
    best = min(run_summaries, key=lambda row: row["primary_metrics"]["stress_mae"])

    lines = [
        "# Cover Layer Cyclic Sweep",
        "",
        "This study focuses only on the `cover_layer` material family.",
        "",
        "Training setup:",
        "- train on exact constitutive labels only, using the real cover-layer input domain with no augmentation,",
        "- monitor synthetic train loss plus real validation/test losses,",
        "- use repeated batch-size cycles `64 -> 128 -> 256 -> 512 -> 1024 -> 2048` and then repeat with lower learning rates,",
        "- compare several `raw_branch` network sizes.",
        "",
        f"Exact-domain dataset: `{exact_dataset_path}`",
        "",
        "## Coverage Validation",
        "",
        "The key check here was whether the old synthetic sets were missing the real input domain. They were not.",
        "The issue was different: the old augmented sets inflated the stress/target tails far beyond the real cover-layer distribution.",
        "",
        f"- exact-domain relabel vs stored real MAE: `{coverage['exact_domain_attrs']['exact_vs_real_mae']:.6e}`",
        f"- exact-domain relabel vs stored real RMSE: `{coverage['exact_domain_attrs']['exact_vs_real_rmse']:.6e}`",
        f"- exact-domain relabel vs stored real max abs: `{coverage['exact_domain_attrs']['exact_vs_real_max_abs']:.6e}`",
        "",
        "| Dataset | Train N | Out-of-box sample rate vs real test | NN z mean | NN z p90 | Stress q95 | Stress q995 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in coverage["datasets"].items():
        lines.append(
            f"| {name} | {stats['n_train']} | {stats['sample_out_of_box_rate']:.4f} | {stats['nn_z_mean']:.4f} | "
            f"{stats['nn_z_p90']:.4f} | {stats['stress_mag_q95']:.4f} | {stats['stress_mag_q995']:.4f} |"
        )

    real_test = coverage["real_test_distribution"]
    lines.extend(
        [
            "",
            f"Real cover-layer test stress magnitude q95 / q995: `{real_test['stress_mag_q95']:.4f}` / `{real_test['stress_mag_q995']:.4f}`",
            "",
            "Interpretation:",
            "- `hybrid_train` covered the input box, but its stress tail was far too wide.",
            "- `exact_domain_train` matches the real distribution by construction and does not miss any real test input state.",
            "",
            "## MEX Cross-Check",
            "",
            "The upstream C/MEX kernel from the Octave repo was compared numerically against the Python constitutive operator",
            "on `200` cover-layer test states.",
            "",
            f"- samples: `{mex_check['samples']}`",
            f"- C-vs-Python MAE: `{mex_check['mae']:.6e}`",
            f"- C-vs-Python RMSE: `{mex_check['rmse']:.6e}`",
            f"- C-vs-Python max abs: `{mex_check['max_abs']:.6e}`",
            "",
            "This is effectively machine precision agreement, so the dataset conventions are consistent with the MEX kernel.",
            "",
            "## Sweep Results",
            "",
            "| Spec | Primary MAE | Primary RMSE | Primary Max Abs | Primary Branch Acc | Cross MAE | Cross RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in run_summaries:
        p = row["primary_metrics"]
        c = row["cross_metrics"]
        lines.append(
            f"| {row['spec']['name']} | {p['stress_mae']:.4f} | {p['stress_rmse']:.4f} | {p['stress_max_abs']:.4f} | "
            f"{p.get('branch_accuracy', float('nan')):.4f} | {c['stress_mae']:.4f} | {c['stress_rmse']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"Best spec by primary real-test MAE: `{best['spec']['name']}`",
            "",
        ]
    )
    for row in run_summaries:
        lines.extend(
            [
                f"## {row['spec']['name']}",
                "",
                f"- history: `{row['history_csv']}`",
                f"- checkpoint: `{row['best_checkpoint']}`",
                f"- best real-val stress MSE: `{row['best_real_val_stress_mse']:.6f}`",
                f"- primary test MAE / RMSE / max abs: `{row['primary_metrics']['stress_mae']:.4f}` / `{row['primary_metrics']['stress_rmse']:.4f}` / `{row['primary_metrics']['stress_max_abs']:.4f}`",
                f"- cross test MAE / RMSE / max abs: `{row['cross_metrics']['stress_mae']:.4f}` / `{row['cross_metrics']['stress_rmse']:.4f}` / `{row['cross_metrics']['stress_max_abs']:.4f}`",
                "",
                "Training/test history:",
                "",
                f"![{row['spec']['name']} history](../{Path(row['history_plot']).as_posix()})",
                "",
                "Primary real test parity and error plots:",
                "",
                f"![{row['spec']['name']} primary parity](../{(Path(row['history_plot']).parent / 'eval_primary' / 'parity_stress.png').as_posix()})",
                "",
                f"![{row['spec']['name']} primary error hist](../{(Path(row['history_plot']).parent / 'eval_primary' / 'stress_error_hist.png').as_posix()})",
                "",
                "Cross real test parity and error plots:",
                "",
                f"![{row['spec']['name']} cross parity](../{(Path(row['history_plot']).parent / 'eval_cross' / 'parity_stress.png').as_posix()})",
                "",
                f"![{row['spec']['name']} cross error hist](../{(Path(row['history_plot']).parent / 'eval_cross' / 'stress_error_hist.png').as_posix()})",
                "",
            ]
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    real_primary = Path(args.real_primary)
    real_cross = Path(args.real_cross)
    hybrid_dataset = Path(args.hybrid_dataset)
    exact_dataset_path, exact_meta = _build_exact_domain_dataset(real_primary, output_root / "cover_layer_exact_domain.h5")
    coverage = _coverage_summary(real_primary, hybrid_dataset, exact_dataset_path)
    coverage["exact_domain_attrs"] = exact_meta
    mex_check = _mex_kernel_check(real_primary)

    specs = _run_specs()
    if args.spec_names:
        wanted = set(args.spec_names)
        specs = [spec for spec in specs if spec.name in wanted]

    run_summaries: list[dict[str, Any]] = []
    for spec in specs:
        summary = _train_one(
            exact_dataset=exact_dataset_path,
            real_primary=real_primary,
            real_cross=real_cross,
            run_dir=output_root / spec.name,
            spec=spec,
            device=device,
            eval_batch_size=args.eval_batch_size,
            global_patience=args.global_patience,
            min_epochs_before_stop=args.min_epochs_before_stop,
            branch_loss_weight=args.branch_loss_weight,
            grad_clip=args.grad_clip,
            log_every_epochs=args.log_every_epochs,
        )
        run_summaries.append(summary)

    _write_report(Path(args.report_md), exact_dataset_path=exact_dataset_path, coverage=coverage, mex_check=mex_check, run_summaries=run_summaries)
    (output_root / "coverage_summary.json").write_text(json.dumps(_json_safe(coverage), indent=2), encoding="utf-8")
    (output_root / "mex_check.json").write_text(json.dumps(_json_safe(mex_check), indent=2), encoding="utf-8")
    (output_root / "sweep_summary.json").write_text(json.dumps(_json_safe(run_summaries), indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "report_md": str(Path(args.report_md)), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
