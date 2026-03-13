#!/usr/bin/env python
"""Fit a B-based displacement copula for the cover layer and train with refreshed synthetic data."""

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
from scipy.stats import norm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import REDUCED_MATERIAL_COLUMNS, SPLIT_NAMES, SPLIT_TO_ID
from mc_surrogate.fe_p2_tetra import (
    LocalIntegrationPointPool,
    build_local_b_pool,
    positive_corner_volume_mask,
    strain_from_local_displacements,
)
from mc_surrogate.models import (
    Standardizer,
    build_model,
    build_raw_features,
    compute_trial_stress,
    spectral_decomposition_from_strain,
)
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.training import (
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    evaluate_checkpoint_on_dataset,
    set_seed,
)
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


COPULA_MODES = ("raw", "pca", "branch_raw", "branch_pca", "branch_raw_blend")


@dataclass(frozen=True)
class RunSpec:
    name: str
    width: int
    depth: int
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class GaussianCopulaModel:
    mode: str
    sorted_values: np.ndarray
    chol: np.ndarray
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None


@dataclass(frozen=True)
class BCopulaFit:
    real_primary: str
    mode: str
    geometry_pool: LocalIntegrationPointPool
    copula_models: dict[int, GaussianCopulaModel]
    train_material_reduced: np.ndarray
    train_pseudo_u_norm: np.ndarray
    train_split_arrays: dict[str, np.ndarray]
    train_branch_id: np.ndarray
    train_stress: np.ndarray
    branch_probs: np.ndarray
    branch_material_indices: tuple[np.ndarray, ...]
    principal_cap: float
    global_stress_cap: float
    branch_principal_cap: np.ndarray
    branch_stress_cap: np.ndarray
    real_distribution_summary: dict[str, Any]
    synthetic_distribution_summary: dict[str, Any]
    distribution_fit_score: dict[str, float]
    min_volume_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-primary",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5",
    )
    parser.add_argument(
        "--mesh-path",
        default="/tmp/slope_stability_octave/slope_stability/meshes/SSR_hetero_ada_L1.h5",
    )
    parser.add_argument("--material-id", type=int, default=0)
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_b_copula_20260312")
    parser.add_argument("--report-md", default="docs/cover_layer_b_copula_refresh.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--train-size", type=int, default=65536)
    parser.add_argument("--synthetic-test-size", type=int, default=16384)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--base-lr", type=float, default=1.0e-4)
    parser.add_argument("--cycle-lr-decay", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--stage-patience", type=int, default=15)
    parser.add_argument("--improvement-rel-tol", type=float, default=1.0e-4)
    parser.add_argument("--improvement-abs-tol", type=float, default=1.0e-7)
    parser.add_argument("--branch-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stress-weight-alpha", type=float, default=0.0)
    parser.add_argument("--stress-weight-scale", type=float, default=250.0)
    parser.add_argument("--copula-modes", nargs="+", default=list(COPULA_MODES), choices=list(COPULA_MODES))
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--pca-variance", type=float, default=0.999)
    parser.add_argument("--right-inverse-damping", type=float, default=1.0e-10)
    parser.add_argument("--min-volume-ratio", type=float, default=0.05)
    parser.add_argument("--principal-cap-quantile", type=float, default=0.995)
    parser.add_argument("--stress-cap-quantile", type=float, default=0.995)
    parser.add_argument("--cap-slack", type=float, default=1.10)
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


def _format_runtime(seconds: float) -> str:
    total_seconds = int(max(seconds, 0.0))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def _stage_initial_lr(*, cycle_idx: int, batch_size: int, base_lr: float, cycle_lr_decay: float) -> float:
    cycle_lr = base_lr * (cycle_lr_decay ** max(cycle_idx - 1, 0))
    batch_scale = math.sqrt(64.0 / float(batch_size))
    return cycle_lr * batch_scale


def _maybe_improved(current: float, best: float, rel_tol: float, abs_tol: float) -> bool:
    if not math.isfinite(best):
        return True
    threshold = max(abs_tol, rel_tol * max(abs(best), 1.0))
    return current < best - threshold


def _load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
    return arrays


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(arrays["strain_eng"].shape[0])
    split_id = np.full(n, SPLIT_TO_ID["test"], dtype=np.int8)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.create_dataset("split_id", data=split_id, compression="gzip", shuffle=True)
        f.attrs["branch_names_json"] = json.dumps(BRANCH_NAMES)
        f.attrs["reduced_material_columns_json"] = json.dumps(REDUCED_MATERIAL_COLUMNS)
        f.attrs["split_names_json"] = json.dumps(SPLIT_NAMES)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(_json_safe(value))
            else:
                f.attrs[key] = value
    return path


def _split_arrays_for_raw_branch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "features": build_raw_features(arrays["strain_eng"], arrays["material_reduced"]).astype(np.float32),
        "target": arrays["stress"].astype(np.float32),
        "stress_true": arrays["stress"].astype(np.float32),
        "branch_id": arrays["branch_id"].astype(np.int64),
        "eigvecs": arrays["eigvecs"].astype(np.float32),
        "trial_stress": compute_trial_stress(arrays["strain_eng"], arrays["material_reduced"]).astype(np.float32),
    }


def _distribution_summary(strain_eng: np.ndarray, stress: np.ndarray, branch_id: np.ndarray) -> dict[str, Any]:
    strain_principal, _ = spectral_decomposition_from_strain(strain_eng)
    max_abs_principal = np.max(np.abs(strain_principal), axis=1)
    stress_mag = np.linalg.norm(stress, axis=1)
    branch_freq = np.bincount(branch_id.astype(int), minlength=len(BRANCH_NAMES)).astype(np.float64)
    branch_freq /= max(branch_freq.sum(), 1.0)
    return {
        "n_samples": int(strain_eng.shape[0]),
        "max_abs_principal_q50": float(np.quantile(max_abs_principal, 0.5)),
        "max_abs_principal_q95": float(np.quantile(max_abs_principal, 0.95)),
        "max_abs_principal_q995": float(np.quantile(max_abs_principal, 0.995)),
        "stress_mag_q50": float(np.quantile(stress_mag, 0.5)),
        "stress_mag_q95": float(np.quantile(stress_mag, 0.95)),
        "stress_mag_q995": float(np.quantile(stress_mag, 0.995)),
        "branch_counts": {name: int(np.sum(branch_id == i)) for i, name in enumerate(BRANCH_NAMES)},
        "branch_freq": {name: float(branch_freq[i]) for i, name in enumerate(BRANCH_NAMES)},
    }


def _distribution_fit_score(real_summary: dict[str, Any], synth_summary: dict[str, Any]) -> dict[str, float]:
    principal_q95 = abs(math.log((synth_summary["max_abs_principal_q95"] + 1.0e-12) / (real_summary["max_abs_principal_q95"] + 1.0e-12)))
    principal_q995 = abs(math.log((synth_summary["max_abs_principal_q995"] + 1.0e-12) / (real_summary["max_abs_principal_q995"] + 1.0e-12)))
    stress_q95 = abs(math.log((synth_summary["stress_mag_q95"] + 1.0e-12) / (real_summary["stress_mag_q95"] + 1.0e-12)))
    stress_q995 = abs(math.log((synth_summary["stress_mag_q995"] + 1.0e-12) / (real_summary["stress_mag_q995"] + 1.0e-12)))
    real_freq = np.array([real_summary["branch_freq"][name] for name in BRANCH_NAMES], dtype=float)
    synth_freq = np.array([synth_summary["branch_freq"][name] for name in BRANCH_NAMES], dtype=float)
    branch_tv = 0.5 * float(np.abs(real_freq - synth_freq).sum())
    total = principal_q95 + principal_q995 + stress_q95 + stress_q995 + branch_tv
    return {
        "principal_q95_logerr": principal_q95,
        "principal_q995_logerr": principal_q995,
        "stress_q95_logerr": stress_q95,
        "stress_q995_logerr": stress_q995,
        "branch_tv": branch_tv,
        "total": total,
    }


def _plot_distribution_fit(
    real_arrays: dict[str, np.ndarray],
    synth_arrays: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    real_principal, _ = spectral_decomposition_from_strain(real_arrays["strain_eng"])
    synth_principal, _ = spectral_decomposition_from_strain(synth_arrays["strain_eng"])
    real_max_abs = np.sort(np.max(np.abs(real_principal), axis=1))
    synth_max_abs = np.sort(np.max(np.abs(synth_principal), axis=1))
    real_stress_mag = np.sort(np.linalg.norm(real_arrays["stress"], axis=1))
    synth_stress_mag = np.sort(np.linalg.norm(synth_arrays["stress"], axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(real_max_abs, np.linspace(0.0, 1.0, real_max_abs.size), label="real test")
    axes[0].plot(synth_max_abs, np.linspace(0.0, 1.0, synth_max_abs.size), label="synthetic holdout")
    axes[0].set_xscale("symlog", linthresh=1.0e-3)
    axes[0].set_xlabel("max |principal strain|")
    axes[0].set_ylabel("empirical CDF")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(real_stress_mag, np.linspace(0.0, 1.0, real_stress_mag.size), label="real test")
    axes[1].plot(synth_stress_mag, np.linspace(0.0, 1.0, synth_stress_mag.size), label="synthetic holdout")
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("stress magnitude")
    axes[1].set_ylabel("empirical CDF")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _rank_gaussianize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = data.shape
    z = np.empty((n, d), dtype=np.float64)
    sorted_values = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        col = data[:, j]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n, dtype=np.int64)
        u = (ranks.astype(np.float64) + 0.5) / float(n)
        u = np.clip(u, 1.0e-6, 1.0 - 1.0e-6)
        z[:, j] = norm.ppf(u)
        sorted_values[:, j] = col[order]
    return z, sorted_values


def _regularize_correlation(corr: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    sym = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, eps)
    repaired = (vecs * vals[None, :]) @ vecs.T
    diag = np.sqrt(np.maximum(np.diag(repaired), eps))
    repaired = repaired / np.outer(diag, diag)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _fit_gaussian_copula(data: np.ndarray) -> GaussianCopulaModel:
    z, sorted_values = _rank_gaussianize(data)
    corr = np.corrcoef(z, rowvar=False)
    if np.ndim(corr) == 0:
        corr = np.array([[1.0]], dtype=np.float64)
    corr = _regularize_correlation(np.asarray(corr, dtype=np.float64))
    chol = np.linalg.cholesky(corr + 1.0e-6 * np.eye(corr.shape[0], dtype=np.float64))
    return GaussianCopulaModel(
        mode="raw",
        sorted_values=sorted_values.astype(np.float32),
        chol=chol.astype(np.float32),
        pca_mean=None,
        pca_components=None,
    )


def _fit_pca_copula(data: np.ndarray, variance_fraction: float) -> GaussianCopulaModel:
    centered = data - data.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = s**2
    frac = np.cumsum(var) / np.maximum(var.sum(), 1.0e-12)
    k = int(np.searchsorted(frac, variance_fraction) + 1)
    components = vt[:k]
    latent = centered @ components.T
    z, sorted_values = _rank_gaussianize(latent)
    corr = np.corrcoef(z, rowvar=False)
    if np.ndim(corr) == 0:
        corr = np.array([[1.0]], dtype=np.float64)
    corr = _regularize_correlation(np.asarray(corr, dtype=np.float64))
    chol = np.linalg.cholesky(corr + 1.0e-6 * np.eye(corr.shape[0], dtype=np.float64))
    return GaussianCopulaModel(
        mode="pca",
        sorted_values=sorted_values.astype(np.float32),
        chol=chol.astype(np.float32),
        pca_mean=data.mean(axis=0, keepdims=False).astype(np.float32),
        pca_components=components.astype(np.float32),
    )


def _inverse_empirical(sorted_values: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    n_ref, d = sorted_values.shape
    pos = np.clip(uniform, 1.0e-6, 1.0 - 1.0e-6) * float(n_ref - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, n_ref - 1)
    frac = pos - lo
    out = np.empty((uniform.shape[0], d), dtype=np.float64)
    rows = np.arange(uniform.shape[0])
    for j in range(d):
        vals = sorted_values[:, j]
        out[:, j] = (1.0 - frac[:, j]) * vals[lo[:, j]] + frac[:, j] * vals[hi[:, j]]
    return out


def _sample_copula(model: GaussianCopulaModel, n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, model.chol.shape[0])).astype(np.float64) @ model.chol.T.astype(np.float64)
    u = norm.cdf(z)
    latent = _inverse_empirical(model.sorted_values.astype(np.float64), u)
    if model.mode == "raw":
        return latent.astype(np.float32)
    assert model.pca_mean is not None and model.pca_components is not None
    return (model.pca_mean[None, :] + latent @ model.pca_components.astype(np.float64)).astype(np.float32)


def _load_real_train_test(real_primary: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    arrays = _load_arrays(real_primary)
    train_mask = arrays["split_id"] == SPLIT_TO_ID["train"]
    test_mask = arrays["split_id"] == SPLIT_TO_ID["test"]
    train_arrays = {
        "strain_eng": arrays["strain_eng"][train_mask].astype(np.float32),
        "stress": arrays["stress"][train_mask].astype(np.float32),
        "material_reduced": arrays["material_reduced"][train_mask].astype(np.float32),
        "branch_id": arrays["branch_id"][train_mask].astype(np.int64),
        "eigvecs": arrays["eigvecs"][train_mask].astype(np.float32),
        "stress_principal": arrays["stress_principal"][train_mask].astype(np.float32),
    }
    test_arrays = {
        "strain_eng": arrays["strain_eng"][test_mask].astype(np.float32),
        "stress": arrays["stress"][test_mask].astype(np.float32),
        "branch_id": arrays["branch_id"][test_mask].astype(np.int64),
    }
    info = {
        "train_size": int(train_arrays["strain_eng"].shape[0]),
        "test_size": int(test_arrays["strain_eng"].shape[0]),
        "unique_material_rows": int(np.unique(train_arrays["material_reduced"], axis=0).shape[0]),
    }
    return train_arrays, test_arrays, info


def _pair_real_strain_with_geometry(
    train_strain: np.ndarray,
    geometry_pool: LocalIntegrationPointPool,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    geom_idx = rng.integers(0, geometry_pool.b_blocks.shape[0], size=train_strain.shape[0])
    right_inverse = geometry_pool.right_inverse[geom_idx].astype(np.float64)
    char_length = geometry_pool.characteristic_length[geom_idx].astype(np.float64)
    pseudo_u = np.einsum("nij,nj->ni", right_inverse, train_strain.astype(np.float64))
    pseudo_u_norm = pseudo_u / char_length[:, None]
    return geom_idx.astype(np.int32), pseudo_u_norm.astype(np.float32)


def _fit_b_copula_mode(
    *,
    real_primary: Path,
    geometry_pool: LocalIntegrationPointPool,
    mode: str,
    pca_variance: float,
    seed: int,
    principal_cap_quantile: float,
    stress_cap_quantile: float,
    cap_slack: float,
    min_volume_ratio: float,
    synthetic_test_size: int,
    output_root: Path,
) -> tuple[BCopulaFit, Path]:
    train_arrays, real_test_arrays, info = _load_real_train_test(real_primary)
    _, pseudo_u_norm = _pair_real_strain_with_geometry(train_arrays["strain_eng"], geometry_pool, seed=seed)
    principal_train, _ = spectral_decomposition_from_strain(train_arrays["strain_eng"])

    branch_probs = np.bincount(train_arrays["branch_id"], minlength=len(BRANCH_NAMES)).astype(np.float64)
    branch_probs /= branch_probs.sum()
    branch_material_indices = tuple(np.flatnonzero(train_arrays["branch_id"] == bid) for bid in range(len(BRANCH_NAMES)))
    branch_principal_cap = np.zeros(len(BRANCH_NAMES), dtype=np.float32)
    branch_stress_cap = np.zeros(len(BRANCH_NAMES), dtype=np.float32)
    branch_max_abs = np.max(np.abs(principal_train), axis=1)
    branch_stress_mag = np.linalg.norm(train_arrays["stress"], axis=1)
    for bid in range(len(BRANCH_NAMES)):
        mask = train_arrays["branch_id"] == bid
        branch_principal_cap[bid] = float(np.quantile(branch_max_abs[mask], principal_cap_quantile) * cap_slack)
        branch_stress_cap[bid] = float(np.quantile(branch_stress_mag[mask], stress_cap_quantile) * cap_slack)

    if mode == "raw":
        copula_models = {-1: _fit_gaussian_copula(pseudo_u_norm.astype(np.float64))}
    elif mode == "pca":
        copula_models = {-1: _fit_pca_copula(pseudo_u_norm.astype(np.float64), variance_fraction=pca_variance)}
    elif mode == "branch_raw":
        copula_models = {
            bid: _fit_gaussian_copula(pseudo_u_norm[train_arrays["branch_id"] == bid].astype(np.float64))
            for bid in range(len(BRANCH_NAMES))
        }
    elif mode == "branch_pca":
        copula_models = {
            bid: _fit_pca_copula(
                pseudo_u_norm[train_arrays["branch_id"] == bid].astype(np.float64),
                variance_fraction=pca_variance,
            )
            for bid in range(len(BRANCH_NAMES))
        }
    elif mode == "branch_raw_blend":
        copula_models = {
            bid: _fit_gaussian_copula(pseudo_u_norm[train_arrays["branch_id"] == bid].astype(np.float64))
            for bid in range(len(BRANCH_NAMES))
        }
    else:
        raise ValueError(f"Unsupported copula mode {mode!r}.")

    principal_cap = float(np.quantile(np.max(np.abs(principal_train), axis=1), principal_cap_quantile) * cap_slack)
    stress_cap = float(np.quantile(np.linalg.norm(train_arrays["stress"], axis=1), stress_cap_quantile) * cap_slack)

    provisional_fit = BCopulaFit(
        real_primary=str(real_primary),
        mode=mode,
        geometry_pool=geometry_pool,
        copula_models=copula_models,
        train_material_reduced=train_arrays["material_reduced"],
        train_pseudo_u_norm=pseudo_u_norm.astype(np.float32),
        train_split_arrays=_load_split_for_training(str(real_primary), "train", "raw_branch"),
        train_branch_id=train_arrays["branch_id"],
        train_stress=train_arrays["stress"],
        branch_probs=branch_probs.astype(np.float32),
        branch_material_indices=branch_material_indices,
        principal_cap=principal_cap,
        global_stress_cap=stress_cap,
        branch_principal_cap=branch_principal_cap,
        branch_stress_cap=branch_stress_cap,
        real_distribution_summary={},
        synthetic_distribution_summary={},
        distribution_fit_score={},
        min_volume_ratio=min_volume_ratio,
    )
    synthetic_test_arrays = generate_from_b_copula(fit=provisional_fit, n_samples=synthetic_test_size, seed=seed + 17)
    real_summary = _distribution_summary(real_test_arrays["strain_eng"], real_test_arrays["stress"], real_test_arrays["branch_id"])
    synth_summary = _distribution_summary(
        synthetic_test_arrays["strain_eng"],
        synthetic_test_arrays["stress"],
        synthetic_test_arrays["branch_id"],
    )
    score = _distribution_fit_score(real_summary, synth_summary)

    fit = BCopulaFit(
        real_primary=str(real_primary),
        mode=mode,
        geometry_pool=geometry_pool,
        copula_models=copula_models,
        train_material_reduced=train_arrays["material_reduced"],
        train_pseudo_u_norm=pseudo_u_norm.astype(np.float32),
        train_split_arrays=_load_split_for_training(str(real_primary), "train", "raw_branch"),
        train_branch_id=train_arrays["branch_id"],
        train_stress=train_arrays["stress"],
        branch_probs=branch_probs.astype(np.float32),
        branch_material_indices=branch_material_indices,
        principal_cap=principal_cap,
        global_stress_cap=stress_cap,
        branch_principal_cap=branch_principal_cap,
        branch_stress_cap=branch_stress_cap,
        real_distribution_summary=real_summary,
        synthetic_distribution_summary=synth_summary,
        distribution_fit_score=score,
        min_volume_ratio=min_volume_ratio,
    )

    plot_path = output_root / f"distribution_fit_{mode}.png"
    _plot_distribution_fit(real_test_arrays, synthetic_test_arrays, plot_path)
    synthetic_test_path = _write_dataset(
        output_root / f"synthetic_test_{mode}.h5",
        synthetic_test_arrays,
        attrs={
            "generator": "cover_layer_b_copula",
            "real_primary": str(real_primary),
            "mesh_path": "cover_layer_geometry_pool",
            "copula_mode": mode,
            "fit_info": info,
            "distribution_fit_score": score,
            "distribution_plot": str(plot_path),
        },
    )
    return fit, synthetic_test_path


def generate_from_b_copula(
    *,
    fit: BCopulaFit,
    n_samples: int,
    seed: int,
    return_aux: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pool = fit.geometry_pool
    chunks: list[dict[str, np.ndarray]] = []
    aux_chunks: list[dict[str, np.ndarray]] = []
    if -1 in fit.copula_models:
        branch_plan = { -1: int(n_samples) }
    else:
        desired = np.floor(fit.branch_probs * n_samples).astype(np.int64)
        desired[-1] += int(n_samples - desired.sum())
        branch_plan = {bid: int(desired[bid]) for bid in range(len(BRANCH_NAMES))}

    for branch_id, target_count in branch_plan.items():
        remaining = int(target_count)
        attempts = 0
        while remaining > 0:
            attempts += 1
            if attempts > 5000:
                raise RuntimeError(f"Failed to generate enough accepted B-copula samples for mode {fit.mode!r}, branch={branch_id}.")
            batch = max(2048, remaining * 4)
            geom_idx = rng.integers(0, pool.b_blocks.shape[0], size=batch)
            u_norm = _sample_copula(fit.copula_models[branch_id], batch, rng)
            base_idx: np.ndarray | None = None
            if branch_id >= 0:
                pool_idx = fit.branch_material_indices[branch_id]
                base_idx = pool_idx[rng.integers(0, pool_idx.shape[0], size=batch)]
                if fit.mode.endswith("_blend"):
                    alpha = np.clip(rng.normal(loc=0.82, scale=0.08, size=batch), 0.60, 0.97).astype(np.float32)
                    empirical_u = fit.train_pseudo_u_norm[base_idx]
                    u_norm = alpha[:, None] * empirical_u + (1.0 - alpha)[:, None] * u_norm
            char_length = pool.characteristic_length[geom_idx][:, None]
            displacement = u_norm * char_length
            volume_ok = positive_corner_volume_mask(
                pool.local_coords[geom_idx],
                displacement,
                min_volume_ratio=fit.min_volume_ratio,
            )
            if not np.any(volume_ok):
                continue
            geom_idx = geom_idx[volume_ok]
            displacement = displacement[volume_ok]
            if base_idx is not None:
                base_idx = base_idx[volume_ok]
            strain_eng = strain_from_local_displacements(pool.b_blocks[geom_idx], displacement)
            principal, _ = spectral_decomposition_from_strain(strain_eng)
            max_abs_principal = np.max(np.abs(principal), axis=1)
            strain_ok = max_abs_principal <= fit.principal_cap
            if branch_id >= 0:
                strain_ok &= max_abs_principal <= fit.branch_principal_cap[branch_id]
            if not np.any(strain_ok):
                continue
            geom_idx = geom_idx[strain_ok]
            strain_eng = strain_eng[strain_ok]
            if base_idx is not None:
                base_idx = base_idx[strain_ok]
            n_keep = strain_eng.shape[0]
            if branch_id == -1:
                mat_idx = rng.integers(0, fit.train_material_reduced.shape[0], size=n_keep)
            else:
                assert base_idx is not None
                mat_idx = base_idx
            material = fit.train_material_reduced[mat_idx].astype(np.float32)
            response = constitutive_update_3d(
                strain_eng,
                c_bar=material[:, 0],
                sin_phi=material[:, 1],
                shear=material[:, 2],
                bulk=material[:, 3],
                lame=material[:, 4],
            )
            stress_mag = np.linalg.norm(response.stress, axis=1)
            accept = stress_mag <= fit.global_stress_cap
            if branch_id >= 0:
                accept &= response.branch_id == branch_id
                accept &= stress_mag <= fit.branch_stress_cap[branch_id]
            take = np.flatnonzero(accept)[:remaining]
            if take.size == 0:
                continue
            chunks.append(
                {
                    "strain_eng": strain_eng[take].astype(np.float32),
                    "stress": response.stress[take].astype(np.float32),
                    "material_reduced": material[take].astype(np.float32),
                    "branch_id": response.branch_id[take].astype(np.int8),
                    "eigvecs": response.eigvecs[take].astype(np.float32),
                    "stress_principal": response.stress_principal[take].astype(np.float32),
                }
            )
            if return_aux:
                aux_chunks.append(
                    {
                        "u_norm": u_norm[take].astype(np.float32),
                        "geom_idx": geom_idx[take].astype(np.int32),
                        "target_branch_id": np.full(take.size, branch_id, dtype=np.int16),
                    }
                )
            remaining -= int(take.size)

    arrays = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0].keys()}
    perm = rng.permutation(arrays["strain_eng"].shape[0])
    arrays = {key: value[perm] for key, value in arrays.items()}
    if not return_aux:
        return arrays
    aux = {key: np.concatenate([chunk[key] for chunk in aux_chunks], axis=0) for key in aux_chunks[0].keys()}
    aux = {key: value[perm] for key, value in aux.items()}
    return arrays, aux


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "stage_name",
                "batch_size",
                "train_seed",
                "lr",
                "train_loss",
                "train_stress_mse",
                "real_val_loss",
                "real_val_stress_mse",
                "real_test_loss",
                "real_test_stress_mse",
                "synthetic_test_loss",
                "synthetic_test_stress_mse",
                "train_branch_accuracy",
                "real_val_branch_accuracy",
                "real_test_branch_accuracy",
                "synthetic_test_branch_accuracy",
                "best_real_val_stress_mse",
                "is_best",
            ]
        )


def _append_history_row(
    path: Path,
    *,
    epoch: int,
    stage_name: str,
    batch_size: int,
    train_seed: int,
    lr: float,
    train_metrics: dict[str, float],
    real_val_metrics: dict[str, float],
    real_test_metrics: dict[str, float],
    synthetic_test_metrics: dict[str, float],
    best_real_val_stress_mse: float,
    is_best: bool,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                stage_name,
                batch_size,
                train_seed,
                lr,
                train_metrics["loss"],
                train_metrics["stress_mse"],
                real_val_metrics["loss"],
                real_val_metrics["stress_mse"],
                real_test_metrics["loss"],
                real_test_metrics["stress_mse"],
                synthetic_test_metrics["loss"],
                synthetic_test_metrics["stress_mse"],
                train_metrics["branch_accuracy"],
                real_val_metrics["branch_accuracy"],
                real_test_metrics["branch_accuracy"],
                synthetic_test_metrics["branch_accuracy"],
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
    axes[0].plot(epoch, arr("real_val_loss"), label="real val")
    axes[0].plot(epoch, arr("real_test_loss"), label="real test")
    axes[0].plot(epoch, arr("synthetic_test_loss"), label="synthetic test")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch, arr("train_stress_mse"), label="train")
    axes[1].plot(epoch, arr("real_val_stress_mse"), label="real val")
    axes[1].plot(epoch, arr("real_test_stress_mse"), label="real test")
    axes[1].plot(epoch, arr("synthetic_test_stress_mse"), label="synthetic test")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("stress mse")
    axes[1].set_xlabel("global epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ymax0 = float(max(arr("train_loss").max(), arr("real_val_loss").max(), arr("real_test_loss").max(), arr("synthetic_test_loss").max()))
    ymax1 = float(max(arr("train_stress_mse").max(), arr("real_val_stress_mse").max(), arr("real_test_stress_mse").max(), arr("synthetic_test_stress_mse").max()))
    for ax in axes:
        for x, _ in boundaries:
            ax.axvline(x, color="k", linestyle="--", alpha=0.2)
    for x, label in boundaries:
        axes[0].text(x + 1, ymax0, label, rotation=90, va="top", ha="left", fontsize=8)
        axes[1].text(x + 1, ymax1, label, rotation=90, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_branch_accuracy(history_csv: Path, output_path: Path) -> Path:
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

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(epoch, arr("train_branch_accuracy"), label="train")
    ax.plot(epoch, arr("real_val_branch_accuracy"), label="real val")
    ax.plot(epoch, arr("real_test_branch_accuracy"), label="real test")
    ax.plot(epoch, arr("synthetic_test_branch_accuracy"), label="synthetic test")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("branch accuracy")
    ax.set_xlabel("global epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for x, _ in boundaries:
        ax.axvline(x, color="k", linestyle="--", alpha=0.15)
    for x, label in boundaries:
        ax.text(x + 1, 0.99, label, rotation=90, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def _evaluate_and_plot(
    best_ckpt: Path,
    *,
    real_primary: Path,
    synthetic_test_path: Path,
    run_dir: Path,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    real_eval = evaluate_checkpoint_on_dataset(best_ckpt, real_primary, split="test", device=str(device), batch_size=eval_batch_size)
    synthetic_eval = evaluate_checkpoint_on_dataset(best_ckpt, synthetic_test_path, split="test", device=str(device), batch_size=eval_batch_size)

    real_dir = run_dir / "eval_real"
    synth_dir = run_dir / "eval_synth"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(real_eval["arrays"]["stress"], real_eval["predictions"]["stress"], real_dir / "parity_stress.png", label="stress")
    error_histogram(real_eval["predictions"]["stress"] - real_eval["arrays"]["stress"], real_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in real_eval["metrics"]:
        branch_confusion_plot(real_eval["metrics"]["branch_confusion"], real_dir / "branch_confusion.png")
    parity_plot(synthetic_eval["arrays"]["stress"], synthetic_eval["predictions"]["stress"], synth_dir / "parity_stress.png", label="stress")
    error_histogram(synthetic_eval["predictions"]["stress"] - synthetic_eval["arrays"]["stress"], synth_dir / "stress_error_hist.png", label="stress error")
    if "branch_confusion" in synthetic_eval["metrics"]:
        branch_confusion_plot(synthetic_eval["metrics"]["branch_confusion"], synth_dir / "branch_confusion.png")
    return real_eval, synthetic_eval


def _train_refresh_run(
    *,
    fit: BCopulaFit,
    synthetic_test_path: Path,
    run_dir: Path,
    spec: RunSpec,
    device: torch.device,
    eval_batch_size: int,
    branch_loss_weight: float,
    grad_clip: float,
    stress_weight_alpha: float,
    stress_weight_scale: float,
    train_size: int,
    cycles: int,
    batch_sizes: list[int],
    base_lr: float,
    cycle_lr_decay: float,
    min_lr: float,
    plateau_patience: int,
    stage_patience: int,
    improvement_rel_tol: float,
    improvement_abs_tol: float,
) -> dict[str, Any]:
    set_seed(spec.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    x_scaler = Standardizer.from_array(fit.train_split_arrays["features"])
    y_scaler = Standardizer.from_array(fit.train_split_arrays["target"])

    real_val_arrays = _load_split_for_training(fit.real_primary, "val", "raw_branch")
    real_test_arrays = _load_split_for_training(fit.real_primary, "test", "raw_branch")
    synthetic_test_arrays = _load_split_for_training(str(synthetic_test_path), "test", "raw_branch")

    real_val_loader = DataLoader(_build_tensor_dataset(real_val_arrays, x_scaler, y_scaler), batch_size=eval_batch_size, shuffle=False, num_workers=0)
    real_test_loader = DataLoader(_build_tensor_dataset(real_test_arrays, x_scaler, y_scaler), batch_size=eval_batch_size, shuffle=False, num_workers=0)
    synthetic_test_loader = DataLoader(_build_tensor_dataset(synthetic_test_arrays, x_scaler, y_scaler), batch_size=eval_batch_size, shuffle=False, num_workers=0)

    model = build_model("raw_branch", input_dim=fit.train_split_arrays["features"].shape[1], width=spec.width, depth=spec.depth, dropout=0.0).to(device)
    metadata = {
        "config": {
            "dataset": "epoch_refreshed_b_copula_synthetic_train",
            "real_primary": fit.real_primary,
            "synthetic_test": str(synthetic_test_path),
            "run_dir": str(run_dir),
            "model_kind": "raw_branch",
            "width": spec.width,
            "depth": spec.depth,
            "dropout": 0.0,
            "seed": spec.seed,
            "weight_decay": spec.weight_decay,
            "cycles": cycles,
            "batch_sizes": batch_sizes,
            "base_lr": base_lr,
            "cycle_lr_decay": cycle_lr_decay,
            "min_lr": min_lr,
            "plateau_patience": plateau_patience,
            "stage_patience": stage_patience,
            "selection_metric": "real_val_stress_mse",
            "copula_mode": fit.mode,
            "train_size": train_size,
            "principal_cap": fit.principal_cap,
            "global_stress_cap": fit.global_stress_cap,
            "stress_weight_alpha": stress_weight_alpha,
            "stress_weight_scale": stress_weight_scale,
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
    global_epoch = 0
    total_start = time.perf_counter()

    for cycle_idx in range(1, cycles + 1):
        for batch_size in batch_sizes:
            stage_name = f"cycle{cycle_idx}_bs{batch_size}"
            stage_lr = _stage_initial_lr(
                cycle_idx=cycle_idx,
                batch_size=batch_size,
                base_lr=base_lr,
                cycle_lr_decay=cycle_lr_decay,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=spec.weight_decay)
            stage_best_val = float("inf")
            stage_bad_epochs = 0
            lr_bad_epochs = 0
            stage_start = time.perf_counter()
            _log_message(
                log_path,
                (
                    f"[stage-start] mode={fit.mode} cycle={cycle_idx}/{cycles} batch={batch_size} lr={stage_lr:.3e} "
                    f"global_epoch={global_epoch} runtime={_format_runtime(time.perf_counter() - total_start)}"
                ),
            )

            while True:
                global_epoch += 1
                train_seed = spec.seed + 100000 + global_epoch * 7919
                train_arrays = generate_from_b_copula(fit=fit, n_samples=train_size, seed=train_seed)
                train_split = _split_arrays_for_raw_branch(train_arrays)
                train_ds = _build_tensor_dataset(train_split, x_scaler, y_scaler)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

                train_metrics = _epoch_loop(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=stress_weight_alpha,
                    stress_weight_scale=stress_weight_scale,
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
                    stress_weight_alpha=stress_weight_alpha,
                    stress_weight_scale=stress_weight_scale,
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
                    stress_weight_alpha=stress_weight_alpha,
                    stress_weight_scale=stress_weight_scale,
                )
                synthetic_test_metrics = _epoch_loop(
                    model=model,
                    loader=synthetic_test_loader,
                    optimizer=None,
                    model_kind="raw_branch",
                    y_scaler=y_scaler,
                    branch_loss_weight=branch_loss_weight,
                    device=device,
                    grad_clip=grad_clip,
                    stress_weight_alpha=stress_weight_alpha,
                    stress_weight_scale=stress_weight_scale,
                )

                current_lr = optimizer.param_groups[0]["lr"]
                checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
                torch.save(checkpoint, last_ckpt)

                is_best = False
                if real_val_metrics["stress_mse"] < best_real_val_stress_mse:
                    best_real_val_stress_mse = real_val_metrics["stress_mse"]
                    best_epoch = global_epoch
                    _save_checkpoint(best_ckpt, model, metadata)
                    is_best = True

                _append_history_row(
                    history_csv,
                    epoch=global_epoch,
                    stage_name=stage_name,
                    batch_size=batch_size,
                    train_seed=train_seed,
                    lr=current_lr,
                    train_metrics=train_metrics,
                    real_val_metrics=real_val_metrics,
                    real_test_metrics=real_test_metrics,
                    synthetic_test_metrics=synthetic_test_metrics,
                    best_real_val_stress_mse=best_real_val_stress_mse,
                    is_best=is_best,
                )
                _log_message(
                    log_path,
                    (
                        f"[adam] mode={fit.mode} epoch={global_epoch} cycle={cycle_idx}/{cycles} batch={batch_size} lr={current_lr:.3e} "
                        f"train_seed={train_seed} runtime={_format_runtime(time.perf_counter() - total_start)} "
                        f"train_loss={train_metrics['loss']:.6f} real_val_stress_mse={real_val_metrics['stress_mse']:.6f} "
                        f"real_test_stress_mse={real_test_metrics['stress_mse']:.6f} synth_test_stress_mse={synthetic_test_metrics['stress_mse']:.6f} "
                        f"best_real_val={best_real_val_stress_mse:.6f}"
                    ),
                )

                if _maybe_improved(
                    real_val_metrics["stress_mse"],
                    stage_best_val,
                    rel_tol=improvement_rel_tol,
                    abs_tol=improvement_abs_tol,
                ):
                    stage_best_val = real_val_metrics["stress_mse"]
                    stage_bad_epochs = 0
                    lr_bad_epochs = 0
                else:
                    stage_bad_epochs += 1
                    lr_bad_epochs += 1
                    if lr_bad_epochs >= plateau_patience and current_lr > min_lr * (1.0 + 1.0e-12):
                        new_lr = max(current_lr * 0.5, min_lr)
                        for group in optimizer.param_groups:
                            group["lr"] = new_lr
                        lr_bad_epochs = 0
                        _log_message(
                            log_path,
                            (
                                f"[lr-drop] mode={fit.mode} epoch={global_epoch} cycle={cycle_idx}/{cycles} batch={batch_size} "
                                f"old_lr={current_lr:.3e} new_lr={new_lr:.3e} "
                                f"runtime={_format_runtime(time.perf_counter() - total_start)}"
                            ),
                        )

                if stage_bad_epochs >= stage_patience:
                    _log_message(
                        log_path,
                        (
                            f"[stage-stop] mode={fit.mode} cycle={cycle_idx}/{cycles} batch={batch_size} "
                            f"stage_runtime={_format_runtime(time.perf_counter() - stage_start)} "
                            f"total_runtime={_format_runtime(time.perf_counter() - total_start)} "
                            f"best_stage_real_val={stage_best_val:.6f}"
                        ),
                    )
                    break

    elapsed = time.perf_counter() - total_start
    history_plot = _plot_history(history_csv, run_dir / "history_log.png")
    branch_plot = _plot_branch_accuracy(history_csv, run_dir / "branch_accuracy.png")
    real_eval, synthetic_eval = _evaluate_and_plot(
        best_ckpt,
        real_primary=Path(fit.real_primary),
        synthetic_test_path=synthetic_test_path,
        run_dir=run_dir,
        device=device,
        eval_batch_size=eval_batch_size,
    )

    summary = {
        "spec": asdict(spec),
        "copula_mode": fit.mode,
        "best_epoch": best_epoch,
        "best_real_val_stress_mse": best_real_val_stress_mse,
        "history_csv": str(history_csv),
        "history_plot": str(history_plot),
        "branch_plot": str(branch_plot),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "log_path": str(log_path),
        "elapsed_seconds": elapsed,
        "elapsed_hms": _format_runtime(elapsed),
        "distribution_fit_score": fit.distribution_fit_score,
        "real_test_metrics": real_eval["metrics"],
        "synthetic_test_metrics": synthetic_eval["metrics"],
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _write_report(
    report_md: Path,
    *,
    output_root: Path,
    mesh_path: Path,
    geometry_pool: LocalIntegrationPointPool,
    fit_results: dict[str, BCopulaFit],
    synthetic_test_paths: dict[str, Path],
    run_summaries: dict[str, dict[str, Any]],
) -> None:
    def rel_repo(path: str | Path) -> str:
        return Path(path).resolve().relative_to(ROOT).as_posix()

    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cover Layer B-Copula Refresh",
        "",
        "This study replaces direct strain-space perturbation with a local FE-driven generator.",
        "Synthetic samples are produced by sampling local P2 tetra nodal displacements, filtering invalid corner-volume changes, and then mapping them through the local `B` operator to engineering strains.",
        "",
        "## FE Geometry Pool",
        "",
        f"- mesh: `{mesh_path}`",
        f"- local integration-point pool size: `{geometry_pool.b_blocks.shape[0]}`",
        f"- element count in pool: `{np.unique(geometry_pool.element_index).size}`",
        f"- quadrature points per element: `{int(np.max(geometry_pool.quadrature_index) + 1)}`",
        f"- mean characteristic length: `{float(np.mean(geometry_pool.characteristic_length)):.4f}`",
        "",
        "## Distribution Fits",
        "",
        "| mode | total score | principal q95 logerr | principal q995 logerr | stress q95 logerr | stress q995 logerr | branch TV | synthetic test |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode, fit in fit_results.items():
        score = fit.distribution_fit_score
        synth_test = synthetic_test_paths[mode]
        lines.append(
            f"| `{mode}` | `{score['total']:.4f}` | `{score['principal_q95_logerr']:.4f}` | `{score['principal_q995_logerr']:.4f}` | "
            f"`{score['stress_q95_logerr']:.4f}` | `{score['stress_q995_logerr']:.4f}` | `{score['branch_tv']:.4f}` | "
            f"`{rel_repo(synth_test)}` |"
        )
    lines.extend([""])

    for mode, fit in fit_results.items():
        plot_path = output_root / f"distribution_fit_{mode}.png"
        if plot_path.exists():
            lines.extend([f"### {mode}", "", f"![distribution fit {mode}](../{rel_repo(plot_path)})", ""])

    if run_summaries:
        lines.extend(
            [
                "## Training Results",
                "",
                "| mode | real MAE | real RMSE | real max abs | real branch acc | synth MAE | synth RMSE | synth max abs | synth branch acc | checkpoint |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for mode, summary in run_summaries.items():
            real_metrics = summary["real_test_metrics"]
            synth_metrics = summary["synthetic_test_metrics"]
            lines.append(
                f"| `{mode}` | `{real_metrics['stress_mae']:.4f}` | `{real_metrics['stress_rmse']:.4f}` | `{real_metrics['stress_max_abs']:.4f}` | "
                f"`{real_metrics.get('branch_accuracy', float('nan')):.4f}` | `{synth_metrics['stress_mae']:.4f}` | `{synth_metrics['stress_rmse']:.4f}` | "
                f"`{synth_metrics['stress_max_abs']:.4f}` | `{synth_metrics.get('branch_accuracy', float('nan')):.4f}` | "
                f"`{rel_repo(summary['best_checkpoint'])}` |"
            )
        lines.extend([""])
        best_mode = min(run_summaries, key=lambda key: run_summaries[key]["real_test_metrics"]["stress_mae"])
        best_summary = run_summaries[best_mode]
        lines.extend(
            [
                "## Best Run",
                "",
                f"- best mode by real-test MAE: `{best_mode}`",
                f"- history: `{rel_repo(best_summary['history_csv'])}`",
                f"- history plot: `{rel_repo(best_summary['history_plot'])}`",
                f"- branch plot: `{rel_repo(best_summary['branch_plot'])}`",
                f"- checkpoint: `{rel_repo(best_summary['best_checkpoint'])}`",
                "",
                f"![history {best_mode}](../{rel_repo(best_summary['history_plot'])})",
                "",
                f"![real parity {best_mode}](../{rel_repo(Path(best_summary['best_checkpoint']).parent / 'eval_real' / 'parity_stress.png')})",
                "",
                f"![real error hist {best_mode}](../{rel_repo(Path(best_summary['best_checkpoint']).parent / 'eval_real' / 'stress_error_hist.png')})",
                "",
                f"![real branch confusion {best_mode}](../{rel_repo(Path(best_summary['best_checkpoint']).parent / 'eval_real' / 'branch_confusion.png')})",
                "",
            ]
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    report_md = ROOT / args.report_md
    device = choose_device(args.device)

    geometry_pool = build_local_b_pool(
        args.mesh_path,
        material_id=args.material_id,
        right_inverse_damping=args.right_inverse_damping,
    )

    fit_results: dict[str, BCopulaFit] = {}
    synthetic_test_paths: dict[str, Path] = {}
    for mode in args.copula_modes:
        fit, synthetic_test_path = _fit_b_copula_mode(
            real_primary=ROOT / args.real_primary,
            geometry_pool=geometry_pool,
            mode=mode,
            pca_variance=args.pca_variance,
            seed=args.seed + {"raw": 0, "pca": 17, "branch_raw": 31, "branch_pca": 47, "branch_raw_blend": 59}[mode],
            principal_cap_quantile=args.principal_cap_quantile,
            stress_cap_quantile=args.stress_cap_quantile,
            cap_slack=args.cap_slack,
            min_volume_ratio=args.min_volume_ratio,
            synthetic_test_size=args.synthetic_test_size,
            output_root=output_root,
        )
        fit_results[mode] = fit
        synthetic_test_paths[mode] = synthetic_test_path

    run_summaries: dict[str, dict[str, Any]] = {}
    if not args.screen_only:
        run_spec = RunSpec(
            name=f"cover_raw_branch_w{args.width}_d{args.depth}",
            width=args.width,
            depth=args.depth,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        for mode in args.copula_modes:
            run_dir = output_root / f"{run_spec.name}_{mode}"
            run_summaries[mode] = _train_refresh_run(
                fit=fit_results[mode],
                synthetic_test_path=synthetic_test_paths[mode],
                run_dir=run_dir,
                spec=RunSpec(
                    name=f"{run_spec.name}_{mode}",
                    width=run_spec.width,
                    depth=run_spec.depth,
                    weight_decay=run_spec.weight_decay,
                    seed=run_spec.seed + (0 if mode == "raw" else 17),
                ),
                device=device,
                eval_batch_size=args.eval_batch_size,
                branch_loss_weight=args.branch_loss_weight,
                grad_clip=args.grad_clip,
                stress_weight_alpha=args.stress_weight_alpha,
                stress_weight_scale=args.stress_weight_scale,
                train_size=args.train_size,
                cycles=args.cycles,
                batch_sizes=list(args.batch_sizes),
                base_lr=args.base_lr,
                cycle_lr_decay=args.cycle_lr_decay,
                min_lr=args.min_lr,
                plateau_patience=args.plateau_patience,
                stage_patience=args.stage_patience,
                improvement_rel_tol=args.improvement_rel_tol,
                improvement_abs_tol=args.improvement_abs_tol,
            )

    _write_report(
        report_md,
        output_root=output_root,
        mesh_path=Path(args.mesh_path),
        geometry_pool=geometry_pool,
        fit_results=fit_results,
        synthetic_test_paths=synthetic_test_paths,
        run_summaries=run_summaries,
    )


if __name__ == "__main__":
    main()
