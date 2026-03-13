#!/usr/bin/env python
"""Fit cover-layer strain samplers to the real E distribution and retrain with fresh synthetic data each epoch."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import REDUCED_MATERIAL_COLUMNS, SPLIT_NAMES, SPLIT_TO_ID
from mc_surrogate.models import (
    Standardizer,
    build_model,
    build_raw_features,
    compute_trial_stress,
    spectral_decomposition_from_strain,
)
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.sampling import _principal_to_global_engineering_strain, random_rotation_matrices
from mc_surrogate.training import (
    _build_tensor_dataset,
    _epoch_loop,
    _load_split_for_training,
    choose_device,
    evaluate_checkpoint_on_dataset,
    set_seed,
)
from mc_surrogate.viz import branch_confusion_plot, error_histogram, parity_plot


SAMPLER_NAMES = ("local_noise", "local_interp", "tail_local")


@dataclass(frozen=True)
class RunSpec:
    name: str
    width: int
    depth: int
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class SamplerFit:
    real_primary: str
    train_strain_eng: np.ndarray
    train_principal: np.ndarray
    train_material_reduced: np.ndarray
    train_branch_id: np.ndarray
    train_stress: np.ndarray
    train_split_arrays: dict[str, np.ndarray]
    branch_to_indices: tuple[np.ndarray, ...]
    branch_probs: np.ndarray
    branch_cholesky: np.ndarray
    branch_scale: np.ndarray
    branch_small_scale: np.ndarray
    branch_eng_floor: np.ndarray
    branch_stress_cap: np.ndarray
    branch_principal_cap: np.ndarray
    max_abs_principal: np.ndarray
    train_stress_mag: np.ndarray
    empirical_weights: np.ndarray
    tail_weights: np.ndarray
    principal_cap: float
    global_stress_cap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-primary",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5",
    )
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_fitted_refresh_20260312")
    parser.add_argument("--report-md", default="docs/cover_layer_fitted_refresh.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=9101)
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
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES), choices=list(SAMPLER_NAMES))
    parser.add_argument("--train-tail-alpha", type=float, default=2.0)
    parser.add_argument("--stress-weight-alpha", type=float, default=0.0)
    parser.add_argument("--stress-weight-scale", type=float, default=250.0)
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
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
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


def _safe_cholesky(cov: np.ndarray, scale: float) -> np.ndarray:
    eye = np.eye(cov.shape[0], dtype=np.float64)
    jitter = max(scale * scale * 1.0e-6, 1.0e-12)
    return np.linalg.cholesky(cov + jitter * eye)


def _load_fit(real_primary: Path, tail_alpha: float) -> tuple[SamplerFit, dict[str, Any], dict[str, np.ndarray]]:
    arrays = _load_arrays(real_primary)
    train_mask = arrays["split_id"] == SPLIT_TO_ID["train"]
    test_mask = arrays["split_id"] == SPLIT_TO_ID["test"]
    train_strain = arrays["strain_eng"][train_mask].astype(np.float32)
    train_material = arrays["material_reduced"][train_mask].astype(np.float32)
    train_branch = arrays["branch_id"][train_mask].astype(np.int64)
    train_stress = arrays["stress"][train_mask].astype(np.float32)
    train_stress_mag = np.linalg.norm(train_stress, axis=1)
    train_principal, _ = spectral_decomposition_from_strain(train_strain)
    max_abs = np.max(np.abs(train_principal), axis=1)
    branch_counts = np.bincount(train_branch, minlength=len(BRANCH_NAMES)).astype(np.float64)
    branch_probs = branch_counts / branch_counts.sum()
    branch_to_indices = tuple(np.flatnonzero(train_branch == i) for i in range(len(BRANCH_NAMES)))
    branch_chol = np.zeros((len(BRANCH_NAMES), 3, 3), dtype=np.float64)
    branch_scale = np.zeros(len(BRANCH_NAMES), dtype=np.float64)
    branch_small_scale = np.zeros(len(BRANCH_NAMES), dtype=np.float64)
    branch_eng_floor = np.zeros((len(BRANCH_NAMES), train_strain.shape[1]), dtype=np.float64)
    branch_stress_cap = np.zeros(len(BRANCH_NAMES), dtype=np.float64)
    branch_principal_cap = np.zeros(len(BRANCH_NAMES), dtype=np.float64)
    branch_info: dict[str, Any] = {}
    for bid, name in enumerate(BRANCH_NAMES):
        idx = branch_to_indices[bid]
        p = train_principal[idx].astype(np.float64)
        s_mag = train_stress_mag[idx]
        e_abs = np.abs(train_strain[idx].astype(np.float64))
        cov = np.cov(p.T) if p.shape[0] > 1 else np.eye(3, dtype=np.float64) * 1.0e-6
        scale = float(np.quantile(np.max(np.abs(p), axis=1), 0.95))
        small = float(np.quantile(np.max(np.abs(p), axis=1), 0.50))
        branch_scale[bid] = max(scale, 1.0e-4)
        branch_small_scale[bid] = max(small, 1.0e-5)
        branch_chol[bid] = _safe_cholesky(cov, branch_scale[bid])
        branch_eng_floor[bid] = np.maximum(np.quantile(e_abs, 0.50, axis=0), 1.0e-5)
        branch_stress_cap[bid] = max(float(np.quantile(s_mag, 0.995) * 1.05), 1.0)
        branch_principal_cap[bid] = max(float(np.quantile(np.max(np.abs(p), axis=1), 0.995) * 1.05), 1.0e-4)
        branch_info[name] = {
            "n_samples": int(idx.size),
            "max_abs_q50": float(np.quantile(np.max(np.abs(p), axis=1), 0.50)),
            "max_abs_q95": float(np.quantile(np.max(np.abs(p), axis=1), 0.95)),
            "max_abs_q995": float(np.quantile(np.max(np.abs(p), axis=1), 0.995)),
            "stress_mag_q95": float(np.quantile(s_mag, 0.95)),
            "stress_mag_q995": float(np.quantile(s_mag, 0.995)),
        }
    principal_cap = float(np.quantile(max_abs, 0.995) * 1.05)
    global_stress_cap = float(np.quantile(train_stress_mag, 0.995) * 1.05)
    empirical_weights = np.full(train_strain.shape[0], 1.0 / train_strain.shape[0], dtype=np.float64)
    tail_boost = 1.0 + tail_alpha * np.clip(train_stress_mag / max(np.quantile(train_stress_mag, 0.95), 1.0e-8), 0.0, 4.0)
    hard_branch_boost = np.ones_like(tail_boost)
    hard_branch_boost[np.isin(train_branch, [1, 2, 3, 4])] += 0.5 * tail_alpha
    tail_weights = tail_boost * hard_branch_boost
    tail_weights /= tail_weights.sum()
    fit = SamplerFit(
        real_primary=str(real_primary),
        train_strain_eng=train_strain,
        train_principal=train_principal.astype(np.float32),
        train_material_reduced=train_material,
        train_branch_id=train_branch,
        train_stress=train_stress,
        train_split_arrays=_load_split_for_training(str(real_primary), "train", "raw_branch"),
        branch_to_indices=branch_to_indices,
        branch_probs=branch_probs,
        branch_cholesky=branch_chol,
        branch_scale=branch_scale,
        branch_small_scale=branch_small_scale,
        branch_eng_floor=branch_eng_floor,
        branch_stress_cap=branch_stress_cap,
        branch_principal_cap=branch_principal_cap,
        max_abs_principal=max_abs,
        train_stress_mag=train_stress_mag,
        empirical_weights=empirical_weights,
        tail_weights=tail_weights,
        principal_cap=principal_cap,
        global_stress_cap=global_stress_cap,
    )
    info = {
        "real_primary": str(real_primary),
        "train_size": int(train_strain.shape[0]),
        "principal_cap": principal_cap,
        "global_stress_cap": global_stress_cap,
        "branch_probs": {name: float(branch_probs[i]) for i, name in enumerate(BRANCH_NAMES)},
        "branch_info": branch_info,
        "tail_alpha": tail_alpha,
    }
    real_test_arrays = {
        "strain_eng": arrays["strain_eng"][test_mask].astype(np.float32),
        "stress": arrays["stress"][test_mask].astype(np.float32),
        "branch_id": arrays["branch_id"][test_mask].astype(np.int64),
    }
    return fit, info, real_test_arrays


def _sample_base_indices(fit: SamplerFit, n: int, rng: np.random.Generator, *, tail_weighted: bool) -> np.ndarray:
    if tail_weighted:
        return rng.choice(fit.train_principal.shape[0], size=n, replace=True, p=fit.tail_weights)
    return rng.integers(0, fit.train_principal.shape[0], size=n)


def _desired_branch_counts(branch_probs: np.ndarray, n_samples: int) -> np.ndarray:
    counts = np.zeros(branch_probs.shape[0], dtype=np.int64)
    assigned = 0
    for bid in range(branch_probs.shape[0] - 1):
        counts[bid] = int(round(float(branch_probs[bid]) * n_samples))
        assigned += counts[bid]
    counts[-1] = n_samples - assigned
    return counts


def _sample_same_branch_companions(fit: SamplerFit, base_branch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(base_branch.shape[0], dtype=np.int64)
    for bid, pool in enumerate(fit.branch_to_indices):
        mask = base_branch == bid
        if not np.any(mask):
            continue
        out[mask] = pool[rng.integers(0, pool.shape[0], size=int(mask.sum()))]
    return out


def _candidate_noise_scale(
    fit: SamplerFit,
    base_strain: np.ndarray,
    base_branch: np.ndarray,
    *,
    sampler_name: str,
) -> np.ndarray:
    scale = np.empty_like(base_strain, dtype=np.float32)
    for bid in range(len(BRANCH_NAMES)):
        mask = base_branch == bid
        if not np.any(mask):
            continue
        if sampler_name == "local_noise":
            rel = 0.18 if bid == 0 else 0.035
            floor = 0.25 if bid == 0 else 0.10
        elif sampler_name == "local_interp":
            rel = 0.10 if bid == 0 else 0.020
            floor = 0.18 if bid == 0 else 0.06
        elif sampler_name == "tail_local":
            rel = 0.14 if bid == 0 else 0.030
            floor = 0.22 if bid == 0 else 0.08
        else:
            raise ValueError(f"Unsupported sampler {sampler_name!r}.")
        scale[mask] = np.maximum(
            np.abs(base_strain[mask]) * rel,
            fit.branch_eng_floor[bid][None, :] * floor,
        )
    return scale


def _accept_candidate_response(
    fit: SamplerFit,
    *,
    response,
    base_branch: np.ndarray,
) -> np.ndarray:
    stress_mag = np.linalg.norm(response.stress, axis=1)
    max_abs_principal = np.max(np.abs(response.strain_principal), axis=1)
    actual_branch = response.branch_id.astype(np.int64)
    accept = actual_branch == base_branch
    accept &= stress_mag <= fit.global_stress_cap
    accept &= stress_mag <= fit.branch_stress_cap[actual_branch]
    accept &= max_abs_principal <= fit.principal_cap
    accept &= max_abs_principal <= fit.branch_principal_cap[actual_branch]
    return accept


def _sample_branch_indices(
    fit: SamplerFit,
    *,
    branch_id: int,
    n: int,
    rng: np.random.Generator,
    tail_weighted: bool,
) -> np.ndarray:
    pool = fit.branch_to_indices[branch_id]
    if tail_weighted:
        w = fit.tail_weights[pool]
        w = w / w.sum()
        return pool[rng.choice(pool.shape[0], size=n, replace=True, p=w)]
    return pool[rng.integers(0, pool.shape[0], size=n)]


def generate_from_fitted_sampler(
    *,
    fit: SamplerFit,
    sampler_name: str,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks: list[dict[str, np.ndarray]] = []
    desired_counts = _desired_branch_counts(fit.branch_probs, n_samples)
    for branch_id, want in enumerate(desired_counts):
        remaining = int(want)
        attempts = 0
        while remaining > 0:
            attempts += 1
            if attempts > 4000:
                raise RuntimeError(f"Failed to generate enough accepted samples for sampler {sampler_name!r} branch {branch_id}.")
            batch = max(2048, remaining * 4)
            tail_weighted = sampler_name == "tail_local"
            base_idx = _sample_branch_indices(fit, branch_id=branch_id, n=batch, rng=rng, tail_weighted=tail_weighted)
            base_strain = fit.train_strain_eng[base_idx].astype(np.float32)
            base_branch = fit.train_branch_id[base_idx].astype(np.int64)
            material = fit.train_material_reduced[base_idx].astype(np.float32)

            if sampler_name == "local_noise":
                noise_scale = _candidate_noise_scale(fit, base_strain, base_branch, sampler_name=sampler_name)
                strain_eng = base_strain + rng.normal(size=base_strain.shape).astype(np.float32) * noise_scale
            elif sampler_name == "local_interp":
                mate_idx = _sample_same_branch_companions(fit, base_branch, rng)
                mate_strain = fit.train_strain_eng[mate_idx].astype(np.float32)
                blend = np.clip(
                    rng.beta(a=0.65, b=0.65, size=batch) + rng.normal(0.0, 0.04, size=batch),
                    -0.08,
                    1.08,
                ).astype(np.float32)
                base_mix = blend[:, None] * base_strain + (1.0 - blend[:, None]) * mate_strain
                noise_scale = _candidate_noise_scale(fit, base_mix, base_branch, sampler_name=sampler_name)
                strain_eng = base_mix + rng.normal(size=base_mix.shape).astype(np.float32) * noise_scale
            elif sampler_name == "tail_local":
                mate_idx = _sample_same_branch_companions(fit, base_branch, rng)
                mate_strain = fit.train_strain_eng[mate_idx].astype(np.float32)
                interp_mask = rng.random(batch) < 0.35
                strain_eng = base_strain.copy()
                if np.any(interp_mask):
                    blend = np.clip(
                        rng.beta(a=0.55, b=0.55, size=int(interp_mask.sum())) + rng.normal(0.0, 0.06, size=int(interp_mask.sum())),
                        -0.10,
                        1.10,
                    ).astype(np.float32)
                    strain_eng[interp_mask] = (
                        blend[:, None] * base_strain[interp_mask]
                        + (1.0 - blend[:, None]) * mate_strain[interp_mask]
                    )
                noise_scale = _candidate_noise_scale(fit, strain_eng, base_branch, sampler_name=sampler_name)
                strain_eng = strain_eng + rng.normal(size=strain_eng.shape).astype(np.float32) * noise_scale
            else:
                raise ValueError(f"Unsupported sampler {sampler_name!r}.")

            response = constitutive_update_3d(
                strain_eng,
                c_bar=material[:, 0],
                sin_phi=material[:, 1],
                shear=material[:, 2],
                bulk=material[:, 3],
                lame=material[:, 4],
            )
            accept = _accept_candidate_response(fit, response=response, base_branch=base_branch)
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
            remaining -= take.size

    arrays: dict[str, np.ndarray] = {}
    for key in chunks[0].keys():
        arrays[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    perm = rng.permutation(arrays["strain_eng"].shape[0])
    arrays = {key: value[perm] for key, value in arrays.items()}
    return arrays


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
    fit: SamplerFit,
    sampler_name: str,
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

    model = build_model(
        "raw_branch",
        input_dim=fit.train_split_arrays["features"].shape[1],
        width=spec.width,
        depth=spec.depth,
        dropout=0.0,
    ).to(device)
    metadata = {
        "config": {
            "dataset": "epoch_refreshed_distribution_fitted_synthetic_train",
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
            "sampler_name": sampler_name,
            "train_size": train_size,
            "principal_cap": fit.principal_cap,
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
                    f"[stage-start] sampler={sampler_name} cycle={cycle_idx}/{cycles} batch={batch_size} lr={stage_lr:.3e} "
                    f"global_epoch={global_epoch} runtime={_format_runtime(time.perf_counter() - total_start)}"
                ),
            )

            while True:
                global_epoch += 1
                train_seed = spec.seed + 100000 + global_epoch * 7919
                train_arrays = generate_from_fitted_sampler(
                    fit=fit,
                    sampler_name=sampler_name,
                    n_samples=train_size,
                    seed=train_seed,
                )
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
                        f"[adam] sampler={sampler_name} epoch={global_epoch} cycle={cycle_idx}/{cycles} batch={batch_size} lr={current_lr:.3e} "
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
                                f"[lr-drop] sampler={sampler_name} epoch={global_epoch} cycle={cycle_idx}/{cycles} batch={batch_size} "
                                f"old_lr={current_lr:.3e} new_lr={new_lr:.3e} "
                                f"runtime={_format_runtime(time.perf_counter() - total_start)}"
                            ),
                        )

                if stage_bad_epochs >= stage_patience:
                    _log_message(
                        log_path,
                        (
                            f"[stage-stop] sampler={sampler_name} cycle={cycle_idx}/{cycles} batch={batch_size} "
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
        "sampler_name": sampler_name,
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
        "real_test_metrics": real_eval["metrics"],
        "synthetic_test_metrics": synthetic_eval["metrics"],
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _write_report(
    report_md: Path,
    *,
    fit_info: dict[str, Any],
    real_test_summary: dict[str, Any],
    sampler_distribution: dict[str, dict[str, Any]],
    run_summaries: dict[str, dict[str, Any]],
) -> None:
    def rel_repo(path: str | Path) -> str:
        return Path(path).resolve().relative_to(ROOT).as_posix()

    prior = json.loads(
        (
            ROOT
            / "experiment_runs"
            / "real_sim"
            / "cover_layer_cyclic_20260312"
            / "cover_raw_branch_w384_d6"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )

    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cover Layer Fitted-Distribution Epoch Refresh",
        "",
        "This study replaces the old branch-targeted synthetic sampler with strain samplers fitted directly to the real cover-layer `E` distribution.",
        "Each run keeps validation and testing fixed on the real cover-layer dataset, while regenerating new exact-labeled synthetic training data every epoch.",
        "",
        "## Real-Domain Fit",
        "",
        f"- real primary dataset: `{fit_info['real_primary']}`",
        f"- train pool size: `{fit_info['train_size']}`",
        f"- fitted principal cap: `{fit_info['principal_cap']:.6f}`",
        f"- empirical train branch probabilities: `{fit_info['branch_probs']}`",
        "",
        "| Real test | N | max |eps_principal| q95 | q995 | stress | q95 | q995 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| real test | {real_test_summary['n_samples']} | {real_test_summary['max_abs_principal_q95']:.4f} | {real_test_summary['max_abs_principal_q995']:.4f} | {real_test_summary['stress_mag_q95']:.4f} | {real_test_summary['stress_mag_q995']:.4f} |",
        "",
        "## Synthetic Holdout Distribution Fit",
        "",
        "| Sampler | max |eps_principal| q95 | q995 | stress | q95 | q995 | branch TV | fit score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sampler_name, info in sampler_distribution.items():
        summary = info["summary"]
        score = info["fit_score"]
        lines.append(
            f"| `{sampler_name}` | {summary['max_abs_principal_q95']:.4f} | {summary['max_abs_principal_q995']:.4f} | "
            f"{summary['stress_mag_q95']:.4f} | {summary['stress_mag_q995']:.4f} | {score['branch_tv']:.4f} | {score['total']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Training Results",
            "",
            "| Sampler | Runtime | Best epoch | Real test MAE | RMSE | Max abs | Branch acc | Synthetic test MAE | RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    best_sampler = min(run_summaries, key=lambda k: run_summaries[k]["real_test_metrics"]["stress_mae"])
    for sampler_name, summary in run_summaries.items():
        real_metrics = summary["real_test_metrics"]
        synth_metrics = summary["synthetic_test_metrics"]
        lines.append(
            f"| `{sampler_name}` | {summary['elapsed_hms']} | {summary['best_epoch']} | {real_metrics['stress_mae']:.4f} | {real_metrics['stress_rmse']:.4f} | "
            f"{real_metrics['stress_max_abs']:.4f} | {real_metrics.get('branch_accuracy', float('nan')):.4f} | {synth_metrics['stress_mae']:.4f} | {synth_metrics['stress_rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Best real-test sampler in this run: `{best_sampler}`.",
            "",
            "Reference fixed-dataset exact-domain baseline (`cover_raw_branch_w384_d6`):",
            "",
            f"- real test MAE / RMSE / max abs: `{prior['primary_metrics']['stress_mae']:.4f}` / "
            f"`{prior['primary_metrics']['stress_rmse']:.4f}` / `{prior['primary_metrics']['stress_max_abs']:.4f}`",
            "",
        ]
    )
    for sampler_name in run_summaries:
        summary = run_summaries[sampler_name]
        info = sampler_distribution[sampler_name]
        run_dir = Path(summary["history_plot"]).parent
        lines.extend(
            [
                f"## {sampler_name}",
                "",
                f"- distribution fit score: `{info['fit_score']['total']:.4f}`",
                f"- real test MAE / RMSE / max abs: `{summary['real_test_metrics']['stress_mae']:.4f}` / "
                f"`{summary['real_test_metrics']['stress_rmse']:.4f}` / `{summary['real_test_metrics']['stress_max_abs']:.4f}`",
                f"- synthetic test MAE / RMSE / max abs: `{summary['synthetic_test_metrics']['stress_mae']:.4f}` / "
                f"`{summary['synthetic_test_metrics']['stress_rmse']:.4f}` / `{summary['synthetic_test_metrics']['stress_max_abs']:.4f}`",
                "",
                f"![{sampler_name} distribution fit](../{rel_repo(info['distribution_plot'])})",
                "",
                f"![{sampler_name} history](../{rel_repo(summary['history_plot'])})",
                "",
                f"![{sampler_name} branch accuracy](../{rel_repo(summary['branch_plot'])})",
                "",
                f"![{sampler_name} real parity](../{rel_repo(run_dir / 'eval_real' / 'parity_stress.png')})",
                "",
                f"![{sampler_name} real error histogram](../{rel_repo(run_dir / 'eval_real' / 'stress_error_hist.png')})",
                "",
                f"![{sampler_name} real branch confusion](../{rel_repo(run_dir / 'eval_real' / 'branch_confusion.png')})",
                "",
                f"![{sampler_name} synthetic parity](../{rel_repo(run_dir / 'eval_synth' / 'parity_stress.png')})",
                "",
                f"![{sampler_name} synthetic error histogram](../{rel_repo(run_dir / 'eval_synth' / 'stress_error_hist.png')})",
                "",
                f"![{sampler_name} synthetic branch confusion](../{rel_repo(run_dir / 'eval_synth' / 'branch_confusion.png')})",
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

    fit, fit_info, real_test_arrays = _load_fit(real_primary, tail_alpha=args.train_tail_alpha)
    real_test_summary = _distribution_summary(real_test_arrays["strain_eng"], real_test_arrays["stress"], real_test_arrays["branch_id"])
    (output_root / "fit_summary.json").write_text(json.dumps(_json_safe(fit_info), indent=2), encoding="utf-8")

    sampler_distribution: dict[str, dict[str, Any]] = {}
    run_summaries: dict[str, dict[str, Any]] = {}
    for sampler_idx, sampler_name in enumerate(args.samplers, start=1):
        synthetic_arrays = generate_from_fitted_sampler(
            fit=fit,
            sampler_name=sampler_name,
            n_samples=args.synthetic_test_size,
            seed=args.seed + 500000 + sampler_idx * 1000,
        )
        synthetic_test_path = _write_dataset(
            output_root / f"synthetic_test_{sampler_name}.h5",
            {
                **synthetic_arrays,
                "split_id": np.full(synthetic_arrays["strain_eng"].shape[0], SPLIT_TO_ID["test"], dtype=np.int8),
            },
            {
                "generator": "distribution_fitted_cover_layer_holdout",
                "sampler_name": sampler_name,
                "seed": int(args.seed + 500000 + sampler_idx * 1000),
                "principal_cap": fit.principal_cap,
            },
        )
        synth_summary = _distribution_summary(
            synthetic_arrays["strain_eng"],
            synthetic_arrays["stress"],
            synthetic_arrays["branch_id"],
        )
        fit_score = _distribution_fit_score(real_test_summary, synth_summary)
        distribution_plot = _plot_distribution_fit(real_test_arrays, synthetic_arrays, output_root / f"distribution_fit_{sampler_name}.png")
        sampler_distribution[sampler_name] = {
            "synthetic_test_path": str(synthetic_test_path),
            "summary": synth_summary,
            "fit_score": fit_score,
            "distribution_plot": str(distribution_plot),
        }

        spec = RunSpec(
            name=f"cover_raw_branch_w{args.width}_d{args.depth}_{sampler_name}",
            width=args.width,
            depth=args.depth,
            weight_decay=args.weight_decay,
            seed=args.seed + sampler_idx * 101,
        )
        run_summaries[sampler_name] = _train_refresh_run(
            fit=fit,
            sampler_name=sampler_name,
            synthetic_test_path=synthetic_test_path,
            run_dir=output_root / spec.name,
            spec=spec,
            device=device,
            eval_batch_size=args.eval_batch_size,
            branch_loss_weight=args.branch_loss_weight,
            grad_clip=args.grad_clip,
            stress_weight_alpha=args.stress_weight_alpha,
            stress_weight_scale=args.stress_weight_scale,
            train_size=args.train_size,
            cycles=args.cycles,
            batch_sizes=args.batch_sizes,
            base_lr=args.base_lr,
            cycle_lr_decay=args.cycle_lr_decay,
            min_lr=args.min_lr,
            plateau_patience=args.plateau_patience,
            stage_patience=args.stage_patience,
            improvement_rel_tol=args.improvement_rel_tol,
            improvement_abs_tol=args.improvement_abs_tol,
        )

    summary_path = output_root / "run_summaries.json"
    summary_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "fit_info": fit_info,
                    "real_test_summary": real_test_summary,
                    "sampler_distribution": sampler_distribution,
                    "run_summaries": run_summaries,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        Path(args.report_md),
        fit_info=fit_info,
        real_test_summary=real_test_summary,
        sampler_distribution=sampler_distribution,
        run_summaries=run_summaries,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "report_md": str(Path(args.report_md)),
                "device": str(device),
                "best_sampler": min(run_summaries, key=lambda k: run_summaries[k]["real_test_metrics"]["stress_mae"]),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
