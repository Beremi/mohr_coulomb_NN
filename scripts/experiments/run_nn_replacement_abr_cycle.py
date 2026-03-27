#!/usr/bin/env python
"""Run the March 24 ACN Stage 0/1 packet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    decode_abr_to_principal,
    encode_principal_to_abr,
    exact_trial_principal_stress_3d,
    infer_branch_from_abr,
    principal_relative_error_3d,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.models import (
    build_trial_abr_features_f1,
    build_trial_abr_features_f2,
    compute_trial_abr_feature_stats,
    compute_trial_stress,
)
from mc_surrogate.training import TrainingConfig, load_checkpoint, predict_with_checkpoint, train_model
from run_hybrid_campaign import _candidate_checkpoint_paths, _sample_indices
from run_return_mapping_accel_scout import build_synthetic_hard_panel


PANEL_MASK_KEYS = (
    "broad_val_mask",
    "broad_test_mask",
    "hard_mask",
    "hard_val_mask",
    "hard_test_mask",
    "ds_valid_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redesign-root", default="experiment_runs/real_sim/hybrid_gate_redesign_20260324")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/nn_replacement_abr_20260324")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--force-rerun", action="store_true")
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


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_phase_report(path: Path, title: str, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")
    return path


def _load_h5(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _write_h5(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
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


def _slice_arrays(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: value[mask] if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0] else value
        for key, value in arrays.items()
    }


def _quantile_or_zero(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _branch_counts(branch_id: np.ndarray) -> dict[str, int]:
    counts = np.bincount(branch_id.astype(np.int64), minlength=len(BRANCH_NAMES))
    return {name: int(counts[idx]) for idx, name in enumerate(BRANCH_NAMES)}


def _rows_to_flat(rows: list[dict[str, Any]], extra_keys: list[str]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        payload = {key: row.get(key) for key in extra_keys}
        payload.update(row["metrics"])
        flat.append(payload)
    return flat


def _percentile_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _plot_histograms(
    output_path: Path,
    values_by_name: dict[str, np.ndarray],
    *,
    bins: int = 80,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(values_by_name)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.0 * n), squeeze=False)
    for ax, (name, values) in zip(axes[:, 0], values_by_name.items(), strict=False):
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        ax.hist(arr, bins=bins, alpha=0.85, color="#244c5a")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _pointwise_prediction_stats(
    arrays: dict[str, np.ndarray],
    *,
    stress_pred: np.ndarray,
    stress_principal_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    stress_true = arrays["stress"]
    principal_true = arrays["stress_principal"]
    material = arrays["material_reduced"]
    stress_abs = np.abs(stress_pred - stress_true)
    principal_abs = np.abs(stress_principal_pred - principal_true)
    return {
        "stress_component_abs": stress_abs.astype(np.float32),
        "stress_sample_mae": np.mean(stress_abs, axis=1).astype(np.float32),
        "stress_sample_rmse": np.sqrt(np.mean((stress_pred - stress_true) ** 2, axis=1)).astype(np.float32),
        "principal_max_abs": np.max(principal_abs, axis=1).astype(np.float32),
        "principal_rel_error": principal_relative_error_3d(
            stress_principal_pred,
            principal_true,
            c_bar=material[:, 0],
        ).astype(np.float32),
        "yield_violation_rel": yield_violation_rel_principal_3d(
            stress_principal_pred,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
        ).astype(np.float32),
    }


def _aggregate_policy_metrics(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    *,
    learned_mask: np.ndarray,
    elastic_mask: np.ndarray,
) -> dict[str, Any]:
    plastic = panel["plastic_mask"].astype(bool)
    hard = panel["hard_mask"].astype(bool)
    hard_plastic = hard & plastic
    yield_subset = stats["yield_violation_rel"][plastic]
    return {
        "n_rows": int(arrays["stress"].shape[0]),
        "broad_mae": float(np.mean(stats["stress_component_abs"])),
        "hard_mae": float(np.mean(stats["stress_component_abs"][hard])) if np.any(hard) else float("nan"),
        "broad_plastic_mae": float(np.mean(stats["stress_component_abs"][plastic])) if np.any(plastic) else float("nan"),
        "hard_plastic_mae": float(np.mean(stats["stress_component_abs"][hard_plastic])) if np.any(hard_plastic) else float("nan"),
        "hard_p95_principal": _quantile_or_zero(stats["principal_max_abs"][hard], 0.95),
        "hard_rel_p95_principal": _quantile_or_zero(stats["principal_rel_error"][hard], 0.95),
        "yield_violation_p95": _quantile_or_zero(yield_subset, 0.95),
        "plastic_coverage": float(np.mean(learned_mask[plastic])) if np.any(plastic) else 0.0,
        "accepted_plastic_rows": int(np.sum(learned_mask & plastic)),
        "route_counts": {
            "elastic": int(np.sum(elastic_mask)),
            "fallback": 0,
            "learned": int(np.sum(learned_mask)),
        },
        "accepted_true_branch_counts": _branch_counts(arrays["branch_id"][learned_mask & plastic]) if np.any(learned_mask & plastic) else {name: 0 for name in BRANCH_NAMES},
    }


def _parameter_count(checkpoint_path: Path, device: str) -> int:
    model, _metadata = load_checkpoint(checkpoint_path, device=device)
    return int(sum(param.numel() for param in model.parameters()))


def _maybe_train(config: TrainingConfig, *, force_rerun: bool) -> dict[str, Any]:
    summary_path = Path(config.run_dir) / "summary.json"
    best_path = Path(config.run_dir) / "best.pt"
    if summary_path.exists() and best_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return train_model(config)


def _train_with_batch_fallback(config: TrainingConfig, *, force_rerun: bool) -> dict[str, Any]:
    try:
        return _maybe_train(config, force_rerun=force_rerun)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or config.batch_size <= 2048:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fallback = TrainingConfig(**{**config.__dict__, "batch_size": 2048})
        return _maybe_train(fallback, force_rerun=force_rerun)


def build_stage0_dataset(
    *,
    real_dataset_path: Path,
    panel_path: Path,
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    dataset_path = output_dir / "derived_abr_dataset.h5"
    summary_path = output_dir / "stage0_summary.json"
    if dataset_path.exists() and summary_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    real_arrays, real_attrs = _load_h5(real_dataset_path)
    panel_arrays, panel_attrs = _load_h5(panel_path)

    train_mask = real_arrays["split_id"] == 0
    trial_stress = compute_trial_stress(real_arrays["strain_eng"], real_arrays["material_reduced"]).astype(np.float32)
    trial_principal = exact_trial_principal_stress_3d(
        real_arrays["strain_eng"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
        shear=real_arrays["material_reduced"][:, 2],
        bulk=real_arrays["material_reduced"][:, 3],
        lame=real_arrays["material_reduced"][:, 4],
    ).astype(np.float32)
    encoded = encode_principal_to_abr(
        real_arrays["stress_principal"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
    )
    decoded = decode_abr_to_principal(
        encoded["abr_raw"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
    ).astype(np.float32)
    inferred_branch = infer_branch_from_abr(
        encoded["abr_raw"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
    ).astype(np.int8)
    feature_stats = compute_trial_abr_feature_stats(trial_principal[train_mask], real_arrays["material_reduced"][train_mask])
    feature_f1 = build_trial_abr_features_f1(trial_principal, real_arrays["material_reduced"], feature_stats)
    feature_f2 = build_trial_abr_features_f2(trial_principal, real_arrays["material_reduced"], feature_stats)

    reconstruction_abs = np.abs(decoded - real_arrays["stress_principal"].astype(np.float32))
    reconstruction_abs_max = np.max(reconstruction_abs, axis=1).astype(np.float32)
    elastic = real_arrays["branch_id"] == 0
    elastic_identity = np.abs(trial_stress - real_arrays["stress"].astype(np.float32))
    elastic_identity_abs_max = np.max(elastic_identity, axis=1).astype(np.float32)

    derived_arrays = {
        "strain_eng": real_arrays["strain_eng"].astype(np.float32),
        "stress": real_arrays["stress"].astype(np.float32),
        "stress_principal": real_arrays["stress_principal"].astype(np.float32),
        "material_reduced": real_arrays["material_reduced"].astype(np.float32),
        "eigvecs": real_arrays["eigvecs"].astype(np.float32),
        "branch_id": real_arrays["branch_id"].astype(np.int8),
        "inferred_branch_id": inferred_branch,
        "split_id": real_arrays["split_id"].astype(np.int8),
        "source_call_id": real_arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": real_arrays["source_row_in_call"].astype(np.int32),
        "trial_stress": trial_stress,
        "trial_principal": trial_principal,
        "abr_raw": encoded["abr_raw"].astype(np.float32),
        "abr_nonneg": encoded["abr_nonneg"].astype(np.float32),
        "feature_f1": feature_f1.astype(np.float32),
        "feature_f2": feature_f2.astype(np.float32),
        "reconstruction_abs_max": reconstruction_abs_max,
        "elastic_identity_abs_max": elastic_identity_abs_max,
        "plastic_mask": panel_arrays["plastic_mask"].astype(np.int8),
        **{key: panel_arrays[key].astype(np.int8) for key in PANEL_MASK_KEYS},
    }
    attrs = {
        "source_real_dataset": str(real_dataset_path),
        "source_panel_path": str(panel_path),
        "acn_feature_stats_json": feature_stats,
        "split_seed": real_attrs.get("split_seed", 20260324),
        "branch_names_json": real_attrs.get("branch_names_json", json.dumps(BRANCH_NAMES)),
        "panel_summary_json": panel_attrs.get("panel_summary_json", ""),
    }
    _write_h5(dataset_path, derived_arrays, attrs)

    sin_phi = real_arrays["material_reduced"][:, 1].astype(float)
    abr_raw = encoded["abr_raw"]
    trial_p = trial_principal.astype(float)
    p_tr = trial_p.mean(axis=1)
    a_tr = trial_p[:, 0] - trial_p[:, 1]
    b_tr = trial_p[:, 1] - trial_p[:, 2]
    f_tr = (1.0 + real_arrays["material_reduced"][:, 1]) * trial_p[:, 0]
    f_tr -= (1.0 - real_arrays["material_reduced"][:, 1]) * trial_p[:, 2]
    f_tr -= real_arrays["material_reduced"][:, 0]
    rho_tr = (a_tr - b_tr) / (a_tr + b_tr + 1.0e-12)

    split_summaries: dict[str, Any] = {}
    for split_id, split_name in ((0, "train"), (1, "val"), (2, "test")):
        mask = real_arrays["split_id"] == split_id
        split_summaries[split_name] = {
            "n_rows": int(np.sum(mask)),
            "a": _percentile_summary(abr_raw[mask, 0]),
            "b": _percentile_summary(abr_raw[mask, 1]),
            "r_raw": _percentile_summary(abr_raw[mask, 2]),
        }

    branch_summaries: dict[str, Any] = {}
    for branch_id, branch_name in enumerate(BRANCH_NAMES):
        mask = real_arrays["branch_id"] == branch_id
        branch_summaries[branch_name] = {
            "n_rows": int(np.sum(mask)),
            "a": _percentile_summary(abr_raw[mask, 0]),
            "b": _percentile_summary(abr_raw[mask, 1]),
            "r_raw": _percentile_summary(abr_raw[mask, 2]),
        }

    _plot_histograms(
        output_dir / "abr_by_split.png",
        {
            "train_a": abr_raw[real_arrays["split_id"] == 0, 0],
            "val_a": abr_raw[real_arrays["split_id"] == 1, 0],
            "test_a": abr_raw[real_arrays["split_id"] == 2, 0],
            "train_b": abr_raw[real_arrays["split_id"] == 0, 1],
            "val_b": abr_raw[real_arrays["split_id"] == 1, 1],
            "test_b": abr_raw[real_arrays["split_id"] == 2, 1],
            "train_r_raw": abr_raw[real_arrays["split_id"] == 0, 2],
            "val_r_raw": abr_raw[real_arrays["split_id"] == 1, 2],
            "test_r_raw": abr_raw[real_arrays["split_id"] == 2, 2],
        },
        title="ABR by split",
    )
    _plot_histograms(
        output_dir / "trial_feature_histograms.png",
        {
            "a_tr": a_tr,
            "b_tr": b_tr,
            "f_tr": f_tr,
            "rho_tr": rho_tr,
        },
        title="Trial feature ingredients",
    )
    _plot_histograms(
        output_dir / "abr_by_branch.png",
        {f"{name}_r_raw": abr_raw[real_arrays["branch_id"] == idx, 2] for idx, name in enumerate(BRANCH_NAMES)},
        title="Raw r by branch",
    )

    summary = {
        "dataset_path": str(dataset_path),
        "feature_stats": feature_stats,
        "sin_phi_stats": {
            "min": float(np.min(sin_phi)),
            "q01": float(np.quantile(sin_phi, 0.01)),
            "q05": float(np.quantile(sin_phi, 0.05)),
            "lt_1e_3": int(np.sum(sin_phi < 1.0e-3)),
            "lt_5e_3": int(np.sum(sin_phi < 5.0e-3)),
        },
        "split_summaries": split_summaries,
        "branch_summaries": branch_summaries,
        "trial_feature_summaries": {
            "a_tr": _percentile_summary(a_tr),
            "b_tr": _percentile_summary(b_tr),
            "f_tr": _percentile_summary(f_tr),
            "rho_tr": _percentile_summary(rho_tr),
        },
        "reconstruction": {
            "mean_abs": float(np.mean(reconstruction_abs)),
            "max_abs": float(np.max(reconstruction_abs)),
        },
        "elastic_identity": {
            "mean_abs": float(np.mean(elastic_identity[elastic])) if np.any(elastic) else 0.0,
            "max_abs": float(np.max(elastic_identity[elastic])) if np.any(elastic) else 0.0,
        },
        "r_raw": {
            "min": float(np.min(abr_raw[:, 2])),
            "negative_count": int(np.sum(abr_raw[:, 2] < 0.0)),
        },
        "inferred_branch_accuracy": float(np.mean(inferred_branch == real_arrays["branch_id"].astype(np.int8))),
    }
    _write_json(summary_path, summary)
    return summary


def build_stage1_dataset(
    *,
    stage0_dataset_path: Path,
    synthetic_dataset_path: Path,
    output_dir: Path,
    seed: int,
    force_rerun: bool,
) -> dict[str, Any]:
    dataset_path = output_dir / "stage1_train_mix.h5"
    summary_path = output_dir / "stage1_train_mix_summary.json"
    if dataset_path.exists() and summary_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    stage0_arrays, stage0_attrs = _load_h5(stage0_dataset_path)
    synth_arrays, _synth_attrs = _load_h5(synthetic_dataset_path)
    feature_stats = json.loads(stage0_attrs["acn_feature_stats_json"]) if isinstance(stage0_attrs["acn_feature_stats_json"], str) else stage0_attrs["acn_feature_stats_json"]

    rng = np.random.default_rng(seed)
    train_mask = stage0_arrays["split_id"] == 0
    broad_train = np.flatnonzero(train_mask)
    hard_train = np.flatnonzero(train_mask & stage0_arrays["hard_mask"].astype(bool))
    broad_idx = _sample_indices(rng, broad_train, 91751)
    hard_idx = _sample_indices(rng, hard_train, 26214)
    synth_idx = _sample_indices(rng, np.arange(synth_arrays["strain_eng"].shape[0], dtype=np.int64), 13107)

    synth_trial_stress = compute_trial_stress(synth_arrays["strain_eng"][synth_idx], synth_arrays["material_reduced"][synth_idx]).astype(np.float32)
    synth_trial_principal = exact_trial_principal_stress_3d(
        synth_arrays["strain_eng"][synth_idx],
        c_bar=synth_arrays["material_reduced"][synth_idx, 0],
        sin_phi=synth_arrays["material_reduced"][synth_idx, 1],
        shear=synth_arrays["material_reduced"][synth_idx, 2],
        bulk=synth_arrays["material_reduced"][synth_idx, 3],
        lame=synth_arrays["material_reduced"][synth_idx, 4],
    ).astype(np.float32)
    synth_abr = encode_principal_to_abr(
        synth_arrays["stress_principal"][synth_idx],
        c_bar=synth_arrays["material_reduced"][synth_idx, 0],
        sin_phi=synth_arrays["material_reduced"][synth_idx, 1],
    )
    synth_feature_f1 = build_trial_abr_features_f1(synth_trial_principal, synth_arrays["material_reduced"][synth_idx], feature_stats)
    synth_feature_f2 = build_trial_abr_features_f2(synth_trial_principal, synth_arrays["material_reduced"][synth_idx], feature_stats)

    val_test_mask = stage0_arrays["split_id"] != 0
    real_train_idx = np.concatenate([broad_idx, hard_idx], axis=0)
    split_train = np.full(real_train_idx.shape[0] + synth_idx.shape[0], 0, dtype=np.int8)

    train_arrays = {
        key: np.concatenate(
            [
                stage0_arrays[key][real_train_idx],
                (
                    synth_arrays[key][synth_idx]
                    if key in synth_arrays
                    else np.zeros_like(stage0_arrays[key][: synth_idx.shape[0]])
                ),
            ],
            axis=0,
        )
        for key in (
            "strain_eng",
            "stress",
            "stress_principal",
            "material_reduced",
            "eigvecs",
            "branch_id",
            "source_call_id",
            "source_row_in_call",
        )
    }
    train_arrays["trial_stress"] = np.concatenate([stage0_arrays["trial_stress"][real_train_idx], synth_trial_stress], axis=0)
    train_arrays["trial_principal"] = np.concatenate([stage0_arrays["trial_principal"][real_train_idx], synth_trial_principal], axis=0)
    train_arrays["abr_raw"] = np.concatenate([stage0_arrays["abr_raw"][real_train_idx], synth_abr["abr_raw"].astype(np.float32)], axis=0)
    train_arrays["abr_nonneg"] = np.concatenate([stage0_arrays["abr_nonneg"][real_train_idx], synth_abr["abr_nonneg"].astype(np.float32)], axis=0)
    train_arrays["feature_f1"] = np.concatenate([stage0_arrays["feature_f1"][real_train_idx], synth_feature_f1], axis=0)
    train_arrays["feature_f2"] = np.concatenate([stage0_arrays["feature_f2"][real_train_idx], synth_feature_f2], axis=0)
    for key in ("plastic_mask", *PANEL_MASK_KEYS):
        train_arrays[key] = np.concatenate(
            [
                stage0_arrays[key][real_train_idx],
                np.zeros(synth_idx.shape[0], dtype=np.int8),
            ],
            axis=0,
        )
    train_arrays["split_id"] = split_train

    final_arrays = {
        key: np.concatenate([train_arrays[key], stage0_arrays[key][val_test_mask]], axis=0)
        for key in train_arrays
    }
    coordinate_scales = {
        "scale_a": float(max(np.quantile(train_arrays["abr_nonneg"][:, 0], 0.95), 1.0)),
        "scale_b": float(max(np.quantile(train_arrays["abr_nonneg"][:, 1], 0.95), 1.0)),
        "scale_r": float(max(np.quantile(train_arrays["abr_nonneg"][:, 2], 0.95), 1.0)),
    }
    attrs = {
        "source_stage0_dataset": str(stage0_dataset_path),
        "source_synthetic_dataset": str(synthetic_dataset_path),
        "acn_feature_stats_json": feature_stats,
        "acn_coordinate_scales_json": coordinate_scales,
    }
    _write_h5(dataset_path, final_arrays, attrs)
    summary = {
        "dataset_path": str(dataset_path),
        "coordinate_scales": coordinate_scales,
        "feature_stats": feature_stats,
        "train_rows": int(train_arrays["strain_eng"].shape[0]),
        "train_real_rows": int(real_train_idx.shape[0]),
        "train_synthetic_rows": int(synth_idx.shape[0]),
    }
    _write_json(summary_path, summary)
    return summary


def _load_eval_split(dataset_path: Path, split_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays_all, _attrs = _load_h5(dataset_path)
    split_id = {"train": 0, "val": 1, "test": 2}[split_name]
    mask = arrays_all["split_id"] == split_id
    arrays = _slice_arrays(arrays_all, mask)
    panel = {
        "plastic_mask": arrays["plastic_mask"],
        "hard_mask": arrays["hard_mask"],
    }
    return arrays, panel


def evaluate_acn_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split_name: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    arrays, panel = _load_eval_split(dataset_path, split_name)
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=batch_size,
    )
    stats = _pointwise_prediction_stats(
        arrays,
        stress_pred=pred["stress"],
        stress_principal_pred=pred["stress_principal"],
    )
    learned_mask = np.ones(arrays["stress"].shape[0], dtype=bool)
    elastic_mask = arrays["branch_id"] == 0
    metrics = _aggregate_policy_metrics(arrays, panel, stats, learned_mask=learned_mask, elastic_mask=elastic_mask)
    metrics["split"] = split_name
    return {
        "metrics": metrics,
        "predictions": pred,
    }


def write_stage0_report(report_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "## Frozen Split",
        "",
        "- source grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`",
        "- split seed: `20260324`",
        "- split fractions: `70 / 15 / 15`",
        "",
        "## Conditioning",
        "",
        f"- sin_phi min / q01 / q05: `{summary['sin_phi_stats']['min']:.7f}` / `{summary['sin_phi_stats']['q01']:.7f}` / `{summary['sin_phi_stats']['q05']:.7f}`",
        f"- sin_phi rows below `1e-3` / `5e-3`: `{summary['sin_phi_stats']['lt_1e_3']}` / `{summary['sin_phi_stats']['lt_5e_3']}`",
        "",
        "## Exactness",
        "",
        f"- reconstruction mean / max abs: `{summary['reconstruction']['mean_abs']:.3e}` / `{summary['reconstruction']['max_abs']:.3e}`",
        f"- elastic identity mean / max abs: `{summary['elastic_identity']['mean_abs']:.3e}` / `{summary['elastic_identity']['max_abs']:.3e}`",
        f"- raw r minimum: `{summary['r_raw']['min']:.6e}` with `{summary['r_raw']['negative_count']}` slightly negative rows",
        f"- inferred branch agreement from raw ABR: `{summary['inferred_branch_accuracy']:.6f}`",
        "",
        "## Derived Dataset",
        "",
        f"- derived dataset: `{summary['dataset_path']}`",
        f"- feature scales: `{json.dumps(summary['feature_stats'])}`",
        "",
        "## Decision",
        "",
        "- Stage 0 passes only if reconstruction remains exact and elastic identity remains exact in Voigt form.",
    ]
    _write_phase_report(report_path, "NN Replacement ABR Stage 0 20260324", lines)


def write_stage1_report(
    report_path: Path,
    *,
    stage1_dataset: dict[str, Any],
    architecture_rows: list[dict[str, Any]],
    winner: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
) -> None:
    lines = [
        "## Training Setup",
        "",
        f"- stage1 dataset: `{stage1_dataset['dataset_path']}`",
        f"- coordinate scales: `{json.dumps(stage1_dataset['coordinate_scales'])}`",
        f"- train rows real / synthetic: `{stage1_dataset['train_real_rows']}` / `{stage1_dataset['train_synthetic_rows']}`",
        "",
        "## Architecture Sweep",
        "",
        "| Architecture | Param Count | Best Checkpoint | Broad Plastic MAE | Hard Plastic MAE | Hard p95 Principal | Yield Violation p95 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in architecture_rows:
        lines.append(
            f"| {row['architecture']} | {row['param_count']} | {row['selected_checkpoint_label']} | "
            f"{row['broad_plastic_mae']:.6f} | {row['hard_plastic_mae']:.6f} | {row['hard_p95_principal']:.6f} | {row['yield_violation_p95']:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Validation Winner",
            "",
            f"- architecture: `{winner['architecture']}`",
            f"- checkpoint: `{winner['selected_checkpoint_path']}`",
            "",
            "## One-Shot Test",
            "",
            f"- broad plastic MAE: `{winner_test['metrics']['broad_plastic_mae']:.6f}`",
            f"- hard plastic MAE: `{winner_test['metrics']['hard_plastic_mae']:.6f}`",
            f"- hard p95 principal: `{winner_test['metrics']['hard_p95_principal']:.6f}`",
            f"- yield violation p95: `{winner_test['metrics']['yield_violation_p95']:.6e}`",
            "",
            "## Decision",
            "",
            f"- Stage 1 credibility bar met: `{success}`",
        ]
    )
    _write_phase_report(report_path, "NN Replacement ABR Stage 1 20260324", lines)


def write_execution_report(
    report_path: Path,
    *,
    stage0_summary: dict[str, Any],
    stage1_dataset: dict[str, Any],
    winner: dict[str, Any],
    winner_test: dict[str, Any],
    success: bool,
) -> None:
    lines = [
        f"- Stage 0 derived dataset: `{stage0_summary['dataset_path']}`",
        f"- Stage 1 mixed dataset: `{stage1_dataset['dataset_path']}`",
        f"- Stage 1 winner: `{winner['architecture']}` at `{winner['selected_checkpoint_label']}`",
        f"- Validation hard plastic MAE / hard p95 / yield p95: `{winner['hard_plastic_mae']:.6f}` / `{winner['hard_p95_principal']:.6f}` / `{winner['yield_violation_p95']:.6e}`",
        f"- One-shot test hard plastic MAE / hard p95 / yield p95: `{winner_test['metrics']['hard_plastic_mae']:.6f}` / `{winner_test['metrics']['hard_p95_principal']:.6f}` / `{winner_test['metrics']['yield_violation_p95']:.6e}`",
        f"- Stage 1 go decision: `{success}`",
    ]
    _write_phase_report(report_path, "NN Replacement ABR Execution 20260324", lines)


def main() -> None:
    args = parse_args()
    redesign_root = (ROOT / args.redesign_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    real_dataset_path = redesign_root / "real_grouped_sampled_512.h5"
    panel_path = redesign_root / "panels" / "panel_sidecar.h5"

    stage0_dir = output_root / "exp0_coordinate_audit"
    stage1_dir = output_root / "exp1_small_acn"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    stage0_summary = build_stage0_dataset(
        real_dataset_path=real_dataset_path,
        panel_path=panel_path,
        output_dir=stage0_dir,
        force_rerun=args.force_rerun,
    )
    write_stage0_report(docs_root / "nn_replacement_abr_exp0_20260324.md", stage0_summary)

    if stage0_summary["reconstruction"]["max_abs"] > 1.0e-10 or stage0_summary["elastic_identity"]["max_abs"] > 1.0e-10:
        raise RuntimeError("Stage 0 failed exactness stop rules; aborting before Stage 1.")

    synth_summary = build_synthetic_hard_panel(
        redesign_root=redesign_root,
        output_root=stage1_dir / "synthetic_hard",
        sample_count=13107,
        seed=args.seed,
    )
    stage1_dataset = build_stage1_dataset(
        stage0_dataset_path=Path(stage0_summary["dataset_path"]),
        synthetic_dataset_path=Path(synth_summary["dataset_path"]),
        output_dir=stage1_dir / "data",
        seed=args.seed,
        force_rerun=args.force_rerun,
    )

    architectures = [
        ("acn_f1_w32_d2", 32, 2),
        ("acn_f1_w64_d2", 64, 2),
        ("acn_f1_w64_d3", 64, 3),
        ("acn_f1_w96_d3", 96, 3),
    ]
    architecture_rows: list[dict[str, Any]] = []
    val_summary_rows: list[dict[str, Any]] = []
    for arch_name, width, depth in architectures:
        run_dir = stage1_dir / arch_name
        config = TrainingConfig(
            dataset=stage1_dataset["dataset_path"],
            run_dir=str(run_dir),
            model_kind="trial_abr_acn_f1",
            epochs=60,
            batch_size=4096,
            lr=3.0e-4,
            weight_decay=1.0e-5,
            width=width,
            depth=depth,
            dropout=0.0,
            seed=args.seed,
            patience=60,
            device=args.device,
            scheduler_kind="cosine",
            warmup_epochs=2,
            min_lr=1.0e-5,
            regression_loss_kind="huber",
            huber_delta=1.0,
            snapshot_every_epochs=10,
        )
        summary = _train_with_batch_fallback(config, force_rerun=args.force_rerun)
        checkpoint_rows: list[dict[str, Any]] = []
        for checkpoint_path in _candidate_checkpoint_paths(run_dir):
            eval_result = evaluate_acn_checkpoint(
                checkpoint_path,
                dataset_path=Path(stage1_dataset["dataset_path"]),
                split_name="val",
                device=args.device,
                batch_size=args.eval_batch_size,
            )
            checkpoint_rows.append(
                {
                    "checkpoint_label": checkpoint_path.stem if checkpoint_path.parent.name == "snapshots" else checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                    **eval_result["metrics"],
                }
            )
        checkpoint_rows.sort(
            key=lambda row: (
                row["broad_plastic_mae"],
                row["hard_plastic_mae"],
                row["hard_p95_principal"],
                row["yield_violation_p95"],
                row["checkpoint_label"],
            )
        )
        selected = checkpoint_rows[0]
        _write_csv(
            run_dir / "checkpoint_sweep_val.csv",
            checkpoint_rows,
            [
                "checkpoint_label",
                "checkpoint_path",
                "split",
                "n_rows",
                "broad_mae",
                "hard_mae",
                "broad_plastic_mae",
                "hard_plastic_mae",
                "hard_p95_principal",
                "hard_rel_p95_principal",
                "yield_violation_p95",
                "plastic_coverage",
                "accepted_plastic_rows",
                "route_counts",
                "accepted_true_branch_counts",
            ],
        )
        row = {
            "architecture": arch_name,
            "run_dir": str(run_dir),
            "param_count": _parameter_count(Path(selected["checkpoint_path"]), args.device),
            "selected_checkpoint_label": selected["checkpoint_label"],
            "selected_checkpoint_path": selected["checkpoint_path"],
            "completed_epochs": summary["completed_epochs"],
            "best_epoch_by_train_loop": summary["best_epoch"],
            "batch_size": config.batch_size,
            "width": width,
            "depth": depth,
            "feature_set": "F1",
            "coordinate_scales": stage1_dataset["coordinate_scales"],
            **{key: selected[key] for key in ("broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95")},
        }
        architecture_rows.append(row)
        val_summary_rows.append(row)

    architecture_rows.sort(
        key=lambda row: (
            row["broad_plastic_mae"],
            row["hard_plastic_mae"],
            row["hard_p95_principal"],
            row["yield_violation_p95"],
            row["selected_checkpoint_label"],
        )
    )
    winner = architecture_rows[0]
    winner_test = evaluate_acn_checkpoint(
        Path(winner["selected_checkpoint_path"]),
        dataset_path=Path(stage1_dataset["dataset_path"]),
        split_name="test",
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    success = (
        winner["yield_violation_p95"] <= 1.0e-4
        and winner["broad_plastic_mae"] <= 5.771003
        and winner["hard_plastic_mae"] <= 6.949571
        and winner["hard_p95_principal"] <= 76.398964
    )

    _write_csv(
        stage1_dir / "architecture_summary_val.csv",
        val_summary_rows,
        [
            "architecture",
            "run_dir",
            "param_count",
            "selected_checkpoint_label",
            "selected_checkpoint_path",
            "completed_epochs",
            "best_epoch_by_train_loop",
            "batch_size",
            "width",
            "depth",
            "feature_set",
            "coordinate_scales",
            "broad_plastic_mae",
            "hard_plastic_mae",
            "hard_p95_principal",
            "hard_rel_p95_principal",
            "yield_violation_p95",
        ],
    )
    _write_json(stage1_dir / "checkpoint_selection.json", {"winner": winner, "all_architectures": architecture_rows})
    _write_json(stage1_dir / "winner_test_summary.json", winner_test)

    write_stage1_report(
        docs_root / "nn_replacement_abr_exp1_20260324.md",
        stage1_dataset=stage1_dataset,
        architecture_rows=architecture_rows,
        winner=winner,
        winner_test=winner_test,
        success=success,
    )
    write_execution_report(
        docs_root / "nn_replacement_abr_execution_20260324.md",
        stage0_summary=stage0_summary,
        stage1_dataset=stage1_dataset,
        winner=winner,
        winner_test=winner_test,
        success=success,
    )


if __name__ == "__main__":
    main()
