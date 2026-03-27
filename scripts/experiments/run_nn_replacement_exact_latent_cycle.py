#!/usr/bin/env python
"""Run Program X exact-latent oracle-ceiling study on the frozen March 24 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_surrogate.models import (
    ExactLatentRegressorNet,
    Standardizer,
    build_trial_exact_latent_features_f1,
    compute_trial_exact_latent_feature_stats,
    decode_exact_branch_latents_to_principal_torch,
    spectral_decomposition_from_strain,
    stress_voigt_from_principal_numpy,
    stress_voigt_from_principal_torch,
)
from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    BRANCH_TO_ID,
    constitutive_update_3d,
    decode_exact_branch_latents_to_principal,
    extract_exact_branch_latents,
    principal_relative_error_3d,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.training import choose_device, load_checkpoint, set_seed
from run_nn_replacement_abr_cycle import (
    PANEL_MASK_KEYS,
    _aggregate_policy_metrics,
    _load_h5,
    _plot_histograms,
    _quantile_or_zero,
    _write_csv,
    _write_json,
    _write_phase_report,
)

PACKET2_REFERENCE_VAL = {
    "broad_plastic_mae": 32.458569,
    "hard_plastic_mae": 37.824432,
    "hard_p95_principal": 328.641602,
    "yield_violation_p95": 6.173982e-08,
}
PACKET3_REFERENCE_VAL_ORACLE = {
    "broad_plastic_mae": 29.512514,
    "hard_plastic_mae": 32.682789,
    "hard_p95_principal": 302.360168,
    "yield_violation_p95": 5.600788e-08,
}
PACKET4_REFERENCE_VAL = {
    "broad_plastic_mae": 41.599422,
    "hard_plastic_mae": 46.363712,
    "hard_p95_principal": 374.894531,
    "yield_violation_p95": 6.008882e-08,
}
PACKET4_REFERENCE_TEST = {
    "broad_plastic_mae": 43.238693,
    "hard_plastic_mae": 48.038338,
    "hard_p95_principal": 388.482666,
    "yield_violation_p95": 5.825863e-08,
}
PROGRAM_X_MIN_BAR = {
    "broad_plastic_mae": 25.0,
    "hard_plastic_mae": 28.0,
    "hard_p95_principal": 250.0,
    "yield_violation_p95": 1.0e-6,
}
PROGRAM_X_STRONG_BAR = {
    "broad_plastic_mae": 20.0,
    "hard_plastic_mae": 24.0,
    "hard_p95_principal": 200.0,
    "yield_violation_p95": 1.0e-6,
}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
BRANCH_MODEL_ORDER = ("smooth", "left_edge", "right_edge")
BRANCH_ARCHITECTURES = {
    "smooth": [(48, 2), (64, 2), (96, 3)],
    "left_edge": [(32, 2), (48, 2), (64, 3)],
    "right_edge": [(32, 2), (48, 2), (64, 3)],
}
BRANCH_LATENT_SCHEMA = {
    "elastic": {"latent_kind": "elastic_analytic", "latent_dim": 0},
    "smooth": {"latent_kind": "plastic_multiplier", "latent_dim": 1},
    "left_edge": {"latent_kind": "plastic_multiplier", "latent_dim": 1},
    "right_edge": {"latent_kind": "plastic_multiplier", "latent_dim": 1},
    "apex": {"latent_kind": "analytic", "latent_dim": 0},
}
EXACT_LATENT_FEATURE_NAMES = [
    "asinh_p_tr",
    "log1p_g_tr",
    "atanh_rho_tr",
    "logit_lambda_tr",
    "asinh_f_tr",
    "m_yield",
    "m_smooth_left",
    "m_smooth_right",
    "m_left_apex",
    "m_right_apex",
    "gap12_norm",
    "gap23_norm",
    "gap_min_norm",
    "m_left_vs_right",
    "d_geom",
    "material_reduced_0",
    "material_reduced_1",
    "material_reduced_2",
    "material_reduced_3",
    "material_reduced_4",
]


@dataclass
class BranchTrainingConfig:
    branch_name: str
    branch_id: int
    width: int
    depth: int
    epochs: int = 60
    batch_size: int = 4096
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-5
    warmup_epochs: int = 2
    min_lr: float = 1.0e-5
    grad_clip: float = 1.0
    patience: int = 12
    device: str = "auto"
    seed: int = 20260324


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grouped-dataset",
        default="experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5",
    )
    parser.add_argument(
        "--panel-path",
        default="experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5",
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/nn_replacement_exact_latent_20260325",
    )
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


def _feature_stats_summary(features: np.ndarray) -> dict[str, Any]:
    feat = np.asarray(features, dtype=float)
    if feat.ndim != 2:
        return {
            "feature_dim": 0,
            "n_rows": 0,
            "all_finite": True,
            "nan_count": 0,
            "inf_count": 0,
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "q05": [],
            "q95": [],
        }
    return {
        "feature_dim": int(feat.shape[1]),
        "n_rows": int(feat.shape[0]),
        "all_finite": bool(np.isfinite(feat).all()),
        "nan_count": int(np.isnan(feat).sum()),
        "inf_count": int(np.isinf(feat).sum()),
        "mean": np.mean(feat, axis=0).astype(float).tolist(),
        "std": np.std(feat, axis=0).astype(float).tolist(),
        "min": np.min(feat, axis=0).astype(float).tolist(),
        "max": np.max(feat, axis=0).astype(float).tolist(),
        "q05": np.quantile(feat, 0.05, axis=0).astype(float).tolist(),
        "q95": np.quantile(feat, 0.95, axis=0).astype(float).tolist(),
    }


def _string_array(values: list[str] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def _latent_matrix(latent_values: np.ndarray, latent_dim: int) -> np.ndarray:
    values = np.asarray(latent_values, dtype=object)
    n = int(values.shape[0])
    if latent_dim == 0:
        return np.zeros((n, 0), dtype=np.float64)
    if values.dtype != object:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            return arr[:, None]
        return arr
    rows = [np.asarray(item, dtype=np.float64).reshape(1, latent_dim) for item in values]
    return np.concatenate(rows, axis=0).astype(np.float64) if rows else np.zeros((0, latent_dim), dtype=np.float64)


def _write_grouped_h5(
    path: Path,
    root_arrays: dict[str, np.ndarray],
    root_attrs: dict[str, Any],
    branch_groups: dict[str, dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in root_arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        for key, value in root_attrs.items():
            f.attrs[key] = json.dumps(_json_safe(value)) if isinstance(value, (dict, list, tuple)) else value
        branches = f.create_group("branches")
        for branch_name, payload in branch_groups.items():
            grp = branches.create_group(branch_name)
            for key, value in payload["datasets"].items():
                arr = np.asarray(value)
                if arr.dtype.kind in {"U", "O"}:
                    grp.create_dataset(key, data=_string_array(arr.astype(str).tolist()))
                else:
                    grp.create_dataset(key, data=arr, compression="gzip", shuffle=True)
            for key, value in payload.get("attrs", {}).items():
                grp.attrs[key] = json.dumps(_json_safe(value)) if isinstance(value, (dict, list, tuple)) else value
    return path


def _load_exact_latent_root(path: Path, keys: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        use_keys = keys if keys is not None else [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
        for key in use_keys:
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _load_exact_latent_branch(path: Path, branch_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        grp = f["branches"][branch_name]
        for key in grp.keys():
            arrays[key] = grp[key][:]
        for key, value in grp.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _slice_arrays_by_mask(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: value[mask] if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0] else value
        for key, value in arrays.items()
    }


def _symlink_or_copy(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(source.resolve(), dest)
    except OSError:
        shutil.copy2(source, dest)


def _reference_summary_sources() -> dict[str, dict[str, Path]]:
    return {
        "packet2": {
            "checkpoint_selection": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp1_branchless_surface/checkpoint_selection.json",
            "winner_test": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet2/exp1_branchless_surface/winner_test_summary.json",
        },
        "packet3": {
            "winner_val_predicted": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/winner_val_predicted_summary.json",
            "winner_val_oracle": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/winner_val_oracle_summary.json",
        },
        "packet4": {
            "winner_val_predicted": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_val_predicted_summary.json",
            "winner_val_oracle": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_val_oracle_summary.json",
            "winner_test": ROOT
            / "experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/winner_test_summary.json",
        },
    }


def build_baseline_freeze(output_dir: Path, *, force_rerun: bool) -> dict[str, Any]:
    baseline_manifest_path = output_dir / "baseline_manifest.json"
    reference_metrics_path = output_dir / "reference_metrics.json"
    execution_plan_path = output_dir / "execution_plan.md"
    summary_path = output_dir / "summary.json"
    if (
        baseline_manifest_path.exists()
        and reference_metrics_path.exists()
        and execution_plan_path.exists()
        and summary_path.exists()
        and not force_rerun
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    linked_dir = output_dir / "reference_links"
    linked_dir.mkdir(parents=True, exist_ok=True)
    linked: dict[str, dict[str, str]] = {}
    for family, sources in _reference_summary_sources().items():
        linked[family] = {}
        family_dir = linked_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for label, source in sources.items():
            if source.exists():
                dest = family_dir / source.name
                _symlink_or_copy(source, dest)
                linked[family][label] = str(dest)

    manifest = {
        "program": "Program X exact-latent branchwise oracle-ceiling study",
        "date_label": "20260325",
        "benchmark": {
            "source_h5": "constitutive_problem_3D_full.h5",
            "grouped_dataset": "experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5",
            "panel_sidecar": "experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5",
            "split_seed": 20260324,
            "split_unit": "constitutive_call",
            "split_fractions": [0.70, 0.15, 0.15],
            "samples_per_call": 512,
            "canonical_sizes": {
                "train": 198656,
                "val": 42496,
                "test": 42496,
                "broad_val": 42496,
                "broad_test": 42496,
                "hard_val": 21198,
                "hard_test": 22705,
                "ds_valid": 73261,
                "ds_valid_val": 11224,
                "ds_valid_test": 9839,
            },
        },
        "decision_bars": {
            "minimum_oracle_success": PROGRAM_X_MIN_BAR,
            "strong_oracle_success": PROGRAM_X_STRONG_BAR,
        },
        "references_linked": linked,
    }
    references = {
        "packet2_deployed_val": PACKET2_REFERENCE_VAL,
        "packet3_oracle_val": PACKET3_REFERENCE_VAL_ORACLE,
        "packet4_deployed_val": PACKET4_REFERENCE_VAL,
        "packet4_deployed_test": PACKET4_REFERENCE_TEST,
    }
    execution_lines = [
        "# Program X Execution Plan",
        "",
        "- Freeze the March 24 grouped benchmark and panel sidecar exactly as packet2/3/4 used them.",
        "- Instrument the exact solver to expose branchwise exact latents without changing the default constitutive API.",
        "- Build the exact-latent dataset, prove numerical exactness, train teacher-forced branchwise latent models once, and compare the oracle against packet2/3/4.",
        "- Stop the direct line unless the validation oracle beats packet2 clearly and reaches the Program X minimum oracle bar.",
    ]
    baseline_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reference_metrics_path.write_text(json.dumps(references, indent=2), encoding="utf-8")
    execution_plan_path.write_text("\n".join(execution_lines) + "\n", encoding="utf-8")
    summary = {
        "baseline_manifest": str(baseline_manifest_path),
        "reference_metrics": str(reference_metrics_path),
        "execution_plan": str(execution_plan_path),
        "linked_references": linked,
    }
    _write_json(summary_path, summary)
    return summary


def build_stage0_exact_latent_dataset(
    *,
    real_dataset_path: Path,
    panel_path: Path,
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    dataset_path = output_dir / "derived_exact_latent_dataset.h5"
    feature_stats_path = output_dir / "stage0_latent_feature_stats.json"
    branch_counts_path = output_dir / "branch_counts.json"
    schema_path = output_dir / "latent_schema_summary.json"
    summary_path = output_dir / "stage0_dataset_summary.json"
    if dataset_path.exists() and feature_stats_path.exists() and branch_counts_path.exists() and schema_path.exists() and summary_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    real_arrays, real_attrs = _load_h5(real_dataset_path)
    panel_arrays, panel_attrs = _load_h5(panel_path)
    split_id = real_arrays["split_id"].astype(np.int8)
    branch_id = real_arrays["branch_id"].astype(np.int8)
    plastic = branch_id > 0
    train_plastic = (split_id == SPLIT_TO_ID["train"]) & plastic

    extracted = extract_exact_branch_latents(
        real_arrays["strain_eng"],
        c_bar=real_arrays["material_reduced"][:, 0],
        sin_phi=real_arrays["material_reduced"][:, 1],
        shear=real_arrays["material_reduced"][:, 2],
        bulk=real_arrays["material_reduced"][:, 3],
        lame=real_arrays["material_reduced"][:, 4],
    )
    trial_principal = extracted["trial_principal"].astype(np.float64)
    trial_stress = stress_voigt_from_principal_numpy(trial_principal, real_arrays["eigvecs"].astype(np.float64)).astype(np.float64)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(real_arrays["strain_eng"])
    feature_stats = compute_trial_exact_latent_feature_stats(
        trial_principal[train_plastic],
        real_arrays["material_reduced"][train_plastic],
    )
    exact_latent_feature_f1 = build_trial_exact_latent_features_f1(
        strain_principal,
        real_arrays["material_reduced"],
        trial_principal,
        feature_stats,
    ).astype(np.float32)
    feature_stats_summary = {
        "feature_names": EXACT_LATENT_FEATURE_NAMES,
        "train_plastic": _feature_stats_summary(exact_latent_feature_f1[train_plastic]),
        "full_dataset": _feature_stats_summary(exact_latent_feature_f1),
    }
    feature_stats_summary["checks"] = {
        "train_plastic_all_finite": bool(feature_stats_summary["train_plastic"]["all_finite"]),
        "full_dataset_all_finite": bool(feature_stats_summary["full_dataset"]["all_finite"]),
        "nan_count": int(np.isnan(exact_latent_feature_f1).sum()),
        "inf_count": int(np.isinf(exact_latent_feature_f1).sum()),
    }

    branch_counts = {
        name: {
            "total": int(np.sum(branch_id == idx)),
            "train": int(np.sum((split_id == SPLIT_TO_ID["train"]) & (branch_id == idx))),
            "val": int(np.sum((split_id == SPLIT_TO_ID["val"]) & (branch_id == idx))),
            "test": int(np.sum((split_id == SPLIT_TO_ID["test"]) & (branch_id == idx))),
            "hard_total": int(np.sum((panel_arrays["hard_mask"] > 0) & (branch_id == idx))),
            "hard_val": int(np.sum((panel_arrays["hard_val_mask"] > 0) & (branch_id == idx))),
            "hard_test": int(np.sum((panel_arrays["hard_test_mask"] > 0) & (branch_id == idx))),
        }
        for idx, name in enumerate(BRANCH_NAMES)
    }

    schema_summary: dict[str, Any] = {}
    branch_groups: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(BRANCH_NAMES):
        if idx == BRANCH_TO_ID["elastic"]:
            schema_summary[name] = {
                "branch_id": idx,
                "latent_kind": BRANCH_LATENT_SCHEMA[name]["latent_kind"],
                "latent_dim": BRANCH_LATENT_SCHEMA[name]["latent_dim"],
                "count": branch_counts[name]["total"],
            }
            continue
        mask = branch_id == idx
        rows = np.flatnonzero(mask)
        latent_dim = int(extracted["latent_dim"][rows][0]) if rows.size else BRANCH_LATENT_SCHEMA[name]["latent_dim"]
        latent_kind = str(extracted["latent_kind"][rows][0]) if rows.size else BRANCH_LATENT_SCHEMA[name]["latent_kind"]
        latent_values = _latent_matrix(extracted["latent_values"][rows], latent_dim)
        branch_groups[name] = {
            "datasets": {
                "row_index": rows.astype(np.int64),
                "split_id": split_id[mask].astype(np.int8),
                "hard_mask": panel_arrays["hard_mask"][mask].astype(np.int8),
                "feature_f1": exact_latent_feature_f1[mask].astype(np.float32),
                "latent_kind": np.asarray([latent_kind] * rows.size),
                "latent_dim": np.full(rows.size, latent_dim, dtype=np.int8),
                "latent_values": latent_values.astype(np.float64),
                "trial_principal": trial_principal[mask].astype(np.float64),
                "stress_principal": real_arrays["stress_principal"][mask].astype(np.float32),
                "stress": real_arrays["stress"][mask].astype(np.float32),
                "eigvecs": real_arrays["eigvecs"][mask].astype(np.float32),
                "material_reduced": real_arrays["material_reduced"][mask].astype(np.float64),
                "f_trial": extracted["f_trial"][mask].astype(np.float32),
                "m_yield": panel_arrays["m_yield"][mask].astype(np.float32),
                "m_smooth_left": panel_arrays["m_smooth_left"][mask].astype(np.float32),
                "m_smooth_right": panel_arrays["m_smooth_right"][mask].astype(np.float32),
                "m_left_apex": panel_arrays["m_left_apex"][mask].astype(np.float32),
                "m_right_apex": panel_arrays["m_right_apex"][mask].astype(np.float32),
                "gap12_norm": panel_arrays["gap12_norm"][mask].astype(np.float32),
                "gap23_norm": panel_arrays["gap23_norm"][mask].astype(np.float32),
                "gap_min_norm": panel_arrays["gap_min_norm"][mask].astype(np.float32),
                "d_geom": panel_arrays["d_geom"][mask].astype(np.float32),
            },
            "attrs": {
                "branch_id": idx,
                "branch_name": name,
                "latent_kind": latent_kind,
                "latent_dim": latent_dim,
            },
        }
        schema_summary[name] = {
            "branch_id": idx,
            "latent_kind": latent_kind,
            "latent_dim": latent_dim,
            "count": int(rows.size),
            "hard_count": int(np.sum(panel_arrays["hard_mask"][mask] > 0)),
        }

    root_arrays = {
        "strain_eng": real_arrays["strain_eng"].astype(np.float32),
        "stress": real_arrays["stress"].astype(np.float32),
        "stress_principal": real_arrays["stress_principal"].astype(np.float32),
        "trial_principal": trial_principal.astype(np.float64),
        "trial_stress": trial_stress.astype(np.float64),
        "eigvecs": real_arrays["eigvecs"].astype(np.float32),
        "material_reduced": real_arrays["material_reduced"].astype(np.float32),
        "branch_id": branch_id.astype(np.int8),
        "split_id": split_id.astype(np.int8),
        "source_call_id": real_arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": real_arrays["source_row_in_call"].astype(np.int32),
        "f_trial": extracted["f_trial"].astype(np.float32),
        "exact_latent_feature_f1": exact_latent_feature_f1.astype(np.float32),
        "plastic_mask": panel_arrays["plastic_mask"].astype(np.int8),
        **{key: panel_arrays[key].astype(np.int8) for key in PANEL_MASK_KEYS},
        "m_yield": panel_arrays["m_yield"].astype(np.float32),
        "m_smooth_left": panel_arrays["m_smooth_left"].astype(np.float32),
        "m_smooth_right": panel_arrays["m_smooth_right"].astype(np.float32),
        "m_left_apex": panel_arrays["m_left_apex"].astype(np.float32),
        "m_right_apex": panel_arrays["m_right_apex"].astype(np.float32),
        "gap12_norm": panel_arrays["gap12_norm"].astype(np.float32),
        "gap23_norm": panel_arrays["gap23_norm"].astype(np.float32),
        "gap_min_norm": panel_arrays["gap_min_norm"].astype(np.float32),
        "d_geom": panel_arrays["d_geom"].astype(np.float32),
    }
    root_attrs = {
        "source_real_dataset": str(real_dataset_path),
        "source_panel_path": str(panel_path),
        "exact_latent_feature_stats_json": feature_stats,
        "exact_latent_schema_json": schema_summary,
        "split_seed": real_attrs.get("split_seed", 20260324),
        "branch_names_json": real_attrs.get("branch_names_json", json.dumps(BRANCH_NAMES)),
        "panel_summary_json": panel_attrs.get("panel_summary_json", ""),
    }
    _write_grouped_h5(dataset_path, root_arrays, root_attrs, branch_groups)
    _write_json(feature_stats_path, feature_stats_summary)
    _write_json(branch_counts_path, branch_counts)
    _write_json(schema_path, schema_summary)
    _plot_histograms(
        output_dir / "exact_latent_feature_histograms.png",
        {
            "train_f_trial": extracted["f_trial"][train_plastic],
            "val_f_trial": extracted["f_trial"][(split_id == SPLIT_TO_ID["val"]) & plastic],
            "test_f_trial": extracted["f_trial"][(split_id == SPLIT_TO_ID["test"]) & plastic],
        },
        title="Program X exact-latent feature inputs",
    )
    summary = {
        "dataset_path": str(dataset_path),
        "feature_stats": feature_stats,
        "feature_stats_path": str(feature_stats_path),
        "branch_counts_path": str(branch_counts_path),
        "latent_schema_path": str(schema_path),
        "checks": {
            "branch_id_matches_source": bool(np.array_equal(extracted["branch_id"].astype(np.int8), branch_id)),
            "principal_matches_source_max_abs": float(np.max(np.abs(extracted["sigma_principal"] - real_arrays["stress_principal"]))),
            "features_all_finite": bool(feature_stats_summary["checks"]["full_dataset_all_finite"]),
        },
    }
    _write_json(summary_path, summary)
    return summary


def run_stage0_exactness_audit(
    *,
    stage0_dataset_path: Path,
    output_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    summary_path = output_dir / "stage0_exactness_summary.json"
    csv_path = output_dir / "stage0_branchwise_exactness.csv"
    if summary_path.exists() and csv_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    root_arrays, _attrs = _load_exact_latent_root(stage0_dataset_path)
    branch_id = root_arrays["branch_id"].astype(np.int64)
    exact = constitutive_update_3d(
        root_arrays["strain_eng"],
        c_bar=root_arrays["material_reduced"][:, 0],
        sin_phi=root_arrays["material_reduced"][:, 1],
        shear=root_arrays["material_reduced"][:, 2],
        bulk=root_arrays["material_reduced"][:, 3],
        lame=root_arrays["material_reduced"][:, 4],
    )
    sigma_true = exact.stress_principal.astype(np.float64)
    sigma_pred = sigma_true.copy()

    for branch_name in ("smooth", "left_edge", "right_edge", "apex"):
        branch_arrays, _branch_attrs = _load_exact_latent_branch(stage0_dataset_path, branch_name)
        rows = branch_arrays["row_index"].astype(np.int64)
        decoded = decode_exact_branch_latents_to_principal(
            branch_id[rows],
            branch_arrays["trial_principal"].astype(np.float64),
            branch_arrays["material_reduced"].astype(np.float64),
            branch_arrays["latent_values"].astype(np.float64),
        ).astype(np.float64)
        sigma_pred[rows] = decoded

    overall_abs = np.abs(sigma_pred - sigma_true)
    yield_rel = yield_violation_rel_principal_3d(
        sigma_pred,
        c_bar=root_arrays["material_reduced"][:, 0],
        sin_phi=root_arrays["material_reduced"][:, 1],
    ).astype(np.float32)
    branch_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(BRANCH_NAMES):
        mask = branch_id == idx
        abs_err = np.abs(sigma_pred[mask] - sigma_true[mask])
        yield_branch = yield_violation_rel_principal_3d(
            sigma_pred[mask],
            c_bar=root_arrays["material_reduced"][mask, 0],
            sin_phi=root_arrays["material_reduced"][mask, 1],
        ).astype(np.float64) if np.any(mask) else np.zeros(0, dtype=np.float64)
        branch_rows.append(
            {
                "branch": name,
                "count": int(np.sum(mask)),
                "mean_abs": float(np.mean(abs_err)) if abs_err.size else 0.0,
                "max_abs": float(np.max(abs_err)) if abs_err.size else 0.0,
                "yield_p95": _quantile_or_zero(yield_branch, 0.95),
                "yield_max": float(np.max(yield_branch)) if yield_branch.size else 0.0,
            }
        )
    stage0_pass = bool(np.max(overall_abs) <= 5.0e-6 and _quantile_or_zero(yield_rel, 0.95) <= 1.0e-6)
    summary = {
        "dataset_path": str(stage0_dataset_path),
        "reconstruction": {
            "mean_abs": float(np.mean(overall_abs)),
            "max_abs": float(np.max(overall_abs)),
        },
        "yield": {
            "p95": _quantile_or_zero(yield_rel, 0.95),
            "max": float(np.max(yield_rel)),
        },
        "branch_agreement": float(np.mean(exact.branch_id.astype(np.int64) == branch_id)),
        "branch_rows": branch_rows,
        "stage0_pass": stage0_pass,
    }
    _write_csv(csv_path, branch_rows, ["branch", "count", "mean_abs", "max_abs", "yield_p95", "yield_max"])
    _write_json(summary_path, summary)
    return summary


def _decode_json_attr(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _branch_split_arrays(dataset_path: Path, branch_name: str, split_name: str) -> dict[str, np.ndarray]:
    arrays, _attrs = _load_exact_latent_branch(dataset_path, branch_name)
    mask = arrays["split_id"].astype(np.int8) == SPLIT_TO_ID[split_name]
    return _slice_arrays_by_mask(arrays, mask)


def _branch_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    feasible = 0.0 if metrics["yield_violation_p95"] <= 1.0e-6 else 1.0
    return (
        feasible,
        float(metrics["principal_abs_p95"]),
        float(metrics["stress_component_mae"]),
        float(metrics["latent_mae"]),
    )


def _save_branch_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    metadata: dict[str, Any],
) -> Path:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _predict_branch_model(
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    device_obj = choose_device(device)
    model = model.to(device_obj)
    x_scaler = Standardizer.from_dict(metadata["x_scaler"])
    y_scaler = Standardizer.from_dict(metadata["y_scaler"])
    branch_id = int(metadata["branch_id"])

    features = arrays["feature_f1"].astype(np.float32)
    trial_principal = arrays["trial_principal"].astype(np.float32)
    material = arrays["material_reduced"].astype(np.float32)
    eigvecs = arrays["eigvecs"].astype(np.float32)
    latent_true = arrays["latent_values"].astype(np.float32)
    if batch_size <= 0:
        batch_size = int(features.shape[0])

    latent_chunks: list[np.ndarray] = []
    principal_chunks: list[np.ndarray] = []
    stress_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            xb = torch.from_numpy(x_scaler.transform(features[start:stop])).to(device_obj)
            out = model(xb)["stress"]
            latent_pred = out * torch.as_tensor(y_scaler.std, device=device_obj) + torch.as_tensor(y_scaler.mean, device=device_obj)
            branch_tensor = torch.full((stop - start,), branch_id, dtype=torch.long, device=device_obj)
            principal_t = decode_exact_branch_latents_to_principal_torch(
                latent_pred,
                branch_tensor,
                torch.from_numpy(trial_principal[start:stop]).to(device_obj),
                torch.from_numpy(material[start:stop]).to(device_obj),
            )
            stress_t = stress_voigt_from_principal_torch(principal_t, torch.from_numpy(eigvecs[start:stop]).to(device_obj))
            latent_chunks.append(latent_pred.cpu().numpy().astype(np.float32))
            principal_chunks.append(principal_t.cpu().numpy().astype(np.float32))
            stress_chunks.append(stress_t.cpu().numpy().astype(np.float32))

    latent_pred = np.concatenate(latent_chunks, axis=0)
    principal_pred = np.concatenate(principal_chunks, axis=0)
    stress_pred = np.concatenate(stress_chunks, axis=0)
    yield_rel = yield_violation_rel_principal_3d(
        principal_pred,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
    ).astype(np.float32)
    principal_abs = np.abs(principal_pred - arrays["stress_principal"].astype(np.float32))
    stress_abs = np.abs(stress_pred - arrays["stress"].astype(np.float32))
    principal_max_abs = np.max(principal_abs, axis=1).astype(np.float32)
    repo_relative = principal_relative_error_3d(
        principal_pred,
        arrays["stress_principal"].astype(np.float32),
        c_bar=material[:, 0],
    ).astype(np.float32)
    hard = arrays["hard_mask"].astype(bool)
    metrics = {
        "split": int(arrays["split_id"][0]) if arrays["split_id"].size else -1,
        "count": int(features.shape[0]),
        "latent_mae": float(np.mean(np.abs(latent_pred - latent_true))) if latent_true.size else 0.0,
        "stress_component_mae": float(np.mean(stress_abs)),
        "principal_mae": float(np.mean(principal_abs)),
        "principal_abs_p95": float(np.quantile(principal_max_abs, 0.95)),
        "principal_rel_p95": float(np.quantile(repo_relative, 0.95)),
        "yield_violation_p95": _quantile_or_zero(yield_rel, 0.95),
        "yield_violation_max": float(np.max(yield_rel)) if yield_rel.size else 0.0,
        "hard_stress_component_mae": float(np.mean(stress_abs[hard])) if np.any(hard) else float("nan"),
        "hard_principal_abs_p95": float(np.quantile(principal_max_abs[hard], 0.95)) if np.any(hard) else float("nan"),
    }
    return {
        "metrics": metrics,
        "predictions": {
            "latent_values": latent_pred.astype(np.float32),
            "stress_principal": principal_pred.astype(np.float32),
            "stress": stress_pred.astype(np.float32),
        },
    }


def _principal_relative_loss(pred: torch.Tensor, true: torch.Tensor, c_bar: torch.Tensor) -> torch.Tensor:
    scale = torch.linalg.vector_norm(true, dim=1, keepdim=True) + torch.clamp(c_bar[:, None], min=0.0) + 1.0e-12
    return F.smooth_l1_loss((pred - true) / scale, torch.zeros_like(pred))


def _train_branch_architecture(
    *,
    dataset_path: Path,
    branch_name: str,
    config: BranchTrainingConfig,
    run_dir: Path,
    force_rerun: bool,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    if summary_path.exists() and best_path.exists() and not force_rerun:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    train_arrays = _branch_split_arrays(dataset_path, branch_name, "train")
    val_arrays = _branch_split_arrays(dataset_path, branch_name, "val")
    x_scaler = Standardizer.from_array(train_arrays["feature_f1"])
    y_scaler = Standardizer.from_array(train_arrays["latent_values"])

    x_train = torch.from_numpy(x_scaler.transform(train_arrays["feature_f1"]))
    y_train = torch.from_numpy(y_scaler.transform(train_arrays["latent_values"]))
    trial_train = torch.from_numpy(train_arrays["trial_principal"].astype(np.float32))
    stress_train = torch.from_numpy(train_arrays["stress"].astype(np.float32))
    principal_train = torch.from_numpy(train_arrays["stress_principal"].astype(np.float32))
    eigvecs_train = torch.from_numpy(train_arrays["eigvecs"].astype(np.float32))
    material_train = torch.from_numpy(train_arrays["material_reduced"].astype(np.float32))
    weights = np.where(train_arrays["hard_mask"].astype(bool), 2.0, 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=int(weights.shape[0]), replacement=True)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train, trial_train, stress_train, principal_train, eigvecs_train, material_train),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    device_obj = choose_device(config.device)
    model = ExactLatentRegressorNet(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        width=config.width,
        depth=config.depth,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.warmup_epochs > 0 and config.epochs > config.warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs - config.warmup_epochs, 1),
            eta_min=config.min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs, 1),
            eta_min=config.min_lr,
        )

    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    checkpoint_metadata = {
        "config": {
            "model_kind": "exact_latent_regressor",
            "width": config.width,
            "depth": config.depth,
            "dropout": 0.0,
        },
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "branch_name": branch_name,
        "branch_id": config.branch_id,
        "latent_kind": BRANCH_LATENT_SCHEMA[branch_name]["latent_kind"],
        "latent_dim": BRANCH_LATENT_SCHEMA[branch_name]["latent_dim"],
    }

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train = 0
        for xb, yb, trial_b, stress_b, principal_b, eigvecs_b, material_b in train_loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            trial_b = trial_b.to(device_obj)
            stress_b = stress_b.to(device_obj)
            principal_b = principal_b.to(device_obj)
            eigvecs_b = eigvecs_b.to(device_obj)
            material_b = material_b.to(device_obj)

            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(xb)["stress"]
            pred_raw = pred_norm * torch.as_tensor(y_scaler.std, device=device_obj) + torch.as_tensor(y_scaler.mean, device=device_obj)
            branch_tensor = torch.full((xb.shape[0],), config.branch_id, dtype=torch.long, device=device_obj)
            principal_pred = decode_exact_branch_latents_to_principal_torch(pred_raw, branch_tensor, trial_b, material_b)
            stress_pred = stress_voigt_from_principal_torch(principal_pred, eigvecs_b)

            loss_latent = F.smooth_l1_loss(pred_norm, yb)
            loss_principal = _principal_relative_loss(principal_pred, principal_b, material_b[:, 0])
            loss_voigt = torch.mean(torch.abs(stress_pred - stress_b))
            loss = loss_latent + 0.5 * loss_principal + 0.25 * loss_voigt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_loss_total += float(loss.detach().cpu()) * xb.shape[0]
            n_train += int(xb.shape[0])

        scheduler.step()
        checkpoint_metadata["current_epoch"] = epoch
        checkpoint_metadata["train_loss"] = train_loss_total / max(n_train, 1)
        last_path = run_dir / "last.pt"
        _save_branch_checkpoint(last_path, model=model, metadata=checkpoint_metadata)

        val_eval = _predict_branch_model(last_path, val_arrays, device=config.device, batch_size=config.batch_size)
        val_metrics = val_eval["metrics"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(n_train, 1),
            "val_latent_mae": val_metrics["latent_mae"],
            "val_stress_component_mae": val_metrics["stress_component_mae"],
            "val_principal_abs_p95": val_metrics["principal_abs_p95"],
            "val_yield_violation_p95": val_metrics["yield_violation_p95"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(history_row)

        if best_metrics is None or _branch_selection_key(val_metrics) < _branch_selection_key(best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
            _save_branch_checkpoint(best_path, model=model, metadata=checkpoint_metadata)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    _write_csv(
        run_dir / "history.csv",
        history_rows,
        ["epoch", "train_loss", "val_latent_mae", "val_stress_component_mae", "val_principal_abs_p95", "val_yield_violation_p95", "lr"],
    )
    train_config = {
        "branch_name": branch_name,
        "branch_id": config.branch_id,
        "width": config.width,
        "depth": config.depth,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "min_lr": config.min_lr,
        "grad_clip": config.grad_clip,
        "patience": config.patience,
        "device": config.device,
        "seed": config.seed,
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    summary = {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "completed_epochs": len(history_rows),
        "param_count": int(sum(param.numel() for param in model.parameters())),
        "best_val_metrics": best_metrics,
    }
    _write_json(summary_path, summary)
    return summary


def _branch_selection_row(
    *,
    architecture: str,
    summary: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "selected_checkpoint_path": summary["best_checkpoint"],
        "best_epoch": summary["best_epoch"],
        "completed_epochs": summary["completed_epochs"],
        "param_count": summary["param_count"],
        "latent_mae": val_metrics["latent_mae"],
        "stress_component_mae": val_metrics["stress_component_mae"],
        "principal_abs_p95": val_metrics["principal_abs_p95"],
        "principal_rel_p95": val_metrics["principal_rel_p95"],
        "yield_violation_p95": val_metrics["yield_violation_p95"],
    }


def train_branchwise_oracle_models(
    *,
    stage0_dataset_path: Path,
    output_dir: Path,
    device: str,
    seed: int,
    force_rerun: bool,
) -> dict[str, Any]:
    winners_path = output_dir / "winner_selection.json"
    if winners_path.exists() and not force_rerun:
        return json.loads(winners_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    winners: dict[str, Any] = {}
    for branch_name in BRANCH_MODEL_ORDER:
        branch_id = BRANCH_TO_ID[branch_name]
        branch_dir = output_dir / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for width, depth in BRANCH_ARCHITECTURES[branch_name]:
            arch_id = f"{branch_name}_w{width}_d{depth}"
            run_dir = branch_dir / arch_id
            config = BranchTrainingConfig(
                branch_name=branch_name,
                branch_id=branch_id,
                width=width,
                depth=depth,
                device=device,
                seed=seed,
            )
            summary = _train_branch_architecture(
                dataset_path=stage0_dataset_path,
                branch_name=branch_name,
                config=config,
                run_dir=run_dir,
                force_rerun=force_rerun,
            )
            val_arrays = _branch_split_arrays(stage0_dataset_path, branch_name, "val")
            val_eval = _predict_branch_model(Path(summary["best_checkpoint"]), val_arrays, device=device, batch_size=config.batch_size)
            rows.append(_branch_selection_row(architecture=arch_id, summary=summary, val_metrics=val_eval["metrics"]))
        rows.sort(key=_branch_selection_key)
        winner = rows[0]
        test_arrays = _branch_split_arrays(stage0_dataset_path, branch_name, "test")
        winner_val = _predict_branch_model(Path(winner["selected_checkpoint_path"]), _branch_split_arrays(stage0_dataset_path, branch_name, "val"), device=device, batch_size=4096)
        winner_test = _predict_branch_model(Path(winner["selected_checkpoint_path"]), test_arrays, device=device, batch_size=4096)
        _write_csv(
            branch_dir / "architecture_summary_val.csv",
            rows,
            [
                "architecture",
                "param_count",
                "selected_checkpoint_path",
                "best_epoch",
                "completed_epochs",
                "latent_mae",
                "stress_component_mae",
                "principal_abs_p95",
                "principal_rel_p95",
                "yield_violation_p95",
            ],
        )
        _write_json(
            branch_dir / "checkpoint_selection.json",
            {
                "selected_row": winner,
                "selection_rule": [
                    "require yield_violation_p95 <= 1e-6 when available",
                    "rank by principal_abs_p95, stress_component_mae, latent_mae",
                ],
            },
        )
        _write_json(branch_dir / "winner_val_summary.json", _compact_branch_eval(winner_val))
        _write_json(branch_dir / "winner_test_summary.json", _compact_branch_eval(winner_test))
        winners[branch_name] = {
            **winner,
            "winner_val_summary": str(branch_dir / "winner_val_summary.json"),
            "winner_test_summary": str(branch_dir / "winner_test_summary.json"),
        }

    apex_rows = {
        "branch_name": "apex",
        "branch_id": BRANCH_TO_ID["apex"],
        "latent_kind": BRANCH_LATENT_SCHEMA["apex"]["latent_kind"],
        "latent_dim": BRANCH_LATENT_SCHEMA["apex"]["latent_dim"],
        "mode": "analytic_only",
    }
    _write_json(output_dir / "apex_analytic_only.json", apex_rows)
    summary = {
        "winners": winners,
        "apex": apex_rows,
    }
    _write_json(winners_path, summary)
    return summary


def _branch_breakdown_rows(
    branch_id: np.ndarray,
    stress_principal_true: np.ndarray,
    stress_principal_pred: np.ndarray,
    stress_true: np.ndarray,
    stress_pred: np.ndarray,
    material_reduced: np.ndarray,
) -> list[dict[str, Any]]:
    stress_abs = np.abs(stress_pred - stress_true)
    principal_abs = np.abs(stress_principal_pred - stress_principal_true)
    principal_abs_max = np.max(principal_abs, axis=1)
    repo_relative = principal_relative_error_3d(
        stress_principal_pred,
        stress_principal_true,
        c_bar=material_reduced[:, 0],
    ).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for branch_value, branch_name in enumerate(BRANCH_NAMES):
        mask = branch_id == branch_value
        if not np.any(mask):
            continue
        rows.append(
            {
                "branch": branch_name,
                "count": int(np.sum(mask)),
                "fraction_of_split": float(np.mean(mask)),
                "stress_component_mae": float(np.mean(stress_abs[mask])),
                "principal_abs_p95": float(np.quantile(principal_abs_max[mask], 0.95)),
                "repo_relative_p95": float(np.quantile(repo_relative[mask], 0.95)),
            }
        )
    return rows


def _evaluate_combined_oracle(
    *,
    stage0_dataset_path: Path,
    split_name: str,
    winners: dict[str, Any],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    root_arrays, _attrs = _load_exact_latent_root(stage0_dataset_path)
    split_mask = root_arrays["split_id"].astype(np.int8) == SPLIT_TO_ID[split_name]
    branch_id = root_arrays["branch_id"].astype(np.int64)
    stress_true = root_arrays["stress"].astype(np.float32)
    principal_true = root_arrays["stress_principal"].astype(np.float32)
    stress_pred = stress_true.copy()
    principal_pred = principal_true.copy()
    branch_prediction_payloads: dict[str, dict[str, Any]] = {}

    for branch_name in BRANCH_MODEL_ORDER:
        branch_arrays, _branch_attrs = _load_exact_latent_branch(stage0_dataset_path, branch_name)
        branch_eval = _predict_branch_model(
            Path(winners[branch_name]["selected_checkpoint_path"]),
            branch_arrays,
            device=device,
            batch_size=batch_size,
        )
        rows = branch_arrays["row_index"].astype(np.int64)
        principal_pred[rows] = branch_eval["predictions"]["stress_principal"]
        stress_pred[rows] = branch_eval["predictions"]["stress"]
        branch_prediction_payloads[branch_name] = {
            "row_index": rows.tolist(),
            "split_id": branch_arrays["split_id"].astype(int).tolist(),
            "hard_mask": branch_arrays["hard_mask"].astype(int).tolist(),
            "latent_true": branch_arrays["latent_values"].astype(np.float32).tolist(),
            "latent_pred": branch_eval["predictions"]["latent_values"].astype(np.float32).tolist(),
            "principal_true": branch_arrays["stress_principal"].astype(np.float32).tolist(),
            "principal_pred": branch_eval["predictions"]["stress_principal"].astype(np.float32).tolist(),
        }

    apex_arrays, _apex_attrs = _load_exact_latent_branch(stage0_dataset_path, "apex")
    apex_rows = apex_arrays["row_index"].astype(np.int64)
    if apex_rows.size:
        apex_principal = decode_exact_branch_latents_to_principal(
            np.full(apex_rows.shape[0], BRANCH_TO_ID["apex"], dtype=np.int64),
            apex_arrays["trial_principal"].astype(np.float32),
            apex_arrays["material_reduced"].astype(np.float32),
            apex_arrays["latent_values"].astype(np.float32),
        ).astype(np.float32)
        principal_pred[apex_rows] = apex_principal
        stress_pred[apex_rows] = stress_true[apex_rows]
        branch_prediction_payloads["apex"] = {
            "row_index": apex_rows.tolist(),
            "split_id": apex_arrays["split_id"].astype(int).tolist(),
            "hard_mask": apex_arrays["hard_mask"].astype(int).tolist(),
            "principal_true": apex_arrays["stress_principal"].astype(np.float32).tolist(),
            "principal_pred": apex_principal.astype(np.float32).tolist(),
        }

    arrays = {
        "stress": stress_true[split_mask],
        "stress_principal": principal_true[split_mask],
        "material_reduced": root_arrays["material_reduced"][split_mask].astype(np.float32),
        "branch_id": branch_id[split_mask].astype(np.int64),
    }
    panel = {
        "plastic_mask": root_arrays["plastic_mask"][split_mask].astype(np.int8),
        "hard_mask": root_arrays["hard_mask"][split_mask].astype(np.int8),
    }
    stress_pred_split = stress_pred[split_mask]
    principal_pred_split = principal_pred[split_mask]
    yield_rel = yield_violation_rel_principal_3d(
        principal_pred_split,
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
    ).astype(np.float32)
    stats = {
        "stress_component_abs": np.abs(stress_pred_split - arrays["stress"]).astype(np.float32),
        "principal_max_abs": np.max(np.abs(principal_pred_split - arrays["stress_principal"]), axis=1).astype(np.float32),
        "principal_rel_error": principal_relative_error_3d(
            principal_pred_split,
            arrays["stress_principal"],
            c_bar=arrays["material_reduced"][:, 0],
        ).astype(np.float32),
        "yield_violation_rel": yield_rel,
    }
    metrics = _aggregate_policy_metrics(
        arrays,
        panel,
        stats,
        learned_mask=np.ones(arrays["stress"].shape[0], dtype=bool),
        elastic_mask=(arrays["branch_id"] == BRANCH_TO_ID["elastic"]),
    )
    metrics.update(
        {
            "split": split_name,
            "yield_violation_max": float(np.max(yield_rel)) if yield_rel.size else 0.0,
        }
    )
    branch_rows = _branch_breakdown_rows(
        arrays["branch_id"],
        arrays["stress_principal"],
        principal_pred_split,
        arrays["stress"],
        stress_pred_split,
        arrays["material_reduced"],
    )
    return {
        "metrics": metrics,
        "branch_rows": branch_rows,
        "predictions": {
            "stress": stress_pred_split.astype(np.float32),
            "stress_principal": principal_pred_split.astype(np.float32),
        },
        "arrays": arrays,
        "branch_prediction_payloads": branch_prediction_payloads,
    }


def _compact_branch_eval(eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": eval_result["metrics"],
    }


def _compact_oracle_eval(eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": eval_result["metrics"],
        "branch_rows": eval_result["branch_rows"],
    }


def _plot_branch_principal_parity(path: Path, evaluation: dict[str, Any], *, split_name: str) -> None:
    branch_payloads = evaluation["branch_prediction_payloads"]
    plot_branches = (*BRANCH_MODEL_ORDER, "apex")
    fig, axes = plt.subplots(1, len(plot_branches), figsize=(5.5 * len(plot_branches), 5), squeeze=False)
    for ax, branch_name in zip(axes[0], plot_branches, strict=False):
        payload = branch_payloads[branch_name]
        split_id = np.asarray(payload["split_id"], dtype=np.int64)
        mask = split_id == SPLIT_TO_ID[split_name]
        true = np.asarray(payload["principal_true"], dtype=np.float32)[mask].reshape(-1)
        pred = np.asarray(payload["principal_pred"], dtype=np.float32)[mask].reshape(-1)
        if true.size:
            step = max(true.size // 4000, 1)
            ax.scatter(true[::step], pred[::step], s=4, alpha=0.25, color="#244c5a")
            lo = float(min(np.min(true), np.min(pred)))
            hi = float(max(np.max(true), np.max(pred)))
            ax.plot([lo, hi], [lo, hi], "--", color="#d26b34", linewidth=1.0)
        ax.set_title(branch_name)
        ax.set_xlabel("true principal")
        ax.set_ylabel("pred principal")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Program X {split_name} principal parity by branch")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_branch_latent_parity(path: Path, evaluation: dict[str, Any], *, split_name: str) -> None:
    fig, axes = plt.subplots(1, len(BRANCH_MODEL_ORDER), figsize=(5.5 * len(BRANCH_MODEL_ORDER), 5), squeeze=False)
    for ax, branch_name in zip(axes[0], BRANCH_MODEL_ORDER, strict=False):
        payload = evaluation["branch_prediction_payloads"][branch_name]
        split_id = np.asarray(payload["split_id"], dtype=np.int64)
        mask = split_id == SPLIT_TO_ID[split_name]
        true = np.asarray(payload["latent_true"], dtype=np.float32)[mask].reshape(-1)
        pred = np.asarray(payload["latent_pred"], dtype=np.float32)[mask].reshape(-1)
        if true.size:
            step = max(true.size // 4000, 1)
            ax.scatter(true[::step], pred[::step], s=4, alpha=0.25, color="#3a7d44")
            lo = float(min(np.min(true), np.min(pred)))
            hi = float(max(np.max(true), np.max(pred)))
            ax.plot([lo, hi], [lo, hi], "--", color="#d26b34", linewidth=1.0)
        ax.set_title(branch_name)
        ax.set_xlabel("true latent")
        ax.set_ylabel("pred latent")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Program X {split_name} latent parity by branch")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_branch_error_cdfs(path: Path, evaluation: dict[str, Any], *, split_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for branch_name in (*BRANCH_MODEL_ORDER, "apex"):
        payload = evaluation["branch_prediction_payloads"][branch_name]
        split_id = np.asarray(payload["split_id"], dtype=np.int64)
        mask = split_id == SPLIT_TO_ID[split_name]
        true = np.asarray(payload["principal_true"], dtype=np.float32)[mask]
        pred = np.asarray(payload["principal_pred"], dtype=np.float32)[mask]
        if not true.size:
            continue
        err = np.max(np.abs(pred - true), axis=1)
        xs = np.sort(err)
        ys = np.linspace(0.0, 1.0, xs.size, endpoint=True)
        ax.plot(xs, ys, label=branch_name)
    ax.set_xlabel("principal max abs error")
    ax.set_ylabel("CDF")
    ax.set_title(f"Program X {split_name} branchwise error CDFs")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_broad_vs_hard_summary(path: Path, metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    axes[0, 0].bar(["broad", "hard"], [metrics["broad_plastic_mae"], metrics["hard_plastic_mae"]], color=["#244c5a", "#d26b34"])
    axes[0, 0].set_title("Plastic stress MAE")
    axes[0, 0].grid(True, axis="y", alpha=0.25)
    axes[0, 1].bar(["hard p95", "yield p95"], [metrics["hard_p95_principal"], metrics["yield_violation_p95"]], color=["#244c5a", "#3a7d44"])
    axes[0, 1].set_title("Validation tail metrics")
    axes[0, 1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Program X broad vs hard validation summary")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _decision_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    clears_min_bar = bool(
        metrics["broad_plastic_mae"] <= PROGRAM_X_MIN_BAR["broad_plastic_mae"]
        and metrics["hard_plastic_mae"] <= PROGRAM_X_MIN_BAR["hard_plastic_mae"]
        and metrics["hard_p95_principal"] <= PROGRAM_X_MIN_BAR["hard_p95_principal"]
        and metrics["yield_violation_p95"] <= PROGRAM_X_MIN_BAR["yield_violation_p95"]
    )
    clears_strong_bar = bool(
        metrics["broad_plastic_mae"] <= PROGRAM_X_STRONG_BAR["broad_plastic_mae"]
        and metrics["hard_plastic_mae"] <= PROGRAM_X_STRONG_BAR["hard_plastic_mae"]
        and metrics["hard_p95_principal"] <= PROGRAM_X_STRONG_BAR["hard_p95_principal"]
        and metrics["yield_violation_p95"] <= PROGRAM_X_STRONG_BAR["yield_violation_p95"]
    )
    beats_packet2 = bool(
        metrics["broad_plastic_mae"] < PACKET2_REFERENCE_VAL["broad_plastic_mae"]
        and metrics["hard_plastic_mae"] < PACKET2_REFERENCE_VAL["hard_plastic_mae"]
        and metrics["hard_p95_principal"] < PACKET2_REFERENCE_VAL["hard_p95_principal"]
    )
    decision = "open_routing_study" if (beats_packet2 and clears_min_bar) else "stop_direct_replacement"
    return {
        "beats_packet2_clearly": beats_packet2,
        "clears_minimum_oracle_bar": clears_min_bar,
        "approaches_strong_oracle_bar": clears_strong_bar,
        "decision": decision,
        "ds_status": "blocked" if not clears_strong_bar else "still_blocked_pending_explicit_followup",
    }


def write_stage0_report(report_path: Path, *, dataset_summary: dict[str, Any], exactness_summary: dict[str, Any]) -> None:
    lines = [
        "## Frozen Benchmark",
        "",
        "- Program X reuses the frozen March 24 grouped dataset and panel sidecar exactly as packet2/3/4 used them.",
        "- Packet4 is closed and this study does not widen the packet4 family.",
        "",
        "## Exact Latent Schema",
        "",
        "- `smooth`: one scalar `plastic_multiplier` latent",
        "- `left_edge`: one scalar `plastic_multiplier` latent",
        "- `right_edge`: one scalar `plastic_multiplier` latent",
        "- `apex`: analytic-only, zero latent dimension",
        "",
        "## Stage 0 Exactness",
        "",
        f"- dataset: `{dataset_summary['dataset_path']}`",
        f"- feature finiteness: `{dataset_summary['checks']['features_all_finite']}`",
        f"- branch-id match vs source: `{dataset_summary['checks']['branch_id_matches_source']}`",
        f"- source principal match max abs: `{dataset_summary['checks']['principal_matches_source_max_abs']:.3e}`",
        f"- round-trip mean / max abs: `{exactness_summary['reconstruction']['mean_abs']:.3e}` / `{exactness_summary['reconstruction']['max_abs']:.3e}`",
        f"- yield p95 / max: `{exactness_summary['yield']['p95']:.3e}` / `{exactness_summary['yield']['max']:.3e}`",
        f"- Stage 0 pass: `{exactness_summary['stage0_pass']}`",
    ]
    _write_phase_report(report_path, "NN Replacement Exact Latent Stage0 20260325", lines)


def write_oracle_report(
    report_path: Path,
    *,
    val_eval: dict[str, Any],
    decision_summary: dict[str, Any],
) -> None:
    metrics = val_eval["metrics"]
    lines = [
        "## Validation Oracle",
        "",
        f"- broad plastic MAE: `{metrics['broad_plastic_mae']:.6f}`",
        f"- hard plastic MAE: `{metrics['hard_plastic_mae']:.6f}`",
        f"- hard p95 principal: `{metrics['hard_p95_principal']:.6f}`",
        f"- hard relative p95 principal: `{metrics['hard_rel_p95_principal']:.6f}`",
        f"- yield violation p95 / max: `{metrics['yield_violation_p95']:.6e}` / `{metrics['yield_violation_max']:.6e}`",
        "",
        "## Comparison",
        "",
        f"- packet2 validation broad / hard / p95: `{PACKET2_REFERENCE_VAL['broad_plastic_mae']:.6f}` / `{PACKET2_REFERENCE_VAL['hard_plastic_mae']:.6f}` / `{PACKET2_REFERENCE_VAL['hard_p95_principal']:.6f}`",
        f"- packet3 oracle validation broad / hard / p95: `{PACKET3_REFERENCE_VAL_ORACLE['broad_plastic_mae']:.6f}` / `{PACKET3_REFERENCE_VAL_ORACLE['hard_plastic_mae']:.6f}` / `{PACKET3_REFERENCE_VAL_ORACLE['hard_p95_principal']:.6f}`",
        f"- packet4 validation broad / hard / p95: `{PACKET4_REFERENCE_VAL['broad_plastic_mae']:.6f}` / `{PACKET4_REFERENCE_VAL['hard_plastic_mae']:.6f}` / `{PACKET4_REFERENCE_VAL['hard_p95_principal']:.6f}`",
        "",
        "## Decision",
        "",
        f"- beats packet2 clearly: `{decision_summary['beats_packet2_clearly']}`",
        f"- clears minimum oracle bar: `{decision_summary['clears_minimum_oracle_bar']}`",
        f"- approaches strong oracle bar: `{decision_summary['approaches_strong_oracle_bar']}`",
        f"- decision: `{decision_summary['decision']}`",
        f"- `DS` status: `{decision_summary['ds_status']}`",
    ]
    _write_phase_report(report_path, "NN Replacement Exact Latent Oracle 20260325", lines)


def write_execution_report(
    report_path: Path,
    *,
    baseline_summary: dict[str, Any],
    exactness_summary: dict[str, Any],
    val_eval: dict[str, Any],
    test_eval: dict[str, Any],
    decision_summary: dict[str, Any],
) -> None:
    val_metrics = val_eval["metrics"]
    test_metrics = test_eval["metrics"]
    lines = [
        "## Program X",
        "",
        "- Authoritative brief: `report11.md`",
        "- Packet4 remains closed; this is a new exact-latent representation study.",
        f"- baseline freeze manifest: `{baseline_summary['baseline_manifest']}`",
        "",
        "## Exactness",
        "",
        f"- Stage 0 pass: `{exactness_summary['stage0_pass']}`",
        f"- round-trip max abs: `{exactness_summary['reconstruction']['max_abs']:.3e}`",
        "",
        "## Oracle Result",
        "",
        f"- validation broad / hard plastic MAE: `{val_metrics['broad_plastic_mae']:.6f}` / `{val_metrics['hard_plastic_mae']:.6f}`",
        f"- validation hard p95 principal / yield p95: `{val_metrics['hard_p95_principal']:.6f}` / `{val_metrics['yield_violation_p95']:.6e}`",
        f"- test broad / hard plastic MAE: `{test_metrics['broad_plastic_mae']:.6f}` / `{test_metrics['hard_plastic_mae']:.6f}`",
        f"- test hard p95 principal / yield p95: `{test_metrics['hard_p95_principal']:.6f}` / `{test_metrics['yield_violation_p95']:.6e}`",
        "",
        "## Decision",
        "",
        f"- final decision: `{decision_summary['decision']}`",
        f"- `DS` status: `{decision_summary['ds_status']}`",
    ]
    _write_phase_report(report_path, "NN Replacement Exact Latent Execution 20260325", lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    grouped_dataset = (ROOT / args.grouped_dataset).resolve()
    panel_path = (ROOT / args.panel_path).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    exp0_dir = output_root / "exp0_baseline_freeze"
    exp1_dir = output_root / "exp1_latent_dataset"
    exp2_dir = output_root / "exp2_stage0_exactness"
    exp3_dir = output_root / "exp3_branchwise_oracle"
    exp4_dir = output_root / "exp4_oracle_eval"
    for path in (exp0_dir, exp1_dir, exp2_dir, exp3_dir, exp4_dir, docs_root):
        path.mkdir(parents=True, exist_ok=True)

    baseline_summary = build_baseline_freeze(exp0_dir, force_rerun=args.force_rerun)
    dataset_summary = build_stage0_exact_latent_dataset(
        real_dataset_path=grouped_dataset,
        panel_path=panel_path,
        output_dir=exp1_dir,
        force_rerun=args.force_rerun,
    )
    exactness_summary = run_stage0_exactness_audit(
        stage0_dataset_path=Path(dataset_summary["dataset_path"]),
        output_dir=exp2_dir,
        force_rerun=args.force_rerun,
    )
    if not exactness_summary["stage0_pass"]:
        decision_summary = {
            "beats_packet2_clearly": False,
            "clears_minimum_oracle_bar": False,
            "approaches_strong_oracle_bar": False,
            "decision": "stop_direct_replacement",
            "ds_status": "blocked",
        }
        write_stage0_report(
            docs_root / "nn_replacement_exact_latent_stage0_20260325.md",
            dataset_summary=dataset_summary,
            exactness_summary=exactness_summary,
        )
        write_execution_report(
            docs_root / "nn_replacement_exact_latent_execution_20260325.md",
            baseline_summary=baseline_summary,
            exactness_summary=exactness_summary,
            val_eval={"metrics": {k: float("nan") for k in ("broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "yield_violation_p95")}},
            test_eval={"metrics": {k: float("nan") for k in ("broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "yield_violation_p95")}},
            decision_summary=decision_summary,
        )
        _write_json(exp4_dir / "decision_logic_summary.json", decision_summary)
        return

    winners_summary = train_branchwise_oracle_models(
        stage0_dataset_path=Path(dataset_summary["dataset_path"]),
        output_dir=exp3_dir,
        device=args.device,
        seed=args.seed,
        force_rerun=args.force_rerun,
    )
    val_eval = _evaluate_combined_oracle(
        stage0_dataset_path=Path(dataset_summary["dataset_path"]),
        split_name="val",
        winners=winners_summary["winners"],
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    test_eval = _evaluate_combined_oracle(
        stage0_dataset_path=Path(dataset_summary["dataset_path"]),
        split_name="test",
        winners=winners_summary["winners"],
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    decision_summary = _decision_summary(val_eval["metrics"])
    _write_json(exp4_dir / "oracle_val_summary.json", _compact_oracle_eval(val_eval))
    _write_json(exp4_dir / "oracle_test_summary.json", _compact_oracle_eval(test_eval))
    _write_json(exp4_dir / "decision_logic_summary.json", decision_summary)
    _write_csv(
        exp4_dir / "winner_val_branch_breakdown.csv",
        val_eval["branch_rows"],
        ["branch", "count", "fraction_of_split", "stress_component_mae", "principal_abs_p95", "repo_relative_p95"],
    )
    _write_csv(
        exp4_dir / "winner_test_branch_breakdown.csv",
        test_eval["branch_rows"],
        ["branch", "count", "fraction_of_split", "stress_component_mae", "principal_abs_p95", "repo_relative_p95"],
    )
    _plot_branch_principal_parity(exp4_dir / "validation_parity_by_branch.png", val_eval, split_name="val")
    _plot_branch_latent_parity(exp4_dir / "validation_latent_parity_by_branch.png", val_eval, split_name="val")
    _plot_branch_error_cdfs(exp4_dir / "validation_error_cdfs_by_branch.png", val_eval, split_name="val")
    _plot_broad_vs_hard_summary(exp4_dir / "validation_broad_vs_hard_summary.png", val_eval["metrics"])

    comparison_lines = [
        "## Program X Validation Comparison",
        "",
        f"- Program X broad / hard plastic MAE: `{val_eval['metrics']['broad_plastic_mae']:.6f}` / `{val_eval['metrics']['hard_plastic_mae']:.6f}`",
        f"- Program X hard p95 principal / yield p95: `{val_eval['metrics']['hard_p95_principal']:.6f}` / `{val_eval['metrics']['yield_violation_p95']:.6e}`",
        f"- packet2 broad / hard / p95: `{PACKET2_REFERENCE_VAL['broad_plastic_mae']:.6f}` / `{PACKET2_REFERENCE_VAL['hard_plastic_mae']:.6f}` / `{PACKET2_REFERENCE_VAL['hard_p95_principal']:.6f}`",
        f"- packet3 oracle broad / hard / p95: `{PACKET3_REFERENCE_VAL_ORACLE['broad_plastic_mae']:.6f}` / `{PACKET3_REFERENCE_VAL_ORACLE['hard_plastic_mae']:.6f}` / `{PACKET3_REFERENCE_VAL_ORACLE['hard_p95_principal']:.6f}`",
        f"- packet4 broad / hard / p95: `{PACKET4_REFERENCE_VAL['broad_plastic_mae']:.6f}` / `{PACKET4_REFERENCE_VAL['hard_plastic_mae']:.6f}` / `{PACKET4_REFERENCE_VAL['hard_p95_principal']:.6f}`",
        f"- decision: `{decision_summary['decision']}`",
    ]
    _write_phase_report(exp4_dir / "programx_vs_packet2_packet3_packet4_val.md", "Program X vs Packet2 / Packet3 / Packet4 20260325", comparison_lines)

    write_stage0_report(
        docs_root / "nn_replacement_exact_latent_stage0_20260325.md",
        dataset_summary=dataset_summary,
        exactness_summary=exactness_summary,
    )
    write_oracle_report(
        docs_root / "nn_replacement_exact_latent_oracle_20260325.md",
        val_eval=val_eval,
        decision_summary=decision_summary,
    )
    write_execution_report(
        docs_root / "nn_replacement_exact_latent_execution_20260325.md",
        baseline_summary=baseline_summary,
        exactness_summary=exactness_summary,
        val_eval=val_eval,
        test_eval=test_eval,
        decision_summary=decision_summary,
    )


if __name__ == "__main__":
    main()
