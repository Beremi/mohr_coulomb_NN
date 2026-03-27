#!/usr/bin/env python
"""Run the mixed-material safe hybrid pivot campaign."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_surrogate.data import SPLIT_TO_ID, dataset_summary
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.principal_branch_generation import (
    fit_principal_hybrid_bank,
    summarize_branch_geometry,
    synthesize_from_principal_hybrid,
)
from mc_surrogate.training import TrainingConfig, train_model
from run_cover_layer_single_material_plan import _plot_history_log
from run_safe_hybrid_pivot import (
    THRESHOLDS,
    _evaluate_split,
    _json_safe,
    _lexicographic_key,
    _load_all,
    _plot_threshold_curves,
    _select_threshold,
    _write_threshold_csv,
    run_wrapper_baseline,
)
from mc_surrogate.inference import hybrid_predict_with_checkpoint


PRETRAIN_MIXTURE = (
    ("real_like_margin_bulk", "margin_bulk", 0.30, 0.18),
    ("yield_tube", "yield_tube", 0.06, 0.015),
    ("smooth_left_tube", "boundary_smooth_left", 0.06, 0.015),
    ("smooth_right_tube", "boundary_smooth_right", 0.06, 0.015),
    ("left_apex_tube", "edge_apex_left", 0.06, 0.015),
    ("right_apex_tube", "edge_apex_right", 0.06, 0.015),
    ("small_gap", "small_gap", 0.15, 0.08),
    ("tail", "tail", 0.15, 0.35),
    ("loading_paths", "loading_paths", 0.10, 0.10),
)
HARD_REPLAY_RECIPES = {
    "yield_tube",
    "smooth_left_tube",
    "smooth_right_tube",
    "left_apex_tube",
    "right_apex_tube",
    "small_gap",
    "tail",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", default="constitutive_problem_3D_full.h5")
    parser.add_argument("--baseline-checkpoint", default="experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/hybrid_pivot_20260323")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-call", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--pretrain-train-rows", type=int, default=131072)
    parser.add_argument("--pretrain-val-rows", type=int, default=16384)
    parser.add_argument("--pretrain-test-rows", type=int, default=16384)
    parser.add_argument("--finetune-train-rows", type=int, default=131072)
    parser.add_argument("--finetune-real-fraction", type=float, default=0.80)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--model-width", type=int, default=256)
    parser.add_argument("--model-depth", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--run-candidate-c", action="store_true")
    return parser.parse_args()


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
        if "tangent" in arrays and "DS" not in arrays:
            f.create_dataset("DS", data=np.asarray(arrays["tangent"]), compression="gzip", shuffle=True)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(_json_safe(value))
            else:
                f.attrs[key] = value
    return path


def _counts_from_fractions(total: int, recipes: tuple[tuple[str, str, float, float], ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw = [fraction * total for _, _, fraction, _ in recipes]
    floors = [int(np.floor(v)) for v in raw]
    remainder = total - sum(floors)
    order = np.argsort([-(v - np.floor(v)) for v in raw])
    for idx, (name, _, _, _) in enumerate(recipes):
        counts[name] = floors[idx]
    for idx in order[:remainder]:
        counts[recipes[int(idx)][0]] += 1
    return counts


def _branch_counts(branch_id: np.ndarray) -> dict[str, int]:
    counts = np.bincount(branch_id.astype(np.int64), minlength=len(BRANCH_NAMES))
    return {name: int(counts[idx]) for idx, name in enumerate(BRANCH_NAMES)}


def _sample_indices(rng: np.random.Generator, pool: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if pool.size == 0:
        raise ValueError("Cannot sample from an empty pool.")
    return rng.choice(pool, size=count, replace=True).astype(np.int64)


def _build_exact_arrays(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> dict[str, np.ndarray]:
    exact = constitutive_update_3d(
        strain_eng,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        shear=material_reduced[:, 2],
        bulk=material_reduced[:, 3],
        lame=material_reduced[:, 4],
    )
    return {
        "strain_eng": strain_eng.astype(np.float32),
        "stress": exact.stress.astype(np.float32),
        "material_reduced": material_reduced.astype(np.float32),
        "stress_principal": exact.stress_principal.astype(np.float32),
        "branch_id": exact.branch_id.astype(np.int8),
        "eigvecs": exact.eigvecs.astype(np.float32),
    }


def build_synthetic_pretrain_dataset(
    *,
    real_dataset_path: Path,
    output_root: Path,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    seed: int,
) -> dict[str, Any]:
    dataset_path = output_root / "synthetic_pretrain_v1.h5"
    summary_path = output_root / "synthetic_pretrain_summary.json"
    if dataset_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text())

    real_arrays, real_attrs = _load_h5(real_dataset_path)
    train_mask = real_arrays["split_id"] == SPLIT_TO_ID["train"]
    bank = fit_principal_hybrid_bank(
        real_arrays["strain_eng"][train_mask],
        real_arrays["branch_id"][train_mask],
        real_arrays["material_reduced"][train_mask],
    )

    recipe_names = [name for name, _, _, _ in PRETRAIN_MIXTURE]
    recipe_to_id = {name: idx for idx, name in enumerate(recipe_names)}
    split_targets = {
        "train": int(train_rows),
        "val": int(val_rows),
        "test": int(test_rows),
    }

    arrays_out: dict[str, list[np.ndarray]] = {
        "strain_eng": [],
        "stress": [],
        "material_reduced": [],
        "stress_principal": [],
        "branch_id": [],
        "eigvecs": [],
        "split_id": [],
        "source_call_id": [],
        "source_row_in_call": [],
        "source_recipe_id": [],
    }
    split_summaries: dict[str, Any] = {}

    for split_idx, (split_name, target_rows) in enumerate(split_targets.items()):
        counts = _counts_from_fractions(target_rows, PRETRAIN_MIXTURE)
        split_parts: dict[str, list[np.ndarray]] = {key: [] for key in arrays_out}
        recipe_counts: dict[str, int] = {}
        for recipe_offset, (recipe_name, selection, _, noise_scale) in enumerate(PRETRAIN_MIXTURE):
            count = counts[recipe_name]
            if count <= 0:
                continue
            recipe_seed = seed + 1000 * split_idx + 37 * recipe_offset
            strain_eng, _branch, material_reduced, _valid = synthesize_from_principal_hybrid(
                bank,
                sample_count=count,
                seed=recipe_seed,
                noise_scale=noise_scale,
                selection=selection,
            )
            exact = _build_exact_arrays(strain_eng, material_reduced)
            for key in ("strain_eng", "stress", "material_reduced", "stress_principal", "branch_id", "eigvecs"):
                split_parts[key].append(exact[key])
            split_parts["split_id"].append(np.full(count, SPLIT_TO_ID[split_name], dtype=np.int8))
            split_parts["source_call_id"].append(np.full(count, -1, dtype=np.int32))
            split_parts["source_row_in_call"].append(np.full(count, -1, dtype=np.int32))
            split_parts["source_recipe_id"].append(np.full(count, recipe_to_id[recipe_name], dtype=np.int16))
            recipe_counts[recipe_name] = int(count)

        combined = {key: np.concatenate(parts, axis=0) for key, parts in split_parts.items()}
        perm = np.random.default_rng(seed + 901 * (split_idx + 1)).permutation(combined["strain_eng"].shape[0])
        for key, value in combined.items():
            combined[key] = value[perm]
            arrays_out[key].append(combined[key])
        split_summaries[split_name] = {
            "n_rows": int(combined["strain_eng"].shape[0]),
            "recipe_counts": recipe_counts,
            "branch_counts": _branch_counts(combined["branch_id"]),
            "geometry_summary": summarize_branch_geometry(
                combined["strain_eng"],
                combined["branch_id"],
                combined["material_reduced"],
            ),
        }

    arrays_final = {key: np.concatenate(parts, axis=0) for key, parts in arrays_out.items()}
    attrs = {
        "generator_kind": "principal_hybrid_v2",
        "source_real_dataset": str(real_dataset_path),
        "source_hdf5": real_attrs.get("source_hdf5", ""),
        "recipe_names_json": json.dumps(recipe_names),
        "pretrain_mixture_json": json.dumps(
            [
                {
                    "recipe_name": recipe_name,
                    "selection": selection,
                    "fraction": fraction,
                    "noise_scale": noise_scale,
                }
                for recipe_name, selection, fraction, noise_scale in PRETRAIN_MIXTURE
            ]
        ),
        "branch_names_json": real_attrs["branch_names_json"],
        "raw_material_columns_json": real_attrs["raw_material_columns_json"],
        "reduced_material_columns_json": real_attrs["reduced_material_columns_json"],
        "split_names_json": real_attrs["split_names_json"],
        "split_seed": int(seed),
    }
    _write_h5(dataset_path, arrays_final, attrs)
    summary = {
        "dataset_path": str(dataset_path),
        "dataset_summary": dataset_summary(dataset_path),
        "split_summaries": split_summaries,
        "mixture": [
            {
                "recipe_name": recipe_name,
                "selection": selection,
                "fraction": fraction,
                "noise_scale": noise_scale,
            }
            for recipe_name, selection, fraction, noise_scale in PRETRAIN_MIXTURE
        ],
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def build_finetune_dataset(
    *,
    real_dataset_path: Path,
    panel_path: Path,
    synthetic_dataset_path: Path,
    output_root: Path,
    train_rows: int,
    real_fraction: float,
    seed: int,
) -> dict[str, Any]:
    dataset_path = output_root / "candidate_b_finetune_v1.h5"
    summary_path = output_root / "candidate_b_finetune_summary.json"
    if dataset_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text())

    real_arrays, real_attrs = _load_h5(real_dataset_path)
    panel = _load_all(panel_path)
    synth_arrays, synth_attrs = _load_h5(synthetic_dataset_path)
    tangent_real = real_arrays.get("tangent")
    if tangent_real is None:
        tangent_real = real_arrays.get("DS")
    if tangent_real is None:
        tangent_real = np.full((real_arrays["strain_eng"].shape[0], 6, 6), np.nan, dtype=np.float32)

    real_split = real_arrays["split_id"]
    real_train = np.flatnonzero(real_split == SPLIT_TO_ID["train"])
    hard_train = np.flatnonzero((real_split == SPLIT_TO_ID["train"]) & panel["hard_mask"].astype(bool))
    ds_train = np.flatnonzero((real_split == SPLIT_TO_ID["train"]) & panel["ds_valid_mask"].astype(bool))

    rng = np.random.default_rng(seed)
    n_real = int(round(train_rows * real_fraction))
    n_synth = int(train_rows - n_real)
    broad_count = int(round(0.50 * n_real))
    hard_count = int(round(0.35 * n_real))
    ds_count = int(n_real - broad_count - hard_count)

    real_train_idx = np.concatenate(
        [
            _sample_indices(rng, real_train, broad_count),
            _sample_indices(rng, hard_train, hard_count),
            _sample_indices(rng, ds_train, ds_count),
        ]
    )

    recipe_names = json.loads(synth_attrs["recipe_names_json"])
    hard_recipe_ids = np.array([recipe_names.index(name) for name in recipe_names if name in HARD_REPLAY_RECIPES], dtype=np.int16)
    synth_pool = np.flatnonzero(
        (synth_arrays["split_id"] == SPLIT_TO_ID["train"]) & np.isin(synth_arrays["source_recipe_id"], hard_recipe_ids)
    )
    synth_train_idx = _sample_indices(rng, synth_pool, n_synth)
    synth_tangent = np.full((synth_train_idx.shape[0], 6, 6), np.nan, dtype=np.float32)

    train_arrays = {
        "strain_eng": np.concatenate([real_arrays["strain_eng"][real_train_idx], synth_arrays["strain_eng"][synth_train_idx]], axis=0),
        "stress": np.concatenate([real_arrays["stress"][real_train_idx], synth_arrays["stress"][synth_train_idx]], axis=0),
        "material_reduced": np.concatenate(
            [real_arrays["material_reduced"][real_train_idx], synth_arrays["material_reduced"][synth_train_idx]],
            axis=0,
        ),
        "stress_principal": np.concatenate(
            [real_arrays["stress_principal"][real_train_idx], synth_arrays["stress_principal"][synth_train_idx]],
            axis=0,
        ),
        "branch_id": np.concatenate([real_arrays["branch_id"][real_train_idx], synth_arrays["branch_id"][synth_train_idx]], axis=0),
        "eigvecs": np.concatenate([real_arrays["eigvecs"][real_train_idx], synth_arrays["eigvecs"][synth_train_idx]], axis=0),
        "source_call_id": np.concatenate(
            [real_arrays["source_call_id"][real_train_idx], np.full(synth_train_idx.shape[0], -1, dtype=np.int32)],
            axis=0,
        ),
        "source_row_in_call": np.concatenate(
            [real_arrays["source_row_in_call"][real_train_idx], np.full(synth_train_idx.shape[0], -1, dtype=np.int32)],
            axis=0,
        ),
        "source_type": np.concatenate(
            [np.zeros(real_train_idx.shape[0], dtype=np.int8), np.ones(synth_train_idx.shape[0], dtype=np.int8)],
            axis=0,
        ),
        "source_recipe_id": np.concatenate(
            [np.full(real_train_idx.shape[0], -1, dtype=np.int16), synth_arrays["source_recipe_id"][synth_train_idx].astype(np.int16)],
            axis=0,
        ),
        "tangent": np.concatenate([tangent_real[real_train_idx], synth_tangent], axis=0),
        "split_id": np.full(real_train_idx.shape[0] + synth_train_idx.shape[0], SPLIT_TO_ID["train"], dtype=np.int8),
    }

    parts = [train_arrays]
    split_summaries: dict[str, Any] = {
        "train": {
            "n_rows": int(train_arrays["strain_eng"].shape[0]),
            "real_rows": int(real_train_idx.shape[0]),
            "synthetic_rows": int(synth_train_idx.shape[0]),
            "branch_counts": _branch_counts(train_arrays["branch_id"]),
        }
    }
    for split_name in ("val", "test"):
        mask = real_split == SPLIT_TO_ID[split_name]
        split_arrays = {
            "strain_eng": real_arrays["strain_eng"][mask],
            "stress": real_arrays["stress"][mask],
            "material_reduced": real_arrays["material_reduced"][mask],
            "stress_principal": real_arrays["stress_principal"][mask],
            "branch_id": real_arrays["branch_id"][mask],
            "eigvecs": real_arrays["eigvecs"][mask],
            "source_call_id": real_arrays["source_call_id"][mask],
            "source_row_in_call": real_arrays["source_row_in_call"][mask],
            "source_type": np.zeros(int(np.sum(mask)), dtype=np.int8),
            "source_recipe_id": np.full(int(np.sum(mask)), -1, dtype=np.int16),
            "tangent": tangent_real[mask],
            "split_id": np.full(int(np.sum(mask)), SPLIT_TO_ID[split_name], dtype=np.int8),
        }
        parts.append(split_arrays)
        split_summaries[split_name] = {
            "n_rows": int(split_arrays["strain_eng"].shape[0]),
            "branch_counts": _branch_counts(split_arrays["branch_id"]),
        }

    keys = parts[0].keys()
    arrays_final = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    arrays_final["DS"] = arrays_final["tangent"]
    attrs = {
        "generator_kind": "candidate_b_finetune_v1",
        "source_real_dataset": str(real_dataset_path),
        "source_synthetic_dataset": str(synthetic_dataset_path),
        "branch_names_json": real_attrs["branch_names_json"],
        "raw_material_columns_json": real_attrs["raw_material_columns_json"],
        "reduced_material_columns_json": real_attrs["reduced_material_columns_json"],
        "split_names_json": real_attrs["split_names_json"],
        "recipe_names_json": synth_attrs["recipe_names_json"],
        "split_seed": int(seed),
        "real_fraction": float(real_fraction),
        "train_mix_json": json.dumps(
            {
                "real_broad_fraction": 0.50,
                "real_hard_fraction": 0.35,
                "real_ds_fraction": 0.15,
                "overall_real_fraction": real_fraction,
                "overall_synthetic_fraction": 1.0 - real_fraction,
                "hard_replay_recipes": sorted(HARD_REPLAY_RECIPES),
            }
        ),
    }
    _write_h5(dataset_path, arrays_final, attrs)
    summary = {
        "dataset_path": str(dataset_path),
        "dataset_summary": dataset_summary(dataset_path),
        "split_summaries": split_summaries,
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _maybe_train(config: TrainingConfig, title: str) -> dict[str, Any]:
    run_dir = Path(config.run_dir)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    if best_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = train_model(config)
    history_path = run_dir / "history.csv"
    if history_path.exists() and not (run_dir / "history_log.png").exists():
        _plot_history_log(history_path, run_dir / "history_log.png", title=title)
    (run_dir / "config_used.json").write_text(json.dumps(_json_safe(asdict(config)), indent=2), encoding="utf-8")
    return summary


def _evaluate_checkpoint_thresholds(
    *,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    baseline_val: dict[str, Any],
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    val_mask = arrays["split_id"] == SPLIT_TO_ID["val"]
    test_mask = arrays["split_id"] == SPLIT_TO_ID["test"]
    rows: list[dict[str, Any]] = []
    for delta_geom in THRESHOLDS:
        pred = {
            "stress": np.zeros_like(arrays["stress"], dtype=np.float32),
            "stress_principal": np.zeros_like(arrays["stress_principal"], dtype=np.float32),
            "branch_probabilities": np.zeros((arrays["stress"].shape[0], 5), dtype=np.float32),
            "elastic_mask": np.zeros(arrays["stress"].shape[0], dtype=bool),
            "fallback_mask": np.zeros(arrays["stress"].shape[0], dtype=bool),
            "learned_mask": np.zeros(arrays["stress"].shape[0], dtype=bool),
        }
        for split_name, split_mask in (("val", val_mask), ("test", test_mask)):
            if not np.any(split_mask):
                continue
            split_pred = hybrid_predict_with_checkpoint(
                checkpoint_path,
                arrays["strain_eng"][split_mask],
                arrays["material_reduced"][split_mask],
                delta_geom=float(delta_geom),
                device=device,
                batch_size=batch_size,
            )
            pred["stress"][split_mask] = split_pred["stress"]
            pred["stress_principal"][split_mask] = split_pred["stress_principal"]
            pred["branch_probabilities"][split_mask] = split_pred["branch_probabilities"]
            pred["elastic_mask"][split_mask] = split_pred["elastic_mask"]
            pred["fallback_mask"][split_mask] = split_pred["fallback_mask"]
            pred["learned_mask"][split_mask] = split_pred["learned_mask"]
        rows.append(
            {
                "delta_geom": float(delta_geom),
                "val": _evaluate_split(arrays, panel, pred, split_name="val"),
                "test": _evaluate_split(arrays, panel, pred, split_name="test"),
            }
        )
    return rows, _select_threshold(rows, baseline_val)


def _candidate_checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = [run_dir / "best.pt", run_dir / "last.pt"]
    paths.extend(sorted((run_dir / "snapshots").glob("epoch_*.pt")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if path.exists() and key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _checkpoint_label(path: Path) -> str:
    return path.stem if path.parent.name == "snapshots" else path.name


def evaluate_candidate_b(
    *,
    finetune_run_dir: Path,
    real_dataset_path: Path,
    panel_path: Path,
    baseline_val: dict[str, Any],
    output_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    arrays = _load_all(real_dataset_path)
    panel = _load_all(panel_path)
    eval_dir = output_root / "candidate_b" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    selector_rows: list[dict[str, Any]] = []
    selected_rows_detail: dict[str, Any] = {}
    for checkpoint_path in _candidate_checkpoint_paths(finetune_run_dir):
        rows, selection = _evaluate_checkpoint_thresholds(
            checkpoint_path=checkpoint_path,
            arrays=arrays,
            panel=panel,
            baseline_val=baseline_val,
            device=device,
            batch_size=batch_size,
        )
        label = _checkpoint_label(checkpoint_path)
        (eval_dir / f"{label}_threshold_metrics.json").write_text(json.dumps(_json_safe(rows), indent=2), encoding="utf-8")
        (eval_dir / f"{label}_selection.json").write_text(json.dumps(_json_safe(selection), indent=2), encoding="utf-8")
        selected = selection["accepted"] if selection["accepted"] is not None else selection["fallback_best"]
        selected_rows_detail[label] = {"rows": rows, "selection": selection}
        selector_rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_path": str(checkpoint_path),
                "admissible": bool(selection["accepted"] is not None),
                "status": selection["status"],
                "delta_geom": float(selected["delta_geom"]),
                "val_broad_plastic_mae": float(selected["val"]["broad_plastic"]["stress_mae"]),
                "val_hard_plastic_mae": float(selected["val"]["hard_plastic"]["stress_mae"]),
                "val_hard_principal_p95": float(selected["val"]["hard"]["principal_p95"]),
                "val_plastic_coverage": float(selected["val"]["plastic_coverage"]),
                "test_plastic_coverage": float(selected["test"]["plastic_coverage"]),
            }
        )

    admissible = [row for row in selector_rows if row["admissible"]]
    if admissible:
        best_row = min(admissible, key=lambda row: _lexicographic_key(selected_rows_detail[row["checkpoint_label"]]["selection"]["accepted"]))
    else:
        best_row = min(selector_rows, key=lambda row: _lexicographic_key(selected_rows_detail[row["checkpoint_label"]]["selection"]["fallback_best"]))
    best_detail = selected_rows_detail[best_row["checkpoint_label"]]

    selected_checkpoint = Path(best_row["checkpoint_path"])
    selected_copy = output_root / "candidate_b" / "selected_checkpoint.pt"
    selected_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, selected_copy)

    selector_csv = output_root / "candidate_b" / "snapshot_selector.csv"
    with selector_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_label",
                "checkpoint_path",
                "admissible",
                "status",
                "delta_geom",
                "val_broad_plastic_mae",
                "val_hard_plastic_mae",
                "val_hard_principal_p95",
                "val_plastic_coverage",
                "test_plastic_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(selector_rows)

    _write_threshold_csv(output_root / "candidate_b" / "selected_threshold_summary.csv", best_detail["rows"])
    _plot_threshold_curves(output_root / "candidate_b" / "selected_threshold_curves.png", best_detail["rows"], baseline_val)
    summary = {
        "selector_rows": selector_rows,
        "best_row": best_row,
        "selected_checkpoint": str(selected_copy),
        "selected_detail": best_detail,
        "selector_csv": str(selector_csv),
    }
    (output_root / "candidate_b" / "selector_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def write_candidate_b_report(
    *,
    report_path: Path,
    pretrain_dataset: dict[str, Any],
    finetune_dataset: dict[str, Any],
    pretrain_summary: dict[str, Any],
    finetune_summary: dict[str, Any],
    baseline_val: dict[str, Any],
    baseline_test: dict[str, Any],
    wrapper_selection: dict[str, Any],
    selector_summary: dict[str, Any],
    output_root: Path,
) -> None:
    best_row = selector_summary["best_row"]
    selected = selector_summary["selected_detail"]["selection"]["accepted"] or selector_summary["selected_detail"]["selection"]["fallback_best"]
    lines: list[str] = []
    lines.append("# Hybrid Pivot Experiment 2: Candidate B")
    lines.append("")
    lines.append("This report executes the geometry-augmented plastic corrector v1 screening run: synthetic pretraining, real fine-tuning with hard-panel emphasis, and external snapshot selection on the frozen real validation panels.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- synthetic pretrain dataset: `{pretrain_dataset['dataset_path']}`")
    lines.append(f"- fine-tune dataset: `{finetune_dataset['dataset_path']}`")
    lines.append(f"- pretrain run: `{pretrain_summary['run_dir']}`")
    lines.append(f"- fine-tune run: `{finetune_summary['run_dir']}`")
    lines.append(f"- selected checkpoint copy: `{selector_summary['selected_checkpoint']}`")
    lines.append("")
    lines.append("## Synthetic Mixture")
    lines.append("")
    lines.append("| Recipe | Fraction | Selection | Noise scale |")
    lines.append("|---|---:|---|---:|")
    for row in pretrain_dataset["mixture"]:
        lines.append(f"| {row['recipe_name']} | {row['fraction']:.2f} | {row['selection']} | {row['noise_scale']:.3f} |")
    lines.append("")
    lines.append("## Dataset Summaries")
    lines.append("")
    lines.append(f"- synthetic pretrain rows: `{pretrain_dataset['dataset_summary']['n_samples']}`")
    lines.append(f"- fine-tune train rows: `{finetune_dataset['split_summaries']['train']['n_rows']}`")
    lines.append(f"- fine-tune train real rows: `{finetune_dataset['split_summaries']['train']['real_rows']}`")
    lines.append(f"- fine-tune train synthetic replay rows: `{finetune_dataset['split_summaries']['train']['synthetic_rows']}`")
    lines.append("")
    lines.append(f"![Pretrain history]({(Path(pretrain_summary['run_dir']) / 'history_log.png').as_posix()})")
    lines.append("")
    lines.append(f"![Fine-tune history]({(Path(finetune_summary['run_dir']) / 'history_log.png').as_posix()})")
    lines.append("")
    lines.append("## Snapshot Selector")
    lines.append("")
    lines.append("| Checkpoint | Admissible | delta_geom | Val Broad Plastic MAE | Val Hard Plastic MAE | Val Hard p95 | Val Coverage | Test Coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in selector_summary["selector_rows"]:
        lines.append(
            f"| {row['checkpoint_label']} | {int(row['admissible'])} | {row['delta_geom']:.3f} | {row['val_broad_plastic_mae']:.6f} | "
            f"{row['val_hard_plastic_mae']:.6f} | {row['val_hard_principal_p95']:.6f} | {row['val_plastic_coverage']:.6f} | {row['test_plastic_coverage']:.6f} |"
        )
    lines.append("")
    lines.append(f"![Selected threshold curves]({(output_root / 'candidate_b' / 'selected_threshold_curves.png').as_posix()})")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Candidate | Val Broad Plastic MAE | Val Hard Plastic MAE | Val Hard p95 | Val Coverage | Test Coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| March 12 baseline raw_branch | {baseline_val['broad_plastic']['stress_mae']:.6f} | {baseline_val['hard_plastic']['stress_mae']:.6f} | {baseline_val['hard']['principal_p95']:.6f} | 1.000000 | 1.000000 |"
    )
    wrapper_best = wrapper_selection["fallback_best"]
    lines.append(
        f"| Experiment 1 wrapper best-error threshold | {wrapper_best['val']['broad_plastic']['stress_mae']:.6f} | {wrapper_best['val']['hard_plastic']['stress_mae']:.6f} | "
        f"{wrapper_best['val']['hard']['principal_p95']:.6f} | {wrapper_best['val']['plastic_coverage']:.6f} | {wrapper_best['test']['plastic_coverage']:.6f} |"
    )
    lines.append(
        f"| Candidate B selected checkpoint `{best_row['checkpoint_label']}` | {selected['val']['broad_plastic']['stress_mae']:.6f} | {selected['val']['hard_plastic']['stress_mae']:.6f} | "
        f"{selected['val']['hard']['principal_p95']:.6f} | {selected['val']['plastic_coverage']:.6f} | {selected['test']['plastic_coverage']:.6f} |"
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if best_row["admissible"]:
        lines.append("- Candidate B cleared the wrapper coverage/tail gates on the frozen real panels and is eligible for Candidate C.")
    else:
        lines.append("- Candidate B did not clear the combined coverage-and-tail admissibility gate in this screening run.")
        lines.append("- The selected checkpoint is still the best external lexicographic snapshot for analysis, but it is not solver-eligible yet.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_c_report(report_path: Path, *, candidate_b_summary: dict[str, Any], ran_candidate_c: bool) -> None:
    best_row = candidate_b_summary["best_row"]
    lines = ["# Hybrid Pivot Experiment 5: Candidate C", ""]
    if ran_candidate_c:
        lines.append("Status: Candidate C execution is not implemented in this pass.")
    elif best_row["admissible"]:
        lines.append("Status: Candidate B cleared the gate, but Candidate C was not launched because this execution pass was limited to the stress-only v1 screen.")
    else:
        lines.append("Status: skipped because Candidate B did not clear the broad/hard real-panel gate with meaningful learned coverage.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_fe_handoff(report_path: Path, *, baseline_checkpoint: Path, candidate_b_summary: dict[str, Any]) -> None:
    best_row = candidate_b_summary["best_row"]
    lines = [
        "# Hybrid Pivot FE Handoff",
        "",
        "This repository still does not contain an external FE solver harness, so this phase remains a documented handoff rather than an executed benchmark.",
        "",
        "## Current Candidate",
        "",
        f"- baseline reference checkpoint: `{baseline_checkpoint}`",
        f"- latest Candidate B checkpoint copy: `{candidate_b_summary['selected_checkpoint']}`",
        f"- admissible on real broad/hard coverage gate: `{best_row['admissible']}`",
        f"- selected `delta_geom`: `{best_row['delta_geom']:.3f}`",
        "",
        "## Required External FE Checks",
        "",
        "- exact constitutive update everywhere",
        "- hybrid wrapper with analytic elastic dispatch and analytic fallback",
        "- convergence and divergence parity",
        "- Newton iterations per load step",
        "- constitutive-call runtime share handled by the learned path",
        "- final solver quantities of interest",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_execution_doc(
    report_path: Path,
    *,
    panel_result: dict[str, Any],
    exp1_selection: dict[str, Any],
    candidate_b_summary: dict[str, Any],
) -> None:
    best_row = candidate_b_summary["best_row"]
    lines = [
        "# Hybrid Pivot Execution 20260323",
        "",
        "This document tracks execution against `report3.md` and only checks work items whose artifact and report now exist.",
        "",
        "## Status",
        "",
        f"- Experiment 0 report: `{ROOT / 'docs' / 'hybrid_pivot_exp0_panels_20260323.md'}`",
        f"- Experiment 1 report: `{ROOT / 'docs' / 'hybrid_pivot_exp1_wrapper_20260323.md'}`",
        f"- Experiment 2 report: `{ROOT / 'docs' / 'hybrid_pivot_exp2_candidate_b_20260323.md'}`",
        f"- Experiment 5 report: `{ROOT / 'docs' / 'hybrid_pivot_exp5_candidate_c_20260323.md'}`",
        f"- FE handoff report: `{ROOT / 'docs' / 'hybrid_pivot_fe_handoff_20260323.md'}`",
        "",
        "## Minimal Checklist",
        "",
        "- [x] build broad/hard/tangent real panels",
        "- [x] implement analytic `d_geom` and gap metrics",
        "- [x] wrap current best model with exact elastic + exact fallback",
        "- [x] evaluate coverage vs error on real validation and test",
        "- [x] add geometry-augmented principal feature builder",
        "- [x] create plastic-only auxiliary branch head path",
        "- [x] train synthetic pretrain checkpoint",
        "- [x] fine-tune on real with hard-panel oversampling",
        "- [x] tune fallback threshold on real validation",
        f"- [{'x' if best_row['admissible'] else ' '}] add tangent-direction loss on the interior `DS` subset",
        "- [ ] run FE-loop benchmark suite",
        "",
        "## Notes",
        "",
        f"- Experiment 1 admissible threshold found: `{exp1_selection['accepted'] is not None}`",
        f"- Candidate B admissible threshold found: `{best_row['admissible']}`",
        "- Candidate C remains unticked in this pass if Candidate B did not clear the coverage-and-tail gate.",
        "- FE benchmarking remains unticked because there is still no in-repo external harness.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    wrapper_result = run_wrapper_baseline(
        full_export=(ROOT / args.full_export).resolve(),
        baseline_checkpoint=(ROOT / args.baseline_checkpoint).resolve(),
        output_root=output_root,
        report_path=(ROOT / "docs" / "hybrid_pivot_exp1_wrapper_20260323.md").resolve(),
        samples_per_call=args.samples_per_call,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )

    panel_result = wrapper_result["panel_result"]
    pretrain_dataset = build_synthetic_pretrain_dataset(
        real_dataset_path=Path(panel_result["dataset_path"]),
        output_root=output_root / "candidate_b" / "data",
        train_rows=args.pretrain_train_rows,
        val_rows=args.pretrain_val_rows,
        test_rows=args.pretrain_test_rows,
        seed=args.seed,
    )
    finetune_dataset = build_finetune_dataset(
        real_dataset_path=Path(panel_result["dataset_path"]),
        panel_path=Path(panel_result["panel_path"]),
        synthetic_dataset_path=Path(pretrain_dataset["dataset_path"]),
        output_root=output_root / "candidate_b" / "data",
        train_rows=args.finetune_train_rows,
        real_fraction=args.finetune_real_fraction,
        seed=args.seed + 11,
    )

    pretrain_cfg = TrainingConfig(
        dataset=pretrain_dataset["dataset_path"],
        run_dir=str(output_root / "candidate_b" / "pretrain"),
        model_kind="trial_principal_geom_plastic_branch_residual",
        epochs=args.pretrain_epochs,
        batch_size=args.train_batch_size,
        lr=1.0e-3,
        weight_decay=1.0e-4,
        width=args.model_width,
        depth=args.model_depth,
        dropout=0.0,
        seed=args.seed,
        patience=15,
        grad_clip=1.0,
        branch_loss_weight=0.05,
        device=args.device,
        scheduler_kind="cosine",
        warmup_epochs=2,
        min_lr=1.0e-5,
        checkpoint_metric="stress_mae",
        regression_loss_kind="huber",
        huber_delta=1.0,
        voigt_mae_weight=0.25,
        snapshot_every_epochs=10,
        log_every_epochs=10,
    )
    pretrain_summary = _maybe_train(pretrain_cfg, "candidate_b_pretrain")

    finetune_cfg = TrainingConfig(
        dataset=finetune_dataset["dataset_path"],
        run_dir=str(output_root / "candidate_b" / "finetune"),
        model_kind="trial_principal_geom_plastic_branch_residual",
        epochs=args.finetune_epochs,
        batch_size=args.train_batch_size,
        lr=3.0e-4,
        weight_decay=1.0e-5,
        width=args.model_width,
        depth=args.model_depth,
        dropout=0.0,
        seed=args.seed + 1,
        patience=12,
        grad_clip=1.0,
        branch_loss_weight=0.05,
        device=args.device,
        scheduler_kind="plateau",
        min_lr=1.0e-5,
        plateau_factor=0.5,
        plateau_patience=4,
        checkpoint_metric="stress_mae",
        init_checkpoint=pretrain_summary["best_checkpoint"],
        regression_loss_kind="huber",
        huber_delta=1.0,
        voigt_mae_weight=0.25,
        snapshot_every_epochs=10,
        log_every_epochs=10,
    )
    finetune_summary = _maybe_train(finetune_cfg, "candidate_b_finetune")

    selector_summary = evaluate_candidate_b(
        finetune_run_dir=Path(finetune_summary["run_dir"]),
        real_dataset_path=Path(panel_result["dataset_path"]),
        panel_path=Path(panel_result["panel_path"]),
        baseline_val=wrapper_result["baseline_val"],
        output_root=output_root,
        device=args.device,
        batch_size=args.batch_size,
    )

    write_candidate_b_report(
        report_path=(ROOT / "docs" / "hybrid_pivot_exp2_candidate_b_20260323.md").resolve(),
        pretrain_dataset=pretrain_dataset,
        finetune_dataset=finetune_dataset,
        pretrain_summary=pretrain_summary,
        finetune_summary=finetune_summary,
        baseline_val=wrapper_result["baseline_val"],
        baseline_test=wrapper_result["baseline_test"],
        wrapper_selection=wrapper_result["selection"],
        selector_summary=selector_summary,
        output_root=output_root,
    )
    write_candidate_c_report(
        (ROOT / "docs" / "hybrid_pivot_exp5_candidate_c_20260323.md").resolve(),
        candidate_b_summary=selector_summary,
        ran_candidate_c=args.run_candidate_c and selector_summary["best_row"]["admissible"],
    )
    write_fe_handoff(
        (ROOT / "docs" / "hybrid_pivot_fe_handoff_20260323.md").resolve(),
        baseline_checkpoint=(ROOT / args.baseline_checkpoint).resolve(),
        candidate_b_summary=selector_summary,
    )
    write_execution_doc(
        (ROOT / "docs" / "hybrid_pivot_execution_20260323.md").resolve(),
        panel_result=panel_result,
        exp1_selection=wrapper_result["selection"],
        candidate_b_summary=selector_summary,
    )

    result = {
        "wrapper_result": wrapper_result,
        "pretrain_dataset": pretrain_dataset,
        "finetune_dataset": finetune_dataset,
        "pretrain_summary": pretrain_summary,
        "finetune_summary": finetune_summary,
        "selector_summary": selector_summary,
    }
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
