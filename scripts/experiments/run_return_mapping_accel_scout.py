#!/usr/bin/env python
"""Run the exact-authority acceleration scout after the hybrid gate redesign cycle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    _build_branch_state_from_principal,
    _build_principal_state_3d,
    _dispatch_from_branch_state,
    constitutive_update_3d,
    profile_constitutive_update_3d,
)
from mc_surrogate.principal_branch_generation import fit_principal_hybrid_bank, synthesize_from_principal_hybrid
from mc_surrogate.training import predict_with_checkpoint


PANEL_MASK_KEYS = {
    "broad_val": "broad_val_mask",
    "broad_test": "broad_test_mask",
    "hard_val": "hard_val_mask",
    "hard_test": "hard_test_mask",
    "ds_valid": "ds_valid_mask",
}
HARD_SYNTH_RECIPES = (
    ("yield_tube", 0.015),
    ("boundary_smooth_left", 0.015),
    ("boundary_smooth_right", 0.015),
    ("edge_apex_left", 0.015),
    ("edge_apex_right", 0.015),
    ("small_gap", 0.08),
    ("tail", 0.35),
)
ENTROPY_BIN_EDGES = (0.0, 0.20, 0.40, 0.60, 0.80, 1.000001)
TOPK_THRESHOLD = 0.98
SHORTLIST_K = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redesign-root", default="experiment_runs/real_sim/hybrid_gate_redesign_20260324")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/return_mapping_accel_20260324")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic-hard-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--shortlist-warmup", type=int, default=2)
    parser.add_argument("--shortlist-repeats", type=int, default=5)
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
    content = [f"# {title}", ""]
    content.extend(lines)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
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


def _branch_counts(branch_id: np.ndarray) -> dict[str, int]:
    counts = np.bincount(branch_id.astype(np.int64), minlength=len(BRANCH_NAMES))
    return {name: int(counts[idx]) for idx, name in enumerate(BRANCH_NAMES)}


def _panel_arrays(
    dataset_arrays: dict[str, np.ndarray],
    panel_arrays: dict[str, np.ndarray],
    panel_name: str,
) -> dict[str, np.ndarray]:
    mask = panel_arrays[PANEL_MASK_KEYS[panel_name]].astype(bool)
    return _slice_arrays(dataset_arrays, mask)


def _counts_from_total(total: int, names: tuple[str, ...]) -> dict[str, int]:
    base = total // len(names)
    rem = total % len(names)
    counts = {}
    for idx, name in enumerate(names):
        counts[name] = base + (1 if idx < rem else 0)
    return counts


def build_synthetic_hard_panel(
    *,
    redesign_root: Path,
    output_root: Path,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    dataset_path = output_root / "synthetic_hard_panel.h5"
    summary_path = output_root / "synthetic_hard_panel_summary.json"
    if dataset_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    real_dataset_path = redesign_root / "real_grouped_sampled_512.h5"
    real_arrays, _ = _load_h5(real_dataset_path)
    train_mask = real_arrays["split_id"] == 0
    bank = fit_principal_hybrid_bank(
        real_arrays["strain_eng"][train_mask],
        real_arrays["branch_id"][train_mask],
        real_arrays["material_reduced"][train_mask],
    )

    recipe_names = tuple(name for name, _ in HARD_SYNTH_RECIPES)
    recipe_to_id = {name: idx for idx, name in enumerate(recipe_names)}
    counts = _counts_from_total(sample_count, recipe_names)

    strain_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    principal_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    recipe_parts: list[np.ndarray] = []

    for recipe_idx, (selection, noise_scale) in enumerate(HARD_SYNTH_RECIPES):
        n_rows = counts[selection]
        if n_rows <= 0:
            continue
        strain_eng, _branch_seed, material_reduced, _valid = synthesize_from_principal_hybrid(
            bank,
            sample_count=n_rows,
            seed=seed + 101 * recipe_idx,
            noise_scale=noise_scale,
            selection=selection,
        )
        exact = constitutive_update_3d(
            strain_eng,
            c_bar=material_reduced[:, 0],
            sin_phi=material_reduced[:, 1],
            shear=material_reduced[:, 2],
            bulk=material_reduced[:, 3],
            lame=material_reduced[:, 4],
        )
        strain_parts.append(strain_eng.astype(np.float32))
        material_parts.append(material_reduced.astype(np.float32))
        stress_parts.append(exact.stress.astype(np.float32))
        principal_parts.append(exact.stress_principal.astype(np.float32))
        branch_parts.append(exact.branch_id.astype(np.int8))
        recipe_parts.append(np.full(n_rows, recipe_to_id[selection], dtype=np.int8))

    arrays = {
        "strain_eng": np.concatenate(strain_parts, axis=0),
        "material_reduced": np.concatenate(material_parts, axis=0),
        "stress": np.concatenate(stress_parts, axis=0),
        "stress_principal": np.concatenate(principal_parts, axis=0),
        "branch_id": np.concatenate(branch_parts, axis=0),
        "source_recipe_id": np.concatenate(recipe_parts, axis=0),
    }
    summary = {
        "dataset_path": str(dataset_path),
        "sample_count": int(arrays["strain_eng"].shape[0]),
        "recipe_counts_requested": counts,
        "recipe_id_to_name": {int(idx): name for name, idx in recipe_to_id.items()},
        "branch_counts": _branch_counts(arrays["branch_id"]),
    }
    _write_h5(dataset_path, arrays, {"summary_json": summary})
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _profile_arrays(
    panel_name: str,
    arrays: dict[str, np.ndarray],
    *,
    warmup: int,
    repeats: int,
    return_tangent: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_rows: list[dict[str, Any]] = []
    profile_ref: dict[str, Any] | None = None
    for run_idx in range(warmup + repeats):
        _result, profile = profile_constitutive_update_3d(
            arrays["strain_eng"],
            c_bar=arrays["material_reduced"][:, 0],
            sin_phi=arrays["material_reduced"][:, 1],
            shear=arrays["material_reduced"][:, 2],
            bulk=arrays["material_reduced"][:, 3],
            lame=arrays["material_reduced"][:, 4],
            return_tangent=return_tangent,
        )
        if profile_ref is None:
            profile_ref = profile
        if run_idx < warmup:
            continue
        explicit_non_eig_s = profile["branch_scalar_s"] + profile["branch_dispatch_s"] + profile["tangent_fd_s"]
        total_s = float(profile["total_s"])
        n_rows = max(int(profile["n_rows"]), 1)
        run_rows.append(
            {
                "panel_name": panel_name,
                "run_idx": int(run_idx - warmup),
                "n_rows": int(profile["n_rows"]),
                "principal_eig_s": float(profile["principal_eig_s"]),
                "branch_scalar_s": float(profile["branch_scalar_s"]),
                "branch_dispatch_s": float(profile["branch_dispatch_s"]),
                "tangent_fd_s": float(profile["tangent_fd_s"]),
                "total_s": total_s,
                "residual_overhead_s": float(profile["residual_overhead_s"]),
                "explicit_non_eig_s": float(explicit_non_eig_s),
                "per_row_total_us": 1.0e6 * total_s / n_rows,
                "per_row_principal_eig_us": 1.0e6 * float(profile["principal_eig_s"]) / n_rows,
                "per_row_explicit_non_eig_us": 1.0e6 * float(explicit_non_eig_s) / n_rows,
                "explicit_non_eig_share": float(explicit_non_eig_s / total_s) if total_s > 0.0 else 0.0,
                "branch_scalar_share": float(profile["branch_scalar_s"] / total_s) if total_s > 0.0 else 0.0,
                "principal_eig_share": float(profile["principal_eig_s"] / total_s) if total_s > 0.0 else 0.0,
                "plastic_fraction": float(profile["plastic_fraction"]),
            }
        )

    if profile_ref is None:
        raise RuntimeError(f"No profiles collected for panel {panel_name}.")

    summary = {
        "panel_name": panel_name,
        "n_rows": int(profile_ref["n_rows"]),
        "branch_counts": profile_ref["branch_counts"],
        "plastic_fraction": float(profile_ref["plastic_fraction"]),
        "median_total_s": _median([float(row["total_s"]) for row in run_rows]),
        "median_principal_eig_s": _median([float(row["principal_eig_s"]) for row in run_rows]),
        "median_branch_scalar_s": _median([float(row["branch_scalar_s"]) for row in run_rows]),
        "median_branch_dispatch_s": _median([float(row["branch_dispatch_s"]) for row in run_rows]),
        "median_tangent_fd_s": _median([float(row["tangent_fd_s"]) for row in run_rows]),
        "median_per_row_total_us": _median([float(row["per_row_total_us"]) for row in run_rows]),
        "median_per_row_principal_eig_us": _median([float(row["per_row_principal_eig_us"]) for row in run_rows]),
        "median_per_row_explicit_non_eig_us": _median([float(row["per_row_explicit_non_eig_us"]) for row in run_rows]),
        "median_explicit_non_eig_share": _median([float(row["explicit_non_eig_share"]) for row in run_rows]),
        "median_branch_scalar_share": _median([float(row["branch_scalar_share"]) for row in run_rows]),
        "median_principal_eig_share": _median([float(row["principal_eig_share"]) for row in run_rows]),
    }
    summary["branch_scalar_is_dominant"] = bool(
        summary["median_branch_scalar_s"] >= summary["median_principal_eig_s"]
        and summary["median_branch_scalar_s"] >= summary["median_branch_dispatch_s"]
    )
    return run_rows, summary


def _normalized_plastic_probabilities(branch_probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(branch_probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D branch probabilities, got {probs.shape}.")
    plastic = probs[:, 1:] if probs.shape[1] >= 5 else probs
    totals = np.sum(plastic, axis=1, keepdims=True)
    return np.divide(plastic, np.maximum(totals, 1.0e-12), out=np.zeros_like(plastic), where=totals > 0.0).astype(np.float32)


def _entropy_from_plastic_probs(plastic_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(plastic_probs, dtype=np.float64)
    entropy = -np.sum(np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1.0e-12)), 0.0), axis=1)
    return np.clip(entropy / np.log(max(probs.shape[1], 2)), 0.0, 1.0).astype(np.float32)


def _branch_hint_artifacts(
    *,
    model_name: str,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    device: str,
) -> dict[str, Any]:
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=device,
        batch_size=32768,
    )
    plastic_mask = arrays["branch_id"] > 0
    true_branch = arrays["branch_id"][plastic_mask].astype(np.int64)
    plastic_probs_full = _normalized_plastic_probabilities(pred["branch_probabilities"])
    plastic_probs = plastic_probs_full[plastic_mask]
    entropy = _entropy_from_plastic_probs(plastic_probs)
    order = np.argsort(-plastic_probs, axis=1)
    top1 = order[:, 0].astype(np.int64) + 1
    top2 = order[:, :SHORTLIST_K].astype(np.int64) + 1
    top1_recall = float(np.mean(top1 == true_branch)) if true_branch.size else 0.0
    top2_recall = float(np.mean(np.any(top2 == true_branch[:, None], axis=1))) if true_branch.size else 0.0

    summary = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "branch_head_kind": pred.get("branch_head_kind", "unknown"),
        "plastic_rows": int(true_branch.shape[0]),
        "top1_recall": top1_recall,
        "top2_recall": top2_recall,
        "missed_true_branch_rate": float(1.0 - top2_recall),
        "mean_entropy": float(np.mean(entropy)) if entropy.size else 1.0,
        "p90_entropy": float(np.quantile(entropy, 0.90)) if entropy.size else 1.0,
        "mean_top1_confidence": float(np.mean(np.max(plastic_probs, axis=1))) if entropy.size else 0.0,
    }

    per_branch_rows: list[dict[str, Any]] = []
    for branch_id in range(1, len(BRANCH_NAMES)):
        mask = true_branch == branch_id
        if not np.any(mask):
            continue
        per_branch_rows.append(
            {
                "model_name": model_name,
                "true_branch": BRANCH_NAMES[branch_id],
                "n_rows": int(np.sum(mask)),
                "top1_recall": float(np.mean(top1[mask] == true_branch[mask])),
                "top2_recall": float(np.mean(np.any(top2[mask] == true_branch[mask, None], axis=1))),
                "mean_entropy": float(np.mean(entropy[mask])),
            }
        )

    confusion_rows: list[dict[str, Any]] = []
    for branch_id in range(1, len(BRANCH_NAMES)):
        for pred_id in range(1, len(BRANCH_NAMES)):
            count = int(np.sum((true_branch == branch_id) & (top1 == pred_id)))
            if count <= 0:
                continue
            confusion_rows.append(
                {
                    "model_name": model_name,
                    "true_branch": BRANCH_NAMES[branch_id],
                    "predicted_branch": BRANCH_NAMES[pred_id],
                    "n_rows": count,
                }
            )

    entropy_rows: list[dict[str, Any]] = []
    for low, high in zip(ENTROPY_BIN_EDGES[:-1], ENTROPY_BIN_EDGES[1:]):
        mask = (entropy >= low) & (entropy < high)
        if not np.any(mask):
            continue
        entropy_rows.append(
            {
                "model_name": model_name,
                "entropy_bin": f"[{low:.2f}, {high:.2f})",
                "n_rows": int(np.sum(mask)),
                "top1_recall": float(np.mean(top1[mask] == true_branch[mask])),
                "top2_recall": float(np.mean(np.any(top2[mask] == true_branch[mask, None], axis=1))),
                "mean_top1_confidence": float(np.mean(np.max(plastic_probs[mask], axis=1))),
            }
        )

    topk_ids_full = np.zeros((arrays["branch_id"].shape[0], SHORTLIST_K), dtype=np.int64)
    topk_ids_full[plastic_mask] = top2
    return {
        "summary": summary,
        "per_branch_rows": per_branch_rows,
        "confusion_rows": confusion_rows,
        "entropy_rows": entropy_rows,
        "topk_ids_full": topk_ids_full,
    }


def _shortlist_exact_proxy(
    arrays: dict[str, np.ndarray],
    topk_ids: np.ndarray,
    *,
    branch_tol: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    principal_state = _build_principal_state_3d(
        arrays["strain_eng"],
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
        shear=arrays["material_reduced"][:, 2],
        bulk=arrays["material_reduced"][:, 3],
        lame=arrays["material_reduced"][:, 4],
    )
    state = _build_branch_state_from_principal(principal_state)
    n = arrays["strain_eng"].shape[0]
    shortlist = np.asarray(topk_ids, dtype=np.int64)
    if shortlist.shape != (n, SHORTLIST_K):
        raise ValueError(f"Expected shortlist shape {(n, SHORTLIST_K)}, got {shortlist.shape}.")

    exact_s = state.lambda_s <= np.minimum(state.gamma_sl, state.gamma_sr) + branch_tol
    exact_l = (
        (~exact_s)
        & (state.gamma_sl < state.gamma_sr + branch_tol)
        & (state.lambda_l >= state.gamma_sl - branch_tol)
        & (state.lambda_l <= state.gamma_la + branch_tol)
    )
    exact_r = (
        (~exact_s)
        & (~exact_l)
        & (state.gamma_sl > state.gamma_sr - branch_tol)
        & (state.lambda_r >= state.gamma_sr - branch_tol)
        & (state.lambda_r <= state.gamma_ra + branch_tol)
    )
    elastic_mask = state.f_trial <= branch_tol
    plastic_mask = ~elastic_mask
    exact_a = plastic_mask & ~(exact_s | exact_l | exact_r)

    has_smooth = np.any(shortlist == 1, axis=1)
    has_left = np.any(shortlist == 2, axis=1)
    has_right = np.any(shortlist == 3, axis=1)
    has_apex = np.any(shortlist == 4, axis=1)

    shortlist_branch = np.zeros(n, dtype=np.int64)
    shortlist_hit_mask = np.zeros(n, dtype=bool)
    shortlist_branch[elastic_mask] = 0
    shortlist_hit_mask[elastic_mask] = True

    accept_s = plastic_mask & exact_s & has_smooth
    accept_l = plastic_mask & exact_l & has_left
    accept_r = plastic_mask & exact_r & has_right
    accept_a = plastic_mask & exact_a & has_apex
    shortlist_branch[accept_s] = 1
    shortlist_branch[accept_l] = 2
    shortlist_branch[accept_r] = 3
    shortlist_branch[accept_a] = 4
    shortlist_hit_mask |= accept_s | accept_l | accept_r | accept_a
    fallback_mask = plastic_mask & ~shortlist_hit_mask

    stress = np.zeros((n, 6), dtype=np.float64)
    stress_principal = np.zeros((n, 3), dtype=np.float64)
    plastic_multiplier = np.zeros(n, dtype=np.float64)
    branch_id = np.zeros(n, dtype=np.int64)

    resolved_branch = np.where(shortlist_hit_mask, shortlist_branch, 0)
    resolved_stress, resolved_principal, resolved_lambda = _dispatch_from_branch_state(state, resolved_branch)
    stress[shortlist_hit_mask] = resolved_stress[shortlist_hit_mask]
    stress_principal[shortlist_hit_mask] = resolved_principal[shortlist_hit_mask]
    plastic_multiplier[shortlist_hit_mask] = resolved_lambda[shortlist_hit_mask]
    branch_id[shortlist_hit_mask] = resolved_branch[shortlist_hit_mask]

    if np.any(fallback_mask):
        fallback = constitutive_update_3d(
            arrays["strain_eng"][fallback_mask],
            c_bar=arrays["material_reduced"][fallback_mask, 0],
            sin_phi=arrays["material_reduced"][fallback_mask, 1],
            shear=arrays["material_reduced"][fallback_mask, 2],
            bulk=arrays["material_reduced"][fallback_mask, 3],
            lame=arrays["material_reduced"][fallback_mask, 4],
        )
        stress[fallback_mask] = fallback.stress
        stress_principal[fallback_mask] = fallback.stress_principal
        plastic_multiplier[fallback_mask] = fallback.plastic_multiplier
        branch_id[fallback_mask] = fallback.branch_id

    return {
        "stress": stress.astype(np.float32),
        "stress_principal": stress_principal.astype(np.float32),
        "branch_id": branch_id.astype(np.int8),
        "plastic_multiplier": plastic_multiplier.astype(np.float32),
        "elastic_mask": elastic_mask.astype(bool),
        "shortlist_hit_mask": (shortlist_hit_mask & plastic_mask).astype(bool),
        "fallback_mask": fallback_mask.astype(bool),
    }


def _time_exact_update(arrays: dict[str, np.ndarray], *, warmup: int, repeats: int) -> tuple[list[float], Any]:
    timings: list[float] = []
    last_result = None
    for run_idx in range(warmup + repeats):
        start = perf_counter()
        result = constitutive_update_3d(
            arrays["strain_eng"],
            c_bar=arrays["material_reduced"][:, 0],
            sin_phi=arrays["material_reduced"][:, 1],
            shear=arrays["material_reduced"][:, 2],
            bulk=arrays["material_reduced"][:, 3],
            lame=arrays["material_reduced"][:, 4],
        )
        elapsed = perf_counter() - start
        last_result = result
        if run_idx >= warmup:
            timings.append(float(elapsed))
    return timings, last_result


def _benchmark_shortlist_panel(
    panel_name: str,
    arrays: dict[str, np.ndarray],
    topk_ids: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_times, full_result = _time_exact_update(arrays, warmup=warmup, repeats=repeats)
    proxy = _shortlist_exact_proxy(arrays, topk_ids)
    proxy_times: list[float] = []
    for run_idx in range(warmup + repeats):
        start = perf_counter()
        _shortlist_exact_proxy(arrays, topk_ids)
        elapsed = perf_counter() - start
        if run_idx >= warmup:
            proxy_times.append(float(elapsed))

    plastic_mask = arrays["branch_id"] > 0
    shortlist_hit_rate = float(np.mean(proxy["shortlist_hit_mask"][plastic_mask])) if np.any(plastic_mask) else 0.0
    fallback_rate = float(np.mean(proxy["fallback_mask"][plastic_mask])) if np.any(plastic_mask) else 0.0
    row_mismatch = proxy["branch_id"].astype(np.int64) != full_result.branch_id.astype(np.int64)
    row_mismatch |= np.any(proxy["stress"] != full_result.stress.astype(np.float32), axis=1)
    row_mismatch |= np.any(proxy["stress_principal"] != full_result.stress_principal.astype(np.float32), axis=1)

    run_rows = []
    for idx, (full_time, proxy_time) in enumerate(zip(full_times, proxy_times)):
        run_rows.append(
            {
                "panel_name": panel_name,
                "run_idx": idx,
                "full_exact_s": float(full_time),
                "proxy_s": float(proxy_time),
                "speedup_fraction": float(1.0 - proxy_time / full_time) if full_time > 0.0 else 0.0,
            }
        )

    summary = {
        "panel_name": panel_name,
        "n_rows": int(arrays["strain_eng"].shape[0]),
        "plastic_rows": int(np.sum(plastic_mask)),
        "shortlist_hit_rate": shortlist_hit_rate,
        "fallback_rate": fallback_rate,
        "exact_state_mismatch_rate": float(np.mean(row_mismatch)),
        "exact_state_mismatch_rows": int(np.sum(row_mismatch)),
        "median_full_exact_s": _median(full_times),
        "median_proxy_s": _median(proxy_times),
        "median_speedup_fraction": _median([float(row["speedup_fraction"]) for row in run_rows]),
        "median_full_exact_per_row_us": 1.0e6 * _median(full_times) / max(int(arrays["strain_eng"].shape[0]), 1),
        "median_proxy_per_row_us": 1.0e6 * _median(proxy_times) / max(int(arrays["strain_eng"].shape[0]), 1),
    }
    return run_rows, summary


def main() -> None:
    args = parse_args()
    redesign_root = (ROOT / args.redesign_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    execution_doc = docs_root / "return_mapping_accel_execution_20260324.md"
    rm0_report = docs_root / "return_mapping_accel_rm0_profile_20260324.md"
    rm1_report = docs_root / "return_mapping_accel_rm1_branch_hints_20260324.md"
    pivot_report = docs_root / "return_mapping_accel_pivot_memo_20260324.md"
    taskgiver_report = docs_root / "return_mapping_accel_taskgiver_report_20260324.md"

    real_dataset_path = redesign_root / "real_grouped_sampled_512.h5"
    panel_path = redesign_root / "panels" / "panel_sidecar.h5"
    snapshot_selection_path = redesign_root / "exp_a_global" / "candidate_b_snapshot_selection.json"
    baseline_selection_path = redesign_root / "exp_a_global" / "baseline_selection.json"
    redesign_final_summary_path = redesign_root / "final" / "final_test_summary.json"
    baseline_summary_path = redesign_root / "baseline" / "rb_staged_w512_d6_valfirst" / "summary.json"

    arrays_all, _ = _load_h5(real_dataset_path)
    panel_all, _ = _load_h5(panel_path)
    snapshot_selection = json.loads(snapshot_selection_path.read_text(encoding="utf-8"))
    baseline_selection = json.loads(baseline_selection_path.read_text(encoding="utf-8"))
    redesign_final_summary = json.loads(redesign_final_summary_path.read_text(encoding="utf-8"))
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))

    baseline_ckpt = Path(baseline_summary["best_checkpoint"])
    candidate_ckpt = Path(snapshot_selection["selected_checkpoint_path"])

    synthetic_summary = build_synthetic_hard_panel(
        redesign_root=redesign_root,
        output_root=output_root / "synthetic_hard",
        sample_count=args.synthetic_hard_rows,
        seed=args.seed,
    )
    synthetic_arrays, _ = _load_h5(Path(synthetic_summary["dataset_path"]))

    profile_panels = {
        "broad_val": _panel_arrays(arrays_all, panel_all, "broad_val"),
        "hard_val": _panel_arrays(arrays_all, panel_all, "hard_val"),
        "hard_val_plastic": _slice_arrays(arrays_all, panel_all["hard_val_mask"].astype(bool) & panel_all["plastic_mask"].astype(bool)),
        "broad_test": _panel_arrays(arrays_all, panel_all, "broad_test"),
        "hard_test": _panel_arrays(arrays_all, panel_all, "hard_test"),
        "ds_valid": _panel_arrays(arrays_all, panel_all, "ds_valid"),
        "synthetic_hard": synthetic_arrays,
    }

    profiling_run_rows: list[dict[str, Any]] = []
    profiling_summary_rows: list[dict[str, Any]] = []
    for panel_name, panel_arrays in profile_panels.items():
        run_rows, summary = _profile_arrays(
            panel_name,
            panel_arrays,
            warmup=args.profile_warmup,
            repeats=args.profile_repeats,
            return_tangent=False,
        )
        profiling_run_rows.extend(run_rows)
        profiling_summary_rows.append(summary)

    profiling_dir = output_root / "profiling"
    profiling_runs_csv = _write_csv(
        profiling_dir / "panel_profile_runs.csv",
        profiling_run_rows,
        ["panel_name", "run_idx", "n_rows", "principal_eig_s", "branch_scalar_s", "branch_dispatch_s", "tangent_fd_s", "total_s", "residual_overhead_s", "explicit_non_eig_s", "per_row_total_us", "per_row_principal_eig_us", "per_row_explicit_non_eig_us", "explicit_non_eig_share", "branch_scalar_share", "principal_eig_share", "plastic_fraction"],
    )
    profiling_summary_csv = _write_csv(
        profiling_dir / "panel_profile_summary.csv",
        profiling_summary_rows,
        ["panel_name", "n_rows", "branch_counts", "plastic_fraction", "median_total_s", "median_principal_eig_s", "median_branch_scalar_s", "median_branch_dispatch_s", "median_tangent_fd_s", "median_per_row_total_us", "median_per_row_principal_eig_us", "median_per_row_explicit_non_eig_us", "median_explicit_non_eig_share", "median_branch_scalar_share", "median_principal_eig_share", "branch_scalar_is_dominant"],
    )

    hard_val_plastic_summary = next(row for row in profiling_summary_rows if row["panel_name"] == "hard_val_plastic")
    rm0_admit = bool(
        hard_val_plastic_summary["median_explicit_non_eig_share"] >= 0.20
        or hard_val_plastic_summary["branch_scalar_is_dominant"]
    )
    profiling_summary_json = _write_json(
        profiling_dir / "panel_profile_summary.json",
        {
            "panels": profiling_summary_rows,
            "rm0_admission": {
                "proceed": rm0_admit,
                "hard_val_plastic_non_eig_share": hard_val_plastic_summary["median_explicit_non_eig_share"],
                "hard_val_plastic_branch_scalar_is_dominant": hard_val_plastic_summary["branch_scalar_is_dominant"],
            },
        },
    )

    val_arrays = _panel_arrays(arrays_all, panel_all, "broad_val")
    hint_dir = output_root / "hints"
    hint_artifacts = [
        _branch_hint_artifacts(model_name="baseline_raw_branch", checkpoint_path=baseline_ckpt, arrays=val_arrays, device=args.device),
        _branch_hint_artifacts(model_name="candidate_b", checkpoint_path=candidate_ckpt, arrays=val_arrays, device=args.device),
    ]
    hint_summary_rows = [artifact["summary"] for artifact in hint_artifacts]
    hint_summary_csv = _write_csv(
        hint_dir / "branch_hint_comparison_val.csv",
        hint_summary_rows,
        ["model_name", "checkpoint_path", "branch_head_kind", "plastic_rows", "top1_recall", "top2_recall", "missed_true_branch_rate", "mean_entropy", "p90_entropy", "mean_top1_confidence"],
    )
    per_branch_rows = [row for artifact in hint_artifacts for row in artifact["per_branch_rows"]]
    per_branch_csv = _write_csv(
        hint_dir / "branch_hint_per_branch_recall_val.csv",
        per_branch_rows,
        ["model_name", "true_branch", "n_rows", "top1_recall", "top2_recall", "mean_entropy"],
    )
    entropy_csv = _write_csv(
        hint_dir / "branch_hint_entropy_calibration_val.csv",
        [row for artifact in hint_artifacts for row in artifact["entropy_rows"]],
        ["model_name", "entropy_bin", "n_rows", "top1_recall", "top2_recall", "mean_top1_confidence"],
    )
    for artifact in hint_artifacts:
        _write_csv(
            hint_dir / f"{artifact['summary']['model_name']}_confusion_val.csv",
            artifact["confusion_rows"],
            ["model_name", "true_branch", "predicted_branch", "n_rows"],
        )

    selected_hint = max(
        hint_artifacts,
        key=lambda artifact: (
            artifact["summary"]["top2_recall"],
            artifact["summary"]["top1_recall"],
            -artifact["summary"]["mean_entropy"],
        ),
    )
    hint_gate_pass = bool(selected_hint["summary"]["top2_recall"] >= TOPK_THRESHOLD)
    hint_selection_json = _write_json(
        hint_dir / "branch_hint_selection.json",
        {
            "selected_model_name": selected_hint["summary"]["model_name"],
            "selected_checkpoint_path": selected_hint["summary"]["checkpoint_path"],
            "selected_top2_recall": selected_hint["summary"]["top2_recall"],
            "passes_top2_gate": hint_gate_pass,
            "threshold": TOPK_THRESHOLD,
        },
    )

    shortlist_run_rows: list[dict[str, Any]] = []
    shortlist_summary_rows: list[dict[str, Any]] = []
    shortlist_dir = output_root / "shortlist"
    shortlist_executed = False
    shortlist_promoted = False
    shortlist_reason = "skipped_no_seam"
    if rm0_admit and hint_gate_pass:
        shortlist_executed = True
        shortlist_reason = "executed"
        topk_ids_full = selected_hint["topk_ids_full"]
        val_mask_full = panel_all["broad_val_mask"].astype(bool)
        hard_val_local_mask = panel_all["hard_val_mask"][val_mask_full].astype(bool)
        plastic_val_local_mask = panel_all["plastic_mask"][val_mask_full].astype(bool)
        shortlist_panels = {
            "broad_val": np.ones(val_arrays["strain_eng"].shape[0], dtype=bool),
            "hard_val": hard_val_local_mask,
            "hard_val_plastic": hard_val_local_mask & plastic_val_local_mask,
        }
        for panel_name, mask in shortlist_panels.items():
            panel_arrays = _slice_arrays(val_arrays, mask)
            panel_topk = topk_ids_full[mask]
            run_rows, summary = _benchmark_shortlist_panel(
                panel_name,
                panel_arrays,
                panel_topk,
                warmup=args.shortlist_warmup,
                repeats=args.shortlist_repeats,
            )
            shortlist_run_rows.extend(run_rows)
            shortlist_summary_rows.append(summary)

        _write_csv(
            shortlist_dir / "shortlist_benchmark_runs.csv",
            shortlist_run_rows,
            ["panel_name", "run_idx", "full_exact_s", "proxy_s", "speedup_fraction"],
        )
        shortlist_summary_csv = _write_csv(
            shortlist_dir / "shortlist_benchmark_summary.csv",
            shortlist_summary_rows,
            ["panel_name", "n_rows", "plastic_rows", "shortlist_hit_rate", "fallback_rate", "exact_state_mismatch_rate", "exact_state_mismatch_rows", "median_full_exact_s", "median_proxy_s", "median_speedup_fraction", "median_full_exact_per_row_us", "median_proxy_per_row_us"],
        )
        shortlist_hard = next(row for row in shortlist_summary_rows if row["panel_name"] == "hard_val_plastic")
        shortlist_promoted = bool(
            shortlist_hard["exact_state_mismatch_rows"] == 0
            and shortlist_hard["median_speedup_fraction"] >= 0.15
        )
        _write_json(
            shortlist_dir / "shortlist_benchmark_summary.json",
            {
                "panels": shortlist_summary_rows,
                "promotion": {
                    "promote": shortlist_promoted,
                    "required_speedup_fraction": 0.15,
                    "hard_val_plastic_speedup_fraction": shortlist_hard["median_speedup_fraction"],
                    "hard_val_plastic_exact_state_mismatch_rows": shortlist_hard["exact_state_mismatch_rows"],
                },
            },
        )
    elif not rm0_admit:
        shortlist_reason = "skipped_no_material_seam"
    else:
        shortlist_reason = "skipped_hint_top2_below_threshold"

    rm0_lines = [
        "This phase profiles the current in-repo exact constitutive update on the frozen March 24 real panels and a hard synthetic panel.",
        "",
        "## Artifacts",
        "",
        f"- profiling runs csv: `{profiling_runs_csv}`",
        f"- profiling summary csv: `{profiling_summary_csv}`",
        f"- profiling summary json: `{profiling_summary_json}`",
        f"- synthetic hard panel summary: `{output_root / 'synthetic_hard' / 'synthetic_hard_panel_summary.json'}`",
        "",
        "## Admission Result",
        "",
        f"- hard_val_plastic median explicit non-eig share: `{hard_val_plastic_summary['median_explicit_non_eig_share']:.6f}`",
        f"- hard_val_plastic branch scalar dominant: `{bool(hard_val_plastic_summary['branch_scalar_is_dominant'])}`",
        f"- proceed past RM-0: `{bool(rm0_admit)}`",
    ]
    _write_phase_report(rm0_report, "Return-Mapping Acceleration Scout RM-0", rm0_lines)

    rm1_lines = [
        "This phase benchmarks existing branch signals only as exact-search helpers on frozen dev validation plastic rows.",
        "",
        "## Artifacts",
        "",
        f"- hint comparison csv: `{hint_summary_csv}`",
        f"- per-branch recall csv: `{per_branch_csv}`",
        f"- entropy calibration csv: `{entropy_csv}`",
        f"- hint selection json: `{hint_selection_json}`",
        "",
        "## Result",
        "",
        f"- selected hint source: `{selected_hint['summary']['model_name']}`",
        f"- selected top-2 recall: `{selected_hint['summary']['top2_recall']:.6f}`",
        f"- passes top-2 gate `{TOPK_THRESHOLD:.2f}`: `{bool(hint_gate_pass)}`",
        f"- shortlist execution status: `{shortlist_reason}`",
    ]
    _write_phase_report(rm1_report, "Return-Mapping Acceleration Scout RM-1", rm1_lines)

    if shortlist_executed and shortlist_summary_rows:
        shortlist_hard = next(row for row in shortlist_summary_rows if row["panel_name"] == "hard_val_plastic")
        pivot_lines = [
            "Exact-authority shortlist benchmarking was executed after RM-0 admission and the branch-hint quality screen.",
            "",
            "## Result",
            "",
            f"- selected hint source: `{selected_hint['summary']['model_name']}`",
            f"- hard_val_plastic shortlist hit rate: `{shortlist_hard['shortlist_hit_rate']:.6f}`",
            f"- hard_val_plastic fallback rate: `{shortlist_hard['fallback_rate']:.6f}`",
            f"- hard_val_plastic exact-state mismatch rows: `{shortlist_hard['exact_state_mismatch_rows']}`",
            f"- hard_val_plastic median speedup fraction: `{shortlist_hard['median_speedup_fraction']:.6f}`",
            f"- shortlist prototype promoted: `{bool(shortlist_promoted)}`",
            "- Further in-repo acceleration is blocked unless the shortlist path gives exact equivalence and at least 15% hard-plastic local speedup.",
        ]
    elif not rm0_admit:
        pivot_lines = [
            "RM-0 showed that the current exact operator does not expose a strong enough non-eigendecomposition seam for an in-repo learning-assisted shortcut.",
            "",
            "## Decision",
            "",
            f"- hard_val_plastic median explicit non-eig share: `{hard_val_plastic_summary['median_explicit_non_eig_share']:.6f}`",
            f"- hard_val_plastic branch scalar dominant: `{bool(hard_val_plastic_summary['branch_scalar_is_dominant'])}`",
            "- The next acceleration work should move to the external FE/local-solver harness rather than add more in-repo ML around this closed-form operator.",
        ]
    else:
        pivot_lines = [
            "RM-0 admitted the scout, but the available learned branch hints were not reliable enough to justify shortlist-based exact narrowing.",
            "",
            "## Decision",
            "",
            f"- selected hint source: `{selected_hint['summary']['model_name']}`",
            f"- selected top-2 recall: `{selected_hint['summary']['top2_recall']:.6f}`",
            f"- required top-2 recall: `{TOPK_THRESHOLD:.6f}`",
            "- Shortlist benchmarking was skipped because the true branch would still be missed too often for an exact-search helper.",
        ]
    _write_phase_report(pivot_report, "Return-Mapping Acceleration Scout Pivot Memo", pivot_lines)

    execution_lines = [
        "This document tracks the report5 follow-on implementation: redesign closeout plus the exact-authority acceleration scout.",
        "",
        "## Completed",
        "",
        f"- [x] Redesign closeout artifacts reused from: `{redesign_root}`",
        f"- [x] RM-0 exact-update profiling report: `{rm0_report}`",
        f"- [x] RM-1 branch-hint report: `{rm1_report}`",
        f"- [{'x' if shortlist_executed else ' '}] Shortlist exact-dispatch proxy benchmark",
        f"- [x] Pivot memo: `{pivot_report}`",
        "",
        "## Outcome",
        "",
        f"- RM-0 admission passed: `{bool(rm0_admit)}`",
        f"- best hint source: `{selected_hint['summary']['model_name']}`",
        f"- best hint top-2 recall: `{selected_hint['summary']['top2_recall']:.6f}`",
        f"- shortlist executed: `{bool(shortlist_executed)}`",
        f"- shortlist promoted: `{bool(shortlist_promoted)}`",
    ]
    _write_phase_report(execution_doc, "Return-Mapping Acceleration Execution 20260324", execution_lines)

    final_summary = {
        "redesign_root": str(redesign_root),
        "baseline_checkpoint": str(baseline_ckpt),
        "candidate_checkpoint": str(candidate_ckpt),
        "redesign_baseline_feasible_gate_test": redesign_final_summary.get("baseline_feasible_gate_test"),
        "rm0_admission": {
            "proceed": rm0_admit,
            "hard_val_plastic_non_eig_share": hard_val_plastic_summary["median_explicit_non_eig_share"],
            "hard_val_plastic_branch_scalar_is_dominant": hard_val_plastic_summary["branch_scalar_is_dominant"],
        },
        "selected_hint": selected_hint["summary"],
        "hint_gate_pass": hint_gate_pass,
        "shortlist_executed": shortlist_executed,
        "shortlist_reason": shortlist_reason,
        "shortlist_promoted": shortlist_promoted,
        "shortlist_panels": shortlist_summary_rows,
    }
    final_summary_json = _write_json(output_root / "final_summary.json", final_summary)

    taskgiver_lines = [
        "Prepared for taskgiver review. This report is self-contained and summarizes the redesign closeout plus the exact-authority acceleration scout.",
        "",
        "## Redesign Closeout",
        "",
        f"- March 24 redesign root reused: `{redesign_root}`",
        "- The redesign closeout now includes the validation-feasible baseline gate on final test and an explicit Candidate B snapshot-sweep selection record.",
        f"- baseline-feasible gate delta: `{float(redesign_final_summary['baseline_feasible_gate_test']['delta_geom']):.6f}`",
        f"- baseline-feasible gate final-test coverage: `{float(redesign_final_summary['baseline_feasible_gate_test']['plastic_coverage']):.6f}`",
        f"- baseline-feasible gate final-test yield violation p95: `{float(redesign_final_summary['baseline_feasible_gate_test']['yield_violation_p95']):.6f}`",
        "",
        "## Exact-Authority Scout Setup",
        "",
        f"- exact update implementation: analytic closed-form `constitutive_update_3d` in `src/mc_surrogate/mohr_coulomb.py`",
        f"- real dataset reused: `{real_dataset_path}`",
        f"- frozen panel sidecar reused: `{panel_path}`",
        f"- profiled real panels: `broad_val`, `hard_val`, `broad_test`, `hard_test`, `ds_valid`, plus derived `hard_val_plastic`",
        f"- hard synthetic panel rows: `{synthetic_summary['sample_count']}`",
        f"- profiling warmup/repeats: `{args.profile_warmup}` / `{args.profile_repeats}`",
        f"- shortlist warmup/repeats: `{args.shortlist_warmup}` / `{args.shortlist_repeats}`",
        "",
        "## RM-0 Profiling Result",
        "",
        f"- hard_val_plastic median total time per row (us): `{hard_val_plastic_summary['median_per_row_total_us']:.6f}`",
        f"- hard_val_plastic median principal/eig share: `{hard_val_plastic_summary['median_principal_eig_share']:.6f}`",
        f"- hard_val_plastic median explicit non-eig share: `{hard_val_plastic_summary['median_explicit_non_eig_share']:.6f}`",
        f"- hard_val_plastic branch scalar dominant: `{bool(hard_val_plastic_summary['branch_scalar_is_dominant'])}`",
        f"- proceed past RM-0: `{bool(rm0_admit)}`",
        "",
        "## RM-1 Branch Hints",
        "",
        f"- baseline top-1 / top-2 recall: `{hint_artifacts[0]['summary']['top1_recall']:.6f}` / `{hint_artifacts[0]['summary']['top2_recall']:.6f}`",
        f"- Candidate B top-1 / top-2 recall: `{hint_artifacts[1]['summary']['top1_recall']:.6f}` / `{hint_artifacts[1]['summary']['top2_recall']:.6f}`",
        f"- selected hint source: `{selected_hint['summary']['model_name']}`",
        f"- selected hint top-2 recall: `{selected_hint['summary']['top2_recall']:.6f}`",
        f"- passes helper threshold `{TOPK_THRESHOLD:.2f}`: `{bool(hint_gate_pass)}`",
        "",
        "## Shortlist Proxy Outcome",
        "",
        f"- shortlist executed: `{bool(shortlist_executed)}`",
        f"- shortlist reason: `{shortlist_reason}`",
        (
            f"- hard_val_plastic shortlist speedup: `{next(row for row in shortlist_summary_rows if row['panel_name'] == 'hard_val_plastic')['median_speedup_fraction']:.6f}`"
            if shortlist_summary_rows
            else "- hard_val_plastic shortlist speedup: `not evaluated`"
        ),
        (
            f"- hard_val_plastic exact-state mismatch rows: `{next(row for row in shortlist_summary_rows if row['panel_name'] == 'hard_val_plastic')['exact_state_mismatch_rows']}`"
            if shortlist_summary_rows
            else "- hard_val_plastic exact-state mismatch rows: `not evaluated`"
        ),
        f"- shortlist promoted: `{bool(shortlist_promoted)}`",
        "",
        "## Recommendation",
        "",
        "- Keep exact authority with the closed-form constitutive update.",
        "- If shortlist promotion failed or RM-0 did not admit the line, move any further acceleration work to the external FE/local-solver harness rather than adding more in-repo ML layers around the current exact operator.",
        f"- Final cycle summary json: `{final_summary_json}`",
    ]
    _write_phase_report(taskgiver_report, "Return-Mapping Acceleration Taskgiver Report", taskgiver_lines)

    print(
        json.dumps(
            _json_safe(
                {
                    "final_summary_json": str(final_summary_json),
                    "rm0_admit": rm0_admit,
                    "selected_hint": selected_hint["summary"],
                    "shortlist_executed": shortlist_executed,
                    "shortlist_promoted": shortlist_promoted,
                    "taskgiver_report": str(taskgiver_report),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
