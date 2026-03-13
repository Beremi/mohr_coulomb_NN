#!/usr/bin/env python
"""Improve hard per-material families with branch-balanced synthetic augmentation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import REDUCED_MATERIAL_COLUMNS, SPLIT_NAMES, SPLIT_TO_ID
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, BRANCH_TO_ID, constitutive_update_3d
from mc_surrogate.models import spectral_decomposition_from_strain
from mc_surrogate.real_materials import RawMaterialSpec, default_slope_material_specs
from mc_surrogate.sampling import (
    _principal_direction_template,
    _principal_to_global_engineering_strain,
    _scale_factor_for_branch,
    _yield_scale_from_direction,
    random_rotation_matrices,
)
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", default="experiment_runs/real_sim/per_material_synth_to_real_20260312")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/per_material_hybrid_hardcases_20260312")
    parser.add_argument("--materials", nargs="*", default=("general_slope", "cover_layer"))
    parser.add_argument("--branch-samples", type=int, default=80000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=550)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--plateau-patience", type=int, default=35)
    parser.add_argument("--lbfgs-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2400)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    return parser.parse_args()


def _load_arrays(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, object]) -> None:
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


def _metric_table_row(label: str, metrics: dict[str, float]) -> str:
    branch_acc = metrics.get("branch_accuracy")
    branch_text = "-" if branch_acc is None else f"{branch_acc:.4f}"
    return (
        f"| {label} | {metrics['n_samples']} | {metrics['stress_mae']:.4f} | "
        f"{metrics['stress_rmse']:.4f} | {metrics['stress_max_abs']:.4f} | {branch_text} |"
    )


def _aggregate(metrics_by_material: dict[str, dict[str, float]]) -> dict[str, float]:
    total = sum(entry["n_samples"] for entry in metrics_by_material.values())
    mae = sum(entry["stress_mae"] * entry["n_samples"] for entry in metrics_by_material.values()) / total
    mse = sum((entry["stress_rmse"] ** 2) * entry["n_samples"] for entry in metrics_by_material.values()) / total
    branch_acc = sum(entry.get("branch_accuracy", 0.0) * entry["n_samples"] for entry in metrics_by_material.values()) / total
    return {
        "n_samples": int(total),
        "stress_mae": float(mae),
        "stress_rmse": float(math.sqrt(mse)),
        "stress_max_abs": float(max(entry["stress_max_abs"] for entry in metrics_by_material.values())),
        "branch_accuracy": float(branch_acc),
    }


def _branch_fraction_blend(branch_id: np.ndarray) -> dict[str, float]:
    counts = np.array([np.sum(branch_id == i) for i in range(len(BRANCH_NAMES))], dtype=float)
    real_frac = counts / max(counts.sum(), 1.0)
    uniform = np.full(len(BRANCH_NAMES), 1.0 / len(BRANCH_NAMES), dtype=float)
    blended = 0.5 * real_frac + 0.5 * uniform
    blended = blended / blended.sum()
    return {name: float(blended[i]) for i, name in enumerate(BRANCH_NAMES)}


def _fixed_material_rows(spec: RawMaterialSpec, strength_reduction: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = strength_reduction.shape[0]
    phi_deg = np.full(n, spec.phi_deg, dtype=float)
    psi_deg = np.full(n, spec.psi_deg, dtype=float)
    raw = {
        "c0": np.full(n, spec.c0, dtype=float),
        "phi_deg": phi_deg,
        "psi_deg": psi_deg,
        "phi_rad": np.deg2rad(phi_deg),
        "psi_rad": np.deg2rad(psi_deg),
        "young": np.full(n, spec.young, dtype=float),
        "poisson": np.full(n, spec.poisson, dtype=float),
        "strength_reduction": strength_reduction.astype(float),
        "davis_type": np.array([spec.davis_type] * n, dtype=object),
    }
    reduced = build_reduced_material_from_raw(
        raw["c0"],
        raw["phi_rad"],
        raw["psi_rad"],
        raw["young"],
        raw["poisson"],
        raw["strength_reduction"],
        raw["davis_type"],
    )
    rows = np.column_stack([reduced.c_bar, reduced.sin_phi, reduced.shear, reduced.bulk, reduced.lame]).astype(np.float32)
    return raw, rows


def _generate_branch_augmented_arrays(
    *,
    spec: RawMaterialSpec,
    strength_reduction_range: tuple[float, float],
    branch_fractions: dict[str, float],
    n_samples: int,
    principal_cap: float,
    seed: int,
    candidate_batch: int = 2048,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    desired_counts: dict[str, int] = {}
    assigned = 0
    for name in BRANCH_NAMES[:-1]:
        count = int(round(branch_fractions[name] * n_samples))
        desired_counts[name] = count
        assigned += count
    desired_counts[BRANCH_NAMES[-1]] = n_samples - assigned

    chunks: list[dict[str, np.ndarray]] = []
    lo, hi = strength_reduction_range

    for branch in BRANCH_NAMES:
        remaining = desired_counts[branch]
        attempts = 0
        while remaining > 0:
            attempts += 1
            if attempts > 5000:
                raise RuntimeError(f"Failed to generate enough fixed-material samples for {spec.name}:{branch}.")
            batch = max(candidate_batch, remaining * 2)
            srf = np.exp(rng.uniform(np.log(lo), np.log(hi), size=batch))
            _, material_reduced = _fixed_material_rows(spec, srf)

            direction = _principal_direction_template(branch, batch, rng)
            alpha = _yield_scale_from_direction(
                direction,
                material_reduced[:, 0],
                material_reduced[:, 1],
                material_reduced[:, 2],
                material_reduced[:, 4],
            )
            factor = _scale_factor_for_branch(branch, batch, rng)
            principal = direction * alpha[:, None] * factor[:, None]
            rotations = random_rotation_matrices(batch, rng)
            strain_eng = _principal_to_global_engineering_strain(principal, rotations)

            exact = constitutive_update_3d(
                strain_eng,
                c_bar=material_reduced[:, 0],
                sin_phi=material_reduced[:, 1],
                shear=material_reduced[:, 2],
                bulk=material_reduced[:, 3],
                lame=material_reduced[:, 4],
            )
            matched = exact.branch_id == BRANCH_TO_ID[branch]
            matched &= np.max(np.abs(principal), axis=1) <= principal_cap
            idx = np.flatnonzero(matched)[:remaining]
            if idx.size == 0:
                continue

            reduced_sel = material_reduced[idx]
            exact_sel = constitutive_update_3d(
                strain_eng[idx],
                c_bar=reduced_sel[:, 0],
                sin_phi=reduced_sel[:, 1],
                shear=reduced_sel[:, 2],
                bulk=reduced_sel[:, 3],
                lame=reduced_sel[:, 4],
            )
            chunks.append(
                {
                    "strain_eng": strain_eng[idx].astype(np.float32),
                    "stress": exact_sel.stress.astype(np.float32),
                    "material_reduced": reduced_sel.astype(np.float32),
                    "stress_principal": exact_sel.stress_principal.astype(np.float32),
                    "branch_id": exact_sel.branch_id.astype(np.int8),
                    "eigvecs": exact_sel.eigvecs.astype(np.float32),
                }
            )
            remaining -= idx.size

    merged: dict[str, np.ndarray] = {}
    for key in chunks[0]:
        merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    perm = rng.permutation(merged["strain_eng"].shape[0])
    return {key: value[perm] for key, value in merged.items()}


def _merge_empirical_and_branch(
    empirical_arrays: dict[str, np.ndarray],
    *,
    branch_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    train_mask = empirical_arrays["split_id"] == SPLIT_TO_ID["train"]
    val_mask = empirical_arrays["split_id"] == SPLIT_TO_ID["val"]
    test_mask = empirical_arrays["split_id"] == SPLIT_TO_ID["test"]

    merged: dict[str, np.ndarray] = {}
    keys = ("strain_eng", "stress", "material_reduced", "stress_principal", "branch_id", "eigvecs")
    for key in keys:
        merged[key] = np.concatenate(
            [
                empirical_arrays[key][train_mask],
                branch_arrays[key],
                empirical_arrays[key][val_mask],
                empirical_arrays[key][test_mask],
            ],
            axis=0,
        )
    merged["split_id"] = np.concatenate(
        [
            np.full(int(np.sum(train_mask)), SPLIT_TO_ID["train"], dtype=np.int8),
            np.full(branch_arrays["strain_eng"].shape[0], SPLIT_TO_ID["train"], dtype=np.int8),
            np.full(int(np.sum(val_mask)), SPLIT_TO_ID["val"], dtype=np.int8),
            np.full(int(np.sum(test_mask)), SPLIT_TO_ID["test"], dtype=np.int8),
        ]
    )
    return merged


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Per-Material Hybrid Hardcase Study",
        "",
        "This follow-up keeps the good per-material empirical-exact models for the two foundation families,",
        "and improves the two hard plastic families with extra branch-balanced fixed-material synthetic data.",
        "",
        f"Base experiment: `{summary['base_root']}`",
        f"Hybrid output root: `{summary['output_root']}`",
        "",
        "## Hardcase Results",
        "",
        "| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["hardcase_results"].items():
        lines.append(_metric_table_row(name, row["primary_eval"]))
    lines.append(_metric_table_row("hardcase aggregate", summary["hardcase_aggregate"]["primary_eval"]))

    lines.extend(
        [
            "",
            "Hardcase comparison against the earlier per-material empirical-only models:",
            "",
            "| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["hardcase_results"].items():
        lines.append(_metric_table_row(name, row["empirical_primary_eval"]))
    lines.append(_metric_table_row("hardcase aggregate", summary["hardcase_aggregate"]["empirical_primary_eval"]))

    lines.extend(
        [
            "",
            "## Mixed Best Aggregate",
            "",
            "This aggregate uses the per-material empirical models for `general_foundation` and `weak_foundation`,",
            "and the new hybrid models for the hard materials.",
            "",
            "| Split | Samples | MAE | RMSE | Max Abs | Branch Acc |",
            "|---|---:|---:|---:|---:|---:|",
            _metric_table_row("primary mixed-best", summary["mixed_best"]["primary_eval"]),
            _metric_table_row("cross mixed-best", summary["mixed_best"]["cross_eval"]),
            _metric_table_row("primary baseline", summary["mixed_best"]["primary_baseline_eval"]),
            _metric_table_row("cross baseline", summary["mixed_best"]["cross_baseline_eval"]),
            "",
            "## Notes",
            "",
            "- If the hardcase hybrid models improve materially, the problem is not that synthetic training is impossible; it is that empirical exact relabeling alone undersupplies hard plastic branches.",
            "- If the mixed-best aggregate still loses to the global real-trained baseline, the current synthetic-only route is still not competitive enough.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prior_summary = json.loads((base_root / "summary.json").read_text(encoding="utf-8"))
    specs = {spec.name: spec for spec in default_slope_material_specs()}

    hardcase_results: dict[str, object] = {}
    primary_hardcase_metrics: dict[str, dict[str, float]] = {}
    primary_hardcase_empirical: dict[str, dict[str, float]] = {}
    cross_hardcase_metrics: dict[str, dict[str, float]] = {}
    cross_hardcase_empirical: dict[str, dict[str, float]] = {}

    for i, material_name in enumerate(args.materials):
        spec = specs[material_name]
        empirical = prior_summary["materials"][material_name]
        primary_real_path = empirical["primary_dataset"]
        cross_real_path = empirical["cross_dataset"]
        empirical_synth_path = empirical["synthetic_dataset"]
        empirical_ckpt = empirical["train_summary"]["best_checkpoint"]

        primary_real, _ = _load_arrays(primary_real_path)
        empirical_synth, _ = _load_arrays(empirical_synth_path)

        train_mask = primary_real["split_id"] == SPLIT_TO_ID["train"]
        srf_train = primary_real["estimated_strength_reduction"][train_mask]
        branch_train = primary_real["branch_id"][train_mask]
        branch_fractions = _branch_fraction_blend(branch_train)
        strain_train = primary_real["strain_eng"][train_mask]
        strain_principal, _ = spectral_decomposition_from_strain(strain_train)
        principal_cap = float(np.quantile(np.max(np.abs(strain_principal), axis=1), 0.995) * 1.10)

        branch_arrays = _generate_branch_augmented_arrays(
            spec=spec,
            strength_reduction_range=(float(np.quantile(srf_train, 0.02)), float(np.quantile(srf_train, 0.98))),
            branch_fractions=branch_fractions,
            n_samples=args.branch_samples,
            principal_cap=max(principal_cap, 5.0e-3),
            seed=args.seed + i,
        )
        hybrid_arrays = _merge_empirical_and_branch(empirical_synth, branch_arrays=branch_arrays)

        hybrid_dataset_path = output_root / "hybrid_datasets" / f"{material_name}_hybrid.h5"
        _write_dataset(
            hybrid_dataset_path,
            hybrid_arrays,
            {
                "material_name": material_name,
                "branch_samples": int(args.branch_samples),
                "branch_fraction_blend": branch_fractions,
                "strength_reduction_range": [
                    float(np.quantile(srf_train, 0.02)),
                    float(np.quantile(srf_train, 0.98)),
                ],
                "principal_cap": principal_cap,
            },
        )

        run_dir = output_root / "runs" / material_name
        train_summary = train_model(
            TrainingConfig(
                dataset=str(hybrid_dataset_path),
                run_dir=str(run_dir),
                model_kind="raw_branch",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                width=args.width,
                depth=args.depth,
                dropout=0.0,
                seed=args.seed + 100 + i,
                patience=args.patience,
                grad_clip=1.0,
                branch_loss_weight=0.1,
                num_workers=0,
                device=args.device,
                scheduler_kind="plateau",
                warmup_epochs=0,
                min_lr=1.0e-6,
                plateau_factor=0.5,
                plateau_patience=args.plateau_patience,
                lbfgs_epochs=args.lbfgs_epochs,
                lbfgs_lr=0.25,
                lbfgs_max_iter=20,
                lbfgs_history_size=100,
                log_every_epochs=50,
            )
        )
        best_ckpt = train_summary["best_checkpoint"]
        primary_eval = evaluate_checkpoint_on_dataset(best_ckpt, primary_real_path, split="test", device=args.device, batch_size=args.eval_batch_size)["metrics"]
        cross_eval = evaluate_checkpoint_on_dataset(best_ckpt, cross_real_path, split="test", device=args.device, batch_size=args.eval_batch_size)["metrics"]
        empirical_primary = evaluate_checkpoint_on_dataset(empirical_ckpt, primary_real_path, split="test", device=args.device, batch_size=args.eval_batch_size)["metrics"]
        empirical_cross = evaluate_checkpoint_on_dataset(empirical_ckpt, cross_real_path, split="test", device=args.device, batch_size=args.eval_batch_size)["metrics"]

        hardcase_results[material_name] = {
            "train_summary": train_summary,
            "primary_eval": primary_eval,
            "cross_eval": cross_eval,
            "empirical_primary_eval": empirical_primary,
            "empirical_cross_eval": empirical_cross,
            "hybrid_dataset": str(hybrid_dataset_path),
            "branch_fraction_blend": branch_fractions,
        }
        primary_hardcase_metrics[material_name] = primary_eval
        cross_hardcase_metrics[material_name] = cross_eval
        primary_hardcase_empirical[material_name] = empirical_primary
        cross_hardcase_empirical[material_name] = empirical_cross

    mixed_primary: dict[str, dict[str, float]] = {}
    mixed_cross: dict[str, dict[str, float]] = {}
    mixed_primary_baseline: dict[str, dict[str, float]] = {}
    mixed_cross_baseline: dict[str, dict[str, float]] = {}
    for material_name, row in prior_summary["materials"].items():
        if material_name in hardcase_results:
            mixed_primary[material_name] = hardcase_results[material_name]["primary_eval"]
            mixed_cross[material_name] = hardcase_results[material_name]["cross_eval"]
        else:
            mixed_primary[material_name] = row["primary_eval"]
            mixed_cross[material_name] = row["cross_eval"]
        mixed_primary_baseline[material_name] = row["primary_baseline_eval"]
        mixed_cross_baseline[material_name] = row["cross_baseline_eval"]

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "hardcase_results": hardcase_results,
        "hardcase_aggregate": {
            "primary_eval": _aggregate(primary_hardcase_metrics),
            "cross_eval": _aggregate(cross_hardcase_metrics),
            "empirical_primary_eval": _aggregate(primary_hardcase_empirical),
            "empirical_cross_eval": _aggregate(cross_hardcase_empirical),
        },
        "mixed_best": {
            "primary_eval": _aggregate(mixed_primary),
            "cross_eval": _aggregate(mixed_cross),
            "primary_baseline_eval": _aggregate(mixed_primary_baseline),
            "cross_baseline_eval": _aggregate(mixed_cross_baseline),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(ROOT / "docs" / "per_material_hybrid_synth_to_real.md", summary)
    print(json.dumps(summary["mixed_best"], indent=2))


if __name__ == "__main__":
    main()
