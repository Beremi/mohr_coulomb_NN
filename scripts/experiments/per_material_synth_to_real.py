#!/usr/bin/env python
"""Train per-material surrogates on exact-labeled synthetic data and test on real export samples."""

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
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from mc_surrogate.models import spectral_decomposition_from_strain
from mc_surrogate.real_materials import assign_material_families, default_slope_material_specs
from mc_surrogate.sampling import random_rotation_matrices
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model
from mc_surrogate.voigt import tensor_to_strain_voigt


REQUIRED_KEYS = (
    "strain_eng",
    "stress",
    "material_reduced",
    "stress_principal",
    "branch_id",
    "eigvecs",
    "split_id",
)
OPTIONAL_KEYS = ("source_call_id", "source_mode_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-real", default="experiment_runs/real_sim/train_sample_512.h5")
    parser.add_argument("--cross-real", default="experiment_runs/real_sim/baseline_sample_256.h5")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/per_material_synth_to_real_20260312")
    parser.add_argument(
        "--baseline-checkpoint",
        default="experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--lbfgs-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    return parser.parse_args()


def _load_dataset_full(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        for key in REQUIRED_KEYS:
            arrays[key] = f[key][:]
        for key in OPTIONAL_KEYS:
            if key in f:
                arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            if isinstance(value, bytes):
                attrs[key] = value.decode()
            else:
                attrs[key] = value
    return arrays, attrs


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], split_id: np.ndarray, attrs: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.create_dataset("split_id", data=np.asarray(split_id, dtype=np.int8), compression="gzip", shuffle=True)
        f.attrs["branch_names_json"] = json.dumps(BRANCH_NAMES)
        f.attrs["reduced_material_columns_json"] = json.dumps(REDUCED_MATERIAL_COLUMNS)
        f.attrs["split_names_json"] = json.dumps(SPLIT_NAMES)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value


def _filter_dataset_by_material(
    arrays: dict[str, np.ndarray],
    assignment: dict[str, np.ndarray | list[str]],
    material_id: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mask = assignment["material_id"] == material_id
    out = {key: value[mask] for key, value in arrays.items() if key != "split_id"}
    out["estimated_strength_reduction"] = assignment["estimated_strength_reduction"][mask]
    out["material_fit_error"] = assignment["fit_error"][mask]
    return out, arrays["split_id"][mask]


def _exact_response_arrays(strain_eng: np.ndarray, material_reduced: np.ndarray) -> dict[str, np.ndarray]:
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


def _augment_train_inputs(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    copies: int,
    seed: int,
    principal_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return (
            np.zeros((0, strain_eng.shape[1]), dtype=np.float32),
            np.zeros((0, material_reduced.shape[1]), dtype=np.float32),
        )

    rng = np.random.default_rng(seed)
    strain_principal, _ = spectral_decomposition_from_strain(strain_eng)

    strain_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    for _ in range(copies):
        scale = np.exp(rng.normal(loc=0.0, scale=0.10, size=strain_principal.shape[0]))
        scale = np.clip(scale, 0.82, 1.22)
        principal_aug = strain_principal * scale[:, None]
        current_max = np.max(np.abs(principal_aug), axis=1)
        clip = np.minimum(1.0, principal_cap / np.maximum(current_max, 1.0e-12))
        principal_aug = principal_aug * clip[:, None]

        rotations = random_rotation_matrices(strain_principal.shape[0], rng)
        strain_aug = tensor_to_strain_voigt(np.einsum("nij,nj,nkj->nik", rotations, principal_aug, rotations))
        strain_parts.append(strain_aug.astype(np.float32))
        material_parts.append(material_reduced.astype(np.float32))

    return np.vstack(strain_parts), np.vstack(material_parts)


def _build_empirical_synthetic_dataset(
    path: Path,
    *,
    real_arrays: dict[str, np.ndarray],
    split_id: np.ndarray,
    augment_copies: int,
    seed: int,
    attrs: dict[str, object],
) -> dict[str, object]:
    train_mask = split_id == SPLIT_TO_ID["train"]
    val_mask = split_id == SPLIT_TO_ID["val"]
    test_mask = split_id == SPLIT_TO_ID["test"]

    train_principal, _ = spectral_decomposition_from_strain(real_arrays["strain_eng"][train_mask])
    principal_cap = float(np.quantile(np.max(np.abs(train_principal), axis=1), 0.995) * 1.05)

    original_train = _exact_response_arrays(real_arrays["strain_eng"][train_mask], real_arrays["material_reduced"][train_mask])
    val_exact = _exact_response_arrays(real_arrays["strain_eng"][val_mask], real_arrays["material_reduced"][val_mask])
    test_exact = _exact_response_arrays(real_arrays["strain_eng"][test_mask], real_arrays["material_reduced"][test_mask])

    aug_strain, aug_material = _augment_train_inputs(
        real_arrays["strain_eng"][train_mask],
        real_arrays["material_reduced"][train_mask],
        copies=augment_copies,
        seed=seed,
        principal_cap=principal_cap,
    )
    aug_exact = _exact_response_arrays(aug_strain, aug_material)

    merged: dict[str, np.ndarray] = {}
    for key in original_train:
        merged[key] = np.concatenate([original_train[key], aug_exact[key], val_exact[key], test_exact[key]], axis=0)

    split = np.concatenate(
        [
            np.full(original_train["strain_eng"].shape[0], SPLIT_TO_ID["train"], dtype=np.int8),
            np.full(aug_exact["strain_eng"].shape[0], SPLIT_TO_ID["train"], dtype=np.int8),
            np.full(val_exact["strain_eng"].shape[0], SPLIT_TO_ID["val"], dtype=np.int8),
            np.full(test_exact["strain_eng"].shape[0], SPLIT_TO_ID["test"], dtype=np.int8),
        ]
    )

    synthetic_attrs = dict(attrs)
    synthetic_attrs.update(
        {
            "generator": "empirical_input_exact_relabel",
            "augment_copies": int(augment_copies),
            "principal_cap": principal_cap,
        }
    )
    _write_dataset(path, merged, split, synthetic_attrs)
    return {
        "principal_cap": principal_cap,
        "n_train_original": int(original_train["strain_eng"].shape[0]),
        "n_train_augmented": int(aug_exact["strain_eng"].shape[0]),
        "n_val": int(val_exact["strain_eng"].shape[0]),
        "n_test": int(test_exact["strain_eng"].shape[0]),
    }


def _metric_table_row(label: str, metrics: dict[str, float]) -> str:
    branch_acc = metrics.get("branch_accuracy")
    branch_text = "-" if branch_acc is None else f"{branch_acc:.4f}"
    return (
        f"| {label} | {metrics['n_samples']} | {metrics['stress_mae']:.4f} | "
        f"{metrics['stress_rmse']:.4f} | {metrics['stress_max_abs']:.4f} | {branch_text} |"
    )


def _aggregate_metrics(metrics_by_material: dict[str, dict[str, float]]) -> dict[str, float]:
    total = sum(entry["n_samples"] for entry in metrics_by_material.values())
    mae = sum(entry["stress_mae"] * entry["n_samples"] for entry in metrics_by_material.values()) / total
    mse = sum((entry["stress_rmse"] ** 2) * entry["n_samples"] for entry in metrics_by_material.values()) / total
    branch_n = sum(entry["n_samples"] for entry in metrics_by_material.values() if "branch_accuracy" in entry)
    branch_acc = None
    if branch_n > 0:
        branch_acc = sum(entry.get("branch_accuracy", 0.0) * entry["n_samples"] for entry in metrics_by_material.values()) / branch_n
    return {
        "n_samples": int(total),
        "stress_mae": float(mae),
        "stress_rmse": float(math.sqrt(mse)),
        "stress_max_abs": float(max(entry["stress_max_abs"] for entry in metrics_by_material.values())),
        **({"branch_accuracy": float(branch_acc)} if branch_acc is not None else {}),
    }


def _material_stats(real_arrays: dict[str, np.ndarray], split_id: np.ndarray) -> dict[str, object]:
    train_mask = split_id == SPLIT_TO_ID["train"]
    srf = real_arrays["estimated_strength_reduction"]
    fit_error = real_arrays["material_fit_error"]
    train_principal, _ = spectral_decomposition_from_strain(real_arrays["strain_eng"][train_mask])
    max_abs_principal = np.max(np.abs(train_principal), axis=1)
    branch_id = real_arrays["branch_id"][train_mask]
    branch_counts = {name: int(np.sum(branch_id == i)) for i, name in enumerate(BRANCH_NAMES)}
    return {
        "n_total": int(real_arrays["strain_eng"].shape[0]),
        "n_train": int(np.sum(train_mask)),
        "n_val": int(np.sum(split_id == SPLIT_TO_ID["val"])),
        "n_test": int(np.sum(split_id == SPLIT_TO_ID["test"])),
        "strength_reduction_min_train": float(np.min(srf[train_mask])),
        "strength_reduction_max_train": float(np.max(srf[train_mask])),
        "strength_reduction_q05_train": float(np.quantile(srf[train_mask], 0.05)),
        "strength_reduction_q95_train": float(np.quantile(srf[train_mask], 0.95)),
        "material_fit_error_max": float(np.max(fit_error)),
        "max_abs_principal_strain_q995_train": float(np.quantile(max_abs_principal, 0.995)),
        "branch_counts_train": branch_counts,
    }


def _report_markdown(
    *,
    output_path: Path,
    primary_dataset: str,
    cross_dataset: str,
    baseline_checkpoint: str | None,
    material_results: dict[str, object],
    aggregate_results: dict[str, object],
) -> None:
    lines = [
        "# Per-Material Synthetic-to-Real Study",
        "",
        "This report tests a narrower version of the surrogate problem:",
        "- split the real sampled data by material family,",
        "- train one surrogate per material family,",
        "- train on exact synthetic labels generated from the real train-split input domain,",
        "- evaluate only on held-out real test data.",
        "",
        "This is not broad synthetic sampling. It is an empirical-domain synthetic setup: the training inputs come",
        "from the real train split of each material family, but the labels are recomputed with the exact Python",
        "constitutive operator. That directly tests whether the mixed-material global model was the main bottleneck.",
        "",
        f"Primary real dataset: `{primary_dataset}`",
        f"Cross real dataset: `{cross_dataset}`",
        f"Global baseline checkpoint: `{baseline_checkpoint}`" if baseline_checkpoint else "Global baseline checkpoint: none",
        "",
        "## Material Mapping",
        "",
        "| Material | Train | Val | Test | Train SRF q05-q95 | Train max | Branch train counts |",
        "|---|---:|---:|---:|---|---:|---|",
    ]

    for name, row in material_results.items():
        stats = row["stats"]
        lines.append(
            "| "
            f"{name} | {stats['n_train']} | {stats['n_val']} | {stats['n_test']} | "
            f"{stats['strength_reduction_q05_train']:.3f} - {stats['strength_reduction_q95_train']:.3f} | "
            f"{stats['max_abs_principal_strain_q995_train']:.4e} | "
            f"{json.dumps(stats['branch_counts_train'])} |"
        )

    lines.extend(
        [
            "",
            "## Primary Real Test",
            "",
            "| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in material_results.items():
        lines.append(_metric_table_row(name, row["primary_eval"]))
    lines.append(_metric_table_row("aggregate", aggregate_results["primary_eval"]))

    if aggregate_results.get("primary_baseline_eval") is not None:
        lines.extend(
            [
                "",
                "Global baseline on the same per-material primary test splits:",
                "",
                "| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in material_results.items():
            lines.append(_metric_table_row(name, row["primary_baseline_eval"]))
        lines.append(_metric_table_row("aggregate", aggregate_results["primary_baseline_eval"]))

    lines.extend(
        [
            "",
            "## Cross Real Test",
            "",
            "| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in material_results.items():
        lines.append(_metric_table_row(name, row["cross_eval"]))
    lines.append(_metric_table_row("aggregate", aggregate_results["cross_eval"]))

    if aggregate_results.get("cross_baseline_eval") is not None:
        lines.extend(
            [
                "",
                "Global baseline on the same per-material cross test splits:",
                "",
                "| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in material_results.items():
            lines.append(_metric_table_row(name, row["cross_baseline_eval"]))
        lines.append(_metric_table_row("aggregate", aggregate_results["cross_baseline_eval"]))

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- If the per-material synthetic models win, the mixed-material coupling was a real part of the error budget.",
            "- If they still lose badly to the real-trained global model, then the dominant issue is not material mixing, but state-distribution and architecture bias.",
            "- The `material_fit_error_max` values should stay very small; if they are not, the material-family reconstruction is suspect.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    real_dir = output_root / "real_material_datasets"
    synth_dir = output_root / "synthetic_material_datasets"
    run_dir = output_root / "runs"

    specs = default_slope_material_specs()

    primary_arrays, primary_attrs = _load_dataset_full(args.primary_real)
    cross_arrays, cross_attrs = _load_dataset_full(args.cross_real)
    primary_assignment = assign_material_families(primary_arrays["material_reduced"], specs=specs)
    cross_assignment = assign_material_families(cross_arrays["material_reduced"], specs=specs)

    baseline_checkpoint = args.baseline_checkpoint if Path(args.baseline_checkpoint).exists() else None

    results: dict[str, object] = {}
    primary_eval_agg: dict[str, dict[str, float]] = {}
    cross_eval_agg: dict[str, dict[str, float]] = {}
    primary_baseline_agg: dict[str, dict[str, float]] = {}
    cross_baseline_agg: dict[str, dict[str, float]] = {}

    for material_id, spec in enumerate(specs):
        material_name = spec.name
        primary_material_arrays, primary_material_split = _filter_dataset_by_material(primary_arrays, primary_assignment, material_id)
        cross_material_arrays, cross_material_split = _filter_dataset_by_material(cross_arrays, cross_assignment, material_id)

        primary_material_path = real_dir / f"{material_name}_primary.h5"
        cross_material_path = real_dir / f"{material_name}_cross.h5"
        _write_dataset(
            primary_material_path,
            primary_material_arrays,
            primary_material_split,
            {**primary_attrs, "material_name": material_name, "material_spec": spec.__dict__},
        )
        _write_dataset(
            cross_material_path,
            cross_material_arrays,
            cross_material_split,
            {**cross_attrs, "material_name": material_name, "material_spec": spec.__dict__},
        )

        stats = _material_stats(primary_material_arrays, primary_material_split)
        synthetic_path = synth_dir / f"{material_name}_empirical_exact.h5"
        synthetic_summary = _build_empirical_synthetic_dataset(
            synthetic_path,
            real_arrays=primary_material_arrays,
            split_id=primary_material_split,
            augment_copies=args.augment_copies,
            seed=args.seed + material_id,
            attrs={"material_name": material_name, "material_spec": spec.__dict__},
        )

        this_run_dir = run_dir / material_name
        train_summary = train_model(
            TrainingConfig(
                dataset=str(synthetic_path),
                run_dir=str(this_run_dir),
                model_kind="raw_branch",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                width=args.width,
                depth=args.depth,
                dropout=0.0,
                seed=args.seed + 100 + material_id,
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
                stress_weight_alpha=0.0,
                stress_weight_scale=250.0,
            )
        )

        best_checkpoint = train_summary["best_checkpoint"]
        primary_eval = evaluate_checkpoint_on_dataset(
            best_checkpoint,
            primary_material_path,
            split="test",
            device=args.device,
            batch_size=args.eval_batch_size,
        )["metrics"]
        cross_eval = evaluate_checkpoint_on_dataset(
            best_checkpoint,
            cross_material_path,
            split="test",
            device=args.device,
            batch_size=args.eval_batch_size,
        )["metrics"]

        baseline_primary = None
        baseline_cross = None
        if baseline_checkpoint is not None:
            baseline_primary = evaluate_checkpoint_on_dataset(
                baseline_checkpoint,
                primary_material_path,
                split="test",
                device=args.device,
                batch_size=args.eval_batch_size,
            )["metrics"]
            baseline_cross = evaluate_checkpoint_on_dataset(
                baseline_checkpoint,
                cross_material_path,
                split="test",
                device=args.device,
                batch_size=args.eval_batch_size,
            )["metrics"]

        results[material_name] = {
            "stats": stats,
            "synthetic_summary": synthetic_summary,
            "train_summary": train_summary,
            "primary_eval": primary_eval,
            "cross_eval": cross_eval,
            "primary_baseline_eval": baseline_primary,
            "cross_baseline_eval": baseline_cross,
            "synthetic_dataset": str(synthetic_path),
            "primary_dataset": str(primary_material_path),
            "cross_dataset": str(cross_material_path),
        }
        primary_eval_agg[material_name] = primary_eval
        cross_eval_agg[material_name] = cross_eval
        if baseline_primary is not None:
            primary_baseline_agg[material_name] = baseline_primary
        if baseline_cross is not None:
            cross_baseline_agg[material_name] = baseline_cross

        (this_run_dir / "material_result.json").write_text(json.dumps(results[material_name], indent=2), encoding="utf-8")

    aggregate = {
        "primary_eval": _aggregate_metrics(primary_eval_agg),
        "cross_eval": _aggregate_metrics(cross_eval_agg),
        "primary_baseline_eval": _aggregate_metrics(primary_baseline_agg) if primary_baseline_agg else None,
        "cross_baseline_eval": _aggregate_metrics(cross_baseline_agg) if cross_baseline_agg else None,
    }

    summary = {
        "primary_real": args.primary_real,
        "cross_real": args.cross_real,
        "baseline_checkpoint": baseline_checkpoint,
        "materials": results,
        "aggregate": aggregate,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _report_markdown(
        output_path=docs_dir / "per_material_synth_to_real.md",
        primary_dataset=args.primary_real,
        cross_dataset=args.cross_real,
        baseline_checkpoint=baseline_checkpoint,
        material_results=results,
        aggregate_results=aggregate,
    )

    print(json.dumps({"output_root": str(output_root), "report": str(docs_dir / "per_material_synth_to_real.md"), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
