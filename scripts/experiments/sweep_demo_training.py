#!/usr/bin/env python
"""Run a small sweep of dataset/training configurations for the demo notebook."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.sampling import DatasetGenerationConfig, MaterialRangeConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model
from mc_surrogate.voigt import tensor_to_strain_voigt


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: DatasetGenerationConfig
    training: TrainingConfig


def focused_material_ranges() -> MaterialRangeConfig:
    return MaterialRangeConfig(
        cohesion_range=(5.0, 25.0),
        friction_deg_range=(25.0, 45.0),
        dilatancy_deg_range=(0.0, 10.0),
        young_range=(1.0e4, 6.0e4),
        poisson_range=(0.25, 0.38),
        strength_reduction_range=(0.9, 2.0),
    )


def summarize_dataset(dataset_path: Path) -> dict[str, float | list[float]]:
    with h5py.File(dataset_path, "r") as f:
        stress = f["stress"][:]
        strain_principal = f["strain_principal"][:]
    rowmax_stress = np.max(np.abs(stress), axis=1)
    return {
        "stress_rowmax_quantiles": np.quantile(rowmax_stress, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist(),
        "max_abs_principal_strain": float(np.max(np.abs(strain_principal))),
    }


def path_principal_mae(checkpoint_path: Path, device: str) -> float:
    from mc_surrogate.inference import ConstitutiveSurrogate

    t = np.linspace(0.0, 1.0, 200)
    principal_strain = np.column_stack([2.0 * t, -1.0 * t, -1.0 * t]) * 1.0e-3
    rotations = np.repeat(np.eye(3)[None, :, :], t.size, axis=0)
    strain_eng = tensor_to_strain_voigt(np.einsum("nij,nj,nkj->nik", rotations, principal_strain, rotations))
    reduced = build_reduced_material_from_raw(
        c0=np.full(t.size, 15.0),
        phi_rad=np.deg2rad(np.full(t.size, 35.0)),
        psi_rad=np.deg2rad(np.full(t.size, 0.0)),
        young=np.full(t.size, 2.0e4),
        poisson=np.full(t.size, 0.30),
        strength_reduction=np.full(t.size, 1.2),
        davis_type=["B"] * t.size,
    )
    exact = constitutive_update_3d(
        strain_eng,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    surrogate = ConstitutiveSurrogate.from_checkpoint(checkpoint_path, device=device)
    pred = surrogate.predict_reduced(
        strain_eng,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    return float(np.mean(np.abs(pred["stress_principal"] - exact.stress_principal)))


def build_experiments(device: str) -> list[ExperimentSpec]:
    base_seed = 7
    focused = focused_material_ranges()
    return [
        ExperimentSpec(
            name="baseline_uncapped_small",
            dataset=DatasetGenerationConfig(
                n_samples=1000,
                seed=base_seed,
                candidate_batch=512,
                max_abs_principal_strain=None,
            ),
            training=TrainingConfig(
                dataset="",
                run_dir="",
                model_kind="principal",
                epochs=12,
                batch_size=256,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                width=128,
                depth=3,
                dropout=0.0,
                seed=base_seed,
                patience=6,
                grad_clip=1.0,
                branch_loss_weight=0.1,
                num_workers=0,
                device=device,
            ),
        ),
        ExperimentSpec(
            name="wide_capped_medium",
            dataset=DatasetGenerationConfig(
                n_samples=3000,
                seed=base_seed,
                candidate_batch=1024,
                max_abs_principal_strain=5.0e-3,
            ),
            training=TrainingConfig(
                dataset="",
                run_dir="",
                model_kind="principal",
                epochs=40,
                batch_size=256,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                width=128,
                depth=3,
                dropout=0.0,
                seed=base_seed,
                patience=10,
                grad_clip=1.0,
                branch_loss_weight=0.1,
                num_workers=0,
                device=device,
            ),
        ),
        ExperimentSpec(
            name="wide_capped_large",
            dataset=DatasetGenerationConfig(
                n_samples=5000,
                seed=base_seed,
                candidate_batch=1024,
                max_abs_principal_strain=5.0e-3,
            ),
            training=TrainingConfig(
                dataset="",
                run_dir="",
                model_kind="principal",
                epochs=60,
                batch_size=256,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                width=256,
                depth=4,
                dropout=0.0,
                seed=base_seed,
                patience=12,
                grad_clip=1.0,
                branch_loss_weight=0.1,
                num_workers=0,
                device=device,
            ),
        ),
        ExperimentSpec(
            name="focused_capped_large",
            dataset=DatasetGenerationConfig(
                n_samples=5000,
                seed=base_seed,
                candidate_batch=1024,
                max_abs_principal_strain=2.0e-3,
                material_ranges=focused,
            ),
            training=TrainingConfig(
                dataset="",
                run_dir="",
                model_kind="principal",
                epochs=60,
                batch_size=256,
                lr=1.0e-3,
                weight_decay=1.0e-4,
                width=256,
                depth=4,
                dropout=0.0,
                seed=base_seed,
                patience=12,
                grad_clip=1.0,
                branch_loss_weight=0.1,
                num_workers=0,
                device=device,
            ),
        ),
        ExperimentSpec(
            name="focused_capped_xlarge",
            dataset=DatasetGenerationConfig(
                n_samples=7000,
                seed=base_seed,
                candidate_batch=1536,
                max_abs_principal_strain=2.0e-3,
                material_ranges=focused,
            ),
            training=TrainingConfig(
                dataset="",
                run_dir="",
                model_kind="principal",
                epochs=80,
                batch_size=256,
                lr=8.0e-4,
                weight_decay=1.0e-4,
                width=384,
                depth=5,
                dropout=0.0,
                seed=base_seed,
                patience=16,
                grad_clip=1.0,
                branch_loss_weight=0.1,
                num_workers=0,
                device=device,
            ),
        ),
    ]


def run() -> None:
    output_root = ROOT / "experiment_runs" / "demo_sweep"
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = build_experiments(device=device)
    aggregate_rows: list[dict[str, object]] = []

    for spec in experiments:
        exp_root = output_root / spec.name
        exp_root.mkdir(parents=True, exist_ok=True)
        dataset_path = exp_root / "dataset.h5"
        run_dir = exp_root / "run"

        print(f"[{spec.name}] generating dataset")
        generate_branch_balanced_dataset(str(dataset_path), spec.dataset)
        dataset_stats = summarize_dataset(dataset_path)

        train_cfg = TrainingConfig(**asdict(spec.training))
        train_cfg.dataset = str(dataset_path)
        train_cfg.run_dir = str(run_dir)

        print(f"[{spec.name}] training model")
        train_summary = train_model(train_cfg)

        print(f"[{spec.name}] evaluating checkpoint")
        evaluation = evaluate_checkpoint_on_dataset(run_dir / "best.pt", dataset_path, split="test", device=device)
        metrics = evaluation["metrics"]
        triaxial_mae = path_principal_mae(run_dir / "best.pt", device=device)

        summary = {
            "name": spec.name,
            "dataset_config": asdict(spec.dataset),
            "training_config": asdict(train_cfg),
            "dataset_stats": dataset_stats,
            "train_summary": train_summary,
            "metrics": metrics,
            "triaxial_principal_mae": triaxial_mae,
        }
        (exp_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        row = {
            "name": spec.name,
            "n_samples": spec.dataset.n_samples,
            "max_abs_principal_strain": spec.dataset.max_abs_principal_strain,
            "width": train_cfg.width,
            "depth": train_cfg.depth,
            "epochs": train_cfg.epochs,
            "stress_mae": metrics["stress_mae"],
            "stress_rmse": metrics["stress_rmse"],
            "stress_max_abs": metrics["stress_max_abs"],
            "principal_mae": metrics.get("principal_mae"),
            "branch_accuracy": metrics.get("branch_accuracy"),
            "triaxial_principal_mae": triaxial_mae,
            "dataset_stress_q99": dataset_stats["stress_rowmax_quantiles"][3],
            "dataset_stress_q100": dataset_stats["stress_rowmax_quantiles"][4],
        }
        aggregate_rows.append(row)
        print(json.dumps(row, indent=2))

    results_csv = output_root / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    best = min(aggregate_rows, key=lambda row: float(row["stress_mae"]))
    print("\nBest experiment:")
    print(json.dumps(best, indent=2))
    print(f"\nAggregate results: {results_csv}")


if __name__ == "__main__":
    run()
