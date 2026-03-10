#!/usr/bin/env python
"""Run pilot and repeated high-capacity surrogate experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.inference import ConstitutiveSurrogate
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model
from mc_surrogate.voigt import tensor_to_strain_voigt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="High-coverage dataset path.")
    parser.add_argument("--output-root", required=True, help="Output directory for experiments.")
    parser.add_argument("--repeats", type=int, default=12, help="Number of repeated runs after pilot selection.")
    parser.add_argument("--skip-pilot", action="store_true", help="Reuse an existing pilot selection under output-root.")
    return parser.parse_args()


def benchmark_materials() -> list[dict[str, float | str]]:
    return [
        {"name": "general_foundation", "c0": 15.0, "phi_deg": 38.0, "psi_deg": 0.0, "young": 50000.0, "poisson": 0.30},
        {"name": "weak_foundation", "c0": 10.0, "phi_deg": 35.0, "psi_deg": 0.0, "young": 50000.0, "poisson": 0.30},
        {"name": "general_slope", "c0": 18.0, "phi_deg": 32.0, "psi_deg": 0.0, "young": 20000.0, "poisson": 0.33},
        {"name": "cover_layer", "c0": 15.0, "phi_deg": 30.0, "psi_deg": 0.0, "young": 10000.0, "poisson": 0.33},
    ]


def make_principal_path(path_kind: str, n_points: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points)
    if path_kind == "triaxial":
        return np.column_stack([2.4 * t, -1.2 * t, -1.2 * t]) * 1.0e-3
    if path_kind == "left_edge":
        return np.column_stack([2.2 * t, 2.12 * t, -2.3 * t]) * 1.0e-3
    if path_kind == "right_edge":
        return np.column_stack([2.3 * t, -1.15 * t, -1.22 * t]) * 1.0e-3
    if path_kind == "apex":
        return np.column_stack([1.8 * t, 1.8 * t, 1.8 * t]) * 1.0e-3
    raise ValueError(f"Unsupported path kind {path_kind!r}.")


def evaluate_benchmark_paths(checkpoint_path: Path, device: str) -> dict[str, object]:
    path_kinds = ("triaxial", "left_edge", "right_edge", "apex")
    strength_reductions = (1.0, 1.25, 1.5, 2.0)
    n_points = 201
    surrogate = ConstitutiveSurrogate.from_checkpoint(checkpoint_path, device=device)

    rows: list[dict[str, object]] = []
    for material in benchmark_materials():
        for srf in strength_reductions:
            for path_kind in path_kinds:
                principal = make_principal_path(path_kind, n_points)
                rotations = np.repeat(np.eye(3)[None, :, :], n_points, axis=0)
                strain_eng = tensor_to_strain_voigt(np.einsum("nij,nj,nkj->nik", rotations, principal, rotations))
                reduced = build_reduced_material_from_raw(
                    c0=np.full(n_points, material["c0"]),
                    phi_rad=np.deg2rad(np.full(n_points, material["phi_deg"])),
                    psi_rad=np.deg2rad(np.full(n_points, material["psi_deg"])),
                    young=np.full(n_points, material["young"]),
                    poisson=np.full(n_points, material["poisson"]),
                    strength_reduction=np.full(n_points, srf),
                    davis_type=["B"] * n_points,
                )
                exact = constitutive_update_3d(
                    strain_eng,
                    c_bar=reduced.c_bar,
                    sin_phi=reduced.sin_phi,
                    shear=reduced.shear,
                    bulk=reduced.bulk,
                    lame=reduced.lame,
                )
                pred = surrogate.predict_reduced(
                    strain_eng,
                    c_bar=reduced.c_bar,
                    sin_phi=reduced.sin_phi,
                    shear=reduced.shear,
                    bulk=reduced.bulk,
                    lame=reduced.lame,
                )
                stress_abs_err = np.abs(pred["stress"] - exact.stress)
                principal_abs_err = np.abs(pred["stress_principal"] - exact.stress_principal)
                rows.append(
                    {
                        "material": material["name"],
                        "path_kind": path_kind,
                        "strength_reduction": srf,
                        "stress_mae": float(np.mean(stress_abs_err)),
                        "stress_rmse": float(np.sqrt(np.mean((pred["stress"] - exact.stress) ** 2))),
                        "stress_max_abs": float(np.max(stress_abs_err)),
                        "principal_mae": float(np.mean(principal_abs_err)),
                        "principal_rmse": float(np.sqrt(np.mean((pred["stress_principal"] - exact.stress_principal) ** 2))),
                        "principal_max_abs": float(np.max(principal_abs_err)),
                    }
                )

    summary = {
        "rows": rows,
        "mean_stress_mae": float(np.mean([row["stress_mae"] for row in rows])),
        "mean_principal_mae": float(np.mean([row["principal_mae"] for row in rows])),
        "worst_stress_mae": float(np.max([row["stress_mae"] for row in rows])),
        "worst_principal_mae": float(np.max([row["principal_mae"] for row in rows])),
        "worst_stress_max_abs": float(np.max([row["stress_max_abs"] for row in rows])),
        "worst_principal_max_abs": float(np.max([row["principal_max_abs"] for row in rows])),
    }
    return summary


def pilot_training_configs(dataset_path: Path, device: str) -> list[TrainingConfig]:
    common = dict(
        dataset=str(dataset_path),
        model_kind="principal",
        batch_size=2048,
        weight_decay=5.0e-5,
        dropout=0.0,
        patience=70,
        grad_clip=1.0,
        branch_loss_weight=0.05,
        num_workers=0,
        device=device,
        scheduler_kind="cosine",
        min_lr=1.0e-6,
    )
    return [
        TrainingConfig(
            run_dir="",
            epochs=240,
            lr=1.0e-3,
            width=512,
            depth=6,
            seed=101,
            warmup_epochs=12,
            lbfgs_epochs=6,
            lbfgs_lr=0.20,
            lbfgs_max_iter=25,
            lbfgs_history_size=100,
            **common,
        ),
        TrainingConfig(
            run_dir="",
            epochs=280,
            lr=1.0e-3,
            width=512,
            depth=8,
            seed=101,
            warmup_epochs=14,
            lbfgs_epochs=8,
            lbfgs_lr=0.20,
            lbfgs_max_iter=25,
            lbfgs_history_size=100,
            **common,
        ),
        TrainingConfig(
            run_dir="",
            epochs=320,
            lr=8.0e-4,
            width=768,
            depth=8,
            seed=101,
            warmup_epochs=16,
            lbfgs_epochs=10,
            lbfgs_lr=0.15,
            lbfgs_max_iter=25,
            lbfgs_history_size=100,
            **common,
        ),
    ]


def pilot_score(test_metrics: dict[str, object], benchmark_paths: dict[str, object]) -> float:
    return float(test_metrics["stress_mae"]) + float(benchmark_paths["mean_principal_mae"])


def run_single_experiment(run_root: Path, cfg: TrainingConfig) -> dict[str, object]:
    run_root.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(run_root)
    train_summary = train_model(cfg)
    dataset_eval = evaluate_checkpoint_on_dataset(run_root / "best.pt", cfg.dataset, split="test", device=cfg.device)
    benchmark_eval = evaluate_benchmark_paths(run_root / "best.pt", device=cfg.device)

    summary = {
        "training_config": asdict(cfg),
        "train_summary": train_summary,
        "test_metrics": dataset_eval["metrics"],
        "benchmark_paths": benchmark_eval,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pilot_root = output_root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    pilot_rows: list[dict[str, object]] = []
    pilot_summaries: list[dict[str, object]] = []

    if args.skip_pilot:
        for summary_path in sorted(pilot_root.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            cfg = TrainingConfig(**summary["training_config"])
            row = {
                "name": summary_path.parent.name,
                "width": cfg.width,
                "depth": cfg.depth,
                "epochs": cfg.epochs,
                "lbfgs_epochs": cfg.lbfgs_epochs,
                "seed": cfg.seed,
                "test_stress_mae": summary["test_metrics"]["stress_mae"],
                "test_stress_rmse": summary["test_metrics"]["stress_rmse"],
                "test_stress_max_abs": summary["test_metrics"]["stress_max_abs"],
                "test_principal_mae": summary["test_metrics"].get("principal_mae"),
                "branch_accuracy": summary["test_metrics"].get("branch_accuracy"),
                "benchmark_mean_principal_mae": summary["benchmark_paths"]["mean_principal_mae"],
                "benchmark_worst_principal_mae": summary["benchmark_paths"]["worst_principal_mae"],
                "pilot_score": pilot_score(summary["test_metrics"], summary["benchmark_paths"]),
            }
            pilot_rows.append(row)
            pilot_summaries.append(summary)
    else:
        for idx, cfg in enumerate(pilot_training_configs(dataset_path, device=device), start=1):
            name = f"pilot_{idx:02d}_w{cfg.width}_d{cfg.depth}"
            summary = run_single_experiment(pilot_root / name, cfg)
            row = {
                "name": name,
                "width": cfg.width,
                "depth": cfg.depth,
                "epochs": cfg.epochs,
                "lbfgs_epochs": cfg.lbfgs_epochs,
                "seed": cfg.seed,
                "test_stress_mae": summary["test_metrics"]["stress_mae"],
                "test_stress_rmse": summary["test_metrics"]["stress_rmse"],
                "test_stress_max_abs": summary["test_metrics"]["stress_max_abs"],
                "test_principal_mae": summary["test_metrics"].get("principal_mae"),
                "branch_accuracy": summary["test_metrics"].get("branch_accuracy"),
                "benchmark_mean_principal_mae": summary["benchmark_paths"]["mean_principal_mae"],
                "benchmark_worst_principal_mae": summary["benchmark_paths"]["worst_principal_mae"],
                "pilot_score": pilot_score(summary["test_metrics"], summary["benchmark_paths"]),
            }
            pilot_rows.append(row)
            pilot_summaries.append(summary)
            print(json.dumps(row, indent=2))
        write_rows_csv(pilot_root / "pilot_results.csv", pilot_rows)

    if not pilot_rows:
        raise RuntimeError("No pilot results available.")

    best_pilot = min(pilot_rows, key=lambda row: float(row["pilot_score"]))
    best_name = str(best_pilot["name"])
    best_summary = pilot_summaries[pilot_rows.index(best_pilot)]
    (output_root / "selected_pilot.json").write_text(json.dumps(best_pilot, indent=2), encoding="utf-8")

    selected_cfg_dict = best_summary["training_config"]
    repeat_root = output_root / "repeats"
    repeat_root.mkdir(parents=True, exist_ok=True)
    repeat_rows: list[dict[str, object]] = []

    base_seed = 1001
    for offset in range(args.repeats):
        seed = base_seed + offset
        cfg = TrainingConfig(**selected_cfg_dict)
        cfg.seed = seed
        name = f"seed_{seed}"
        summary = run_single_experiment(repeat_root / name, cfg)
        row = {
            "name": name,
            "seed": seed,
            "selected_from_pilot": best_name,
            "width": cfg.width,
            "depth": cfg.depth,
            "epochs": cfg.epochs,
            "lbfgs_epochs": cfg.lbfgs_epochs,
            "test_stress_mae": summary["test_metrics"]["stress_mae"],
            "test_stress_rmse": summary["test_metrics"]["stress_rmse"],
            "test_stress_max_abs": summary["test_metrics"]["stress_max_abs"],
            "test_principal_mae": summary["test_metrics"].get("principal_mae"),
            "branch_accuracy": summary["test_metrics"].get("branch_accuracy"),
            "benchmark_mean_principal_mae": summary["benchmark_paths"]["mean_principal_mae"],
            "benchmark_worst_principal_mae": summary["benchmark_paths"]["worst_principal_mae"],
            "benchmark_worst_stress_max_abs": summary["benchmark_paths"]["worst_stress_max_abs"],
        }
        repeat_rows.append(row)
        print(json.dumps(row, indent=2))

    write_rows_csv(repeat_root / "repeat_results.csv", repeat_rows)
    repeat_summary = {
        "n_runs": len(repeat_rows),
        "mean_test_stress_mae": float(np.mean([row["test_stress_mae"] for row in repeat_rows])) if repeat_rows else None,
        "std_test_stress_mae": float(np.std([row["test_stress_mae"] for row in repeat_rows])) if repeat_rows else None,
        "mean_test_stress_rmse": float(np.mean([row["test_stress_rmse"] for row in repeat_rows])) if repeat_rows else None,
        "std_test_stress_rmse": float(np.std([row["test_stress_rmse"] for row in repeat_rows])) if repeat_rows else None,
        "mean_benchmark_principal_mae": float(np.mean([row["benchmark_mean_principal_mae"] for row in repeat_rows])) if repeat_rows else None,
        "std_benchmark_principal_mae": float(np.std([row["benchmark_mean_principal_mae"] for row in repeat_rows])) if repeat_rows else None,
    }

    aggregate = {
        "dataset": str(dataset_path),
        "device": device,
        "selected_pilot": best_pilot,
        "pilot_results_csv": str(pilot_root / "pilot_results.csv"),
        "repeat_results_csv": str(repeat_root / "repeat_results.csv"),
        "repeat_summary": repeat_summary,
    }
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
