import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.models import (
    build_trial_principal_geom_features,
    exact_trial_principal_from_strain,
    spectral_decomposition_from_strain,
)
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model


def test_train_model_with_cosine_and_lbfgs(tmp_path: Path):
    dataset_path = tmp_path / "train_small.h5"
    run_dir = tmp_path / "run"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=50,
            seed=3,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )
    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="principal",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=3,
        patience=5,
        device="cpu",
        scheduler_kind="cosine",
        warmup_epochs=1,
        min_lr=1.0e-5,
        lbfgs_epochs=1,
        lbfgs_lr=0.2,
        lbfgs_max_iter=5,
        lbfgs_history_size=20,
    )
    summary = train_model(config)
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "history.csv").exists()
    assert summary["completed_epochs"] == 3
    assert summary["best_epoch"] >= 1


def test_train_model_with_plateau_scheduler_and_logging(tmp_path: Path):
    dataset_path = tmp_path / "train_small_plateau.h5"
    run_dir = tmp_path / "run_plateau"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=50,
            seed=4,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )
    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="principal",
        epochs=3,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=4,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        log_every_epochs=2,
    )
    summary = train_model(config)
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "history.csv").exists()
    assert summary["completed_epochs"] == 3
    assert summary["best_epoch"] >= 1


def test_train_trial_raw_without_branch_labels(tmp_path: Path):
    dataset_path = tmp_path / "train_no_branch.h5"
    run_dir = tmp_path / "run_trial_raw"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=40,
            seed=5,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )
    with h5py.File(dataset_path, "a") as f:
        del f["branch_id"]

    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_raw",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=5,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
    )
    summary = train_model(config)
    assert (run_dir / "best.pt").exists()
    assert summary["completed_epochs"] >= 1


def test_train_trial_raw_branch_residual_and_evaluate(tmp_path: Path):
    dataset_path = tmp_path / "train_branch_residual.h5"
    run_dir = tmp_path / "run_branch_residual"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=40,
            seed=6,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )

    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_raw_branch_residual",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=6,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
        branch_loss_weight=0.1,
    )
    summary = train_model(config)
    eval_result = evaluate_checkpoint_on_dataset(summary["best_checkpoint"], dataset_path, split="test", device="cpu")
    assert (run_dir / "best.pt").exists()
    assert "stress_mae" in eval_result["metrics"]


def test_evaluate_checkpoint_accepts_route_temperature_for_non_soft_atlas_model(tmp_path: Path):
    dataset_path = tmp_path / "train_principal_temp_eval.h5"
    run_dir = tmp_path / "run_principal_temp_eval"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=40,
            seed=7,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )

    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="principal",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=7,
        patience=5,
        device="cpu",
        scheduler_kind="cosine",
        warmup_epochs=1,
        min_lr=1.0e-5,
    )
    summary = train_model(config)
    eval_default = evaluate_checkpoint_on_dataset(summary["best_checkpoint"], dataset_path, split="test", device="cpu")
    eval_temp = evaluate_checkpoint_on_dataset(
        summary["best_checkpoint"],
        dataset_path,
        split="test",
        device="cpu",
        route_temperature=0.60,
    )

    assert eval_default["metrics"]["stress_mae"] == pytest.approx(eval_temp["metrics"]["stress_mae"])


def test_train_projected_student_and_evaluate(tmp_path: Path):
    dataset_path = tmp_path / "train_projected_student_base.h5"
    projected_dataset_path = tmp_path / "train_projected_student.h5"
    run_dir = tmp_path / "run_projected_student"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=80,
            seed=8,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )

    with h5py.File(dataset_path, "r") as src, h5py.File(projected_dataset_path, "w") as dst:
        plastic_mask = src["branch_id"][:] > 0
        for key in src.keys():
            value = src[key][:]
            if value.shape[0] == plastic_mask.shape[0]:
                value = value[plastic_mask]
            dst.create_dataset(key, data=value)
        dst.create_dataset("teacher_stress_principal", data=src["stress_principal"][:][plastic_mask])
        for key, value in src.attrs.items():
            dst.attrs[key] = value

    config = TrainingConfig(
        dataset=str(projected_dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_geom_projected_student",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=8,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
        regression_loss_kind="huber",
    )
    summary = train_model(config)
    eval_result = evaluate_checkpoint_on_dataset(summary["best_checkpoint"], projected_dataset_path, split="test", device="cpu")

    assert (run_dir / "best.pt").exists()
    assert "principal_mae" in eval_result["metrics"]
    assert "provisional_stress_principal" in eval_result["predictions"]


def test_train_projected_student_with_preservation_fields_and_sampling_weight(tmp_path: Path):
    dataset_path = tmp_path / "train_projected_student_preserve_base.h5"
    projected_dataset_path = tmp_path / "train_projected_student_preserve.h5"
    run_dir = tmp_path / "run_projected_student_preserve"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=96,
            seed=9,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )

    with h5py.File(dataset_path, "r") as src, h5py.File(projected_dataset_path, "w") as dst:
        plastic_mask = src["branch_id"][:] > 0
        for key in src.keys():
            value = src[key][:]
            if value.shape[0] == plastic_mask.shape[0]:
                value = value[plastic_mask]
            dst.create_dataset(key, data=value)

        strain_eng = src["strain_eng"][:][plastic_mask]
        material_reduced = src["material_reduced"][:][plastic_mask]
        stress_principal = src["stress_principal"][:][plastic_mask]
        trial_principal = exact_trial_principal_from_strain(strain_eng, material_reduced)
        strain_principal, _eigvecs = spectral_decomposition_from_strain(strain_eng)
        geom_features = build_trial_principal_geom_features(strain_principal, material_reduced, trial_principal)

        dst.create_dataset("teacher_provisional_stress_principal", data=stress_principal)
        dst.create_dataset("teacher_projected_stress_principal", data=stress_principal)
        dst.create_dataset("teacher_projection_delta_principal", data=np.zeros_like(stress_principal, dtype=np.float32))
        dst.create_dataset("teacher_projection_candidate_id", data=np.zeros(stress_principal.shape[0], dtype=np.int8))
        dst.create_dataset("teacher_projection_disp_norm", data=np.zeros(stress_principal.shape[0], dtype=np.float32))
        dst.create_dataset("ds_valid_mask", data=np.ones(stress_principal.shape[0], dtype=np.int8))
        dst.create_dataset("sampling_weight", data=np.linspace(1.0, 2.0, stress_principal.shape[0], dtype=np.float32))
        dst.create_dataset("trial_principal_geom_feature_f1", data=geom_features.astype(np.float32))
        for key, value in src.attrs.items():
            dst.attrs[key] = value

    config = TrainingConfig(
        dataset=str(projected_dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_geom_projected_student",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        seed=9,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
        regression_loss_kind="huber",
    )
    summary = train_model(config)
    eval_result = evaluate_checkpoint_on_dataset(summary["best_checkpoint"], projected_dataset_path, split="test", device="cpu")

    assert (run_dir / "best.pt").exists()
    assert "principal_mae" in eval_result["metrics"]
