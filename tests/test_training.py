import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.models import (
    Standardizer,
    build_trial_principal_geom_features,
    exact_trial_principal_from_strain,
    spectral_decomposition_from_strain,
)
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import (
    TrainingConfig,
    _load_split_for_training,
    _regression_loss,
    _decode_projected_student_outputs,
    evaluate_checkpoint_on_dataset,
    train_model,
)


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


def test_load_split_for_training_handles_optional_boundary_fields(tmp_path: Path):
    dataset_path = tmp_path / "train_projected_student_schema.h5"
    projected_dataset_path = tmp_path / "train_projected_student_schema_projected.h5"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=64,
            seed=10,
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
        stress_principal = src["stress_principal"][:][plastic_mask]
        dst.create_dataset("teacher_provisional_stress_principal", data=stress_principal)
        dst.create_dataset("teacher_projected_stress_principal", data=stress_principal)
        dst.create_dataset("teacher_projection_delta_principal", data=np.zeros_like(stress_principal, dtype=np.float32))
        dst.create_dataset("teacher_projection_candidate_id", data=np.zeros(stress_principal.shape[0], dtype=np.int8))
        dst.create_dataset("teacher_projection_disp_norm", data=np.linspace(0.0, 4.0, stress_principal.shape[0], dtype=np.float32))
        dst.create_dataset("sampling_weight", data=np.ones(stress_principal.shape[0], dtype=np.float32))
        dst.create_dataset(
            "near_yield_mask",
            data=np.asarray(np.arange(stress_principal.shape[0]) % 2 == 0, dtype=np.int8),
        )
        dst.create_dataset(
            "near_smooth_left_mask",
            data=np.asarray(np.arange(stress_principal.shape[0]) % 3 == 0, dtype=np.int8),
        )
        dst.create_dataset("any_boundary_mask", data=np.asarray(np.arange(stress_principal.shape[0]) % 4 == 0, dtype=np.int8))
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["high_disp_focus_threshold"] = 2.0

    split_with_masks = _load_split_for_training(
        str(projected_dataset_path),
        "train",
        "trial_principal_geom_projected_student",
    )
    assert np.any(split_with_masks["any_boundary_mask"])
    assert np.any(split_with_masks["high_disp_mask"])

    with h5py.File(projected_dataset_path, "a") as dst:
        del dst["near_yield_mask"]
        del dst["near_smooth_left_mask"]
        del dst["any_boundary_mask"]

    split_without_masks = _load_split_for_training(
        str(projected_dataset_path),
        "train",
        "trial_principal_geom_projected_student",
    )
    assert split_without_masks["any_boundary_mask"].shape[0] == split_with_masks["any_boundary_mask"].shape[0]
    assert not np.any(split_without_masks["any_boundary_mask"])


def test_projected_student_regression_loss_defaults_match_preservation_formula():
    y_scaler = Standardizer.identity(3)
    pred_norm = torch.tensor(
        [
            [0.4, -0.2, 0.1],
            [0.1, 0.3, -0.1],
        ],
        dtype=torch.float32,
    )
    branch_true = torch.tensor([1, 2], dtype=torch.int64)
    eigvecs = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    trial_principal = torch.tensor(
        [
            [4.0, 2.0, 0.0],
            [3.5, 1.5, -0.2],
        ],
        dtype=torch.float32,
    )
    trial_stress = torch.zeros((2, 6), dtype=torch.float32)
    stress_true = torch.tensor(
        [
            [2.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.8, 0.9, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    stress_principal_true = torch.tensor(
        [
            [2.0, 1.0, 0.5],
            [1.8, 0.9, 0.4],
        ],
        dtype=torch.float32,
    )
    material_reduced = torch.tensor(
        [
            [1.0, 0.2, 1.0, 1.0, 1.0],
            [1.0, 0.2, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    teacher_projected = stress_principal_true + 0.1
    teacher_provisional = stress_principal_true + 0.2
    teacher_delta = teacher_projected - teacher_provisional
    branch_logits = torch.tensor([[2.0, 0.1, -1.0, -2.0], [0.2, 1.2, -0.5, -1.0]], dtype=torch.float32)

    total, metrics = _regression_loss(
        model_kind="trial_principal_geom_projected_student",
        pred_norm=pred_norm,
        target_norm=pred_norm,
        y_scaler=y_scaler,
        xb=None,
        x_scaler=None,
        model=None,
        branch_true=branch_true,
        eigvecs=eigvecs,
        stress_true=stress_true,
        stress_principal_true=stress_principal_true,
        trial_stress=trial_stress,
        trial_principal=trial_principal,
        abr_true_nonneg=torch.full((2, 3), float("nan"), dtype=torch.float32),
        grho_true=torch.full((2, 2), float("nan"), dtype=torch.float32),
        soft_route_target=None,
        material_reduced=material_reduced,
        teacher_provisional_stress_principal=teacher_provisional,
        teacher_projected_stress_principal=teacher_projected,
        teacher_projection_delta_principal=teacher_delta,
        hard_mask=torch.tensor([1, 1], dtype=torch.int8),
        any_boundary_mask=torch.tensor([0, 0], dtype=torch.int8),
        high_disp_mask=torch.tensor([0, 0], dtype=torch.int8),
        teacher_projection_candidate_id=torch.tensor([0, 0], dtype=torch.int64),
        branch_logits=branch_logits,
        stress_weight_alpha=0.0,
        stress_weight_scale=250.0,
        regression_loss_kind="huber",
        huber_delta=1.0,
        voigt_mae_weight=0.0,
    )
    _stress_pred, principal_pred, provisional = _decode_projected_student_outputs(
        pred_norm,
        y_scaler,
        material_reduced,
        eigvecs,
        trial_stress,
        trial_principal,
        projection_mode="exact",
        projection_tau=0.05,
    )
    branch_target = branch_true - 1
    manual = (
        1.00 * torch.nn.functional.huber_loss(principal_pred, stress_principal_true, delta=1.0)
        + 0.75 * torch.nn.functional.huber_loss(principal_pred, teacher_projected, delta=1.0)
        + 0.50 * torch.nn.functional.huber_loss(provisional, teacher_provisional, delta=1.0)
        + 0.25 * torch.nn.functional.huber_loss(principal_pred - provisional, teacher_delta, delta=1.0)
        + 0.10 * torch.nn.functional.cross_entropy(branch_logits, branch_target)
    )
    assert float(total.detach().cpu()) == pytest.approx(float(manual.detach().cpu()))
    assert metrics["hard_quantile_huber_loss"] == pytest.approx(0.0)


def test_projected_student_regression_loss_tail_focus_is_finite():
    y_scaler = Standardizer.identity(3)
    pred_norm = torch.tensor(
        [
            [0.6, -0.1, 0.2],
            [0.1, 0.4, -0.2],
            [1.2, -0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    branch_true = torch.tensor([1, 2, 3], dtype=torch.int64)
    eigvecs = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    trial_principal = torch.tensor(
        [
            [4.0, 2.0, 0.0],
            [3.5, 1.5, -0.2],
            [4.5, 2.5, 0.1],
        ],
        dtype=torch.float32,
    )
    trial_stress = torch.zeros((3, 6), dtype=torch.float32)
    stress_true = torch.tensor(
        [
            [2.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.8, 0.9, 0.4, 0.0, 0.0, 0.0],
            [2.4, 1.2, 0.6, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    stress_principal_true = torch.tensor(
        [
            [2.0, 1.0, 0.5],
            [1.8, 0.9, 0.4],
            [2.4, 1.2, 0.6],
        ],
        dtype=torch.float32,
    )
    material_reduced = torch.tensor(
        [
            [1.0, 0.2, 1.0, 1.0, 1.0],
            [1.0, 0.2, 1.0, 1.0, 1.0],
            [1.0, 0.2, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    teacher_projected = stress_principal_true + 0.15
    teacher_provisional = stress_principal_true + 0.30
    teacher_delta = teacher_projected - teacher_provisional
    branch_logits = torch.tensor(
        [[2.0, 0.1, -1.0, -2.0], [0.2, 1.2, -0.5, -1.0], [0.1, -0.2, 1.8, -1.0]],
        dtype=torch.float32,
    )

    total, metrics = _regression_loss(
        model_kind="trial_principal_geom_projected_student",
        pred_norm=pred_norm,
        target_norm=pred_norm,
        y_scaler=y_scaler,
        xb=None,
        x_scaler=None,
        model=None,
        branch_true=branch_true,
        eigvecs=eigvecs,
        stress_true=stress_true,
        stress_principal_true=stress_principal_true,
        trial_stress=trial_stress,
        trial_principal=trial_principal,
        abr_true_nonneg=torch.full((3, 3), float("nan"), dtype=torch.float32),
        grho_true=torch.full((3, 2), float("nan"), dtype=torch.float32),
        soft_route_target=None,
        material_reduced=material_reduced,
        teacher_provisional_stress_principal=teacher_provisional,
        teacher_projected_stress_principal=teacher_projected,
        teacher_projection_delta_principal=teacher_delta,
        hard_mask=torch.tensor([1, 0, 1], dtype=torch.int8),
        any_boundary_mask=torch.tensor([0, 1, 1], dtype=torch.int8),
        high_disp_mask=torch.tensor([0, 0, 1], dtype=torch.int8),
        teacher_projection_candidate_id=torch.tensor([0, 2, 2], dtype=torch.int64),
        branch_logits=branch_logits,
        stress_weight_alpha=0.0,
        stress_weight_scale=250.0,
        regression_loss_kind="huber",
        huber_delta=1.0,
        voigt_mae_weight=0.0,
        projected_student_any_boundary_loss_multiplier=1.5,
        projected_student_high_disp_loss_multiplier=1.5,
        projected_student_candidate_loss_weights={2: 1.75},
        projected_student_teacher_alignment_focus_multiplier=1.5,
        projected_student_hard_quantile=0.5,
        projected_student_hard_quantile_weight=0.30,
    )
    assert np.isfinite(float(total.detach().cpu()))
    assert metrics["projected_student_focus_fraction"] > 0.0
    assert metrics["hard_quantile_huber_loss"] >= 0.0
