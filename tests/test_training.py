from pathlib import Path

import h5py

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
