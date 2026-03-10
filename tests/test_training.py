from pathlib import Path

from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, train_model


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
