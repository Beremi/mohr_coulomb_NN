from pathlib import Path

import h5py

from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset


def test_small_dataset_generation(tmp_path: Path):
    path = tmp_path / "small.h5"
    cfg = DatasetGenerationConfig(n_samples=20, seed=1, candidate_batch=128, include_tangent=False)
    counts = generate_branch_balanced_dataset(str(path), cfg)
    assert path.exists()
    assert sum(counts.values()) == 20
    with h5py.File(path, "r") as f:
        assert f["strain_eng"].shape == (20, 6)
        assert f["stress"].shape == (20, 6)
        assert f["branch_id"].shape == (20,)


def test_small_dataset_generation_with_principal_strain_cap(tmp_path: Path):
    path = tmp_path / "small_capped.h5"
    cfg = DatasetGenerationConfig(
        n_samples=20,
        seed=1,
        candidate_batch=128,
        include_tangent=False,
        max_abs_principal_strain=5.0e-3,
    )
    counts = generate_branch_balanced_dataset(str(path), cfg)
    assert path.exists()
    assert sum(counts.values()) == 20
    with h5py.File(path, "r") as f:
        assert f["strain_principal"].shape == (20, 3)
        assert float(abs(f["strain_principal"][:]).max()) <= 5.0e-3 + 1.0e-12
