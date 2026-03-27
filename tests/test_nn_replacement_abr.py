import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.models import (
    build_model,
    build_trial_abr_features_f1,
    build_trial_abr_features_f2,
    compute_trial_abr_feature_stats,
    decode_abr_to_principal_torch,
    exact_trial_principal_from_strain,
)
from mc_surrogate.mohr_coulomb import (
    BRANCH_TO_ID,
    constitutive_update_3d,
    decode_abr_to_principal,
    encode_principal_to_abr,
    exact_trial_principal_stress_3d,
    yield_function_principal_3d,
)
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, train_model
from run_nn_replacement_abr_cycle import build_stage0_dataset


def _default_material(n: int = 1):
    return build_reduced_material_from_raw(
        c0=np.full(n, 15.0),
        phi_rad=np.deg2rad(np.full(n, 35.0)),
        psi_rad=np.deg2rad(np.full(n, 0.0)),
        young=np.full(n, 2.0e4),
        poisson=np.full(n, 0.3),
        strength_reduction=np.full(n, 1.2),
        davis_type=["B"] * n,
    )


def _build_small_real_and_panel(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "real_small.h5"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=36,
            seed=19,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )
    with h5py.File(dataset_path, "a") as f:
        n = f["strain_eng"].shape[0]
        if "source_call_id" not in f:
            f.create_dataset("source_call_id", data=np.arange(n, dtype=np.int32))
        if "source_row_in_call" not in f:
            f.create_dataset("source_row_in_call", data=np.zeros(n, dtype=np.int32))

    panel_path = tmp_path / "panel_sidecar.h5"
    with h5py.File(dataset_path, "r") as f:
        branch_id = f["branch_id"][:].astype(np.int8)
        split_id = f["split_id"][:].astype(np.int8)
    plastic = (branch_id > 0).astype(np.int8)
    hard = plastic.copy()
    with h5py.File(panel_path, "w") as f:
        f.create_dataset("plastic_mask", data=plastic)
        f.create_dataset("broad_val_mask", data=(split_id == 1).astype(np.int8))
        f.create_dataset("broad_test_mask", data=(split_id == 2).astype(np.int8))
        f.create_dataset("hard_mask", data=hard)
        f.create_dataset("hard_val_mask", data=((split_id == 1) & (hard > 0)).astype(np.int8))
        f.create_dataset("hard_test_mask", data=((split_id == 2) & (hard > 0)).astype(np.int8))
        f.create_dataset("ds_valid_mask", data=np.zeros_like(split_id, dtype=np.int8))
    return dataset_path, panel_path


def test_encode_decode_abr_round_trip_and_yield_consistency():
    sigma = np.array(
        [
            [80.0, 50.0, 20.0],
            [30.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    c_bar = np.array([25.0, 18.0], dtype=np.float32)
    sin_phi = np.array([0.35, 0.35], dtype=np.float32)
    encoded = encode_principal_to_abr(sigma, c_bar=c_bar, sin_phi=sin_phi)
    decoded = decode_abr_to_principal(encoded["abr_raw"], c_bar=c_bar, sin_phi=sin_phi)
    assert np.allclose(decoded, np.sort(sigma, axis=1)[:, ::-1], atol=1.0e-10)
    f_sigma = yield_function_principal_3d(sigma, c_bar=c_bar, sin_phi=sin_phi)
    assert np.allclose(encoded["r_raw"], -f_sigma, atol=1.0e-10)
    assert np.all(encoded["abr_nonneg"][:, :2] >= 0.0)


def test_exact_trial_principal_matches_elastic_rows():
    reduced = _default_material(2)
    strain = np.array(
        [
            [1.0e-5, 1.0e-5, 1.0e-5, 0.0, 0.0, 0.0],
            [2.0e-5, 1.0e-5, -5.0e-6, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    exact = constitutive_update_3d(
        strain,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    trial_principal = exact_trial_principal_stress_3d(
        strain,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    assert np.all(exact.branch_id == BRANCH_TO_ID["elastic"])
    assert np.allclose(trial_principal, exact.stress_principal, atol=1.0e-5)


def test_trial_abr_feature_builders_are_finite():
    reduced = _default_material(4)
    strain = np.array(
        [
            [2.0e-4, 1.0e-4, -1.0e-4, 0.0, 0.0, 0.0],
            [3.0e-4, -1.5e-4, -1.0e-4, 0.0, 0.0, 0.0],
            [4.0e-4, 2.0e-4, -2.0e-4, 0.0, 0.0, 0.0],
            [5.0e-4, 1.0e-4, -3.0e-4, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    material = np.column_stack([reduced.c_bar, reduced.sin_phi, reduced.shear, reduced.bulk, reduced.lame]).astype(np.float32)
    trial_principal = exact_trial_principal_from_strain(strain, material)
    stats = compute_trial_abr_feature_stats(trial_principal, material)
    f1 = build_trial_abr_features_f1(trial_principal, material, stats)
    f2 = build_trial_abr_features_f2(trial_principal, material, stats)
    assert f1.shape == (4, 10)
    assert f2.shape == (4, 8)
    assert np.isfinite(f1).all()
    assert np.isfinite(f2).all()


def test_acn_decoder_and_model_smoke():
    abr = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]], dtype=torch.float32)
    c_bar = torch.tensor([20.0, 20.0], dtype=torch.float32)
    sin_phi = torch.tensor([0.35, 0.35], dtype=torch.float32)
    decoded = decode_abr_to_principal_torch(abr, c_bar=c_bar, sin_phi=sin_phi)
    assert torch.all(decoded[:, 0] >= decoded[:, 1])
    assert torch.all(decoded[:, 1] >= decoded[:, 2])

    model = build_model("trial_abr_acn_f1", input_dim=10, width=32, depth=2, dropout=0.0)
    out = model(torch.randn(5, 10))
    assert out["stress"].shape == (5, 3)


def test_stage0_small_batch_dataset_smoke(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path)
    summary = build_stage0_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    derived_path = Path(summary["dataset_path"])
    assert derived_path.exists()
    assert summary["reconstruction"]["max_abs"] < 1.0e-6
    with h5py.File(derived_path, "r") as f:
        assert "feature_f1" in f
        assert "abr_nonneg" in f


def test_stage1_training_smoke_on_tiny_stage0_dataset(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data")
    summary = build_stage0_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    derived_path = Path(summary["dataset_path"])
    with h5py.File(derived_path, "a") as f:
        abr_nonneg = f["abr_nonneg"][:]
        coord_scales = {
            "scale_a": float(max(np.quantile(abr_nonneg[:, 0], 0.95), 1.0)),
            "scale_b": float(max(np.quantile(abr_nonneg[:, 1], 0.95), 1.0)),
            "scale_r": float(max(np.quantile(abr_nonneg[:, 2], 0.95), 1.0)),
        }
        f.attrs["acn_coordinate_scales_json"] = json.dumps(coord_scales)

    config = TrainingConfig(
        dataset=str(derived_path),
        run_dir=str(tmp_path / "run"),
        model_kind="trial_abr_acn_f1",
        epochs=2,
        batch_size=8,
        lr=1.0e-3,
        width=32,
        depth=2,
        seed=23,
        patience=5,
        device="cpu",
        scheduler_kind="cosine",
        warmup_epochs=1,
        min_lr=1.0e-5,
        regression_loss_kind="huber",
        snapshot_every_epochs=1,
    )
    train_summary = train_model(config)
    eval_result = evaluate_checkpoint_on_dataset(train_summary["best_checkpoint"], derived_path, split="test", device="cpu")
    assert Path(train_summary["best_checkpoint"]).exists()
    assert "stress_mae" in eval_result["metrics"]
