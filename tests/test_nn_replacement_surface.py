import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.models import (
    Standardizer,
    build_model,
    build_trial_principal_surface_features,
    build_trial_soft_atlas_features_f1,
    build_trial_surface_features_f1,
    compute_trial_soft_atlas_feature_stats,
    compute_trial_surface_feature_stats,
    decode_grho_to_principal_torch,
    exact_trial_principal_from_strain,
    spectral_decomposition_from_strain,
)
from mc_surrogate.mohr_coulomb import (
    BRANCH_TO_ID,
    constitutive_update_3d,
    decode_abr_to_principal,
    decode_branch_specialized_grho_to_principal,
    decode_grho_to_principal,
    decode_grho_to_principal_plastic,
    encode_principal_to_grho,
    exact_trial_principal_stress_3d,
    project_grho_to_branch_specialized,
    yield_function_principal_3d,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, predict_with_checkpoint, train_model
from run_nn_replacement_surface_cycle import build_stage0_surface_dataset
from run_nn_replacement_surface_branch_structured_cycle import build_stage0_branch_structured_audit
from run_nn_replacement_surface_soft_atlas_cycle import (
    _soft_atlas_decode_from_chart,
    _select_best_temperature_row,
    build_stage0_soft_atlas_dataset,
    evaluate_soft_atlas_checkpoint,
)


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
            n_samples=48,
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


def _write_surface_checkpoint(
    checkpoint_path: Path,
    *,
    input_dim: int,
    feature_stats: dict[str, object],
    coordinate_scales: dict[str, float],
) -> Path:
    model = build_model("trial_surface_acn_f1", input_dim=input_dim, width=8, depth=1, dropout=0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "config": {
                "model_kind": "trial_surface_acn_f1",
                "width": 8,
                "depth": 1,
                "dropout": 0.0,
            },
            "x_scaler": Standardizer.identity(input_dim).to_dict(),
            "y_scaler": Standardizer.identity(2).to_dict(),
            "feature_stats": feature_stats,
            "coordinate_scales": coordinate_scales,
            "branch_names": ["elastic", "smooth", "left_edge", "right_edge", "apex"],
            "branch_head_kind": "full",
            "elastic_handling": "exact_upstream",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _write_branch_structured_surface_checkpoint(
    checkpoint_path: Path,
    *,
    input_dim: int,
    feature_stats: dict[str, object],
    coordinate_scales: dict[str, float],
) -> Path:
    model = build_model("trial_surface_branch_structured_f1", input_dim=input_dim, width=8, depth=1, dropout=0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "config": {
                "model_kind": "trial_surface_branch_structured_f1",
                "width": 8,
                "depth": 1,
                "dropout": 0.0,
            },
            "x_scaler": Standardizer.identity(input_dim).to_dict(),
            "y_scaler": Standardizer.identity(4).to_dict(),
            "feature_stats": feature_stats,
            "coordinate_scales": coordinate_scales,
            "branch_names": ["elastic", "smooth", "left_edge", "right_edge", "apex"],
            "plastic_branch_names": ["smooth", "left_edge", "right_edge", "apex"],
            "branch_coordinate_scheme": "branch_specialized",
            "branch_head_kind": "learned_plastic_4way",
            "elastic_handling": "exact_upstream",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _write_soft_atlas_checkpoint(
    checkpoint_path: Path,
    *,
    model_kind: str,
    input_dim: int,
    feature_stats: dict[str, object],
    route_bias: list[float] | None = None,
) -> Path:
    model = build_model(model_kind, input_dim=input_dim, width=8, depth=1, dropout=0.0)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        if route_bias is not None and hasattr(model, "head_route"):
            model.head_route.bias.copy_(torch.tensor(route_bias, dtype=torch.float32))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "config": {
                "model_kind": model_kind,
                "width": 8,
                "depth": 1,
                "dropout": 0.0,
            },
            "x_scaler": Standardizer.identity(input_dim).to_dict(),
            "y_scaler": Standardizer.identity(4).to_dict(),
            "feature_stats": feature_stats,
            "coordinate_scales": None,
            "branch_names": ["elastic", "smooth", "left_edge", "right_edge", "apex"],
            "plastic_branch_names": ["smooth", "left_edge", "right_edge", "apex"],
            "branch_coordinate_scheme": "soft_atlas",
            "branch_head_kind": "soft_atlas",
            "elastic_handling": "exact_upstream",
            "soft_atlas_route_temperature": 1.0,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def test_encode_decode_grho_round_trip_on_plastic_states(tmp_path: Path):
    dataset_path, _panel_path = _build_small_real_and_panel(tmp_path)
    with h5py.File(dataset_path, "r") as f:
        branch_id = f["branch_id"][:]
        plastic = branch_id > 0
        stress_principal = f["stress_principal"][:][plastic]
        material_reduced = f["material_reduced"][:][plastic]

    encoded = encode_principal_to_grho(stress_principal, c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1])
    decoded = decode_grho_to_principal(encoded["grho"], c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1])
    decoded_alias = decode_grho_to_principal_plastic(encoded["grho"], c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1])

    assert np.allclose(decoded, stress_principal, atol=1.0e-10)
    assert np.allclose(decoded_alias, stress_principal, atol=1.0e-10)
    assert np.all(encoded["g"] >= -1.0e-12)
    assert np.all(np.abs(encoded["rho"]) <= 1.0 + 1.0e-12)
    assert np.all(yield_function_principal_3d(decoded, c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1]) <= 1.0e-10)


def test_branch_specialized_surface_projection_and_decode(tmp_path: Path):
    dataset_path, _panel_path = _build_small_real_and_panel(tmp_path)
    with h5py.File(dataset_path, "r") as f:
        branch_id = f["branch_id"][:]
        plastic = branch_id > 0
        stress_principal = f["stress_principal"][:][plastic]
        material_reduced = f["material_reduced"][:][plastic]
        grho = encode_principal_to_grho(
            stress_principal,
            c_bar=material_reduced[:, 0],
            sin_phi=material_reduced[:, 1],
        )["grho"]
        branch_plastic = branch_id[plastic]

    projected = project_grho_to_branch_specialized(grho, branch_plastic)
    decoded = decode_branch_specialized_grho_to_principal(
        projected,
        branch_plastic,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
    )

    assert np.allclose(decoded, stress_principal, atol=1.0e-6)
    assert np.all(projected[branch_plastic == BRANCH_TO_ID["left_edge"], 1] == -1.0)
    assert np.all(projected[branch_plastic == BRANCH_TO_ID["right_edge"], 1] == 1.0)
    assert np.all(projected[branch_plastic == BRANCH_TO_ID["apex"]] == 0.0)
    assert np.all(yield_function_principal_3d(decoded, c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1]) <= 1.0e-8)


def test_surface_feature_builders_are_finite():
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
    strain_principal, _eigvecs = spectral_decomposition_from_strain(strain)
    stats = compute_trial_surface_feature_stats(trial_principal, material)
    f1 = build_trial_surface_features_f1(strain_principal, material, trial_principal, stats)
    alias = build_trial_principal_surface_features(strain_principal, material, trial_principal, stats)

    assert f1.shape == alias.shape
    assert f1.shape[0] == 4
    assert f1.shape[1] >= 10
    assert np.isfinite(f1).all()
    assert np.allclose(f1, alias)


def test_surface_decoder_and_model_smoke():
    grho = torch.tensor([[10.0, 0.2], [0.0, 0.0]], dtype=torch.float32)
    c_bar = torch.tensor([20.0, 20.0], dtype=torch.float32)
    sin_phi = torch.tensor([0.35, 0.35], dtype=torch.float32)
    decoded = decode_grho_to_principal_torch(grho, c_bar=c_bar, sin_phi=sin_phi)
    assert torch.all(decoded[:, 0] >= decoded[:, 1])
    assert torch.all(decoded[:, 1] >= decoded[:, 2])

    model = build_model("trial_surface_acn_f1", input_dim=18, width=32, depth=2, dropout=0.0)
    out = model(torch.randn(5, 18))
    assert out["stress"].shape == (5, 2)

    branch_model = build_model("trial_surface_branch_structured_f1", input_dim=18, width=32, depth=2, dropout=0.0)
    branch_out = branch_model(torch.randn(5, 18))
    assert branch_out["stress"].shape == (5, 4)
    assert branch_out["branch_logits"].shape == (5, 4)

    for model_kind in ("trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"):
        soft_model = build_model(model_kind, input_dim=17, width=32, depth=2, dropout=0.0)
        soft_out = soft_model(torch.randn(5, 17))
        assert soft_out["stress"].shape == (5, 4)
        assert soft_out["branch_logits"].shape == (5, 4)
        assert soft_out["route_logits"].shape == (5, 4)


def test_surface_prediction_keeps_elastic_rows_exact(tmp_path: Path):
    dataset_path, _panel_path = _build_small_real_and_panel(tmp_path / "data")
    with h5py.File(dataset_path, "r") as f:
        strain_eng = f["strain_eng"][:]
        material_reduced = f["material_reduced"][:]
        stress = f["stress"][:]
        stress_principal = f["stress_principal"][:]
        branch_id = f["branch_id"][:]
        split_id = f["split_id"][:]

    elastic_idx = np.flatnonzero(branch_id == BRANCH_TO_ID["elastic"])[0:1]
    trial_principal = exact_trial_principal_stress_3d(
        strain_eng,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        shear=material_reduced[:, 2],
        bulk=material_reduced[:, 3],
        lame=material_reduced[:, 4],
    ).astype(np.float32)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(strain_eng)
    stats = compute_trial_surface_feature_stats(trial_principal[split_id == 0], material_reduced[split_id == 0])
    features = build_trial_surface_features_f1(strain_principal, material_reduced, trial_principal, stats)
    encoded = encode_principal_to_grho(stress_principal[split_id == 0], c_bar=material_reduced[split_id == 0, 0], sin_phi=material_reduced[split_id == 0, 1])
    c_bar_safe = np.maximum(material_reduced[split_id == 0, 0], 1.0e-6)
    g_tilde = ((1.0 + material_reduced[split_id == 0, 1]) * encoded["g"]) / c_bar_safe
    coordinate_scales = {"scale_g": float(max(np.quantile(np.maximum(g_tilde, 0.0), 0.95), 1.0))}
    checkpoint_path = _write_surface_checkpoint(
        tmp_path / "surface_checkpoint.pt",
        input_dim=features.shape[1],
        feature_stats=stats,
        coordinate_scales=coordinate_scales,
    )

    pred = predict_with_checkpoint(
        checkpoint_path,
        strain_eng[elastic_idx],
        material_reduced[elastic_idx],
        device="cpu",
        batch_size=1,
    )
    exact = constitutive_update_3d(
        strain_eng[elastic_idx],
        c_bar=material_reduced[elastic_idx, 0],
        sin_phi=material_reduced[elastic_idx, 1],
        shear=material_reduced[elastic_idx, 2],
        bulk=material_reduced[elastic_idx, 3],
        lame=material_reduced[elastic_idx, 4],
    )

    assert np.allclose(pred["stress"], exact.stress.astype(np.float32), atol=1.0e-6)
    assert np.allclose(pred["stress_principal"], exact.stress_principal.astype(np.float32), atol=1.0e-6)


def test_stage0_surface_dataset_smoke(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path)
    summary = build_stage0_surface_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    derived_path = Path(summary["dataset_path"])
    assert derived_path.exists()
    assert summary["reconstruction"]["plastic_max_abs"] < 1.0e-6
    with h5py.File(derived_path, "r") as f:
        assert "surface_feature_f1" in f
        assert "grho" in f
        assert json.loads(f.attrs["surface_feature_stats_json"])
        assert json.loads(f.attrs["surface_coordinate_scales_json"])


def test_stage0_branch_structured_surface_audit_smoke(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data")
    stage0_summary = build_stage0_surface_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    audit = build_stage0_branch_structured_audit(
        stage0_dataset_path=Path(stage0_summary["dataset_path"]),
        output_dir=tmp_path / "stage0_branch_structured",
        force_rerun=True,
    )
    assert audit["stop_rules"]["plastic_reconstruction_max_le_5e4"]
    assert audit["stop_rules"]["plastic_yield_max_le_1e6"]
    assert audit["stop_rules"]["apex_reconstruction_max_le_1e6"]
    assert audit["stop_rules"]["apex_yield_max_le_1e6"]


def test_stage1_surface_training_smoke_on_tiny_stage0_dataset(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data")
    summary = build_stage0_surface_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    config = TrainingConfig(
        dataset=summary["dataset_path"],
        run_dir=str(tmp_path / "run"),
        model_kind="trial_surface_acn_f1",
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
    eval_result = evaluate_checkpoint_on_dataset(train_summary["best_checkpoint"], summary["dataset_path"], split="test", device="cpu")
    assert Path(train_summary["best_checkpoint"]).exists()
    assert "stress_mae" in eval_result["metrics"]


def test_branch_structured_surface_prediction_metadata_and_exact_elastic_override(tmp_path: Path):
    dataset_path, _panel_path = _build_small_real_and_panel(tmp_path / "data")
    with h5py.File(dataset_path, "r") as f:
        strain_eng = f["strain_eng"][:]
        material_reduced = f["material_reduced"][:]
        branch_id = f["branch_id"][:]
        split_id = f["split_id"][:]

    keep = np.concatenate([np.flatnonzero(branch_id == BRANCH_TO_ID["elastic"])[0:1], np.flatnonzero(branch_id > 0)[0:3]])
    trial_principal = exact_trial_principal_stress_3d(
        strain_eng,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        shear=material_reduced[:, 2],
        bulk=material_reduced[:, 3],
        lame=material_reduced[:, 4],
    ).astype(np.float32)
    strain_principal, _eigvecs = spectral_decomposition_from_strain(strain_eng)
    stats = compute_trial_surface_feature_stats(trial_principal[split_id == 0], material_reduced[split_id == 0])
    features = build_trial_surface_features_f1(strain_principal, material_reduced, trial_principal, stats)
    encoded = encode_principal_to_grho(
        trial_principal[split_id == 0],
        c_bar=material_reduced[split_id == 0, 0],
        sin_phi=material_reduced[split_id == 0, 1],
    )
    coordinate_scales = {
        "scale_g_smooth": 1.0,
        "scale_g_left": 1.0,
        "scale_g_right": 1.0,
    }
    checkpoint_path = _write_branch_structured_surface_checkpoint(
        tmp_path / "surface_branch_structured_checkpoint.pt",
        input_dim=features.shape[1],
        feature_stats=stats,
        coordinate_scales=coordinate_scales,
    )

    pred = predict_with_checkpoint(
        checkpoint_path,
        strain_eng[keep],
        material_reduced[keep],
        device="cpu",
        batch_size=2,
        branch_override=branch_id[keep],
    )
    exact = constitutive_update_3d(
        strain_eng[keep],
        c_bar=material_reduced[keep, 0],
        sin_phi=material_reduced[keep, 1],
        shear=material_reduced[keep, 2],
        bulk=material_reduced[keep, 3],
        lame=material_reduced[keep, 4],
    )

    assert pred["branch_probabilities"].shape == (keep.shape[0], 5)
    assert pred["predicted_branch_id"].shape == (keep.shape[0],)
    assert np.array_equal(pred["route_branch_id"], branch_id[keep])
    assert np.allclose(pred["stress"][0:1], exact.stress.astype(np.float32)[0:1], atol=1.0e-6)
    assert np.allclose(pred["stress_principal"][0:1], exact.stress_principal.astype(np.float32)[0:1], atol=1.0e-6)


def test_stage1_branch_structured_surface_training_smoke_on_tiny_stage0_dataset(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data")
    summary = build_stage0_surface_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    config = TrainingConfig(
        dataset=summary["dataset_path"],
        run_dir=str(tmp_path / "run"),
        model_kind="trial_surface_branch_structured_f1",
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
        branch_loss_weight=0.1,
    )
    train_summary = train_model(config)
    eval_result = evaluate_checkpoint_on_dataset(train_summary["best_checkpoint"], summary["dataset_path"], split="test", device="cpu")
    assert Path(train_summary["best_checkpoint"]).exists()
    assert "stress_mae" in eval_result["metrics"]


def _build_tiny_packet4_stage0(tmp_path: Path) -> tuple[dict[str, object], Path]:
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data")
    packet2 = build_stage0_surface_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0_packet2",
        force_rerun=True,
    )
    packet4 = build_stage0_soft_atlas_dataset(
        stage0_dataset_path=Path(packet2["dataset_path"]),
        output_dir=tmp_path / "stage0_packet4",
        force_rerun=True,
    )
    return packet4, Path(packet4["dataset_path"])


def test_stage0_packet4_soft_atlas_dataset_smoke(tmp_path: Path):
    summary, derived_path = _build_tiny_packet4_stage0(tmp_path)
    assert derived_path.exists()
    assert summary["reconstruction"]["plastic_max_abs"] < 1.0e-5
    assert summary["yield"]["plastic_max"] < 1.0e-6
    assert summary["stage0_pass"] is True
    route_target_summary_path = derived_path.parent / "stage0_route_target_summary.json"
    feature_stats_summary_path = derived_path.parent / "stage0_feature_stats.json"
    assert route_target_summary_path.exists()
    assert feature_stats_summary_path.exists()
    route_target_summary = json.loads(route_target_summary_path.read_text(encoding="utf-8"))
    feature_stats_summary = json.loads(feature_stats_summary_path.read_text(encoding="utf-8"))
    assert route_target_summary["checks"]["probability_vectors_valid"] is True
    assert route_target_summary["checks"]["forbidden_smooth_apex_mixing_absent"] is True
    assert feature_stats_summary["checks"]["train_plastic_all_finite"] is True
    assert feature_stats_summary["checks"]["full_dataset_all_finite"] is True
    with h5py.File(derived_path, "r") as f:
        assert "soft_atlas_feature_f1" in f
        assert "soft_atlas_route_target" in f
        assert "soft_atlas_target" in f
        assert json.loads(f.attrs["soft_atlas_feature_stats_json"])
        assert json.loads(f.attrs["soft_atlas_target_stats_json"])


def test_packet4_soft_atlas_target_round_trip_and_admissibility(tmp_path: Path):
    summary, derived_path = _build_tiny_packet4_stage0(tmp_path)
    with h5py.File(derived_path, "r") as f:
        branch_id = f["branch_id"][:]
        plastic = branch_id > 0
        stress_principal_true = f["stress_principal"][:][plastic]
        trial_principal = f["trial_principal"][:][plastic]
        material_reduced = f["material_reduced"][:][plastic]
        soft_target = f["soft_atlas_target"][:][plastic]

    a_tr = trial_principal[:, 0] - trial_principal[:, 1]
    b_tr = trial_principal[:, 1] - trial_principal[:, 2]
    g_tr = a_tr + b_tr
    lambda_tr = np.clip(a_tr / np.maximum(g_tr, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    g_s = np.maximum(g_tr, 1.0e-12) * np.exp(soft_target[:, 0])
    lambda_s = 1.0 / (1.0 + np.exp(-(np.log(lambda_tr / (1.0 - lambda_tr)) + soft_target[:, 1])))
    a = np.zeros_like(g_s)
    b = np.zeros_like(g_s)
    smooth = branch_id[plastic] == BRANCH_TO_ID["smooth"]
    left = branch_id[plastic] == BRANCH_TO_ID["left_edge"]
    right = branch_id[plastic] == BRANCH_TO_ID["right_edge"]
    apex = branch_id[plastic] == BRANCH_TO_ID["apex"]
    a[smooth] = lambda_s[smooth] * g_s[smooth]
    b[smooth] = (1.0 - lambda_s[smooth]) * g_s[smooth]
    b[left] = np.maximum(b_tr[left], 1.0e-12) * np.exp(soft_target[left, 2])
    a[right] = np.maximum(a_tr[right], 1.0e-12) * np.exp(soft_target[right, 3])
    a[apex] = 0.0
    b[apex] = 0.0
    decoded = decode_abr_to_principal(
        np.column_stack([a, b, np.zeros_like(a)]).astype(np.float32),
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
    ).astype(np.float32)

    assert np.allclose(decoded, stress_principal_true, atol=1.0e-6)
    assert np.all(yield_function_principal_3d(decoded, c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1]) <= 5.0e-6)


def test_packet4_random_soft_mixture_remains_admissible(tmp_path: Path):
    _summary, derived_path = _build_tiny_packet4_stage0(tmp_path)
    with h5py.File(derived_path, "r") as f:
        branch_id = f["branch_id"][:]
        plastic_idx = np.flatnonzero(branch_id > 0)[0]
        trial_principal = f["trial_principal"][plastic_idx : plastic_idx + 1]
        material_reduced = f["material_reduced"][plastic_idx : plastic_idx + 1]
        soft_target = f["soft_atlas_target"][plastic_idx : plastic_idx + 1]

    a_tr = trial_principal[:, 0] - trial_principal[:, 1]
    b_tr = trial_principal[:, 1] - trial_principal[:, 2]
    g_tr = a_tr + b_tr
    lambda_tr = np.clip(a_tr / np.maximum(g_tr, 1.0e-12), 1.0e-6, 1.0 - 1.0e-6)
    g_s = np.maximum(g_tr, 1.0e-12) * np.exp(soft_target[:, 0])
    lambda_s = 1.0 / (1.0 + np.exp(-(np.log(lambda_tr / (1.0 - lambda_tr)) + soft_target[:, 1])))
    a_s = lambda_s * g_s
    b_s = (1.0 - lambda_s) * g_s
    b_l = np.maximum(b_tr, 1.0e-12) * np.exp(soft_target[:, 2])
    a_r = np.maximum(a_tr, 1.0e-12) * np.exp(soft_target[:, 3])

    rng = np.random.default_rng(17)
    for _ in range(8):
        w = rng.random(4)
        w = w / w.sum()
        a = w[0] * a_s + w[2] * a_r
        b = w[0] * b_s + w[1] * b_l
        decoded = decode_abr_to_principal(
            np.column_stack([a, b, np.zeros_like(a)]).astype(np.float32),
            c_bar=material_reduced[:, 0],
            sin_phi=material_reduced[:, 1],
        ).astype(np.float32)
        assert np.all(a >= -1.0e-12)
        assert np.all(b >= -1.0e-12)
        assert np.all(
            yield_function_principal_3d(decoded, c_bar=material_reduced[:, 0], sin_phi=material_reduced[:, 1]) <= 5.0e-6
        )


@pytest.mark.parametrize("model_kind", ["trial_surface_soft_atlas_f1_concat", "trial_surface_soft_atlas_f1_film"])
def test_packet4_checkpoint_keeps_exact_elastic_override(tmp_path: Path, model_kind: str):
    summary, derived_path = _build_tiny_packet4_stage0(tmp_path)
    with h5py.File(derived_path, "r") as f:
        feature_dim = f["soft_atlas_feature_f1"].shape[1]
        feature_stats = json.loads(f.attrs["soft_atlas_feature_stats_json"])
        strain_eng = f["strain_eng"][:]
        material_reduced = f["material_reduced"][:]
        branch_id = f["branch_id"][:]
        elastic_idx = np.flatnonzero(branch_id == BRANCH_TO_ID["elastic"])[0:1]
        exact = constitutive_update_3d(
            strain_eng[elastic_idx],
            c_bar=material_reduced[elastic_idx, 0],
            sin_phi=material_reduced[elastic_idx, 1],
            shear=material_reduced[elastic_idx, 2],
            bulk=material_reduced[elastic_idx, 3],
            lame=material_reduced[elastic_idx, 4],
        )

    checkpoint_path = _write_soft_atlas_checkpoint(
        tmp_path / f"{model_kind}.pt",
        model_kind=model_kind,
        input_dim=feature_dim,
        feature_stats=feature_stats,
    )
    pred = predict_with_checkpoint(
        checkpoint_path,
        strain_eng[elastic_idx],
        material_reduced[elastic_idx],
        device="cpu",
        batch_size=1,
    )

    assert pred["branch_probabilities"].shape == (1, 5)
    assert pred["predicted_branch_id"].shape == (1,)
    assert pred["branch_head_kind"] == "soft_atlas"
    assert np.allclose(pred["stress"], exact.stress.astype(np.float32), atol=1.0e-6)
    assert np.allclose(pred["stress_principal"], exact.stress_principal.astype(np.float32), atol=1.0e-6)


def test_packet4_route_temperature_changes_deployed_route_and_stress(tmp_path: Path):
    summary, derived_path = _build_tiny_packet4_stage0(tmp_path)
    with h5py.File(derived_path, "r") as f:
        feature_dim = f["soft_atlas_feature_f1"].shape[1]
        feature_stats = json.loads(f.attrs["soft_atlas_feature_stats_json"])
        strain_eng = f["strain_eng"][:]
        material_reduced = f["material_reduced"][:]
        branch_id = f["branch_id"][:]
        plastic_idx = np.flatnonzero(branch_id > 0)[0:3]

    checkpoint_path = _write_soft_atlas_checkpoint(
        tmp_path / "soft_atlas_temp.pt",
        model_kind="trial_surface_soft_atlas_f1_concat",
        input_dim=feature_dim,
        feature_stats=feature_stats,
        route_bias=[3.0, 1.0, -1.0, -3.0],
    )
    pred_cold = predict_with_checkpoint(
        checkpoint_path,
        strain_eng[plastic_idx],
        material_reduced[plastic_idx],
        device="cpu",
        batch_size=1,
        route_temperature=0.60,
    )
    pred_hot = predict_with_checkpoint(
        checkpoint_path,
        strain_eng[plastic_idx],
        material_reduced[plastic_idx],
        device="cpu",
        batch_size=1,
        route_temperature=2.00,
    )

    assert not np.allclose(pred_cold["soft_atlas_route_probs"], pred_hot["soft_atlas_route_probs"])
    assert not np.allclose(pred_cold["branch_probabilities"], pred_hot["branch_probabilities"])
    assert pred_cold["grho"].shape == (plastic_idx.shape[0], 2)
    assert pred_hot["grho"].shape == (plastic_idx.shape[0], 2)
    assert pred_cold["predicted_branch_id"].shape == (plastic_idx.shape[0],)
    assert pred_hot["predicted_branch_id"].shape == (plastic_idx.shape[0],)
    assert not np.allclose(pred_cold["stress"], pred_hot["stress"])


def test_packet4_temperature_selection_excludes_infeasible_rows():
    rows = [
        {
            "temperature": 0.60,
            "checkpoint_label": "best.pt",
            "hard_p95_principal": 100.0,
            "hard_plastic_mae": 10.0,
            "broad_plastic_mae": 9.0,
            "oracle_gap_hard_p95_principal": 2.0,
            "yield_violation_p95": 1.0e-4,
        },
        {
            "temperature": 1.00,
            "checkpoint_label": "best.pt",
            "hard_p95_principal": 120.0,
            "hard_plastic_mae": 12.0,
            "broad_plastic_mae": 11.0,
            "oracle_gap_hard_p95_principal": 3.0,
            "yield_violation_p95": 1.0e-8,
        },
    ]
    selected = _select_best_temperature_row(rows)
    assert selected is not None
    assert selected["temperature"] == 1.00


def test_stage1_packet4_soft_atlas_training_smoke_on_tiny_stage0_dataset(tmp_path: Path):
    summary, _derived_path = _build_tiny_packet4_stage0(tmp_path)
    config = TrainingConfig(
        dataset=summary["dataset_path"],
        run_dir=str(tmp_path / "run"),
        model_kind="trial_surface_soft_atlas_f1_concat",
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
        branch_loss_weight=0.0,
    )
    train_summary = train_model(config)
    eval_result = evaluate_soft_atlas_checkpoint(
        Path(train_summary["best_checkpoint"]),
        dataset_path=Path(summary["dataset_path"]),
        split_name="test",
        device="cpu",
        batch_size=8,
    )
    assert Path(train_summary["best_checkpoint"]).exists()
    assert "broad_plastic_mae" in eval_result["metrics"]
    assert "soft_atlas_route_entropy" in eval_result["metrics"]
