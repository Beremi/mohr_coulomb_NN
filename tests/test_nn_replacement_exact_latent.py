import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

from mc_surrogate.branch_geometry import compute_branch_geometry_principal
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.models import (
    build_model,
    build_trial_exact_latent_features_f1,
    compute_trial_exact_latent_feature_stats,
    decode_exact_branch_latents_to_principal_torch,
    exact_trial_principal_from_strain,
)
from mc_surrogate.mohr_coulomb import (
    BRANCH_NAMES,
    BRANCH_TO_ID,
    audit_exact_branch_latent_roundtrip,
    decode_exact_branch_latents_to_principal,
    extract_exact_branch_latents,
    yield_violation_rel_principal_3d,
)
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from run_nn_replacement_exact_latent_cycle import (
    BRANCH_MODEL_ORDER,
    BranchTrainingConfig,
    _evaluate_combined_oracle,
    _load_exact_latent_root,
    _train_branch_architecture,
    build_stage0_exact_latent_dataset,
    run_stage0_exactness_audit,
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


def _balanced_split_ids(branch_id: np.ndarray, seed: int = 11) -> np.ndarray:
    split_id = np.zeros(branch_id.shape[0], dtype=np.int8)
    rng = np.random.default_rng(seed)
    for branch_value in range(len(BRANCH_NAMES)):
        rows = np.flatnonzero(branch_id == branch_value)
        rng.shuffle(rows)
        n = int(rows.shape[0])
        if n < 3:
            split_id[rows] = 0
            continue
        n_val = max(1, int(round(0.15 * n)))
        n_test = max(1, int(round(0.15 * n)))
        while n - n_val - n_test < 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        n_train = n - n_val - n_test
        split_id[rows[:n_train]] = 0
        split_id[rows[n_train : n_train + n_val]] = 1
        split_id[rows[n_train + n_val :]] = 2
    return split_id


def _build_small_real_and_panel(tmp_path: Path, n_samples: int = 160) -> tuple[Path, Path]:
    dataset_path = tmp_path / "real_small.h5"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=n_samples,
            seed=19,
            candidate_batch=256,
            max_abs_principal_strain=2.0e-3,
            split_fractions=(0.70, 0.15, 0.15),
        ),
    )
    with h5py.File(dataset_path, "a") as f:
        branch_id = f["branch_id"][:].astype(np.int8)
        split_id = _balanced_split_ids(branch_id)
        f["split_id"][...] = split_id
        n = f["strain_eng"].shape[0]
        if "source_call_id" not in f:
            f.create_dataset("source_call_id", data=np.arange(n, dtype=np.int32))
        if "source_row_in_call" not in f:
            f.create_dataset("source_row_in_call", data=np.zeros(n, dtype=np.int32))

    with h5py.File(dataset_path, "r") as f:
        branch_id = f["branch_id"][:].astype(np.int8)
        split_id = f["split_id"][:].astype(np.int8)
        strain_principal = f["strain_principal"][:].astype(np.float32)
        material_reduced = f["material_reduced"][:].astype(np.float32)

    geom = compute_branch_geometry_principal(
        strain_principal,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        shear=material_reduced[:, 2],
        bulk=material_reduced[:, 3],
        lame=material_reduced[:, 4],
    )
    plastic = (branch_id > 0).astype(np.int8)
    hard = plastic.copy()

    panel_path = tmp_path / "panel_sidecar.h5"
    with h5py.File(panel_path, "w") as f:
        f.create_dataset("plastic_mask", data=plastic)
        f.create_dataset("broad_val_mask", data=(split_id == 1).astype(np.int8))
        f.create_dataset("broad_test_mask", data=(split_id == 2).astype(np.int8))
        f.create_dataset("hard_mask", data=hard)
        f.create_dataset("hard_val_mask", data=((split_id == 1) & (hard > 0)).astype(np.int8))
        f.create_dataset("hard_test_mask", data=((split_id == 2) & (hard > 0)).astype(np.int8))
        f.create_dataset("ds_valid_mask", data=np.zeros_like(split_id, dtype=np.int8))
        f.create_dataset("m_yield", data=geom.m_yield.astype(np.float32))
        f.create_dataset("m_smooth_left", data=geom.m_smooth_left.astype(np.float32))
        f.create_dataset("m_smooth_right", data=geom.m_smooth_right.astype(np.float32))
        f.create_dataset("m_left_apex", data=geom.m_left_apex.astype(np.float32))
        f.create_dataset("m_right_apex", data=geom.m_right_apex.astype(np.float32))
        f.create_dataset("gap12_norm", data=geom.gap12_norm.astype(np.float32))
        f.create_dataset("gap23_norm", data=geom.gap23_norm.astype(np.float32))
        f.create_dataset("gap_min_norm", data=geom.gap_min_norm.astype(np.float32))
        f.create_dataset("m_left_vs_right", data=geom.m_left_vs_right.astype(np.float32))
        f.create_dataset("d_geom", data=geom.d_geom.astype(np.float32))
    return dataset_path, panel_path


def _latent_matrix(extracted: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    latent_dim = int(extracted["latent_dim"][mask][0])
    if latent_dim == 0:
        return np.zeros((int(np.sum(mask)), 0), dtype=np.float32)
    rows = [np.asarray(item, dtype=np.float32).reshape(1, latent_dim) for item in extracted["latent_values"][mask]]
    return np.concatenate(rows, axis=0).astype(np.float32)


def test_exact_latent_feature_builder_and_model_are_finite():
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
    stats = compute_trial_exact_latent_feature_stats(trial_principal, material)
    features = build_trial_exact_latent_features_f1(strain[:, :3], material, trial_principal, stats)

    assert features.shape == (4, 20)
    assert np.isfinite(features).all()

    model = build_model("exact_latent_regressor", input_dim=features.shape[1], width=16, depth=2, dropout=0.0)
    out = model(torch.randn(5, features.shape[1]))
    assert out["stress"].shape == (5, 1)

    decoded = decode_exact_branch_latents_to_principal_torch(
        torch.zeros(3, 1),
        torch.tensor([BRANCH_TO_ID["smooth"], BRANCH_TO_ID["left_edge"], BRANCH_TO_ID["right_edge"]], dtype=torch.long),
        torch.tensor(
            [
                [20.0, 10.0, -5.0],
                [25.0, 12.5, -8.0],
                [28.0, 9.0, -10.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            np.column_stack([_default_material(3).c_bar, _default_material(3).sin_phi, _default_material(3).shear, _default_material(3).bulk, _default_material(3).lame]),
            dtype=torch.float32,
        ),
    )
    assert decoded.shape == (3, 3)
    assert torch.all(decoded[:, 0] >= decoded[:, 1])
    assert torch.all(decoded[:, 1] >= decoded[:, 2])


def test_exact_latent_extraction_and_roundtrip_cover_every_branch(tmp_path: Path):
    dataset_path, _panel_path = _build_small_real_and_panel(tmp_path)
    with h5py.File(dataset_path, "r") as f:
        strain_eng = f["strain_eng"][:]
        material_reduced = f["material_reduced"][:]
        stress_principal = f["stress_principal"][:]
        branch_id = f["branch_id"][:].astype(np.int64)

    extracted = extract_exact_branch_latents(
        strain_eng,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        shear=material_reduced[:, 2],
        bulk=material_reduced[:, 3],
        lame=material_reduced[:, 4],
    )
    assert np.array_equal(extracted["branch_id"], branch_id)

    for branch_name in BRANCH_NAMES:
        bid = BRANCH_TO_ID[branch_name]
        mask = branch_id == bid
        assert np.any(mask), f"Expected at least one {branch_name} row."
        decoded = decode_exact_branch_latents_to_principal(
            extracted["branch_id"][mask],
            extracted["trial_principal"][mask],
            extracted["material_reduced"][mask],
            _latent_matrix(extracted, mask),
        )
        assert np.allclose(decoded, stress_principal[mask], atol=1.0e-6)

    apex_mask = branch_id == BRANCH_TO_ID["apex"]
    assert np.all(extracted["latent_dim"][apex_mask] == 0)
    assert np.all(extracted["latent_kind"][apex_mask] == "analytic")

    audit = audit_exact_branch_latent_roundtrip(
        strain_eng,
        c_bar=material_reduced[:, 0],
        sin_phi=material_reduced[:, 1],
        shear=material_reduced[:, 2],
        bulk=material_reduced[:, 3],
        lame=material_reduced[:, 4],
    )
    assert audit["max_abs"] <= 1.0e-6
    assert audit["yield_p95"] <= 1.0e-6
    assert audit["branchwise"]["apex"]["count"] > 0


def test_stage0_exact_latent_dataset_and_exactness_smoke(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data")
    summary = build_stage0_exact_latent_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    derived_path = Path(summary["dataset_path"])
    assert derived_path.exists()
    assert summary["checks"]["branch_id_matches_source"]
    assert summary["checks"]["principal_matches_source_max_abs"] <= 1.0e-6
    assert summary["checks"]["features_all_finite"]

    with h5py.File(derived_path, "r") as f:
        assert "exact_latent_feature_f1" in f
        assert "branches" in f
        for branch_name in ("smooth", "left_edge", "right_edge", "apex"):
            assert branch_name in f["branches"]
        assert json.loads(f.attrs["exact_latent_feature_stats_json"])
        assert json.loads(f.attrs["exact_latent_schema_json"])
        assert f["branches"]["apex"]["latent_values"].shape[1] == 0

    audit = run_stage0_exactness_audit(
        stage0_dataset_path=derived_path,
        output_dir=tmp_path / "stage0_exactness",
        force_rerun=True,
    )
    assert audit["stage0_pass"]
    assert audit["reconstruction"]["max_abs"] <= 5.0e-6
    assert audit["yield"]["p95"] <= 1.0e-6


def test_tiny_exact_latent_branchwise_oracle_smoke(tmp_path: Path):
    dataset_path, panel_path = _build_small_real_and_panel(tmp_path / "data", n_samples=200)
    summary = build_stage0_exact_latent_dataset(
        real_dataset_path=dataset_path,
        panel_path=panel_path,
        output_dir=tmp_path / "stage0",
        force_rerun=True,
    )
    stage0_dataset_path = Path(summary["dataset_path"])

    winners: dict[str, dict[str, str]] = {}
    for branch_name in BRANCH_MODEL_ORDER:
        config = BranchTrainingConfig(
            branch_name=branch_name,
            branch_id=BRANCH_TO_ID[branch_name],
            width=16,
            depth=1,
            epochs=2,
            batch_size=16,
            patience=3,
            device="cpu",
            seed=23,
        )
        run_dir = tmp_path / "oracle_runs" / branch_name
        train_summary = _train_branch_architecture(
            dataset_path=stage0_dataset_path,
            branch_name=branch_name,
            config=config,
            run_dir=run_dir,
            force_rerun=True,
        )
        assert Path(train_summary["best_checkpoint"]).exists()
        winners[branch_name] = {
            "selected_checkpoint_path": train_summary["best_checkpoint"],
        }

    val_eval = _evaluate_combined_oracle(
        stage0_dataset_path=stage0_dataset_path,
        split_name="val",
        winners=winners,
        device="cpu",
        batch_size=256,
    )
    test_eval = _evaluate_combined_oracle(
        stage0_dataset_path=stage0_dataset_path,
        split_name="test",
        winners=winners,
        device="cpu",
        batch_size=256,
    )

    for evaluation in (val_eval, test_eval):
        metrics = evaluation["metrics"]
        assert np.isfinite(metrics["broad_plastic_mae"])
        assert np.isfinite(metrics["hard_plastic_mae"])
        assert np.isfinite(metrics["hard_p95_principal"])
        assert np.isfinite(metrics["yield_violation_p95"])
        assert "apex" in evaluation["branch_prediction_payloads"]
        branch_names = {row["branch"] for row in evaluation["branch_rows"]}
        assert {"elastic", "smooth", "left_edge", "right_edge", "apex"} <= branch_names

        arrays = evaluation["arrays"]
        preds = evaluation["predictions"]
        elastic = arrays["branch_id"] == BRANCH_TO_ID["elastic"]
        apex = arrays["branch_id"] == BRANCH_TO_ID["apex"]
        assert np.allclose(preds["stress"][elastic], arrays["stress"][elastic], atol=1.0e-6)
        assert np.allclose(preds["stress_principal"][apex], arrays["stress_principal"][apex], atol=1.0e-6)
        assert np.allclose(preds["stress"][apex], arrays["stress"][apex], atol=1.0e-6)

        yield_rel = yield_violation_rel_principal_3d(
            preds["stress_principal"],
            c_bar=arrays["material_reduced"][:, 0],
            sin_phi=arrays["material_reduced"][:, 1],
        )
        assert np.isfinite(yield_rel).all()

    root_arrays, _attrs = _load_exact_latent_root(stage0_dataset_path)
    assert root_arrays["exact_latent_feature_f1"].shape[1] == 20
