import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

from mc_surrogate.data import load_arrays
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d, profile_constitutive_update_3d
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, train_model
from run_return_mapping_accel_scout import _branch_hint_artifacts, _profile_arrays, _shortlist_exact_proxy, _write_csv


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


def _train_small_geom_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "geom_train.h5"
    run_dir = tmp_path / "geom_run"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=80,
            seed=19,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
        ),
    )
    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="trial_principal_geom_plastic_branch_residual",
        epochs=2,
        batch_size=16,
        lr=1.0e-3,
        width=32,
        depth=1,
        dropout=0.0,
        seed=19,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
        branch_loss_weight=0.1,
        voigt_mae_weight=0.1,
        snapshot_every_epochs=1,
    )
    summary = train_model(config)
    return Path(summary["best_checkpoint"]), dataset_path


def test_profile_constitutive_update_matches_exact():
    reduced = _default_material(3)
    material = np.column_stack([reduced.c_bar, reduced.sin_phi, reduced.shear, reduced.bulk, reduced.lame]).astype(np.float32)
    strain = np.array(
        [
            [1.0e-5, 1.0e-5, 1.0e-5, 0.0, 0.0, 0.0],
            [3.0e-3, 1.0e-3, -2.0e-3, 0.0, 0.0, 0.0],
            [4.0e-3, 1.0e-3, -3.0e-3, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    exact = constitutive_update_3d(
        strain,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    profiled, profile = profile_constitutive_update_3d(
        strain,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    assert np.array_equal(profiled.branch_id, exact.branch_id)
    assert np.allclose(profiled.stress, exact.stress)
    assert np.allclose(profiled.stress_principal, exact.stress_principal)
    assert profile["branch_counts"]["elastic"] + profile["branch_counts"]["smooth"] + profile["branch_counts"]["left_edge"] + profile["branch_counts"]["right_edge"] + profile["branch_counts"]["apex"] == strain.shape[0]
    assert profile["plastic_rows"] + profile["elastic_rows"] == strain.shape[0]


def test_branch_hint_artifacts_and_profile_smoke(tmp_path: Path):
    checkpoint_path, dataset_path = _train_small_geom_checkpoint(tmp_path)
    arrays = load_arrays(str(dataset_path), ["strain_eng", "stress", "material_reduced", "stress_principal", "branch_id"], split="val")
    artifact = _branch_hint_artifacts(
        model_name="candidate_b",
        checkpoint_path=checkpoint_path,
        arrays=arrays,
        device="cpu",
    )
    assert artifact["summary"]["plastic_rows"] > 0
    assert artifact["topk_ids_full"].shape[0] == arrays["branch_id"].shape[0]
    assert artifact["topk_ids_full"].shape[1] == 2

    summary_csv = _write_csv(
        tmp_path / "hint_summary.csv",
        [artifact["summary"]],
        ["model_name", "checkpoint_path", "branch_head_kind", "plastic_rows", "top1_recall", "top2_recall", "missed_true_branch_rate", "mean_entropy", "p90_entropy", "mean_top1_confidence"],
    )
    assert summary_csv.exists()

    run_rows, summary = _profile_arrays("val_panel", arrays, warmup=0, repeats=1, return_tangent=False)
    assert len(run_rows) == 1
    assert summary["panel_name"] == "val_panel"
    assert summary["n_rows"] == arrays["strain_eng"].shape[0]


def test_shortlist_proxy_falls_back_and_preserves_exact_outputs(tmp_path: Path):
    checkpoint_path, dataset_path = _train_small_geom_checkpoint(tmp_path)
    arrays = load_arrays(str(dataset_path), ["strain_eng", "stress", "material_reduced", "stress_principal", "branch_id"], split="val")
    artifact = _branch_hint_artifacts(
        model_name="candidate_b",
        checkpoint_path=checkpoint_path,
        arrays=arrays,
        device="cpu",
    )
    exact = constitutive_update_3d(
        arrays["strain_eng"],
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
        shear=arrays["material_reduced"][:, 2],
        bulk=arrays["material_reduced"][:, 3],
        lame=arrays["material_reduced"][:, 4],
    )

    topk = artifact["topk_ids_full"].copy()
    plastic_rows = np.flatnonzero(arrays["branch_id"] > 0)
    for idx, row in enumerate(plastic_rows):
        true_branch = int(exact.branch_id[row])
        if idx % 2 == 0:
            topk[row, 0] = true_branch
            topk[row, 1] = 1 if true_branch != 1 else 2
        else:
            topk[row, 0] = 1 if true_branch != 1 else 2
            topk[row, 1] = 3 if true_branch not in (1, 3) else 4

    proxy = _shortlist_exact_proxy(arrays, topk)
    assert np.any(proxy["fallback_mask"])
    assert np.array_equal(proxy["branch_id"], exact.branch_id.astype(np.int8))
    assert np.allclose(proxy["stress"], exact.stress.astype(np.float32))
    assert np.allclose(proxy["stress_principal"], exact.stress_principal.astype(np.float32))
