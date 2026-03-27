import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

from build_hybrid_real_panels import build_hybrid_real_panels
from mc_surrogate.branch_geometry import compute_branch_geometry_principal, normalized_gap_metrics_principal
from mc_surrogate.inference import hybrid_predict_with_checkpoint
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, load_checkpoint, predict_with_checkpoint, train_model


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


def _train_small_geom_checkpoint(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "geom_train.h5"
    run_dir = tmp_path / "geom_run"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=60,
            seed=17,
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
        seed=17,
        patience=5,
        device="cpu",
        scheduler_kind="plateau",
        branch_loss_weight=0.1,
        voigt_mae_weight=0.1,
        snapshot_every_epochs=1,
    )
    summary = train_model(config)
    return Path(summary["best_checkpoint"])


def _write_mini_full_export(path: Path, *, n_calls: int = 10, rows_per_call: int = 12) -> Path:
    rng = np.random.default_rng(123)
    with h5py.File(path, "w") as f:
        calls = f.create_group("calls")
        for call_idx in range(n_calls):
            grp = calls.create_group(f"call_{call_idx + 1:06d}")
            strain = rng.normal(scale=2.0e-3, size=(rows_per_call, 6)).astype(np.float32)
            strain[:, :3] += rng.normal(loc=5.0e-4, scale=8.0e-4, size=(rows_per_call, 3)).astype(np.float32)
            reduced = build_reduced_material_from_raw(
                c0=np.full(rows_per_call, 12.0 + 0.1 * call_idx),
                phi_rad=np.deg2rad(np.full(rows_per_call, 30.0 + 0.1 * call_idx)),
                psi_rad=np.deg2rad(np.zeros(rows_per_call)),
                young=np.full(rows_per_call, 2.0e4 + 20.0 * call_idx),
                poisson=np.full(rows_per_call, 0.28),
                strength_reduction=np.full(rows_per_call, 1.1),
                davis_type=["B"] * rows_per_call,
            )
            exact = constitutive_update_3d(
                strain,
                c_bar=reduced.c_bar,
                sin_phi=reduced.sin_phi,
                shear=reduced.shear,
                bulk=reduced.bulk,
                lame=reduced.lame,
            )
            grp.create_dataset("E", data=strain)
            grp.create_dataset("S", data=exact.stress.astype(np.float32))
            grp.create_dataset("DS", data=np.repeat(np.eye(6, dtype=np.float32)[None, :, :], rows_per_call, axis=0))
            grp.create_dataset("c_bar", data=reduced.c_bar[:, None].astype(np.float32))
            grp.create_dataset("sin_phi", data=reduced.sin_phi[:, None].astype(np.float32))
            grp.create_dataset("shear", data=reduced.shear[:, None].astype(np.float32))
            grp.create_dataset("bulk", data=reduced.bulk[:, None].astype(np.float32))
            grp.create_dataset("lame", data=reduced.lame[:, None].astype(np.float32))
    return path


def test_geometry_safety_metrics_are_consistent():
    principal = np.array(
        [
            [3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    gap12, gap23, gap_min = normalized_gap_metrics_principal(principal)
    assert np.isclose(gap12[0], 1.0 / 8.0)
    assert np.isclose(gap23[0], 1.0 / 8.0)
    assert np.isclose(gap_min[0], 1.0 / 8.0)
    assert np.isclose(gap12[1], 0.0)
    assert np.isclose(gap23[1], 0.0)
    assert np.isclose(gap_min[1], 0.0)

    reduced = _default_material(2)
    geom = compute_branch_geometry_principal(
        principal,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    expected = np.minimum.reduce(
        [
            np.abs(geom.m_yield),
            np.abs(geom.m_smooth_left),
            np.abs(geom.m_smooth_right),
            np.abs(geom.m_left_apex),
            np.abs(geom.m_right_apex),
            geom.gap12_norm,
            geom.gap23_norm,
        ]
    )
    assert np.allclose(geom.gap_min_norm, np.minimum(geom.gap12_norm, geom.gap23_norm))
    assert np.allclose(geom.d_geom, expected)


def test_geom_plastic_checkpoint_snapshots_and_hybrid_exact_paths(tmp_path: Path):
    checkpoint_path = _train_small_geom_checkpoint(tmp_path)
    run_dir = checkpoint_path.parent
    snapshots_dir = run_dir / "snapshots"
    assert snapshots_dir.exists()
    assert (snapshots_dir / "epoch_0001.pt").exists()
    assert (snapshots_dir / "epoch_0002.pt").exists()

    _, metadata = load_checkpoint(checkpoint_path, device="cpu")
    assert metadata["branch_head_kind"] == "plastic_only"
    assert metadata["elastic_handling"] == "exact_upstream"
    assert metadata["branch_names"] == ["smooth", "left_edge", "right_edge", "apex"]

    reduced = _default_material(2)
    strain = np.array(
        [
            [1.0e-5, 1.0e-5, 1.0e-5, 0.0, 0.0, 0.0],
            [5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    material = np.column_stack(
        [
            reduced.c_bar,
            reduced.sin_phi,
            reduced.shear,
            reduced.bulk,
            reduced.lame,
        ]
    ).astype(np.float32)
    pred_plain = predict_with_checkpoint(checkpoint_path, strain, material, device="cpu", batch_size=2)
    assert pred_plain["branch_head_kind"] == "plastic_only"
    assert pred_plain["branch_probabilities"].shape == (2, 5)
    assert np.allclose(pred_plain["branch_probabilities"][:, 0], 0.0)

    hybrid = hybrid_predict_with_checkpoint(
        checkpoint_path,
        strain,
        material,
        delta_geom=1.0,
        device="cpu",
        batch_size=2,
    )
    exact = constitutive_update_3d(
        strain,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    assert hybrid["elastic_mask"].tolist() == [True, False]
    assert hybrid["fallback_mask"].tolist() == [False, True]
    assert hybrid["learned_mask"].tolist() == [False, False]
    assert np.allclose(hybrid["stress"], exact.stress.astype(np.float32), atol=1.0e-5)
    assert np.allclose(hybrid["stress_principal"], exact.stress_principal.astype(np.float32), atol=1.0e-5)


def test_build_hybrid_real_panels_smoke(tmp_path: Path):
    full_export = _write_mini_full_export(tmp_path / "mini_full_export.h5")
    output_root = tmp_path / "hybrid_panels"
    report_path = tmp_path / "hybrid_panels.md"

    result = build_hybrid_real_panels(
        full_export=full_export,
        output_root=output_root,
        report_path=report_path,
        samples_per_call=6,
        seed=20260323,
        split_fractions=(0.6, 0.2, 0.2),
    )

    dataset_path = Path(result["dataset_path"])
    panel_path = Path(result["panel_path"])
    call_split_path = Path(result["call_split_path"])
    assert dataset_path.exists()
    assert panel_path.exists()
    assert call_split_path.exists()
    assert report_path.exists()

    summary = result["summary"]
    assert summary["split_counts"]["train"] > 0
    assert summary["split_counts"]["val"] > 0
    assert summary["split_counts"]["test"] > 0
    assert summary["panel_counts"]["broad_val"] > 0
    assert "rare_branch_summary" in summary

    call_split = json.loads(call_split_path.read_text())
    assert len(call_split["source_call_names"]) == 10
    assert len(call_split["train_call_names"]) + len(call_split["val_call_names"]) + len(call_split["test_call_names"]) == 10

    with h5py.File(panel_path, "r") as f:
        assert "d_geom" in f
        assert "gap_min_norm" in f

    report_text = report_path.read_text()
    assert "call split json" in report_text
    assert "rare-branch policy" in report_text
