import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(SCRIPTS_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EXPERIMENTS))

from mc_surrogate.branch_geometry import compute_branch_geometry_principal, global_min_term_names, select_branch_conditioned_distance
from mc_surrogate.data import load_arrays
from mc_surrogate.inference import apply_hybrid_gate, normalized_branch_entropy, prepare_hybrid_gate_inputs
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import yield_function_principal_3d, yield_violation_rel_principal_3d
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset
from mc_surrogate.training import TrainingConfig, train_model
from run_hybrid_gate_redesign_cycle import (
    AdamStage,
    _evaluate_global_frontier,
    _rows_to_flat,
    _write_csv,
    predict_rejector,
    train_rejector,
    train_validation_first_staged_baseline,
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


def _train_small_geom_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "geom_train.h5"
    run_dir = tmp_path / "geom_run"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=80,
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
    return Path(summary["best_checkpoint"]), dataset_path


def test_yield_violation_rel_and_branch_conditioned_distance_helpers():
    sigma = np.array(
        [
            [10.0, 8.0, 6.0],
            [80.0, 25.0, -20.0],
        ],
        dtype=np.float32,
    )
    c_bar = np.array([30.0, 10.0], dtype=np.float32)
    sin_phi = np.array([0.3, 0.3], dtype=np.float32)
    f_sigma = yield_function_principal_3d(sigma, c_bar=c_bar, sin_phi=sin_phi)
    violation = yield_violation_rel_principal_3d(sigma, c_bar=c_bar, sin_phi=sin_phi)
    assert f_sigma[0] < 0.0
    assert np.isclose(violation[0], 0.0)
    assert f_sigma[1] > 0.0
    assert violation[1] > 0.0

    principal = np.array(
        [
            [3.0, 2.0, 1.0],
            [4.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    reduced = _default_material(2)
    geom = compute_branch_geometry_principal(
        principal,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    terms = global_min_term_names(geom)
    assert terms.shape == (2,)
    assert np.all(np.isin(terms, ["yield", "smooth_left", "smooth_right", "left_apex", "right_apex", "gap12_norm", "gap23_norm"]))

    dist, family, min_term = select_branch_conditioned_distance(geom, np.array([1, 4]))
    assert np.isclose(dist[0], geom.d_smooth[0])
    assert np.isclose(dist[1], geom.d_apex[1])
    assert family.tolist() == ["smooth", "apex"]
    assert min_term.shape == (2,)


def test_normalized_entropy_and_gate_routing_modes(tmp_path: Path):
    assert np.isclose(normalized_branch_entropy(np.array([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32))[0], 0.0)
    assert normalized_branch_entropy(np.array([[0.0, 0.25, 0.25, 0.25, 0.25]], dtype=np.float32))[0] > 0.99

    checkpoint_path, _dataset_path = _train_small_geom_checkpoint(tmp_path)
    reduced = _default_material(2)
    strain = np.array(
        [
            [3.0e-3, 1.0e-3, -2.0e-3, 0.0, 0.0, 0.0],
            [4.0e-3, 1.0e-3, -3.0e-3, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    material = np.column_stack([reduced.c_bar, reduced.sin_phi, reduced.shear, reduced.bulk, reduced.lame]).astype(np.float32)
    prepared = prepare_hybrid_gate_inputs(checkpoint_path, strain, material, device="cpu", batch_size=2, include_exact=True)
    prepared["elastic_mask"][:] = False
    prepared["predicted_branch_distance"] = np.array([0.50, 0.50], dtype=np.float32)
    prepared["predicted_branch_family"] = np.array(["smooth", "smooth"], dtype=object)
    prepared["predicted_branch_min_term"] = np.array(["yield", "yield"], dtype=object)
    prepared["branch_entropy"] = np.array([0.10, 0.90], dtype=np.float32)

    predicted = apply_hybrid_gate(
        prepared,
        delta_geom=0.05,
        gate_mode="predicted_branch",
        entropy_threshold=0.50,
    )
    assert predicted["route_reason"].tolist() == ["learned", "entropy_fallback"]

    rejector = apply_hybrid_gate(
        prepared,
        delta_geom=10.0,
        gate_mode="rejector",
        rejector_score=np.array([0.80, 0.20], dtype=np.float32),
        rejector_threshold=0.50,
    )
    assert rejector["route_reason"].tolist() == ["rejector_fallback", "learned"]


def test_validation_first_baseline_smoke_and_frontier_csv(tmp_path: Path):
    dataset_path = tmp_path / "baseline_small.h5"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=72,
            seed=11,
            candidate_batch=96,
            max_abs_principal_strain=2.0e-3,
        ),
    )
    summary = train_validation_first_staged_baseline(
        dataset_path=dataset_path,
        run_dir=tmp_path / "baseline_run",
        device="cpu",
        width=32,
        depth=1,
        seed=11,
        schedule=[AdamStage("mini", batch_size=16, lr_start=1.0e-3, lr_end=1.0e-3, epochs=1, lbfgs_steps_after=0)],
        min_delta=0.0,
        stop_patience=2,
        force_rerun=True,
    )
    assert summary["selection_metric"] == "val_stress_mse"
    assert Path(summary["best_checkpoint"]).exists()
    history_text = Path(summary["history_csv"]).read_text()
    assert "best_val_stress_mse" in history_text
    assert "test_stress_mse" not in history_text

    checkpoint_path, geom_dataset = _train_small_geom_checkpoint(tmp_path / "frontier")
    arrays = load_arrays(str(geom_dataset), ["strain_eng", "stress", "material_reduced", "stress_principal", "branch_id"], split="val")
    panel = {
        "plastic_mask": (arrays["branch_id"] > 0).astype(np.int8),
        "hard_mask": np.ones(arrays["branch_id"].shape[0], dtype=np.int8),
    }
    prepared = prepare_hybrid_gate_inputs(checkpoint_path, arrays["strain_eng"], arrays["material_reduced"], device="cpu", batch_size=64, include_exact=True)
    rows = _evaluate_global_frontier(prepared=prepared, arrays=arrays, panel=panel, delta_grid=(0.0, 0.1))
    csv_path = _write_csv(
        tmp_path / "frontier.csv",
        _rows_to_flat(rows, ["gate_mode", "delta_geom"]),
        ["gate_mode", "delta_geom", "n_rows", "broad_mae", "hard_mae", "broad_plastic_mae", "hard_plastic_mae", "hard_p95_principal", "hard_rel_p95_principal", "yield_violation_p95", "plastic_coverage", "accepted_plastic_rows", "route_counts", "accepted_true_branch_counts"],
    )
    assert csv_path.exists()
    assert len(csv_path.read_text().strip().splitlines()) == 3


def test_rejector_training_and_prediction_smoke(tmp_path: Path):
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(64, 8)).astype(np.float32)
    train_y = (rng.random(64) > 0.6).astype(np.float32)
    val_x = rng.normal(size=(32, 8)).astype(np.float32)
    val_y = (rng.random(32) > 0.6).astype(np.float32)
    summary = train_rejector(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        run_dir=tmp_path / "rejector",
        device="cpu",
        epochs=2,
        force_rerun=True,
    )
    preds = predict_rejector(Path(summary["best_checkpoint"]), val_x, device="cpu")
    assert preds.shape == (32,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_redesign_closeout_artifacts_include_baseline_gate_and_snapshot_selection():
    root = ROOT / "experiment_runs" / "real_sim" / "hybrid_gate_redesign_20260324"
    final_summary_path = root / "final" / "final_test_summary.json"
    baseline_selection_path = root / "exp_a_global" / "baseline_selection.json"
    snapshot_selection_path = root / "exp_a_global" / "candidate_b_snapshot_selection.json"
    snapshot_csv_path = root / "exp_a_global" / "candidate_b_snapshot_sweep_val.csv"
    if not all(path.exists() for path in [final_summary_path, baseline_selection_path, snapshot_selection_path, snapshot_csv_path]):
        pytest.skip("Hybrid gate redesign closeout artifacts are not available in this workspace.")

    import json

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    baseline_selection = json.loads(baseline_selection_path.read_text(encoding="utf-8"))
    snapshot_selection = json.loads(snapshot_selection_path.read_text(encoding="utf-8"))
    snapshot_csv = snapshot_csv_path.read_text(encoding="utf-8")

    assert "baseline_feasible_gate_test" in final_summary
    assert np.isclose(
        float(final_summary["baseline_feasible_gate_test"]["delta_geom"]),
        float(baseline_selection["best_row"]["delta_geom"]),
    )
    assert snapshot_selection["selected_checkpoint_label"] in snapshot_csv
    assert snapshot_selection["selection"]["best_row"]["delta_geom"] == snapshot_selection["selected_delta_geom"]
