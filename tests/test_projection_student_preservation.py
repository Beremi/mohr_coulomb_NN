import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.models import stress_voigt_from_principal_numpy
from mc_surrogate.principal_projection import PROJECTION_CANDIDATE_TO_ID
from mc_surrogate.projection_student_preservation import (
    build_call_concentration_rows,
    build_control_zero_rowwise_arrays,
    build_slice_summary_rows,
    build_top_fraction_mask,
    build_teacher_projection_cache_arrays,
    directional_error_arrays,
    project_teacher_checkpoint_stress,
    projection_switch_mask,
)
from mc_surrogate.sampling import DatasetGenerationConfig, generate_branch_balanced_dataset


def _load_small_exact_arrays(tmp_path: Path) -> dict[str, np.ndarray]:
    dataset_path = tmp_path / "small_exact.h5"
    generate_branch_balanced_dataset(
        str(dataset_path),
        DatasetGenerationConfig(
            n_samples=48,
            seed=17,
            candidate_batch=128,
            max_abs_principal_strain=2.0e-3,
            include_tangent=True,
        ),
    )
    with h5py.File(dataset_path, "r") as f:
        arrays = {key: f[key][:] for key in f.keys()}
    return arrays


def test_build_teacher_projection_cache_arrays_preserves_selected_candidate_and_displacement(tmp_path: Path):
    arrays = _load_small_exact_arrays(tmp_path)
    n_rows = arrays["stress"].shape[0]
    panel = {
        "plastic_mask": (arrays["branch_id"] > 0).astype(np.int8),
        "hard_mask": np.zeros(n_rows, dtype=np.int8),
        "ds_valid_mask": np.ones(n_rows, dtype=np.int8),
        "near_yield_mask": np.zeros(n_rows, dtype=np.int8),
        "near_smooth_left_mask": np.zeros(n_rows, dtype=np.int8),
        "near_smooth_right_mask": np.zeros(n_rows, dtype=np.int8),
        "near_left_apex_mask": np.zeros(n_rows, dtype=np.int8),
        "near_right_apex_mask": np.zeros(n_rows, dtype=np.int8),
    }
    teacher_principal = arrays["stress_principal"].copy()
    teacher_principal[:, 0] += 5.0
    teacher_stress = stress_voigt_from_principal_numpy(teacher_principal, arrays["eigvecs"]).astype(np.float32)
    teacher_pred = {
        "stress": teacher_stress,
        "branch_probabilities": np.zeros((n_rows, 5), dtype=np.float32),
        "predicted_branch_id": np.zeros(n_rows, dtype=np.int64),
    }

    cache = build_teacher_projection_cache_arrays(arrays, panel, teacher_pred)
    disp = cache["teacher_projected_stress_principal"] - cache["teacher_dispatch_stress_principal"]
    assert np.allclose(cache["teacher_projection_disp_principal"], disp, atol=1.0e-6)
    assert np.allclose(cache["teacher_projection_delta_principal"], disp, atol=1.0e-6)

    rows = np.arange(n_rows)
    selected = cache["teacher_projection_candidate_principal"][rows, cache["teacher_projection_candidate_id"].astype(np.int64)]
    assert np.allclose(selected, cache["teacher_projected_stress_principal"], atol=1.0e-6)
    assert np.all(cache["teacher_projection_feasible_mask"][rows, cache["teacher_projection_candidate_id"].astype(np.int64)] > 0)


def test_directional_probe_smoke_exact_and_softmin(tmp_path: Path):
    arrays = _load_small_exact_arrays(tmp_path)
    strain = arrays["strain_eng"][:8].astype(np.float32)
    material = arrays["material_reduced"][:8].astype(np.float32)
    tangent = arrays["tangent"][:8].astype(np.float32)
    base = constitutive_update_3d(
        strain,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
        return_tangent=True,
    )
    rng = np.random.default_rng(20260327)
    direction = rng.normal(size=(strain.shape[0], 6)).astype(np.float32)
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-12)
    h = 1.0e-6 * np.maximum(np.amax(np.abs(strain), axis=1, keepdims=True), 1.0)
    strain_plus = strain + h * direction
    strain_minus = strain - h * direction
    plus = constitutive_update_3d(
        strain_plus,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
        return_tangent=False,
    )
    minus = constitutive_update_3d(
        strain_minus,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
        return_tangent=False,
    )

    for mode in ("exact", "softmin"):
        base_proj = project_teacher_checkpoint_stress(strain, material, base.stress, mode=mode, tau=0.05)
        plus_proj = project_teacher_checkpoint_stress(strain_plus, material, plus.stress, mode=mode, tau=0.05)
        minus_proj = project_teacher_checkpoint_stress(strain_minus, material, minus.stress, mode=mode, tau=0.05)
        jv_pred = (plus_proj["teacher_projected_stress"] - minus_proj["teacher_projected_stress"]) / (2.0 * h)
        jv_true = np.einsum("nij,nj->ni", tangent, direction, optimize=True)
        metrics = directional_error_arrays(jv_pred, jv_true)
        switch = projection_switch_mask(
            base_proj["teacher_projection_candidate_id"],
            plus_proj["teacher_projection_candidate_id"],
            minus_proj["teacher_projection_candidate_id"],
        )
        assert np.isfinite(metrics["abs_error_norm"]).all()
        assert np.isfinite(metrics["relative_error"]).all()
        assert np.isfinite(metrics["cosine_similarity"]).all()
        assert switch.shape == (strain.shape[0],)


def test_control_zero_rowwise_and_slice_helpers(tmp_path: Path):
    arrays = _load_small_exact_arrays(tmp_path)
    n_rows = 12
    exact_principal = arrays["stress_principal"][:n_rows].astype(np.float32)
    branch_id = np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4], dtype=np.int8)
    split_arrays = {
        "split_id": np.full(n_rows, 1, dtype=np.int8),
        "source_call_id": np.asarray([10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 40, 40], dtype=np.int32),
        "source_row_in_call": np.arange(n_rows, dtype=np.int32),
        "branch_id": branch_id,
        "hard_mask": np.asarray([0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
        "ds_valid_mask": np.ones(n_rows, dtype=np.int8),
        "plastic_mask": np.ones(n_rows, dtype=np.int8),
        "near_yield_mask": np.asarray([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8),
        "near_smooth_left_mask": np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8),
        "near_smooth_right_mask": np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        "near_left_apex_mask": np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
        "near_right_apex_mask": np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8),
        "teacher_projection_candidate_id": np.asarray(
            [
                PROJECTION_CANDIDATE_TO_ID["smooth"],
                PROJECTION_CANDIDATE_TO_ID["left_edge"],
                PROJECTION_CANDIDATE_TO_ID["left_edge"],
                PROJECTION_CANDIDATE_TO_ID["smooth"],
                PROJECTION_CANDIDATE_TO_ID["right_edge"],
                PROJECTION_CANDIDATE_TO_ID["right_edge"],
                PROJECTION_CANDIDATE_TO_ID["apex"],
                PROJECTION_CANDIDATE_TO_ID["apex"],
                PROJECTION_CANDIDATE_TO_ID["smooth"],
                PROJECTION_CANDIDATE_TO_ID["left_edge"],
                PROJECTION_CANDIDATE_TO_ID["right_edge"],
                PROJECTION_CANDIDATE_TO_ID["apex"],
            ],
            dtype=np.int8,
        ),
        "teacher_projection_disp_norm": np.linspace(0.5, 12.0, n_rows, dtype=np.float32),
        "exact_stress_principal": exact_principal,
        "teacher_projected_stress_principal": exact_principal,
    }
    offsets = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    control_predictions = {
        "stress_principal": exact_principal + offsets,
        "provisional_stress_principal": exact_principal + offsets + 0.25,
        "predicted_branch_id": branch_id.astype(np.int64),
    }
    rowwise, meta = build_control_zero_rowwise_arrays(split_arrays, control_predictions)
    assert rowwise["teacher_projection_disp_decile"].shape == (n_rows,)
    assert len(meta["teacher_projection_disp_decile_edges"]) == 11
    expected_any_boundary = (
        split_arrays["near_yield_mask"]
        | split_arrays["near_smooth_left_mask"]
        | split_arrays["near_smooth_right_mask"]
        | split_arrays["near_left_apex_mask"]
        | split_arrays["near_right_apex_mask"]
    )
    assert np.array_equal(rowwise["any_boundary_mask"], expected_any_boundary.astype(np.int8))
    assert np.allclose(rowwise["control_zero_principal_max_abs_error"], offsets[:, 0], atol=1.0e-6)

    hard_top25, _threshold = build_top_fraction_mask(
        rowwise["control_zero_principal_max_abs_error"],
        pool_mask=rowwise["hard_mask"] > 0,
        fraction=0.25,
    )
    slice_rows = build_slice_summary_rows(
        scope_name="val",
        rowwise_arrays=rowwise,
        group_name="teacher_projection_candidate",
        group_values=rowwise["teacher_projection_candidate_id"],
        labels={idx: name for idx, name in enumerate(("pass_through", "smooth", "left_edge", "right_edge", "apex"))},
        base_mask=rowwise["plastic_mask"] > 0,
        hard_top_fraction_masks={"25": hard_top25},
    )
    left_row = next(row for row in slice_rows if row["group_value"] == "left_edge")
    apex_row = next(row for row in slice_rows if row["group_value"] == "apex")
    assert left_row["n_rows"] > 0
    assert apex_row["hard_top25_share"] > 0.0

    call_rows = build_call_concentration_rows(
        scope_name="val",
        source_call_id=rowwise["source_call_id"],
        base_mask=rowwise["plastic_mask"] > 0,
        hard_mask=rowwise["hard_mask"] > 0,
        principal_max_abs_error=rowwise["control_zero_principal_max_abs_error"],
        hard_top_mask=hard_top25,
        top_n=5,
    )
    assert call_rows
    assert int(call_rows[0]["source_call_id"]) in {20, 40}
