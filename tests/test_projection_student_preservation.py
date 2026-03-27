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
from mc_surrogate.projection_student_preservation import (
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
