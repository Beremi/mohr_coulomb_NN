from __future__ import annotations

import numpy as np
import h5py

from mc_surrogate.full_export import canonicalize_p2_element_states, split_full_export_call_names


def _rotation_z(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def test_canonicalize_p2_element_states_is_rigid_motion_invariant() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.75],
            [0.0, 0.5, 0.75],
            [1.0, 0.0, 0.75],
        ],
        dtype=np.float32,
    )[None, :, :]
    disp = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, 0.08],
            [0.05, 0.01, 0.0],
            [0.04, -0.02, 0.0],
            [0.0, -0.03, 0.01],
            [0.0, 0.0, 0.04],
            [0.0, -0.01, 0.03],
            [0.03, 0.0, 0.02],
        ],
        dtype=np.float32,
    )[None, :, :]

    canonical = canonicalize_p2_element_states(coords, disp)

    rot = _rotation_z(47.0)
    shift = np.array([8.0, -5.0, 3.0], dtype=np.float32)
    scale = 2.75
    coords_moved = (coords @ rot.T) * scale + shift
    disp_moved = (disp @ rot.T) * scale
    canonical_moved = canonicalize_p2_element_states(coords_moved, disp_moved)

    np.testing.assert_allclose(canonical.local_coords, canonical_moved.local_coords, atol=2.0e-6, rtol=2.0e-6)
    np.testing.assert_allclose(
        canonical.local_displacements,
        canonical_moved.local_displacements,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_split_full_export_call_names_is_deterministic(tmp_path) -> None:
    path = tmp_path / "mini.h5"
    with h5py.File(path, "w") as f:
        calls = f.create_group("calls")
        for idx in range(5):
            calls.create_group(f"call_{idx:06d}")

    split_a = split_full_export_call_names(path, split_fractions=(0.6, 0.2, 0.2), seed=17)
    split_b = split_full_export_call_names(path, split_fractions=(0.6, 0.2, 0.2), seed=17)

    assert split_a == split_b
    assert sum(len(v) for v in split_a.values()) == 5
    assert set(split_a["generator_fit"]) | set(split_a["real_val"]) | set(split_a["real_test"]) == {
        f"call_{idx:06d}" for idx in range(5)
    }
