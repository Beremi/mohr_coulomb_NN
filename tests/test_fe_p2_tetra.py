from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mc_surrogate.fe_p2_tetra import (
    build_local_b_pool,
    positive_corner_volume_mask,
    strain_from_local_displacements,
)


MESH_PATH = Path("/tmp/slope_stability_octave/slope_stability/meshes/SSR_hetero_ada_L1.h5")


@pytest.mark.skipif(not MESH_PATH.exists(), reason="Upstream slope_stability_octave mesh is not available.")
def test_local_b_pool_right_inverse_reconstructs_strain() -> None:
    pool = build_local_b_pool(MESH_PATH, material_id=0)
    assert pool.b_blocks.shape[1:] == (6, 30)
    assert pool.right_inverse.shape[1:] == (30, 6)

    rng = np.random.default_rng(123)
    strain = rng.normal(size=(6, 6)).astype(np.float32)
    disp = np.einsum("nij,nj->ni", pool.right_inverse[:6], strain)
    recon = strain_from_local_displacements(pool.b_blocks[:6], disp)
    np.testing.assert_allclose(recon, strain, atol=2.0e-6, rtol=2.0e-6)


@pytest.mark.skipif(not MESH_PATH.exists(), reason="Upstream slope_stability_octave mesh is not available.")
def test_positive_corner_volume_mask_rejects_inversion() -> None:
    pool = build_local_b_pool(MESH_PATH, material_id=0)
    local_coords = pool.local_coords[:1].copy()

    ok = positive_corner_volume_mask(local_coords, np.zeros((1, 30), dtype=np.float32), min_volume_ratio=0.05)
    assert ok.tolist() == [True]

    corners = local_coords[0, :4, :]
    face_centroid = np.mean(corners[1:], axis=0)
    move = 2.0 * (face_centroid - corners[0])
    bad_disp = np.zeros((1, 30), dtype=np.float32)
    bad_disp[0, :3] = move.astype(np.float32)
    ok_bad = positive_corner_volume_mask(local_coords, bad_disp, min_volume_ratio=0.05)
    assert ok_bad.tolist() == [False]
