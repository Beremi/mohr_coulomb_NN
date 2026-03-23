"""Local P2 tetrahedral FE helpers for displacement-to-strain sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class LocalIntegrationPointPool:
    """Precomputed local geometry and B-operator data for one material family."""

    local_coords: np.ndarray
    b_blocks: np.ndarray
    right_inverse: np.ndarray
    characteristic_length: np.ndarray
    corner_volume: np.ndarray
    element_index: np.ndarray
    quadrature_index: np.ndarray


def quadrature_volume_p2_tetra() -> tuple[np.ndarray, np.ndarray]:
    """Return the 11-point tetrahedral quadrature used upstream for P2 elements."""
    xi = np.array(
        [
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.0714285714285714,
                0.399403576166799,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
                0.100596423833201,
            ],
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
            ],
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
            ],
        ],
        dtype=np.float64,
    )
    wf = np.array(
        [-0.013155555555555]
        + [0.007622222222222] * 4
        + [0.024888888888888] * 6,
        dtype=np.float64,
    )
    return xi, wf


def local_basis_derivatives_p2_tetra(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P2 tetrahedral basis derivatives on the reference element."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape[0] != 3:
        raise ValueError(f"Expected xi shape (3, n_q), got {xi.shape}.")
    xi_1 = xi[0]
    xi_2 = xi[1]
    xi_3 = xi[2]
    xi_0 = 1.0 - xi_1 - xi_2 - xi_3
    n_q = xi.shape[1]

    d1 = np.vstack(
        [
            -4.0 * xi_0 + 1.0,
            4.0 * xi_1 - 1.0,
            np.zeros(n_q, dtype=np.float64),
            np.zeros(n_q, dtype=np.float64),
            4.0 * (xi_0 - xi_1),
            4.0 * xi_2,
            -4.0 * xi_2,
            4.0 * xi_3,
            np.zeros(n_q, dtype=np.float64),
            -4.0 * xi_3,
        ]
    )
    d2 = np.vstack(
        [
            -4.0 * xi_0 + 1.0,
            np.zeros(n_q, dtype=np.float64),
            4.0 * xi_2 - 1.0,
            np.zeros(n_q, dtype=np.float64),
            -4.0 * xi_1,
            4.0 * xi_1,
            4.0 * (xi_0 - xi_2),
            np.zeros(n_q, dtype=np.float64),
            4.0 * xi_3,
            -4.0 * xi_3,
        ]
    )
    d3 = np.vstack(
        [
            -4.0 * xi_0 + 1.0,
            np.zeros(n_q, dtype=np.float64),
            np.zeros(n_q, dtype=np.float64),
            4.0 * xi_3 - 1.0,
            -4.0 * xi_1,
            np.zeros(n_q, dtype=np.float64),
            -4.0 * xi_2,
            4.0 * xi_1,
            4.0 * xi_2,
            4.0 * (xi_0 - xi_3),
        ]
    )
    return d1, d2, d3


def load_mesh_p2_hdf5(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the upstream P2 mesh and reorder coordinates to match Octave."""
    with h5py.File(path, "r") as f:
        node = f["node"][:].astype(np.float64)
        elem = f["elem"][:].astype(np.int64)
        material = f["material"][:].astype(np.int64)
    coord = node[:, [0, 2, 1]]
    return coord, elem, material


def corner_signed_volumes(corner_coords: np.ndarray) -> np.ndarray:
    """Return signed corner-tetra volumes for coordinates (..., 4, 3)."""
    corners = np.asarray(corner_coords, dtype=np.float64)
    v1 = corners[..., 1, :] - corners[..., 0, :]
    v2 = corners[..., 2, :] - corners[..., 0, :]
    v3 = corners[..., 3, :] - corners[..., 0, :]
    det = np.einsum("...i,...i->...", np.cross(v1, v2), v3)
    return det / 6.0


def characteristic_length_from_corners(corner_coords: np.ndarray) -> np.ndarray:
    """Use the mean corner-edge length as a stable local normalization scale."""
    corners = np.asarray(corner_coords, dtype=np.float64)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = [np.linalg.norm(corners[..., i, :] - corners[..., j, :], axis=-1) for i, j in pairs]
    mean_len = np.mean(np.stack(lengths, axis=-1), axis=-1)
    return np.maximum(mean_len, 1.0e-8)


def build_local_b_pool(
    mesh_path: str | Path,
    *,
    material_id: int,
    right_inverse_damping: float = 1.0e-10,
) -> LocalIntegrationPointPool:
    """Build a local integration-point pool for one material family."""
    coord, elem, material = load_mesh_p2_hdf5(mesh_path)
    element_index = np.flatnonzero(material == int(material_id))
    if element_index.size == 0:
        raise ValueError(f"No elements found for material_id={material_id}.")

    local_coords = coord[elem[element_index]]
    b_blocks = build_local_b_blocks_from_coords(local_coords)
    n_elem = local_coords.shape[0]
    n_q = b_blocks.shape[1]
    b_blocks = b_blocks.reshape(n_elem * n_q, 6, 30)
    local_coords_ip = np.repeat(local_coords, repeats=n_q, axis=0)
    corner_coords = local_coords_ip[:, :4, :]
    char_length = characteristic_length_from_corners(corner_coords)
    corner_volume = corner_signed_volumes(corner_coords)

    bb_t = np.einsum("nij,nkj->nik", b_blocks, b_blocks)
    eye6 = np.eye(6, dtype=np.float64)[None, :, :]
    inv_term = np.linalg.inv(bb_t + right_inverse_damping * eye6)
    right_inverse = np.einsum("nji,njk->nik", b_blocks, inv_term)

    q_index = np.tile(np.arange(n_q, dtype=np.int16), n_elem)
    e_index_ip = np.repeat(element_index.astype(np.int32), repeats=n_q)
    return LocalIntegrationPointPool(
        local_coords=local_coords_ip.astype(np.float32),
        b_blocks=b_blocks.astype(np.float32),
        right_inverse=right_inverse.astype(np.float32),
        characteristic_length=char_length.astype(np.float32),
        corner_volume=corner_volume.astype(np.float32),
        element_index=e_index_ip,
        quadrature_index=q_index,
    )


def build_local_b_blocks_from_coords(local_coords: np.ndarray) -> np.ndarray:
    """Build local P2 tetrahedral B blocks for coordinates of shape (n_elem, 10, 3)."""
    local_coords = np.asarray(local_coords, dtype=np.float64)
    if local_coords.ndim != 3 or local_coords.shape[1:] != (10, 3):
        raise ValueError(f"Expected local_coords shape (n_elem, 10, 3), got {local_coords.shape}.")

    xi, _ = quadrature_volume_p2_tetra()
    dh1, dh2, dh3 = local_basis_derivatives_p2_tetra(xi)
    n_elem = local_coords.shape[0]
    n_q = xi.shape[1]

    b_blocks = np.empty((n_elem, n_q, 6, 30), dtype=np.float64)

    x = local_coords[:, :, 0]
    y = local_coords[:, :, 1]
    z = local_coords[:, :, 2]

    for q in range(n_q):
        d1 = dh1[:, q]
        d2 = dh2[:, q]
        d3 = dh3[:, q]

        j11 = x @ d1
        j12 = y @ d1
        j13 = z @ d1
        j21 = x @ d2
        j22 = y @ d2
        j23 = z @ d2
        j31 = x @ d3
        j32 = y @ d3
        j33 = z @ d3

        det = j11 * (j22 * j33 - j32 * j23) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)
        if np.any(np.abs(det) < 1.0e-14):
            raise ValueError("Encountered degenerate tetrahedral Jacobian while building B blocks.")

        jinv11 = (j22 * j33 - j23 * j32) / det
        jinv12 = -(j12 * j33 - j13 * j32) / det
        jinv13 = (j12 * j23 - j13 * j22) / det
        jinv21 = -(j21 * j33 - j23 * j31) / det
        jinv22 = (j11 * j33 - j13 * j31) / det
        jinv23 = -(j11 * j23 - j13 * j21) / det
        jinv31 = (j21 * j32 - j22 * j31) / det
        jinv32 = -(j11 * j32 - j12 * j31) / det
        jinv33 = (j11 * j22 - j12 * j21) / det

        dphi1 = jinv11[:, None] * d1[None, :] + jinv12[:, None] * d2[None, :] + jinv13[:, None] * d3[None, :]
        dphi2 = jinv21[:, None] * d1[None, :] + jinv22[:, None] * d2[None, :] + jinv23[:, None] * d3[None, :]
        dphi3 = jinv31[:, None] * d1[None, :] + jinv32[:, None] * d2[None, :] + jinv33[:, None] * d3[None, :]

        bq = np.zeros((n_elem, 6, 30), dtype=np.float64)
        for node_idx in range(10):
            c = 3 * node_idx
            d1n = dphi1[:, node_idx]
            d2n = dphi2[:, node_idx]
            d3n = dphi3[:, node_idx]
            # Local engineering strain order for this repo:
            # [e11, e22, e33, g12, g13, g23]
            bq[:, 0, c + 0] = d1n
            bq[:, 1, c + 1] = d2n
            bq[:, 2, c + 2] = d3n
            bq[:, 3, c + 0] = d2n
            bq[:, 3, c + 1] = d1n
            bq[:, 4, c + 0] = d3n
            bq[:, 4, c + 2] = d1n
            bq[:, 5, c + 1] = d3n
            bq[:, 5, c + 2] = d2n
        b_blocks[:, q] = bq
    return b_blocks.astype(np.float32)


def strain_from_local_displacements(b_block: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    """Apply a local B block to nodal displacement vectors."""
    b = np.asarray(b_block, dtype=np.float64)
    u = np.asarray(displacements, dtype=np.float64)
    if b.ndim == 2:
        b = b[None, :, :]
    if u.ndim == 1:
        u = u[None, :]
    return np.einsum("nij,nj->ni", b, u).astype(np.float32)


def positive_corner_volume_mask(
    local_coords: np.ndarray,
    displacements: np.ndarray,
    *,
    min_volume_ratio: float = 0.05,
) -> np.ndarray:
    """Reject deformations that invert or nearly collapse the corner tetrahedron."""
    coords = np.asarray(local_coords, dtype=np.float64)
    disp = np.asarray(displacements, dtype=np.float64)
    if coords.ndim == 3:
        coords = coords
    else:
        raise ValueError(f"Expected local_coords shape (n, 10, 3), got {coords.shape}.")
    if disp.ndim == 1:
        disp = disp[None, :]
    disp_nodes = disp.reshape(disp.shape[0], 10, 3)
    ref_corners = coords[:, :4, :]
    def_corners = ref_corners + disp_nodes[:, :4, :]
    ref_vol = corner_signed_volumes(ref_corners)
    def_vol = corner_signed_volumes(def_corners)
    ref_abs = np.maximum(np.abs(ref_vol), 1.0e-14)
    ratio = def_vol / ref_vol
    return (ratio > 0.0) & (np.abs(def_vol) >= min_volume_ratio * ref_abs)
