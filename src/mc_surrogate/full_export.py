"""Utilities for the full constitutive export with stored U and sparse B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .data import RAW_MATERIAL_COLUMNS, REDUCED_MATERIAL_COLUMNS, SPLIT_NAMES, SPLIT_TO_ID
from .fe_p2_tetra import quadrature_volume_p2_tetra
from .mohr_coulomb import BRANCH_NAMES, constitutive_update_3d
from .real_materials import assign_material_families

FULL_EXPORT_MATERIAL_KEYS = ("c_bar", "sin_phi", "shear", "bulk", "lame")
FULL_EXPORT_IPS_PER_ELEMENT = 11


@dataclass(frozen=True)
class CoverElementCallBlock:
    """Element-grouped cover-layer data for one constitutive call."""

    call_name: str
    element_index: np.ndarray
    quadrature_reference: np.ndarray
    local_coords: np.ndarray
    local_displacements: np.ndarray
    strain_eng: np.ndarray
    stress: np.ndarray
    material_reduced: np.ndarray
    branch_id: np.ndarray
    stress_exact: np.ndarray
    stress_export: np.ndarray
    branch_exact_matches_export_stress: bool


@dataclass(frozen=True)
class CanonicalizedElementStates:
    """Canonical local geometry/displacement representation for one element batch."""

    local_coords: np.ndarray
    local_displacements: np.ndarray
    basis: np.ndarray
    centroid: np.ndarray
    characteristic_length: np.ndarray


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _read_material_rows(group: h5py.Group) -> np.ndarray:
    return np.column_stack([group[key][:, 0] for key in FULL_EXPORT_MATERIAL_KEYS]).astype(np.float32)


def _normalized(v: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def _problem_material_identifier(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["problem"]["material_identifier"][0].astype(np.int16)


def infer_problem_material_family_map(
    path: str | Path,
    *,
    reference_call: str = "call_000001",
) -> dict[str, int]:
    """
    Infer a map from material-family name to problem material id.

    The full export stores both per-element material ids and per-integration-point reduced
    material rows. We align them once on a reference call so later grouping does not rely
    on hard-coded ids.
    """
    problem_material = _problem_material_identifier(path)
    with h5py.File(path, "r") as f:
        reduced = _read_material_rows(f["calls"][reference_call])

    n_elem = problem_material.size
    if reduced.shape[0] % n_elem != 0:
        raise RuntimeError(
            f"Expected integration-point rows to be divisible by element count; got {reduced.shape[0]} rows "
            f"for {n_elem} elements."
        )
    ips_per_element = reduced.shape[0] // n_elem
    family_by_problem_id: dict[str, int] = {}

    for problem_id in np.unique(problem_material):
        elem_mask = problem_material == int(problem_id)
        row_mask = np.repeat(elem_mask, ips_per_element)
        unique = np.unique(reduced[row_mask], axis=0)
        if unique.shape[0] != 1:
            raise RuntimeError(
                f"Expected exactly one reduced material state for problem material id {problem_id}, "
                f"got {unique.shape[0]}."
            )
        fam = assign_material_families(unique)
        family_name = fam["material_names"][int(fam["material_id"][0])]
        family_by_problem_id[family_name] = int(problem_id)
    return family_by_problem_id


def family_element_indices(
    path: str | Path,
    *,
    family_name: str,
    reference_call: str = "call_000001",
) -> np.ndarray:
    """Return 0-based element indices for one material family from the full export."""
    family_map = infer_problem_material_family_map(path, reference_call=reference_call)
    if family_name not in family_map:
        known = ", ".join(sorted(family_map))
        raise ValueError(f"Unknown family name {family_name!r}; available: {known}.")
    problem_material = _problem_material_identifier(path)
    return np.flatnonzero(problem_material == family_map[family_name]).astype(np.int32)


def family_ip_rows(
    path: str | Path,
    *,
    family_name: str,
    reference_call: str = "call_000001",
) -> np.ndarray:
    """Return the flattened integration-point rows for one material family."""
    element_idx = family_element_indices(path, family_name=family_name, reference_call=reference_call)
    q = np.arange(FULL_EXPORT_IPS_PER_ELEMENT, dtype=np.int32)
    return (element_idx[:, None] * FULL_EXPORT_IPS_PER_ELEMENT + q[None, :]).reshape(-1)


def load_sparse_B(path: str | Path) -> csr_matrix:
    """Load the sparse global B operator from the full export."""
    with h5py.File(path, "r") as f:
        g = f["problem/B_coo"]
        row = g["row"][0].astype(np.int64) - 1
        col = g["col"][0].astype(np.int64) - 1
        val = g["val"][0].astype(np.float64)
        shape = tuple(int(x) for x in g["shape"][:, 0])
    return coo_matrix((val, (row, col)), shape=shape).tocsr()


def canonicalize_p2_element_states(
    local_coords: np.ndarray,
    local_displacements: np.ndarray,
) -> CanonicalizedElementStates:
    """
    Canonicalize P2 tetrahedral element states by translation, rotation, and scale.

    The frame is built from the corner tetrahedron:
    - origin at the corner centroid
    - e1 along corner 0 -> 1
    - e2 in the plane spanned by corner 0 -> 1 and corner 0 -> 2
    - e3 orthogonal to e1/e2, oriented so corner 3 has positive coordinate in e3
    """
    coords = np.asarray(local_coords, dtype=np.float64)
    disp = np.asarray(local_displacements, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (10, 3):
        raise ValueError(f"Expected local_coords shape (n, 10, 3), got {coords.shape}.")
    if disp.shape != coords.shape:
        raise ValueError(f"Expected local_displacements shape {coords.shape}, got {disp.shape}.")

    corner = coords[:, :4, :]
    centroid = np.mean(corner, axis=1, keepdims=True)
    h = np.linalg.norm(corner[:, 1, :] - corner[:, 0, :], axis=1, keepdims=True)
    h = np.maximum(h, 1.0e-8)
    h_scalar = h[:, 0]

    e1 = _normalized(corner[:, 1, :] - corner[:, 0, :])
    v2 = corner[:, 2, :] - corner[:, 0, :]
    v2 = v2 - np.sum(v2 * e1, axis=1, keepdims=True) * e1
    bad_v2 = np.linalg.norm(v2, axis=1) < 1.0e-10
    if np.any(bad_v2):
        alt = corner[bad_v2, 3, :] - corner[bad_v2, 0, :]
        alt = alt - np.sum(alt * e1[bad_v2], axis=1, keepdims=True) * e1[bad_v2]
        v2[bad_v2] = alt
    e2 = _normalized(v2)
    e3 = _normalized(np.cross(e1, e2))

    rel4 = corner[:, 3, :] - corner[:, 0, :]
    flip = np.sum(rel4 * e3, axis=1) < 0.0
    e2[flip] *= -1.0
    e3[flip] *= -1.0

    basis = np.stack([e1, e2, e3], axis=2)
    coords_local = np.einsum("nij,njk->nik", coords - centroid, basis) / h_scalar[:, None, None]
    disp_local = np.einsum("nij,njk->nik", disp, basis) / h_scalar[:, None, None]
    return CanonicalizedElementStates(
        local_coords=coords_local.astype(np.float32),
        local_displacements=disp_local.astype(np.float32),
        basis=basis.astype(np.float32),
        centroid=centroid[:, 0, :].astype(np.float32),
        characteristic_length=h_scalar.astype(np.float32),
    )


def infer_material_family_mask(
    path: str | Path,
    *,
    family_name: str,
    reference_call: str = "call_000001",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer row masks for one material family from a reference call.

    Returns:
    - ip_mask: shape (n_ip,)
    - b_row_mask: shape (6*n_ip,)
    """
    with h5py.File(path, "r") as f:
        mat = _read_material_rows(f["calls"][reference_call])
    fam = assign_material_families(mat)
    try:
        family_idx = fam["material_names"].index(family_name)
    except ValueError as exc:
        raise ValueError(f"Unknown family name {family_name!r}.") from exc
    ip_mask = fam["material_id"] == family_idx
    b_row_mask = np.repeat(ip_mask, 6)
    return ip_mask, b_row_mask


def sample_cover_family_dataset(
    path: str | Path,
    output_path: str | Path,
    *,
    family_name: str = "cover_layer",
    samples_per_call: int = 256,
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    use_exact_stress: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """
    Build a sampled single-family dataset from the full export with call-level splits.
    """
    path = Path(path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ip_mask, _ = infer_material_family_mask(path, family_name=family_name)

    with h5py.File(path, "r") as f:
        call_names = list(f["calls"].keys())

    perm_calls = rng.permutation(len(call_names))
    n_train = int(round(split_fractions[0] * len(call_names)))
    n_val = int(round(split_fractions[1] * len(call_names)))
    train_calls = perm_calls[:n_train]
    val_calls = perm_calls[n_train : n_train + n_val]
    test_calls = perm_calls[n_train + n_val :]

    call_to_split = np.full(len(call_names), SPLIT_TO_ID["test"], dtype=np.int8)
    call_to_split[train_calls] = SPLIT_TO_ID["train"]
    call_to_split[val_calls] = SPLIT_TO_ID["val"]
    call_to_split[test_calls] = SPLIT_TO_ID["test"]

    strain_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    stress_principal_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    eigvec_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    call_id_parts: list[np.ndarray] = []
    row_in_call_parts: list[np.ndarray] = []

    exact_abs_sum = 0.0
    exact_sq_sum = 0.0
    exact_count = 0
    exact_max = 0.0

    with h5py.File(path, "r") as f:
        calls = f["calls"]
        for call_idx, call_name in enumerate(call_names):
            grp = calls[call_name]
            all_rows = np.flatnonzero(ip_mask)
            k = min(samples_per_call, all_rows.size)
            sample_rows = np.sort(rng.choice(all_rows, size=k, replace=False))
            strain = grp["E"][sample_rows].astype(np.float32)
            stress_export = grp["S"][sample_rows].astype(np.float32)
            material = _read_material_rows(grp)[sample_rows]

            exact = constitutive_update_3d(
                strain,
                c_bar=material[:, 0],
                sin_phi=material[:, 1],
                shear=material[:, 2],
                bulk=material[:, 3],
                lame=material[:, 4],
            )
            exact_stress = exact.stress.astype(np.float32)
            diff = exact_stress - stress_export
            exact_abs_sum += float(np.abs(diff).sum())
            exact_sq_sum += float(np.square(diff).sum())
            exact_count += int(diff.size)
            exact_max = max(exact_max, float(np.abs(diff).max()))

            strain_parts.append(strain)
            stress_parts.append(exact_stress if use_exact_stress else stress_export)
            material_parts.append(material.astype(np.float32))
            stress_principal_parts.append(exact.stress_principal.astype(np.float32))
            branch_parts.append(exact.branch_id.astype(np.int8))
            eigvec_parts.append(exact.eigvecs.astype(np.float32))
            split_parts.append(np.full(k, call_to_split[call_idx], dtype=np.int8))
            call_id_parts.append(np.full(k, call_idx, dtype=np.int32))
            row_in_call_parts.append(sample_rows.astype(np.int32))

        arrays = {
            "strain_eng": np.vstack(strain_parts),
            "stress": np.vstack(stress_parts),
            "material_reduced": np.vstack(material_parts),
            "stress_principal": np.vstack(stress_principal_parts),
            "branch_id": np.concatenate(branch_parts),
            "eigvecs": np.vstack(eigvec_parts),
            "split_id": np.concatenate(split_parts),
            "source_call_id": np.concatenate(call_id_parts),
            "source_row_in_call": np.concatenate(row_in_call_parts),
        }

        attrs = {
            "source_hdf5": str(path),
            "family_name": family_name,
            "samples_per_call": samples_per_call,
            "use_exact_stress": bool(use_exact_stress),
            "source_call_names_json": json.dumps(call_names),
            "train_call_names_json": json.dumps([call_names[i] for i in train_calls]),
            "val_call_names_json": json.dumps([call_names[i] for i in val_calls]),
            "test_call_names_json": json.dumps([call_names[i] for i in test_calls]),
            "exact_match_mae": exact_abs_sum / max(exact_count, 1),
            "exact_match_rmse": float(np.sqrt(exact_sq_sum / max(exact_count, 1))),
            "exact_match_max_abs": exact_max,
            "cover_row_count_per_call": int(ip_mask.sum()),
            "branch_names_json": json.dumps(BRANCH_NAMES),
            "raw_material_columns_json": json.dumps(RAW_MATERIAL_COLUMNS),
            "reduced_material_columns_json": json.dumps(REDUCED_MATERIAL_COLUMNS),
            "split_names_json": json.dumps(SPLIT_NAMES),
        }
        for key, value in f.attrs.items():
            attrs[f"source_attr_{key}"] = _json_safe(value)

    with h5py.File(output_path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(_json_safe(value))
            else:
                f.attrs[key] = value

    summary = {
        "output_path": str(output_path),
        "n_samples": int(arrays["strain_eng"].shape[0]),
        "split_counts": {
            name: int(np.sum(arrays["split_id"] == SPLIT_TO_ID[name]))
            for name in SPLIT_NAMES
        },
        "branch_counts": {
            BRANCH_NAMES[i]: int(np.sum(arrays["branch_id"] == i))
            for i in range(len(BRANCH_NAMES))
        },
        "exact_match_mae": attrs["exact_match_mae"],
        "exact_match_rmse": attrs["exact_match_rmse"],
        "exact_match_max_abs": attrs["exact_match_max_abs"],
        "cover_row_count_per_call": attrs["cover_row_count_per_call"],
    }
    return output_path, summary


def load_cover_call_archive(
    path: str | Path,
    *,
    family_name: str = "cover_layer",
    split_call_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load full-call U traces and cover-layer reduced parameters for selected calls.
    """
    ip_mask, b_row_mask = infer_material_family_mask(path, family_name=family_name)
    call_names: list[str] = []
    u_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []

    with h5py.File(path, "r") as f:
        for call_name, grp in f["calls"].items():
            if split_call_names is not None and call_name not in split_call_names:
                continue
            mat = _read_material_rows(grp)[ip_mask]
            # The family is fixed; reduced state should be constant inside the family for one call.
            unique = np.unique(mat, axis=0)
            if unique.shape[0] != 1:
                raise RuntimeError(f"Expected one reduced state for {family_name!r} in {call_name}, got {unique.shape[0]}.")
            call_names.append(call_name)
            u_parts.append(grp["U"][0].astype(np.float32))
            material_parts.append(unique[0].astype(np.float32))

    return {
        "call_names": call_names,
        "U": np.stack(u_parts, axis=0),
        "material_reduced": np.stack(material_parts, axis=0),
        "ip_mask": ip_mask,
        "b_row_mask": b_row_mask,
    }


def iter_family_element_blocks(
    path: str | Path,
    *,
    family_name: str = "cover_layer",
    call_names: list[str] | None = None,
    reference_call: str = "call_000001",
    use_exact_stress: bool = True,
    max_elements_per_call: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[CoverElementCallBlock]]:
    """
    Group one material family into element blocks with 11 integration points per element.

    Returns:
    - cover element indices for the chosen family
    - one `CoverElementCallBlock` per selected constitutive call
    """
    path = Path(path)
    element_idx = family_element_indices(path, family_name=family_name, reference_call=reference_call)
    if max_elements_per_call is not None and max_elements_per_call < element_idx.size:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(element_idx.size, size=max_elements_per_call, replace=False))
        element_idx = element_idx[chosen]
    ip_rows = (element_idx[:, None] * FULL_EXPORT_IPS_PER_ELEMENT + np.arange(FULL_EXPORT_IPS_PER_ELEMENT)[None, :]).reshape(-1)
    q_ref = np.asarray(quadrature_volume_p2_tetra()[0].T, dtype=np.float32)

    with h5py.File(path, "r") as f:
        coord = f["problem"]["coord"][:].astype(np.float32)
        elem = f["problem"]["elem"][:].astype(np.int64) - 1
        local_coords = coord[elem[element_idx]]
        selected_calls = list(f["calls"].keys()) if call_names is None else list(call_names)
        blocks: list[CoverElementCallBlock] = []

        local_dof = np.stack([3 * elem[element_idx] + i for i in range(3)], axis=-1).reshape(element_idx.size, 30)
        for call_name in selected_calls:
            grp = f["calls"][call_name]
            strain = grp["E"][ip_rows].astype(np.float32).reshape(element_idx.size, FULL_EXPORT_IPS_PER_ELEMENT, 6)
            stress_export = grp["S"][ip_rows].astype(np.float32).reshape(element_idx.size, FULL_EXPORT_IPS_PER_ELEMENT, 6)
            material_rows = _read_material_rows(grp)[ip_rows].reshape(element_idx.size, FULL_EXPORT_IPS_PER_ELEMENT, 5)
            unique_mat = material_rows[:, 0, :]
            if not np.allclose(material_rows, unique_mat[:, None, :], atol=1.0e-7, rtol=1.0e-7):
                raise RuntimeError(f"Expected one reduced material row per element in {call_name}.")

            u_global = grp["U"][0].astype(np.float32)
            local_disp = u_global[local_dof].reshape(element_idx.size, 10, 3)

            exact = constitutive_update_3d(
                strain.reshape(-1, 6),
                c_bar=np.repeat(unique_mat[:, 0], FULL_EXPORT_IPS_PER_ELEMENT),
                sin_phi=np.repeat(unique_mat[:, 1], FULL_EXPORT_IPS_PER_ELEMENT),
                shear=np.repeat(unique_mat[:, 2], FULL_EXPORT_IPS_PER_ELEMENT),
                bulk=np.repeat(unique_mat[:, 3], FULL_EXPORT_IPS_PER_ELEMENT),
                lame=np.repeat(unique_mat[:, 4], FULL_EXPORT_IPS_PER_ELEMENT),
            )
            exact_stress = exact.stress.astype(np.float32).reshape(element_idx.size, FULL_EXPORT_IPS_PER_ELEMENT, 6)
            branch_id = exact.branch_id.astype(np.int8).reshape(element_idx.size, FULL_EXPORT_IPS_PER_ELEMENT)

            blocks.append(
                CoverElementCallBlock(
                    call_name=call_name,
                    element_index=element_idx.copy(),
                    quadrature_reference=q_ref.copy(),
                    local_coords=local_coords.copy(),
                    local_displacements=local_disp,
                    strain_eng=strain,
                    stress=exact_stress if use_exact_stress else stress_export,
                    material_reduced=unique_mat.astype(np.float32),
                    branch_id=branch_id,
                    stress_exact=exact_stress,
                    stress_export=stress_export,
                    branch_exact_matches_export_stress=bool(
                        np.allclose(exact_stress, stress_export, atol=1.0e-5, rtol=1.0e-5)
                    ),
                )
            )
    return element_idx, blocks


def build_test_only_dataset(
    output_path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    attrs: dict[str, Any] | None = None,
) -> Path:
    """Write a dataset where all rows belong to the test split."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.create_dataset("split_id", data=np.full(arrays["strain_eng"].shape[0], SPLIT_TO_ID["test"], dtype=np.int8))
        f.attrs["branch_names_json"] = json.dumps(BRANCH_NAMES)
        f.attrs["raw_material_columns_json"] = json.dumps(RAW_MATERIAL_COLUMNS)
        f.attrs["reduced_material_columns_json"] = json.dumps(REDUCED_MATERIAL_COLUMNS)
        f.attrs["split_names_json"] = json.dumps(SPLIT_NAMES)
        if attrs:
            for key, value in attrs.items():
                if isinstance(value, (dict, list, tuple)):
                    f.attrs[key] = json.dumps(_json_safe(value))
                else:
                    f.attrs[key] = value
    return output_path


def split_full_export_call_names(
    path: str | Path,
    *,
    split_fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
    seed: int = 0,
) -> dict[str, list[str]]:
    """Create a deterministic call-level split for the full export."""
    if len(split_fractions) != 3:
        raise ValueError("split_fractions must have length 3.")
    if not np.isclose(sum(split_fractions), 1.0):
        raise ValueError(f"split_fractions must sum to 1.0, got {split_fractions}.")

    with h5py.File(path, "r") as f:
        call_names = list(f["calls"].keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(call_names))
    n_train = int(round(split_fractions[0] * len(call_names)))
    n_val = int(round(split_fractions[1] * len(call_names)))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return {
        "generator_fit": [call_names[i] for i in train_idx],
        "real_val": [call_names[i] for i in val_idx],
        "real_test": [call_names[i] for i in test_idx],
    }
