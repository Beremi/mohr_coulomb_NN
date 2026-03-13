"""Utilities for sampling constitutive states from captured real simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mohr_coulomb import constitutive_update_3d

REAL_EXPORT_MATERIAL_KEYS = ("c_bar", "sin_phi", "shear", "bulk", "lame")


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    return value


def _call_mode(group: h5py.Group) -> str:
    return str(_decode_attr(group.attrs["mode"]))


def _decode_stress(group: h5py.Group, sample_idx: np.ndarray) -> np.ndarray:
    base = int(np.asarray(group.attrs["stress_index_base"]).reshape(-1)[0])
    stress_idx = group["stress_index"][sample_idx, 0].astype(np.int64) - base
    return group["stress_unique"][:][stress_idx]


def sample_real_export(
    path: str | Path,
    *,
    samples_per_call: int = 256,
    seed: int = 0,
    include_modes: tuple[str, ...] | None = None,
    attach_exact_labels: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Sample states from a captured constitutive export and convert them into the
    framework dataset schema.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)

    strain_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    stress_principal_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    eigvec_parts: list[np.ndarray] = []
    call_index_parts: list[np.ndarray] = []
    mode_id_parts: list[np.ndarray] = []

    exact_abs_err_sum = 0.0
    exact_sq_err_sum = 0.0
    exact_count = 0
    exact_max_abs = 0.0

    with h5py.File(path, "r") as f:
        calls = f["calls"]
        call_names = list(calls.keys())
        mode_names: list[str] = []
        mode_to_id: dict[str, int] = {}

        for call_idx, call_name in enumerate(call_names):
            group = calls[call_name]
            mode = _call_mode(group)
            if include_modes is not None and mode not in include_modes:
                continue

            if mode not in mode_to_id:
                mode_to_id[mode] = len(mode_names)
                mode_names.append(mode)

            n = group["E"].shape[0]
            k = min(samples_per_call, n)
            sample_idx = np.sort(rng.choice(n, size=k, replace=False))

            strain = group["E"][sample_idx]
            stress = _decode_stress(group, sample_idx)
            material = np.column_stack([group[key][sample_idx, 0] for key in REAL_EXPORT_MATERIAL_KEYS])

            if attach_exact_labels:
                exact = constitutive_update_3d(
                    strain,
                    c_bar=material[:, 0],
                    sin_phi=material[:, 1],
                    shear=material[:, 2],
                    bulk=material[:, 3],
                    lame=material[:, 4],
                )
                stress_principal = exact.stress_principal.astype(np.float32)
                branch_id = exact.branch_id.astype(np.int8)
                eigvecs = exact.eigvecs.astype(np.float32)

                diff = exact.stress - stress
                exact_abs_err_sum += float(np.abs(diff).sum())
                exact_sq_err_sum += float(np.square(diff).sum())
                exact_count += int(diff.size)
                exact_max_abs = max(exact_max_abs, float(np.abs(diff).max()))
            else:
                stress_principal = np.zeros((k, 3), dtype=np.float32)
                branch_id = np.full(k, -1, dtype=np.int8)
                eigvecs = np.zeros((k, 3, 3), dtype=np.float32)

            strain_parts.append(strain.astype(np.float32))
            stress_parts.append(stress.astype(np.float32))
            material_parts.append(material.astype(np.float32))
            stress_principal_parts.append(stress_principal)
            branch_parts.append(branch_id)
            eigvec_parts.append(eigvecs)
            call_index_parts.append(np.full(k, call_idx, dtype=np.int32))
            mode_id_parts.append(np.full(k, mode_to_id[mode], dtype=np.int8))

        arrays = {
            "strain_eng": np.vstack(strain_parts),
            "stress": np.vstack(stress_parts),
            "material_reduced": np.vstack(material_parts),
            "stress_principal": np.vstack(stress_principal_parts),
            "branch_id": np.concatenate(branch_parts),
            "eigvecs": np.vstack(eigvec_parts),
            "source_call_id": np.concatenate(call_index_parts),
            "source_mode_id": np.concatenate(mode_id_parts),
        }

        attrs: dict[str, Any] = {
            "source_hdf5": str(path),
            "source_call_count": len(call_names),
            "samples_per_call": samples_per_call,
            "mode_names_json": json.dumps(mode_names),
            "source_call_names_json": json.dumps(call_names),
        }
        for key, value in f.attrs.items():
            attrs[f"source_attr_{key}"] = _decode_attr(value)

    if attach_exact_labels and exact_count > 0:
        attrs["exact_match_mae"] = exact_abs_err_sum / exact_count
        attrs["exact_match_rmse"] = float(np.sqrt(exact_sq_err_sum / exact_count))
        attrs["exact_match_max_abs"] = exact_max_abs

    return arrays, attrs
