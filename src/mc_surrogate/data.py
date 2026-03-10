"""HDF5 dataset I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .materials import DAVIS_TYPES
from .mohr_coulomb import BRANCH_NAMES

RAW_MATERIAL_COLUMNS = ("c0", "phi_deg", "psi_deg", "young", "poisson", "strength_reduction", "davis_id")
REDUCED_MATERIAL_COLUMNS = ("c_bar", "sin_phi", "shear", "bulk", "lame")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_TO_ID = {name: i for i, name in enumerate(SPLIT_NAMES)}


def save_dataset_hdf5(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    attrs: dict[str, Any] | None = None,
) -> Path:
    """Save arrays to an HDF5 file with train/val/test split ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = None
    for key, value in arrays.items():
        if n is None:
            n = int(np.asarray(value).shape[0])
        elif int(np.asarray(value).shape[0]) != n:
            raise ValueError(f"Array {key!r} has incompatible first dimension.")
    if n is None:
        raise ValueError("No arrays provided.")

    train_frac, val_frac, test_frac = split_fractions
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    split_id = np.full(n, SPLIT_TO_ID["test"], dtype=np.int8)
    split_id[perm[:n_train]] = SPLIT_TO_ID["train"]
    split_id[perm[n_train : n_train + n_val]] = SPLIT_TO_ID["val"]
    split_id[perm[n_train + n_val : n_train + n_val + n_test]] = SPLIT_TO_ID["test"]

    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            arr = np.asarray(value)
            f.create_dataset(key, data=arr, compression="gzip", shuffle=True)
        f.create_dataset("split_id", data=split_id, compression="gzip", shuffle=True)

        f.attrs["branch_names_json"] = json.dumps(BRANCH_NAMES)
        f.attrs["davis_types_json"] = json.dumps(DAVIS_TYPES)
        f.attrs["raw_material_columns_json"] = json.dumps(RAW_MATERIAL_COLUMNS)
        f.attrs["reduced_material_columns_json"] = json.dumps(REDUCED_MATERIAL_COLUMNS)
        f.attrs["split_names_json"] = json.dumps(SPLIT_NAMES)
        if attrs:
            for key, value in attrs.items():
                if isinstance(value, (dict, list, tuple)):
                    f.attrs[key] = json.dumps(value)
                else:
                    f.attrs[key] = value
    return path


def load_split_indices(path: str | Path, split: str) -> np.ndarray:
    """Return integer indices for the requested split."""
    if split not in SPLIT_TO_ID:
        raise ValueError(f"Unknown split {split!r}. Expected one of {SPLIT_NAMES}.")
    with h5py.File(path, "r") as f:
        split_id = f["split_id"][:]
    return np.flatnonzero(split_id == SPLIT_TO_ID[split])


def load_arrays(path: str | Path, keys: list[str] | tuple[str, ...], split: str | None = None) -> dict[str, np.ndarray]:
    """Load selected arrays from HDF5, optionally restricting to a split."""
    indices = None if split is None else load_split_indices(path, split)
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in keys:
            data = f[key]
            out[key] = data[:] if indices is None else data[indices]
    return out


def dataset_summary(path: str | Path) -> dict[str, Any]:
    """Return a small metadata summary for a dataset."""
    with h5py.File(path, "r") as f:
        summary = {
            "path": str(path),
            "n_samples": int(f["strain_eng"].shape[0]),
            "contains_tangent": "tangent" in f,
            "branch_names": json.loads(f.attrs["branch_names_json"]),
            "raw_material_columns": json.loads(f.attrs["raw_material_columns_json"]),
            "reduced_material_columns": json.loads(f.attrs["reduced_material_columns_json"]),
        }
        split_id = f["split_id"][:]
        branch_id = f["branch_id"][:]
    summary["split_counts"] = {
        name: int(np.sum(split_id == SPLIT_TO_ID[name])) for name in SPLIT_NAMES
    }
    summary["branch_counts"] = {
        BRANCH_NAMES[i]: int(np.sum(branch_id == i)) for i in range(len(BRANCH_NAMES))
    }
    return summary
