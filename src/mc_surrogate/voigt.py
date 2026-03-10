"""Utilities for Voigt/tensor conversions used by the constitutive law."""

from __future__ import annotations

import numpy as np


def _atleast_2d_last6(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError(f"Expected shape (6,), got {arr.shape}.")
        return arr[None, :]
    if arr.ndim == 2 and arr.shape[1] == 6:
        return arr
    raise ValueError(f"Expected shape (n, 6), got {arr.shape}.")


def strain_voigt_to_tensor(strain_eng: np.ndarray) -> np.ndarray:
    """Convert engineering strain Voigt vectors [e11,e22,e33,g12,g13,g23] to 3x3 tensors."""
    e = _atleast_2d_last6(strain_eng)
    out = np.zeros((e.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = e[:, 0]
    out[:, 1, 1] = e[:, 1]
    out[:, 2, 2] = e[:, 2]
    out[:, 0, 1] = out[:, 1, 0] = 0.5 * e[:, 3]
    out[:, 0, 2] = out[:, 2, 0] = 0.5 * e[:, 4]
    out[:, 1, 2] = out[:, 2, 1] = 0.5 * e[:, 5]
    return out


def tensor_to_strain_voigt(strain_tensor: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 strain tensors to engineering strain Voigt vectors."""
    eps = np.asarray(strain_tensor, dtype=float)
    if eps.ndim == 2:
        eps = eps[None, :, :]
    if eps.ndim != 3 or eps.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (n,3,3) or (3,3), got {eps.shape}.")
    out = np.zeros((eps.shape[0], 6), dtype=float)
    out[:, 0] = eps[:, 0, 0]
    out[:, 1] = eps[:, 1, 1]
    out[:, 2] = eps[:, 2, 2]
    out[:, 3] = 2.0 * eps[:, 0, 1]
    out[:, 4] = 2.0 * eps[:, 0, 2]
    out[:, 5] = 2.0 * eps[:, 1, 2]
    return out


def stress_voigt_to_tensor(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert stress Voigt vectors [s11,s22,s33,s12,s13,s23] to symmetric tensors."""
    s = _atleast_2d_last6(stress_voigt)
    out = np.zeros((s.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = s[:, 0]
    out[:, 1, 1] = s[:, 1]
    out[:, 2, 2] = s[:, 2]
    out[:, 0, 1] = out[:, 1, 0] = s[:, 3]
    out[:, 0, 2] = out[:, 2, 0] = s[:, 4]
    out[:, 1, 2] = out[:, 2, 1] = s[:, 5]
    return out


def tensor_to_stress_voigt(stress_tensor: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 stress tensors to stress Voigt vectors."""
    sig = np.asarray(stress_tensor, dtype=float)
    if sig.ndim == 2:
        sig = sig[None, :, :]
    if sig.ndim != 3 or sig.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (n,3,3) or (3,3), got {sig.shape}.")
    out = np.zeros((sig.shape[0], 6), dtype=float)
    out[:, 0] = sig[:, 0, 0]
    out[:, 1] = sig[:, 1, 1]
    out[:, 2] = sig[:, 2, 2]
    out[:, 3] = sig[:, 0, 1]
    out[:, 4] = sig[:, 0, 2]
    out[:, 5] = sig[:, 1, 2]
    return out


def principal_values_and_vectors_from_strain(strain_eng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return descending principal strains and associated eigenvectors."""
    eps = strain_voigt_to_tensor(strain_eng)
    vals_asc, vecs_asc = np.linalg.eigh(eps)
    vals = vals_asc[:, ::-1]
    vecs = vecs_asc[:, :, ::-1]
    return vals, vecs


def reconstruct_from_principal(values: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Reconstruct symmetric tensors from principal values and eigenvectors."""
    vals = np.asarray(values, dtype=float)
    vecs = np.asarray(eigvecs, dtype=float)
    if vals.ndim == 1:
        vals = vals[None, :]
    if vecs.ndim == 2:
        vecs = vecs[None, :, :]
    return np.einsum("nij,nj,nkj->nik", vecs, vals, vecs)
