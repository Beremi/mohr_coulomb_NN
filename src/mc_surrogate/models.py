"""Neural-network models and feature builders for constitutive surrogates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .voigt import principal_values_and_vectors_from_strain, tensor_to_stress_voigt


def _safe_log(x: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), floor))


def lode_cos3theta_from_principal(principal_values: np.ndarray) -> np.ndarray:
    """Compute cos(3θ) from ordered principal values of a symmetric tensor."""
    p = np.asarray(principal_values, dtype=float)
    mean = np.mean(p, axis=1, keepdims=True)
    s = p - mean
    j2 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) / 6.0
    j3 = s[:, 0] * s[:, 1] * s[:, 2]
    denom = np.maximum(j2 ** 1.5, 1.0e-14)
    cos3theta = (3.0 * np.sqrt(3.0) / 2.0) * j3 / denom
    return np.clip(cos3theta, -1.0, 1.0)


def equivalent_deviatoric_measure(principal_values: np.ndarray) -> np.ndarray:
    """Simple equivalent magnitude of the deviatoric principal values."""
    p = np.asarray(principal_values, dtype=float)
    mean = np.mean(p, axis=1, keepdims=True)
    s = p - mean
    return np.sqrt(np.sum(s**2, axis=1))


def build_principal_features(
    strain_principal: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Build invariant-aware features for the recommended principal-stress model."""
    eps_p = np.asarray(strain_principal, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]
    trace = np.sum(eps_p, axis=1)
    eq_dev = equivalent_deviatoric_measure(eps_p)
    lode = lode_cos3theta_from_principal(eps_p)
    return np.column_stack(
        [
            eps_p,
            trace,
            eq_dev,
            lode,
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def build_raw_features(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Build raw-Voigt features for the baseline model."""
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]
    return np.column_stack(
        [
            strain_eng,
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def build_trial_features(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """
    Build raw-stress features around the elastic trial stress.

    This representation is better aligned with the actual return mapping and is
    more stable on real simulation states with very large signed strain ranges.
    """
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    c_bar = mat[:, 0]
    sin_phi = np.clip(mat[:, 1], -0.999999, 0.999999)
    shear = mat[:, 2]
    bulk = mat[:, 3]
    lame = mat[:, 4]

    trial = compute_trial_stress(strain_eng, material_reduced)

    scale = np.maximum(c_bar, 1.0)
    return np.column_stack(
        [
            np.arcsinh(strain_eng),
            np.arcsinh(trial / scale[:, None]),
            _safe_log(c_bar),
            np.arctanh(sin_phi),
            _safe_log(shear),
            _safe_log(bulk),
            _safe_log(lame),
        ]
    ).astype(np.float32)


def compute_trial_stress(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> np.ndarray:
    """Compute elastic trial stress in Voigt notation from engineering strain."""
    strain_eng = np.asarray(strain_eng, dtype=float)
    mat = np.asarray(material_reduced, dtype=float)
    shear = mat[:, 2]
    lame = mat[:, 4]
    trace = np.sum(strain_eng[:, :3], axis=1)
    trial = np.empty_like(strain_eng, dtype=float)
    trial[:, :3] = lame[:, None] * trace[:, None] + 2.0 * shear[:, None] * strain_eng[:, :3]
    trial[:, 3:] = shear[:, None] * strain_eng[:, 3:]
    return trial.astype(np.float32)


@dataclass
class Standardizer:
    """Feature or target standardization parameters."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Standardizer":
        arr = np.asarray(array, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1.0e-12, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=float)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=float)
        return (arr * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, dct: dict[str, list[float]]) -> "Standardizer":
        return cls(mean=np.asarray(dct["mean"], dtype=np.float32), std=np.asarray(dct["std"], dtype=np.float32))


class ResidualBlock(nn.Module):
    """Small residual block for tabular MLPs."""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.out_norm = nn.LayerNorm(width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.out_norm(x + self.net(x)))


class PrincipalStressNet(nn.Module):
    """Recommended spectral MLP with auxiliary branch classification head."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        n_branches: int = 5,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_stress = nn.Linear(width, 3)
        self.head_branch = nn.Linear(width, n_branches)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        stress = self.head_stress(h)
        stress, _ = torch.sort(stress, dim=-1, descending=True)
        branch_logits = self.head_branch(h)
        return {"stress": stress, "branch_logits": branch_logits}


class RawStressNet(nn.Module):
    """Baseline raw-input MLP that predicts all six stress components directly."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_stress = nn.Linear(width, 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return {"stress": self.head_stress(h)}


class RawStressBranchNet(nn.Module):
    """Raw-stress MLP with an auxiliary branch classification head."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        n_branches: int = 5,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])
        self.head_stress = nn.Linear(width, 6)
        self.head_branch = nn.Linear(width, n_branches)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return {
            "stress": self.head_stress(h),
            "branch_logits": self.head_branch(h),
        }


def build_model(
    model_kind: str,
    input_dim: int,
    width: int = 256,
    depth: int = 4,
    dropout: float = 0.0,
) -> nn.Module:
    """Factory for supported surrogate architectures."""
    if model_kind == "principal":
        return PrincipalStressNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind in {"raw", "trial_raw", "trial_raw_residual"}:
        return RawStressNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    if model_kind in {"raw_branch", "trial_raw_branch", "trial_raw_branch_residual"}:
        return RawStressBranchNet(input_dim=input_dim, width=width, depth=depth, dropout=dropout)
    raise ValueError(f"Unsupported model kind {model_kind!r}.")


def stress_tensor_from_principal_torch(principal_stress: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    """Reconstruct stress tensors V diag(s) V^T in torch."""
    return torch.einsum("nij,nj,nkj->nik", eigvecs, principal_stress, eigvecs)


def stress_voigt_from_principal_torch(principal_stress: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
    """Reconstruct stress Voigt vectors from principal stresses and eigenvectors."""
    tensor = stress_tensor_from_principal_torch(principal_stress, eigvecs)
    out = torch.zeros((tensor.shape[0], 6), dtype=tensor.dtype, device=tensor.device)
    out[:, 0] = tensor[:, 0, 0]
    out[:, 1] = tensor[:, 1, 1]
    out[:, 2] = tensor[:, 2, 2]
    out[:, 3] = tensor[:, 0, 1]
    out[:, 4] = tensor[:, 0, 2]
    out[:, 5] = tensor[:, 1, 2]
    return out


def stress_voigt_from_principal_numpy(principal_stress: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Reconstruct stress Voigt vectors from principal stresses and eigenvectors."""
    tensor = np.einsum("nij,nj,nkj->nik", eigvecs, principal_stress, eigvecs)
    return tensor_to_stress_voigt(tensor)


def spectral_decomposition_from_strain(strain_eng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute principal strains and eigenvectors from engineering strains."""
    return principal_values_and_vectors_from_strain(strain_eng)
