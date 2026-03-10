"""Inference helpers and a constitutive-style surrogate wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .materials import build_reduced_material_from_raw
from .training import predict_with_checkpoint


@dataclass
class ConstitutiveSurrogate:
    """Load a checkpoint and expose a constitutive-operator-like prediction API."""
    checkpoint_path: str
    device: str = "cpu"

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> "ConstitutiveSurrogate":
        return cls(checkpoint_path=str(checkpoint_path), device=device)

    def predict_reduced(
        self,
        strain_eng: np.ndarray,
        *,
        c_bar: np.ndarray | float,
        sin_phi: np.ndarray | float,
        shear: np.ndarray | float,
        bulk: np.ndarray | float,
        lame: np.ndarray | float,
    ) -> dict[str, np.ndarray]:
        strain = np.asarray(strain_eng, dtype=float)
        if strain.ndim == 1:
            strain = strain[None, :]
        n = strain.shape[0]
        material_reduced = np.column_stack(
            [
                np.broadcast_to(np.asarray(c_bar, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(sin_phi, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(shear, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(bulk, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(lame, dtype=float), (n,)).reshape(-1),
            ]
        )
        return predict_with_checkpoint(
            self.checkpoint_path,
            strain,
            material_reduced,
            device=self.device,
        )

    def predict_raw(
        self,
        strain_eng: np.ndarray,
        *,
        c0: np.ndarray | float,
        phi_rad: np.ndarray | float,
        psi_rad: np.ndarray | float,
        young: np.ndarray | float,
        poisson: np.ndarray | float,
        strength_reduction: np.ndarray | float,
        davis_type: str | int | list[str | int],
    ) -> dict[str, np.ndarray]:
        reduced = build_reduced_material_from_raw(
            c0=c0,
            phi_rad=phi_rad,
            psi_rad=psi_rad,
            young=young,
            poisson=poisson,
            strength_reduction=strength_reduction,
            davis_type=davis_type,
        )
        return self.predict_reduced(
            strain_eng,
            c_bar=reduced.c_bar,
            sin_phi=reduced.sin_phi,
            shear=reduced.shear,
            bulk=reduced.bulk,
            lame=reduced.lame,
        )
