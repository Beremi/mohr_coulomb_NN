#!/usr/bin/env python
"""Compare the exact constitutive law and a trained surrogate along a loading path."""

from __future__ import annotations

import argparse
import numpy as np

from mc_surrogate.inference import ConstitutiveSurrogate
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.viz import plot_path_comparison
from mc_surrogate.voigt import tensor_to_strain_voigt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--path-kind", choices=("triaxial", "left_edge", "right_edge", "apex"), default="triaxial")
    parser.add_argument("--c0", type=float, default=15.0)
    parser.add_argument("--phi-deg", type=float, default=35.0)
    parser.add_argument("--psi-deg", type=float, default=0.0)
    parser.add_argument("--young", type=float, default=2.0e4)
    parser.add_argument("--poisson", type=float, default=0.30)
    parser.add_argument("--strength-reduction", type=float, default=1.2)
    parser.add_argument("--davis-type", default="B", choices=("A", "B", "C"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def make_path(path_kind: str, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_points)
    if path_kind == "triaxial":
        principal = np.column_stack([2.0 * t, -1.0 * t, -1.0 * t]) * 1.0e-3
    elif path_kind == "left_edge":
        principal = np.column_stack([2.0 * t, 1.95 * t, -2.0 * t]) * 1.0e-3
    elif path_kind == "right_edge":
        principal = np.column_stack([2.0 * t, -1.0 * t, -1.05 * t]) * 1.0e-3
    else:
        principal = np.column_stack([t, t, t]) * 1.0e-3
    rotations = np.repeat(np.eye(3)[None, :, :], n_points, axis=0)
    strain_eng = tensor_to_strain_voigt(np.einsum("nij,nj,nkj->nik", rotations, principal, rotations))
    return t, strain_eng


def main() -> None:
    args = parse_args()
    t, strain_eng = make_path(args.path_kind, args.n_points)

    reduced = build_reduced_material_from_raw(
        c0=np.full(args.n_points, args.c0),
        phi_rad=np.deg2rad(np.full(args.n_points, args.phi_deg)),
        psi_rad=np.deg2rad(np.full(args.n_points, args.psi_deg)),
        young=np.full(args.n_points, args.young),
        poisson=np.full(args.n_points, args.poisson),
        strength_reduction=np.full(args.n_points, args.strength_reduction),
        davis_type=[args.davis_type] * args.n_points,
    )
    exact = constitutive_update_3d(
        strain_eng,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    surrogate = ConstitutiveSurrogate.from_checkpoint(args.checkpoint, device=args.device)
    pred = surrogate.predict_reduced(
        strain_eng,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    pred_principal = pred.get("stress_principal")
    if pred_principal is None:
        raise RuntimeError("Path comparison expects a principal-stress surrogate.")
    plot_path_comparison(
        t,
        exact.stress_principal,
        pred_principal,
        args.output,
        title=f"Path comparison: {args.path_kind}",
    )
    print(args.output)


if __name__ == "__main__":
    main()
