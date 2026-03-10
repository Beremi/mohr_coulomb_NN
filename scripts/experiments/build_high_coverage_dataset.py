#!/usr/bin/env python
"""Build a high-coverage constitutive dataset for large-capacity surrogate studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import dataset_summary, save_dataset_hdf5
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.sampling import (
    DatasetGenerationConfig,
    MaterialRangeConfig,
    _concat_dicts,
    _pack_samples,
    generate_branch_balanced_arrays,
    random_rotation_matrices,
)
from mc_surrogate.voigt import tensor_to_strain_voigt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output HDF5 dataset path.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--wide-samples", type=int, default=30000)
    parser.add_argument("--focused-samples", type=int, default=30000)
    parser.add_argument("--path-points", type=int, default=151)
    parser.add_argument("--path-rotations", type=int, default=2)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser.parse_args()


def benchmark_materials() -> list[dict[str, float | str]]:
    return [
        {"name": "general_foundation", "c0": 15.0, "phi_deg": 38.0, "psi_deg": 0.0, "young": 50000.0, "poisson": 0.30},
        {"name": "weak_foundation", "c0": 10.0, "phi_deg": 35.0, "psi_deg": 0.0, "young": 50000.0, "poisson": 0.30},
        {"name": "general_slope", "c0": 18.0, "phi_deg": 32.0, "psi_deg": 0.0, "young": 20000.0, "poisson": 0.33},
        {"name": "cover_layer", "c0": 15.0, "phi_deg": 30.0, "psi_deg": 0.0, "young": 10000.0, "poisson": 0.33},
    ]


def focused_material_ranges() -> MaterialRangeConfig:
    return MaterialRangeConfig(
        cohesion_range=(5.0, 25.0),
        friction_deg_range=(25.0, 45.0),
        dilatancy_deg_range=(0.0, 10.0),
        young_range=(1.0e4, 6.0e4),
        poisson_range=(0.25, 0.38),
        strength_reduction_range=(0.9, 2.0),
    )


def make_principal_path(path_kind: str, n_points: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_points)
    if path_kind == "triaxial":
        return np.column_stack([2.4 * t, -1.2 * t, -1.2 * t]) * 1.0e-3
    if path_kind == "left_edge":
        return np.column_stack([2.2 * t, 2.12 * t, -2.3 * t]) * 1.0e-3
    if path_kind == "right_edge":
        return np.column_stack([2.3 * t, -1.15 * t, -1.22 * t]) * 1.0e-3
    if path_kind == "apex":
        return np.column_stack([1.8 * t, 1.8 * t, 1.8 * t]) * 1.0e-3
    raise ValueError(f"Unsupported path kind {path_kind!r}.")


def build_path_augmentation(
    *,
    n_points: int,
    rotations_per_path: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(seed)
    path_kinds = ("triaxial", "left_edge", "right_edge", "apex")
    strength_reductions = (1.0, 1.25, 1.5, 2.0)

    chunks: list[dict[str, np.ndarray]] = []
    meta_rows: list[dict[str, object]] = []

    for material in benchmark_materials():
        for srf in strength_reductions:
            for path_kind in path_kinds:
                principal = make_principal_path(path_kind, n_points)
                for rotation_id in range(rotations_per_path):
                    rotations = random_rotation_matrices(n_points, rng)
                    strain_eng = tensor_to_strain_voigt(np.einsum("nij,nj,nkj->nik", rotations, principal, rotations))
                    reduced = build_reduced_material_from_raw(
                        c0=np.full(n_points, material["c0"]),
                        phi_rad=np.deg2rad(np.full(n_points, material["phi_deg"])),
                        psi_rad=np.deg2rad(np.full(n_points, material["psi_deg"])),
                        young=np.full(n_points, material["young"]),
                        poisson=np.full(n_points, material["poisson"]),
                        strength_reduction=np.full(n_points, srf),
                        davis_type=["B"] * n_points,
                    )
                    response = constitutive_update_3d(
                        strain_eng,
                        c_bar=reduced.c_bar,
                        sin_phi=reduced.sin_phi,
                        shear=reduced.shear,
                        bulk=reduced.bulk,
                        lame=reduced.lame,
                    )
                    response._input_strain = strain_eng
                    material_dict = {
                        "c0": np.full(n_points, material["c0"]),
                        "phi_deg": np.full(n_points, material["phi_deg"]),
                        "psi_deg": np.full(n_points, material["psi_deg"]),
                        "phi_rad": np.deg2rad(np.full(n_points, material["phi_deg"])),
                        "psi_rad": np.deg2rad(np.full(n_points, material["psi_deg"])),
                        "young": np.full(n_points, material["young"]),
                        "poisson": np.full(n_points, material["poisson"]),
                        "strength_reduction": np.full(n_points, srf),
                        "davis_id": np.full(n_points, 1, dtype=int),
                        "davis_type": np.array(["B"] * n_points, dtype=object),
                        "reduced": reduced,
                    }
                    chunks.append(_pack_samples(material_dict, response))
                    meta_rows.append(
                        {
                            "material": material["name"],
                            "strength_reduction": srf,
                            "path_kind": path_kind,
                            "rotation_id": rotation_id,
                            "n_points": n_points,
                        }
                    )

    return _concat_dicts(chunks), {"path_kinds": path_kinds, "strength_reductions": strength_reductions, "rows": meta_rows}


def concatenate_arrays(chunks: list[dict[str, np.ndarray]], seed: int) -> dict[str, np.ndarray]:
    arrays = _concat_dicts(chunks)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(arrays["strain_eng"].shape[0])
    return {key: value[perm] for key, value in arrays.items()}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    wide_cfg = DatasetGenerationConfig(
        n_samples=args.wide_samples,
        seed=args.seed,
        candidate_batch=1536,
        include_tangent=False,
        max_abs_principal_strain=5.0e-3,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
    )
    focused_cfg = DatasetGenerationConfig(
        n_samples=args.focused_samples,
        seed=args.seed + 1,
        candidate_batch=1536,
        include_tangent=False,
        max_abs_principal_strain=2.5e-3,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
        material_ranges=focused_material_ranges(),
    )

    wide_arrays, wide_counts = generate_branch_balanced_arrays(wide_cfg)
    focused_arrays, focused_counts = generate_branch_balanced_arrays(focused_cfg)
    path_arrays, path_meta = build_path_augmentation(
        n_points=args.path_points,
        rotations_per_path=args.path_rotations,
        seed=args.seed + 2,
    )

    arrays = concatenate_arrays([wide_arrays, focused_arrays, path_arrays], seed=args.seed + 3)
    attrs = {
        "generator": "high_coverage_composite",
        "seed": args.seed,
        "components": {
            "wide_branch_balanced": {
                "n_samples": args.wide_samples,
                "counts": wide_counts,
                "max_abs_principal_strain": wide_cfg.max_abs_principal_strain,
                "material_ranges": wide_cfg.material_ranges.__dict__,
            },
            "focused_branch_balanced": {
                "n_samples": args.focused_samples,
                "counts": focused_counts,
                "max_abs_principal_strain": focused_cfg.max_abs_principal_strain,
                "material_ranges": focused_cfg.material_ranges.__dict__,
            },
            "benchmark_paths": path_meta,
        },
    }
    save_dataset_hdf5(
        output_path,
        arrays,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        attrs=attrs,
    )

    summary = dataset_summary(output_path)
    summary["component_sizes"] = {
        "wide_branch_balanced": int(wide_arrays["strain_eng"].shape[0]),
        "focused_branch_balanced": int(focused_arrays["strain_eng"].shape[0]),
        "benchmark_paths": int(path_arrays["strain_eng"].shape[0]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
