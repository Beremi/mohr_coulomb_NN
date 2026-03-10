#!/usr/bin/env python
"""Compare the Python constitutive implementation against the original Octave code."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat, loadmat

from mc_surrogate.data import load_arrays
from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import constitutive_update_3d
from mc_surrogate.sampling import MaterialRangeConfig, random_rotation_matrices, sample_raw_materials
from mc_surrogate.voigt import tensor_to_strain_voigt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octave-repo", required=True, help="Path to local slope_stability repository clone.")
    parser.add_argument("--dataset", help="Optional HDF5 dataset to sample from.")
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", help="Optional output JSON path.")
    return parser.parse_args()


def _resolve_matlab_root(repo_path: Path) -> Path:
    repo_path = repo_path.resolve()
    if (repo_path / "+CONSTITUTIVE_PROBLEM").exists():
        return repo_path
    if (repo_path / "slope_stability" / "+CONSTITUTIVE_PROBLEM").exists():
        return repo_path / "slope_stability"
    raise FileNotFoundError(
        f"Could not find +CONSTITUTIVE_PROBLEM under {repo_path} or {repo_path / 'slope_stability'}."
    )


def _random_states(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mats = sample_raw_materials(n, rng, MaterialRangeConfig())
    principal = rng.normal(size=(n, 3))
    principal.sort(axis=1)
    principal = principal[:, ::-1] * 5.0e-4
    rotations = random_rotation_matrices(n, rng)
    strain_eng = tensor_to_strain_voigt(np.einsum("nij,nj,nkj->nik", rotations, principal, rotations))
    material_raw = np.column_stack(
        [
            mats["c0"],
            mats["phi_deg"],
            mats["psi_deg"],
            mats["young"],
            mats["poisson"],
            mats["strength_reduction"],
            mats["davis_id"].astype(float),
        ]
    )
    return strain_eng, material_raw


def _load_reference_states(dataset_path: str, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_arrays(dataset_path, ["strain_eng", "material_raw"], split=None)
    n_total = arrays["strain_eng"].shape[0]
    n = min(max_points, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n, replace=False)
    return arrays["strain_eng"][idx], arrays["material_raw"][idx]


def main() -> None:
    args = parse_args()
    matlab_root = _resolve_matlab_root(Path(args.octave_repo))
    octave_exe = shutil.which("octave") or shutil.which("octave-cli")
    if octave_exe is None:
        raise RuntimeError("Octave executable not found on PATH.")

    if args.dataset:
        strain_eng, material_raw = _load_reference_states(args.dataset, args.max_points, args.seed)
    else:
        strain_eng, material_raw = _random_states(args.max_points, args.seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_mat = tmpdir / "input.mat"
        output_mat = tmpdir / "output.mat"
        script_path = tmpdir / "run_compare.m"

        savemat(
            input_mat,
            {
                "strain_eng": strain_eng,
                "material_raw": material_raw,
            },
        )

        script_path.write_text(
            f"""
            addpath(genpath('{str(matlab_root).replace("'", "''")}'));
            data = load('{str(input_mat).replace("'", "''")}');
            strain_eng = data.strain_eng';
            material_raw = data.material_raw;
            c0 = material_raw(:,1)';
            phi = deg2rad(material_raw(:,2)');
            psi = deg2rad(material_raw(:,3)');
            young = material_raw(:,4)';
            poisson = material_raw(:,5)';
            strength_reduction = material_raw(:,6)';
            davis_id = material_raw(:,7)';
            shear = young ./ (2 * (1 + poisson));
            bulk = young ./ (3 * (1 - 2 * poisson));
            lame = bulk - (2/3) * shear;
            n = size(strain_eng, 2);
            S = zeros(6, n);
            for did = 0:2
              idx = find(round(davis_id) == did);
              if isempty(idx)
                continue;
              end
              if did == 0
                dchar = 'A';
              elseif did == 1
                dchar = 'B';
              else
                dchar = 'C';
              end
              [c_bar, sin_phi] = CONSTITUTIVE_PROBLEM.reduction(c0(idx), phi(idx), psi(idx), strength_reduction(idx), dchar);
              [S(:, idx), ~] = CONSTITUTIVE_PROBLEM.constitutive_problem_3D(strain_eng(:, idx), c_bar, sin_phi, shear(idx), bulk(idx), lame(idx));
            end
            save('-mat', '{str(output_mat).replace("'", "''")}', 'S');
            """,
            encoding="utf-8",
        )

        subprocess.run(
            [octave_exe, "--quiet", "--no-gui", str(script_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        octave_out = loadmat(output_mat)
        stress_octave = np.asarray(octave_out["S"].T, dtype=float)

    reduced = build_reduced_material_from_raw(
        c0=material_raw[:, 0],
        phi_rad=np.deg2rad(material_raw[:, 1]),
        psi_rad=np.deg2rad(material_raw[:, 2]),
        young=material_raw[:, 3],
        poisson=material_raw[:, 4],
        strength_reduction=material_raw[:, 5],
        davis_type=material_raw[:, 6].astype(int),
    )
    stress_python = constitutive_update_3d(
        strain_eng,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    ).stress

    diff = stress_python - stress_octave
    result = {
        "n_points": int(stress_python.shape[0]),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
        "per_component_mae": np.mean(np.abs(diff), axis=0).tolist(),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
