"""Dataset generation for constitutive surrogate training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .data import save_dataset_hdf5
from .materials import DAVIS_TYPES, build_reduced_material_from_raw
from .mohr_coulomb import BRANCH_NAMES, BRANCH_TO_ID, constitutive_update_3d
from .voigt import tensor_to_strain_voigt


@dataclass
class MaterialRangeConfig:
    """Ranges used when drawing raw geotechnical material parameters."""
    cohesion_range: tuple[float, float] = (2.0, 40.0)
    friction_deg_range: tuple[float, float] = (20.0, 50.0)
    dilatancy_deg_range: tuple[float, float] = (0.0, 15.0)
    young_range: tuple[float, float] = (5.0e3, 1.0e5)
    poisson_range: tuple[float, float] = (0.20, 0.45)
    strength_reduction_range: tuple[float, float] = (0.8, 3.0)
    davis_probabilities: tuple[float, float, float] = (0.15, 0.70, 0.15)
    zero_dilatancy_probability: float = 0.70


@dataclass
class DatasetGenerationConfig:
    """High-level configuration for HDF5 dataset generation."""
    n_samples: int
    seed: int = 0
    candidate_batch: int = 4096
    include_tangent: bool = False
    max_abs_principal_strain: float | None = None
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    branch_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "elastic": 0.2,
            "smooth": 0.2,
            "left_edge": 0.2,
            "right_edge": 0.2,
            "apex": 0.2,
        }
    )
    material_ranges: MaterialRangeConfig = field(default_factory=MaterialRangeConfig)


def _loguniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def sample_raw_materials(
    n: int,
    rng: np.random.Generator,
    cfg: MaterialRangeConfig,
) -> dict[str, np.ndarray]:
    """Sample raw materials from broad geotechnically plausible ranges."""
    c0 = _loguniform(rng, *cfg.cohesion_range, size=n)
    phi_deg = rng.uniform(*cfg.friction_deg_range, size=n)
    zero_mask = rng.random(n) < cfg.zero_dilatancy_probability
    psi_upper = np.minimum(phi_deg - 1.0, cfg.dilatancy_deg_range[1])
    psi_upper = np.maximum(psi_upper, cfg.dilatancy_deg_range[0])
    psi_deg = np.where(
        zero_mask,
        0.0,
        rng.uniform(cfg.dilatancy_deg_range[0], psi_upper, size=n),
    )
    young = _loguniform(rng, *cfg.young_range, size=n)
    poisson = rng.uniform(*cfg.poisson_range, size=n)
    strength_reduction = _loguniform(rng, *cfg.strength_reduction_range, size=n)
    davis_id = rng.choice(np.arange(3, dtype=int), size=n, p=np.asarray(cfg.davis_probabilities))
    davis_type = np.array([DAVIS_TYPES[i] for i in davis_id], dtype=object)

    return {
        "c0": c0,
        "phi_deg": phi_deg,
        "psi_deg": psi_deg,
        "phi_rad": np.deg2rad(phi_deg),
        "psi_rad": np.deg2rad(psi_deg),
        "young": young,
        "poisson": poisson,
        "strength_reduction": strength_reduction,
        "davis_id": davis_id,
        "davis_type": davis_type,
    }


def random_rotation_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random proper orthogonal matrices via QR decomposition."""
    a = rng.normal(size=(n, 3, 3))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    d[d == 0.0] = 1.0
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0.0, :, 0] *= -1.0
    return q


def _yield_scale_from_direction(
    direction: np.ndarray,
    c_bar: np.ndarray,
    sin_phi: np.ndarray,
    shear: np.ndarray,
    lame: np.ndarray,
) -> np.ndarray:
    d1 = direction[:, 0]
    d2 = direction[:, 1]
    d3 = direction[:, 2]
    i1 = d1 + d2 + d3
    a = 2.0 * shear * ((1.0 + sin_phi) * d1 - (1.0 - sin_phi) * d3) + 2.0 * lame * sin_phi * i1
    a = np.where(a <= 1.0e-12, 1.0e-12, a)
    return c_bar / a


def _principal_direction_template(
    branch: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if branch == "elastic" or branch == "smooth":
        m = rng.uniform(0.1, 1.0, size=n)
        a = rng.uniform(0.3, 1.8, size=n)
        b = rng.uniform(0.3, 1.8, size=n)
        e1 = m + a
        e2 = m + rng.uniform(-0.2, 0.2, size=n)
        e3 = m - b
    elif branch == "left_edge":
        m = rng.uniform(0.2, 1.1, size=n)
        spread = rng.uniform(0.6, 2.0, size=n)
        delta = rng.uniform(0.0, 0.05, size=n)
        e1 = m + spread + delta
        e2 = m + spread - delta
        e3 = m - 1.4 * spread
    elif branch == "right_edge":
        m = rng.uniform(0.2, 1.1, size=n)
        spread = rng.uniform(0.6, 2.0, size=n)
        delta = rng.uniform(0.0, 0.05, size=n)
        e1 = m + 1.4 * spread
        e2 = m - spread + delta
        e3 = m - spread - delta
    elif branch == "apex":
        base = rng.uniform(0.5, 2.0, size=n)
        pert = rng.uniform(-0.03, 0.03, size=(n, 3))
        vals = base[:, None] * (1.0 + pert)
        vals.sort(axis=1)
        return vals[:, ::-1]
    else:
        raise ValueError(f"Unsupported branch {branch!r}.")
    vals = np.column_stack([e1, e2, e3])
    vals.sort(axis=1)
    return vals[:, ::-1]


def _scale_factor_for_branch(branch: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if branch == "elastic":
        mix = rng.random(n)
        out = np.empty(n)
        small = mix < 0.30
        out[small] = np.exp(rng.uniform(np.log(1.0e-3), np.log(0.1), size=np.sum(small)))
        out[~small] = rng.uniform(0.10, 0.98, size=np.sum(~small))
        return out
    if branch == "smooth":
        return rng.uniform(1.02, 1.35, size=n)
    if branch == "left_edge":
        return rng.uniform(1.05, 1.80, size=n)
    if branch == "right_edge":
        return rng.uniform(1.05, 1.80, size=n)
    if branch == "apex":
        return rng.uniform(1.50, 4.00, size=n)
    raise ValueError(f"Unsupported branch {branch!r}.")


def _principal_to_global_engineering_strain(principal_strain: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    eps_tensor = np.einsum("nij,nj,nkj->nik", rotations, principal_strain, rotations)
    return tensor_to_strain_voigt(eps_tensor)


def _pack_samples(materials: dict[str, np.ndarray], response) -> dict[str, np.ndarray]:
    n = response.stress.shape[0]
    material_raw = np.column_stack(
        [
            materials["c0"],
            materials["phi_deg"],
            materials["psi_deg"],
            materials["young"],
            materials["poisson"],
            materials["strength_reduction"],
            materials["davis_id"].astype(float),
        ]
    )
    material_reduced = np.column_stack(
        [
            materials["reduced"].c_bar,
            materials["reduced"].sin_phi,
            materials["reduced"].shear,
            materials["reduced"].bulk,
            materials["reduced"].lame,
        ]
    )
    tangent = response.tangent if response.tangent is not None else None
    arrays = {
        "strain_eng": response._input_strain,  # injected by caller
        "stress": response.stress,
        "strain_principal": response.strain_principal,
        "stress_principal": response.stress_principal,
        "eigvecs": response.eigvecs,
        "branch_id": response.branch_id.astype(np.int8),
        "plastic_multiplier": response.plastic_multiplier,
        "f_trial": response.f_trial,
        "material_raw": material_raw,
        "material_reduced": material_reduced,
    }
    if tangent is not None:
        arrays["tangent"] = tangent
    assert arrays["strain_eng"].shape[0] == n
    return arrays


def _concat_dicts(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = chunks[0].keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        out[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return out


def _dataset_attrs_from_config(cfg: DatasetGenerationConfig) -> dict[str, object]:
    return {
        "generator": "branch_targeted_rejection_sampling",
        "seed": cfg.seed,
        "config": {
            "n_samples": cfg.n_samples,
            "candidate_batch": cfg.candidate_batch,
            "include_tangent": cfg.include_tangent,
            "max_abs_principal_strain": cfg.max_abs_principal_strain,
            "split_fractions": cfg.split_fractions,
            "branch_fractions": cfg.branch_fractions,
            "material_ranges": cfg.material_ranges.__dict__,
        },
    }


def generate_branch_balanced_arrays(
    cfg: DatasetGenerationConfig,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Generate branch-balanced constitutive samples and return arrays in memory."""
    rng = np.random.default_rng(cfg.seed)
    fractions = cfg.branch_fractions
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("branch_fractions must sum to 1.")
    desired_counts: Dict[str, int] = {}
    assigned = 0
    names = list(BRANCH_NAMES)
    for name in names[:-1]:
        count = int(round(fractions[name] * cfg.n_samples))
        desired_counts[name] = count
        assigned += count
    desired_counts[names[-1]] = cfg.n_samples - assigned

    saved_chunks: list[dict[str, np.ndarray]] = []

    for branch in BRANCH_NAMES:
        target_count = desired_counts[branch]
        remaining = target_count
        attempts = 0
        while remaining > 0:
            attempts += 1
            if attempts > 4000:
                raise RuntimeError(f"Failed to realize enough samples for branch {branch!r}.")
            batch = max(cfg.candidate_batch, remaining * 2)

            mats = sample_raw_materials(batch, rng, cfg.material_ranges)
            mats["reduced"] = build_reduced_material_from_raw(
                mats["c0"],
                mats["phi_rad"],
                mats["psi_rad"],
                mats["young"],
                mats["poisson"],
                mats["strength_reduction"],
                mats["davis_type"],
            )

            direction = _principal_direction_template(branch, batch, rng)
            alpha_yield = _yield_scale_from_direction(
                direction,
                mats["reduced"].c_bar,
                mats["reduced"].sin_phi,
                mats["reduced"].shear,
                mats["reduced"].lame,
            )
            factor = _scale_factor_for_branch(branch, batch, rng)
            principal_strain = direction * alpha_yield[:, None] * factor[:, None]
            rotations = random_rotation_matrices(batch, rng)
            strain_eng = _principal_to_global_engineering_strain(principal_strain, rotations)

            response = constitutive_update_3d(
                strain_eng,
                c_bar=mats["reduced"].c_bar,
                sin_phi=mats["reduced"].sin_phi,
                shear=mats["reduced"].shear,
                bulk=mats["reduced"].bulk,
                lame=mats["reduced"].lame,
                return_tangent=cfg.include_tangent,
            )
            response._input_strain = strain_eng  # small convenience for packing

            matched = response.branch_id == BRANCH_TO_ID[branch]
            if cfg.max_abs_principal_strain is not None:
                matched = matched & (np.max(np.abs(principal_strain), axis=1) <= cfg.max_abs_principal_strain)
            idx = np.flatnonzero(matched)
            if idx.size == 0:
                continue
            idx = idx[:remaining]

            selected_materials = {}
            for key, value in mats.items():
                if key == "reduced":
                    continue
                selected_materials[key] = np.asarray(value)[idx]
            selected_materials["reduced"] = build_reduced_material_from_raw(
                selected_materials["c0"],
                selected_materials["phi_rad"],
                selected_materials["psi_rad"],
                selected_materials["young"],
                selected_materials["poisson"],
                selected_materials["strength_reduction"],
                selected_materials["davis_type"],
            )

            selected_response = constitutive_update_3d(
                strain_eng[idx],
                c_bar=selected_materials["reduced"].c_bar,
                sin_phi=selected_materials["reduced"].sin_phi,
                shear=selected_materials["reduced"].shear,
                bulk=selected_materials["reduced"].bulk,
                lame=selected_materials["reduced"].lame,
                return_tangent=cfg.include_tangent,
            )
            selected_response._input_strain = strain_eng[idx]
            saved_chunks.append(_pack_samples(selected_materials, selected_response))
            remaining -= idx.size

    arrays = _concat_dicts(saved_chunks)
    perm = rng.permutation(arrays["strain_eng"].shape[0])
    arrays = {key: value[perm] for key, value in arrays.items()}
    counts = {name: int(np.sum(arrays["branch_id"] == BRANCH_TO_ID[name])) for name in BRANCH_NAMES}
    return arrays, counts


def generate_branch_balanced_dataset(
    output_path: str,
    cfg: DatasetGenerationConfig,
) -> dict[str, int]:
    """
    Generate an HDF5 dataset using branch-targeted rejection sampling.

    Returns
    -------
    dict
        Final branch counts in the saved dataset.
    """
    arrays, counts = generate_branch_balanced_arrays(cfg)
    attrs = _dataset_attrs_from_config(cfg)
    save_dataset_hdf5(
        output_path,
        arrays,
        split_fractions=cfg.split_fractions,
        seed=cfg.seed,
        attrs=attrs,
    )
    return counts
