"""Shared helpers for synthetic cover-layer branch-generation experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

from .fe_p2_tetra import build_local_b_blocks_from_coords, positive_corner_volume_mask
from .full_export import canonicalize_p2_element_states, iter_family_element_blocks
from .mohr_coulomb import constitutive_update_3d


def load_split_calls(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in payload["splits"].items()}


def load_call_regimes(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["call_name"]: row for row in payload["calls"]}


def pick_calls(
    call_names: list[str],
    *,
    count: int,
    selection: str,
    regimes: dict[str, dict[str, float]] | None,
) -> list[str]:
    if count >= len(call_names):
        return list(call_names)
    if selection == "first":
        return list(call_names[:count])
    if selection == "spread_p95":
        if regimes is None:
            raise ValueError("spread_p95 selection requires regime summaries.")
        ranked = sorted(call_names, key=lambda name: regimes[name]["strain_norm_p95"])
        positions = np.linspace(0, len(ranked) - 1, num=count)
        picked = [ranked[int(round(pos))] for pos in positions]
        return list(dict.fromkeys(picked))
    raise ValueError(f"Unknown call selection mode {selection!r}.")


def collect_blocks(
    export_path: Path,
    *,
    call_names: list[str],
    max_elements_per_call: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, blocks = iter_family_element_blocks(
        export_path,
        family_name="cover_layer",
        call_names=call_names,
        max_elements_per_call=max_elements_per_call,
        seed=seed,
    )
    coords = np.concatenate([block.local_coords for block in blocks], axis=0)
    disp = np.concatenate([block.local_displacements for block in blocks], axis=0)
    strain = np.concatenate([block.strain_eng for block in blocks], axis=0)
    branch = np.concatenate([block.branch_id for block in blocks], axis=0)
    material = np.concatenate([block.material_reduced for block in blocks], axis=0)
    return coords, disp, strain, branch, material


def fit_pca(samples: np.ndarray, *, explained_variance: float, max_rank: int) -> dict[str, np.ndarray]:
    mean = np.mean(samples, axis=0, dtype=np.float64)
    centered = samples - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    rank = int(np.searchsorted(energy, explained_variance) + 1)
    rank = max(1, min(rank, max_rank, vt.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    latent = centered @ basis
    latent_mean = np.mean(latent, axis=0, dtype=np.float64)
    latent_cov = np.cov(latent, rowvar=False).astype(np.float64)
    if rank == 1:
        latent_cov = np.array([[float(latent_cov)]], dtype=np.float64)
    latent_std = np.std(latent, axis=0, dtype=np.float64)
    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "latent": latent.astype(np.float32),
        "latent_mean": latent_mean.astype(np.float32),
        "latent_cov": latent_cov.astype(np.float32),
        "latent_std": np.maximum(latent_std, 1.0e-6).astype(np.float32),
        "rank": np.array([rank], dtype=np.int32),
    }


def fit_latent_mixture(latent: np.ndarray, *, n_clusters: int, seed: int) -> dict[str, np.ndarray]:
    _, labels = kmeans2(latent, n_clusters, minit="points", seed=seed)
    weights = []
    means = []
    covs = []
    for cluster in range(n_clusters):
        subset = latent[labels == cluster]
        if subset.shape[0] < 2:
            continue
        weights.append(subset.shape[0])
        means.append(np.mean(subset, axis=0, dtype=np.float64))
        cov = np.cov(subset, rowvar=False).astype(np.float64)
        if subset.shape[1] == 1:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov += np.eye(subset.shape[1], dtype=np.float64) * 1.0e-6
        covs.append(cov)
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= np.sum(weights_arr)
    return {
        "weights": weights_arr.astype(np.float32),
        "means": np.stack(means, axis=0).astype(np.float32),
        "covs": np.stack(covs, axis=0).astype(np.float32),
    }


def fit_seed_noise_bank(
    coords_fit: np.ndarray,
    disp_fit: np.ndarray,
    branch_fit: np.ndarray,
    material_fit: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a real-state seed bank for local-noise synthetic generation."""
    canonical = canonicalize_p2_element_states(coords_fit, disp_fit)
    branch_counts = np.bincount(branch_fit.reshape(-1), minlength=5).astype(np.float64)
    branch_inv = 1.0 / np.maximum(branch_counts, 1.0)
    seed_weights = np.mean(branch_inv[branch_fit], axis=1)
    seed_weights = seed_weights / np.sum(seed_weights)
    smooth_frac = np.mean(branch_fit == 1, axis=1).astype(np.float64)
    right_frac = np.mean(branch_fit == 3, axis=1).astype(np.float64)
    smooth_flag = np.any(branch_fit == 1, axis=1).astype(np.float64)
    right_flag = np.any(branch_fit == 3, axis=1).astype(np.float64)
    transition_flag = (smooth_flag * right_flag).astype(np.float64)
    disp_energy = np.sqrt(np.mean(canonical.local_displacements.reshape(canonical.local_displacements.shape[0], -1) ** 2, axis=1))
    disp_energy = disp_energy / max(float(np.mean(disp_energy)), 1.0e-8)
    smooth_weights = 1.0 + 2.0 * smooth_flag + 4.0 * smooth_frac + 0.75 * disp_energy
    smooth_weights = smooth_weights / np.sum(smooth_weights)
    smooth_edge_weights = (
        1.0
        + 4.0 * transition_flag
        + 3.0 * np.sqrt(np.maximum(smooth_frac * right_frac, 0.0))
        + 1.5 * smooth_frac
        + 1.0 * right_frac
        + 0.75 * disp_energy
    )
    smooth_edge_weights = smooth_edge_weights / np.sum(smooth_edge_weights)
    disp_std = canonical.local_displacements.reshape(canonical.local_displacements.shape[0], -1).std(axis=0)
    disp_std = np.maximum(disp_std.reshape(10, 3), 1.0e-5).astype(np.float32)
    return {
        "coords_fit": coords_fit.astype(np.float32),
        "material_fit": material_fit.astype(np.float32),
        "canonical_disp": canonical.local_displacements.astype(np.float32),
        "basis": canonical.basis.astype(np.float32),
        "characteristic_length": canonical.characteristic_length.astype(np.float32),
        "seed_weights_uniform": np.full(coords_fit.shape[0], 1.0 / coords_fit.shape[0], dtype=np.float32),
        "seed_weights_branch_balanced": seed_weights.astype(np.float32),
        "seed_weights_smooth_focus": smooth_weights.astype(np.float32),
        "seed_weights_smooth_edge": smooth_edge_weights.astype(np.float32),
        "disp_std": disp_std,
        "disp_abs_mean": np.mean(np.abs(canonical.local_displacements), axis=0).astype(np.float32),
    }


def draw_latent(
    mode: str,
    *,
    pca: dict[str, np.ndarray],
    mixture: dict[str, np.ndarray],
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rank = int(pca["rank"][0])
    if mode == "pca_gaussian":
        return rng.multivariate_normal(pca["latent_mean"], pca["latent_cov"], size=sample_count).astype(np.float32)
    if mode == "pca_cluster_mixture":
        cluster = rng.choice(mixture["weights"].shape[0], size=sample_count, p=mixture["weights"])
        latent = np.empty((sample_count, rank), dtype=np.float32)
        for k in range(mixture["weights"].shape[0]):
            mask = cluster == k
            if not np.any(mask):
                continue
            latent[mask] = rng.multivariate_normal(
                mixture["means"][k],
                mixture["covs"][k],
                size=int(np.sum(mask)),
            ).astype(np.float32)
        return latent
    if mode == "empirical_local_noise":
        idx = rng.choice(pca["latent"].shape[0], size=sample_count, replace=True)
        noise = rng.normal(size=(sample_count, rank)).astype(np.float32) * pca["latent_std"][None, :] * noise_scale
        return pca["latent"][idx] + noise
    raise ValueError(f"Unknown mode {mode!r}.")


def _evaluate_element_states(
    coords_valid: np.ndarray,
    disp_valid: np.ndarray,
    material_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    b_blocks = build_local_b_blocks_from_coords(coords_valid)
    strain = np.einsum("nqij,nj->nqi", b_blocks, disp_valid.reshape(coords_valid.shape[0], 30)).astype(np.float32)
    exact = constitutive_update_3d(
        strain.reshape(-1, 6),
        c_bar=np.repeat(material_valid[:, 0], 11),
        sin_phi=np.repeat(material_valid[:, 1], 11),
        shear=np.repeat(material_valid[:, 2], 11),
        bulk=np.repeat(material_valid[:, 3], 11),
        lame=np.repeat(material_valid[:, 4], 11),
    )
    branch = exact.branch_id.reshape(coords_valid.shape[0], 11).astype(np.int8)
    return strain, branch


def synthesize_element_states_from_seeded_noise(
    seed_bank: dict[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str = "branch_balanced",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic element states by perturbing real seed elements locally."""
    rng = np.random.default_rng(seed)
    if selection == "branch_balanced":
        weights = seed_bank["seed_weights_branch_balanced"]
    elif selection == "smooth_focus":
        weights = seed_bank["seed_weights_smooth_focus"]
    elif selection == "smooth_edge":
        weights = seed_bank["seed_weights_smooth_edge"]
    elif selection == "uniform":
        weights = seed_bank["seed_weights_uniform"]
    else:
        raise ValueError(f"Unknown seed selection {selection!r}.")

    idx = rng.choice(seed_bank["coords_fit"].shape[0], size=sample_count, replace=True, p=weights)
    coords_pick = seed_bank["coords_fit"][idx]
    material_pick = seed_bank["material_fit"][idx]
    basis = seed_bank["basis"][idx]
    h = seed_bank["characteristic_length"][idx]
    canonical_disp = seed_bank["canonical_disp"][idx]
    local_scale = np.maximum(
        0.10 * seed_bank["disp_std"][None, :, :],
        0.20 * np.abs(canonical_disp),
    ).astype(np.float32)
    noise = rng.normal(size=canonical_disp.shape).astype(np.float32) * local_scale * float(noise_scale)
    disp_std = canonical_disp + noise
    disp_phys = np.einsum(
        "nij,njk->nik",
        disp_std * h[:, None, None],
        np.transpose(basis, (0, 2, 1)),
    ).astype(np.float32)

    valid = positive_corner_volume_mask(coords_pick, disp_phys.reshape(sample_count, 30), min_volume_ratio=0.01)
    coords_valid = coords_pick[valid]
    disp_valid = disp_phys[valid]
    material_valid = material_pick[valid]
    strain, branch = _evaluate_element_states(coords_valid, disp_valid, material_valid)
    return coords_valid, disp_valid, strain, branch, material_valid, valid


def synthesize_from_seeded_noise(
    seed_bank: dict[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
    noise_scale: float,
    selection: str = "branch_balanced",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic strains/branches/materials from locally perturbed seed elements."""
    coords_valid, disp_valid, strain, branch, material_valid, valid = synthesize_element_states_from_seeded_noise(
        seed_bank,
        sample_count=sample_count,
        seed=seed,
        noise_scale=noise_scale,
        selection=selection,
    )
    return strain, branch, material_valid, valid


def synthesize_element_states_from_latent(
    latent: np.ndarray,
    *,
    coords_fit: np.ndarray,
    material_fit: np.ndarray,
    pca: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_count = latent.shape[0]
    idx = rng.choice(coords_fit.shape[0], size=sample_count, replace=True)
    coords_pick = coords_fit[idx]
    material_pick = material_fit[idx]

    canonical = canonicalize_p2_element_states(coords_pick, np.zeros_like(coords_pick, dtype=np.float32))
    disp_flat = pca["mean"][None, :] + latent @ pca["basis"].T
    disp_std = disp_flat.reshape(sample_count, 10, 3)
    disp_phys = np.einsum(
        "nij,nkj->nik",
        disp_std * canonical.characteristic_length[:, None, None],
        np.transpose(canonical.basis, (0, 2, 1)),
    ).astype(np.float32)

    valid = positive_corner_volume_mask(coords_pick, disp_phys.reshape(sample_count, 30), min_volume_ratio=0.01)
    coords_valid = coords_pick[valid]
    disp_valid = disp_phys[valid]
    material_valid = material_pick[valid]
    strain, branch = _evaluate_element_states(coords_valid, disp_valid, material_valid)
    return coords_valid, disp_valid, strain, branch, material_valid, valid


def synthesize_from_latent(
    latent: np.ndarray,
    *,
    coords_fit: np.ndarray,
    material_fit: np.ndarray,
    pca: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_valid, disp_valid, strain, branch, material_valid, valid = synthesize_element_states_from_latent(
        latent,
        coords_fit=coords_fit,
        material_fit=material_fit,
        pca=pca,
        seed=seed,
    )
    return strain, branch, material_valid, valid
