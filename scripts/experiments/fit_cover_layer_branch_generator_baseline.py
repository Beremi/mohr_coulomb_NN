from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mc_surrogate.fe_p2_tetra import build_local_b_blocks_from_coords, positive_corner_volume_mask
from mc_surrogate.full_export import canonicalize_p2_element_states, iter_family_element_blocks
from mc_surrogate.mohr_coulomb import BRANCH_NAMES, constitutive_update_3d


def _load_split_calls(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in payload["splits"].items()}


def _collect_blocks(
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


def _fit_pca_gaussian(samples: np.ndarray, *, explained_variance: float, max_rank: int) -> dict[str, np.ndarray]:
    mean = np.mean(samples, axis=0, dtype=np.float64)
    centered = samples - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        raise RuntimeError("Cannot fit PCA on an empty sample matrix.")
    energy = np.cumsum(s**2) / np.sum(s**2)
    rank = int(np.searchsorted(energy, explained_variance) + 1)
    rank = max(1, min(rank, max_rank, vt.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    latent = centered @ basis
    latent_mean = np.mean(latent, axis=0, dtype=np.float64)
    latent_cov = np.cov(latent, rowvar=False).astype(np.float64)
    if rank == 1:
        latent_cov = np.array([[float(latent_cov)]], dtype=np.float64)
    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "latent_mean": latent_mean.astype(np.float32),
        "latent_cov": latent_cov.astype(np.float32),
        "rank": np.array([rank], dtype=np.int32),
    }


def _sample_synthetic(
    coords: np.ndarray,
    canonical_disp: np.ndarray,
    material: np.ndarray,
    fit: dict[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(coords.shape[0], size=sample_count, replace=True)
    coords_pick = coords[idx]
    material_pick = material[idx]

    canonical = canonicalize_p2_element_states(coords_pick, np.zeros_like(coords_pick, dtype=np.float32))
    rank = int(fit["rank"][0])
    latent = rng.multivariate_normal(fit["latent_mean"], fit["latent_cov"], size=sample_count).astype(np.float32)
    disp_flat = fit["mean"][None, :] + latent @ fit["basis"].T
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
    if coords_valid.shape[0] == 0:
        raise RuntimeError("All synthetic samples were rejected by the deformation validity filter.")

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
    return coords_valid, disp_valid, strain, branch


def _branch_hist(branch: np.ndarray) -> dict[str, int]:
    counts = np.bincount(branch.reshape(-1), minlength=len(BRANCH_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(BRANCH_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a first synthetic generator baseline for cover-layer branch prediction.")
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_generator_baseline_20260314"),
    )
    parser.add_argument("--fit-calls", type=int, default=12, help="Number of generator-fit calls to load.")
    parser.add_argument("--val-calls", type=int, default=4, help="Number of real validation calls to summarize.")
    parser.add_argument("--max-elements-per-call", type=int, default=512, help="Cap loaded elements per call for fast iteration.")
    parser.add_argument("--sample-count", type=int, default=12000, help="Number of synthetic element states to sample.")
    parser.add_argument("--explained-variance", type=float, default=0.995)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_split_calls(args.split_json)
    fit_calls = splits["generator_fit"][: args.fit_calls]
    val_calls = splits["real_val"][: args.val_calls]

    coords_fit, disp_fit, _, _, material_fit = _collect_blocks(
        args.export,
        call_names=fit_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed,
    )
    canonical_fit = canonicalize_p2_element_states(coords_fit, disp_fit)
    disp_fit_flat = canonical_fit.local_displacements.reshape(canonical_fit.local_displacements.shape[0], -1)
    fit = _fit_pca_gaussian(disp_fit_flat, explained_variance=args.explained_variance, max_rank=args.max_rank)

    coords_syn, disp_syn, strain_syn, branch_syn = _sample_synthetic(
        coords_fit,
        canonical_fit.local_displacements,
        material_fit,
        fit,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    _, _, strain_val, branch_val, _ = _collect_blocks(
        args.export,
        call_names=val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 1,
    )

    summary = {
        "export_path": str(args.export),
        "split_json": str(args.split_json),
        "fit_calls": fit_calls,
        "real_val_calls": val_calls,
        "fit_elements": int(coords_fit.shape[0]),
        "synthetic_kept_elements": int(coords_syn.shape[0]),
        "latent_rank": int(fit["rank"][0]),
        "explained_variance_target": float(args.explained_variance),
        "branch_counts_real_val": _branch_hist(branch_val),
        "branch_counts_synthetic": _branch_hist(branch_syn),
        "strain_stack_norm_real_val": {
            "mean": float(np.mean(np.linalg.norm(strain_val.reshape(strain_val.shape[0], -1), axis=1))),
            "p95": float(np.quantile(np.linalg.norm(strain_val.reshape(strain_val.shape[0], -1), axis=1), 0.95)),
        },
        "strain_stack_norm_synthetic": {
            "mean": float(np.mean(np.linalg.norm(strain_syn.reshape(strain_syn.shape[0], -1), axis=1))),
            "p95": float(np.quantile(np.linalg.norm(strain_syn.reshape(strain_syn.shape[0], -1), axis=1), 0.95)),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
