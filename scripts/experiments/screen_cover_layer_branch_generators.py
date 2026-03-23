from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mc_surrogate.cover_branch_generation import (
    collect_blocks as _collect_blocks,
    draw_latent as _draw_latent,
    fit_latent_mixture as _fit_latent_mixture,
    fit_pca as _fit_pca,
    fit_seed_noise_bank as _fit_seed_noise_bank,
    load_call_regimes as _load_call_regimes,
    load_split_calls as _load_split_calls,
    pick_calls as _pick_calls,
    synthesize_from_seeded_noise as _synthesize_from_seeded_noise,
    synthesize_from_latent as _synthesize_from_latent,
)
from mc_surrogate.full_export import canonicalize_p2_element_states
from mc_surrogate.mohr_coulomb import BRANCH_NAMES


def _branch_hist(branch: np.ndarray) -> np.ndarray:
    return np.bincount(branch.reshape(-1), minlength=len(BRANCH_NAMES)).astype(np.float64)


def _pattern_hist(branch: np.ndarray) -> Counter[str]:
    def key(row: np.ndarray) -> str:
        return "|".join(BRANCH_NAMES[int(x)] for x in row.tolist())

    return Counter(key(row) for row in branch)


def _evaluate_mode(
    mode: str,
    *,
    coords_fit: np.ndarray,
    material_fit: np.ndarray,
    pca: dict[str, np.ndarray],
    mixture: dict[str, np.ndarray],
    seed_bank: dict[str, np.ndarray],
    strain_real: np.ndarray,
    branch_real: np.ndarray,
    sample_count: int,
    seed: int,
    noise_scale: float,
) -> dict[str, object]:
    if mode in {"seeded_local_noise_uniform", "seeded_local_noise_branch_balanced"}:
        selection = "uniform" if mode.endswith("uniform") else "branch_balanced"
        strain_syn, branch_syn, _, valid_mask = _synthesize_from_seeded_noise(
            seed_bank,
            sample_count=sample_count,
            seed=seed + 17,
            noise_scale=noise_scale,
            selection=selection,
        )
    else:
        latent = _draw_latent(mode, pca=pca, mixture=mixture, sample_count=sample_count, seed=seed, noise_scale=noise_scale)
        strain_syn, branch_syn, _, valid_mask = _synthesize_from_latent(
            latent,
            coords_fit=coords_fit,
            material_fit=material_fit,
            pca=pca,
            seed=seed + 17,
        )
    real_branch_hist = _branch_hist(branch_real)
    syn_branch_hist = _branch_hist(branch_syn)
    real_branch_prob = real_branch_hist / np.maximum(np.sum(real_branch_hist), 1.0)
    syn_branch_prob = syn_branch_hist / np.maximum(np.sum(syn_branch_hist), 1.0)
    branch_tv = 0.5 * float(np.sum(np.abs(real_branch_prob - syn_branch_prob)))

    real_norm = np.linalg.norm(strain_real.reshape(strain_real.shape[0], -1), axis=1)
    syn_norm = np.linalg.norm(strain_syn.reshape(strain_syn.shape[0], -1), axis=1)
    mean_rel_err = float(abs(np.mean(syn_norm) - np.mean(real_norm)) / max(np.mean(real_norm), 1.0e-8))
    p95_rel_err = float(abs(np.quantile(syn_norm, 0.95) - np.quantile(real_norm, 0.95)) / max(np.quantile(real_norm, 0.95), 1.0e-8))

    real_patterns = _pattern_hist(branch_real)
    syn_patterns = _pattern_hist(branch_syn)
    top_real = real_patterns.most_common(5)
    top_syn = syn_patterns.most_common(5)
    top_overlap = sum(1 for pat, _ in top_real if pat in dict(top_syn))

    score = branch_tv + mean_rel_err + p95_rel_err - 0.05 * top_overlap
    return {
        "mode": mode,
        "synthetic_kept_elements": int(strain_syn.shape[0]),
        "valid_ratio_before_filter": float(np.mean(valid_mask)),
        "branch_hist_real": {BRANCH_NAMES[i]: int(real_branch_hist[i]) for i in range(len(BRANCH_NAMES))},
        "branch_hist_synthetic": {BRANCH_NAMES[i]: int(syn_branch_hist[i]) for i in range(len(BRANCH_NAMES))},
        "branch_total_variation": branch_tv,
        "strain_norm_mean_real": float(np.mean(real_norm)),
        "strain_norm_mean_synthetic": float(np.mean(syn_norm)),
        "strain_norm_p95_real": float(np.quantile(real_norm, 0.95)),
        "strain_norm_p95_synthetic": float(np.quantile(syn_norm, 0.95)),
        "strain_norm_mean_rel_err": mean_rel_err,
        "strain_norm_p95_rel_err": p95_rel_err,
        "top_pattern_overlap_5": int(top_overlap),
        "score": float(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen synthetic generator modes for cover-layer branch-predictor training.")
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_generator_screen_20260314"),
    )
    parser.add_argument(
        "--regime-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_regimes.json"),
    )
    parser.add_argument("--fit-calls", type=int, default=6)
    parser.add_argument("--val-calls", type=int, default=2)
    parser.add_argument("--call-selection", choices=("first", "spread_p95"), default="spread_p95")
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--explained-variance", type=float, default=0.995)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_split_calls(args.split_json)
    regimes = _load_call_regimes(args.regime_json) if args.regime_json.exists() else None
    fit_calls = _pick_calls(
        splits["generator_fit"],
        count=args.fit_calls,
        selection=args.call_selection,
        regimes=regimes,
    )
    val_calls = _pick_calls(
        splits["real_val"],
        count=args.val_calls,
        selection=args.call_selection,
        regimes=regimes,
    )

    coords_fit, disp_fit, _, branch_fit, material_fit = _collect_blocks(
        args.export,
        call_names=fit_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed,
    )
    _, _, strain_val, branch_val, _ = _collect_blocks(
        args.export,
        call_names=val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 1,
    )

    canonical_fit = canonicalize_p2_element_states(coords_fit, disp_fit)
    disp_fit_flat = canonical_fit.local_displacements.reshape(canonical_fit.local_displacements.shape[0], -1)
    pca = _fit_pca(disp_fit_flat, explained_variance=args.explained_variance, max_rank=args.max_rank)
    mixture = _fit_latent_mixture(pca["latent"], n_clusters=min(args.clusters, pca["latent"].shape[0]), seed=args.seed)
    seed_bank = _fit_seed_noise_bank(coords_fit, disp_fit, branch_fit=branch_fit, material_fit=material_fit)

    results = []
    for mode in (
        "pca_gaussian",
        "pca_cluster_mixture",
        "empirical_local_noise",
        "seeded_local_noise_uniform",
        "seeded_local_noise_branch_balanced",
    ):
        results.append(
            _evaluate_mode(
                mode,
                coords_fit=coords_fit,
                material_fit=material_fit,
                pca=pca,
                mixture=mixture,
                seed_bank=seed_bank,
                strain_real=strain_val,
                branch_real=branch_val,
                sample_count=args.sample_count,
                seed=args.seed,
                noise_scale=args.noise_scale,
            )
        )

    best = min(results, key=lambda x: x["score"])
    payload = {
        "export_path": str(args.export),
        "split_json": str(args.split_json),
        "regime_json": str(args.regime_json),
        "fit_calls": fit_calls,
        "real_val_calls": val_calls,
        "fit_elements": int(coords_fit.shape[0]),
        "real_val_elements": int(strain_val.shape[0]),
        "latent_rank": int(pca["rank"][0]),
        "generator_results": results,
        "best_mode": best["mode"],
    }
    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
