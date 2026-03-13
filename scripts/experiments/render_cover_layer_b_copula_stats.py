#!/usr/bin/env python
"""Render detailed statistics for the cover-layer B-copula fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for path in (str(SRC), str(THIS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mc_surrogate.fe_p2_tetra import build_local_b_pool

import cover_layer_b_copula_refresh as bmod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-primary",
        default="experiment_runs/real_sim/per_material_synth_to_real_20260312/real_material_datasets/cover_layer_primary.h5",
    )
    parser.add_argument(
        "--mesh-path",
        default="/tmp/slope_stability_octave/slope_stability/meshes/SSR_hetero_ada_L1.h5",
    )
    parser.add_argument("--material-id", type=int, default=0)
    parser.add_argument("--focus-mode", default="branch_raw", choices=list(bmod.COPULA_MODES))
    parser.add_argument("--compare-modes", nargs="+", default=list(bmod.COPULA_MODES), choices=list(bmod.COPULA_MODES))
    parser.add_argument("--sample-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--pca-variance", type=float, default=0.999)
    parser.add_argument("--right-inverse-damping", type=float, default=1.0e-10)
    parser.add_argument("--min-volume-ratio", type=float, default=0.05)
    parser.add_argument("--principal-cap-quantile", type=float, default=0.995)
    parser.add_argument("--stress-cap-quantile", type=float, default=0.995)
    parser.add_argument("--cap-slack", type=float, default=1.10)
    parser.add_argument("--output-root", default="experiment_runs/real_sim/cover_layer_b_copula_stats_20260312")
    parser.add_argument("--report-md", default="docs/cover_layer_b_copula_stats.md")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _matrix_to_text(matrix: np.ndarray, precision: int = 6) -> str:
    return np.array2string(np.asarray(matrix, dtype=np.float64), precision=precision, suppress_small=False, max_line_width=200)


def _save_matrix_json(path: Path, matrix: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(np.asarray(matrix, dtype=np.float64).tolist(), indent=2), encoding="utf-8")
    return path


def _save_vector_json(path: Path, vector: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(np.asarray(vector, dtype=np.float64).tolist(), indent=2), encoding="utf-8")
    return path


def _plot_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> Path:
    mat = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _covariance(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    return np.cov(arr, rowvar=False)


def _mean(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).mean(axis=0)


def _rel_repo(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    report_md = ROOT / args.report_md

    geometry_pool = build_local_b_pool(
        args.mesh_path,
        material_id=args.material_id,
        right_inverse_damping=args.right_inverse_damping,
    )

    fit_results: dict[str, tuple[bmod.BCopulaFit, Path]] = {}
    for mode in args.compare_modes:
        fit_results[mode] = bmod._fit_b_copula_mode(
            real_primary=ROOT / args.real_primary,
            geometry_pool=geometry_pool,
            mode=mode,
            pca_variance=args.pca_variance,
            seed=args.seed + {"raw": 0, "pca": 17, "branch_raw": 31, "branch_pca": 47, "branch_raw_blend": 59}[mode],
            principal_cap_quantile=args.principal_cap_quantile,
            stress_cap_quantile=args.stress_cap_quantile,
            cap_slack=args.cap_slack,
            min_volume_ratio=args.min_volume_ratio,
            synthetic_test_size=args.sample_size,
            output_root=output_root / "fits",
        )

    focus_fit, focus_synth_path = fit_results[args.focus_mode]
    train_arrays, test_arrays, _ = bmod._load_real_train_test(ROOT / args.real_primary)
    _, real_u_norm = bmod._pair_real_strain_with_geometry(train_arrays["strain_eng"], geometry_pool, seed=args.seed)
    generated_arrays, generated_aux = bmod.generate_from_b_copula(
        fit=focus_fit,
        n_samples=args.sample_size,
        seed=args.seed + 1000,
        return_aux=True,
    )

    real_train_u_mean = _mean(real_u_norm)
    real_train_u_cov = _covariance(real_u_norm)
    generated_u_mean = _mean(generated_aux["u_norm"])
    generated_u_cov = _covariance(generated_aux["u_norm"])

    real_train_e_mean = _mean(train_arrays["strain_eng"])
    real_train_e_cov = _covariance(train_arrays["strain_eng"])
    real_test_e_mean = _mean(test_arrays["strain_eng"])
    real_test_e_cov = _covariance(test_arrays["strain_eng"])
    generated_e_mean = _mean(generated_arrays["strain_eng"])
    generated_e_cov = _covariance(generated_arrays["strain_eng"])

    stats_dir = output_root / args.focus_mode
    stats_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}
    for name, vector in (
        ("real_train_u_mean.json", real_train_u_mean),
        ("generated_u_mean.json", generated_u_mean),
        ("real_train_e_mean.json", real_train_e_mean),
        ("real_test_e_mean.json", real_test_e_mean),
        ("generated_e_mean.json", generated_e_mean),
    ):
        artifact_paths[name] = _save_vector_json(stats_dir / name, vector)

    for name, matrix in (
        ("real_train_u_cov.json", real_train_u_cov),
        ("generated_u_cov.json", generated_u_cov),
        ("real_train_e_cov.json", real_train_e_cov),
        ("real_test_e_cov.json", real_test_e_cov),
        ("generated_e_cov.json", generated_e_cov),
    ):
        artifact_paths[name] = _save_matrix_json(stats_dir / name, matrix)

    for name, matrix, title in (
        ("real_train_u_cov.png", real_train_u_cov, "Real train pseudo-U covariance"),
        ("generated_u_cov.png", generated_u_cov, f"Generated pseudo-U covariance ({args.focus_mode})"),
        ("real_train_e_cov.png", real_train_e_cov, "Real train E covariance"),
        ("real_test_e_cov.png", real_test_e_cov, "Real test E covariance"),
        ("generated_e_cov.png", generated_e_cov, f"Generated E covariance ({args.focus_mode})"),
    ):
        artifact_paths[name] = _plot_heatmap(matrix, stats_dir / name, title)

    latent_stats: dict[str, Any] = {}
    if -1 in focus_fit.copula_models:
        z, _ = bmod._rank_gaussianize(real_u_norm.astype(np.float64))
        latent_cov = _covariance(z)
        latent_mean = _mean(z)
        label = "global"
        latent_stats[label] = {
            "mean": latent_mean,
            "cov": latent_cov,
            "cov_path": _save_matrix_json(stats_dir / f"latent_cov_{label}.json", latent_cov),
            "mean_path": _save_vector_json(stats_dir / f"latent_mean_{label}.json", latent_mean),
            "heatmap_path": _plot_heatmap(latent_cov, stats_dir / f"latent_cov_{label}.png", f"Latent covariance ({label})"),
        }
    else:
        for bid, name in enumerate(bmod.BRANCH_NAMES):
            mask = focus_fit.train_branch_id == bid
            z, _ = bmod._rank_gaussianize(real_u_norm[mask].astype(np.float64))
            latent_cov = _covariance(z)
            latent_mean = _mean(z)
            label = name
            latent_stats[label] = {
                "mean": latent_mean,
                "cov": latent_cov,
                "cov_path": _save_matrix_json(stats_dir / f"latent_cov_{label}.json", latent_cov),
                "mean_path": _save_vector_json(stats_dir / f"latent_mean_{label}.json", latent_mean),
                "heatmap_path": _plot_heatmap(latent_cov, stats_dir / f"latent_cov_{label}.png", f"Latent covariance ({label})"),
            }

    summary_json = {
        "focus_mode": args.focus_mode,
        "compare_modes": {
            mode: {
                "distribution_fit_score": fit.distribution_fit_score,
                "synthetic_test_path": str(path),
            }
            for mode, (fit, path) in fit_results.items()
        },
        "real_train_u_mean": real_train_u_mean,
        "real_train_u_cov": real_train_u_cov,
        "generated_u_mean": generated_u_mean,
        "generated_u_cov": generated_u_cov,
        "real_train_e_mean": real_train_e_mean,
        "real_train_e_cov": real_train_e_cov,
        "real_test_e_mean": real_test_e_mean,
        "real_test_e_cov": real_test_e_cov,
        "generated_e_mean": generated_e_mean,
        "generated_e_cov": generated_e_cov,
        "latent_stats": {
            label: {
                "mean": payload["mean"],
                "cov": payload["cov"],
            }
            for label, payload in latent_stats.items()
        },
    }
    (stats_dir / "summary.json").write_text(json.dumps(_json_safe(summary_json), indent=2), encoding="utf-8")

    lines = [
        "# Cover Layer B-Copula Detailed Statistics",
        "",
        "This report focuses on the current best-screened B-copula mode and compares its pseudo-displacement and strain statistics against the real cover-layer data.",
        "",
        "## Copula Reminder",
        "",
        "The current model is a Gaussian copula, not a multivariate normal directly in `U`-space.",
        "That means:",
        "",
        "- the fitted latent normal variable `Z` has approximately zero mean",
        "- dependence is carried by the latent covariance/correlation matrix of `Z`",
        "- the physical `U` marginals are empirical, not Gaussian",
        "",
        f"Focused mode: `{args.focus_mode}`",
        "",
        "## Screen Summary",
        "",
        "| mode | total score | principal q95 logerr | principal q995 logerr | stress q95 logerr | stress q995 logerr | branch TV | synthetic test |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode, (fit, synth_path) in fit_results.items():
        score = fit.distribution_fit_score
        lines.append(
            f"| `{mode}` | `{score['total']:.4f}` | `{score['principal_q95_logerr']:.4f}` | `{score['principal_q995_logerr']:.4f}` | "
            f"`{score['stress_q95_logerr']:.4f}` | `{score['stress_q995_logerr']:.4f}` | `{score['branch_tv']:.4f}` | "
            f"`{_rel_repo(synth_path)}` |"
        )

    lines.extend(
        [
            "",
            "## U Statistics",
            "",
            "Real train pseudo-displacements `U/h`:",
            "",
            "```text",
            _matrix_to_text(real_train_u_mean),
            "```",
            "",
            "Generated pseudo-displacements `U/h`:",
            "",
            "```text",
            _matrix_to_text(generated_u_mean),
            "```",
            "",
            f"Full real-train `U` covariance matrix: `{_rel_repo(artifact_paths['real_train_u_cov.json'])}`",
            "",
            f"Full generated `U` covariance matrix: `{_rel_repo(artifact_paths['generated_u_cov.json'])}`",
            "",
            f"![Real train U covariance](../{_rel_repo(artifact_paths['real_train_u_cov.png'])})",
            "",
            f"![Generated U covariance](../{_rel_repo(artifact_paths['generated_u_cov.png'])})",
            "",
            "## E Statistics",
            "",
            "Real train `E` mean:",
            "",
            "```text",
            _matrix_to_text(real_train_e_mean),
            "```",
            "",
            "Real test `E` mean:",
            "",
            "```text",
            _matrix_to_text(real_test_e_mean),
            "```",
            "",
            "Generated `E` mean:",
            "",
            "```text",
            _matrix_to_text(generated_e_mean),
            "```",
            "",
            "Real train `E` covariance:",
            "",
            "```text",
            _matrix_to_text(real_train_e_cov),
            "```",
            "",
            "Real test `E` covariance:",
            "",
            "```text",
            _matrix_to_text(real_test_e_cov),
            "```",
            "",
            "Generated `E` covariance:",
            "",
            "```text",
            _matrix_to_text(generated_e_cov),
            "```",
            "",
            f"![Real train E covariance](../{_rel_repo(artifact_paths['real_train_e_cov.png'])})",
            "",
            f"![Real test E covariance](../{_rel_repo(artifact_paths['real_test_e_cov.png'])})",
            "",
            f"![Generated E covariance](../{_rel_repo(artifact_paths['generated_e_cov.png'])})",
            "",
            "## Latent Gaussian Statistics",
            "",
            "These are the actual Gaussian-copula statistics, in latent normal space.",
            "",
        ]
    )

    for label, payload in latent_stats.items():
        lines.extend(
            [
                f"### {label}",
                "",
                "Latent mean:",
                "",
                "```text",
                _matrix_to_text(payload["mean"]),
                "```",
                "",
                f"Full latent covariance matrix: `{_rel_repo(payload['cov_path'])}`",
                "",
                f"![Latent covariance {label}](../{_rel_repo(payload['heatmap_path'])})",
                "",
            ]
        )

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
