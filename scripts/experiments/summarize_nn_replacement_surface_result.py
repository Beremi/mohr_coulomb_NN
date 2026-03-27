#!/usr/bin/env python
"""Create a human-readable summary of the current best surface surrogate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.mohr_coulomb import BRANCH_NAMES, principal_relative_error_3d, yield_violation_rel_principal_3d
from mc_surrogate.training import predict_with_checkpoint


ABR_REFERENCE_TEST = {
    "broad_plastic_mae": 44.258335,
    "hard_plastic_mae": 48.720104,
    "hard_p95_principal": 391.887390,
    "yield_violation_p95": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-json",
        default="experiment_runs/real_sim/nn_replacement_surface_20260324/exp1_branchless_surface/checkpoint_selection.json",
    )
    parser.add_argument(
        "--dataset",
        default="experiment_runs/real_sim/nn_replacement_surface_20260324/exp0_surface_dataset/derived_surface_dataset.h5",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_runs/real_sim/nn_replacement_surface_20260324/exp1_branchless_surface/interpretation",
    )
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--report-path", default="docs/nn_replacement_surface_readable_summary_20260324.md")
    parser.add_argument("--report-title", default="NN Replacement Surface Readable Summary 20260324")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260324)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_report(path: Path, title: str, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")
    return path


def _load_test_arrays(dataset_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(dataset_path, "r") as f:
        mask = f["split_id"][:] == 2
        return {
            key: f[key][:][mask]
            for key in (
                "strain_eng",
                "material_reduced",
                "stress",
                "stress_principal",
                "branch_id",
                "plastic_mask",
                "hard_mask",
                "grho",
            )
        }


def _quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {f"q{int(100 * q):02d}": float(np.quantile(arr, q)) for q in qs}


def _sample_indices(size: int, sample_size: int, seed: int) -> np.ndarray:
    if size <= sample_size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=sample_size, replace=False).astype(np.int64))


def _plot_principal_parity(
    output_path: Path,
    true_principal: np.ndarray,
    pred_principal: np.ndarray,
    sample_idx: np.ndarray,
) -> Path:
    labels = ("sigma1", "sigma2", "sigma3")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for i, ax in enumerate(axes):
        x = true_principal[sample_idx, i]
        y = pred_principal[sample_idx, i]
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.hexbin(x, y, gridsize=55, bins="log", cmap="viridis", mincnt=1)
        ax.plot([lo, hi], [lo, hi], color="#d1495b", lw=1.5)
        ax.set_title(labels[i])
        ax.set_xlabel("Exact MC")
        ax.set_ylabel("Surrogate")
    fig.suptitle("Held-Out Plastic Test: Principal Stress Parity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_surface_parity(
    output_path: Path,
    true_grho: np.ndarray,
    pred_grho: np.ndarray,
    sample_idx: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    configs = (
        ("g", true_grho[sample_idx, 0], pred_grho[sample_idx, 0]),
        ("rho", true_grho[sample_idx, 1], pred_grho[sample_idx, 1]),
    )
    for ax, (label, x, y) in zip(axes, configs, strict=False):
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.hexbin(x, y, gridsize=55, bins="log", cmap="magma", mincnt=1)
        ax.plot([lo, hi], [lo, hi], color="#edae49", lw=1.5)
        ax.set_title(label)
        ax.set_xlabel(f"Exact {label}")
        ax.set_ylabel(f"Predicted {label}")
    fig.suptitle("Held-Out Plastic Test: Surface Coordinate Parity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_error_cdfs(
    output_path: Path,
    principal_abs_max: np.ndarray,
    repo_relative: np.ndarray,
    plastic: np.ndarray,
    hard_plastic: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    curves = (
        ("All plastic", plastic, "#2a9d8f"),
        ("Hard plastic", hard_plastic, "#e76f51"),
    )
    for ax, values, title in (
        (axes[0], principal_abs_max, "Max Principal Abs Error"),
        (axes[1], repo_relative, "Repo Relative Principal Error"),
    ):
        for label, mask, color in curves:
            arr = np.sort(np.asarray(values[mask], dtype=float))
            y = np.linspace(0.0, 1.0, arr.size, endpoint=True)
            ax.plot(arr, y, label=label, color=color, lw=2.0)
        ax.set_title(title)
        ax.set_xlabel("Error")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Held-Out Real Test: Error Distributions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_branch_summary(
    output_path: Path,
    branch_rows: list[dict[str, Any]],
) -> Path:
    names = [row["branch"] for row in branch_rows]
    counts = np.asarray([row["count"] for row in branch_rows], dtype=float)
    maes = np.asarray([row["stress_component_mae"] for row in branch_rows], dtype=float)
    p95s = np.asarray([row["principal_abs_p95"] for row in branch_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(names, counts, color="#457b9d")
    axes[0].set_title("Rows by Branch")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, maes, color="#2a9d8f")
    axes[1].set_title("Stress Component MAE")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(names, p95s, color="#e76f51")
    axes[2].set_title("Max Principal Error p95")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Held-Out Real Test: Coverage and Error by Exact Branch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    selection_path = (ROOT / args.selection_json).resolve()
    dataset_path = (ROOT / args.dataset).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    docs_root = (ROOT / args.docs_root).resolve()
    docs_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner = selection["winner"]
    checkpoint_path = Path(winner["selected_checkpoint_path"]).resolve()

    arrays = _load_test_arrays(dataset_path)
    pred = predict_with_checkpoint(
        checkpoint_path,
        arrays["strain_eng"],
        arrays["material_reduced"],
        device=args.device,
        batch_size=args.batch_size,
    )

    branch_id = arrays["branch_id"].astype(np.int64)
    elastic = branch_id == 0
    plastic = arrays["plastic_mask"].astype(bool)
    hard = arrays["hard_mask"].astype(bool)
    hard_plastic = hard & plastic

    stress_abs = np.abs(pred["stress"] - arrays["stress"])
    principal_abs = np.abs(pred["stress_principal"] - arrays["stress_principal"])
    principal_abs_max = np.max(principal_abs, axis=1)
    repo_relative = principal_relative_error_3d(
        pred["stress_principal"],
        arrays["stress_principal"],
        c_bar=arrays["material_reduced"][:, 0],
    )
    yield_relative = yield_violation_rel_principal_3d(
        pred["stress_principal"],
        c_bar=arrays["material_reduced"][:, 0],
        sin_phi=arrays["material_reduced"][:, 1],
    )
    simple_relative = principal_abs_max / np.maximum(np.max(np.abs(arrays["stress_principal"]), axis=1), 1.0e-12)

    broad_plastic_mae = float(np.mean(stress_abs[plastic]))
    hard_plastic_mae = float(np.mean(stress_abs[hard_plastic]))
    mean_abs_true_plastic = float(np.mean(np.abs(arrays["stress"][plastic])))
    mean_abs_true_hard = float(np.mean(np.abs(arrays["stress"][hard_plastic])))

    branch_rows: list[dict[str, Any]] = []
    for branch_value, branch_name in enumerate(BRANCH_NAMES):
        mask = branch_id == branch_value
        if not np.any(mask):
            continue
        branch_rows.append(
            {
                "branch": branch_name,
                "count": int(np.sum(mask)),
                "fraction_of_test": float(np.mean(mask)),
                "stress_component_mae": float(np.mean(stress_abs[mask])),
                "principal_abs_p95": float(np.quantile(principal_abs_max[mask], 0.95)),
                "repo_relative_p95": float(np.quantile(repo_relative[mask], 0.95)),
            }
        )

    readable = {
        "winner_checkpoint": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "row_composition": {
            "total_test_rows": int(arrays["stress"].shape[0]),
            "elastic_rows": int(np.sum(elastic)),
            "elastic_fraction": float(np.mean(elastic)),
            "plastic_rows": int(np.sum(plastic)),
            "plastic_fraction": float(np.mean(plastic)),
            "hard_rows": int(np.sum(hard)),
            "hard_fraction": float(np.mean(hard)),
            "hard_plastic_rows": int(np.sum(hard_plastic)),
            "hard_plastic_fraction": float(np.mean(hard_plastic)),
            "plastic_coverage_fraction": 1.0,
            "fallback_fraction": 0.0,
        },
        "plain_metrics": {
            "broad_plastic_component_mae": broad_plastic_mae,
            "hard_plastic_component_mae": hard_plastic_mae,
            "broad_plastic_mae_as_fraction_of_mean_abs_component": broad_plastic_mae / mean_abs_true_plastic,
            "hard_plastic_mae_as_fraction_of_mean_abs_component": hard_plastic_mae / mean_abs_true_hard,
            "hard_p95_max_principal_abs_error": float(np.quantile(principal_abs_max[hard_plastic], 0.95)),
            "yield_violation_p95": float(np.quantile(yield_relative[plastic], 0.95)),
            "repo_relative_error_plastic": _quantiles(repo_relative[plastic]),
            "repo_relative_error_hard_plastic": _quantiles(repo_relative[hard_plastic]),
            "simple_max_principal_relative_error_plastic": _quantiles(simple_relative[plastic]),
            "simple_max_principal_relative_error_hard_plastic": _quantiles(simple_relative[hard_plastic]),
            "g_mae": float(np.mean(np.abs(pred["grho"][plastic, 0] - arrays["grho"][plastic, 0]))),
            "rho_mae": float(np.mean(np.abs(pred["grho"][plastic, 1] - arrays["grho"][plastic, 1]))),
        },
        "improvement_vs_first_abr_test": {
            key: float(ABR_REFERENCE_TEST[key] - winner_value)
            for key, winner_value in (
                ("broad_plastic_mae", float(np.mean(stress_abs[plastic]))),
                ("hard_plastic_mae", float(np.mean(stress_abs[hard_plastic]))),
                ("hard_p95_principal", float(np.quantile(principal_abs_max[hard_plastic], 0.95))),
            )
        },
        "branch_breakdown": branch_rows,
    }

    summary_json_path = _write_json(output_dir / "readable_test_summary.json", readable)
    branch_csv_path = _write_csv(
        output_dir / "branch_breakdown_test.csv",
        branch_rows,
        [
            "branch",
            "count",
            "fraction_of_test",
            "stress_component_mae",
            "principal_abs_p95",
            "repo_relative_p95",
        ],
    )

    plastic_sample = _sample_indices(int(np.sum(plastic)), args.sample_size, args.seed)

    figures = {
        "principal_parity": _plot_principal_parity(
            output_dir / "test_principal_parity.png",
            arrays["stress_principal"][plastic],
            pred["stress_principal"][plastic],
            plastic_sample,
        ),
        "surface_parity": _plot_surface_parity(
            output_dir / "test_surface_parity.png",
            arrays["grho"][plastic],
            pred["grho"][plastic],
            plastic_sample,
        ),
        "error_cdfs": _plot_error_cdfs(
            output_dir / "test_error_cdfs.png",
            principal_abs_max,
            repo_relative,
            plastic,
            hard_plastic,
        ),
        "branch_summary": _plot_branch_summary(
            output_dir / "test_branch_summary.png",
            branch_rows,
        ),
    }

    report_lines = [
        "## Plain-Language Verdict",
        "",
        "- Elastic rows are exact by construction: the surrogate returns the exact trial elastic stress whenever `f_trial <= 0`.",
        f"- On the held-out real test split, `{readable['row_composition']['elastic_rows']}` / `{readable['row_composition']['total_test_rows']}` rows are elastic (`{100.0 * readable['row_composition']['elastic_fraction']:.1f}%`) and `{readable['row_composition']['plastic_rows']}` / `{readable['row_composition']['total_test_rows']}` are learned plastic (`{100.0 * readable['row_composition']['plastic_fraction']:.1f}%`).",
        "- Plastic coverage is 100%: there is no learned rejector and no fallback path inside this packet.",
        f"- Average plastic error is still large in relative terms: broad-plastic component MAE is `{100.0 * readable['plain_metrics']['broad_plastic_mae_as_fraction_of_mean_abs_component']:.1f}%` of the mean absolute exact stress-component magnitude on plastic rows; the hard-plastic value is `{100.0 * readable['plain_metrics']['hard_plastic_mae_as_fraction_of_mean_abs_component']:.1f}%`.",
        f"- The tail is still the problem: hard-plastic max principal abs error p95 is `{readable['plain_metrics']['hard_p95_max_principal_abs_error']:.3f}` even though admissibility remains essentially exact.",
        "",
        "## Test Metrics",
        "",
        f"- broad plastic MAE: `{readable['plain_metrics']['broad_plastic_component_mae']:.6f}`",
        f"- hard plastic MAE: `{readable['plain_metrics']['hard_plastic_component_mae']:.6f}`",
        f"- hard p95 max principal abs error: `{readable['plain_metrics']['hard_p95_max_principal_abs_error']:.6f}`",
        f"- yield violation p95: `{readable['plain_metrics']['yield_violation_p95']:.6e}`",
        f"- repo relative principal error q50 / q90 / q95 on plastic rows: `{readable['plain_metrics']['repo_relative_error_plastic']['q50']:.3f}` / `{readable['plain_metrics']['repo_relative_error_plastic']['q90']:.3f}` / `{readable['plain_metrics']['repo_relative_error_plastic']['q95']:.3f}`",
        f"- repo relative principal error q50 / q90 / q95 on hard plastic rows: `{readable['plain_metrics']['repo_relative_error_hard_plastic']['q50']:.3f}` / `{readable['plain_metrics']['repo_relative_error_hard_plastic']['q90']:.3f}` / `{readable['plain_metrics']['repo_relative_error_hard_plastic']['q95']:.3f}`",
        f"- learned geometry quality on plastic rows: `g` MAE `{readable['plain_metrics']['g_mae']:.6f}`, `rho` MAE `{readable['plain_metrics']['rho_mae']:.6f}`",
        "",
        "## Where It Works And Fails",
        "",
        "- The smooth branch is much better than the edges and apex.",
        "- Left and right edge rows carry the largest stress MAE and principal-error tail.",
        "- Apex rows have lower absolute MAE than the edges, but still large normalized principal error because the true stress scale is smaller there.",
        "",
        "## Improvement Vs The First ABR Packet",
        "",
        f"- broad plastic MAE improvement on held-out real test: `{readable['improvement_vs_first_abr_test']['broad_plastic_mae']:.6f}`",
        f"- hard plastic MAE improvement on held-out real test: `{readable['improvement_vs_first_abr_test']['hard_plastic_mae']:.6f}`",
        f"- hard p95 principal improvement on held-out real test: `{readable['improvement_vs_first_abr_test']['hard_p95_principal']:.6f}`",
        "",
        "## Generated Files",
        "",
        f"- summary JSON: `{summary_json_path}`",
        f"- branch CSV: `{branch_csv_path}`",
        f"- principal parity plot: `{figures['principal_parity']}`",
        f"- surface parity plot: `{figures['surface_parity']}`",
        f"- error CDF plot: `{figures['error_cdfs']}`",
        f"- branch summary plot: `{figures['branch_summary']}`",
    ]
    report_path = (ROOT / args.report_path).resolve()
    _write_report(report_path, args.report_title, report_lines)


if __name__ == "__main__":
    main()
