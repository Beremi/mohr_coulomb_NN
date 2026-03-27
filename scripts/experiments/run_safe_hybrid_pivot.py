#!/usr/bin/env python
"""Run Experiment 1 of the mixed-material safe hybrid pivot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.data import SPLIT_TO_ID
from mc_surrogate.inference import hybrid_predict_with_checkpoint
from mc_surrogate.mohr_coulomb import branch_harm_metrics_3d
from mc_surrogate.training import predict_with_checkpoint
from mc_surrogate.voigt import stress_voigt_to_tensor

from build_hybrid_real_panels import build_hybrid_real_panels


THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
BUCKETS = (
    ("near_yield", "near_yield_mask"),
    ("near_smooth_left", "near_smooth_left_mask"),
    ("near_smooth_right", "near_smooth_right_mask"),
    ("near_left_apex", "near_left_apex_mask"),
    ("near_right_apex", "near_right_apex_mask"),
    ("repeated_gap", "repeated_gap_mask"),
    ("tail", "tail_mask"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", default="constitutive_problem_3D_full.h5")
    parser.add_argument("--baseline-checkpoint", default="experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/hybrid_pivot_20260323")
    parser.add_argument("--report-md", default="docs/hybrid_pivot_exp1_wrapper_20260323.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-call", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--batch-size", type=int, default=32768)
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


def _load_all(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = f[key][:]
    return out


def _sample_abs_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - true), axis=1)


def _principal_sample_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.max(np.abs(pred - true), axis=1)


def _principal_from_stress(stress_voigt: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(stress_voigt_to_tensor(stress_voigt))[:, ::-1].astype(np.float32)


def _mask_metrics(
    arrays: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    if not np.any(mask):
        return {
            "n": 0,
            "stress_mae": float("nan"),
            "stress_rmse": float("nan"),
            "sample_mae": float("nan"),
            "principal_p95": float("nan"),
        }
    stress_true = arrays["stress"][mask]
    stress_pred = pred["stress"][mask]
    diff = stress_pred - stress_true
    sample_mae = _sample_abs_err(stress_pred, stress_true)
    principal_err = _principal_sample_err(pred["stress_principal"][mask], arrays["stress_principal"][mask])
    return {
        "n": int(np.sum(mask)),
        "stress_mae": float(np.mean(np.abs(diff))),
        "stress_rmse": float(np.sqrt(np.mean(diff**2))),
        "sample_mae": float(np.mean(sample_mae)),
        "principal_p95": float(np.quantile(principal_err, 0.95)),
    }


def _bucket_metrics(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    split_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name, key in BUCKETS:
        mask = split_mask & panel[key].astype(bool)
        rows[name] = _mask_metrics(arrays, pred, mask)
    return rows


def _coverage(mask_learned: np.ndarray | None, plastic_mask: np.ndarray) -> float:
    if mask_learned is None:
        return 1.0 if np.any(plastic_mask) else 0.0
    if not np.any(plastic_mask):
        return 0.0
    return float(np.mean(mask_learned[plastic_mask]))


def _harm_metrics(arrays: dict[str, np.ndarray], pred: dict[str, np.ndarray], learned_mask: np.ndarray | None) -> dict[str, float] | None:
    if pred.get("branch_probabilities") is None:
        return None
    if learned_mask is None:
        learned_mask = np.ones(arrays["strain_eng"].shape[0], dtype=bool)
    if not np.any(learned_mask):
        return None
    branch_pred = np.argmax(pred["branch_probabilities"][learned_mask], axis=1)
    harm = branch_harm_metrics_3d(
        arrays["strain_eng"][learned_mask],
        branch_pred,
        c_bar=arrays["material_reduced"][learned_mask, 0],
        sin_phi=arrays["material_reduced"][learned_mask, 1],
        shear=arrays["material_reduced"][learned_mask, 2],
        bulk=arrays["material_reduced"][learned_mask, 3],
        lame=arrays["material_reduced"][learned_mask, 4],
    )
    return {
        "n_learned": int(np.sum(learned_mask)),
        "wrong_rate": float(np.mean(harm.wrong_branch)),
        "harmful_fail_rate": float(np.mean(harm.harmful_fail)),
    }


def _evaluate_split(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    *,
    split_name: str,
) -> dict[str, Any]:
    split_mask = arrays["split_id"] == SPLIT_TO_ID[split_name]
    broad = split_mask
    hard = split_mask & panel["hard_mask"].astype(bool)
    plastic = split_mask & panel["plastic_mask"].astype(bool)
    hard_plastic = hard & panel["plastic_mask"].astype(bool)
    learned_mask = pred.get("learned_mask")
    learned_split_mask = None if learned_mask is None else (split_mask & learned_mask)

    result = {
        "split": split_name,
        "broad": _mask_metrics(arrays, pred, broad),
        "hard": _mask_metrics(arrays, pred, hard),
        "broad_plastic": _mask_metrics(arrays, pred, plastic),
        "hard_plastic": _mask_metrics(arrays, pred, hard_plastic),
        "plastic_coverage": _coverage(learned_mask, plastic),
        "bucket_metrics": _bucket_metrics(arrays, panel, pred, split_mask),
        "harm": _harm_metrics(arrays, pred, learned_split_mask),
    }
    if learned_mask is not None:
        result["learned_rows"] = int(np.sum(learned_mask[split_mask]))
        result["fallback_rows"] = int(np.sum(pred["fallback_mask"][split_mask]))
        result["elastic_rows"] = int(np.sum(pred["elastic_mask"][split_mask]))
    return result


def _lexicographic_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        row["val"]["broad_plastic"]["stress_mae"],
        row["val"]["hard_plastic"]["stress_mae"],
        row["val"]["hard"]["principal_p95"],
        -row["val"]["plastic_coverage"],
        row["delta_geom"],
    )


def _select_threshold(rows: list[dict[str, Any]], baseline_val: dict[str, Any]) -> dict[str, Any]:
    passing: list[dict[str, Any]] = []
    for row in rows:
        if row["val"]["plastic_coverage"] < 0.60:
            continue
        if row["test"]["plastic_coverage"] < 0.40:
            continue
        if row["val"]["broad_plastic"]["stress_mae"] >= baseline_val["broad_plastic"]["stress_mae"]:
            continue
        if row["val"]["hard_plastic"]["stress_mae"] >= baseline_val["hard_plastic"]["stress_mae"]:
            continue
        if row["val"]["hard"]["principal_p95"] >= baseline_val["hard"]["principal_p95"]:
            continue
        passing.append(row)
    if passing:
        accepted = min(passing, key=lambda row: row["delta_geom"])
    else:
        accepted = None
    fallback_best = min(rows, key=_lexicographic_key)
    return {
        "accepted": accepted,
        "fallback_best": fallback_best,
        "passing_count": len(passing),
        "status": "accepted" if accepted is not None else "no_threshold_passed_coverage_and_tail_gates",
    }


def _write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "delta_geom",
                "val_broad_plastic_mae",
                "val_hard_plastic_mae",
                "val_hard_principal_p95",
                "val_plastic_coverage",
                "test_broad_plastic_mae",
                "test_hard_plastic_mae",
                "test_hard_principal_p95",
                "test_plastic_coverage",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["delta_geom"],
                    row["val"]["broad_plastic"]["stress_mae"],
                    row["val"]["hard_plastic"]["stress_mae"],
                    row["val"]["hard"]["principal_p95"],
                    row["val"]["plastic_coverage"],
                    row["test"]["broad_plastic"]["stress_mae"],
                    row["test"]["hard_plastic"]["stress_mae"],
                    row["test"]["hard"]["principal_p95"],
                    row["test"]["plastic_coverage"],
                ]
            )


def _plot_threshold_curves(path: Path, rows: list[dict[str, Any]], baseline_val: dict[str, Any]) -> None:
    delta = np.array([row["delta_geom"] for row in rows], dtype=float)
    val_cov = np.array([row["val"]["plastic_coverage"] for row in rows], dtype=float)
    test_cov = np.array([row["test"]["plastic_coverage"] for row in rows], dtype=float)
    val_broad = np.array([row["val"]["broad_plastic"]["stress_mae"] for row in rows], dtype=float)
    val_hard = np.array([row["val"]["hard_plastic"]["stress_mae"] for row in rows], dtype=float)
    val_p95 = np.array([row["val"]["hard"]["principal_p95"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(delta, val_cov, marker="o", label="val coverage")
    axes[0].plot(delta, test_cov, marker="s", label="test coverage")
    axes[0].axhline(0.60, color="tab:blue", linestyle="--", alpha=0.6)
    axes[0].axhline(0.40, color="tab:orange", linestyle="--", alpha=0.6)
    axes[0].set_xlabel("delta_geom")
    axes[0].set_ylabel("plastic coverage")
    axes[0].set_title("Coverage")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(delta, val_broad, marker="o", label="val broad plastic MAE")
    axes[1].plot(delta, val_hard, marker="s", label="val hard plastic MAE")
    axes[1].axhline(baseline_val["broad_plastic"]["stress_mae"], color="tab:blue", linestyle="--", alpha=0.6)
    axes[1].axhline(baseline_val["hard_plastic"]["stress_mae"], color="tab:orange", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("delta_geom")
    axes[1].set_ylabel("stress MAE")
    axes[1].set_title("Validation MAE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(delta, val_p95, marker="o", label="val hard p95")
    axes[2].axhline(baseline_val["hard"]["principal_p95"], color="tab:red", linestyle="--", alpha=0.6, label="baseline p95")
    axes[2].set_xlabel("delta_geom")
    axes[2].set_ylabel("principal p95")
    axes[2].set_title("Validation Tail Error")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    report_path: Path,
    *,
    baseline_ckpt: Path,
    baseline_val: dict[str, Any],
    baseline_test: dict[str, Any],
    selection: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    accepted = selection["accepted"]
    fallback_best = selection["fallback_best"]
    lines: list[str] = []
    lines.append("# Hybrid Pivot Experiment 1: Wrapper Baseline")
    lines.append("")
    lines.append("This report evaluates the March 12, 2026 staged mixed-material checkpoint under an analytic exact-elastic plus exact-fallback wrapper.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- baseline checkpoint: `{baseline_ckpt}`")
    lines.append(f"- thresholds swept: `{list(THRESHOLDS)}`")
    lines.append(f"- output root: `{output_root}`")
    lines.append(f"- coverage/tail gate status: `{selection['status']}`")
    lines.append("")
    lines.append("## Baseline On Frozen Panels")
    lines.append("")
    lines.append("| Split | Broad MAE | Hard MAE | Broad Plastic MAE | Hard Plastic MAE | Hard p95 Principal | Plastic Coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| val raw_branch | {baseline_val['broad']['stress_mae']:.6f} | {baseline_val['hard']['stress_mae']:.6f} | {baseline_val['broad_plastic']['stress_mae']:.6f} | {baseline_val['hard_plastic']['stress_mae']:.6f} | {baseline_val['hard']['principal_p95']:.6f} | 1.000000 |"
    )
    lines.append(
        f"| test raw_branch | {baseline_test['broad']['stress_mae']:.6f} | {baseline_test['hard']['stress_mae']:.6f} | {baseline_test['broad_plastic']['stress_mae']:.6f} | {baseline_test['hard_plastic']['stress_mae']:.6f} | {baseline_test['hard']['principal_p95']:.6f} | 1.000000 |"
    )
    lines.append("")
    lines.append("## Threshold Sweep")
    lines.append("")
    lines.append("| delta_geom | Val Broad MAE | Val Hard MAE | Val Broad Plastic MAE | Val Hard Plastic MAE | Val Hard p95 Principal | Val Coverage | Test Broad Plastic MAE | Test Hard Plastic MAE | Test Hard p95 Principal | Test Coverage |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['delta_geom']:.3f} | {row['val']['broad']['stress_mae']:.6f} | {row['val']['hard']['stress_mae']:.6f} | {row['val']['broad_plastic']['stress_mae']:.6f} | {row['val']['hard_plastic']['stress_mae']:.6f} | "
            f"{row['val']['hard']['principal_p95']:.6f} | {row['val']['plastic_coverage']:.6f} | {row['test']['broad_plastic']['stress_mae']:.6f} | "
            f"{row['test']['hard_plastic']['stress_mae']:.6f} | {row['test']['hard']['principal_p95']:.6f} | {row['test']['plastic_coverage']:.6f} |"
        )
    lines.append("")
    lines.append(f"![Threshold curves]({(output_root / 'threshold_curves.png').as_posix()})")
    lines.append("")
    lines.append("## Threshold Selection")
    lines.append("")
    if accepted is None:
        lines.append("- No swept `delta_geom` satisfied the required `val >= 60%` and `test >= 40%` plastic-coverage gates while also improving hard-tail validation metrics over the March 12 baseline.")
        lines.append(f"- Best error-only fallback threshold: `{fallback_best['delta_geom']:.3f}`")
        lines.append(f"- Best error-only val broad plastic MAE: `{fallback_best['val']['broad_plastic']['stress_mae']:.6f}`")
        lines.append(f"- Best error-only val hard plastic MAE: `{fallback_best['val']['hard_plastic']['stress_mae']:.6f}`")
        lines.append(f"- Best error-only val hard p95 principal: `{fallback_best['val']['hard']['principal_p95']:.6f}`")
        lines.append(f"- Best error-only val plastic coverage: `{fallback_best['val']['plastic_coverage']:.6f}`")
        lines.append(f"- Best error-only test plastic coverage: `{fallback_best['test']['plastic_coverage']:.6f}`")
        lines.append("- Recommendation: continue to Candidate B rather than ship the wrapper-only policy.")
    else:
        lines.append(f"- Accepted `delta_geom`: `{accepted['delta_geom']:.3f}`")
        lines.append(f"- val broad plastic MAE: `{accepted['val']['broad_plastic']['stress_mae']:.6f}`")
        lines.append(f"- val hard plastic MAE: `{accepted['val']['hard_plastic']['stress_mae']:.6f}`")
        lines.append(f"- val hard p95 principal: `{accepted['val']['hard']['principal_p95']:.6f}`")
        lines.append(f"- val plastic coverage: `{accepted['val']['plastic_coverage']:.6f}`")
        lines.append(f"- test plastic coverage: `{accepted['test']['plastic_coverage']:.6f}`")
    lines.append("")
    lines.append("## Hard-Bucket Test Metrics At Best Error-Only Threshold")
    lines.append("")
    lines.append("| Bucket | Rows | Stress MAE | Principal p95 |")
    lines.append("|---|---:|---:|---:|")
    for name, bucket in fallback_best["test"]["bucket_metrics"].items():
        lines.append(f"| {name} | {bucket['n']} | {bucket['stress_mae']:.6f} | {bucket['principal_p95']:.6f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Coverage is measured only on plastic rows.")
    lines.append("- The wrapper uses exact constitutive geometry for dispatch and ignores the checkpoint branch head when deciding elastic or fallback behavior.")
    lines.append("- A threshold is only considered admissible if it clears both plastic-coverage gates and improves the hard validation tail against the March 12 baseline.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stub(path: Path, title: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\nStatus: not run yet in this execution pass.\n", encoding="utf-8")


def run_wrapper_baseline(
    *,
    full_export: Path,
    baseline_checkpoint: Path,
    output_root: Path,
    report_path: Path,
    samples_per_call: int,
    seed: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    panel_result = build_hybrid_real_panels(
        full_export=full_export,
        output_root=output_root,
        report_path=(ROOT / "docs" / "hybrid_pivot_exp0_panels_20260323.md").resolve(),
        samples_per_call=samples_per_call,
        seed=seed,
        split_fractions=(0.70, 0.15, 0.15),
    )
    arrays = _load_all(Path(panel_result["dataset_path"]))
    panel = _load_all(Path(panel_result["panel_path"]))

    val_mask = arrays["split_id"] == SPLIT_TO_ID["val"]
    test_mask = arrays["split_id"] == SPLIT_TO_ID["test"]

    base_pred = {
        "stress": np.zeros_like(arrays["stress"], dtype=np.float32),
        "stress_principal": np.zeros_like(arrays["stress_principal"], dtype=np.float32),
        "branch_probabilities": None,
    }
    if np.any(val_mask):
        pred_val = predict_with_checkpoint(
            baseline_checkpoint,
            arrays["strain_eng"][val_mask],
            arrays["material_reduced"][val_mask],
            device=device,
            batch_size=batch_size,
        )
        base_pred["stress"][val_mask] = pred_val["stress"]
        base_pred["stress_principal"][val_mask] = pred_val.get("stress_principal", _principal_from_stress(pred_val["stress"]))
        if pred_val["branch_probabilities"] is not None:
            if base_pred["branch_probabilities"] is None:
                base_pred["branch_probabilities"] = np.zeros((arrays["stress"].shape[0], pred_val["branch_probabilities"].shape[1]), dtype=np.float32)
            base_pred["branch_probabilities"][val_mask] = pred_val["branch_probabilities"]
    if np.any(test_mask):
        pred_test = predict_with_checkpoint(
            baseline_checkpoint,
            arrays["strain_eng"][test_mask],
            arrays["material_reduced"][test_mask],
            device=device,
            batch_size=batch_size,
        )
        base_pred["stress"][test_mask] = pred_test["stress"]
        base_pred["stress_principal"][test_mask] = pred_test.get("stress_principal", _principal_from_stress(pred_test["stress"]))
        if pred_test["branch_probabilities"] is not None:
            if base_pred["branch_probabilities"] is None:
                base_pred["branch_probabilities"] = np.zeros((arrays["stress"].shape[0], pred_test["branch_probabilities"].shape[1]), dtype=np.float32)
            base_pred["branch_probabilities"][test_mask] = pred_test["branch_probabilities"]

    baseline_val = _evaluate_split(arrays, panel, base_pred, split_name="val")
    baseline_test = _evaluate_split(arrays, panel, base_pred, split_name="test")

    wrapper_rows: list[dict[str, Any]] = []
    for delta_geom in THRESHOLDS:
        pred = {
            "stress": np.zeros_like(arrays["stress"], dtype=np.float32),
            "stress_principal": np.zeros_like(arrays["stress_principal"], dtype=np.float32),
            "branch_probabilities": np.zeros((arrays["stress"].shape[0], 5), dtype=np.float32),
            "elastic_mask": np.zeros(arrays["stress"].shape[0], dtype=bool),
            "fallback_mask": np.zeros(arrays["stress"].shape[0], dtype=bool),
            "learned_mask": np.zeros(arrays["stress"].shape[0], dtype=bool),
        }
        for split_name, split_mask in (("val", val_mask), ("test", test_mask)):
            if not np.any(split_mask):
                continue
            split_pred = hybrid_predict_with_checkpoint(
                baseline_checkpoint,
                arrays["strain_eng"][split_mask],
                arrays["material_reduced"][split_mask],
                delta_geom=delta_geom,
                device=device,
                batch_size=batch_size,
            )
            pred["stress"][split_mask] = split_pred["stress"]
            pred["stress_principal"][split_mask] = split_pred["stress_principal"]
            pred["branch_probabilities"][split_mask] = split_pred["branch_probabilities"]
            pred["elastic_mask"][split_mask] = split_pred["elastic_mask"]
            pred["fallback_mask"][split_mask] = split_pred["fallback_mask"]
            pred["learned_mask"][split_mask] = split_pred["learned_mask"]
        wrapper_rows.append(
            {
                "delta_geom": float(delta_geom),
                "val": _evaluate_split(arrays, panel, pred, split_name="val"),
                "test": _evaluate_split(arrays, panel, pred, split_name="test"),
            }
        )

    selection = _select_threshold(wrapper_rows, baseline_val)
    exp_dir = output_root / "exp1_wrapper"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "baseline_metrics.json").write_text(
        json.dumps(_json_safe({"val": baseline_val, "test": baseline_test}), indent=2),
        encoding="utf-8",
    )
    (exp_dir / "threshold_metrics.json").write_text(json.dumps(_json_safe(wrapper_rows), indent=2), encoding="utf-8")
    (exp_dir / "chosen_threshold.json").write_text(json.dumps(_json_safe(selection), indent=2), encoding="utf-8")
    _write_threshold_csv(exp_dir / "threshold_summary.csv", wrapper_rows)
    _plot_threshold_curves(exp_dir / "threshold_curves.png", wrapper_rows, baseline_val)
    _write_report(
        report_path,
        baseline_ckpt=baseline_checkpoint,
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        selection=selection,
        rows=wrapper_rows,
        output_root=exp_dir,
    )
    _write_stub(ROOT / "docs" / "hybrid_pivot_exp2_candidate_b_20260323.md", "Hybrid Pivot Experiment 2: Candidate B")
    _write_stub(ROOT / "docs" / "hybrid_pivot_exp5_candidate_c_20260323.md", "Hybrid Pivot Experiment 5: Candidate C")
    _write_stub(ROOT / "docs" / "hybrid_pivot_fe_handoff_20260323.md", "Hybrid Pivot FE Handoff")
    return {
        "panel_result": panel_result,
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "threshold_rows": wrapper_rows,
        "selection": selection,
        "report_path": str(report_path),
    }


def main() -> None:
    args = parse_args()
    result = run_wrapper_baseline(
        full_export=(ROOT / args.full_export).resolve(),
        baseline_checkpoint=(ROOT / args.baseline_checkpoint).resolve(),
        output_root=(ROOT / args.output_root).resolve(),
        report_path=(ROOT / args.report_md).resolve(),
        samples_per_call=args.samples_per_call,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
