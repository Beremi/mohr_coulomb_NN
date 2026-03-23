#!/usr/bin/env python
"""Train plastic branch experts for the cover-layer surrogate and evaluate gated ensembles."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.models import compute_trial_stress
from mc_surrogate.mohr_coulomb import BRANCH_NAMES
from mc_surrogate.training import TrainingConfig, evaluate_checkpoint_on_dataset, predict_with_checkpoint, train_model

sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from run_cover_layer_single_material_plan import (  # noqa: E402
    _compute_real_dissection,
    _json_safe,
    _plot_error_vs_magnitude,
    _plot_history_log,
    _plot_relative_error_cdf,
)


PLASTIC_BRANCHES = (
    ("smooth", 1),
    ("left_edge", 2),
    ("right_edge", 3),
    ("apex", 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        default="experiment_runs/real_sim/cover_layer_single_material_20260313",
    )
    parser.add_argument(
        "--output-root",
        default="experiment_runs/real_sim/cover_layer_branch_experts_20260313",
    )
    parser.add_argument("--report-md", default="docs/cover_layer_branch_experts.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--plateau-patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lbfgs-epochs", type=int, default=8)
    parser.add_argument("--lbfgs-lr", type=float, default=0.25)
    parser.add_argument("--log-every-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1551)
    return parser.parse_args()


def _load_dataset_all(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        attrs = {key: f.attrs[key] for key in f.attrs.keys()}
    return arrays, arrays["split_id"], attrs


def _write_dataset(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


def _build_branch_dataset(real_dataset_path: Path, output_path: Path, branch_id: int) -> Path:
    arrays, split_id, attrs = _load_dataset_all(real_dataset_path)
    mask = arrays["branch_id"] == branch_id
    out = {key: value[mask] for key, value in arrays.items() if key != "split_id"}
    out["split_id"] = split_id[mask]
    attrs = dict(attrs)
    attrs["expert_branch_id"] = int(branch_id)
    attrs["expert_branch_name"] = BRANCH_NAMES[branch_id]
    return _write_dataset(output_path, out, attrs)


def _train_expert(
    *,
    branch_name: str,
    dataset_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    run_dir = output_root / f"expert_{branch_name}"
    config = TrainingConfig(
        dataset=str(dataset_path),
        run_dir=str(run_dir),
        model_kind="raw",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        dropout=0.0,
        seed=args.seed + seed_offset,
        patience=args.patience,
        grad_clip=1.0,
        branch_loss_weight=0.0,
        num_workers=0,
        device=args.device,
        scheduler_kind="plateau",
        min_lr=args.min_lr,
        plateau_factor=0.5,
        plateau_patience=args.plateau_patience,
        lbfgs_epochs=args.lbfgs_epochs,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_max_iter=20,
        lbfgs_history_size=100,
        log_every_epochs=args.log_every_epochs,
        stress_weight_alpha=0.0,
        stress_weight_scale=250.0,
        checkpoint_metric="stress_mae",
    )
    summary = train_model(config)
    (run_dir / "config_used.json").write_text(json.dumps(_json_safe(asdict(config)), indent=2), encoding="utf-8")
    _plot_history_log(run_dir / "history.csv", run_dir / "history_log.png", title=f"expert_{branch_name}")
    return summary


def _eval_metrics(stress_true: np.ndarray, stress_pred: np.ndarray, branch_true: np.ndarray | None = None, branch_pred: np.ndarray | None = None) -> dict[str, Any]:
    abs_err = np.abs(stress_pred - stress_true)
    metrics = {
        "n_samples": int(stress_true.shape[0]),
        "stress_mae": float(np.mean(abs_err)),
        "stress_rmse": float(np.sqrt(np.mean((stress_pred - stress_true) ** 2))),
        "stress_max_abs": float(np.max(abs_err)),
        "per_component_mae": np.mean(abs_err, axis=0).tolist(),
    }
    if branch_true is not None and branch_pred is not None:
        metrics["branch_accuracy"] = float(np.mean(branch_true == branch_pred))
        metrics["branch_confusion"] = [
            [
                int(np.sum((branch_true == i) & (branch_pred == j)))
                for j in range(len(BRANCH_NAMES))
            ]
            for i in range(len(BRANCH_NAMES))
        ]
    if branch_true is not None:
        metrics["per_branch_stress_mae"] = {
            BRANCH_NAMES[i]: float(np.mean(np.abs(stress_pred[branch_true == i] - stress_true[branch_true == i])))
            for i in range(len(BRANCH_NAMES))
            if np.any(branch_true == i)
        }
    return metrics


def _ensemble_predict(
    *,
    baseline_ckpt: Path,
    expert_ckpts: dict[int, Path],
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    branch_true: np.ndarray | None,
    mode: str,
    threshold: float | None,
    device: str,
) -> dict[str, Any]:
    baseline_pred = predict_with_checkpoint(baseline_ckpt, strain_eng, material_reduced, device=device, batch_size=16384)
    baseline_stress = baseline_pred["stress"]
    branch_probs = baseline_pred["branch_probabilities"]
    pred_branch = np.argmax(branch_probs, axis=1)
    trial_stress = compute_trial_stress(strain_eng, material_reduced)

    expert_predictions: dict[int, np.ndarray] = {}
    for branch_id, ckpt in expert_ckpts.items():
        expert_predictions[branch_id] = predict_with_checkpoint(
            ckpt,
            strain_eng,
            material_reduced,
            device=device,
            batch_size=16384,
        )["stress"]

    if mode == "baseline":
        stress = baseline_stress
        used_branch = pred_branch
    else:
        if mode == "oracle":
            if branch_true is None:
                raise ValueError("oracle mode requires branch_true.")
            gate_branch = branch_true.astype(int)
        else:
            gate_branch = pred_branch.astype(int)
        stress = np.empty_like(baseline_stress)
        for branch_id in range(len(BRANCH_NAMES)):
            mask = gate_branch == branch_id
            if not np.any(mask):
                continue
            if branch_id == 0:
                stress[mask] = trial_stress[mask]
            else:
                stress[mask] = expert_predictions[branch_id][mask]
        used_branch = gate_branch
        if mode == "threshold":
            if threshold is None:
                raise ValueError("threshold mode requires threshold.")
            conf = np.max(branch_probs, axis=1)
            fallback_mask = conf < threshold
            if np.any(fallback_mask):
                stress[fallback_mask] = baseline_stress[fallback_mask]

    return {
        "stress": stress.astype(np.float32),
        "branch_pred": used_branch.astype(np.int64),
        "branch_probs": branch_probs.astype(np.float32),
    }


def _plot_parity(stress_true: np.ndarray, stress_pred: np.ndarray, output_path: Path) -> Path:
    if stress_true.size > 4000 * 6:
        rng = np.random.default_rng(0)
        idx = rng.choice(stress_true.shape[0], size=4000, replace=False)
        stress_true = stress_true[idx]
        stress_pred = stress_pred[idx]
    lo = float(min(stress_true.min(), stress_pred.min()))
    hi = float(max(stress_true.max(), stress_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(stress_true.reshape(-1), stress_pred.reshape(-1), s=6, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("true stress")
    plt.ylabel("predicted stress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_confusion(confusion: list[list[int]], output_path: Path) -> Path:
    mat = np.asarray(confusion, dtype=float)
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(BRANCH_NAMES)), BRANCH_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(BRANCH_NAMES)), BRANCH_NAMES)
    plt.xlabel("predicted branch")
    plt.ylabel("true branch")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _evaluate_mode(
    *,
    name: str,
    predictions: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _eval_metrics(
        arrays["stress"],
        predictions["stress"],
        branch_true=arrays["branch_id"].astype(int),
        branch_pred=predictions["branch_pred"].astype(int),
    )
    (output_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _plot_parity(arrays["stress"], predictions["stress"], output_dir / "parity.png")
    _plot_relative_error_cdf(arrays["stress"], predictions["stress"], output_dir / "relative_error_cdf.png", title=f"{name} relative error")
    _plot_error_vs_magnitude(arrays["stress"], predictions["stress"], output_dir / "error_vs_magnitude.png", title=f"{name} error vs stress magnitude")
    _plot_confusion(metrics["branch_confusion"], output_dir / "branch_confusion.png")
    return {"name": name, "metrics": metrics}


def _plot_compare(rows: list[dict[str, Any]], output_path: Path, key: str, title: str) -> Path:
    names = [row["name"] for row in rows]
    real = [row["real"][key] for row in rows]
    synth = [row["synthetic"][key] for row in rows]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, real, width=width, label="real")
    ax.bar(x + width / 2, synth, width=width, label="synthetic")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_report(
    *,
    report_path: Path,
    output_root: Path,
    branch_summaries: dict[str, Any],
    rows: list[dict[str, Any]],
    best_real_name: str,
    best_dissection: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Cover-Layer Branch Experts")
    lines.append("")
    lines.append("This report tests a branch-specialized plastic surrogate on top of the existing strong baseline branch gate.")
    lines.append("")
    lines.append("Setup:")
    lines.append("")
    lines.append("- branch gate: existing `baseline_raw_branch` checkpoint")
    lines.append("- elastic branch: exact elastic trial stress")
    lines.append("- plastic branches: separate raw stress experts for `smooth`, `left_edge`, `right_edge`, `apex`")
    lines.append("- deployable ensemble: predicted-branch routing")
    lines.append("- optimistic ceiling: oracle routing with true branch labels")
    lines.append("- robust variant: confidence-threshold routing with baseline fallback")
    lines.append("")
    lines.append("## Expert Training Sets")
    lines.append("")
    lines.append("| Expert | Train | Val | Test |")
    lines.append("|---|---:|---:|---:|")
    for branch_name, summary in branch_summaries.items():
        lines.append(f"| {branch_name} | {summary['train']} | {summary['val']} | {summary['test']} |")
    lines.append("")
    lines.append("## Ensemble Results")
    lines.append("")
    lines.append("| Mode | Real MAE | Real RMSE | Real Max Abs | Real Branch Acc | Synthetic MAE | Synthetic RMSE | Synthetic Branch Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        r = row["real"]
        s = row["synthetic"]
        lines.append(
            f"| {row['name']} | {r['stress_mae']:.4f} | {r['stress_rmse']:.4f} | {r['stress_max_abs']:.4f} | {r['branch_accuracy']:.4f} | "
            f"{s['stress_mae']:.4f} | {s['stress_rmse']:.4f} | {s['branch_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append(f"Best real-holdout mode: `{best_real_name}`")
    lines.append("")
    lines.append(f"![MAE comparison]({(output_root / 'compare_mae.png').as_posix()})")
    lines.append("")
    lines.append(f"![RMSE comparison]({(output_root / 'compare_rmse.png').as_posix()})")
    lines.append("")
    lines.append("## Per-Expert Histories")
    lines.append("")
    for branch_name, _ in PLASTIC_BRANCHES:
        lines.append(f"### {branch_name}")
        lines.append("")
        lines.append(f"![history]({(output_root / f'expert_{branch_name}' / 'history_log.png').as_posix()})")
        lines.append("")
    lines.append("## Ensemble Figures")
    lines.append("")
    for row in rows:
        mode_dir = output_root / row["name"]
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- real parity: ![real parity]({(mode_dir / 'real' / 'parity.png').as_posix()})")
        lines.append(f"- real relative error: ![real rel]({(mode_dir / 'real' / 'relative_error_cdf.png').as_posix()})")
        lines.append(f"- real error vs magnitude: ![real mag]({(mode_dir / 'real' / 'error_vs_magnitude.png').as_posix()})")
        lines.append(f"- real branch confusion: ![real branch]({(mode_dir / 'real' / 'branch_confusion.png').as_posix()})")
        lines.append(f"- synthetic parity: ![synth parity]({(mode_dir / 'synthetic' / 'parity.png').as_posix()})")
        lines.append(f"- synthetic relative error: ![synth rel]({(mode_dir / 'synthetic' / 'relative_error_cdf.png').as_posix()})")
        lines.append(f"- synthetic error vs magnitude: ![synth mag]({(mode_dir / 'synthetic' / 'error_vs_magnitude.png').as_posix()})")
        lines.append(f"- synthetic branch confusion: ![synth branch]({(mode_dir / 'synthetic' / 'branch_confusion.png').as_posix()})")
        lines.append("")
    lines.append("## Best Real-Holdout Dissection")
    lines.append("")
    lines.append(f"- mean relative sample error: `{best_dissection['relative_error_mean']:.4f}`")
    lines.append(f"- median relative sample error: `{best_dissection['relative_error_median']:.4f}`")
    lines.append(f"- p90 relative sample error: `{best_dissection['relative_error_p90']:.4f}`")
    lines.append(f"- p99 relative sample error: `{best_dissection['relative_error_p99']:.4f}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- If oracle routing improves a lot but predicted routing does not, the branch experts are good and the gate is the bottleneck.")
    lines.append("- If predicted routing plus baseline fallback beats the baseline, this is immediately useful.")
    lines.append("- If even oracle routing barely helps, branch-specialized raw experts are not enough and we should change the plastic target again.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reference_root = (ROOT / args.reference_root).resolve()
    output_root = ROOT / args.output_root
    report_path = ROOT / args.report_md
    output_root.mkdir(parents=True, exist_ok=True)

    real_dataset = reference_root / "cover_layer_full_real_exact_256.h5"
    baseline_ckpt = reference_root / "baseline_raw_branch" / "best.pt"
    synthetic_holdout = reference_root / "cover_layer_full_synthetic_holdout.h5"

    branch_summaries: dict[str, Any] = {}
    expert_ckpts: dict[int, Path] = {}
    for idx, (branch_name, branch_id) in enumerate(PLASTIC_BRANCHES):
        dataset_path = output_root / f"{branch_name}_dataset.h5"
        if not dataset_path.exists():
            _build_branch_dataset(real_dataset, dataset_path, branch_id)
        with h5py.File(dataset_path, "r") as f:
            split = f["split_id"][:]
        branch_summaries[branch_name] = {
            "train": int(np.sum(split == 0)),
            "val": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
        }
        run_dir = output_root / f"expert_{branch_name}"
        if not (run_dir / "best.pt").exists():
            _train_expert(branch_name=branch_name, dataset_path=dataset_path, output_root=output_root, args=args, seed_offset=idx)
        expert_ckpts[branch_id] = run_dir / "best.pt"

    real_arrays = {
        "strain_eng": evaluate_checkpoint_on_dataset(baseline_ckpt, real_dataset, split="test", device=args.device, batch_size=16384)["arrays"]["strain_eng"],
        "stress": evaluate_checkpoint_on_dataset(baseline_ckpt, real_dataset, split="test", device=args.device, batch_size=16384)["arrays"]["stress"],
        "material_reduced": evaluate_checkpoint_on_dataset(baseline_ckpt, real_dataset, split="test", device=args.device, batch_size=16384)["arrays"]["material_reduced"],
        "branch_id": evaluate_checkpoint_on_dataset(baseline_ckpt, real_dataset, split="test", device=args.device, batch_size=16384)["arrays"]["branch_id"],
    }
    synth_arrays = {
        "strain_eng": evaluate_checkpoint_on_dataset(baseline_ckpt, synthetic_holdout, split="test", device=args.device, batch_size=16384)["arrays"]["strain_eng"],
        "stress": evaluate_checkpoint_on_dataset(baseline_ckpt, synthetic_holdout, split="test", device=args.device, batch_size=16384)["arrays"]["stress"],
        "material_reduced": evaluate_checkpoint_on_dataset(baseline_ckpt, synthetic_holdout, split="test", device=args.device, batch_size=16384)["arrays"]["material_reduced"],
        "branch_id": evaluate_checkpoint_on_dataset(baseline_ckpt, synthetic_holdout, split="test", device=args.device, batch_size=16384)["arrays"]["branch_id"],
    }

    baseline_pred_real = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=real_arrays["strain_eng"],
        material_reduced=real_arrays["material_reduced"],
        branch_true=real_arrays["branch_id"],
        mode="baseline",
        threshold=None,
        device=args.device,
    )
    baseline_pred_synth = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=synth_arrays["strain_eng"],
        material_reduced=synth_arrays["material_reduced"],
        branch_true=synth_arrays["branch_id"],
        mode="baseline",
        threshold=None,
        device=args.device,
    )

    oracle_real = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=real_arrays["strain_eng"],
        material_reduced=real_arrays["material_reduced"],
        branch_true=real_arrays["branch_id"],
        mode="oracle",
        threshold=None,
        device=args.device,
    )
    oracle_synth = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=synth_arrays["strain_eng"],
        material_reduced=synth_arrays["material_reduced"],
        branch_true=synth_arrays["branch_id"],
        mode="oracle",
        threshold=None,
        device=args.device,
    )

    pred_real = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=real_arrays["strain_eng"],
        material_reduced=real_arrays["material_reduced"],
        branch_true=real_arrays["branch_id"],
        mode="predicted",
        threshold=None,
        device=args.device,
    )
    pred_synth = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=synth_arrays["strain_eng"],
        material_reduced=synth_arrays["material_reduced"],
        branch_true=synth_arrays["branch_id"],
        mode="predicted",
        threshold=None,
        device=args.device,
    )

    val_arrays = evaluate_checkpoint_on_dataset(baseline_ckpt, real_dataset, split="val", device=args.device, batch_size=16384)["arrays"]
    best_threshold = None
    best_val_mae = float("inf")
    for threshold in [0.45, 0.55, 0.65, 0.75, 0.85]:
        val_pred = _ensemble_predict(
            baseline_ckpt=baseline_ckpt,
            expert_ckpts=expert_ckpts,
            strain_eng=val_arrays["strain_eng"],
            material_reduced=val_arrays["material_reduced"],
            branch_true=val_arrays["branch_id"],
            mode="threshold",
            threshold=threshold,
            device=args.device,
        )
        val_metrics = _eval_metrics(val_arrays["stress"], val_pred["stress"], val_arrays["branch_id"], val_pred["branch_pred"])
        if val_metrics["stress_mae"] < best_val_mae:
            best_val_mae = val_metrics["stress_mae"]
            best_threshold = threshold

    thresh_real = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=real_arrays["strain_eng"],
        material_reduced=real_arrays["material_reduced"],
        branch_true=real_arrays["branch_id"],
        mode="threshold",
        threshold=best_threshold,
        device=args.device,
    )
    thresh_synth = _ensemble_predict(
        baseline_ckpt=baseline_ckpt,
        expert_ckpts=expert_ckpts,
        strain_eng=synth_arrays["strain_eng"],
        material_reduced=synth_arrays["material_reduced"],
        branch_true=synth_arrays["branch_id"],
        mode="threshold",
        threshold=best_threshold,
        device=args.device,
    )

    mode_payloads = [
        ("baseline_reference", baseline_pred_real, baseline_pred_synth),
        ("oracle_branch_experts", oracle_real, oracle_synth),
        ("predicted_branch_experts", pred_real, pred_synth),
        (f"threshold_branch_experts_t{best_threshold:.2f}", thresh_real, thresh_synth),
    ]

    rows: list[dict[str, Any]] = []
    for mode_name, pred_real_mode, pred_synth_mode in mode_payloads:
        real_row = _evaluate_mode(name=mode_name, predictions=pred_real_mode, arrays=real_arrays, output_dir=output_root / mode_name / "real")
        synth_row = _evaluate_mode(name=mode_name, predictions=pred_synth_mode, arrays=synth_arrays, output_dir=output_root / mode_name / "synthetic")
        rows.append({"name": mode_name, "real": real_row["metrics"], "synthetic": synth_row["metrics"]})

    _plot_compare(rows, output_root / "compare_mae.png", key="stress_mae", title="Branch-expert ensemble MAE")
    _plot_compare(rows, output_root / "compare_rmse.png", key="stress_rmse", title="Branch-expert ensemble RMSE")

    best_row = min(rows, key=lambda row: row["real"]["stress_mae"])
    best_name = best_row["name"]
    best_predictions = {
        "baseline_reference": baseline_pred_real["stress"],
        "oracle_branch_experts": oracle_real["stress"],
        "predicted_branch_experts": pred_real["stress"],
        f"threshold_branch_experts_t{best_threshold:.2f}": thresh_real["stress"],
    }[best_name]
    best_dissection = _compute_real_dissection(real_dataset_path=real_dataset, predictions=best_predictions)
    (output_root / "best_real_dissection.json").write_text(json.dumps(_json_safe(best_dissection), indent=2), encoding="utf-8")
    (output_root / "threshold_selection.json").write_text(json.dumps({"best_threshold": best_threshold, "val_stress_mae": best_val_mae}, indent=2), encoding="utf-8")

    _write_report(
        report_path=report_path,
        output_root=output_root,
        branch_summaries=branch_summaries,
        rows=rows,
        best_real_name=best_name,
        best_dissection=best_dissection,
    )


if __name__ == "__main__":
    main()
