from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mc_surrogate.branch_geometry import compute_branch_geometry_principal
from mc_surrogate.mohr_coulomb import branch_harm_metrics_3d
from mc_surrogate.voigt import principal_values_and_vectors_from_strain

BRANCH_NAMES = ("elastic", "smooth", "left_edge", "right_edge", "apex")


def _load_trainer_module():
    script_path = Path(__file__).with_name("train_cover_layer_strain_branch_predictor_synth_only.py")
    spec = importlib.util.spec_from_file_location("cover_branch_trainer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load trainer module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict_numpy(model, x_np: np.ndarray, *, batch_size: int = 16384) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.from_numpy(x_np[start:start + batch_size])
            elastic_logits, plastic_logits = model(xb)
            elastic_pred = torch.argmax(elastic_logits, dim=1)
            plastic_pred = torch.argmax(plastic_logits, dim=1) + 1
            pred = torch.where(elastic_pred == 0, torch.zeros_like(elastic_pred), plastic_pred)
            preds.append(pred.numpy())
    return np.concatenate(preds, axis=0)


def _metrics(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = pred.reshape(-1)
    labels = labels.reshape(-1)
    out = {"accuracy": float(np.mean(pred == labels))}
    recalls = []
    for idx, name in enumerate(BRANCH_NAMES):
        mask = labels == idx
        val = float(np.mean(pred[mask] == labels[mask])) if np.any(mask) else float("nan")
        out[f"recall_{name}"] = val
        recalls.append(val)
    out["macro_recall"] = float(np.nanmean(recalls))
    return out


def _harm_summary(strain: np.ndarray, material: np.ndarray, labels: np.ndarray, pred: np.ndarray, *, tau: float) -> dict[str, float]:
    if material.ndim == 2 and material.shape[0] * 11 == strain.shape[0]:
        material_point = np.repeat(material.astype(np.float32), 11, axis=0)
    else:
        material_point = material.astype(np.float32).reshape(-1, 5)
    harm = branch_harm_metrics_3d(
        strain,
        pred,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
        tau=tau,
    )
    wrong_rate = float(np.mean(harm.wrong_branch))
    soft_rate = float(np.mean(harm.benign_fail))
    hard_rate = float(np.mean(harm.harmful_fail))
    denom_wrong = max(wrong_rate, 1.0e-12)
    return {
        "wrong_rate": wrong_rate,
        "soft_fail_rate": soft_rate,
        "hard_fail_rate": hard_rate,
        "soft_share_of_wrong": soft_rate / denom_wrong,
        "hard_share_of_wrong": hard_rate / denom_wrong,
        "hard_adjacent_fail_rate": float(np.mean(harm.harmful_adjacent_fail)),
        "hard_non_adjacent_fail_rate": float(np.mean(harm.harmful_non_adjacent_fail)),
        "median_rel_e_sigma_wrong": float(np.median(harm.rel_e_sigma[harm.wrong_branch])) if np.any(harm.wrong_branch) else 0.0,
        "p95_rel_e_sigma_wrong": float(np.quantile(harm.rel_e_sigma[harm.wrong_branch], 0.95)) if np.any(harm.wrong_branch) else 0.0,
    }


def _harm_confusions(strain: np.ndarray, material: np.ndarray, labels: np.ndarray, pred: np.ndarray, *, tau: float) -> list[dict[str, object]]:
    if material.ndim == 2 and material.shape[0] * 11 == strain.shape[0]:
        material_point = np.repeat(material.astype(np.float32), 11, axis=0)
    else:
        material_point = material.astype(np.float32).reshape(-1, 5)
    harm = branch_harm_metrics_3d(
        strain,
        pred,
        c_bar=material_point[:, 0],
        sin_phi=material_point[:, 1],
        shear=material_point[:, 2],
        bulk=material_point[:, 3],
        lame=material_point[:, 4],
        tau=tau,
    )
    rows: list[dict[str, object]] = []
    for true_id, true_name in enumerate(BRANCH_NAMES):
        for pred_id, pred_name in enumerate(BRANCH_NAMES):
            if true_id == pred_id:
                continue
            mask = (labels.reshape(-1) == true_id) & (pred.reshape(-1) == pred_id)
            if not np.any(mask):
                continue
            rel = harm.rel_e_sigma[mask]
            rows.append(
                {
                    "true_branch": true_name,
                    "pred_branch": pred_name,
                    "count": int(np.sum(mask)),
                    "harmful_rate": float(np.mean(harm.harmful_fail[mask])),
                    "soft_rate": float(np.mean(harm.benign_fail[mask])),
                    "median_rel_e_sigma": float(np.median(rel)),
                    "p95_rel_e_sigma": float(np.quantile(rel, 0.95)),
                }
            )
    rows.sort(key=lambda row: (row["harmful_rate"], row["count"]), reverse=True)
    return rows


def _build_split(trainer, *, export_path: Path, call_names: list[str], max_elements_per_call: int, seed: int, ckpt) -> dict[str, np.ndarray]:
    _, _, strain, branch, material = trainer.collect_blocks(
        export_path,
        call_names=call_names,
        max_elements_per_call=max_elements_per_call,
        seed=seed,
    )
    x_base = trainer._build_point_features(strain, material, feature_set=str(ckpt["feature_set"]))
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_scaled = ((x_base - x_mean) / np.where(np.abs(x_std) < 1.0e-6, 1.0, x_std)).astype(np.float32)
    return {
        "x": x_scaled,
        "y": branch.reshape(-1).astype(np.int64),
        "strain": strain.reshape(-1, 6).astype(np.float32),
        "material": material.astype(np.float32),
    }


def main() -> None:
    trainer = _load_trainer_module()

    parser = argparse.ArgumentParser(description="Report soft vs hard identification failures for the best saved branch model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_100cycles_20260315/loop_17_accepted.pt"),
    )
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json"),
    )
    parser.add_argument(
        "--regime-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_regimes.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_best_model_soft_hard_fails_20260316"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_best_model_soft_hard_fails.md"),
    )
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--tau-harm", type=float, default=1.0e-2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = trainer.HierarchicalBranchNet(
        in_dim=int(ckpt["input_dim"]),
        width=int(ckpt["width"]),
        depth=int(ckpt["depth"]),
    )
    model.load_state_dict(ckpt["state_dict"])

    splits = trainer.load_split_calls(args.split_json)
    regimes = trainer.load_call_regimes(args.regime_json)
    real_val_slice_calls = trainer._spread_pick_exact(splits["real_val"], count=4, regimes=regimes)
    real_val_large_calls = trainer._spread_pick_exact(splits["real_val"], count=32, regimes=regimes)
    real_test_calls = trainer._spread_pick_exact(splits["real_test"], count=4, regimes=regimes)

    split_data = {
        "real_val_slice": _build_split(
            trainer,
            export_path=args.export,
            call_names=real_val_slice_calls,
            max_elements_per_call=args.max_elements_per_call,
            seed=20,
            ckpt=ckpt,
        ),
        "real_val_large": _build_split(
            trainer,
            export_path=args.export,
            call_names=real_val_large_calls,
            max_elements_per_call=args.max_elements_per_call,
            seed=21,
            ckpt=ckpt,
        ),
        "real_val_full": _build_split(
            trainer,
            export_path=args.export,
            call_names=splits["real_val"],
            max_elements_per_call=args.max_elements_per_call,
            seed=701,
            ckpt=ckpt,
        ),
        "real_test": _build_split(
            trainer,
            export_path=args.export,
            call_names=real_test_calls,
            max_elements_per_call=args.max_elements_per_call,
            seed=22,
            ckpt=ckpt,
        ),
    }

    large = split_data["real_val_large"]
    principal_large, _ = principal_values_and_vectors_from_strain(large["strain"])
    material_large = np.repeat(large["material"].astype(np.float32), 11, axis=0)
    geom_large = compute_branch_geometry_principal(
        principal_large,
        c_bar=material_large[:, 0],
        sin_phi=material_large[:, 1],
        shear=material_large[:, 2],
        bulk=material_large[:, 3],
        lame=material_large[:, 4],
    )
    strain_norm_large = np.linalg.norm(large["strain"], axis=1)
    gap_ratio_large = np.minimum(geom_large.delta12, geom_large.delta23) / (
        np.abs(principal_large[:, 0]) + np.abs(principal_large[:, 1]) + np.abs(principal_large[:, 2]) + 1.0e-12
    )
    hard_mask = (
        (np.abs(geom_large.m_yield) <= 0.05)
        | (np.abs(geom_large.m_smooth_left) <= 0.05)
        | (np.abs(geom_large.m_smooth_right) <= 0.05)
        | (np.abs(geom_large.m_left_apex) <= 0.05)
        | (np.abs(geom_large.m_right_apex) <= 0.05)
        | (gap_ratio_large <= 0.03)
        | (strain_norm_large >= float(np.quantile(strain_norm_large, 0.90)))
    )
    split_data["real_val_hard"] = {
        "x": large["x"][hard_mask],
        "y": large["y"][hard_mask],
        "strain": large["strain"][hard_mask],
        "material": material_large[hard_mask],
    }

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "splits": {
            "real_val_slice_calls": real_val_slice_calls,
            "real_val_large_calls": real_val_large_calls,
            "real_test_calls": real_test_calls,
            "real_val_full_calls": len(splits["real_val"]),
        },
        "metrics": {},
        "harm": {},
        "top_hard_confusions": {},
    }

    for split_name, split in split_data.items():
        pred = _predict_numpy(model, split["x"])
        summary["metrics"][split_name] = _metrics(pred, split["y"])
        summary["harm"][split_name] = _harm_summary(split["strain"], split["material"], split["y"], pred, tau=args.tau_harm)
        summary["top_hard_confusions"][split_name] = _harm_confusions(
            split["strain"],
            split["material"],
            split["y"],
            pred,
            tau=args.tau_harm,
        )[:12]

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Best Branch Model: Soft vs Hard Fail Report",
        "",
        "## Model",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- feature set: `{ckpt['feature_set']}`",
        f"- architecture: `hierarchical w{ckpt['width']} d{ckpt['depth']}`",
        f"- hard-fail threshold `tau`: `{args.tau_harm}`",
        "",
        "## Split Summary",
        "",
        "| split | accuracy | macro recall | wrong | soft fail | hard fail | hard adjacent | hard non-adjacent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("real_val_slice", "real_val_large", "real_val_hard", "real_val_full", "real_test"):
        metrics = summary["metrics"][split_name]
        harm = summary["harm"][split_name]
        lines.append(
            f"| {split_name} | {metrics['accuracy']:.4f} | {metrics['macro_recall']:.4f} | "
            f"{harm['wrong_rate']:.4f} | {harm['soft_fail_rate']:.4f} | {harm['hard_fail_rate']:.4f} | "
            f"{harm['hard_adjacent_fail_rate']:.4f} | {harm['hard_non_adjacent_fail_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    for split_name in ("real_val_large", "real_val_hard", "real_val_full", "real_test"):
        harm = summary["harm"][split_name]
        lines.append(
            f"- `{split_name}`: soft among wrong = `{harm['soft_share_of_wrong']:.4f}`, hard among wrong = `{harm['hard_share_of_wrong']:.4f}`, "
            f"median / p95 `rel_e_sigma` on wrong = `{harm['median_rel_e_sigma_wrong']:.4f}` / `{harm['p95_rel_e_sigma_wrong']:.4f}`"
        )
    for split_name in ("real_val_large", "real_test"):
        lines.extend(
            [
                "",
                f"## Top Hard Confusions: {split_name}",
                "",
                "| true | predicted | count | hard rate | soft rate | median rel_e_sigma | p95 rel_e_sigma |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["top_hard_confusions"][split_name]:
            lines.append(
                f"| {row['true_branch']} | {row['pred_branch']} | {row['count']} | {row['harmful_rate']:.4f} | "
                f"{row['soft_rate']:.4f} | {row['median_rel_e_sigma']:.4f} | {row['p95_rel_e_sigma']:.4f} |"
            )
    args.report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
