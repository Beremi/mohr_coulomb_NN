#!/usr/bin/env python
"""Build grouped mixed-material real panels for the safe hybrid pivot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.branch_geometry import compute_branch_geometry_principal
from mc_surrogate.data import SPLIT_NAMES, SPLIT_TO_ID
from mc_surrogate.full_export import sample_full_export_dataset
from mc_surrogate.models import compute_trial_stress, spectral_decomposition_from_strain
from mc_surrogate.voigt import stress_voigt_to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export", default="constitutive_problem_3D_full.h5")
    parser.add_argument("--output-root", default="experiment_runs/real_sim/hybrid_pivot_20260323")
    parser.add_argument("--report-md", default="docs/hybrid_pivot_exp0_panels_20260323.md")
    parser.add_argument("--samples-per-call", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
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


def _load_all_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays[key] = f[key][:]
        for key, value in f.attrs.items():
            attrs[key] = value.decode() if isinstance(value, bytes) else value
    return arrays, attrs


def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _decode_attr_json(attrs: dict[str, Any], key: str) -> list[str]:
    raw = attrs.get(key)
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, str):
        return list(json.loads(raw))
    return list(raw)


def _rare_branch_mask(
    branch_id: np.ndarray,
    *,
    plastic_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    plastic_val = plastic_mask & val_mask
    plastic_val_count = int(np.sum(plastic_val))
    if plastic_val_count <= 0:
        return np.zeros_like(plastic_mask, dtype=bool), {
            "mode": "none",
            "plastic_val_count": 0,
            "rare_branch_ids": [],
            "rare_branch_names": [],
            "branch_val_counts": {},
        }

    branch_val_counts = np.bincount(branch_id[plastic_val], minlength=5)
    include_all_if_small = plastic_val_count <= 20_000
    rare_cutoff = max(256, int(round(0.05 * plastic_val_count)))
    if include_all_if_small:
        rare_branch_ids = [1, 2, 3, 4]
        mode = "all_plastic_branches_small_dataset"
    else:
        rare_branch_ids = [idx for idx in range(1, 5) if int(branch_val_counts[idx]) <= rare_cutoff]
        mode = "frequency_threshold" if rare_branch_ids else "disabled_large_dataset"

    rare_mask = np.isin(branch_id, rare_branch_ids) if rare_branch_ids else np.zeros_like(plastic_mask, dtype=bool)
    return rare_mask, {
        "mode": mode,
        "plastic_val_count": plastic_val_count,
        "rare_branch_cutoff": rare_cutoff,
        "rare_branch_ids": rare_branch_ids,
        "rare_branch_names": [name for idx, name in enumerate(("elastic", "smooth", "left_edge", "right_edge", "apex")) if idx in rare_branch_ids],
        "branch_val_counts": {
            "elastic": int(branch_val_counts[0]),
            "smooth": int(branch_val_counts[1]),
            "left_edge": int(branch_val_counts[2]),
            "right_edge": int(branch_val_counts[3]),
            "apex": int(branch_val_counts[4]),
        },
    }


def _compute_panel_arrays(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    strain_eng = arrays["strain_eng"].astype(np.float32)
    material = arrays["material_reduced"].astype(np.float32)
    branch_id = arrays["branch_id"].astype(np.int64)
    split_id = arrays["split_id"].astype(np.int8)
    tangent = arrays.get("tangent")

    strain_principal, _ = spectral_decomposition_from_strain(strain_eng)
    geom = compute_branch_geometry_principal(
        strain_principal,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    trial_stress = compute_trial_stress(strain_eng, material)
    trial_principal = np.linalg.eigvalsh(stress_voigt_to_tensor(trial_stress))[:, ::-1].astype(np.float32)
    trial_mag = np.linalg.norm(trial_principal, axis=1) / np.maximum(material[:, 0], 1.0)

    plastic = branch_id > 0
    val_mask = split_id == SPLIT_TO_ID["val"]
    test_mask = split_id == SPLIT_TO_ID["test"]
    plastic_val = plastic & val_mask

    abs_m_yield = np.abs(geom.m_yield)
    abs_m_smooth_left = np.abs(geom.m_smooth_left)
    abs_m_smooth_right = np.abs(geom.m_smooth_right)
    abs_m_left_apex = np.abs(geom.m_left_apex)
    abs_m_right_apex = np.abs(geom.m_right_apex)

    thresholds = {
        "near_yield_abs": _quantile_or_nan(abs_m_yield[plastic_val], 0.15),
        "near_smooth_left_abs": _quantile_or_nan(abs_m_smooth_left[plastic_val], 0.15),
        "near_smooth_right_abs": _quantile_or_nan(abs_m_smooth_right[plastic_val], 0.15),
        "near_left_apex_abs": _quantile_or_nan(abs_m_left_apex[plastic_val], 0.15),
        "near_right_apex_abs": _quantile_or_nan(abs_m_right_apex[plastic_val], 0.15),
        "repeated_gap_norm": _quantile_or_nan(geom.gap_min_norm[plastic_val], 0.15),
        "tail_trial_mag": _quantile_or_nan(trial_mag[plastic_val], 0.85),
    }

    near_yield = plastic & (abs_m_yield <= thresholds["near_yield_abs"])
    near_smooth_left = plastic & (abs_m_smooth_left <= thresholds["near_smooth_left_abs"])
    near_smooth_right = plastic & (abs_m_smooth_right <= thresholds["near_smooth_right_abs"])
    near_left_apex = plastic & (abs_m_left_apex <= thresholds["near_left_apex_abs"])
    near_right_apex = plastic & (abs_m_right_apex <= thresholds["near_right_apex_abs"])
    repeated_gap = plastic & (geom.gap_min_norm <= thresholds["repeated_gap_norm"])
    tail = plastic & (trial_mag >= thresholds["tail_trial_mag"])
    rare_branch, rare_branch_summary = _rare_branch_mask(
        branch_id,
        plastic_mask=plastic,
        val_mask=val_mask,
    )
    hard = near_yield | near_smooth_left | near_smooth_right | near_left_apex | near_right_apex | repeated_gap | tail | rare_branch

    ds_valid = np.zeros(strain_eng.shape[0], dtype=bool)
    if tangent is not None:
        tangent_reshaped = tangent.astype(np.float32).reshape(-1, 6, 6)
        ds_valid = np.isfinite(tangent_reshaped).all(axis=(1, 2))
        ds_valid &= np.linalg.norm(tangent_reshaped.reshape(tangent_reshaped.shape[0], -1), axis=1) > 0.0

    panel_arrays = {
        "split_id": split_id,
        "source_call_id": arrays["source_call_id"].astype(np.int32),
        "source_row_in_call": arrays["source_row_in_call"].astype(np.int32),
        "branch_id": branch_id.astype(np.int8),
        "plastic_mask": plastic.astype(np.int8),
        "broad_val_mask": val_mask.astype(np.int8),
        "broad_test_mask": test_mask.astype(np.int8),
        "hard_mask": hard.astype(np.int8),
        "hard_val_mask": (hard & val_mask).astype(np.int8),
        "hard_test_mask": (hard & test_mask).astype(np.int8),
        "near_yield_mask": near_yield.astype(np.int8),
        "near_smooth_left_mask": near_smooth_left.astype(np.int8),
        "near_smooth_right_mask": near_smooth_right.astype(np.int8),
        "near_left_apex_mask": near_left_apex.astype(np.int8),
        "near_right_apex_mask": near_right_apex.astype(np.int8),
        "repeated_gap_mask": repeated_gap.astype(np.int8),
        "tail_mask": tail.astype(np.int8),
        "rare_branch_mask": rare_branch.astype(np.int8),
        "ds_valid_mask": ds_valid.astype(np.int8),
        "m_yield": geom.m_yield.astype(np.float32),
        "m_smooth_left": geom.m_smooth_left.astype(np.float32),
        "m_smooth_right": geom.m_smooth_right.astype(np.float32),
        "m_left_apex": geom.m_left_apex.astype(np.float32),
        "m_right_apex": geom.m_right_apex.astype(np.float32),
        "gap12_norm": geom.gap12_norm.astype(np.float32),
        "gap23_norm": geom.gap23_norm.astype(np.float32),
        "gap_min_norm": geom.gap_min_norm.astype(np.float32),
        "d_geom": geom.d_geom.astype(np.float32),
        "trial_mag": trial_mag.astype(np.float32),
    }

    split_names = {idx: name for name, idx in SPLIT_TO_ID.items()}
    summary = {
        "n_samples": int(strain_eng.shape[0]),
        "split_counts": {
            split_names[idx]: int(np.sum(split_id == idx))
            for idx in range(len(SPLIT_NAMES))
        },
        "panel_counts": {
            "broad_val": int(np.sum(val_mask)),
            "broad_test": int(np.sum(test_mask)),
            "hard_val": int(np.sum(hard & val_mask)),
            "hard_test": int(np.sum(hard & test_mask)),
            "ds_valid": int(np.sum(ds_valid)),
            "ds_valid_val": int(np.sum(ds_valid & val_mask)),
            "ds_valid_test": int(np.sum(ds_valid & test_mask)),
        },
        "component_counts": {
            "near_yield_val": int(np.sum(near_yield & val_mask)),
            "near_smooth_left_val": int(np.sum(near_smooth_left & val_mask)),
            "near_smooth_right_val": int(np.sum(near_smooth_right & val_mask)),
            "near_left_apex_val": int(np.sum(near_left_apex & val_mask)),
            "near_right_apex_val": int(np.sum(near_right_apex & val_mask)),
            "repeated_gap_val": int(np.sum(repeated_gap & val_mask)),
            "tail_val": int(np.sum(tail & val_mask)),
            "rare_branch_val": int(np.sum(rare_branch & val_mask)),
            "near_yield_test": int(np.sum(near_yield & test_mask)),
            "near_smooth_left_test": int(np.sum(near_smooth_left & test_mask)),
            "near_smooth_right_test": int(np.sum(near_smooth_right & test_mask)),
            "near_left_apex_test": int(np.sum(near_left_apex & test_mask)),
            "near_right_apex_test": int(np.sum(near_right_apex & test_mask)),
            "repeated_gap_test": int(np.sum(repeated_gap & test_mask)),
            "tail_test": int(np.sum(tail & test_mask)),
            "rare_branch_test": int(np.sum(rare_branch & test_mask)),
        },
        "thresholds": thresholds,
        "rare_branch_summary": rare_branch_summary,
        "d_geom_quantiles": {
            "plastic_val_q05": _quantile_or_nan(geom.d_geom[plastic_val], 0.05),
            "plastic_val_q50": _quantile_or_nan(geom.d_geom[plastic_val], 0.50),
            "plastic_val_q95": _quantile_or_nan(geom.d_geom[plastic_val], 0.95),
        },
    }
    return panel_arrays, summary


def _write_panel_sidecar(path: Path, panel_arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in panel_arrays.items():
            f.create_dataset(key, data=np.asarray(value), compression="gzip", shuffle=True)
        f.attrs["summary_json"] = json.dumps(_json_safe(summary))


def _write_report(
    report_path: Path,
    dataset_path: Path,
    panel_path: Path,
    call_split_path: Path,
    summary: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Hybrid Pivot Experiment 0: Real Panels")
    lines.append("")
    lines.append("This report freezes the grouped mixed-material real dataset and the validation/test panel sidecars used for the safe hybrid pivot.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- grouped dataset: `{dataset_path}`")
    lines.append(f"- panel sidecar: `{panel_path}`")
    lines.append(f"- call split json: `{call_split_path}`")
    lines.append("")
    lines.append("## Split Counts")
    lines.append("")
    lines.append("| Split | Rows |")
    lines.append("|---|---:|")
    for split_name, count in summary["split_counts"].items():
        lines.append(f"| {split_name} | {count} |")
    lines.append("")
    lines.append("## Panel Counts")
    lines.append("")
    lines.append("| Panel | Rows |")
    lines.append("|---|---:|")
    for name, count in summary["panel_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append("| Quantity | Threshold |")
    lines.append("|---|---:|")
    for name, value in summary["thresholds"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.append("")
    lines.append("## Split Policy")
    lines.append("")
    lines.append(f"- split seed: `{summary['dataset_attrs'].get('split_seed', 'unknown')}`")
    lines.append(f"- rare-branch policy: `{summary['rare_branch_summary']['mode']}`")
    lines.append(f"- rare branches included in hard panel: `{summary['rare_branch_summary']['rare_branch_names']}`")
    lines.append("")
    lines.append("## Hard-Panel Component Counts")
    lines.append("")
    lines.append("| Component | Rows |")
    lines.append("|---|---:|")
    for name, count in summary["component_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Thresholds were computed on plastic validation rows only and then applied unchanged to the test split.")
    lines.append("- Rare-branch inclusion is disabled automatically on large validation populations unless a plastic branch is genuinely low-frequency.")
    lines.append("- `ds_valid` marks rows with finite, nonzero exported `DS`; the stricter interior tangent subset will be derived later after choosing the fallback threshold.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_hybrid_real_panels(
    *,
    full_export: Path,
    output_root: Path,
    report_path: Path,
    samples_per_call: int,
    seed: int,
    split_fractions: tuple[float, float, float],
) -> dict[str, Any]:
    dataset_path = output_root / f"real_grouped_sampled_{samples_per_call}.h5"
    if not dataset_path.exists():
        sample_full_export_dataset(
            full_export,
            dataset_path,
            samples_per_call=samples_per_call,
            split_fractions=split_fractions,
            seed=seed,
            use_exact_stress=True,
            include_tangent=True,
        )

    arrays, attrs = _load_all_arrays(dataset_path)
    panel_arrays, summary = _compute_panel_arrays(arrays)
    summary["dataset_attrs"] = attrs

    panel_path = output_root / "panels" / "panel_sidecar.h5"
    summary_path = output_root / "panels" / "panel_summary.json"
    call_split_path = output_root / "panels" / "call_splits.json"
    call_split = {
        "source_call_names": _decode_attr_json(attrs, "source_call_names_json"),
        "train_call_names": _decode_attr_json(attrs, "train_call_names_json"),
        "val_call_names": _decode_attr_json(attrs, "val_call_names_json"),
        "test_call_names": _decode_attr_json(attrs, "test_call_names_json"),
        "split_seed": attrs.get("split_seed"),
        "samples_per_call": attrs.get("samples_per_call"),
    }
    _write_panel_sidecar(panel_path, panel_arrays, summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    call_split_path.write_text(json.dumps(_json_safe(call_split), indent=2), encoding="utf-8")
    _write_report(report_path, dataset_path, panel_path, call_split_path, summary)

    return {
        "dataset_path": str(dataset_path),
        "panel_path": str(panel_path),
        "summary_path": str(summary_path),
        "call_split_path": str(call_split_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    result = build_hybrid_real_panels(
        full_export=(ROOT / args.full_export).resolve(),
        output_root=(ROOT / args.output_root).resolve(),
        report_path=(ROOT / args.report_md).resolve(),
        samples_per_call=args.samples_per_call,
        seed=args.seed,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
