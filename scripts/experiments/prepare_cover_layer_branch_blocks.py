from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mc_surrogate.fe_p2_tetra import positive_corner_volume_mask
from mc_surrogate.full_export import (
    canonicalize_p2_element_states,
    infer_problem_material_family_map,
    iter_family_element_blocks,
)
from mc_surrogate.mohr_coulomb import BRANCH_NAMES


def _pattern_key(branch_row: np.ndarray) -> str:
    row = np.asarray(branch_row)
    return "|".join(BRANCH_NAMES[int(x)] for x in row.tolist())


def build_summary(export_path: Path, *, max_calls: int | None) -> dict[str, object]:
    family_map = infer_problem_material_family_map(export_path)
    _, blocks = iter_family_element_blocks(
        export_path,
        family_name="cover_layer",
        call_names=None if max_calls is None else [f"call_{i:06d}" for i in range(1, max_calls + 1)],
    )
    if not blocks:
        raise RuntimeError("No cover-layer blocks were loaded.")

    branch_counts = Counter()
    pattern_counts = Counter()
    exact_vs_export = []
    valid_ratios = []
    disp_norm = []
    strain_norm = []
    canonical_coord_abs = []
    canonical_disp_abs = []

    total_element_states = 0
    n_elements_per_call = int(blocks[0].element_index.shape[0])

    for block in blocks:
        total_element_states += int(block.element_index.shape[0])
        canonical = canonicalize_p2_element_states(block.local_coords, block.local_displacements)

        branch_counts.update(int(x) for x in block.branch_id.reshape(-1).tolist())
        pattern_counts.update(_pattern_key(row) for row in block.branch_id.tolist())

        diff = block.stress_exact - block.stress_export
        exact_vs_export.append(
            {
                "call_name": block.call_name,
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "max_abs": float(np.max(np.abs(diff))),
            }
        )

        valid = positive_corner_volume_mask(
            block.local_coords,
            block.local_displacements.reshape(block.local_displacements.shape[0], 30),
            min_volume_ratio=0.01,
        )
        valid_ratios.append(float(np.mean(valid)))
        disp_norm.append(np.linalg.norm(block.local_displacements.reshape(block.local_displacements.shape[0], -1), axis=1))
        strain_norm.append(np.linalg.norm(block.strain_eng.reshape(block.strain_eng.shape[0], -1), axis=1))
        canonical_coord_abs.append(np.abs(canonical.local_coords.reshape(canonical.local_coords.shape[0], -1)))
        canonical_disp_abs.append(np.abs(canonical.local_displacements.reshape(canonical.local_displacements.shape[0], -1)))

    disp_norm_all = np.concatenate(disp_norm)
    strain_norm_all = np.concatenate(strain_norm)
    canonical_coord_all = np.concatenate(canonical_coord_abs, axis=0)
    canonical_disp_all = np.concatenate(canonical_disp_abs, axis=0)
    exact_mae = np.mean([x["mae"] for x in exact_vs_export])
    exact_rmse = np.mean([x["rmse"] for x in exact_vs_export])
    exact_max = max(x["max_abs"] for x in exact_vs_export)

    summary = {
        "export_path": str(export_path),
        "family_name": "cover_layer",
        "problem_material_family_map": family_map,
        "n_calls_loaded": len(blocks),
        "n_elements_per_call": n_elements_per_call,
        "n_element_states": total_element_states,
        "n_integration_points_per_element": 11,
        "branch_counts": {BRANCH_NAMES[k]: int(v) for k, v in sorted(branch_counts.items())},
        "top_branch_patterns": [{"pattern": k, "count": int(v)} for k, v in pattern_counts.most_common(10)],
        "exact_vs_export": {
            "mae_mean_over_calls": float(exact_mae),
            "rmse_mean_over_calls": float(exact_rmse),
            "max_abs_over_calls": float(exact_max),
        },
        "valid_deformation_ratio_mean": float(np.mean(valid_ratios)),
        "valid_deformation_ratio_min": float(np.min(valid_ratios)),
        "displacement_norm": {
            "mean": float(np.mean(disp_norm_all)),
            "p95": float(np.quantile(disp_norm_all, 0.95)),
            "p99": float(np.quantile(disp_norm_all, 0.99)),
            "max": float(np.max(disp_norm_all)),
        },
        "strain_stack_norm": {
            "mean": float(np.mean(strain_norm_all)),
            "p95": float(np.quantile(strain_norm_all, 0.95)),
            "p99": float(np.quantile(strain_norm_all, 0.99)),
            "max": float(np.max(strain_norm_all)),
        },
        "canonical_abs_coord": {
            "mean": float(np.mean(canonical_coord_all)),
            "p95": float(np.quantile(canonical_coord_all, 0.95)),
            "max": float(np.max(canonical_coord_all)),
        },
        "canonical_abs_displacement": {
            "mean": float(np.mean(canonical_disp_all)),
            "p95": float(np.quantile(canonical_disp_all, 0.95)),
            "max": float(np.max(canonical_disp_all)),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and summarize cover-layer element blocks from the full export.")
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("constitutive_problem_3D_full.h5"),
        help="Path to the full constitutive export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314"),
        help="Directory for the generated summary.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=4,
        help="Limit the summary run to the first N calls for a fast smoke check. Use 0 for all calls.",
    )
    args = parser.parse_args()

    max_calls = None if args.max_calls == 0 else args.max_calls
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.export, max_calls=max_calls)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
