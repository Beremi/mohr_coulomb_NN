from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from mc_surrogate.full_export import family_ip_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-call cover-layer deformation regimes from the full export.")
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_regimes.json"),
    )
    args = parser.parse_args()

    ip_rows = family_ip_rows(args.export, family_name="cover_layer")
    rows = []
    with h5py.File(args.export, "r") as f:
        for call_name, grp in f["calls"].items():
            strain = grp["E"][ip_rows].astype(np.float32)
            stress = grp["S"][ip_rows].astype(np.float32)
            strain_norm = np.linalg.norm(strain, axis=1)
            stress_norm = np.linalg.norm(stress, axis=1)
            rows.append(
                {
                    "call_name": call_name,
                    "strain_norm_mean": float(np.mean(strain_norm)),
                    "strain_norm_p95": float(np.quantile(strain_norm, 0.95)),
                    "strain_norm_p99": float(np.quantile(strain_norm, 0.99)),
                    "strain_norm_max": float(np.max(strain_norm)),
                    "stress_norm_mean": float(np.mean(stress_norm)),
                    "stress_norm_p95": float(np.quantile(stress_norm, 0.95)),
                    "stress_norm_p99": float(np.quantile(stress_norm, 0.99)),
                    "stress_norm_max": float(np.max(stress_norm)),
                }
            )

    payload = {
        "source_hdf5": str(args.export),
        "n_calls": len(rows),
        "calls": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["calls"][:5], indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
