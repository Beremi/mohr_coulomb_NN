from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_surrogate.full_export import split_full_export_call_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the call-level real-data split for cover-layer branch experiments.")
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("constitutive_problem_3D_full.h5"),
        help="Path to the full constitutive export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json"),
        help="Where to write the split JSON.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the call permutation.")
    args = parser.parse_args()

    splits = split_full_export_call_names(args.export, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_hdf5": str(args.export),
        "seed": args.seed,
        "splits": splits,
        "counts": {name: len(value) for name, value in splits.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
