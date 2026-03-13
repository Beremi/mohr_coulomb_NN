#!/usr/bin/env python
"""Sample a captured constitutive HDF5 export into the framework dataset format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_surrogate.data import save_dataset_hdf5
from mc_surrogate.real_export import sample_real_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--samples-per-call", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="*", default=None, help="Optional subset of call modes to include.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays, attrs = sample_real_export(
        args.input_h5,
        samples_per_call=args.samples_per_call,
        seed=args.seed,
        include_modes=None if args.modes is None else tuple(args.modes),
        attach_exact_labels=True,
    )
    out_path = save_dataset_hdf5(
        args.output_h5,
        arrays,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        attrs=attrs,
    )
    summary = {
        "output_h5": str(out_path),
        "n_samples": int(arrays["strain_eng"].shape[0]),
        "samples_per_call": int(args.samples_per_call),
        "exact_match_mae": attrs.get("exact_match_mae"),
        "exact_match_rmse": attrs.get("exact_match_rmse"),
        "exact_match_max_abs": attrs.get("exact_match_max_abs"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
