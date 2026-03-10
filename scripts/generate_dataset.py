#!/usr/bin/env python
"""Generate an HDF5 dataset of exact Mohr-Coulomb constitutive samples."""

from __future__ import annotations

import argparse
import json

from mc_surrogate.sampling import DatasetGenerationConfig, MaterialRangeConfig, generate_branch_balanced_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--n-samples", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-batch", type=int, default=4096)
    parser.add_argument("--include-tangent", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)

    parser.add_argument("--elastic-frac", type=float, default=0.2)
    parser.add_argument("--smooth-frac", type=float, default=0.2)
    parser.add_argument("--left-frac", type=float, default=0.2)
    parser.add_argument("--right-frac", type=float, default=0.2)
    parser.add_argument("--apex-frac", type=float, default=0.2)

    parser.add_argument("--cohesion-min", type=float, default=2.0)
    parser.add_argument("--cohesion-max", type=float, default=40.0)
    parser.add_argument("--phi-min", type=float, default=20.0)
    parser.add_argument("--phi-max", type=float, default=50.0)
    parser.add_argument("--psi-min", type=float, default=0.0)
    parser.add_argument("--psi-max", type=float, default=15.0)
    parser.add_argument("--young-min", type=float, default=5.0e3)
    parser.add_argument("--young-max", type=float, default=1.0e5)
    parser.add_argument("--nu-min", type=float, default=0.20)
    parser.add_argument("--nu-max", type=float, default=0.45)
    parser.add_argument("--srf-min", type=float, default=0.8)
    parser.add_argument("--srf-max", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    material_ranges = MaterialRangeConfig(
        cohesion_range=(args.cohesion_min, args.cohesion_max),
        friction_deg_range=(args.phi_min, args.phi_max),
        dilatancy_deg_range=(args.psi_min, args.psi_max),
        young_range=(args.young_min, args.young_max),
        poisson_range=(args.nu_min, args.nu_max),
        strength_reduction_range=(args.srf_min, args.srf_max),
    )
    cfg = DatasetGenerationConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        candidate_batch=args.candidate_batch,
        include_tangent=args.include_tangent,
        split_fractions=(args.train_frac, args.val_frac, args.test_frac),
        branch_fractions={
            "elastic": args.elastic_frac,
            "smooth": args.smooth_frac,
            "left_edge": args.left_frac,
            "right_edge": args.right_frac,
            "apex": args.apex_frac,
        },
        material_ranges=material_ranges,
    )
    counts = generate_branch_balanced_dataset(args.output, cfg)
    print(json.dumps({"output": args.output, "branch_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
