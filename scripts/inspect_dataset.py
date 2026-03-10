#!/usr/bin/env python
"""Print a compact summary of an HDF5 dataset."""

from __future__ import annotations

import argparse
import json

from mc_surrogate.data import dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to HDF5 dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(dataset_summary(args.dataset), indent=2))


if __name__ == "__main__":
    main()
