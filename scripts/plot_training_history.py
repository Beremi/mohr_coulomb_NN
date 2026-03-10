#!/usr/bin/env python
"""Plot training history from history.csv."""

from __future__ import annotations

import argparse

from mc_surrogate.viz import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, help="Path to history.csv.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_history(args.history, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
