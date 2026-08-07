#!/usr/bin/env python
from __future__ import annotations

import argparse

from low_rank_eval.analysis.aggregate_results import write_aggregates
from low_rank_eval.analysis.plots import plot_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    write_aggregates(args.results_dir)
    plot_all(args.results_dir)


if __name__ == "__main__":
    main()
