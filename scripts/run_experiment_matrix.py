#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from low_rank_eval.config import load_config
from low_rank_eval.training.sequential import run_sequential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["uniform", "early_heavy", "middle_heavy", "late_heavy"],
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["ifeval_to_math", "math_to_ifeval"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    for name in args.configs:
        path = config_dir / f"{name}.yaml"
        if name == "spectral" and not path.exists():
            raise FileNotFoundError("Run the spectral pilot before the spectral matrix")
        for seed in args.seeds:
            for order in args.orders:
                run_sequential(load_config(path), task_order=order, seed=seed)


if __name__ == "__main__":
    main()
