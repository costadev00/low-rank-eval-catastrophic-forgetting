#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from low_rank_eval.config import load_config
from low_rank_eval.training.sequential import run_sequential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["ifeval_to_math", "math_to_ifeval"],
    )
    args = parser.parse_args()
    selection_path = Path(args.results_dir) / "aggregates" / "manual_selection.json"
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    selected = selection.get("selected_configuration")
    if not selected or selection.get("ifeval_or_gsm8k_used_for_selection") is not False:
        raise RuntimeError("A complete, benchmark-free manual selection manifest is required")
    for name in ("uniform", selected, "spectral"):
        config = load_config(Path(args.config_dir) / f"{name}.yaml")
        for seed in args.seeds:
            for order in args.orders:
                run_sequential(config, task_order=order, seed=seed)


if __name__ == "__main__":
    main()
