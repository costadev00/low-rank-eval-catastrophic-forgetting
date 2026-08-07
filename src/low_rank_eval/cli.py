from __future__ import annotations

import argparse

import torch.distributed as dist

from low_rank_eval.config import load_config
from low_rank_eval.data.prepare import build_token_budget_splits
from low_rank_eval.training.sequential import run_sequential


def prepare_data_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = build_token_budget_splits(load_config(args.config), force=args.force)
    print(outputs)


def run_sequential_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task_order",
        choices=["ifeval_to_math", "math_to_ifeval"],
        required=True,
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        run_sequential(load_config(args.config), task_order=args.task_order, seed=args.seed)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def analyze_results_main() -> None:
    from low_rank_eval.analysis.aggregate_results import main

    main()
