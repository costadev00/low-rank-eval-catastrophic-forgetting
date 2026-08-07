#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from low_rank_eval.config import load_config
from low_rank_eval.data.prepare import build_token_budget_splits
from low_rank_eval.evaluation.evaluator import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    prepared = build_token_budget_splits(config)
    evaluate_checkpoint(
        config,
        adapter_path=Path(args.adapter) if args.adapter else None,
        output_dir=args.output_dir,
        calibration_paths={task: paths["calibration"] for task, paths in prepared.items()},
    )


if __name__ == "__main__":
    main()
