#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from low_rank_eval.analysis.paper_artifacts import build_first_paper_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the audited eight-protocol, seed-42 paper artifacts."
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--paper_dir", default="paper")
    args = parser.parse_args()
    snapshot = build_first_paper_artifacts(
        results_dir=args.results_dir,
        config_dir=args.config_dir,
        paper_dir=args.paper_dir,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "validated_protocols": snapshot["validated_protocols"],
                "paper_dir": args.paper_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
