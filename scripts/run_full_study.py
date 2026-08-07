#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_state(step: str, status: str, command: list[str] | None = None) -> None:
    destination = ROOT / "results" / "full_study_state.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "command": command,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(step: str, command: list[str]) -> None:
    _write_state(step, "running", command)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    _write_state(step, "complete", command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete resumable study and build the result-backed paper."
    )
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    accelerate = str(Path(sys.executable).with_name("accelerate"))
    python = sys.executable
    launch = [
        accelerate,
        "launch",
        "--num_processes",
        str(args.num_processes),
        "--mixed_precision",
        "bf16",
    ]
    _run(
        "initial_matrix",
        [
            *launch,
            "scripts/run_experiment_matrix.py",
            "--config_dir",
            args.config_dir,
            "--configs",
            "uniform",
            "early_heavy",
            "middle_heavy",
            "late_heavy",
            "spectral",
            "--orders",
            "ifeval_to_math",
            "math_to_ifeval",
            "--seeds",
            "42",
        ],
    )
    _run(
        "initial_analysis",
        [python, "scripts/analyze_results.py", "--results_dir", args.results_dir],
    )
    _run(
        "confirmatory_matrix",
        [
            *launch,
            "scripts/run_confirmatory.py",
            "--config_dir",
            args.config_dir,
            "--results_dir",
            args.results_dir,
            "--seeds",
            "7",
            "42",
            "123",
        ],
    )
    _run(
        "final_analysis",
        [python, "scripts/analyze_results.py", "--results_dir", args.results_dir],
    )
    _run("paper", [python, "scripts/build_paper.py"])
    _write_state("complete", "complete")


if __name__ == "__main__":
    main()
