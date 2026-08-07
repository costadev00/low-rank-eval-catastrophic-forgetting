import json
from pathlib import Path

import pandas as pd

from low_rank_eval.analysis.aggregate_results import load_run_results, pareto_front
from low_rank_eval.config import load_config


def _write_run(
    root: Path,
    run_id: str,
    *,
    scientific: bool,
    fingerprint: str,
) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "configuration": "uniform",
        "seed": 42,
        "task_order": ["ifeval", "math"],
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "scientific": scientific,
        "config_fingerprint": fingerprint,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_only_current_scientific_results_are_loaded(tmp_path: Path) -> None:
    fingerprint = load_config("configs/uniform.yaml").fingerprint()
    valid_id = f"uniform-ifeval_to_math-seed42-{fingerprint}"
    _write_run(tmp_path, valid_id, scientific=True, fingerprint=fingerprint)
    _write_run(tmp_path, "smoke-result", scientific=False, fingerprint=fingerprint)
    _write_run(tmp_path, "stale-result", scientific=True, fingerprint="obsolete")

    runs = load_run_results(tmp_path, config_dir="configs")

    assert [run["run_id"] for run in runs] == [valid_id]


def test_pareto_front_compares_only_within_task_order() -> None:
    rows = []
    # Configuration a is dominated by b in IFEval->Math.
    # Its lower absolute scores must not make it dominated by c from the
    # reverse order, whose axes refer to different benchmarks.
    values = {
        ("a", "ifeval_to_math"): {"ifeval": (20.0, 0.0), "gsm8k": (0.0, 30.0)},
        ("b", "ifeval_to_math"): {"ifeval": (21.0, 0.0), "gsm8k": (0.0, 31.0)},
        ("c", "math_to_ifeval"): {"ifeval": (0.0, 90.0), "gsm8k": (90.0, 0.0)},
    }
    for (configuration, order), benchmarks in values.items():
        for benchmark, (stage2_score, plasticity) in benchmarks.items():
            rows.append(
                {
                    "configuration": configuration,
                    "task_order": order,
                    "benchmark": benchmark,
                    "stage2_score": stage2_score,
                    "plasticity": plasticity,
                    "trainable_parameters": 10,
                }
            )

    frontier = pareto_front(pd.DataFrame(rows)).set_index(["configuration", "task_order"])

    assert bool(frontier.loc[("a", "ifeval_to_math"), "pareto"]) is False
    assert bool(frontier.loc[("b", "ifeval_to_math"), "pareto"]) is True
    assert bool(frontier.loc[("c", "math_to_ifeval"), "pareto"]) is True
