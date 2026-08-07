from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

MANUAL_CONFIGS = {"early_heavy", "middle_heavy", "late_heavy"}


def load_run_results(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    results = []
    for path in sorted((root / "runs").glob("*/result.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        results.append(payload)
    return results


def _flatten_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    base = run["base"]["benchmarks"]
    stage1 = run["stage1"]["benchmarks"]
    stage2 = run["stage2"]["benchmarks"]
    continual = run["continual_learning"]["tasks"]
    rank_manifest = run["rank_manifest"]
    elapsed = (
        run["training"]["stage1"]["elapsed_seconds"] + run["training"]["stage2"]["elapsed_seconds"]
    )
    peaks = [
        *run["training"]["stage1"].get("peak_memory_bytes_by_rank", []),
        *run["training"]["stage2"].get("peak_memory_bytes_by_rank", []),
    ]
    rows = []
    for benchmark in sorted(base):
        task_metrics = continual[benchmark]
        rows.append(
            {
                "run_id": run["run_id"],
                "configuration": run["configuration"],
                "model": run["model"],
                "model_revision": run["model_revision"],
                "seed": run["seed"],
                "task_order": "_to_".join(run["task_order"]),
                "benchmark": benchmark,
                "rank_by_layer": json.dumps(rank_manifest["block_ranks"], sort_keys=True),
                "trainable_parameters": rank_manifest["lora_parameters"],
                "trainable_percent": rank_manifest["instantiated_model"]["trainable_percent"],
                "train_token_budget_per_task": run["train_token_budget_per_task"],
                "training_time_seconds": elapsed,
                "peak_memory_bytes": max(peaks, default=0),
                "base_score": base[benchmark],
                "stage1_score": stage1[benchmark],
                "stage2_score": stage2[benchmark],
                "gain": task_metrics["gain"],
                "forgetting": task_metrics["forgetting"],
                "bwt": task_metrics["bwt"],
                "net": task_metrics["net"],
                "plasticity": task_metrics["plasticity"],
                "utility": run["continual_learning"]["utility"],
                "result_path": run["_path"],
            }
        )
    return rows


def aggregate_runs(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = load_run_results(results_dir)
    rows = [row for run in runs for row in _flatten_run(run)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, frame
    statistics = []
    group_columns = ["configuration", "task_order", "benchmark"]
    measures = ["stage2_score", "gain", "forgetting", "bwt", "net", "plasticity"]
    for keys, group in frame.groupby(group_columns, sort=True):
        for measure in measures:
            values = group[measure].astype(float)
            count = len(values)
            average = float(values.mean())
            std = float(values.std(ddof=1)) if count > 1 else math.nan
            if count > 1:
                margin = float(stats.t.ppf(0.975, count - 1) * std / math.sqrt(count))
                lower, upper = average - margin, average + margin
            else:
                lower = upper = math.nan
            statistics.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "measure": measure,
                    "n": count,
                    "mean": average,
                    "std": std,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    return frame, pd.DataFrame(statistics)


def select_best_manual(runs: list[dict[str, Any]], utility_lambda: float = 1.0) -> dict[str, Any]:
    candidates: dict[str, list[float]] = {}
    evidence: list[dict[str, Any]] = []
    for run in runs:
        if run["configuration"] not in MANUAL_CONFIGS or run["seed"] != 42:
            continue
        first, second = run["task_order"]
        base = run["base"]["calibration_nll"]
        stage1 = run["stage1"]["calibration_nll"]
        stage2 = run["stage2"]["calibration_nll"]
        nets = {task: 100.0 * (base[task] - stage2[task]) / base[task] for task in base}
        first_after = stage1[first]
        forgetting = {
            first: max(0.0, 100.0 * (stage2[first] - first_after) / first_after),
            second: 0.0,
        }
        utility = sum(nets.values()) / len(nets) - utility_lambda * (
            sum(forgetting.values()) / len(forgetting)
        )
        candidates.setdefault(run["configuration"], []).append(utility)
        evidence.append(
            {
                "run_id": run["run_id"],
                "configuration": run["configuration"],
                "task_order": run["task_order"],
                "calibration_net_percent": nets,
                "calibration_forgetting_percent": forgetting,
                "calibration_utility": utility,
            }
        )
    complete = {name: values for name, values in candidates.items() if len(values) == 2}
    if set(complete) != MANUAL_CONFIGS:
        missing = MANUAL_CONFIGS - set(complete)
        raise RuntimeError(
            f"Manual selection needs seed-42 results for both orders: missing {missing}"
        )
    averages = {name: sum(values) / len(values) for name, values in complete.items()}
    selected = max(averages, key=averages.get)
    return {
        "selected_configuration": selected,
        "mean_calibration_utility_by_configuration": averages,
        "utility_lambda": utility_lambda,
        "selection_data": "decontaminated training-derived calibration splits",
        "benchmark_fields_used": [],
        "ifeval_or_gsm8k_used_for_selection": False,
        "evidence": evidence,
    }


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    points = []
    if frame.empty:
        return frame
    for (configuration, order), group in frame.groupby(["configuration", "task_order"]):
        first, second = order.split("_to_")
        first = "gsm8k" if first == "math" else first
        second = "gsm8k" if second == "math" else second
        by_benchmark = group.set_index("benchmark")
        if first not in by_benchmark.index or second not in by_benchmark.index:
            continue
        points.append(
            {
                "configuration": configuration,
                "task_order": order,
                "new_task_plasticity": float(by_benchmark.loc[second, "plasticity"]),
                "previous_task_retention": float(by_benchmark.loc[first, "stage2_score"]),
                "trainable_parameters": int(by_benchmark["trainable_parameters"].iloc[0]),
            }
        )
    points_frame = pd.DataFrame(points)
    is_pareto = []
    for _, point in points_frame.iterrows():
        dominated = False
        for _, other in points_frame.iterrows():
            no_worse = (
                other.new_task_plasticity >= point.new_task_plasticity
                and other.previous_task_retention >= point.previous_task_retention
                and other.trainable_parameters <= point.trainable_parameters
            )
            strictly_better = (
                other.new_task_plasticity > point.new_task_plasticity
                or other.previous_task_retention > point.previous_task_retention
                or other.trainable_parameters < point.trainable_parameters
            )
            if no_worse and strictly_better:
                dominated = True
                break
        is_pareto.append(not dominated)
    points_frame["pareto"] = is_pareto
    return points_frame


def write_aggregates(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(results_dir)
    output = root / "aggregates"
    output.mkdir(parents=True, exist_ok=True)
    frame, statistics = aggregate_runs(root)
    if frame.empty:
        raise RuntimeError(f"No completed run results found under {root / 'runs'}")
    frame.to_csv(output / "all_results.csv", index=False)
    statistics.to_csv(output / "multi_seed_statistics.csv", index=False)
    pareto_front(frame).to_csv(output / "pareto_front.csv", index=False)
    with (output / "all_results.json").open("w", encoding="utf-8") as handle:
        json.dump(frame.to_dict(orient="records"), handle, indent=2)
    runs = load_run_results(root)
    try:
        selection = select_best_manual(runs)
    except RuntimeError as error:
        selection = {"status": "incomplete", "reason": str(error)}
    with (output / "manual_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
    return frame, statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    write_aggregates(args.results_dir)


if __name__ == "__main__":
    main()
