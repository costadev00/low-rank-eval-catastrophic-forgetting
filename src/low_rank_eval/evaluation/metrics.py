from __future__ import annotations

from statistics import mean
from typing import Any


def continual_learning_metrics(
    *,
    base: dict[str, float],
    stage1: dict[str, float],
    stage2: dict[str, float],
    task_order: tuple[str, str],
    utility_lambda: float = 1.0,
) -> dict[str, Any]:
    tasks = sorted(base)
    if set(tasks) != set(stage1) or set(tasks) != set(stage2):
        raise ValueError("All stages must contain the same benchmark metrics")
    first, second = task_order
    if set(task_order) != set(tasks):
        raise ValueError("task_order and metric names differ")
    per_task: dict[str, dict[str, float]] = {}
    for task in tasks:
        after_learning = stage1[task] if task == first else stage2[task]
        final = stage2[task]
        forgetting = max(base[task], stage1[task], stage2[task]) - final
        per_task[task] = {
            "base": base[task],
            "stage1": stage1[task],
            "stage2": stage2[task],
            "gain": after_learning - base[task],
            "forgetting": forgetting,
            "net": final - base[task],
            "plasticity": after_learning,
            "bwt": final - after_learning,
        }
    net_mean = mean(item["net"] for item in per_task.values())
    forgetting_mean = mean(item["forgetting"] for item in per_task.values())
    return {
        "tasks": per_task,
        "first_task": first,
        "second_task": second,
        "forgetting_first": stage1[first] - stage2[first],
        "utility": net_mean - utility_lambda * forgetting_mean,
        "utility_lambda": utility_lambda,
    }
