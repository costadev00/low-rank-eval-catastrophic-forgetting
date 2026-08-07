from __future__ import annotations

import json
import random
from pathlib import Path

from low_rank_eval.lora.parameter_budget import ModuleShape, parameter_count


def contiguous_depth_groups(num_blocks: int) -> tuple[list[int], list[int], list[int]]:
    if num_blocks < 3:
        raise ValueError("At least three transformer blocks are required")
    base, remainder = divmod(num_blocks, 3)
    sizes = [base + (1 if index < remainder else 0) for index in range(3)]
    boundaries = [0, sizes[0], sizes[0] + sizes[1], num_blocks]
    return tuple(list(range(boundaries[index], boundaries[index + 1])) for index in range(3))  # type: ignore[return-value]


def _group_values(strategy: str, low: int, high: int, uniform: int) -> tuple[int, int, int]:
    mapping = {
        "uniform": (uniform, uniform, uniform),
        "early_heavy": (high, low, low),
        "middle_heavy": (low, high, low),
        "late_heavy": (low, low, high),
    }
    if strategy not in mapping:
        raise ValueError(f"Unsupported manual strategy: {strategy}")
    return mapping[strategy]


def _boundary_priority(groups: tuple[list[int], list[int], list[int]]) -> list[int]:
    priority: list[int] = []
    for group in groups:
        center = (group[0] + group[-1]) / 2
        priority.extend(sorted(group, key=lambda block: abs(block - center), reverse=True))
    return list(dict.fromkeys(priority))


def _adjust_to_budget(
    ranks: dict[int, int],
    module_shapes: list[ModuleShape],
    target: int,
    tolerance: float,
    priority: list[int],
) -> dict[int, int]:
    costs = {
        block: sum(shape.cost_per_rank for shape in module_shapes if shape.block == block)
        for block in ranks
    }
    current = parameter_count(module_shapes, ranks)
    maximum_error = target * tolerance
    attempts = 0
    while abs(current - target) > maximum_error:
        direction = 1 if current < target else -1
        candidates = [
            block
            for block in priority
            if ranks[block] + direction > 0
            and abs((current + direction * costs[block]) - target) < abs(current - target)
        ]
        if not candidates:
            break
        block = min(
            candidates,
            key=lambda candidate: abs((current + direction * costs[candidate]) - target),
        )
        ranks[block] += direction
        current += direction * costs[block]
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Rank budget adjustment did not converge")
    if abs(current - target) > maximum_error:
        raise ValueError(
            f"Unable to match reference budget within {tolerance:.2%}: "
            f"actual={current:,}, target={target:,}"
        )
    return ranks


def build_block_ranks(
    module_shapes: list[ModuleShape],
    *,
    strategy: str,
    uniform_rank: int = 16,
    low_rank: int = 8,
    high_rank: int = 32,
    tolerance: float = 0.01,
    random_seed: int = 42,
    rank_file: str | Path | None = None,
) -> dict[int, int]:
    blocks = sorted({shape.block for shape in module_shapes})
    if blocks != list(range(len(blocks))):
        raise ValueError("Transformer blocks must use contiguous zero-based indices")
    if strategy == "spectral":
        if rank_file is None:
            raise ValueError("spectral strategy requires rank_file")
        with Path(rank_file).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = payload.get("block_ranks", payload)
        ranks = {int(key): int(value) for key, value in raw.items()}
        if set(ranks) != set(blocks):
            raise ValueError("Spectral rank file does not cover every block")
        return ranks

    groups = contiguous_depth_groups(len(blocks))
    values = _group_values(
        "early_heavy" if strategy == "random" else strategy,
        low_rank,
        high_rank,
        uniform_rank,
    )
    ranks = {block: value for group, value in zip(groups, values, strict=True) for block in group}
    if strategy == "random":
        values_to_shuffle = list(ranks.values())
        random.Random(random_seed).shuffle(values_to_shuffle)
        ranks = dict(zip(blocks, values_to_shuffle, strict=True))

    reference = parameter_count(module_shapes, {block: uniform_rank for block in blocks})
    return _adjust_to_budget(
        ranks,
        module_shapes,
        reference,
        tolerance,
        _boundary_priority(groups),
    )


def peft_patterns(
    module_shapes: list[ModuleShape], block_ranks: dict[int, int]
) -> tuple[dict[str, int], dict[str, int]]:
    rank_pattern = {shape.name: block_ranks[shape.block] for shape in module_shapes}
    return rank_pattern, dict(rank_pattern)


def verify_applied_ranks(
    peft_model: object,
    expected_rank_pattern: dict[str, int],
    *,
    adapter_name: str = "default",
) -> list[dict[str, int | float | str]]:
    modules = dict(peft_model.named_modules())  # type: ignore[attr-defined]
    records: list[dict[str, int | float | str]] = []
    for expected_name, expected_rank in expected_rank_pattern.items():
        matches = [
            (name, module)
            for name, module in modules.items()
            if name == expected_name or name.endswith(f".{expected_name}")
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected exactly one applied module for {expected_name}, got {len(matches)}"
            )
        applied_name, module = matches[0]
        rank = int(module.r[adapter_name])
        alpha = float(module.lora_alpha[adapter_name])
        scaling = float(module.scaling[adapter_name])
        a_shape = list(module.lora_A[adapter_name].weight.shape)
        b_shape = list(module.lora_B[adapter_name].weight.shape)
        if rank != expected_rank or alpha != expected_rank or scaling != 1.0:
            raise AssertionError(
                f"{applied_name}: rank={rank}, alpha={alpha}, scaling={scaling}, "
                f"expected rank=alpha={expected_rank}, scaling=1"
            )
        if a_shape[0] != rank or b_shape[1] != rank:
            raise AssertionError(f"{applied_name}: inconsistent LoRA factor shapes")
        records.append(
            {
                "module": applied_name,
                "rank": rank,
                "alpha": alpha,
                "scaling": scaling,
                "lora_a_rows": a_shape[0],
                "lora_b_columns": b_shape[1],
            }
        )
    return records
