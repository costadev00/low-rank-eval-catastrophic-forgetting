from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import Any


@dataclass(frozen=True)
class AllocationResult:
    block_ranks: dict[int, int]
    parameters: int
    reference_parameters: int
    utility: float
    forced_non_uniform: bool


def aggregate_module_energy(
    module_curves: dict[int, dict[str, list[float]]],
    candidate_ranks: list[int],
) -> dict[int, dict[int, float]]:
    aggregated: dict[int, dict[int, float]] = {}
    for block, modules in module_curves.items():
        if not modules:
            raise ValueError(f"Block {block} has no module energy curves")
        aggregated[block] = {}
        for rank in candidate_ranks:
            values = [
                curve[min(rank, len(curve)) - 1] if curve else 0.0 for curve in modules.values()
            ]
            aggregated[block][rank] = sum(values) / len(values)
    return aggregated


def _gcd(values: list[int]) -> int:
    return reduce(math.gcd, values)


def allocate_spectral_ranks(
    block_energy: dict[int, dict[int, float]],
    cost_per_rank: dict[int, int],
    *,
    reference_parameters: int,
    r_min: int = 4,
    r_max: int = 32,
    rank_step: int = 4,
    tolerance: float = 0.01,
    require_non_uniform: bool = True,
) -> AllocationResult:
    blocks = sorted(block_energy)
    if set(blocks) != set(cost_per_rank):
        raise ValueError("Energy and cost maps must cover the same blocks")
    candidates = list(range(r_min, r_max + 1, rank_step))
    divisor = _gcd([cost_per_rank[block] * rank_step for block in blocks])
    scaled_budget = reference_parameters // divisor
    scaled_cost = {
        (block, rank): (cost_per_rank[block] * rank) // divisor
        for block in blocks
        for rank in candidates
    }
    states: dict[int, tuple[float, dict[int, int]]] = {0: (0.0, {})}
    for block in blocks:
        next_states: dict[int, tuple[float, dict[int, int]]] = {}
        for used, (utility, ranks) in states.items():
            for rank in candidates:
                new_used = used + scaled_cost[(block, rank)]
                if new_used > scaled_budget:
                    continue
                new_utility = utility + block_energy[block][rank]
                previous = next_states.get(new_used)
                if previous is None or new_utility > previous[0]:
                    next_states[new_used] = (new_utility, {**ranks, block: rank})
        states = next_states
        if not states:
            raise ValueError("No feasible spectral allocation remains")

    minimum = reference_parameters * (1 - tolerance)
    feasible = [
        (used, utility, ranks)
        for used, (utility, ranks) in states.items()
        if used * divisor >= minimum
    ]
    if not feasible:
        raise ValueError("No spectral allocation meets the parameter-budget tolerance")
    used, utility, ranks = max(feasible, key=lambda item: (item[1], item[0]))
    forced = False
    if require_non_uniform and len(set(ranks.values())) == 1:
        alternatives = [
            (alt_used, alt_utility, alt_ranks)
            for alt_used, (alt_utility, alt_ranks) in states.items()
            if alt_used * divisor >= minimum and len(set(alt_ranks.values())) > 1
        ]
        if not alternatives:
            raise ValueError("The budget admits no non-uniform spectral allocation")
        used, utility, ranks = max(alternatives, key=lambda item: (item[1], item[0]))
        forced = True
    return AllocationResult(
        block_ranks=ranks,
        parameters=used * divisor,
        reference_parameters=reference_parameters,
        utility=utility,
        forced_non_uniform=forced,
    )


def allocation_payload(result: AllocationResult) -> dict[str, Any]:
    return {
        "block_ranks": {str(key): value for key, value in sorted(result.block_ranks.items())},
        "parameters": result.parameters,
        "reference_parameters": result.reference_parameters,
        "difference_parameters": result.parameters - result.reference_parameters,
        "difference_percent": (
            100.0 * (result.parameters - result.reference_parameters) / result.reference_parameters
        ),
        "spectral_utility": result.utility,
        "forced_non_uniform": result.forced_non_uniform,
    }
