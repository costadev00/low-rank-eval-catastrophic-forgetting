from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BLOCK_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ModuleShape:
    name: str
    block: int
    in_features: int
    out_features: int

    @property
    def cost_per_rank(self) -> int:
        return self.in_features + self.out_features


def block_index(module_name: str) -> int:
    match = _BLOCK_RE.search(module_name)
    if not match:
        raise ValueError(f"Cannot infer transformer block from module {module_name!r}")
    return int(match.group(1))


def discover_target_modules(model: Any, targets: list[str]) -> list[ModuleShape]:
    discovered: list[ModuleShape] = []
    suffixes = tuple(f".{target}" for target in targets)
    for name, module in model.named_modules():
        if not (name in targets or name.endswith(suffixes)):
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            weight = getattr(module, "weight", None)
            if weight is None or len(weight.shape) != 2:
                raise TypeError(f"Target {name} is not a supported linear module")
            out_features, in_features = map(int, weight.shape)
        discovered.append(
            ModuleShape(
                name=name,
                block=block_index(name),
                in_features=int(in_features),
                out_features=int(out_features),
            )
        )
    if not discovered:
        raise ValueError(f"No target modules found for {targets}")
    return sorted(discovered, key=lambda item: (item.block, item.name))


def parameter_count(module_shapes: list[ModuleShape], block_ranks: dict[int, int]) -> int:
    return sum(shape.cost_per_rank * block_ranks[shape.block] for shape in module_shapes)


def per_module_parameter_count(
    module_shapes: list[ModuleShape], block_ranks: dict[int, int]
) -> dict[str, int]:
    return {shape.name: shape.cost_per_rank * block_ranks[shape.block] for shape in module_shapes}


def trainable_parameter_summary(
    model: Any, *, reference_total_parameters: int | None = None
) -> dict[str, float | int]:
    instantiated_total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = reference_total_parameters or instantiated_total
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "instantiated_quantized_numel": instantiated_total,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }


def write_budget_manifest(
    path: str | Path,
    *,
    module_shapes: list[ModuleShape],
    block_ranks: dict[int, int],
    reference_parameters: int,
    model_summary: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    actual = parameter_count(module_shapes, block_ranks)
    payload: dict[str, Any] = {
        "block_ranks": {str(key): value for key, value in sorted(block_ranks.items())},
        "module_ranks": {shape.name: block_ranks[shape.block] for shape in module_shapes},
        "module_shapes": [asdict(shape) for shape in module_shapes],
        "module_parameters": per_module_parameter_count(module_shapes, block_ranks),
        "lora_parameters": actual,
        "reference_parameters": reference_parameters,
        "difference_parameters": actual - reference_parameters,
        "difference_percent": (
            100.0 * (actual - reference_parameters) / reference_parameters
            if reference_parameters
            else 0.0
        ),
    }
    if model_summary:
        payload["instantiated_model"] = model_summary
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload
