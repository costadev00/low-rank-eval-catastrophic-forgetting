from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def singular_values_from_factors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return singular values of B @ A by decomposing only a rank-by-rank matrix."""
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(f"Incompatible factors B{tuple(b.shape)} and A{tuple(a.shape)}")
    working_a = a.detach().to(device="cpu", dtype=torch.float64)
    working_b = b.detach().to(device="cpu", dtype=torch.float64)
    _, r_b = torch.linalg.qr(working_b, mode="reduced")
    _, r_at = torch.linalg.qr(working_a.T, mode="reduced")
    middle = r_b @ r_at.T
    return torch.linalg.svdvals(middle).to(dtype=torch.float32)


def cumulative_energy(singular_values: torch.Tensor) -> list[float]:
    energy = singular_values.double().square()
    total = float(energy.sum())
    if total == 0.0:
        return [0.0] * singular_values.numel()
    return torch.cumsum(energy, dim=0).div(total).tolist()


def analyze_peft_model(
    model: Any,
    target_modules: list[str],
    *,
    adapter_name: str = "default",
) -> dict[str, dict[str, Any]]:
    suffixes = tuple(f".{target}" for target in target_modules)
    results: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not (name in target_modules or name.endswith(suffixes)):
            continue
        if not hasattr(module, "lora_A") or adapter_name not in module.lora_A:
            continue
        a = module.lora_A[adapter_name].weight
        b = module.lora_B[adapter_name].weight
        values = singular_values_from_factors(a, b)
        results[name] = {
            "singular_values": values.tolist(),
            "cumulative_energy": cumulative_energy(values),
            "rank": len(values),
        }
    if not results:
        raise ValueError("No trained LoRA factors were found")
    return results


def write_spectral_analysis(path: str | Path, results: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
