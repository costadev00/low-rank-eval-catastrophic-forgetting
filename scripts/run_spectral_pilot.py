#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json

import torch
from accelerate import Accelerator
from datasets import Dataset, load_from_disk

from low_rank_eval.config import load_config
from low_rank_eval.data.prepare import build_token_budget_splits
from low_rank_eval.lora.parameter_budget import (
    block_index,
    parameter_count,
)
from low_rank_eval.lora.spectral_allocator import (
    aggregate_module_energy,
    allocate_spectral_ranks,
    allocation_payload,
)
from low_rank_eval.lora.spectral_analysis import analyze_peft_model, write_spectral_analysis
from low_rank_eval.training.modeling import create_or_load_adapter
from low_rank_eval.training.train_stage import train_stage


def _alternating_rows(left: Dataset, right: Dataset) -> list[dict]:
    rows: list[dict] = []
    maximum = max(len(left), len(right))
    for index in range(maximum):
        if index < len(left):
            rows.append(left[index])
        if index < len(right):
            rows.append(right[index])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    accelerator = Accelerator()
    prepared = build_token_budget_splits(config, force=args.force)
    pilot_config = config.model_copy(
        update={
            "lora": config.lora.model_copy(
                update={
                    "strategy": "uniform",
                    "uniform_rank": config.spectral.pilot_rank,
                    "rank_file": None,
                }
            )
        }
    )
    pilot_data = (
        config.data.processed_dir / config.data_fingerprint() / "spectral_pilot" / "balanced"
    )
    if accelerator.is_main_process and (args.force or not pilot_data.exists()):
        ifeval = load_from_disk(str(prepared["ifeval"]["calibration"]))
        math = load_from_disk(str(prepared["math"]["calibration"]))
        Dataset.from_list(_alternating_rows(ifeval, math)).save_to_disk(str(pilot_data))
    accelerator.wait_for_everyone()
    pilot_dir = config.output.checkpoints_dir / "spectral_pilot" / config.fingerprint()
    train_stage(pilot_config, dataset_path=pilot_data, stage_dir=pilot_dir)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, shapes, _ = create_or_load_adapter(
        pilot_config,
        adapter_path=pilot_dir / "adapter",
        manifest_dir=pilot_dir / "analysis_manifest",
    )
    if accelerator.is_main_process:
        raw = analyze_peft_model(model, config.lora.target_modules)
        run_kind = "full" if config.evaluation.limit is None else "smoke"
        spectral_root = (
            config.output.results_dir / "spectral" / run_kind / config.data_fingerprint()
        )
        write_spectral_analysis(spectral_root / "spectral_analysis.json", raw)
        module_curves: dict[int, dict[str, list[float]]] = {}
        for name, values in raw.items():
            module_curves.setdefault(block_index(name), {})[name] = values["cumulative_energy"]
        candidates = list(
            range(
                config.spectral.r_min,
                config.spectral.r_max + 1,
                config.spectral.rank_step,
            )
        )
        energy = aggregate_module_energy(module_curves, candidates)
        costs = {
            block: sum(shape.cost_per_rank for shape in shapes if shape.block == block)
            for block in sorted({shape.block for shape in shapes})
        }
        reference = parameter_count(
            shapes, {block: config.spectral.baseline_rank for block in costs}
        )
        allocation = allocate_spectral_ranks(
            energy,
            costs,
            reference_parameters=reference,
            r_min=config.spectral.r_min,
            r_max=config.spectral.r_max,
            rank_step=config.spectral.rank_step,
            tolerance=config.lora.budget_tolerance,
            require_non_uniform=config.spectral.require_non_uniform,
        )
        spectral_root.mkdir(parents=True, exist_ok=True)
        with (spectral_root / "rank_allocation.json").open("w", encoding="utf-8") as handle:
            payload = allocation_payload(allocation)
            json.dump(payload, handle, indent=2, sort_keys=True)
        if run_kind == "full":
            canonical = config.output.results_dir / "spectral" / "rank_allocation.json"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            with canonical.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
