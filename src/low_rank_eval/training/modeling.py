from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from low_rank_eval.config import ExperimentConfig
from low_rank_eval.lora.parameter_budget import (
    discover_target_modules,
    parameter_count,
    trainable_parameter_summary,
    write_budget_manifest,
)
from low_rank_eval.lora.rank_patterns import (
    build_block_ranks,
    peft_patterns,
    verify_applied_ranks,
)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def local_device_map() -> dict[str, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def load_tokenizer(config: ExperimentConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_base(config: ExperimentConfig, *, trainable: bool) -> Any:
    if config.model.attention_backend == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "attention_backend=flash_attention_2 requires the optional flash dependency"
            ) from error
    quantization = None
    if config.model.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.quant_type,
            bnb_4bit_use_double_quant=config.model.double_quant,
            bnb_4bit_compute_dtype=_dtype(config.model.compute_dtype),
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization,
        dtype=_dtype(config.model.compute_dtype),
        attn_implementation=config.model.attention_backend,
        device_map=local_device_map() if config.model.load_in_4bit else None,
    )
    model.config.use_cache = not trainable
    if trainable and config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.model.gradient_checkpointing,
        )
    elif trainable and config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def create_or_load_adapter(
    config: ExperimentConfig,
    *,
    adapter_path: str | Path | None = None,
    manifest_dir: str | Path | None = None,
) -> tuple[Any, Any, dict[str, int]]:
    base = load_quantized_base(config, trainable=True)
    shapes = discover_target_modules(base, config.lora.target_modules)
    ranks = build_block_ranks(
        shapes,
        strategy=config.lora.strategy,
        uniform_rank=config.lora.uniform_rank,
        low_rank=config.lora.low_rank,
        high_rank=config.lora.high_rank,
        tolerance=config.lora.budget_tolerance,
        random_seed=config.lora.random_seed,
        rank_file=config.lora.rank_file,
    )
    rank_pattern, alpha_pattern = peft_patterns(shapes, ranks)
    reference = parameter_count(
        shapes, {block: config.lora.uniform_rank for block in sorted(ranks)}
    )
    if adapter_path is None:
        peft_config = PeftLoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora.uniform_rank,
            lora_alpha=config.lora.uniform_rank,
            target_modules=config.lora.target_modules,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            lora_dropout=config.lora.dropout,
            bias="none",
            use_rslora=False,
        )
        model = get_peft_model(base, peft_config)
    else:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    records = verify_applied_ranks(model, rank_pattern)
    summary = trainable_parameter_summary(
        model, reference_total_parameters=config.model.reference_parameter_count
    )
    instantiated = int(summary["trainable_parameters"])
    if abs(instantiated - reference) > reference * config.lora.budget_tolerance:
        raise AssertionError(
            f"Instantiated trainable parameters {instantiated:,} differ from "
            f"reference {reference:,} by more than {config.lora.budget_tolerance:.2%}"
        )
    if manifest_dir is not None and int(os.environ.get("RANK", "0")) == 0:
        target = Path(manifest_dir)
        target.mkdir(parents=True, exist_ok=True)
        write_budget_manifest(
            target / "rank_manifest.json",
            module_shapes=shapes,
            block_ranks=ranks,
            reference_parameters=reference,
            model_summary=summary,
        )
        with (target / "applied_ranks.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
    return model, shapes, ranks


def load_adapter_for_evaluation(config: ExperimentConfig, adapter_path: str | Path | None) -> Any:
    model = load_quantized_base(config, trainable=False)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model
