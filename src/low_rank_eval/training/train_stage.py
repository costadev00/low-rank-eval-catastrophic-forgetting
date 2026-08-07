from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from low_rank_eval.config import ExperimentConfig
from low_rank_eval.training.modeling import create_or_load_adapter, load_tokenizer


def _training_dataset(path: str | Path) -> Any:
    dataset = load_from_disk(str(path))
    keep = {"input_ids", "attention_mask", "labels"}
    remove = [column for column in dataset.column_names if column not in keep]
    return dataset.remove_columns(remove) if remove else dataset


def _load_optimizer_ablation(trainer: SFTTrainer, checkpoint: Path) -> None:
    trainer.create_optimizer_and_scheduler(num_training_steps=trainer.state.max_steps)
    optimizer_path = checkpoint / "optimizer.pt"
    scheduler_path = checkpoint / "scheduler.pt"
    if not optimizer_path.exists() or not scheduler_path.exists():
        raise FileNotFoundError(
            "Optimizer-preservation ablation requires optimizer.pt and scheduler.pt"
        )
    trainer.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    trainer.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))


def train_stage(
    config: ExperimentConfig,
    *,
    dataset_path: str | Path,
    stage_dir: str | Path,
    adapter_path: str | Path | None = None,
    optimizer_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(stage_dir)
    output.mkdir(parents=True, exist_ok=True)
    done = output / "stage_complete.json"
    if config.training.resume and done.exists():
        with done.open(encoding="utf-8") as handle:
            return json.load(handle)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, _, _ = create_or_load_adapter(config, adapter_path=adapter_path, manifest_dir=output)
    tokenizer = load_tokenizer(config)
    dataset = _training_dataset(dataset_path)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation = config.training.accumulation_for_world_size(world_size)
    args = SFTConfig(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optimizer,
        bf16=config.model.compute_dtype == "bfloat16",
        fp16=config.model.compute_dtype == "float16",
        gradient_checkpointing=config.model.gradient_checkpointing,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config.training.max_sequence_length,
        packing=False,
        seed=config.training.seed,
        data_seed=config.training.seed,
        ddp_find_unused_parameters=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    if optimizer_checkpoint is not None:
        _load_optimizer_ablation(trainer, Path(optimizer_checkpoint))
    resume_checkpoint = None
    trainer_dir = output / "trainer"
    if config.training.resume and trainer_dir.exists() and optimizer_checkpoint is None:
        resume_checkpoint = get_last_checkpoint(str(trainer_dir))
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    adapter_output = output / "adapter"
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(str(adapter_output), safe_serialization=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    trainer.save_state()
    elapsed = time.perf_counter() - started
    local_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    peaks = [local_peak]
    if dist.is_available() and dist.is_initialized():
        gathered: list[int | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_peak)
        peaks = [int(value or 0) for value in gathered]
    payload = {
        "status": "complete",
        "adapter_path": str(adapter_output),
        "train_metrics": train_result.metrics,
        "elapsed_seconds": elapsed,
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "peak_memory_bytes_local": local_peak,
        "peak_memory_bytes_by_rank": peaks,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
    }
    if trainer.is_world_process_zero():
        with done.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    return payload
