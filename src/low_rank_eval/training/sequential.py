from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from low_rank_eval.config import ExperimentConfig, save_resolved_config
from low_rank_eval.data.prepare import build_token_budget_splits
from low_rank_eval.evaluation.evaluator import evaluate_checkpoint
from low_rank_eval.evaluation.metrics import continual_learning_metrics
from low_rank_eval.training.train_stage import train_stage

TASK_ORDERS = {
    "ifeval_to_math": ("ifeval", "math"),
    "math_to_ifeval": ("math", "ifeval"),
}


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _base_cache_key(config: ExperimentConfig) -> str:
    payload = {
        "model": config.model.model_dump(mode="json"),
        "ifeval": config.data.ifeval_eval.model_dump(mode="json"),
        "gsm8k": config.data.gsm8k_eval.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "processed_data": config.data_fingerprint(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:20]


def _evaluate_cached(
    config: ExperimentConfig,
    *,
    adapter_path: Path | None,
    output: Path,
    calibration_paths: dict[str, Path],
) -> dict[str, Any]:
    summary = output / "evaluation_summary.json"
    if config.training.resume and summary.exists():
        if int(os.environ.get("RANK", "0")) == 0:
            with summary.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {}
    result = evaluate_checkpoint(
        config,
        adapter_path=adapter_path,
        output_dir=output,
        calibration_paths=calibration_paths,
    )
    _cleanup()
    return result


def _last_trainer_checkpoint(stage_dir: Path) -> Path:
    candidates = sorted(
        (stage_dir / "trainer").glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No Trainer checkpoint found under {stage_dir}")
    return candidates[-1]


def run_sequential(
    config: ExperimentConfig,
    *,
    task_order: str,
    seed: int | None = None,
) -> dict[str, Any]:
    if task_order not in TASK_ORDERS:
        raise ValueError(f"task_order must be one of {sorted(TASK_ORDERS)}")
    if seed is not None:
        config = config.model_copy(
            update={
                "training": config.training.model_copy(update={"seed": seed}),
                "lora": config.lora.model_copy(update={"random_seed": seed}),
            }
        )
    accelerator = Accelerator()
    prepared = build_token_budget_splits(config)
    order = TASK_ORDERS[task_order]
    run_id = (
        f"{config.lora.strategy}-{task_order}-seed{config.training.seed}-{config.fingerprint()}"
    )
    run_dir = config.output.results_dir / "runs" / run_id
    checkpoint_root = config.output.checkpoints_dir / run_id
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, run_dir / "resolved_config.yaml")
        with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "task_order": list(order),
                    "seed": config.training.seed,
                    "config_fingerprint": config.fingerprint(),
                    "scientific": config.evaluation.limit is None,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    accelerator.wait_for_everyone()
    calibration = {task: paths["calibration"] for task, paths in prepared.items()}
    base_dir = config.output.results_dir / "base_cache" / _base_cache_key(config)
    base = _evaluate_cached(
        config,
        adapter_path=None,
        output=base_dir,
        calibration_paths=calibration,
    )
    stage1_dir = checkpoint_root / "stage_1"
    stage1_training = train_stage(
        config,
        dataset_path=prepared[order[0]]["train"],
        stage_dir=stage1_dir,
    )
    _cleanup()
    stage1_adapter = stage1_dir / "adapter"
    stage1 = _evaluate_cached(
        config,
        adapter_path=stage1_adapter,
        output=run_dir / "stage_1_evaluation",
        calibration_paths=calibration,
    )
    optimizer_checkpoint = None
    if not config.training.reset_optimizer_between_tasks:
        optimizer_checkpoint = _last_trainer_checkpoint(stage1_dir)
    stage2_dir = checkpoint_root / "stage_2"
    stage2_training = train_stage(
        config,
        dataset_path=prepared[order[1]]["train"],
        stage_dir=stage2_dir,
        adapter_path=stage1_adapter,
        optimizer_checkpoint=optimizer_checkpoint,
    )
    _cleanup()
    stage2 = _evaluate_cached(
        config,
        adapter_path=stage2_dir / "adapter",
        output=run_dir / "stage_2_evaluation",
        calibration_paths=calibration,
    )
    payload: dict[str, Any] = {}
    if accelerator.is_main_process:
        with (stage1_dir / "rank_manifest.json").open(encoding="utf-8") as handle:
            rank_manifest = json.load(handle)
        metrics = continual_learning_metrics(
            base=base["benchmarks"],
            stage1=stage1["benchmarks"],
            stage2=stage2["benchmarks"],
            task_order=tuple("gsm8k" if task == "math" else task for task in order),
            utility_lambda=config.utility_lambda,
        )
        payload = {
            "run_id": run_id,
            "model": config.model.name,
            "model_revision": config.model.revision,
            "configuration": config.lora.strategy,
            "seed": config.training.seed,
            "task_order": list(order),
            "train_token_budget_per_task": config.training.train_token_budget_per_task,
            "base": base,
            "stage1": stage1,
            "stage2": stage2,
            "continual_learning": metrics,
            "training": {
                "stage1": stage1_training,
                "stage2": stage2_training,
            },
            "rank_manifest": rank_manifest,
            "stage1_adapter": str(stage1_adapter),
            "stage2_adapter": str(stage2_dir / "adapter"),
        }
        with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    accelerator.wait_for_everyone()
    return payload
