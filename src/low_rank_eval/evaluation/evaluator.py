from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq

from low_rank_eval.config import ExperimentConfig
from low_rank_eval.evaluation.gsm8k import gsm8k_exact_match
from low_rank_eval.training.modeling import load_adapter_for_evaluation, load_tokenizer


def _ifeval_score(document: dict[str, Any], response: str) -> dict[str, Any]:
    try:
        from lm_eval.tasks.ifeval.utils import process_results
    except ImportError as error:
        raise RuntimeError("lm-eval with the official IFEval task is required") from error
    return process_results(document, [response])


def assert_lm_eval_compatibility() -> None:
    from importlib.metadata import version

    installed = version("lm-eval")
    if installed != "0.4.12":
        raise RuntimeError(f"Expected lm-eval==0.4.12, found {installed}")
    sample = {
        "key": 0,
        "prompt": "Write at least one sentence.",
        "instruction_id_list": ["length_constraints:number_sentences"],
        "kwargs": [{"num_sentences": 1, "relation": "at least"}],
    }
    result = _ifeval_score(sample, "This is one sentence.")
    required = {"prompt_level_strict_acc", "inst_level_strict_acc"}
    if not required <= set(result):
        raise RuntimeError(f"IFEval compatibility probe missing {required - set(result)}")


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = generated[:, encoded["input_ids"].shape[1] :]
        return tokenizer.batch_decode(completions, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = previous_padding_side


def _dataset(config: ExperimentConfig, benchmark: str) -> Any:
    ref = config.data.ifeval_eval if benchmark == "ifeval" else config.data.gsm8k_eval
    dataset = load_dataset(
        ref.name,
        ref.config,
        split=ref.split,
        revision=ref.revision,
    )
    if config.evaluation.limit is not None:
        dataset = dataset.select(range(min(config.evaluation.limit, len(dataset))))
    return dataset


def evaluate_checkpoint(
    config: ExperimentConfig,
    *,
    adapter_path: str | Path | None,
    output_dir: str | Path,
    calibration_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    accelerator = Accelerator()
    assert_lm_eval_compatibility()
    model = load_adapter_for_evaluation(config, adapter_path)
    tokenizer = load_tokenizer(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    for benchmark in ("ifeval", "gsm8k"):
        dataset = _dataset(config, benchmark)
        rank_rows: list[dict[str, Any]] = []
        indices = list(range(accelerator.process_index, len(dataset), accelerator.num_processes))
        progress = tqdm(
            total=len(indices),
            disable=not accelerator.is_local_main_process,
            desc=f"{benchmark} rank {accelerator.process_index}",
        )
        for start in range(0, len(indices), config.evaluation.batch_size):
            batch_indices = indices[start : start + config.evaluation.batch_size]
            documents = [dataset[index] for index in batch_indices]
            prompts = [
                str(document["prompt"] if benchmark == "ifeval" else document["question"])
                for document in documents
            ]
            responses = _generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=(
                    config.evaluation.ifeval_max_new_tokens
                    if benchmark == "ifeval"
                    else config.evaluation.gsm8k_max_new_tokens
                ),
            )
            for index, document, prompt, response in zip(
                batch_indices, documents, prompts, responses, strict=True
            ):
                if benchmark == "ifeval":
                    score = _ifeval_score(document, response)
                    record = {
                        "index": index,
                        "prompt_hash": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                        "response": response,
                        **score,
                    }
                else:
                    correct, extracted, gold = gsm8k_exact_match(response, document["answer"])
                    record = {
                        "index": index,
                        "prompt_hash": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                        "response": response,
                        "extracted_answer": extracted,
                        "gold_answer": gold,
                        "exact_match": correct,
                    }
                rank_rows.append(record)
            progress.update(len(batch_indices))
        progress.close()
        rank_file = output / f"{benchmark}.rank_{accelerator.process_index}.jsonl"
        with rank_file.open("w", encoding="utf-8") as handle:
            for row in rank_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            all_rows = []
            for rank in range(accelerator.num_processes):
                with (output / f"{benchmark}.rank_{rank}.jsonl").open(encoding="utf-8") as handle:
                    all_rows.extend(json.loads(line) for line in handle)
            all_rows.sort(key=lambda row: row["index"])
            metric_name = "prompt_level_strict_acc" if benchmark == "ifeval" else "exact_match"
            scores[benchmark] = (
                100.0 * sum(bool(row[metric_name]) for row in all_rows) / len(all_rows)
            )
            with (output / f"{benchmark}.metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "benchmark": benchmark,
                        "metric": metric_name,
                        "score": scores[benchmark],
                        "scale": "0-100",
                        "examples": len(all_rows),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
        accelerator.wait_for_everyone()
    calibration_nll: dict[str, float] = {}
    if calibration_paths:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        for task, dataset_path in calibration_paths.items():
            from datasets import load_from_disk

            dataset = load_from_disk(str(dataset_path))
            keep = {"input_ids", "attention_mask", "labels"}
            remove = [column for column in dataset.column_names if column not in keep]
            if remove:
                dataset = dataset.remove_columns(remove)
            dataset = dataset.shard(
                num_shards=accelerator.num_processes,
                index=accelerator.process_index,
                contiguous=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.evaluation.calibration_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            nll_sum = 0.0
            token_count = 0
            for batch in loader:
                batch = {key: value.to(accelerator.device) for key, value in batch.items()}
                count = int((batch["labels"] != -100).sum())
                with torch.inference_mode():
                    loss = model(**batch).loss
                loss_value = float(loss)
                nll_sum += loss_value * count
                token_count += count
                del loss, batch
            totals = torch.tensor(
                [nll_sum, float(token_count)],
                dtype=torch.float64,
                device=accelerator.device,
            )
            totals = accelerator.reduce(totals, reduction="sum")
            if accelerator.is_main_process:
                calibration_nll[task] = float(totals[0] / totals[1])
        if accelerator.is_main_process:
            with (output / "calibration_nll.json").open("w", encoding="utf-8") as handle:
                json.dump(calibration_nll, handle, indent=2, sort_keys=True)
    if accelerator.is_main_process:
        summary = {"benchmarks": scores, "calibration_nll": calibration_nll}
        with (output / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    accelerator.wait_for_everyone()
    return (
        {"benchmarks": scores, "calibration_nll": calibration_nll}
        if accelerator.is_main_process
        else {}
    )
