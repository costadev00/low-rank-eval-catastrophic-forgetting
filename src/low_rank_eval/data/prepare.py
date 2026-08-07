from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from low_rank_eval.config import ExperimentConfig
from low_rank_eval.data.decontamination import (
    Decontaminator,
    ReferenceIndex,
    ReferenceText,
    text_hash,
)
from low_rank_eval.data.token_budget import (
    TokenizedExample,
    pack_examples,
    take_exact_assistant_budget,
    tokenize_conversation,
    write_token_manifest,
)


def _load(ref: Any) -> Dataset:
    return load_dataset(
        ref.name,
        ref.config,
        split=ref.split,
        revision=ref.revision,
    )


def _gsm_references(config: ExperimentConfig) -> Dataset:
    return concatenate_datasets([_load(config.data.gsm8k_train), _load(config.data.gsm8k_eval)])


def _reference_texts(dataset: Dataset, field: str, prefix: str) -> list[ReferenceText]:
    return [
        ReferenceText(f"{prefix}:{index}", str(row[field])) for index, row in enumerate(dataset)
    ]


def _decontaminate(
    dataset: Dataset,
    references: Iterable[ReferenceText],
    *,
    text_field: str,
    output_dir: Path,
    config: ExperimentConfig,
    source_field: str | None = None,
    reject_gsm_source: bool = False,
) -> Dataset:
    settings = config.decontamination
    index = ReferenceIndex(
        references,
        approximate_threshold=settings.approximate_threshold,
        candidate_threshold=settings.candidate_threshold,
        num_perm=settings.num_perm,
        ngram_size=settings.char_ngram_size,
    )
    cleaner = Decontaminator(index, remove_internal_duplicates=settings.remove_internal_duplicates)

    def predicate(row: dict[str, Any], row_index: int) -> bool:
        return cleaner.keep(
            str(row[text_field]),
            dataset_index=row_index,
            source=str(row[source_field]) if source_field else None,
            reject_gsm_source=reject_gsm_source,
        )

    cleaned = dataset.filter(predicate, with_indices=True, desc=f"Decontaminating {text_field}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.save_to_disk(str(output_dir / "clean"))
    cleaner.write_audit(
        output_dir / "audit",
        {
            "input_rows": len(dataset),
            "output_rows": len(cleaned),
            "text_field": text_field,
            "approximate_threshold": settings.approximate_threshold,
            "candidate_threshold": settings.candidate_threshold,
            "num_perm": settings.num_perm,
            "char_ngram_size": settings.char_ngram_size,
        },
    )
    return cleaned


def decontaminate_training_data(
    config: ExperimentConfig, *, force: bool = False
) -> dict[str, Path]:
    root = config.data.processed_dir / "clean" / config.decontamination_fingerprint()
    ifeval_path = root / "ifeval_like"
    numina_path = root / "numinamath"
    if not force and (ifeval_path / "clean").exists() and (numina_path / "clean").exists():
        return {"ifeval": ifeval_path / "clean", "math": numina_path / "clean"}

    ifeval_eval = _load(config.data.ifeval_eval)
    gsm_all = _gsm_references(config)
    ifeval_train = _load(config.data.ifeval_train)
    numina_train = _load(config.data.numina_train)
    if config.data.preparation_limit_per_task is not None:
        limit = config.data.preparation_limit_per_task
        ifeval_train = ifeval_train.select(range(min(limit, len(ifeval_train))))
        numina_train = numina_train.select(range(min(limit, len(numina_train))))

    _decontaminate(
        ifeval_train,
        _reference_texts(ifeval_eval, "prompt", "ifeval"),
        text_field="prompt",
        output_dir=ifeval_path,
        config=config,
    )
    _decontaminate(
        numina_train,
        _reference_texts(gsm_all, "question", "gsm8k"),
        text_field="problem",
        source_field="source",
        reject_gsm_source=True,
        output_dir=numina_path,
        config=config,
    )
    return {"ifeval": ifeval_path / "clean", "math": numina_path / "clean"}


def _messages(task: str, row: dict[str, Any]) -> list[dict[str, str]]:
    if task == "ifeval":
        return [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["response"])},
        ]
    if task == "math":
        messages = row.get("messages")
        if messages:
            return [
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in messages
            ]
        return [
            {"role": "user", "content": str(row["problem"])},
            {"role": "assistant", "content": str(row["solution"])},
        ]
    raise ValueError(f"Unknown task {task}")


def _tokenized_stream(
    dataset: Dataset,
    tokenizer: Any,
    *,
    task: str,
    max_sequence_length: int,
) -> Iterable[TokenizedExample]:
    for row in dataset:
        messages = _messages(task, row)
        source_text = messages[0]["content"]
        tokenized = tokenize_conversation(
            tokenizer,
            messages,
            max_sequence_length=max_sequence_length,
            source_id=text_hash(source_text),
        )
        if tokenized is not None:
            yield tokenized


def build_token_budget_splits(
    config: ExperimentConfig,
    *,
    force: bool = False,
) -> dict[str, dict[str, Path]]:
    clean_paths = decontaminate_training_data(config, force=force)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    output_root = config.data.processed_dir / config.data_fingerprint() / "tokenized"
    outputs: dict[str, dict[str, Path]] = {}
    for task, clean_path in clean_paths.items():
        task_root = output_root / task
        train_path, calibration_path = task_root / "train", task_root / "calibration"
        if not force and train_path.exists() and calibration_path.exists():
            outputs[task] = {"train": train_path, "calibration": calibration_path}
            continue
        dataset = load_from_disk(str(clean_path)).shuffle(seed=config.training.seed)
        stream = iter(
            _tokenized_stream(
                dataset,
                tokenizer,
                task=task,
                max_sequence_length=config.training.max_sequence_length,
            )
        )
        calibration, calibration_stats = take_exact_assistant_budget(
            stream, config.training.calibration_token_budget_per_task
        )
        train, train_stats = take_exact_assistant_budget(
            stream, config.training.train_token_budget_per_task
        )
        train_rows = (
            list(pack_examples(train, config.training.max_sequence_length))
            if config.training.packing
            else [example.as_dict() for example in train]
        )
        calibration_rows = [example.as_dict() for example in calibration]
        Dataset.from_list(train_rows).save_to_disk(str(train_path))
        Dataset.from_list(calibration_rows).save_to_disk(str(calibration_path))
        overlap = {item.source_id for item in calibration} & {item.source_id for item in train}
        if overlap:
            raise RuntimeError("Calibration and training splits overlap")
        write_token_manifest(
            task_root / "token_manifest.json",
            {
                "task": task,
                "packing": config.training.packing,
                "train": train_stats,
                "calibration": calibration_stats,
                "source_fingerprint": dataset._fingerprint,
                "tokenizer": config.model.name,
                "tokenizer_revision": config.model.revision,
                "max_sequence_length": config.training.max_sequence_length,
            },
        )
        outputs[task] = {"train": train_path, "calibration": calibration_path}
    with (output_root / "prepared_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_fingerprint": config.data_fingerprint(),
                "outputs": {
                    task: {split: str(path) for split, path in paths.items()}
                    for task, paths in outputs.items()
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return outputs
