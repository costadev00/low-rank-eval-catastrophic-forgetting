from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100


@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    assistant_tokens: int
    prompt_tokens: int
    total_tokens: int
    source_id: str
    budget_truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def tokenize_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_sequence_length: int,
    source_id: str,
) -> TokenizedExample | None:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected a conversation ending with an assistant message")
    prompt_messages = messages[:-1]
    prompt_encoding = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = (
        prompt_encoding["input_ids"] if isinstance(prompt_encoding, Mapping) else prompt_encoding
    )
    full_ids = full_encoding["input_ids"] if isinstance(full_encoding, Mapping) else full_encoding
    prefix = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids, strict=False):
        if prompt_token != full_token:
            break
        prefix += 1
    if prefix == 0:
        raise ValueError("The official chat template did not preserve a prompt prefix")
    full_ids = list(full_ids[:max_sequence_length])
    if prefix >= len(full_ids):
        return None
    labels = [IGNORE_INDEX] * prefix + full_ids[prefix:]
    attention_mask = [1] * len(full_ids)
    assistant_tokens = sum(label != IGNORE_INDEX for label in labels)
    return TokenizedExample(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=labels,
        assistant_tokens=assistant_tokens,
        prompt_tokens=prefix,
        total_tokens=len(full_ids),
        source_id=source_id,
    )


def trim_to_assistant_budget(example: TokenizedExample, remaining: int) -> TokenizedExample:
    if remaining <= 0 or remaining > example.assistant_tokens:
        raise ValueError("remaining must be within the example's assistant-token count")
    assistant_positions = [
        index for index, label in enumerate(example.labels) if label != IGNORE_INDEX
    ]
    end = assistant_positions[remaining - 1] + 1
    return TokenizedExample(
        input_ids=example.input_ids[:end],
        attention_mask=example.attention_mask[:end],
        labels=example.labels[:end],
        assistant_tokens=remaining,
        prompt_tokens=example.prompt_tokens,
        total_tokens=end,
        source_id=example.source_id,
        budget_truncated=True,
    )


def take_exact_assistant_budget(
    examples: Iterable[TokenizedExample], budget: int
) -> tuple[list[TokenizedExample], dict[str, int]]:
    selected: list[TokenizedExample] = []
    assistant_tokens = prompt_tokens = total_tokens = 0
    truncated_examples = 0
    for example in examples:
        remaining = budget - assistant_tokens
        if remaining <= 0:
            break
        chosen = example
        if example.assistant_tokens > remaining:
            chosen = trim_to_assistant_budget(example, remaining)
            truncated_examples += 1
        selected.append(chosen)
        assistant_tokens += chosen.assistant_tokens
        prompt_tokens += chosen.prompt_tokens
        total_tokens += chosen.total_tokens
    if assistant_tokens != budget:
        raise ValueError(
            f"Dataset exhausted at {assistant_tokens:,} assistant tokens, need {budget:,}"
        )
    return selected, {
        "examples": len(selected),
        "assistant_tokens": assistant_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "budget_truncated_examples": truncated_examples,
    }


def pack_examples(
    examples: Iterable[TokenizedExample], max_sequence_length: int
) -> Iterator[dict[str, Any]]:
    current = {"input_ids": [], "attention_mask": [], "labels": [], "source_ids": []}
    for example in examples:
        if len(example.input_ids) > max_sequence_length:
            raise ValueError("Input must be truncated before packing")
        if (
            current["input_ids"]
            and len(current["input_ids"]) + len(example.input_ids) > max_sequence_length
        ):
            yield current
            current = {"input_ids": [], "attention_mask": [], "labels": [], "source_ids": []}
        current["input_ids"].extend(example.input_ids)
        current["attention_mask"].extend(example.attention_mask)
        current["labels"].extend(example.labels)
        current["source_ids"].append(example.source_id)
    if current["input_ids"]:
        yield current


def write_token_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
