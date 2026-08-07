from low_rank_eval.data.token_budget import (
    IGNORE_INDEX,
    TokenizedExample,
    pack_examples,
    take_exact_assistant_budget,
)


def _example(source: str, assistant: int, prompt: int = 2) -> TokenizedExample:
    ids = list(range(prompt + assistant))
    return TokenizedExample(
        input_ids=ids,
        attention_mask=[1] * len(ids),
        labels=[IGNORE_INDEX] * prompt + ids[prompt:],
        assistant_tokens=assistant,
        prompt_tokens=prompt,
        total_tokens=len(ids),
        source_id=source,
    )


def test_exact_budget_truncates_only_final_assistant() -> None:
    selected, stats = take_exact_assistant_budget([_example("a", 4), _example("b", 4)], 6)
    assert stats["assistant_tokens"] == 6
    assert selected[-1].assistant_tokens == 2
    assert selected[-1].budget_truncated
    assert all(label == IGNORE_INDEX for label in selected[-1].labels[:2])


def test_packing_preserves_labels_and_boundaries() -> None:
    examples = [_example("a", 2), _example("b", 3), _example("c", 4)]
    packed = list(pack_examples(examples, max_sequence_length=9))
    assert [row["source_ids"] for row in packed] == [["a", "b"], ["c"]]
    assert sum(label != IGNORE_INDEX for row in packed for label in row["labels"]) == 9
