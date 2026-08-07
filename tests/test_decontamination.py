from low_rank_eval.data.decontamination import (
    Decontaminator,
    ReferenceIndex,
    ReferenceText,
    normalize_text,
    text_hash,
)


def test_normalization_and_exact_hash() -> None:
    assert normalize_text("  Héllo,\nWORLD! ") == "héllo world"
    assert text_hash("A + B") == text_hash("a b")


def test_exact_approximate_source_and_internal_duplicate_are_exclusive() -> None:
    index = ReferenceIndex(
        [ReferenceText("eval:1", "Janet has sixteen ducks and sells nine eggs every day")],
        approximate_threshold=0.75,
        candidate_threshold=0.50,
        num_perm=64,
        ngram_size=3,
    )
    cleaner = Decontaminator(index)
    assert not cleaner.keep(
        "Janet has sixteen ducks and sells nine eggs every day", dataset_index=0
    )
    assert not cleaner.keep("unrelated", dataset_index=1, source="GSM8K", reject_gsm_source=True)
    assert cleaner.keep("a genuinely different prompt", dataset_index=2)
    assert not cleaner.keep("A genuinely different prompt!", dataset_index=3)
    assert cleaner.counts() == {
        "benchmark_exact": 1,
        "source_benchmark": 1,
        "internal_duplicate": 1,
    }
