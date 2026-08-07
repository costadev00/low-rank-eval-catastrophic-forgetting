from low_rank_eval.evaluation.gsm8k import extract_final_number, gsm8k_exact_match


def test_gsm8k_extraction_prefers_explicit_final_answer() -> None:
    text = "We first compute 12. The answer is $1,234."
    assert extract_final_number(text) == "1234"
    correct, predicted, gold = gsm8k_exact_match(text, "work\n#### 1234")
    assert correct and predicted == gold == "1234"
