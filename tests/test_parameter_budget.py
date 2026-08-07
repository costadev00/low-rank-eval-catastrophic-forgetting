from low_rank_eval.lora.parameter_budget import ModuleShape, parameter_count
from low_rank_eval.lora.rank_patterns import build_block_ranks, peft_patterns


def _qwen_shapes() -> list[ModuleShape]:
    shapes = []
    for block in range(36):
        shapes.append(ModuleShape(f"model.layers.{block}.self_attn.q_proj", block, 2560, 4096))
        shapes.append(ModuleShape(f"model.layers.{block}.self_attn.v_proj", block, 2560, 1024))
    return shapes


def test_qwen_manual_patterns_have_exact_budget() -> None:
    shapes = _qwen_shapes()
    reference = parameter_count(shapes, {block: 16 for block in range(36)})
    assert reference == 5_898_240
    for strategy in ("uniform", "early_heavy", "middle_heavy", "late_heavy", "random"):
        ranks = build_block_ranks(shapes, strategy=strategy)
        assert parameter_count(shapes, ranks) == reference
        rank_pattern, alpha_pattern = peft_patterns(shapes, ranks)
        assert rank_pattern == alpha_pattern
        assert len(rank_pattern) == 72
