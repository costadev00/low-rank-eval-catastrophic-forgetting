from low_rank_eval.lora.spectral_allocator import allocate_spectral_ranks


def test_allocator_respects_budget_and_is_non_uniform() -> None:
    curves = {
        0: {4: 0.5, 8: 0.8, 12: 0.9, 16: 0.94, 20: 0.96, 24: 0.97, 28: 0.98, 32: 1.0},
        1: {4: 0.1, 8: 0.2, 12: 0.3, 16: 0.4, 20: 0.55, 24: 0.7, 28: 0.9, 32: 1.0},
        2: {4: 0.6, 8: 0.9, 12: 0.96, 16: 0.98, 20: 0.99, 24: 1.0, 28: 1.0, 32: 1.0},
    }
    result = allocate_spectral_ranks(
        curves,
        {0: 10, 1: 10, 2: 10},
        reference_parameters=3 * 16 * 10,
    )
    assert result.parameters == result.reference_parameters
    assert len(set(result.block_ranks.values())) > 1
