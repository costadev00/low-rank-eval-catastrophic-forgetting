import torch

from low_rank_eval.lora.spectral_analysis import singular_values_from_factors


def test_factor_singular_values_match_explicit_product() -> None:
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(5, 11, generator=generator)
    b = torch.randn(13, 5, generator=generator)
    expected = torch.linalg.svdvals(b @ a)
    actual = singular_values_from_factors(a, b)
    torch.testing.assert_close(actual, expected[:5], rtol=1e-5, atol=1e-5)
    assert torch.all(expected[5:] < 1e-5)
