from pathlib import Path

import pytest

from low_rank_eval.config import load_config


def test_config_inheritance_and_fingerprint() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.training.train_token_budget_per_task == 20_000
    assert config.lora.strategy == "uniform"
    assert len(config.fingerprint()) == 16
    uniform = load_config("configs/uniform.yaml")
    early = load_config("configs/early_heavy.yaml")
    assert uniform.fingerprint() != early.fingerprint()
    assert uniform.data_fingerprint() == early.data_fingerprint()


def test_gradient_accumulation_is_world_size_aware() -> None:
    config = load_config("configs/base.yaml")
    assert config.training.accumulation_for_world_size(4) == 4
    assert config.training.accumulation_for_world_size(1) == 16
    with pytest.raises(ValueError):
        config.training.accumulation_for_world_size(3)
