from pathlib import Path

import pytest

from low_rank_eval.config import load_config
from low_rank_eval.training.sequential import _base_cache_key


def test_config_inheritance_and_fingerprint() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.training.train_token_budget_per_task == 20_000
    assert config.lora.strategy == "uniform"
    assert config.evaluation.calibration_batch_size == 1
    assert len(config.fingerprint()) == 16
    uniform = load_config("configs/uniform.yaml")
    early = load_config("configs/early_heavy.yaml")
    assert uniform.fingerprint() != early.fingerprint()
    assert uniform.data_fingerprint() == early.data_fingerprint()


def test_calibration_batch_size_is_operational_and_preserves_run_id() -> None:
    config = load_config("configs/uniform.yaml")
    changed = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(update={"calibration_batch_size": 2})
        }
    )
    assert config.fingerprint() == changed.fingerprint() == "e0102e4958087b5e"
    assert _base_cache_key(config) == _base_cache_key(changed)


def test_gradient_accumulation_is_world_size_aware() -> None:
    config = load_config("configs/base.yaml")
    assert config.training.accumulation_for_world_size(4) == 4
    assert config.training.accumulation_for_world_size(1) == 16
    with pytest.raises(ValueError):
        config.training.accumulation_for_world_size(3)
