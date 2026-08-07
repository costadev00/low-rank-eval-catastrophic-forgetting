from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from low_rank_eval.lora.rank_patterns import verify_applied_ranks


def _tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(config)


def test_adapter_safetensors_reload_preserves_logits(tmp_path: Path) -> None:
    torch.manual_seed(42)
    base = _tiny_model()
    base_state = {key: value.clone() for key, value in base.state_dict().items()}
    patterns = {
        "model.layers.0.self_attn.q_proj": 4,
        "model.layers.0.self_attn.v_proj": 4,
        "model.layers.1.self_attn.q_proj": 8,
        "model.layers.1.self_attn.v_proj": 8,
    }
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            rank_pattern=patterns,
            alpha_pattern=patterns,
            use_rslora=False,
        ),
    )
    verify_applied_ranks(model, patterns)
    for module in model.modules():
        if hasattr(module, "lora_B") and "default" in module.lora_B:
            torch.nn.init.normal_(module.lora_B["default"].weight)
    inputs = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    expected = model(inputs).logits.detach()
    adapter_path = tmp_path / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    assert (adapter_path / "adapter_model.safetensors").exists()

    restored_base = _tiny_model()
    restored_base.load_state_dict(base_state)
    restored = PeftModel.from_pretrained(restored_base, adapter_path)
    restored.eval()
    actual = restored(inputs).logits.detach()
    torch.testing.assert_close(actual, expected)
