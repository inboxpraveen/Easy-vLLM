"""Tests for the HF config.json parser."""
from __future__ import annotations

import json

from easy_vllm.config_parser import parse_config_bytes, parse_config_dict


QWEN3_8B = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 22016,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
}


MIXTRAL_8X7B = {
    "model_type": "mixtral",
    "architectures": ["MixtralForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "torch_dtype": "bfloat16",
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}


LLAVA_VLM = {
    "model_type": "llava",
    "architectures": ["LlavaForConditionalGeneration"],
    "vision_config": {"hidden_size": 1024},
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
    },
}


def test_dense_model_is_not_uncertain():
    info = parse_config_dict(QWEN3_8B)
    assert info.model_type == "qwen3"
    assert info.num_hidden_layers == 32
    assert info.num_key_value_heads == 8
    assert info.head_dim == 128
    assert info.is_moe is False
    assert info.is_multimodal is False
    assert info.is_uncertain is False
    assert info.estimated_param_count is not None


def test_moe_detection_marks_uncertain():
    info = parse_config_dict(MIXTRAL_8X7B)
    assert info.is_moe is True
    assert info.is_uncertain is True
    assert any("Mixture-of-Experts" in n for n in info.notes)


def test_multimodal_detection_marks_uncertain():
    info = parse_config_dict(LLAVA_VLM)
    assert info.is_multimodal is True
    assert info.is_uncertain is True


def test_head_dim_inferred_when_missing():
    raw = dict(QWEN3_8B)
    raw.pop("head_dim")
    info = parse_config_dict(raw)
    assert info.head_dim == 128


def test_kv_heads_default_to_attention_heads_when_missing():
    raw = dict(QWEN3_8B)
    raw.pop("num_key_value_heads")
    info = parse_config_dict(raw)
    assert info.num_key_value_heads == raw["num_attention_heads"]


def test_parse_bytes_roundtrip():
    blob = json.dumps(QWEN3_8B).encode()
    info = parse_config_bytes(blob)
    assert info.model_type == "qwen3"
