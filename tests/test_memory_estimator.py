"""Tests for the memory estimator."""
from __future__ import annotations

import pytest

from easy_vllm.memory_estimator import (
    bytes_per_weight,
    estimate_memory,
    kv_dtype_bytes,
    resolve_param_count,
)
from easy_vllm.schemas import DeploymentRequest, ModelConfigInfo


def make_qwen3_8b_config() -> ModelConfigInfo:
    return ModelConfigInfo(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=32768,
        torch_dtype="bfloat16",
        estimated_param_count=8.0e9,
    )


def test_bytes_per_weight_picks_quantization_when_set():
    assert bytes_per_weight("auto", "awq") == 0.5
    assert bytes_per_weight("bfloat16", "fp8") == 1.0


def test_bytes_per_weight_defaults_to_bfloat16_for_auto():
    assert bytes_per_weight("auto", "none") == 2.0


def test_kv_dtype_bytes_explicit_overrides_dtype():
    assert kv_dtype_bytes("fp8", "bfloat16") == 1.0
    assert kv_dtype_bytes("auto", "bfloat16") == 2.0
    assert kv_dtype_bytes("auto", "float32") == 4.0


def test_resolve_param_count_prefers_manual():
    req = DeploymentRequest(manual_param_count_b=7.0)
    params, is_est, note = resolve_param_count(req)
    assert params == pytest.approx(7e9)
    assert is_est is False
    assert note is None


def test_resolve_param_count_uses_config_estimate():
    cfg = make_qwen3_8b_config()
    req = DeploymentRequest(config_info=cfg)
    params, is_est, _ = resolve_param_count(req)
    assert params == pytest.approx(8e9)
    assert is_est is False


def test_estimate_memory_qwen3_8b_on_24gb_bf16_is_risky_or_oom():
    req = DeploymentRequest(
        config_info=make_qwen3_8b_config(),
        gpu_memory_gb=24.0,
        gpu_memory_utilization=0.90,
        input_tokens=4096,
        output_tokens=1024,
        max_num_seqs=32,
        dtype="bfloat16",
    )
    mem = estimate_memory(req)
    assert mem.weight_gb > 14.0
    assert mem.kv_cache_gb > 0.5
    assert mem.fit_status in ("risky", "oom")


def test_estimate_memory_qwen3_8b_awq_fits_24gb():
    cfg = make_qwen3_8b_config()
    cfg.quantization_config = {"quant_method": "awq"}
    req = DeploymentRequest(
        config_info=cfg,
        gpu_memory_gb=24.0,
        gpu_memory_utilization=0.90,
        input_tokens=2048,
        output_tokens=512,
        max_num_seqs=16,
        quantization="awq",
    )
    mem = estimate_memory(req)
    assert mem.fit_status == "good"
    assert mem.weight_gb < 6.0


def test_estimate_memory_tensor_parallel_halves_per_gpu_weights():
    cfg = make_qwen3_8b_config()
    base = DeploymentRequest(
        config_info=cfg, gpu_memory_gb=80.0, tensor_parallel_size=1, gpu_count=1
    )
    tp2 = DeploymentRequest(
        config_info=cfg, gpu_memory_gb=80.0, tensor_parallel_size=2, gpu_count=2
    )
    mem1 = estimate_memory(base)
    mem2 = estimate_memory(tp2)
    assert mem2.weight_gb == pytest.approx(mem1.weight_gb / 2, rel=0.05)


def test_estimate_memory_unknown_when_no_params_or_config():
    req = DeploymentRequest(gpu_memory_gb=24.0)
    mem = estimate_memory(req)
    assert mem.fit_status == "unknown"
