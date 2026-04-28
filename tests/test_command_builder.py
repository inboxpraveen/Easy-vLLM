"""Tests for the vLLM command builder."""
from __future__ import annotations

from easy_vllm.command_builder import build_args, build_command_strings
from easy_vllm.schemas import DeploymentRequest


def test_default_command_includes_core_flags():
    req = DeploymentRequest(model_id="Qwen/Qwen3-8B-Instruct")
    args = build_args(req)
    joined = " ".join(args)
    assert "--model Qwen/Qwen3-8B-Instruct" in joined
    assert "--served-model-name qwen3-8b-instruct" in joined
    assert "--host 0.0.0.0" in joined
    assert "--port 8000" in joined
    assert "--dtype auto" in joined
    assert "--max-model-len 5120" in joined
    assert "--tensor-parallel-size 1" in joined
    assert "--enable-prefix-caching" in joined
    assert "--generation-config vllm" in joined


def test_quantization_emits_flag():
    req = DeploymentRequest(model_id="x/y", quantization="awq")
    joined = " ".join(build_args(req))
    assert "--quantization awq" in joined


def test_bitsandbytes_forces_load_format():
    req = DeploymentRequest(model_id="x/y", quantization="bitsandbytes")
    joined = " ".join(build_args(req))
    assert "--quantization bitsandbytes" in joined
    assert "--load-format bitsandbytes" in joined


def test_pipeline_parallel_only_when_gt_one():
    req = DeploymentRequest(model_id="x/y", pipeline_parallel_size=1)
    assert "--pipeline-parallel-size" not in build_args(req)
    req2 = DeploymentRequest(model_id="x/y", pipeline_parallel_size=2)
    assert "--pipeline-parallel-size" in build_args(req2)


def test_api_key_substitutes_env():
    req = DeploymentRequest(model_id="x/y", api_key_required=True)
    joined = " ".join(build_args(req))
    assert "--api-key ${VLLM_API_KEY}" in joined


def test_local_model_path_used_when_source_local():
    req = DeploymentRequest(
        model_source="local",
        local_model_path="/models/qwen3-8b",
        served_model_name="qwen3-8b-local",
    )
    joined = " ".join(build_args(req))
    assert "--model /models/qwen3-8b" in joined
    assert "--served-model-name qwen3-8b-local" in joined


def test_extra_flags_passthrough():
    req = DeploymentRequest(model_id="x/y", extra_flags="--swap-space 8 --seed 42")
    args = build_args(req)
    assert "--swap-space" in args
    assert "8" in args
    assert "--seed" in args
    assert "42" in args


def test_multiline_renders_flags_pairwise():
    req = DeploymentRequest(model_id="Qwen/Qwen3-8B-Instruct")
    oneline, multiline = build_command_strings(req)
    assert "Qwen/Qwen3-8B-Instruct" in oneline
    assert "\\\n" in multiline
    assert "--model Qwen/Qwen3-8B-Instruct" in multiline
