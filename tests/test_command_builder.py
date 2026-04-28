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


# ---------------------------------------------------------------------------
# New vLLM options coverage
# ---------------------------------------------------------------------------


def test_chunked_prefill_explicit_on_off():
    on = " ".join(build_args(DeploymentRequest(model_id="x/y", enable_chunked_prefill=True)))
    off = " ".join(build_args(DeploymentRequest(model_id="x/y", enable_chunked_prefill=False)))
    assert "--enable-chunked-prefill" in on and "--no-enable-chunked-prefill" not in on
    assert "--no-enable-chunked-prefill" in off


def test_enforce_eager_and_seed_emit():
    args = build_args(DeploymentRequest(model_id="x/y", enforce_eager=True, seed=42))
    joined = " ".join(args)
    assert "--enforce-eager" in joined
    assert "--seed 42" in joined


def test_swap_space_emitted_when_positive():
    args = build_args(DeploymentRequest(model_id="x/y", swap_space_gb=8))
    joined = " ".join(args)
    assert "--swap-space" in joined
    # Pydantic stores the float; the builder stringifies it.
    assert "--swap-space 8" in joined
    args0 = build_args(DeploymentRequest(model_id="x/y", swap_space_gb=0))
    assert "--swap-space" not in args0


def test_lora_emits_and_lora_modules_split():
    req = DeploymentRequest(
        model_id="x/y",
        enable_lora=True,
        max_loras=4,
        max_lora_rank=16,
        lora_modules="sql=/loras/sql\nchat=/loras/chat",
    )
    args = build_args(req)
    joined = " ".join(args)
    assert "--enable-lora" in joined
    assert "--max-loras 4" in joined
    assert "--max-lora-rank 16" in joined
    # Each line becomes a separate --lora-modules arg
    pairs = [(args[i], args[i + 1]) for i in range(len(args) - 1) if args[i] == "--lora-modules"]
    assert ("--lora-modules", "sql=/loras/sql") in pairs
    assert ("--lora-modules", "chat=/loras/chat") in pairs


def test_speculative_config_draft_model_json():
    req = DeploymentRequest(
        model_id="x/y",
        speculative_method="draft_model",
        speculative_model="meta-llama/Llama-3.2-1B",
        num_speculative_tokens=5,
    )
    args = build_args(req)
    assert "--speculative-config" in args
    idx = args.index("--speculative-config")
    payload = args[idx + 1]
    import json as _json
    parsed = _json.loads(payload)
    assert parsed["method"] == "draft_model"
    assert parsed["model"] == "meta-llama/Llama-3.2-1B"
    assert parsed["num_speculative_tokens"] == 5


def test_speculative_ngram_omits_model_key():
    req = DeploymentRequest(
        model_id="x/y",
        speculative_method="ngram",
        num_speculative_tokens=4,
    )
    args = build_args(req)
    idx = args.index("--speculative-config")
    import json as _json
    parsed = _json.loads(args[idx + 1])
    assert parsed["method"] == "ngram"
    assert "model" not in parsed
    assert parsed["num_speculative_tokens"] == 4


def test_tools_chat_and_reasoning_flags():
    req = DeploymentRequest(
        model_id="x/y",
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
        chat_template="/templates/chatml.jinja",
        reasoning_parser="qwen3",
    )
    joined = " ".join(build_args(req))
    assert "--enable-auto-tool-choice" in joined
    assert "--tool-call-parser hermes" in joined
    assert "--chat-template /templates/chatml.jinja" in joined
    assert "--reasoning-parser qwen3" in joined


def test_server_logging_and_cors():
    req = DeploymentRequest(
        model_id="x/y",
        allowed_origins="https://a.com,https://b.com",
        enable_log_requests=True,
        max_log_len=200,
    )
    joined = " ".join(build_args(req))
    assert "--allowed-origins https://a.com,https://b.com" in joined
    assert "--enable-log-requests" in joined
    assert "--max-log-len 200" in joined


def test_loading_distribution_and_mm():
    req = DeploymentRequest(
        model_id="x/y",
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        revision="main",
        download_dir="/cache",
        data_parallel_size=2,
        distributed_executor_backend="ray",
        limit_mm_per_prompt="image=4,video=1",
    )
    joined = " ".join(build_args(req))
    assert "--tokenizer meta-llama/Llama-3.1-8B-Instruct" in joined
    assert "--revision main" in joined
    assert "--download-dir /cache" in joined
    assert "--data-parallel-size 2" in joined
    assert "--distributed-executor-backend ray" in joined
    assert "--limit-mm-per-prompt image=4,video=1" in joined


def test_scheduling_policy_and_async():
    req = DeploymentRequest(
        model_id="x/y",
        scheduling_policy="priority",
        async_scheduling=True,
        max_num_partial_prefills=2,
        long_prefill_token_threshold=8192,
    )
    joined = " ".join(build_args(req))
    assert "--scheduling-policy priority" in joined
    assert "--async-scheduling" in joined
    assert "--max-num-partial-prefills 2" in joined
    assert "--long-prefill-token-threshold 8192" in joined


def test_sliding_window_and_cascade_attn_toggles():
    req = DeploymentRequest(
        model_id="x/y",
        disable_sliding_window=True,
        disable_cascade_attn=True,
    )
    joined = " ".join(build_args(req))
    assert "--disable-sliding-window" in joined
    assert "--disable-cascade-attn" in joined
