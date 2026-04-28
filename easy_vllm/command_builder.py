"""Build the ``vllm serve`` argument list from a :class:`DeploymentRequest`.

Flag emission order is deliberate: it mirrors the wizard's section order so
the live preview reads top-to-bottom the same way the user filled the form.
"""
from __future__ import annotations

import json
import shlex
from typing import Iterable

from .schemas import DeploymentRequest


def _resolved_model_path(req: DeploymentRequest) -> str:
    if req.model_source == "local" and req.local_model_path:
        return req.local_model_path
    return req.model_id or "PLEASE_SET_MODEL_ID"


def _resolved_served_name(req: DeploymentRequest) -> str:
    if req.served_model_name:
        return req.served_model_name
    raw = _resolved_model_path(req)
    base = raw.rstrip("/").split("/")[-1] or "local-model"
    return base.lower().replace(" ", "-")


def _build_speculative_config(req: DeploymentRequest) -> str | None:
    """Render the ``--speculative-config`` JSON value if speculative decoding is on."""
    method = req.speculative_method
    if not method or method == "none":
        return None

    payload: dict[str, object] = {"method": method}
    if req.speculative_model and method in {"draft_model", "mtp", "eagle3"}:
        payload["model"] = req.speculative_model
    if req.num_speculative_tokens:
        payload["num_speculative_tokens"] = int(req.num_speculative_tokens)
    return json.dumps(payload, separators=(",", ":"))


def build_args(req: DeploymentRequest) -> list[str]:
    """Return a flat list of CLI tokens for ``vllm serve``.

    Order: identity → precision/quant → context/memory → parallelism →
    scheduling → KV cache → loading → LoRA → speculative → tools/chat →
    server/CORS/logging → multimodal → trust/api → extra raw passthrough.
    """
    args: list[str] = []

    # ---- identity & networking ---------------------------------------------
    args += ["--model", _resolved_model_path(req)]
    args += ["--served-model-name", _resolved_served_name(req)]
    args += ["--host", "0.0.0.0"]
    args += ["--port", "8000"]

    # ---- precision & quantization ------------------------------------------
    args += ["--dtype", req.dtype]

    if req.quantization and req.quantization != "none":
        args += ["--quantization", req.quantization]

    # ---- context length & GPU memory ---------------------------------------
    args += ["--max-model-len", str(req.input_tokens + req.output_tokens)]
    args += ["--gpu-memory-utilization", f"{req.gpu_memory_utilization:.2f}"]
    if req.cpu_offload_gb:
        args += ["--cpu-offload-gb", str(req.cpu_offload_gb)]
    if req.swap_space_gb is not None and req.swap_space_gb > 0:
        args += ["--swap-space", str(req.swap_space_gb)]

    # ---- parallelism --------------------------------------------------------
    args += ["--tensor-parallel-size", str(req.tensor_parallel_size)]
    if req.pipeline_parallel_size and req.pipeline_parallel_size > 1:
        args += ["--pipeline-parallel-size", str(req.pipeline_parallel_size)]
    if req.data_parallel_size and req.data_parallel_size > 1:
        args += ["--data-parallel-size", str(req.data_parallel_size)]
    if req.distributed_executor_backend and req.distributed_executor_backend != "auto":
        args += ["--distributed-executor-backend", req.distributed_executor_backend]

    # ---- scheduling / batching ---------------------------------------------
    args += ["--max-num-seqs", str(req.max_num_seqs)]
    if req.max_num_batched_tokens:
        args += ["--max-num-batched-tokens", str(req.max_num_batched_tokens)]

    if req.enable_chunked_prefill is True:
        args += ["--enable-chunked-prefill"]
    elif req.enable_chunked_prefill is False:
        args += ["--no-enable-chunked-prefill"]

    if req.max_num_partial_prefills:
        args += ["--max-num-partial-prefills", str(req.max_num_partial_prefills)]
    if req.long_prefill_token_threshold:
        args += [
            "--long-prefill-token-threshold",
            str(req.long_prefill_token_threshold),
        ]
    if req.scheduling_policy and req.scheduling_policy != "fcfs":
        args += ["--scheduling-policy", req.scheduling_policy]
    if req.async_scheduling:
        args += ["--async-scheduling"]

    # ---- KV cache & runtime tuning -----------------------------------------
    if req.kv_cache_dtype and req.kv_cache_dtype != "auto":
        args += ["--kv-cache-dtype", req.kv_cache_dtype]
    if req.enable_prefix_caching:
        args += ["--enable-prefix-caching"]
    if req.enforce_eager:
        args += ["--enforce-eager"]
    if req.disable_sliding_window:
        args += ["--disable-sliding-window"]
    if req.disable_cascade_attn:
        args += ["--disable-cascade-attn"]
    if req.seed is not None:
        args += ["--seed", str(req.seed)]

    # ---- loading & tokenizer -----------------------------------------------
    load_format = req.load_format
    if req.quantization == "bitsandbytes" and load_format == "auto":
        load_format = "bitsandbytes"
    if req.quantization == "gguf" and load_format == "auto":
        load_format = "gguf"
    if load_format != "auto":
        args += ["--load-format", load_format]

    if req.tokenizer:
        args += ["--tokenizer", req.tokenizer]
    if req.revision:
        args += ["--revision", req.revision]
    if req.download_dir:
        args += ["--download-dir", req.download_dir]

    # ---- LoRA adapters ------------------------------------------------------
    if req.enable_lora:
        args += ["--enable-lora"]
        if req.max_loras:
            args += ["--max-loras", str(req.max_loras)]
        if req.max_lora_rank:
            args += ["--max-lora-rank", str(req.max_lora_rank)]
        if req.lora_modules:
            for line in req.lora_modules.splitlines():
                line = line.strip()
                if line:
                    args += ["--lora-modules", line]

    # ---- speculative decoding ----------------------------------------------
    spec_cfg = _build_speculative_config(req)
    if spec_cfg:
        args += ["--speculative-config", spec_cfg]

    # ---- tools, chat & reasoning ------------------------------------------
    if req.enable_auto_tool_choice:
        args += ["--enable-auto-tool-choice"]
    if req.tool_call_parser:
        args += ["--tool-call-parser", req.tool_call_parser]
    if req.chat_template:
        args += ["--chat-template", req.chat_template]
    if req.reasoning_parser:
        args += ["--reasoning-parser", req.reasoning_parser]
    if req.generation_config_vllm:
        args += ["--generation-config", "vllm"]

    # ---- server / CORS / logging -------------------------------------------
    if req.allowed_origins:
        args += ["--allowed-origins", req.allowed_origins]
    if req.enable_log_requests:
        args += ["--enable-log-requests"]
    if req.max_log_len:
        args += ["--max-log-len", str(req.max_log_len)]

    # ---- multimodal --------------------------------------------------------
    if req.limit_mm_per_prompt:
        args += ["--limit-mm-per-prompt", req.limit_mm_per_prompt]

    # ---- security / auth ---------------------------------------------------
    if req.trust_remote_code:
        args += ["--trust-remote-code"]
    if req.api_key_required:
        args += ["--api-key", "${VLLM_API_KEY}"]

    # ---- raw passthrough ---------------------------------------------------
    if req.extra_flags:
        try:
            extra = shlex.split(req.extra_flags)
            args += extra
        except ValueError:
            args += req.extra_flags.split()

    return args


def args_to_oneline(args: Iterable[str]) -> str:
    return " ".join(_quote_if_needed(a) for a in args)


def args_to_multiline(args: Iterable[str], indent: str = "      ") -> str:
    """Render args as docker-compose ``command: >`` block content.

    Pairs each ``--flag value`` on a single line for readability.
    """
    items = list(args)
    lines: list[str] = []
    i = 0
    while i < len(items):
        token = items[i]
        if token.startswith("--") and i + 1 < len(items) and not items[i + 1].startswith("--"):
            lines.append(f"{token} {_quote_if_needed(items[i + 1])}")
            i += 2
        else:
            lines.append(token)
            i += 1
    joined = (" \\\n" + indent).join(lines)
    return joined


def _quote_if_needed(token: str) -> str:
    if not token:
        return '""'
    if any(c in token for c in (" ", "\t", "\"", "'")):
        return shlex.quote(token)
    return token


def build_command_strings(req: DeploymentRequest) -> tuple[str, str]:
    args = build_args(req)
    return args_to_oneline(args), args_to_multiline(args)
