"""Build the ``vllm serve`` argument list from a :class:`DeploymentRequest`."""
from __future__ import annotations

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


def build_args(req: DeploymentRequest) -> list[str]:
    """Return a flat list of CLI tokens for ``vllm serve``."""
    args: list[str] = []

    args += ["--model", _resolved_model_path(req)]
    args += ["--served-model-name", _resolved_served_name(req)]
    args += ["--host", "0.0.0.0"]
    args += ["--port", "8000"]
    args += ["--dtype", req.dtype]

    args += ["--max-model-len", str(req.input_tokens + req.output_tokens)]
    args += ["--gpu-memory-utilization", f"{req.gpu_memory_utilization:.2f}"]

    args += ["--tensor-parallel-size", str(req.tensor_parallel_size)]
    if req.pipeline_parallel_size and req.pipeline_parallel_size > 1:
        args += ["--pipeline-parallel-size", str(req.pipeline_parallel_size)]

    args += ["--max-num-seqs", str(req.max_num_seqs)]
    if req.max_num_batched_tokens:
        args += ["--max-num-batched-tokens", str(req.max_num_batched_tokens)]

    if req.quantization and req.quantization != "none":
        args += ["--quantization", req.quantization]

    load_format = req.load_format
    if req.quantization == "bitsandbytes" and load_format == "auto":
        load_format = "bitsandbytes"
    if req.quantization == "gguf" and load_format == "auto":
        load_format = "gguf"
    if load_format != "auto":
        args += ["--load-format", load_format]

    if req.kv_cache_dtype and req.kv_cache_dtype != "auto":
        args += ["--kv-cache-dtype", req.kv_cache_dtype]

    if req.cpu_offload_gb:
        args += ["--cpu-offload-gb", str(req.cpu_offload_gb)]

    if req.enable_prefix_caching:
        args += ["--enable-prefix-caching"]

    if req.generation_config_vllm:
        args += ["--generation-config", "vllm"]

    if req.trust_remote_code:
        args += ["--trust-remote-code"]

    if req.api_key_required:
        args += ["--api-key", "${VLLM_API_KEY}"]

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
