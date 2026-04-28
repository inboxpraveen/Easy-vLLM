"""Memory estimator for vLLM deployments.

Estimates VRAM required for weights and the KV cache and produces a Good /
Risky / Likely-OOM verdict plus actionable suggestions. The numbers are
intentionally rough - vLLM profiles memory at startup and the real number
also depends on CUDA graphs, kernels, fragmentation, and activations - but
they're accurate enough to catch obvious OOM disasters before deployment.
"""
from __future__ import annotations

import math
from typing import Optional

from .schemas import (
    DeploymentRequest,
    FitStatus,
    MemoryBreakdown,
    ModelConfigInfo,
    Suggestion,
)


GIB = 1024.0 ** 3

BYTES_PER_WEIGHT: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
    "marlin": 0.5,
    "bitsandbytes": 0.5,
    "gguf": 0.5,
}

KV_DTYPE_BYTES: dict[str, float] = {
    "auto": 2.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
}

WEIGHT_OVERHEAD = 1.15
RUNTIME_GB = 2.0


def bytes_per_weight(dtype: str, quantization: str) -> float:
    """Pick the smaller of dtype and quantization bytes-per-weight."""
    q = (quantization or "none").lower()
    d = (dtype or "auto").lower()
    if q != "none" and q in BYTES_PER_WEIGHT:
        return BYTES_PER_WEIGHT[q]
    if d == "auto":
        return BYTES_PER_WEIGHT["bfloat16"]
    return BYTES_PER_WEIGHT.get(d, BYTES_PER_WEIGHT["bfloat16"])


def kv_dtype_bytes(kv_cache_dtype: str, dtype: str) -> float:
    if kv_cache_dtype and kv_cache_dtype != "auto":
        return KV_DTYPE_BYTES.get(kv_cache_dtype, 2.0)
    if dtype in ("float16", "bfloat16"):
        return 2.0
    if dtype == "float32":
        return 4.0
    return 2.0


def approximate_param_count_from_config(cfg: ModelConfigInfo) -> Optional[float]:
    """Rough parameter count estimate for a dense decoder-only transformer.

    ``params ~= 12 * L * H^2 + vocab * H``. This is a deliberate
    approximation: it ignores MoE expert counts, tied embeddings, vision
    towers, adapters, and similar. We mark the result as estimated and ask
    the user for an explicit count when accuracy matters.
    """
    if cfg.num_hidden_layers and cfg.hidden_size:
        L = cfg.num_hidden_layers
        H = cfg.hidden_size
        V = cfg.vocab_size or 32_000
        return 12.0 * L * (H ** 2) + V * H
    return None


def resolve_param_count(req: DeploymentRequest) -> tuple[Optional[float], bool, Optional[str]]:
    """Return (param_count, is_estimate, note)."""
    if req.manual_param_count_b and req.manual_param_count_b > 0:
        return req.manual_param_count_b * 1e9, False, None

    cfg = req.config_info
    if cfg is None:
        return None, True, "No config.json uploaded and no manual parameter count provided."

    if cfg.estimated_param_count:
        is_estimate = bool(cfg.is_uncertain or cfg.is_moe or cfg.is_multimodal)
        note = None
        if cfg.is_moe:
            note = "MoE detected - param count is approximate; enter explicit count for accuracy."
        elif cfg.is_multimodal:
            note = "Multimodal model - param count is approximate; enter explicit count for accuracy."
        return cfg.estimated_param_count, is_estimate, note

    estimate = approximate_param_count_from_config(cfg)
    if estimate is None:
        return None, True, "Insufficient config.json fields to estimate parameter count."
    return estimate, True, "Parameter count estimated from config.json (rough)."


def max_model_len(req: DeploymentRequest) -> int:
    return int(req.input_tokens + req.output_tokens)


def estimate_memory(req: DeploymentRequest) -> MemoryBreakdown:
    tp = max(1, req.tensor_parallel_size)
    pp = max(1, req.pipeline_parallel_size)
    bpw = bytes_per_weight(req.dtype, req.quantization)
    kv_b = kv_dtype_bytes(req.kv_cache_dtype, req.dtype)

    params, is_estimate, note = resolve_param_count(req)
    cfg = req.config_info

    if params is not None:
        weight_gb = (params * bpw * WEIGHT_OVERHEAD) / GIB / (tp * pp)
    else:
        weight_gb = 0.0

    kv_cache_gb = 0.0
    if cfg and cfg.num_hidden_layers and cfg.head_dim:
        num_layers = cfg.num_hidden_layers
        kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads or 1
        head_dim = cfg.head_dim
        kv_heads_per_gpu = max(1, math.ceil(kv_heads / tp))
        kv_bpt = 2 * num_layers * kv_heads_per_gpu * head_dim * kv_b
        ml = max_model_len(req)
        kv_cache_gb = (kv_bpt * ml * req.max_num_seqs) / GIB

    runtime_gb = RUNTIME_GB
    total = weight_gb + kv_cache_gb + runtime_gb

    usable = req.gpu_memory_gb * req.gpu_memory_utilization
    if req.cpu_offload_gb:
        usable += float(req.cpu_offload_gb)
    percent = (total / usable) * 100.0 if usable > 0 else 0.0

    if params is None and kv_cache_gb == 0:
        fit: FitStatus = "unknown"
    elif total > usable:
        fit = "oom"
    elif total >= 0.85 * usable:
        fit = "risky"
    else:
        fit = "good"

    param_count_b = (params / 1e9) if params is not None else None

    return MemoryBreakdown(
        weight_gb=round(weight_gb, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        runtime_gb=round(runtime_gb, 2),
        total_required_gb=round(total, 2),
        usable_gb=round(usable, 2),
        gpu_total_gb=round(req.gpu_memory_gb, 2),
        percent_used=round(percent, 1),
        fit_status=fit,
        is_estimate=is_estimate,
        bytes_per_weight=bpw,
        kv_dtype_bytes=kv_b,
        param_count_b=round(param_count_b, 3) if param_count_b else None,
        note=note,
    )


def build_suggestions(req: DeploymentRequest, mem: MemoryBreakdown) -> list[Suggestion]:
    """Return ordered, actionable fixes when fit is risky/OOM."""
    if mem.fit_status not in ("risky", "oom"):
        return []

    out: list[Suggestion] = []
    ml = max_model_len(req)

    if mem.kv_cache_gb > 0.4 * mem.total_required_gb and ml > 4096:
        target = max(2048, (req.input_tokens + req.output_tokens) // 2)
        out.append(
            Suggestion(
                title=f"Reduce max context to ~{target} tokens",
                detail=(
                    f"KV cache is {mem.kv_cache_gb:.1f} GiB. Halving "
                    f"input+output tokens roughly halves the KV cache."
                ),
            )
        )

    if req.max_num_seqs > 8 and mem.kv_cache_gb > 0.3 * mem.total_required_gb:
        target = max(4, req.max_num_seqs // 2)
        out.append(
            Suggestion(
                title=f"Reduce max concurrent sequences to {target}",
                detail="KV cache scales linearly with the number of sequences vLLM batches in parallel.",
            )
        )

    if req.quantization == "none" and mem.weight_gb > 0.3 * mem.total_required_gb:
        out.append(
            Suggestion(
                title="Switch to AWQ, GPTQ or BitsAndBytes 4-bit quantization",
                detail="4-bit quantization roughly quarters weight VRAM compared to bfloat16.",
            )
        )

    if req.tensor_parallel_size < req.gpu_count:
        out.append(
            Suggestion(
                title=f"Increase tensor-parallel-size to {req.gpu_count}",
                detail="Splitting the model across more GPUs reduces per-GPU weight and KV memory.",
            )
        )

    if not req.cpu_offload_gb:
        out.append(
            Suggestion(
                title="Enable CPU offload (e.g. cpu-offload-gb=8)",
                detail="vLLM can offload some weights to CPU RAM at the cost of throughput.",
            )
        )

    if req.gpu_memory_utilization < 0.92 and mem.fit_status == "risky":
        out.append(
            Suggestion(
                title=f"Raise gpu-memory-utilization to 0.92 from {req.gpu_memory_utilization:.2f}",
                detail="Only do this on a dedicated GPU with no other processes competing for VRAM.",
            )
        )

    return out
