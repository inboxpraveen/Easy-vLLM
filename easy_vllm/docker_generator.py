"""Render artifact templates into a dict of file content."""
from __future__ import annotations

import secrets
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .command_builder import build_command_strings, _resolved_served_name
from .memory_estimator import estimate_memory, max_model_len
from .schemas import DeploymentRequest, EstimateResult, WarningItem


_TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "artifacts"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(default=False),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
)


ARTIFACT_FILES = (
    ("docker-compose.yml", "docker-compose.yml.j2"),
    (".env", "env.j2"),
    ("test_client.py", "test_client.py.j2"),
    ("test_curl.sh", "test_curl.sh.j2"),
    ("README.md", "README.generated.md.j2"),
    ("config_summary.json", "config_summary.json.j2"),
)


def _render_context(req: DeploymentRequest, warnings: list[WarningItem]) -> dict:
    oneline, multiline = build_command_strings(req)
    mem = estimate_memory(req)
    summary = {
        "model": {
            "model_source": req.model_source,
            "model_id": req.model_id,
            "local_model_path": req.local_model_path,
            "served_model_name": _resolved_served_name(req),
            "is_private_hf_model": req.is_private_hf_model,
            "trust_remote_code": req.trust_remote_code,
            "config_info": req.config_info.model_dump() if req.config_info else None,
            "manual_param_count_b": req.manual_param_count_b,
        },
        "hardware": {
            "gpu_preset": req.gpu_preset,
            "gpu_memory_gb": req.gpu_memory_gb,
            "gpu_count": req.gpu_count,
            "tensor_parallel_size": req.tensor_parallel_size,
            "pipeline_parallel_size": req.pipeline_parallel_size,
            "gpu_memory_utilization": req.gpu_memory_utilization,
        },
        "workload": {
            "input_tokens": req.input_tokens,
            "output_tokens": req.output_tokens,
            "max_num_seqs": req.max_num_seqs,
            "max_model_len": max_model_len(req),
        },
        "optimization": {
            "dtype": req.dtype,
            "quantization": req.quantization,
            "enable_prefix_caching": req.enable_prefix_caching,
            "kv_cache_dtype": req.kv_cache_dtype,
            "max_num_batched_tokens": req.max_num_batched_tokens,
            "cpu_offload_gb": req.cpu_offload_gb,
            "load_format": req.load_format,
            "generation_config_vllm": req.generation_config_vllm,
            "api_key_required": req.api_key_required,
            "image_tag": req.image_tag,
            "extra_flags": req.extra_flags,
        },
        "memory_estimate": mem.model_dump(),
        "warnings": [w.model_dump() for w in warnings],
    }

    return {
        "vllm_command_oneline": oneline,
        "vllm_command_multiline": multiline,
        "image_tag": req.image_tag,
        "served_model_name": _resolved_served_name(req),
        "is_private_hf_model": req.is_private_hf_model,
        "api_key_required": req.api_key_required,
        "gpu_count": req.gpu_count,
        "gpu_memory_utilization": req.gpu_memory_utilization,
        "max_model_len": max_model_len(req),
        "max_num_seqs": req.max_num_seqs,
        "quantization": req.quantization,
        "memory": mem.model_dump(),
        "warnings": [w.model_dump() for w in warnings],
        "random_token": secrets.token_hex(8),
        "summary": summary,
    }


def render_artifacts(req: DeploymentRequest, warnings: list[WarningItem]) -> dict[str, str]:
    """Render every artifact and return ``{filename: content}``."""
    ctx = _render_context(req, warnings)
    out: dict[str, str] = {}
    for filename, template_name in ARTIFACT_FILES:
        template = _env.get_template(template_name)
        out[filename] = template.render(**ctx)
    return out


def build_estimate_result(req: DeploymentRequest, warnings: list[WarningItem]) -> EstimateResult:
    from .memory_estimator import build_suggestions

    mem = estimate_memory(req)
    suggestions = build_suggestions(req, mem)
    oneline, multiline = build_command_strings(req)

    return EstimateResult(
        memory=mem,
        warnings=warnings,
        suggestions=suggestions,
        vllm_command_oneline=oneline,
        vllm_command_multiline=multiline,
        max_model_len=max_model_len(req),
        served_model_name=_resolved_served_name(req),
        image_tag=req.image_tag,
    )
