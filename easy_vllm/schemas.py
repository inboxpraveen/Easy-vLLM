"""Pydantic schemas for the Easy-vLLM wizard.

These models intentionally accept lenient inputs (most fields optional) so the
live estimator can fire on every keystroke without forcing the user to fill
every field first. Hard validation only happens when generating the final
deployment artifacts.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


DType = Literal["auto", "bfloat16", "float16", "float32"]
Quantization = Literal[
    "none", "awq", "gptq", "fp8", "bitsandbytes", "gguf", "marlin"
]
KvCacheDType = Literal["auto", "fp16", "bf16", "fp8", "fp8_e4m3", "fp8_e5m2"]
LoadFormat = Literal["auto", "bitsandbytes", "gguf", "safetensors", "pt"]
FitStatus = Literal["good", "risky", "oom", "unknown"]
SeverityLevel = Literal["info", "warning", "danger", "error"]


class ModelConfigInfo(BaseModel):
    """Fields extracted from an uploaded Hugging Face ``config.json``."""

    model_config = ConfigDict(extra="ignore")

    model_type: Optional[str] = None
    architectures: list[str] = Field(default_factory=list)
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    torch_dtype: Optional[str] = None
    quantization_config: Optional[dict[str, Any]] = None

    is_moe: bool = False
    is_multimodal: bool = False
    is_uncertain: bool = False
    estimated_param_count: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class DeploymentRequest(BaseModel):
    """Full wizard state submitted from the front-end.

    All fields are optional during the live-estimate phase; the front-end fills
    them in progressively as the user advances through the wizard.
    """

    model_config = ConfigDict(extra="ignore")

    # Step 1 - Model
    model_source: Literal["huggingface", "local"] = "huggingface"
    model_id: Optional[str] = None
    local_model_path: Optional[str] = None
    served_model_name: Optional[str] = None
    is_private_hf_model: bool = False
    trust_remote_code: bool = False

    config_info: Optional[ModelConfigInfo] = None
    manual_param_count_b: Optional[float] = Field(
        default=None,
        description="Approximate parameter count in billions (used when no config.json).",
        ge=0.001,
        le=2000.0,
    )

    # Step 2 - Hardware & workload
    gpu_preset: str = "rtx_4090_24gb"
    gpu_memory_gb: float = Field(default=24.0, gt=0, le=2048)
    gpu_count: int = Field(default=1, ge=1, le=32)
    tensor_parallel_size: int = Field(default=1, ge=1, le=32)
    pipeline_parallel_size: int = Field(default=1, ge=1, le=16)
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=0.99)

    input_tokens: int = Field(default=4096, ge=1, le=2_000_000)
    output_tokens: int = Field(default=1024, ge=1, le=2_000_000)
    max_num_seqs: int = Field(default=32, ge=1, le=4096)

    # Step 3 - Optimization
    dtype: DType = "auto"
    quantization: Quantization = "none"
    enable_prefix_caching: bool = True

    # Advanced
    kv_cache_dtype: KvCacheDType = "auto"
    max_num_batched_tokens: Optional[int] = Field(default=None, ge=1, le=2_000_000)
    cpu_offload_gb: Optional[float] = Field(default=None, ge=0, le=1024)
    load_format: LoadFormat = "auto"
    generation_config_vllm: bool = True
    api_key_required: bool = False
    image_tag: str = "vllm/vllm-openai:latest"
    extra_flags: Optional[str] = None

    @field_validator("served_model_name", mode="before")
    @classmethod
    def _empty_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class MemoryBreakdown(BaseModel):
    weight_gb: float
    kv_cache_gb: float
    runtime_gb: float
    total_required_gb: float
    usable_gb: float
    gpu_total_gb: float
    percent_used: float
    fit_status: FitStatus
    is_estimate: bool = False
    bytes_per_weight: float = 0.0
    kv_dtype_bytes: float = 0.0
    param_count_b: Optional[float] = None
    note: Optional[str] = None


class WarningItem(BaseModel):
    severity: SeverityLevel
    message: str
    field: Optional[str] = None


class Suggestion(BaseModel):
    title: str
    detail: str


class EstimateResult(BaseModel):
    memory: MemoryBreakdown
    warnings: list[WarningItem] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    vllm_command_oneline: str
    vllm_command_multiline: str
    max_model_len: int
    served_model_name: str
    image_tag: str


class GpuPreset(BaseModel):
    id: str
    label: str
    vram_gb: float
    family: str
    notes: Optional[str] = None
