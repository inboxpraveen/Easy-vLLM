"""Cross-field validators producing actionable warnings.

These validators don't reject input; they surface a list of human-readable
warnings the live panel renders as colored chips. Hard errors (status=
``error``) prevent the final ``/api/generate`` call from succeeding.
"""
from __future__ import annotations

from .schemas import DeploymentRequest, WarningItem


GATED_NAMESPACES = (
    "meta-llama/",
    "google/gemma",
    "mistralai/",
    "tiiuae/falcon",
    "Qwen/Qwen3-",
    "deepseek-ai/",
)


def validate(req: DeploymentRequest) -> list[WarningItem]:
    out: list[WarningItem] = []

    if not (req.model_id or req.local_model_path):
        out.append(
            WarningItem(
                severity="info",
                message="Enter a Hugging Face model ID or local model path to continue.",
                field="model_id",
            )
        )

    if req.tensor_parallel_size > req.gpu_count:
        out.append(
            WarningItem(
                severity="error",
                message=(
                    f"tensor-parallel-size ({req.tensor_parallel_size}) cannot "
                    f"exceed GPU count ({req.gpu_count})."
                ),
                field="tensor_parallel_size",
            )
        )

    if req.tensor_parallel_size > 1 and req.gpu_count % req.tensor_parallel_size != 0:
        out.append(
            WarningItem(
                severity="warning",
                message=(
                    f"GPU count ({req.gpu_count}) should be a multiple of "
                    f"tensor-parallel-size ({req.tensor_parallel_size})."
                ),
                field="tensor_parallel_size",
            )
        )

    if req.gpu_memory_utilization > 0.95:
        out.append(
            WarningItem(
                severity="warning",
                message=(
                    f"gpu-memory-utilization {req.gpu_memory_utilization:.2f} "
                    "is risky unless this GPU has nothing else running on it."
                ),
                field="gpu_memory_utilization",
            )
        )

    if req.trust_remote_code:
        out.append(
            WarningItem(
                severity="warning",
                message="trust-remote-code executes the model repo's Python on load. Only enable for repos you trust.",
                field="trust_remote_code",
            )
        )

    cfg = req.config_info
    if req.quantization in ("awq", "gptq") and cfg is not None:
        qc = cfg.quantization_config or {}
        method = (qc.get("quant_method") or "").lower()
        if not method or (req.quantization not in method):
            out.append(
                WarningItem(
                    severity="warning",
                    message=(
                        f"You picked --quantization {req.quantization} but the "
                        "uploaded config.json doesn't show a matching "
                        "quantization_config. Make sure the model is actually "
                        f"pre-quantized to {req.quantization.upper()}."
                    ),
                    field="quantization",
                )
            )

    if req.quantization == "bitsandbytes":
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "BitsAndBytes needs `bitsandbytes>=0.49.2` installed in the "
                    "container. The official vLLM image does not ship it - see "
                    "the generated README for a one-line custom Dockerfile."
                ),
                field="quantization",
            )
        )

    if req.quantization == "gguf":
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "GGUF deployment is advanced; you'll typically also need "
                    "to specify a tokenizer model ID alongside the GGUF file."
                ),
                field="quantization",
            )
        )

    model_ref = (req.model_id or "").strip()
    if model_ref and not req.is_private_hf_model:
        if any(model_ref.startswith(ns) for ns in GATED_NAMESPACES):
            out.append(
                WarningItem(
                    severity="warning",
                    message=(
                        f"'{model_ref}' looks like a gated Hugging Face repo. "
                        "Toggle 'Private HF model' so the generated .env "
                        "includes an HF_TOKEN slot."
                    ),
                    field="is_private_hf_model",
                )
            )

    if req.pipeline_parallel_size > 1:
        out.append(
            WarningItem(
                severity="warning",
                message=(
                    "Pipeline parallelism usually implies multi-node deployment "
                    "with Ray and NCCL/IB networking. The generated compose "
                    "file targets a single host - additional setup is required."
                ),
                field="pipeline_parallel_size",
            )
        )

    if req.image_tag.endswith(":latest"):
        out.append(
            WarningItem(
                severity="info",
                message="Pin a specific vLLM image tag (e.g. vllm/vllm-openai:v0.11.0) for reproducible deployments.",
                field="image_tag",
            )
        )

    if req.enforce_eager:
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "--enforce-eager disables CUDA graph capture. Saves a bit "
                    "of VRAM and helps debugging, but lowers throughput."
                ),
                field="enforce_eager",
            )
        )

    if req.swap_space_gb is not None and req.swap_space_gb > 0:
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "--swap-space is honored on V0; vLLM V1 no longer swaps KV "
                    "cache between GPU and CPU - this flag may be ignored."
                ),
                field="swap_space_gb",
            )
        )

    if req.enable_lora and not req.max_loras:
        out.append(
            WarningItem(
                severity="warning",
                message=(
                    "LoRA is enabled but max-loras is not set. Set a positive "
                    "value (e.g. 4) so vLLM can pre-allocate slots."
                ),
                field="max_loras",
            )
        )

    if req.speculative_method != "none":
        if req.speculative_method in {"draft_model", "mtp", "eagle3"} and not req.speculative_model:
            out.append(
                WarningItem(
                    severity="error",
                    message=(
                        f"Speculative method '{req.speculative_method}' "
                        "requires a draft model id."
                    ),
                    field="speculative_model",
                )
            )
        if not req.num_speculative_tokens:
            out.append(
                WarningItem(
                    severity="warning",
                    message=(
                        "Speculative decoding is on but num-speculative-tokens "
                        "is not set. vLLM may pick a default; specifying it is "
                        "safer."
                    ),
                    field="num_speculative_tokens",
                )
            )
        if req.pipeline_parallel_size and req.pipeline_parallel_size > 1:
            out.append(
                WarningItem(
                    severity="warning",
                    message=(
                        "Speculative decoding is not composable with pipeline "
                        "parallelism in older vLLM releases - test before "
                        "shipping."
                    ),
                    field="speculative_method",
                )
            )

    if req.async_scheduling and req.scheduling_policy == "priority":
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "Combining --async-scheduling with priority scheduling is "
                    "still experimental in vLLM. Verify behavior under load."
                ),
                field="async_scheduling",
            )
        )

    if req.enable_auto_tool_choice and not req.tool_call_parser:
        out.append(
            WarningItem(
                severity="warning",
                message=(
                    "Auto tool choice needs a --tool-call-parser. Pick one "
                    "matching your model (e.g. hermes, llama3_json, mistral)."
                ),
                field="tool_call_parser",
            )
        )

    if req.chat_template:
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "--chat-template expects a path inside the container. "
                    "Mount the file via a volume in docker-compose.yml."
                ),
                field="chat_template",
            )
        )

    if req.data_parallel_size and req.data_parallel_size > 1:
        out.append(
            WarningItem(
                severity="info",
                message=(
                    "Data parallelism replicates the model across GPUs. Make "
                    "sure (data-parallel * tensor-parallel) <= total GPUs."
                ),
                field="data_parallel_size",
            )
        )

    return out


def has_blocking_errors(warnings: list[WarningItem]) -> bool:
    return any(w.severity == "error" for w in warnings)
