"""Parse a Hugging Face ``config.json`` into :class:`ModelConfigInfo`.

The parser is conservative: when it can't be confident about parameter count
(MoE, multimodal, tied embeddings, custom architectures) it sets ``is_uncertain``
and lets the wizard ask the user for an explicit number.
"""
from __future__ import annotations

import json
from typing import Any

from .schemas import ModelConfigInfo


MOE_HINT_KEYS = {
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
}

MULTIMODAL_HINT_KEYS = {
    "vision_config",
    "vision_tower",
    "image_token_id",
    "audio_config",
    "audio_token_id",
    "video_config",
}

MULTIMODAL_ARCH_HINTS = (
    "VisionLanguage",
    "MultiModal",
    "Vision",
    "AudioLanguage",
    "Vlm",
    "VLM",
)


def _is_moe(raw: dict[str, Any]) -> bool:
    if any(k in raw for k in MOE_HINT_KEYS):
        return True
    text = json.dumps(raw).lower()
    return "moe" in text or "mixture_of_experts" in text


def _is_multimodal(raw: dict[str, Any]) -> bool:
    if any(k in raw for k in MULTIMODAL_HINT_KEYS):
        return True
    archs = raw.get("architectures") or []
    if isinstance(archs, list):
        for a in archs:
            if isinstance(a, str) and any(h in a for h in MULTIMODAL_ARCH_HINTS):
                return True
    return False


def _coerce_head_dim(raw: dict[str, Any]) -> int | None:
    head_dim = raw.get("head_dim")
    if head_dim:
        return int(head_dim)
    h = raw.get("hidden_size")
    n_head = raw.get("num_attention_heads")
    if h and n_head:
        return int(h) // int(n_head)
    return None


def _approx_param_count(raw: dict[str, Any]) -> float | None:
    L = raw.get("num_hidden_layers")
    H = raw.get("hidden_size")
    V = raw.get("vocab_size") or 32_000
    if not (L and H):
        return None
    return 12.0 * int(L) * (int(H) ** 2) + int(V) * int(H)


def parse_config_dict(raw: dict[str, Any]) -> ModelConfigInfo:
    notes: list[str] = []

    is_moe = _is_moe(raw)
    is_multimodal = _is_multimodal(raw)
    is_uncertain = is_moe or is_multimodal

    head_dim = _coerce_head_dim(raw)
    estimated_params = _approx_param_count(raw)

    if is_moe:
        notes.append(
            "Mixture-of-Experts model detected - parameter count from "
            "config.json is approximate. Enter explicit total parameter "
            "count for an accurate memory estimate."
        )
    if is_multimodal:
        notes.append(
            "Multimodal architecture detected - vision/audio towers add "
            "parameters. Enter explicit parameter count for accuracy."
        )
    if raw.get("tie_word_embeddings"):
        notes.append("Tied word embeddings - estimate slightly overcounts.")
    if raw.get("quantization_config"):
        notes.append("Pre-quantized weights detected via quantization_config.")

    return ModelConfigInfo(
        model_type=raw.get("model_type"),
        architectures=raw.get("architectures") or [],
        hidden_size=raw.get("hidden_size"),
        intermediate_size=raw.get("intermediate_size"),
        num_hidden_layers=raw.get("num_hidden_layers"),
        num_attention_heads=raw.get("num_attention_heads"),
        num_key_value_heads=raw.get("num_key_value_heads") or raw.get("num_attention_heads"),
        head_dim=head_dim,
        vocab_size=raw.get("vocab_size"),
        max_position_embeddings=raw.get("max_position_embeddings"),
        torch_dtype=raw.get("torch_dtype"),
        quantization_config=raw.get("quantization_config"),
        is_moe=is_moe,
        is_multimodal=is_multimodal,
        is_uncertain=is_uncertain,
        estimated_param_count=estimated_params,
        notes=notes,
    )


def parse_config_bytes(data: bytes | str) -> ModelConfigInfo:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError("config.json must be a JSON object")
    return parse_config_dict(raw)
