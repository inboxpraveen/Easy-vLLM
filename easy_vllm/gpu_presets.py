"""Canonical GPU preset list for the hardware step dropdown.

Values are the manufacturer's nominal VRAM in GiB. Users can always fall back
to the ``custom`` preset and type any number.
"""
from __future__ import annotations

from .schemas import GpuPreset


GPU_PRESETS: list[GpuPreset] = [
    GpuPreset(
        id="rtx_3090_24gb",
        label="NVIDIA RTX 3090 (24 GB)",
        vram_gb=24.0,
        family="Ampere consumer",
    ),
    GpuPreset(
        id="rtx_4090_24gb",
        label="NVIDIA RTX 4090 (24 GB)",
        vram_gb=24.0,
        family="Ada consumer",
    ),
    GpuPreset(
        id="rtx_5090_32gb",
        label="NVIDIA RTX 5090 (32 GB)",
        vram_gb=32.0,
        family="Blackwell consumer",
    ),
    GpuPreset(
        id="l4_24gb",
        label="NVIDIA L4 (24 GB)",
        vram_gb=24.0,
        family="Ada datacenter",
    ),
    GpuPreset(
        id="l40s_48gb",
        label="NVIDIA L40S (48 GB)",
        vram_gb=48.0,
        family="Ada datacenter",
    ),
    GpuPreset(
        id="rtx_6000_ada_48gb",
        label="NVIDIA RTX 6000 Ada (48 GB)",
        vram_gb=48.0,
        family="Ada workstation",
    ),
    GpuPreset(
        id="a100_40gb",
        label="NVIDIA A100 (40 GB)",
        vram_gb=40.0,
        family="Ampere datacenter",
    ),
    GpuPreset(
        id="a100_80gb",
        label="NVIDIA A100 (80 GB)",
        vram_gb=80.0,
        family="Ampere datacenter",
    ),
    GpuPreset(
        id="h100_80gb",
        label="NVIDIA H100 (80 GB)",
        vram_gb=80.0,
        family="Hopper datacenter",
        notes="FP8 W8A8 supported",
    ),
    GpuPreset(
        id="h200_141gb",
        label="NVIDIA H200 (141 GB)",
        vram_gb=141.0,
        family="Hopper datacenter",
        notes="FP8 W8A8 supported",
    ),
    GpuPreset(
        id="b200_192gb",
        label="NVIDIA B200 (192 GB)",
        vram_gb=192.0,
        family="Blackwell datacenter",
    ),
    GpuPreset(
        id="mi300x_192gb",
        label="AMD MI300X (192 GB)",
        vram_gb=192.0,
        family="AMD CDNA3",
        notes="ROCm image required",
    ),
    GpuPreset(
        id="custom",
        label="Custom (enter VRAM manually)",
        vram_gb=24.0,
        family="custom",
    ),
]


PRESET_BY_ID: dict[str, GpuPreset] = {p.id: p for p in GPU_PRESETS}


def get_preset(preset_id: str) -> GpuPreset | None:
    return PRESET_BY_ID.get(preset_id)
