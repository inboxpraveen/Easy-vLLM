"""Smoke tests for the Flask routes - exercises the full pipeline end-to-end."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.data.decode()
    assert "easy-vLLM" in body or "easy-" in body
    assert "Live estimate" in body


def test_gpu_presets_endpoint(client):
    res = client.get("/api/gpu-presets")
    assert res.status_code == 200
    data = res.get_json()
    assert any(p["id"] == "rtx_4090_24gb" for p in data)
    assert any(p["id"] == "h100_80gb" for p in data)


def test_parse_config_endpoint(client):
    payload = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    }
    data = io.BytesIO(json.dumps(payload).encode())
    res = client.post(
        "/api/parse-config",
        data={"config": (data, "config.json")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    out = res.get_json()
    assert out["model_type"] == "qwen3"
    assert out["num_hidden_layers"] == 32
    assert out["is_uncertain"] is False


def test_estimate_endpoint_returns_memory_breakdown(client):
    body = {
        "model_id": "Qwen/Qwen3-8B-Instruct",
        "manual_param_count_b": 8.0,
        "gpu_memory_gb": 24.0,
        "input_tokens": 4096,
        "output_tokens": 1024,
        "max_num_seqs": 16,
        "dtype": "bfloat16",
    }
    res = client.post("/api/estimate", json=body)
    assert res.status_code == 200
    data = res.get_json()
    assert "memory" in data
    assert "vllm_command_oneline" in data
    assert "--model Qwen/Qwen3-8B-Instruct" in data["vllm_command_oneline"]
    assert data["memory"]["fit_status"] in ("good", "risky", "oom")


def test_generate_endpoint_returns_zip(client):
    body = {
        "model_id": "Qwen/Qwen3-8B-Instruct",
        "manual_param_count_b": 8.0,
        "gpu_memory_gb": 24.0,
        "quantization": "awq",
        "input_tokens": 2048,
        "output_tokens": 512,
        "max_num_seqs": 8,
    }
    res = client.post("/api/generate", json=body)
    assert res.status_code == 200
    assert res.mimetype == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(res.data))
    names = {n.split("/", 1)[1] for n in zf.namelist()}
    assert "docker-compose.yml" in names
    assert ".env" in names
    assert "test_client.py" in names
    assert "test_curl.sh" in names
    assert "README.md" in names
    assert "config_summary.json" in names

    compose = zf.read("easy-vllm-output/docker-compose.yml").decode()
    assert "vllm/vllm-openai" in compose
    assert "--model Qwen/Qwen3-8B-Instruct" in compose
    assert "--quantization awq" in compose

    summary = json.loads(zf.read("easy-vllm-output/config_summary.json"))
    assert summary["model"]["model_id"] == "Qwen/Qwen3-8B-Instruct"
    assert summary["optimization"]["quantization"] == "awq"


def test_generate_endpoint_blocks_on_validation_error(client):
    body = {
        "model_id": "x/y",
        "tensor_parallel_size": 4,
        "gpu_count": 1,
        "manual_param_count_b": 1.0,
    }
    res = client.post("/api/generate", json=body)
    assert res.status_code == 400
    data = res.get_json()
    assert data["error"] == "blocking_validation_errors"
    assert any(w["severity"] == "error" for w in data["warnings"])
