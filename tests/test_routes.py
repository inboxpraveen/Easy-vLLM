"""Smoke tests for the Flask routes - exercises the full pipeline end-to-end."""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile

import pytest

from app import create_app


@pytest.fixture()
def client(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="easy-vllm-test-")
    monkeypatch.setenv("EASY_VLLM_DB", os.path.join(tmpdir, "test.db"))
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.data.decode()
    assert "Easy-vLLM" in body or "easy-" in body
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


def test_generate_endpoint_saves_and_returns_record(client):
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
    data = res.get_json()

    assert data["id"]
    assert data["name"]
    assert data["model_id"] == "Qwen/Qwen3-8B-Instruct"
    assert data["quantization"] == "awq"
    assert data["download_url"].endswith("/zip")

    artifacts = data["artifacts"]
    assert "docker-compose.yml" in artifacts
    assert ".env" in artifacts
    assert "test_client.py" in artifacts
    assert "test_curl.sh" in artifacts
    assert "README.md" in artifacts
    assert "config_summary.json" in artifacts

    assert "vllm/vllm-openai" in artifacts["docker-compose.yml"]
    assert "--model Qwen/Qwen3-8B-Instruct" in artifacts["docker-compose.yml"]
    assert "--quantization awq" in artifacts["docker-compose.yml"]


def test_generate_then_list_get_and_zip(client):
    body = {
        "model_id": "Qwen/Qwen3-8B-Instruct",
        "manual_param_count_b": 8.0,
        "gpu_memory_gb": 24.0,
        "input_tokens": 1024,
        "output_tokens": 256,
        "max_num_seqs": 4,
    }
    res = client.post("/api/generate", json=body)
    assert res.status_code == 200
    deployment_id = res.get_json()["id"]

    # list
    res2 = client.get("/api/deployments")
    assert res2.status_code == 200
    rows = res2.get_json()
    assert any(r["id"] == deployment_id for r in rows)

    # get
    res3 = client.get(f"/api/deployments/{deployment_id}")
    assert res3.status_code == 200
    rec = res3.get_json()
    assert rec["id"] == deployment_id
    assert "request" in rec and "artifacts" in rec
    assert "memory" in rec
    assert "command_oneline" in rec

    # zip
    res4 = client.get(f"/api/deployments/{deployment_id}/zip")
    assert res4.status_code == 200
    assert res4.mimetype == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(res4.data))
    names = {n.split("/", 1)[1] for n in zf.namelist()}
    assert {"docker-compose.yml", ".env", "README.md"}.issubset(names)


def test_delete_deployment(client):
    body = {
        "model_id": "x/y",
        "manual_param_count_b": 1.0,
        "gpu_memory_gb": 24.0,
    }
    deployment_id = client.post("/api/generate", json=body).get_json()["id"]
    res = client.delete(f"/api/deployments/{deployment_id}")
    assert res.status_code == 200
    assert res.get_json()["ok"] is True
    # second delete returns 404
    res2 = client.delete(f"/api/deployments/{deployment_id}")
    assert res2.status_code == 404


def test_get_unknown_deployment_returns_404(client):
    res = client.get("/api/deployments/does-not-exist")
    assert res.status_code == 404


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
