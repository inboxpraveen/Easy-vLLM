"""Smoke tests for the SQLite-backed deployment history."""
from __future__ import annotations

import os
import tempfile

import pytest

from easy_vllm.schemas import DeploymentRequest, MemoryBreakdown, WarningItem
from easy_vllm.storage import (
    delete_deployment,
    get_deployment,
    init_db,
    list_deployments,
    save_deployment,
)


@pytest.fixture()
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.db")
        init_db(path)
        yield path


def _fixture_inputs():
    req = DeploymentRequest(
        model_id="Qwen/Qwen3-8B-Instruct",
        served_model_name="qwen3-8b",
        manual_param_count_b=8.0,
        gpu_preset="rtx_4090_24gb",
        gpu_count=1,
        quantization="awq",
    )
    mem = MemoryBreakdown(
        weight_gb=4.6, kv_cache_gb=2.1, runtime_gb=2.0,
        total_required_gb=8.7, usable_gb=21.6, gpu_total_gb=24.0,
        percent_used=40.3, fit_status="good",
    )
    warnings = [WarningItem(severity="info", message="all good")]
    artifacts = {
        "docker-compose.yml": "services: {}",
        ".env": "VLLM_PORT=8000",
        "test_client.py": "print('hi')",
        "test_curl.sh": "#!/bin/sh",
        "README.md": "# generated",
        "config_summary.json": "{}",
    }
    return req, mem, warnings, artifacts


def test_save_and_get_roundtrip(db_path):
    req, mem, warnings, artifacts = _fixture_inputs()
    new_id = save_deployment(
        req=req, memory=mem, warnings=warnings, artifacts=artifacts,
        command_oneline="--model x", command_multiline="--model x \\\n--port 8000",
        db_path=db_path,
    )
    assert new_id

    rec = get_deployment(new_id, db_path=db_path)
    assert rec is not None
    assert rec["id"] == new_id
    assert rec["name"] == "qwen3-8b"
    assert rec["model_id"] == "Qwen/Qwen3-8B-Instruct"
    assert rec["fit_status"] == "good"
    assert rec["memory"]["fit_status"] == "good"
    assert rec["request"]["model_id"] == "Qwen/Qwen3-8B-Instruct"
    assert rec["artifacts"]["docker-compose.yml"] == "services: {}"
    assert rec["warnings"][0]["severity"] == "info"


def test_list_returns_newest_first(db_path):
    import time

    req, mem, warnings, artifacts = _fixture_inputs()
    a = save_deployment(req, mem, warnings, artifacts, "a", "a", db_path=db_path)
    time.sleep(0.005)  # ensure distinct microsecond-precision timestamps on Windows
    b = save_deployment(req, mem, warnings, artifacts, "b", "b", db_path=db_path)
    rows = list_deployments(db_path=db_path)
    ids = [r["id"] for r in rows]
    assert a in ids and b in ids
    # newest first: b (later) should appear before a
    assert ids.index(b) < ids.index(a)


def test_delete_removes_row(db_path):
    req, mem, warnings, artifacts = _fixture_inputs()
    new_id = save_deployment(req, mem, warnings, artifacts, "c", "c", db_path=db_path)
    assert delete_deployment(new_id, db_path=db_path) is True
    assert get_deployment(new_id, db_path=db_path) is None
    assert delete_deployment(new_id, db_path=db_path) is False


def test_init_db_is_idempotent(db_path):
    init_db(db_path)
    init_db(db_path)  # second call must not raise
    assert list_deployments(db_path=db_path) == []
