"""SQLite-backed storage for saved Easy-vLLM deployments.

A tiny stdlib-only persistence layer so users can revisit, copy, or duplicate
past deployments without re-typing the entire wizard. The DB lives at
``instance/easy_vllm.db`` next to the Flask app and is created on first use.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schemas import DeploymentRequest, MemoryBreakdown, WarningItem


DEFAULT_DB_PATH = Path("instance") / "easy_vllm.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deployments (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    model_id        TEXT,
    gpu_preset      TEXT,
    gpu_count       INTEGER,
    quantization    TEXT,
    fit_status      TEXT,
    percent_used    REAL,
    created_at      TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    artifacts_json  TEXT NOT NULL,
    memory_json     TEXT NOT NULL,
    warnings_json   TEXT NOT NULL,
    command_oneline TEXT NOT NULL,
    command_multiline TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS deployments_created_at_idx
    ON deployments (created_at);
"""


def _resolve_db_path(db_path: str | os.PathLike[str] | None) -> str:
    if db_path is None:
        return str(DEFAULT_DB_PATH)
    if str(db_path) == ":memory:":
        return ":memory:"
    return str(db_path)


def _connect(db_path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    resolved = _resolve_db_path(db_path)
    if resolved != ":memory:":
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | os.PathLike[str] | None = None) -> None:
    """Create the deployments table if it doesn't exist yet."""
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _derive_name(req: DeploymentRequest) -> str:
    if req.served_model_name:
        return req.served_model_name
    raw = req.model_id or req.local_model_path or "deployment"
    base = raw.rstrip("/").split("/")[-1] or "deployment"
    return base.lower().replace(" ", "-")


def save_deployment(
    req: DeploymentRequest,
    memory: MemoryBreakdown,
    warnings: list[WarningItem],
    artifacts: dict[str, str],
    command_oneline: str,
    command_multiline: str,
    db_path: str | os.PathLike[str] | None = None,
) -> str:
    """Persist a generated deployment and return its new id."""
    new_id = uuid.uuid4().hex[:12]
    name = _derive_name(req)
    record = {
        "id": new_id,
        "name": name,
        "model_id": req.model_id or req.local_model_path or "",
        "gpu_preset": req.gpu_preset,
        "gpu_count": req.gpu_count,
        "quantization": req.quantization,
        "fit_status": memory.fit_status,
        "percent_used": memory.percent_used,
        "created_at": _now_iso(),
        "request_json": req.model_dump_json(),
        "artifacts_json": json.dumps(artifacts),
        "memory_json": memory.model_dump_json(),
        "warnings_json": json.dumps([w.model_dump() for w in warnings]),
        "command_oneline": command_oneline,
        "command_multiline": command_multiline,
    }
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO deployments (
                id, name, model_id, gpu_preset, gpu_count, quantization,
                fit_status, percent_used, created_at,
                request_json, artifacts_json, memory_json, warnings_json,
                command_oneline, command_multiline
            ) VALUES (
                :id, :name, :model_id, :gpu_preset, :gpu_count, :quantization,
                :fit_status, :percent_used, :created_at,
                :request_json, :artifacts_json, :memory_json, :warnings_json,
                :command_oneline, :command_multiline
            )
            """,
            record,
        )
        conn.commit()
    finally:
        conn.close()
    return new_id


def list_deployments(
    limit: int = 100,
    db_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a list of summary rows ordered newest-first."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, name, model_id, gpu_preset, gpu_count, quantization,
                   fit_status, percent_used, created_at
            FROM deployments
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_deployment(
    deployment_id: str,
    db_path: str | os.PathLike[str] | None = None,
) -> Optional[dict[str, Any]]:
    """Return the full deployment record (with parsed JSON fields) or ``None``."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None

    out = dict(row)
    out["request"] = json.loads(out.pop("request_json"))
    out["artifacts"] = json.loads(out.pop("artifacts_json"))
    out["memory"] = json.loads(out.pop("memory_json"))
    out["warnings"] = json.loads(out.pop("warnings_json"))
    return out


def delete_deployment(
    deployment_id: str,
    db_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Delete a deployment and return ``True`` if a row was removed."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM deployments WHERE id = ?", (deployment_id,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
