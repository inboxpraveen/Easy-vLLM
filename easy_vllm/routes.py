"""Flask routes for the easy-vLLM web app."""
from __future__ import annotations

from flask import Flask, Response, current_app, jsonify, render_template, request, send_file
from io import BytesIO
from pydantic import ValidationError

from .command_builder import build_command_strings
from .config_parser import parse_config_bytes
from .docker_generator import build_estimate_result, render_artifacts
from .gpu_presets import GPU_PRESETS
from .memory_estimator import estimate_memory
from .schemas import DeploymentRequest
from .storage import (
    delete_deployment,
    get_deployment,
    list_deployments,
    save_deployment,
)
from .validators import has_blocking_errors, validate
from .zip_exporter import build_zip


def _db_path() -> str:
    return current_app.config.get("EASY_VLLM_DB", "instance/easy_vllm.db")


def register_routes(app: Flask) -> None:
    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            gpu_presets=[p.model_dump() for p in GPU_PRESETS],
        )

    # ---------------------------------------------------------------- presets
    @app.get("/api/gpu-presets")
    def api_gpu_presets() -> Response:
        return jsonify([p.model_dump() for p in GPU_PRESETS])

    # -------------------------------------------------------------- HF config
    @app.post("/api/parse-config")
    def api_parse_config() -> Response:
        if "config" not in request.files:
            return jsonify({"error": "Upload a 'config' file (HF config.json)."}), 400
        f = request.files["config"]
        try:
            data = f.read()
            if len(data) > 1_000_000:
                return jsonify({"error": "config.json is too large (>1 MB)."}), 400
            info = parse_config_bytes(data)
        except Exception as exc:
            return jsonify({"error": f"Could not parse config.json: {exc}"}), 400
        return jsonify(info.model_dump())

    # ---------------------------------------------------------------- estimate
    @app.post("/api/estimate")
    def api_estimate() -> Response:
        try:
            req = DeploymentRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return jsonify({"error": "validation_failed", "details": exc.errors()}), 400
        warnings = validate(req)
        result = build_estimate_result(req, warnings)
        return jsonify(result.model_dump())

    # ---------------------------------------------------------------- generate
    @app.post("/api/generate")
    def api_generate() -> Response:
        try:
            req = DeploymentRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return jsonify({"error": "validation_failed", "details": exc.errors()}), 400

        warnings = validate(req)
        if has_blocking_errors(warnings):
            return jsonify(
                {
                    "error": "blocking_validation_errors",
                    "warnings": [w.model_dump() for w in warnings],
                }
            ), 400

        artifacts = render_artifacts(req, warnings)
        memory = estimate_memory(req)
        oneline, multiline = build_command_strings(req)

        deployment_id = save_deployment(
            req=req,
            memory=memory,
            warnings=warnings,
            artifacts=artifacts,
            command_oneline=oneline,
            command_multiline=multiline,
            db_path=_db_path(),
        )

        record = get_deployment(deployment_id, db_path=_db_path())
        if record is None:
            return jsonify({"error": "save_failed"}), 500

        return jsonify(
            {
                "id": record["id"],
                "name": record["name"],
                "model_id": record["model_id"],
                "gpu_preset": record["gpu_preset"],
                "gpu_count": record["gpu_count"],
                "quantization": record["quantization"],
                "fit_status": record["fit_status"],
                "percent_used": record["percent_used"],
                "created_at": record["created_at"],
                "artifacts": record["artifacts"],
                "command_oneline": record["command_oneline"],
                "command_multiline": record["command_multiline"],
                "memory": record["memory"],
                "warnings": record["warnings"],
                "request": record["request"],
                "download_url": f"/api/deployments/{record['id']}/zip",
            }
        )

    # ------------------------------------------------------------ deployments
    @app.get("/api/deployments")
    def api_list_deployments() -> Response:
        rows = list_deployments(limit=200, db_path=_db_path())
        return jsonify(rows)

    @app.get("/api/deployments/<deployment_id>")
    def api_get_deployment(deployment_id: str) -> Response:
        record = get_deployment(deployment_id, db_path=_db_path())
        if record is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify(record)

    @app.get("/api/deployments/<deployment_id>/zip")
    def api_get_deployment_zip(deployment_id: str) -> Response:
        record = get_deployment(deployment_id, db_path=_db_path())
        if record is None:
            return jsonify({"error": "not_found"}), 404
        zip_bytes = build_zip(record["artifacts"])
        filename = f"easy-vllm-{record['name']}.zip"
        return send_file(
            BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )

    @app.delete("/api/deployments/<deployment_id>")
    def api_delete_deployment(deployment_id: str) -> Response:
        ok = delete_deployment(deployment_id, db_path=_db_path())
        if not ok:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"ok": True})

    # --------------------------------------------------------------- handlers
    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "Uploaded file too large."}), 413
