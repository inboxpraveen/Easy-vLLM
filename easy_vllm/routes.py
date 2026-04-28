"""Flask routes for the easy-vLLM web app."""
from __future__ import annotations

from flask import Flask, Response, jsonify, render_template, request, send_file
from io import BytesIO
from pydantic import ValidationError

from .command_builder import _resolved_served_name
from .config_parser import parse_config_bytes
from .docker_generator import build_estimate_result, render_artifacts
from .gpu_presets import GPU_PRESETS
from .schemas import DeploymentRequest
from .validators import has_blocking_errors, validate
from .zip_exporter import build_zip


def register_routes(app: Flask) -> None:
    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            gpu_presets=[p.model_dump() for p in GPU_PRESETS],
        )

    @app.get("/api/gpu-presets")
    def api_gpu_presets() -> Response:
        return jsonify([p.model_dump() for p in GPU_PRESETS])

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

    @app.post("/api/estimate")
    def api_estimate() -> Response:
        try:
            req = DeploymentRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return jsonify({"error": "validation_failed", "details": exc.errors()}), 400
        warnings = validate(req)
        result = build_estimate_result(req, warnings)
        return jsonify(result.model_dump())

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

        files = render_artifacts(req, warnings)
        zip_bytes = build_zip(files)
        served_name = _resolved_served_name(req)
        filename = f"easy-vllm-{served_name}.zip"
        return send_file(
            BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "Uploaded file too large."}), 413
