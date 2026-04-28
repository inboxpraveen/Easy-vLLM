"""Easy-vLLM Flask application entrypoint.

Routes are wired in :mod:`easy_vllm.routes` so that the entrypoint stays small
and the package remains testable in isolation.
"""
from __future__ import annotations

import os

from flask import Flask

from easy_vllm.routes import register_routes
from easy_vllm.storage import init_db


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
        instance_relative_config=True,
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    db_path = os.environ.get(
        "EASY_VLLM_DB",
        os.path.join(app.instance_path, "easy_vllm.db"),
    )
    app.config["EASY_VLLM_DB"] = db_path
    os.makedirs(app.instance_path, exist_ok=True)
    init_db(db_path)

    register_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
