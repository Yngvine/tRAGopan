from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .agent_graph import AgentGeoService, run_sync
from .idena_client import IdenaClient

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


def _env_timeout(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        return default


client = IdenaClient(timeout=_env_timeout("IDENA_TIMEOUT", 60.0))
agent_service = AgentGeoService(client=client)


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.get("/api/toponimos")
def buscar_toponimos():
    q = request.args.get("q", "").strip()
    limit_raw = request.args.get("limit", "10").strip()
    if not q:
        return {"error": "Query parameter 'q' is required."}, 400
    try:
        limit = max(1, min(100, int(limit_raw)))
    except ValueError:
        return {"error": "Query parameter 'limit' must be an integer."}, 400

    try:
        data = run_sync(client.search_toponym(q=q, limit=limit))
        return jsonify(data), 200
    except Exception as exc:
        return {"error": f"IDENA request failed: {exc}"}, 502


@app.get("/api/municipios/<name>/limite")
def limite_municipio(name: str):
    try:
        data = run_sync(client.get_municipality_boundary(name))
        if not data.get("features"):
            return {"error": f"Municipality '{name}' not found."}, 404
        return jsonify(data), 200
    except Exception as exc:
        return {"error": f"IDENA request failed: {exc}"}, 502


@app.get("/api/valles/<name>/limite")
def limite_valle(name: str):
    try:
        data = run_sync(client.get_valley_boundary(name))
        if not data.get("features"):
            return {"error": f"Valley '{name}' not found."}, 404
        return jsonify(data), 200
    except Exception as exc:
        return {"error": f"IDENA request failed: {exc}"}, 502


@app.post("/api/agent/chat")
def agent_chat():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()
    selected_layer = payload.get("selected_layer")

    if not prompt:
        return {"error": "Field 'prompt' is required."}, 400

    if selected_layer is not None:
        selected_layer = str(selected_layer).strip() or None

    try:
        result = run_sync(agent_service.run(prompt=prompt, selected_layer=selected_layer))
        return jsonify(result), 200
    except Exception as exc:
        return {"error": f"Agent workflow failed: {exc}"}, 502


@app.get("/")
def serve_root():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/legacy")
def serve_legacy():
    return send_from_directory(BASE_DIR / "web", "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
