from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

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
conversation_store: dict[str, dict] = {}


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


@app.get("/api/tools/municipios")
def tool_listar_municipios():
    try:
        municipalities = run_sync(client.list_municipalities())
        return jsonify({"municipalities": municipalities, "count": len(municipalities)}), 200
    except Exception as exc:
        return {"error": f"Tool failed: {exc}"}, 502


@app.get("/api/tools/municipios/<name>")
def tool_municipio_geometria(name: str):
    try:
        data = run_sync(client.get_municipality_boundary(name))
        if not data.get("features"):
            return {"error": f"Municipality '{name}' not found."}, 404
        return jsonify(
            {
                "municipality": name,
                "geojson": data,
                "actions": [
                    {"type": "clear_layers"},
                    {"type": "add_layer", "layer_role": "municipality_boundary", "source": "geojson"},
                    {"type": "fit_bounds", "source": "geojson"},
                ],
                "warnings": [],
            }
        ), 200
    except Exception as exc:
        return {"error": f"Tool failed: {exc}"}, 502


@app.get("/api/tools/municipios/<name>/toponimia")
def tool_toponimia_municipio(name: str):
    q = request.args.get("q")
    limit_raw = request.args.get("limit", "500").strip()
    start_index_raw = request.args.get("startIndex", "0").strip()
    try:
        limit = max(1, min(2000, int(limit_raw)))
        start_index = max(0, int(start_index_raw))
    except ValueError:
        return {"error": "Query parameter 'limit' and 'startIndex' must be integers."}, 400

    try:
        data = run_sync(
            client.get_toponyms_in_municipality(
                municipality_name=name,
                q=q,
                limit=limit,
                start_index=start_index,
                page_scan_size=500,
                max_scan_pages=8,
            )
        )
        if not data.get("features") and data.get("warnings"):
            for warning in data.get("warnings", []):
                if "not found" in warning.lower():
                    return {"error": warning}, 404
        return jsonify(
            {
                "municipality": name,
                "geojson": {"type": "FeatureCollection", "features": data.get("features", [])},
                "count": data.get("count", len(data.get("features", []))),
                "pagination": data.get("pagination", {}),
                "actions": [
                    {"type": "clear_layers"},
                    {
                        "type": "add_layer",
                        "layer_role": "toponymy_points",
                        "source": "geojson",
                        "style": {"pointColor": "#4f8cff", "radius": 5},
                    },
                    {"type": "fit_bounds", "source": "geojson"},
                ],
                "warnings": data.get("warnings", []),
            }
        ), 200
    except Exception as exc:
        return {"error": f"Tool failed: {exc}"}, 502


@app.post("/api/tools/layer-in-municipality")
def tool_capa_en_municipio():
    payload = request.get_json(silent=True) or {}
    layer = str(payload.get("layer", "")).strip()
    municipality = str(payload.get("municipality", "")).strip()
    selected_layer = payload.get("selected_layer")
    limit_raw = str(payload.get("limit", "500")).strip()
    start_index_raw = str(payload.get("startIndex", "0")).strip()

    if not layer:
        return {"error": "Field 'layer' is required."}, 400
    if not municipality:
        return {"error": "Field 'municipality' is required."}, 400

    try:
        limit = max(1, min(2000, int(limit_raw)))
        start_index = max(0, int(start_index_raw))
    except ValueError:
        return {"error": "Field 'limit' and 'startIndex' must be integers."}, 400

    if selected_layer is not None:
        selected_layer = str(selected_layer).strip() or None

    try:
        result = run_sync(
            client.get_layer_in_municipality(
                layer_hint=layer,
                municipality_name=municipality,
                limit=limit,
                selected_layer=selected_layer,
                start_index=start_index,
                page_scan_size=500,
                max_scan_pages=8,
            )
        )
        if result.get("error"):
            return {"error": result["error"]}, 404

        actions = [
            {"type": "clear_layers"},
            {
                "type": "add_layer",
                "layer_role": "filtered_layer",
                "source": "geojson",
                "style": {"stroke": "#ffd166", "fill": "#ffd166", "fillOpacity": 0.15},
            },
            {"type": "fit_bounds", "source": "geojson"},
        ]

        return jsonify({**result, "actions": actions}), 200
    except Exception as exc:
        return {"error": f"Tool failed: {exc}"}, 502


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
    conversation_id = str(payload.get("conversation_id", "")).strip() or str(uuid4())
    page_size_raw = str(payload.get("page_size", "500")).strip()
    pagination_cursor_raw = str(payload.get("pagination_cursor", "0")).strip()

    if not prompt:
        return {"error": "Field 'prompt' is required."}, 400

    if selected_layer is not None:
        selected_layer = str(selected_layer).strip() or None

    try:
        page_size = max(50, min(2000, int(page_size_raw)))
        pagination_cursor = max(0, int(pagination_cursor_raw))
    except ValueError:
        return {"error": "Fields 'page_size' and 'pagination_cursor' must be integers."}, 400

    try:
        context = conversation_store.get(conversation_id, {})
        result = run_sync(
            agent_service.run(
                prompt=prompt,
                selected_layer=selected_layer,
                conversation_context=context,
                pagination_cursor=pagination_cursor,
                page_size=page_size,
            )
        )
        if result.get("context_update"):
            conversation_store[conversation_id] = result.get("context_update", {})
        result["conversation_id"] = conversation_id
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
