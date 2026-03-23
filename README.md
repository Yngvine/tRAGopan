# IDENA Navarra MCP + Prototipo Leaflet

Servicio en **Python** para consultar IDENA (Navarra) y exponer:

- Herramientas MCP para uso con LLM/LangChain
- API HTTP (FastAPI)
- Prototipo web HTML + Leaflet (sin LLM)

## Qué incluye

- `idena_service/mcp_server.py`: servidor MCP con tools:
  - `buscar_toponimo`
  - `obtener_limite_municipio`
  - `obtener_limite_valle`
- `idena_service/app.py`: API FastAPI:
  - `GET /api/toponimos?q=...`
  - `GET /api/municipios/{name}/limite`
  - `GET /api/valles/{name}/limite`
- `web/index.html`: mapa Leaflet para visualizar resultados

## Requisitos

- Python 3.10+

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar API + mapa

```bash
uvicorn idena_service.app:app --reload --host 0.0.0.0 --port 8000
```

Abrir en navegador:

- `http://localhost:8000`

## Probar rápido sin dejar procesos en background

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from idena_service.app import app

client = TestClient(app)
for path in [
  '/api/municipios/Villava/limite',
  '/api/toponimos?q=Villava&limit=5',
  '/api/valles/Erro/limite',
]:
  r = client.get(path)
  print(path, r.status_code, len(r.json().get('features', [])) if r.status_code == 200 else r.text)
PY
```

## Ejecutar servidor MCP

```bash
python -m idena_service.mcp_server
```

## Integración rápida con LangChain (idea)

Puedes registrar este servidor MCP y usar sus herramientas para que el agente consulte geometrías en GeoJSON.

## Notas

- Las capas WFS de IDENA pueden evolucionar (nombres de capa o campos). Si cambia algo, ajusta los `typeNames` y filtros CQL en `idena_service/idena_client.py`.
- En este MVP, el endpoint de `valles` usa la capa de `concejos` como delimitación de zonas administrativas para Navarra.
- El prototipo está centrado en Navarra y pensado como MVP para crecer con más capas (valles explícitos, merindades, etc.).
