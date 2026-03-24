# IDENA Navarra Agentic Chat + Leaflet Viewer

Servicio en **Python** para consultar IDENA (Navarra) y exponer:

- Herramientas MCP para uso con LLM/LangChain
- API HTTP en Flask
- Flujo agéntico con LangGraph y tools WFS
- Visor web con chat y render de GeoJSON

## Qué incluye

- `idena_service/mcp_server.py`: servidor MCP con tools:
  - `buscar_toponimo`
  - `obtener_limite_municipio`
  - `obtener_limite_valle`
- `idena_service/app.py`: API Flask:
  - `GET /api/toponimos?q=...`
  - `GET /api/municipios/{name}/limite`
  - `GET /api/valles/{name}/limite`
  - `POST /api/agent/chat`
- `idena_service/agent_graph.py`: grafo LangGraph del agente
- `index.html`: visor principal con chat oscuro y render GeoJSON

## Requisitos

- Python 3.10+

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración de LLM (independiente de proveedor)

Por defecto el servicio usa Ollama por URL:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.1:8b
```

Si no quieres usar LLM remoto, puedes forzar modo reglas:

```bash
export LLM_PROVIDER=rules
```

## Ejecutar API + mapa

```bash
python -m idena_service.app
```

Abrir en navegador:

- `http://localhost:8000`

Visor legado:

- `http://localhost:8000/legacy`

## Probar rápido sin dejar procesos en background

```bash
curl -s -X POST http://127.0.0.1:8000/api/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Show schools layer in municipality of Pamplona"}' | jq
```

## Ejecutar servidor MCP

```bash
python -m idena_service.mcp_server
```

## Flujo agéntico implementado

1. El agente detecta municipio y contexto de capa desde el prompt.
2. Consulta WFS `GetCapabilities` (con caché en memoria).
3. Busca capas candidatas (top-3 si hay ambigüedad).
4. Descarga features de la capa elegida.
5. Filtra espacialmente por el municipio delimitado.
6. Devuelve GeoJSON para render directo en el visor.

## Notas

- Las capas WFS de IDENA pueden evolucionar (nombres de capa o campos). Si cambia algo, ajusta los `typeNames` y filtros CQL en `idena_service/idena_client.py`.
- En este MVP, el endpoint de `valles` usa la capa de `concejos` como delimitación de zonas administrativas para Navarra.
- Si el agente devuelve `needs_layer_selection=true`, el frontend muestra candidatos y permite seleccionar capa.
