from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .idena_client import IdenaClient

app = FastAPI(title="IDENA Navarra Geo API", version="0.1.0")
client = IdenaClient()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/toponimos")
async def buscar_toponimos(
    q: str = Query(..., min_length=1, description="Texto a buscar"),
    limit: int = Query(10, ge=1, le=100),
):
    try:
        data = await client.search_toponym(q=q, limit=limit)
        return JSONResponse(data)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando IDENA: {exc}") from exc


@app.get("/api/municipios/{name}/limite")
async def limite_municipio(name: str):
    try:
        data = await client.get_municipality_boundary(name)
        if not data.get("features"):
            raise HTTPException(status_code=404, detail=f"No se encontró municipio '{name}'")
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando IDENA: {exc}") from exc


@app.get("/api/valles/{name}/limite")
async def limite_valle(name: str):
    try:
        data = await client.get_valley_boundary(name)
        if not data.get("features"):
            raise HTTPException(status_code=404, detail=f"No se encontró valle '{name}'")
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando IDENA: {exc}") from exc


app.mount("/", StaticFiles(directory="web", html=True), name="web")
