from __future__ import annotations

import json
from mcp.server.fastmcp import FastMCP

from .idena_client import IdenaClient

mcp = FastMCP("idena-navarra")
client = IdenaClient()


@mcp.tool()
async def buscar_toponimo(texto: str, limite: int = 10) -> str:
    """Busca topónimos en Navarra (IDENA) y devuelve GeoJSON serializado."""
    data = await client.search_toponym(texto, limite)
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
async def obtener_limite_municipio(nombre: str) -> str:
    """Obtiene geometría de límite municipal en Navarra (IDENA) en GeoJSON serializado."""
    data = await client.get_municipality_boundary(nombre)
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
async def obtener_limite_valle(nombre: str) -> str:
    """Obtiene geometría de límite de valle en Navarra (IDENA) en GeoJSON serializado."""
    data = await client.get_valley_boundary(nombre)
    return json.dumps(data, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
