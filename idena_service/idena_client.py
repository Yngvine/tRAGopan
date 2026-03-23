from __future__ import annotations

import re
import unicodedata
from typing import Any
import httpx


class IdenaClient:
    """Cliente simple para consultar WFS de IDENA y devolver GeoJSON."""

    # Nota: endpoint público WFS de IDENA (GeoServer)
    BASE_WFS_URL = "https://idena.navarra.es/ogc/wfs"

    MUNICIPIOS_TYPENAME = "IDENA:DIADMI_Pol_Municipio"
    CONCEJOS_TYPENAME = "IDENA:DIADMI_Pol_Concejo"
    TOPONIMIA_TYPENAME = "IDENA:TOPONI_Txt_Toponimos"

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    async def _get_wfs(self, params: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(self.BASE_WFS_URL, params=params)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _pick_name_value(props: dict[str, Any]) -> str:
        preferred_keys = ["nombre", "nombre_oficial", "name", "nom", "toponimo", "denominacion"]
        lower_map = {str(k).lower(): k for k in props.keys()}
        for key in preferred_keys:
            if key in lower_map:
                value = props.get(lower_map[key])
                if value is not None:
                    return str(value)
        # fallback: primer valor string no vacío
        for value in props.values():
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _escape_cql(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = unicodedata.normalize("NFD", value)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.lower().strip()
        text = text.replace("/", " ")
        text = re.sub(r"\s+", " ", text)
        return text

    def _find_best_features(
        self,
        features: list[dict[str, Any]],
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        query_norm = self._normalize_text(query)
        query_tokens = set(query_norm.split())

        exact: list[dict[str, Any]] = []
        contains: list[dict[str, Any]] = []
        token_match: list[dict[str, Any]] = []

        for feat in features:
            name = self._pick_name_value(feat.get("properties", {}))
            name_norm = self._normalize_text(name)
            if not name_norm:
                continue

            if name_norm == query_norm:
                exact.append(feat)
                continue

            if query_norm in name_norm:
                contains.append(feat)
                continue

            name_tokens = set(name_norm.split())
            if query_tokens and query_tokens.issubset(name_tokens):
                token_match.append(feat)

        ordered = exact + contains + token_match
        return ordered[:max_results]

    async def search_toponym(self, q: str, limit: int = 10) -> dict[str, Any]:
        q_escaped = self._escape_cql(q)
        cql = (
            f"strToLowerCase(TOPONIMO) like '%{q_escaped.lower()}%' "
            f"OR strToLowerCase(MUNICIPIO) like '%{q_escaped.lower()}%' "
            f"OR strToLowerCase(CONCEJO) like '%{q_escaped.lower()}%'"
        )

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.TOPONIMIA_TYPENAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": str(limit),
            "cql_filter": cql,
        }
        data = await self._get_wfs(params)

        if data.get("features"):
            return data

        # Fallback sin CQL por si hay diferencias de acentos o grafías bilingües.
        fallback_params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.TOPONIMIA_TYPENAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": "1000",
        }
        fallback_data = await self._get_wfs(fallback_params)
        matched = self._find_best_features(fallback_data.get("features", []), q, max_results=limit)
        return {**fallback_data, "features": matched, "numberReturned": len(matched)}

    async def get_municipality_boundary(self, name: str) -> dict[str, Any]:
        name_escaped = self._escape_cql(name)
        cql = f"strToLowerCase(MUNICIPIO) like '%{name_escaped.lower()}%'"

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.MUNICIPIOS_TYPENAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": "5",
            "cql_filter": cql,
        }
        data = await self._get_wfs(params)
        if data.get("features"):
            return data

        # Fallback robusto para nombres con barra, acentos o variantes bilingües.
        fallback_params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.MUNICIPIOS_TYPENAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": "500",
        }
        fallback_data = await self._get_wfs(fallback_params)
        matched = self._find_best_features(fallback_data.get("features", []), name, max_results=5)
        return {**fallback_data, "features": matched, "numberReturned": len(matched)}

    async def get_valley_boundary(self, name: str) -> dict[str, Any]:
        # En IDENA abierto no aparece una capa de "valles" homogénea en WFS.
        # Como alternativa operativa para delimitar zonas, usamos concejos.
        name_escaped = self._escape_cql(name)
        cql = (
            f"strToLowerCase(CONCEJO)='{name_escaped.lower()}' "
            f"OR strToLowerCase(MUNICIPIO)='{name_escaped.lower()}'"
        )

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.CONCEJOS_TYPENAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": "10",
            "cql_filter": cql,
        }
        data = await self._get_wfs(params)
        if data.get("features"):
            return data

        fallback_params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.CONCEJOS_TYPENAME,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": "1200",
        }
        fallback_data = await self._get_wfs(fallback_params)
        matched = self._find_best_features(fallback_data.get("features", []), name, max_results=10)
        return {**fallback_data, "features": matched, "numberReturned": len(matched)}
