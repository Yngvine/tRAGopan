from __future__ import annotations

import re
import time
import unicodedata
from typing import Any
import xml.etree.ElementTree as ET
import httpx
from shapely.geometry import shape
from shapely.ops import unary_union


class IdenaClient:
    """Cliente simple para consultar WFS de IDENA y devolver GeoJSON."""

    # Nota: endpoint público WFS de IDENA (GeoServer)
    BASE_WFS_URL = "https://idena.navarra.es/ogc/wfs"

    MUNICIPIOS_TYPENAME = "IDENA:DIADMI_Pol_Municipio"
    CONCEJOS_TYPENAME = "IDENA:DIADMI_Pol_Concejo"
    TOPONIMIA_TYPENAME = "IDENA:TOPONI_Txt_Toponimos"

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self._capabilities_cache: list[dict[str, str]] = []
        self._capabilities_cached_at: float = 0.0
        self._municipalities_cache: list[str] = []
        self._municipalities_cached_at: float = 0.0

    async def _request_wfs(self, params: dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(self.BASE_WFS_URL, params=params)
            response.raise_for_status()
            return response

    async def _get_wfs(self, params: dict[str, Any]) -> dict[str, Any]:
        response = await self._request_wfs(params)
        return response.json()

    async def get_capabilities(self, ttl_seconds: int = 900) -> list[dict[str, str]]:
        now = time.time()
        if self._capabilities_cache and (now - self._capabilities_cached_at) < ttl_seconds:
            return self._capabilities_cache

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetCapabilities",
        }
        response = await self._request_wfs(params)
        root = ET.fromstring(response.text)
        ns = {
            "wfs": "http://www.opengis.net/wfs/2.0",
            "ows": "http://www.opengis.net/ows/1.1",
        }

        layers: list[dict[str, str]] = []
        feature_types = root.findall(".//wfs:FeatureType", ns)
        for feature_type in feature_types:
            name = (feature_type.findtext("wfs:Name", default="", namespaces=ns) or "").strip()
            title = (feature_type.findtext("wfs:Title", default="", namespaces=ns) or "").strip()
            abstract = (feature_type.findtext("wfs:Abstract", default="", namespaces=ns) or "").strip()
            default_crs = (
                feature_type.findtext("wfs:DefaultCRS", default="", namespaces=ns) or ""
            ).strip()
            if not name:
                continue
            layers.append(
                {
                    "name": name,
                    "title": title,
                    "abstract": abstract,
                    "default_crs": default_crs,
                }
            )

        self._capabilities_cache = layers
        self._capabilities_cached_at = now
        return layers

    def search_layers(self, layers: list[dict[str, str]], layer_hint: str, limit: int = 3) -> list[dict[str, str]]:
        hint_norm = self._normalize_text(layer_hint)
        hint_tokens = set(hint_norm.split())
        if not hint_norm:
            return layers[:limit]

        scored: list[tuple[int, dict[str, str]]] = []
        for layer in layers:
            text = " ".join(
                [
                    layer.get("name", ""),
                    layer.get("title", ""),
                    layer.get("abstract", ""),
                ]
            )
            text_norm = self._normalize_text(text)
            score = 0
            if hint_norm in text_norm:
                score += 100
            for token in hint_tokens:
                if token and token in text_norm:
                    score += 15
            if score > 0:
                scored.append((score, layer))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [layer for _, layer in scored[:limit]]

    async def get_layer_features(
        self,
        type_name: str,
        bbox: tuple[float, float, float, float] | None = None,
        count: int = 2000,
        start_index: int = 0,
        cql_filter: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": type_name,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": str(count),
        }
        if start_index > 0:
            params["startIndex"] = str(start_index)
        if bbox is not None:
            params["bbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326"
        if cql_filter:
            params["cql_filter"] = cql_filter
        return await self._get_wfs(params)

    async def list_municipalities(self, ttl_seconds: int = 3600) -> list[str]:
        now = time.time()
        if self._municipalities_cache and (now - self._municipalities_cached_at) < ttl_seconds:
            return self._municipalities_cache

        data = await self.get_layer_features(type_name=self.MUNICIPIOS_TYPENAME, count=1500)
        names: set[str] = set()
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            for key in ("MUNICIPIO", "municipio", "nombre", "name"):
                value = props.get(key)
                if isinstance(value, str) and value.strip():
                    names.add(value.strip())
                    break

        ordered = sorted(names)
        self._municipalities_cache = ordered
        self._municipalities_cached_at = now
        return ordered

    async def get_toponyms_in_municipality(
        self,
        municipality_name: str,
        q: str | None = None,
        limit: int = 500,
        start_index: int = 0,
        page_scan_size: int = 500,
        max_scan_pages: int = 8,
    ) -> dict[str, Any]:
        boundary = await self.get_municipality_boundary(municipality_name)
        muni_features = boundary.get("features", [])
        if not muni_features:
            return {
                "type": "FeatureCollection",
                "features": [],
                "municipality": municipality_name,
                "warnings": [f"Municipality '{municipality_name}' not found."],
            }

        muni_geometries = [shape(feat["geometry"]) for feat in muni_features if feat.get("geometry")]
        muni_geometry = unary_union(muni_geometries)
        minx, miny, maxx, maxy = muni_geometry.bounds

        query_norm = self._normalize_text(q or "")
        target_count = max(1, min(2000, int(limit)))
        scan_size = max(50, min(2000, int(page_scan_size)))
        cursor = max(0, int(start_index))

        filtered_features: list[dict[str, Any]] = []
        scanned_pages = 0
        reached_end = False

        while len(filtered_features) < target_count and scanned_pages < max_scan_pages:
            data = await self.get_layer_features(
                type_name=self.TOPONIMIA_TYPENAME,
                bbox=(minx, miny, maxx, maxy),
                count=scan_size,
                start_index=cursor,
            )
            raw_features = data.get("features", [])
            raw_count = len(raw_features)
            if raw_count == 0:
                reached_end = True
                break

            for feature in raw_features:
                geom_data = feature.get("geometry")
                if not geom_data:
                    continue
                try:
                    feature_geom = shape(geom_data)
                except Exception:
                    continue
                if not (feature_geom.is_valid and feature_geom.intersects(muni_geometry)):
                    continue

                if query_norm:
                    props = feature.get("properties", {})
                    name_value = self._pick_name_value(props)
                    if query_norm not in self._normalize_text(name_value):
                        continue

                filtered_features.append(feature)
                if len(filtered_features) >= target_count:
                    break

            cursor += raw_count
            scanned_pages += 1
            if raw_count < scan_size:
                reached_end = True
                break

        warnings: list[str] = []
        if scanned_pages >= max_scan_pages and len(filtered_features) < target_count:
            warnings.append(
                "Toponym query scanned max pages. Results may be partial; use 'cargar mas'."
            )

        has_more = not reached_end

        return {
            "type": "FeatureCollection",
            "features": filtered_features,
            "municipality": municipality_name,
            "query": q,
            "count": len(filtered_features),
            "pagination": {
                "has_more": has_more,
                "next_cursor": cursor if has_more else None,
                "page_size": target_count,
            },
            "warnings": warnings,
        }

    async def get_layer_in_municipality(
        self,
        layer_hint: str,
        municipality_name: str,
        limit: int = 500,
        selected_layer: str | None = None,
        start_index: int = 0,
        page_scan_size: int = 500,
        max_scan_pages: int = 8,
    ) -> dict[str, Any]:
        capabilities = await self.get_capabilities(ttl_seconds=900)
        chosen_layer: dict[str, str] | None = None
        if selected_layer:
            selected_norm = selected_layer.strip().lower()
            for layer in capabilities:
                if layer.get("name", "").lower() == selected_norm:
                    chosen_layer = layer
                    break

        candidates = self.search_layers(capabilities, layer_hint, limit=3)
        if chosen_layer is None and candidates:
            chosen_layer = candidates[0]

        if not chosen_layer:
            return {
                "error": "No compatible WFS layers were found for that request.",
                "layer_candidates": candidates,
                "needs_layer_selection": False,
            }

        if len(candidates) > 1 and not selected_layer:
            return {
                "layer_candidates": candidates,
                "needs_layer_selection": True,
            }

        boundary = await self.get_municipality_boundary(municipality_name)
        muni_features = boundary.get("features", [])
        if not muni_features:
            return {
                "error": f"Municipality '{municipality_name}' was not found.",
                "needs_layer_selection": False,
            }

        muni_geometries = [shape(feat["geometry"]) for feat in muni_features if feat.get("geometry")]
        muni_geometry = unary_union(muni_geometries)
        minx, miny, maxx, maxy = muni_geometry.bounds

        target_count = max(1, min(2000, int(limit)))
        scan_size = max(50, min(2000, int(page_scan_size)))
        cursor = max(0, int(start_index))
        filtered_features: list[dict[str, Any]] = []
        scanned_pages = 0
        reached_end = False
        while len(filtered_features) < target_count and scanned_pages < max_scan_pages:
            raw = await self.get_layer_features(
                type_name=chosen_layer["name"],
                bbox=(minx, miny, maxx, maxy),
                count=scan_size,
                start_index=cursor,
            )
            raw_features = raw.get("features", [])
            raw_count = len(raw_features)
            if raw_count == 0:
                reached_end = True
                break

            for feature in raw_features:
                geom_data = feature.get("geometry")
                if not geom_data:
                    continue
                try:
                    feature_geom = shape(geom_data)
                except Exception:
                    continue
                if feature_geom.is_valid and feature_geom.intersects(muni_geometry):
                    filtered_features.append(feature)
                    if len(filtered_features) >= target_count:
                        break

            cursor += raw_count
            scanned_pages += 1
            if raw_count < scan_size:
                reached_end = True
                break

        warnings: list[str] = []
        if scanned_pages >= max_scan_pages and len(filtered_features) < target_count:
            warnings.append(
                "Layer query scanned max pages. Results may be partial; use 'cargar mas'."
            )

        has_more = not reached_end

        return {
            "reply": (
                f"Found {len(filtered_features)} features in layer '{chosen_layer['name']}' "
                f"intersecting municipality '{municipality_name}'."
            ),
            "municipality": municipality_name,
            "selected_layer": chosen_layer["name"],
            "layer_candidates": candidates,
            "needs_layer_selection": False,
            "geojson": {
                "type": "FeatureCollection",
                "features": filtered_features,
                "selected_layer": chosen_layer["name"],
                "municipality": municipality_name,
            },
            "count": len(filtered_features),
            "pagination": {
                "has_more": has_more,
                "next_cursor": cursor if has_more else None,
                "page_size": target_count,
            },
            "warnings": warnings,
        }

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
