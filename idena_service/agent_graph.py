from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama
from shapely.geometry import shape
from shapely.ops import unary_union

from .idena_client import IdenaClient


class AgentState(TypedDict, total=False):
    prompt: str
    selected_layer: str | None
    municipality_hint: str | None
    layer_hint: str | None
    municipality_name: str | None
    municipality_geojson: dict[str, Any] | None
    capabilities: list[dict[str, str]]
    layer_candidates: list[dict[str, str]]
    chosen_layer: dict[str, str] | None
    needs_layer_selection: bool
    raw_geojson: dict[str, Any] | None
    filtered_geojson: dict[str, Any] | None
    municipalities: list[str]
    query_mode: str
    is_geo_request: bool
    reply: str
    error: str | None


class LLMAdapter:
    async def extract_scope(self, prompt: str) -> dict[str, str | None]:
        raise NotImplementedError

    async def generate_chat_reply(self, prompt: str) -> str:
        raise NotImplementedError


class RuleBasedAdapter(LLMAdapter):
    MUNICIPALITY_PATTERNS = [
        r"municipality of\s+([a-zA-ZÀ-ÿ\-\s/]+)",
        r"municipio de\s+([a-zA-ZÀ-ÿ\-\s/]+)",
        r"in\s+([a-zA-ZÀ-ÿ\-\s/]+)",
        r"en\s+([a-zA-ZÀ-ÿ\-\s/]+)",
    ]

    async def extract_scope(self, prompt: str) -> dict[str, str | None]:
        text = prompt.strip()
        municipality: str | None = None
        for pattern in self.MUNICIPALITY_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                municipality = match.group(1).strip(" .,;:")
                break

        # Simple layer hint extraction: anything after "layer"/"capa" if present.
        layer_hint: str | None = None
        layer_match = re.search(
            r"(?:layer|capa)\s*(?:named|name|called|de)?\s*[:\-]?\s*([a-zA-Z0-9_:À-ÿ\-\s/]+)",
            text,
            flags=re.IGNORECASE,
        )
        if layer_match:
            layer_hint = layer_match.group(1).strip(" .,;:")

        if not layer_hint:
            # Fallback: use the full prompt as layer search context.
            layer_hint = text

        return {
            "municipality": municipality,
            "layer_hint": layer_hint,
        }

    async def generate_chat_reply(self, prompt: str) -> str:
        text = prompt.strip()
        if not text:
            return "Hi! Ask me anything."
        return (
            "I can chat normally, and I can also run geospatial workflows over IDENA layers. "
            f"About your message: '{text}', I understand it as a general chat request."
        )


class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._chat_model = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0,
            client_kwargs={"timeout": self.timeout},
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        candidate = text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            raise ValueError(f"Ollama response did not contain valid JSON: {candidate[:200]}")
        return json.loads(match.group(0))

    @staticmethod
    def _parse_content(data: dict[str, Any]) -> dict[str, str | None]:
        # Ollama /api/chat returns content in message.content.
        raw_content = str(data.get("message", {}).get("content", "")).strip()
        parsed = OllamaAdapter._extract_json_object(raw_content)
        return {
            "municipality": parsed.get("municipality"),
            "layer_hint": parsed.get("layer_hint"),
        }

    @staticmethod
    def _parse_langchain_content(content: Any) -> dict[str, str | None]:
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and "text" in item:
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(item))
            text = "\n".join(chunks)
        else:
            text = str(content)

        parsed = OllamaAdapter._extract_json_object(text)
        return {
            "municipality": parsed.get("municipality"),
            "layer_hint": parsed.get("layer_hint"),
        }

    async def extract_scope(self, prompt: str) -> dict[str, str | None]:
        system_instruction = (
            "Extract municipality and layer_hint from the user request. "
            "Return ONLY valid JSON with keys municipality and layer_hint. "
            "Use null when missing."
        )

        # Preferred path: LangChain Ollama client.
        langchain_error: Exception | None = None
        try:
            response = await self._chat_model.ainvoke(
                [
                    ("system", system_instruction),
                    ("human", prompt),
                ],
                format="json",
            )
            return self._parse_langchain_content(response.content)
        except Exception as exc:
            # Fallback path: direct HTTP for compatibility and debugging.
            langchain_error = exc

        request_body_schema = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "format": {
                "type": "object",
                "properties": {
                    "municipality": {"type": ["string", "null"]},
                    "layer_hint": {"type": ["string", "null"]},
                },
                "required": ["municipality", "layer_hint"],
            },
        }

        # Compatibility fallback for servers that do not support schema format.
        request_body_json = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "options": {"temperature": 0},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=request_body_schema)
                response.raise_for_status()
                data = response.json()
                return self._parse_content(data)
            except (httpx.HTTPStatusError, json.JSONDecodeError, ValueError):
                # Retry once with format=json, which is supported by older servers.
                try:
                    response = await client.post(f"{self.base_url}/api/chat", json=request_body_json)
                    response.raise_for_status()
                    data = response.json()
                    return self._parse_content(data)
                except httpx.HTTPError as exc:
                    previous = f" LangChain error: {langchain_error}" if langchain_error else ""
                    raise RuntimeError(
                        f"Failed to call Ollama at {self.base_url}/api/chat: {exc}.{previous}"
                    ) from exc

    async def generate_chat_reply(self, prompt: str) -> str:
        system_instruction = (
            "You are a helpful assistant. Reply in plain text, concise and clear."
        )

        langchain_error: Exception | None = None
        try:
            response = await self._chat_model.ainvoke(
                [
                    ("system", system_instruction),
                    ("human", prompt),
                ]
            )
            content = response.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
        except Exception as exc:
            langchain_error = exc

        request_body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.2},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=request_body)
                response.raise_for_status()
                data = response.json()
                return str(data.get("message", {}).get("content", "")).strip() or ""
            except httpx.HTTPError as exc:
                previous = f" LangChain error: {langchain_error}" if langchain_error else ""
                raise RuntimeError(
                    f"Failed to call Ollama chat at {self.base_url}/api/chat: {exc}.{previous}"
                ) from exc


def build_llm_adapter() -> LLMAdapter:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        timeout_raw = os.getenv("OLLAMA_TIMEOUT", "60")
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 60.0
        if timeout <= 0:
            timeout = 60.0
        return OllamaAdapter(base_url=base_url, model=model, timeout=timeout)
    return RuleBasedAdapter()


class AgentGeoService:
    def __init__(self, client: IdenaClient, llm: LLMAdapter | None = None) -> None:
        self.client = client
        self.llm = llm or build_llm_adapter()
        self.rule_fallback = RuleBasedAdapter()
        self.graph = self._build_graph()

    @staticmethod
    def _is_geospatial_prompt(prompt: str) -> bool:
        text = prompt.lower()
        geo_hints = [
            "layer",
            "layers",
            "capa",
            "capas",
            "municipality",
            "municipio",
            "wfs",
            "geojson",
            "idena",
            "map",
            "mapa",
            "feature",
            "features",
            "geometry",
            "geometr",
            "bbox",
            "river",
            "road",
            "parcela",
            "parcel",
            "boundary",
            "limite",
            "municipalities",
            "municipios",
            "navarra",
        ]
        return any(token in text for token in geo_hints)

    @staticmethod
    def _is_capabilities_prompt(prompt: str) -> bool:
        text = prompt.lower().strip()
        if not text:
            return False

        hints = [
            "what layers",
            "which layers",
            "list layers",
            "available layers",
            "capabilities",
            "capas hay",
            "que capas",
            "capas disponibles",
            "idena layers",
            "layers of idena",
            "idena layer list",
        ]
        if any(hint in text for hint in hints):
            return True

        # Robust intent match for natural wording in English and Spanish.
        patterns = [
            r"\b(list|show|tell|give|display)\b[\s\w]{0,40}\b(layers|capabilities)\b[\s\w]{0,40}\b(idena|wfs)?\b",
            r"\b(cuales|cu[aá]les|dime|lista|listar|ens[eé]n?ame|mostrar)\b[\s\w]{0,40}\b(capas|capas disponibles)\b[\s\w]{0,40}\b(idena)?\b",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _is_municipalities_prompt(prompt: str) -> bool:
        text = prompt.lower()
        mentions_municipality = any(
            token in text for token in ["municipalities", "municipios", "municipality list", "lista de municipios"]
        )
        mentions_global_scope = any(
            token in text for token in ["all navarra", "whole navarra", "toda navarra", "navarra entera", "de navarra"]
        )
        return mentions_municipality and (mentions_global_scope or "all" in text or "todos" in text)

    @staticmethod
    def _is_municipality_boundary_prompt(prompt: str) -> bool:
        text = prompt.lower().strip()
        if not text:
            return False

        # If the user explicitly asks for a layer/capa, keep filtered-layer flow.
        if any(token in text for token in ["layer", "layers", "capa", "capas", "wfs"]):
            return False

        direct_hints = [
            "cargame el municipio",
            "cargame municipio",
            "mostrar municipio",
            "muestrame el municipio",
            "show municipality",
            "load municipality",
            "municipality boundary",
            "limite del municipio",
            "límite del municipio",
        ]
        if any(hint in text for hint in direct_hints):
            return True

        # Natural wording: "municipio de X" with a display verb and without layer intent.
        return bool(
            re.search(
                r"\b(cargar|cargame|mostrar|muestrame|ver|ensename|enséname|show|load)\b[\s\w]{0,30}\bmunicip(io|ality)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    async def _parse_prompt(self, state: AgentState) -> AgentState:
        prompt = state["prompt"]
        if self._is_municipality_boundary_prompt(prompt):
            try:
                parsed = await self.llm.extract_scope(state["prompt"])
            except Exception:
                parsed = await self.rule_fallback.extract_scope(state["prompt"])
            return {
                "is_geo_request": True,
                "query_mode": "municipality_boundary",
                "municipality_hint": parsed.get("municipality"),
                "needs_layer_selection": False,
            }

        if self._is_municipalities_prompt(prompt):
            return {
                "is_geo_request": True,
                "query_mode": "municipalities",
                "needs_layer_selection": False,
            }

        if self._is_capabilities_prompt(prompt):
            return {
                "is_geo_request": True,
                "query_mode": "capabilities",
                "needs_layer_selection": False,
            }

        if not self._is_geospatial_prompt(prompt):
            try:
                reply = await self.llm.generate_chat_reply(prompt)
            except Exception:
                # If remote LLM is unavailable, keep chat working with local fallback.
                reply = await self.rule_fallback.generate_chat_reply(prompt)
            return {
                "is_geo_request": False,
                "query_mode": "general",
                "needs_layer_selection": False,
                "reply": reply,
            }

        try:
            parsed = await self.llm.extract_scope(state["prompt"])
        except Exception:
            # Keep geospatial pipeline available even if Ollama endpoint fails.
            parsed = await self.rule_fallback.extract_scope(state["prompt"])
        selected_layer = state.get("selected_layer")
        return {
            "is_geo_request": True,
            "query_mode": "filtered_layer",
            "municipality_hint": parsed.get("municipality"),
            "layer_hint": selected_layer or parsed.get("layer_hint") or state["prompt"],
        }

    def _parse_branch(self, state: AgentState) -> str:
        if not state.get("is_geo_request"):
            return "finalize"

        mode = state.get("query_mode", "filtered_layer")
        if mode == "capabilities":
            return "load_capabilities"
        if mode == "municipalities":
            return "fetch_municipalities"
        if mode == "municipality_boundary":
            return "resolve_municipality"
        if mode == "filtered_layer":
            return "resolve_municipality"
        return "finalize"

    async def _fetch_municipalities(self, state: AgentState) -> AgentState:
        data = await self.client.get_layer_features(
            type_name=self.client.MUNICIPIOS_TYPENAME,
            count=1500,
        )
        names: set[str] = set()
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            for key in ("MUNICIPIO", "municipio", "nombre", "name"):
                value = props.get(key)
                if isinstance(value, str) and value.strip():
                    names.add(value.strip())
                    break

        ordered = sorted(names)
        return {
            "municipalities": ordered,
        }

    async def _resolve_municipality(self, state: AgentState) -> AgentState:
        municipality_hint = state.get("municipality_hint")
        if not municipality_hint:
            return {
                "error": "Please include a municipality to define the area filter.",
                "needs_layer_selection": False,
            }

        boundary = await self.client.get_municipality_boundary(municipality_hint)
        features = boundary.get("features", [])
        if not features:
            return {
                "error": f"Municipality '{municipality_hint}' was not found.",
                "needs_layer_selection": False,
            }

        name = municipality_hint
        props = features[0].get("properties", {})
        for key in ("MUNICIPIO", "municipio", "nombre", "name"):
            if props.get(key):
                name = str(props[key])
                break

        return {
            "municipality_name": name,
            "municipality_geojson": boundary,
        }

    async def _load_capabilities(self, state: AgentState) -> AgentState:
        layers = await self.client.get_capabilities(ttl_seconds=900)
        return {"capabilities": layers}

    async def _select_layer(self, state: AgentState) -> AgentState:
        layer_hint = state.get("layer_hint") or state["prompt"]
        layers = state.get("capabilities", [])
        candidates = self.client.search_layers(layers, layer_hint, limit=3)

        if not candidates:
            return {
                "error": "No compatible WFS layers were found for that request.",
                "layer_candidates": [],
                "needs_layer_selection": False,
            }

        if state.get("selected_layer"):
            selected = state["selected_layer"].strip().lower()
            exact = [layer for layer in layers if layer.get("name", "").lower() == selected]
            if exact:
                return {
                    "chosen_layer": exact[0],
                    "layer_candidates": candidates,
                    "needs_layer_selection": False,
                }

        if len(candidates) > 1 and not state.get("selected_layer"):
            return {
                "layer_candidates": candidates,
                "needs_layer_selection": True,
                "reply": "I found multiple layer candidates. Please choose one and send it back.",
            }

        return {
            "chosen_layer": candidates[0],
            "layer_candidates": candidates,
            "needs_layer_selection": False,
        }

    async def _fetch_features(self, state: AgentState) -> AgentState:
        if state.get("needs_layer_selection"):
            return {}

        chosen_layer = state.get("chosen_layer")
        municipality_geojson = state.get("municipality_geojson")
        if not chosen_layer or not municipality_geojson:
            return {"error": "Missing chosen layer or municipality geometry."}

        muni_features = municipality_geojson.get("features", [])
        if not muni_features:
            return {"error": "Municipality geometry is empty."}

        muni_geom = shape(muni_features[0]["geometry"])
        minx, miny, maxx, maxy = muni_geom.bounds

        raw = await self.client.get_layer_features(
            type_name=chosen_layer["name"],
            bbox=(minx, miny, maxx, maxy),
            count=3000,
        )
        return {"raw_geojson": raw}

    async def _spatial_filter(self, state: AgentState) -> AgentState:
        if state.get("needs_layer_selection"):
            return {}

        raw_geojson = state.get("raw_geojson") or {}
        municipality_geojson = state.get("municipality_geojson") or {}
        raw_features = raw_geojson.get("features", [])
        muni_features = municipality_geojson.get("features", [])

        if not raw_features:
            return {
                "filtered_geojson": {
                    "type": "FeatureCollection",
                    "features": [],
                }
            }

        if not muni_features:
            return {"error": "Municipality geometry is missing for spatial filtering."}

        # Merge municipality geometry in case a municipality returns multiple polygons.
        muni_geometry = unary_union([shape(feat["geometry"]) for feat in muni_features if feat.get("geometry")])

        filtered_features: list[dict[str, Any]] = []
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

        filtered_geojson = {
            "type": "FeatureCollection",
            "features": filtered_features,
            "selected_layer": state.get("chosen_layer", {}).get("name"),
            "municipality": state.get("municipality_name"),
        }
        return {"filtered_geojson": filtered_geojson}

    async def _finalize(self, state: AgentState) -> AgentState:
        if not state.get("is_geo_request", True):
            return {
                "reply": state.get("reply", ""),
            }

        mode = state.get("query_mode", "filtered_layer")

        if mode == "capabilities":
            layers = state.get("capabilities", [])
            if not layers:
                return {"reply": "No WFS layers were found in IDENA capabilities."}

            preview = layers[:25]
            lines = [f"IDENA WFS has {len(layers)} available layers. First {len(preview)}:"]
            for idx, layer in enumerate(preview, start=1):
                title = layer.get("title") or "No title"
                lines.append(f"{idx}. {layer.get('name', 'unknown')} - {title}")
            lines.append("Ask me for a specific layer name if you want to query it.")
            return {"reply": "\n".join(lines)}

        if mode == "municipalities":
            municipalities = state.get("municipalities", [])
            if not municipalities:
                return {"reply": "I could not retrieve municipalities for Navarra right now."}

            preview = municipalities[:40]
            lines = [f"Navarra has {len(municipalities)} municipalities in the available dataset. First {len(preview)}:"]
            lines.append(", ".join(preview))
            lines.append("Ask for a municipality by name and I can continue with a layer query.")
            return {"reply": "\n".join(lines)}

        if mode == "municipality_boundary":
            if state.get("error"):
                return {"reply": state["error"]}

            municipality_geojson = state.get("municipality_geojson")
            municipality_name = state.get("municipality_name", "the selected municipality")
            if not municipality_geojson:
                return {"reply": "I could not retrieve the municipality boundary right now."}
            return {
                "reply": f"Loaded boundary for municipality '{municipality_name}'.",
                "filtered_geojson": municipality_geojson,
            }

        if state.get("error"):
            return {
                "reply": state["error"],
            }

        if state.get("needs_layer_selection"):
            candidates = state.get("layer_candidates", [])
            lines = ["I found multiple matching layers. Please pick one:"]
            for idx, layer in enumerate(candidates, start=1):
                title = layer.get("title") or "No title"
                lines.append(f"{idx}. {layer['name']} - {title}")
            return {
                "reply": "\n".join(lines),
            }

        filtered_geojson = state.get("filtered_geojson") or {"type": "FeatureCollection", "features": []}
        total = len(filtered_geojson.get("features", []))
        municipality = state.get("municipality_name", "the selected municipality")
        chosen_layer = state.get("chosen_layer", {}).get("name", "unknown layer")
        return {
            "reply": (
                f"Found {total} features in layer '{chosen_layer}' "
                f"intersecting municipality '{municipality}'."
            ),
        }

    def _layer_branch(self, state: AgentState) -> str:
        if state.get("error"):
            return "finalize"
        return "fetch_features"

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("parse_prompt", self._parse_prompt)
        graph_builder.add_node("resolve_municipality", self._resolve_municipality)
        graph_builder.add_node("load_capabilities", self._load_capabilities)
        graph_builder.add_node("fetch_municipalities", self._fetch_municipalities)
        graph_builder.add_node("select_layer", self._select_layer)
        graph_builder.add_node("fetch_features", self._fetch_features)
        graph_builder.add_node("spatial_filter", self._spatial_filter)
        graph_builder.add_node("finalize", self._finalize)

        graph_builder.set_entry_point("parse_prompt")
        graph_builder.add_conditional_edges(
            "parse_prompt",
            self._parse_branch,
            {
                "load_capabilities": "load_capabilities",
                "fetch_municipalities": "fetch_municipalities",
                "resolve_municipality": "resolve_municipality",
                "finalize": "finalize",
            },
        )
        graph_builder.add_conditional_edges(
            "load_capabilities",
            lambda state: "select_layer" if state.get("query_mode") == "filtered_layer" else "finalize",
            {
                "select_layer": "select_layer",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("fetch_municipalities", "finalize")
        graph_builder.add_conditional_edges(
            "resolve_municipality",
            lambda state: "finalize" if state.get("query_mode") == "municipality_boundary" else "load_capabilities",
            {
                "load_capabilities": "load_capabilities",
                "finalize": "finalize",
            },
        )
        graph_builder.add_conditional_edges(
            "select_layer",
            self._layer_branch,
            {
                "fetch_features": "fetch_features",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("fetch_features", "spatial_filter")
        graph_builder.add_edge("spatial_filter", "finalize")
        graph_builder.add_edge("finalize", END)

        return graph_builder.compile()

    async def run(self, prompt: str, selected_layer: str | None = None) -> dict[str, Any]:
        initial_state: AgentState = {
            "prompt": prompt,
            "selected_layer": selected_layer,
            "is_geo_request": True,
            "needs_layer_selection": False,
            "error": None,
        }
        result = await self.graph.ainvoke(initial_state)
        return {
            "reply": result.get("reply", "No response generated."),
            "chat_mode": "geospatial" if result.get("is_geo_request", True) else "general",
            "query_mode": result.get("query_mode"),
            "municipality": result.get("municipality_name"),
            "selected_layer": (result.get("chosen_layer") or {}).get("name"),
            "municipalities": result.get("municipalities", []),
            "layer_candidates": result.get("layer_candidates", []),
            "needs_layer_selection": bool(result.get("needs_layer_selection")),
            "geojson": result.get("filtered_geojson"),
        }


def run_sync(coro: Any) -> Any:
    return asyncio.run(coro)
