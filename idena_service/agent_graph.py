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
    municipality_hints: list[str]
    layer_hint: str | None
    municipality_name: str | None
    municipality_names: list[str]
    municipality_geojson: dict[str, Any] | None
    capabilities: list[dict[str, str]]
    layer_candidates: list[dict[str, str]]
    chosen_layer: dict[str, str] | None
    needs_layer_selection: bool
    raw_geojson: dict[str, Any] | None
    filtered_geojson: dict[str, Any] | None
    municipalities: list[str]
    query_mode: str
    global_scope: bool
    is_geo_request: bool
    conversation_context: dict[str, Any]
    pagination_cursor: int
    page_size: int
    pagination: dict[str, Any]
    tool_plan: list[str]
    reply: str
    actions: list[dict[str, Any]]
    trace: list[str]
    warnings: list[str]
    error: str | None


class LLMAdapter:
    async def extract_scope(self, prompt: str) -> dict[str, Any]:
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

    async def extract_scope(self, prompt: str) -> dict[str, Any]:
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
            "municipalities": [municipality] if municipality else [],
            "layer_hint": layer_hint,
            "query_mode": None,
            "planned_tools": [],
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
    def _tool_definitions() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_municipalities_navarra",
                    "description": "List all municipalities in Navarra.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_municipality_boundary",
                    "description": "Get municipality boundary by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "municipality": {"type": "string"},
                            "municipalities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_toponyms_in_municipality",
                    "description": "Get toponyms inside a municipality. Optional text filter query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "municipality": {"type": "string"},
                            "query": {"type": "string"},
                        },
                        "required": ["municipality"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_layer_in_municipality",
                    "description": "Get WFS layer features clipped to a municipality.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "municipality": {"type": "string"},
                            "layer_hint": {"type": "string"},
                        },
                        "required": ["municipality", "layer_hint"],
                    },
                },
            },
        ]

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
    def _mode_from_tool_name(tool_name: str) -> str | None:
        mapping = {
            "list_municipalities_navarra": "municipalities",
            "get_municipality_boundary": "municipality_boundary",
            "get_toponyms_in_municipality": "toponymy_in_municipality",
            "get_layer_in_municipality": "filtered_layer",
        }
        return mapping.get(tool_name)

    @staticmethod
    def _parse_tool_calls(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        municipality: str | None = None
        municipalities: list[str] = []
        layer_hint: str | None = None
        query_mode: str | None = None
        planned_tools: list[str] = []

        for call in tool_calls:
            function_data = call.get("function", {}) if isinstance(call, dict) else {}
            tool_name = str(function_data.get("name", "")).strip()
            if not tool_name:
                continue
            planned_tools.append(tool_name)

            mode = OllamaAdapter._mode_from_tool_name(tool_name)
            if mode:
                query_mode = mode

            args_raw = function_data.get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}

            municipality = str(args.get("municipality", municipality or "")).strip() or municipality
            raw_munis = args.get("municipalities", [])
            if isinstance(raw_munis, list):
                for item in raw_munis:
                    value = str(item).strip()
                    if value and value not in municipalities:
                        municipalities.append(value)

            if tool_name == "get_toponyms_in_municipality":
                layer_hint = str(args.get("query", layer_hint or "")).strip() or layer_hint
            if tool_name == "get_layer_in_municipality":
                layer_hint = str(args.get("layer_hint", layer_hint or "")).strip() or layer_hint

        if municipality and municipality not in municipalities:
            municipalities.insert(0, municipality)

        return {
            "municipality": municipality,
            "municipalities": municipalities,
            "layer_hint": layer_hint,
            "query_mode": query_mode,
            "planned_tools": planned_tools,
        }

    @staticmethod
    def _parse_content(data: dict[str, Any]) -> dict[str, Any]:
        message = data.get("message", {}) if isinstance(data, dict) else {}
        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, list) and tool_calls:
            parsed_tools = OllamaAdapter._parse_tool_calls(tool_calls)
            if parsed_tools.get("query_mode"):
                return parsed_tools

        # Fallback to JSON content.
        raw_content = str(message.get("content", "")).strip()
        parsed = OllamaAdapter._extract_json_object(raw_content)
        return {
            "municipality": parsed.get("municipality"),
            "municipalities": parsed.get("municipalities") or [],
            "layer_hint": parsed.get("layer_hint"),
            "query_mode": parsed.get("query_mode"),
            "planned_tools": parsed.get("planned_tools") or [],
        }

    @staticmethod
    def _parse_langchain_content(content: Any) -> dict[str, Any]:
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
            "municipalities": parsed.get("municipalities") or [],
            "layer_hint": parsed.get("layer_hint"),
            "query_mode": parsed.get("query_mode"),
            "planned_tools": parsed.get("planned_tools") or [],
        }

    async def extract_scope(self, prompt: str) -> dict[str, Any]:
        system_instruction = (
            "You are a planner for a geospatial agent over IDENA WFS. "
            "Use a strict think-then-act approach internally (ReAct style), then choose tools and infer query_mode. "
            "Return ONLY valid JSON with keys municipality, municipalities, layer_hint, query_mode, planned_tools. "
            "query_mode must be one of: municipalities, municipality_boundary, toponymy_in_municipality, filtered_layer, general. "
            "planned_tools must be an array of tool names from the provided catalog. "
            "If user asks multiple municipalities, fill municipalities as a list and set municipality to first item. "
            "If the user mentions a place with patterns like 'en X', 'in X', or trailing 'de X', treat X as municipality when plausible. "
            "Use null when municipality or layer_hint are missing."
        )

        langchain_error: Exception | None = None

        request_body_schema = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "tools": self._tool_definitions(),
            "format": {
                "type": "object",
                "properties": {
                    "municipality": {"type": ["string", "null"]},
                    "municipalities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "layer_hint": {"type": ["string", "null"]},
                    "query_mode": {
                        "type": "string",
                        "enum": [
                            "municipalities",
                            "municipality_boundary",
                            "toponymy_in_municipality",
                            "filtered_layer",
                            "general",
                        ],
                    },
                    "planned_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["municipality", "municipalities", "layer_hint", "query_mode", "planned_tools"],
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
            "tools": self._tool_definitions(),
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
                except httpx.HTTPError:
                    pass

        # Last fallback path: LangChain client without explicit tool payload.
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
            langchain_error = exc
            raise RuntimeError(
                f"Failed to call Ollama at {self.base_url}/api/chat and LangChain fallback: {langchain_error}"
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
        model = os.getenv("OLLAMA_MODEL", "llama3-groq-tool-use")
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
            "riego",
            "bocas",
        ]
        return any(token in text for token in geo_hints)

    @staticmethod
    def _semantic_layer_catalog() -> list[tuple[str, str]]:
        return [
            ("acometida", "acometidas abastecimiento saneamiento"),
            ("acometidas", "acometidas abastecimiento saneamiento"),
            ("abastecimiento", "abastecimiento agua red"),
            ("saneamiento", "saneamiento alcantarillado"),
            ("alcantarillado", "saneamiento alcantarillado"),
            ("toponimo", "toponimos toponimia"),
            ("toponimos", "toponimos toponimia"),
            ("topónimo", "toponimos toponimia"),
            ("topónimos", "toponimos toponimia"),
            ("colegio", "escuelas centros educativos"),
            ("colegios", "escuelas centros educativos"),
            ("escuela", "escuelas centros educativos"),
            ("escuelas", "escuelas centros educativos"),
            ("carretera", "carreteras viario"),
            ("carreteras", "carreteras viario"),
            ("rio", "rios hidrografia"),
            ("río", "rios hidrografia"),
            ("rios", "rios hidrografia"),
            ("ríos", "rios hidrografia"),
            ("parcela", "parcelas catastral"),
            ("parcelas", "parcelas catastral"),
            ("riego", "riego regadio bocas hidrantes"),
            ("boca de riego", "riego regadio bocas hidrantes"),
            ("bocas de riego", "riego regadio bocas hidrantes"),
            ("hidrante", "riego regadio bocas hidrantes"),
            ("hidrantes", "riego regadio bocas hidrantes"),
        ]

    def _infer_semantic_layer_hint(self, prompt: str) -> str | None:
        prompt_norm = self.client._normalize_text(prompt)
        for token, hint in self._semantic_layer_catalog():
            if token in prompt_norm:
                return hint
        return None

    @staticmethod
    def _clean_layer_hint_from_prompt(prompt: str) -> str:
        text = prompt.lower()
        text = re.sub(
            r"\b(muestrame|mu[eé]strame|mostrar|quiero|ver|ensename|ens[eé]name|"
            r"dame|dime|de|en|para|con|sin|el|la|los|las|un|una|unos|unas|"
            r"capa|capas|layer|layers|municipio|municipios|municipality|municipalities|"
            r"navarra|toda|todo|entera|completa)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _has_explicit_municipality_reference(prompt: str) -> bool:
        text = prompt.lower()
        return bool(
            re.search(r"\bmunicip(?:io|ality)(?:s)?\b", text, flags=re.IGNORECASE)
            or re.search(r"\ben\s+[a-zà-ÿ\-\s/]+$", text, flags=re.IGNORECASE)
            or re.search(r"\bin\s+[a-zà-ÿ\-\s/]+$", text, flags=re.IGNORECASE)
            or re.search(r"\bde\s+[a-zà-ÿ\-\s/']+$", text, flags=re.IGNORECASE)
        )

    @staticmethod
    def _is_likely_municipality_candidate(value: str) -> bool:
        candidate = value.strip().lower()
        if not candidate or len(candidate) < 3:
            return False

        forbidden_substrings = [
            "acomet",
            "riego",
            "hidr",
            "topon",
            "capa",
            "layer",
            "wfs",
            "aerogener",
            "eolic",
            "parque",
            "energia",
            "agua",
            "saneamiento",
            "alcantar",
            "carretera",
            "rio",
            "río",
            "parcela",
        ]
        if any(token in candidate for token in forbidden_substrings):
            return False

        # Avoid treating generic helper words as municipality values.
        generic_words = {
            "navarra",
            "toda",
            "todo",
            "entera",
            "completa",
            "disponibles",
            "available",
            "capas",
            "layers",
        }
        if candidate in generic_words:
            return False

        return True

    @staticmethod
    def _is_followup_municipality_prompt(prompt: str) -> bool:
        text = prompt.lower().strip(" ?!.,;")
        patterns = [
            r"^y\s+en\s+[a-zà-ÿ\-\s/]+$",
            r"^en\s+[a-zà-ÿ\-\s/]+$",
            r"^ahora\s+en\s+[a-zà-ÿ\-\s/]+$",
        ]
        return any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _extract_direct_layer_municipality(prompt: str) -> tuple[str | None, str | None]:
        text = prompt.strip()
        if not text:
            return None, None

        # Example: "muestrame la capa de acometidas de pamplona"
        match = re.search(
            r"\b(?:muestrame|mu[eé]strame|mostrar|quiero|dame|ens[eé]n?ame|cargar)?\s*"
            r"(?:la\s+|el\s+)?(?:capa|layer)\s+de\s+(.+?)\s+(?:en|de)\s+"
            r"([a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\-\s/']*)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None, None

        layer_hint = match.group(1).strip(" .,:;!?") or None
        municipality = match.group(2).strip(" .,:;!?") or None
        return layer_hint, municipality

    @staticmethod
    def _is_global_scope_prompt(prompt: str) -> bool:
        text = prompt.lower().strip()
        if not text:
            return False

        direct_hints = [
            "toda navarra",
            "todo navarra",
            "all navarra",
            "whole navarra",
            "de navarra",
            "en navarra",
        ]
        if any(hint in text for hint in direct_hints):
            return True

        return bool(re.search(r"\bnavarra\b", text, flags=re.IGNORECASE))

    @staticmethod
    def _split_municipality_candidates(raw: str) -> list[str]:
        if not raw:
            return []
        text = raw.strip(" .,:;!?")
        parts = re.split(r"\s+(?:y|e|and)\s+|,\s*", text, flags=re.IGNORECASE)
        cleaned: list[str] = []
        for part in parts:
            value = part.strip(" .,:;!?")
            if value and value.lower() not in {"municipio", "municipios", "municipality", "municipalities"}:
                cleaned.append(value)
        unique: list[str] = []
        for value in cleaned:
            if value not in unique:
                unique.append(value)
        return unique

    def _extract_municipality_hints(
        self,
        prompt: str,
        parsed: dict[str, Any],
        context: dict[str, Any],
        allow_context: bool = True,
    ) -> list[str]:
        hints: list[str] = []

        text = prompt.strip()
        pattern = r"\bmunicip(?:io|ality)(?:s)?\b\s*(?:de|of)?\s+(.+)$"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            tail = re.sub(r"\b(en|in)\b.*$", "", match.group(1), flags=re.IGNORECASE).strip()
            for value in self._split_municipality_candidates(tail):
                if self._is_likely_municipality_candidate(value) and value not in hints:
                    hints.append(value)

        # Support natural prompts like "capa X en Tudela" where municipality keyword
        # is not explicitly written.
        location_pattern = r"\b(?:en|in)\s+([a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\-\s/']*)"
        for location_match in re.finditer(location_pattern, text, flags=re.IGNORECASE):
            tail = location_match.group(1)
            tail = re.split(
                r"\b(?:con|sin|de|del|para|por|where|with|without|using|que|which)\b",
                tail,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;!?")
            for value in self._split_municipality_candidates(tail):
                if self._is_likely_municipality_candidate(value) and value not in hints:
                    hints.append(value)

        # Support Spanish prompts like "capa de X de Pamplona" by only taking
        # the trailing final "de <location>" candidate.
        trailing_de = re.search(
            r"\bde\s+([a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\-\s/']*)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if trailing_de:
            tail = trailing_de.group(1).strip(" .,:;!?")
            for value in self._split_municipality_candidates(tail):
                if self._is_likely_municipality_candidate(value) and value not in hints:
                    hints.append(value)

        # Only trust LLM municipality fields when the prompt references municipality location,
        # and never accept obvious layer-like tokens as municipality names.
        prompt_has_muni_ref = self._has_explicit_municipality_reference(prompt)
        raw_list = parsed.get("municipalities") or []
        if prompt_has_muni_ref and isinstance(raw_list, list):
            for item in raw_list:
                for value in self._split_municipality_candidates(str(item)):
                    if self._is_likely_municipality_candidate(value) and value not in hints:
                        hints.append(value)

        parsed_single = parsed.get("municipality")
        if prompt_has_muni_ref and parsed_single:
            for value in self._split_municipality_candidates(str(parsed_single)):
                if self._is_likely_municipality_candidate(value) and value not in hints:
                    hints.append(value)

        if allow_context and not hints:
            context_muni = context.get("municipality")
            if isinstance(context_muni, str) and context_muni.strip():
                hints.append(context_muni.strip())

        return hints

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
            "qué capas",
            "capas disponibles",
            "muestrame las capas",
            "muéstrame las capas",
            "mostrar capas",
            "layers disponibles",
            "que layers",
            "qué layers",
            "layers hay",
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
            r"\b(muestrame|mu[eé]strame|mostrar|dime|lista|listar|que|qu[eé])\b[\s\w]{0,40}\b(capas|layers)\b[\s\w]{0,40}\b(disponibles|hay|idena|wfs)?\b",
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

    @staticmethod
    def _is_toponymy_in_municipality_prompt(prompt: str) -> bool:
        text = prompt.lower().strip()
        if not text:
            return False

        toponymy_hints = [
            "toponimo",
            "toponimos",
            "topónimo",
            "topónimos",
            "toponym",
            "toponyms",
        ]
        municipality_hints = ["municipio", "municipality", "en ", "in "]
        return any(hint in text for hint in toponymy_hints) and any(
            hint in text for hint in municipality_hints
        )

    async def _parse_prompt(self, state: AgentState) -> AgentState:
        prompt = state["prompt"]
        context = state.get("conversation_context") or {}

        if self._is_toponymy_in_municipality_prompt(prompt):
            try:
                parsed = await self.llm.extract_scope(state["prompt"])
            except Exception:
                parsed = await self.rule_fallback.extract_scope(state["prompt"])
            municipality_hints = self._extract_municipality_hints(prompt, parsed, context)
            municipality = municipality_hints[0] if municipality_hints else None
            semantic_hint = self._infer_semantic_layer_hint(prompt)
            return {
                "is_geo_request": True,
                "query_mode": "toponymy_in_municipality",
                "global_scope": False,
                "municipality_hint": municipality,
                "municipality_hints": municipality_hints,
                "layer_hint": semantic_hint or parsed.get("layer_hint") or prompt,
                "tool_plan": parsed.get("planned_tools") or [],
                "needs_layer_selection": False,
            }

        if self._is_municipality_boundary_prompt(prompt):
            try:
                parsed = await self.llm.extract_scope(state["prompt"])
            except Exception:
                parsed = await self.rule_fallback.extract_scope(state["prompt"])
            municipality_hints = self._extract_municipality_hints(prompt, parsed, context)
            municipality = municipality_hints[0] if municipality_hints else None
            return {
                "is_geo_request": True,
                "query_mode": "municipality_boundary",
                "global_scope": False,
                "municipality_hint": municipality,
                "municipality_hints": municipality_hints,
                "tool_plan": parsed.get("planned_tools") or [],
                "needs_layer_selection": False,
            }

        if self._is_municipalities_prompt(prompt):
            return {
                "is_geo_request": True,
                "query_mode": "municipalities",
                "global_scope": True,
                "needs_layer_selection": False,
            }

        if self._is_capabilities_prompt(prompt):
            return {
                "is_geo_request": True,
                "query_mode": "capabilities",
                "global_scope": True,
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
                "global_scope": False,
                "needs_layer_selection": False,
                "reply": reply,
            }

        try:
            parsed = await self.llm.extract_scope(state["prompt"])
        except Exception:
            # Keep geospatial pipeline available even if Ollama endpoint fails.
            parsed = await self.rule_fallback.extract_scope(state["prompt"])

        semantic_hint = self._infer_semantic_layer_hint(prompt)
        direct_layer_hint, direct_municipality = self._extract_direct_layer_municipality(prompt)
        followup = self._is_followup_municipality_prompt(prompt)
        selected_layer = state.get("selected_layer")

        municipality_hints = self._extract_municipality_hints(prompt, parsed, context, allow_context=False)
        if direct_municipality and self._is_likely_municipality_candidate(direct_municipality):
            if direct_municipality not in municipality_hints:
                municipality_hints.insert(0, direct_municipality)
        municipality = municipality_hints[0] if municipality_hints else None
        layer_hint = selected_layer or direct_layer_hint or semantic_hint or parsed.get("layer_hint")
        global_scope = self._is_global_scope_prompt(prompt)

        if municipality and self.client._normalize_text(municipality) == "navarra":
            municipality = None
            municipality_hints = []
            global_scope = True

        if not municipality_hints:
            global_scope = True

        if followup and not layer_hint:
            layer_hint = context.get("layer_hint")

        query_mode = "filtered_layer"
        llm_query_mode = str(parsed.get("query_mode") or "").strip().lower()
        if llm_query_mode in {
            "municipalities",
            "municipality_boundary",
            "toponymy_in_municipality",
            "filtered_layer",
        }:
            query_mode = llm_query_mode
        if followup and context.get("query_mode") in {"filtered_layer", "toponymy_in_municipality"}:
            query_mode = str(context.get("query_mode"))

        if query_mode == "toponymy_in_municipality" and global_scope:
            query_mode = "filtered_layer"

        return {
            "is_geo_request": True,
            "query_mode": query_mode,
            "global_scope": global_scope,
            "municipality_hint": municipality,
            "municipality_hints": municipality_hints,
            "layer_hint": layer_hint or state["prompt"],
            "tool_plan": parsed.get("planned_tools") or [],
        }

    def _build_trace(self, result: AgentState) -> list[str]:
        trace: list[str] = []
        prompt = str(result.get("prompt") or "").strip()
        if prompt:
            trace.append(f"Analyzing request: '{prompt}'.")

        if not result.get("is_geo_request", True):
            trace.append("Classified as general chat (no geospatial workflow).")
            return trace

        mode = str(result.get("query_mode") or "filtered_layer")
        trace.append(f"Inferred mode: {mode}.")

        muni_hints = result.get("municipality_hints") or []
        if muni_hints:
            trace.append(f"Searching municipality(ies): {', '.join(muni_hints)}.")

        municipality_name = result.get("municipality_name")
        if municipality_name:
            trace.append(f"Resolved municipality: {municipality_name}.")
        elif result.get("global_scope"):
            trace.append("No explicit municipality: using Navarra-wide scope.")

        if mode in {"filtered_layer", "toponymy_in_municipality"}:
            layer_hint = str(result.get("layer_hint") or "").strip()
            if layer_hint:
                trace.append(f"Searching layers with hint: '{layer_hint}'.")

        candidates = result.get("layer_candidates") or []
        if candidates:
            names = [str(layer.get("name", "")) for layer in candidates if layer.get("name")]
            if names:
                trace.append(f"Candidate layers: {', '.join(names)}.")

        chosen_layer = (result.get("chosen_layer") or {}).get("name")
        if chosen_layer:
            trace.append(f"Selected layer: {chosen_layer}.")

        if mode == "filtered_layer":
            if result.get("global_scope"):
                trace.append("Applying query without municipality filter (Navarra-wide).")
            else:
                trace.append("Applying spatial filter by municipality boundary.")

        filtered_geojson = result.get("filtered_geojson") or {}
        total = len(filtered_geojson.get("features", []) or [])
        if total:
            trace.append(f"Renderable results: {total} feature(s).")

        actions = result.get("actions") or []
        has_boundary = any(
            action.get("type") == "add_layer" and action.get("layer_role") == "municipality_boundary"
            for action in actions
            if isinstance(action, dict)
        )
        has_filtered = any(
            action.get("type") == "add_layer" and action.get("layer_role") in {"filtered_layer", "toponymy_points"}
            for action in actions
            if isinstance(action, dict)
        )
        if has_boundary and has_filtered:
            trace.append("Final render: municipality boundary + filtered requested layer.")

        if result.get("error"):
            trace.append(f"Execution error: {result.get('error')}.")

        return trace

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
        if mode == "toponymy_in_municipality":
            return "resolve_municipality"
        if mode == "filtered_layer":
            if state.get("global_scope"):
                return "load_capabilities"
            return "resolve_municipality"
        return "finalize"

    async def _fetch_municipalities(self, state: AgentState) -> AgentState:
        ordered = await self.client.list_municipalities()
        return {
            "municipalities": ordered,
        }

    async def _fetch_toponymy_in_municipality(self, state: AgentState) -> AgentState:
        municipality_name = state.get("municipality_name")
        if not municipality_name:
            return {"error": "Please include a municipality for toponymy queries."}

        pagination_cursor = max(0, int(state.get("pagination_cursor") or 0))
        page_size = max(50, min(2000, int(state.get("page_size") or 500)))

        data = await self.client.get_toponyms_in_municipality(
            municipality_name=municipality_name,
            q=state.get("layer_hint"),
            limit=page_size,
            start_index=pagination_cursor,
            page_scan_size=500,
            max_scan_pages=8,
        )
        return {
            "filtered_geojson": {
                "type": "FeatureCollection",
                "features": data.get("features", []),
                "municipality": municipality_name,
                "selected_layer": self.client.TOPONIMIA_TYPENAME,
            },
            "warnings": data.get("warnings", []),
            "pagination": data.get("pagination", {}),
        }

    async def _resolve_municipality(self, state: AgentState) -> AgentState:
        municipality_hints = state.get("municipality_hints") or []
        municipality_hint = state.get("municipality_hint")
        if not municipality_hints and municipality_hint:
            municipality_hints = [municipality_hint]

        if not municipality_hints:
            return {
                "error": "Please include a municipality to define the area filter.",
                "needs_layer_selection": False,
            }

        all_features: list[dict[str, Any]] = []
        resolved_names: list[str] = []
        missing: list[str] = []

        for hint in municipality_hints:
            boundary = await self.client.get_municipality_boundary(hint)
            features = boundary.get("features", [])
            if not features:
                missing.append(hint)
                continue

            name = hint
            props = features[0].get("properties", {})
            for key in ("MUNICIPIO", "municipio", "nombre", "name"):
                if props.get(key):
                    name = str(props[key])
                    break

            if name not in resolved_names:
                resolved_names.append(name)
            all_features.extend(features)

        if not all_features:
            return {
                "error": f"Municipality '{municipality_hints[0]}' was not found.",
                "needs_layer_selection": False,
            }

        warnings = state.get("warnings", [])
        if missing:
            warnings = warnings + [f"Municipalities not found: {', '.join(missing)}"]

        combined_geojson = {
            "type": "FeatureCollection",
            "features": all_features,
        }
        return {
            "municipality_name": ", ".join(resolved_names),
            "municipality_names": resolved_names,
            "municipality_geojson": combined_geojson,
            "warnings": warnings,
        }

    async def _load_capabilities(self, state: AgentState) -> AgentState:
        layers = await self.client.get_capabilities(ttl_seconds=900)
        return {"capabilities": layers}

    async def _select_layer(self, state: AgentState) -> AgentState:
        layer_hint = state.get("layer_hint") or state["prompt"]
        layers = state.get("capabilities", [])

        hints: list[str] = []
        for value in [
            layer_hint,
            state.get("prompt"),
            self._infer_semantic_layer_hint(state.get("prompt", "")),
            self._clean_layer_hint_from_prompt(state.get("prompt", "")),
        ]:
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned and cleaned not in hints:
                    hints.append(cleaned)

        seen: set[str] = set()
        candidates: list[dict[str, str]] = []
        for hint in hints:
            for layer in self.client.search_layers(layers, hint, limit=5):
                name = layer.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    candidates.append(layer)
                if len(candidates) >= 5:
                    break
            if len(candidates) >= 5:
                break

        candidates = candidates[:3]

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
        if not chosen_layer:
            return {"error": "Missing chosen layer."}

        global_scope = bool(state.get("global_scope"))

        bbox: tuple[float, float, float, float] | None = None
        if not global_scope:
            if not municipality_geojson:
                return {"error": "Missing municipality geometry."}

            muni_features = municipality_geojson.get("features", [])
            if not muni_features:
                return {"error": "Municipality geometry is empty."}

            muni_geom = shape(muni_features[0]["geometry"])
            minx, miny, maxx, maxy = muni_geom.bounds
            bbox = (minx, miny, maxx, maxy)

        pagination_cursor = max(0, int(state.get("pagination_cursor") or 0))
        page_size = max(50, min(2000, int(state.get("page_size") or 500)))

        raw = await self.client.get_layer_features(
            type_name=chosen_layer["name"],
            bbox=bbox,
            count=500,
            start_index=pagination_cursor,
        )
        raw_features = raw.get("features", [])
        has_more = len(raw_features) >= 500
        next_cursor = pagination_cursor + len(raw_features) if has_more else None
        return {
            "raw_geojson": raw,
            "pagination": {
                "has_more": has_more,
                "next_cursor": next_cursor,
                "page_size": page_size,
            },
        }

    async def _spatial_filter(self, state: AgentState) -> AgentState:
        if state.get("needs_layer_selection"):
            return {}

        raw_geojson = state.get("raw_geojson") or {}
        municipality_geojson = state.get("municipality_geojson") or {}
        raw_features = raw_geojson.get("features", [])
        muni_features = municipality_geojson.get("features", [])
        global_scope = bool(state.get("global_scope"))

        if not raw_features:
            return {
                "filtered_geojson": {
                    "type": "FeatureCollection",
                    "features": [],
                }
            }

        if global_scope:
            page_size = max(50, min(2000, int(state.get("page_size") or 500)))
            return {
                "filtered_geojson": {
                    "type": "FeatureCollection",
                    "features": raw_features[:page_size],
                    "selected_layer": state.get("chosen_layer", {}).get("name"),
                    "municipality": None,
                },
                "warnings": (state.get("warnings") or []) + [
                    "No municipal filter applied: showing Navarra-wide results."
                ],
            }

        if not muni_features:
            return {"error": "Municipality geometry is missing for spatial filtering."}

        # Merge municipality geometry in case a municipality returns multiple polygons.
        muni_geometry = unary_union([shape(feat["geometry"]) for feat in muni_features if feat.get("geometry")])

        page_size = max(50, min(2000, int(state.get("page_size") or 500)))
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
                if len(filtered_features) >= page_size:
                    break

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
        warnings = state.get("warnings", [])

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
            return {"reply": "\n".join(lines), "actions": []}

        if mode == "municipalities":
            municipalities = state.get("municipalities", [])
            if not municipalities:
                return {"reply": "I could not retrieve municipalities for Navarra right now."}

            preview = municipalities[:40]
            lines = [f"Navarra has {len(municipalities)} municipalities in the available dataset. First {len(preview)}:"]
            lines.append(", ".join(preview))
            lines.append("Ask for a municipality by name and I can continue with a layer query.")
            return {"reply": "\n".join(lines), "actions": []}

        if mode == "municipality_boundary":
            if state.get("error"):
                return {"reply": state["error"]}

            municipality_geojson = state.get("municipality_geojson")
            municipality_name = state.get("municipality_name", "the selected municipality")
            municipality_names = state.get("municipality_names", [])
            if not municipality_geojson:
                return {"reply": "I could not retrieve the municipality boundary right now.", "actions": []}

            if len(municipality_names) > 1:
                reply = f"Loaded boundaries for municipalities: {', '.join(municipality_names)}."
            else:
                reply = f"Loaded boundary for municipality '{municipality_name}'."
            return {
                "reply": reply,
                "filtered_geojson": municipality_geojson,
                "actions": [
                    {"type": "clear_layers"},
                    {
                        "type": "add_layer",
                        "layer_role": "municipality_boundary",
                        "source": "municipality_geojson",
                        "style": {"stroke": "#63d9c3", "fill": "#63d9c3", "fillOpacity": 0.06},
                        "count_in_stats": False,
                    },
                    {"type": "fit_bounds", "source": "geojson"},
                ],
                "warnings": warnings,
                "conversation_context": {
                    "municipality": municipality_name,
                    "query_mode": "municipality_boundary",
                },
            }

        if mode == "toponymy_in_municipality":
            if state.get("error"):
                return {"reply": state["error"], "actions": []}

            filtered_geojson = state.get("filtered_geojson") or {"type": "FeatureCollection", "features": []}
            total = len(filtered_geojson.get("features", []))
            municipality = state.get("municipality_name", "the selected municipality")
            municipality_geojson = state.get("municipality_geojson")
            actions: list[dict[str, Any]] = [{"type": "clear_layers"}]
            if municipality_geojson and municipality_geojson.get("features"):
                actions.append(
                    {
                        "type": "add_layer",
                        "layer_role": "municipality_boundary",
                        "source": "municipality_geojson",
                        "style": {"stroke": "#63d9c3", "fill": "#63d9c3", "fillOpacity": 0.05},
                        "count_in_stats": False,
                    }
                )
            actions.extend(
                [
                    {
                        "type": "add_layer",
                        "layer_role": "toponymy_points",
                        "source": "geojson",
                        "style": {"pointColor": "#4f8cff", "radius": 5},
                    },
                    {"type": "fit_bounds", "source": "geojson"},
                ]
            )
            return {
                "reply": f"Found {total} toponyms inside municipality '{municipality}'.",
                "actions": actions,
                "warnings": warnings,
                "conversation_context": {
                    "municipality": municipality,
                    "layer_hint": state.get("layer_hint") or "toponimia",
                    "query_mode": "toponymy_in_municipality",
                },
            }

        if state.get("error"):
            return {
                "reply": state["error"],
                "actions": [],
            }

        if state.get("needs_layer_selection"):
            candidates = state.get("layer_candidates", [])
            lines = ["I found multiple matching layers. Please pick one:"]
            for idx, layer in enumerate(candidates, start=1):
                title = layer.get("title") or "No title"
                lines.append(f"{idx}. {layer['name']} - {title}")
            return {
                "reply": "\n".join(lines),
                "actions": [],
            }

        filtered_geojson = state.get("filtered_geojson") or {"type": "FeatureCollection", "features": []}
        if not state.get("municipality_name"):
            if state.get("global_scope"):
                pass
            else:
                return {
                    "reply": "No pude resolver el municipio para esa consulta. Indica un municipio explicitamente.",
                    "actions": [],
                }
        if not state.get("chosen_layer"):
            return {
                "reply": "No pude resolver una capa valida para esa consulta. Pide 'capas disponibles' o especifica la capa.",
                "actions": [],
            }
        total = len(filtered_geojson.get("features", []))
        municipality = state.get("municipality_name", "Navarra")
        chosen_layer = state.get("chosen_layer", {}).get("name", "unknown layer")
        if state.get("global_scope"):
            reply_text = f"Found {total} features in layer '{chosen_layer}' across Navarra (no municipality filter)."
        else:
            reply_text = (
                f"Found {total} features in layer '{chosen_layer}' "
                f"intersecting municipality '{municipality}'."
            )
        return {
            "reply": reply_text,
            "actions": (
                [
                    {"type": "clear_layers"},
                    {
                        "type": "add_layer",
                        "layer_role": "municipality_boundary",
                        "source": "municipality_geojson",
                        "style": {"stroke": "#63d9c3", "fill": "#63d9c3", "fillOpacity": 0.05},
                        "count_in_stats": False,
                    },
                    {
                        "type": "add_layer",
                        "layer_role": "filtered_layer",
                        "source": "geojson",
                        "style": {"stroke": "#ffd166", "fill": "#ffd166", "fillOpacity": 0.15},
                    },
                    {"type": "fit_bounds", "source": "geojson"},
                ]
                if not state.get("global_scope")
                else [
                    {"type": "clear_layers"},
                    {
                        "type": "add_layer",
                        "layer_role": "filtered_layer",
                        "source": "geojson",
                        "style": {"stroke": "#ffd166", "fill": "#ffd166", "fillOpacity": 0.15},
                    },
                    {"type": "fit_bounds", "source": "geojson"},
                ]
            ),
            "warnings": warnings,
            "conversation_context": {
                "municipality": None if state.get("global_scope") else municipality,
                "layer_hint": state.get("layer_hint") or chosen_layer,
                "query_mode": "filtered_layer",
            },
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
        graph_builder.add_node("fetch_toponymy_in_municipality", self._fetch_toponymy_in_municipality)
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
            lambda state: (
                "finalize"
                if state.get("query_mode") == "municipality_boundary"
                else "fetch_toponymy_in_municipality"
                if state.get("query_mode") == "toponymy_in_municipality"
                else "load_capabilities"
            ),
            {
                "load_capabilities": "load_capabilities",
                "fetch_toponymy_in_municipality": "fetch_toponymy_in_municipality",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("fetch_toponymy_in_municipality", "finalize")
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

    async def run(
        self,
        prompt: str,
        selected_layer: str | None = None,
        conversation_context: dict[str, Any] | None = None,
        pagination_cursor: int = 0,
        page_size: int = 500,
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            "prompt": prompt,
            "selected_layer": selected_layer,
            "is_geo_request": True,
            "needs_layer_selection": False,
            "conversation_context": conversation_context or {},
            "pagination_cursor": pagination_cursor,
            "page_size": page_size,
            "error": None,
        }
        result = await self.graph.ainvoke(initial_state)
        trace = self._build_trace(result)
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
            "municipality_geojson": result.get("municipality_geojson"),
            "actions": result.get("actions", []),
            "warnings": result.get("warnings", []),
            "pagination": result.get("pagination", {}),
            "context_update": result.get("conversation_context", {}),
            "tool_plan": result.get("tool_plan", []),
            "trace": trace,
        }


def run_sync(coro: Any) -> Any:
    return asyncio.run(coro)
