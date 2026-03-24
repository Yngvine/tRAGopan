#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: ./test_agent.sh [options]

Test the agent in two modes:
- direct: calls AgentGeoService directly in Python (best to debug 502 causes)
- api: calls a running Flask endpoint /api/agent/chat

Options:
  --mode <direct|api>         Test mode (default: direct)
  --prompt <text>             Prompt to test (default provided)
  --selected-layer <name>     Optional selected layer name
  --provider <ollama|rules>   LLM provider for direct mode (default: rules)
  --ollama-url <url>          Ollama base URL for direct mode
  --ollama-model <model>      Ollama model for direct mode
  --host <host>               API host for api mode (default: 127.0.0.1)
  --port <port>               API port for api mode (default: 8001)
  --timeout <seconds>         Request timeout (default: 90)
  -h, --help                  Show this help

Examples:
  ./test_agent.sh
  ./test_agent.sh --mode direct --provider ollama --ollama-url https://host:11434
  ./test_agent.sh --mode api --host 127.0.0.1 --port 8001
  ./test_agent.sh --mode api --selected-layer IDENA:DIADMI_Pol_Municipio
EOF
}

MODE="direct"
PROMPT="Show schools layer in municipality of Pamplona"
SELECTED_LAYER=""
PROVIDER="rules"
OLLAMA_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="llama3.1:8b"
HOST="127.0.0.1"
PORT="8001"
TIMEOUT="90"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --selected-layer)
      SELECTED_LAYER="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --ollama-url)
      OLLAMA_URL="$2"
      shift 2
      ;;
    --ollama-model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Error: .venv not found. Run: python -m venv .venv" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if [[ "$MODE" != "direct" && "$MODE" != "api" ]]; then
  echo "Error: --mode must be direct or api" >&2
  exit 1
fi

if [[ "$MODE" == "direct" ]]; then
  echo "[test_agent] mode=direct provider=${PROVIDER}"
  export LLM_PROVIDER="$PROVIDER"
  export OLLAMA_BASE_URL="$OLLAMA_URL"
  export OLLAMA_MODEL="$OLLAMA_MODEL"
  export TEST_PROMPT="$PROMPT"
  export TEST_SELECTED_LAYER="$SELECTED_LAYER"

  python - <<'PY'
import asyncio
import json
import os
import traceback

from idena_service.agent_graph import AgentGeoService
from idena_service.idena_client import IdenaClient

prompt = os.environ.get("TEST_PROMPT", "")
selected_layer = os.environ.get("TEST_SELECTED_LAYER") or None

async def main() -> None:
    client = IdenaClient()
    service = AgentGeoService(client=client)
    result = await service.run(prompt=prompt, selected_layer=selected_layer)
    print(json.dumps(result, indent=2, ensure_ascii=False))

try:
    asyncio.run(main())
except Exception as exc:
    print("[test_agent] direct mode failed with exception:")
    traceback.print_exc()
    raise SystemExit(1) from exc
PY
  exit 0
fi

echo "[test_agent] mode=api url=http://${HOST}:${PORT}/api/agent/chat"

if ! curl -sS --max-time "$TIMEOUT" "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "Error: API health check failed on http://${HOST}:${PORT}/health" >&2
  exit 1
fi

PAYLOAD=$(TEST_PROMPT="$PROMPT" TEST_SELECTED_LAYER="$SELECTED_LAYER" python - <<'PY'
import json
import os

payload = {"prompt": os.environ.get("TEST_PROMPT", "")}
selected = os.environ.get("TEST_SELECTED_LAYER", "").strip()
if selected:
    payload["selected_layer"] = selected
print(json.dumps(payload, ensure_ascii=False))
PY
)

RESP_FILE="$(mktemp)"
HTTP_CODE=$(curl -sS --max-time "$TIMEOUT" \
  -o "$RESP_FILE" \
  -w "%{http_code}" \
  -X POST "http://${HOST}:${PORT}/api/agent/chat" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

if [[ "$HTTP_CODE" -ge 400 ]]; then
  echo "[test_agent] API returned HTTP ${HTTP_CODE}" >&2
  cat "$RESP_FILE" >&2
  rm -f "$RESP_FILE"
  exit 1
fi

echo "[test_agent] API returned HTTP ${HTTP_CODE}"
cat "$RESP_FILE"
rm -f "$RESP_FILE"
