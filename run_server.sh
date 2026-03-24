#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: ./run_server.sh [options]

Options:
  --host <host>             Host to bind (default: 127.0.0.1)
  --port <port>             Preferred port (default: 8001)
  --provider <provider>     LLM provider: ollama|rules (default: ollama)
  --ollama-url <url>        Ollama base URL (default: https://eim-alu-83071.tail3405b4.ts.net/)
  --ollama-model <model>    Ollama model (default: llama3-groq-tool-use)
  --no-auto-port            Do not search for next free port if occupied
  -h, --help                Show this help

Examples:
  ./run_server.sh
  ./run_server.sh --provider rules
  ./run_server.sh --ollama-url https://myhost:11434 --port 8002
EOF
}

HOST="127.0.0.1"
PORT="8001"
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="https://eim-alu-83071.tail3405b4.ts.net/"
OLLAMA_MODEL="llama3-groq-tool-use"
AUTO_PORT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --provider)
      LLM_PROVIDER="$2"
      shift 2
      ;;
    --ollama-url)
      OLLAMA_BASE_URL="$2"
      shift 2
      ;;
    --ollama-model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --no-auto-port)
      AUTO_PORT=0
      shift
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
  echo "Error: .venv not found. Create it first: python -m venv .venv" >&2
  exit 1
fi

# shellcheck source=/dev/null
source ".venv/bin/activate"

is_port_in_use() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -q ":${p}$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -i ":${p}" >/dev/null 2>&1
  else
    return 1
  fi
}

if [[ "$AUTO_PORT" -eq 1 ]]; then
  if is_port_in_use "$PORT"; then
    ORIGINAL_PORT="$PORT"
    while is_port_in_use "$PORT"; do
      PORT="$((PORT + 1))"
    done
    echo "Port ${ORIGINAL_PORT} is busy. Using ${PORT} instead."
  fi
else
  if is_port_in_use "$PORT"; then
    echo "Error: Port ${PORT} is already in use."
    exit 1
  fi
fi

export HOST
export PORT
export LLM_PROVIDER
export OLLAMA_BASE_URL
export OLLAMA_MODEL

echo "Starting server with:"
echo "  HOST=${HOST}"
echo "  PORT=${PORT}"
echo "  LLM_PROVIDER=${LLM_PROVIDER}"
if [[ "$LLM_PROVIDER" == "ollama" ]]; then
  echo "  OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"
  echo "  OLLAMA_MODEL=${OLLAMA_MODEL}"
fi

action="from idena_service.app import app; import os; app.run(host=os.environ['HOST'], port=int(os.environ['PORT']), debug=False)"
python -c "$action"
