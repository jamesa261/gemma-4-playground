#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DATA_DIR="${DATA_DIR:-$ROOT_DIR/.open-webui-data}"
export ENABLE_PERSISTENT_CONFIG="${ENABLE_PERSISTENT_CONFIG:-False}"
export ENABLE_OLLAMA_API="${ENABLE_OLLAMA_API:-False}"
export OPENAI_API_BASE_URLS="${OPENAI_API_BASE_URLS:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEYS="${OPENAI_API_KEYS:-EMPTY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST="${AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST:-30}"

exec uvx --python 3.11 open-webui@latest serve "$@"
