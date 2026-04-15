#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/.venv}"
PYTHON_BIN="$ENV_DIR/bin/python"

uv venv "$ENV_DIR"

# Install the base runtime first so vLLM can resolve its declared dependency
# set, then override the Hugging Face stack to the Gemma-4-capable versions.
uv pip install --python "$PYTHON_BIN" \
  "https://wheels.vllm.ai/21e5a9f48e773e36e916bc8d10c4ab4aed3887a7/vllm-0.19.1rc1.dev311%2Bg21e5a9f48.cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
  nvidia-modelopt \
  ninja
uv pip install --python "$PYTHON_BIN" --no-deps transformers==5.5.4 huggingface_hub==1.9.2

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as m
import torch

print("vllm", m.version("vllm"))
print("transformers", m.version("transformers"))
print("huggingface_hub", m.version("huggingface_hub"))
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
PY
