#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/.venv-vllm-cu130}"
PYTHON_BIN="$ENV_DIR/bin/python"
VLLM_WHEEL_URL="https://wheels.vllm.ai/739e5945dc4b3ba30d84bdf6e637657abd4136b8/vllm-0.19.1rc1.dev242%2Bg739e5945d.cu130-cp38-abi3-manylinux_2_35_x86_64.whl"

uv venv "$ENV_DIR"

# Install the cu130 vLLM wheel with its normal dependency set first. The
# current vLLM wheel metadata still caps transformers at <5, so we layer the
# newer Gemma-4-capable Transformers stack on top afterwards.
uv pip install --python "$PYTHON_BIN" "$VLLM_WHEEL_URL" nvidia-modelopt ninja
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
