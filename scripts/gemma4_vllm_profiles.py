#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MOE_26B_MODEL = "bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4"
LILA_MODEL = "LilaRest/gemma-4-31B-it-NVFP4-turbo"
REDHAT_MODEL = "RedHatAI/gemma-4-31B-it-NVFP4"
DEFAULT_GGUF_MODEL = "unsloth/gemma-4-31B-it-GGUF:Q4_K_M"
BASE_31B_TOKENIZER = "google/gemma-4-31B-it"

NVFP4_GEMM_BACKENDS = [
    "auto",
    "cutlass",
    "marlin",
    "flashinfer-cutlass",
    "flashinfer-trtllm",
    "flashinfer-cudnn",
    "fbgemm",
    "emulation",
]
MOE_BACKENDS = ["auto", "marlin"]

MODEL_SPECIFIC_DEFAULTS = {
    MOE_26B_MODEL: {
        "quantization": "modelopt",
        "dtype": "auto",
        "trust_remote_code": True,
        "kv_cache_dtype": "auto",
        "max_model_len": 50000,
        "nvfp4_gemm_backend": "marlin",
    },
    REDHAT_MODEL: {
        "quantization": "none",
        "dtype": "auto",
        "trust_remote_code": False,
        "kv_cache_dtype": "auto",
        "max_model_len": 32768,
    },
    LILA_MODEL: {
        "quantization": "modelopt",
        "dtype": "auto",
        "trust_remote_code": True,
        "kv_cache_dtype": "fp8",
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.95,
    },
    DEFAULT_GGUF_MODEL: {
        "tokenizer": BASE_31B_TOKENIZER,
        "quantization": "none",
        "dtype": "auto",
        "trust_remote_code": False,
        "kv_cache_dtype": "auto",
        "max_model_len": 16384,
    },
}


@dataclass(frozen=True)
class ServerProfile:
    key: str
    model: str
    description: str
    recommended_python: str
    defaults: dict[str, Any] = field(default_factory=dict)
    requires_cuda_major: int | None = None
    requires_moe_loader_patch: bool = False


SERVER_PROFILES = {
    "dense-31b": ServerProfile(
        key="dense-31b",
        model=LILA_MODEL,
        description="Default 31B dense profile for this RTX 5090: LilaRest turbo NVFP4 on CUDA 13.",
        recommended_python=".venv-vllm-cu130/bin/python",
        requires_cuda_major=13,
        defaults={
            "quantization": "modelopt",
            "trust_remote_code": True,
            "kv_cache_dtype": "fp8",
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.95,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
            "reasoning_parser": "gemma4",
        },
    ),
    "dense-31b-redhat": ServerProfile(
        key="dense-31b-redhat",
        model=REDHAT_MODEL,
        description="Alternative 31B dense profile: Red Hat NVFP4 with a conservative context budget on this RTX 5090.",
        recommended_python=".venv-vllm-cu130/bin/python",
        requires_cuda_major=13,
        defaults={
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.95,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
            "reasoning_parser": "gemma4",
        },
    ),
    "moe-26b": ServerProfile(
        key="moe-26b",
        model=MOE_26B_MODEL,
        description="Patched 26B MoE NVFP4 profile using the community checkpoint and Marlin MoE kernels.",
        recommended_python=".venv-vllm/bin/python",
        requires_moe_loader_patch=True,
        defaults={
            "quantization": "modelopt",
            "trust_remote_code": True,
            "max_model_len": 50000,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 4096,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
            "moe_backend": "marlin",
            "reasoning_parser": "gemma4",
        },
    ),
}


def ensure_runtime_bin_on_path() -> None:
    runtime_bin_dir = os.path.dirname(sys.executable)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if runtime_bin_dir and runtime_bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([runtime_bin_dir, *path_entries]) if path_entries else runtime_bin_dir


def detect_installed_vllm_gemma4_source() -> Path | None:
    spec = importlib.util.find_spec("vllm.model_executor.models.gemma4")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def has_moe_loader_patch() -> bool:
    source_path = detect_installed_vllm_gemma4_source()
    if source_path is None or not source_path.exists():
        return False

    source = source_path.read_text()
    return "Gemma 4 MoE NVFP4 checkpoints include expert weights plus" in source and "weight_scale_2" in source
