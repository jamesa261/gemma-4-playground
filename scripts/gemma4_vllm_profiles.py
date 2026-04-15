#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any


MOE_26B_MODEL = "cklaus/gemma-4-26B-A4B-it-NVFP4"
MOE_26B_REDHAT_MODEL = "RedHatAI/gemma-4-26B-A4B-it-NVFP4"
LILA_MODEL = "LilaRest/gemma-4-31B-it-NVFP4-turbo"
REDHAT_MODEL = "RedHatAI/gemma-4-31B-it-NVFP4"
DEFAULT_GGUF_MODEL = "unsloth/gemma-4-31B-it-GGUF:Q4_K_M"
BASE_26B_TOKENIZER = "google/gemma-4-26B-A4B-it"
BASE_31B_TOKENIZER = "google/gemma-4-31B-it"

TURBOQUANT_KV_CACHE_DTYPES = [
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]
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

MOE_26B_TURBOQUANT_SKIP_LAYERS = [str(i) for i in range(30) if i % 6 != 5]
DENSE_31B_TURBOQUANT_SKIP_LAYERS = [str(i) for i in range(60) if i % 6 != 5]

MODEL_SPECIFIC_DEFAULTS = {
    MOE_26B_MODEL: {
        "tokenizer": BASE_26B_TOKENIZER,
        "quantization": "modelopt",
        "dtype": "auto",
        "trust_remote_code": False,
        "kv_cache_dtype": "turboquant_k8v4",
        "kv_cache_dtype_skip_layers": MOE_26B_TURBOQUANT_SKIP_LAYERS,
        "disable_hybrid_kv_cache_manager": True,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "moe_backend": "auto",
        "max_num_batched_tokens": 8192,
    },
    MOE_26B_REDHAT_MODEL: {
        "tokenizer": BASE_26B_TOKENIZER,
        "quantization": "none",
        "dtype": "auto",
        "trust_remote_code": False,
        "kv_cache_dtype": "auto",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "moe_backend": "auto",
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
        "kv_cache_dtype": "turboquant_k8v4",
        "kv_cache_dtype_skip_layers": DENSE_31B_TURBOQUANT_SKIP_LAYERS,
        "disable_hybrid_kv_cache_manager": True,
        "max_model_len": 9984,
        "gpu_memory_utilization": 0.95,
        "max_num_batched_tokens": 8192,
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


SERVER_PROFILES = {
    "dense-31b": ServerProfile(
        key="dense-31b",
        model=LILA_MODEL,
        description="Default 31B dense profile for this RTX 5090: LilaRest turbo NVFP4 with selective TurboQuant KV on CUDA 13.",
        recommended_python=".venv/bin/python",
        requires_cuda_major=13,
        defaults={
            "quantization": "modelopt",
            "trust_remote_code": True,
            "kv_cache_dtype": "turboquant_k8v4",
            "kv_cache_dtype_skip_layers": DENSE_31B_TURBOQUANT_SKIP_LAYERS,
            "disable_hybrid_kv_cache_manager": True,
            "max_model_len": 9984,
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
        recommended_python=".venv/bin/python",
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
        description="Default 26B MoE profile for this RTX 5090: selective NVFP4 checkpoint plus selective TurboQuant KV on CUDA 13.",
        recommended_python=".venv/bin/python",
        requires_cuda_major=13,
        defaults={
            "quantization": "modelopt",
            "trust_remote_code": False,
            "kv_cache_dtype": "turboquant_k8v4",
            "kv_cache_dtype_skip_layers": MOE_26B_TURBOQUANT_SKIP_LAYERS,
            "disable_hybrid_kv_cache_manager": True,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
            "moe_backend": "auto",
            "reasoning_parser": "gemma4",
        },
    ),
    "moe-26b-redhat": ServerProfile(
        key="moe-26b-redhat",
        model=MOE_26B_REDHAT_MODEL,
        description="Upstream 26B MoE NVFP4 profile on CUDA 13 for comparison against the 5090-tuned selective quantization.",
        recommended_python=".venv/bin/python",
        requires_cuda_major=13,
        defaults={
            "kv_cache_dtype": "auto",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "language_model_only": True,
            "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
            "moe_backend": "auto",
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
