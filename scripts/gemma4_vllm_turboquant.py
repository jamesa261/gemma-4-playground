#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from transformers import AutoConfig

from gemma4_vllm_profiles import TURBOQUANT_KV_CACHE_DTYPES


def is_turboquant_kv_cache_dtype(cache_dtype: str | None) -> bool:
    return cache_dtype in TURBOQUANT_KV_CACHE_DTYPES


def infer_gemma4_sliding_window_skip_layers(
    model: str,
    *,
    trust_remote_code: bool,
) -> list[str]:
    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        return []
    return [str(i) for i, layer_type in enumerate(layer_types) if layer_type == "sliding_attention"]


def resolve_selective_turboquant_settings(
    *,
    model: str,
    trust_remote_code: bool,
    kv_cache_dtype: str | None,
    kv_cache_dtype_skip_layers: list[str] | None,
    disable_hybrid_kv_cache_manager: bool | None,
) -> dict[str, Any]:
    if not is_turboquant_kv_cache_dtype(kv_cache_dtype):
        return {}

    updates: dict[str, Any] = {}
    if disable_hybrid_kv_cache_manager is None:
        updates["disable_hybrid_kv_cache_manager"] = True
    if not kv_cache_dtype_skip_layers:
        skip_layers = infer_gemma4_sliding_window_skip_layers(
            model,
            trust_remote_code=trust_remote_code,
        )
        if skip_layers:
            updates["kv_cache_dtype_skip_layers"] = skip_layers
    return updates


_PATCH_APPLIED = False


def apply_vllm_turboquant_workarounds() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import vllm.model_executor.models.config as model_config_mod
    import vllm.v1.attention.backends.turboquant_attn as turboquant_attn_mod

    original_verify = model_config_mod.Gemma4Config.verify_and_update_config

    def patched_verify_and_update_config(vllm_config: Any) -> None:
        cache_config = getattr(vllm_config, "cache_config", None)
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if is_turboquant_kv_cache_dtype(cache_dtype):
            return
        return original_verify(vllm_config)

    model_config_mod.Gemma4Config.verify_and_update_config = staticmethod(
        patched_verify_and_update_config
    )
    # TurboQuant currently routes prefill through FA2, which fails on Gemma 4's
    # 512-dim global-attention layers. Let the backend fall back to SDPA instead.
    turboquant_attn_mod._HAS_FLASH_ATTN = False
    _PATCH_APPLIED = True
