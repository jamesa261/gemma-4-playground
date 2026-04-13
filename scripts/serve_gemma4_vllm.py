#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from typing import Any

import torch

from gemma4_vllm_profiles import (
    MOE_BACKENDS,
    NVFP4_GEMM_BACKENDS,
    SERVER_PROFILES,
    ensure_runtime_bin_on_path,
    has_moe_loader_patch,
)


FLAG_MAP = {
    "model": "--model",
    "tokenizer": "--tokenizer",
    "quantization": "--quantization",
    "trust_remote_code": "--trust-remote-code",
    "kv_cache_dtype": "--kv-cache-dtype",
    "max_model_len": "--max-model-len",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "enable_prefix_caching": "--enable-prefix-caching",
    "language_model_only": "--language-model-only",
    "limit_mm_per_prompt": "--limit-mm-per-prompt",
    "reasoning_parser": "--reasoning-parser",
    "default_chat_template_kwargs": "--default-chat-template-kwargs",
    "tool_call_parser": "--tool-call-parser",
    "enable_auto_tool_choice": "--enable-auto-tool-choice",
    "moe_backend": "--moe-backend",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a preset vLLM OpenAI-compatible server for the Gemma 4 checkpoints validated in this repo."
    )
    parser.add_argument(
        "--profile",
        default="dense-31b",
        choices=sorted(SERVER_PROFILES),
        help="Model profile to launch.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for vLLM serve.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for vLLM serve.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override the profile's max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Override the profile's max concurrent sequences.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override the profile's max batched tokens.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override the profile's GPU memory reservation fraction.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        choices=["auto", "fp8"],
        help="Override the profile's KV cache dtype.",
    )
    parser.add_argument(
        "--moe-backend",
        default=None,
        choices=MOE_BACKENDS,
        help="Override the profile's MoE backend.",
    )
    parser.add_argument(
        "--nvfp4-gemm-backend",
        default="auto",
        choices=NVFP4_GEMM_BACKENDS,
        help="Override VLLM_NVFP4_GEMM_BACKEND. Leave at auto unless you are comparing kernels.",
    )
    parser.add_argument(
        "--disable-reasoning-parser",
        action="store_true",
        help="Do not set --reasoning-parser gemma4 on the server.",
    )
    parser.add_argument(
        "--enable-thinking-by-default",
        action="store_true",
        help=(
            "Set Gemma 4 thinking mode on for every request by passing "
            "--default-chat-template-kwargs '{\"enable_thinking\": true}'. "
            "This is useful for generic OpenAI-compatible UIs that do not expose chat_template_kwargs."
        ),
    )
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Add Gemma 4 tool-call parsing and auto tool choice flags to the server.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep vLLM at INFO logging instead of the quieter WARNING default used here.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and environment without launching the server.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append a raw token to the vLLM command. Repeat for additional tokens.",
    )
    return parser.parse_args()


def validate_runtime(args: argparse.Namespace) -> None:
    profile = SERVER_PROFILES[args.profile]
    cuda_version = torch.version.cuda or ""
    if profile.requires_cuda_major is not None:
        expected_prefix = f"{profile.requires_cuda_major}."
        if not cuda_version.startswith(expected_prefix):
            raise RuntimeError(
                f"{profile.key} expects a CUDA {profile.requires_cuda_major}.x vLLM environment for native Blackwell FP4. "
                f"Current torch CUDA version is {cuda_version or 'unknown'}. "
                f"Use {profile.recommended_python}."
            )

    if profile.requires_moe_loader_patch and not has_moe_loader_patch():
        raise RuntimeError(
            "The selected MoE profile needs the patched Gemma 4 loader. "
            "Apply patches/vllm-gemma4-modelopt-moe-loader.patch in the active vLLM environment first."
        )

    if shutil.which("vllm") is None:
        raise RuntimeError("Could not find the vllm CLI on PATH in the active environment.")

    if profile.requires_cuda_major is not None and shutil.which("ninja") is None:
        raise RuntimeError(
            "Could not find ninja on PATH in the active environment. "
            "The native FlashInfer NVFP4 path needs it for JIT kernel builds."
        )


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    profile = SERVER_PROFILES[args.profile]
    settings = {"model": profile.model, **profile.defaults}

    overrides = {
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "moe_backend": args.moe_backend,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    if args.disable_reasoning_parser:
        settings.pop("reasoning_parser", None)

    if args.enable_auto_tool_choice:
        settings["tool_call_parser"] = "gemma4"
        settings["enable_auto_tool_choice"] = True

    if args.enable_thinking_by_default:
        settings["default_chat_template_kwargs"] = json.dumps({"enable_thinking": True})

    return settings


def build_command(args: argparse.Namespace, settings: dict[str, Any]) -> list[str]:
    cmd = ["vllm", "serve", settings["model"], "--host", args.host, "--port", str(args.port)]
    for key, value in settings.items():
        if key == "model":
            continue
        if value is None:
            continue
        flag = FLAG_MAP[key]
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])

    cmd.extend(args.extra_arg)
    return cmd


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if not args.verbose:
        env["VLLM_LOGGING_LEVEL"] = "WARNING"
    if args.nvfp4_gemm_backend != "auto":
        env["VLLM_NVFP4_GEMM_BACKEND"] = args.nvfp4_gemm_backend
    return env


def main() -> int:
    args = parse_args()
    ensure_runtime_bin_on_path()
    validate_runtime(args)

    profile = SERVER_PROFILES[args.profile]
    settings = resolve_settings(args)
    cmd = build_command(args, settings)
    env = build_env(args)

    print(f"profile: {profile.key}")
    print(f"description: {profile.description}")
    print(f"recommended_python: {profile.recommended_python}")
    print(f"runtime_python: {sys.executable}")
    print(f"runtime_cuda: {torch.version.cuda}")
    if args.nvfp4_gemm_backend != "auto":
        print(f"VLLM_NVFP4_GEMM_BACKEND={args.nvfp4_gemm_backend}")
    print("command:")
    print("  " + shlex.join(cmd))

    if args.dry_run:
        return 0

    os.execvpe(cmd[0], cmd, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
