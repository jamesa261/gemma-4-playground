#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from gemma4_vllm_profiles import (
    BASE_26B_TOKENIZER,
    BASE_31B_TOKENIZER,
    DEFAULT_GGUF_MODEL,
    LILA_MODEL,
    MODEL_SPECIFIC_DEFAULTS,
    MOE_26B_MODEL,
    MOE_26B_REDHAT_MODEL,
    REDHAT_MODEL,
    ensure_runtime_bin_on_path,
)


DEFAULT_MODEL = MOE_26B_MODEL
DEFAULT_PROMPT = "Explain NVFP4 in one sentence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 text-only inference in vLLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model repo to load")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer / processor repo to use. Defaults to the model unless a model-specific preset overrides it.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Base user prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of generated tokens per request")
    parser.add_argument("--warmup-tokens", type=int, default=8, help="Warmup generation length")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4], help="Batch sizes to benchmark")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "bfloat16", "float16"],
        help="Model dtype. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max model length. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["none", "modelopt"],
        help="Quantization mode for vLLM. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        default="slow",
        choices=["auto", "slow"],
        help="Tokenizer mode to use in vLLM",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="Allow remote code for model loading",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        choices=["auto", "fp8"],
        help="KV cache dtype for vLLM. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--moe-backend",
        default=None,
        choices=["auto", "marlin"],
        help="MoE backend override",
    )
    parser.add_argument(
        "--nvfp4-gemm-backend",
        default="auto",
        choices=["auto", "cutlass", "marlin", "flashinfer-cutlass", "flashinfer-trtllm", "flashinfer-cudnn", "fbgemm", "emulation"],
        help="NVFP4 GEMM backend override via VLLM_NVFP4_GEMM_BACKEND",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum concurrent sequences for the vLLM engine",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM may reserve",
    )
    args = parser.parse_args()
    args._specified_flags = collect_specified_flags(sys.argv[1:])
    return args


def collect_specified_flags(argv: list[str]) -> set[str]:
    specified: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        specified.add(token.split("=", 1)[0])
    return specified


def apply_model_specific_defaults(args: argparse.Namespace) -> list[str]:
    applied: list[str] = []
    defaults = MODEL_SPECIFIC_DEFAULTS.get(args.model, {})
    if not defaults:
        if args.tokenizer is None:
            args.tokenizer = args.model
        if args.quantization is None:
            args.quantization = "none"
        if args.dtype is None:
            args.dtype = "bfloat16"
        if args.trust_remote_code is None:
            args.trust_remote_code = False
        if args.kv_cache_dtype is None:
            args.kv_cache_dtype = "auto"
        if args.max_model_len is None:
            args.max_model_len = 4096
        return applied

    option_to_attr = {
        "--tokenizer": "tokenizer",
        "--quantization": "quantization",
        "--dtype": "dtype",
        "--trust-remote-code": "trust_remote_code",
        "--kv-cache-dtype": "kv_cache_dtype",
        "--nvfp4-gemm-backend": "nvfp4_gemm_backend",
        "--max-model-len": "max_model_len",
        "--gpu-memory-utilization": "gpu_memory_utilization",
        "--moe-backend": "moe_backend",
    }
    for option, attr in option_to_attr.items():
        if option not in args._specified_flags and attr in defaults:
            value = defaults[attr]
            setattr(args, attr, value)
            applied.append(f"{attr}={value!r}")

    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.quantization is None:
        args.quantization = "none"
    if args.dtype is None:
        args.dtype = "bfloat16"
    if args.trust_remote_code is None:
        args.trust_remote_code = False
    if args.kv_cache_dtype is None:
        args.kv_cache_dtype = "auto"
    if args.max_model_len is None:
        args.max_model_len = 4096
    if args.moe_backend is None:
        args.moe_backend = "auto"
    return applied


def main() -> int:
    args = parse_args()
    applied_defaults = apply_model_specific_defaults(args)
    ensure_runtime_bin_on_path()
    if args.nvfp4_gemm_backend != "auto":
        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = args.nvfp4_gemm_backend
    quantization = None if args.quantization == "none" else args.quantization
    tokenizer_id = args.tokenizer or args.model

    load_t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(tokenizer_id, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_id,
        dtype=args.dtype,
        quantization=quantization,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        kv_cache_dtype=args.kv_cache_dtype,
        moe_backend=args.moe_backend,
        max_num_seqs=args.max_num_seqs,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
    )
    print(f"load_time_s {time.perf_counter() - load_t0:.3f}")
    if applied_defaults:
        print("auto_defaults", ", ".join(applied_defaults))
    print("tokenizer", tokenizer_id)

    warmup = SamplingParams(temperature=0.0, max_tokens=args.warmup_tokens)
    measure = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    for batch_size in args.batch_sizes:
        prompts = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "user",
                    "content": f"{args.prompt} Example {i}.",
                }
            ]
            prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        _ = llm.generate(prompts, warmup)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, measure)
        dt = time.perf_counter() - t0
        total_new = sum(len(output.outputs[0].token_ids) for output in outputs)

        print(
            "batch",
            batch_size,
            "seconds",
            f"{dt:.3f}",
            "total_new",
            total_new,
            "tokps_total",
            f"{total_new / dt:.3f}",
            "tokps_per_req",
            f"{(total_new / batch_size) / dt:.3f}",
        )
        print("sample", outputs[0].outputs[0].text[:160].replace("\n", "\\n"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
