#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import argparse
import contextlib
import os
import sys
import time
import uuid
from typing import Any

from transformers import AutoProcessor
from gemma4_vllm_profiles import (
    BASE_26B_TOKENIZER,
    BASE_31B_TOKENIZER,
    DEFAULT_GGUF_MODEL,
    LILA_MODEL,
    MODEL_SPECIFIC_DEFAULTS,
    MOE_26B_MODEL,
    MOE_BACKENDS,
    MOE_26B_REDHAT_MODEL,
    NVFP4_GEMM_BACKENDS,
    REDHAT_MODEL,
    TURBOQUANT_KV_CACHE_DTYPES,
    ensure_runtime_bin_on_path,
)
from gemma4_vllm_turboquant import apply_vllm_turboquant_workarounds, resolve_selective_turboquant_settings

DEFAULT_MODEL = MOE_26B_MODEL
DEFAULT_PROMPT = "Explain NVFP4 in one sentence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Gemma 4 through vLLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model repo to load")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer / processor repo to use. Defaults to the model unless a model-specific preset overrides it.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional one-shot prompt. If omitted, the script starts an interactive chat session.",
    )
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["none", "modelopt"],
        help="Quantization mode for vLLM. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "bfloat16", "float16"],
        help="Model dtype. If omitted, model-specific defaults may apply.",
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
        choices=["auto", "fp8", *TURBOQUANT_KV_CACHE_DTYPES],
        help="KV cache dtype for vLLM. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--moe-backend",
        default=None,
        choices=MOE_BACKENDS,
        help="MoE backend override",
    )
    parser.add_argument(
        "--nvfp4-gemm-backend",
        default="auto",
        choices=NVFP4_GEMM_BACKENDS,
        help="NVFP4 GEMM backend override via VLLM_NVFP4_GEMM_BACKEND",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max model length. If omitted, model-specific defaults may apply.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Number of tokens to generate. Total prompt plus generation must still fit within max_model_len.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Gemma 4 model cards recommend 1.0 for general use.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling value. Gemma 4 model cards recommend 0.95.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Top-k sampling value. Gemma 4 model cards recommend 64.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Gemma 4 thinking mode through the processor chat template.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live token streaming and print the full completion at the end.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show vLLM and Hugging Face load logs instead of the quiet default.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even if --prompt is provided",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum concurrent sequences for the vLLM engine",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum total tokens scheduled in one vLLM iteration.",
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
        "--max-num-batched-tokens": "max_num_batched_tokens",
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


def apply_turboquant_defaults(args: argparse.Namespace) -> list[str]:
    updates = resolve_selective_turboquant_settings(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_dtype_skip_layers=getattr(args, "kv_cache_dtype_skip_layers", None),
        disable_hybrid_kv_cache_manager=getattr(args, "disable_hybrid_kv_cache_manager", None),
    )
    applied: list[str] = []
    for key, value in updates.items():
        setattr(args, key, value)
        if key == "kv_cache_dtype_skip_layers":
            applied.append(f"{key}={len(value)} layers")
        else:
            applied.append(f"{key}={value!r}")
    if args.kv_cache_dtype in TURBOQUANT_KV_CACHE_DTYPES:
        apply_vllm_turboquant_workarounds()
    return applied


def configure_environment(args: argparse.Namespace) -> None:
    ensure_runtime_bin_on_path()

    if args.verbose:
        os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
        os.environ["FLASHINFER_LOGGING_LEVEL"] = "INFO"
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        os.environ.pop("PYTHONWARNINGS", None)
    else:
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["FLASHINFER_LOGGING_LEVEL"] = "ERROR"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    if args.nvfp4_gemm_backend != "auto":
        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = args.nvfp4_gemm_backend


def get_runtime_cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return torch.version.cuda


def print_runtime_notes(args: argparse.Namespace) -> None:
    runtime_cuda = get_runtime_cuda_version()
    if args.model in {REDHAT_MODEL, LILA_MODEL, MOE_26B_MODEL, MOE_26B_REDHAT_MODEL} and runtime_cuda is not None:
        major = int(runtime_cuda.split(".", 1)[0])
        if major < 13:
            print(
                "note: this Python env is using CUDA "
                f"{runtime_cuda}. "
                f"{args.model} is best run from a CUDA 13.0 vLLM env on Blackwell to pick up the native FP4 path.",
            )
    if args.model.startswith("unsloth/gemma-4-31B-it-GGUF:"):
        print("note: vLLM GGUF support is still experimental and under-optimized; use the base HF tokenizer.")


def build_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": text}


def build_base_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    system_prompt = args.system_prompt
    if not system_prompt:
        return []
    return [build_message("system", system_prompt)]


def build_sampling_params(
    args: argparse.Namespace, *, stream: bool, max_tokens: int | None = None
) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_new_tokens if max_tokens is None else max_tokens,
        "output_kind": RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
    }
    if args.enable_thinking:
        kwargs["skip_special_tokens"] = False
        kwargs["spaces_between_special_tokens"] = False
    return SamplingParams(**kwargs)


def build_prompt(processor: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return processor.apply_chat_template(
        messages,
        **kwargs,
    )


def build_engine_input(processor: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> tuple[Any, int]:
    prompt = build_prompt(processor, messages, enable_thinking)
    prompt_token_ids = processor.tokenizer(
        prompt,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    return {
        "type": "token",
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
    }, len(prompt_token_ids)


@contextlib.contextmanager
def suppress_fd_output(enabled: bool):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(sys.stdout.fileno())
    stderr_fd = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, sys.stdout.fileno())
            os.dup2(stderr_fd, sys.stderr.fileno())
            os.close(stdout_fd)
            os.close(stderr_fd)


def print_generation_summary(summary: dict[str, Any]) -> None:
    print(f"prompt_tokens: {summary['prompt_tokens']}")
    print(f"generated_tokens: {summary['generated_tokens']}")
    print(f"generate_time_s: {summary['generate_time_s']:.2f}")
    print(f"tokens_per_second: {summary['tokens_per_second']:.2f}")
    if summary["first_token_latency_s"] is not None:
        print(f"first_token_latency_s: {summary['first_token_latency_s']:.2f}")


def parse_reasoning_output(processor: Any, raw_text: str) -> tuple[str | None, str]:
    parsed = processor.parse_response(raw_text)
    thinking = parsed.get("thinking")
    answer = (parsed.get("content") or "").strip()
    return thinking, answer


async def run_completion(
    llm: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str | None, str, dict[str, Any]]:
    from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser

    stream = not args.no_stream
    engine_input, prompt_tokens = build_engine_input(processor, messages, args.enable_thinking)
    request_id = uuid.uuid4().hex
    generate_start = time.perf_counter()
    first_token_latency_s: float | None = None
    raw_pieces: list[str] = []
    generated_tokens = 0
    emitted_reasoning = False
    emitted_answer = False
    reasoning_parser = Gemma4ReasoningParser(processor.tokenizer) if args.enable_thinking else None
    previous_text = ""
    current_text = ""
    previous_token_ids: list[int] = []
    current_token_ids: list[int] = []

    async for output in llm.generate(
        engine_input,
        build_sampling_params(args, stream=stream),
        request_id,
    ):
        if not output.outputs:
            continue
        piece = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        if not piece and not token_ids:
            continue
        if first_token_latency_s is None:
            first_token_latency_s = time.perf_counter() - generate_start
        raw_pieces.append(piece)
        generated_tokens += len(token_ids)
        current_text += piece
        current_token_ids.extend(token_ids)
        if stream:
            if not args.enable_thinking:
                if piece:
                    print(piece, end="", flush=True)
            else:
                delta = reasoning_parser.extract_reasoning_streaming(
                    previous_text,
                    current_text,
                    piece,
                    previous_token_ids,
                    current_token_ids,
                    token_ids,
                )
                if delta is not None and delta.reasoning:
                    if not emitted_reasoning:
                        print("thinking> ", end="", flush=True)
                        emitted_reasoning = True
                    print(delta.reasoning, end="", flush=True)
                if delta is not None and delta.content:
                    if emitted_reasoning and not emitted_answer:
                        print()
                    if not emitted_answer:
                        print("assistant> ", end="", flush=True)
                        emitted_answer = True
                    print(delta.content, end="", flush=True)
                previous_text = current_text
                previous_token_ids = list(current_token_ids)

    raw_completion = "".join(raw_pieces).strip()
    thinking, completion = parse_reasoning_output(processor, raw_completion)

    if stream:
        if args.enable_thinking and thinking and not emitted_reasoning:
            print("thinking> ", end="", flush=True)
            print(thinking, end="", flush=True)
            emitted_reasoning = True
        if args.enable_thinking and not emitted_answer and completion:
            if emitted_reasoning:
                print()
            print("assistant> ", end="", flush=True)
            print(completion, end="", flush=True)
        print()

    generate_seconds = time.perf_counter() - generate_start
    tokens_per_second = generated_tokens / generate_seconds if generate_seconds > 0 else 0.0

    return thinking, completion, {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generate_time_s": generate_seconds,
        "tokens_per_second": tokens_per_second,
        "first_token_latency_s": first_token_latency_s,
    }


async def interactive_chat_loop(
    llm: Any,
    processor: Any,
    args: argparse.Namespace,
) -> None:
    base_messages = build_base_messages(args)
    messages = list(base_messages)

    session_turns = 0
    session_generated_tokens = 0
    session_generate_time = 0.0

    print("Interactive mode. Commands: /exit, /quit, /reset, /stats")
    while True:
        try:
            user_text = (await asyncio.to_thread(input, "\n>>> ")).strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/exit", "/quit"}:
            break
        if lowered == "/reset":
            messages = list(base_messages)
            print("history reset")
            continue
        if lowered == "/stats":
            avg_toks = session_generated_tokens / session_generate_time if session_generate_time > 0 else 0.0
            print(f"session_turns: {session_turns}")
            print(f"session_generated_tokens: {session_generated_tokens}")
            print(f"session_generate_time_s: {session_generate_time:.2f}")
            print(f"session_tokens_per_second: {avg_toks:.2f}")
            continue

        messages.append(build_message("user", user_text))
        if not args.no_stream and not args.enable_thinking:
            print("assistant> ", end="", flush=True)
        thinking, completion, summary = await run_completion(llm, processor, messages, args)
        if args.no_stream:
            if thinking:
                print(f"thinking> {thinking}")
            print(f"assistant> {completion}")
        messages.append(build_message("assistant", completion))

        session_turns += 1
        session_generated_tokens += summary["generated_tokens"]
        session_generate_time += summary["generate_time_s"]

        print(f"turn: {session_turns}")
        print_generation_summary(summary)

    avg_toks = session_generated_tokens / session_generate_time if session_generate_time > 0 else 0.0
    print("session_end")
    print(f"session_turns: {session_turns}")
    print(f"session_generated_tokens: {session_generated_tokens}")
    print(f"session_generate_time_s: {session_generate_time:.2f}")
    print(f"session_tokens_per_second: {avg_toks:.2f}")


def create_llm(args: argparse.Namespace) -> tuple[Any, Any, float]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    quantization = None if args.quantization == "none" else args.quantization
    tokenizer_id = args.tokenizer or args.model

    load_t0 = time.perf_counter()
    with suppress_fd_output(not args.verbose):
        processor = AutoProcessor.from_pretrained(tokenizer_id, trust_remote_code=args.trust_remote_code)
        engine_args = AsyncEngineArgs(
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
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_cache_dtype_skip_layers=getattr(args, "kv_cache_dtype_skip_layers", None),
            disable_hybrid_kv_cache_manager=getattr(args, "disable_hybrid_kv_cache_manager", None),
            disable_log_stats=True,
            enable_log_requests=False,
            use_tqdm_on_load=args.verbose,
            stream_interval=1,
            language_model_only=True,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        )
        llm = AsyncLLM.from_engine_args(engine_args)
    return llm, processor, time.perf_counter() - load_t0


def print_load_summary(args: argparse.Namespace, load_seconds: float, applied_defaults: list[str]) -> None:
    print(f"model: {args.model}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"quantization: {args.quantization}")
    print(f"dtype: {args.dtype}")
    print(f"tokenizer_mode: {args.tokenizer_mode}")
    print(f"trust_remote_code: {args.trust_remote_code}")
    print(f"kv_cache_dtype: {args.kv_cache_dtype}")
    skip_layers = getattr(args, "kv_cache_dtype_skip_layers", None)
    if skip_layers:
        print(f"kv_cache_dtype_skip_layers: {len(skip_layers)} layers")
    disable_hybrid = getattr(args, "disable_hybrid_kv_cache_manager", None)
    if disable_hybrid is not None:
        print(f"disable_hybrid_kv_cache_manager: {disable_hybrid}")
    print(f"moe_backend: {args.moe_backend}")
    print(f"nvfp4_gemm_backend: {os.environ.get('VLLM_NVFP4_GEMM_BACKEND', 'auto')}")
    print(f"max_model_len: {args.max_model_len}")
    print(f"max_num_seqs: {args.max_num_seqs}")
    if args.max_num_batched_tokens is not None:
        print(f"max_num_batched_tokens: {args.max_num_batched_tokens}")
    print(f"gpu_memory_utilization: {args.gpu_memory_utilization:.2f}")
    runtime_cuda = get_runtime_cuda_version()
    if runtime_cuda is not None:
        print(f"runtime_cuda: {runtime_cuda}")
    if applied_defaults:
        print(f"auto_defaults: {', '.join(applied_defaults)}")
    print(f"load_time_s: {load_seconds:.2f}")


async def run_app(args: argparse.Namespace, applied_defaults: list[str]) -> int:
    if not args.verbose:
        print("loading model...", flush=True)

    llm, processor, load_seconds = create_llm(args)
    try:
        print_load_summary(args, load_seconds, applied_defaults)
        print_runtime_notes(args)

        if args.interactive or not args.prompt:
            await interactive_chat_loop(llm, processor, args)
            return 0

        messages = build_base_messages(args)
        messages.append(build_message("user", args.prompt))
        if not args.no_stream and not args.enable_thinking:
            print("assistant> ", end="", flush=True)
        thinking, completion, summary = await run_completion(llm, processor, messages, args)
        if args.no_stream:
            if thinking:
                print(f"thinking> {thinking}")
            print(f"assistant> {completion}")
        print_generation_summary(summary)
        return 0
    finally:
        llm.shutdown()


def main() -> int:
    args = parse_args()
    applied_defaults = apply_model_specific_defaults(args)
    configure_environment(args)
    applied_defaults.extend(apply_turboquant_defaults(args))
    return asyncio.run(run_app(args, applied_defaults))


if __name__ == "__main__":
    raise SystemExit(main())
