# Gemma 4 Playground

This repo is set up to run Gemma 4 through vLLM on an RTX 5090, using a single CUDA 13 environment and two TurboQuant-first profiles that are currently the most useful here:

- `dense-31b`: `LilaRest/gemma-4-31B-it-NVFP4-turbo`
- `moe-26b`: `cklaus/gemma-4-26B-A4B-it-NVFP4`
- `dense-31b-redhat`: `RedHatAI/gemma-4-31B-it-NVFP4`
- `moe-26b-redhat`: `RedHatAI/gemma-4-26B-A4B-it-NVFP4`

The main entrypoint is [`scripts/serve_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/serve_gemma4_vllm.py). It resolves the right model, tokenizer, memory knobs, reasoning parser, and TurboQuant workarounds for each profile, then launches `vllm serve`.

## Layout

- [`scripts/serve_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/serve_gemma4_vllm.py): primary server launcher
- [`scripts/gemma4_vllm_profiles.py`](/home/jamesa261/gemma-4-playground/scripts/gemma4_vllm_profiles.py): shared presets and runtime checks
- [`scripts/chat_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/chat_gemma4_vllm.py): local REPL for qualitative testing
- [`scripts/benchmark_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/benchmark_gemma4_vllm.py): in-process throughput benchmark
- [`scripts/run_open_webui.sh`](/home/jamesa261/gemma-4-playground/scripts/run_open_webui.sh): launch Open WebUI against the local vLLM OpenAI endpoint
- [`scripts/test_structured_output.py`](/home/jamesa261/gemma-4-playground/scripts/test_structured_output.py): minimal structured-output test against the server API
- [`scripts/setup_vllm.sh`](/home/jamesa261/gemma-4-playground/scripts/setup_vllm.sh): setup helper for the shared CUDA 13 vLLM environment
- [`scripts/gemma4_vllm_turboquant.py`](/home/jamesa261/gemma-4-playground/scripts/gemma4_vllm_turboquant.py): selective-TurboQuant helpers and Gemma 4 runtime patch

## Environment

There is now one vLLM runtime in this repo:

- `.venv`
  Shared by the 31B dense profiles and the 26B MoE profiles. The launcher validates that it is actually running against CUDA 13 before it will start any Blackwell NVFP4 profile.

The manifest for that environment is:

- [`requirements.in`](/home/jamesa261/gemma-4-playground/requirements.in)
- [`requirements.txt`](/home/jamesa261/gemma-4-playground/requirements.txt)

Bootstrap or refresh it with:

```bash
bash scripts/setup_vllm.sh
```

That helper intentionally installs in two phases: first the vLLM wheel with its
declared dependency set, then the Gemma-4-capable `transformers==5.5.4` and
`huggingface_hub==1.9.2` override. The current vLLM wheel still declares
`transformers<5`, so a plain `pip_compile` / `uv pip sync requirements.txt`
workflow is not reliable for this environment yet.

The old patched cu129 MoE path has been removed. The remaining patching in this repo is now narrowly scoped to Gemma 4 plus TurboQuant on current upstream vLLM.

## TurboQuant Notes

Current upstream TurboQuant is not drop-in for Gemma 4's mixed sliding-window and global-attention layout. This repo applies the smallest workaround that was stable on this 5090:

- only the full-attention layers use TurboQuant KV
- sliding-window layers stay on native KV
- `disable_hybrid_kv_cache_manager=True` is enabled for the TurboQuant profiles
- the TurboQuant backend is forced off FlashAttention so Gemma 4's `global_head_dim=512` prefills fall back to SDPA instead of hitting the FA2 `head_dim <= 256` limit

That keeps decode working on both the dense 31B turbo checkpoint and the preferred 26B MoE checkpoint, while still letting the native CUDA 13 / Blackwell NVFP4 weight path do the heavy lifting.

Observed MoE fidelity with `turboquant_k8v4` on `cklaus/gemma-4-26B-A4B-it-NVFP4`:

- `k8v4` means FP8 keys plus 4-bit uniformly quantized values, so the loss is expected to show up mostly in `V`, not `K`
- measured on real vLLM captures from the four compressed full-attention layers, `K` cosine similarity was effectively near-lossless: mean about `0.99965` to `0.99968`
- the same probe showed `V` cosine similarity in the modest-loss range: mean about `0.955` on a short 29-token chat prompt and about `0.974` on a longer 3396-token retrieval prompt
- practical spot checks matched that picture: long retrieval outputs stayed faithful to the uncompressed-KV baseline, while short prompts could still drift a bit in formatting or numeric representation
- this is a KV-cache comparison on the same NVFP4-weight checkpoint, not a full-BF16-weights comparison

## Quick Start

Default dense server:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py
```

Default MoE server:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py --profile moe-26b
```

Upstream MoE baseline:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py --profile moe-26b-redhat
```

All of these start an OpenAI-compatible vLLM server on `http://127.0.0.1:8000/v1`.

To inspect the resolved command without launching the server:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py --profile moe-26b --dry-run
```

If you want generic OpenAI-compatible clients to get Gemma 4 thinking mode without having to send `chat_template_kwargs` manually:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py \
  --profile moe-26b \
  --enable-thinking-by-default
```

If you want Gemma 4 tool-call parsing enabled on the server as well:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py \
  --profile moe-26b \
  --enable-auto-tool-choice
```

## Default Profiles

`dense-31b`

- Model: `LilaRest/gemma-4-31B-it-NVFP4-turbo`
- Runtime: CUDA 13 vLLM env
- Quantization: `modelopt`
- KV cache: selective `turboquant_k8v4`
- TurboQuant scope: only full-attention layers are compressed; sliding-window layers are skipped
- Max model length: `9984`
- GPU memory utilization: `0.95`
- Max sequences: `8`
- Max batched tokens: `8192`
- Reasoning parser: enabled
- Text-only optimization: `--language-model-only` plus `--limit-mm-per-prompt image=0,audio=0,video=0`

This profile is bounded by KV memory, not model weights. On this card and wheel, vLLM estimated a practical TurboQuant ceiling of about `10064` tokens, so the saved default stays slightly below that at `9984`.

`dense-31b-redhat`

- Model: `RedHatAI/gemma-4-31B-it-NVFP4`
- Runtime: CUDA 13 vLLM env
- Quantization: model-native compressed tensors
- KV cache: `auto`
- Max model length: `4096`
- GPU memory utilization: `0.95`
- Max sequences: `8`
- Max batched tokens: `4096`
- Reasoning parser: enabled
- Text-only optimization: enabled

`moe-26b`

- Model: `cklaus/gemma-4-26B-A4B-it-NVFP4`
- Tokenizer: `google/gemma-4-26B-A4B-it`
- Runtime: CUDA 13 vLLM env
- Quantization: `modelopt`
- KV cache: selective `turboquant_k8v4`
- TurboQuant scope: only full-attention layers are compressed; sliding-window layers are skipped
- Max model length: `32768`
- GPU memory utilization: `0.90`
- Max sequences: `16`
- Max batched tokens: `8192`
- MoE backend: `auto`
- Reasoning parser: enabled
- Text-only optimization: enabled

This is the current default MoE profile because it cleanly picks up the native Blackwell NVFP4 path on this 5090 and still leaves enough TurboQuant-compressed KV room for a much larger context budget than the dense 31B turbo path.

`moe-26b-redhat`

- Model: `RedHatAI/gemma-4-26B-A4B-it-NVFP4`
- Tokenizer: `google/gemma-4-26B-A4B-it`
- Runtime: CUDA 13 vLLM env
- Quantization: model-native compressed tensors
- KV cache: `auto`
- Max model length: `4096`
- GPU memory utilization: `0.90`
- Max sequences: `16`
- Max batched tokens: `8192`
- MoE backend: `auto`
- Reasoning parser: enabled
- Text-only optimization: enabled

This is the upstream baseline profile for comparison against the more 5090-tuned `cklaus` selective NVFP4 checkpoint.

## Useful Overrides

The launcher exposes the small set of knobs that matter for downstream testing:

- `--max-model-len`
- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--gpu-memory-utilization`
- `--kv-cache-dtype`
- `--moe-backend`
- `--nvfp4-gemm-backend`
- `--host`
- `--port`
- `--extra-arg`

Examples:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py \
  --profile moe-26b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92
```

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py \
  --profile dense-31b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

## Chat And Benchmarking

Qualitative testing is handled by [`scripts/chat_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/chat_gemma4_vllm.py).

MoE chat:

```bash
.venv/bin/python scripts/chat_gemma4_vllm.py \
  --model cklaus/gemma-4-26B-A4B-it-NVFP4
```

Dense chat:

```bash
.venv/bin/python scripts/chat_gemma4_vllm.py \
  --model LilaRest/gemma-4-31B-it-NVFP4-turbo
```

Throughput benchmarking is handled by [`scripts/benchmark_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/benchmark_gemma4_vllm.py).

MoE benchmark:

```bash
.venv/bin/python scripts/benchmark_gemma4_vllm.py \
  --model cklaus/gemma-4-26B-A4B-it-NVFP4 \
  --tokenizer google/gemma-4-26B-A4B-it \
  --batch-sizes 1 4 8 16 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-new-tokens 64
```

Upstream MoE baseline benchmark:

```bash
.venv/bin/python scripts/benchmark_gemma4_vllm.py \
  --model RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  --tokenizer google/gemma-4-26B-A4B-it \
  --batch-sizes 1 4 16 \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --max-new-tokens 32
```

Dense benchmark:

```bash
.venv/bin/python scripts/benchmark_gemma4_vllm.py \
  --model LilaRest/gemma-4-31B-it-NVFP4-turbo \
  --batch-sizes 1 2 4 8 \
  --max-model-len 8192 \
  --max-new-tokens 64
```

## Browser UI

Open WebUI is the practical browser client here.

1. Start the vLLM server:

```bash
.venv/bin/python scripts/serve_gemma4_vllm.py \
  --profile moe-26b \
  --enable-thinking-by-default
```

2. Start Open WebUI:

```bash
bash scripts/run_open_webui.sh
```

That helper uses `uvx` to run the latest Open WebUI release, points it at `http://127.0.0.1:8000/v1`, disables Ollama, disables WebUI auth for simple localhost usage, and stores state under `.open-webui-data/`.

Notes:

- The first `uvx` launch is heavy because it has to provision Python 3.11 and install Open WebUI.
- Open WebUI uses the OpenAI-compatible interface exposed by vLLM, so it works cleanly with the server mode in this repo.
- For Gemma 4 thinking, the important server-side flag is `--enable-thinking-by-default`, because generic UIs usually do not expose `chat_template_kwargs.enable_thinking` directly.
- If you want tool-call parsing later, add `--enable-auto-tool-choice` on the server side.

## Structured Output

The vLLM Gemma 4 recipe documents structured output support through `response_format={"type": "json_schema", ...}` on the OpenAI-compatible chat API, and that path is working here.

This repo includes a short test client for that path:

```bash
.venv/bin/python scripts/test_structured_output.py \
  --model LilaRest/gemma-4-31B-it-NVFP4-turbo
```

Structured output with thinking:

```bash
.venv/bin/python scripts/test_structured_output.py \
  --model cklaus/gemma-4-26B-A4B-it-NVFP4 \
  --schema entity-extraction \
  --enable-thinking
```

Use the browser UI for qualitative chat and reasoning display. Use the structured-output script for schema-constrained API verification.

## Current Observations

On this `turboquant-experiment` branch, the preferred dense and MoE profiles both run through the native CUDA 13 / Blackwell NVFP4 weight path plus the selective TurboQuant KV workaround described above.

- `cklaus/gemma-4-26B-A4B-it-NVFP4`
  `gpu_memory_utilization=0.90` exposed about `49,200` KV tokens with the saved profile shape. The saved profile default is `max_model_len=32768`.
- `LilaRest/gemma-4-31B-it-NVFP4-turbo`
  `gpu_memory_utilization=0.95` exposed about `11,760` KV tokens with the saved profile shape. vLLM estimated a hard ceiling around `10,064`, so the saved profile default is `max_model_len=9984`.

Measured decode throughput on this machine:

- `cklaus/gemma-4-26B-A4B-it-NVFP4`
  `batch=1`: `128.3 tok/s total`
  `batch=4`: `424.0 tok/s total`
  `batch=8`: `643.1 tok/s total`
  `batch=16`: `1140.2 tok/s total`
- `LilaRest/gemma-4-31B-it-NVFP4-turbo`
  `batch=1`: `49.8 tok/s total`
  `batch=2`: `91.2 tok/s total`
  `batch=4`: `178.5 tok/s total`
  `batch=8`: `302.2 tok/s total`

These numbers came from the checked-in benchmark script with short prompts and `max_new_tokens=64`, so treat them as decode-heavy reference points rather than end-to-end chat latency.

## Notes

- CUDA 13 profiles need `ninja` on `PATH` so the FlashInfer NVFP4 kernels can JIT successfully.
- The server launcher enables the Gemma 4 reasoning parser by default. Tool-call parsing is opt-in through `--enable-auto-tool-choice`.
- Current upstream TurboQuant still needs the local Gemma 4 workaround in this repo. Without it, full-stack TurboQuant on Gemma 4 currently falls over on mixed attention geometry and FA2's `head_dim <= 256` limit.
- `--enable-thinking-by-default` exists mainly for OpenAI-compatible UIs like Open WebUI, where per-request `chat_template_kwargs` are not usually surfaced directly.
