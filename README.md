# Gemma 4 Playground

This repo is set up to run Gemma 4 through vLLM on an RTX 5090, using a single CUDA 13 environment and model-specific defaults for the profiles that are currently worth testing here:

- `dense-31b`: `LilaRest/gemma-4-31B-it-NVFP4-turbo`
- `dense-31b-redhat`: `RedHatAI/gemma-4-31B-it-NVFP4`
- `moe-26b`: `cklaus/gemma-4-26B-A4B-it-NVFP4`
- `moe-26b-redhat`: `RedHatAI/gemma-4-26B-A4B-it-NVFP4`

The main entrypoint is [`scripts/serve_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/serve_gemma4_vllm.py). It resolves the right model, tokenizer, memory knobs, reasoning parser, and backend defaults for each profile, then launches `vllm serve`.

## Layout

- [`scripts/serve_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/serve_gemma4_vllm.py): primary server launcher
- [`scripts/gemma4_vllm_profiles.py`](/home/jamesa261/gemma-4-playground/scripts/gemma4_vllm_profiles.py): shared presets and runtime checks
- [`scripts/chat_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/chat_gemma4_vllm.py): local REPL for qualitative testing
- [`scripts/benchmark_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/benchmark_gemma4_vllm.py): in-process throughput benchmark
- [`scripts/run_open_webui.sh`](/home/jamesa261/gemma-4-playground/scripts/run_open_webui.sh): launch Open WebUI against the local vLLM OpenAI endpoint
- [`scripts/test_structured_output.py`](/home/jamesa261/gemma-4-playground/scripts/test_structured_output.py): minimal structured-output test against the server API
- [`scripts/setup_vllm.sh`](/home/jamesa261/gemma-4-playground/scripts/setup_vllm.sh): setup helper for the shared CUDA 13 vLLM environment

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

The old patched cu129 MoE path has been removed. Current vLLM already contains the upstream Gemma 4 MoE support needed for the CUDA 13 / Blackwell-native NVFP4 path.

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
- KV cache: `fp8`
- Max model length: `16384`
- GPU memory utilization: `0.95`
- Max sequences: `8`
- Max batched tokens: `8192`
- Reasoning parser: enabled
- Text-only optimization: `--language-model-only` plus `--limit-mm-per-prompt image=0,audio=0,video=0`

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
- KV cache: `auto`
- Max model length: `4096`
- GPU memory utilization: `0.90`
- Max sequences: `16`
- Max batched tokens: `8192`
- MoE backend: `auto`
- Reasoning parser: enabled
- Text-only optimization: enabled

This is the current default MoE profile because it cleanly picks up the native Blackwell NVFP4 path on this 5090 while landing close to the checkpoint author's published decode numbers.

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
  --max-model-len 32768 \
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
  --batch-sizes 1 4 16 \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --max-new-tokens 32
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
  --batch-sizes 1 2 4 8
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

On this 5090, the dense and MoE defaults are now both native CUDA 13 / Blackwell NVFP4 paths.

- `LilaRest/gemma-4-31B-it-NVFP4-turbo`
  Single-request decode was validated around `45 tok/s`, with batch-8 in-process decode around `309 tok/s`.
- `RedHatAI/gemma-4-31B-it-NVFP4`
  Similar single-request decode was validated around `46 tok/s`, with batch-8 in-process decode around `281 tok/s`, but with a tighter usable context budget on this card.
- `cklaus/gemma-4-26B-A4B-it-NVFP4`
  With `max_model_len=4096`, `max_num_seqs=16`, `gpu_memory_utilization=0.90`, and `max_new_tokens=32`, local in-process decode measured about `129 tok/s` at batch 1, `357 tok/s` at batch 4, and `1311 tok/s` at batch 16.
- `RedHatAI/gemma-4-26B-A4B-it-NVFP4`
  Under the same benchmark shape, local in-process decode measured about `138 tok/s` at batch 1, `348 tok/s` at batch 4, and `1343 tok/s` at batch 16. vLLM also emitted an NVFP4 warning that fused parallel layers are using different global scales, which may reduce accuracy. Treat this as the upstream baseline, not the preferred default.

Both 26B checkpoints loaded through the native CUDA 13 stack with NVFP4 GEMM and vLLM's Cutlass MoE path rather than the old patch-era compatibility route.

## Notes

- CUDA 13 profiles need `ninja` on `PATH` so the FlashInfer NVFP4 kernels can JIT successfully.
- The server launcher enables the Gemma 4 reasoning parser by default. Tool-call parsing is opt-in through `--enable-auto-tool-choice`.
- For the 26B Gemma 4 MoE models, vLLM currently forces the Triton attention backend because of the model's heterogeneous head dimensions. That is expected and does not mean the NVFP4 GEMM or MoE kernels failed to activate.
- `--enable-thinking-by-default` exists mainly for OpenAI-compatible UIs like Open WebUI, where per-request `chat_template_kwargs` are not usually surfaced directly.
