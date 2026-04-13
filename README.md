# Gemma 4 Playground

This repo is set up to run Gemma 4 through vLLM on an RTX 5090, with model-specific defaults for the two serving targets that are currently most practical here:

- `dense-31b`: `LilaRest/gemma-4-31B-it-NVFP4-turbo` on a CUDA 13 vLLM stack for native Blackwell NVFP4 kernels
- `moe-26b`: `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` on a patched vLLM stack for the community NVFP4 MoE checkpoint

The main entrypoint is [`scripts/serve_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/serve_gemma4_vllm.py). It resolves the right model, memory knobs, reasoning parser, and backend defaults for each profile, then launches `vllm serve`.

## Layout

- [`scripts/serve_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/serve_gemma4_vllm.py): primary server launcher
- [`scripts/gemma4_vllm_profiles.py`](/home/jamesa261/gemma-4-playground/scripts/gemma4_vllm_profiles.py): shared model presets and runtime checks
- [`scripts/chat_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/chat_gemma4_vllm.py): local REPL for qualitative testing
- [`scripts/benchmark_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/benchmark_gemma4_vllm.py): in-process throughput benchmark
- [`scripts/run_open_webui.sh`](/home/jamesa261/gemma-4-playground/scripts/run_open_webui.sh): launch Open WebUI against the local vLLM OpenAI endpoint
- [`scripts/setup_vllm_cu130.sh`](/home/jamesa261/gemma-4-playground/scripts/setup_vllm_cu130.sh): setup helper for the CUDA 13 dense environment
- [`patches/vllm-gemma4-modelopt-moe-loader.patch`](/home/jamesa261/gemma-4-playground/patches/vllm-gemma4-modelopt-moe-loader.patch): required patch for the community 26B MoE checkpoint

## Environments

Two vLLM environments are currently useful:

- `.venv-vllm-cu130`
  Used for the 31B dense NVFP4 checkpoints. This is the environment that exposes the native Blackwell FP4 path. The launcher validates that it is running against CUDA 13 before it will start a dense profile.
- `.venv-vllm`
  Used for the patched community 26B MoE checkpoint. This is the environment where the Gemma 4 loader patch has already been applied.

The corresponding manifests are kept in the repo on purpose:

- [`requirements-vllm-cu130.in`](/home/jamesa261/gemma-4-playground/requirements-vllm-cu130.in) and [`requirements-vllm-cu130.txt`](/home/jamesa261/gemma-4-playground/requirements-vllm-cu130.txt) for the dense CUDA 13 environment
- [`requirements-vllm-moe-cu129.in`](/home/jamesa261/gemma-4-playground/requirements-vllm-moe-cu129.in) and [`requirements-vllm-moe-cu129.txt`](/home/jamesa261/gemma-4-playground/requirements-vllm-moe-cu129.txt) for the patched MoE environment

The dense setup helper is:

```bash
bash scripts/setup_vllm_cu130.sh
```

The MoE environment still needs the loader patch after rebuilding the env:

```bash
patch -p0 < patches/vllm-gemma4-modelopt-moe-loader.patch
```

## Quick Start

Best default dense server:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py
```

Alternative dense server using the Red Hat checkpoint:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py --profile dense-31b-redhat
```

MoE server:

```bash
.venv-vllm/bin/python scripts/serve_gemma4_vllm.py --profile moe-26b
```

All of these start an OpenAI-compatible vLLM server on `http://127.0.0.1:8000/v1`.

To inspect the resolved command without launching the server:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py --dry-run
```

If you want Gemma 4 tool-call parsing enabled on the server as well:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py --enable-auto-tool-choice
```

If you want generic OpenAI-compatible clients to get Gemma 4 thinking mode without having to send `chat_template_kwargs` manually:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py --enable-thinking-by-default
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
- Max model length: `4096`
- GPU memory utilization: `0.95`
- Max sequences: `8`
- Max batched tokens: `4096`
- Reasoning parser: enabled
- Text-only optimization: enabled

The Red Hat model card suggests a much larger context window, but on this 32 GB 5090 that setting did not leave enough KV cache headroom. The preset is deliberately conservative so the server starts reliably.

`moe-26b`

- Model: `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`
- Runtime: patched cu129 vLLM env
- Quantization: `modelopt`
- Max model length: `50000`
- GPU memory utilization: `0.9`
- Max sequences: `1`
- Max batched tokens: `4096`
- MoE backend: `marlin`
- Reasoning parser: enabled
- Text-only optimization: enabled

## Useful Overrides

The launcher accepts the common knobs you are likely to vary during downstream testing:

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
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

```bash
.venv-vllm/bin/python scripts/serve_gemma4_vllm.py \
  --profile moe-26b \
  --max-model-len 32768
```

## Chat And Benchmarking

Qualitative testing against a loaded model is easiest through [`scripts/chat_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/chat_gemma4_vllm.py).

Dense chat:

```bash
.venv-vllm-cu130/bin/python scripts/chat_gemma4_vllm.py --model LilaRest/gemma-4-31B-it-NVFP4-turbo
```

MoE chat:

```bash
.venv-vllm/bin/python scripts/chat_gemma4_vllm.py --model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
```

Throughput benchmarking is handled by [`scripts/benchmark_gemma4_vllm.py`](/home/jamesa261/gemma-4-playground/scripts/benchmark_gemma4_vllm.py).

Dense benchmark:

```bash
.venv-vllm-cu130/bin/python scripts/benchmark_gemma4_vllm.py \
  --model LilaRest/gemma-4-31B-it-NVFP4-turbo \
  --batch-sizes 1 2 4 8
```

Red Hat benchmark:

```bash
.venv-vllm-cu130/bin/python scripts/benchmark_gemma4_vllm.py \
  --model RedHatAI/gemma-4-31B-it-NVFP4 \
  --max-model-len 4096 \
  --batch-sizes 1 2 4 8
```

## Browser UI

For a browser chat UI, the pragmatic path is Open WebUI rather than building another client in this repo.

Start the vLLM server first:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py
```

If you want reasoning enabled for all UI requests, start the server with:

```bash
.venv-vllm-cu130/bin/python scripts/serve_gemma4_vllm.py --enable-thinking-by-default
```

Then start Open WebUI:

```bash
bash scripts/run_open_webui.sh
```

That helper uses `uvx` to run the latest Open WebUI release, points it at `http://127.0.0.1:8000/v1`, disables Ollama in the UI, and stores Open WebUI state under `.open-webui-data/`.

Notes:

- The first `uvx` launch is heavy because it has to provision Python 3.11 and install Open WebUI.
- Open WebUI uses the OpenAI-compatible interface exposed by vLLM, so it works cleanly with the server mode in this repo.
- For Gemma 4 thinking, the important server-side flag is `--enable-thinking-by-default`, because generic UIs usually do not expose `chat_template_kwargs.enable_thinking` directly.
- For Gemma 4 tool calling, the vLLM recipe also recommends `--tool-call-parser gemma4`, `--enable-auto-tool-choice`, and the Gemma 4 tool chat template from the vLLM examples. That alignment is still partial in this repo today; the current helper is aimed at browser chat and reasoning rather than end-to-end tool use.

## Current Observations

On this machine, the dense CUDA 13 NVFP4 path is the best serving default.

- `LilaRest/gemma-4-31B-it-NVFP4-turbo`
  Single-request decode was validated around `45 tok/s`, with batch-8 in-process decode around `309 tok/s`.
- `RedHatAI/gemma-4-31B-it-NVFP4`
  Similar single-request decode was validated around `46 tok/s`, with batch-8 in-process decode around `281 tok/s`, but with a tighter usable context budget on this card.
- `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`
  Works with the patch, but is substantially slower and should be treated as a compatibility path rather than the default serving path.

## Notes

- Dense profiles require `ninja` on `PATH` so the FlashInfer NVFP4 kernels can JIT successfully.
- The server launcher enables the Gemma 4 reasoning parser by default. Tool-call parsing is opt-in through `--enable-auto-tool-choice`.
- `--enable-thinking-by-default` exists mainly for OpenAI-compatible UIs like Open WebUI, where per-request `chat_template_kwargs` are not usually surfaced directly.
- The non-CUDA-13 vLLM manifest is still kept because the current MoE workflow is pinned to the patched `.venv-vllm` environment rather than the CUDA 13 dense stack.
