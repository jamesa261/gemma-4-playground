#!/usr/bin/env python3
from __future__ import annotations

import os

from gemma4_vllm_turboquant import apply_vllm_turboquant_workarounds


def main() -> int:
    if os.environ.get("GEMMA4_VLLM_ENABLE_TURBOQUANT_PATCH") == "1":
        apply_vllm_turboquant_workarounds()

    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
