#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from openai import OpenAI


SCHEMA_PRESETS: dict[str, dict[str, Any]] = {
    "city-info": {
        "system_prompt": "Extract city information as structured JSON.",
        "user_prompt": "Tell me about Paris, France.",
        "schema_name": "city-info",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "population": {"type": "integer"},
                "landmarks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["city", "country", "population", "landmarks"],
        },
    },
    "entity-extraction": {
        "system_prompt": (
            "Analyze the text and extract entities. Output JSON with:\n"
            "- people: list of person names mentioned\n"
            "- organizations: list of organization names\n"
            "- locations: list of location names\n"
            "- summary: one-sentence summary of the text"
        ),
        "user_prompt": (
            "Dr. Elena Torres, lead researcher at the Riverside Institute, "
            "presented her findings on marine biodiversity at the annual symposium in "
            "Cape Marina. The Oceanic Wildlife Fund and the Global Conservation Alliance "
            "both pledged support."
        ),
        "schema_name": "entity-extraction",
        "schema": {
            "type": "object",
            "properties": {
                "people": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["people", "organizations", "locations", "summary"],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a Gemma 4 structured-output request to a local vLLM OpenAI-compatible server."
    )
    parser.add_argument("--model", required=True, help="Model ID exposed by the running vLLM server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the local server.")
    parser.add_argument(
        "--schema",
        choices=sorted(SCHEMA_PRESETS),
        default="city-info",
        help="Built-in structured-output preset to test.",
    )
    parser.add_argument("--system-prompt", default=None, help="Override the preset system prompt.")
    parser.add_argument("--prompt", default=None, help="Override the preset user prompt.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Send chat_template_kwargs.enable_thinking in extra_body for structured-output-plus-thinking tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = SCHEMA_PRESETS[args.schema]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    messages = [
        {
            "role": "system",
            "content": args.system_prompt or preset["system_prompt"],
        },
        {
            "role": "user",
            "content": args.prompt or preset["user_prompt"],
        },
    ]

    # Make the request deterministic even when the server was launched with
    # --enable-thinking-by-default. Structured-output tests should only spend
    # output budget on reasoning when the caller explicitly opts in.
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {
            "enable_thinking": args.enable_thinking,
        }
    }

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": preset["schema_name"],
                "schema": preset["schema"],
            },
        },
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )

    choice = response.choices[0]
    message = choice.message

    print("=== Response Metadata ===")
    print(f"finish_reason: {choice.finish_reason}")
    if response.usage is not None:
        print(f"prompt_tokens: {response.usage.prompt_tokens}")
        print(f"completion_tokens: {response.usage.completion_tokens}")
        print(f"total_tokens: {response.usage.total_tokens}")
    print()

    if getattr(message, "reasoning", None):
        print("=== Thinking ===")
        print(message.reasoning)
        print()

    print("=== Raw Content ===")
    print(message.content)
    print()

    if message.content is None:
        print("No final content was returned.")
        if choice.finish_reason == "length":
            print("Hint: the response hit max_tokens before the model emitted the final JSON content.")
            if args.enable_thinking:
                print("Hint: when thinking is enabled, reasoning tokens share the same output budget as the final answer.")
        return 1

    try:
        parsed = json.loads(message.content)
    except json.JSONDecodeError as exc:
        print(f"JSON decode failed: {exc}")
        if choice.finish_reason == "length":
            print("Hint: the response hit max_tokens. If thinking is enabled, the reasoning trace shares that same output budget.")
        return 1

    print("=== Parsed JSON ===")
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
