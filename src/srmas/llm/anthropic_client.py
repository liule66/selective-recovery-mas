"""Claude client (official `anthropic` SDK) with structured JSON outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic

from .base import LLM, LLMResult, Oracle

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_api_key() -> str | None:
    """ANTHROPIC_API_KEY if set, else ./api.txt at the project root (gitignored)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    path = PROJECT_ROOT / "api.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


class AnthropicLLM(LLM):
    def __init__(self, model: str | None = None, effort: str = "low",
                 max_tokens: int = 8000, cache_dir: str | Path | None = None):
        super().__init__(cache_dir=cache_dir)
        self.model = model or os.environ.get("SRMAS_MODEL", "claude-haiku-5-5")
        self.name = self.model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=load_api_key(), max_retries=8)

    def _call(self, system: str, user: str, schema: dict[str, Any],
              oracle: Oracle | None) -> LLMResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        usage = response.usage
        if response.stop_reason == "refusal":
            return LLMResult({}, usage.input_tokens, usage.output_tokens, refused=True)
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{self.model} hit max_tokens={self.max_tokens}")
        text = next(b.text for b in response.content if b.type == "text")
        return LLMResult(json.loads(text), usage.input_tokens, usage.output_tokens)
