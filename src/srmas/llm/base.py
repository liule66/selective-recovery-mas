"""LLM client interface with an on-disk response cache.

Identical prompts can yield different outputs. The cache makes an unchanged step return
exactly the output it returned before. That is what both counterfactual replay and
selective recovery need: only steps whose inputs changed are re-sampled, so any
changes in unchanged steps are avoided. Changed prompts can still resample;
counterfactual comparisons do not remove every stochastic confounder.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Oracle = Callable[[], dict[str, Any]]


@dataclass
class LLMResult:
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cached: bool = False
    refused: bool = False
    elapsed_s: float = 0.0


@dataclass
class LLMStats:
    live_calls: int = 0
    cache_hits: int = 0
    live_input_tokens: int = 0
    live_output_tokens: int = 0
    refusals: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items() if k != "lock"}


class LLM:
    """Base class. Subclasses implement `_call`."""

    name = "base"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = LLMStats()

    def complete(self, system: str, user: str, schema: dict[str, Any],
                 oracle: Oracle | None = None, use_cache: bool = True) -> LLMResult:
        started = time.perf_counter()
        key = hashlib.sha256(
            json.dumps([self.name, system, user, schema], sort_keys=True).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir and use_cache else None
        if path is not None and path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            with self.stats.lock:
                self.stats.cache_hits += 1
            return LLMResult(rec["data"], rec["input_tokens"], rec["output_tokens"],
                             cached=True, refused=rec.get("refused", False),
                             elapsed_s=time.perf_counter() - started)

        result = self._call(system, user, schema, oracle)
        with self.stats.lock:
            self.stats.live_calls += 1
            self.stats.live_input_tokens += result.input_tokens
            self.stats.live_output_tokens += result.output_tokens
            self.stats.refusals += int(result.refused)
        if path is not None:
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({
                "data": result.data, "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens, "refused": result.refused,
            }), encoding="utf-8")
            tmp.replace(path)
        result.elapsed_s = time.perf_counter() - started
        return result

    def _call(self, system: str, user: str, schema: dict[str, Any],
              oracle: Oracle | None) -> LLMResult:
        raise NotImplementedError


def make_llm(kind: str, cache_dir: str | Path | None = None, **kwargs: Any) -> LLM:
    if kind == "mock":
        from .mock import MockLLM
        return MockLLM(cache_dir=cache_dir)
    if kind in ("haiku", "anthropic"):
        from .anthropic_client import AnthropicLLM
        return AnthropicLLM(cache_dir=cache_dir, **kwargs)
    if kind == "triton":
        from .triton_client import TritonLLM
        return TritonLLM(cache_dir=cache_dir, **kwargs)
    raise ValueError(f"unknown llm {kind!r}")
