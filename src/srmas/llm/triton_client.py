"""OpenAI-compatible campus gateway with explicit rates and a persistent budget.

No automatic retries: a timed-out request may already have been billed.
Unknown request costs retain their conservative reservation in the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import validate

from .base import LLM, LLMResult, Oracle


class BudgetExceeded(RuntimeError):
    pass


class GatewayRequestError(RuntimeError):
    def __init__(self, kind: str, http_status: int | None):
        self.http_status = http_status
        super().__init__(f"Gateway request failed ({kind}, HTTP {http_status}); reservation retained")


class TritonLLM(LLM):
    def __init__(self, cache_dir: str | Path | None = None, *, model: str | None = None,
                 budget_usd: float = 5.0, input_rate: float | None = None,
                 output_rate: float | None = None, max_tokens: int = 4096,
                 ledger: str | Path = ".local/api-budget.json", timeout: float = 90,
                 min_interval_s: float = 1.25):
        super().__init__(cache_dir)
        self.model = model or os.environ.get("SRMAS_MODEL", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.model or not self.base_url.startswith("https://") or not self.api_key:
            raise ValueError("Set SRMAS_MODEL, HTTPS OPENAI_BASE_URL and OPENAI_API_KEY")
        self.input_rate = input_rate if input_rate is not None else float(os.environ["SRMAS_INPUT_USD_PER_M"])
        self.output_rate = output_rate if output_rate is not None else float(os.environ["SRMAS_OUTPUT_USD_PER_M"])
        if (not all(math.isfinite(x) for x in (self.input_rate, self.output_rate, budget_usd))
                or min(self.input_rate, self.output_rate) < 0 or not 0 < budget_usd <= 5 or max_tokens < 1):
            raise ValueError("Use nonnegative verified rates, budget in (0, 5], and positive max_tokens")
        self.budget_usd, self.max_tokens, self.timeout = budget_usd, max_tokens, timeout
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.budget_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.next_request = 0.0
        self.min_interval_s = min_interval_s
        self.costs = json.loads(self.ledger.read_text()) if self.ledger.exists() else []
        # Scope cache to endpoint, model and generation settings, never the key.
        identity = hashlib.sha256(json.dumps([self.base_url, self.max_tokens, "json_schema"]).encode()).hexdigest()[:12]
        self.name = f"{self.model}@{identity}"

    def _save(self) -> None:
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.costs, indent=1), encoding="utf-8")
        tmp.replace(self.ledger)

    def _call(self, system: str, user: str, schema: dict[str, Any],
              oracle: Oracle | None) -> LLMResult:
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "agent_output", "strict": True, "schema": schema}}}
        body = json.dumps(payload).encode("utf-8")
        # UTF-8 bytes bound byte-level tokens; allow additional gateway framing.
        reserved = ((len(body) + 8192) * self.input_rate + self.max_tokens * self.output_rate) / 1e6
        with self.budget_lock:
            if sum(c["charged_or_reserved_usd"] for c in self.costs) + reserved > self.budget_usd:
                raise BudgetExceeded("API budget reservation would exceed the configured cap")
            rec = {"model": self.model, "charged_or_reserved_usd": reserved, "status": "reserved"}
            self.costs.append(rec)
            self._save()
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body,
                  headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with self.request_lock:
                time.sleep(max(0.0, self.next_request - time.monotonic()))
                self.next_request = time.monotonic() + self.min_interval_s
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except Exception as exc:
            # Do not persist API error bodies, headers or a URL containing credentials.
            with self.budget_lock:
                rec["status"] = "request_failed_cost_unknown"
                rec["http_status"] = getattr(exc, "code", None)
                self._save()
            raise GatewayRequestError(type(exc).__name__, getattr(exc, "code", None)) from None
        usage = data.get("usage") or {}
        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise RuntimeError("Gateway omitted usage; budget reservation retained")
        ti, to = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        with self.budget_lock:
            rec.update(status="completed", input_tokens=ti, output_tokens=to,
                       charged_or_reserved_usd=(ti * self.input_rate + to * self.output_rate) / 1e6)
            self._save()
        choice = data["choices"][0]
        message = choice["message"]
        if message.get("refusal"):
            return LLMResult({}, ti, to, refused=True)
        if choice.get("finish_reason") != "stop":
            raise RuntimeError("Gateway did not complete the structured response")
        result = json.loads(message["content"])
        validate(result, schema)
        return LLMResult(result, ti, to)
