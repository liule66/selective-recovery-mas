"""Deterministic mock LLM: returns whatever the agent's oracle computes.

Each agent passes an oracle that produces the "ideal" answer from the agent's
structured inputs (which may already contain injected faults), so the full
pipeline can be exercised without API calls.
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLM, LLMResult, Oracle


class MockLLM(LLM):
    name = "mock"

    def _call(self, system: str, user: str, schema: dict[str, Any],
              oracle: Oracle | None) -> LLMResult:
        if oracle is None:
            raise ValueError("MockLLM needs an oracle for every call")
        data = oracle()
        return LLMResult(data, (len(system) + len(user)) // 4, len(json.dumps(data)) // 4)
