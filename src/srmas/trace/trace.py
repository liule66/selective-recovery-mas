"""Provenance trace of one workflow run.

A *unit* is one agent invocation (research:<source>, plan, exec:<step>); it is the
granularity at which work can be recomputed. Within units the trace also keeps
finer provenance: the facts each research unit produced, the facts each plan
step cites, and the shared-state resources each executor read and wrote.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..tasks.base import Fact, as_float, norm_num

STAGES = ("research", "plan", "exec")


@dataclass
class Node:
    node_id: str
    stage: str
    parents: list[str]  # message-level inputs (unit ids)
    output: Any
    produces_facts: list[str] = field(default_factory=list)
    uses_facts: list[str] = field(default_factory=list)
    reads: list[tuple[str, str | None]] = field(default_factory=list)  # (resource, writer unit)
    writes: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    refused: bool = False
    injected: bool = False
    error: str = ""
    verify: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    retry: dict[str, Any] | None = None

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class Trace:
    task_id: str
    fault: str | None
    nodes: dict[str, Node] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    step_facts: dict[str, list[str]] = field(default_factory=dict)  # plan step -> cited facts
    origin: dict[str, Any] = field(default_factory=dict)  # where the fault entered

    def add(self, node: Node) -> Node:
        self.nodes[node.node_id] = node
        return node

    def units(self, stage: str | None = None) -> list[str]:
        return [n for n, node in self.nodes.items() if stage is None or node.stage == stage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fault": self.fault,
            "origin": self.origin,
            "facts": [asdict(f) for f in self.facts],
            "step_facts": self.step_facts,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
        }

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=str)


def facts_from_records(source_id: str, records: list[dict[str, str]]) -> list[Fact]:
    from ..tasks.templates import fact_id

    out, seen = [], {}
    for r in records:
        fid = fact_id(source_id, r["entity"], r["attribute"])
        seen[fid] = seen.get(fid, 0) + 1
        if seen[fid] > 1:
            fid = f"{fid}#{seen[fid]}"
        out.append(Fact(fid, source_id, r["entity"], r["attribute"], str(r["value"])))
    return out


def value_matched(facts: list[Fact], target: str, amount: float, content: str) -> list[str]:
    """Facts whose value visibly flows into a step or call (detected provenance).

    Short numbers are skipped because they would match almost any text.
    """
    text = norm_num(f"{target}\n{content}").lower()
    hits = []
    for f in facts:
        v = norm_num(f.value).strip().lower()
        num = as_float(v)
        if num is not None and not re.search(r"[a-z]", v):
            if amount and abs(num - amount) < 0.005 and len(v) >= 3:
                hits.append(f.fact_id)
            elif len(v) >= 4 and re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", text):
                hits.append(f.fact_id)
        elif len(v) >= 4 and v in text:
            hits.append(f.fact_id)
    return hits
