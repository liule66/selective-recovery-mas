"""Dependency tracking at three granularities and the affected-set predictions
they imply, given where a fault entered.

  stage   : invalidate the faulty stage and every later stage.
  message : transitive closure over unit -> unit messages (research -> plan ->
            exec, plus exec -> exec edges through shared state).
  fact    : transitive closure over fact provenance: faulty facts -> plan steps
            that cite them -> their executors -> readers of what they wrote.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .trace import STAGES, Trace

GRANULARITIES = ("stage", "message", "fact")


def _state_children(trace: Trace) -> dict[str, set[str]]:
    """exec unit -> exec units that read a resource it wrote."""
    kids: dict[str, set[str]] = defaultdict(set)
    for nid, node in trace.nodes.items():
        for _, writer in node.reads:
            if writer and writer in trace.nodes and writer != nid:
                kids[writer].add(nid)
    return kids


def _closure(start: set[str], kids: dict[str, set[str]]) -> set[str]:
    seen, todo = set(start), list(start)
    while todo:
        for k in kids.get(todo.pop(), ()):
            if k not in seen:
                seen.add(k)
                todo.append(k)
    return seen


def message_edges(trace: Trace) -> dict[str, set[str]]:
    kids: dict[str, set[str]] = defaultdict(set)
    for nid, node in trace.nodes.items():
        for p in node.parents:
            kids[p].add(nid)
    return kids


def predict(trace: Trace, granularity: str) -> set[str]:
    """Units that dependency tracking at `granularity` would invalidate."""
    origin = trace.origin
    if not origin:
        return set()
    unit = origin["unit"]
    stage = trace.nodes[unit].stage
    if granularity == "stage":
        later = set(STAGES[STAGES.index(stage):])
        return {n for n, node in trace.nodes.items() if node.stage in later}
    if granularity == "message":
        return _closure({unit}, message_edges(trace))
    if granularity != "fact":
        raise ValueError(granularity)

    state_kids = _state_children(trace)
    if stage == "research":
        bad = set(origin.get("facts", []))
        steps = {sid for sid, fids in trace.step_facts.items() if bad & set(fids)}
        # executors whose own call visibly used a faulty value also count
        execs = {f"exec:{sid}" for sid in steps} | {
            n for n, node in trace.nodes.items() if node.stage == "exec" and bad & set(node.uses_facts)}
        hit = {unit} | ({"plan"} if steps else set())
        return hit | _closure({e for e in execs if e in trace.nodes}, state_kids)
    if stage == "plan":
        execs = {f"exec:{sid}" for sid in origin.get("steps", [])}
        return {"plan"} | _closure({e for e in execs if e in trace.nodes}, state_kids)
    return _closure({unit}, state_kids)


def bookkeeping(trace: Trace, granularity: str) -> dict[str, int]:
    """Size of the dependency metadata each granularity has to keep."""
    if granularity == "stage":
        recs: Any = {n: node.stage for n, node in trace.nodes.items()}
        n_items = len(recs)
    elif granularity == "message":
        recs = {n: node.parents for n, node in trace.nodes.items()}
        n_items = sum(len(v) for v in recs.values()) + len(recs)
    else:
        recs = {
            "facts": {n: node.produces_facts for n, node in trace.nodes.items() if node.produces_facts},
            "step_facts": trace.step_facts,
            "exec_facts": {n: node.uses_facts for n, node in trace.nodes.items() if node.stage == "exec"},
            "state": {n: [node.reads, node.writes] for n, node in trace.nodes.items() if node.stage == "exec"},
        }
        n_items = (sum(len(v) for v in recs["facts"].values())
                   + sum(len(v) for v in recs["step_facts"].values())
                   + sum(len(v) for v in recs["exec_facts"].values())
                   + sum(len(r) + len(w) for r, w in recs["state"].values()))
    return {"items": n_items, "bytes": len(json.dumps(recs, default=str))}
