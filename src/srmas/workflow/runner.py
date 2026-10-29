"""Researcher -> planner -> executor workflow with fan-out, tracing and optional
fault injection / always-verify."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..agents import agents
from ..env.environment import TOOLS, Environment
from ..faults.injector import FaultSpec, Injector
from ..llm.base import LLM, LLMResult
from ..tasks.base import Fact, Task, check
from ..trace.trace import Node, Trace, facts_from_records, value_matched


@dataclass
class RunResult:
    trace: Trace
    env: Environment
    check: dict[str, Any]
    verify_calls: int = 0
    verify_tokens: int = 0
    retry_tokens: int = 0
    flagged: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def tokens(self) -> int:
        return sum(n.tokens for n in self.trace.nodes.values())

    def dump(self, path: str | Path) -> None:
        """Synthetic artifacts and tool actions; never client credentials."""
        payload = self.trace.to_dict() | {
            "environment": asdict(self.env), "check": self.check,
            "elapsed_s": self.elapsed_s, "tokens": self.tokens,
            "verify_tokens": self.verify_tokens, "retry_tokens": self.retry_tokens,
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _clean_records(data: dict[str, Any]) -> list[dict[str, str]]:
    return [{"entity": str(r.get("entity", "")), "attribute": str(r.get("attribute", "")),
             "value": str(r.get("value", ""))} for r in data.get("records", [])]


def _clean_steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    steps = []
    for i, s in enumerate(data.get("steps", []), 1):
        steps.append({
            "step_id": str(s.get("step_id") or f"s{i}"),
            "tool": s.get("tool", ""),
            "target": str(s.get("target", "")),
            "amount": round(float(s.get("amount") or 0.0), 2),
            "content": str(s.get("content", "")),
            "subject": str(s.get("subject", "")),
            "uses_facts": [str(x) for x in s.get("uses_facts", [])],
            "reads_state": [str(x) for x in s.get("reads_state", [])],
            "depends_on": [str(x) for x in s.get("depends_on", [])],
            "instruction": str(s.get("instruction", "")),
        })
    # de-duplicate step ids
    seen: dict[str, int] = {}
    for s in steps:
        seen[s["step_id"]] = seen.get(s["step_id"], 0) + 1
        if seen[s["step_id"]] > 1:
            s["step_id"] = f"{s['step_id']}_{seen[s['step_id']]}"
    return steps


def _clean_call(data: dict[str, Any]) -> dict[str, Any]:
    return {"tool": data.get("tool", ""), "target": str(data.get("target", "")).strip(),
            "amount": round(float(data.get("amount") or 0.0), 2),
            "content": str(data.get("content", "")), "subject": str(data.get("subject", ""))}


def _order(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plan order, moved only as needed to respect depends_on."""
    ids = {s["step_id"] for s in steps}
    done: set[str] = set()
    out: list[dict[str, Any]] = []
    pending = list(steps)
    while pending:
        for s in pending:
            if all(d in done or d not in ids for d in s["depends_on"]):
                break
        else:  # cycle: fall back to plan order
            s = pending[0]
        pending.remove(s)
        done.add(s["step_id"])
        out.append(s)
    return out


def _fill(node: Node, res: LLMResult) -> None:
    node.tokens_in, node.tokens_out = res.input_tokens, res.output_tokens
    node.cached, node.refused = res.cached, res.refused
    node.elapsed_s = res.elapsed_s


def run_workflow(task: Task, llm: LLM, fault: FaultSpec | None = None, verify: bool = False,
                 env: Environment | None = None, tag: str = "", use_cache: bool = True) -> RunResult:
    """Run the workflow once.

    env: continue on an existing environment (used by the full-restart baseline).
    tag: prefix for step names written into the environment's access log.
    verify: always-verify baseline; a flagged output is regenerated once with the
            (transient) fault removed, and the verifier's and retry's tokens are counted.
    """
    started = time.perf_counter()
    env = env if env is not None else Environment(workspace=task.files())
    inj = Injector(task, fault) if fault else None
    trace = Trace(task.task_id, fault.label if fault else None)
    out = RunResult(trace, env, {})

    def run_verifier(node: Node, role: str, inputs: str, output: Any, bad: bool, res: LLMResult) -> bool:
        """Returns True when the output must be regenerated."""
        v = agents.verify(llm, role, inputs, output, truly_ok=not bad, use_cache=use_cache)
        out.verify_calls += 1
        out.verify_tokens += v.input_tokens + v.output_tokens
        ok = bool(v.data.get("ok", False)) and not v.refused
        node.verify = {"ok": ok, "issue": v.data.get("issue", ""), "faulty": bad,
                       "tokens": v.input_tokens + v.output_tokens,
                       "cached": v.cached, "elapsed_s": v.elapsed_s}
        if not ok:
            out.flagged.append(node.node_id)
        return not ok

    def record_retry(node: Node, res: LLMResult) -> None:
        out.retry_tokens += res.input_tokens + res.output_tokens
        node.retry = asdict(res)

    # ------------------------------------------------------------- research (fan-out)
    facts: list[Fact] = []
    for src in task.sources:
        nid = f"research:{src.source_id}"
        text = env.read(tag + nid, f"file:sources/{src.source_id}.md") or ""
        res = agents.research(llm, src, text, use_cache)
        raw = _clean_records(res.data)
        recs, changed = inj.research(src.source_id, raw) if inj else (raw, [])
        node = trace.add(Node(nid, "research", [], recs, injected=bool(changed)))
        _fill(node, res)
        if verify and run_verifier(node, "researcher", f"{text}\n\nInstructions: {src.extract}",
                                   {"records": recs}, bool(changed), res):
            retry = agents.research(llm, src, text, use_cache=False)
            record_retry(node, retry)
            recs, changed = _clean_records(retry.data), []
            node.output, node.injected = recs, False
        src_facts = facts_from_records(src.source_id, recs)
        node.produces_facts = [f.fact_id for f in src_facts]
        if changed:
            trace.origin = {"unit": nid, "facts": [src_facts[i].fact_id for i in changed],
                            "applied": inj.applied}
        facts += src_facts
    trace.facts = facts
    by_id = {f.fact_id: f for f in facts}

    # ------------------------------------------------------------- plan
    res = agents.plan(llm, task, facts, use_cache)
    raw_steps = _clean_steps(res.data)
    steps, changed_steps = inj.plan(raw_steps) if inj else (raw_steps, [])
    node = trace.add(Node("plan", "plan", trace.units("research"), steps, injected=bool(changed_steps)))
    _fill(node, res)
    plan_inputs = f"Goal: {task.goal}\n\nFacts:\n" + "\n".join(f.render() for f in facts)
    if verify and run_verifier(node, "planner", plan_inputs, {"steps": steps}, bool(changed_steps), res):
        retry = agents.plan(llm, task, facts, use_cache=False)
        record_retry(node, retry)
        steps, changed_steps = _clean_steps(retry.data), []
        node.output, node.injected = steps, False
    if changed_steps:
        trace.origin = {"unit": "plan", "steps": changed_steps, "applied": inj.applied}
    for s in steps:
        declared = [f for f in s["uses_facts"] if f in by_id]
        matched = value_matched(facts, s["target"], s["amount"], s["content"])
        trace.step_facts[s["step_id"]] = sorted(set(declared) | set(matched))
    node.uses_facts = sorted({f for ids in trace.step_facts.values() for f in ids})

    # ------------------------------------------------------------- execute
    if inj:
        inj.choose_exec_target(steps)
    for s in _order(steps):
        nid = f"exec:{s['step_id']}"
        state: dict[str, str | None] = {}
        reads = []
        for r in s["reads_state"]:
            r = r.strip()
            state[r] = env.read(tag + nid, r)
            w = env.accesses[-1].writer
            reads.append((r, w[len(tag):] if w and w.startswith(tag) else w))
        cited = [by_id[f] for f in trace.step_facts[s["step_id"]]]
        res = agents.execute(llm, s, cited, state, use_cache)
        raw_call = _clean_call(res.data)
        call, injected = inj.exec(s["step_id"], raw_call) if inj else (raw_call, False)
        node = trace.add(Node(nid, "exec", ["plan"] + [w for _, w in reads if w], call,
                              reads=reads, injected=injected))
        _fill(node, res)
        if verify:
            inputs = (f"Plan step: {json.dumps(s)}\nFacts:\n" + "\n".join(f.render() for f in cited)
                      + "\nState:\n" + "\n".join(f"{k} = {v}" for k, v in state.items()))
            if run_verifier(node, "executor", inputs, call, injected, res):
                retry = agents.execute(llm, s, cited, state, use_cache=False)
                record_retry(node, retry)
                call, injected = _clean_call(retry.data), False
                node.output, node.injected = call, False
        if injected:
            trace.origin = {"unit": nid, "applied": inj.applied}
        node.uses_facts = sorted(set(trace.step_facts[s["step_id"]])
                                 | set(value_matched(facts, call["target"], call["amount"], call["content"])))
        if call["tool"] not in TOOLS or not call["target"]:
            node.error = f"invalid call {call['tool']!r} -> {call['target']!r}"
            continue
        n_before = len(env.accesses)
        try:
            env.call(tag + nid, call["tool"], call["target"], call["amount"], call["content"],
                     call["subject"])
        except Exception as exc:  # malformed tool calls are recorded, not fatal
            node.error = repr(exc)
        node.writes = [a.resource for a in env.accesses[n_before:] if a.kind == "write"]

    out.check = check(task, env)
    out.elapsed_s = time.perf_counter() - started
    return out
