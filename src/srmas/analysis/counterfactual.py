"""Counterfactual comparison of a faulty run against the replay in which the
injected result is replaced by the clean one.

With the response cache, a unit whose prompt did not change returns exactly its
recorded output, so a unit counts as *actually affected* only if the fault changed
its inputs and that in turn changed what it did.

Two details keep the comparison honest:
  * Outputs are compared on decision-relevant content (tool, target, amount, KV
    value, and the amounts / accounts / addresses / codes mentioned in free text),
    so rewording by a re-sampled step does not count as propagation.
  * Executor units are aligned across the two runs by content before falling back
    to plan position, so a fault that inserts or drops a plan step does not shift
    every later step into a false mismatch.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..tasks.base import norm_num, slug
from ..trace.trace import Trace

TOKEN_RE = re.compile(
    r"ACCT-[A-Z]+-\d+"  # account ids
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"  # email addresses
    r"|\b[A-Z]{2}\d{3}\b"  # flight codes
    r"|\d+\.\d+"  # amounts
)


def _tokens(text: str) -> tuple[str, ...]:
    toks = set()
    for t in TOKEN_RE.findall(norm_num(text)):
        toks.add(f"{float(t):.2f}" if t[0].isdigit() else t.lower())
    return tuple(sorted(toks))


def call_sig(call: dict[str, Any]) -> tuple:
    tool = call.get("tool", "")
    content = call.get("content", "")
    body = content.strip().lower() if tool == "kv_set" else _tokens(content + " " + call.get("subject", ""))
    return (tool, str(call.get("target", "")).strip().lower(), round(float(call.get("amount") or 0), 2), body)


def unit_sig(trace: Trace, unit: str | None) -> Any:
    node = trace.nodes.get(unit) if unit else None
    if node is None:
        return None
    if node.stage == "research":
        return tuple(sorted((slug(r["entity"]), slug(r["attribute"]), norm_num(r["value"]).strip().lower())
                            for r in node.output))
    if node.stage == "plan":
        # The plan's decision is its set of actions; draft message text in a step is
        # rewritten by the executor, so it is compared there, not here.
        return tuple(sorted(Counter(
            call_sig(s)[:3] + ((call_sig(s)[3],) if s.get("tool") == "kv_set" else ())
            for s in node.output).items()))
    return call_sig(node.output) + (bool(node.error),)


def align(faulty: Trace, clean: Trace) -> list[tuple[str | None, str | None]]:
    """Pairs of (faulty unit, clean unit); None marks a unit present in one run only."""
    pairs: list[tuple[str | None, str | None]] = [
        (u if u in faulty.nodes else None, u if u in clean.nodes else None)
        for u in dict.fromkeys(faulty.units() + clean.units())
        if (faulty.nodes.get(u) or clean.nodes[u]).stage != "exec"
    ]
    fe, ce = faulty.units("exec"), clean.units("exec")
    keys = [
        lambda t, u: unit_sig(t, u),
        lambda t, u: unit_sig(t, u)[:2],  # same tool and target
        lambda t, u: u,  # same plan position
    ]
    for key in keys:
        for f in list(fe):
            kf = key(faulty, f)
            c = next((c for c in ce if key(clean, c) == kf), None)
            if c is not None:
                pairs.append((f, c))
                fe.remove(f)
                ce.remove(c)
    pairs += [(f, None) for f in fe] + [(None, c) for c in ce]
    return pairs


def affected_pairs(faulty: Trace, clean: Trace) -> list[tuple[str | None, str | None]]:
    return [(f, c) for f, c in align(faulty, clean) if unit_sig(faulty, f) != unit_sig(clean, c)]


def actual_affected(faulty: Trace, clean: Trace) -> set[str]:
    return {f or c for f, c in affected_pairs(faulty, clean)}


def score(predicted: set[str], faulty: Trace, clean: Trace) -> dict[str, float]:
    """Precision/recall of a predicted invalidation set over aligned units.

    A unit that exists only in the clean replay (an action the faulty run dropped)
    is regenerated when the planner, or the faulty step at the same position, is
    re-run, so it counts as predicted in that case.
    """
    pred_n = tp = 0
    actual_n = 0
    for f, c in align(faulty, clean):
        if f is not None:
            p = f in predicted
        else:
            p = "plan" in predicted or c in predicted
        a = unit_sig(faulty, f) != unit_sig(clean, c)
        pred_n += p
        actual_n += a
        tp += p and a
    return {
        "precision": tp / pred_n if pred_n else 1.0,
        "recall": tp / actual_n if actual_n else 1.0,
        "over": pred_n - tp,
        "under": actual_n - tp,
        "n_pred": pred_n,
    }
