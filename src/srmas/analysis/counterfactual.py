"""Counterfactual comparison of a faulty run against the replay in which the
injected result is replaced by the clean one.

With the response cache, a unit whose prompt did not change returns exactly its
recorded output, so a unit counts as *actually affected* only if the fault changed
its inputs and that in turn changed what it did.

Outputs are compared semantically (decision-relevant content: tools, targets,
amounts, numbers and identifiers), not character by character, so harmless
rewording by a re-sampled step does not count as propagation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..tasks.base import as_float, norm_num, slug
from ..trace.trace import Trace

TOKEN_RE = re.compile(r"ACCT-[A-Z]+-\d+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\d+(?:\.\d+)?")


def _tokens(text: str) -> tuple[str, ...]:
    toks = []
    for t in TOKEN_RE.findall(norm_num(text)):
        f = as_float(t)
        toks.append(f"{f:.2f}" if f is not None and not t.startswith("ACCT") and "@" not in t else t.lower())
    return tuple(sorted(set(toks)))


def call_sig(call: dict[str, Any]) -> tuple:
    tool = call.get("tool", "")
    content = call.get("content", "")
    body = content.strip().lower() if tool == "kv_set" else _tokens(content + " " + call.get("subject", ""))
    return (tool, str(call.get("target", "")).strip().lower(), round(float(call.get("amount") or 0), 2), body)


def unit_sig(trace: Trace, unit: str) -> Any:
    node = trace.nodes.get(unit)
    if node is None:
        return None
    if node.stage == "research":
        return tuple(sorted((slug(r["entity"]), slug(r["attribute"]), norm_num(r["value"]).strip().lower())
                            for r in node.output))
    if node.stage == "plan":
        return tuple(sorted(Counter(call_sig(s) for s in node.output).items()))
    return call_sig(node.output) + (bool(node.error),)


def actual_affected(faulty: Trace, clean: Trace) -> set[str]:
    units = set(faulty.nodes) | set(clean.nodes)
    return {u for u in units if unit_sig(faulty, u) != unit_sig(clean, u)}


def score(predicted: set[str], actual: set[str], faulty: Trace, clean: Trace) -> dict[str, float]:
    """Precision/recall of a predicted invalidation set.

    Units that exist only in the clean replay (actions the faulty plan dropped) are
    regenerated whenever the planner is re-run, so they count as predicted if `plan` is.
    """
    pred = set(predicted)
    if "plan" in pred:
        pred |= set(clean.nodes) - set(faulty.nodes)
    tp = len(pred & actual)
    return {
        "precision": tp / len(pred) if pred else 1.0,
        "recall": tp / len(actual) if actual else 1.0,
        "over": len(pred - actual),
        "under": len(actual - pred),
        "n_pred": len(pred),
    }
