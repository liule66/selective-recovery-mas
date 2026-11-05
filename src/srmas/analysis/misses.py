"""Whether a unit that a tracking policy missed actually holds a wrong result.

Under-invalidation counts every unit whose decision-relevant signature changed. A re-sampled
report can add a number and still be right, and reusing it would do no harm, so a missed
executor's call is judged here against the task's reference outcome.
"""

from __future__ import annotations

from typing import Any

from ..tasks.base import Task, _contains, _kv_match


def call_correct(task: Task, call: dict[str, Any]) -> bool:
    """Whether one tool call matches what the checker requires of its target."""
    exp = task.expected
    tool, target = call.get("tool"), str(call.get("target", "")).strip()
    content = str(call.get("content", ""))
    if tool == "pay":
        amount = float(call.get("amount") or 0)
        return any(target == acct and abs(amount - amt) < 0.011 and all(_contains(content, m) for m in memo)
                   for acct, amt, *memo in exp.payments)
    if tool == "kv_set":
        return target in exp.kv and _kv_match(content, exp.kv[target])
    if tool == "write_file":
        return target in exp.files and all(_contains(content, s) for s in exp.files[target])
    if tool == "send_email":
        text = str(call.get("subject", "")) + "\n" + content
        return any(target.lower() == to.lower() and all(_contains(text, s) for s in subs)
                   for to, subs in exp.emails)
    return False


def missed_wrong_units(task: Task, nodes: dict[str, Any], missed: list[str]) -> list[str]:
    """The missed units whose result is wrong. `nodes` are a faulty run's trace nodes.

    The plan acts only through its executors, so it is judged by them. A missed unit that is
    not in the faulty run is an action that run dropped.
    """
    wrong = []
    for unit in missed:
        if unit == "plan":
            continue
        node = nodes.get(unit)
        if node is None or node.get("error") or not call_correct(task, node["output"]):
            wrong.append(unit)
    return wrong
