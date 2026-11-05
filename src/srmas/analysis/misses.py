"""Whether a unit that a tracking policy missed actually holds a wrong result, and what could
have carried a wrong value into an email.

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


def _step_text(trace: dict[str, Any], unit: str) -> str:
    """The plan's text for an executor's step; the plan can state a value the state also holds."""
    step_id = unit.split(":", 1)[1]
    step = next((s for s in trace["nodes"]["plan"]["output"] if s["step_id"] == step_id), {})
    return f"{step.get('content', '')} {step.get('instruction', '')}"


def wrong_state_emails(task: Task, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Incorrect emails in a faulty run that read a wrong stored value.

    A read shows only that the path existed. The same error can also arrive through a corrupted
    fact the email cites or through the plan's text for its step. Plan text that reformats the
    value is not recognized, so `read_only` is an upper bound.
    """
    kv = trace["environment"]["kv"]
    bad = {f"kv:{k}": kv.get(k) for k, v in task.expected.kv.items() if not _kv_match(kv.get(k), v)}
    corrupted = set(trace["origin"].get("facts", []))
    emails = []
    for unit, node in trace["nodes"].items():
        call = node["output"]
        if node["stage"] != "exec" or call.get("tool") != "send_email" or call_correct(task, call):
            continue
        read = {resource for resource, _ in node["reads"]} & set(bad)
        if not read:
            continue
        cites = bool(corrupted & set(node["uses_facts"]))
        in_plan = any(bad[r] and _contains(_step_text(trace, unit), bad[r]) for r in read)
        emails.append({"unit": unit, "cites_fault": cites, "in_plan": in_plan, "read_only": not cites and not in_plan})
    return emails


def state_fault_readers(task: Task, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """What each reader did after a stored value was corrupted directly (a state_value fault)."""
    writer = trace["nodes"][trace["origin"]["unit"]]["output"]
    key, stored = writer["target"], writer["content"]
    correct = task.expected.kv.get(key, "")
    readers = []
    for unit, node in trace["nodes"].items():
        if node["stage"] != "exec" or f"kv:{key}" not in {resource for resource, _ in node["reads"]}:
            continue
        call = node["output"]
        text = f"{call.get('subject', '')} {call.get('content', '')}"
        readers.append({"unit": unit, "tool": call.get("tool"), "plan_states_value": _contains(_step_text(trace, unit), correct),
                        "carried": _contains(text, stored), "correct": call_correct(task, call)})
    return readers
