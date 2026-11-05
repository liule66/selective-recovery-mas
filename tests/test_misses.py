from srmas.analysis.misses import call_correct, missed_wrong_units
from srmas.llm.mock import MockLLM
from srmas.tasks.templates import make_suite
from srmas.workflow.runner import run_workflow

TASKS = {t.template: t for t in make_suite((0,))}


def exec_calls(task):
    run = run_workflow(task, MockLLM())
    assert run.check["success"]
    return {n: node for n, node in run.trace.nodes.items() if node.stage == "exec"}


def test_every_call_of_a_clean_run_is_correct():
    for task in TASKS.values():
        assert all(call_correct(task, node.output) for node in exec_calls(task).values())


def test_wrong_calls_are_rejected():
    task = TASKS["procurement"]
    acct, amount, memo = task.expected.payments[0]
    assert call_correct(task, {"tool": "pay", "target": acct, "amount": amount, "content": memo})
    assert not call_correct(task, {"tool": "pay", "target": acct, "amount": amount - 1, "content": memo})
    assert not call_correct(task, {"tool": "pay", "target": acct, "amount": amount, "content": "other"})
    assert not call_correct(task, {"tool": "pay", "target": "", "amount": amount, "content": memo})
    assert not call_correct(task, {"tool": "kv_set", "target": "hold/s1", "content": "HOLD"})
    to, subs = task.expected.emails[0]
    assert call_correct(task, {"tool": "send_email", "target": to, "subject": "Order",
                               "content": " ".join(subs) + " (7 units)"})
    assert not call_correct(task, {"tool": "send_email", "target": to, "subject": "Order", "content": "done"})


def test_missed_units_are_judged_by_their_output():
    task = TASKS["procurement"]
    acct, amount, memo = task.expected.payments[0]
    good = {"output": {"tool": "pay", "target": acct, "amount": amount, "content": memo}}
    bad = {"output": {"tool": "pay", "target": acct, "amount": amount - 1, "content": memo}}
    nodes = {"plan": {"output": []}, "exec:s1": good, "exec:s2": bad}
    assert missed_wrong_units(task, nodes, ["plan", "exec:s1"]) == []
    assert missed_wrong_units(task, nodes, ["plan", "exec:s2", "exec:s9"]) == ["exec:s2", "exec:s9"]
    assert missed_wrong_units(task, {"exec:s1": good | {"error": "boom"}}, ["exec:s1"]) == ["exec:s1"]
