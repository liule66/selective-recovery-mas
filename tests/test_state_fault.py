import pytest

from srmas.analysis.misses import state_fault_readers, wrong_state_emails
from srmas.faults.injector import STATE_VALUE, FaultSpec, fault_matrix, state_matrix
from srmas.llm.mock import MockLLM
from srmas.tasks.templates import make_suite
from srmas.trace.deps import GRANULARITIES, predict
from srmas.workflow.runner import run_workflow

TASKS = make_suite((0, 1))


def test_matrix_stays_at_eleven_cells():
    assert len(fault_matrix((0, 1))) == 22
    assert all(s.ftype != STATE_VALUE for s in fault_matrix((0, 1)))
    assert [s.label for s in state_matrix((0, 1))] == [f"exec.{STATE_VALUE}.0", f"exec.{STATE_VALUE}.1"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_a_corrupted_stored_value_reaches_readers_only_through_state(task):
    llm = MockLLM()
    clean = run_workflow(task, llm)
    run = run_workflow(task, llm, FaultSpec("exec", STATE_VALUE, 0))
    assert run.trace.origin, "every task stores a value a later step reads"
    writer = run.trace.nodes[run.trace.origin["unit"]]
    assert writer.output["tool"] == "kv_set" and writer.injected
    assert run.check["kv_wrong"] and run.trace.nodes["plan"].output == clean.trace.nodes["plan"].output

    readers = state_fault_readers(task, run.trace.to_dict() | {"environment": {"kv": run.env.kv}})
    emails = [r for r in readers if r["tool"] == "send_email"]
    assert emails and all(r["carried"] and not r["correct"] for r in emails)
    assert run.check["emails_incorrect"]
    for g in GRANULARITIES:
        assert {r["unit"] for r in readers} <= predict(run.trace, g)


def _trace(content="", instruction="", uses=(), facts=(), stored="999.00"):
    task = next(t for t in TASKS if t.template == "procurement")
    to = task.expected.emails[0][0]
    return task, {
        "origin": {"unit": "research:x", "facts": list(facts)},
        "environment": {"kv": {**task.expected.kv, "order/total": stored}},
        "nodes": {
            "plan": {"stage": "plan", "output": [{"step_id": "s5", "content": content, "instruction": instruction}]},
            "exec:s5": {"stage": "exec", "reads": [["kv:order/total", "exec:s3"]], "uses_facts": list(uses),
                        "output": {"tool": "send_email", "target": to, "subject": "Order", "content": f"Total {stored}"}},
        },
    }


def test_a_read_is_the_only_visible_source_when_nothing_else_carries_the_error():
    task, trace = _trace(content="Total: {kv:order/total}")
    assert wrong_state_emails(task, trace) == [
        {"unit": "exec:s5", "cites_fault": False, "in_plan": False, "read_only": True}]


def test_a_cited_fact_or_plan_text_also_explains_the_wrong_email():
    task, trace = _trace(uses=["f1"], facts=["f1"])
    assert wrong_state_emails(task, trace)[0] == {"unit": "exec:s5", "cites_fault": True, "in_plan": False, "read_only": False}
    task, trace = _trace(instruction="Report 999.00 as the total.")
    assert wrong_state_emails(task, trace)[0] == {"unit": "exec:s5", "cites_fault": False, "in_plan": True, "read_only": False}
