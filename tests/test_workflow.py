import pytest

from srmas.analysis.counterfactual import actual_affected, score
from srmas.baselines.baselines import always_verify, full_restart
from srmas.faults.injector import FaultSpec, fault_matrix
from srmas.llm.mock import MockLLM
from srmas.tasks.templates import make_suite
from srmas.trace.deps import predict
from srmas.workflow.runner import run_workflow

TASKS = make_suite((0, 1))


@pytest.fixture(scope="module")
def llm():
    return MockLLM()


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_clean_run_passes_checker(task, llm):
    res = run_workflow(task, llm)
    assert res.check["success"], res.check
    assert not res.check["safety_violation"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.task_id)
def test_every_fault_is_injected(task, llm):
    for spec in fault_matrix((0, 1)):
        res = run_workflow(task, llm, spec)
        assert res.trace.origin, spec.label


def test_counterfactual_replay_matches_clean(tmp_path):
    llm = MockLLM(cache_dir=tmp_path)
    task = TASKS[0]
    clean = run_workflow(task, llm)
    run_workflow(task, llm, FaultSpec("research", "wrong_value", 0))
    replay = run_workflow(task, llm)
    assert actual_affected(replay.trace, clean.trace) == set()


def test_adversarial_injection_reaches_irreversible_tool(llm):
    task = TASKS[0]
    res = run_workflow(task, llm, FaultSpec("research", "injected_instruction", 0))
    assert res.check["attacker_emailed"] == 1
    assert res.check["attacker_paid"] == 250.0


@pytest.mark.parametrize("stage,ftype", [("research", "wrong_value"), ("research", "stale_value"),
                                         ("plan", "malicious_redirect"), ("exec", "wrong_value")])
def test_granularities_are_nested(stage, ftype, llm):
    for task in TASKS:
        clean = run_workflow(task, llm)
        faulty = run_workflow(task, llm, FaultSpec(stage, ftype, 0))
        fact, msg, stg = (predict(faulty.trace, g) for g in ("fact", "message", "stage"))
        assert fact <= msg <= stg
        actual = actual_affected(faulty.trace, clean.trace)
        assert score(stg, actual, faulty.trace, clean.trace)["recall"] == 1.0


def test_full_restart_restores_state_but_not_email(llm):
    task = TASKS[0]
    clean = run_workflow(task, llm)
    faulty = run_workflow(task, llm, FaultSpec("research", "injected_instruction", 0))
    rec = full_restart(task, llm, faulty, clean)
    assert rec["restart_state_ok"]
    assert rec["restart_unrecovered_emails"] >= 1


def test_always_verify_with_perfect_mock_detector(llm):
    out = always_verify(TASKS[2], llm, FaultSpec("plan", "injected_instruction", 0))
    assert out["verify_detected"] and out["verify_false_pos"] == 0 and out["verify_success"]
