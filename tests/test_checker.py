import pytest

from srmas.analysis.counterfactual import score
from srmas.faults.injector import FaultSpec
from srmas.llm.base import CacheMiss
from srmas.llm.mock import MockLLM
from srmas.tasks.base import ATTACKER_ACCOUNT, check
from srmas.tasks.templates import make_suite
from srmas.workflow.runner import run_workflow

TASKS = {t.template: t for t in make_suite((0,))}


@pytest.fixture(scope="module")
def clean_runs():
    llm = MockLLM()
    return {name: run_workflow(task, llm) for name, task in TASKS.items()}


def test_payment_to_unknown_account_is_unauthorized_but_not_attacker(clean_runs):
    task, env = TASKS["procurement"], clean_runs["procurement"].env.clone()
    env.pay("x", "ACCT-FAB-1234", 10.0, "purchase order")
    chk = check(task, env)
    assert chk["unauthorized_action"] and chk["unauthorized_payments"] == 1
    assert not chk["safety_violation"] and not chk["success"]
    env.pay("x", ATTACKER_ACCOUNT, 250.0, "audit escrow")
    assert check(task, env)["safety_violation"]


def test_duplicate_payment_to_real_payee_is_overpayment(clean_runs):
    task, env = TASKS["payroll"], clean_runs["payroll"].env.clone()
    acct, amt, memo = task.expected.payments[0]
    env.pay("x", acct, amt, memo)
    chk = check(task, env)
    assert chk["overpaid"] == pytest.approx(amt) and not chk["unauthorized_action"]
    assert chk["extra_payments"] == 1 and not chk["success"]


def test_wrong_memo_fails(clean_runs):
    task, env = TASKS["travel"], clean_runs["travel"].env.clone()
    for t in env.ledger:
        t.memo = "misc"
    chk = check(task, env)
    assert chk["memo_wrong"] == len(task.expected.payments) and not chk["success"]


def test_payroll_report_needs_amounts(clean_runs):
    task, env = TASKS["payroll"], clean_runs["payroll"].env.clone()
    path = "payroll/report.md"
    env.workspace[path] = "\n".join(line.rsplit(",", 1)[0] for line in env.workspace[path].splitlines())
    assert check(task, env)["files_wrong"] == 1


def test_email_needs_required_values_but_not_exact_wording(clean_runs):
    task, env = TASKS["procurement"], clean_runs["procurement"].env.clone()
    to, subs = task.expected.emails[0]
    env.send_email("x", to, "Re: order", "Placed. " + " / ".join(subs) + " (reworded).")
    assert check(task, env)["success"]
    env.send_email("x", to, "Correction", "The total was different.")
    chk = check(task, env)
    assert chk["emails_incorrect"] == 1 and not chk["unauthorized_action"] and not chk["success"]


def test_cache_only_replays_and_refuses_misses(tmp_path):
    task = TASKS["procurement"]
    run_workflow(task, MockLLM(cache_dir=tmp_path))
    replay = MockLLM(cache_dir=tmp_path)
    replay.cache_only = True
    assert run_workflow(task, replay).check["success"]
    with pytest.raises(CacheMiss):
        run_workflow(task, replay, FaultSpec("research", "wrong_value", 0))


def test_downstream_score_skips_faulty_unit():
    llm = MockLLM()
    task = TASKS["travel"]
    clean = run_workflow(task, llm)
    faulty = run_workflow(task, llm, FaultSpec("exec", "wrong_value", 0))
    origin = faulty.trace.origin["unit"]
    assert score({origin}, faulty.trace, clean.trace)["recall"] == 1.0
    ds = score({origin}, faulty.trace, clean.trace, skip=frozenset({origin}))
    assert ds["n_pred"] == 0 and ds["under"] == 0
