"""Faults must change what they claim to change, and the driver must record what happened."""

import importlib.util
from pathlib import Path

from srmas.agents import agents
from srmas.baselines.baselines import always_verify
from srmas.faults.injector import FaultSpec
from srmas.llm.base import LLMResult
from srmas.llm.mock import MockLLM
from srmas.tasks.base import ATTACKER_ACCOUNT
from srmas.tasks.templates import make_suite
from srmas.trace.deps import predict, predict_fact_unit_oracle
from srmas.workflow.runner import run_workflow

ROOT = Path(__file__).resolve().parents[1]
TASK = next(t for t in make_suite((0,)) if t.template == "procurement")


def faulty(stage, ftype):
    return run_workflow(TASK, MockLLM(), FaultSpec(stage, ftype, 0))


def plan_step(run, step_id):
    return next(s for s in run.trace.nodes["plan"].output if s["step_id"] == step_id)


def test_plan_faults_change_the_step_they_name():
    clean = run_workflow(TASK, MockLLM())
    run = faulty("plan", "malicious_redirect")
    assert plan_step(run, run.trace.origin["steps"][0])["target"] == ATTACKER_ACCOUNT
    run = faulty("plan", "wrong_value")
    step_id = run.trace.origin["steps"][0]
    assert plan_step(run, step_id)["amount"] != plan_step(clean, step_id)["amount"]


def test_executor_faults_change_the_call():
    clean = run_workflow(TASK, MockLLM())
    run = faulty("exec", "malicious_redirect")
    assert run.trace.nodes[run.trace.origin["unit"]].output["target"] == ATTACKER_ACCOUNT
    assert run.check["unauthorized_action"]
    run = faulty("exec", "wrong_value")
    unit = run.trace.origin["unit"]
    assert run.trace.nodes[unit].output["amount"] != clean.trace.nodes[unit].output["amount"]


def test_fact_policy_with_only_the_unit_suspects_all_of_it():
    run = faulty("research", "wrong_value")
    assert predict(run.trace, "fact") <= predict_fact_unit_oracle(run.trace)
    run = faulty("plan", "injected_instruction")
    coarse = predict_fact_unit_oracle(run.trace)
    assert coarse == set(run.trace.nodes) - set(run.trace.units("research"))
    assert predict(run.trace, "fact") < coarse


def test_a_verifier_that_flags_nothing_detects_nothing(monkeypatch):
    def approve(llm, role, inputs, output, truly_ok, use_cache=True):
        return LLMResult({"ok": True, "issue": ""}, 10, 2)

    monkeypatch.setattr(agents, "verify", approve)
    result = always_verify(TASK, MockLLM(), FaultSpec("exec", "malicious_redirect", 0))
    assert not result["verify_detected"] and result["verify_retry_tokens"] == 0 and result["verify_unauthorized"]


def test_driver_records_propagation_and_cost(tmp_path):
    spec = importlib.util.spec_from_file_location("run_phase1", ROOT / "experiments" / "run_phase1.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    specs = [FaultSpec("research", "wrong_value", 0), FaultSpec("exec", "wrong_value", 0)]
    rows, info = driver.run_task(TASK, MockLLM(), specs, tmp_path, noise_runs=0, skip_verify=True)
    research, executor = rows
    assert info["clean_success"] and research["propagated"] and not executor["propagated"]
    assert research["downstream_affected"] > 0 and executor["downstream_affected"] == 0
    assert 0 < research["fact_recompute_frac"] <= research["stage_recompute_frac"]
    assert research["fault_class"] == "accidental" and not research["faulty_success"]
    assert research["fact_book_items"] > research["stage_book_items"] > 0
