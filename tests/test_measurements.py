import json
from io import BytesIO
from unittest.mock import patch

import pandas as pd
import pytest

from srmas.faults.injector import FaultSpec
from srmas.llm.mock import MockLLM
from srmas.llm.triton_client import BudgetExceeded, TritonLLM
from srmas.tasks.templates import make_suite
from srmas.workflow.runner import run_workflow


def test_verification_retries_uncached_and_records_actual_usage(tmp_path):
    llm = MockLLM(cache_dir=tmp_path / "cache")
    task = make_suite((0,))[0]
    run_workflow(task, llm)
    before = llm.stats.live_calls
    run = run_workflow(task, llm, FaultSpec("plan", "injected_instruction", 0), verify=True)
    retried = [n for n in run.trace.nodes.values() if n.retry]
    assert retried and all(not n.retry["cached"] for n in retried)
    assert llm.stats.live_calls > before
    assert run.retry_tokens == sum(n.retry["input_tokens"] + n.retry["output_tokens"] for n in retried)
    run.dump(tmp_path / "trace.json")
    rec = json.loads((tmp_path / "trace.json").read_text())
    assert rec["environment"]["accesses"] and rec["check"]["success"]
    assert rec["elapsed_s"] > 0


def test_gateway_budget_cache_and_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    client = TritonLLM(model="fake-model", input_rate=1, output_rate=2,
                       cache_dir=tmp_path / "cache", ledger=tmp_path / "budget.json")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    response = {"usage": {"prompt_tokens": 20, "completion_tokens": 10},
                "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
    with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as send:
        first = client.complete("system", "question", schema)
        second = client.complete("system", "question", schema)
    assert send.call_count == 1 and second.cached and first.elapsed_s > 0
    assert client.costs[0]["charged_or_reserved_usd"] == pytest.approx(.00004)
    assert "fake-test-key" not in client.ledger.read_text()
    assert "fake-test-key" not in "".join(f.read_text() for f in (tmp_path / "cache").iterdir())
    client.budget_usd = .00001
    with patch("urllib.request.urlopen") as send, pytest.raises(BudgetExceeded):
        client.complete("system", "new question", schema)
    send.assert_not_called()


def test_timeout_retains_reservation_without_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    client = TritonLLM(model="fake-model", input_rate=1, output_rate=2,
                       ledger=tmp_path / "budget.json")
    with patch("urllib.request.urlopen", side_effect=TimeoutError("secret-server-detail")) as send:
        with pytest.raises(RuntimeError, match="reservation retained") as exc:
            client.complete("system", "question", {})
    assert send.call_count == 1 and "secret-server-detail" not in str(exc.value)
    assert client.costs[0]["charged_or_reserved_usd"] > 0


def test_restart_summary_uses_clean_tokens(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    spec = spec_from_file_location("make_figures", Path(__file__).parents[1] / "experiments/make_figures.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    df = pd.read_csv(Path(__file__).parents[1] / "results/phase1_mock/runs.csv").head(2)
    df["clean_tokens"] = 100
    df["restart_tokens"] = 200
    df["clean_success"] = [True, False]
    meta = {"tasks": [{"task_id": df.task_id.iloc[0], "clean_success": True, "clean_tokens": 100}]}
    text = module.tables(df, meta, tmp_path)
    assert "2.0" in text.split("full restart")[1].splitlines()[0]
    assert "Clean-capable subset: 1/2" in text


def test_all_failed_run_keeps_manifest_and_returns_failure(monkeypatch, tmp_path):
    import sys
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    spec = spec_from_file_location("run_phase1", Path(__file__).parents[1] / "experiments/run_phase1.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    def fail(task, llm, seeds, raw, *args):
        (raw / task.task_id).mkdir(parents=True)
        raise TimeoutError("sensitive server error not for publication")
    monkeypatch.setattr(module, "run_task", fail)
    out = tmp_path / "failed"
    monkeypatch.setattr(sys, "argv", ["run_phase1", "--templates", "procurement", "--task-seeds", "0", "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
    meta = json.loads((out / "tasks.json").read_text())
    assert meta["expected_fault_runs"] == 22 and meta["failures"][0]["error_type"] == "TimeoutError"
    assert "sensitive server" not in (out / "tasks.json").read_text()


@pytest.mark.parametrize("bad_rate", [float("nan"), float("inf"), -1])
def test_invalid_prices_cannot_disable_budget(monkeypatch, tmp_path, bad_rate):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    with pytest.raises(ValueError):
        TritonLLM(model="fake-model", input_rate=bad_rate, output_rate=1, ledger=tmp_path / "budget.json")
