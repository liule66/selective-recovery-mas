"""The report must agree with the stored results, and the scorers must still reproduce them."""

import importlib.util
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import pytest

from srmas.analysis.counterfactual import score
from srmas.tasks.templates import make_suite
from srmas.trace.deps import GRANULARITIES, bookkeeping, predict, predict_fact_unit_oracle
from srmas.trace.trace import Trace

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIRST = RESULTS / "phase1_haiku_revised"
TEX = (ROOT / "report" / "phase1" / "main.tex").read_text(encoding="utf-8")
TASKS = {t.task_id: t for t in make_suite((0, 1))}

spec = importlib.util.spec_from_file_location("phase1_analysis", ROOT / "experiments" / "phase1_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def num(x, digits=2):
    return str(Decimal(str(float(x))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def pct(x):
    return num(100 * x, 0) + r"\%"


def row(label, nth=0):
    lines = [line for line in TEX.splitlines() if line.startswith(label)]
    return [cell.strip().rstrip("\\").strip() for cell in lines[nth].split("&")[1:]]


def load(directory, r):
    path = directory / "traces" / r.task_id / f"{r.stage}.{r.ftype}.{r.fault_seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def clean(directory, task_id):
    return json.loads((directory / "traces" / task_id / "clean.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runs():
    return pd.read_csv(FIRST / "runs.csv")


@pytest.fixture(scope="module")
def unit_oracle(runs):
    rows = []
    for r in runs[runs.propagated].itertuples():
        faulty, base = Trace.from_dict(load(FIRST, r)), Trace.from_dict(clean(FIRST, r.task_id))
        units = predict_fact_unit_oracle(faulty)
        s = score(units, faulty, base, skip=frozenset({faulty.origin["unit"]}))
        rows.append(s | {"recompute": sum(faulty.nodes[u].tokens for u in units) / r.clean_tokens})
    return pd.DataFrame(rows)


def test_scorers_reproduce_the_stored_results(runs):
    for r in runs.itertuples():
        faulty, base = Trace.from_dict(load(FIRST, r)), Trace.from_dict(clean(FIRST, r.task_id))
        skip = frozenset({faulty.origin["unit"]})
        for g in GRANULARITIES:
            predicted = predict(faulty, g)
            s = score(predicted, faulty, base, skip=skip)
            assert s["precision"] == pytest.approx(getattr(r, f"{g}_ds_precision")), (r.task_id, g)
            assert s["recall"] == pytest.approx(getattr(r, f"{g}_ds_recall")), (r.task_id, g)
            assert s["under"] == getattr(r, f"{g}_ds_under") and s["over"] == getattr(r, f"{g}_ds_over")
            assert bookkeeping(faulty, g)["items"] == getattr(r, f"{g}_book_items")
            cost = sum(faulty.nodes[u].tokens for u in predicted) / r.clean_tokens
            assert cost == pytest.approx(getattr(r, f"{g}_recompute_frac"))


def test_table_1_matches_the_results(runs, unit_oracle):
    p = runs[runs.propagated]
    res = p[p.stage == "research"]

    def cells(sub, g, over_digits):
        return [num(sub[f"{g}_ds_precision"].mean()), num(sub[f"{g}_ds_recall"].mean()), pct((sub[f"{g}_ds_under"] > 0).mean()),
                num(sub[f"{g}_ds_over"].mean(), over_digits), num(sub[f"{g}_recompute_frac"].mean())]

    for label, g, digits in (("Stage (82)", "stage", 1), ("Message (82)", "message", 1), ("Fact (82)", "fact", 2)):
        records = f"{num(p[f'{g}_book_items'].mean(), 0)} / {num(p[f'{g}_book_bytes'].mean() / 1000, 1)}\\,KB"
        assert row(label) == cells(p, g, digits) + [records], label
    assert len(p) == 82 and len(res) == 50
    assert row("Fact: researcher (50)")[:5] == cells(res, "fact", 2)
    assert row(r"Fact: told only $v_0$ (82)")[:5] == [
        num(unit_oracle.precision.mean()), num(unit_oracle.recall.mean()), pct((unit_oracle.under > 0).mean()),
        num(unit_oracle.over.mean(), 1), num(unit_oracle.recompute.mean())]


def test_table_2_matches_the_results(runs):
    bad = runs.faulty_emails_unexpected + runs.faulty_emails_incorrect
    state = (runs.faulty_missing_payments + runs.faulty_extra_payments + runs.faulty_kv_wrong + runs.faulty_files_wrong) == 0
    assert row("No recovery")[:3] == [pct(state.mean()), pct(runs.faulty_unauthorized_action.mean()), pct((bad > 0).mean())]
    assert row("Full restart")[:4] == [pct(runs.restart_state_ok.mean()) + "*", pct(runs.restart_unauthorized.mean()),
                                       pct((runs.restart_unrecovered_emails > 0).mean()),
                                       num((runs.restart_tokens / runs.clean_tokens).mean()) + r"$\times$"]
    assert row("Verify + restore")[:4] == [pct(runs.verify_state_ok.mean()) + "*", pct(runs.verify_unauthorized.mean()) + "*",
                                           pct((runs.verify_bad_emails > 0).mean()) + "*",
                                           num((runs.verify_overhead_tokens / runs.clean_tokens).mean()) + r"$\times$"]
    assert row("Fact-level (est.)")[3] == num(runs.fact_recompute_frac.mean()) + r"$\times$"

    live = pd.read_csv(RESULTS / "phase1_haiku_live" / "runs.csv")
    assert len(live) == 44 and live.restart_cache_hits.sum() == 0 and live.verify_cache_hits.sum() == 0
    assert row("Full restart", 1) == [pct(live.restart_state_ok.mean()), pct(live.restart_unauthorized.mean()),
                                      pct((live.restart_unrecovered_emails > 0).mean()),
                                      num((live.restart_tokens / live.clean_tokens).mean()) + r"$\times$",
                                      num(live.restart_elapsed_s.median(), 1) + r"\,s"]
    assert row("Verify + retry") == [pct(live.verify_state_ok.mean()), pct(live.verify_unauthorized.mean()),
                                     pct((live.verify_bad_emails > 0).mean()),
                                     num((live.verify_overhead_tokens / live.clean_tokens).mean()) + r"$\times$",
                                     num(live.verify_elapsed_s.median(), 1) + r"\,s"]


def test_ranges_across_samples_are_the_reported_ones():
    dirs = [FIRST, RESULTS / "phase1_haiku_repeat1", RESULTS / "phase1_haiku_repeat2"]
    stats = pd.DataFrame([analysis.sample_stats(d, TASKS) for d in dirs])

    def span(column, fmt):
        return f"{fmt(stats[column].min())}--{fmt(stats[column].max())}"

    plain_pct = lambda x: num(100 * x, 0)  # noqa: E731
    claims = [
        f"recall {span('fact recall', num)}",
        f"miss {span('fact miss runs', plain_pct)}\\%",
        f"cost {span('fact recompute', num)}",
        f"{span('researcher fact recall', num)}, {span('researcher fact miss runs', plain_pct)}\\%",
        f"only {span('pass', plain_pct)}\\%",
        f"or {span('pass without planner wrong/redirect', plain_pct)}\\%",
        f"{int(stats.propagated.min())}--{int(stats.propagated.max())} of 176",
        f"{span('unauthorized accidental', plain_pct)}\\% of accidental",
        f"did in {span('researcher redirect unauthorized', plain_pct)}\\%",
        f"12\\%; {span('researcher wrong-result runs', plain_pct)}\\% across samples",
    ]
    flat = " ".join(TEX.split())
    for claim in claims:
        assert " ".join(claim.split()) in flat, claim
    assert (stats["researcher injection unauthorized"] == 0).all()
    assert stats["planner wrong/redirect ignored"].sum() == 94
    assert stats["planner wrong/redirect ignored, text states correct value"].sum() == 84
    assert stats["planner fabricated, real payee overpaid"].sum() == 36
    assert (stats["planner fabricated, overpaid iff it names one"] == 16).all()
    assert stats["injected plan steps, attacker paid"].sum() == 48 and stats["injected plan steps, attacker emailed"].sum() == 32
    assert (stats["executor faults propagated"] == 0).all() and (stats["executor-fault runs with a ledger read"] == 0).all()


def test_clean_run_count_is_the_reported_one():
    dirs = [FIRST, RESULTS / "phase1_haiku_repeat1", RESULTS / "phase1_haiku_repeat2"]
    current = [t["clean_success"] for d in dirs for t in json.loads((d / "tasks.json").read_text(encoding="utf-8"))["tasks"]]
    latency = json.loads((RESULTS / "phase1_haiku_latency" / "tasks.json").read_text(encoding="utf-8"))
    current += [n["success"] for t in latency["tasks"] for n in t["noise"]]
    assert (sum(current), len(current)) == (39, 40)
    assert re.search(r"Of 40 clean runs under the current checker, 39 pass", " ".join(TEX.split()))


def test_task_files_match_the_generated_tasks():
    for task_id, task in TASKS.items():
        stored = json.loads((ROOT / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        assert stored == json.loads(json.dumps(task.to_json())), task_id
