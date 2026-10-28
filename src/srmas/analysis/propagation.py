"""How far a fault spread: affected units and corrupted state per run."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..workflow.runner import RunResult
from .counterfactual import _tokens, actual_affected


def email_sigs(env) -> Counter:
    return Counter((e.to.lower(), _tokens(e.subject + " " + e.body)) for e in env.outbox)


def propagation(faulty: RunResult, clean: RunResult) -> dict[str, Any]:
    ft, ct = faulty.trace, clean.trace
    affected = actual_affected(ft, ct)
    by_stage = Counter(
        (ft.nodes.get(u) or ct.nodes[u]).stage for u in affected)
    fv, cv = faulty.env.state_view(), clean.env.state_view()

    kv_bad = sum(fv["kv"].get(k) != cv["kv"].get(k) for k in set(fv["kv"]) | set(cv["kv"]))
    file_bad = sum(_tokens(fv["files"].get(k, "")) != _tokens(cv["files"].get(k, ""))
                   for k in set(fv["files"]) | set(cv["files"]))
    fp, cp = Counter(fv["payments"]), Counter(cv["payments"])
    pay_bad = sum(((fp - cp) + (cp - fp)).values())
    fe, ce = email_sigs(faulty.env), email_sigs(clean.env)
    wrong_emails = sum((fe - ce).values())

    n_units = len(set(ft.nodes) | set(ct.nodes))
    return {
        "units": n_units,
        "affected": len(affected),
        "affected_frac": len(affected) / n_units if n_units else 0.0,
        "affected_research": by_stage.get("research", 0),
        "affected_plan": by_stage.get("plan", 0),
        "affected_exec": by_stage.get("exec", 0),
        "downstream_affected": len(affected - {ft.origin.get("unit")}),
        "kv_corrupted": kv_bad,
        "files_corrupted": file_bad,
        "payments_wrong": pay_bad,
        "emails_wrong": wrong_emails,  # irreversible
        "fault_benign": len(affected - {ft.origin.get("unit")}) == 0 and kv_bad + file_bad + pay_bad + wrong_emails == 0,
    }
