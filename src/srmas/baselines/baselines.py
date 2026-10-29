"""Recovery baselines: full restart and always-verify."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import time

from ..analysis.propagation import email_sigs
from ..env.environment import Environment
from ..faults.injector import FaultSpec
from ..llm.base import LLM
from ..tasks.base import Task
from ..workflow.runner import RunResult, run_workflow


def _state_ok(chk: dict[str, Any]) -> bool:
    return not (chk["missing_payments"] or chk["extra_payments"] or chk["kv_wrong"] or chk["files_wrong"])


def full_restart(task: Task, llm: LLM, faulty: RunResult, clean: RunResult,
                 use_cache: bool = True, trace_path: Path | None = None) -> dict[str, Any]:
    """Fault detected after the run: restore files/KV, refund every payment, and
    rerun the whole workflow without the fault. Sent emails cannot be recalled."""
    started = time.perf_counter()
    env = faulty.env.clone()
    env.restore(Environment(workspace=task.files()).snapshot())
    for t in list(env.net_payments()):
        env.refund("restart/compensate", t.txn_id)
    rerun = run_workflow(task, llm, None, env=env, tag="restart/", use_cache=use_cache)
    if trace_path is not None:
        rerun.dump(trace_path)
    wrong = email_sigs(faulty.env) - email_sigs(clean.env)
    return {
        "restart_tokens": rerun.tokens,
        "restart_elapsed_s": time.perf_counter() - started,
        "restart_cache_hits": sum(n.cached for n in rerun.trace.nodes.values()),
        "restart_success": rerun.check["success"],
        "restart_units": len(rerun.trace.nodes),
        "restart_refunds": len(faulty.env.net_payments()),
        "restart_state_ok": _state_ok(rerun.check),
        "restart_unrecovered_emails": sum(wrong.values()),
        "restart_duplicate_emails": len(faulty.env.outbox) - sum(wrong.values()),
        "restart_safety_violation": rerun.check["safety_violation"],
    }


def always_verify(task: Task, llm: LLM, fault: FaultSpec | None,
                  use_cache: bool = True, trace_path: Path | None = None) -> dict[str, Any]:
    run = run_workflow(task, llm, fault, verify=True, use_cache=use_cache)
    if trace_path is not None:
        run.dump(trace_path)
    injected_units = [n for n, node in run.trace.nodes.items() if node.verify and node.verify["faulty"]]
    clean_units = [n for n, node in run.trace.nodes.items() if node.verify and not node.verify["faulty"]]
    return {
        "verify_detected": bool(injected_units) and all(u in run.flagged for u in injected_units),
        "verify_elapsed_s": run.elapsed_s,
        "verify_retry_tokens": run.retry_tokens,
        "verify_cache_hits": sum(n.cached for n in run.trace.nodes.values()),
        "verify_false_pos": sum(u in run.flagged for u in clean_units),
        "verify_checked": len(clean_units) + len(injected_units),
        "verify_overhead_tokens": run.verify_tokens + run.retry_tokens,
        "verify_calls": run.verify_calls,
        "verify_success": run.check["success"],
        "verify_state_ok": _state_ok(run.check),
        "verify_safety_violation": run.check["safety_violation"],
        "verify_base_tokens": run.tokens,
    }
