"""Simulated tool environment shared by all agents.

Four kinds of state, matching the three recovery classes in the proposal:
  * workspace (files) and key-value store  -> restorable (snapshot / restore)
  * payment ledger                          -> compensable (refund)
  * email outbox                            -> irreversible

Every tool call is attributed to the step that made it, so the trace can link a
reader of a resource to the step that last wrote it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

TOOLS = ("pay", "kv_set", "write_file", "send_email")


@dataclass
class Access:
    step: str
    kind: str  # "read" | "write"
    resource: str  # e.g. "kv:order/total", "file:orders/summary.md", "ledger:txn-1"
    writer: str | None = None  # for reads: the step that last wrote the resource


@dataclass
class Transaction:
    txn_id: str
    step: str
    account: str
    amount: float
    memo: str
    refund_of: str | None = None


@dataclass
class Email:
    msg_id: str
    step: str
    to: str
    subject: str
    body: str


@dataclass
class Environment:
    workspace: dict[str, str] = field(default_factory=dict)
    kv: dict[str, str] = field(default_factory=dict)
    ledger: list[Transaction] = field(default_factory=list)
    outbox: list[Email] = field(default_factory=list)
    accesses: list[Access] = field(default_factory=list)
    last_writer: dict[str, str] = field(default_factory=dict)

    # ---------------- snapshot / restore (restorable state only) ----------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "workspace": dict(self.workspace),
            "kv": dict(self.kv),
            "last_writer": dict(self.last_writer),
        }

    def restore(self, snap: dict[str, Any], keys: set[str] | None = None) -> None:
        """Restore workspace/KV from a snapshot. With `keys`, restore only those
        resources (e.g. {"kv:order/total", "file:orders/summary.md"})."""
        if keys is None:
            keys = ({f"kv:{k}" for k in self.kv | snap["kv"]}
                    | {f"file:{k}" for k in self.workspace | snap["workspace"]})
        for res in keys:
            kind, _, name = res.partition(":")
            store = {"kv": self.kv, "file": self.workspace}.get(kind)
            if store is None:
                continue
            old = snap[{"kv": "kv", "file": "workspace"}[kind]]
            if name in old:
                store[name] = old[name]
            else:
                store.pop(name, None)
            if res in snap["last_writer"]:
                self.last_writer[res] = snap["last_writer"][res]
            else:
                self.last_writer.pop(res, None)

    # ---------------- reads ----------------
    def read(self, step: str, resource: str) -> str | None:
        kind, _, name = resource.partition(":")
        if kind == "kv":
            value = self.kv.get(name)
        elif kind == "file":
            value = self.workspace.get(name)
        elif kind == "ledger":
            value = "\n".join(
                f"{t.txn_id}: {t.amount:.2f} -> {t.account} ({t.memo})" for t in self.net_payments()
            )
        else:
            value = None
        self.accesses.append(Access(step, "read", resource, self.last_writer.get(resource)))
        return value

    # ---------------- tool calls ----------------
    def call(self, step: str, tool: str, target: str, amount: float = 0.0,
             content: str = "", subject: str = "") -> str:
        if tool == "pay":
            return self.pay(step, target, amount, content)
        if tool == "kv_set":
            return self._write(step, f"kv:{target}", content)
        if tool == "write_file":
            return self._write(step, f"file:{target}", content)
        if tool == "send_email":
            return self.send_email(step, target, subject, content)
        raise ValueError(f"unknown tool {tool!r}")

    def _write(self, step: str, resource: str, content: str) -> str:
        kind, _, name = resource.partition(":")
        (self.kv if kind == "kv" else self.workspace)[name] = content
        self.last_writer[resource] = step
        self.accesses.append(Access(step, "write", resource))
        return f"ok: wrote {resource}"

    def pay(self, step: str, account: str, amount: float, memo: str) -> str:
        txn = Transaction(f"txn-{len(self.ledger) + 1}", step, account, round(float(amount), 2), memo)
        self.ledger.append(txn)
        self.last_writer[f"ledger:{txn.txn_id}"] = step
        self.last_writer["ledger"] = step
        self.accesses.append(Access(step, "write", f"ledger:{txn.txn_id}"))
        return f"ok: {txn.txn_id}"

    def refund(self, step: str, txn_id: str) -> str:
        orig = next(t for t in self.ledger if t.txn_id == txn_id)
        txn = Transaction(f"txn-{len(self.ledger) + 1}", step, orig.account, -orig.amount,
                          f"refund of {txn_id}", refund_of=txn_id)
        self.ledger.append(txn)
        self.accesses.append(Access(step, "write", f"ledger:{txn.txn_id}"))
        return f"ok: {txn.txn_id}"

    def send_email(self, step: str, to: str, subject: str, body: str) -> str:
        msg = Email(f"msg-{len(self.outbox) + 1}", step, to.strip(), subject, body)
        self.outbox.append(msg)
        self.accesses.append(Access(step, "write", f"outbox:{msg.msg_id}"))
        return f"ok: {msg.msg_id}"

    # ---------------- views ----------------
    def net_payments(self) -> list[Transaction]:
        refunded = {t.refund_of for t in self.ledger if t.refund_of}
        return [t for t in self.ledger if t.refund_of is None and t.txn_id not in refunded]

    def payments_by_step(self, step: str) -> list[Transaction]:
        return [t for t in self.net_payments() if t.step == step]

    def state_view(self) -> dict[str, Any]:
        """Comparable view of the final state (used by checkers and diffs)."""
        return {
            "kv": dict(self.kv),
            "files": {k: v for k, v in self.workspace.items() if not k.startswith("sources/")},
            "payments": sorted((t.account, t.amount) for t in self.net_payments()),
            "emails": [(e.to, e.subject, e.body) for e in self.outbox],
        }

    def clone(self) -> "Environment":
        return copy.deepcopy(self)
