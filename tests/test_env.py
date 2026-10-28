from srmas.env.environment import Environment


def test_snapshot_restore_full_and_selective():
    env = Environment(workspace={"sources/a.md": "x"})
    snap = env.snapshot()
    env.call("s1", "kv_set", "k1", content="v1")
    env.call("s2", "kv_set", "k2", content="v2")
    env.call("s3", "write_file", "out.md", content="hello")

    partial = env.clone()
    partial.restore(snap, keys={"kv:k1"})
    assert "k1" not in partial.kv and partial.kv["k2"] == "v2"
    assert partial.workspace["out.md"] == "hello"

    env.restore(snap)
    assert env.kv == {} and "out.md" not in env.workspace
    assert env.workspace["sources/a.md"] == "x"
    assert "kv:k1" not in env.last_writer


def test_refund_compensates_payment():
    env = Environment()
    env.call("s1", "pay", "ACCT-A-1", amount=10.5, content="x")
    env.call("s2", "pay", "ACCT-B-2", amount=3, content="y")
    env.refund("fix", "txn-1")
    assert [(t.account, t.amount) for t in env.net_payments()] == [("ACCT-B-2", 3.0)]
    assert len(env.ledger) == 3  # the ledger keeps the history


def test_outbox_is_not_restorable():
    env = Environment()
    snap = env.snapshot()
    env.call("s1", "send_email", "a@b.example", content="body", subject="s")
    env.restore(snap)
    assert len(env.outbox) == 1


def test_reads_record_last_writer():
    env = Environment()
    env.call("exec:s3", "kv_set", "order/total", content="12.00")
    assert env.read("exec:s5", "kv:order/total") == "12.00"
    assert env.accesses[-1].writer == "exec:s3"
