import sqlite3
import time

import pytest

from agent_runtime.context import RunContext
from agent_runtime.contracts import BudgetExceeded, OutcomeUnknown, RunConflict, RunRequest, RunStopped
from agent_runtime.store import RunStore


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs.sqlite")


def active(store, **kwargs):
    row = store.create(RunRequest(query="task", **kwargs))
    token = store.claim(row["run_id"])
    return row["run_id"], token


def test_durable_namespaces_and_single_executor_across_instances(store):
    a = store.create(RunRequest(query="first", project_id="a"))
    b = store.create(RunRequest(query="second", project_id="b"))
    token = store.claim(a["run_id"])
    reopened = RunStore(store.path)
    assert reopened.claim(a["run_id"]) is None
    assert reopened.claim(b["run_id"]) is None
    assert [r["run_id"] for r in reopened.list_runs("a")] == [a["run_id"]]
    assert store.finish(a["run_id"], 0, token, "succeeded", result={"answer": "ok"})
    assert reopened.claim(b["run_id"])


def test_cancel_fences_old_executor_and_late_result(store):
    run, token = active(store)
    store.cancel(run)
    with pytest.raises(RunStopped):
        store.heartbeat(run, 0, token)
    assert not store.finish(run, 0, token, "succeeded", result={"wrong": True})
    assert store.get(run)["status"] == "cancelled"


def test_expired_lease_changes_owner_and_old_owner_cannot_write(store):
    run, old = active(store)
    with store.connection() as db:
        db.execute("UPDATE runs SET lease_until=0 WHERE run_id=?", (run,))
    new = store.claim(run)
    assert new != old
    with pytest.raises(RunStopped):
        store.consume(run, 0, old, node_steps=1)
    store.consume(run, 0, new, node_steps=1)


def test_deadline_and_total_call_budget_are_explicit(store):
    run, token = active(store, budgets={"model_calls": 1})
    store.consume(run, 0, token, model_calls=1)
    with pytest.raises(BudgetExceeded, match="model_calls"):
        store.consume(run, 0, token, model_calls=1)
    with store.connection() as db:
        db.execute("UPDATE runs SET deadline=0 WHERE run_id=?", (run,))
    with pytest.raises(BudgetExceeded, match="deadline"):
        store.heartbeat(run, 0, token)


def test_tool_result_survives_restart_and_nested_calls_have_stable_identity(store):
    run, token = active(store)
    ctx = RunContext(store, run, 0, token)
    ctx.begin_node("read", {})
    calls = []
    def outer():
        return ctx.tool("inner", {}, lambda: calls.append("inner") or "value")
    assert ctx.tool("outer", {}, outer) == "value"
    assert ctx.tool("after", {}, lambda: calls.append("after") or "next") == "next"
    ctx.begin_node("read", {})
    assert ctx.tool("outer", {}, outer) == "value"
    assert ctx.tool("after", {}, lambda: pytest.fail("Repeated completed effect")) == "next"
    assert calls == ["inner", "after"]


def test_interrupted_write_is_never_blindly_retried(store):
    run, token = active(store)
    store.begin_effect(run, 0, token, "write", "external.write", "hash", False)
    with pytest.raises(OutcomeUnknown):
        store.begin_effect(run, 0, token, "write", "external.write", "hash", False)
    store.finish(run, 0, token, "failed")
    with pytest.raises(OutcomeUnknown):
        store.rewind(run, "checkpoint")


def test_outbox_submission_is_durable_and_stale_epoch_is_rejected(store):
    row = store.create(RunRequest(query="task"), queued_topic="pending")
    assert len(RunStore(store.path).pending_outbox()) == 1
    store.cancel(row["run_id"])
    store.rewind(row["run_id"], "checkpoint")
    assert store.claim(row["run_id"], epoch=0) is None
    assert store.claim(row["run_id"], epoch=1)


def test_resume_only_accepts_waiting_runs_and_preserves_original_deadline(store):
    run, token = active(store)
    deadline = store.get(run)["deadline"]
    with pytest.raises(RunConflict):
        store.requeue(run, answer="facts")
    store.finish(run, 0, token, "waiting_input")
    resumed = store.requeue(run, answer="facts")
    assert resumed["deadline"] == deadline and resumed["resume"] == "facts"


def test_sqlite_backup_restores_records_and_ledger(store, tmp_path):
    run, token = active(store)
    store.begin_effect(run, 0, token, "lookup", "lookup", "hash", True)
    store.end_effect(run, 0, token, "lookup", {"value": "kept"})
    store.finish(run, 0, token, "succeeded", result={"answer": "kept"})
    path = store.backup(tmp_path / "backup.sqlite")
    assert RunStore(path).get(run)["result"] == {"answer": "kept"}
    with pytest.raises(ValueError):
        store.backup(path)


def test_cancelled_executor_cannot_append_late_graph_checkpoint(store, tmp_path):
    from agent_runtime.checkpointer import FencedSqliteSaver
    run, token = active(store)
    connection = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    try:
        saver = FencedSqliteSaver(connection, RunContext(store, run, 0, token))
        store.cancel(run)
        with pytest.raises(RunStopped):
            saver.put({}, {}, {}, {})
    finally:
        connection.close()


def test_known_tool_failure_is_recorded_and_read_only_retry_can_succeed(store):
    run, token = active(store)
    ctx = RunContext(store, run, 0, token)
    ctx.begin_node("Read", {})
    def fail():
        raise ConnectionError("temporary")
    with pytest.raises(ConnectionError):
        ctx.tool("read", {}, fail)
    assert ctx.tool("read", {}, lambda: "recovered") == "recovered"
    with store.connection() as db:
        row = db.execute("SELECT status,attempts FROM effects").fetchone()
        assert (row[0], row[1]) == ("done", 2)


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_requests_rejected(query):
    with pytest.raises(ValueError):
        RunRequest(query=query)
