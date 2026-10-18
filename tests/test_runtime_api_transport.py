from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent_runtime.api import create_app
from agent_runtime.service import AgentService
from agent_runtime.store import RunStore
from agent_runtime.transport import KafkaTransport
from tests.test_agent_runtime import minimal_graph


def test_task_api_authentication_lifecycle_and_events(tmp_path):
    service = AgentService(RunStore(tmp_path / "runs.sqlite"), workflow_factory=minimal_graph)
    with TestClient(create_app(service, token="test-secret", execute_local=False)) as client:
        assert client.post("/runs", json={"query": "task"}).status_code == 401
        assert client.get("/ready").json() == {"ready": True}
        headers = {"Authorization": "Bearer test-secret"}
        response = client.post("/runs", headers=headers, json={"query": "task"})
        assert response.status_code == 202
        run = response.json()["run_id"]
        assert client.get("/runs/absent", headers=headers).status_code == 404
        assert service.execute(run)["status"] == "succeeded"
        row = client.get("/runs/" + run, headers=headers).json()
        assert row["status"] == "succeeded" and "lease_token" not in row
        assert client.get(f"/runs/{run}/events", headers=headers).json()
        assert client.post(f"/runs/{run}/resume", headers=headers, json={"answer": "facts"}).status_code == 409


def test_api_requires_explicit_secret(tmp_path, monkeypatch):
    from config_loader import cfg
    monkeypatch.setitem(cfg._data["runtime"], "api_token", "")
    with pytest.raises(ValueError, match="AGENT_API_TOKEN"):
        create_app(AgentService(RunStore(tmp_path / "runs.sqlite")))


def test_outbox_failure_is_not_acknowledged_and_duplicate_delivery_is_safe(tmp_path):
    service = AgentService(RunStore(tmp_path / "runs.sqlite"), workflow_factory=minimal_graph)
    row = service.submit({"query": "task"}, transport="kafka")
    pending = service.store.pending_outbox()[0]
    def fail(timeout):
        raise ConnectionError("broker unavailable")
    producer = SimpleNamespace(send=lambda *a, **kw: SimpleNamespace(get=fail))
    bus = KafkaTransport(service, producer)
    with pytest.raises(ConnectionError):
        bus.dispatch()
    assert len(service.store.pending_outbox()) == 1
    assert bus.receive(pending["payload"])["status"] == "succeeded"
    assert bus.receive(pending["payload"])["status"] == "succeeded"
    producer.send = lambda *a, **kw: SimpleNamespace(get=lambda timeout: None)
    assert bus.dispatch() == 2
    assert not service.store.pending_outbox()


def test_dlq_failure_prevents_offset_commit(monkeypatch):
    import observability
    from asynchronization.kafka_utils import quarantine_message
    monkeypatch.setattr(observability, "push_to_dlq", lambda *args: False)
    with pytest.raises(RuntimeError, match="must not be committed"):
        quarantine_message(SimpleNamespace(topic="pending", key="k", value={}), ValueError("bad"))
