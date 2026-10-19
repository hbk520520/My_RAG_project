"""Offline tests for the optional SPACE sandbox backend.

The tests use an injected HTTP transport.  No SPACE deployment, network,
Docker daemon or real bearer token is used.
"""
from __future__ import annotations

import base64
import json
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from agent_runtime.context import RunContext, current_run
from agent_runtime.contracts import (Budgets, OutcomeUnknown, RunConflict, RunRequest,
                                     RunStatus)
from agent_runtime.store import RunStore
from agent_tools.sandbox.errors import (SandboxCleanupError, SandboxConfigError,
                                        SandboxOutcomeUnknown, SandboxProtocolError,
                                        SandboxUnavailableError)
from agent_tools.sandbox.executor import run_code_once
from agent_tools.sandbox.manager import create_sandbox_manager, sandbox_backend
from agent_tools.sandbox.space_client import (HttpResponse, SpaceClient, SpaceConfig,
                                               SpaceEventStream, UrllibTransport)
from agent_tools.sandbox.space_manager import SpaceSandboxManager
from agent_tools.sandbox.space_runner import SPACE_RUNNER_SOURCE

SANDBOX_ID = "sbx_00000000-0000-0000-0000-000000000001"
CREATE_OP = "op_00000000-0000-0000-0000-000000000002"
DELETE_OP = "op_00000000-0000-0000-0000-000000000003"
SESSION_ID = "proc_00000000-0000-0000-0000-000000000004"
TEMPLATE = "seed@sha256:" + "a" * 64
TOKEN = "management-secret-token"
ATTACH_TOKEN = "attachment-secret-token"


class FakeResponse:
    def __init__(self, status=200, *, payload=None, chunks=None, headers=None,
                 fail_after=None):
        self.status = status
        self.headers = headers or {}
        if payload is not None:
            chunks = [json.dumps(payload).encode("utf-8")]
        self._chunks = list(chunks or [])
        self._buffer = b""
        self._fail_after = fail_after
        self.reads = 0
        self.closed = False

    def read1(self, size):
        if self._fail_after is not None and self.reads >= self._fail_after:
            raise OSError("connection reset")
        self.reads += 1
        if not self._buffer and self._chunks:
            self._buffer = self._chunks.pop(0)
        if not self._buffer:
            return b""
        result, self._buffer = self._buffer[:size], self._buffer[size:]
        return result

    def read(self, size=-1):
        data = self._buffer + b"".join(self._chunks)
        self._buffer, self._chunks = b"", []
        return data if size is None or size < 0 else data[:size]

    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def open(self, method, url, headers, body, timeout, read_timeout=None,
             total_timeout=None):
        self.requests.append({
            "method": method, "url": url, "headers": dict(headers),
            "body": body, "timeout": timeout, "read_timeout": read_timeout,
            "total_timeout": total_timeout,
        })
        result = self.handler(method, url, dict(headers), body, timeout)
        if isinstance(result, BaseException):
            raise result
        return result


def json_response(status, payload):
    return FakeResponse(status, payload=payload,
                        headers={"content-type": "application/json"})


def space_config(**overrides):
    values = {
        "enabled": True,
        "api_url": "https://space.example.test",
        "data_plane_url": "https://data.example.test",
        "api_token": TOKEN,
        "template": TEMPLATE,
        "runtime": "fc",
        "api_contract": "m1",
        "vcpu": 2,
        "memory_mib": 768,
        "disk_mib": 2048,
        "connect_timeout": 0.1,
        "request_timeout": 0.1,
        "operation_timeout": 0.2,
        "stream_timeout": 0.2,
        "command_timeout": 5,
        "poll_interval": 0,
        "max_total_seconds": 10,
        "idle_timeout": 60,
        "max_output_bytes": 1024,
        "max_stderr_bytes": 1024,
        "max_frame_bytes": 65536,
        "max_stream_bytes": 262144,
        "max_request_bytes": 262144,
        "idempotency_window_seconds": 86400,
        "network_isolation": {
            "mode": "egress_deny_all",
            "verified_deployment": "space-test-deployment",
            "verified_at": "2026-10-19",
            "evidence": "docs/validation/space-test.txt",
        },
    }
    values.update(overrides)
    return {"backend": "space", "space": values}


def make_manager(handler, **overrides):
    transport = FakeTransport(handler)
    manager = SpaceSandboxManager.from_config(space_config(**overrides),
                                              transport=transport)
    return manager, transport


def sse_bytes(frames, *, multiline=False):
    chunks = []
    for frame in frames:
        data = json.dumps(frame)
        if multiline and frame.get("type") == "stdout":
            head, tail = data.split(",", 1)
            chunks.append(b"data: " + head.encode() + b",\n")
            chunks.append(b"data: " + tail.encode() + b"\n\n")
        else:
            chunks.append(b"data: " + data.encode() + b"\n\n")
    return b"".join(chunks)


def runner_payload(output=None, error=None, kind=None):
    return json.dumps({"output": output or "", "error": error,
                       "error_kind": kind}).encode()


def default_frames(output="42"):
    return [
        {"type": "attached", "session_id": SESSION_ID},
        {"type": "stdout",
         "data_base64": base64.b64encode(runner_payload(output)).decode()},
        {"type": "process_exited", "process_id": SESSION_ID,
         "exit_code": 0, "status": "exited"},
    ]


class Lifecycle:
    """Scripted native API for create -> ready -> command -> stream -> delete."""

    def __init__(self, *, frames=None, create_state="succeeded",
                 sandbox_state="running", exit_code=0, exit_status="exited",
                 stderr=b"", fail_stream_after=None, delete_status=202,
                 create_error=None, command_error=None, cancel=None,
                 multiline=False, chunk_size=None, extra_frames=None,
                 create_payload=None, command_sandbox_id=SANDBOX_ID,
                 exit_process_id=SESSION_ID):
        self.frames = list(frames or default_frames())
        if extra_frames:
            self.frames.extend(extra_frames)
        if stderr:
            self.frames.insert(-1, {"type": "stderr",
                                    "data_base64": base64.b64encode(stderr).decode()})
        if self.frames and self.frames[-1].get("type") == "process_exited":
            if exit_code or exit_status != "exited":
                self.frames[-1] = {"type": "process_exited",
                                   "process_id": exit_process_id,
                                   "exit_code": exit_code, "status": exit_status}
            else:
                self.frames[-1] = {**self.frames[-1],
                                   "process_id": exit_process_id}
        self.create_state = create_state
        self.sandbox_state = sandbox_state
        self.create_payload = create_payload
        self.command_sandbox_id = command_sandbox_id
        self.fail_stream_after = fail_stream_after
        self.delete_status = delete_status
        self.create_error = create_error
        self.command_error = command_error
        self.cancel = cancel
        self.multiline = multiline
        self.chunk_size = chunk_size
        self.creates = 0
        self.commands = 0
        self.deletes = 0
        self.event_streams = 0

    def __call__(self, method, url, headers, body, timeout):
        parsed = urlsplit(url)
        path = parsed.path
        if method == "POST" and path == "/v1/sandboxes":
            self.creates += 1
            if self.create_error:
                raise self.create_error
            payload = self.create_payload or {"sandbox_id": SANDBOX_ID,
                                              "operation_id": CREATE_OP}
            return json_response(202, payload)
        if method == "GET" and path == f"/v1/operations/{CREATE_OP}":
            return json_response(200, {"state": self.create_state})
        if method == "GET" and path == f"/v1/operations/{DELETE_OP}":
            return json_response(200, {"state": "succeeded"})
        if method == "GET" and path == f"/v1/sandboxes/{SANDBOX_ID}":
            return json_response(200, {"state": self.sandbox_state})
        if method == "GET" and path == f"/v1/sandboxes/{SANDBOX_ID}/processes":
            return json_response(200, {"data": [], "next_cursor": None})
        if method == "POST" and path == f"/v1/sandboxes/{SANDBOX_ID}/commands":
            self.commands += 1
            if self.command_error:
                raise self.command_error
            return json_response(201, {
                "session_id": SESSION_ID, "process_id": SESSION_ID,
                "sandbox_id": self.command_sandbox_id, "state": "open",
                "attach": {
                    "url": "wss://data.example.test/v1/sessions/" + SESSION_ID + "/attach",
                    "events_url": f"/v1/sessions/{SESSION_ID}/events",
                    "token": ATTACH_TOKEN,
                    "expires_at": "2099-01-01T00:00:00Z",
                    "token_transport": ["authorization_header"],
                },
            })
        if method == "GET" and path == f"/v1/sessions/{SESSION_ID}/events":
            self.event_streams += 1
            if self.cancel:
                self.cancel()
            raw = sse_bytes(self.frames, multiline=self.multiline)
            chunks = ([raw] if not self.chunk_size else
                      [raw[i:i + self.chunk_size]
                       for i in range(0, len(raw), self.chunk_size)])
            return FakeResponse(200, chunks=chunks,
                                headers={"content-type": "text/event-stream"},
                                fail_after=self.fail_stream_after)
        if method == "DELETE" and path == f"/v1/sandboxes/{SANDBOX_ID}":
            self.deletes += 1
            if self.delete_status != 202:
                return json_response(self.delete_status, {
                    "code": "not_found", "message": "absent",
                    "retryable": False})
            return json_response(202, {"sandbox_id": SANDBOX_ID,
                                       "operation_id": DELETE_OP})
        raise AssertionError(f"unexpected SPACE request: {method} {path}")


# ---------------------------------------------------------------------------
# Configuration / factory
# ---------------------------------------------------------------------------
def test_default_backend_remains_docker():
    assert sandbox_backend({"image": "x"}) == "docker"
    assert sandbox_backend({"backend": "docker"}) == "docker"


def test_invalid_backend_is_explicit_and_never_falls_back():
    with pytest.raises(SandboxConfigError, match="sandbox.backend"):
        sandbox_backend({"backend": "firecracker"})


def test_docker_default_never_constructs_a_space_client(monkeypatch):
    import agent_tools.sandbox.manager as manager_module
    import agent_tools.sandbox.space_client as space_client

    class DummyDocker:
        via = "docker"

    monkeypatch.setattr(manager_module, "DockerSandboxManager", DummyDocker)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("SPACE transport must not be used for the Docker backend")

    monkeypatch.setattr(space_client.UrllibTransport, "open", forbidden)
    manager = manager_module.create_sandbox_manager({"backend": "docker"})
    assert isinstance(manager, DummyDocker)


def test_space_disabled_is_an_explicit_configuration_error():
    disabled = space_config()
    disabled["space"]["enabled"] = False
    with pytest.raises(SandboxConfigError, match="enabled"):
        create_sandbox_manager(disabled)


def test_space_requires_digest_pinned_template_and_verification_record():
    mutable = space_config()
    mutable["space"]["template"] = "seed:latest"
    with pytest.raises(SandboxConfigError, match="digest-pinned"):
        SpaceConfig.from_mapping(mutable)

    truncated = space_config()
    truncated["space"]["template"] = "seed@sha256:" + "a" * 12
    assert SpaceConfig.from_mapping(truncated).template.endswith("a" * 12)

    unverified = space_config()
    unverified["space"]["network_isolation"]["evidence"] = ""
    with pytest.raises(SandboxConfigError, match="evidence"):
        SpaceConfig.from_mapping(unverified)


def test_space_config_identity_excludes_the_bearer_token():
    first = SpaceConfig.from_mapping(space_config())
    rotated = space_config()
    rotated["space"]["api_token"] = "rotated-token"
    second = SpaceConfig.from_mapping(rotated)
    assert first.identity() == second.identity()
    assert TOKEN not in json.dumps(first.identity())
    assert TOKEN not in repr(first)


def test_space_resources_are_explicit_and_not_derived_from_docker():
    config = SpaceConfig.from_mapping(space_config())
    assert (config.vcpu, config.memory_mib, config.disk_mib) == (2, 768, 2048)
    assert "cpu_quota" not in config.identity()


def test_run_pin_identity_keeps_docker_shape_and_ignores_token_rotation():
    from agent_runtime.service import sandbox_policy_identity
    legacy = {"image": "legal-sandbox:v2", "max_output_bytes": 65536}
    assert sandbox_policy_identity(legacy) == legacy
    assert sandbox_policy_identity({
        "backend": "docker", **legacy, "space": {"api_token": TOKEN},
    }) == legacy

    first = space_config()["space"]
    rotated = dict(first, api_token="rotated-token")
    identity_one = sandbox_policy_identity({"backend": "space", "space": first})
    identity_two = sandbox_policy_identity({"backend": "space", "space": rotated})
    assert identity_one == identity_two
    assert TOKEN not in json.dumps(identity_one)
    changed = dict(first, template="seed@sha256:" + "b" * 64)
    assert sandbox_policy_identity({"backend": "space", "space": changed}) != identity_one


# ---------------------------------------------------------------------------
# Normal lifecycle and output contract
# ---------------------------------------------------------------------------
def test_run_code_once_space_success_uses_via_space_and_releases_resource():
    lifecycle = Lifecycle()
    manager, transport = make_manager(lifecycle)

    result = run_code_once("result = 42", manager=manager)

    assert result == {"calc_result": "42", "error": None,
                      "session_id": None, "via": "space"}
    assert lifecycle.creates == 1 and lifecycle.commands == 1
    assert lifecycle.event_streams == 1 and lifecycle.deletes == 1
    assert manager.active_count() == 0

    create = transport.requests[0]
    assert create["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert create["headers"]["Idempotency-Key"].startswith("legal-agent-create-")
    body = json.loads(create["body"])
    assert body["template"] == TEMPLATE
    assert body["runtime"] == "fc"
    assert body["resources"] == {"vcpu": 2, "memory_mib": 768, "disk_mib": 2048}
    assert body["network_policy"] == {
        "version": "v1",
        "egress": {"default_action": "deny", "allow": [], "deny": []},
    }
    assert body["idle_action"] == "delete"
    assert TOKEN not in json.dumps(body)

    command = next(r for r in transport.requests if r["url"].endswith("/commands"))
    assert command["headers"]["Idempotency-Key"].startswith("legal-agent-command-")
    command_body = json.loads(command["body"])
    assert command_body["timeout"] == "5s"
    assert command_body["command"][:3] == ["python3", "-c", SPACE_RUNNER_SOURCE]

    stream = next(r for r in transport.requests if "/v1/sessions/" in r["url"])
    assert stream["headers"]["Authorization"] == f"Bearer {ATTACH_TOKEN}"
    assert TOKEN not in json.dumps(stream["headers"])


def test_transport_receives_separate_connect_and_read_timeouts():
    lifecycle = Lifecycle()
    manager, transport = make_manager(lifecycle, connect_timeout=5,
                                      request_timeout=30,
                                      max_total_seconds=60)
    result = run_code_once("result = 42", manager=manager)
    assert result["calc_result"] == "42"
    create = transport.requests[0]
    assert create["timeout"] == 5
    assert create["read_timeout"] == 30


def test_partial_create_response_cleans_up_a_known_sandbox():
    lifecycle = Lifecycle(create_payload={"sandbox_id": SANDBOX_ID})
    manager, _ = make_manager(lifecycle)

    with pytest.raises(OutcomeUnknown, match="create request outcome"):
        run_code_once("result = 1", manager=manager)
    assert lifecycle.commands == 0
    assert lifecycle.deletes == 1
    assert manager.active_count() == 0


def test_command_response_for_another_sandbox_is_outcome_unknown():
    lifecycle = Lifecycle(command_sandbox_id=(
        "sbx_00000000-0000-0000-0000-000000000099"))
    manager, _ = make_manager(lifecycle)

    with pytest.raises(OutcomeUnknown, match="command create request outcome"):
        run_code_once("result = 1", manager=manager)
    assert lifecycle.deletes == 1
    assert manager.active_count() == 0


def test_process_exit_for_another_process_is_outcome_unknown():
    lifecycle = Lifecycle(exit_process_id=(
        "proc_00000000-0000-0000-0000-000000000099"))
    manager, _ = make_manager(lifecycle)

    with pytest.raises(OutcomeUnknown, match="stream could not be used"):
        run_code_once("result = 1", manager=manager)
    assert lifecycle.deletes == 1
    assert manager.active_count() == 0


def test_attachment_requires_authorization_header_and_valid_expiry():
    client = SpaceClient(SpaceConfig.from_mapping(space_config()))
    attach = {
        "url": "wss://data.example.test/v1/sessions/" + SESSION_ID + "/attach",
        "events_url": f"/v1/sessions/{SESSION_ID}/events",
        "token": ATTACH_TOKEN,
        "expires_at": "2099-01-01T00:00:00Z",
        "token_transport": ["authorization_header"],
    }
    command = {"session_id": SESSION_ID, "attach": attach}
    assert client.validate_attach(command)[1] == ATTACH_TOKEN

    for broken in (
        {**attach, "token_transport": ["first_frame"]},
        {**attach, "token_transport": []},
        {**attach, "expires_at": None},
        {**attach, "expires_at": "2000-01-01T00:00:00Z"},
        {**attach, "expires_at": "not-a-date"},
    ):
        with pytest.raises(SandboxProtocolError):
            client.validate_attach({"session_id": SESSION_ID, "attach": broken})


def test_command_body_limit_fails_before_a_command_intent(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    lifecycle = Lifecycle()
    manager, _ = make_manager(lifecycle, max_request_bytes=1024)

    mark = current_run.set(context)
    try:
        result = run_code_once("result = 1", manager=manager)
    finally:
        current_run.reset(mark)

    assert result["calc_result"] is None
    assert "body size limit" in result["error"]
    assert lifecycle.commands == 0
    assert lifecycle.deletes == 1
    records = _effect_records(store, run_id)
    actions = [item["action"] for item in records[0]["result"]["_intents"]]
    assert actions == ["create_sandbox", "delete_sandbox"]


def test_epoch_change_rejects_reuse_of_an_owned_handle(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    lifecycle = Lifecycle()
    manager, _ = make_manager(lifecycle)

    mark = current_run.set(context)
    try:
        handle = manager.start_session()
    finally:
        current_run.reset(mark)

    newer = RunContext(store, run_id, context.epoch + 1, context.token)
    mark = current_run.set(newer)
    try:
        with pytest.raises(SandboxUnavailableError, match="epoch"):
            manager.execute_code(handle, "result = 1")
    finally:
        current_run.reset(mark)
        manager.destroy_session(handle)
    assert lifecycle.deletes == 1


def test_cleanup_from_another_run_is_rejected(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    lifecycle = Lifecycle()
    manager, _ = make_manager(lifecycle)

    mark = current_run.set(context)
    try:
        handle = manager.start_session()
    finally:
        current_run.reset(mark)

    other = RunContext(store, "another-run", 0, context.token)
    mark = current_run.set(other)
    try:
        with pytest.raises(SandboxCleanupError, match="different run"):
            manager.destroy_session(handle)
    finally:
        current_run.reset(mark)
    assert lifecycle.deletes == 0

    manager.destroy_session(handle)
    assert lifecycle.deletes == 1
    assert manager.active_count() == 0


def test_stdout_fallback_and_multiline_fragmented_sse():
    lifecycle = Lifecycle(frames=[
        {"type": "stdout", "data_base64": base64.b64encode(
            runner_payload("printed value")).decode()},
        {"type": "process_exited", "process_id": SESSION_ID,
         "exit_code": 0, "status": "exited"},
    ], multiline=True, chunk_size=3)
    manager, _ = make_manager(lifecycle)

    result = run_code_once("print('x')", manager=manager)
    assert result["calc_result"] == "printed value"
    assert result["via"] == "space"


def test_code_exception_is_reported_and_not_reclassified():
    lifecycle = Lifecycle(frames=[
        {"type": "stdout", "data_base64": base64.b64encode(
            runner_payload(error="ValueError: bad", kind="execution")).decode()},
        {"type": "process_exited", "process_id": SESSION_ID,
         "exit_code": 0, "status": "exited"},
    ])
    manager, _ = make_manager(lifecycle)

    result = run_code_once("raise ValueError('bad')", manager=manager)
    assert result["calc_result"] is None
    assert result["error"] == "ValueError: bad"
    assert result["via"] == "space"
    assert lifecycle.deletes == 1


def test_nonzero_exit_includes_bounded_stderr():
    lifecycle = Lifecycle(exit_code=7, stderr=b"boom")
    manager, _ = make_manager(lifecycle)

    result = run_code_once("result = 1", manager=manager)
    assert result["calc_result"] is None
    assert "code 7" in result["error"]
    assert "boom" in result["error"]
    assert lifecycle.deletes == 1


def test_create_operation_failure_is_a_definite_error():
    lifecycle = Lifecycle(create_state="failed")
    manager, _ = make_manager(lifecycle)

    result = run_code_once("result = 1", manager=manager)
    assert result["calc_result"] is None
    assert "failed" in result["error"]
    assert lifecycle.commands == 0
    assert lifecycle.deletes == 1  # best-effort cleanup of the accepted sandbox


def test_readiness_timeout_is_outcome_unknown_and_cleans_up():
    lifecycle = Lifecycle(create_state="running")
    manager, _ = make_manager(lifecycle, operation_timeout=0.1,
                              request_timeout=0.1, poll_interval=0)

    with pytest.raises(OutcomeUnknown, match="did not finish"):
        run_code_once("result = 1", manager=manager)
    assert lifecycle.commands == 0
    assert lifecycle.deletes == 1


def test_task_total_deadline_is_outcome_unknown_and_cleans_up(d_tmp):
    store, run_id, context = _claimed_run(d_tmp, seconds=4)
    lifecycle = Lifecycle(create_state="running")
    manager, _ = make_manager(lifecycle, max_total_seconds=10,
                              operation_timeout=10, request_timeout=0.1,
                              poll_interval=0.05)

    mark = current_run.set(context)
    try:
        with pytest.raises(OutcomeUnknown, match="did not finish"):
            run_code_once("result = 1", manager=manager)
    finally:
        current_run.reset(mark)
    assert lifecycle.deletes == 1
    assert manager.active_count() == 0


def test_incomplete_sse_frame_is_a_protocol_error():
    response = HttpResponse(200, {}, FakeResponse(
        200, chunks=[b'data: {"type":"stdout"']))
    stream = SpaceEventStream(response, max_frame_bytes=1024,
                              max_stream_bytes=1024, deadline=None,
                              clock=time.monotonic, check=None)
    with pytest.raises(SandboxProtocolError, match="incomplete"):
        list(stream)


def test_space_handle_is_one_shot():
    lifecycle = Lifecycle()
    manager, _ = make_manager(lifecycle)
    handle = manager.start_session()

    first = manager.execute_code(handle, "result = 1")
    assert first["output"] == "42"
    with pytest.raises(SandboxUnavailableError, match="one-shot"):
        manager.execute_code(handle, "result = 2")
    assert lifecycle.commands == 1

    manager.destroy_session(handle)
    assert lifecycle.deletes == 1


def test_two_managers_clean_only_their_own_sandbox():
    first = Lifecycle()
    second = Lifecycle()
    first_manager, _ = make_manager(first)
    second_manager, _ = make_manager(second)
    first_handle = first_manager.start_session()
    second_handle = second_manager.start_session()

    first_manager.destroy_session(first_handle)
    assert first.deletes == 1
    assert second.deletes == 0
    assert first_manager.active_count() == 0
    assert second_manager.active_count() == 1

    second_manager.destroy_session(second_handle)
    assert second.deletes == 1
    assert second_manager.active_count() == 0


def test_tool_slot_is_released_when_space_outcome_is_unknown():
    from agent_runtime.resources import resource

    lifecycle = Lifecycle(create_error=ConnectionResetError("lost"))
    manager, _ = make_manager(lifecycle)
    with pytest.raises(OutcomeUnknown):
        with resource("tool.sandbox", 1):
            run_code_once("result = 1", manager=manager)
    with resource("tool.sandbox", 1):
        pass


def test_output_limit_is_not_success():
    lifecycle = Lifecycle(frames=[
        {"type": "stdout", "data_base64": base64.b64encode(
            runner_payload("x" * 4096)).decode()},
        {"type": "process_exited", "process_id": SESSION_ID,
         "exit_code": 0, "status": "exited"},
    ])
    manager, _ = make_manager(lifecycle, max_output_bytes=64)

    result = run_code_once("result = 'x' * 4096", manager=manager)
    assert result["calc_result"] is None
    assert "limit" in result["error"]
    assert lifecycle.deletes == 1


@pytest.mark.parametrize("frames", [
    [{"type": "stdout", "data_base64": base64.b64encode(runner_payload("42")).decode()}],
    [{"type": "error", "code": "backend_stream_error", "message": "boom",
      "retryable": False}],
    [{"type": "resync_required", "reason": "cursor_replay_unavailable"}],
])
def test_missing_exit_or_stream_error_is_outcome_unknown(frames):
    lifecycle = Lifecycle(frames=frames)
    manager, _ = make_manager(lifecycle)

    with pytest.raises(OutcomeUnknown):
        run_code_once("result = 1", manager=manager)
    assert lifecycle.deletes == 1


def test_corrupt_frame_and_stream_interruption_are_outcome_unknown():
    corrupt = Lifecycle(frames=[{"type": "stdout", "data_base64": "not-base64!"},
                                {"type": "process_exited", "process_id": SESSION_ID,
                                 "exit_code": 0, "status": "exited"}])
    manager, _ = make_manager(corrupt)
    with pytest.raises(OutcomeUnknown):
        run_code_once("result = 1", manager=manager)

    interrupted = Lifecycle(frames=[{
        "type": "stdout",
        "data_base64": base64.b64encode(runner_payload("42")).decode()}],
        fail_stream_after=1)
    manager, _ = make_manager(interrupted)
    with pytest.raises(OutcomeUnknown):
        run_code_once("result = 1", manager=manager)


def test_cleanup_failure_does_not_overwrite_a_successful_calculation(caplog):
    lifecycle = Lifecycle(delete_status=404)
    manager, _ = make_manager(lifecycle)

    result = run_code_once("result = 42", manager=manager)
    assert result["calc_result"] == "42"
    assert result["error"] is None
    assert "cleanup_error" in result
    assert "not proven" in result["cleanup_error"]
    assert manager.active_count() == 1
    assert any("清理失败" in record.message for record in caplog.records)


def test_unknown_cleanup_handle_does_not_delete_another_task():
    lifecycle = Lifecycle()
    manager, transport = make_manager(lifecycle)
    with pytest.raises(SandboxCleanupError, match="cannot be confirmed"):
        manager.destroy_session("space:deadbeef:1:" + SANDBOX_ID)
    assert lifecycle.deletes == 0
    assert not any(r["method"] == "DELETE" for r in transport.requests)


# ---------------------------------------------------------------------------
# Real SQLite ledger: side-effect intent, recovery and cancellation
# ---------------------------------------------------------------------------
@pytest.fixture
def d_tmp():
    root = (Path(__file__).resolve().parents[1] / "outputs" / "test-space"
            / uuid.uuid4().hex)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _claimed_run(tmp_path, *, seconds=30):
    store = RunStore(tmp_path / "runs.sqlite")
    run = store.create(RunRequest(query="计算赔偿", allow_code=True,
                                  budgets=Budgets(seconds=seconds)))
    token = store.claim(run["run_id"], max_running=1)
    assert token
    context = RunContext(store, run["run_id"], run["epoch"], token)
    return store, run["run_id"], context


def _effect_records(store, run_id):
    with store.connection() as db:
        rows = [dict(row) for row in db.execute(
            "SELECT * FROM effects WHERE run_id=? ORDER BY rowid", (run_id,))]
    for row in rows:
        if row["result"]:
            row["result"] = json.loads(row["result"])
    return rows


def test_create_response_loss_persists_original_intent_and_blocks_retry(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    lifecycle = Lifecycle(create_error=ConnectionResetError("response lost"))
    manager, transport = make_manager(lifecycle)

    mark = current_run.set(context)
    try:
        with pytest.raises(OutcomeUnknown, match="outcome is unknown"):
            run_code_once("result = 1", manager=manager)
        assert lifecycle.creates == 1
        assert lifecycle.commands == 0
        records = _effect_records(store, run_id)
        assert len(records) == 1
        assert records[0]["read_only"] == 0
        intents = records[0]["result"]["_intents"]
        create = intents[0]
        assert create["method"] == "POST"
        assert create["path"] == "/v1/sandboxes"
        assert create["idempotency_key"].startswith("legal-agent-create-")
        assert create["body"]["template"] == TEMPLATE
        assert TOKEN not in json.dumps(records[0]["result"])
        assert ATTACH_TOKEN not in json.dumps(records[0]["result"])

        # Reconciliation returns the original request verbatim; it does not
        # send it.  A caller that chooses to retry must reuse this key/body.
        recovered = manager.reconcile_intent(create)
        assert recovered["idempotency_key"] == create["idempotency_key"]
        assert recovered["method"] == create["method"]
        assert recovered["path"] == create["path"]
        assert recovered["body"] == create["body"]

        # The same tool call must not be replayed automatically.
        with pytest.raises(OutcomeUnknown, match="Interrupted write tool"):
            run_code_once("result = 1", manager=manager)
        assert lifecycle.creates == 1
    finally:
        current_run.reset(mark)


def test_effect_request_timestamp_and_request_identity_are_immutable(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    call_id = "sandbox-call"
    store.begin_effect(run_id, 0, context.token, call_id,
                       "sandbox.execute", "hash", False)
    base = {
        "format": 1,
        "backend": "space",
        "api_contract": "m1",
        "upstream_commit": "commit",
        "deployment": "deployment",
        "api_url": "https://space.example.test",
        "data_plane_url": "https://data.example.test",
        "template": TEMPLATE,
        "runner_version": "runner/1",
        "action": "create_sandbox",
        "method": "POST",
        "path": "/v1/sandboxes",
        "body": {"template": TEMPLATE},
        "idempotency_key": "key-1",
        "created_at": 100.0,
        "run_id": run_id,
        "epoch": 0,
        "task_id": "",
        "revision": 0,
    }
    store.record_effect_request(run_id, 0, context.token, call_id, base)
    store.record_effect_request(run_id, 0, context.token, call_id, {
        **base, "created_at": 200.0, "sandbox_id": SANDBOX_ID,
        "operation_id": CREATE_OP,
    })
    intent = store.effect_intents(call_id)[0]
    assert intent["created_at"] == 100.0
    assert intent["sandbox_id"] == SANDBOX_ID
    assert intent["operation_id"] == CREATE_OP

    with pytest.raises(RunConflict, match="identity changed"):
        store.record_effect_request(run_id, 0, context.token, call_id, {
            **base, "method": "PUT",
        })


def test_command_response_loss_keeps_create_and_command_intents(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    lifecycle = Lifecycle(command_error=ConnectionResetError("response lost"))
    manager, _ = make_manager(lifecycle)

    mark = current_run.set(context)
    try:
        with pytest.raises(OutcomeUnknown, match="command create request outcome"):
            run_code_once("result = 1", manager=manager)
        records = _effect_records(store, run_id)
        intents = records[0]["result"]["_intents"]
        actions = [item["action"] for item in intents]
        assert actions == ["create_sandbox", "create_command", "delete_sandbox"]
        command = intents[1]
        assert command["path"] == f"/v1/sandboxes/{SANDBOX_ID}/commands"
        assert command["idempotency_key"].startswith("legal-agent-command-")
        assert command["body"]["timeout"] == "5s"
        delete = intents[2]
        assert delete["method"] == "DELETE"
        assert delete["path"] == f"/v1/sandboxes/{SANDBOX_ID}"
        assert delete["idempotency_key"].startswith("legal-agent-delete-")
        assert TOKEN not in json.dumps(records[0]["result"])
        assert ATTACH_TOKEN not in json.dumps(records[0]["result"])
        assert lifecycle.deletes == 1
    finally:
        current_run.reset(mark)


def test_successful_run_keeps_intents_without_polluting_the_tool_result(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)
    lifecycle = Lifecycle()
    manager, _ = make_manager(lifecycle)

    mark = current_run.set(context)
    try:
        result = run_code_once("result = 42", manager=manager)
    finally:
        current_run.reset(mark)

    assert result == {"calc_result": "42", "error": None,
                      "session_id": None, "via": "space"}
    records = _effect_records(store, run_id)
    assert len(records) == 1
    intents = records[0]["result"]["_intents"]
    assert [item["action"] for item in intents] == [
        "create_sandbox", "create_command", "delete_sandbox"]
    assert intents[2]["operation_id"] == DELETE_OP
    assert all("token" not in item for item in intents)
    assert TOKEN not in json.dumps(records[0]["result"])
    assert ATTACH_TOKEN not in json.dumps(records[0]["result"])

    # A completed effect is still returned through the original result shape;
    # private ledger metadata must not leak into a cached tool result.
    cached, value = store.begin_effect(
        run_id, 0, context.token, records[0]["call_id"], records[0]["tool"],
        records[0]["input_hash"], False)
    assert cached is True
    assert value == result


def test_cancellation_rejects_late_result_but_still_cleans_owned_sandbox(d_tmp):
    store, run_id, context = _claimed_run(d_tmp)

    def cancel():
        store.cancel(run_id)

    lifecycle = Lifecycle(cancel=cancel)
    manager, _ = make_manager(lifecycle)

    mark = current_run.set(context)
    try:
        from agent_runtime.contracts import RunStopped
        with pytest.raises(RunStopped):
            run_code_once("result = 1", manager=manager)
    finally:
        current_run.reset(mark)

    assert store.get(run_id)["status"] == RunStatus.CANCELLED
    records = _effect_records(store, run_id)
    assert records[0]["status"] == "started"  # no successful commit
    assert lifecycle.deletes == 1             # independent cleanup still ran
    assert manager.active_count() == 0


def test_reconcile_intent_rejects_expired_window_and_other_deployment():
    manager, _ = make_manager(Lifecycle())
    intent = manager._intent_base("create_sandbox", "POST", "/v1/sandboxes",
                                  {"template": TEMPLATE}, "key-1")
    assert manager.reconcile_intent(intent)["idempotency_key"] == "key-1"

    stale = dict(intent, created_at=time.time() - 90000)
    with pytest.raises(SandboxOutcomeUnknown, match="window expired"):
        manager.reconcile_intent(stale)

    other = dict(intent, deployment="another-deployment")
    with pytest.raises(SandboxOutcomeUnknown, match="does not belong"):
        manager.reconcile_intent(other)

    moved = dict(intent, api_url="https://other.example.test")
    with pytest.raises(SandboxOutcomeUnknown, match="api_url"):
        manager.reconcile_intent(moved)

    changed_path = dict(intent, path="/v1/sandboxes/not-the-create-path")
    with pytest.raises(SandboxOutcomeUnknown, match="path changed"):
        manager.reconcile_intent(changed_path)

    unknown_action = dict(intent, action="pause_sandbox")
    with pytest.raises(SandboxOutcomeUnknown, match="unknown action"):
        manager.reconcile_intent(unknown_action)


def test_rotation_does_not_change_identity_used_for_reconciliation():
    first, _ = make_manager(Lifecycle())
    intent = first._intent_base("create_sandbox", "POST", "/v1/sandboxes",
                                {"template": TEMPLATE}, "key-1")
    rotated = space_config()
    rotated["space"]["api_token"] = "rotated-token"
    second = SpaceSandboxManager.from_config(rotated,
                                             transport=FakeTransport(Lifecycle()))
    assert second.reconcile_intent(intent)["idempotency_key"] == "key-1"


def test_runner_source_executes_locally_with_the_same_result_contract():
    import subprocess
    import sys
    encoded = base64.b64encode(b"result = 6 * 7").decode()
    completed = subprocess.run(
        [sys.executable, "-c", SPACE_RUNNER_SOURCE, encoded, "1024"],
        capture_output=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr.decode()
    payload = json.loads(completed.stdout)
    assert payload == {"output": "42", "error": None, "error_kind": None}


def test_runner_source_reports_errors_and_caps_output():
    import subprocess
    import sys
    encoded = base64.b64encode(b"raise ValueError('bad')").decode()
    completed = subprocess.run(
        [sys.executable, "-c", SPACE_RUNNER_SOURCE, encoded, "1024"],
        capture_output=True, check=False, timeout=30)
    payload = json.loads(completed.stdout)
    assert payload["output"] == ""
    assert "ValueError" in payload["error"]

    encoded = base64.b64encode(b"result = 'x' * 100").decode()
    completed = subprocess.run(
        [sys.executable, "-c", SPACE_RUNNER_SOURCE, encoded, "8"],
        capture_output=True, check=False, timeout=30)
    payload = json.loads(completed.stdout)
    assert payload["error"] and "limit" in payload["error"].lower()


def test_stdlib_transport_reads_json_and_does_not_follow_redirects():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/json":
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(302)
                self.send_header("Location", "https://evil.example.test/v1/json")
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        config = space_config()
        config["space"]["api_url"] = f"http://127.0.0.1:{port}"
        config["space"]["data_plane_url"] = f"http://127.0.0.1:{port}"
        client = SpaceClient(SpaceConfig.from_mapping(config),
                             transport=UrllibTransport())
        assert client.request_json("GET", "/v1/json") == {"ok": True}
        from agent_tools.sandbox.errors import SandboxRemoteError
        with pytest.raises(SandboxRemoteError) as error:
            client.request_json("GET", "/v1/redirect")
        assert error.value.status == 302
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
