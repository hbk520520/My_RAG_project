"""Optional SPACE sandbox backend.

Docker remains the default.  This backend is one-shot only: one sandbox per
``run_code_once`` call, one command, one event stream, then deletion.  It does
not share a guest workspace between tasks and does not expose pause, resume,
snapshot, warm-pool or file-transfer features.

The upstream contract is pinned in ``space_client``.  Every mutation is recorded
in the existing effects ledger before the network call, including the original
idempotency key, method, path and body.  Bearer and attachment tokens are never
recorded.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from typing import Any, Mapping

from agent_runtime.context import current_run, record_effect_intent

from .errors import (SandboxCleanupError, SandboxOutcomeUnknown,
                     SandboxProtocolError, SandboxRemoteError, SandboxTimeoutError,
                     SandboxUnavailableError)
from .space_client import (API_CONTRACT, UPSTREAM_COMMIT, SpaceClient, SpaceConfig,
                           UrllibTransport)
from .space_runner import RUNNER_VERSION, SPACE_RUNNER_SOURCE

import logging

logger = logging.getLogger("SandboxSpace")

CLEANUP_SECONDS = 5.0
_HANDLE_RE = re.compile(r"^space:([A-Za-z0-9_-]+):(\d+):(sbx_[A-Za-z0-9_-]+)$")
_TERMINAL_BAD_OPERATION = {"failed", "canceled", "expired"}
_TERMINAL_BAD_SANDBOX = {"failed", "deleted", "deleting", "stopped", "paused",
                         "pausing", "suspending", "suspended", "stopping",
                         "restoring", "migrating"}
_RETRYABLE_READINESS = {404, 409, 412, 429, 500, 503}


class _Clock:
    monotonic = staticmethod(time.monotonic)
    wall_time = staticmethod(time.time)
    sleep = staticmethod(time.sleep)


class SpaceSandboxManager:
    """SPACE implementation of the Docker manager surface.

    ``start_session`` returns an ownership-bound handle, not a raw sandbox ID.
    ``execute_code`` only accepts a handle created by this manager instance.
    """

    via = "space"
    one_shot = True
    side_effecting = True

    def __init__(self, config: SpaceConfig, *, transport=None, clock=None):
        self.config = config
        self.client = SpaceClient(config, transport or UrllibTransport())
        self.clock = clock or _Clock()
        self._active: dict[str, dict[str, Any]] = {}
        self._cleaned: set[str] = set()
        self._lock = threading.Lock()
        self._last_check = 0.0

    @classmethod
    def from_config(cls, sandbox: Mapping[str, Any] | None, *, transport=None,
                    clock=None) -> "SpaceSandboxManager":
        return cls(SpaceConfig.from_mapping(sandbox), transport=transport, clock=clock)

    # ------------------------------------------------------------------
    # Context, deadlines and the execution ledger
    # ------------------------------------------------------------------
    def _check_context(self, *, force: bool = False) -> None:
        now = self.clock.monotonic()
        if not force and now - self._last_check < 0.25:
            return
        self._last_check = now
        context = current_run.get()
        if context is not None:
            context.check()

    def _deadline(self) -> float:
        budget = self.config.max_total_seconds
        context = current_run.get()
        if context is not None:
            row = context.store.get(context.run_id)
            remaining = row["deadline"] - self.clock.wall_time() - 2.0
            budget = min(budget, max(0.0, remaining))
        if budget <= 0:
            raise SandboxTimeoutError("Task deadline leaves no time for the SPACE sandbox")
        return self.clock.monotonic() + budget

    def _remaining(self, deadline: float, *, check_context: bool = True) -> float:
        if check_context:
            self._check_context(force=True)
        remaining = deadline - self.clock.monotonic()
        if remaining <= 0.05:
            raise SandboxTimeoutError("SPACE sandbox deadline expired")
        return remaining

    def _bounded(self, deadline: float, limit: float, *,
                 check_context: bool = True) -> float:
        remaining = self._remaining(deadline, check_context=check_context)
        if remaining <= 0.05:
            raise SandboxTimeoutError("SPACE request deadline expired")
        return min(limit, remaining)

    def _sleep(self, deadline: float, seconds: float, *,
               check_context: bool = True) -> None:
        if check_context:
            self._check_context(force=True)
        if seconds <= 0:
            # A zero poll interval means "yield and re-check", not "deadline
            # expired".  The next loop iteration still checks the deadline.
            self.clock.sleep(0)
            return
        delay = min(seconds, deadline - self.clock.monotonic())
        if delay <= 0:
            raise SandboxTimeoutError("SPACE sandbox deadline expired")
        self.clock.sleep(delay)

    def _phase_deadline(self, deadline: float, limit: float) -> float:
        """Bound one asynchronous phase while preserving the task deadline."""
        return min(deadline, self.clock.monotonic() + limit)

    def _intent_base(self, action: str, method: str, path: str,
                     body: Mapping[str, Any], key: str) -> dict[str, Any]:
        context = current_run.get()
        record = {
            "format": 1,
            "backend": "space",
            "api_contract": API_CONTRACT,
            "upstream_commit": UPSTREAM_COMMIT,
            "deployment": self.config.verified_deployment,
            "api_url": self.config.api_url,
            "data_plane_url": self.config.data_plane_url,
            "template": self.config.template,
            "runner_version": RUNNER_VERSION,
            "action": action,
            "method": method,
            "path": path,
            "body": dict(body),
            "idempotency_key": key,
            "created_at": self.clock.wall_time(),
        }
        if context is not None:
            record.update({
                "run_id": context.run_id,
                "epoch": context.epoch,
                "task_id": context.task_id,
                "revision": context.task_revision,
            })
        return record

    def _persist(self, action: str, method: str, path: str,
                 body: Mapping[str, Any], key: str, **extra) -> None:
        record = self._intent_base(action, method, path, body, key)
        record.update(extra)
        # Raises before the network call if the run is no longer owned.
        record_effect_intent(record)

    def intent_is_replayable(self, intent: Mapping[str, Any]) -> bool:
        created = intent.get("created_at") if isinstance(intent, Mapping) else None
        if isinstance(created, bool) or not isinstance(created, (int, float)):
            return False
        age = self.clock.wall_time() - created
        return 0 <= age <= self.config.idempotency_window_seconds

    def reconcile_intent(self, intent: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a persisted mutation before any manual replay.

        Replay is never automatic.  This method only proves that the original
        key/body belong to this backend deployment/template and are still
        inside the documented 24-hour idempotency window.
        """
        if not isinstance(intent, Mapping):
            raise SandboxOutcomeUnknown("Persisted SPACE intent is not an object")
        expected = {
            "backend": "space",
            "api_contract": API_CONTRACT,
            "upstream_commit": UPSTREAM_COMMIT,
            "deployment": self.config.verified_deployment,
            "api_url": self.config.api_url,
            "data_plane_url": self.config.data_plane_url,
            "template": self.config.template,
            "runner_version": RUNNER_VERSION,
        }
        for key, value in expected.items():
            if intent.get(key) != value:
                raise SandboxOutcomeUnknown(
                    f"Persisted SPACE intent does not belong to this deployment "
                    f"({key} differs)")
        if not self.intent_is_replayable(intent):
            raise SandboxOutcomeUnknown(
                "SPACE idempotency window expired; do not replay the recorded request")
        for key in ("idempotency_key", "method", "path", "body"):
            if key not in intent:
                raise SandboxOutcomeUnknown(
                    f"Persisted SPACE intent is missing {key}")
        action = intent.get("action")
        method = intent.get("method")
        path = intent.get("path")
        body = intent.get("body")
        if (not isinstance(action, str) or not isinstance(method, str)
                or not isinstance(path, str)
                or not isinstance(body, (Mapping, type(None)))):
            raise SandboxOutcomeUnknown("Persisted SPACE intent has invalid field types")
        if action == "create_sandbox":
            if method != "POST" or path != "/v1/sandboxes":
                raise SandboxOutcomeUnknown("Persisted create request path changed")
            if not isinstance(body, Mapping) or body.get("template") != self.config.template:
                raise SandboxOutcomeUnknown("Persisted create request body is incompatible")
        elif action == "create_command":
            match = re.fullmatch(r"/v1/sandboxes/(sbx_[A-Za-z0-9_-]+)/commands",
                                 path or "")
            if method != "POST" or match is None:
                raise SandboxOutcomeUnknown("Persisted command request path changed")
            if not isinstance(body, Mapping) or not isinstance(body.get("command"), list):
                raise SandboxOutcomeUnknown("Persisted command request body is incompatible")
            recorded_sandbox = intent.get("sandbox_id")
            if recorded_sandbox and recorded_sandbox != match.group(1):
                raise SandboxOutcomeUnknown("Persisted command sandbox identity changed")
        elif action == "delete_sandbox":
            match = re.fullmatch(r"/v1/sandboxes/(sbx_[A-Za-z0-9_-]+)", path or "")
            if method != "DELETE" or match is None:
                raise SandboxOutcomeUnknown("Persisted delete request path changed")
            if body not in ({}, None):
                raise SandboxOutcomeUnknown("Persisted delete request body is incompatible")
            recorded_sandbox = intent.get("sandbox_id")
            if recorded_sandbox and recorded_sandbox != match.group(1):
                raise SandboxOutcomeUnknown("Persisted delete sandbox identity changed")
        else:
            raise SandboxOutcomeUnknown("Persisted SPACE intent has an unknown action")
        return dict(intent)

    def _handle(self, sandbox_id: str) -> str:
        context = current_run.get()
        run_id = context.run_id if context is not None else "local"
        epoch = context.epoch if context is not None else 0
        return f"space:{run_id}:{epoch}:{sandbox_id}"

    @staticmethod
    def _parse_handle(handle: str) -> tuple[str, int, str]:
        match = _HANDLE_RE.fullmatch(handle or "")
        if not match:
            raise SandboxUnavailableError("SPACE sandbox handle is malformed")
        return match.group(1), int(match.group(2)), match.group(3)

    def _entry(self, session_id: str) -> dict[str, Any]:
        handle_run, handle_epoch, handle_sandbox = self._parse_handle(session_id)
        with self._lock:
            entry = self._active.get(session_id)
        if entry is None:
            raise SandboxUnavailableError(
                "SPACE sandbox is not owned by this manager; it was not reused")
        if entry["sandbox_id"] != handle_sandbox:
            raise SandboxUnavailableError(
                "SPACE sandbox handle does not match the owned resource")
        context = current_run.get()
        if context is not None:
            if entry["run_id"] != context.run_id or handle_run != context.run_id:
                raise SandboxUnavailableError(
                    "SPACE sandbox belongs to a different run; refusing to reuse it")
            if entry["epoch"] != context.epoch or handle_epoch != context.epoch:
                raise SandboxUnavailableError(
                    "SPACE sandbox belongs to a different run epoch; refusing late reuse")
            if entry["task_id"] and context.task_id and entry["task_id"] != context.task_id:
                raise SandboxUnavailableError(
                    "SPACE sandbox belongs to a different task; refusing to reuse it")
        return entry

    # ------------------------------------------------------------------
    # Create -> operation -> running -> process readiness
    # ------------------------------------------------------------------
    def _metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "managed_by": "legal-agent",
            "backend": "space",
            "api_contract": API_CONTRACT,
            "template": self.config.template,
            "runner_version": RUNNER_VERSION,
        }
        context = current_run.get()
        if context is not None:
            metadata.update({
                "run_id": context.run_id,
                "epoch": context.epoch,
                "task_id": context.task_id,
                "revision": context.task_revision,
            })
            try:
                metadata["project_id"] = context.store.get(context.run_id)["project_id"]
            except KeyError:
                pass
        return metadata

    def _create_body(self) -> dict[str, Any]:
        resources: dict[str, Any] = {
            "vcpu": self.config.vcpu,
            "memory_mib": self.config.memory_mib,
        }
        if self.config.disk_mib is not None:
            resources["disk_mib"] = self.config.disk_mib
        return {
            "template": self.config.template,
            "runtime": self.config.runtime,
            "metadata": self._metadata(),
            "resources": resources,
            "network_policy": {
                "version": "v1",
                "egress": {"default_action": "deny", "allow": [], "deny": []},
            },
            "idle_timeout": f"{self.config.idle_timeout}s",
            "idle_action": "delete",
        }

    def _check_request_body(self, body: Mapping[str, Any], label: str) -> None:
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if len(payload) > self.config.max_request_bytes:
            # Local limit: checked before the idempotency intent is recorded
            # and before any request is sent.
            raise SandboxProtocolError(
                f"SPACE {label} request body size limit exceeded")

    def start_session(self) -> str:
        deadline = self._deadline()
        key = f"legal-agent-create-{uuid.uuid4().hex}"
        body = self._create_body()
        self._check_request_body(body, "create sandbox")
        request_timeout = self._bounded(deadline, self.config.request_timeout)
        self._persist("create_sandbox", "POST", "/v1/sandboxes", body, key)

        sandbox_id = None
        operation_id = None
        try:
            try:
                created = self.client.create_sandbox(body, key, request_timeout)
            except (SandboxTimeoutError, SandboxProtocolError) as exc:
                # A partial response may still name a sandbox that exists.
                sandbox_id = getattr(exc, "sandbox_id", None)
                raise SandboxOutcomeUnknown(
                    "SPACE sandbox create request outcome is unknown; inspect the "
                    "recorded idempotency key before retrying") from exc
            except SandboxRemoteError as exc:
                if exc.status in {500, 502, 503, 504}:
                    raise SandboxOutcomeUnknown(
                        "SPACE sandbox create returned a server error; its outcome is "
                        "unknown") from exc
                raise
            sandbox_id = created["sandbox_id"]
            operation_id = created["operation_id"]
            self._persist("create_sandbox", "POST", "/v1/sandboxes", body, key,
                          sandbox_id=sandbox_id, operation_id=operation_id)
            self._wait_operation(
                operation_id, self._phase_deadline(deadline, self.config.operation_timeout),
                "create_sandbox")
            self._wait_sandbox_running(
                sandbox_id, self._phase_deadline(deadline, self.config.operation_timeout))
            self._wait_process_ready(
                sandbox_id, self._phase_deadline(deadline, self.config.operation_timeout))
        except BaseException as exc:
            if sandbox_id is not None:
                cleaned = self._destroy_best_effort(sandbox_id, "create/readiness failure")
                if not cleaned and hasattr(exc, "add_note"):
                    exc.add_note(
                        f"SPACE cleanup was not proven; sandbox_id={sandbox_id}")
            raise

        handle = self._handle(sandbox_id)
        context = current_run.get()
        project_id = ""
        if context is not None:
            try:
                project_id = context.store.get(context.run_id)["project_id"]
            except KeyError:
                project_id = ""
        with self._lock:
            self._active[handle] = {
                "handle": handle,
                "sandbox_id": sandbox_id,
                "operation_id": operation_id,
                "run_id": context.run_id if context is not None else "local",
                "epoch": context.epoch if context is not None else 0,
                "task_id": context.task_id if context is not None else "",
                "revision": context.task_revision if context is not None else 0,
                "project_id": project_id,
                "executed": False,
                "deadline": deadline,
                "created_at": self.clock.wall_time(),
                "last_active": self.clock.wall_time(),
            }
        return handle

    def _wait_operation(self, operation_id: str, deadline: float, action: str, *,
                        check_context: bool = True) -> dict[str, Any]:
        try:
            while True:
                remaining = self._remaining(deadline, check_context=check_context)
                record = self.client.get_operation(
                    operation_id, min(self.config.request_timeout, remaining))
                self._remaining(deadline, check_context=check_context)
                state = record["state"]
                if state == "succeeded":
                    return record
                if state in _TERMINAL_BAD_OPERATION:
                    raise SandboxRemoteError(
                        f"SPACE {action} operation ended in state {state!r}")
                if state not in {"queued", "running"}:
                    logger.warning("SPACE %s operation returned unknown state %r",
                                   action, state)
                self._sleep(deadline, self.config.poll_interval,
                            check_context=check_context)
        except SandboxTimeoutError as exc:
            raise SandboxOutcomeUnknown(
                f"SPACE {action} operation did not finish before the deadline; "
                "the remote side may still complete it") from exc

    def _wait_sandbox_running(self, sandbox_id: str, deadline: float) -> None:
        try:
            while True:
                remaining = self._remaining(deadline)
                record = self.client.get_sandbox(
                    sandbox_id, min(self.config.request_timeout, remaining))
                self._remaining(deadline)
                state = record["state"]
                if state == "running":
                    return
                if state in _TERMINAL_BAD_SANDBOX:
                    raise SandboxRemoteError(
                        f"SPACE sandbox ended in state {state!r} before execution")
                if state not in {"creating", "starting"}:
                    logger.warning("SPACE sandbox returned unknown state %r", state)
                self._sleep(deadline, self.config.poll_interval)
        except SandboxTimeoutError as exc:
            raise SandboxOutcomeUnknown(
                "SPACE sandbox did not become running before the deadline") from exc

    def _wait_process_ready(self, sandbox_id: str, deadline: float) -> None:
        try:
            while True:
                remaining = self._remaining(deadline)
                try:
                    page = self.client.list_processes(
                        sandbox_id, min(self.config.request_timeout, remaining))
                except SandboxRemoteError as exc:
                    if exc.status not in _RETRYABLE_READINESS:
                        raise
                    self._sleep(deadline, self.config.poll_interval)
                    continue
                if not isinstance(page.get("data"), list):
                    raise SandboxProtocolError(
                        "SPACE process readiness response did not contain a list")
                self._remaining(deadline)
                return
        except SandboxTimeoutError as exc:
            raise SandboxOutcomeUnknown(
                "SPACE guest process service was not ready before the deadline") from exc

    # ------------------------------------------------------------------
    # Command creation and event stream
    # ------------------------------------------------------------------
    def _command_body(self, code: str, deadline: float) -> dict[str, Any]:
        remaining = self._remaining(deadline)
        if remaining <= 1.5:
            raise SandboxTimeoutError("Not enough task budget for a SPACE command")
        seconds = max(1, min(self.config.command_timeout, int(remaining - 1)))
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        body = {
            "command": ["python3", "-c", SPACE_RUNNER_SOURCE, encoded,
                        str(self.config.max_output_bytes)],
            "timeout": f"{seconds}s",
            "stdin": False,
        }
        self._check_request_body(body, "command")
        return body

    def execute_code(self, session_id: str, code: str) -> dict[str, Any]:
        entry = self._entry(session_id)
        if not isinstance(code, str) or not code.strip():
            raise SandboxProtocolError("No code to execute")
        if len(code.encode("utf-8")) > 65536:
            raise SandboxProtocolError("SPACE code input limit exceeded")
        with self._lock:
            if entry.get("executed"):
                raise SandboxUnavailableError(
                    "SPACE sandbox is one-shot; refusing a second command")
            entry["executed"] = True
        deadline = entry["deadline"]
        sandbox_id = entry["sandbox_id"]
        body = self._command_body(code, deadline)
        request_timeout = self._bounded(deadline, self.config.request_timeout)
        path = f"/v1/sandboxes/{sandbox_id}/commands"
        key = f"legal-agent-command-{uuid.uuid4().hex}"
        self._persist("create_command", "POST", path, body, key,
                      sandbox_id=sandbox_id)
        with self._lock:
            entry["last_active"] = self.clock.wall_time()

        command_created = False
        try:
            try:
                command = self.client.create_command(
                    sandbox_id, body, key, request_timeout)
            except (SandboxTimeoutError, SandboxProtocolError) as exc:
                raise SandboxOutcomeUnknown(
                    "SPACE command create request outcome is unknown; inspect the "
                    "recorded idempotency key before retrying") from exc
            except SandboxRemoteError as exc:
                if exc.status in {500, 502, 503, 504}:
                    raise SandboxOutcomeUnknown(
                        "SPACE command create returned a server error; its outcome is "
                        "unknown") from exc
                raise
            command_created = True
            self._persist("create_command", "POST", path, body, key,
                          sandbox_id=sandbox_id,
                          session_id=command["session_id"],
                          process_id=command["process_id"])
            events_path, event_token = self.client.validate_attach(command)
            return self._consume_command(events_path, event_token,
                                         command["process_id"], deadline)
        except SandboxRemoteError as exc:
            if command_created:
                raise SandboxOutcomeUnknown(
                    f"SPACE command was created but attachment failed: {exc}") from exc
            raise
        except SandboxProtocolError as exc:
            if command_created:
                raise SandboxOutcomeUnknown(
                    f"SPACE command was created but its stream could not be used: {exc}") from exc
            raise
        except SandboxTimeoutError as exc:
            if command_created:
                raise SandboxOutcomeUnknown(
                    "SPACE command was created but no exit event was received before "
                    "the deadline") from exc
            raise

    def _consume_command(self, events_path: str, event_token: str,
                         process_id: str, deadline: float) -> dict[str, Any]:
        remaining = self._remaining(deadline)
        stream = self.client.open_events(
            events_path, event_token,
            timeout=remaining,
            max_frame_bytes=self.config.max_frame_bytes,
            max_stream_bytes=self.config.max_stream_bytes,
            deadline=deadline, clock=self.clock.monotonic,
            check=self._check_context)
        stdout = bytearray()
        stderr = bytearray()
        # The runner emits a JSON envelope, not raw result bytes. ASCII JSON
        # escaping can require six bytes per result byte, while an error is
        # capped at 4096 Unicode characters (up to twelve escaped bytes each).
        stdout_limit = max(6 * self.config.max_output_bytes, 12 * 4096) + 1024
        exit_frame = None
        try:
            for frame in stream:
                kind = frame.get("type")
                if kind == "stdout" or kind == "stderr":
                    encoded = frame.get("data_base64")
                    if not isinstance(encoded, str):
                        raise SandboxProtocolError(
                            f"SPACE {kind} frame has no data_base64")
                    try:
                        chunk = base64.b64decode(encoded, validate=True)
                    except (ValueError, TypeError) as exc:
                        raise SandboxProtocolError(
                            f"SPACE {kind} frame has invalid base64") from exc
                    target = stdout if kind == "stdout" else stderr
                    target.extend(chunk)
                    limit = (stdout_limit if kind == "stdout"
                             else self.config.max_stderr_bytes)
                    if len(target) > limit:
                        return {"output": "", "error": f"SPACE {kind} limit exceeded",
                                "error_kind": "output_limit"}
                elif kind == "process_exited":
                    if exit_frame is not None:
                        raise SandboxProtocolError(
                            "SPACE stream contained more than one process_exited frame")
                    exit_frame = frame
                    break
                elif kind in {"error", "resync_required"}:
                    raise SandboxOutcomeUnknown(
                        f"SPACE command stream reported {kind}; output was not replayed")
                elif kind in {"attached", "process_started", "control_accepted",
                              "input_accepted", "closed"}:
                    continue
                else:
                    # Response vocabularies are extensible; ignore unknown frames.
                    continue
        except (SandboxTimeoutError, SandboxProtocolError) as exc:
            raise SandboxOutcomeUnknown(
                f"SPACE command stream ended without a clean exit: {exc}") from exc
        finally:
            stream.close()

        if exit_frame is None:
            raise SandboxOutcomeUnknown(
                "SPACE command stream ended without process_exited")
        if exit_frame.get("process_id") != process_id:
            raise SandboxProtocolError(
                "SPACE process_exited names a different process")
        exit_code = exit_frame.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise SandboxProtocolError("SPACE process_exited has a non-integer exit_code")
        status = exit_frame.get("status")
        if status != "exited":
            return {"output": "", "error": f"SPACE command status was {status!r}",
                    "error_kind": "timeout" if status == "timed_out" else "execution"}
        if exit_code != 0:
            detail = stderr.decode("utf-8", "replace").strip()
            error = f"SPACE command exited with code {exit_code}"
            if detail:
                error += f": {detail[:4096]}"
            return {"output": "", "error": error, "error_kind": "execution"}
        if stderr:
            logger.warning("SPACE command wrote stderr before a successful exit: %s",
                           stderr.decode("utf-8", "replace")[:4096])

        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise SandboxProtocolError(
                "SPACE runner did not return a valid UTF-8 JSON result") from exc
        if not isinstance(payload, dict):
            raise SandboxProtocolError("SPACE runner result must be a JSON object")
        if payload.get("error"):
            return {"output": "", "error": str(payload["error"])[:4096],
                    "error_kind": payload.get("error_kind") or "execution"}
        output = payload.get("output")
        if not isinstance(output, str) or not output.strip():
            raise SandboxProtocolError("SPACE runner returned an empty result")
        if len(output.encode("utf-8")) > self.config.max_output_bytes:
            return {"output": "", "error": "SPACE output limit exceeded",
                    "error_kind": "output_limit"}
        return {"output": output, "error": None}

    # ------------------------------------------------------------------
    # Cleanup and ownership
    # ------------------------------------------------------------------
    def _record_delete_intent(self, sandbox_id: str, key: str) -> None:
        """Record the delete request while the tool effect is still open.

        Cleanup must proceed even after cancellation/epoch loss, so an
        ownership error while recording the intent is logged and does not
        prevent the explicit DELETE attempt.
        """
        try:
            self._persist("delete_sandbox", "DELETE",
                          f"/v1/sandboxes/{sandbox_id}", {}, key,
                          sandbox_id=sandbox_id)
        except Exception as exc:
            logger.warning(
                "SPACE delete intent could not be recorded (sandbox_id=%s): %s",
                sandbox_id, exc)

    def _record_delete_result(self, sandbox_id: str, key: str,
                              operation_id: str) -> None:
        try:
            self._persist("delete_sandbox", "DELETE",
                          f"/v1/sandboxes/{sandbox_id}", {}, key,
                          sandbox_id=sandbox_id, operation_id=operation_id)
        except Exception as exc:
            logger.warning(
                "SPACE delete result could not be recorded (sandbox_id=%s): %s",
                sandbox_id, exc)

    def _destroy_best_effort(self, sandbox_id: str, reason: str) -> bool:
        deadline = self.clock.monotonic() + CLEANUP_SECONDS
        key = f"legal-agent-delete-{uuid.uuid4().hex}"
        timeout = min(self.config.request_timeout, CLEANUP_SECONDS)
        self._record_delete_intent(sandbox_id, key)
        try:
            created = self.client.delete_sandbox(sandbox_id, key, timeout)
            self._record_delete_result(sandbox_id, key, created["operation_id"])
            self._wait_operation(created["operation_id"], deadline, "delete_sandbox",
                                 check_context=False)
            return True
        except Exception as exc:
            logger.exception("SPACE cleanup failed after %s: sandbox_id=%s",
                             reason, sandbox_id)
            return False

    def destroy_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            entry = self._active.get(session_id)
            if entry is None:
                if session_id in self._cleaned:
                    return
                raise SandboxCleanupError(
                    "SPACE sandbox cleanup cannot be confirmed: session is not "
                    "owned by this manager",
                    session_id=session_id)
            sandbox_id = entry["sandbox_id"]
            key = entry.get("delete_key")
            if key is None:
                key = f"legal-agent-delete-{uuid.uuid4().hex}"
                entry["delete_key"] = key
        handle_run, _, _ = self._parse_handle(session_id)
        context = current_run.get()
        if context is not None:
            if handle_run != context.run_id:
                raise SandboxCleanupError(
                    "SPACE sandbox belongs to a different run; refusing cleanup",
                    session_id=session_id, sandbox_id=sandbox_id)
            if entry.get("task_id") and context.task_id and entry["task_id"] != context.task_id:
                raise SandboxCleanupError(
                    "SPACE sandbox belongs to a different task; refusing cleanup",
                    session_id=session_id, sandbox_id=sandbox_id)
            if entry.get("project_id"):
                try:
                    project_id = context.store.get(context.run_id)["project_id"]
                except KeyError:
                    project_id = ""
                if project_id and project_id != entry["project_id"]:
                    raise SandboxCleanupError(
                        "SPACE sandbox belongs to a different project; refusing cleanup",
                        session_id=session_id, sandbox_id=sandbox_id)
        deadline = self.clock.monotonic() + CLEANUP_SECONDS
        timeout = min(self.config.request_timeout, CLEANUP_SECONDS)
        self._record_delete_intent(sandbox_id, key)
        try:
            created = self.client.delete_sandbox(sandbox_id, key, timeout)
            self._record_delete_result(sandbox_id, key, created["operation_id"])
            self._wait_operation(created["operation_id"], deadline, "delete_sandbox",
                                 check_context=False)
        except (SandboxRemoteError, SandboxTimeoutError, SandboxOutcomeUnknown,
                SandboxProtocolError) as exc:
            raise SandboxCleanupError(
                f"SPACE sandbox cleanup was not proven for {sandbox_id}: {exc}",
                session_id=session_id, sandbox_id=sandbox_id) from exc
        with self._lock:
            self._active.pop(session_id, None)
            self._cleaned.add(session_id)

    def cleanup_idle(self, max_idle_seconds: float | None = None) -> None:
        limit = self.config.idle_timeout if max_idle_seconds is None else max_idle_seconds
        now = self.clock.wall_time()
        with self._lock:
            stale = [handle for handle, entry in self._active.items()
                     if now - entry["last_active"] > limit]
        for handle in stale:
            try:
                self.destroy_session(handle)
            except SandboxCleanupError as exc:
                logger.error("SPACE idle cleanup failed: session_id=%s sandbox_id=%s "
                             "error=%s", handle, exc.sandbox_id, exc)

    def cleanup_orphans(self) -> None:
        """Only clean resources this manager still owns.

        Listing sandboxes globally and deleting by age would cross task
        ownership, so that is deliberately not implemented.
        """
        self.cleanup_idle()

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)
