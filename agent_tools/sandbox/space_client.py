"""Minimal SPACE native API client.

Pinned upstream contract: commit ``57c7c0c47a73373a974c5b542d180599795465da``,
native ``/v1`` M1 surface.  Only the operations this project needs are used:

* ``POST /v1/sandboxes`` (202, operation handle)
* ``GET /v1/operations/{operation_id}``
* ``GET /v1/sandboxes/{sandbox_id}``
* ``GET /v1/sandboxes/{sandbox_id}/processes?limit=1``
* ``POST /v1/sandboxes/{sandbox_id}/commands`` (201, PublicCommand)
* ``GET {data_plane}/v1/sessions/{session_id}/events`` (SSE)
* ``DELETE /v1/sandboxes/{sandbox_id}`` (202, operation handle)

No Python SDK or E2B compatibility layer is used.  The transport is injectable
so the lifecycle and parser can be tested without a SPACE deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import http.client
import json
import re
import socket
import ssl
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .errors import (SandboxConfigError, SandboxProtocolError, SandboxRemoteError,
                     SandboxTimeoutError)

UPSTREAM_COMMIT = "57c7c0c47a73373a974c5b542d180599795465da"
API_CONTRACT = "m1"
MAX_API_BODY_BYTES = 4 * 1024 * 1024
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_TEMPLATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*@sha256:[0-9a-f]{12,64}$")
_ID_RE = {
    "sbx": re.compile(r"^sbx_[A-Za-z0-9_-]+$"),
    "op": re.compile(r"^op_[A-Za-z0-9_-]+$"),
    "proc": re.compile(r"^proc_[A-Za-z0-9_-]+$"),
}


def _fail(key: str, message: str) -> SandboxConfigError:
    return SandboxConfigError(f"sandbox.space.{key}: {message}")


def _number(value: Any, key: str, *, minimum: float, maximum: float | None = None,
            integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(key, "must be a number")
    if integer and not isinstance(value, int):
        raise _fail(key, "must be an integer")
    if value < minimum:
        raise _fail(key, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise _fail(key, f"must be <= {maximum}")
    return value


def _text(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(key, "must be a non-empty string")
    if "\x00" in value or any(ord(char) < 32 for char in value.strip()):
        raise _fail(key, "must not contain control characters")
    return value.strip()


def _origin(value: Any, key: str) -> str:
    text = _text(value, key)
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"}:
        raise _fail(key, "must use http or https")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise _fail(key, "must be an origin without credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise _fail(key, "must be an origin without a path, query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise _fail(key, "has a malformed port") from exc
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK:
        raise _fail(key, "requires HTTPS outside loopback")
    return f"{parsed.scheme}://{parsed.netloc}"


def validate_id(value: Any, prefix: str, key: str) -> str:
    if not isinstance(value, str) or not _ID_RE[prefix].fullmatch(value):
        raise SandboxProtocolError(f"{key} is not a valid {prefix}_ identifier")
    return value


@dataclass(frozen=True)
class SpaceConfig:
    """Validated SPACE settings.

    Resource values are explicit SPACE units.  Docker ``cpu_quota`` and
    ``mem_limit`` are never reinterpreted as vCPU/memory here.
    """

    api_url: str
    data_plane_url: str
    api_token: str
    template: str
    runtime: str = "fc"
    api_contract: str = API_CONTRACT
    vcpu: int = 1
    memory_mib: int = 512
    disk_mib: int | None = None
    connect_timeout: float = 5.0
    request_timeout: float = 30.0
    operation_timeout: float = 60.0
    stream_timeout: float = 45.0
    command_timeout: int = 10
    poll_interval: float = 1.0
    max_total_seconds: float = 120.0
    idle_timeout: int = 600
    max_output_bytes: int = 65536
    max_stderr_bytes: int = 16384
    max_frame_bytes: int = 1048576
    max_stream_bytes: int = 4 * 1024 * 1024
    max_request_bytes: int = 262144
    idempotency_window_seconds: int = 86400
    verified_deployment: str = ""
    verified_at: str = ""
    verification_evidence: str = ""

    @classmethod
    def from_mapping(cls, sandbox: Mapping[str, Any] | None) -> "SpaceConfig":
        sandbox = sandbox or {}
        if not isinstance(sandbox, Mapping):
            raise SandboxConfigError("sandbox must be a mapping")
        backend = sandbox.get("backend", "docker")
        if backend != "space":
            raise _fail("backend", "must be 'space' to construct this backend")
        space = sandbox.get("space")
        if not isinstance(space, Mapping):
            raise _fail("enabled", "missing sandbox.space mapping")
        if space.get("enabled") is not True:
            raise _fail("enabled", "must be the boolean true")

        api_url = _origin(space.get("api_url"), "api_url")
        data_plane_url = _origin(space.get("data_plane_url"), "data_plane_url")
        token = _text(space.get("api_token"), "api_token")
        if token.startswith("${"):
            raise _fail("api_token", "environment variable is unresolved")

        template = _text(space.get("template"), "template")
        if not _TEMPLATE_RE.fullmatch(template):
            raise _fail("template", "must be a digest-pinned reference such as "
                                   "'seed@sha256:<12-64 lowercase hex>'")

        runtime = _text(space.get("runtime", "fc"), "runtime")
        if runtime != "fc":
            raise _fail("runtime", "only 'fc' is supported by this adapter")
        contract = _text(space.get("api_contract", API_CONTRACT), "api_contract")
        if contract != API_CONTRACT:
            raise _fail("api_contract", f"only '{API_CONTRACT}' is supported")

        isolation = space.get("network_isolation")
        if not isinstance(isolation, Mapping):
            raise _fail("network_isolation", "missing verification record")
        if isolation.get("mode") != "egress_deny_all":
            raise _fail("network_isolation.mode", "must be 'egress_deny_all'")
        verified_deployment = _text(isolation.get("verified_deployment"),
                                    "network_isolation.verified_deployment")
        verified_at = _text(isolation.get("verified_at"),
                            "network_isolation.verified_at")
        try:
            date.fromisoformat(verified_at)
        except ValueError as exc:
            raise _fail("network_isolation.verified_at",
                        "must be an ISO date (YYYY-MM-DD)") from exc
        evidence = _text(isolation.get("evidence"), "network_isolation.evidence")

        vcpu = int(_number(space.get("vcpu"), "vcpu", minimum=1, maximum=256,
                           integer=True))
        memory_mib = int(_number(space.get("memory_mib"), "memory_mib",
                                 minimum=64, maximum=1048576, integer=True))
        disk_mib = space.get("disk_mib")
        if disk_mib is not None:
            disk_mib = int(_number(disk_mib, "disk_mib", minimum=1,
                                   maximum=1048576, integer=True))

        connect_timeout = float(_number(space.get("connect_timeout", 5),
                                        "connect_timeout", minimum=0.1,
                                        maximum=300))
        request_timeout = float(_number(space.get("request_timeout", 30),
                                        "request_timeout", minimum=0.1,
                                        maximum=600))
        operation_timeout = float(_number(space.get("operation_timeout", 60),
                                          "operation_timeout", minimum=0.1,
                                          maximum=3600))
        stream_timeout = float(_number(space.get("stream_timeout", 45),
                                       "stream_timeout", minimum=0.1,
                                       maximum=3600))
        command_timeout = int(_number(space.get("command_timeout", 10),
                                      "command_timeout", minimum=1,
                                      maximum=3600, integer=True))
        poll_interval = float(_number(space.get("poll_interval", 1),
                                      "poll_interval", minimum=0,
                                      maximum=60))
        max_total_seconds = float(_number(space.get("max_total_seconds", 120),
                                          "max_total_seconds", minimum=1,
                                          maximum=3600))
        idle_timeout = int(_number(space.get("idle_timeout", 600),
                                   "idle_timeout", minimum=1,
                                   maximum=86400, integer=True))
        max_output_bytes = int(_number(space.get("max_output_bytes", 65536),
                                       "max_output_bytes", minimum=1,
                                       maximum=10 * 1024 * 1024, integer=True))
        max_stderr_bytes = int(_number(space.get("max_stderr_bytes", 16384),
                                       "max_stderr_bytes", minimum=1,
                                       maximum=10 * 1024 * 1024, integer=True))
        max_frame_bytes = int(_number(space.get("max_frame_bytes", 1048576),
                                      "max_frame_bytes", minimum=1024,
                                      maximum=16 * 1024 * 1024, integer=True))
        max_stream_bytes = int(_number(space.get("max_stream_bytes", 4 * 1024 * 1024),
                                       "max_stream_bytes", minimum=1024,
                                       maximum=64 * 1024 * 1024, integer=True))
        max_request_bytes = int(_number(space.get("max_request_bytes", 262144),
                                        "max_request_bytes", minimum=1024,
                                        maximum=MAX_API_BODY_BYTES, integer=True))
        idempotency_window = int(_number(space.get("idempotency_window_seconds", 86400),
                                         "idempotency_window_seconds", minimum=1,
                                         maximum=86400, integer=True))

        return cls(
            api_url=api_url, data_plane_url=data_plane_url, api_token=token,
            template=template, runtime=runtime, api_contract=contract,
            vcpu=vcpu, memory_mib=memory_mib, disk_mib=disk_mib,
            connect_timeout=connect_timeout, request_timeout=request_timeout,
            operation_timeout=operation_timeout, stream_timeout=stream_timeout,
            command_timeout=command_timeout, poll_interval=poll_interval,
            max_total_seconds=max_total_seconds, idle_timeout=idle_timeout,
            max_output_bytes=max_output_bytes, max_stderr_bytes=max_stderr_bytes,
            max_frame_bytes=max_frame_bytes, max_stream_bytes=max_stream_bytes,
            max_request_bytes=max_request_bytes,
            idempotency_window_seconds=idempotency_window,
            verified_deployment=verified_deployment, verified_at=verified_at,
            verification_evidence=evidence,
        )

    def __repr__(self):
        return (f"SpaceConfig(api_url={self.api_url!r}, "
                f"data_plane_url={self.data_plane_url!r}, "
                f"template={self.template!r}, api_token=<redacted>)")

    def identity(self) -> dict[str, Any]:
        """Semantic identity; deliberately excludes the bearer token."""
        return {
            "backend": "space",
            "api_contract": self.api_contract,
            "upstream_commit": UPSTREAM_COMMIT,
            "api_url": self.api_url,
            "data_plane_url": self.data_plane_url,
            "template": self.template,
            "runtime": self.runtime,
            "vcpu": self.vcpu,
            "memory_mib": self.memory_mib,
            "disk_mib": self.disk_mib,
            "network_isolation": "egress_deny_all",
            "verified_deployment": self.verified_deployment,
        }


class HttpResponse:
    """Small response wrapper shared by the real and injected transports."""

    def __init__(self, status: int, headers: Mapping[str, str], stream, *,
                 close_stream: bool = True):
        self.status = int(status)
        self.headers = {str(key).lower(): str(value)
                        for key, value in (headers or {}).items()}
        self._stream = stream
        self._close_stream = close_stream

    def read1(self, size: int) -> bytes:
        reader = getattr(self._stream, "read1", None)
        if reader is not None:
            return reader(size)
        return self._stream.read(size)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        if self._close_stream:
            try:
                self._stream.close()
            except Exception:
                # Closing a failed stream is best-effort and must not replace
                # the execution result.
                pass


class _HttpClientStream:
    def __init__(self, response, connection, *, deadline, timer):
        self._response = response
        self._connection = connection
        self._deadline = deadline
        self._timer = timer

    def _check_deadline(self):
        if time.monotonic() >= self._deadline:
            raise TimeoutError("SPACE HTTP request deadline expired")

    def read1(self, size: int) -> bytes:
        self._check_deadline()
        data = self._response.read1(size)
        self._check_deadline()
        return data

    def read(self, size: int = -1) -> bytes:
        self._check_deadline()
        data = self._response.read(size)
        self._check_deadline()
        return data

    def close(self) -> None:
        self._timer.cancel()
        try:
            self._response.close()
        finally:
            self._connection.close()


class UrllibTransport:
    """Standard-library transport with connect, idle-read and total budgets.

    ``http.client`` does not follow redirects or use environment proxies, so a
    bearer token cannot be forwarded to an unexpected destination.
    """

    def open(self, method: str, url: str, headers: Mapping[str, str],
             body: bytes | None, timeout: float, read_timeout: float | None = None,
             total_timeout: float | None = None
             ) -> HttpResponse:
        read_timeout = timeout if read_timeout is None else read_timeout
        budget = read_timeout if total_timeout is None else total_timeout
        deadline = time.monotonic() + budget
        timer = None
        parsed = urlsplit(url)
        if parsed.scheme == "https":
            connection = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443, timeout=timeout,
                context=ssl.create_default_context())
        elif parsed.scheme == "http":
            connection = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=timeout)
        else:
            raise SandboxProtocolError("SPACE transport only supports HTTP(S)")
        try:
            connection.connect()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SPACE HTTP request deadline expired during connection")
            connected_socket = connection.sock
            connected_socket.settimeout(min(read_timeout, remaining))

            def expire():
                # Closing a socket alone does not reliably interrupt another
                # thread's recv on Windows. Shutdown also unblocks reads of
                # headers and trickling/chunked bodies at the total deadline.
                try:
                    connected_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # The caller may already have closed the response.

            timer = threading.Timer(remaining, expire)
            timer.daemon = True
            timer.start()
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            connection.request(method, target, body=body, headers=dict(headers))
            response = connection.getresponse()
            if time.monotonic() >= deadline:
                response.close()
                raise TimeoutError("SPACE HTTP request deadline expired during headers")
            return HttpResponse(response.status, dict(response.getheaders()),
                                _HttpClientStream(response, connection,
                                                  deadline=deadline, timer=timer))
        except BaseException:
            if timer is not None:
                timer.cancel()
            connection.close()
            raise


class SpaceEventStream:
    """Incremental SSE parser for the SPACE data plane.

    It handles arbitrary chunk boundaries, multi-line ``data:`` fields, CRLF,
    comments/heartbeats and a final line without a newline.  A stream that ends
    with a partial frame is an error, never an empty success.
    """

    def __init__(self, response: HttpResponse, *, max_frame_bytes: int,
                 max_stream_bytes: int, deadline: float | None,
                 clock: Callable[[], float], check: Callable[[], None] | None):
        self._response = response
        self._max_frame_bytes = max_frame_bytes
        self._max_stream_bytes = max_stream_bytes
        self._deadline = deadline
        self._clock = clock
        self._check = check
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._response.close()

    def __iter__(self):
        total = 0
        buffer = bytearray()
        data_lines: list[bytes] = []
        frame_bytes = 0

        def process_line(line: bytes):
            nonlocal data_lines, frame_bytes
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                if data_lines:
                    raw = b"\n".join(data_lines)
                    data_lines = []
                    frame_bytes = 0
                    try:
                        frame = json.loads(raw)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise SandboxProtocolError("SPACE event frame is not valid JSON") from exc
                    if not isinstance(frame, dict):
                        raise SandboxProtocolError("SPACE event frame must be a JSON object")
                    yield frame
                return
            if line.startswith(b":"):
                return
            name, separator, value = line.partition(b":")
            if not separator or name != b"data":
                return
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value)
            frame_bytes += len(value) + 1
            if frame_bytes > self._max_frame_bytes:
                raise SandboxProtocolError("SPACE SSE frame size limit exceeded")

        while True:
            if self._deadline is not None and self._clock() >= self._deadline:
                raise SandboxTimeoutError("SPACE stream deadline expired")
            if self._check is not None:
                self._check()
            try:
                chunk = self._response.read1(65536)
            except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
                raise SandboxTimeoutError(f"SPACE stream read failed: {exc}") from exc
            if self._deadline is not None and self._clock() >= self._deadline:
                raise SandboxTimeoutError("SPACE stream deadline expired")
            if self._check is not None:
                self._check()
            if not chunk:
                break
            total += len(chunk)
            if total > self._max_stream_bytes:
                raise SandboxProtocolError("SPACE stream size limit exceeded")
            buffer.extend(chunk)
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                line = bytes(buffer[:newline])
                del buffer[:newline + 1]
                yield from process_line(line)
            if len(buffer) > self._max_frame_bytes:
                raise SandboxProtocolError("SPACE SSE line size limit exceeded")
        if buffer:
            yield from process_line(bytes(buffer))
        if data_lines:
            raise SandboxProtocolError("SPACE stream ended in an incomplete SSE frame")
        self.close()


class SpaceClient:
    def __init__(self, config: SpaceConfig, transport=None):
        self.config = config
        self.transport = transport or UrllibTransport()

    def __repr__(self):
        return f"SpaceClient(api_url={self.config.api_url!r}, template={self.config.template!r})"

    def _headers(self, *, idempotency_key: str | None = None,
                 accept: str = "application/json") -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.config.api_token}",
                   "Accept": accept}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    @staticmethod
    def _path(path: str) -> str:
        if not path.startswith("/v1/") or path.startswith("//") or ".." in path:
            raise SandboxProtocolError("SPACE API path must be an absolute /v1 path")
        if any(ord(char) < 33 for char in path):
            raise SandboxProtocolError("SPACE API path contains whitespace")
        return path

    def _open(self, method: str, url: str, headers: Mapping[str, str],
              body: bytes | None, *, connect_timeout: float,
              read_timeout: float, total_timeout: float) -> HttpResponse:
        if connect_timeout <= 0 or read_timeout <= 0 or total_timeout <= 0:
            raise SandboxTimeoutError("SPACE request deadline expired")
        try:
            return self.transport.open(method, url, headers, body, connect_timeout,
                                       read_timeout=read_timeout,
                                       total_timeout=total_timeout)
        except (SandboxTimeoutError, SandboxProtocolError):
            raise
        except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
            raise SandboxTimeoutError(f"SPACE transport failed: {exc}") from exc

    def request_json(self, method: str, path: str, *, body: Mapping[str, Any] | None = None,
                     idempotency_key: str | None = None,
                     timeout: float | None = None) -> dict[str, Any]:
        path = self._path(path)
        payload = None
        headers = self._headers(idempotency_key=idempotency_key)
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            if len(payload) > self.config.max_request_bytes:
                raise SandboxProtocolError("SPACE request body size limit exceeded")
            headers["Content-Type"] = "application/json"
        budget = self.config.request_timeout if timeout is None else timeout
        response = self._open(
            method, self.config.api_url + path, headers, payload,
            connect_timeout=min(self.config.connect_timeout, budget),
            read_timeout=min(self.config.request_timeout, budget),
            total_timeout=budget)
        try:
            raw = response.read(1024 * 1024 + 1)
        except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
            raise SandboxTimeoutError(f"SPACE response read failed: {exc}") from exc
        finally:
            response.close()
        if len(raw) > 1024 * 1024:
            raise SandboxProtocolError("SPACE API response size limit exceeded")
        if not 200 <= response.status < 300:
            raise self._remote_error(method, path, response.status, raw)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise SandboxProtocolError("SPACE API response is not valid JSON") from exc
        if not isinstance(value, dict):
            raise SandboxProtocolError("SPACE API response must be a JSON object")
        return value

    @staticmethod
    def _remote_error(method: str, path: str, status: int, raw: bytes) -> SandboxRemoteError:
        code = None
        message = raw[:4096].decode("utf-8", "replace")
        retryable = None
        request_id = None
        try:
            envelope = json.loads(raw)
            if isinstance(envelope, dict):
                # Management errors are top-level; data-plane errors wrap the
                # same fields in {"error": {...}}.
                detail = envelope.get("error")
                if isinstance(detail, dict):
                    envelope = detail
                code = envelope.get("code")
                message = str(envelope.get("message") or message)[:4096]
                retryable = envelope.get("retryable")
                request_id = envelope.get("request_id")
        except (ValueError, UnicodeDecodeError):
            pass
        return SandboxRemoteError(
            f"{method} {path}: HTTP {status}"
            + (f" {code}" if code else "")
            + (f": {message}" if message else ""),
            status=status, code=code, retryable=retryable, request_id=request_id)

    def create_sandbox(self, body: Mapping[str, Any], idempotency_key: str,
                       timeout: float) -> dict[str, Any]:
        result = self.request_json("POST", "/v1/sandboxes", body=body,
                                   idempotency_key=idempotency_key, timeout=timeout)
        sandbox_id = result.get("sandbox_id")
        try:
            validate_id(sandbox_id, "sbx", "sandbox_id")
            validate_id(result.get("operation_id"), "op", "operation_id")
        except SandboxProtocolError as exc:
            # The example preserves every valid ID even when another required
            # field is missing.  Expose the valid sandbox id so the manager can
            # still attempt cleanup of a resource that may exist.
            if isinstance(sandbox_id, str) and _ID_RE["sbx"].fullmatch(sandbox_id):
                exc.sandbox_id = sandbox_id
            raise
        return result

    def get_operation(self, operation_id: str, timeout: float) -> dict[str, Any]:
        operation_id = validate_id(operation_id, "op", "operation_id")
        result = self.request_json("GET", f"/v1/operations/{operation_id}",
                                   timeout=timeout)
        if not isinstance(result.get("state"), str):
            raise SandboxProtocolError("SPACE operation response has no state")
        return result

    def get_sandbox(self, sandbox_id: str, timeout: float, *,
                    allow_not_found: bool = False) -> dict[str, Any] | None:
        sandbox_id = validate_id(sandbox_id, "sbx", "sandbox_id")
        try:
            result = self.request_json("GET", f"/v1/sandboxes/{sandbox_id}",
                                       timeout=timeout)
        except SandboxRemoteError as exc:
            if allow_not_found and exc.status == 404:
                return None
            raise
        if not isinstance(result.get("state"), str):
            raise SandboxProtocolError("SPACE sandbox response has no state")
        return result

    def list_processes(self, sandbox_id: str, timeout: float) -> dict[str, Any]:
        sandbox_id = validate_id(sandbox_id, "sbx", "sandbox_id")
        return self.request_json("GET",
                                 f"/v1/sandboxes/{sandbox_id}/processes?limit=1",
                                 timeout=timeout)

    def create_command(self, sandbox_id: str, body: Mapping[str, Any],
                       idempotency_key: str, timeout: float) -> dict[str, Any]:
        sandbox_id = validate_id(sandbox_id, "sbx", "sandbox_id")
        result = self.request_json("POST",
                                   f"/v1/sandboxes/{sandbox_id}/commands",
                                   body=body, idempotency_key=idempotency_key,
                                   timeout=timeout)
        validate_id(result.get("sandbox_id"), "sbx", "sandbox_id")
        validate_id(result.get("session_id"), "proc", "session_id")
        validate_id(result.get("process_id"), "proc", "process_id")
        if result["sandbox_id"] != sandbox_id:
            raise SandboxProtocolError(
                "SPACE command response names a different sandbox")
        if not isinstance(result.get("attach"), Mapping):
            raise SandboxProtocolError("SPACE command response has no attachment")
        return result

    def delete_sandbox(self, sandbox_id: str, idempotency_key: str,
                       timeout: float) -> dict[str, Any]:
        sandbox_id = validate_id(sandbox_id, "sbx", "sandbox_id")
        result = self.request_json("DELETE", f"/v1/sandboxes/{sandbox_id}",
                                   idempotency_key=idempotency_key, timeout=timeout)
        validate_id(result.get("operation_id"), "op", "operation_id")
        if "sandbox_id" in result:
            validate_id(result.get("sandbox_id"), "sbx", "sandbox_id")
            if result["sandbox_id"] != sandbox_id:
                raise SandboxProtocolError(
                    "SPACE delete response names a different sandbox")
        return result

    def event_stream_path(self, session_id: str) -> str:
        session_id = validate_id(session_id, "proc", "session_id")
        return f"/v1/sessions/{session_id}/events"

    def validate_attach(self, command: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(events_path, token)`` after origin/path validation.

        The returned token is used immediately and is never persisted.
        """
        session_id = validate_id(command.get("session_id"), "proc", "session_id")
        expected = self.event_stream_path(session_id)
        attach = command.get("attach")
        if not isinstance(attach, Mapping):
            raise SandboxProtocolError("SPACE command has no attach object")
        events_url = attach.get("events_url")
        if not isinstance(events_url, str) or not events_url:
            raise SandboxProtocolError("SPACE command has no events_url")
        if events_url not in {expected, self.config.data_plane_url + expected}:
            raise SandboxProtocolError(
                "SPACE events_url does not match the configured data-plane origin")
        transport = attach.get("token_transport")
        if (not isinstance(transport, list)
                or "authorization_header" not in transport):
            raise SandboxProtocolError(
                "SPACE attachment does not accept the Authorization header")
        token = attach.get("token")
        if not isinstance(token, str) or not token:
            raise SandboxProtocolError("SPACE command attachment token is missing")
        expires_at = attach.get("expires_at")
        if not isinstance(expires_at, str):
            raise SandboxProtocolError("SPACE attachment expiry is missing")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SandboxProtocolError("SPACE attachment expiry is invalid") from exc
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= datetime.now(timezone.utc):
            raise SandboxProtocolError("SPACE attachment token has expired")
        return expected, token

    def open_events(self, events_path: str, token: str, *, timeout: float,
                    max_frame_bytes: int, max_stream_bytes: int,
                    deadline: float | None, clock: Callable[[], float],
                    check: Callable[[], None] | None) -> SpaceEventStream:
        if not events_path.startswith("/v1/sessions/") or "/.." in events_path:
            raise SandboxProtocolError("SPACE events path is not a session path")
        headers = {"Authorization": f"Bearer {token}",
                   "Accept": "text/event-stream"}
        response = self._open(
            "GET", self.config.data_plane_url + events_path, headers, None,
            connect_timeout=min(self.config.connect_timeout, timeout),
            read_timeout=min(timeout, self.config.stream_timeout),
            total_timeout=timeout)
        if response.status != 200:
            try:
                raw = response.read(65536)
            except (TimeoutError, socket.timeout, http.client.HTTPException, OSError):
                raw = b""
            finally:
                response.close()
            raise self._remote_error("GET", events_path, response.status, raw)
        return SpaceEventStream(response, max_frame_bytes=max_frame_bytes,
                                max_stream_bytes=max_stream_bytes,
                                deadline=deadline, clock=clock, check=check)
