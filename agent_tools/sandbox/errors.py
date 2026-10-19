"""Explicit sandbox backend errors.

These errors are intentionally narrow.  Configuration problems must not be
converted into a different backend, and an ambiguous remote outcome must be
propagated as ``OutcomeUnknown`` by the executor, never re-labelled as a normal
code error.
"""
from __future__ import annotations


class SandboxError(RuntimeError):
    """Base class for sandbox backend failures."""


class SandboxConfigError(SandboxError):
    """The selected backend is not configured validly.

    The caller must fail; it must not fall back to Docker or host execution.
    """


class SandboxUnavailableError(SandboxError):
    """The backend cannot be used in this deployment."""


class SandboxRemoteError(SandboxError):
    """A remote API returned a definite error."""

    def __init__(self, message, *, status=None, code=None, retryable=None,
                 request_id=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.request_id = request_id


class SandboxProtocolError(SandboxError):
    """The remote protocol response was malformed or incomplete."""


class SandboxTimeoutError(SandboxError):
    """A bounded client/operation deadline expired before a definite result."""


class SandboxOutcomeUnknown(SandboxError):
    """A side effect may have happened, but its result is not known.

    The executor maps this to ``agent_runtime.contracts.OutcomeUnknown`` so the
    run is not automatically retried and the code is not rewritten.
    """


class SandboxCleanupError(SandboxError):
    """A resource could not be proven released.

    The message and ``session_id``/``sandbox_id`` attributes are safe to log;
    they must never contain a bearer token.
    """

    def __init__(self, message, *, session_id=None, sandbox_id=None):
        super().__init__(message)
        self.session_id = session_id
        self.sandbox_id = sandbox_id
