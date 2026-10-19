"""Per-run context, durable budgets and replayable read-only tool calls."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import time

from .store import dumps

current_run = ContextVar("agent_run", default=None)
# Set only while a tool body is executing. Sandbox backends use this to persist
# their original remote request (method/path/body/idempotency key) before they
# send it. It is deliberately not part of graph state.
current_effect = ContextVar("agent_effect", default=None)


def record_effect_intent(record: dict) -> bool:
    """Persist a side-effect intent for the currently executing tool call.

    Returns False when no run context is active (offline/trusted direct use).
    Secrets must already have been removed by the caller.
    """
    effect = current_effect.get()
    if effect is None:
        return False
    store, run_id, epoch, token, call_id = effect
    store.record_effect_request(run_id, epoch, token, call_id, record)
    return True


@dataclass
class RunContext:
    store: object
    run_id: str
    epoch: int
    token: str
    node: str = ""
    step: str = ""
    ordinal: int = 0
    task_id: str = ""
    task_revision: int = 0

    def fork(self, task_id, revision=0):
        return RunContext(self.store, self.run_id, self.epoch, self.token,
                          self.node, self.step, task_id=task_id, task_revision=revision)

    def check(self):
        self.store.heartbeat(self.run_id, self.epoch, self.token)

    def begin_node(self, name, state):
        self.check()
        self.store.consume(self.run_id, self.epoch, self.token, node_steps=1)
        self.node = name
        self.step = str(state.get("recursion_depth", 0))
        self.ordinal = 0
        self.store.event(self.run_id, self.epoch, self.token, "node_started", {"node": name})

    def tool(self, name, inputs, call, *, read_only=True):
        self.check()
        self.ordinal += 1
        digest = hashlib.sha256(dumps(inputs).encode()).hexdigest()
        branch = f":{self.task_id}:{self.task_revision}" if self.task_id else ""
        identity = f"{self.run_id}:{self.epoch}:{self.node}:{self.step}{branch}:{name}:{digest}"
        cached, result = self.store.begin_effect(self.run_id, self.epoch, self.token,
                                                identity, name, digest, read_only)
        if cached:
            return result
        started = time.perf_counter()
        marker = current_effect.set((self.store, self.run_id, self.epoch,
                                     self.token, identity))
        try:
            result = call()
        except Exception as exc:
            failure = {"error": f"{type(exc).__name__}: {exc}"}
            # Preserve the original remote request so a later reconciliation
            # can inspect the key/body even though this attempt failed.
            intents = self.store.effect_intents(identity)
            if intents:
                failure["_intents"] = intents
            self.store.end_effect(self.run_id, self.epoch, self.token, identity,
                                  failure, status="failed")
            raise
        finally:
            current_effect.reset(marker)
        self.check()
        self.store.end_effect(self.run_id, self.epoch, self.token, identity, result)
        self.store.event(self.run_id, self.epoch, self.token, "tool_completed",
                         {"tool": name, "task_id": self.task_id, "revision": self.task_revision,
                          "seconds": time.perf_counter() - started})
        return result

    def model_budget(self, messages, max_output):
        # UTF-8 bytes upper-bound text tokens; protocol overhead/billing is separate.
        bound = sum(len(m["content"].encode("utf-8")) for m in messages) + max_output
        self.store.consume(self.run_id, self.epoch, self.token, model_calls=1, text_tokens=bound)


@contextmanager
def task_context(task_id, revision=0):
    parent = current_run.get()
    mark = current_run.set(parent.fork(task_id, revision) if parent else None)
    try:
        yield current_run.get()
    finally:
        current_run.reset(mark)


def tool_call(name, inputs, call, *, read_only=True):
    context = current_run.get()
    return context.tool(name, inputs, call, read_only=read_only) if context else call()
