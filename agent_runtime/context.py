"""Per-run context, durable budgets and replayable read-only tool calls."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib

from .store import dumps

current_run = ContextVar("agent_run", default=None)


@dataclass
class RunContext:
    store: object
    run_id: str
    epoch: int
    token: str
    node: str = ""
    step: str = ""
    ordinal: int = 0

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
        identity = f"{self.run_id}:{self.epoch}:{self.node}:{self.step}:{name}:{digest}"
        cached, result = self.store.begin_effect(self.run_id, self.epoch, self.token,
                                                identity, name, digest, read_only)
        if cached:
            return result
        try:
            result = call()
        except Exception as exc:
            self.store.end_effect(self.run_id, self.epoch, self.token, identity,
                                  {"error": f"{type(exc).__name__}: {exc}"}, status="failed")
            raise
        self.check()
        self.store.end_effect(self.run_id, self.epoch, self.token, identity, result)
        self.store.event(self.run_id, self.epoch, self.token, "tool_completed", {"tool": name})
        return result

    def model_budget(self, messages, max_output):
        # UTF-8 bytes upper-bound text tokens; protocol overhead/billing is separate.
        bound = sum(len(m["content"].encode("utf-8")) for m in messages) + max_output
        self.store.consume(self.run_id, self.epoch, self.token, model_calls=1, text_tokens=bound)


def tool_call(name, inputs, call, *, read_only=True):
    context = current_run.get()
    return context.tool(name, inputs, call, read_only=read_only) if context else call()
