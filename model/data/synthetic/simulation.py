"""Small in-memory tool environment, not an executor for external systems."""
from __future__ import annotations

from .contracts import Outcome, Step, Task, Trajectory


def resolve(record, field):
    sources = [s for s in record.sources if s.active and s.authority == "primary"]
    values = {s.facts.get(field) for s in sources}
    if len(values) > 1:
        return "source_conflict", None
    if not values or None in values:
        return "missing_input", None
    return "ok", next(iter(values))


def oracle(task: Task):
    if task.write_required and not task.write_allowed:
        return Outcome(status="blocked", reason="permission_denied")
    if task.max_calls == 0:
        return Outcome(status="blocked", reason="budget_exhausted")
    values = [resolve(record, task.field) for record in task.records]
    evidence = sorted(s.id for r in task.records for s in r.sources if s.active and s.authority == "primary")
    for reason, status in (("source_conflict", "needs_evidence"), ("missing_input", "needs_input")):
        if any(state == reason for state, _ in values):
            return Outcome(status=status, reason=reason, evidence_ids=evidence)
    if task.operation == "copy":
        value = values[-1][1]
    elif task.operation == "sum":
        value = sum(value for _, value in values)
    else:
        value = {r.id: value for r, (_, value) in zip(task.records, values)}
    return Outcome(status="completed", value=value, evidence_ids=evidence)


class Environment:
    def __init__(self, task):
        self.task = task
        public = task.public_input()
        self.roles, self.assignments = public["roles"], public["assignments"]
        self.read = {}
        self.attempts = {}
        self.values = {}
        self.calls = 0
        self.result = None
        self.store = {}
        self.requested = False
        self.finished = False

    def execute(self, actor, name, arguments):
        operations = {name: op for op, name in self.task.tool_names.items()}
        if name not in operations or actor not in self.roles:
            raise ValueError("Unknown tool or actor")
        if self.finished:
            raise ValueError("Task already finished")
        op = operations[name]
        if op == "finish":
            if actor != "coordinator":
                raise ValueError("Only coordinator can submit the outcome")
            Outcome.model_validate(arguments)
            self.finished = True
            return {"status": "submitted"}
        if self.calls >= self.task.max_calls:
            raise ValueError("Tool budget exhausted")
        self.calls += 1
        seen = self.read.setdefault(actor, set())
        values = self.values.setdefault(actor, {})
        if op == "read":
            if set(arguments) != {"record_id"}:
                raise ValueError("read requires record_id")
            record = next((r for r in self.task.records if r.id == arguments["record_id"]), None)
            if record is None or not set(record.requires).issubset(seen):
                raise ValueError("Record unavailable or dependency not read")
            if self.assignments and self.assignments[record.id] != actor:
                raise ValueError("Record belongs to another role's branch")
            self.attempts[record.id] = self.attempts.get(record.id, 0) + 1
            if record.id == self.task.transient_record and self.attempts[record.id] == 1:
                return {"status": "retryable_error"}
            seen.add(record.id)
            state, value = resolve(record, self.task.field)
            if state == "ok":
                values[record.id] = value
            return {"status": state, self.task.response_format: [s.model_dump(mode="json") for s in record.sources],
                    "next_id": record.next_id}
        if op == "handoff":
            if set(arguments) != {"record_id"} or arguments["record_id"] not in seen or actor == "coordinator":
                raise ValueError("handoff requires the sender's successful read")
            record = arguments["record_id"]
            self.read.setdefault("coordinator", set()).add(record)
            if record in values:
                self.values.setdefault("coordinator", {})[record] = values[record]
            return {"status": "delivered", "recipient": "coordinator", "record_id": record}
        if op == "combine":
            if (arguments or self.task.operation == "copy" or actor != "coordinator"
                    or set(values) != {r.id for r in self.task.records}):
                raise ValueError("combine requires all resolved inputs delivered to coordinator")
            self.result = sum(values.values()) if self.task.operation == "sum" else dict(values)
            return {"value": self.result}
        if op == "write":
            if not self.task.write_allowed:
                raise ValueError("Write permission denied")
            if actor != "coordinator" or self.result is None or arguments != {"value": self.result}:
                raise ValueError("write requires coordinator's computed result")
            self.store["result"] = self.result
            return {"status": "stored"}
        if op == "request":
            expected = oracle(self.task)
            if (actor != "coordinator" or expected.status not in ("needs_input", "needs_evidence") or arguments != {"reason": expected.reason}
                    or seen != {r.id for r in self.task.records}):
                raise ValueError("request requires unresolved evidence and completed reads")
            self.requested = True
            return {"status": expected.status}
        if arguments:
            raise ValueError("weather tool takes no arguments")
        return {"weather": "晴"}


def reference(task):
    environment, steps = Environment(task), []
    def call(op, args, actor="coordinator"):
        tool = task.tool_names[op]
        observation = environment.execute(actor, tool, args)
        steps.append(Step(actor=actor, tool=tool, arguments=args, observation=observation))
        return observation
    answer = oracle(task)
    if answer.status != "blocked":
        for i, record in enumerate(task.records):
            actor = ("researcher_a" if i % 2 == 0 else "researcher_b") if task.family == "parallel" else "coordinator"
            result = call("read", {"record_id": record.id}, actor)
            if result["status"] == "retryable_error":
                call("read", {"record_id": record.id}, actor)
            if task.family == "parallel":
                call("handoff", {"record_id": record.id}, actor)
        if answer.status != "completed":
            call("request", {"reason": answer.reason})
        elif task.operation != "copy":
            call("combine", {})
            if task.write_required:
                call("write", {"value": environment.result})
    call("finish", answer.model_dump(mode="json"))
    return Trajectory(name="reference", steps=steps, answer=answer)


def score(task, trajectory):
    environment, errors = Environment(task), []
    for i, step in enumerate(trajectory.steps):
        try:
            observation = environment.execute(step.actor, step.tool, step.arguments)
        except ValueError as error:
            errors.append({"step": i, "reason": str(error)})
        else:
            if observation != step.observation:
                errors.append({"step": i, "reason": "observation_mismatch"})
    expected = oracle(task)
    normalize = lambda a: {**a.model_dump(), "evidence_ids": sorted(a.evidence_ids)}
    final_correct = normalize(expected) == normalize(trajectory.answer)
    submitted = (bool(trajectory.steps) and trajectory.steps[-1].tool == task.tool_names["finish"]
                 and trajectory.steps[-1].arguments == trajectory.answer.model_dump(mode="json"))
    expected_store = {"result": expected.value} if task.write_required and expected.status == "completed" else {}
    state_correct = environment.store == expected_store
    if expected.status == "completed":
        ready = environment.read.get("coordinator", set()) == {r.id for r in task.records}
        if task.operation != "copy":
            ready = ready and environment.result == expected.value
    else:
        ready = expected.status == "blocked" or environment.requested
    steps_correct = not errors and environment.finished and submitted and ready
    # A fixed task-local call cost; no fitted normalizer or training loss here.
    reference_calls = max(len(reference(task).steps) - 1, 1)
    reward = round(1 - 0.1 * min(environment.calls / reference_calls, 2), 6) if final_correct and state_correct and steps_correct else 0
    return {"final_correct": final_correct, "steps_correct": steps_correct, "state_correct": state_correct,
            "errors": errors, "tool_calls": environment.calls, "reward": reward,
            "reward_version": "verified-outcome-and-steps-minus-cost-v2"}
