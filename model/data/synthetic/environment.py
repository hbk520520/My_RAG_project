"""Executable fixture semantics; no legal inference, external tools or learned judge."""
from __future__ import annotations

from decimal import Decimal

from .schemas import Answer, Sample, ToolStep, Trajectory


def resolve(documents, record_ids, *, missing_user=False):
    """An explicit fixture policy, not a claim about legal authority in real documents."""
    values, evidence, missing = {}, [], []
    for record in record_ids:
        candidates = [d for d in documents if d.record_id == record and d.active and d.authority == "primary"]
        amounts = {d.amount for d in candidates}
        if len(amounts) != 1 or None in amounts:
            missing.append(record)
        else:
            values[record] = next(iter(amounts))
            evidence.extend(d.id for d in candidates)
    amount = str(sum((Decimal(n) / 100 for n in values.values()), Decimal(0)).quantize(Decimal("0.01")))
    return Answer(status="partial" if missing else "sufficient", amount=None if missing else amount,
                  evidence_ids=sorted(evidence), missing_records=missing,
                  missing_info_kind="user" if missing_user else "evidence")


class Environment:
    """Read-only records, one transient failure, bounded arithmetic and clarification."""
    def __init__(self, sample: Sample):
        self.sample = sample
        self.attempts = {}
        self.read = set()
        self.values = {}

    def execute(self, tool, arguments):
        operations = {name: operation for operation, name in self.sample.tools.items()}
        if tool not in operations:
            raise ValueError("Unknown fixture tool")
        operation = operations[tool]
        if operation == "read":
            if set(arguments) != {"record_id"} or arguments["record_id"] not in self.sample.record_ids:
                raise ValueError("Read requires a requested record_id")
            record = arguments["record_id"]
            self.attempts[record] = self.attempts.get(record, 0) + 1
            if record == self.sample.transient_record and self.attempts[record] == 1:
                return {"status": "retryable_error"}
            documents = [d for d in self.sample.documents if d.record_id == record]
            self.read.add(record)
            result = resolve(documents, [record], missing_user=self.sample.family == "missing")
            if result.amount is not None:
                self.values[record] = result.amount
            return {"status": result.status, "documents": [d.model_dump(mode="json") for d in documents]}
        if operation == "sum":
            if arguments != {"record_ids": self.sample.record_ids}:
                raise ValueError("Sum requires the exact requested record sequence")
            if set(self.values) != set(self.sample.record_ids):
                raise ValueError("Sum requires resolved successful reads of every requested record")
            return {"amount": str(sum((Decimal(self.values[r]) for r in self.sample.record_ids),
                                      Decimal(0)).quantize(Decimal("0.01")))}
        if operation == "ask":
            missing = resolve(self.sample.documents, self.sample.record_ids,
                              missing_user=self.sample.family == "missing").missing_records
            if not missing or self.read != set(self.sample.record_ids) or arguments != {"record_ids": missing}:
                raise ValueError("Clarification must follow reads and name exactly the unresolved records")
            return {"status": "waiting_input", "record_ids": missing}
        raise ValueError("Distractor tools cannot advance this task")


def reference_trajectory(sample):
    environment, steps = Environment(sample), []
    def call(operation, arguments):
        name = sample.tools[operation]
        observation = environment.execute(name, arguments)
        steps.append(ToolStep(tool=name, arguments=arguments, observation=observation))
        return observation
    for record in sample.record_ids:
        response = call("read", {"record_id": record})
        if response.get("status") == "retryable_error":
            call("read", {"record_id": record})
    if sample.expected.status == "sufficient":
        call("sum", {"record_ids": sample.record_ids})
    else:
        call("ask", {"record_ids": sample.expected.missing_records})
    return Trajectory(name="reference", steps=steps, answer=sample.expected)


def score_trajectory(sample, trajectory):
    """Replay observations independently; a right final answer cannot conceal bad steps."""
    environment, errors = Environment(sample), []
    for i, step in enumerate(trajectory.steps):
        try:
            observation = environment.execute(step.tool, step.arguments)
        except ValueError as exc:
            errors.append({"step": i, "reason": str(exc)})
        else:
            if observation != step.observation:
                errors.append({"step": i, "reason": "observation_mismatch"})
    expected = resolve(sample.documents, sample.record_ids, missing_user=sample.family == "missing")
    def comparable(answer):
        return {**answer.model_dump(), "evidence_ids": sorted(answer.evidence_ids),
                "missing_records": sorted(answer.missing_records)}
    final_correct = comparable(trajectory.answer) == comparable(expected)
    reference_calls = len(sample.record_ids) + 1 + bool(sample.transient_record)
    # Fixed interpretable scale: no fitted quantiles, no rewards learned from held-out data.
    cost = 0.1 * min(len(trajectory.steps) / reference_calls, 2)
    complete = environment.read == set(sample.record_ids) and bool(trajectory.steps)
    terminal = sample.tools["sum" if expected.status == "sufficient" else "ask"]
    complete = complete and trajectory.steps[-1].tool == terminal
    if not complete:
        errors.append({"step": len(trajectory.steps), "reason": "incomplete_execution"})
    step_correct = not errors and complete
    reward = round(1 - cost, 6) if final_correct and step_correct else 0.0
    return {"final_correct": final_correct, "steps_correct": step_correct, "errors": errors,
            "tool_calls": len(trajectory.steps), "reward": reward,
            "reward_version": "success-and-valid-steps-minus-cost-v1"}
